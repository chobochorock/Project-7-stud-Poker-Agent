// Shadow-Regret Splitting on C++ 7-stud v3 (port of the Toy kernel
// Toy-Card-Game-Agent/agents/adaptive_buckets/shadow_split_native.cpp).
// Usage: python -m agents.cpp_mccfr.shadow_split.run --help; see README.md.
// Tree arm: one card tree per public node; leaves keep signed regret / strategy
// shadows per (feature, bin, action) and split on the cancellation statistic
// Lambda = (g(L) + g(R) - g(L + R)) / W, g(x) = max_a x_a^+.
// Baseline arm: the unchanged dynamic_tree_validated hard256 k-means arm.
#define VALIDATED_NO_MAIN
#include "../dynamic_tree_validated/solver.cpp"

namespace shadow7 {
using hand_recall::require;
using hand_recall::read_text;
using hand_recall::write_text;
using epoch7::read;
using epoch7::write;
using epoch7::Features;
using epoch7::Values;
namespace fs = std::filesystem;

constexpr int kBins = 8;
// Card power vector (0-17) plus the public chip/history summaries that still vary
// inside an own-recall public node: pot, opponent invested, aggression, length.
constexpr std::array<int, 22> kFeatureIndex{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 30, 31};
constexpr int kM = int(kFeatureIndex.size());
constexpr int kCells = kM * kBins;

struct Config {
    int key_mode = 2;      // 0 own-recall public node, 1 exact betting history, 2 k-means betting summary
    int rule = 1;          // 0 time-uniform radius x rho_scale, 1 z score without range term
    double rho_scale = 1.0, z = 0.66, delta = 0.05, span = 2000.0;
    double eps_min = 0.0, kappa = 1.0, eps_unit = 2.0;  // eps(t) = max(eps_min, kappa eps_unit / sqrt(t))
    uint32_t grace = 100, shadow_after = 100;
    int inherit = 2;       // 0 A (shadow only), 2 B (shadow + proportional rest), 3 reset
    int64_t max_splits = -1;
    uint64_t freeze_after = 0;
    double shadow_gib = 8.0;
};

// Quantile edges per street (5, 6, 7) and feature; bin(x) = #edges strictly below x,
// so a cut after bin k is the routing rule x <= edges[k].
struct Edges {
    std::array<std::array<std::array<float, kBins - 1>, kM>, 3> value{};
    int bin(int street, int j, float x) const {
        const auto& e = value[street - 5][j];
        return int(std::lower_bound(e.begin(), e.end(), x) - e.begin());
    }
};

std::array<float, kM> project(const Features& x) {
    std::array<float, kM> result{};
    for (int j = 0; j < kM; ++j) result[j] = x[kFeatureIndex[j]];
    return result;
}

// Data source: no-fold uniform-random play from fresh H4 roots (a data-policy bias:
// bins follow states that uniform play reaches, not the learned policy's states).
Edges fit_edges(const PowerAtlas& atlas, uint64_t seed, size_t per_street) {
    epoch7::FeatureCache cache{atlas};
    std::array<std::array<std::vector<float>, kM>, 3> values;
    std::mt19937_64 rng(seed);
    const auto enough = [&]() {
        for (const auto& street : values) { if (street[0].size() < per_street) return false; }
        return true;
    };
    for (uint64_t root = 1; !enough(); ++root) {
        require(root <= 100 * per_street, "edge sampling did not reach every street");
        State state = epoch7::root_for(seed ^ (root * 0x9e3779b97f4a7c15ull));
        while (!state.terminal) {
            const int actor = state.actor;
            const auto mask = valid_mask(state, actor);
            auto& bucket = values[state.street - 5];
            if (bucket[0].size() < per_street) {
                const auto x = project(cache(state, actor));
                for (int j = 0; j < kM; ++j) bucket[j].push_back(x[j]);
            }
            std::vector<Action> choices;
            for (auto action : actions_from_mask(mask)) { if (action != FOLD) choices.push_back(action); }
            require(!choices.empty(), "no non-fold action");
            state = epoch7::child(std::move(state),
                choices[std::uniform_int_distribution<size_t>(0, choices.size() - 1)(rng)]);
        }
    }
    Edges edges;
    for (int street = 0; street < 3; ++street) {
        for (int j = 0; j < kM; ++j) {
            auto& v = values[street][j];
            std::sort(v.begin(), v.end());
            for (int k = 0; k + 1 < kBins; ++k) {
                edges.value[street][j][k] = v[(k + 1) * v.size() / kBins];
            }
        }
    }
    return edges;
}

double positive_max(const double* values, int count) {
    double best = 0;
    for (int a = 0; a < count; ++a) best = std::max(best, values[a]);
    return best;
}

// Time-uniform empirical-Bernstein radius (Fan-Grama-Liu supermartingale, lambda grid
// 2^-k/span with weights 1/((k+1)(k+2))); identical to the Toy kernel with w_t = 1.
double radius(double q, double span, double log_cell) {
    const double w = 1.0;
    int lowest = int(std::floor(std::log2(w))) + 1;
    double scale = std::sqrt(std::max(q, 1e-300) / (2.0 * (log_cell + 3.0))) / span;
    int centre = int(std::lround(std::log2(std::max(scale, 1e-300))));
    double best = std::numeric_limits<double>::infinity();
    auto value = [&](int k) {
        const double down = std::ldexp(1.0, -k);
        return span * std::ldexp(1.0, k) * (log_cell + std::log((k + 1.0) * (k + 2.0))) +
            down * q / (2.0 * span * (1.0 - down * w));
    };
    for (int k = lowest; k <= lowest + 2; ++k) best = std::min(best, value(k));
    for (int k = std::max(lowest, centre - 2); k <= centre + 2; ++k) best = std::min(best, value(k));
    return best;
}

constexpr int kCuts = kM * (kBins - 1);

struct Shadow {
    // h, g, bh: [(j * kBins + b) * k + slot]; n, ns: [j * kBins + b]; k = legal actions.
    // ql, qr: [(j * (kBins - 1) + cut) * k + slot] sums over traversals of the squared
    // per-traversal side totals; qp: squared per-traversal parent totals. Squaring per
    // traversal (not per visit) keeps the variance right when a leaf recurs in one traversal.
    std::vector<double> h, g, bh, n, ns, ql, qr, qp;
    double w = 0;       // regret visits since allocation
    uint64_t life = 0;  // creation index of this shadow life (union bound)
    bool dirty = false;
};

size_t shadow_bytes_of(int k) {
    return (3 * size_t(kCells) * k + 2 * size_t(kCells) + 2 * size_t(kCuts) * k + k) * sizeof(double);
}

struct Leaf {
    uint32_t root = 0, depth = 0;
    uint8_t street = 5, mask = 0;
    int feature = -1, left = -1, right = -1, cut = -1;
    float threshold = 0;
    uint64_t updates = 0, stamp = 0;
    std::unique_ptr<Shadow> shadow;
};

std::string leaf_key(int id) { return dynamic_tree::leaf_key(id); }

std::string public_key(const State& state, int viewer, int mode, const PowerAtlas* atlas = nullptr) {
    require(viewer == state.actor && !state.terminal, "public key requires the acting player");
    std::string key;
    if (mode == 2) {
        // The hard256 baseline's betting summary and seat, without its card cluster: both arms
        // then share one betting partition and differ only in the card abstraction.
        require(atlas != nullptr, "summary key needs the atlas");
        const auto k = make_power_key(state, viewer, *atlas);
        for (int value : {int(k.street), viewer, int(k.legal_mask), int(k.pot_odds_bucket),
                int(k.stack_pot_bucket), int(k.own_bet_count), int(k.opponent_bet_count), int(k.checked),
                int(k.last_action_class), int(k.betting_goal)}) {
            key.push_back(char(value));
        }
        return key;
    }
    key.push_back(char(viewer));
    key.push_back(char(state.street));
    key.push_back(char(valid_mask(state, viewer)));
    for (const auto& event : state.history) {
        const bool own = event.actor == viewer;
        // own: exact own actions on earlier streets, exact current street;
        // exact: every event (opponent actions on earlier streets as well).
        if (mode == 0 && event.street < state.street && !own) { continue; }
        key.push_back(char(event.street));
        key.push_back(char(own));
        key.push_back(char(event.action));
    }
    return key;
}

struct Solver : hand_recall::Solver {
    Config config;
    Edges edges;
    epoch7::FeatureCache features;
    std::unordered_map<std::string, int> roots;
    std::vector<std::string> root_keys;
    std::vector<Leaf> tree;
    std::vector<int> pending;
    std::unordered_map<int, Values> frozen;  // one policy snapshot per leaf per traversal
    std::vector<int> dirty;                   // leaves with per-traversal sums to square
    uint64_t hand = 0, splits = 0, checks = 0, stuck = 0, revisits = 0, traversal = 0;
    uint64_t lives = 0, shadow_bytes = 0, shadow_leaves = 0, shadow_denied = 0;

    Solver(const PowerAtlas& atlas, uint64_t seed, Config options, Edges bins)
        : hand_recall::Solver(atlas, true, seed), config(options), edges(bins), features{atlas} {}

    static int actions_of(uint8_t mask) { return __builtin_popcount(mask); }
    static int slot(uint8_t mask, int action) {
        return __builtin_popcount(mask & ((1u << action) - 1));
    }

    bool frozen_splits() const {
        return (config.max_splits >= 0 && splits >= uint64_t(config.max_splits)) ||
            (config.freeze_after && hand > config.freeze_after);
    }

    int new_leaf(uint32_t root, uint8_t street, uint8_t mask, uint32_t depth) {
        require(tree.size() < size_t(std::numeric_limits<int>::max()) - 2, "tree index capacity exceeded");
        Leaf leaf;
        leaf.root = root; leaf.street = street; leaf.mask = mask; leaf.depth = depth;
        tree.push_back(std::move(leaf));
        return int(tree.size()) - 1;
    }

    int route(const State& state, int viewer, const std::array<float, kM>& x, bool create) {
        const auto key = public_key(state, viewer, config.key_mode, &atlas);
        auto found = roots.find(key);
        if (found == roots.end()) {
            if (!create) { return -1; }
            const uint32_t root = uint32_t(root_keys.size());
            const int id = new_leaf(root, uint8_t(state.street), valid_mask(state, viewer), 0);
            root_keys.push_back(key);
            found = roots.emplace(key, id).first;
            key_bytes += key.size();
        }
        int id = found->second;
        while (tree[id].feature >= 0) {
            id = x[tree[id].feature] <= tree[id].threshold ? tree[id].left : tree[id].right;
        }
        return id;
    }

    bool allocate(int id) {
        auto& leaf = tree[id];
        if (leaf.shadow || frozen_splits()) { return bool(leaf.shadow); }
        const size_t k = actions_of(leaf.mask);
        const size_t bytes = shadow_bytes_of(int(k));
        if (double(shadow_bytes + bytes) > config.shadow_gib * double(1ull << 30)) {
            ++shadow_denied;
            return false;
        }
        auto shadow = std::make_unique<Shadow>();
        shadow->h.assign(kCells * k, 0); shadow->g.assign(kCells * k, 0); shadow->bh.assign(kCells * k, 0);
        shadow->n.assign(kCells, 0); shadow->ns.assign(kCells, 0);
        shadow->ql.assign(kCuts * k, 0); shadow->qr.assign(kCuts * k, 0); shadow->qp.assign(k, 0);
        shadow->life = ++lives;
        leaf.shadow = std::move(shadow);
        shadow_bytes += bytes;
        ++shadow_leaves;
        return true;
    }

    void release(int id) {
        auto& leaf = tree[id];
        if (!leaf.shadow) { return; }
        shadow_bytes -= shadow_bytes_of(actions_of(leaf.mask));
        --shadow_leaves;
        leaf.shadow.reset();
    }

    double traverse(State state, int traverser) {
        ++nodes;
        if (state.terminal) { return terminal_net_search(state, traverser); }
        const int actor = state.actor;
        const auto mask = valid_mask(state, actor);
        const auto x = project(features(state, actor));
        const int id = route(state, actor, x, true);
        if (actor == traverser) {
            // A traverser leaf must be updated at most once per traversal (own history is in the key).
            // Opponent leaves may recur (earlier traverser actions are merged); they are not updated here.
            if (tree[id].stamp == traversal) { ++revisits; }
            tree[id].stamp = traversal;
        }
        auto inserted = table.try_emplace(leaf_key(id));
        if (inserted.second) { key_bytes += 4; }
        ++inserted.first->second.touches;
        auto snapshot = frozen.find(id);
        if (snapshot == frozen.end()) {
            snapshot = frozen.emplace(id, current_strategy(inserted.first->second, mask)).first;
        }
        const Values strategy = snapshot->second;
        const int k = actions_of(mask);
        if (actor != traverser) {
            auto& node = table.at(leaf_key(id));
            for (auto action : actions_from_mask(mask)) { node.strategy_sum[action] += strategy[action]; }
            if (auto* shadow = tree[id].shadow.get()) {
                for (int j = 0; j < kM; ++j) {
                    const int cell = j * kBins + edges.bin(state.street, j, x[j]);
                    for (auto action : actions_from_mask(mask)) {
                        shadow->g[cell * k + slot(mask, action)] += strategy[action];
                    }
                    shadow->ns[cell] += 1;
                }
            }
            return traverse(epoch7::child(std::move(state), epoch7::sample(strategy, mask, rng)), traverser);
        }
        Values values{}, regret{};
        double expected = 0;
        for (auto action : actions_from_mask(mask)) {
            values[action] = traverse(epoch7::child(state, action), traverser);
            expected += strategy[action] * values[action];
        }
        auto& node = table.at(leaf_key(id));
        for (auto action : actions_from_mask(mask)) {
            regret[action] = values[action] - expected;
            node.regrets[action] += regret[action];
        }
        auto& leaf = tree[id];
        ++leaf.updates;
        if (!leaf.shadow && leaf.updates >= config.shadow_after) { allocate(id); }
        if (leaf.shadow) {
            std::array<int, kM> bins{};
            for (int j = 0; j < kM; ++j) { bins[j] = edges.bin(state.street, j, x[j]); }
            record(id, bins, regret);
            // Test every `grace` visits recorded in this shadow (not since leaf creation).
            if (uint64_t(leaf.shadow->w) % config.grace == 0) { pending.push_back(id); }
        }
        return expected;
    }

    // One regret visit into the leaf shadow; squares are taken per traversal in flush().
    void record(int id, const std::array<int, kM>& bins, const Values& regret) {
        auto& leaf = tree[id];
        auto& shadow = *leaf.shadow;
        const uint8_t mask = leaf.mask;
        const int k = actions_of(mask);
        for (int j = 0; j < kM; ++j) {
            const int cell = j * kBins + bins[j];
            for (auto action : actions_from_mask(mask)) {
                shadow.h[cell * k + slot(mask, action)] += regret[action];
                shadow.bh[cell * k + slot(mask, action)] += regret[action];
            }
            shadow.n[cell] += 1;
        }
        shadow.w += 1;
        if (!shadow.dirty) { shadow.dirty = true; dirty.push_back(id); }
    }

    void flush() {
        for (int id : dirty) {
            auto* shadow = tree[id].shadow.get();
            if (!shadow) { continue; }
            const int k = actions_of(tree[id].mask);
            double total[kActionCount] = {};
            for (int b = 0; b < kBins; ++b) {
                for (int a = 0; a < k; ++a) total[a] += shadow->bh[b * k + a];
            }
            for (int a = 0; a < k; ++a) shadow->qp[a] += total[a] * total[a];
            for (int j = 0; j < kM; ++j) {
                double left[kActionCount] = {};
                for (int cut = 0; cut + 1 < kBins; ++cut) {
                    for (int a = 0; a < k; ++a) {
                        left[a] += shadow->bh[(j * kBins + cut) * k + a];
                        const double right = total[a] - left[a];
                        shadow->ql[(j * (kBins - 1) + cut) * k + a] += left[a] * left[a];
                        shadow->qr[(j * (kBins - 1) + cut) * k + a] += right * right;
                    }
                }
            }
            std::fill(shadow->bh.begin(), shadow->bh.end(), 0.0);
            shadow->dirty = false;
        }
        dirty.clear();
    }

    void train(const State& root, uint64_t index) {
        hand = index;
        for (int player : {0, 1}) {
            ++traversal;
            frozen.clear();
            traverse(root, player);
            flush();
        }
        frozen.clear();
    }

    struct Choice {
        int feature = -1, cut = -1;
        double lambda = 0, rho = 0, score = -std::numeric_limits<double>::infinity();
        double left_n = 0, right_n = 0;
    };

    Choice candidate(int id) const {
        const auto& leaf = tree[id];
        const auto& s = *leaf.shadow;
        const int k = actions_of(leaf.mask);
        const double total_cells = 2.0 * kM * (kBins - 1) + 1.0;
        const double life = double(s.life);
        const double log_cell = std::log(life * (life + 1.0) * total_cells * k * 2.0 / config.delta);
        // Parent cell: the same for every feature (feature 0 totals).
        std::array<double, kActionCount> parent{};
        for (int b = 0; b < kBins; ++b) {
            for (int a = 0; a < k; ++a) parent[a] += s.h[b * k + a];
        }
        const auto bound = [&](double q) {
            return config.rule == 0 ? radius(q, config.span, log_cell) : std::sqrt(q);
        };
        double parent_radius = 0;
        for (int a = 0; a < k; ++a) { parent_radius = std::max(parent_radius, bound(s.qp[a])); }
        const double parent_positive = positive_max(parent.data(), k);
        Choice best;
        const int candidates = config.key_mode == 2 ? kPowerDimensions : kM;  // summary: card features only
        for (int j = 0; j < candidates; ++j) {
            std::array<double, kActionCount> left{};
            double left_n = 0, all_n = 0;
            for (int b = 0; b < kBins; ++b) all_n += s.n[j * kBins + b];
            for (int cut = 0; cut + 1 < kBins; ++cut) {
                const int cell = j * kBins + cut;
                for (int a = 0; a < k; ++a) { left[a] += s.h[cell * k + a]; }
                left_n += s.n[cell];
                if (left_n <= 0 || left_n >= all_n) { continue; }
                double right[kActionCount], radius_left = 0, radius_right = 0;
                for (int a = 0; a < k; ++a) {
                    right[a] = parent[a] - left[a];
                    radius_left = std::max(radius_left, bound(s.ql[(j * (kBins - 1) + cut) * k + a]));
                    radius_right = std::max(radius_right, bound(s.qr[(j * (kBins - 1) + cut) * k + a]));
                }
                Choice c;
                c.feature = j; c.cut = cut; c.left_n = left_n; c.right_n = all_n - left_n;
                c.lambda = (positive_max(left.data(), k) + positive_max(right, k) - parent_positive) / s.w;
                const double scale = config.rule == 0 ? config.rho_scale : config.z;
                c.rho = scale * (radius_left + radius_right + parent_radius) / s.w;
                c.score = c.lambda - c.rho;
                if (c.score > best.score) { best = c; }
            }
        }
        return best;
    }

    void split(int id, const Choice& c) {
        auto& parent_leaf = tree[id];
        const auto& s = *parent_leaf.shadow;
        const uint8_t mask = parent_leaf.mask;
        const int k = actions_of(mask);
        std::array<double, kActionCount> hl{}, hr{}, gl{}, gr{};
        double nl = 0, nr = 0, sl = 0, sr = 0;
        for (int b = 0; b < kBins; ++b) {
            const int cell = c.feature * kBins + b;
            const bool is_left = b <= c.cut;
            for (int a = 0; a < k; ++a) {
                (is_left ? hl : hr)[a] += s.h[cell * k + a];
                (is_left ? gl : gr)[a] += s.g[cell * k + a];
            }
            (is_left ? nl : nr) += s.n[cell];
            (is_left ? sl : sr) += s.ns[cell];
        }
        const RegretNode parent = table.at(leaf_key(id));
        std::array<RegretNode, 2> children{};
        const double regret_share = nl + nr > 0 ? nl / (nl + nr) : 0.5;
        const double strategy_share = sl + sr > 0 ? sl / (sl + sr) : 0.5;
        const auto actions = actions_from_mask(mask);
        for (int a = 0; a < k; ++a) {
            const int action = actions[a];
            const double inherited = parent.regrets[action] - (hl[a] + hr[a]);
            const double inherited_strategy = parent.strategy_sum[action] - (gl[a] + gr[a]);
            if (config.inherit == 0) {
                children[0].regrets[action] = hl[a]; children[1].regrets[action] = hr[a];
                children[0].strategy_sum[action] = std::max(0.0, gl[a]);
                children[1].strategy_sum[action] = std::max(0.0, gr[a]);
            } else if (config.inherit == 2) {
                children[0].regrets[action] = hl[a] + regret_share * inherited;
                children[1].regrets[action] = hr[a] + (1 - regret_share) * inherited;
                children[0].strategy_sum[action] = std::max(0.0, gl[a] + strategy_share * inherited_strategy);
                children[1].strategy_sum[action] = std::max(0.0, gr[a] + (1 - strategy_share) * inherited_strategy);
            }
        }
        children[0].touches = uint64_t(std::llround(regret_share * double(parent.touches)));
        children[1].touches = parent.touches - children[0].touches;
        const float threshold = edges.value[parent_leaf.street - 5][c.feature][c.cut];
        const uint32_t root = parent_leaf.root, depth = parent_leaf.depth;
        const uint8_t street = parent_leaf.street;
        release(id);
        const int left = new_leaf(root, street, mask, depth + 1);
        const int right = new_leaf(root, street, mask, depth + 1);
        auto& node = tree[id];
        node.feature = c.feature; node.cut = c.cut; node.threshold = threshold;
        node.left = left; node.right = right;
        table.erase(leaf_key(id));
        table.emplace(leaf_key(left), children[0]);
        table.emplace(leaf_key(right), children[1]);
        key_bytes += 4;
        ++splits;
    }

    void refine(std::ostream& log) {
        std::sort(pending.begin(), pending.end());
        pending.erase(std::unique(pending.begin(), pending.end()), pending.end());
        for (int id : pending) {
            if (frozen_splits()) { break; }
            if (tree[id].feature >= 0 || !tree[id].shadow) { continue; }
            ++checks;
            const Choice c = candidate(id);
            const double eps = std::max(config.eps_min, config.kappa * config.eps_unit / std::sqrt(double(std::max<uint64_t>(hand, 1))));
            const bool accept = c.feature >= 0 && c.score > eps;
            if (c.feature < 0) { ++stuck; }
            const auto& leaf = tree[id];
            log << hand << ',' << id << ',' << leaf.root << ',' << int(leaf.street) << ',' << int(leaf.mask) << ','
                << leaf.depth << ',' << accept << ',' << (c.feature >= 0 ? kFeatureIndex[c.feature] : -1) << ','
                << c.cut << ',' << (c.feature >= 0 ? edges.value[leaf.street - 5][c.feature][c.cut] : 0.0f) << ','
                << c.lambda << ',' << c.rho << ',' << eps << ',' << leaf.shadow->w << ',' << c.left_n << ','
                << c.right_n << '\n';
            if (accept) { split(id, c); }
        }
        pending.clear();
        if (frozen_splits()) {
            for (int id = 0; id < int(tree.size()); ++id) { release(id); }
        }
    }

    void save(std::ostream& out) const {
        hand_recall::save_solver(out, *this);
        write(out, int32_t(config.key_mode));
        for (uint64_t value : {splits, checks, stuck, revisits, traversal, lives, shadow_denied}) { write(out, value); }
        write(out, edges.value);
        write(out, uint64_t(root_keys.size()));
        for (const auto& key : root_keys) { write_text(out, key); write(out, roots.at(key)); }
        write(out, uint64_t(tree.size()));
        for (const auto& leaf : tree) {
            write(out, leaf.root); write(out, leaf.depth); write(out, leaf.street); write(out, leaf.mask);
            write(out, leaf.feature); write(out, leaf.left); write(out, leaf.right); write(out, leaf.cut);
            write(out, leaf.threshold); write(out, leaf.updates); write(out, leaf.stamp);
            write(out, uint8_t(bool(leaf.shadow)));
            if (leaf.shadow) {
                const auto& s = *leaf.shadow;
                require(!s.dirty, "unflushed shadow at checkpoint");
                write(out, s.w); write(out, s.life); write(out, uint64_t(s.h.size()));
                for (const auto* array : {&s.h, &s.g, &s.n, &s.ns, &s.ql, &s.qr, &s.qp}) {
                    out.write(reinterpret_cast<const char*>(array->data()), array->size() * sizeof(double));
                }
            }
        }
        require(bool(out), "shadow checkpoint write failed");
    }

    void load(std::istream& in, bool keep_shadows = true) {
        hand_recall::load_solver(in, *this);
        // The public-node key decides routing, so a frozen model takes it from the checkpoint.
        const auto key_mode = read<int32_t>(in);
        require(key_mode >= 0 && key_mode <= 2, "invalid key mode");
        config.key_mode = key_mode;
        for (uint64_t* value : {&splits, &checks, &stuck, &revisits, &traversal, &lives, &shadow_denied}) {
            *value = read<uint64_t>(in);
        }
        edges.value = read<decltype(edges.value)>(in);
        roots.clear(); root_keys.clear(); tree.clear(); pending.clear(); dirty.clear(); frozen.clear();
        shadow_bytes = 0; shadow_leaves = 0;
        const auto root_count = read<uint64_t>(in);
        require(root_count <= 100000000, "invalid root count");
        for (uint64_t i = 0; i < root_count; ++i) {
            auto key = read_text(in, 10000);
            const int id = read<int>(in);
            require(roots.emplace(key, id).second, "duplicate public root");
            root_keys.push_back(std::move(key));
        }
        const auto count = read<uint64_t>(in);
        require(count <= uint64_t(std::numeric_limits<int>::max()), "invalid tree size");
        tree.resize(count);
        for (uint64_t id = 0; id < count; ++id) {
            auto& leaf = tree[id];
            leaf.root = read<uint32_t>(in); leaf.depth = read<uint32_t>(in);
            leaf.street = read<uint8_t>(in); leaf.mask = read<uint8_t>(in);
            leaf.feature = read<int>(in); leaf.left = read<int>(in); leaf.right = read<int>(in);
            leaf.cut = read<int>(in); leaf.threshold = read<float>(in);
            leaf.updates = read<uint64_t>(in); leaf.stamp = read<uint64_t>(in);
            require(leaf.street >= 5 && leaf.street <= 7 && leaf.feature >= -1 && leaf.feature < kM &&
                leaf.root < root_count && std::isfinite(leaf.threshold), "invalid leaf");
            if (leaf.feature >= 0) {
                require(leaf.left > int(id) && leaf.right > int(id) && leaf.left < int(count) &&
                    leaf.right < int(count), "invalid child indices");
            }
            if (read<uint8_t>(in)) {
                auto shadow = std::make_unique<Shadow>();
                shadow->w = read<double>(in); shadow->life = read<uint64_t>(in);
                const auto size = read<uint64_t>(in);
                const int k = actions_of(leaf.mask);
                require(size == uint64_t(kCells) * k, "invalid shadow size");
                shadow->h.resize(size); shadow->g.resize(size); shadow->bh.assign(size, 0.0);
                shadow->n.resize(kCells); shadow->ns.resize(kCells);
                shadow->ql.resize(size_t(kCuts) * k); shadow->qr.resize(size_t(kCuts) * k); shadow->qp.resize(k);
                for (auto* array : {&shadow->h, &shadow->g, &shadow->n, &shadow->ns, &shadow->ql, &shadow->qr,
                        &shadow->qp}) {
                    in.read(reinterpret_cast<char*>(array->data()), array->size() * sizeof(double));
                }
                require(bool(in), "truncated shadow");
                if (keep_shadows) {
                    shadow_bytes += shadow_bytes_of(k);
                    ++shadow_leaves;
                    leaf.shadow = std::move(shadow);
                }
            }
        }
        for (const auto& item : roots) {
            require(item.second >= 0 && item.second < int(tree.size()) && tree[item.second].depth == 0,
                "invalid root id");
        }
    }
};

struct Target {
    Solver& solver;
    bool average;
    uint64_t queries = 0, missing = 0;
    Values policy(const State& state, int viewer) {
        require(viewer == state.actor, "policy must be acting information set");
        ++queries;
        const auto x = project(solver.features(state, viewer));
        const int id = solver.route(state, viewer, x, false);
        const RegretNode empty{};
        const RegretNode* node = &empty;
        if (id >= 0) {
            auto found = solver.table.find(leaf_key(id));
            if (found != solver.table.end()) { node = &found->second; }
        }
        missing += node == &empty;
        return average ? average_strategy(*node, valid_mask(state, viewer))
            : current_strategy(*node, valid_mask(state, viewer));
    }
};

// Copy of dynamic_tree::measure for any target type (self-test checks equality).
template <class Policy>
dynamic_tree::Gap measure(const State& root, const std::vector<State>& particles, Policy& policy, uint64_t seed) {
    const auto actions = actions_from_mask(valid_mask(root, root.actor));
    const auto strategy = policy.policy(root, root.actor);
    Values q{}, sum{}, squares{};
    std::array<Values, 2> half{};
    const size_t middle = particles.size() / 2;
    for (size_t i = 0; i < particles.size(); ++i) {
        for (auto a : actions) {
            const double value = epoch7::rollout(epoch7::child(particles[i], a), root.actor,
                policy, seed + i * 0x9e3779b97f4a7c15ull);
            q[a] += value / particles.size();
            sum[a] += value; squares[a] += value * value;
            half[i >= middle][a] += value / (i < middle ? middle : particles.size() - middle);
        }
    }
    dynamic_tree::Gap result;
    double best = -std::numeric_limits<double>::infinity(), expected = 0;
    for (auto a : actions) {
        best = std::max(best, q[a]); expected += strategy[a] * q[a];
        result.action_se += std::sqrt(std::max(0.0, (squares[a] - sum[a] * sum[a] / particles.size()) /
            (particles.size() * (particles.size() - 1)))) / actions.size();
    }
    result.gap = std::max(0.0, best - expected);
    for (int side : {0, 1}) {
        const auto action = *std::max_element(actions.begin(), actions.end(), [&](Action a, Action b) {
            return half[side][a] < half[side][b];
        });
        double reference = 0;
        for (auto a : actions) { reference += strategy[a] * half[1 - side][a]; }
        result.crossfit += (half[1 - side][action] - reference) / 2;
    }
    return result;
}

void header(std::ostream& out) {
    // Same leading columns as dynamic_tree_validated (tree = shadow-split arm).
    out << "hands,tree_current,kmeans_current,tree_average,kmeans_average,root_seed";
    for (auto p : {"tree_current", "kmeans_current", "tree_average", "kmeans_average"}) {
        out << ',' << p << "_crossfit," << p << "_action_se," << p << "_missing," << p << "_queries";
    }
    out << ",tree_leaves,kmeans_states,splits,checks,tree_nodes,kmeans_nodes,peak_rss_bytes"
        << ",public_roots,shadow_leaves,shadow_bytes,stuck,revisits,shadow_denied\n";
}

void evaluate(std::ostream& out, uint64_t hand, uint64_t seed, int count, Solver& tree,
              dynamic_tree::Solver& baseline) {
    const auto root = epoch7::root_for(seed);
    std::mt19937_64 rng(seed ^ 0x123456789ull);
    const auto particles = epoch7::root_particles(root, count, rng);
    Target tree_current{tree, false}, tree_average{tree, true};
    dynamic_tree::Target base_current{baseline, false}, base_average{baseline, true};
    std::array<dynamic_tree::Gap, 4> gaps{
        measure(root, particles, tree_current, seed ^ 0xabcdefull),
        measure(root, particles, base_current, seed ^ 0xabcdefull),
        measure(root, particles, tree_average, seed ^ 0xabcdefull),
        measure(root, particles, base_average, seed ^ 0xabcdefull)};
    const std::array<std::pair<uint64_t, uint64_t>, 4> counts{{
        {tree_current.missing, tree_current.queries}, {base_current.missing, base_current.queries},
        {tree_average.missing, tree_average.queries}, {base_average.missing, base_average.queries}}};
    out << hand;
    for (const auto& g : gaps) { out << ',' << g.gap; }
    out << ',' << seed;
    for (size_t i = 0; i < gaps.size(); ++i) {
        out << ',' << gaps[i].crossfit << ',' << gaps[i].action_se << ',' << counts[i].first << ',' << counts[i].second;
    }
    out << ',' << tree.table.size() << ',' << baseline.table.size() << ',' << tree.splits << ',' << tree.checks
        << ',' << tree.nodes << ',' << baseline.nodes << ',' << epoch7::peak_rss()
        << ',' << tree.root_keys.size() << ',' << tree.shadow_leaves << ',' << tree.shadow_bytes
        << ',' << tree.stuck << ',' << tree.revisits << ',' << tree.shadow_denied << '\n';
    require(bool(out), "metric write failed");
}
}  // namespace shadow7

namespace shadow7 {
struct Progress { uint64_t hand = 0; std::array<uint64_t, 3> bytes{}; };
const std::array<std::string, 3> kLogs{"per_hand.csv", "panel.csv", "splits.csv"};

void save(const fs::path& path, const std::string& identity, Progress progress, const Solver& tree,
          const dynamic_tree::Solver& baseline) {
    std::ofstream out(path / "checkpoint.tmp", std::ios::binary | std::ios::trunc);
    write_text(out, "SHADOW7_1"); write_text(out, identity); write(out, progress.hand);
    for (const auto& name : kLogs) { write(out, uint64_t(fs::file_size(path / name))); }
    tree.save(out); baseline.save(out);
    out.close(); require(bool(out), "checkpoint close failed");
    require(MoveFileExW((path / "checkpoint.tmp").c_str(), (path / "checkpoint.bin").c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "checkpoint replace failed");
}

Progress read_header(std::istream& in, const std::string& identity) {
    require(read_text(in) == "SHADOW7_1" && read_text(in) == identity, "checkpoint identity mismatch");
    Progress result;
    result.hand = read<uint64_t>(in);
    for (auto& bytes : result.bytes) { bytes = read<uint64_t>(in); }
    return result;
}

Progress restore(const fs::path& path, const std::string& identity, Solver& tree, dynamic_tree::Solver& baseline) {
    std::ifstream in(path / "checkpoint.bin", std::ios::binary);
    const Progress result = read_header(in, identity);
    tree.load(in); baseline.load(in);
    require(in.peek() == std::char_traits<char>::eof(), "trailing checkpoint data");
    for (size_t i = 0; i < kLogs.size(); ++i) {
        require(fs::file_size(path / kLogs[i]) >= result.bytes[i], "log shorter than checkpoint");
    }
    for (size_t i = 0; i < kLogs.size(); ++i) { fs::resize_file(path / kLogs[i], result.bytes[i]); }
    return result;
}

void write_edges(const fs::path& path, const Edges& edges) {
    std::ofstream out(path);
    out << std::setprecision(9) << "street,feature";
    for (int k = 0; k + 1 < kBins; ++k) out << ",edge" << k;
    out << '\n';
    for (int street = 0; street < 3; ++street) {
        for (int j = 0; j < kM; ++j) {
            out << street + 5 << ',' << kFeatureIndex[j];
            for (float e : edges.value[street][j]) out << ',' << e;
            out << '\n';
        }
    }
    require(bool(out), "edge write failed");
}

// Synthetic shadow signal for tests: feature 0 decides the sign of the regrets.
void synthetic_signal(Solver& solver, int id, bool signal) {
    const auto actions = actions_from_mask(solver.tree[id].mask);
    for (int i = 0; i < 64; ++i) {
        const bool low = i % 2;
        const double r0 = signal ? (low ? 4.0 : -4.0) : 1.0;
        std::array<int, kM> bins{};
        if (signal && !low) { bins[0] = kBins - 1; }
        Values regret{};
        regret[actions[0]] = r0;
        regret[actions[1]] = -r0;
        solver.record(id, bins, regret);
        solver.flush();  // one visit per synthetic traversal
    }
}

void self_test(const std::string& atlas_path) {
    PowerAtlas atlas; atlas.load(atlas_path); atlas.set_assignment_cache_limit(100000);
    const Edges edges = fit_edges(atlas, 7, 400);
    for (const auto& street : edges.value) {
        for (const auto& e : street) { require(std::is_sorted(e.begin(), e.end()), "edges not sorted"); }
    }
    require(fit_edges(atlas, 7, 400).value == edges.value, "edge fitting not deterministic");
    const State root = epoch7::root_for(51);
    State hidden = root;
    hidden.players[1 - root.actor].hidden.clear(); hidden.players[1 - root.actor].has_discard = false;
    hidden.simulation_deck.clear();
    for (int mode : {0, 1, 2}) {
        require(public_key(root, root.actor, mode, &atlas) == public_key(hidden, hidden.actor, mode, &atlas),
            "key hidden leakage");
    }
    std::ostringstream log;
    Config config; config.grace = 4; config.shadow_after = 1; config.z = 0.5;
    // Cancellation on feature 0 must split on feature 0 and keep regret/average mass (rule B).
    Solver solver(atlas, 17, config, edges);
    std::array<float, kM> x{};
    const int parent = solver.route(root, root.actor, x, true);
    const uint8_t mask = solver.tree[parent].mask;
    const auto actions = actions_from_mask(mask);
    require(actions.size() >= 2, "need two actions");
    require(solver.allocate(parent), "shadow allocation failed");
    RegretNode original; original.regrets[actions[0]] = -3; original.regrets[actions[1]] = 9;
    original.strategy_sum[actions[0]] = 7; original.strategy_sum[actions[1]] = 3; original.touches = 31;
    solver.table[leaf_key(parent)] = original;
    synthetic_signal(solver, parent, true);
    solver.hand = 100;
    solver.pending.push_back(parent);
    solver.refine(log);
    require(solver.splits == 1 && solver.tree[parent].feature == 0, "synthetic cancellation did not split feature 0");
    require(!solver.tree[parent].shadow && solver.shadow_leaves == 0, "parent shadow not released");
    const auto& left = solver.table.at(leaf_key(solver.tree[parent].left));
    const auto& right = solver.table.at(leaf_key(solver.tree[parent].right));
    for (auto action : actions) {
        require(std::abs(left.regrets[action] + right.regrets[action] - original.regrets[action]) < 1e-9 &&
            std::abs(left.strategy_sum[action] + right.strategy_sum[action] - original.strategy_sum[action]) < 1e-9,
            "inheritance B lost mass");
    }
    require(left.regrets[actions[0]] > right.regrets[actions[0]], "children did not take their own shadows");
    require(solver.table.count(leaf_key(parent)) == 0, "parent table entry kept");
    // A common shift in one bin is stuck: no cut has visits on both sides.
    Solver flat(atlas, 3, config, edges);
    const int flat_id = flat.route(root, root.actor, x, true);
    require(flat.allocate(flat_id), "shadow allocation failed");
    flat.table[leaf_key(flat_id)] = RegretNode{};
    synthetic_signal(flat, flat_id, false);
    flat.hand = 100; flat.pending.push_back(flat_id); flat.refine(log);
    require(flat.splits == 0 && flat.stuck == 1, "single-bin leaf must be stuck, not split");
    // Real traversals in both key modes: one visit per leaf per traversal; exact resume.
    for (int mode : {0, 1, 2}) {
        Config real; real.key_mode = mode; real.grace = 8; real.shadow_after = 4; real.z = 0.0;
        Solver a(atlas, 23, real, edges), b(atlas, 23, real, edges);
        for (uint64_t h = 1; h <= 30; ++h) {
            const auto deal = epoch7::root_for(1000 + h);
            a.train(deal, h); a.refine(log);
        }
        require(mode == 2 || a.revisits == 0, "a traverser leaf was updated twice in one traversal");
        std::stringstream stream; a.save(stream); b.load(stream);
        require(b.table.size() == a.table.size() && b.tree.size() == a.tree.size() && b.rng == a.rng &&
            b.splits == a.splits && b.shadow_leaves == a.shadow_leaves && b.roots == a.roots, "resume differs");
        for (uint64_t h = 31; h <= 40; ++h) {
            const auto deal = epoch7::root_for(1000 + h);
            a.train(deal, h); a.refine(log); b.train(deal, h); b.refine(log);
        }
        require(a.splits == b.splits && a.table.size() == b.table.size(), "resumed run diverged");
        for (const auto& item : a.table) {
            const auto& other = b.table.at(item.first);
            require(item.second.regrets == other.regrets && item.second.strategy_sum == other.strategy_sum,
                "resumed statistics differ");
        }
        Target target{a, false};
        std::mt19937_64 rng(31);
        const auto particles = epoch7::root_particles(root, 4, rng);
        const auto before = a.table.size();
        require(measure(root, particles, target, 93).gap == epoch7::local_gap(root, particles, target, 93),
            "gap differs from existing evaluator");
        require(before == a.table.size(), "evaluation learned new states");
        require(target.policy(root, root.actor) == target.policy(hidden, hidden.actor), "policy hidden leakage");
        std::cout << "SELF_TEST key_mode=" << mode << " roots=" << a.root_keys.size() << " leaves=" << a.table.size()
                  << " splits=" << a.splits << " revisits=" << a.revisits << '\n';
    }
    dynamic_tree::Config dt;
    dynamic_tree::Solver base(atlas, false, 5, dt);
    dynamic_tree::Target base_target{base, false};
    std::mt19937_64 rng(33);
    const auto particles = epoch7::root_particles(root, 4, rng);
    require(measure(root, particles, base_target, 7).gap == dynamic_tree::measure(root, particles, base_target, 7).gap,
        "baseline gap differs from dynamic_tree::measure");
    std::cout << "SELF_TEST edges, keys, cancellation split, inheritance B, stuck, revisits, resume, gap: PASS\n";
}

volatile LONG stopping = 0;
BOOL WINAPI stop_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) { InterlockedExchange(&stopping, 1); return TRUE; }
    return FALSE;
}

int run(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--self-test") { self_test(argv[2]); return 0; }
    require(argc == 29, "Use python -m agents.cpp_mccfr.shadow_split.run --help");
    const fs::path path(argv[1]);
    const std::string identity(argv[2]);
    const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]), eval_seed = std::stoull(argv[5]);
    const int particles = std::stoi(argv[6]);
    const uint64_t panel_every = std::stoull(argv[7]);
    const int panel_roots = std::stoi(argv[8]);
    const uint64_t save_every = std::stoull(argv[9]);
    const bool resume = std::stoi(argv[10]);
    Config config;
    config.key_mode = std::stoi(argv[11]); config.rule = std::stoi(argv[12]);
    config.rho_scale = std::stod(argv[13]); config.z = std::stod(argv[14]); config.delta = std::stod(argv[15]);
    config.span = std::stod(argv[16]); config.eps_min = std::stod(argv[17]); config.kappa = std::stod(argv[18]);
    config.eps_unit = std::stod(argv[19]); config.grace = std::stoul(argv[20]);
    config.shadow_after = std::stoul(argv[21]); config.inherit = std::stoi(argv[22]);
    config.max_splits = std::stoll(argv[23]); config.freeze_after = std::stoull(argv[24]);
    config.shadow_gib = std::stod(argv[25]);
    const uint64_t max_states = std::stoull(argv[26]), reserve = std::stoull(argv[27]);
    const uint64_t max_tree_states = std::stoull(argv[28]);
    require(hands > 0 && particles >= 2 && panel_roots > 0 && panel_every && save_every &&
        config.key_mode >= 0 && config.key_mode <= 2 && (config.rule == 0 || config.rule == 1) &&
        config.rho_scale >= 0 && config.z >= 0 && config.delta > 0 && config.delta < 1 && config.span > 0 &&
        config.eps_min >= 0 && config.kappa >= 0 && config.eps_unit >= 0 && config.grace >= 1 &&
        (config.inherit == 0 || config.inherit == 2 || config.inherit == 3) && config.max_splits >= -1 &&
        config.shadow_gib >= 0, "invalid parameters");
    std::cout << "CONFIG key_mode=" << (config.key_mode == 0 ? "own" : config.key_mode == 1 ? "exact" : "summary")
              << " rule=" << (config.rule ? "z" : "time_uniform") << " rho_scale=" << config.rho_scale
              << " z=" << config.z << " grace=" << config.grace << " shadow_after=" << config.shadow_after
              << " inherit=" << config.inherit << " max_splits=" << config.max_splits
              << " shadow_gib=" << config.shadow_gib << std::endl;
    PowerAtlas atlas; atlas.load((path / "hand_atlas.bin").string()); atlas.set_assignment_cache_limit(100000);
    // Bin edges use a fixed seed so every run of this source shares the same bins.
    const Edges edges = fit_edges(atlas, 2026100801ull, 20000);
    Solver tree(atlas, seed, config, edges);
    dynamic_tree::Config dt;
    dynamic_tree::Solver baseline(atlas, false, seed, dt);
    Progress progress;
    if (resume) {
        const int key_mode = tree.config.key_mode;
        progress = restore(path, identity, tree, baseline);
        require(tree.edges.value == edges.value && tree.config.key_mode == key_mode, "bin edges or key mode changed");
    } else {
        write_edges(path / "bin_edges.csv", edges);
    }
    tree.hand = baseline.hand = progress.hand;
    require(progress.hand <= hands, "target below saved hand count");
    std::array<std::ofstream, 3> logs;
    for (size_t i = 0; i < logs.size(); ++i) {
        logs[i].open(path / kLogs[i], resume ? std::ios::app : std::ios::trunc);
        logs[i] << std::setprecision(17);
    }
    const auto panel = [&]() {
        for (int i = 0; i < panel_roots; ++i) {
            evaluate(logs[1], progress.hand, eval_seed ^ ((i + 1) * 0xd1b54a32d192ed03ull), particles, tree, baseline);
        }
    };
    const auto checkpoint = [&]() {
        for (auto& log : logs) { log.flush(); require(bool(log), "log flush failed"); }
        save(path, identity, progress, tree, baseline);
    };
    if (!resume) {
        header(logs[0]); header(logs[1]);
        logs[2] << "hands,leaf,root,street,mask,depth,accepted,feature,cut,threshold,lambda,rho,eps,weight,left_n,right_n\n";
        panel(); checkpoint();
    }
    SetConsoleCtrlHandler(stop_handler, TRUE);
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t hand = progress.hand + 1; hand <= hands; ++hand) {
        const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        tree.train(root, hand);
        baseline.train(root, hand);
        tree.refine(logs[2]);
        progress.hand = hand;
        evaluate(logs[0], hand, eval_seed ^ (hand * 0x94d049bb133111ebull), particles, tree, baseline);
        if (hand % panel_every == 0) { panel(); }
        const bool state_cap = baseline.table.size() >= max_states || tree.table.size() >= max_tree_states;
        const bool low_disk = fs::space(path).available < reserve;
        const bool capped = state_cap || low_disk;
        if (hand % save_every == 0 || hand == hands || stopping || capped) { checkpoint(); }
        if (hand % 100 == 0 || hand == hands || stopping || capped) {
            logs[0].flush(); logs[2].flush();
            std::cout << "PROGRESS hands=" << hand << " target=" << hands << " roots=" << tree.root_keys.size()
                      << " tree_leaves=" << tree.table.size() << " splits=" << tree.splits << " checks=" << tree.checks
                      << " stuck=" << tree.stuck << " shadow_leaves=" << tree.shadow_leaves
                      << " shadow_mib=" << tree.shadow_bytes / 1048576.0 << " revisits=" << tree.revisits
                      << " kmeans_states=" << baseline.table.size() << " tree_nodes=" << tree.nodes
                      << " kmeans_nodes=" << baseline.nodes << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0
                      << " wall_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << std::endl;
        }
        if (stopping || capped) {
            std::cout << "PAUSED reason=" << (stopping ? "user_stop" : state_cap ? "state_limit" : "low_disk")
                      << "; checkpoint saved; use --resume after addressing the limit\n";
            return 2;
        }
    }
    return 0;
}
}  // namespace shadow7

#ifndef SHADOW7_NO_MAIN
int main(int argc, char** argv) {
    try { return shadow7::run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 1; }
}
#endif
