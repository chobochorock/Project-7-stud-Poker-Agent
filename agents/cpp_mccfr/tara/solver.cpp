// TARA (tree-accumulated regret abstraction) on C++ 7-stud v3.
// Usage: python -m agents.cpp_mccfr.tara.run --help; see README.md.
// Tree arm: regret/average tables keyed by the leaves of forests (one small regression tree per
// public node); a new forest is fitted (in Python, sklearn) on every interval's data and added;
// older forests keep accumulating (RM+ per forest, per action). Action = RM(sum_j R_j[leaf_j]).
// Baseline arm: the unchanged dynamic_tree_validated hard256 k-means arm on the same deals.
#define VALIDATED_NO_MAIN
#include "../dynamic_tree_validated/solver.cpp"

namespace tara7 {
using hand_recall::require;
using hand_recall::read_text;
using hand_recall::write_text;
using epoch7::read;
using epoch7::write;
using epoch7::Values;
namespace fs = std::filesystem;

// Card features: power vector 0-17, EHS vs a uniform opponent, dead flush / straight / pair cards.
constexpr int kM = 22;
using Phi = std::array<float, kM>;
constexpr int kPhiPower = kPowerDimensions;

// Exposed cards (opponent up-cards and the viewer's own discard) that kill the viewer's draws.
std::array<float, 3> dead_counts(const State& state, int viewer) {
    const auto& me = state.players[viewer];
    std::vector<Card> mine = all_cards(me);
    std::vector<Card> dead = state.players[1 - viewer].shown;
    if (me.has_discard) { dead.push_back(me.discarded); }
    std::array<int, 4> suits{};
    std::array<bool, 15> held{};
    for (const auto& c : mine) {
        ++suits[c.suit];
        held[c.rank] = true;
        if (c.rank == 14) { held[1] = true; }
    }
    const int flush_suit = int(std::max_element(suits.begin(), suits.end()) - suits.begin());
    int best_lo = 1, best_held = -1;
    for (int lo = 1; lo <= 10; ++lo) {
        int count = 0;
        for (int r = lo; r < lo + 5; ++r) { count += held[r]; }
        if (count >= best_held) { best_held = count; best_lo = lo; }
    }
    float flush = 0, straight = 0, pair = 0;
    for (const auto& c : dead) {
        flush += c.suit == flush_suit;
        pair += held[c.rank];
        for (int r = best_lo; r < best_lo + 5; ++r) {
            if (!held[r] && (c.rank == r || (r == 1 && c.rank == 14))) { straight += 1; break; }
        }
    }
    return {flush, straight, pair};
}

// Monte Carlo showdown equity against a uniformly random opponent completion (ties count 1/2).
float equity(const State& state, int viewer, uint64_t seed, int samples) {
    const auto& me = state.players[viewer];
    const auto& op = state.players[1 - viewer];
    std::vector<Card> mine = all_cards(me);
    uint64_t known = card_mask(mine) | card_mask(op.shown);
    if (me.has_discard) { known |= card_mask({me.discarded}); }
    std::vector<Card> unseen;
    for (Card c : fresh_deck()) { if (!(card_mask({c}) & known)) unseen.push_back(c); }
    const int need_me = 7 - int(mine.size()), need_op = 7 - int(op.shown.size());
    require(need_me >= 0 && need_op >= 0 && need_me + need_op <= int(unseen.size()), "equity card counts");
    std::mt19937_64 rng(seed);
    double score = 0;
    std::vector<Card> a = mine, b = op.shown;
    for (int i = 0; i < samples; ++i) {
        for (int k = 0; k < need_me + need_op; ++k) {
            std::swap(unseen[k], unseen[k + std::uniform_int_distribution<int>(0, int(unseen.size()) - 1 - k)(rng)]);
        }
        a.resize(mine.size()); b.resize(op.shown.size());
        for (int k = 0; k < need_me; ++k) a.push_back(unseen[k]);
        for (int k = 0; k < need_op; ++k) b.push_back(unseen[need_me + k]);
        const auto x = best_hand_encoded(a), y = best_hand_encoded(b);
        score += x > y ? 1.0 : x == y ? 0.5 : 0.0;
    }
    return float(score / samples);
}

struct CardFeatures {
    const PowerAtlas& atlas;
    int samples = 64;
    std::unordered_map<PowerObservationKey, Phi, PowerObservationHash> cache;
    Phi operator()(const State& state, int viewer) {
        const auto key = power_observation_key(state, viewer);
        auto it = cache.find(key);
        if (it != cache.end()) { return it->second; }
        if (cache.size() >= 200000) { cache.clear(); }
        Phi x{};
        const auto power = atlas.vector(state, viewer);
        for (int i = 0; i < kPhiPower; ++i) x[i] = float(power[i]);
        x[kPhiPower] = equity(state, viewer, PowerObservationHash{}(key) ^ 0x5bd1e9955bd1e995ull, samples);
        const auto dead = dead_counts(state, viewer);
        for (int i = 0; i < 3; ++i) x[kPhiPower + 1 + i] = dead[i];
        return cache.emplace(key, x).first->second;
    }
};

// Public node key. 2 (default): the hard256 baseline betting summary and seat without its card
// cluster (both arms share one betting partition); 1: exact betting history (the design's key).
std::string public_key(const State& state, int viewer, int mode, const PowerAtlas& atlas) {
    require(viewer == state.actor && !state.terminal, "public key requires the acting player");
    std::string key;
    if (mode == 2) {
        const auto k = make_power_key(state, viewer, atlas);
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
        key.push_back(char(event.street));
        key.push_back(char(event.actor == viewer));
        key.push_back(char(event.action));
    }
    return key;
}

struct TreeNode { int32_t feature = -1; float threshold = 0; int32_t left = -1, right = -1, leaf = -1; };

// One abstraction "tree" = forest of per-public-node regression trees (kind 0), a fixed
// nearest-centroid card partition shared by the public nodes of a street (kind 1, baseline B2), or
// the hard256 atlas cluster of the co-trained k-means arm (kind 2, baseline B0: same partition as
// that arm when key_mode is summary, but trained with TARA's RM+ and linear average).
struct Forest {
    int32_t kind = 0, index = 1;  // index = creation order 1, 2, ... (weights use it)
    uint64_t born = 0;
    std::unordered_map<std::string, int32_t> roots;
    std::vector<TreeNode> nodes;
    std::array<std::vector<Phi>, 3> centroids;
    std::array<Phi, 3> mean{}, scale{};
    std::unordered_map<uint64_t, int32_t> cluster_leaf;
    std::vector<Values> R, S;
    std::vector<uint8_t> mask;

    int32_t new_leaf(uint8_t m) {
        R.emplace_back(); S.emplace_back(); mask.push_back(m);
        return int32_t(R.size()) - 1;
    }
    int32_t nearest(int street, const Phi& x) const {
        const auto& c = centroids[street - 5];
        int32_t best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < c.size(); ++i) {
            double d = 0;
            for (int j = 0; j < kM; ++j) {
                const double z = (x[j] - mean[street - 5][j]) / scale[street - 5][j] - c[i][j];
                d += z * z;
            }
            if (d < best_d) { best_d = d; best = int32_t(i); }
        }
        return best;
    }
    // Leaf of an information set; -1 when the public node was never seen and create is false.
    int32_t leaf(const std::string& key, int street, uint8_t m, const Phi& x, bool create, int cluster = -1) {
        auto it = roots.find(key);
        if (it == roots.end()) {
            if (!create) { return -1; }
            TreeNode node;
            if (kind == 0) { node.leaf = new_leaf(m); }  // kinds 1/2 keep a placeholder root
            nodes.push_back(node);
            it = roots.emplace(key, int32_t(nodes.size()) - 1).first;
        }
        int32_t id = it->second, result;
        if (kind >= 1) {
            require(kind == 1 || (cluster >= 0 && cluster < 65536), "atlas cluster missing");
            const uint64_t cell = (uint64_t(id) << 16) | uint64_t(kind == 1 ? nearest(street, x) : cluster);
            auto found = cluster_leaf.find(cell);
            if (found == cluster_leaf.end()) {
                if (!create) { return -1; }
                found = cluster_leaf.emplace(cell, new_leaf(m)).first;
            }
            result = found->second;
        } else {
            while (nodes[id].leaf < 0) { id = x[nodes[id].feature] <= nodes[id].threshold ? nodes[id].left : nodes[id].right; }
            result = nodes[id].leaf;
        }
        require(mask[result] == m, "bucket mixes legal action sets");
        return result;
    }
    uint64_t entries() const {
        uint64_t total = 0;
        for (auto m : mask) total += __builtin_popcount(m);
        return total;
    }
    void save(std::ostream& out) const {
        write(out, kind); write(out, index); write(out, born);
        write(out, uint64_t(roots.size()));
        for (const auto& item : roots) { write_text(out, item.first); write(out, item.second); }
        write(out, uint64_t(nodes.size()));
        for (const auto& n : nodes) { write(out, n); }
        for (int s = 0; s < 3; ++s) {
            write(out, uint64_t(centroids[s].size()));
            for (const auto& c : centroids[s]) write(out, c);
            write(out, mean[s]); write(out, scale[s]);
        }
        write(out, uint64_t(cluster_leaf.size()));
        for (const auto& item : cluster_leaf) { write(out, item.first); write(out, item.second); }
        write(out, uint64_t(R.size()));
        for (size_t i = 0; i < R.size(); ++i) { write(out, R[i]); write(out, S[i]); write(out, mask[i]); }
    }
    void load(std::istream& in) {
        kind = read<int32_t>(in); index = read<int32_t>(in); born = read<uint64_t>(in);
        require(kind >= 0 && kind <= 2, "invalid forest kind");
        roots.clear(); nodes.clear(); cluster_leaf.clear(); R.clear(); S.clear(); mask.clear();
        const auto root_count = read<uint64_t>(in);
        require(root_count <= 100000000, "invalid root count");
        for (uint64_t i = 0; i < root_count; ++i) {
            auto key = read_text(in, 10000);
            roots.emplace(std::move(key), read<int32_t>(in));
        }
        const auto node_count = read<uint64_t>(in);
        require(node_count <= 400000000, "invalid node count");
        nodes.resize(node_count);
        for (auto& n : nodes) { n = read<TreeNode>(in); }
        for (int s = 0; s < 3; ++s) {
            centroids[s].resize(read<uint64_t>(in));
            for (auto& c : centroids[s]) c = read<Phi>(in);
            mean[s] = read<Phi>(in); scale[s] = read<Phi>(in);
        }
        const auto cells = read<uint64_t>(in);
        for (uint64_t i = 0; i < cells; ++i) {
            const auto cell = read<uint64_t>(in);
            cluster_leaf.emplace(cell, read<int32_t>(in));
        }
        const auto leaves = read<uint64_t>(in);
        R.resize(leaves); S.resize(leaves); mask.resize(leaves);
        for (uint64_t i = 0; i < leaves; ++i) { R[i] = read<Values>(in); S[i] = read<Values>(in); mask[i] = read<uint8_t>(in); }
        for (const auto& n : nodes) {
            // Centroid forests keep placeholder roots (leaf -1, no children); leaves live in cluster_leaf.
            require(n.leaf < int32_t(leaves) && (kind >= 1 || n.leaf >= 0 || (n.feature >= 0 && n.feature < kM &&
                n.left >= 0 && n.right >= 0 && n.left < int32_t(node_count) && n.right < int32_t(node_count))),
                "invalid forest node");
        }
    }
};

// Fitted forest file written by run.py: per public node a list of nodes, leaves numbered 0..L-1.
Forest read_fitted(const fs::path& path, int32_t index, uint64_t born) {
    std::ifstream in(path, std::ios::binary);
    require(read_text(in) == "TARAFOREST1", "invalid forest file");
    Forest forest;
    forest.kind = read<int32_t>(in);
    forest.index = index; forest.born = born;
    if (forest.kind == 1) {
        for (int s = 0; s < 3; ++s) {
            forest.mean[s] = read<Phi>(in); forest.scale[s] = read<Phi>(in);
            forest.centroids[s].resize(read<uint32_t>(in));
            for (auto& c : forest.centroids[s]) c = read<Phi>(in);
            require(!forest.centroids[s].empty(), "empty centroid set");
        }
    }
    const auto keys = read<uint32_t>(in);
    for (uint32_t i = 0; i < keys; ++i) {
        const auto key = read_text(in, 10000);
        const auto m = read<uint8_t>(in);
        const auto count = read<uint32_t>(in);
        const int32_t base = int32_t(forest.nodes.size());
        std::vector<int32_t> leaf_index;
        for (uint32_t k = 0; k < count; ++k) {
            TreeNode n;
            n.feature = read<int32_t>(in); n.threshold = read<float>(in);
            n.left = read<int32_t>(in); n.right = read<int32_t>(in);
            const auto local_leaf = read<int32_t>(in);
            if (n.feature < 0) {
                n.leaf = forest.new_leaf(m);
                n.left = n.right = -1;
            } else {
                require(n.feature < kM && n.left > int32_t(k) && n.right > int32_t(k) &&
                    n.left < int32_t(count) && n.right < int32_t(count), "invalid fitted node");
                n.left += base; n.right += base; n.leaf = -1;
            }
            (void)local_leaf;
            forest.nodes.push_back(n);
        }
        require(count > 0 && forest.roots.emplace(key, base).second, "duplicate or empty fitted root");
    }
    require(in.peek() == std::char_traits<char>::eof(), "trailing forest data");
    return forest;
}

struct Config {
    int key_mode = 2;
    int weighting = 0;    // 0 pow2 (2^(t-1) previous, 2^t current), 1 halving equivalent, 2 uniform, 3 index j, 4 index frozen
    int prev_last = 0;    // 1: only the previous forest keeps updating (older forests frozen)
    int latest_only = 0;  // 1: B3, previous forests are discarded
    int avg_mode = 0;     // 0 linear (hand index), 1 same as the forest's regret weight
    int collect = 1;      // keep the interval buffer for the next forest
    uint32_t reservoir = 4096;
    int avg_fallback = 1; // average with zero strategy mass: 1 current RM of sum_j R_j (k-means arm convention), 0 uniform
};

struct Row { Phi x; std::array<float, kActionCount> r; };
struct Slot { uint8_t mask = 0; uint8_t street = 5; uint64_t visits = 0; double mass = 0; std::vector<Row> rows; };

struct Solver {
    const PowerAtlas& atlas;
    Config config;
    CardFeatures features;
    std::vector<Forest> forests;
    std::vector<double> weights;
    uint64_t interval = 1, hand = 0, nodes = 0, traversal = 0, forests_made = 1;
    uint64_t interval_end = 0;  // end of the current interval (set per call; 0 = unknown, e.g. frozen LBR models)
    std::mt19937_64 rng, reservoir_rng;
    std::unordered_map<uint64_t, Values> frozen;
    std::unordered_map<std::string, Slot> buffer;

    Solver(const PowerAtlas& source, uint64_t seed, Config options)
        : atlas(source), config(options), features{source}, rng(seed), reservoir_rng(seed ^ 0x3c6ef372fe94f82bull) {
        Forest trivial;  // tree 1: one leaf per public node, card-blind
        forests.push_back(std::move(trivial));
        set_weights();
    }

    void set_weights() {
        const size_t count = forests.size();
        weights.assign(count, 0.0);
        const double t = double(interval);
        for (size_t j = 0; j < count; ++j) {
            const bool current = j + 1 == count;
            double w = 1.0;
            switch (config.weighting) {
                case 0: w = current ? std::ldexp(1.0, int(t)) : std::ldexp(1.0, int(t) - 1); break;
                case 1: w = current ? 1.0 : 0.5; break;
                case 2: w = 1.0; break;
                case 3: w = double(forests[j].index); break;
                case 4: w = current ? t : 0.0; break;
            }
            if (!current && config.prev_last && j + 2 < count) { w = 0.0; }
            weights[j] = w;
        }
    }

    void add_forest(Forest forest) {
        if (config.weighting == 1) {  // halving: rescale history so the next interval uses weights 0.5 / 1
            for (auto& f : forests) {
                for (auto& r : f.R) for (auto& v : r) v *= 0.5;
                if (config.avg_mode == 1) { for (auto& s : f.S) for (auto& v : s) v *= 0.5; }
            }
        }
        forest.index = int32_t(++forests_made);
        if (config.latest_only) { forests.clear(); }
        forests.push_back(std::move(forest));
        ++interval;
        set_weights();
    }

    struct Lookup { std::array<int32_t, 64> leaf{}; size_t count = 0; uint64_t hash = 1469598103934665603ull; };

    Lookup locate(const State& state, int viewer, const Phi& x, bool create) {
        const auto key = public_key(state, viewer, config.key_mode, atlas);
        const auto m = valid_mask(state, viewer);
        Lookup result;
        result.count = forests.size();
        require(result.count <= result.leaf.size(), "too many forests");
        int cluster = -1;
        for (const auto& f : forests) {
            if (f.kind == 2) { cluster = int(make_power_key(state, viewer, atlas).power_cluster); break; }
        }
        for (size_t j = 0; j < forests.size(); ++j) {
            result.leaf[j] = forests[j].leaf(key, state.street, m, x, create, cluster);
            result.hash = (result.hash ^ uint64_t(uint32_t(result.leaf[j]) + j * 0x9e3779b9ull)) * 1099511628211ull;
        }
        return result;
    }

    Values strategy(const Lookup& at, uint8_t m) const {
        RegretNode node;
        for (size_t j = 0; j < at.count; ++j) {
            if (at.leaf[j] < 0) continue;
            const auto& r = forests[j].R[at.leaf[j]];
            for (int a = 0; a < kActionCount; ++a) node.regrets[a] += r[a];
        }
        return current_strategy(node, m);
    }

    // Newest-forest average: until the newest forest has seen half of its interval, use the previous one
    // (as in the Leduc reference); at boundary snapshots and in frozen models the newest forest is used.
    size_t newest_mature(const Lookup& at) const {
        size_t j = at.count - 1;
        const uint64_t born = forests[j].born;
        if (j > 0 && interval_end > born && 2 * (hand - born) < interval_end - born) { --j; }
        return j;
    }

    Values average(const Lookup& at, uint8_t m, bool latest_only) const {
        RegretNode node;
        const size_t first = latest_only ? newest_mature(at) : 0, last = latest_only ? first + 1 : at.count;
        for (size_t j = first; j < last; ++j) {
            if (at.leaf[j] < 0) continue;
            const auto& s = forests[j].S[at.leaf[j]];
            for (int a = 0; a < kActionCount; ++a) node.strategy_sum[a] += s[a];
        }
        double mass = 0;
        for (auto a : actions_from_mask(m)) mass += node.strategy_sum[a];
        // dynamic_tree::Target falls back to the bucket's current RM strategy; match it (option 1).
        if (mass <= 0 && config.avg_fallback == 1) { return strategy(at, m); }
        return average_strategy(node, m);
    }

    void record(const std::string& key, int street, uint8_t m, const Phi& x, const Values& regret) {
        auto& slot = buffer[key];
        slot.mask = m; slot.street = uint8_t(street);
        ++slot.visits;
        double norm = 0;
        Row row; row.x = x;
        for (int a = 0; a < kActionCount; ++a) { row.r[a] = float(regret[a]); norm += regret[a] * regret[a]; }
        slot.mass += std::sqrt(norm);
        if (slot.rows.size() < config.reservoir) { slot.rows.push_back(row); return; }
        const uint64_t j = std::uniform_int_distribution<uint64_t>(0, slot.visits - 1)(reservoir_rng);
        if (j < config.reservoir) { slot.rows[j] = row; }
    }

    double traverse(State state, int traverser) {
        ++nodes;
        if (state.terminal) { return terminal_net_search(state, traverser); }
        const int actor = state.actor;
        const auto m = valid_mask(state, actor);
        const Phi x = features(state, actor);
        const Lookup at = locate(state, actor, x, true);
        // Buckets can recur within one traversal: one policy snapshot per bucket tuple.
        auto snapshot = frozen.find(at.hash);
        if (snapshot == frozen.end()) { snapshot = frozen.emplace(at.hash, strategy(at, m)).first; }
        const Values sigma = snapshot->second;
        if (actor != traverser) {
            const double linear = double(hand);
            for (size_t j = 0; j < at.count; ++j) {
                if (weights[j] <= 0) continue;
                const double w = config.avg_mode == 0 ? linear : weights[j];
                auto& s = forests[j].S[at.leaf[j]];
                for (auto a : actions_from_mask(m)) s[a] += w * sigma[a];
            }
            return traverse(epoch7::child(std::move(state), epoch7::sample(sigma, m, rng)), traverser);
        }
        Values values{}, regret{};
        double expected = 0;
        for (auto a : actions_from_mask(m)) {
            values[a] = traverse(epoch7::child(state, a), traverser);
            expected += sigma[a] * values[a];
        }
        for (auto a : actions_from_mask(m)) regret[a] = values[a] - expected;
        for (size_t j = 0; j < at.count; ++j) {
            if (weights[j] <= 0) continue;
            auto& r = forests[j].R[at.leaf[j]];
            for (auto a : actions_from_mask(m)) {
                const double v = r[a] + weights[j] * regret[a];
                r[a] = v > 0 ? v : 0.0;  // RM+: clip per forest, per action, every update
            }
        }
        if (config.collect) { record(public_key(state, actor, config.key_mode, atlas), state.street, m, x, regret); }
        return expected;
    }

    void train(const State& root, uint64_t index) {
        hand = index;
        for (int player : {0, 1}) {
            ++traversal;
            frozen.clear();
            traverse(root, player);
        }
        frozen.clear();
    }

    uint64_t table_entries() const {
        uint64_t total = 0;
        for (const auto& f : forests) total += f.entries();
        return total;
    }
    // Partition structure units: tree nodes (kind 0), plus centroids and cell->leaf entries (kinds 1/2).
    uint64_t tree_nodes() const {
        uint64_t total = 0;
        for (const auto& f : forests) {
            total += f.nodes.size() + f.cluster_leaf.size();
            for (const auto& c : f.centroids) total += c.size();
        }
        return total;
    }
    uint64_t leaves() const {
        uint64_t total = 0;
        for (const auto& f : forests) total += f.R.size();
        return total;
    }

    void write_buffer(const fs::path& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        write_text(out, "TARABUFFER1");
        write(out, uint32_t(buffer.size()));
        for (const auto& item : buffer) {
            write_text(out, item.first);
            const auto& slot = item.second;
            write(out, slot.mask); write(out, slot.street); write(out, slot.visits); write(out, slot.mass);
            write(out, uint32_t(slot.rows.size()));
            for (const auto& row : slot.rows) { write(out, row.x); write(out, row.r); }
        }
        out.close();
        require(bool(out), "buffer write failed");
    }

    void save(std::ostream& out, bool with_buffer = true) const {
        for (int32_t v : {config.key_mode, config.weighting, config.prev_last, config.latest_only, config.avg_mode,
                config.collect, config.avg_fallback}) { write(out, v); }
        for (uint64_t v : {interval, hand, nodes, traversal, forests_made}) { write(out, v); }
        write(out, int32_t(features.samples));  // equity samples define the feature map
        std::ostringstream state; state << rng << ' ' << reservoir_rng;
        write_text(out, state.str());
        write(out, uint64_t(forests.size()));
        for (const auto& f : forests) f.save(out);
        write(out, uint64_t(with_buffer ? buffer.size() : 0));
        if (with_buffer) {
            for (const auto& item : buffer) {
                write_text(out, item.first);
                const auto& slot = item.second;
                write(out, slot.mask); write(out, slot.street); write(out, slot.visits); write(out, slot.mass);
                write(out, uint64_t(slot.rows.size()));
                for (const auto& row : slot.rows) write(out, row);
            }
        }
        require(bool(out), "solver checkpoint write failed");
    }

    void load(std::istream& in) {
        for (int* v : {&config.key_mode, &config.weighting, &config.prev_last, &config.latest_only, &config.avg_mode,
                &config.collect, &config.avg_fallback}) { *v = read<int32_t>(in); }
        for (uint64_t* v : {&interval, &hand, &nodes, &traversal, &forests_made}) { *v = read<uint64_t>(in); }
        features.samples = read<int32_t>(in);
        require(features.samples >= 1, "invalid equity samples");
        std::istringstream state(read_text(in)); state >> rng >> reservoir_rng;
        require(bool(state), "invalid RNG state");
        forests.resize(read<uint64_t>(in));
        require(!forests.empty() && forests.size() <= 64, "invalid forest count");
        for (auto& f : forests) f.load(in);
        buffer.clear();
        const auto slots = read<uint64_t>(in);
        for (uint64_t i = 0; i < slots; ++i) {
            auto key = read_text(in, 10000);
            Slot slot;
            slot.mask = read<uint8_t>(in); slot.street = read<uint8_t>(in);
            slot.visits = read<uint64_t>(in); slot.mass = read<double>(in);
            slot.rows.resize(read<uint64_t>(in));
            for (auto& row : slot.rows) row = read<Row>(in);
            buffer.emplace(std::move(key), std::move(slot));
        }
        frozen.clear();
        set_weights();
    }
};

// Policy views: kind 0 current RM(sum R), 1 average from sum_j S_j, 2 average from the newest forest.
struct Target {
    Solver& solver;
    int kind;
    uint64_t queries = 0, missing = 0;
    Values policy(const State& state, int viewer) {
        require(viewer == state.actor, "policy must be acting information set");
        ++queries;
        const Phi x = solver.features(state, viewer);
        const auto at = solver.locate(state, viewer, x, false);
        bool any = false;
        for (size_t j = 0; j < at.count; ++j) any = any || at.leaf[j] >= 0;
        missing += !any;
        const auto m = valid_mask(state, viewer);
        if (kind == 0) return solver.strategy(at, m);
        return solver.average(at, m, kind == 2);
    }
};

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
}  // namespace tara7

namespace tara7 {
void header(std::ostream& out, bool panel) {
    // Leading columns match dynamic_tree_validated (tree = TARA arm, tree_average = sum_j S_j output).
    out << "hands,tree_current,kmeans_current,tree_average,kmeans_average,root_seed";
    for (auto p : {"tree_current", "kmeans_current", "tree_average", "kmeans_average"}) {
        out << ',' << p << "_crossfit," << p << "_action_se," << p << "_missing," << p << "_queries";
    }
    out << ",tree_leaves,kmeans_states,splits,checks,tree_nodes,kmeans_nodes,peak_rss_bytes"
        << ",forests,table_entries,struct_nodes,buffer_rows";
    if (panel) { out << ",tree_latest,tag"; }
    out << '\n';
}

uint64_t buffer_rows(const Solver& s) {
    uint64_t total = 0;
    for (const auto& item : s.buffer) total += item.second.rows.size();
    return total;
}

void evaluate(std::ostream& out, uint64_t hand, uint64_t seed, int count, Solver& tree,
              dynamic_tree::Solver& baseline, const char* tag) {
    const auto root = epoch7::root_for(seed);
    std::mt19937_64 rng(seed ^ 0x123456789ull);
    const auto particles = epoch7::root_particles(root, count, rng);
    Target tree_current{tree, 0}, tree_average{tree, 1};
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
    out << ',' << tree.leaves() << ',' << baseline.table.size() << ',' << tree.forests_made - 1 << ',' << 0
        << ',' << tree.nodes << ',' << baseline.nodes << ',' << epoch7::peak_rss()
        << ',' << tree.forests.size() << ',' << tree.table_entries() << ',' << tree.tree_nodes()
        << ',' << buffer_rows(tree);
    if (tag) {
        Target latest{tree, 2};
        out << ',' << measure(root, particles, latest, seed ^ 0xabcdefull).gap << ',' << tag;
    }
    out << '\n';
    require(bool(out), "metric write failed");
}

void forest_rows(std::ostream& out, const Solver& s, const char* tag) {
    for (size_t j = 0; j < s.forests.size(); ++j) {
        const auto& f = s.forests[j];
        double sum_r = 0, sum_s = 0;
        for (size_t i = 0; i < f.R.size(); ++i) {
            for (int a = 0; a < kActionCount; ++a) { sum_r += f.R[i][a]; sum_s += f.S[i][a]; }
        }
        out << s.hand << ',' << tag << ',' << f.index << ',' << f.kind << ',' << f.born << ',' << f.R.size() << ','
            << f.entries() << ',' << f.nodes.size() << ',' << f.roots.size() << ',' << sum_r << ',' << sum_s << ','
            << s.weights[j] << '\n';
    }
}

// Probe states for stability: decision states met in self-play of the average output (sum_j S_j)
// at B - delta, stored as (deal seed, action prefix) so they can be replayed after the tree change.
struct Probe { uint64_t seed = 0; std::vector<uint8_t> actions; };

State replay(const Probe& p) {
    State state = epoch7::root_for(p.seed);
    for (auto a : p.actions) state = epoch7::child(std::move(state), Action(a));
    return state;
}

std::vector<Probe> make_probes(Solver& tree, uint64_t boundary, uint64_t eval_seed, int episodes) {
    std::vector<Probe> probes;
    Target average{tree, 1};
    for (int e = 0; e < episodes; ++e) {
        Probe p;
        p.seed = eval_seed ^ (boundary * 0x9e3779b97f4a7c15ull) ^ (uint64_t(e + 1) * 0xd1b54a32d192ed03ull);
        std::mt19937_64 rng(p.seed ^ 0x51ed270b27f3a2c9ull);
        State state = epoch7::root_for(p.seed);
        while (!state.terminal) {
            probes.push_back(p);
            const auto m = valid_mask(state, state.actor);
            const Action a = epoch7::sample(average.policy(state, state.actor), m, rng);
            p.actions.push_back(uint8_t(a));
            state = epoch7::child(std::move(state), a);
        }
    }
    return probes;
}

void save_probes(const fs::path& path, const std::vector<Probe>& probes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    write(out, uint64_t(probes.size()));
    for (const auto& p : probes) {
        write(out, p.seed); write(out, uint32_t(p.actions.size()));
        for (auto a : p.actions) write(out, a);
    }
    require(bool(out), "probe write failed");
}

std::vector<Probe> load_probes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "missing probe file");
    std::vector<Probe> probes(read<uint64_t>(in));
    for (auto& p : probes) {
        p.seed = read<uint64_t>(in);
        p.actions.resize(read<uint32_t>(in));
        for (auto& a : p.actions) a = read<uint8_t>(in);
    }
    return probes;
}

void probe_rows(std::ostream& out, Solver& tree, const std::vector<Probe>& probes, uint64_t boundary, const char* tag) {
    static const char* const names[] = {"current", "average", "average_latest"};
    for (size_t i = 0; i < probes.size(); ++i) {
        const State state = replay(probes[i]);
        const auto m = valid_mask(state, state.actor);
        for (int kind : {0, 1, 2}) {
            Target target{tree, kind};
            const auto p = target.policy(state, state.actor);
            out << tree.hand << ',' << boundary << ',' << tag << ',' << i << ',' << int(state.street) << ','
                << int(m) << ',' << names[kind];
            for (double v : p) out << ',' << v;
            out << '\n';
        }
    }
    require(bool(out), "probe rows failed");
}

struct Progress { uint64_t hand = 0; std::array<uint64_t, 4> bytes{}; };
const std::array<std::string, 4> kLogs{"per_hand.csv", "panel.csv", "forests.csv", "probes.csv"};

void save(const fs::path& path, const std::string& identity, Progress progress, const Solver& tree,
          const dynamic_tree::Solver& baseline) {
    std::ofstream out(path / "checkpoint.tmp", std::ios::binary | std::ios::trunc);
    write_text(out, "TARA7_1"); write_text(out, identity); write(out, progress.hand);
    for (const auto& name : kLogs) { write(out, uint64_t(fs::file_size(path / name))); }
    tree.save(out); baseline.save(out);
    out.close(); require(bool(out), "checkpoint close failed");
    require(MoveFileExW((path / "checkpoint.tmp").c_str(), (path / "checkpoint.bin").c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "checkpoint replace failed");
}

Progress read_header(std::istream& in, const std::string& identity) {
    require(read_text(in) == "TARA7_1" && read_text(in) == identity, "checkpoint identity mismatch");
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

// Policy-only snapshot (no interval buffer) for frozen Policy-LBR at interval boundaries.
void save_model(const fs::path& file, const std::string& identity, uint64_t hand, const Solver& tree,
                const dynamic_tree::Solver& baseline) {
    std::ofstream out(file.string() + ".tmp", std::ios::binary | std::ios::trunc);
    write_text(out, "TARA7_MODEL1"); write_text(out, identity); write(out, hand);
    tree.save(out, false); baseline.save(out);
    out.close(); require(bool(out), "model snapshot write failed");
    require(MoveFileExW(fs::path(file.string() + ".tmp").c_str(), file.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "model snapshot replace failed");
}

// Two-centroid forest on the equity feature (tests and synthetic checks only).
Forest equity_split_forest() {
    Forest f;
    f.kind = 1;
    for (int s = 0; s < 3; ++s) {
        f.mean[s].fill(0.0f); f.scale[s].fill(1.0f);
        Phi low{}, high{};
        high[kPhiPower] = 1.0f;
        f.centroids[s] = {low, high};
    }
    return f;
}

void self_test(const std::string& atlas_path) {
    PowerAtlas atlas; atlas.load(atlas_path); atlas.set_assignment_cache_limit(100000);
    const State root = epoch7::root_for(51);
    State hidden = root;
    hidden.players[1 - root.actor].hidden.clear(); hidden.players[1 - root.actor].has_discard = false;
    hidden.simulation_deck.clear();
    CardFeatures features{atlas};
    require(features(root, root.actor) == features(hidden, hidden.actor), "feature hidden leakage");
    const auto x = features(root, root.actor);
    require(x[kPhiPower] >= 0 && x[kPhiPower] <= 1, "equity out of range");
    for (int mode : {1, 2}) {
        require(public_key(root, root.actor, mode, atlas) == public_key(hidden, hidden.actor, mode, atlas),
            "key hidden leakage");
    }
    // Tree 1 alone is card-blind: two deals with the same public node get the same policy.
    {
        Config config; config.collect = 0;
        Solver solver(atlas, 5, config);
        for (uint64_t h = 1; h <= 30; ++h) solver.train(epoch7::root_for(700 + h), h);
        Target t0{solver, 0};
        int compared = 0;
        for (uint64_t a = 0; a < 40 && compared < 5; ++a) {
            const auto r1 = epoch7::root_for(900 + a), r2 = epoch7::root_for(1900 + a);
            if (public_key(r1, r1.actor, 2, atlas) != public_key(r2, r2.actor, 2, atlas)) continue;
            require(t0.policy(r1, r1.actor) == t0.policy(r2, r2.actor), "tree 1 used card information");
            ++compared;
        }
        require(compared > 0, "no comparable public nodes");
    }
    // 2^t weighting and the halving implementation give the same strategies.
    {
        Config pow2, halving; halving.weighting = 1; pow2.collect = halving.collect = 0;
        Solver a(atlas, 9, pow2), b(atlas, 9, halving);
        for (uint64_t h = 1; h <= 60; ++h) {
            if (h == 21 || h == 41) {
                auto fa = equity_split_forest(), fb = equity_split_forest();
                fa.born = fb.born = h - 1;
                a.add_forest(std::move(fa)); b.add_forest(std::move(fb));
            }
            const auto deal = epoch7::root_for(3000 + h);
            a.train(deal, h); b.train(deal, h);
        }
        require(a.forests.size() == 3 && a.weights[2] == 8.0 && a.weights[0] == 4.0 && b.weights[2] == 1.0 &&
            b.weights[0] == 0.5, "weights");
        Target ta{a, 0}, tb{b, 0};
        double worst = 0;
        for (uint64_t k = 0; k < 50; ++k) {
            State s = epoch7::root_for(5000 + k);
            std::mt19937_64 walk(k);
            while (!s.terminal) {
                const auto pa = ta.policy(s, s.actor), pb = tb.policy(s, s.actor);
                for (int i = 0; i < kActionCount; ++i) worst = std::max(worst, std::abs(pa[i] - pb[i]));
                s = epoch7::child(std::move(s), epoch7::sample(pa, valid_mask(s, s.actor), walk));
            }
        }
        require(worst < 1e-9, "halving implementation differs from 2^t weighting");
        std::cout << "SELF_TEST pow2 vs halving max |dp| = " << worst << '\n';
    }
    // Weights of the ablations and the latest-only rule.
    {
        Config last; last.prev_last = 1; last.collect = 0;
        Solver s(atlas, 1, last);
        s.add_forest(equity_split_forest()); s.add_forest(equity_split_forest());
        require(s.weights[0] == 0.0 && s.weights[1] == 4.0 && s.weights[2] == 8.0, "prev_scope last weights");
        Config index; index.weighting = 3; index.collect = 0;
        Solver i(atlas, 1, index);
        i.add_forest(equity_split_forest()); i.add_forest(equity_split_forest());
        require(i.weights[0] == 1.0 && i.weights[1] == 2.0 && i.weights[2] == 3.0, "index weights");
        Config latest; latest.latest_only = 1; latest.collect = 0;
        Solver l(atlas, 1, latest);
        l.add_forest(equity_split_forest());
        require(l.forests.size() == 1 && l.forests[0].index == 2 && l.weights[0] == 4.0, "latest only");
    }
    // Reservoir buffer, resume equality, and the gap evaluator.
    {
        Config config; config.reservoir = 8;
        Solver a(atlas, 13, config), b(atlas, 13, config);
        for (uint64_t h = 1; h <= 20; ++h) a.train(epoch7::root_for(8000 + h), h);
        uint64_t visits = 0;
        for (const auto& item : a.buffer) {
            require(item.second.rows.size() <= 8 && item.second.rows.size() == std::min<uint64_t>(8, item.second.visits),
                "reservoir size");
            visits += item.second.visits;
        }
        require(visits > 0, "no buffer visits");
        a.add_forest(equity_split_forest());
        std::stringstream stream; a.save(stream); b.load(stream);
        for (uint64_t h = 21; h <= 30; ++h) {
            const auto deal = epoch7::root_for(8000 + h);
            a.train(deal, h); b.train(deal, h);
        }
        for (size_t j = 0; j < a.forests.size(); ++j) {
            require(a.forests[j].R == b.forests[j].R && a.forests[j].S == b.forests[j].S, "resume differs");
        }
        Target target{a, 0};
        std::mt19937_64 rng(31);
        const auto particles = epoch7::root_particles(root, 4, rng);
        const auto leaves = a.leaves();
        require(measure(root, particles, target, 93).gap == epoch7::local_gap(root, particles, target, 93),
            "gap differs from existing evaluator");
        require(leaves == a.leaves(), "evaluation created leaves");
        require(target.policy(root, root.actor) == target.policy(hidden, hidden.actor), "policy hidden leakage");
    }
    {  // B0: kind-2 leaves are in one-to-one correspondence with the k-means arm's keys
        Forest atlas_forest; atlas_forest.kind = 2;
        Solver b0(atlas, 5, Config{});
        b0.forests.clear(); b0.forests.push_back(atlas_forest); b0.set_weights();
        dynamic_tree::Solver km(atlas, false, 5, dynamic_tree::Config{});
        std::unordered_map<int32_t, std::string> leaf_key;
        std::unordered_map<std::string, int32_t> key_leaf;
        std::mt19937_64 walk(77);
        size_t checked = 0;
        for (uint64_t h = 0; h < 300; ++h) {
            State s = epoch7::root_for(9100 + h);
            while (!s.terminal) {
                const auto at = b0.locate(s, s.actor, b0.features(s, s.actor), true);
                epoch7::Features unused{}; int id = -1;
                const auto k = km.key(s, unused, id, true);
                require(leaf_key.emplace(at.leaf[0], k).first->second == k &&
                        key_leaf.emplace(k, at.leaf[0]).first->second == at.leaf[0], "B0 partition differs from k-means");
                ++checked;
                const auto actions = actions_from_mask(valid_mask(s, s.actor));
                s = epoch7::child(std::move(s), actions[walk() % actions.size()]);
            }
        }
        std::cout << "SELF_TEST B0 atlas forest = k-means partition on " << checked << " decisions, "
                  << key_leaf.size() << " buckets\n";
    }
    std::cout << "SELF_TEST features, keys, card-blind tree 1, pow2=halving, weights, latest, reservoir, resume, gap, B0: PASS\n";
}

volatile LONG stopping = 0;
BOOL WINAPI stop_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) { InterlockedExchange(&stopping, 1); return TRUE; }
    return FALSE;
}

int run(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--self-test") { self_test(argv[2]); return 0; }
    require(argc == 29, "Use python -m agents.cpp_mccfr.tara.run --help");
    const fs::path path(argv[1]);
    const std::string identity(argv[2]);
    const uint64_t stop_hand = std::stoull(argv[3]);
    const bool boundary = std::stoi(argv[4]);
    const uint64_t next_boundary = std::stoull(argv[5]), prev_boundary = std::stoull(argv[6]);
    const std::string add_path(argv[7]), initial_path(argv[8]);
    const bool resume = std::stoi(argv[9]);
    const uint64_t seed = std::stoull(argv[10]), eval_seed = std::stoull(argv[11]);
    const int particles = std::stoi(argv[12]);
    const uint64_t panel_every = std::stoull(argv[13]);
    const int panel_roots = std::stoi(argv[14]);
    const uint64_t save_every = std::stoull(argv[15]);
    Config config;
    config.key_mode = std::stoi(argv[16]); config.weighting = std::stoi(argv[17]);
    config.prev_last = std::stoi(argv[18]); config.latest_only = std::stoi(argv[19]);
    config.avg_mode = std::stoi(argv[20]); config.collect = std::stoi(argv[21]);
    config.reservoir = std::stoul(argv[22]);
    const int ehs_samples = std::stoi(argv[23]);
    const uint64_t probe_delta = std::stoull(argv[24]);
    const int probe_episodes = std::stoi(argv[25]);
    const uint64_t max_states = std::stoull(argv[26]), reserve = std::stoull(argv[27]);
    config.avg_fallback = std::stoi(argv[28]);
    require(stop_hand > 0 && particles >= 2 && panel_roots > 0 && panel_every && save_every &&
        (config.key_mode == 1 || config.key_mode == 2) && config.weighting >= 0 && config.weighting <= 4 &&
        config.reservoir >= 2 && ehs_samples >= 1 && probe_episodes >= 0 &&
        (config.avg_fallback == 0 || config.avg_fallback == 1), "invalid parameters");
    PowerAtlas atlas; atlas.load((path / "hand_atlas.bin").string()); atlas.set_assignment_cache_limit(100000);
    Solver tree(atlas, seed, config);
    tree.features.samples = ehs_samples;
    dynamic_tree::Config dt;
    dynamic_tree::Solver baseline(atlas, false, seed, dt);
    Progress progress;
    if (resume) {
        progress = restore(path, identity, tree, baseline);
    } else if (initial_path != "-") {
        tree.forests.clear();
        tree.forests.push_back(read_fitted(initial_path, 1, 0));
        tree.set_weights();
    }
    tree.hand = baseline.hand = progress.hand;
    tree.interval_end = stop_hand;  // a call always ends at the current interval's boundary or at the final hand
    if (resume && progress.hand == stop_hand) {
        // This segment already finished (its buffer and model were written before the checkpoint), e.g. a
        // stop request arrived during the last hand and the caller did not record it: report it again.
        std::cout << "SEGMENT already complete at hand " << stop_hand << std::endl;
        return boundary ? 3 : 0;
    }
    require(progress.hand < stop_hand, "stop hand must be ahead of the saved hand");
    std::array<std::ofstream, 4> logs;
    for (size_t i = 0; i < logs.size(); ++i) {
        logs[i].open(path / kLogs[i], resume ? std::ios::app : std::ios::trunc);
        logs[i] << std::setprecision(17);
    }
    const auto panel = [&](const char* tag) {
        for (int i = 0; i < panel_roots; ++i) {
            evaluate(logs[1], progress.hand, eval_seed ^ ((i + 1) * 0xd1b54a32d192ed03ull), particles, tree, baseline, tag);
        }
    };
    const auto checkpoint = [&]() {
        for (auto& log : logs) { log.flush(); require(bool(log), "log flush failed"); }
        save(path, identity, progress, tree, baseline);
    };
    if (!resume) {
        header(logs[0], false); header(logs[1], true);
        logs[2] << "hands,tag,forest,kind,born,leaves,entries,struct_nodes,public_nodes,sum_regret,sum_average,weight\n";
        logs[3] << "hands,boundary,tag,probe,street,mask,policy,p0,p1,p2,p3,p4,p5,p6,p7\n";
        panel("regular"); forest_rows(logs[2], tree, "start"); checkpoint();
    }
    // forest_<k>.bin is forest number k; skip it when the checkpoint already contains it.
    uint64_t number = 0;
    if (add_path != "-") {
        const auto stem = fs::path(add_path).stem().string();
        number = std::stoull(stem.substr(stem.find('_') + 1));
        require(number <= tree.forests_made + 1, "forest number ahead of the checkpoint");
    }
    if (number == tree.forests_made + 1) {
        auto forest = read_fitted(add_path, 0, progress.hand);
        tree.add_forest(std::move(forest));
        forest_rows(logs[2], tree, "added");
        // Same probe states right after the zero-initialized tree is added (expected bit-identical to "at").
        const auto file = path / ("probes_" + std::to_string(progress.hand) + ".bin");
        if (probe_delta && probe_episodes && fs::exists(file)) {
            probe_rows(logs[3], tree, load_probes(file), progress.hand, "added");
        }
        tree.buffer.clear();  // the interval buffer only holds data of the current interval
        std::cout << "FOREST added interval=" << tree.interval << " forests=" << tree.forests.size()
                  << " leaves=" << tree.leaves() << std::endl;
    }
    SetConsoleCtrlHandler(stop_handler, TRUE);
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t hand = progress.hand + 1; hand <= stop_hand; ++hand) {
        const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        tree.train(root, hand);
        baseline.train(root, hand);
        progress.hand = hand;
        evaluate(logs[0], hand, eval_seed ^ (hand * 0x94d049bb133111ebull), particles, tree, baseline, nullptr);
        if (hand % panel_every == 0) { panel("regular"); }
        if (probe_delta && next_boundary > probe_delta && hand == next_boundary - probe_delta && probe_episodes) {
            const auto probes = make_probes(tree, next_boundary, eval_seed, probe_episodes);
            save_probes(path / ("probes_" + std::to_string(next_boundary) + ".bin"), probes);
            probe_rows(logs[3], tree, probes, next_boundary, "before");
            panel("before");
        }
        if (probe_delta && prev_boundary && hand == prev_boundary + probe_delta && probe_episodes) {
            const auto probes = load_probes(path / ("probes_" + std::to_string(prev_boundary) + ".bin"));
            probe_rows(logs[3], tree, probes, prev_boundary, "after");
            panel("after");
        }
        const bool state_cap = baseline.table.size() >= max_states;
        const bool low_disk = fs::space(path).available < reserve;
        const bool capped = state_cap || low_disk;
        if (hand == stop_hand && boundary) {
            if (probe_delta && probe_episodes && fs::exists(path / ("probes_" + std::to_string(hand) + ".bin"))) {
                const auto probes = load_probes(path / ("probes_" + std::to_string(hand) + ".bin"));
                probe_rows(logs[3], tree, probes, hand, "at");
                panel("at");
            }
            tree.write_buffer(path / ("buffer_" + std::to_string(tree.interval) + ".bin"));
            forest_rows(logs[2], tree, "boundary");
        }
        if (hand == stop_hand) {
            fs::create_directories(path / "models");
            save_model(path / "models" / ("model_" + std::to_string(hand) + ".bin"), identity, hand, tree, baseline);
        }
        if (hand % save_every == 0 || hand == stop_hand || stopping || capped) { checkpoint(); }
        if (hand % 100 == 0 || hand == stop_hand || stopping || capped) {
            logs[0].flush();
            std::cout << "PROGRESS hands=" << hand << " stop=" << stop_hand << " interval=" << tree.interval
                      << " forests=" << tree.forests.size() << " leaves=" << tree.leaves()
                      << " entries=" << tree.table_entries() << " buffer_rows=" << buffer_rows(tree)
                      << " kmeans_states=" << baseline.table.size() << " tree_nodes=" << tree.nodes
                      << " kmeans_nodes=" << baseline.nodes << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0
                      << " wall_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                      << std::endl;
        }
        if ((stopping || capped) && hand != stop_hand) {
            std::cout << "PAUSED reason=" << (stopping ? "user_stop" : state_cap ? "kmeans_state_limit" : "low_disk")
                      << "; checkpoint saved\n";
            return 2;
        }
    }
    return boundary ? 3 : 0;
}
}  // namespace tara7

#ifndef TARA7_NO_MAIN
int main(int argc, char** argv) {
    try { return tara7::run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 1; }
}
#endif
