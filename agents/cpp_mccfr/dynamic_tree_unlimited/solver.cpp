// Incremental regret regression splits, not the proved CFR+IRA algorithm.
// Usage: python -m agents.cpp_mccfr.dynamic_tree_unlimited.run --help; see README.md.
#define HAND_RECALL_NO_MAIN
#include "../hand_recall.cpp"
#include <map>
#include <set>

namespace dynamic_tree {
using hand_recall::require;
using hand_recall::read_text;
using hand_recall::write_text;
using epoch7::read;
using epoch7::write;
using epoch7::Features;
using epoch7::Values;
namespace fs = std::filesystem;

struct Config {
    uint32_t window = 256, min_child = 32;
    int64_t max_splits = -1, max_depth = -1;
    double gain = 0.20, min_rmse = 1.0;
};
struct Sample { Features x{}; Values regret{}; uint64_t hand = 0; };
struct Node {
    uint32_t group = 0, depth = 0;
    int feature = -1, left = -1, right = -1;
    float threshold = 0;
    uint64_t seen = 0, sample_hand = 0, hand_visits = 0;
    std::vector<Sample> samples;
};
struct Split {
    int feature = -1;
    float threshold = 0;
    double gain = 0, rmse = 0, left_mass = 0;
};

std::string leaf_key(int id) {
    std::string result;
    hand_recall::number(result, static_cast<uint32_t>(id));
    return result;
}

struct Solver : hand_recall::Solver {
    Config config;
    epoch7::FeatureCache features;
    std::map<uint32_t, int> roots;
    std::vector<Node> tree;
    std::set<int> pending;
    std::unordered_map<std::string, Values> frozen;
    std::mt19937_64 monitor_rng;
    uint64_t hand = 0, splits = 0, checks = 0;

    Solver(const PowerAtlas& atlas, bool dynamic, uint64_t seed, Config options)
        : hand_recall::Solver(atlas, dynamic, seed), config(options), features{atlas},
          monitor_rng(seed ^ 0x243f6a8885a308d3ull) {}

    bool split_limit_reached() const {
        return config.max_splits >= 0 && splits >= uint64_t(config.max_splits);
    }

    bool depth_limit_reached(uint32_t depth) const {
        return config.max_depth >= 0 && depth >= uint64_t(config.max_depth);
    }

    int route(uint32_t group, const Features& x, bool create) {
        auto found = roots.find(group);
        if (found == roots.end()) {
            if (!create) { return -1; }
            require(tree.size() < size_t(std::numeric_limits<int>::max()), "tree index capacity exceeded");
            const int id = int(tree.size());
            roots.emplace(group, id);
            Node node;
            node.group = group;
            tree.push_back(node);
            found = roots.find(group);
        }
        int id = found->second;
        while (tree[id].feature >= 0) {
            const auto& node = tree[id];
            id = x[node.feature] <= node.threshold ? node.left : node.right;
        }
        return id;
    }

    std::string key(const State& state, Features& x, int& id, bool create) {
        const int actor = state.actor;
        if (recall) {
            x = features(state, actor);
            id = route(epoch7::group_id(state, actor), x, create);
            return id < 0 ? std::string{} : leaf_key(id);
        }
        const auto k = make_power_key(state, actor, atlas);
        // Existing hard256 betting-summary partition, with explicit seat.
        std::string result;
        for (int value : {int(k.street), actor, int(k.legal_mask),
                int(k.power_cluster & 255), int(k.power_cluster >> 8),
                int(k.pot_odds_bucket), int(k.stack_pot_bucket),
                int(k.own_bet_count), int(k.opponent_bet_count), int(k.checked),
                int(k.last_action_class), int(k.betting_goal)}) {
            result.push_back(static_cast<char>(value));
        }
        return result;
    }

    void observe(int id, const Features& x, const Values& regret) {
        if (split_limit_reached() || depth_limit_reached(tree[id].depth)) { return; }
        auto& node = tree[id];
        const Sample sample{x, regret, hand};
        // One uniform visit per leaf/hand prevents a single traversal filling the buffer.
        if (node.sample_hand == hand && node.hand_visits) {
            ++node.hand_visits;
            if (std::uniform_int_distribution<uint64_t>(1, node.hand_visits)(monitor_rng) == 1) {
                node.samples[(node.seen - 1) % config.window] = sample;
            }
            return;
        }
        node.sample_hand = hand;
        node.hand_visits = 1;
        if (node.samples.size() < config.window) { node.samples.push_back(sample); }
        else { node.samples[node.seen % config.window] = sample; }
        ++node.seen;
        if (node.seen % config.window == 0) { pending.insert(id); }
    }

    double traverse(State state, int traverser) {
        ++nodes;
        if (state.terminal) { return terminal_net_search(state, traverser); }
        const int actor = state.actor;
        const auto mask = valid_mask(state, actor);
        Features x{};
        int id = -1;
        const std::string identity = key(state, x, id, true);
        auto inserted = table.try_emplace(identity);
        if (inserted.second) { key_bytes += identity.size(); }
        ++inserted.first->second.touches;
        // Merged leaves can recur within a traversal: use one policy snapshot.
        auto strategy_it = frozen.find(identity);
        if (strategy_it == frozen.end()) {
            strategy_it = frozen.emplace(identity, current_strategy(inserted.first->second, mask)).first;
        }
        const Values strategy = strategy_it->second;
        if (actor != traverser) {
            for (auto action : actions_from_mask(mask)) {
                table.at(identity).strategy_sum[action] += strategy[action];
            }
            return traverse(epoch7::child(std::move(state), epoch7::sample(strategy, mask, rng)), traverser);
        }
        Values values{}, regret{};
        double expected = 0;
        for (auto action : actions_from_mask(mask)) {
            values[action] = traverse(epoch7::child(state, action), traverser);
            expected += strategy[action] * values[action];
        }
        for (auto action : actions_from_mask(mask)) {
            regret[action] = values[action] - expected;
            table.at(identity).regrets[action] += regret[action];
        }
        if (recall) { observe(id, x, regret); }
        return expected;
    }

    void train(const State& root, uint64_t index) {
        hand = index;
        for (int player : {0, 1}) {
            frozen.clear();
            traverse(root, player);
        }
        frozen.clear();
    }

    Split candidate(int id) const {
        const auto& node = tree[id];
        const auto& samples = node.samples;
        Split best;
        if (samples.size() < config.window) { return best; }
        Values total{};
        double squares = 0;
        const auto actions = actions_from_mask(node.group & 255);
        for (const auto& sample : samples) {
            for (auto action : actions) {
                total[action] += sample.regret[action];
                squares += sample.regret[action] * sample.regret[action];
            }
        }
        double parent_sse = squares;
        for (auto action : actions) { parent_sse -= total[action] * total[action] / samples.size(); }
        best.rmse = std::sqrt(std::max(0.0, parent_sse) / (samples.size() * actions.size()));
        if (parent_sse <= 1e-12 || best.rmse < config.min_rmse) { return best; }
        std::vector<size_t> order(samples.size());
        std::iota(order.begin(), order.end(), 0);
        for (int feature = 0; feature < epoch7::kFeatures; ++feature) {
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return samples[a].x[feature] < samples[b].x[feature];
            });
            Values left{};
            for (size_t cut = 1; cut < order.size(); ++cut) {
                for (auto action : actions) { left[action] += samples[order[cut - 1]].regret[action]; }
                if (cut < config.min_child || order.size() - cut < config.min_child ||
                    samples[order[cut - 1]].x[feature] == samples[order[cut]].x[feature]) { continue; }
                double child_sse = squares;
                for (auto action : actions) {
                    const double right = total[action] - left[action];
                    child_sse -= left[action] * left[action] / cut + right * right / (order.size() - cut);
                }
                const double gain = (parent_sse - std::max(0.0, child_sse)) / parent_sse;
                if (gain <= best.gain) { continue; }
                best.feature = feature;
                // A <= boundary at the observed left endpoint cannot round into the right child.
                best.threshold = samples[order[cut - 1]].x[feature];
                best.gain = gain;
                best.left_mass = double(cut) / order.size();
            }
        }
        return best;
    }

    void refine(std::ostream& log) {
        for (int id : pending) {
            if (split_limit_reached()) { break; }
            ++checks;
            const Split choice = candidate(id);
            const bool accept = choice.feature >= 0 && choice.gain >= config.gain;
            log << hand << ',' << id << ',' << tree[id].group << ',' << accept << ','
                << choice.feature << ',' << choice.threshold << ',' << choice.rmse << ',' << choice.gain << '\n';
            if (!accept) { continue; }
            const auto children = partition_regret_node(table.at(leaf_key(id)), 1 - choice.left_mass);
            Node left, right;
            left.group = right.group = tree[id].group;
            left.depth = right.depth = tree[id].depth + 1;
            require(tree.size() <= size_t(std::numeric_limits<int>::max()) - 2, "tree index capacity exceeded");
            const int next = int(tree.size());
            tree[id].feature = choice.feature;
            tree[id].threshold = choice.threshold;
            tree[id].left = next;
            tree[id].right = next + 1;
            std::vector<Sample>().swap(tree[id].samples);
            tree.push_back(left);
            tree.push_back(right);
            table.erase(leaf_key(id));
            table.emplace(leaf_key(next), children[0]);
            table.emplace(leaf_key(next + 1), children[1]);
            key_bytes += 4;
            ++splits;
        }
        pending.clear();
        if (split_limit_reached()) {
            for (auto& node : tree) { std::vector<Sample>().swap(node.samples); }
        }
    }

    void save(std::ostream& out) const {
        hand_recall::save_solver(out, *this);
        write(out, splits); write(out, checks);
        std::ostringstream monitor_state; monitor_state << monitor_rng;
        write_text(out, monitor_state.str());
        write(out, uint32_t(tree.size()));
        for (const auto& node : tree) {
            write(out, node.group); write(out, node.depth); write(out, node.feature);
            write(out, node.left); write(out, node.right); write(out, node.threshold); write(out, node.seen);
            write(out, node.sample_hand); write(out, node.hand_visits);
            write(out, uint32_t(node.samples.size()));
            for (const auto& sample : node.samples) {
                write(out, sample.x); write(out, sample.regret); write(out, sample.hand);
            }
        }
        write(out, uint32_t(roots.size()));
        for (const auto& item : roots) { write(out, item.first); write(out, item.second); }
    }

    void load(std::istream& in) {
        hand_recall::load_solver(in, *this);
        splits = read<uint64_t>(in); checks = read<uint64_t>(in);
        std::istringstream monitor_state(read_text(in)); monitor_state >> monitor_rng;
        require(bool(monitor_state), "invalid monitor RNG");
        const auto count = read<uint32_t>(in);
        require(count <= uint32_t(std::numeric_limits<int>::max()) &&
            splits <= count / 2 && count - 2 * splits <= 1536 &&
            (config.max_splits < 0 || splits <= uint64_t(config.max_splits)), "invalid tree size");
        tree.clear(); roots.clear(); pending.clear(); frozen.clear();
        for (uint32_t id = 0; id < count; ++id) {
            Node node;
            node.group = read<uint32_t>(in); node.depth = read<uint32_t>(in);
            node.feature = read<int>(in); node.left = read<int>(in); node.right = read<int>(in);
            node.threshold = read<float>(in); node.seen = read<uint64_t>(in);
            node.sample_hand = read<uint64_t>(in); node.hand_visits = read<uint64_t>(in);
            const auto size = read<uint32_t>(in);
            require(size <= config.window && node.feature >= -1 && node.feature < epoch7::kFeatures &&
                std::isfinite(node.threshold) && node.group < 1536 && (node.group & 255), "invalid tree node");
            if (node.feature >= 0) {
                require(node.left > int(id) && node.right > int(id) &&
                    node.left < int(count) && node.right < int(count), "invalid child indices");
            } else { require(table.count(leaf_key(id)), "missing leaf statistics"); }
            for (uint32_t i = 0; i < size; ++i) {
                Sample sample;
                sample.x = read<Features>(in); sample.regret = read<Values>(in); sample.hand = read<uint64_t>(in);
                for (auto value : sample.x) { require(std::isfinite(value), "invalid feature"); }
                for (auto value : sample.regret) { require(std::isfinite(value), "invalid monitor regret"); }
                node.samples.push_back(sample);
            }
            tree.push_back(std::move(node));
        }
        const auto groups = read<uint32_t>(in);
        require(groups <= 1536 && count == groups + 2 * splits, "invalid group count");
        for (uint32_t i = 0; i < groups; ++i) {
            const auto group = read<uint32_t>(in);
            const auto id = read<int>(in);
            require(id >= 0 && id < int(tree.size()) && tree[id].group == group &&
                roots.emplace(group, id).second, "invalid root");
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
        Features x{};
        int id = -1;
        const auto key = solver.key(state, x, id, false);
        auto found = solver.table.find(key);
        missing += found == solver.table.end();
        const RegretNode empty{};
        const auto& node = found == solver.table.end() ? empty : found->second;
        return average ? average_strategy(node, valid_mask(state, viewer)) : current_strategy(node, valid_mask(state, viewer));
    }
};

struct Gap { double gap = 0, crossfit = 0, action_se = 0; };
Gap measure(const State& root, const std::vector<State>& particles, Target& policy, uint64_t seed) {
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
    Gap result;
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
    out << "hands,tree_current,kmeans_current,tree_average,kmeans_average,root_seed";
    for (auto p : {"tree_current", "kmeans_current", "tree_average", "kmeans_average"}) {
        out << ',' << p << "_crossfit," << p << "_action_se," << p << "_missing," << p << "_queries";
    }
    out << ",tree_leaves,kmeans_states,splits,checks,tree_nodes,kmeans_nodes,peak_rss_bytes\n";
}
void evaluate(std::ostream& out, uint64_t hand, uint64_t seed, int count, Solver& tree, Solver& baseline) {
    const auto root = epoch7::root_for(seed);
    std::mt19937_64 rng(seed ^ 0x123456789ull);
    const auto particles = epoch7::root_particles(root, count, rng);
    std::array<Target, 4> policies{{{tree, false}, {baseline, false}, {tree, true}, {baseline, true}}};
    std::array<Gap, 4> gaps{};
    out << hand;
    for (size_t i = 0; i < policies.size(); ++i) {
        gaps[i] = measure(root, particles, policies[i], seed ^ 0xabcdefull);
        out << ',' << gaps[i].gap;
    }
    out << ',' << seed;
    for (size_t i = 0; i < policies.size(); ++i) {
        out << ',' << gaps[i].crossfit << ',' << gaps[i].action_se << ',' << policies[i].missing << ',' << policies[i].queries;
    }
    out << ',' << tree.table.size() << ',' << baseline.table.size() << ',' << tree.splits << ',' << tree.checks
        << ',' << tree.nodes << ',' << baseline.nodes << ',' << epoch7::peak_rss() << '\n';
    require(bool(out), "metric write failed");
}

struct Progress { uint64_t hand = 0; std::array<uint64_t, 3> bytes{}; };
const std::array<std::string, 3> kLogs{"per_hand.csv", "panel.csv", "splits.csv"};
void save(const fs::path& path, const std::string& identity, Progress progress, const Solver& tree, const Solver& baseline) {
    std::ofstream out(path / "checkpoint.tmp", std::ios::binary | std::ios::trunc);
    write_text(out, "DYNAMIC_TREE1"); write_text(out, identity); write(out, progress.hand);
    for (const auto& name : kLogs) { write(out, fs::file_size(path / name)); }
    tree.save(out); baseline.save(out);
    out.close(); require(bool(out), "checkpoint close failed");
    require(MoveFileExW((path / "checkpoint.tmp").c_str(), (path / "checkpoint.bin").c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "checkpoint replace failed");
}
Progress restore(const fs::path& path, const std::string& identity, Solver& tree, Solver& baseline) {
    std::ifstream in(path / "checkpoint.bin", std::ios::binary);
    require(read_text(in) == "DYNAMIC_TREE1" && read_text(in) == identity, "checkpoint identity mismatch");
    Progress result;
    result.hand = read<uint64_t>(in);
    for (auto& bytes : result.bytes) { bytes = read<uint64_t>(in); }
    tree.load(in); baseline.load(in);
    require(in.peek() == std::char_traits<char>::eof(), "trailing checkpoint data");
    for (size_t i = 0; i < kLogs.size(); ++i) {
        require(fs::file_size(path / kLogs[i]) >= result.bytes[i], "log shorter than checkpoint");
    }
    for (size_t i = 0; i < kLogs.size(); ++i) { fs::resize_file(path / kLogs[i], result.bytes[i]); }
    return result;
}

void self_test(const std::string& atlas_path) {
    PowerAtlas atlas; atlas.load(atlas_path);
    Config config; config.window = 16; config.min_child = 4;
    Solver solver(atlas, true, 17, config);
    Features x{};
    const uint32_t group = 3;
    const int parent = solver.route(group, x, true);
    RegretNode original;
    original.regrets[0] = -3; original.regrets[1] = 9;
    original.strategy_sum[0] = 7; original.strategy_sum[1] = 3; original.touches = 31;
    solver.table[leaf_key(parent)] = original;
    solver.key_bytes = 4;
    for (int i = 0; i < 16; ++i) {
        x[0] = i < 8 ? 0 : 1;
        Values r{}; r[0] = i < 8 ? 4 : -4; r[1] = -r[0];
        solver.hand = i + 1; solver.observe(parent, x, r);
    }
    for (int i = 0; i < 100; ++i) { solver.observe(parent, x, solver.tree[parent].samples.back().regret); }
    require(solver.tree[parent].seen == 16 && solver.tree[parent].samples.size() == 16,
        "one hand displaced the monitor history");
    std::ostringstream log;
    solver.refine(log);
    require(solver.splits == 1 && solver.table.size() == 2, "synthetic split failed");
    const auto& left = solver.table.at(leaf_key(solver.tree[parent].left));
    const auto& right = solver.table.at(leaf_key(solver.tree[parent].right));
    for (int a = 0; a < kActionCount; ++a) {
        require(left.regrets[a] + right.regrets[a] == original.regrets[a] &&
            left.strategy_sum[a] + right.strategy_sum[a] == original.strategy_sum[a], "inheritance lost mass");
    }
    require(current_strategy(left, 3) == current_strategy(original, 3) &&
        average_strategy(right, 3) == average_strategy(original, 3), "split changed initial policy");
    require(solver.route(7, x, false) == -1, "action masks mixed");
    const int left_id = solver.tree[parent].left, right_id = solver.tree[parent].right;
    for (int i = 0; i < 16; ++i) {
        x[0] = 0; x[1] = i < 8 ? 0 : 1;
        Values r{}; r[0] = i < 8 ? 4 : -4; r[1] = -r[0];
        solver.hand = i + 17; solver.observe(left_id, x, r);
    }
    solver.refine(log);
    require(solver.splits == 2, "child could not split again");
    x[0] = 1; x[1] = -100;
    require(solver.route(group, x, false) == right_id, "local split changed sibling region");
    x[0] = 0; x[1] = 100;
    require(solver.route(group, x, false) == solver.tree[left_id].right, "unseen point did not reach a leaf");
    Solver restored(atlas, true, 0, config);
    std::stringstream stream; solver.save(stream); restored.load(stream);
    const State root = epoch7::root_for(51);
    State hidden = root;
    hidden.players[1 - root.actor].hidden.clear(); hidden.players[1 - root.actor].has_discard = false;
    hidden.simulation_deck.clear();
    require(solver.features(root, root.actor) == solver.features(hidden, root.actor), "feature hidden leakage");
    // A tiny real traversal tests recursive leaf reuse and restored RNG/buffers.
    solver.train(root, 100); restored.train(root, 100);
    solver.refine(log); restored.refine(log);
    require(solver.table.size() == restored.table.size() && solver.rng == restored.rng &&
        solver.monitor_rng == restored.monitor_rng &&
        solver.splits == restored.splits, "resume differs");
    for (const auto& item : solver.table) {
        const auto& other = restored.table.at(item.first);
        require(item.second.regrets == other.regrets && item.second.strategy_sum == other.strategy_sum &&
            item.second.touches == other.touches, "resumed statistics differ");
    }
    Target target{solver, false};
    std::mt19937_64 rng(31);
    const auto particles = epoch7::root_particles(root, 4, rng);
    const auto before = solver.table.size();
    require(measure(root, particles, target, 93).gap == epoch7::local_gap(root, particles, target, 93),
        "gap differs from existing evaluator");
    require(before == solver.table.size(), "evaluation learned new states");
    Target masked_target{solver, false};
    require(masked_target.policy(root, root.actor) == masked_target.policy(hidden, hidden.actor), "policy hidden leakage");
    Solver constant(atlas, true, 4, config);
    Features same{};
    const int constant_id = constant.route(3, same, true);
    constant.table[leaf_key(constant_id)] = RegretNode{};
    for (int i = 0; i < 16; ++i) {
        Values regret{}; regret[0] = i % 2 ? -5 : 5;
        constant.hand = i + 1; constant.observe(constant_id, same, regret);
    }
    constant.refine(log);
    require(constant.splits == 0, "identical features must not split");
    Config unlimited;
    unlimited.window = 4; unlimited.min_child = 2;
    Solver growing(atlas, true, 7, unlimited);
    Features growing_x{};
    int growing_id = growing.route(3, growing_x, true);
    growing.table[leaf_key(growing_id)] = original;
    growing.key_bytes = 4;
    for (int split = 0; split < 1025; ++split) {
        for (int sample = 0; sample < 4; ++sample) {
            growing_x[0] = float(split + (sample < 2 ? 1 : 2));
            Values regret{}; regret[0] = sample < 2 ? 4 : -4; regret[1] = -regret[0];
            growing.hand = 4 * split + sample + 1;
            growing.observe(growing_id, growing_x, regret);
        }
        growing.refine(log);
        require(growing.splits == uint64_t(split + 1), "unlimited splitting stopped");
        growing_id = growing.tree[growing_id].right;
    }
    require(growing.tree[growing_id].depth == 1025, "unlimited depth stopped");
    Solver grown_restored(atlas, true, 0, unlimited);
    std::stringstream grown_stream; growing.save(grown_stream); grown_restored.load(grown_stream);
    require(grown_restored.splits == growing.splits && grown_restored.tree.size() == growing.tree.size() &&
        grown_restored.monitor_rng == growing.monitor_rng, "deep unlimited checkpoint differs");
    for (const auto& item : growing.table) {
        const auto& other = grown_restored.table.at(item.first);
        require(item.second.regrets == other.regrets && item.second.strategy_sum == other.strategy_sum &&
            item.second.touches == other.touches, "deep restored statistics differ");
    }
    require(grown_restored.route(3, growing_x, false) == growing_id, "deep routing differs");
    grown_restored.config.max_splits = 1025;
    require(grown_restored.split_limit_reached(), "explicit finite split cap ignored");
    grown_restored.config.max_splits = 0;
    require(grown_restored.split_limit_reached(), "no-split control ignored");
    grown_restored.config.max_splits = -1;
    grown_restored.config.max_depth = 12;
    require(!grown_restored.split_limit_reached() && grown_restored.depth_limit_reached(1025),
        "finite depth cap ignored");
    grown_restored.config.max_depth = -1;
    require(!grown_restored.depth_limit_reached(1025), "unlimited depth treated as finite");
    std::cout << "SELF_TEST split, inheritance, masks, hidden invariance, resume, gap, 1025 splits/depth: PASS\n";

}

volatile LONG stopping = 0;
BOOL WINAPI stop_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) { InterlockedExchange(&stopping, 1); return TRUE; }
    return FALSE;
}
int run(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--self-test") { self_test(argv[2]); return 0; }
    require(argc == 19, "Use python -m agents.cpp_mccfr.dynamic_tree_unlimited.run --help");
    const fs::path path(argv[1]);
    const std::string identity(argv[2]);
    const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]), eval_seed = std::stoull(argv[5]);
    const int particles = std::stoi(argv[6]), panel_roots = std::stoi(argv[8]);
    const uint64_t panel_every = std::stoull(argv[7]), save_every = std::stoull(argv[9]);
    const bool resume = std::stoi(argv[10]);
    Config config;
    config.window = std::stoul(argv[11]); config.min_child = std::stoul(argv[12]);
    config.max_splits = std::stoll(argv[13]); config.max_depth = std::stoll(argv[14]);
    config.gain = std::stod(argv[15]); config.min_rmse = std::stod(argv[16]);
    const uint64_t max_states = std::stoull(argv[17]), reserve = std::stoull(argv[18]);
    require(hands > 0 && particles >= 2 && panel_roots > 0 && panel_every && save_every &&
        config.window >= 4 && config.min_child >= 2 && 2 * config.min_child <= config.window &&
        config.max_splits >= -1 && config.max_depth >= -1 &&
        config.gain > 0 && config.gain <= 1 && std::isfinite(config.min_rmse) && config.min_rmse >= 0,
        "invalid parameters");
    std::cout << "CONFIG max_splits=" << (config.max_splits < 0 ? "unlimited" : std::to_string(config.max_splits))
        << " max_depth=" << (config.max_depth < 0 ? "unlimited" : std::to_string(config.max_depth))
        << " kmeans_state_limit=" << max_states << " reserve_bytes=" << reserve
        << " gap_particles_per_hand=" << particles << std::endl;
    PowerAtlas atlas; atlas.load((path / "hand_atlas.bin").string()); atlas.set_assignment_cache_limit(100000);
    Solver tree(atlas, true, seed, config), baseline(atlas, false, seed, config);
    Progress progress;
    if (resume) { progress = restore(path, identity, tree, baseline); }
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
        logs[2] << "hands,parent,group,accepted,feature,threshold,parent_rmse,relative_gain\n";
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
        const bool state_cap = baseline.table.size() >= max_states;
        const bool low_disk = fs::space(path).available < reserve;
        const bool capped = state_cap || low_disk;
        if (hand % save_every == 0 || hand == hands || stopping || capped) { checkpoint(); }
        if (hand % 100 == 0 || hand == hands || stopping || capped) {
            logs[0].flush(); logs[2].flush();
            std::cout << "PROGRESS hands=" << hand << " target=" << hands << " tree_leaves=" << tree.table.size()
                << " splits=" << tree.splits << " checks=" << tree.checks << " kmeans_states=" << baseline.table.size()
                << " tree_nodes=" << tree.nodes << " kmeans_nodes=" << baseline.nodes
                << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0 << " wall_seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << std::endl;
        }
        if (stopping || capped) {
            std::cout << "PAUSED reason=" << (stopping ? "user_stop" : state_cap ? "kmeans_state_limit" : "low_disk")
                << "; checkpoint saved; use --resume after addressing the limit\n";
            return 2;
        }
    }
    return 0;
}
}  // namespace dynamic_tree

int main(int argc, char** argv) {
    try { return dynamic_tree::run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 1; }
}
