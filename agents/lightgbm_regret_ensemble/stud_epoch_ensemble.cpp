// Seven-stud epoch-regret experiment. Build/run via run_epoch_ensemble.py.
// Existing H4 rules; external-sampling signed MCCFR over 5th--7th street.
// stdin: TRAIN <hands>, LOAD <model path>, QUIT. Outputs are run-local.
// Every-hand metric: particle/rollout estimate of the FIRST 5th-street
// infoset's one-step deviation gap, NOT whole-game exploitability.
#define STUD_MCCFR_NO_MAIN
#include "../cpp_mccfr/stud_mccfr.cpp"
#include <filesystem>
#include <windows.h>
#include <psapi.h>

namespace epoch7 {
using Values = std::array<double, kActionCount>;
constexpr int kFeatures = 34;
using Features = std::array<float, kFeatures>;

template <typename T> void write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("binary write failed");
}
template <typename T> T read(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated binary input");
    return value;
}

void advance(State& state) {
    if (state.street == 7) { state.terminal = true; return; }
    ++state.street;
    for (int seat = 0; seat < 2; ++seat) {
        if (state.simulation_deck.empty()) throw std::runtime_error("empty deck");
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (state.street == 7) state.players[seat].hidden.push_back(card);
        else state.players[seat].shown.push_back(card);
        state.players[seat].round_bet = 0;
    }
    state.highest_bet = 0;
    state.raise_count = 0;
    if (state.players[0].all_in || state.players[1].all_in) advance(state);
    else state.actor = first_bettor(state);
}
State child(State state, Action action) {
    const int actor = state.actor;
    const auto result = apply_action(state, actor, action);
    if (result == ActionResult::RoundEnd) advance(state);
    else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
    return state;
}
Action sample(const Values& policy, uint8_t mask, std::mt19937_64& rng) {
    const double draw = std::uniform_real_distribution<double>(0, 1)(rng);
    double sum = 0;
    Action last = FOLD;
    for (Action action : actions_from_mask(mask)) {
        sum += policy[action]; last = action;
        if (draw <= sum) return action;
    }
    return last;
}
Values matching(const Values& regrets, uint8_t mask) {
    RegretNode node;
    node.regrets = regrets;
    return current_strategy(node, mask);
}

// Lossless observation key. No opponent hidden cards or future deck.
std::string exact_key(const State& state, int viewer) {
    std::string key;
    const auto number = [&](int value) {
        for (int shift = 0; shift < 32; shift += 8) {
            key.push_back(static_cast<char>((uint32_t(value) >> shift) & 255));
        }
    };
    number(viewer); number(state.street); number(state.ante);
    number(state.pot); number(state.highest_bet); number(state.raise_count);
    number(valid_mask(state, viewer));
    for (int seat = 0; seat < 2; ++seat) {
        const auto& p = state.players[seat];
        number(stack_cap(state, seat)); number(p.invested); number(p.round_bet);
        number(p.folded); number(p.all_in);
    }
    const auto cards = [&](const std::vector<Card>& hand) {
        key.push_back(static_cast<char>(hand.size()));
        for (Card card : hand) key.push_back(static_cast<char>(card_token(card)));
    };
    cards(state.players[viewer].hidden);
    cards(state.players[viewer].shown);
    cards(state.players[1 - viewer].shown);
    key.push_back(state.players[viewer].has_discard
        ? static_cast<char>(card_token(state.players[viewer].discarded)) : 0);
    number(static_cast<int>(state.history.size()));
    for (auto e : state.history) {
        key.push_back(e.street); key.push_back(e.actor); key.push_back(e.action);
    }
    return key;
}
uint32_t group_id(const State& state, int viewer) {
    return ((state.street - 5) * 2 + viewer) * 256 + valid_mask(state, viewer);
}

struct FeatureCache {
    const PowerAtlas& atlas;
    std::unordered_map<PowerObservationKey, PowerVector, PowerObservationHash> power;
    Features operator()(const State& state, int viewer) {
        const auto key = power_observation_key(state, viewer);
        auto it = power.find(key);
        if (it == power.end()) {
            if (power.size() >= 100000) power.clear();
            it = power.emplace(key, atlas.vector(state, viewer)).first;
        }
        Features f{};
        std::copy(it->second.begin(), it->second.end(), f.begin());
        int i = kPowerDimensions;
        const auto chips = [&](int amount) {
            return float(std::log1p(double(std::max(0, amount)) / state.ante));
        };
        f[i++] = chips(state.pot); f[i++] = chips(state.highest_bet);
        for (int seat : {viewer, 1 - viewer}) {
            f[i++] = chips(state.players[seat].invested);
            f[i++] = chips(state.players[seat].round_bet);
            f[i++] = chips(stack_cap(state, seat) - state.players[seat].invested);
            f[i++] = float(bets_this_street(state, seat));
            f[i++] = float(checked_this_street(state, seat));
        }
        f[i++] = float(cumulative_street_aggression(state, viewer));
        f[i++] = float(state.history.size());
        f[i++] = state.history.empty() ? 0 : float(state.history.back().action + 1);
        f[i++] = state.history.empty() ? 0 : float(state.history.back().actor == viewer);
        assert(i == kFeatures);
        return f;
    }
};

struct TreeNode { int32_t feature, left, right, leaf; double threshold; };
struct LeafStats { Values regrets{}, strategy_sum{}; uint64_t rows = 0; };
struct GroupModel {
    std::vector<std::vector<TreeNode>> trees;
    std::unordered_map<std::string, Values> buckets;
    std::vector<std::vector<LeafStats>> leaves;
    static int leaf_id(const std::vector<TreeNode>& tree, const Features& f) {
        int index = 0;
        while (tree[index].leaf < 0) {
            const auto& node = tree[index];
            index = f[node.feature] <= node.threshold ? node.left : node.right;
        }
        return tree[index].leaf;
    }
    std::string route(const Features& f) const {
        std::string signature;
        for (const auto& tree : trees) {
            const int32_t leaf = leaf_id(tree, f);
            signature.append(reinterpret_cast<const char*>(&leaf), sizeof(leaf));
        }
        return signature;
    }
};
struct EpochModel {
    uint32_t weight = 0;
    bool leafwise = false;
    std::unordered_map<uint32_t, GroupModel> groups;
    Values query(uint32_t group, const Features& f, bool* found = nullptr,
        bool strategy_mass = false) const {
        if (strategy_mass && !leafwise) throw std::runtime_error("legacy model has no strategy sums");
        if (found) *found = false;
        const auto it = groups.find(group);
        if (it == groups.end()) return {};
        if (leafwise) {
            Values result{};
            const auto& g = it->second;
            for (size_t t = 0; t < g.trees.size(); ++t) {
                const auto& stats = g.leaves[t][GroupModel::leaf_id(g.trees[t], f)];
                const auto& values = strategy_mass ? stats.strategy_sum : stats.regrets;
                for (int a = 0; a < kActionCount; ++a) result[a] += values[a];
            }
            for (double& v : result) v /= g.trees.size();
            if (found) *found = true;
            return result;
        }
        const auto bucket = it->second.buckets.find(it->second.route(f));
        if (found) *found = bucket != it->second.buckets.end();
        return bucket == it->second.buckets.end() ? Values{} : bucket->second;
    }
    static EpochModel load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        std::array<char, 8> magic{}; in.read(magic.data(), 8);
        const std::string format(magic.data(), 8);
        if (format != "EPOCH7V1" && format != "EPOCH7L1") {
            throw std::runtime_error("bad epoch model header");
        }
        EpochModel model;
        model.leafwise = format == "EPOCH7L1";
        model.weight = read<uint32_t>(in);
        if (!model.weight) throw std::runtime_error("invalid epoch weight");
        const auto count = read<uint32_t>(in);
        if (count > 1536) throw std::runtime_error("invalid group count");
        for (uint32_t g = 0; g < count; ++g) {
            const auto id = read<uint32_t>(in);
            if (id >= 1536 || !(id % 256) || model.groups.count(id)) {
                throw std::runtime_error("invalid or duplicate group id");
            }
            auto& group = model.groups[id];
            const auto trees = read<uint32_t>(in);
            if (trees > 10000 || (model.leafwise && !trees)) throw std::runtime_error("invalid tree count");
            for (uint32_t t = 0; t < trees; ++t) {
                const auto n = read<uint32_t>(in);
                if (!n || n > 1000) throw std::runtime_error("invalid tree size");
                std::vector<TreeNode> tree;
                for (uint32_t j = 0; j < n; ++j) {
                    TreeNode node;
                    node.feature = read<int32_t>(in); node.left = read<int32_t>(in);
                    node.right = read<int32_t>(in); node.leaf = read<int32_t>(in);
                    node.threshold = read<double>(in);
                    if (!std::isfinite(node.threshold)) throw std::runtime_error("invalid threshold");
                    if (node.leaf < 0 && (node.feature < 0 || node.feature >= kFeatures ||
                        node.left <= int(j) || node.right <= int(j) ||
                        node.left >= int(n) || node.right >= int(n))) {
                        throw std::runtime_error("invalid tree edge");
                    }
                    tree.push_back(node);
                }
                group.trees.push_back(std::move(tree));
            }
            if (model.leafwise) {
                for (const auto& tree : group.trees) {
                    const auto n = read<uint32_t>(in);
                    if (!n || n > tree.size()) throw std::runtime_error("invalid leaf count");
                    std::vector<bool> seen(n, false);
                    for (const auto& node : tree) if (node.leaf >= 0) {
                        if (uint32_t(node.leaf) >= n || seen[node.leaf]) throw std::runtime_error("invalid leaf id");
                        seen[node.leaf] = true;
                    }
                    if (std::count(seen.begin(), seen.end(), true) != int(n)) throw std::runtime_error("missing leaf id");
                    std::vector<LeafStats> stats(n);
                    for (auto& leaf : stats) {
                        leaf.regrets = read<Values>(in);
                        leaf.strategy_sum = read<Values>(in);
                        leaf.rows = read<uint64_t>(in);
                        if (!leaf.rows) throw std::runtime_error("empty fitted leaf");
                        for (int a = 0; a < kActionCount; ++a) {
                            if (!std::isfinite(leaf.regrets[a]) || !std::isfinite(leaf.strategy_sum[a]) ||
                                leaf.strategy_sum[a] < 0 ||
                                (!(id & (1u << a)) && (leaf.regrets[a] != 0 || leaf.strategy_sum[a] != 0))) {
                                throw std::runtime_error("invalid leaf payload");
                            }
                        }
                    }
                    group.leaves.push_back(std::move(stats));
                }
                continue;
            }
            const auto buckets = read<uint32_t>(in);
            if (buckets > 10000000) throw std::runtime_error("invalid bucket count");
            for (uint32_t b = 0; b < buckets; ++b) {
                std::string key(trees * sizeof(int32_t), '\0');
                in.read(key.data(), key.size());
                const auto values = read<Values>(in);
                for (double v : values) if (!std::isfinite(v)) throw std::runtime_error("bad regret");
                if (!group.buckets.emplace(std::move(key), values).second) {
                    throw std::runtime_error("duplicate bucket signature");
                }
            }
        }
        if (in.peek() != EOF) throw std::runtime_error("trailing model bytes");
        return model;
    }
};
struct Entry {
    uint32_t group = 0;
    Features features{};
    Values regrets{}, strategy_sum{}, past{};
    uint64_t visits = 0;
};

struct Ensemble {
    FeatureCache& features;
    std::unordered_map<std::string, Entry> entries;
    std::vector<EpochModel> models;
    std::mt19937_64 rng;
    uint64_t node_visits = 0;
    double sampled_gap = 0;
    uint64_t sampled_decisions = 0;
    Values released_mass{}, released_strategy_mass{};
    std::vector<Entry> released_probes;
    bool released = false;
    struct Segment { uint64_t offset = 0, count = 0; };
    std::unordered_map<std::string, Segment> segments;
    std::fstream spool;
    std::string spool_path;
    std::array<std::string, 2> active_shards;
    std::unordered_map<std::string, std::unordered_map<std::string, Entry>> read_cache;
    uint64_t stored_entries = 0;
    Ensemble(FeatureCache& f, uint64_t seed, const std::string& path = "")
        : features(f), rng(seed), spool_path(path) {
        if (!path.empty()) {
            spool.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
            if (!spool) throw std::runtime_error("cannot open residual spool");
        }
    }
    // The initial private/public observation is retained by every descendant
    // infoset. Sharding on it is lossless; repeated observations reload D_k.
    static std::string shard_key(const State& state, int viewer) {
        State root = state;
        root.street = 5; root.history.clear(); root.raise_count = 0;
        root.highest_bet = 0; root.pot = 2 * root.ante; root.terminal = false;
        for (auto& p : root.players) {
            p.invested = root.ante; p.round_bet = 0; p.folded = false; p.all_in = false;
            p.hidden.resize(2); p.shown.resize(3);
        }
        return exact_key(root, viewer);
    }
    template <class Callback> void visit_segment(const Segment& segment, Callback callback) {
        spool.flush(); spool.clear(); spool.seekg(segment.offset);
        for (uint64_t i = 0; i < segment.count; ++i) {
            const auto length = read<uint32_t>(spool);
            if (length > 4096) throw std::runtime_error("bad spool key length");
            std::string key(length, '\0'); spool.read(key.data(), length);
            Entry entry = read<Entry>(spool);
            callback(std::move(key), entry);
        }
    }
    void begin_hand(const State& root) {
        if (!spool.is_open()) return;
        entries.clear(); read_cache.clear();
        for (int seat = 0; seat < 2; ++seat) {
            active_shards[seat] = shard_key(root, seat);
            const auto found = segments.find(active_shards[seat]);
            if (found != segments.end()) {
                visit_segment(found->second, [&](std::string key, const Entry& e) {
                    entries.emplace(std::move(key), e);
                });
            }
        }
    }
    void commit_hand() {
        if (!spool.is_open()) return;
        read_cache.clear();
        for (int seat = 0; seat < 2; ++seat) {
            spool.clear(); spool.seekp(0, std::ios::end);
            Segment next{uint64_t(spool.tellp()), 0};
            for (const auto& [key, e] : entries) {
                if ((e.group / 256) % 2 != uint32_t(seat)) continue;
                write(spool, uint32_t(key.size())); spool.write(key.data(), key.size());
                write(spool, e); ++next.count;
            }
            auto& old = segments[active_shards[seat]];
            stored_entries = stored_entries - old.count + next.count;
            old = next;
        }
        spool.flush();
    }
    const Entry* find_entry(const State& state, int viewer, const std::string& key) {
        const auto it = entries.find(key);
        if (it != entries.end()) return &it->second;
        if (!spool.is_open()) return nullptr;
        const auto shard = shard_key(state, viewer);
        if (shard == active_shards[viewer]) return nullptr;
        const auto segment = segments.find(shard);
        if (segment == segments.end()) return nullptr;
        auto cached = read_cache.find(shard);
        if (cached == read_cache.end()) {
            if (read_cache.size() >= 8) read_cache.clear();
            auto& table = read_cache[shard];
            visit_segment(segment->second, [&](std::string k, const Entry& e) { table.emplace(std::move(k), e); });
            cached = read_cache.find(shard);
        }
        const auto found = cached->second.find(key);
        return found == cached->second.end() ? nullptr : &found->second;
    }
    uint64_t table_size() const { return spool.is_open() ? stored_entries : entries.size(); }
    Values past(uint32_t group, const Features& f, bool* found = nullptr) const {
        Values result{};
        if (found) *found = false;
        for (const auto& model : models) {
            bool matched = false;
            const auto values = model.query(group, f, &matched);
            if (found) *found = *found || matched;
            for (int a = 0; a < kActionCount; ++a) result[a] += model.weight * values[a];
        }
        return result;
    }
    Values policy(const State& state, int viewer, bool* found = nullptr) {
        const auto entry = find_entry(state, viewer, exact_key(state, viewer));
        if (found) *found = entry != nullptr;
        Values total{};
        if (entry) {
            total = entry->past;
            for (int a = 0; a < kActionCount; ++a) {
                total[a] += (models.size() + 1) * entry->regrets[a];
            }
        } else if (!models.empty()) {
            total = past(group_id(state, viewer), features(state, viewer), found);
        }
        return matching(total, valid_mask(state, viewer));
    }
    Values average_policy(const State& state, int viewer, bool* found = nullptr) {
        Values total{};
        if (found) *found = false;
        if (!models.empty()) {
            const auto f = features(state, viewer);
            for (const auto& model : models) {
                bool matched = false;
                const auto values = model.query(group_id(state, viewer), f, &matched, true);
                if (found) *found = *found || matched;
                for (int a = 0; a < kActionCount; ++a) total[a] += model.weight * values[a];
            }
        }
        if (const auto* e = find_entry(state, viewer, exact_key(state, viewer))) {
            if (found) *found = true;
            for (int a = 0; a < kActionCount; ++a) total[a] += (models.size() + 1) * e->strategy_sum[a];
        }
        return matching(total, valid_mask(state, viewer));
    }
    double traverse(State state, int traverser) {
        ++node_visits;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const auto key = exact_key(state, actor);
        auto [it, inserted] = entries.try_emplace(key);
        if (inserted) {
            if (entries.size() > 40000000) throw std::runtime_error("epoch exceeds 40M infoset safety limit");
            if (entries.size() % 100000 == 0) {
                MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
                if (GlobalMemoryStatusEx(&memory) && memory.ullAvailPhys < 2ull * 1024 * 1024 * 1024) {
                    throw std::runtime_error("less than 2 GiB physical RAM available");
                }
            }
            it->second.group = group_id(state, actor);
            it->second.features = features(state, actor);
            it->second.past = past(it->second.group, it->second.features);
        }
        ++it->second.visits;
        const auto strategy = policy(state, actor);
        if (actor != traverser) {
            for (int a = 0; a < kActionCount; ++a) it->second.strategy_sum[a] += strategy[a];
            return traverse(child(std::move(state), sample(strategy, mask, rng)), traverser);
        }
        Values values{};
        double value = 0, best = -std::numeric_limits<double>::infinity();
        for (Action action : actions_from_mask(mask)) {
            values[action] = traverse(child(state, action), traverser);
            value += strategy[action] * values[action];
            best = std::max(best, values[action]);
        }
        // Recursive inserts can rehash the table; retrieve the entry again.
        auto& updated = entries.at(key);
        for (Action action : actions_from_mask(mask)) {
            updated.regrets[action] += values[action] - value;
        }
        sampled_gap += best - value;
        ++sampled_decisions;
        return value;
    }
    void dump(const std::string& path) {
        std::ofstream out(path, std::ios::binary);
        out.write("ROWS7V02", 8);
        write(out, table_size());
        released_mass = {}; released_strategy_mass = {}; released_probes.clear();
        std::array<int, 1536> probe_counts{};
        const auto emit = [&](const std::string&, const Entry& entry) {
            write(out, entry.group); write(out, entry.features);
            write(out, entry.regrets);
            Values teacher = entry.strategy_sum;
            double mass = std::accumulate(teacher.begin(), teacher.end(), 0.0);
            if (mass > 0) {
                for (double& p : teacher) p /= mass;
            } else {
                Values total = entry.past;
                for (int a = 0; a < kActionCount; ++a) total[a] += (models.size() + 1) * entry.regrets[a];
                teacher = matching(total, entry.group % 256);
            }
            write(out, teacher);
            write(out, entry.visits);
            write(out, entry.strategy_sum);
            for (int a = 0; a < kActionCount; ++a) {
                released_mass[a] += entry.regrets[a];
                released_strategy_mass[a] += entry.strategy_sum[a];
            }
            if (probe_counts[entry.group]++ < 32) released_probes.push_back(entry);
        };
        if (spool.is_open()) {
            for (const auto& [shard, segment] : segments) visit_segment(segment, emit);
        } else {
            for (const auto& [key, entry] : entries) emit(key, entry);
        }
    }
    void load_epoch(const std::string& path) {
        auto model = EpochModel::load(path);
        if (model.weight != models.size() + 1) throw std::runtime_error("epoch order mismatch");
        Values before = released ? released_mass : Values{}, after{};
        Values strategy_before = released ? released_strategy_mass : Values{}, strategy_after{};
        for (const auto& [key, e] : entries) {
            bool found = false;
            model.query(e.group, e.features, &found);
            if (!found) {
                throw std::runtime_error("training row missing from new bucket table");
            }
            for (int a = 0; a < kActionCount; ++a) {
                before[a] += e.regrets[a];
                strategy_before[a] += e.strategy_sum[a];
            }
        }
        for (const auto& [id, group] : model.groups) {
            if (model.leafwise) {
                for (const auto& tree : group.leaves) for (const auto& leaf : tree) {
                    for (int a = 0; a < kActionCount; ++a) {
                        after[a] += leaf.regrets[a] / group.trees.size();
                        strategy_after[a] += leaf.strategy_sum[a] / group.trees.size();
                    }
                }
            }
            for (const auto& [key, values] : group.buckets) {
                for (int a = 0; a < kActionCount; ++a) after[a] += values[a];
            }
        }
        for (const auto& e : released_probes) {
            bool found = false;
            model.query(e.group, e.features, &found);
            if (!found) {
                throw std::runtime_error("released training probe missing from model");
            }
        }
        for (int a = 0; a < kActionCount; ++a) {
            if (std::abs(before[a] - after[a]) > 1e-6 * (1 + std::abs(before[a]))) {
                throw std::runtime_error("new-regret mass not conserved");
            }
            if (model.leafwise && std::abs(strategy_before[a] - strategy_after[a]) >
                1e-6 * (1 + std::abs(strategy_before[a]))) {
                throw std::runtime_error("strategy mass not conserved");
            }
        }
        models.push_back(std::move(model)); entries.clear();
        released = false; released_mass = {}; released_strategy_mass = {}; released_probes.clear();
    }
    void release_epoch() {
        if (released) throw std::runtime_error("epoch already released");
        if (spool.is_open()) {
            spool.close();
            spool.open(spool_path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
            if (!spool) throw std::runtime_error("cannot reset spool");
        }
        segments.clear(); read_cache.clear(); stored_entries = 0;
        active_shards = {};
        entries.clear(); entries.rehash(0); released = true;
    }
};
struct AverageEnsemble {
    Ensemble& solver;
    Values policy(const State& s, int viewer, bool* found = nullptr) {
        return solver.average_policy(s, viewer, found);
    }
};
struct CurrentBaseline {
    MCCFR& solver;
    Values policy(const State& s, int viewer, bool* found = nullptr) {
        return solver.instantaneous_policy(s, viewer, found);
    }
};

// Evaluators must name a matching hand checkpoint and complete epoch prefix.
void load_frozen_checkpoint(const std::filesystem::path& run, Ensemble& ensemble,
    MCCFR& baseline, uint64_t hands, uint64_t epoch_hands) {
    if (!hands || !epoch_hands || hands % epoch_hands) {
        throw std::runtime_error("HANDS must be a positive multiple of EPOCH_HANDS");
    }
    const auto baseline_path = run / ("hard256_" + std::to_string(hands) + ".bin");
    if (!std::filesystem::is_regular_file(baseline_path)) {
        throw std::runtime_error("missing baseline checkpoint: " + baseline_path.string());
    }
    ensemble.models.clear();
    for (uint64_t i = 1; i <= hands / epoch_hands; ++i) {
        std::ostringstream folder;
        folder << "epoch_" << std::setw(3) << std::setfill('0') << i;
        ensemble.models.push_back(EpochModel::load((run / folder.str() / "model.bin").string()));
        if (ensemble.models.back().weight != i) throw std::runtime_error("wrong epoch weight");
    }
    baseline.load(baseline_path.string());
    std::cout << "CHECKPOINT training_hands=" << hands << " epoch_models=" << ensemble.models.size()
        << " baseline=" << baseline_path.filename().string() << '\n' << std::flush;
}

// Conditional H4 prior at a 5th-street root with no betting history.
// Rejection integrates the fixed discard/reveal rule, not the actual hidden hand.
std::vector<State> root_particles(const State& root, int count, std::mt19937_64& rng) {
    if (root.street != 5 || !root.history.empty()) throw std::runtime_error("root-only posterior");
    const int viewer = root.actor, opponent = 1 - viewer;
    uint64_t known = card_mask(all_cards(root.players[viewer])) |
        card_mask(root.players[opponent].shown);
    if (root.players[viewer].has_discard) known |= card_mask({root.players[viewer].discarded});
    std::vector<Card> unseen;
    for (Card c : fresh_deck()) if (!(card_mask({c}) & known)) unseen.push_back(c);
    std::vector<State> result;
    for (int tries = 0; int(result.size()) < count; ++tries) {
        if (tries > count * 10000) throw std::runtime_error("H4 particle rejection exhausted");
        std::shuffle(unseen.begin(), unseen.end(), rng);
        std::vector<Card> initial{unseen[0], unseen[1], unseen[2], root.players[opponent].shown[0]};
        std::shuffle(initial.begin(), initial.end(), rng);
        const auto [discard, reveal] = discard_reveal(initial);
        if (!(initial[reveal] == root.players[opponent].shown[0])) continue;
        State particle = root;
        auto& p = particle.players[opponent];
        p.hidden.clear(); p.discarded = initial[discard]; p.has_discard = true;
        for (int i = 0; i < 4; ++i) if (i != discard && i != reveal) p.hidden.push_back(initial[i]);
        particle.simulation_deck.assign(unseen.begin() + 3, unseen.end());
        result.push_back(std::move(particle));
    }
    return result;
}
template <class Policy> double rollout(State state, int viewer, Policy& policy, uint64_t seed) {
    std::mt19937_64 rng(seed);
    while (!state.terminal) {
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const auto strategy = policy.policy(state, actor);
        state = child(std::move(state), sample(strategy, mask, rng));
    }
    return terminal_net_search(state, viewer);
}
template <class Policy> double local_gap(
    const State& root, const std::vector<State>& particles, Policy& policy, uint64_t seed) {
    const int viewer = root.actor;
    const uint8_t mask = valid_mask(root, viewer);
    const auto strategy = policy.policy(root, viewer);
    Values q{};
    for (size_t i = 0; i < particles.size(); ++i) {
        for (Action action : actions_from_mask(mask)) {
            q[action] += rollout(child(particles[i], action), viewer, policy,
                seed + i * 0x9e3779b97f4a7c15ull) / particles.size();
        }
    }
    double value = 0, best = -std::numeric_limits<double>::infinity();
    for (Action action : actions_from_mask(mask)) {
        value += strategy[action] * q[action]; best = std::max(best, q[action]);
    }
    // terminal_net_search already returns net chips divided by ante.
    return std::max(0.0, best - value);
}
State root_for(uint64_t seed) {
    auto deck = fresh_deck(); std::mt19937_64 rng(seed);
    std::shuffle(deck.begin(), deck.end(), rng);
    return sample_fifth_street_root(deck, 1000, 1000);
}
uint64_t peak_rss() {
    PROCESS_MEMORY_COUNTERS info{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) return 0;
    return info.PeakWorkingSetSize;
}

void self_test(const std::string& atlas_path) {
    // Training routes (0,0) and (1,1) need not include the query route (0,1).
    EpochModel leaf_model; leaf_model.weight = 1; leaf_model.leafwise = true;
    auto& group = leaf_model.groups[3];
    group.trees = {{{0, 1, 2, -1, 0.5}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 1, 0}},
                   {{1, 1, 2, -1, 0.5}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 1, 0}}};
    LeafStats low, high;
    low.regrets[0] = -2; low.regrets[1] = 4; low.strategy_sum[0] = 1; low.rows = 1;
    high.regrets[0] = 6; high.regrets[1] = -8; high.strategy_sum[1] = 3; high.rows = 1;
    group.leaves = {{low, high}, {low, high}};
    Features unseen{}; unseen[1] = 1;
    bool found = false;
    const auto mixed = leaf_model.query(3, unseen, &found);
    assert(found && mixed[0] == 2 && mixed[1] == -2);
    const auto mass = leaf_model.query(3, unseen, nullptr, true);
    assert(mass[0] == 0.5 && mass[1] == 1.5);
    const auto average = matching(mass, 3);
    assert(average[0] == 0.25 && average[1] == 0.75);
    const auto original_trees = group.trees;
    const auto original_leaves = group.leaves;
    group.trees.insert(group.trees.end(), original_trees.begin(), original_trees.end());
    group.leaves.insert(group.leaves.end(), original_leaves.begin(), original_leaves.end());
    assert(leaf_model.query(3, unseen) == mixed);
    leaf_model.query(7, unseen, &found); assert(!found);
    EpochModel legacy; legacy.groups[3].buckets[""] = mixed;
    assert(legacy.query(3, unseen, &found) == mixed && found);
    bool rejected = false;
    try { legacy.query(3, unseen, nullptr, true); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    PowerAtlas atlas; atlas.load(atlas_path);
    FeatureCache features{atlas}; Ensemble agent(features, 7);
    State root = root_for(7), altered = root;
    altered.players[1 - root.actor].hidden.clear();
    altered.players[1 - root.actor].has_discard = false;
    altered.simulation_deck.clear();
    assert(exact_key(root, root.actor) == exact_key(altered, root.actor));
    assert(features(root, root.actor) == features(altered, root.actor));
    assert(agent.policy(root, root.actor) == agent.policy(altered, root.actor));
    for (int p = 0; p < 2; ++p) agent.traverse(root, p);
    bool negative = false;
    for (const auto& [key, e] : agent.entries) {
        for (double regret : e.regrets) negative |= regret < 0;
    }
    assert(negative);
    std::mt19937_64 rng(13);
    const auto particles = root_particles(root, 16, rng);
    for (const auto& p : particles) {
        assert(exact_key(root, root.actor) == exact_key(p, root.actor));
        auto all = all_cards(p.players[0]);
        const auto other = all_cards(p.players[1]); all.insert(all.end(), other.begin(), other.end());
        all.push_back(p.players[0].discarded); all.push_back(p.players[1].discarded);
        assert(__builtin_popcountll(card_mask(all)) == int(all.size()));
        State terminal = p;
        while (!terminal.terminal) {
            const auto mask = valid_mask(terminal, terminal.actor);
            terminal = child(terminal, mask & (1u << CHECK) ? CHECK : CALL);
        }
        assert(std::abs(terminal_net_search(terminal, 0) + terminal_net_search(terminal, 1)) < 1e-9);
    }
    const double gap = local_gap(root, particles, agent, 17);
    assert(std::isfinite(gap) && gap >= 0);
    State folded = root;
    folded.players[root.actor].folded = true;
    assert(terminal_net_search(folded, root.actor) == -1.0);
    Values regrets{}; regrets[CHECK] = 2; regrets[FOLD] = 1;
    const auto policy = matching(regrets, (1u << CHECK) | (1u << FOLD));
    assert(std::abs(policy[CHECK] - 2.0 / 3) < 1e-12);
    std::cout << "{\"self_test\":\"ok\",\"entries\":" << agent.entries.size() << "}\n";
}

void spool_test(const std::string& atlas_path, const std::string& path) {
    PowerAtlas atlas; atlas.load(atlas_path);
    FeatureCache features{atlas};
    Ensemble memory(features, 7), disk(features, 7, path);
    // Include repeated and alternating private-root observations.
    for (int hand = 0; hand < 12; ++hand) {
        const auto root = root_for(7 + hand % 3);
        disk.begin_hand(root);
        for (int p = 0; p < 2; ++p) {
            const double a = memory.traverse(root, p), b = disk.traverse(root, p);
            assert(a == b);
        }
        disk.commit_hand();
        assert(memory.node_visits == disk.node_visits);
        assert(memory.entries.size() == disk.table_size());
        for (const auto& [key, e] : disk.entries) {
            const auto& reference = memory.entries.at(key);
            assert(e.regrets == reference.regrets);
            assert(e.strategy_sum == reference.strategy_sum);
        }
    }
    memory.dump(path + ".memory.rows"); disk.dump(path + ".disk.rows");
    assert(memory.released_mass == disk.released_mass ||
        std::equal(memory.released_mass.begin(), memory.released_mass.end(), disk.released_mass.begin(),
            [](double a, double b) { return std::abs(a - b) < 1e-7 * (1 + std::abs(a)); }));
    std::cout << "{\"spool_test\":\"ok\",\"entries\":" << disk.table_size() << "}\n";
}
} // namespace epoch7

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--self-test") {
            epoch7::self_test(argv[2]); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--engine-self-test") {
            self_test(); return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--spool-test") {
            epoch7::spool_test(argv[2], argv[3]); return 0;
        }
        if (argc != 5) throw std::runtime_error("usage: exe RUN_DIR ATLAS SEED PARTICLES");
        using namespace epoch7;
        const std::filesystem::path dir(argv[1]);
        std::filesystem::create_directories(dir);
        const uint64_t seed = std::stoull(argv[3]);
        const int particle_count = std::stoi(argv[4]);
        if (particle_count < 1) throw std::runtime_error("particles must be positive");
        PowerAtlas atlas; atlas.load(argv[2]); atlas.set_assignment_cache_limit(100000);
        for (int s = 5; s <= 7; ++s) if (atlas.clusters(s) != 256) throw std::runtime_error("expected hard256");
        MCCFR baseline(false, seed, 5, &atlas);
        baseline.use_cumulative_street_summary(true);
        CurrentBaseline current{baseline};
        FeatureCache features{atlas};
        Ensemble ensemble(features, seed, (dir / "residual_spool.bin").string());
        std::ofstream csv(dir / "per_hand.csv");
        std::ofstream audit(dir / "boundary_audit.csv");
        audit << "hand,epoch_models,root,ensemble_gap_ante,hard256_gap_ante\n";
        audit << std::setprecision(12);
        csv << "hand,ensemble_root_local_gap_ante,hard256_root_local_gap_ante,ensemble_training_nodes,hard256_training_nodes,epoch_infosets,hard256_buckets,peak_cpp_rss_bytes,train_seconds,eval_seconds\n";
        csv << std::setprecision(12);
        uint64_t hand = 0;
        double train_seconds = 0, eval_seconds = 0;
        std::string line;
        std::cout << "READY\n" << std::flush;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            if (line == "RELEASE") {
                ensemble.release_epoch();
                std::cout << "RELEASED\n" << std::flush; continue;
            }
            if (line == "AUDIT") {
                double egap_sum = 0, bgap_sum = 0;
                for (int r = 0; r < 128; ++r) {
                    const uint64_t eval_seed = seed ^ (uint64_t(r + 1) * 0x6a09e667f3bcc909ull);
                    const State evaluation = root_for(eval_seed);
                    std::mt19937_64 rng(eval_seed ^ 0xfedcba987ull);
                    const auto particles = root_particles(evaluation, 128, rng);
                    const double egap = local_gap(evaluation, particles, ensemble, eval_seed);
                    const double bgap = local_gap(evaluation, particles, current, eval_seed);
                    audit << hand << ',' << ensemble.models.size() << ',' << r << ',' << egap << ',' << bgap << '\n';
                    egap_sum += egap; bgap_sum += bgap;
                }
                audit.flush();
                std::cout << "AUDITED hand=" << hand << " ensemble_mean_gap_ante=" << egap_sum / 128
                    << " hard256_mean_gap_ante=" << bgap_sum / 128
                    << '\n' << std::flush;
                continue;
            }
            if (line.rfind("LOAD ", 0) == 0) {
                ensemble.load_epoch(line.substr(5));
                std::cout << "LOADED\n" << std::flush; continue;
            }
            if (line.rfind("TRAIN ", 0) != 0) throw std::runtime_error("unknown command");
            if (ensemble.released) throw std::runtime_error("LOAD required before further training");
            const auto count = std::stoull(line.substr(6));
            for (uint64_t i = 0; i < count; ++i) {
                ++hand;
                const State training = root_for(seed ^ (hand * 0xa0761d6478bd642full));
                auto start = std::chrono::steady_clock::now();
                ensemble.begin_hand(training);
                for (int player = 0; player < 2; ++player) {
                    ensemble.traverse(training, player); baseline.train_root(training, player);
                }
                ensemble.commit_hand();
                train_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                start = std::chrono::steady_clock::now();
                const uint64_t eval_seed = seed ^ (hand * 0xe7037ed1a0b428dbull);
                const State evaluation = root_for(eval_seed);
                std::mt19937_64 rng(eval_seed ^ 0x123456789ull);
                const auto particles = root_particles(evaluation, particle_count, rng);
                const double egap = local_gap(evaluation, particles, ensemble, eval_seed);
                const double bgap = local_gap(evaluation, particles, current, eval_seed);
                eval_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (!std::isfinite(egap) || !std::isfinite(bgap)) throw std::runtime_error("nonfinite gap");
                csv << hand << ',' << egap << ',' << bgap << ',' << ensemble.node_visits << ','
                    << baseline.node_visits() << ',' << ensemble.table_size() << ',' << baseline.bucket_count()
                    << ',' << peak_rss() << ',' << train_seconds << ',' << eval_seconds << '\n';
                if (hand % 100 == 0) {
                    csv.flush();
                    std::cout << "PROGRESS hand=" << hand << " ensemble_gap_ante=" << egap
                        << " hard256_gap_ante=" << bgap << " epoch_infosets=" << ensemble.table_size()
                        << " cpp_train_eval_seconds=" << train_seconds + eval_seconds << '\n' << std::flush;
                }
            }
            csv.flush();
            ensemble.dump((dir / ("rows_" + std::to_string(hand) + ".bin")).string());
            baseline.save((dir / ("hard256_" + std::to_string(hand) + ".bin")).string());
            std::cout << "EPOCH hand=" << hand << '\n' << std::flush;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
