// Frozen-path recall diagnostic, not training or an exploitability measurement.
// Build from project root:
// g++ -O3 -std=c++17 agents/cpp_mccfr/dynamic_tree_validated/audit_recall.cpp -o agents/cpp_mccfr/bin/dt_audit_recall.exe -lpsapi
// Usage: dt_audit_recall.exe RUN IDENTITY HANDS SEED > RUN/recall_audit.json
#define VALIDATED_NO_MAIN
#include "solver.cpp"

int main(int argc, char** argv) {
    try {
        using namespace dynamic_tree;
        require(argc == 5, "Expected RUN IDENTITY HANDS SEED");
        const fs::path path(argv[1]);
        const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]);
        require(hands > 0, "Positive hand count required");
        PowerAtlas atlas;
        atlas.load((path / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        Config config;
        config.window = 100000;
        Solver solver(atlas, true, 0, config);
        std::ifstream in(path / "checkpoint.bin", std::ios::binary);
        require(read_text(in) == "DYNAMIC_TREE_VALIDATED1" && read_text(in) == argv[2],
            "Checkpoint identity mismatch");
        const auto trained_hands = read<uint64_t>(in);
        for (int i = 0; i < 3; ++i) { read<uint64_t>(in); }
        solver.load(in);
        for (auto& node : solver.tree) { std::vector<Sample>().swap(node.samples); }
        const auto before = solver.table.size();
        std::cout << "{\"training_hands\":" << trained_hands
            << ",\"sample_hands_per_policy\":" << hands << ",\"seed\":" << seed
            << ",\"policies\":{";
        for (int mode = 0; mode < 3; ++mode) {
            uint64_t repeated_hands = 0, repeated_visits = 0, decisions = 0, missing = 0;
            uint64_t root_repeated_hands = 0;
            std::set<uint32_t> groups;
            std::set<std::pair<uint32_t, size_t>> depth_groups;
            Target policy{solver, mode == 2};
            for (uint64_t hand = 1; hand <= hands; ++hand) {
                State state = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
                std::mt19937_64 rng(seed ^ (hand * 0xd1b54a32d192ed03ull));
                std::set<int> visited;
                std::set<uint32_t> root_visited;
                std::set<std::pair<uint32_t, size_t>> depth_visited;
                bool repeated = false, root_repeated = false;
                while (!state.terminal) {
                    ++decisions;
                    Features features{};
                    int leaf = -1;
                    solver.key(state, features, leaf, false);
                    missing += leaf < 0;
                    if (leaf >= 0 && !visited.insert(leaf).second) {
                        repeated = true;
                        ++repeated_visits;
                    }
                    const auto group = epoch7::group_id(state, state.actor);
                    root_repeated |= !root_visited.insert(group).second;
                    groups.insert(group);
                    const auto depth_group = std::make_pair(group, state.history.size());
                    depth_groups.insert(depth_group);
                    require(depth_visited.insert(depth_group).second,
                        "Public depth did not distinguish sequential decisions");
                    Values strategy = mode ? policy.policy(state, state.actor)
                        : epoch7::matching(Values{}, valid_mask(state, state.actor));
                    const auto previous_depth = state.history.size();
                    const auto action = epoch7::sample(strategy, valid_mask(state, state.actor), rng);
                    state = epoch7::child(std::move(state), action);
                    require(state.history.size() > previous_depth, "History did not grow");
                }
                repeated_hands += repeated;
                root_repeated_hands += root_repeated;
            }
            if (mode) { std::cout << ','; }
            std::cout << '"' << (mode == 0 ? "uniform" : mode == 1 ? "current" : "average")
                << "\":{\"decisions\":" << decisions << ",\"repeated_leaf_hands\":" << repeated_hands
                << ",\"repeated_leaf_visits\":" << repeated_visits
                << ",\"unsplit_root_repeated_hands\":" << root_repeated_hands
                << ",\"missing_leaf_visits\":" << missing << ",\"root_groups\":" << groups.size()
                << ",\"root_plus_depth_groups\":" << depth_groups.size() << '}';
        }
        require(solver.table.size() == before, "Audit changed model table");
        std::cout << "},\"depth_repeated_visits\":0,\"limitations\":\"Sampled frozen paths; not MCCFR traversal frequencies. Adding depth alone is not perfect recall.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
