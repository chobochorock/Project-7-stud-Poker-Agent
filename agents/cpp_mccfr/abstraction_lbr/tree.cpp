// Frozen dynamic-tree checkpoint adapter; build/run through run.py.
#include "tree_engine.inc"
#include "evaluate.hpp"

struct TreeModel {
    PowerAtlas atlas;
    std::unique_ptr<dynamic_tree::Solver> tree, baseline;
    uint64_t hands = 0;
    explicit TreeModel(const std::filesystem::path& run) {
        atlas.load((run / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        tree = std::make_unique<dynamic_tree::Solver>(atlas, true, 0, dynamic_tree::Config{});
        baseline = std::make_unique<dynamic_tree::Solver>(atlas, false, 0, dynamic_tree::Config{});
        std::ifstream in(run / "checkpoint.bin", std::ios::binary);
        hand_recall::require(hand_recall::read_text(in) == "DYNAMIC_TREE1", "invalid tree checkpoint");
        const auto identity = hand_recall::read_text(in);
        std::cout << "INPUT identity=" << identity << std::endl;
        hands = epoch7::read<uint64_t>(in);
        for (int i = 0; i < 3; ++i) epoch7::read<uint64_t>(in);
        tree->load(in);
        baseline->load(in);
        hand_recall::require(in.peek() == std::char_traits<char>::eof(), "trailing checkpoint bytes");
        // Keep the original loader/parity checks, then discard training-only buffers.
        for (auto& node : tree->tree) std::vector<dynamic_tree::Sample>().swap(node.samples);
        std::cout << "LOADED hands=" << hands << " leaves=" << tree->table.size() << std::endl;
    }
    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        auto& solver = kind % 2 == 0 ? *tree : *baseline;
        epoch7::Features x{};
        int id = -1;
        const auto key = solver.key(state, x, id, false);
        const auto it = solver.table.find(key);
        if (found) *found = it != solver.table.end();
        const RegretNode empty{};
        const auto& node = it == solver.table.end() ? empty : it->second;
        const auto result = kind >= 2 ? average_strategy(node, valid_mask(state, viewer))
                                      : current_strategy(node, valid_mask(state, viewer));
        dynamic_tree::Target original{solver, kind >= 2};
        hand_recall::require(result == original.policy(state, viewer), "tree adapter policy mismatch");
        return result;
    }
};

int main(int argc, char** argv) {
    try {
        hand_recall::require(argc == 6 || (argc == 3 && std::string(argv[1]) == "--self-test"),
            "usage: exe RUN NEW_OUTPUT PAIRS PARTICLES SEED; or --self-test RUN");
        TreeModel model(argv[argc == 3 ? 2 : 1]);
        return abstraction_lbr::evaluate(model, argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
