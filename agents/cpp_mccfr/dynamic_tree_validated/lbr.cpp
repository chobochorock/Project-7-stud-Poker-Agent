// Read-only frozen target adapter. Usage: run.py lbr --help.
#define VALIDATED_NO_MAIN
#include "solver.cpp"
#include "../abstraction_lbr/evaluate.hpp"

struct FrozenModel {
    PowerAtlas atlas;
    std::unique_ptr<dynamic_tree::Solver> tree, baseline;
    uint64_t hands = 0;
    FrozenModel(const std::filesystem::path& path, const std::string& identity) {
        atlas.load((path / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        dynamic_tree::Config config;
        config.window = 100000;
        tree = std::make_unique<dynamic_tree::Solver>(atlas, true, 0, config);
        baseline = std::make_unique<dynamic_tree::Solver>(atlas, false, 0, config);
        std::ifstream in(path / "checkpoint.bin", std::ios::binary);
        hand_recall::require(hand_recall::read_text(in) == "DYNAMIC_TREE_VALIDATED1" &&
            hand_recall::read_text(in) == identity, "checkpoint identity mismatch");
        hands = epoch7::read<uint64_t>(in);
        for (int i = 0; i < 3; ++i) { epoch7::read<uint64_t>(in); }
        tree->load(in); baseline->load(in);
        hand_recall::require(in.peek() == std::char_traits<char>::eof(), "trailing model bytes");
        for (auto& node : tree->tree) { std::vector<dynamic_tree::Sample>().swap(node.samples); }
    }
    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        auto& solver = kind % 2 == 0 ? *tree : *baseline;
        dynamic_tree::Target target{solver, kind >= 2};
        const auto result = target.policy(state, viewer);
        if (found) { *found = target.missing == 0; }
        return result;
    }
};

int main(int argc, char** argv) {
    try {
        hand_recall::require(argc == 7 || (argc == 4 && std::string(argv[1]) == "--self-test"),
            "Use run.py lbr --help");
        FrozenModel model(argv[argc == 4 ? 2 : 1], argv[argc - 1]);
        return abstraction_lbr::evaluate(model, argc - 1, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
