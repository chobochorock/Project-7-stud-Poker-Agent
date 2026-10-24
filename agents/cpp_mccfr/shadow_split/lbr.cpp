// Read-only frozen target adapter for Policy-LBR. Usage: run.py lbr --help.
// kind 0/2: shadow-split current/average; 1/3: co-trained hard256 k-means current/average.
#define SHADOW7_NO_MAIN
#include "solver.cpp"
#include "../abstraction_lbr/evaluate.hpp"

struct FrozenModel {
    PowerAtlas atlas;
    std::unique_ptr<shadow7::Solver> tree;
    std::unique_ptr<dynamic_tree::Solver> baseline;
    uint64_t hands = 0;
    FrozenModel(const std::filesystem::path& path, const std::string& identity) {
        atlas.load((path / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        tree = std::make_unique<shadow7::Solver>(atlas, 0, shadow7::Config{}, shadow7::Edges{});
        baseline = std::make_unique<dynamic_tree::Solver>(atlas, false, 0, dynamic_tree::Config{});
        std::ifstream in(path / "checkpoint.bin", std::ios::binary);
        hands = shadow7::read_header(in, identity).hand;
        tree->load(in, false);  // shadows are training statistics, not part of the policy
        baseline->load(in);
        hand_recall::require(in.peek() == std::char_traits<char>::eof(), "trailing model bytes");
    }
    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        if (kind % 2 == 0) {
            shadow7::Target target{*tree, kind >= 2};
            const auto result = target.policy(state, viewer);
            if (found) { *found = target.missing == 0; }
            return result;
        }
        dynamic_tree::Target target{*baseline, kind >= 2};
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
