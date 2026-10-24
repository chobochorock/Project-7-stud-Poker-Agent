// Read-only frozen target adapter for Policy-LBR. Usage: run.py lbr --help.
// argv: MODEL OUT PAIRS PARTICLES SEED IDENTITY AVG_KIND (1 sum_j S_j, 2 newest forest), or
// --self-test MODEL IDENTITY AVG_KIND. kind 0/2: TARA current/average; 1/3: co-trained k-means.
#define TARA7_NO_MAIN
#include "solver.cpp"
#include "../abstraction_lbr/evaluate.hpp"

struct FrozenModel {
    PowerAtlas atlas;
    std::unique_ptr<tara7::Solver> tree;
    std::unique_ptr<dynamic_tree::Solver> baseline;
    uint64_t hands = 0;
    int average_kind = 1;
    FrozenModel(const std::filesystem::path& model, const std::string& identity, int avg_kind)
        : average_kind(avg_kind) {
        hand_recall::require(avg_kind == 1 || avg_kind == 2, "average kind must be 1 or 2");
        atlas.load((model.parent_path().parent_path() / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        tree = std::make_unique<tara7::Solver>(atlas, 0, tara7::Config{});
        baseline = std::make_unique<dynamic_tree::Solver>(atlas, false, 0, dynamic_tree::Config{});
        std::ifstream in(model, std::ios::binary);
        hand_recall::require(hand_recall::read_text(in) == "TARA7_MODEL1" &&
            hand_recall::read_text(in) == identity, "model identity mismatch");
        hands = epoch7::read<uint64_t>(in);
        tree->load(in);
        baseline->load(in);
        hand_recall::require(in.peek() == std::char_traits<char>::eof(), "trailing model bytes");
    }
    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        if (kind % 2 == 0) {
            tara7::Target target{*tree, kind == 0 ? 0 : average_kind};
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
        const bool test = argc == 5 && std::string(argv[1]) == "--self-test";
        hand_recall::require(argc == 8 || test, "Use run.py lbr --help");
        FrozenModel model(argv[test ? 2 : 1], argv[test ? 3 : 6], std::stoi(argv[test ? 4 : 7]));
        return abstraction_lbr::evaluate(model, test ? 2 : 6, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
