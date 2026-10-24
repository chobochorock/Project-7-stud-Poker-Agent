// Audit the frozen solver's split labels, sample window and inheritance.
// Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.audit_split
#define VALIDATED_NO_MAIN
#include "solver.cpp"

int main(int argc, char** argv) {
    using namespace dynamic_tree;
    try {
        require(argc == 2, "Expected the frozen hand atlas path");
        PowerAtlas atlas;
        atlas.load(argv[1]);
        Config config;
        config.split_rule = 1;
        config.gain = 0.05;
        Solver solver(atlas, true, 222, config);
        Features x{};
        const int parent = solver.route(3, x, true);
        RegretNode original;
        original.regrets[0] = -120;
        original.regrets[1] = 360;
        original.strategy_sum[0] = 700;
        original.strategy_sum[1] = 300;
        original.touches = 400;
        solver.table[leaf_key(parent)] = original;

        // The first 256 leaf hands must not enter the latest-window fit.
        for (uint64_t hand = 1; hand <= 512; ++hand) {
            const bool recent = hand > 256;
            const bool left = (hand - 1) % 256 < 192;
            x[0] = recent ? (left ? 0 : 1) : 99;
            Values regret{};
            regret[0] = recent ? (left ? 8 : 0) : 1000;
            regret[1] = recent ? (left ? 0 : 8) : 1000;
            solver.hand = hand;
            solver.observe(parent, x, regret);
        }
        const auto& monitor = solver.tree[parent];
        require(monitor.seen == 512 && monitor.samples.size() == 256,
            "Monitor did not retain a fixed-size recent window");
        double sse = 0;
        for (const auto& sample : monitor.samples) {
            require(sample.hand > 256 && sample.x[0] != 99,
                "Old samples affected the current window");
            const bool left = sample.x[0] == 0;
            require(sample.regret[0] == (left ? 4 : -4) &&
                sample.regret[1] == (left ? -4 : 4), "Labels were not action centered");
            sse += std::pow(sample.regret[0] - 2, 2) +
                std::pow(sample.regret[1] + 2, 2);
        }
        const Split choice = solver.candidate(parent);
        require(sse == 6144 && choice.feature == 0 && choice.threshold == 0 &&
            choice.gain == 1 && choice.left_mass == 0.75 &&
            std::abs(choice.rmse - std::sqrt(12.0)) < 1e-12,
            "Split SSE disagrees with the hand calculation");
        require(solver.table.at(leaf_key(parent)).regrets == original.regrets,
            "Monitoring modified learning regrets");

        std::ostringstream log;
        solver.refine(log);
        const int left_id = solver.tree[parent].left;
        const int right_id = solver.tree[parent].right;
        const auto& left = solver.table.at(leaf_key(left_id));
        const auto& right = solver.table.at(leaf_key(right_id));
        require(solver.splits == 1 && !solver.table.count(leaf_key(parent)),
            "Parent was not replaced by two leaves");
        require(left.regrets[0] == -90 && left.regrets[1] == 270 &&
            right.regrets[0] == -30 && right.regrets[1] == 90,
            "Regrets were reset or inherited at the wrong mass");
        require(left.strategy_sum[0] == 525 && left.strategy_sum[1] == 225 &&
            right.strategy_sum[0] == 175 && right.strategy_sum[1] == 75,
            "Average strategy sums were inherited incorrectly");
        for (int action = 0; action < kActionCount; ++action) {
            require(left.regrets[action] + right.regrets[action] == original.regrets[action] &&
                left.strategy_sum[action] + right.strategy_sum[action] == original.strategy_sum[action],
                "Split lost cumulative mass");
        }
        for (int id : {left_id, right_id}) {
            const auto& child = solver.table.at(leaf_key(id));
            require(current_strategy(child, 3) == current_strategy(original, 3) &&
                average_strategy(child, 3) == average_strategy(original, 3),
                "Split immediately changed the normalized policy");
            require(solver.tree[id].samples.empty() && solver.tree[id].seen == 0,
                "Children unexpectedly inherited monitor samples");
        }
        require(left.touches == 300 && right.touches == 100 &&
            solver.tree[parent].samples.empty(), "Counters or monitor reset differ");
        // Recent label means favor opposite actions, yet both children initially
        // retain the same parent policy: fitting is not per-child regret replay.
        require(current_strategy(left, 3)[0] == 0 && current_strategy(right, 3)[0] == 0,
            "Child policy was initialized from its fitted label mean");
        std::cout << "{\"passed\":true,\"seen\":512,\"fit_samples\":256,"
            "\"left_samples\":192,\"right_samples\":64,\"parent_sse\":6144,"
            "\"relative_gain\":1,\"left_mass\":0.75,"
            "\"parent_regret\":[-120,360],\"left_regret\":[-90,270],"
            "\"right_regret\":[-30,90],\"policies_preserved\":true,"
            "\"child_monitor_reset\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
