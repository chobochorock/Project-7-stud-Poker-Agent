// Fresh centered DT with gain(t) = 1 / (1 + log10(total training hands)).
// Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain --help
// Reuses the frozen solver/checkpoint/LBR format; no changes to past experiments.
#define VALIDATED_NO_MAIN
#include "solver.cpp"

namespace log_gain {
using namespace dynamic_tree;

double threshold(uint64_t hand, double fixed_gain) {
    return fixed_gain > 0 ? fixed_gain : 1.0 / (1.0 + std::log10(double(std::max<uint64_t>(1, hand))));
}

void append_gain(std::ostream& out, const std::string& rows, double gain) {
    std::istringstream input(rows);
    std::string row;
    while (std::getline(input, row)) { out << row << ',' << gain << '\n'; }
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        require(threshold(0, 0) == 1 && threshold(1, 0) == 1, "initial threshold");
        require(std::abs(threshold(1000000, 0) - 1.0 / 7) < 1e-15, "1M threshold");
        require(threshold(1000000000, 0) == 0.1, "1B threshold");
        require(threshold(1000000, 0.15) == 0.15, "fixed control");
        double previous = 1;
        for (uint64_t hand = 1; hand <= 1000000; hand *= 10) {
            require(threshold(hand, 0) <= previous, "monotonic threshold");
            previous = threshold(hand, 0);
        }
        std::cout << "SELF_TEST log gain: PASS\n";
        return 0;
    }
    require(argc == 17, "Use log_gain.py --help");
    const fs::path path(argv[1]);
    const std::string identity(argv[2]);
    const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]);
    const uint64_t eval_seed = std::stoull(argv[5]), every = std::stoull(argv[7]);
    const int particles = std::stoi(argv[6]), roots = std::stoi(argv[8]);
    const uint64_t save_every = std::stoull(argv[9]);
    const bool resume = std::stoi(argv[10]) != 0;
    Config config;
    config.window = std::stoul(argv[11]); config.min_child = std::stoul(argv[12]);
    config.min_rmse = std::stod(argv[13]); config.split_rule = 1;
    const double fixed_gain = std::stod(argv[14]);
    const uint64_t max_states = std::stoull(argv[15]), reserve = std::stoull(argv[16]);
    require(hands && every && save_every && max_states && seed && particles >= 2 && roots >= 2 &&
            config.window >= 4 && config.min_child >= 2 && 2 * config.min_child <= config.window &&
            std::isfinite(config.min_rmse) && config.min_rmse >= 0 &&
            std::isfinite(fixed_gain) && fixed_gain >= 0 && fixed_gain <= 1,
            "invalid log-gain arguments");
    require(resume || !fs::exists(path / "checkpoint.bin"), "existing checkpoint; use resume");
    PowerAtlas atlas;
    atlas.load((path / "hand_atlas.bin").string());
    atlas.set_assignment_cache_limit(100000);
    Solver tree(atlas, true, seed, config), baseline(atlas, false, seed, config);
    Progress progress;
    if (resume) { progress = restore(path, identity, tree, baseline); }
    require(progress.hand <= hands, "target below saved hand count");
    tree.hand = baseline.hand = progress.hand;
    tree.config.gain = threshold(progress.hand, fixed_gain);
    std::array<std::ofstream, 3> logs;
    for (size_t i = 0; i < logs.size(); ++i) {
        logs[i].open(path / kLogs[i], resume ? std::ios::app : std::ios::trunc);
        logs[i] << std::setprecision(17);
        require(bool(logs[i]), "cannot open log");
    }
    const auto panel = [&]() {
        std::array<double, 4> sums{};
        for (int i = 0; i < roots; ++i) {
            std::ostringstream row;
            row << std::setprecision(17);
            evaluate(row, progress.hand, eval_seed ^ ((i + 1) * 0xd1b54a32d192ed03ull),
                     particles, tree, baseline);
            append_gain(logs[1], row.str(), tree.config.gain);
            std::istringstream values(row.str());
            std::string value;
            std::getline(values, value, ',');
            for (auto& sum : sums) { std::getline(values, value, ','); sum += std::stod(value) / roots; }
        }
        logs[1].flush();
        require(bool(logs[1]), "panel write failed");
        std::cout << "LOCAL_BR_GAP hands=" << progress.hand << " tree_current=" << sums[0]
                  << " kmeans_current=" << sums[1] << " tree_average=" << sums[2]
                  << " kmeans_average=" << sums[3] << " gain_threshold=" << tree.config.gain
                  << " roots=" << roots << " particles_per_root=" << particles
                  << " unit=ante\n" << std::flush;
    };
    const auto checkpoint = [&]() {
        for (auto& log : logs) { log.flush(); require(bool(log), "log flush failed"); }
        save(path, identity, progress, tree, baseline);
    };
    if (!resume) {
        // Header-only per_hand.csv preserves checkpoint compatibility, not per-hand evaluation.
        header(logs[0]);
        std::ostringstream columns;
        header(columns);
        std::string line = columns.str();
        line.pop_back();
        logs[1] << line << ",gain_threshold\n";
        logs[2] << "hands,parent,group,accepted,feature,threshold,parent_rmse,relative_gain,validation_gain,validation_se,gain_threshold\n";
        panel(); checkpoint();
    }
    SetConsoleCtrlHandler(stop_handler, TRUE);
    const auto start = std::chrono::steady_clock::now();
    std::cout << "CONFIG gain_mode=" << (fixed_gain > 0 ? "fixed" : "inverse_log10")
              << " fixed_gain=" << fixed_gain << " t_unit=total_training_hands target=" << hands
              << " window=" << config.window << " min_child=" << config.min_child
              << " min_rmse=" << config.min_rmse << " eval_every=" << every
              << " checkpoint_every=" << save_every
              << " max_splits=unlimited max_depth=unlimited ram_cap=none"
              << " kmeans_state_limit=" << max_states << " disk_reserve_bytes=" << reserve << std::endl;
    for (uint64_t hand = progress.hand + 1; hand <= hands; ++hand) {
        tree.config.gain = threshold(hand, fixed_gain);
        const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        tree.train(root, hand);
        baseline.train(root, hand);
        if (!tree.pending.empty()) {
            std::ostringstream decisions;
            decisions << std::setprecision(17);
            tree.refine(decisions);
            append_gain(logs[2], decisions.str(), tree.config.gain);
        }
        progress.hand = hand;
        const bool state_cap = baseline.table.size() >= max_states;
        const bool low_disk = fs::space(path).available < reserve;
        const bool stop = stopping || state_cap || low_disk;
        if (hand % every == 0 || hand == hands || stop) { panel(); }
        if (hand % save_every == 0 || hand == hands || stop) { checkpoint(); }
        if (hand % 1000 == 0 || hand == hands || stop) {
            std::cout << "PROGRESS hands=" << hand << " target=" << hands
                      << " gain_threshold=" << tree.config.gain << " tree_leaves=" << tree.table.size()
                      << " splits=" << tree.splits << " checks=" << tree.checks
                      << " kmeans_states=" << baseline.table.size() << " tree_nodes=" << tree.nodes
                      << " kmeans_nodes=" << baseline.nodes
                      << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0
                      << " wall_seconds=" << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count() << std::endl;
        }
        if (stop) {
            std::cout << "PAUSED reason=" << (stopping ? "user_stop" : state_cap ? "kmeans_state_limit" : "low_disk")
                      << "; checkpoint saved\n";
            return 2;
        }
    }
    std::cout << "COMPLETE hands=" << progress.hand << " checkpoint_saved=1\n";
    return 0;
}
}  // namespace log_gain

int main(int argc, char** argv) {
    try { return log_gain::run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 1; }
}
