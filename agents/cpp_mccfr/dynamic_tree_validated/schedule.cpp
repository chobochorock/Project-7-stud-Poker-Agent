// Fork a frozen DT checkpoint and compare split schedules without changing its solver.
// Usage: python -m agents.cpp_mccfr.dynamic_tree_validated.schedule --help
#define VALIDATED_NO_MAIN
#include "solver.cpp"

namespace split_schedule {
using namespace dynamic_tree;

bool split_due(const Node& node, uint64_t minimum, uint64_t interval) {
    return node.seen >= minimum && node.seen % interval == 0;
}

void discard_monitors(Solver& solver) {
    for (auto& node : solver.tree) { std::vector<Sample>().swap(node.samples); }
    solver.pending.clear();
}

Progress read_parent(const fs::path& path, const std::string& identity,
                     Solver& tree, Solver& baseline) {
    std::ifstream in(path / "checkpoint.bin", std::ios::binary);
    require(read_text(in) == "DYNAMIC_TREE_VALIDATED1" && read_text(in) == identity,
            "parent checkpoint identity mismatch");
    Progress result;
    result.hand = read<uint64_t>(in);
    for (auto& bytes : result.bytes) { bytes = read<uint64_t>(in); }
    tree.load(in);
    baseline.load(in);
    require(in.peek() == std::char_traits<char>::eof(), "trailing parent model bytes");
    for (size_t i = 0; i < kLogs.size(); ++i) {
        require(fs::file_size(path / kLogs[i]) == result.bytes[i], "parent has uncommitted logs");
    }
    return result;
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        Node node;
        node.seen = 256;
        require(split_due(node, 256, 256), "default schedule");
        require(!split_due(node, 1024, 1024), "minimum grace period");
        node.seen = 1024;
        require(split_due(node, 1024, 1024), "delayed split check");
        node.seen = 1280;
        require(!split_due(node, 1024, 1024), "split check interval");
        std::cout << "SELF_TEST split schedule: PASS\n";
        return 0;
    }
    require(argc == 24 || argc == 26, "Use schedule.py --help");
    const fs::path path(argv[1]);
    const std::string identity(argv[2]);
    const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]);
    const uint64_t eval_seed = std::stoull(argv[5]), every = std::stoull(argv[7]);
    const int particles = std::stoi(argv[6]), roots = std::stoi(argv[8]);
    const uint64_t save_every = std::stoull(argv[9]);
    const bool initialize = argc == 26;
    Config config;
    config.window = std::stoul(argv[11]); config.min_child = std::stoul(argv[12]);
    config.max_splits = std::stoll(argv[13]); config.max_depth = std::stoll(argv[14]);
    config.gain = std::stod(argv[15]); config.min_rmse = std::stod(argv[16]);
    const uint64_t max_states = std::stoull(argv[17]), reserve = std::stoull(argv[18]);
    config.split_rule = std::stoi(argv[19]); config.validation_z = std::stod(argv[20]);
    config.freeze_after = std::stoull(argv[21]);
    const uint64_t minimum = std::stoull(argv[22]), interval = std::stoull(argv[23]);
    require(hands && every && save_every && particles >= 2 && roots >= 2 &&
            config.window >= 4 && minimum >= config.window && interval >= config.window &&
            minimum % config.window == 0 && interval % config.window == 0,
            "invalid schedule/evaluation arguments");
    PowerAtlas atlas;
    atlas.load((path / "hand_atlas.bin").string());
    atlas.set_assignment_cache_limit(100000);
    Solver tree(atlas, true, seed, config), baseline(atlas, false, seed, config);
    Progress progress = initialize ? read_parent(argv[24], argv[25], tree, baseline)
                                   : restore(path, identity, tree, baseline);
    tree.hand = baseline.hand = progress.hand;
    require(progress.hand <= hands, "target below saved hand count");
    if (initialize) { require(hands == progress.hand, "fork must not train"); }
    const bool frozen = config.freeze_after && progress.hand >= config.freeze_after;
    if (frozen) { discard_monitors(tree); }
    std::array<std::ofstream, 3> logs;
    for (size_t i = 0; i < logs.size(); ++i) {
        logs[i].open(path / kLogs[i], initialize ? std::ios::trunc : std::ios::app);
        logs[i] << std::setprecision(17);
    }
    const auto panel = [&]() {
        std::array<double, 4> sums{};
        for (int i = 0; i < roots; ++i) {
            std::ostringstream row;
            row << std::setprecision(17);
            evaluate(row, progress.hand, eval_seed ^ ((i + 1) * 0xd1b54a32d192ed03ull),
                     particles, tree, baseline);
            logs[1] << row.str();
            std::istringstream values(row.str());
            std::string value;
            std::getline(values, value, ',');
            for (auto& sum : sums) { std::getline(values, value, ','); sum += std::stod(value) / roots; }
        }
        logs[1].flush();
        require(bool(logs[1]), "panel write failed");
        std::cout << "LOCAL_BR_GAP hands=" << progress.hand << " tree_current=" << sums[0]
                  << " kmeans_current=" << sums[1] << " tree_average=" << sums[2]
                  << " kmeans_average=" << sums[3] << " roots=" << roots
                  << " particles_per_root=" << particles << " unit=ante\n" << std::flush;
    };
    const auto checkpoint = [&]() {
        for (auto& log : logs) { log.flush(); require(bool(log), "log flush failed"); }
        save(path, identity, progress, tree, baseline);
    };
    if (initialize) {
        // Header-only compatibility file: this experiment does NOT evaluate every hand.
        header(logs[0]); header(logs[1]);
        logs[2] << "hands,parent,group,accepted,feature,threshold,parent_rmse,relative_gain,validation_gain,validation_se\n";
        panel(); checkpoint();
        std::cout << "FORK saved_hands=" << progress.hand << " frozen=" << frozen
                  << " training_performed=0\n";
        return 0;
    }
    SetConsoleCtrlHandler(stop_handler, TRUE);
    const auto start = std::chrono::steady_clock::now();
    std::cout << "CONFIG freeze_after=" << config.freeze_after << " min_split_visits=" << minimum
              << " split_check_visits=" << interval << " eval_every=" << every
              << " checkpoint_every=" << save_every << " max_splits=" << config.max_splits
              << " max_depth=" << config.max_depth << " kmeans_state_limit=" << max_states
              << " reserve_bytes=" << reserve << std::endl;
    for (uint64_t hand = progress.hand + 1; hand <= hands; ++hand) {
        const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        tree.train(root, hand);
        baseline.train(root, hand);
        if (!frozen) {
            for (auto it = tree.pending.begin(); it != tree.pending.end();) {
                if (!split_due(tree.tree[*it], minimum, interval)) { it = tree.pending.erase(it); }
                else { ++it; }
            }
            tree.refine(logs[2]);
        }
        progress.hand = hand;
        const bool state_cap = baseline.table.size() >= max_states;
        const bool low_disk = fs::space(path).available < reserve;
        const bool stop = stopping || state_cap || low_disk;
        if (hand % every == 0 || hand == hands || stop) { panel(); }
        if (hand % save_every == 0 || hand == hands || stop) { checkpoint(); }
        if (hand % 1000 == 0 || hand == hands || stop) {
            std::cout << "PROGRESS hands=" << hand << " target=" << hands
                      << " tree_leaves=" << tree.table.size() << " splits=" << tree.splits
                      << " checks=" << tree.checks << " kmeans_states=" << baseline.table.size()
                      << " tree_nodes=" << tree.nodes << " kmeans_nodes=" << baseline.nodes
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
    return 0;
}
}  // namespace split_schedule

int main(int argc, char** argv) {
    try { return split_schedule::run(argc, argv); }
    catch (const std::exception& error) { std::cerr << "ERROR " << error.what() << '\n'; return 1; }
}
