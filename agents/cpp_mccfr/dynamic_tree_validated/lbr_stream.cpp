// Read-only, monitor-skipping LBR batch. Usage: lbr_stream.py --help / LBR_STREAM.md.
#include "stream_adapter.inc"

void loader_test(const PowerAtlas& atlas) {
    using namespace dynamic_tree;
    Config config;
    Solver original(atlas, true, 51, config);
    const State state = epoch7::root_for(197);
    const auto group = epoch7::group_id(state, state.actor);
    original.tree.resize(3);
    original.roots[group] = 0;
    original.splits = 1;
    original.checks = 7;
    for (int id = 0; id < 3; ++id) {
        auto& node = original.tree[id];
        node.group = group;
        node.depth = id != 0;
        if (!id) {
            node.feature = 0;
            node.left = 1;
            node.right = 2;
            node.threshold = 0.5;
        } else {
            RegretNode stats;
            for (int a = 0; a < kActionCount; ++a) {
                stats.regrets[a] = (a % 2 ? -1 : 1) * double(a + id);
                stats.strategy_sum[a] = a + id;
            }
            stats.touches = 11;
            original.table.emplace(leaf_key(id), stats);
            Sample sample;
            sample.x = original.features(state, state.actor);
            sample.regret.fill(2.0);
            sample.hand = 17;
            node.samples.assign(id + 1, sample);
            node.seen = 21;
            node.sample_hand = 17;
            node.hand_visits = 4;
        }
    }
    std::stringstream saved;
    original.save(saved);
    Solver reference(atlas, true, 0, config), streamed(atlas, true, 0, config);
    std::stringstream a(saved.str()), b(saved.str());
    reference.load_reference(a);
    streamed.load(b);
    require(a.tellg() == b.tellg(), "stream loader changed checkpoint cursor");
    for (size_t i = 0; i < reference.tree.size(); ++i) {
        require(streamed.tree[i].samples.empty(), "stream loader retained monitor samples");
        reference.tree[i].samples.clear();
    }
    std::stringstream x, y;
    reference.save(x);
    streamed.save(y);
    require(x.str() == y.str(), "non-monitor solver state differs from original loader");
    for (bool average : {false, true}) {
        Target left{reference, average}, right{streamed, average};
        require(left.policy(state, state.actor) == right.policy(state, state.actor),
                "stream loader changes policy");
    }
    bool rejected = false;
    try {
        std::stringstream truncated(saved.str().substr(0, saved.str().size() - 100));
        streamed.load(truncated);
    } catch (const std::exception&) { rejected = true; }
    require(rejected, "stream loader accepted truncated checkpoint");
    std::cout << "LOADER_TEST original/stream non-monitor bytes, policies, cursor, truncation: PASS\n";
}

int main(int argc, char** argv) {
    try {
        hand_recall::require(argc >= 7, "Use python -m agents.cpp_mccfr.dynamic_tree_validated.lbr_stream --help");
        const auto start = std::chrono::steady_clock::now();
        std::cout << "LOAD_START run=" << argv[1] << " monitor_payload=skip" << std::endl;
        FrozenModel model(argv[1], argv[2]);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "LOAD_DONE hands=" << model.hands << " seconds=" << seconds
                  << " peak_rss_mib=" << epoch7::peak_rss()/1048576.0 << std::endl;
        loader_test(model.atlas);
        abstraction_lbr::self_test(model);
        const std::filesystem::path output(argv[3]);
        std::ofstream metadata(output / "loader.json");
        metadata << "{\"hands\":" << model.hands << ",\"load_seconds\":" << seconds
                 << ",\"load_peak_rss_bytes\":" << epoch7::peak_rss()
                 << ",\"monitor_payload_loaded\":false,\"loader_parity_test\":true}\n";
        metadata.close();
        hand_recall::require(bool(metadata), "loader metadata write failed");
        for (int index = 6; index < argc; ++index) {
            std::cout << "EVALUATE seed=" << argv[index] << std::endl;
            std::array<std::string,6> arguments{
                argv[0], argv[1], (output/argv[index]).string(), argv[4], argv[5], argv[index]};
            std::array<char*,6> pointers;
            for (size_t i = 0; i < arguments.size(); ++i) { pointers[i] = arguments[i].data(); }
            abstraction_lbr::evaluate(model, int(pointers.size()), pointers.data());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
}
