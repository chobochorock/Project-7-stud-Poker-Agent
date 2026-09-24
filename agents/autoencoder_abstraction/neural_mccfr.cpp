// Frozen AE/VQ card codes with the existing signed external-sampling MCCFR.
// Usage/build/validation: python -m agents.autoencoder_abstraction.run_cfr --help
// Archived controls: python -m agents.autoencoder_abstraction.analyze_10m --help
// Reuses the existing root-posterior Local BR-gap and posterior-aware PolicyLBR.
// Neither metric is exact exploitability. See CFR.md for budget and recall limits.
#define main epoch_training_entrypoint
#include "../lightgbm_regret_ensemble/stud_epoch_ensemble.cpp"
#undef main

namespace neural7 {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class T> T read(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(input), "truncated neural input");
    return value;
}

std::vector<float> read_floats(std::istream& input, size_t count) {
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
    require(bool(input), "truncated neural weights");
    for (float value : values) {
        require(std::isfinite(value), "nonfinite neural weight");
    }
    return values;
}

struct Network {
    uint32_t hidden, latent, codes;
    std::vector<float> first, first_bias, second, second_bias, centers;
};

class NeuralAtlas : public PowerAtlas {
public:
    explicit NeuralAtlas(const std::string& path) : PowerAtlas(128) {
        std::ifstream input(path, std::ios::binary);
        char magic[8]{};
        input.read(magic, 8);
        require(std::string(magic, 8) == std::string("NAEAT01\0", 8), "bad neural atlas header");
        for (auto& network : networks_) {
            require(read<uint32_t>(input) == 18, "only 18D power inputs supported");
            network.hidden = read<uint32_t>(input);
            network.latent = read<uint32_t>(input);
            network.codes = read<uint32_t>(input);
            require(network.hidden > 0 && network.hidden <= 1024 &&
                network.latent > 0 && network.latent <= 128 &&
                network.codes > 0 && network.codes <= 32768, "invalid network dimensions");
            network.first = read_floats(input, network.hidden * 18);
            network.first_bias = read_floats(input, network.hidden);
            network.second = read_floats(input, network.latent * network.hidden);
            network.second_bias = read_floats(input, network.latent);
            network.centers = read_floats(input, network.codes * network.latent);
        }
        require(input.peek() == std::char_traits<char>::eof(), "trailing neural atlas data");
        set_assignment_cache_limit(100000);
    }

    uint16_t assign_vector(int street, const PowerVector& vector) const override {
        const auto& network = networks_.at(street - 5);
        std::array<float, 1024> hidden{};
        std::array<float, 128> latent{};
        for (size_t i = 0; i < network.hidden; ++i) {
            float value = network.first_bias[i];
            for (size_t j = 0; j < 18; ++j) {
                value += network.first[i * 18 + j] * static_cast<float>(vector[j]);
            }
            hidden[i] = std::max(0.0f, value);
        }
        for (size_t i = 0; i < network.latent; ++i) {
            float value = network.second_bias[i];
            for (size_t j = 0; j < network.hidden; ++j) {
                value += network.second[i * network.hidden + j] * hidden[j];
            }
            latent[i] = value;
        }
        uint16_t best = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (size_t code = 0; code < network.codes; ++code) {
            double distance = 0;
            for (size_t j = 0; j < network.latent; ++j) {
                const double difference = double(latent[j]) - network.centers[code * network.latent + j];
                distance += difference * difference;
            }
            if (distance < best_distance) {
                best = static_cast<uint16_t>(code);
                best_distance = distance;
            }
        }
        return best;
    }

    uint16_t assign(const State& state, int viewer) const override {
        require(viewer == 0 || viewer == 1, "invalid viewer");
        // The inherited observation cache stores seat-independent CARD codes only.
        return PowerAtlas::assign(state, viewer) + viewer * networks_.at(state.street - 5).codes;
    }

    size_t clusters(int street) const override {
        return 2 * networks_.at(street - 5).codes;
    }

private:
    std::array<Network, 3> networks_;
};

struct Target {
    MCCFR& solver;
    bool average;
    std::mt19937_64 rng{0};
    epoch7::Values policy(const State& state, int viewer, bool* found = nullptr) const {
        return average ? solver.policy(state, viewer, found)
            : solver.instantaneous_policy(state, viewer, found);
    }
    Action choose(const State& state, int viewer, int) {
        return epoch7::sample(policy(state, viewer), valid_mask(state, viewer), rng);
    }
};

void test(NeuralAtlas& atlas, const std::string& fixture_path) {
    std::ifstream input(fixture_path, std::ios::binary);
    const auto count = read<uint32_t>(input);
    require(count > 0 && count <= 10000000, "invalid parity fixture count");
    uint64_t mismatches = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto street = read<uint8_t>(input);
        const auto expected = read<uint16_t>(input);
        PowerVector vector{};
        for (auto& value : vector) { value = read<float>(input); }
        if (atlas.assign_vector(street, vector) != expected) { ++mismatches; }
    }
    std::cout << "PARITY rows=" << count << " mismatches=" << mismatches << '\n';
    require(mismatches == 0, "Python/C++ bucket mismatch");
    require(input.peek() == std::char_traits<char>::eof(), "trailing fixture data");
    MCCFR solver(false, 7, 5, &atlas);
    solver.use_cumulative_street_summary(true);
    for (int i = 0; i < 20; ++i) {
        const auto root = epoch7::root_for(7 + i);
        solver.train_root(root, 0);
        solver.train_root(root, 1);
    }
    for (int i = 0; i < 20; ++i) {
        State state = epoch7::root_for(7 + i);
        while (!state.terminal) {
            const int viewer = state.actor;
            State masked = state;
            masked.players[1 - viewer].hidden.clear();
            masked.players[1 - viewer].has_discard = false;
            masked.simulation_deck.clear();
            const auto code = atlas.assign_vector(state.street, atlas.vector(state, viewer));
            require(atlas.assign(state, viewer) == code + viewer * atlas.clusters(state.street) / 2,
                "seat partition/cache mismatch");
            require(atlas.assign(state, viewer) == atlas.assign(masked, viewer), "hidden card leakage");
            for (bool average : {false, true}) {
                Target target{solver, average};
                const auto policy = target.policy(state, viewer);
                require(policy == target.policy(masked, viewer), "hidden policy leakage");
                double total = 0;
                for (int action = 0; action < kActionCount; ++action) {
                    require(std::isfinite(policy[action]) && policy[action] >= 0,
                        "invalid policy probability");
                    require((valid_mask(state, viewer) & (1u << action)) || policy[action] == 0,
                        "illegal policy action");
                    total += policy[action];
                }
                require(std::abs(total - 1) < 1e-9, "unnormalized policy");
            }
            const auto mask = valid_mask(state, viewer);
            state = epoch7::child(state, (mask & (1u << CALL)) ? CALL : CHECK);
        }
    }
    std::cout << "SELF_TEST passed: parity, seat partition, privacy, legal policies\n";
}

void evaluate_gap(MCCFR& solver, std::ostream& output, uint64_t hands,
    int roots, int particles, uint64_t seed) {
    for (int index = 0; index < roots; ++index) {
        const uint64_t root_seed = seed ^ (uint64_t(index + 1) * 0xd1b54a32d192ed03ull);
        const auto root = epoch7::root_for(root_seed);
        std::mt19937_64 rng(root_seed ^ 0x123456789ull);
        const auto hidden = epoch7::root_particles(root, particles, rng);
        output << hands << ',' << index << ',' << root_seed;
        for (bool average : {false, true}) {
            Target target{solver, average};
            const auto gap = epoch7::local_gap(root, hidden, target, root_seed ^ 0xabcdefull);
            require(std::isfinite(gap), "nonfinite Local BR-gap");
            output << ',' << gap;
        }
        output << '\n';
    }
    output.flush();
}

void evaluate_lbr(MCCFR& solver, const std::filesystem::path& output,
    int pairs, int particles, uint64_t seed) {
    std::ofstream csv(output / "lbr_pairs.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(12)
        << "pair,deal_seed,current_seat0,current_seat1,average_seat0,average_seat1\n";
    std::array<uint64_t, 2> queries{}, misses{};
    for (int index = 0; index < pairs; ++index) {
        const uint64_t deal_seed = seed ^ (uint64_t(index + 1) * 0xd1b54a32d192ed03ull);
        auto deck = fresh_deck();
        std::mt19937_64 rng(deal_seed);
        std::shuffle(deck.begin(), deck.end(), rng);
        csv << index << ',' << deal_seed;
        for (int kind = 0; kind < 2; ++kind) {
            Target target{solver, kind == 1};
            for (int seat = 0; seat < 2; ++seat) {
                const auto action_seed = deal_seed ^ (uint64_t(seat + 1) * 0x94d049bb133111ebull);
                target.rng.seed(action_seed);
                PolicyLBR<Target> lbr(target, particles, action_seed ^ 0xabcdef012345ull);
                const double payoff = play_hand_policy_lbr(deck, seat, target, lbr, 1000, 1000);
                require(std::isfinite(payoff) && std::abs(payoff) <= 1000, "invalid LBR payoff");
                csv << ',' << payoff;
                queries[kind] += lbr.stats().policy_queries;
                misses[kind] += lbr.stats().policy_misses;
            }
        }
        csv << '\n';
        if ((index + 1) % 25 == 0 || index + 1 == pairs) {
            csv.flush();
            std::cout << "LBR pairs=" << index + 1 << '/' << pairs << '\n' << std::flush;
        }
    }
    std::ofstream stats(output / "lbr_queries.csv");
    stats.exceptions(std::ios::badbit | std::ios::failbit);
    stats << "policy,queries,misses\ncurrent," << queries[0] << ',' << misses[0]
        << "\naverage," << queries[1] << ',' << misses[1] << '\n';
}

// Evaluate retained legacy checkpoints with exactly the neural-run evaluator.
// No train_root/choose calls or writes to the input models.
void evaluate_kmeans(int argc, char** argv) {
    require(argc == 12, "usage: exe --evaluate-kmeans ATLAS CHECKPOINT_DIR NEW_OUT HANDS EVERY ROOTS PARTICLES LBR_PAIRS LBR_PARTICLES SEED");
    const std::filesystem::path checkpoints(argv[3]), output(argv[4]);
    const uint64_t hands = std::stoull(argv[5]), every = std::stoull(argv[6]);
    const int roots = std::stoi(argv[7]), particles = std::stoi(argv[8]);
    const int pairs = std::stoi(argv[9]), lbr_particles = std::stoi(argv[10]);
    const uint64_t seed = std::stoull(argv[11]);
    require(hands > 0 && every > 0 && hands % every == 0 && roots > 1 &&
        particles > 0 && pairs > 1 && lbr_particles > 0, "invalid evaluation budget");
    require(!std::filesystem::exists(output), "evaluation output already exists");
    for (uint64_t hand = every; hand <= hands; hand += every) {
        require(std::filesystem::is_regular_file(checkpoints / ("hard256_" +
            std::to_string(hand) + ".bin")), "missing k-means checkpoint");
    }
    PowerAtlas atlas;
    atlas.load(argv[2]);
    atlas.set_assignment_cache_limit(100000);
    MCCFR solver(false, 7, 5, &atlas);
    solver.use_cumulative_street_summary(true);
    std::filesystem::create_directories(output);
    std::ofstream gap(output / "local_gap.csv");
    gap.exceptions(std::ios::badbit | std::ios::failbit);
    gap << std::setprecision(12) << "hands,root,root_seed,current,average\n";
    evaluate_gap(solver, gap, 0, roots, particles, seed);
    for (uint64_t hand = every; hand <= hands; hand += every) {
        solver.load((checkpoints / ("hard256_" + std::to_string(hand) + ".bin")).string());
        evaluate_gap(solver, gap, hand, roots, particles, seed);
        if (hand == 100000 || hand == hands) {
            const auto lbr_output = output / ("lbr_" + std::to_string(hand));
            std::filesystem::create_directories(lbr_output);
            evaluate_lbr(solver, lbr_output, pairs, lbr_particles, seed ^ 0x71834719ull);
        }
        std::cout << "EVALUATION model=kmeans hands=" << hand << '/' << hands
            << " roots=" << roots << " particles=" << particles << '\n' << std::flush;
    }
    std::cout << "DONE evaluation_only hands=" << hands << '\n';
}
// Single frozen neural/power/power-memory16 model; identical evaluation to training.
void evaluate_checkpoint(int argc, char** argv) {
    require(argc == 12, "usage: exe --evaluate-checkpoint ATLAS MODEL NEW_OUT HANDS BUCKET ROOTS PARTICLES LBR_PAIRS LBR_PARTICLES SEED");
    const std::filesystem::path output(argv[4]);
    const uint64_t hands = std::stoull(argv[5]), seed = std::stoull(argv[11]);
    const std::string bucket(argv[6]);
    const int roots = std::stoi(argv[7]), particles = std::stoi(argv[8]);
    const int pairs = std::stoi(argv[9]), lbr_particles = std::stoi(argv[10]);
    require(hands > 0 && roots > 1 && particles > 0 && pairs > 1 && lbr_particles > 0,
        "invalid evaluation budget");
    require(bucket == "power" || bucket == "power-memory16" || bucket == "neural", "unsupported checkpoint bucket");
    require(!std::filesystem::exists(output), "evaluation output already exists");
    std::unique_ptr<PowerAtlas> atlas;
    if (bucket == "neural") {
        atlas = std::make_unique<NeuralAtlas>(argv[2]);
    } else {
        atlas = std::make_unique<PowerAtlas>();
        atlas->load(argv[2]);
    }
    atlas->set_assignment_cache_limit(100000);
    MCCFR solver(false, 7, 5, atlas.get());
    solver.use_cumulative_street_summary(bucket != "power");
    solver.load(argv[3]);
    const auto before = solver.bucket_count();
    std::filesystem::create_directories(output);
    std::ofstream gap(output / "local_gap.csv");
    gap.exceptions(std::ios::badbit | std::ios::failbit);
    gap << std::setprecision(12) << "hands,root,root_seed,current,average\n";
    evaluate_gap(solver, gap, hands, roots, particles, seed);
    evaluate_lbr(solver, output, pairs, lbr_particles, seed ^ 0x71834719ull);
    require(solver.traversals() == 0 && solver.bucket_count() == before, "evaluation trained the target");
    std::cout << "DONE evaluation_only hands=" << hands << " buckets=" << before << '\n';
}
} // namespace neural7

int main(int argc, char** argv) {
    try {
        using namespace neural7;
        if (argc > 1 && std::string(argv[1]) == "--evaluate-checkpoint") {
            evaluate_checkpoint(argc, argv);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--evaluate-kmeans") {
            evaluate_kmeans(argc, argv);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--engine-test") {
            self_test();
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--self-test") {
            NeuralAtlas atlas(argv[2]);
            test(atlas, argv[3]);
            return 0;
        }
        require(argc == 11, "usage: exe ATLAS NEW_OUT HANDS SEED EVAL_EVERY ROOTS PARTICLES LBR_PAIRS LBR_PARTICLES EVAL_SEED");
        NeuralAtlas atlas(argv[1]);
        const std::filesystem::path output(argv[2]);
        const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]);
        const uint64_t every = std::stoull(argv[5]);
        const int roots = std::stoi(argv[6]), particles = std::stoi(argv[7]);
        const int pairs = std::stoi(argv[8]), lbr_particles = std::stoi(argv[9]);
        const uint64_t eval_seed = std::stoull(argv[10]);
        require(hands > 0 && every > 0 && roots > 1 && particles > 0 && pairs > 1 && lbr_particles > 0,
            "invalid run budget");
        require(!std::filesystem::exists(output), "solver output already exists");
        std::filesystem::create_directories(output);
        MCCFR solver(false, seed, 5, &atlas);
        solver.use_cumulative_street_summary(true);
        std::ofstream gap(output / "local_gap.csv"), progress(output / "progress.csv");
        gap.exceptions(std::ios::badbit | std::ios::failbit);
        progress.exceptions(std::ios::badbit | std::ios::failbit);
        gap << std::setprecision(12) << "hands,root,root_seed,current,average\n";
        progress << "hands,traversals,node_visits,buckets,train_seconds,elapsed_seconds,peak_rss_bytes\n";
        const auto started = std::chrono::steady_clock::now();
        double train_seconds = 0;
        evaluate_gap(solver, gap, 0, roots, particles, eval_seed);
        for (uint64_t hand = 1; hand <= hands; ++hand) {
            const auto before = std::chrono::steady_clock::now();
            const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
            solver.train_root(root, 0);
            solver.train_root(root, 1);
            train_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
            if (hand % every == 0 || hand == hands) {
                evaluate_gap(solver, gap, hand, roots, particles, eval_seed);
            }
            if (hand % 1000 == 0 || hand == hands) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                progress << hand << ',' << solver.traversals() << ',' << solver.node_visits() << ','
                    << solver.bucket_count() << ',' << train_seconds << ',' << elapsed << ',' << epoch7::peak_rss() << '\n';
                progress.flush();
                std::cout << "PROGRESS hands=" << hand << '/' << hands << " traversals=" << solver.traversals()
                    << " node_visits=" << solver.node_visits() << " buckets=" << solver.bucket_count()
                    << " train_seconds=" << train_seconds << " elapsed_seconds=" << elapsed
                    << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0 << '\n' << std::flush;
            }
        }
        const auto path = output / ("policy_" + std::to_string(hands) + ".bin");
        solver.save(path.string());
        // Model-only checkpoint: verify both policies survive loading, then evaluate the loaded tables.
        std::vector<epoch7::Values> before;
        for (int i = 0; i < 50; ++i) {
            const auto state = epoch7::root_for(eval_seed + i);
            before.push_back(solver.policy(state, state.actor));
            before.push_back(solver.instantaneous_policy(state, state.actor));
        }
        solver.load(path.string());
        for (int i = 0; i < 50; ++i) {
            const auto state = epoch7::root_for(eval_seed + i);
            require(before[2 * i] == solver.policy(state, state.actor) &&
                before[2 * i + 1] == solver.instantaneous_policy(state, state.actor), "checkpoint policy mismatch");
        }
        evaluate_lbr(solver, output, pairs, lbr_particles, eval_seed ^ 0x71834719ull);
        std::cout << "DONE hands=" << hands << " checkpoint=" << path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
