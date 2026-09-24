#define STUD_REBEL_SEVENTH_NO_MAIN
#include "stud_rebel_seventh.cpp"

namespace {

class V7SampleWriter {
public:
    explicit V7SampleWriter(const std::string& path)
        : output_(path, std::ios::binary) {
        if (!output_) throw std::runtime_error("cannot write V7 samples: " + path);
        output_.write("RBV7S3\0", 8);
        write_u32(3);
        write_u32(kRebelV7FeatureSize);
        write_u64(0);
    }

    void add(
        const RebelSeventhResolverPolicy::ValueSample& sample,
        uint64_t root_id) {
        const float group = static_cast<float>(root_id);
        output_.write(reinterpret_cast<const char*>(&group), sizeof(group));
        output_.write(
            reinterpret_cast<const char*>(sample.features.data()),
            sample.features.size() * sizeof(float));
        output_.write(
            reinterpret_cast<const char*>(&sample.value), sizeof(sample.value));
        if (!output_) throw std::runtime_error("failed to write V7 sample");
        ++count_;
    }

    void finish() {
        output_.seekp(16);
        write_u64(count_);
        output_.close();
    }

    uint64_t count() const { return count_; }

private:
    void write_u32(uint32_t value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    void write_u64(uint64_t value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    std::ofstream output_;
    uint64_t count_ = 0;
};

struct GenerateOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string output = "cpp_mccfr\\rebel_v7_samples.bin";
    std::string bucket = "power-memory16";
    int roots = 1000;
    int particles = 240;
    int iterations = 100;
    int samples_per_root = 8;
    int report_every = 50;
    int ante = 1000;
    int stack_ante = 1000;
    double prior = 100.0;
    double behavior_epsilon = 0.1;
    uint64_t power_cache_max = 250000;
    uint64_t seed = 73001;
    bool self_test = false;
};

GenerateOptions parse_generate_options(int argc, char** argv) {
    GenerateOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") options.model = value(index);
        else if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--output") options.output = value(index);
        else if (argument == "--bucket") options.bucket = value(index);
        else if (argument == "--roots") options.roots = std::stoi(value(index));
        else if (argument == "--particles") {
            options.particles = std::stoi(value(index));
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(value(index));
        } else if (argument == "--samples-per-root") {
            options.samples_per_root = std::stoi(value(index));
        } else if (argument == "--report-every") {
            options.report_every = std::stoi(value(index));
        } else if (argument == "--ante") {
            options.ante = std::stoi(value(index));
        } else if (argument == "--stack-ante") {
            options.stack_ante = std::stoi(value(index));
        } else if (argument == "--prior") {
            options.prior = std::stod(value(index));
        } else if (argument == "--behavior-epsilon") {
            options.behavior_epsilon = std::stod(value(index));
        } else if (argument == "--power-cache-max") {
            options.power_cache_max = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.roots <= 0 || options.particles <= 0 ||
        options.iterations <= 0 || options.samples_per_root <= 0 ||
        options.ante <= 0 || options.stack_ante <= 0 ||
        options.prior < 0.0 || options.behavior_epsilon < 0.0 ||
        options.behavior_epsilon > 1.0) {
        throw std::runtime_error("invalid V7 generation configuration");
    }
    if (options.bucket != "power" &&
        options.bucket != "power-tree" &&
        options.bucket != "power-memory16") {
        throw std::runtime_error("unsupported V7 bucket mode");
    }
    return options;
}

Action sample_strategy(
    const std::array<double, kActionCount>& strategy,
    uint8_t mask,
    double epsilon,
    std::mt19937_64& rng) {
    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < epsilon) {
        const auto legal = actions_from_mask(mask);
        return legal[std::uniform_int_distribution<size_t>(
            0, legal.size() - 1)(rng)];
    }
    const double threshold =
        std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double cumulative = 0.0;
    Action last = FOLD;
    for (int action = 0; action < kActionCount; ++action) {
        if (!(mask & (1u << action))) continue;
        last = static_cast<Action>(action);
        cumulative += strategy[action];
        if (threshold <= cumulative) return last;
    }
    return last;
}

void advance_generation_street(State& state) {
    if (state.street == 7) {
        state.terminal = true;
        return;
    }
    ++state.street;
    const bool public_card = state.street == 6;
    for (int seat = 0; seat < 2; ++seat) {
        if (state.simulation_deck.empty()) {
            throw std::runtime_error("V7 generation deck exhausted");
        }
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (public_card) state.players[seat].shown.push_back(card);
        else state.players[seat].hidden.push_back(card);
    }
    reset_round(state, state.street);
}

bool reach_seventh(
    State& state,
    MCCFR& blueprint,
    double epsilon,
    std::mt19937_64& rng) {
    while (!state.terminal && state.street < 7) {
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const Action action = sample_strategy(
            blueprint.policy(state, actor), mask, epsilon, rng);
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd) advance_generation_street(state);
        else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
    }
    return !state.terminal && state.street == 7;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        GenerateOptions options = parse_generate_options(argc, argv);
        if (options.self_test) {
            options.roots = 2;
            options.particles = 16;
            options.iterations = 4;
            options.samples_per_root = 2;
            options.report_every = 0;
        }
        PowerAtlas atlas;
        atlas.load(options.atlas);
        atlas.set_assignment_cache_limit(options.power_cache_max);
        MCCFR blueprint(false, options.seed, 5, &atlas);
        blueprint.use_cumulative_street_summary(
            options.bucket == "power-memory16");
        blueprint.load(options.model);
        RebelSeventhResolverPolicy resolver(
            blueprint,
            atlas,
            options.iterations,
            options.particles,
            options.prior,
            options.seed ^ 0x6a09e667f3bcc909ull,
            options.behavior_epsilon);
        V7SampleWriter writer(options.output);
        std::mt19937_64 rng(options.seed);
        uint64_t attempted_hands = 0;
        const auto started = std::chrono::steady_clock::now();
        for (int root = 1; root <= options.roots;) {
            ++attempted_hands;
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), rng);
            State state = sample_fifth_street_root(
                deck, options.ante, options.stack_ante);
            if (!reach_seventh(
                    state, blueprint, options.behavior_epsilon, rng)) {
                continue;
            }
            for (const auto& sample : resolver.value_samples(
                    state, options.samples_per_root)) {
                writer.add(sample, static_cast<uint64_t>(root));
            }
            if (options.report_every && root % options.report_every == 0) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << "{\"v7_roots\":" << root
                          << ",\"attempted_hands\":" << attempted_hands
                          << ",\"samples\":" << writer.count()
                          << ",\"elapsed_seconds\":" << seconds << "}\n";
            }
            ++root;
        }
        writer.finish();
        const auto stats = resolver.stats();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "{\"generator\":\"rebel-v7-values\""
                  << ",\"output\":\"" << options.output << "\""
                  << ",\"dimensions\":" << kRebelV7FeatureSize
                  << ",\"belief_dimensions\":" << kRebelBeliefSize
                  << ",\"belief_epsilon\":" << options.behavior_epsilon
                  << ",\"roots\":" << options.roots
                  << ",\"attempted_hands\":" << attempted_hands
                  << ",\"samples\":" << writer.count()
                  << ",\"subgames\":" << stats.subgames
                  << ",\"node_visits\":" << stats.node_visits
                  << ",\"elapsed_seconds\":" << elapsed << "}\n";
        if (options.self_test) {
            if (writer.count() != 8 || stats.subgames != 2) {
                throw std::runtime_error("V7 generator self-test failed");
            }
            std::cerr << "{\"self_test\":\"ok\"}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
