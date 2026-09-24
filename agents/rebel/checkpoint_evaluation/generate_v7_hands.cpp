#define main stud_rebel_v7_generate_original_main
#if __has_include("../../cpp_mccfr/stud_rebel_v7_generate.cpp")
#include "../../cpp_mccfr/stud_rebel_v7_generate.cpp"
#else
#include "../cpp_mccfr/stud_rebel_v7_generate.cpp"  // Legacy directory alias.
#endif
#undef main

#include <filesystem>

namespace {

struct HandGenerateOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string output = "rebel_7stud_checkpoints\\v7_hands.bin";
    std::string bucket = "power-memory16";
    int hands = 10000;
    int particles = 64;
    int iterations = 32;
    int samples_per_root = 4;
    int report_every = 1000;
    int ante = 1000;
    int stack_ante = 1000;
    double prior = 100.0;
    double behavior_epsilon = 0.1;
    uint64_t group_offset = 0;
    uint64_t power_cache_max = 250000;
    uint64_t seed = 75001;
    bool self_test = false;
};

HandGenerateOptions parse_hand_options(int argc, char** argv) {
    HandGenerateOptions options;
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
        else if (argument == "--hands") options.hands = std::stoi(value(index));
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
        } else if (argument == "--group-offset") {
            options.group_offset = std::stoull(value(index));
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
    if (options.hands <= 0 || options.particles <= 0 ||
        options.iterations <= 0 || options.samples_per_root <= 0 ||
        options.ante <= 0 || options.stack_ante <= 0 ||
        options.prior < 0.0 || options.behavior_epsilon < 0.0 ||
        options.behavior_epsilon > 1.0) {
        throw std::runtime_error("invalid V7 hand-generation configuration");
    }
    if (options.bucket != "power" && options.bucket != "power-tree" &&
        options.bucket != "power-memory16") {
        throw std::runtime_error("unsupported V7 bucket mode");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        HandGenerateOptions options = parse_hand_options(argc, argv);
        if (options.self_test) {
            options.hands = 20;
            options.particles = 8;
            options.iterations = 2;
            options.samples_per_root = 1;
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
        const std::filesystem::path output_path(options.output);
        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        V7SampleWriter writer(options.output);
        std::mt19937_64 rng(options.seed);
        uint64_t roots = 0;
        const auto started = std::chrono::steady_clock::now();
        for (int hand = 1; hand <= options.hands; ++hand) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), rng);
            State state = sample_fifth_street_root(
                deck, options.ante, options.stack_ante);
            if (reach_seventh(
                    state, blueprint, options.behavior_epsilon, rng)) {
                ++roots;
                const uint64_t group = options.group_offset + hand;
                for (const auto& sample : resolver.value_samples(
                         state, options.samples_per_root)) {
                    writer.add(sample, group);
                }
            }
            if (options.report_every && hand % options.report_every == 0) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << "{\"v7_hands\":" << hand
                          << ",\"v7_roots\":" << roots
                          << ",\"samples\":" << writer.count()
                          << ",\"elapsed_seconds\":" << seconds << "}\n";
            }
        }
        writer.finish();
        const auto stats = resolver.stats();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "{\"generator\":\"partial-rebel-v7-hands\""
                  << ",\"output\":\"" << options.output << "\""
                  << ",\"hands\":" << options.hands
                  << ",\"roots\":" << roots
                  << ",\"samples\":" << writer.count()
                  << ",\"root_rate\":"
                  << roots / static_cast<double>(options.hands)
                  << ",\"node_visits\":" << stats.node_visits
                  << ",\"elapsed_seconds\":" << elapsed << "}\n";
        if (options.self_test && (!roots || !writer.count())) {
            throw std::runtime_error("V7 hand generator self-test failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
