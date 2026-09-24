#define STUD_MCCFR_NO_MAIN
#if __has_include("../../cpp_mccfr/stud_mccfr.cpp")
#include "../../cpp_mccfr/stud_mccfr.cpp"
#else
#include "../cpp_mccfr/stud_mccfr.cpp"  // Legacy directory alias.
#endif

#include "sampled_br_gap.h"

#include <filesystem>

namespace {

struct GapExperimentOptions {
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string csv = "br_gap_experiment\\results\\per_hand.csv";
    std::string model = "br_gap_experiment\\results\\k512_100k.bin";
    uint64_t hands = 100000;
    uint64_t report_every = 1000;
    uint64_t seed = 81001;
    uint64_t power_cache_max = 250000;
    int ante = 1000;
    int stack_ante = 1000;
    bool self_test = false;
};

GapExperimentOptions parse_gap_options(int argc, char** argv) {
    GapExperimentOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--csv") options.csv = value(index);
        else if (argument == "--save") options.model = value(index);
        else if (argument == "--hands") {
            options.hands = std::stoull(value(index));
        } else if (argument == "--report-every") {
            options.report_every = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--power-cache-max") {
            options.power_cache_max = std::stoull(value(index));
        } else if (argument == "--ante") {
            options.ante = std::stoi(value(index));
        } else if (argument == "--stack-ante") {
            options.stack_ante = std::stoi(value(index));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (!options.hands || options.ante <= 0 || options.stack_ante <= 0) {
        throw std::runtime_error("invalid BR-gap experiment configuration");
    }
    return options;
}

struct RunningMean {
    uint64_t count = 0;
    double mean = 0.0;
    double squared_error = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / count;
        squared_error += delta * (value - mean);
    }

    double standard_error() const {
        return count > 1
            ? std::sqrt(squared_error / (count - 1) / count)
            : 0.0;
    }
};

State next_root(std::mt19937_64& rng, int ante, int stack_ante) {
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    return sample_fifth_street_root(deck, ante, stack_ante);
}

void ensure_parent(const std::string& path) {
    const std::filesystem::path file(path);
    if (!file.parent_path().empty()) {
        std::filesystem::create_directories(file.parent_path());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        GapExperimentOptions options = parse_gap_options(argc, argv);
        if (options.self_test) {
            options.hands = 20;
            options.report_every = 0;
            options.csv = "br_gap_experiment\\results\\self_test.csv";
            options.model = "br_gap_experiment\\results\\self_test.bin";
        }

        PowerAtlas atlas;
        atlas.load(options.atlas);
        atlas.set_assignment_cache_limit(options.power_cache_max);
        MCCFR cluster(false, options.seed, 5, &atlas);
        cluster.use_cumulative_street_summary(true);
        HeuristicPolicy heuristic;

        ensure_parent(options.csv);
        ensure_parent(options.model);
        std::ofstream csv(options.csv);
        if (!csv) throw std::runtime_error("cannot write CSV: " + options.csv);
        csv << "hand,heuristic_gap_ante,cluster_gap_ante,"
               "heuristic_p0_gap_ante,heuristic_p1_gap_ante,"
               "cluster_p0_gap_ante,cluster_p1_gap_ante,"
               "heuristic_gap_5th,heuristic_gap_6th,heuristic_gap_7th,"
               "cluster_gap_5th,cluster_gap_6th,cluster_gap_7th,"
               "heuristic_node_visits,cluster_node_visits,"
               "training_node_visits,buckets\n";

        std::mt19937_64 train_rng(options.seed ^ 0xa0761d6478bd642full);
        std::mt19937_64 evaluation_rng(options.seed ^ 0xe7037ed1a0b428dbull);
        RunningMean heuristic_mean;
        RunningMean cluster_mean;
        const auto started = std::chrono::steady_clock::now();
        for (uint64_t hand = 1; hand <= options.hands; ++hand) {
            const State training_root = next_root(
                train_rng, options.ante, options.stack_ante);
            cluster.train_root(training_root, 0);
            cluster.train_root(training_root, 1);

            const State evaluation_root = next_root(
                evaluation_rng, options.ante, options.stack_ante);
            const uint64_t gap_seed =
                options.seed ^ (hand * 0x9e3779b97f4a7c15ull);
            const auto heuristic_gap =
                SampledBrGapEstimator<HeuristicPolicy>::evaluate(
                    heuristic, evaluation_root, gap_seed);
            const auto cluster_gap =
                SampledBrGapEstimator<MCCFR>::evaluate(
                    cluster, evaluation_root, gap_seed);
            const double heuristic_value =
                heuristic_gap.exploitability_proxy(options.ante);
            const double cluster_value =
                cluster_gap.exploitability_proxy(options.ante);
            heuristic_mean.add(heuristic_value);
            cluster_mean.add(cluster_value);
            const auto stats = cluster.stats();

            csv << hand << ',' << heuristic_value << ',' << cluster_value
                << ',' << heuristic_gap.positive_deviation_gap[0] / options.ante
                << ',' << heuristic_gap.positive_deviation_gap[1] / options.ante
                << ',' << cluster_gap.positive_deviation_gap[0] / options.ante
                << ',' << cluster_gap.positive_deviation_gap[1] / options.ante;
            for (double value : heuristic_gap.gap_by_street) csv << ',' << value;
            for (double value : cluster_gap.gap_by_street) csv << ',' << value;
            csv << ',' << heuristic_gap.node_visits
                << ',' << cluster_gap.node_visits
                << ',' << cluster.node_visits()
                << ',' << stats.buckets << '\n';

            if (options.report_every &&
                (hand % options.report_every == 0 || hand == options.hands)) {
                csv.flush();
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << std::fixed << std::setprecision(8)
                          << "{\"hands\":" << hand
                          << ",\"heuristic_gap_mean_ante\":"
                          << heuristic_mean.mean
                          << ",\"cluster_gap_mean_ante\":"
                          << cluster_mean.mean
                          << ",\"cluster_gap_ci95\":["
                          << cluster_mean.mean -
                                 1.96 * cluster_mean.standard_error()
                          << ',' << cluster_mean.mean +
                                 1.96 * cluster_mean.standard_error()
                          << "]"
                          << ",\"training_node_visits\":"
                          << cluster.node_visits()
                          << ",\"buckets\":" << stats.buckets
                          << ",\"hands_per_second\":" << hand / elapsed
                          << ",\"elapsed_seconds\":" << elapsed << "}\n";
            }
        }
        csv.close();
        cluster.save(options.model);
        const auto stats = cluster.stats();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << std::fixed << std::setprecision(8)
                  << "{\n"
                  << "  \"metric\": \"sampled-trajectory-relaxed-br-gap\",\n"
                  << "  \"scope\": \"5th+; heuristic H4\",\n"
                  << "  \"hands\": " << options.hands << ",\n"
                  << "  \"heuristic_gap_mean_ante\": "
                  << heuristic_mean.mean << ",\n"
                  << "  \"heuristic_gap_ci95\": ["
                  << heuristic_mean.mean -
                         1.96 * heuristic_mean.standard_error()
                  << ", " << heuristic_mean.mean +
                         1.96 * heuristic_mean.standard_error() << "],\n"
                  << "  \"cluster_gap_mean_ante\": "
                  << cluster_mean.mean << ",\n"
                  << "  \"cluster_gap_ci95\": ["
                  << cluster_mean.mean -
                         1.96 * cluster_mean.standard_error()
                  << ", " << cluster_mean.mean +
                         1.96 * cluster_mean.standard_error() << "],\n"
                  << "  \"training_node_visits\": "
                  << cluster.node_visits() << ",\n"
                  << "  \"buckets\": " << stats.buckets << ",\n"
                  << "  \"csv\": \"" << json_escape(options.csv) << "\",\n"
                  << "  \"model\": \"" << json_escape(options.model) << "\",\n"
                  << "  \"elapsed_seconds\": " << elapsed << "\n"
                  << "}\n";
        if (options.self_test &&
            (!std::isfinite(heuristic_mean.mean) ||
             !std::isfinite(cluster_mean.mean) ||
             heuristic_mean.mean < 0.0 || cluster_mean.mean < 0.0 ||
             stats.buckets == 0)) {
            throw std::runtime_error("BR-gap self-test failed");
        }
        if (options.self_test) std::cerr << "{\"self_test\":\"ok\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
