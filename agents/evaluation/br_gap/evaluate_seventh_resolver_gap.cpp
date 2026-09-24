#define STUD_MCCFR_NO_MAIN
#if __has_include("../../cpp_mccfr/stud_mccfr.cpp")
#include "../../cpp_mccfr/stud_mccfr.cpp"
#else
#include "../cpp_mccfr/stud_mccfr.cpp"  // Legacy directory alias.
#endif

#include "sampled_br_gap.h"

#include <filesystem>
#include <memory>

namespace {

struct ResolverGapOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    uint64_t hands = 10000;
    uint64_t report_every = 1000;
    uint64_t seed = 83101;
    uint64_t infoset_cap = 2000000;
    int iterations = 100;
    int posterior_particles = 0;
    double prior = 100.0;
    int ante = 1000;
    int stack_ante = 1000;
    bool hand_history = true;
    bool self_test = false;
};

ResolverGapOptions parse_resolver_gap_options(int argc, char** argv) {
    ResolverGapOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") options.model = value(index);
        else if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--hands") {
            options.hands = std::stoull(value(index));
        } else if (argument == "--report-every") {
            options.report_every = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--infoset-cap") {
            options.infoset_cap = std::stoull(value(index));
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(value(index));
        } else if (argument == "--posterior-particles") {
            options.posterior_particles = std::stoi(value(index));
        } else if (argument == "--prior") {
            options.prior = std::stod(value(index));
        } else if (argument == "--ante") {
            options.ante = std::stoi(value(index));
        } else if (argument == "--stack-ante") {
            options.stack_ante = std::stoi(value(index));
        } else if (argument == "--no-hand-history") {
            options.hand_history = false;
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (!options.hands || !options.infoset_cap || options.iterations < 0 ||
        options.posterior_particles < 0 ||
        options.prior < 0.0 ||
        options.ante <= 0 || options.stack_ante <= 0) {
        throw std::runtime_error("invalid resolver gap configuration");
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

constexpr size_t kResolverGapHistorySlots = 32;

struct ResolverGapInfoKey {
    uint8_t viewer = 0;
    uint8_t street = 0;
    uint8_t legal_mask = 0;
    uint8_t own_hidden_count = 0;
    uint8_t own_shown_count = 0;
    uint8_t opponent_shown_count = 0;
    uint8_t history_length = 0;
    uint8_t discarded = 0;
    std::array<uint8_t, 3> own_hidden{};
    std::array<uint8_t, 4> own_shown{};
    std::array<uint8_t, 4> opponent_shown{};
    std::array<uint8_t, kResolverGapHistorySlots> history{};

    bool operator==(const ResolverGapInfoKey& other) const {
        return viewer == other.viewer && street == other.street &&
            legal_mask == other.legal_mask &&
            own_hidden_count == other.own_hidden_count &&
            own_shown_count == other.own_shown_count &&
            opponent_shown_count == other.opponent_shown_count &&
            history_length == other.history_length &&
            discarded == other.discarded && own_hidden == other.own_hidden &&
            own_shown == other.own_shown &&
            opponent_shown == other.opponent_shown && history == other.history;
    }
};

struct ResolverGapInfoKeyHash {
    size_t operator()(const ResolverGapInfoKey& key) const {
        size_t hash = 1469598103934665603ull;
        const auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * 1099511628211ull;
        };
        mix(key.viewer);
        mix(key.street);
        mix(key.legal_mask);
        mix(key.own_hidden_count);
        mix(key.own_shown_count);
        mix(key.opponent_shown_count);
        mix(key.history_length);
        mix(key.discarded);
        for (uint8_t value : key.own_hidden) mix(value);
        for (uint8_t value : key.own_shown) mix(value);
        for (uint8_t value : key.opponent_shown) mix(value);
        for (size_t index = 0; index < key.history_length; ++index) {
            mix(key.history[index]);
        }
        return hash;
    }
};

ResolverGapInfoKey resolver_gap_info_key(const State& state, int viewer) {
    ResolverGapInfoKey key;
    key.viewer = static_cast<uint8_t>(viewer);
    key.street = static_cast<uint8_t>(state.street);
    key.legal_mask = valid_mask(state, viewer);
    encode_exact_cards(
        state.players[viewer].hidden,
        true,
        key.own_hidden,
        key.own_hidden_count);
    encode_exact_cards(
        state.players[viewer].shown,
        true,
        key.own_shown,
        key.own_shown_count);
    encode_exact_cards(
        state.players[1 - viewer].shown,
        true,
        key.opponent_shown,
        key.opponent_shown_count);
    if (state.players[viewer].has_discard) {
        key.discarded = card_token(state.players[viewer].discarded);
    }
    if (state.history.size() > key.history.size()) {
        throw std::runtime_error("resolver gap history exceeds capacity");
    }
    key.history_length = static_cast<uint8_t>(state.history.size());
    for (size_t index = 0; index < state.history.size(); ++index) {
        const Event& event = state.history[index];
        const uint8_t relative_actor = event.actor == viewer ? 0 : 1;
        key.history[index] = static_cast<uint8_t>(
            1 + (event.street - 5) * 2 * kActionCount +
            relative_actor * kActionCount + event.action);
    }
    return key;
}

struct ResolverGapEstimate {
    std::array<double, kActionCount> deviation_sum{};
    uint64_t visits = 0;
    uint8_t legal_mask = 0;
    uint8_t street = 0;
};

struct ResolverGapExactSummary {
    double mean_ante = 0.0;
    std::array<double, 2> player_gap_ante{};
    std::array<double, 3> gap_by_street{};
    uint64_t infosets = 0;
    uint64_t positive_infosets = 0;
};

class ResolverGapAccumulator {
public:
    explicit ResolverGapAccumulator(uint64_t infoset_cap)
        : infoset_cap_(infoset_cap) {
        estimates_[0].reserve(1 << 18);
        estimates_[1].reserve(1 << 18);
    }

    void record(
        const State& state,
        int target,
        uint8_t mask,
        const std::array<double, kActionCount>& action_values,
        double value) {
        const ResolverGapInfoKey key = resolver_gap_info_key(state, target);
        auto position = estimates_[target].find(key);
        if (position == estimates_[target].end()) {
            if (infosets() >= infoset_cap_) {
                throw std::runtime_error(
                    "resolver gap infoset cap reached; raise --infoset-cap");
            }
            position = estimates_[target].emplace(
                key, ResolverGapEstimate{}).first;
        }
        ResolverGapEstimate& estimate = position->second;
        estimate.legal_mask = mask;
        estimate.street = static_cast<uint8_t>(state.street);
        ++estimate.visits;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                estimate.deviation_sum[action] += action_values[action] - value;
            }
        }
    }

    void finish_sample() { ++samples_; }

    size_t infosets() const {
        return estimates_[0].size() + estimates_[1].size();
    }

    ResolverGapExactSummary summarize(int ante) const {
        ResolverGapExactSummary summary;
        summary.infosets = infosets();
        if (!samples_) return summary;
        for (int player = 0; player < 2; ++player) {
            for (const auto& [key, estimate] : estimates_[player]) {
                double best = -std::numeric_limits<double>::infinity();
                for (int action = 0; action < kActionCount; ++action) {
                    if (estimate.legal_mask & (1u << action)) {
                        best = std::max(
                            best,
                            estimate.deviation_sum[action] /
                                static_cast<double>(samples_));
                    }
                }
                const double contribution = std::max(0.0, best);
                summary.player_gap_ante[player] += contribution / ante;
                if (estimate.street >= 5 && estimate.street <= 7) {
                    summary.gap_by_street[estimate.street - 5] +=
                        0.5 * contribution / ante;
                }
                summary.positive_infosets += contribution > 0.0;
            }
        }
        summary.mean_ante = 0.5 * (
            summary.player_gap_ante[0] + summary.player_gap_ante[1]);
        return summary;
    }

private:
    std::array<std::unordered_map<
        ResolverGapInfoKey,
        ResolverGapEstimate,
        ResolverGapInfoKeyHash>, 2> estimates_;
    uint64_t samples_ = 0;
    uint64_t infoset_cap_ = 0;
};

class BlueprintGapProfile {
public:
    BlueprintGapProfile(MCCFR& policy, uint64_t infoset_cap)
        : policy_(policy), exact_gap_(infoset_cap) {}

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        return policy_.policy(state, viewer, found);
    }

    void record(
        const State& state,
        int target,
        uint8_t mask,
        const std::array<double, kActionCount>& action_values,
        double value) {
        exact_gap_.record(state, target, mask, action_values, value);
    }

    void finish_sample() { exact_gap_.finish_sample(); }
    ResolverGapExactSummary exact_summary(int ante) const {
        return exact_gap_.summarize(ante);
    }

private:
    MCCFR& policy_;
    ResolverGapAccumulator exact_gap_;
};

class ResolvedSeventhProfile {
public:
    ResolvedSeventhProfile(
        MCCFR& blueprint,
        int iterations,
        double prior,
        bool hand_history,
        uint64_t seed,
        uint64_t infoset_cap,
        int posterior_particles)
        : exact_gap_(infoset_cap) {
        for (int seat = 0; seat < 2; ++seat) {
            seats_[seat] = std::make_unique<ExactSeventhResolverPolicy>(
                blueprint,
                iterations,
                prior,
                hand_history,
                seed ^ ((seat + 1) * 0x9e3779b97f4a7c15ull),
                posterior_particles);
        }
    }

    void begin_subgame(const State& state) {
        for (int seat = 0; seat < 2; ++seat) {
            seats_[seat]->begin_subgame(state, seat);
        }
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        return seats_[viewer]->policy(state, viewer, found);
    }

    ExactSeventhResolverPolicy::Stats stats() const {
        ExactSeventhResolverPolicy::Stats result;
        for (const auto& seat : seats_) {
            const auto local = seat->stats();
            result.subgames += local.subgames;
            result.traversals += local.traversals;
            result.node_visits += local.node_visits;
            result.nodes_created += local.nodes_created;
            result.blueprint_fallbacks += local.blueprint_fallbacks;
            result.posterior_builds += local.posterior_builds;
            result.posterior_proposals += local.posterior_proposals;
            result.posterior_policy_queries += local.posterior_policy_queries;
            result.posterior_policy_misses += local.posterior_policy_misses;
            result.posterior_fallbacks += local.posterior_fallbacks;
            result.posterior_effective_particles_sum +=
                local.posterior_effective_particles_sum;
            result.maximum_local_nodes = std::max(
                result.maximum_local_nodes, local.maximum_local_nodes);
        }
        return result;
    }

    void record(
        const State& state,
        int target,
        uint8_t mask,
        const std::array<double, kActionCount>& action_values,
        double value) {
        exact_gap_.record(state, target, mask, action_values, value);
    }

    void finish_sample() { exact_gap_.finish_sample(); }
    ResolverGapExactSummary exact_summary(int ante) const {
        return exact_gap_.summarize(ante);
    }

private:
    std::array<std::unique_ptr<ExactSeventhResolverPolicy>, 2> seats_;
    ResolverGapAccumulator exact_gap_;
};

State next_root(std::mt19937_64& rng, int ante, int stack_ante) {
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    return sample_fifth_street_root(deck, ante, stack_ante);
}

}  // namespace

template <>
struct SampledBrGapSubgameHook<ResolvedSeventhProfile> {
    static void begin(ResolvedSeventhProfile& policy, const State& state) {
        if (state.street == 7 && !state.terminal) policy.begin_subgame(state);
    }
};

template <>
struct SampledBrGapInfoHook<BlueprintGapProfile> {
    static void record(
        BlueprintGapProfile& policy,
        const State& state,
        int target,
        uint8_t mask,
        const std::array<double, kActionCount>& action_values,
        double value) {
        policy.record(state, target, mask, action_values, value);
    }
};

template <>
struct SampledBrGapInfoHook<ResolvedSeventhProfile> {
    static void record(
        ResolvedSeventhProfile& policy,
        const State& state,
        int target,
        uint8_t mask,
        const std::array<double, kActionCount>& action_values,
        double value) {
        policy.record(state, target, mask, action_values, value);
    }
};

int main(int argc, char** argv) {
    try {
        ResolverGapOptions options = parse_resolver_gap_options(argc, argv);
        if (options.self_test) {
            options.hands = 10;
            options.report_every = 0;
            options.iterations = 2;
        }
        PowerAtlas atlas;
        atlas.load(options.atlas);
        MCCFR blueprint(false, options.seed, 5, &atlas);
        blueprint.use_cumulative_street_summary(true);
        blueprint.load(options.model);
        BlueprintGapProfile blueprint_profile(
            blueprint, options.infoset_cap);
        ResolvedSeventhProfile resolved(
            blueprint,
            options.iterations,
            options.prior,
            options.hand_history,
            options.seed ^ 0xd1b54a32d192ed03ull,
            options.infoset_cap,
            options.posterior_particles);

        RunningMean blueprint_mean;
        RunningMean resolved_mean;
        RunningMean difference_mean;
        std::array<double, 3> blueprint_street{};
        std::array<double, 3> resolved_street{};
        uint64_t blueprint_nodes = 0;
        uint64_t resolved_nodes = 0;
        std::mt19937_64 deal_rng(options.seed);
        const auto started = std::chrono::steady_clock::now();
        for (uint64_t hand = 1; hand <= options.hands; ++hand) {
            const State root = next_root(
                deal_rng, options.ante, options.stack_ante);
            const uint64_t gap_seed =
                options.seed ^ (hand * 0xa0761d6478bd642full);
            const auto base = SampledBrGapEstimator<BlueprintGapProfile>::evaluate(
                blueprint_profile, root, gap_seed);
            const auto local =
                SampledBrGapEstimator<ResolvedSeventhProfile>::evaluate(
                    resolved, root, gap_seed);
            blueprint_profile.finish_sample();
            resolved.finish_sample();
            const double blueprint_gap =
                base.exploitability_proxy(options.ante);
            const double resolved_gap =
                local.exploitability_proxy(options.ante);
            blueprint_mean.add(blueprint_gap);
            resolved_mean.add(resolved_gap);
            difference_mean.add(resolved_gap - blueprint_gap);
            blueprint_nodes += base.node_visits;
            resolved_nodes += local.node_visits;
            for (int street = 0; street < 3; ++street) {
                blueprint_street[street] += base.gap_by_street[street];
                resolved_street[street] += local.gap_by_street[street];
            }
            if (options.report_every &&
                (hand % options.report_every == 0 || hand == options.hands)) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << "{\"hands\":" << hand
                          << ",\"blueprint_gap_ante\":"
                          << blueprint_mean.mean
                          << ",\"resolved_gap_ante\":"
                          << resolved_mean.mean
                          << ",\"resolved_minus_blueprint_ante\":"
                          << difference_mean.mean
                          << ",\"elapsed_seconds\":" << elapsed << "}\n";
            }
        }
        const auto stats = resolved.stats();
        const auto blueprint_exact = blueprint_profile.exact_summary(options.ante);
        const auto resolved_exact = resolved.exact_summary(options.ante);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << std::fixed << std::setprecision(8)
                  << "{\n"
                  << "  \"metric\": \"sampled-br-gap-comparison\",\n"
                  << "  \"hands\": " << options.hands << ",\n"
                  << "  \"resolver_iterations\": " << options.iterations
                  << ",\n"
                  << "  \"resolver_prior\": " << options.prior << ",\n"
                  << "  \"posterior_particles\": "
                  << options.posterior_particles << ",\n"
                  << "  \"hand_history\": "
                  << (options.hand_history ? "true" : "false") << ",\n"
                  << "  \"blueprint_gap_mean_ante\": "
                  << blueprint_mean.mean << ",\n"
                  << "  \"blueprint_gap_ci95\": ["
                  << blueprint_mean.mean - 1.96 * blueprint_mean.standard_error()
                  << ", "
                  << blueprint_mean.mean + 1.96 * blueprint_mean.standard_error()
                  << "],\n"
                  << "  \"resolved_gap_mean_ante\": "
                  << resolved_mean.mean << ",\n"
                  << "  \"resolved_gap_ci95\": ["
                  << resolved_mean.mean - 1.96 * resolved_mean.standard_error()
                  << ", "
                  << resolved_mean.mean + 1.96 * resolved_mean.standard_error()
                  << "],\n"
                  << "  \"resolved_minus_blueprint_mean_ante\": "
                  << difference_mean.mean << ",\n"
                  << "  \"resolved_minus_blueprint_ci95\": ["
                  << difference_mean.mean - 1.96 * difference_mean.standard_error()
                  << ", "
                  << difference_mean.mean + 1.96 * difference_mean.standard_error()
                  << "],\n"
                  << "  \"exact_infoset_plugin\": {\n"
                  << "    \"blueprint_gap_mean_ante\": "
                  << blueprint_exact.mean_ante << ",\n"
                  << "    \"resolved_gap_mean_ante\": "
                  << resolved_exact.mean_ante << ",\n"
                  << "    \"resolved_minus_blueprint_ante\": "
                  << resolved_exact.mean_ante - blueprint_exact.mean_ante
                  << ",\n"
                  << "    \"blueprint_infosets\": "
                  << blueprint_exact.infosets << ",\n"
                  << "    \"resolved_infosets\": "
                  << resolved_exact.infosets << ",\n"
                  << "    \"blueprint_positive_infosets\": "
                  << blueprint_exact.positive_infosets << ",\n"
                  << "    \"resolved_positive_infosets\": "
                  << resolved_exact.positive_infosets << ",\n"
                  << "    \"blueprint_gap_by_street\": ["
                  << blueprint_exact.gap_by_street[0] << ", "
                  << blueprint_exact.gap_by_street[1] << ", "
                  << blueprint_exact.gap_by_street[2] << "],\n"
                  << "    \"resolved_gap_by_street\": ["
                  << resolved_exact.gap_by_street[0] << ", "
                  << resolved_exact.gap_by_street[1] << ", "
                  << resolved_exact.gap_by_street[2] << "]\n"
                  << "  },\n"
                  << "  \"blueprint_gap_by_street\": ["
                  << blueprint_street[0] / options.hands << ", "
                  << blueprint_street[1] / options.hands << ", "
                  << blueprint_street[2] / options.hands << "],\n"
                  << "  \"resolved_gap_by_street\": ["
                  << resolved_street[0] / options.hands << ", "
                  << resolved_street[1] / options.hands << ", "
                  << resolved_street[2] / options.hands << "],\n"
                  << "  \"resolver\": {\"subgames\":" << stats.subgames
                  << ",\"traversals\":" << stats.traversals
                  << ",\"node_visits\":" << stats.node_visits
                  << ",\"nodes_created\":" << stats.nodes_created
                  << ",\"maximum_local_nodes\":"
                  << stats.maximum_local_nodes
                  << ",\"blueprint_fallbacks\":"
                  << stats.blueprint_fallbacks
                  << ",\"posterior_builds\":" << stats.posterior_builds
                  << ",\"posterior_proposals\":"
                  << stats.posterior_proposals
                  << ",\"posterior_policy_queries\":"
                  << stats.posterior_policy_queries
                  << ",\"posterior_policy_misses\":"
                  << stats.posterior_policy_misses
                  << ",\"posterior_fallbacks\":"
                  << stats.posterior_fallbacks
                  << ",\"average_effective_posterior_particles\":"
                  << (stats.posterior_builds
                        ? stats.posterior_effective_particles_sum /
                            stats.posterior_builds
                        : 0.0)
                  << "},\n"
                  << "  \"blueprint_eval_nodes\": " << blueprint_nodes
                  << ",\n"
                  << "  \"resolved_eval_nodes\": " << resolved_nodes
                  << ",\n"
                  << "  \"elapsed_seconds\": " << elapsed << "\n}\n";
        if (options.self_test &&
            (!std::isfinite(blueprint_mean.mean) ||
             !std::isfinite(resolved_mean.mean) || !stats.subgames ||
             (options.posterior_particles > 0 &&
              (!stats.posterior_builds || stats.posterior_fallbacks)))) {
            throw std::runtime_error("resolver gap self-test failed");
        }
        if (options.self_test) std::cerr << "{\"self_test\":\"ok\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
