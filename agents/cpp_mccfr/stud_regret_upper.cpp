#define STUD_MCCFR_NO_MAIN
#include "stud_mccfr.cpp"

namespace {

constexpr size_t kUpperHistorySlots = 32;

struct OriginalInfoKey {
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
    std::array<uint8_t, kUpperHistorySlots> history{};

    bool operator==(const OriginalInfoKey& other) const {
        return viewer == other.viewer &&
            street == other.street &&
            legal_mask == other.legal_mask &&
            own_hidden_count == other.own_hidden_count &&
            own_shown_count == other.own_shown_count &&
            opponent_shown_count == other.opponent_shown_count &&
            history_length == other.history_length &&
            discarded == other.discarded &&
            own_hidden == other.own_hidden &&
            own_shown == other.own_shown &&
            opponent_shown == other.opponent_shown &&
            history == other.history;
    }
};

struct OriginalInfoKeyHash {
    size_t operator()(const OriginalInfoKey& key) const {
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

OriginalInfoKey original_info_key(const State& state, int viewer) {
    OriginalInfoKey key;
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
        throw std::runtime_error("upper-bound history exceeds capacity");
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

struct RegretEstimate {
    std::array<double, kActionCount> sum{};
    uint64_t visits = 0;
    uint8_t legal_mask = 0;
    uint8_t street = 0;
};

struct PlayerUpperSummary {
    double bound = 0.0;
    uint64_t infosets = 0;
    uint64_t positive_infosets = 0;
    double maximum_contribution = 0.0;
    std::array<uint64_t, 5> infosets_by_visits{};
    std::array<double, 5> bound_by_visits{};
    std::array<double, 3> bound_by_street{};
};

struct RunningEstimate {
    uint64_t count = 0;
    double mean = 0.0;
    double squared_error = 0.0;

    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        squared_error += delta * (value - mean);
    }

    double standard_error() const {
        if (count < 2) return 0.0;
        return std::sqrt(
            squared_error / static_cast<double>(count - 1) /
            static_cast<double>(count));
    }
};

class RegretUpperEstimator {
public:
    RegretUpperEstimator(MCCFR& policy, uint64_t seed, uint64_t infoset_cap)
        : policy_(policy), rng_(seed), infoset_cap_(infoset_cap) {
        estimates_[0].reserve(1 << 18);
        estimates_[1].reserve(1 << 18);
        bucket_estimates_[0].reserve(1 << 14);
        bucket_estimates_[1].reserve(1 << 14);
    }

    void sample(const State& root) {
        current_relaxed_ = 0.0;
        traverse(root, 0);
        const double player_0 = current_relaxed_;
        relaxed_[0].add(player_0);
        current_relaxed_ = 0.0;
        traverse(root, 1);
        const double player_1 = current_relaxed_;
        relaxed_[1].add(player_1);
        relaxed_total_.add(player_0 + player_1);
        ++samples_;
    }

    PlayerUpperSummary summarize(int player) const {
        return summarize_map(estimates_[player]);
    }

    PlayerUpperSummary summarize_buckets(int player) const {
        return summarize_map(bucket_estimates_[player]);
    }

    template <class Map>
    PlayerUpperSummary summarize_map(const Map& estimates) const {
        PlayerUpperSummary summary;
        summary.infosets = estimates.size();
        for (const auto& [key, estimate] : estimates) {
            double best = -std::numeric_limits<double>::infinity();
            for (int action = 0; action < kActionCount; ++action) {
                if (estimate.legal_mask & (1u << action)) {
                    best = std::max(
                        best, estimate.sum[action] / samples_);
                }
            }
            const double contribution = std::max(0.0, best);
            const int visit_bin = estimate.visits == 1 ? 0
                : estimate.visits <= 4 ? 1
                : estimate.visits <= 16 ? 2
                : estimate.visits <= 64 ? 3
                : 4;
            ++summary.infosets_by_visits[visit_bin];
            summary.bound_by_visits[visit_bin] += contribution;
            if (estimate.street >= 5 && estimate.street <= 7) {
                summary.bound_by_street[estimate.street - 5] += contribution;
            }
            summary.bound += contribution;
            summary.positive_infosets += contribution > 0.0;
            summary.maximum_contribution = std::max(
                summary.maximum_contribution, contribution);
        }
        return summary;
    }

    uint64_t samples() const { return samples_; }
    uint64_t node_visits() const { return node_visits_; }
    uint64_t policy_queries() const { return policy_queries_; }
    uint64_t policy_misses() const { return policy_misses_; }
    const RunningEstimate& relaxed(int player) const {
        return relaxed_[player];
    }
    const RunningEstimate& relaxed_total() const { return relaxed_total_; }
    size_t infosets() const {
        return estimates_[0].size() + estimates_[1].size();
    }
    size_t bucket_infosets() const {
        return bucket_estimates_[0].size() + bucket_estimates_[1].size();
    }

private:
    void advance(State& state) {
        if (state.street == 7) {
            state.terminal = true;
            return;
        }
        ++state.street;
        const bool public_card = state.street != 7;
        for (int seat = 0; seat < 2; ++seat) {
            if (state.simulation_deck.empty()) {
                throw std::runtime_error("upper-bound simulation deck exhausted");
            }
            const Card card = state.simulation_deck.back();
            state.simulation_deck.pop_back();
            if (public_card) state.players[seat].shown.push_back(card);
            else state.players[seat].hidden.push_back(card);
            state.players[seat].round_bet = 0;
        }
        state.highest_bet = 0;
        state.raise_count = 0;
        if (state.players[0].all_in || state.players[1].all_in) {
            advance(state);
            return;
        }
        state.actor = first_bettor(state);
    }

    State child(State state, int actor, Action action) {
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd) advance(state);
        else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
        return state;
    }

    Action sample_action(
        const std::array<double, kActionCount>& strategy,
        uint8_t mask) {
        const double threshold =
            std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
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

    double traverse(State state, int target) {
        ++node_visits_;
        if (state.terminal) return terminal_net_search(state, target);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        bool found = false;
        const auto strategy = policy_.policy(state, actor, &found);
        ++policy_queries_;
        policy_misses_ += !found;

        if (actor != target) {
            const Action action = sample_action(strategy, mask);
            return traverse(child(std::move(state), actor, action), target);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            action_values[action] = traverse(child(state, actor, action), target);
        }
        double value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                value += strategy[action] * action_values[action];
            }
        }
        double local_best = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                local_best = std::max(
                    local_best, action_values[action] - value);
            }
        }
        current_relaxed_ += local_best;
        const OriginalInfoKey key = original_info_key(state, actor);
        auto position = estimates_[target].find(key);
        if (position == estimates_[target].end()) {
            if (infosets() >= infoset_cap_) {
                throw std::runtime_error(
                    "original infoset cap reached; lower --samples or raise "
                    "--infoset-cap deliberately");
            }
            position = estimates_[target].emplace(
                key, RegretEstimate{}).first;
        }
        RegretEstimate& estimate = position->second;
        estimate.legal_mask = mask;
        estimate.street = static_cast<uint8_t>(state.street);
        ++estimate.visits;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                estimate.sum[action] += action_values[action] - value;
            }
        }
        const InfoKey bucket_key = policy_.information_key(state, actor);
        RegretEstimate& bucket = bucket_estimates_[target][bucket_key];
        bucket.legal_mask = mask;
        bucket.street = static_cast<uint8_t>(state.street);
        ++bucket.visits;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                bucket.sum[action] += action_values[action] - value;
            }
        }
        return value;
    }

    MCCFR& policy_;
    std::mt19937_64 rng_;
    std::array<
        std::unordered_map<OriginalInfoKey, RegretEstimate, OriginalInfoKeyHash>,
        2> estimates_;
    std::array<
        std::unordered_map<InfoKey, RegretEstimate, InfoKeyHash>,
        2> bucket_estimates_;
    uint64_t samples_ = 0;
    uint64_t node_visits_ = 0;
    uint64_t policy_queries_ = 0;
    uint64_t policy_misses_ = 0;
    double current_relaxed_ = 0.0;
    std::array<RunningEstimate, 2> relaxed_{};
    RunningEstimate relaxed_total_{};
    uint64_t infoset_cap_ = 0;
};

struct UpperOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string bucket = "power-memory16";
    uint64_t samples = 10000;
    uint64_t report_every = 1000;
    uint64_t seed = 72001;
    uint64_t power_cache_max = 250000;
    uint64_t infoset_cap = 2000000;
    int ante = 1000;
    int stack_ante = 1000;
    bool self_test = false;
};

UpperOptions parse_upper_options(int argc, char** argv) {
    UpperOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") options.model = value(index);
        else if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--bucket") options.bucket = value(index);
        else if (argument == "--samples") {
            options.samples = std::stoull(value(index));
        } else if (argument == "--report-every") {
            options.report_every = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--power-cache-max") {
            options.power_cache_max = std::stoull(value(index));
        } else if (argument == "--infoset-cap") {
            options.infoset_cap = std::stoull(value(index));
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
    if (!options.samples || !options.infoset_cap ||
        options.ante <= 0 || options.stack_ante <= 0) {
        throw std::runtime_error("invalid upper-bound configuration");
    }
    if (options.bucket != "power" &&
        options.bucket != "power-tree" &&
        options.bucket != "power-memory16") {
        throw std::runtime_error("unsupported upper-bound bucket mode");
    }
    return options;
}

void print_player_summary(
    std::ostream& output,
    const PlayerUpperSummary& summary,
    const RunningEstimate& relaxed,
    double ante_scale) {
    output << "{\"original_infoset_plugin_estimate_ante\":"
           << summary.bound * ante_scale
           << ",\"infosets\":" << summary.infosets
           << ",\"positive_infosets\":" << summary.positive_infosets
           << ",\"maximum_contribution_ante\":"
           << summary.maximum_contribution * ante_scale
           << ",\"infosets_by_visits\":[";
    for (size_t index = 0; index < summary.infosets_by_visits.size(); ++index) {
        if (index) output << ',';
        output << summary.infosets_by_visits[index];
    }
    output << "],\"bound_by_visits\":[";
    for (size_t index = 0; index < summary.bound_by_visits.size(); ++index) {
        if (index) output << ',';
        output << summary.bound_by_visits[index] * ante_scale;
    }
    output << "],\"bound_by_street_ante\":["
           << summary.bound_by_street[0] * ante_scale << ','
           << summary.bound_by_street[1] * ante_scale << ','
           << summary.bound_by_street[2] * ante_scale
           << "],\"trajectory_relaxed_upper_mean_ante\":"
           << relaxed.mean * ante_scale
           << ",\"trajectory_relaxed_standard_error_ante\":"
           << relaxed.standard_error() * ante_scale
           << ",\"trajectory_relaxed_ci95_upper_ante\":"
           << (relaxed.mean + 1.96 * relaxed.standard_error()) * ante_scale
           << '}';
}

void print_bucket_summary(
    std::ostream& output,
    const PlayerUpperSummary& summary,
    double ante_scale) {
    output << "{\"plugin_estimate_ante\":" << summary.bound * ante_scale
           << ",\"buckets\":" << summary.infosets
           << ",\"positive_buckets\":" << summary.positive_infosets
           << ",\"maximum_contribution_ante\":"
           << summary.maximum_contribution * ante_scale
           << ",\"buckets_by_visits\":[";
    for (size_t index = 0; index < summary.infosets_by_visits.size(); ++index) {
        if (index) output << ',';
        output << summary.infosets_by_visits[index];
    }
    output << "],\"bound_by_visits_ante\":[";
    for (size_t index = 0; index < summary.bound_by_visits.size(); ++index) {
        if (index) output << ',';
        output << summary.bound_by_visits[index] * ante_scale;
    }
    output
           << "],\"bound_by_street_ante\":["
           << summary.bound_by_street[0] * ante_scale << ','
           << summary.bound_by_street[1] * ante_scale << ','
           << summary.bound_by_street[2] * ante_scale << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        UpperOptions options = parse_upper_options(argc, argv);
        if (options.self_test) {
            options.samples = 100;
            options.report_every = 0;
        }
        PowerAtlas atlas;
        atlas.load(options.atlas);
        atlas.set_assignment_cache_limit(options.power_cache_max);
        MCCFR policy(false, options.seed, 5, &atlas);
        policy.use_cumulative_street_summary(
            options.bucket == "power-memory16");
        policy.load(options.model);
        RegretUpperEstimator estimator(
            policy,
            options.seed ^ 0x6a09e667f3bcc909ull,
            options.infoset_cap);
        std::mt19937_64 deal_rng(options.seed);
        const auto started = std::chrono::steady_clock::now();
        for (uint64_t sample = 1; sample <= options.samples; ++sample) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            estimator.sample(sample_fifth_street_root(
                deck, options.ante, options.stack_ante));
            if (options.report_every &&
                (sample % options.report_every == 0 ||
                 sample == options.samples)) {
                const PlayerUpperSummary p0 = estimator.summarize(0);
                const PlayerUpperSummary p1 = estimator.summarize(1);
                const PlayerUpperSummary b0 = estimator.summarize_buckets(0);
                const PlayerUpperSummary b1 = estimator.summarize_buckets(1);
                const double ante_scale = 1.0 / options.ante;
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << "{\"samples\":" << sample
                          << ",\"infosets\":" << estimator.infosets()
                          << ",\"bucket_infosets\":"
                          << estimator.bucket_infosets()
                          << ",\"player_plugin_estimates_ante\":["
                          << p0.bound * ante_scale << ','
                          << p1.bound * ante_scale << ']'
                          << ",\"exploitability_plugin_estimate_ante\":"
                          << 0.5 * (p0.bound + p1.bound) * ante_scale
                          << ",\"bucket_exploitability_plugin_estimate_ante\":"
                          << 0.5 * (b0.bound + b1.bound) * ante_scale
                          << ",\"trajectory_relaxed_nash_conv_mean_ante\":"
                          << estimator.relaxed_total().mean * ante_scale
                          << ",\"node_visits\":" << estimator.node_visits()
                          << ",\"elapsed_seconds\":" << seconds << "}\n";
            }
        }
        const PlayerUpperSummary p0 = estimator.summarize(0);
        const PlayerUpperSummary p1 = estimator.summarize(1);
        const PlayerUpperSummary b0 = estimator.summarize_buckets(0);
        const PlayerUpperSummary b1 = estimator.summarize_buckets(1);
        const double ante_scale = 1.0 / options.ante;
        const PowerCacheStats cache = atlas.assignment_cache_stats();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << std::fixed << std::setprecision(8)
                  << "{\n  \"estimator\": "
                     "\"t1-counterfactual-regret-upper\",\n"
                  << "  \"scope\": \"5th+; heuristic H4; exact original "
                     "infoset accounting; external chance/opponent "
                     "sampling\",\n"
                  << "  \"samples\": " << estimator.samples() << ",\n"
                  << "  \"player_0\": ";
        print_player_summary(
            std::cout, p0, estimator.relaxed(0), ante_scale);
        std::cout << ",\n  \"player_1\": ";
        print_player_summary(
            std::cout, p1, estimator.relaxed(1), ante_scale);
        std::cout << ",\n  \"bucket_player_0\": ";
        print_bucket_summary(std::cout, b0, ante_scale);
        std::cout << ",\n  \"bucket_player_1\": ";
        print_bucket_summary(std::cout, b1, ante_scale);
        const RunningEstimate& relaxed_total = estimator.relaxed_total();
        std::cout << ",\n  \"nash_conv_plugin_estimate_ante\": "
                  << (p0.bound + p1.bound) * ante_scale << ",\n"
                  << "  \"exploitability_plugin_estimate_ante\": "
                  << 0.5 * (p0.bound + p1.bound) * ante_scale << ",\n"
                  << "  \"bucket_exploitability_plugin_estimate_ante\": "
                  << 0.5 * (b0.bound + b1.bound) * ante_scale << ",\n"
                  << "  \"trajectory_relaxed_nash_conv_mean_ante\": "
                  << relaxed_total.mean * ante_scale << ",\n"
                  << "  \"trajectory_relaxed_nash_conv_standard_error_ante\": "
                  << relaxed_total.standard_error() * ante_scale << ",\n"
                  << "  \"trajectory_relaxed_nash_conv_ci95_upper_ante\": "
                  << (relaxed_total.mean +
                         1.96 * relaxed_total.standard_error()) * ante_scale
                  << ",\n"
                  << "  \"trajectory_relaxed_exploitability_ci95_upper_ante\": "
                  << 0.5 * (relaxed_total.mean +
                         1.96 * relaxed_total.standard_error()) * ante_scale
                  << ",\n"
                  << "  \"node_visits\": " << estimator.node_visits() << ",\n"
                  << "  \"policy_queries\": " << estimator.policy_queries()
                  << ",\n  \"policy_misses\": " << estimator.policy_misses()
                  << ",\n  \"power_cache\": {\"entries\":" << cache.entries
                  << ",\"limit\":" << cache.limit
                  << ",\"resets\":" << cache.resets << "},\n"
                  << "  \"elapsed_seconds\": " << elapsed << "\n}\n";
        if (options.self_test) {
            if (!std::isfinite(p0.bound) || !std::isfinite(p1.bound) ||
                !std::isfinite(b0.bound) || !std::isfinite(b1.bound) ||
                p0.bound < 0.0 || p1.bound < 0.0 ||
                b0.bound < 0.0 || b1.bound < 0.0 ||
                !p0.infosets || !p1.infosets ||
                !b0.infosets || !b1.infosets) {
                throw std::runtime_error("upper-bound self-test failed");
            }
            std::cerr << "{\"self_test\":\"ok\"}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
