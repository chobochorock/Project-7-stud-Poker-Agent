// Play frozen policies against the existing posterior-aware PolicyLBR.
// Build: g++ -O3 -std=c++17 -DNOMINMAX -static-libstdc++ -static-libgcc
//   evaluate_epoch_lbr.cpp -o bin/evaluate_epoch_lbr.exe -lpsapi
// Usage: exe RUN_DIR ATLAS NEW_OUT_DIR PAIRS PARTICLES SEED HANDS EPOCH_HANDS
// Test: exe --self-test RUN_DIR ATLAS HANDS EPOCH_HANDS
// Each pair is one deal played in both seats. Profit is LBR ante/hand;
// no local max-Q estimates are reused as observed payoff. See README.md.
// Leaf-format checkpoints additionally evaluate the ensemble average policy.
#define main epoch_training_entrypoint
#include "stud_epoch_ensemble.cpp"
#undef main

namespace {
constexpr int kPolicies = 5;
const std::array<const char*, kPolicies> names{
    "ensemble_current", "hard256_current", "hard256_average", "uniform", "ensemble_average"};

struct FrozenTarget {
    epoch7::Ensemble& ensemble;
    MCCFR& baseline;
    int kind;
    std::mt19937_64 rng;

    epoch7::Values policy(const State& state, int viewer, bool* found = nullptr) const {
        epoch7::Values result;
        if (kind == 0) result = ensemble.policy(state, viewer, found);
        else if (kind == 1) result = baseline.instantaneous_policy(state, viewer, found);
        else if (kind == 2) result = baseline.policy(state, viewer, found);
        else if (kind == 4) result = ensemble.average_policy(state, viewer, found);
        else {
            if (found) *found = true;
            result = uniform_strategy(valid_mask(state, viewer));
        }
        const auto mask = valid_mask(state, viewer);
        double total = 0;
        for (int a = 0; a < kActionCount; ++a) {
            if (!std::isfinite(result[a]) || result[a] < 0 ||
                (!(mask & (1u << a)) && result[a] != 0)) {
                throw std::runtime_error("invalid frozen target policy");
            }
            total += result[a];
        }
        if (std::abs(total - 1) > 1e-8) throw std::runtime_error("policy not normalized");
        return result;
    }

    Action choose(const State& state, int viewer, int) {
        return epoch7::sample(policy(state, viewer), valid_mask(state, viewer), rng);
    }
};

using LBR = PolicyLBR<FrozenTarget>;

void add_stats(LBR::Stats& total, const LBR::Stats& next) {
    const auto count = total.decisions + next.decisions;
    total.average_effective_particles = count ?
        (total.average_effective_particles * total.decisions +
         next.average_effective_particles * next.decisions) / count : 0;
    total.decisions = count;
    total.policy_queries += next.policy_queries;
    total.policy_misses += next.policy_misses;
    total.h4_particle_proposals += next.h4_particle_proposals;
    total.h4_particle_rejections += next.h4_particle_rejections;
    for (int a = 0; a < kActionCount; ++a) total.actions[a] += next.actions[a];
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void self_test(epoch7::Ensemble& ensemble, MCCFR& baseline, int policy_count) {
    for (int kind = 0; kind < policy_count; ++kind) {
        FrozenTarget target{ensemble, baseline, kind, std::mt19937_64(21)};
        State state = epoch7::root_for(381);
        for (int step = 0; !state.terminal && step < 24; ++step) {
            const int seat = state.actor;
            State masked = state;
            masked.players[1 - seat].hidden.clear();
            masked.players[1 - seat].has_discard = false;
            masked.simulation_deck.clear();
            check(target.policy(state, seat) == target.policy(masked, seat),
                "target depends on unobserved cards");
            LBR first(target, 16, 41), second(target, 16, 41);
            check(first.choose(state, seat) == second.choose(masked, seat),
                "LBR depends on unobserved cards");
            const auto actions = actions_from_mask(valid_mask(state, seat));
            const Action action = std::find(actions.begin(), actions.end(), CALL) != actions.end()
                ? CALL : CHECK;
            state = epoch7::child(state, action);
        }
        auto deck = fresh_deck();
        std::mt19937_64 rng(418); std::shuffle(deck.begin(), deck.end(), rng);
        LBR first(target, 16, 415);
        target.rng.seed(91);
        const auto value = play_hand_policy_lbr(deck, 0, target, first, 1000, 1000);
        LBR second(target, 16, 415);
        target.rng.seed(91);
        check(value == play_hand_policy_lbr(deck, 0, target, second, 1000, 1000),
            "evaluation is not reproducible");
        check(std::isfinite(value) && std::abs(value) <= 1000,
            "bad ante-normalized payoff");
    }
    epoch7::EpochModel missing; missing.weight = 1;
    bool found = true;
    missing.query(0, {}, &found);
    check(!found, "missing bucket reported as found");
    // Seat-paired LBR takes both antes from an always-fold target, one per hand.
    ConditionalParticipationPolicy folds("fold", 1);
    PolicyLBR<ConditionalParticipationPolicy> lbr(folds, 16, 61);
    double pair = 0;
    for (int seat = 0; seat < 2; ++seat) {
        pair += play_hand_policy_lbr(fresh_deck(), seat, folds, lbr, 1000, 1000);
    }
    check(std::abs(pair / 2 - 1) < 1e-9, "payoff unit/seat check failed");
    std::cout << "self-test passed: legal policies, privacy, reproducibility, ante units\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool testing = argc >= 2 && std::string(argv[1]) == "--self-test";
        if ((testing && argc != 6) || (!testing && argc != 9)) throw std::runtime_error(
            "usage: exe RUN_DIR ATLAS NEW_OUT_DIR PAIRS PARTICLES SEED HANDS EPOCH_HANDS; "
            "or exe --self-test RUN_DIR ATLAS HANDS EPOCH_HANDS");
        const std::filesystem::path run(argv[testing ? 2 : 1]);
        const std::string atlas_path(argv[testing ? 3 : 2]);
        const int pairs = testing ? 1 : std::stoi(argv[4]);
        const int particles = testing ? 16 : std::stoi(argv[5]);
        const uint64_t seed = testing ? 21 : std::stoull(argv[6]);
        const uint64_t hands = std::stoull(argv[testing ? 4 : 7]);
        const uint64_t epoch_hands = std::stoull(argv[testing ? 5 : 8]);
        if (pairs < 2 || particles < 1) {
            if (!testing) throw std::runtime_error("PAIRS >= 2 and PARTICLES >= 1 required");
        }
        PowerAtlas atlas; atlas.load(atlas_path); atlas.set_assignment_cache_limit(100000);
        epoch7::FeatureCache features{atlas}; epoch7::Ensemble ensemble(features, seed);
        MCCFR baseline(false, seed, 5, &atlas);
        baseline.use_cumulative_street_summary(true);
        epoch7::load_frozen_checkpoint(run, ensemble, baseline, hands, epoch_hands);
        const bool has_average = std::all_of(ensemble.models.begin(), ensemble.models.end(),
            [](const epoch7::EpochModel& model) { return model.leafwise; });
        const int policy_count = has_average ? kPolicies : kPolicies - 1;
        if (testing) { self_test(ensemble, baseline, policy_count); return 0; }

        const std::filesystem::path output(argv[3]);
        if (std::filesystem::exists(output)) throw std::runtime_error("output already exists");
        std::filesystem::create_directories(output);
        std::ofstream csv(output / "paired_payoffs.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << std::setprecision(12) << "pair,deal_seed";
        for (int k = 0; k < policy_count; ++k) {
            csv << ',' << names[k] << "_seat0," << names[k] << "_seat1";
        }
        csv << '\n';
        std::array<LBR::Stats, kPolicies> stats{};
        std::array<double, kPolicies> sums{}, seconds{};
        for (int index = 0; index < pairs; ++index) {
            const uint64_t deal_seed = seed ^ (uint64_t(index + 1) * 0xd1b54a32d192ed03ull);
            std::mt19937_64 deal_rng(deal_seed);
            auto deck = fresh_deck(); std::shuffle(deck.begin(), deck.end(), deal_rng);
            csv << index << ',' << deal_seed;
            for (int kind = 0; kind < policy_count; ++kind) {
                const auto start = std::chrono::steady_clock::now();
                FrozenTarget target{ensemble, baseline, kind, std::mt19937_64(0)};
                for (int seat = 0; seat < 2; ++seat) {
                    const uint64_t action_seed = deal_seed ^ (uint64_t(seat + 1) * 0x94d049bb133111ebull);
                    target.rng.seed(action_seed);
                    LBR lbr(target, particles, action_seed ^ 0xabcdef012345ull);
                    const double payoff = play_hand_policy_lbr(deck, seat, target, lbr, 1000, 1000);
                    check(std::isfinite(payoff) && std::abs(payoff) <= 1000, "invalid LBR payoff");
                    csv << ',' << payoff;
                    sums[kind] += payoff / 2;
                    add_stats(stats[kind], lbr.stats());
                }
                seconds[kind] += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
            }
            csv << '\n';
            if ((index + 1) % 25 == 0 || index + 1 == pairs) {
                csv.flush();
                std::cout << "PAIRS " << index + 1 << '/' << pairs;
                for (int k = 0; k < policy_count; ++k) std::cout << ' ' << names[k] << '=' << sums[k] / (index + 1);
                std::cout << " seconds=" << std::accumulate(seconds.begin(), seconds.end(), 0.0) << '\n' << std::flush;
            }
        }
        std::ofstream meta(output / "evaluation.json");
        meta.exceptions(std::ios::badbit | std::ios::failbit);
        meta << std::setprecision(12) << "{\n\"seed\":" << seed << ",\"pairs\":" << pairs
             << ",\"training_hands\":" << hands << ",\"epoch_hands\":" << epoch_hands
             << ",\"epoch_models\":" << ensemble.models.size()
             << ",\"peak_rss_bytes\":" << epoch7::peak_rss()
             << ",\"hands_per_policy\":" << 2 * pairs << ",\"particles\":" << particles
             << ",\"unit\":\"LBR ante/hand\",\"policies\":{\n";
        for (int k = 0; k < policy_count; ++k) {
            const auto& s = stats[k];
            meta << '"' << names[k] << "\":{\"mean\":" << sums[k] / pairs
                 << ",\"seconds\":" << seconds[k] << ",\"decisions\":" << s.decisions
                 << ",\"queries\":" << s.policy_queries << ",\"misses\":" << s.policy_misses
                 << ",\"mean_ess\":" << s.average_effective_particles
                 << ",\"h4_proposals\":" << s.h4_particle_proposals
                 << ",\"h4_rejections\":" << s.h4_particle_rejections << '}'
                 << (k + 1 == policy_count ? "\n" : ",\n");
        }
        meta << "}}\n";
        std::cout << "DONE " << output.string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
