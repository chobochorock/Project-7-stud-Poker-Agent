// Shared frozen-policy matches. See README.md and run.py; no training here.
#pragma once
#include <fstream>
#include <iomanip>

namespace abstraction_lbr {
const std::array<const char*, 5> names{
    "model_current", "baseline_current", "model_average", "baseline_average", "uniform"};

template <class Model> struct Target {
    Model& model;
    int kind;
    std::mt19937_64 rng{0};
    epoch7::Values policy(const State& state, int viewer, bool* found = nullptr) const {
        const auto mask = valid_mask(state, viewer);
        epoch7::Values p;
        if (kind == 4) {
            p = uniform_strategy(mask);
            if (found) *found = true;
        } else {
            p = model.policy(state, viewer, kind, found);
        }
        double sum = 0;
        for (int a = 0; a < kActionCount; ++a) {
            hand_recall::require(std::isfinite(p[a]) && p[a] >= 0 &&
                ((mask & (1u << a)) || p[a] == 0), "invalid target policy");
            sum += p[a];
        }
        hand_recall::require(std::abs(sum - 1) < 1e-8, "policy normalization failed");
        return p;
    }
    Action choose(const State& state, int viewer, int) {
        return epoch7::sample(policy(state, viewer), valid_mask(state, viewer), rng);
    }
};

template <class Model> void self_test(Model& model) {
    for (int kind = 0; kind < 5; ++kind) {
        Target<Model> target{model, kind};
        auto state = epoch7::root_for(381);
        for (int step = 0; !state.terminal && step < 24; ++step) {
            const int seat = state.actor;
            auto masked = state;
            masked.players[1 - seat].hidden.clear();
            masked.players[1 - seat].has_discard = false;
            masked.simulation_deck.clear();
            hand_recall::require(target.policy(state, seat) == target.policy(masked, seat),
                "target sees hidden information");
            PolicyLBR<Target<Model>> a(target, 8, 41), b(target, 8, 41);
            hand_recall::require(a.choose(state, seat) == b.choose(masked, seat),
                "LBR sees hidden information");
            auto actions = actions_from_mask(valid_mask(state, seat));
            auto action = std::find(actions.begin(), actions.end(), CALL) != actions.end() ? CALL : CHECK;
            state = epoch7::child(state, action);
        }
        auto deck = fresh_deck();
        std::mt19937_64 deal_rng(418);
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        PolicyLBR<Target<Model>> a(target, 8, 415), b(target, 8, 415);
        target.rng.seed(91);
        const auto first = play_hand_policy_lbr(deck, 0, target, a, 1000, 1000);
        target.rng.seed(91);
        hand_recall::require(first == play_hand_policy_lbr(deck, 0, target, b, 1000, 1000),
            "nonreproducible payoff");
    }
    ConditionalParticipationPolicy fold("fold", 1);
    PolicyLBR<ConditionalParticipationPolicy> lbr(fold, 8, 61);
    double total = 0;
    for (int seat = 0; seat < 2; ++seat) {
        total += play_hand_policy_lbr(fresh_deck(), seat, fold, lbr, 1000, 1000);
    }
    hand_recall::require(std::abs(total / 2 - 1) < 1e-9, "seat/payoff unit check failed");
    std::cout << "SELF_TEST privacy, legal policies, reproducibility, ante/seat units: PASS\n";
}

template <class Model> int evaluate(Model& model, int argc, char** argv) {
    if (std::string(argv[1]) == "--self-test") {
        self_test(model);
        return 0;
    }
    const std::filesystem::path output(argv[2]);
    const int pairs = std::stoi(argv[3]), particles = std::stoi(argv[4]);
    const uint64_t seed = std::stoull(argv[5]);
    hand_recall::require(pairs >= 2 && particles > 0 && !std::filesystem::exists(output),
        "invalid budget or existing output");
    std::filesystem::create_directories(output);
    std::ofstream csv(output / "paired_payoffs.csv");
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv << std::setprecision(17) << "pair,deal_seed";
    for (auto name : names) csv << ',' << name << "_seat0," << name << "_seat1";
    csv << '\n';
    std::array<double, 5> sums{}, seconds{}, ess{};
    std::array<uint64_t, 5> queries{}, misses{}, decisions{};
    for (int index = 0; index < pairs; ++index) {
        const uint64_t deal_seed = seed ^ (uint64_t(index + 1) * 0xd1b54a32d192ed03ull);
        std::mt19937_64 deal_rng(deal_seed);
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        csv << index << ',' << deal_seed;
        for (int kind = 0; kind < 5; ++kind) {
            const auto start = std::chrono::steady_clock::now();
            Target<Model> target{model, kind};
            for (int seat = 0; seat < 2; ++seat) {
                const uint64_t action_seed = deal_seed ^ (uint64_t(seat + 1) * 0x94d049bb133111ebull);
                target.rng.seed(action_seed);
                PolicyLBR<Target<Model>> lbr(target, particles, action_seed ^ 0xabcdef012345ull);
                const double value = play_hand_policy_lbr(deck, seat, target, lbr, 1000, 1000);
                hand_recall::require(std::isfinite(value) && std::abs(value) <= 1000,
                    "invalid ante payoff");
                csv << ',' << value;
                sums[kind] += value / 2;
                const auto s = lbr.stats();
                queries[kind] += s.policy_queries;
                misses[kind] += s.policy_misses;
                decisions[kind] += s.decisions;
                ess[kind] += s.average_effective_particles * s.decisions;
            }
            seconds[kind] += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
        csv << '\n';
        if ((index + 1) % 25 == 0 || index + 1 == pairs) {
            csv.flush();
            std::cout << "PROGRESS pairs=" << index + 1 << " target_pairs=" << pairs;
            for (int k = 0; k < 5; ++k) std::cout << ' ' << names[k] << "_lbr_ante=" << sums[k] / (index + 1);
            std::cout << " seconds=" << std::accumulate(seconds.begin(), seconds.end(), 0.0) << std::endl;
        }
    }
    std::ofstream meta(output / "evaluation.json");
    meta << std::setprecision(17) << "{\"training_hands\":" << model.hands
         << ",\"pairs\":" << pairs << ",\"particles\":" << particles << ",\"seed\":" << seed
         << ",\"unit\":\"LBR attacker ante/hand\",\"peak_rss_bytes\":" << epoch7::peak_rss()
         << ",\"policies\":{";
    for (int k = 0; k < 5; ++k) {
        if (k) meta << ',';
        meta << '"' << names[k] << "\":{\"mean\":" << sums[k] / pairs
             << ",\"seconds\":" << seconds[k] << ",\"queries\":" << queries[k]
             << ",\"misses\":" << misses[k] << ",\"decisions\":" << decisions[k]
             << ",\"mean_ess\":" << (decisions[k] ? ess[k] / decisions[k] : 0) << '}';
    }
    meta << "}}\n";
    meta.close();
    hand_recall::require(bool(meta), "metadata write failed");
    return 0;
}
} // namespace abstraction_lbr
