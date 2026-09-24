#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

struct SampledBrGapResult {
    std::array<double, 2> policy_value{};
    std::array<double, 2> positive_deviation_gap{};
    std::array<double, 3> gap_by_street{};
    std::array<uint64_t, 2> decisions{};
    uint64_t node_visits = 0;

    double exploitability_proxy(int ante) const {
        return 0.5 * (
            positive_deviation_gap[0] + positive_deviation_gap[1]) /
            std::max(1, ante);
    }
};

template <class Policy>
struct SampledBrGapSubgameHook {
    static void begin(Policy&, const State&) {}
};

template <class Policy>
struct SampledBrGapInfoHook {
    static void record(
        Policy&,
        const State&,
        int,
        uint8_t,
        const std::array<double, kActionCount>&,
        double) {}
};

// External-sampling, trajectory-relaxed T=1 counterfactual deviation gap.
// The policy type only needs policy(state, actor, found).
template <class Policy>
class SampledBrGapEstimator {
public:
    static SampledBrGapResult evaluate(
        Policy& policy,
        const State& root,
        uint64_t seed) {
        SampledBrGapEstimator estimator(policy, seed);
        for (int player = 0; player < 2; ++player) {
            estimator.target_ = player;
            estimator.current_gap_ = 0.0;
            estimator.result_.policy_value[player] = estimator.traverse(root);
            estimator.result_.positive_deviation_gap[player] =
                estimator.current_gap_;
        }
        return estimator.result_;
    }

private:
    SampledBrGapEstimator(Policy& policy, uint64_t seed)
        : policy_(policy), rng_(seed) {}

    void advance_street(State& state) {
        if (state.street == 7) {
            state.terminal = true;
            return;
        }
        ++state.street;
        const bool public_card = state.street != 7;
        for (int seat = 0; seat < 2; ++seat) {
            if (state.simulation_deck.empty()) {
                throw std::runtime_error("BR-gap simulation deck exhausted");
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
            advance_street(state);
            return;
        }
        state.actor = first_bettor(state);
        SampledBrGapSubgameHook<Policy>::begin(policy_, state);
    }

    State child(State state, int actor, Action action) {
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd) advance_street(state);
        else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
        return state;
    }

    Action sample(
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

    double traverse(State state) {
        ++result_.node_visits;
        if (state.terminal) return terminal_net_search(state, target_);

        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const auto strategy = policy_.policy(state, actor);
        if (actor != target_) {
            return traverse(child(
                std::move(state), actor, sample(strategy, mask)));
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            action_values[action] = traverse(child(state, actor, action));
        }
        double value = 0.0;
        double best = -std::numeric_limits<double>::infinity();
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            value += strategy[action] * action_values[action];
            best = std::max(best, action_values[action]);
        }
        SampledBrGapInfoHook<Policy>::record(
            policy_, state, target_, mask, action_values, value);
        const double gap = std::max(0.0, best - value);
        current_gap_ += gap;
        if (state.street >= 5 && state.street <= 7) {
            result_.gap_by_street[state.street - 5] +=
                0.5 * gap / std::max(1, state.ante);
        }
        ++result_.decisions[target_];
        return value;
    }

    Policy& policy_;
    std::mt19937_64 rng_;
    SampledBrGapResult result_{};
    int target_ = 0;
    double current_gap_ = 0.0;
};
