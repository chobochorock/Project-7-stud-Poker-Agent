#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace local_split {

template <std::size_t Dimension, std::size_t ActionCount>
struct Sample {
    static_assert(Dimension > 0, "Dimension must be positive");
    static_assert(ActionCount > 0, "ActionCount must be positive");

    std::array<double, Dimension> feature{};
    std::array<double, ActionCount> reward{};
};

template <std::size_t Dimension, std::size_t ActionCount>
struct Result {
    std::array<std::array<double, Dimension>, 2> centroids{};
    std::vector<std::uint8_t> child;
    std::array<std::size_t, 2> child_count{};
    std::array<double, 2> child_mass_ratio{};
    double parent_reward_sse = 0.0;
    std::array<double, 2> child_reward_sse{};
    double reward_sse_gain = 0.0;
    std::array<std::array<double, ActionCount>, 2> child_mean_reward{};
    std::array<std::size_t, 2> child_best_action{};
    bool best_actions_differ = false;
};

template <std::size_t Dimension>
double squared_distance(
    const std::array<double, Dimension>& lhs,
    const std::array<double, Dimension>& rhs) {
    double distance = 0.0;
    for (std::size_t d = 0; d < Dimension; ++d) {
        const double delta = lhs[d] - rhs[d];
        distance += delta * delta;
    }
    return distance;
}

template <std::size_t ActionCount>
std::size_t best_action(const std::array<double, ActionCount>& reward) {
    return static_cast<std::size_t>(
        std::max_element(reward.begin(), reward.end()) - reward.begin());
}

template <std::size_t Dimension, std::size_t ActionCount>
std::optional<Result<Dimension, ActionCount>> reward_aware_two_means(
    const std::vector<Sample<Dimension, ActionCount>>& samples,
    double minimum_centroid_squared_distance = 1e-24) {
    if (samples.size() < 2) return std::nullopt;

    std::array<std::size_t, 2> seeds{0, 0};
    double farthest_distance = -1.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        for (std::size_t j = i + 1; j < samples.size(); ++j) {
            const double distance = squared_distance(
                samples[i].feature, samples[j].feature);
            if (distance > farthest_distance) {
                farthest_distance = distance;
                seeds = {i, j};
            }
        }
    }
    if (farthest_distance <= minimum_centroid_squared_distance) {
        return std::nullopt;
    }

    Result<Dimension, ActionCount> result;
    result.centroids = {
        samples[seeds[0]].feature,
        samples[seeds[1]].feature};
    result.child.assign(samples.size(), 2);

    for (int iteration = 0; iteration < 20; ++iteration) {
        std::array<std::array<double, Dimension>, 2> sums{};
        std::array<std::size_t, 2> counts{};
        bool changed = false;

        for (std::size_t i = 0; i < samples.size(); ++i) {
            const std::uint8_t child =
                squared_distance(samples[i].feature, result.centroids[1]) <
                        squared_distance(samples[i].feature, result.centroids[0])
                    ? 1
                    : 0;
            changed = changed || result.child[i] != child;
            result.child[i] = child;
            ++counts[child];
            for (std::size_t d = 0; d < Dimension; ++d) {
                sums[child][d] += samples[i].feature[d];
            }
        }

        if (counts[0] == 0 || counts[1] == 0) return std::nullopt;
        for (std::size_t child = 0; child < 2; ++child) {
            for (std::size_t d = 0; d < Dimension; ++d) {
                result.centroids[child][d] =
                    sums[child][d] / static_cast<double>(counts[child]);
            }
        }
        result.child_count = counts;
        if (squared_distance(result.centroids[0], result.centroids[1]) <=
            minimum_centroid_squared_distance) {
            return std::nullopt;
        }
        if (!changed) break;
    }

    std::array<double, ActionCount> parent_mean{};
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::size_t child = result.child[i];
        for (std::size_t a = 0; a < ActionCount; ++a) {
            parent_mean[a] += samples[i].reward[a];
            result.child_mean_reward[child][a] += samples[i].reward[a];
        }
    }
    for (std::size_t a = 0; a < ActionCount; ++a) {
        parent_mean[a] /= static_cast<double>(samples.size());
        for (std::size_t child = 0; child < 2; ++child) {
            result.child_mean_reward[child][a] /=
                static_cast<double>(result.child_count[child]);
        }
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::size_t child = result.child[i];
        for (std::size_t a = 0; a < ActionCount; ++a) {
            const double parent_delta = samples[i].reward[a] - parent_mean[a];
            const double child_delta =
                samples[i].reward[a] - result.child_mean_reward[child][a];
            result.parent_reward_sse += parent_delta * parent_delta;
            result.child_reward_sse[child] += child_delta * child_delta;
        }
    }

    const double child_sse =
        result.child_reward_sse[0] + result.child_reward_sse[1];
    if (result.parent_reward_sse > std::numeric_limits<double>::epsilon()) {
        result.reward_sse_gain = std::clamp(
            1.0 - child_sse / result.parent_reward_sse, 0.0, 1.0);
    }
    for (std::size_t child = 0; child < 2; ++child) {
        result.child_mass_ratio[child] =
            static_cast<double>(result.child_count[child]) /
            static_cast<double>(samples.size());
        result.child_best_action[child] =
            best_action(result.child_mean_reward[child]);
    }
    result.best_actions_differ =
        result.child_best_action[0] != result.child_best_action[1];
    return result;
}

template <std::size_t Dimension, std::size_t ActionCount>
bool passes_reward_split(
    const Result<Dimension, ActionCount>& result,
    double minimum_reward_sse_gain) {
    return result.best_actions_differ &&
           result.reward_sse_gain >= minimum_reward_sse_gain;
}

}  // namespace local_split
