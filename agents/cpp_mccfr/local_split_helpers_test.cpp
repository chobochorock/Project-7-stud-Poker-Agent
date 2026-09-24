#include "local_split_helpers.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using AuditSample = local_split::Sample<2, 3>;
    const std::vector<AuditSample> samples{
        {{{-3.0, 0.0}}, {{3.0, 0.0, 0.0}}},
        {{{-2.0, 0.0}}, {{2.0, 0.2, 0.0}}},
        {{{-1.0, 0.0}}, {{2.5, 0.1, 0.0}}},
        {{{1.0, 0.0}}, {{0.0, 2.5, 0.1}}},
        {{{2.0, 0.0}}, {{0.2, 2.0, 0.0}}},
        {{{3.0, 0.0}}, {{0.0, 3.0, 0.0}}},
    };

    const auto split = local_split::reward_aware_two_means(samples);
    assert(split);
    assert(split->child_count[0] == 3);
    assert(split->child_count[1] == 3);
    assert(std::abs(split->child_mass_ratio[0] - 0.5) < 1e-12);
    assert(std::abs(split->child_mass_ratio[1] - 0.5) < 1e-12);
    assert(std::abs(
        split->child_mass_ratio[0] + split->child_mass_ratio[1] - 1.0) < 1e-12);
    assert(split->best_actions_differ);
    assert(local_split::passes_reward_split(*split, 0.20));
    assert(split->reward_sse_gain > 0.90);
    assert(std::abs(
        split->reward_sse_gain -
        (1.0 -
         (split->child_reward_sse[0] + split->child_reward_sse[1]) /
             split->parent_reward_sse)) < 1e-12);

    const std::vector<AuditSample> identical{
        {{{1.0, 1.0}}, {{1.0, 0.0, 0.0}}},
        {{{1.0, 1.0}}, {{0.0, 1.0, 0.0}}},
    };
    assert(!local_split::reward_aware_two_means(identical));
    assert(!local_split::reward_aware_two_means(
        std::vector<AuditSample>{samples.front()}));

    std::cout << "local_split_helpers self-check: ok\n";
}
