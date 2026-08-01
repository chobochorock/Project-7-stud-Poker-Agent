#define main stud2_embedded_main
#include "stud_mccfr.cpp"
#undef main

namespace {

constexpr int kSeats5 = 5;
constexpr int kPower5Dimensions = 18;
constexpr uint32_t kFiveModelVersion = 5;

struct FivePlayer {
    std::vector<Card> hidden;
    std::vector<Card> shown;
    Card discarded{};
    int stack_cap = 1000;
    int invested = 0;
    int round_bet = 0;
    int bet_count = 0;
    bool checked = false;
    bool folded = false;
    bool all_in = false;
};

struct FiveEvent {
    uint8_t actor = 0;
    Action action = CHECK;
};

struct FiveState {
    std::array<FivePlayer, kSeats5> players;
    std::vector<FiveEvent> history;
    int ante = 1;
    int pot = 0;
    int highest_bet = 0;
    int actor = 0;
    uint8_t pending = 0;
    bool terminal = false;
};

int remaining(const FiveState& state, int seat) {
    return std::max(
        0, state.players[seat].stack_cap - state.players[seat].invested);
}

bool can_act(const FiveState& state, int seat) {
    const auto& player = state.players[seat];
    return !player.folded && !player.all_in && remaining(state, seat) > 0;
}

int survivors(const FiveState& state) {
    return static_cast<int>(std::count_if(
        state.players.begin(), state.players.end(),
        [](const FivePlayer& player) { return !player.folded; }));
}

uint8_t five_valid_mask(const FiveState& state, int seat) {
    if (state.terminal || !can_act(state, seat)) return 0;
    const auto& player = state.players[seat];
    const int call = std::max(0, state.highest_bet - player.round_bet);
    uint8_t mask = static_cast<uint8_t>(1u << FOLD);
    mask |= static_cast<uint8_t>(1u << (call ? CALL : CHECK));
    const bool can_raise = !player.checked && player.bet_count < 3;
    if (can_raise && state.highest_bet == 0 && remaining(state, seat) > 0) {
        mask |= static_cast<uint8_t>(1u << BBING);
    }
    if (can_raise && state.pot > 0 && remaining(state, seat) > call) {
        if (state.highest_bet > 0) mask |= static_cast<uint8_t>(1u << DDADANG);
        mask |= static_cast<uint8_t>((1u << QUARTER) | (1u << HALF));
    }
    return mask;
}

int five_raise_amount(
    const FiveState& state,
    Action action,
    int call) {
    const int pot_after_call = state.pot + call;
    if (action == BBING) return state.ante;
    if (action == DDADANG) return std::max(1, state.highest_bet);
    if (action == QUARTER) return std::max(1, (pot_after_call + 3) / 4);
    if (action == HALF) return std::max(1, (pot_after_call + 1) / 2);
    return 0;
}

int next_pending(const FiveState& state, int after) {
    for (int offset = 1; offset <= kSeats5; ++offset) {
        const int seat = (after + offset) % kSeats5;
        if (state.pending & (1u << seat)) return seat;
    }
    return -1;
}

void finish_if_needed(FiveState& state, int previous_actor) {
    if (survivors(state) <= 1 || state.pending == 0) {
        state.terminal = true;
        return;
    }
    const int next = next_pending(state, previous_actor);
    if (next < 0) {
        state.terminal = true;
    } else {
        state.actor = next;
    }
}

void five_apply(FiveState& state, Action action) {
    const int seat = state.actor;
    const uint8_t mask = five_valid_mask(state, seat);
    if (!(mask & (1u << action))) {
        throw std::runtime_error("invalid five-player action");
    }
    auto& player = state.players[seat];
    const int old_highest = state.highest_bet;
    const int call = std::max(0, old_highest - player.round_bet);

    if (action == FOLD) {
        player.folded = true;
        state.pending &= static_cast<uint8_t>(~(1u << seat));
    } else if (action == CHECK) {
        player.checked = true;
        state.pending &= static_cast<uint8_t>(~(1u << seat));
    } else {
        const int requested =
            call + five_raise_amount(state, action, call);
        const int paid = std::min(requested, remaining(state, seat));
        player.round_bet += paid;
        player.invested += paid;
        player.all_in = remaining(state, seat) == 0;
        state.pot += paid;
        state.highest_bet = std::max(
            state.highest_bet, player.round_bet);
        if (aggressive(action)) ++player.bet_count;

        if (state.highest_bet > old_highest) {
            state.pending = 0;
            for (int other = 0; other < kSeats5; ++other) {
                if (other != seat && can_act(state, other)) {
                    state.pending |= static_cast<uint8_t>(1u << other);
                }
            }
        } else {
            state.pending &= static_cast<uint8_t>(~(1u << seat));
        }
    }
    state.history.push_back(FiveEvent{
        static_cast<uint8_t>(seat), action});
    finish_if_needed(state, seat);
}

std::vector<Card> five_cards(const FivePlayer& player) {
    std::vector<Card> cards = player.hidden;
    cards.insert(cards.end(), player.shown.begin(), player.shown.end());
    return cards;
}

std::array<double, kSeats5> five_payoffs(const FiveState& state) {
    std::array<int, kSeats5> awards{};
    std::vector<int> live;
    for (int seat = 0; seat < kSeats5; ++seat) {
        if (!state.players[seat].folded) live.push_back(seat);
    }
    if (live.size() == 1) {
        awards[live.front()] = state.pot;
    } else {
        std::array<Score, kSeats5> scores{};
        for (int seat : live) {
            scores[seat] = best_hand(five_cards(state.players[seat]));
        }
        std::vector<int> levels;
        for (const auto& player : state.players) {
            if (player.invested > 0) levels.push_back(player.invested);
        }
        std::sort(levels.begin(), levels.end());
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        int previous = 0;
        for (int level : levels) {
            int amount = 0;
            for (const auto& player : state.players) {
                amount += std::max(
                    0, std::min(player.invested, level) - previous);
            }
            previous = level;
            if (!amount) continue;
            std::vector<int> eligible;
            for (int seat : live) {
                if (state.players[seat].invested >= level) {
                    eligible.push_back(seat);
                }
            }
            if (eligible.empty()) eligible = live;
            const Score best = std::max_element(
                eligible.begin(), eligible.end(),
                [&](int left, int right) {
                    return scores[left] < scores[right];
                }) == eligible.end()
                ? Score{}
                : scores[*std::max_element(
                    eligible.begin(), eligible.end(),
                    [&](int left, int right) {
                        return scores[left] < scores[right];
                    })];
            std::vector<int> winners;
            for (int seat : eligible) {
                if (scores[seat] == best) winners.push_back(seat);
            }
            const int share = amount / static_cast<int>(winners.size());
            int remainder_chips = amount % static_cast<int>(winners.size());
            for (int seat : winners) {
                awards[seat] += share + (remainder_chips-- > 0 ? 1 : 0);
            }
        }
    }

    std::array<double, kSeats5> utility{};
    for (int seat = 0; seat < kSeats5; ++seat) {
        utility[seat] =
            (awards[seat] - state.players[seat].invested) /
            static_cast<double>(state.ante);
    }
    return utility;
}

int five_first_bettor(const FiveState& state) {
    int best = -1;
    for (int seat = 0; seat < kSeats5; ++seat) {
        if (!can_act(state, seat)) continue;
        if (best < 0 ||
            public_priority(state.players[seat].shown) >
                public_priority(state.players[best].shown)) {
            best = seat;
        }
    }
    return best;
}

void begin_five_round(FiveState& state) {
    state.highest_bet = 0;
    state.history.clear();
    state.pending = 0;
    state.terminal = false;
    for (int seat = 0; seat < kSeats5; ++seat) {
        auto& player = state.players[seat];
        player.round_bet = 0;
        player.bet_count = 0;
        player.checked = false;
        if (can_act(state, seat)) {
            state.pending |= static_cast<uint8_t>(1u << seat);
        }
    }
    state.actor = five_first_bettor(state);
    int actors = 0;
    for (int seat = 0; seat < kSeats5; ++seat) {
        actors += (state.pending & (1u << seat)) != 0;
    }
    if (state.actor < 0 || survivors(state) <= 1 || actors <= 1) {
        state.terminal = true;
    }
}

Action five_heuristic(const FiveState& state, int seat) {
    const uint8_t mask = five_valid_mask(state, seat);
    const auto legal = actions_from_mask(mask);
    if (legal.empty()) return FOLD;
    const auto& player = state.players[seat];
    const double strength =
        std::clamp(made_strength(best_hand(five_cards(player))), 0.0, 1.0);
    const int call = std::max(0, state.highest_bet - player.round_bet);
    const double odds =
        call / std::max(1.0, static_cast<double>(state.pot + call));
    auto has = [&](Action action) { return mask & (1u << action); };

    if (call > 0) {
        if (strength + 0.08 < odds && has(FOLD)) return FOLD;
        if (strength >= 0.82 && has(HALF)) return HALF;
        if (strength >= 0.68 && has(QUARTER)) return QUARTER;
        return has(CALL) ? CALL : FOLD;
    }
    if (strength >= 0.82 && has(HALF)) return HALF;
    if (strength >= 0.62 && has(QUARTER)) return QUARTER;
    if (strength >= 0.48 && has(BBING)) return BBING;
    return has(CHECK) ? CHECK : legal.front();
}

void play_five_heuristic_round(FiveState& state) {
    begin_five_round(state);
    while (!state.terminal) {
        five_apply(state, five_heuristic(state, state.actor));
    }
    if (survivors(state) > 1) state.terminal = false;
}

struct FiveRoot {
    FiveState seventh;
};

FiveRoot make_five_root(
    const std::vector<Card>& deck,
    const std::array<int, kSeats5>& stack_antes,
    int ante) {
    FiveState state;
    state.ante = ante;
    state.pot = ante * kSeats5;
    for (int seat = 0; seat < kSeats5; ++seat) {
        state.players[seat].stack_cap = stack_antes[seat] * ante;
        state.players[seat].invested = ante;
    }
    size_t cursor = 0;
    for (int round = 0; round < 4; ++round) {
        for (int seat = 0; seat < kSeats5; ++seat) {
            state.players[seat].hidden.push_back(deck[cursor++]);
        }
    }
    for (int seat = 0; seat < kSeats5; ++seat) {
        auto& player = state.players[seat];
        const auto [discard, reveal] = discard_reveal(player.hidden);
        player.discarded = player.hidden[discard];
        player.shown.push_back(player.hidden[reveal]);
        std::vector<Card> kept;
        for (int index = 0; index < 4; ++index) {
            if (index != discard && index != reveal) {
                kept.push_back(player.hidden[index]);
            }
        }
        player.hidden = std::move(kept);
    }

    for (int seat = 0; seat < kSeats5; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);  // 4th
    }
    for (int seat = 0; seat < kSeats5; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);  // 5th
    }
    play_five_heuristic_round(state);

    if (!state.terminal) {
        for (int seat = 0; seat < kSeats5; ++seat) {
            if (!state.players[seat].folded) {
                state.players[seat].shown.push_back(deck[cursor++]);  // 6th
            }
        }
        play_five_heuristic_round(state);
    }

    if (!state.terminal) {
        for (int seat = 0; seat < kSeats5; ++seat) {
            if (!state.players[seat].folded) {
                state.players[seat].hidden.push_back(deck[cursor++]);  // 7th
            }
        }
        begin_five_round(state);
    }
    return FiveRoot{state};
}

uint64_t five_card_mask(const std::vector<Card>& cards) {
    uint64_t mask = 0;
    for (const Card& card : cards) {
        mask |= 1ull << (card.suit * 13 + card.rank - 2);
    }
    return mask;
}

struct FiveObservationKey {
    uint64_t own = 0;
    std::array<uint64_t, 4> opponents{};

    bool operator==(const FiveObservationKey& other) const {
        return own == other.own && opponents == other.opponents;
    }
};

struct FiveObservationHash {
    size_t operator()(const FiveObservationKey& key) const {
        size_t hash = key.own;
        for (uint64_t mask : key.opponents) {
            hash ^= mask + 0x9e3779b97f4a7c15ull +
                (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

using FivePower = std::array<double, kPower5Dimensions>;
using FiveStackFeature = std::array<double, kSeats5>;

FiveStackFeature five_stack_features(
    const FiveState& state,
    int viewer) {
    FiveStackFeature features{};
    const double scale = std::log1p(1000.0);
    for (int offset = 0; offset < kSeats5; ++offset) {
        const int seat = (viewer + offset) % kSeats5;
        features[offset] = std::clamp(
            std::log1p(
                remaining(state, seat) /
                static_cast<double>(state.ante)) / scale,
            0.0,
            1.0);
    }
    return features;
}

class FivePowerEstimator {
public:
    explicit FivePowerEstimator(int samples) : samples_(samples) {}

    const FivePower& get(const FiveState& state, int viewer) {
        FiveObservationKey key;
        key.own = five_card_mask(five_cards(state.players[viewer]));
        key.own |= 1ull << (
            state.players[viewer].discarded.suit * 13 +
            state.players[viewer].discarded.rank - 2);
        for (int offset = 1; offset < kSeats5; ++offset) {
            const int seat = (viewer + offset) % kSeats5;
            key.opponents[offset - 1] =
                five_card_mask(state.players[seat].shown);
            if (state.players[seat].folded) {
                key.opponents[offset - 1] |= 1ull << 63;
            }
        }
        auto [it, inserted] = cache_.try_emplace(key);
        if (inserted) it->second = compute(state, viewer, key);
        return it->second;
    }

    size_t cache_size() const { return cache_.size(); }

private:
    FivePower compute(
        const FiveState& state,
        int viewer,
        const FiveObservationKey& key) const {
        FivePower power{};
        const Score own_score =
            best_hand(five_cards(state.players[viewer]));
        power[std::clamp(own_score[0], 0, 8)] = 1.0;

        uint64_t known = key.own;
        for (uint64_t mask : key.opponents) {
            known |= mask & ((1ull << 52) - 1);
        }
        std::vector<Card> unknown;
        for (const Card& card : fresh_deck()) {
            const uint64_t bit =
                1ull << (card.suit * 13 + card.rank - 2);
            if (!(known & bit)) unknown.push_back(card);
        }

        uint64_t seed = key.own ^ 0xd1b54a32d192ed03ull;
        for (uint64_t mask : key.opponents) {
            seed ^= mask + 0x9e3779b97f4a7c15ull +
                (seed << 6) + (seed >> 2);
        }
        std::mt19937_64 rng(seed);
        int used_samples = 0;
        for (int sample = 0; sample < samples_; ++sample) {
            std::vector<Card> draw = unknown;
            const int needed = 3 * (kSeats5 - 1);
            if (static_cast<int>(draw.size()) < needed) break;
            for (int i = 0; i < needed; ++i) {
                std::uniform_int_distribution<int> pick(
                    i, static_cast<int>(draw.size()) - 1);
                std::swap(draw[i], draw[pick(rng)]);
            }

            std::array<Score, kSeats5> scores{};
            scores[viewer] = own_score;
            int cursor = 0;
            for (int offset = 1; offset < kSeats5; ++offset) {
                const int seat = (viewer + offset) % kSeats5;
                std::vector<Card> cards = state.players[seat].shown;
                for (int card = 0; card < 3; ++card) {
                    cards.push_back(draw[cursor++]);
                }
                scores[seat] = best_hand(cards);
                power[13 + offset] += scores[seat][0] / 8.0;
                if (own_score > scores[seat]) {
                    power[9 + offset] += 1.0;
                } else if (own_score == scores[seat]) {
                    power[9 + offset] += 0.5;
                }
            }
            Score best = own_score;
            for (int seat = 0; seat < kSeats5; ++seat) {
                if (!state.players[seat].folded) {
                    best = std::max(best, scores[seat]);
                }
            }
            if (own_score == best && !state.players[viewer].folded) {
                int ties = 0;
                for (int seat = 0; seat < kSeats5; ++seat) {
                    if (!state.players[seat].folded &&
                        scores[seat] == best) {
                        ++ties;
                    }
                }
                power[9] += 1.0 / std::max(1, ties);
            }
            ++used_samples;
        }
        if (used_samples) {
            for (int index = 9; index < kPower5Dimensions; ++index) {
                power[index] /= used_samples;
            }
        }
        return power;
    }

    int samples_;
    std::unordered_map<
        FiveObservationKey,
        FivePower,
        FiveObservationHash> cache_;
};

template <size_t Dimensions>
double five_distance(
    const std::array<double, Dimensions>& left,
    const std::array<double, Dimensions>& right) {
    double sum = 0.0;
    for (size_t index = 0; index < Dimensions; ++index) {
        const double delta = left[index] - right[index];
        sum += delta * delta;
    }
    return sum;
}

class FiveAtlas {
public:
    void fit(
        const std::vector<FivePower>& samples,
        const std::vector<FiveStackFeature>& stack_samples,
        int power_clusters,
        int stack_clusters,
        uint64_t seed) {
        if (samples.empty() || stack_samples.empty() ||
            power_clusters <= 0 || stack_clusters <= 0) {
            throw std::runtime_error("five-player atlas needs samples");
        }
        fit_centers(samples, power_clusters, seed, centers_);
        fit_centers(
            stack_samples,
            stack_clusters,
            seed ^ 0x8cb92baa3f3d8dd7ull,
            stack_centers_);
    }

    int assign(const FivePower& value) const {
        return nearest(value, centers_);
    }

    int assign_stack(const FiveStackFeature& value) const {
        return nearest(value, stack_centers_);
    }

    const std::vector<FivePower>& centers() const { return centers_; }
    std::vector<FivePower>& centers() { return centers_; }
    const std::vector<FiveStackFeature>& stack_centers() const {
        return stack_centers_;
    }
    std::vector<FiveStackFeature>& stack_centers() {
        return stack_centers_;
    }

private:
    template <size_t Dimensions>
    static int nearest(
        const std::array<double, Dimensions>& value,
        const std::vector<std::array<double, Dimensions>>& centers) {
        if (centers.empty()) return 0;
        int best = 0;
        double distance = five_distance(value, centers[0]);
        for (int index = 1; index < static_cast<int>(centers.size()); ++index) {
            const double candidate = five_distance(value, centers[index]);
            if (candidate < distance) {
                best = index;
                distance = candidate;
            }
        }
        return best;
    }

    template <size_t Dimensions>
    static void fit_centers(
        const std::vector<std::array<double, Dimensions>>& samples,
        int clusters,
        uint64_t seed,
        std::vector<std::array<double, Dimensions>>& centers) {
        clusters = std::min(clusters, static_cast<int>(samples.size()));
        std::mt19937_64 rng(seed);
        std::vector<size_t> order(samples.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        centers.clear();
        for (int index = 0; index < clusters; ++index) {
            centers.push_back(samples[order[index]]);
        }

        std::vector<int> assignment(samples.size());
        for (int iteration = 0; iteration < 12; ++iteration) {
            std::vector<std::array<double, Dimensions>> sums(centers.size());
            std::vector<int> counts(centers.size());
            for (size_t row = 0; row < samples.size(); ++row) {
                const int cluster = nearest(samples[row], centers);
                assignment[row] = cluster;
                ++counts[cluster];
                for (size_t dim = 0; dim < Dimensions; ++dim) {
                    sums[cluster][dim] += samples[row][dim];
                }
            }
            for (size_t cluster = 0; cluster < centers.size(); ++cluster) {
                if (!counts[cluster]) {
                    centers[cluster] =
                        samples[order[(cluster + iteration) % order.size()]];
                    continue;
                }
                for (size_t dim = 0; dim < Dimensions; ++dim) {
                    centers[cluster][dim] =
                        sums[cluster][dim] / counts[cluster];
                }
            }
        }
    }

    std::vector<FivePower> centers_;
    std::vector<FiveStackFeature> stack_centers_;
};

struct FiveInfoKey {
    uint16_t power_cluster = 0;
    uint16_t stack_profile = 0;
    uint8_t active_mask = 0;
    uint8_t all_in_mask = 0;
    uint8_t checked_mask = 0;
    uint8_t legal_mask = 0;
    uint8_t pot_odds_bucket = 0;
    uint8_t stack_pot_bucket = 0;
    uint8_t own_bet_count = 0;
    uint8_t opponent_bet_count = 0;
    uint8_t last_action_class = 0;
    uint8_t betting_goal = 0;

    bool operator==(const FiveInfoKey& other) const {
        return power_cluster == other.power_cluster &&
            stack_profile == other.stack_profile &&
            active_mask == other.active_mask &&
            all_in_mask == other.all_in_mask &&
            checked_mask == other.checked_mask &&
            legal_mask == other.legal_mask &&
            pot_odds_bucket == other.pot_odds_bucket &&
            stack_pot_bucket == other.stack_pot_bucket &&
            own_bet_count == other.own_bet_count &&
            opponent_bet_count == other.opponent_bet_count &&
            last_action_class == other.last_action_class &&
            betting_goal == other.betting_goal;
    }
};

struct FiveInfoHash {
    size_t operator()(const FiveInfoKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * 1099511628211ull;
        };
        mix(key.power_cluster);
        mix(key.stack_profile);
        mix(key.active_mask);
        mix(key.all_in_mask);
        mix(key.checked_mask);
        mix(key.legal_mask);
        mix(key.pot_odds_bucket);
        mix(key.stack_pot_bucket);
        mix(key.own_bet_count);
        mix(key.opponent_bet_count);
        mix(key.last_action_class);
        mix(key.betting_goal);
        return hash;
    }
};

struct FiveNode {
    std::array<double, kActionCount> regret{};
    std::array<double, kActionCount> strategy_sum{};
    uint64_t touches = 0;
};

struct FiveEvalSummary {
    std::vector<double> deal_values;
    std::array<std::vector<double>, 4> stack_values;
    std::array<uint64_t, kActionCount> actions{};
};

class FiveMCCFR {
public:
    FiveMCCFR(
        FiveAtlas atlas,
        int mc_samples,
        uint64_t seed)
        : atlas_(std::move(atlas)),
          power_(mc_samples),
          rng_(seed) {}

    FiveInfoKey make_key(const FiveState& state, int viewer) {
        FiveInfoKey key;
        key.power_cluster = static_cast<uint16_t>(
            atlas_.assign(power_.get(state, viewer)));
        key.stack_profile = static_cast<uint16_t>(
            atlas_.assign_stack(five_stack_features(state, viewer)));
        for (int offset = 0; offset < kSeats5; ++offset) {
            const int seat = (viewer + offset) % kSeats5;
            const auto& player = state.players[seat];
            if (!player.folded) key.active_mask |= 1u << offset;
            if (player.all_in) key.all_in_mask |= 1u << offset;
        }
        key.legal_mask = five_valid_mask(state, viewer);
        const int call = std::max(
            0, state.highest_bet - state.players[viewer].round_bet);
        const double pot = std::max(1, state.pot);
        const double odds =
            call / std::max(1.0, pot + call);
        key.pot_odds_bucket =
            ratio_bucket(odds, {0.1, 0.2, 0.33, 0.5});
        return key;
    }

    std::array<double, kActionCount> current_strategy(
        FiveNode& node,
        uint8_t mask) {
        std::array<double, kActionCount> strategy{};
        double positive = 0.0;
        int legal = 0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = std::max(0.0, node.regret[action]);
            positive += strategy[action];
            ++legal;
        }
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = positive > 0.0
                ? strategy[action] / positive
                : 1.0 / legal;
        }
        return strategy;
    }

    std::array<double, kActionCount> average_strategy(
        const FiveNode& node,
        uint8_t mask) const {
        std::array<double, kActionCount> strategy{};
        double total = 0.0;
        int legal = 0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = std::max(0.0, node.strategy_sum[action]);
            total += strategy[action];
            ++legal;
        }
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = total > 0.0
                ? strategy[action] / total
                : 1.0 / legal;
        }
        return strategy;
    }

    Action sample(
        const std::array<double, kActionCount>& strategy,
        uint8_t mask) {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        double choice = unit(rng_);
        Action fallback = FOLD;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            fallback = static_cast<Action>(action);
            choice -= strategy[action];
            if (choice <= 0.0) return static_cast<Action>(action);
        }
        return fallback;
    }

    double traverse(FiveState state, int traverser) {
        ++node_visits_;
        if (state.terminal) return five_payoffs(state)[traverser];
        const int actor = state.actor;
        const uint8_t mask = five_valid_mask(state, actor);
        if (!mask) {
            state.pending &= static_cast<uint8_t>(~(1u << actor));
            finish_if_needed(state, actor);
            return traverse(std::move(state), traverser);
        }
        const FiveInfoKey key = make_key(state, actor);
        FiveNode& node = nodes_[key];
        ++node.touches;
        const auto strategy = current_strategy(node, mask);
        for (int action = 0; action < kActionCount; ++action) {
            node.strategy_sum[action] += strategy[action];
        }

        if (actor != traverser) {
            const Action action = sample(strategy, mask);
            five_apply(state, action);
            return traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> values{};
        double expected = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            FiveState child = state;
            five_apply(child, static_cast<Action>(action));
            values[action] = traverse(std::move(child), traverser);
            expected += strategy[action] * values[action];
        }
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            node.regret[action] = std::max(
                0.0, node.regret[action] + values[action] - expected);
        }
        return expected;
    }

    void train_root(const FiveState& root) {
        for (int traverser = 0; traverser < kSeats5; ++traverser) {
            if (!root.players[traverser].folded) {
                traverse(root, traverser);
                ++traversals_;
            }
        }
    }

    Action choose(FiveState& state, int seat) {
        const uint8_t mask = five_valid_mask(state, seat);
        const FiveInfoKey key = make_key(state, seat);
        auto it = nodes_.find(key);
        if (it == nodes_.end()) {
            ++policy_misses_;
            FiveNode empty;
            return sample(average_strategy(empty, mask), mask);
        }
        return sample(average_strategy(it->second, mask), mask);
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot save five-player model");
        const char magic[8] = {'S','T','U','D','5','C','F','R'};
        out.write(magic, sizeof(magic));
        out.write(
            reinterpret_cast<const char*>(&kFiveModelVersion),
            sizeof(kFiveModelVersion));
        const uint64_t center_count = atlas_.centers().size();
        out.write(
            reinterpret_cast<const char*>(&center_count),
            sizeof(center_count));
        for (const auto& center : atlas_.centers()) {
            out.write(
                reinterpret_cast<const char*>(center.data()),
                sizeof(double) * center.size());
        }
        const uint64_t stack_center_count = atlas_.stack_centers().size();
        out.write(
            reinterpret_cast<const char*>(&stack_center_count),
            sizeof(stack_center_count));
        for (const auto& center : atlas_.stack_centers()) {
            out.write(
                reinterpret_cast<const char*>(center.data()),
                sizeof(double) * center.size());
        }
        const uint64_t node_count = nodes_.size();
        out.write(
            reinterpret_cast<const char*>(&node_count),
            sizeof(node_count));
        for (const auto& [key, node] : nodes_) {
            out.write(reinterpret_cast<const char*>(&key), sizeof(key));
            out.write(reinterpret_cast<const char*>(&node), sizeof(node));
        }
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot load five-player model");
        char magic[8]{};
        uint32_t version = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (std::string(magic, sizeof(magic)) != "STUD5CFR" ||
            version != kFiveModelVersion) {
            throw std::runtime_error("invalid five-player model");
        }
        uint64_t center_count = 0;
        in.read(
            reinterpret_cast<char*>(&center_count),
            sizeof(center_count));
        atlas_.centers().resize(center_count);
        for (auto& center : atlas_.centers()) {
            in.read(
                reinterpret_cast<char*>(center.data()),
                sizeof(double) * center.size());
        }
        uint64_t stack_center_count = 0;
        in.read(
            reinterpret_cast<char*>(&stack_center_count),
            sizeof(stack_center_count));
        atlas_.stack_centers().resize(stack_center_count);
        for (auto& center : atlas_.stack_centers()) {
            in.read(
                reinterpret_cast<char*>(center.data()),
                sizeof(double) * center.size());
        }
        uint64_t node_count = 0;
        in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
        nodes_.clear();
        for (uint64_t index = 0; index < node_count; ++index) {
            FiveInfoKey key;
            FiveNode node;
            in.read(reinterpret_cast<char*>(&key), sizeof(key));
            in.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes_.emplace(key, node);
        }
        if (!in) throw std::runtime_error("truncated five-player model");
    }

    size_t buckets() const { return nodes_.size(); }
    uint64_t traversals() const { return traversals_; }
    uint64_t node_visits() const { return node_visits_; }
    uint64_t policy_misses() const { return policy_misses_; }
    size_t power_cache_size() const { return power_.cache_size(); }
    const FiveAtlas& atlas() const { return atlas_; }

private:
    FiveAtlas atlas_;
    FivePowerEstimator power_;
    std::mt19937_64 rng_;
    std::unordered_map<FiveInfoKey, FiveNode, FiveInfoHash> nodes_;
    uint64_t traversals_ = 0;
    uint64_t node_visits_ = 0;
    uint64_t policy_misses_ = 0;
};

std::array<int, kSeats5> sample_five_stacks(
    std::mt19937_64& rng,
    int minimum,
    int maximum) {
    std::uniform_real_distribution<double> log_stack(
        std::log(minimum), std::log(maximum));
    std::array<int, kSeats5> stacks{};
    for (int& stack : stacks) {
        stack = std::clamp(
            static_cast<int>(std::llround(std::exp(log_stack(rng)))),
            minimum,
            maximum);
    }
    return stacks;
}

struct FiveAtlasSamples {
    std::vector<FivePower> power;
    std::vector<FiveStackFeature> stacks;
};

FiveAtlasSamples collect_five_atlas_samples(
    int roots,
    int samples,
    uint64_t seed,
    int stack_min,
    int stack_max) {
    FivePowerEstimator estimator(samples);
    std::mt19937_64 rng(seed);
    FiveAtlasSamples observations;
    observations.power.reserve(static_cast<size_t>(roots) * kSeats5);
    observations.stacks.reserve(static_cast<size_t>(roots) * kSeats5);
    for (int root_index = 0; root_index < roots; ++root_index) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        const auto stacks = sample_five_stacks(rng, stack_min, stack_max);
        const FiveState state = make_five_root(deck, stacks, 1).seventh;
        if (state.terminal) continue;
        for (int seat = 0; seat < kSeats5; ++seat) {
            if (!state.players[seat].folded) {
                observations.power.push_back(estimator.get(state, seat));
                observations.stacks.push_back(
                    five_stack_features(state, seat));
            }
        }
    }
    return observations;
}

int stack_group(int stack) {
    if (stack < 100) return 0;
    if (stack < 200) return 1;
    if (stack < 500) return 2;
    return 3;
}

FiveEvalSummary evaluate_five_solver(
    FiveMCCFR& solver,
    int deals,
    uint64_t seed,
    int stack_min,
    int stack_max) {
    FiveEvalSummary summary;
    summary.deal_values.reserve(deals);
    std::mt19937_64 rng(seed);
    for (int deal = 0; deal < deals; ++deal) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        const auto stacks = sample_five_stacks(rng, stack_min, stack_max);
        const FiveState root = make_five_root(deck, stacks, 1).seventh;
        std::vector<double> rotations;
        rotations.reserve(kSeats5);
        std::array<std::vector<double>, 4> deal_stack_values;
        for (int target = 0; target < kSeats5; ++target) {
            FiveState state = root;
            if (!state.terminal) {
                while (!state.terminal) {
                    const int actor = state.actor;
                    const Action action = actor == target
                        ? solver.choose(state, actor)
                        : five_heuristic(state, actor);
                    if (actor == target) ++summary.actions[action];
                    five_apply(state, action);
                }
            }
            const double value = five_payoffs(state)[target];
            rotations.push_back(value);
            deal_stack_values[stack_group(stacks[target])].push_back(value);
        }
        summary.deal_values.push_back(
            std::accumulate(rotations.begin(), rotations.end(), 0.0) /
            rotations.size());
        for (int group = 0; group < 4; ++group) {
            if (deal_stack_values[group].empty()) continue;
            summary.stack_values[group].push_back(
                std::accumulate(
                    deal_stack_values[group].begin(),
                    deal_stack_values[group].end(),
                    0.0) /
                deal_stack_values[group].size());
        }
    }
    return summary;
}

struct NumericSummary {
    double mean = 0.0;
    double standard_error = 0.0;
};

NumericSummary summarize_five(const std::vector<double>& values) {
    NumericSummary summary;
    if (values.empty()) return summary;
    summary.mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    if (values.size() > 1) {
        double squared = 0.0;
        for (double value : values) {
            squared += (value - summary.mean) * (value - summary.mean);
        }
        summary.standard_error = std::sqrt(
            squared / (values.size() - 1) / values.size());
    }
    return summary;
}

struct FiveOptions {
    uint64_t root_iterations = 0;
    uint64_t report_every = 0;
    uint64_t seed = 7;
    int eval_deals = 1000;
    int fit_roots = 1000;
    int clusters = 64;
    int stack_clusters = 16;
    int mc_samples = 16;
    int stack_min = 50;
    int stack_max = 1000;
    bool self_test = false;
    std::string load;
    std::string save;
};

FiveOptions parse_five_options(int argc, char** argv) {
    FiveOptions options;
    auto value = [&](int& index) {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return std::string(argv[index]);
    };
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--root-iterations") {
            options.root_iterations = std::stoull(value(index));
        } else if (arg == "--report-every") {
            options.report_every = std::stoull(value(index));
        } else if (arg == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (arg == "--eval-deals") {
            options.eval_deals = std::stoi(value(index));
        } else if (arg == "--fit-roots") {
            options.fit_roots = std::stoi(value(index));
        } else if (arg == "--clusters") {
            options.clusters = std::stoi(value(index));
        } else if (arg == "--stack-clusters") {
            options.stack_clusters = std::stoi(value(index));
        } else if (arg == "--mc-samples") {
            options.mc_samples = std::stoi(value(index));
        } else if (arg == "--stack-min") {
            options.stack_min = std::stoi(value(index));
        } else if (arg == "--stack-max") {
            options.stack_max = std::stoi(value(index));
        } else if (arg == "--load") {
            options.load = value(index);
        } else if (arg == "--save") {
            options.save = value(index);
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown five-player option: " + arg);
        }
    }
    if (options.eval_deals <= 0 ||
        options.fit_roots <= 0 ||
        options.clusters <= 0 ||
        options.stack_clusters <= 0 ||
        options.mc_samples <= 0 ||
        options.stack_min <= 0 ||
        options.stack_min > options.stack_max) {
        throw std::runtime_error("invalid five-player options");
    }
    return options;
}

void five_self_test() {
    std::mt19937_64 rng(19);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    const std::array<int, kSeats5> stacks = {50, 100, 200, 500, 1000};
    FiveState root = make_five_root(deck, stacks, 1).seventh;
    assert(root.pot >= kSeats5);
    const auto utility = five_payoffs(root);
    const double total =
        std::accumulate(utility.begin(), utility.end(), 0.0);
    assert(std::abs(total) < 1e-9);

    FivePowerEstimator estimator(8);
    std::vector<FivePower> powers;
    if (!root.terminal) {
        for (int seat = 0; seat < kSeats5; ++seat) {
            if (!root.players[seat].folded) {
                const auto& power = estimator.get(root, seat);
                assert(std::all_of(
                    power.begin(), power.end(),
                    [](double value) {
                        return value >= 0.0 && value <= 1.0;
                    }));
                powers.push_back(power);
            }
        }
    }
    if (powers.empty()) powers.push_back(FivePower{});
    std::vector<FiveStackFeature> stack_samples = {
        five_stack_features(root, 0)};
    FiveAtlas atlas;
    atlas.fit(
        powers,
        stack_samples,
        std::min(4, static_cast<int>(powers.size())),
        1,
        23);
    FiveMCCFR solver(std::move(atlas), 8, 29);
    if (!root.terminal) solver.train_root(root);
    std::cout << "{\"self_test\":\"ok\",\"buckets\":"
              << solver.buckets() << "}\n";
}

void print_five_result(
    const FiveOptions& options,
    const FiveMCCFR& solver,
    const FiveEvalSummary& evaluation,
    double training_seconds) {
    const auto overall = summarize_five(evaluation.deal_values);
    static const std::array<const char*, 4> labels = {
        "50-100", "100-200", "200-500", "500-1000"};
    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"solver\": \"five-player-external-sampling-mccfr-plus\",\n"
              << "  \"trained_street\": \"7th\",\n"
              << "  \"pre_seventh_policy\": \"heuristic\",\n"
              << "  \"root_iterations\": " << options.root_iterations << ",\n"
              << "  \"training_traversals\": " << solver.traversals() << ",\n"
              << "  \"training_node_visits\": " << solver.node_visits() << ",\n"
              << "  \"buckets\": " << solver.buckets() << ",\n"
              << "  \"power_clusters\": " << solver.atlas().centers().size() << ",\n"
              << "  \"stack_clusters\": "
              << solver.atlas().stack_centers().size() << ",\n"
              << "  \"power_mc_samples\": " << options.mc_samples << ",\n"
              << "  \"power_cache_entries\": " << solver.power_cache_size() << ",\n"
              << "  \"stack_sampling\": {\"distribution\":\"log-uniform\","
              << "\"min\":" << options.stack_min
              << ",\"max\":" << options.stack_max << "},\n"
              << "  \"evaluation\": {\n"
              << "    \"base_deals\": " << options.eval_deals << ",\n"
              << "    \"hands\": " << options.eval_deals * kSeats5 << ",\n"
              << "    \"target_average_profit_ante\": {"
              << "\"mean\":" << overall.mean
              << ",\"standard_error\":" << overall.standard_error
              << ",\"ci95\":[" << overall.mean - 1.96 * overall.standard_error
              << "," << overall.mean + 1.96 * overall.standard_error << "]},\n"
              << "    \"by_target_stack\": {";
    for (int group = 0; group < 4; ++group) {
        if (group) std::cout << ",";
        const auto summary = summarize_five(evaluation.stack_values[group]);
        std::cout << "\"" << labels[group] << "\":{"
                  << "\"base_deal_samples\":"
                  << evaluation.stack_values[group].size()
                  << ",\"mean\":" << summary.mean
                  << ",\"standard_error\":" << summary.standard_error
                  << ",\"ci95\":["
                  << summary.mean - 1.96 * summary.standard_error << ","
                  << summary.mean + 1.96 * summary.standard_error << "]}";
    }
    std::cout << "},\n"
              << "    \"policy_misses\": " << solver.policy_misses() << ",\n"
              << "    \"action_counts\": {";
    bool first = true;
    for (int action = 0; action < kActionCount; ++action) {
        if (!evaluation.actions[action]) continue;
        if (!first) std::cout << ",";
        first = false;
        std::cout << "\"" << kActionNames[action] << "\":"
                  << evaluation.actions[action];
    }
    std::cout << "}\n"
              << "  },\n"
              << "  \"training_elapsed_seconds\": "
              << training_seconds << "\n"
              << "}\n";
}

}  // namespace

#ifndef STUD5_NO_MAIN
int main(int argc, char** argv) {
    try {
        const FiveOptions options = parse_five_options(argc, argv);
        if (options.self_test) {
            five_self_test();
            return 0;
        }

        FiveAtlas atlas;
        if (options.load.empty()) {
            std::cerr << "collecting public-card-aware power samples...\n";
            const auto samples = collect_five_atlas_samples(
                options.fit_roots,
                options.mc_samples,
                options.seed ^ 0x517cc1b727220a95ull,
                options.stack_min,
                options.stack_max);
            std::cerr << "fitting " << options.clusters
                      << " hard power centroids from "
                      << samples.power.size() << " observations and "
                      << options.stack_clusters
                      << " hard stack centroids...\n";
            atlas.fit(
                samples.power,
                samples.stacks,
                options.clusters,
                options.stack_clusters,
                options.seed);
        }
        FiveMCCFR solver(
            std::move(atlas),
            options.mc_samples,
            options.seed ^ 0x9e3779b97f4a7c15ull);
        if (!options.load.empty()) solver.load(options.load);

        std::mt19937_64 rng(
            options.seed ^ 0xa0761d6478bd642full);
        const auto training_started = std::chrono::steady_clock::now();
        for (uint64_t iteration = 1;
             iteration <= options.root_iterations;
             ++iteration) {
            FiveState root;
            do {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), rng);
                const auto stacks = sample_five_stacks(
                    rng, options.stack_min, options.stack_max);
                root = make_five_root(deck, stacks, 1).seventh;
            } while (root.terminal);
            solver.train_root(root);
            if (options.report_every &&
                (iteration % options.report_every == 0 ||
                 iteration == options.root_iterations)) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    training_started).count();
                std::cerr
                    << "{\"root_iteration\":" << iteration
                    << ",\"traversals\":" << solver.traversals()
                    << ",\"training_node_visits\":" << solver.node_visits()
                    << ",\"buckets\":" << solver.buckets()
                    << ",\"power_cache_entries\":"
                    << solver.power_cache_size()
                    << ",\"elapsed_seconds\":" << elapsed << "}\n";
            }
        }
        const double training_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            training_started).count();
        if (!options.save.empty()) solver.save(options.save);

        const auto evaluation = evaluate_five_solver(
            solver,
            options.eval_deals,
            options.seed ^ 0xe7037ed1a0b428dbull,
            options.stack_min,
            options.stack_max);
        print_five_result(
            options, solver, evaluation, training_seconds);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
#endif
