// Game rules shared by the existing solver translation units.
// Included inside their anonymous namespace, after constants/action IDs.
// Build and self-test through the owning agent; see PROJECT_STRUCTURE.md.
#pragma once

struct Card {
    uint8_t rank = 2;
    uint8_t suit = 0;

    bool operator==(const Card& other) const {
        return rank == other.rank && suit == other.suit;
    }
};

std::vector<Card> fresh_deck() {
    std::vector<Card> deck;
    deck.reserve(52);
    for (int suit = 0; suit < 4; ++suit) {
        for (int rank = 2; rank <= 14; ++rank) {
            deck.push_back(Card{static_cast<uint8_t>(rank), static_cast<uint8_t>(suit)});
        }
    }
    return deck;
}

using Score = std::array<int, 6>;

int straight_rank(const std::array<int, 5>& values) {
    if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
        return 0;
    }
    if (values == std::array<int, 5>{14, 13, 12, 11, 10}) {
        return 15;
    }
    if (values == std::array<int, 5>{14, 5, 4, 3, 2}) {
        return 14;  // Match the existing Python evaluator.
    }
    return values.front() - values.back() == 4 ? values.front() : 0;
}

Score evaluate_five(const std::array<Card, 5>& cards) {
    std::array<int, 15> counts{};
    std::array<int, 5> values{};
    for (int i = 0; i < 5; ++i) {
        values[i] = cards[i].rank;
        ++counts[cards[i].rank];
    }
    std::sort(values.begin(), values.end(), std::greater<int>());
    const bool flush = std::all_of(
        cards.begin() + 1, cards.end(),
        [&](const Card& card) { return card.suit == cards[0].suit; });
    const int straight = straight_rank(values);

    std::array<std::pair<int, int>, 5> groups{};
    int group_count = 0;
    for (int rank = 2; rank <= 14; ++rank) {
        if (counts[rank]) groups[group_count++] = {counts[rank], rank};
    }
    std::sort(groups.begin(), groups.begin() + group_count, std::greater<>());

    Score score{};
    if (straight && flush) {
        score = {8, straight, 0, 0, 0, 0};
    } else if (groups[0].first == 4) {
        score = {7, groups[0].second, groups[1].second, 0, 0, 0};
    } else if (groups[0].first == 3 && groups[1].first == 2) {
        score = {6, groups[0].second, groups[1].second, 0, 0, 0};
    } else if (flush) {
        score = {5, values[0], values[1], values[2], values[3], values[4]};
    } else if (straight) {
        score = {4, straight, 0, 0, 0, 0};
    } else if (groups[0].first == 3) {
        score[0] = 3;
        score[1] = groups[0].second;
        int out = 2;
        for (int value : values) if (value != groups[0].second) score[out++] = value;
    } else if (groups[0].first == 2 && groups[1].first == 2) {
        const int high = std::max(groups[0].second, groups[1].second);
        const int low = std::min(groups[0].second, groups[1].second);
        int kicker = 0;
        for (int value : values) if (value != high && value != low) kicker = value;
        score = {2, high, low, kicker, 0, 0};
    } else if (groups[0].first == 2) {
        score[0] = 1;
        score[1] = groups[0].second;
        int out = 2;
        for (int value : values) if (value != groups[0].second) score[out++] = value;
    } else {
        score = {0, values[0], values[1], values[2], values[3], values[4]};
    }
    return score;
}

Score best_hand_direct(const std::vector<Card>& cards) {
    if (cards.size() < 5) return {};
    Score best{};
    bool initialized = false;
    for (size_t a = 0; a + 4 < cards.size(); ++a)
        for (size_t b = a + 1; b + 3 < cards.size(); ++b)
            for (size_t c = b + 1; c + 2 < cards.size(); ++c)
                for (size_t d = c + 1; d + 1 < cards.size(); ++d)
                    for (size_t e = d + 1; e < cards.size(); ++e) {
                        const Score score = evaluate_five(
                            {cards[a], cards[b], cards[c], cards[d], cards[e]});
                        if (!initialized || score > best) {
                            best = score;
                            initialized = true;
                        }
                    }
    return best;
}

uint32_t encode_score(const Score& score) {
    uint32_t encoded = 0;
    for (int value : score) encoded = (encoded << 4) | value;
    return encoded;
}

Score decode_score(uint32_t encoded) {
    Score score{};
    for (int index = 5; index >= 0; --index) {
        score[index] = encoded & 15u;
        encoded >>= 4;
    }
    return score;
}

const std::array<std::array<uint32_t, 6>, 53>& binomial_table() {
    static const auto table = [] {
        std::array<std::array<uint32_t, 6>, 53> result{};
        for (int n = 0; n <= 52; ++n) {
            result[n][0] = 1;
            for (int k = 1; k <= 5; ++k) {
                result[n][k] = n == 0
                    ? 0
                    : result[n - 1][k - 1] + result[n - 1][k];
            }
        }
        return result;
    }();
    return table;
}

uint32_t five_card_index(std::array<int, 5> indices) {
    std::sort(indices.begin(), indices.end());
    const auto& choose = binomial_table();
    uint32_t index = 0;
    for (int i = 0; i < 5; ++i) index += choose[indices[i]][i + 1];
    return index;
}

int card_index(const Card& card) {
    return card.suit * 13 + card.rank - 2;
}

const std::vector<uint32_t>& five_card_score_table() {
    static const auto table = [] {
        constexpr uint32_t count = 2598960;
        std::vector<uint32_t> result(count);
        const auto deck = fresh_deck();
        const auto& choose = binomial_table();
        for (int a = 0; a + 4 < 52; ++a)
            for (int b = a + 1; b + 3 < 52; ++b)
                for (int c = b + 1; c + 2 < 52; ++c)
                    for (int d = c + 1; d + 1 < 52; ++d)
                        for (int e = d + 1; e < 52; ++e) {
                            const uint32_t index = choose[a][1] + choose[b][2] +
                                choose[c][3] + choose[d][4] + choose[e][5];
                            result[index] = encode_score(evaluate_five({
                                deck[a], deck[b], deck[c], deck[d], deck[e]}));
                        }
        return result;
    }();
    return table;
}

uint32_t score_five_encoded(const std::array<Card, 5>& cards) {
    return five_card_score_table()[five_card_index({
        card_index(cards[0]), card_index(cards[1]), card_index(cards[2]),
        card_index(cards[3]), card_index(cards[4])})];
}

uint32_t best_hand_encoded(const std::vector<Card>& cards) {
    if (cards.size() < 5) return 0;
    const auto& table = five_card_score_table();
    const auto& choose = binomial_table();
    std::array<int, 7> indices{};
    for (size_t index = 0; index < cards.size(); ++index) {
        indices[index] = card_index(cards[index]);
    }
    std::sort(indices.begin(), indices.begin() + cards.size());
    uint32_t best = 0;
    for (size_t a = 0; a + 4 < cards.size(); ++a)
        for (size_t b = a + 1; b + 3 < cards.size(); ++b)
            for (size_t c = b + 1; c + 2 < cards.size(); ++c)
                for (size_t d = c + 1; d + 1 < cards.size(); ++d)
                    for (size_t e = d + 1; e < cards.size(); ++e) {
                        const uint32_t index = choose[indices[a]][1] +
                            choose[indices[b]][2] + choose[indices[c]][3] +
                            choose[indices[d]][4] + choose[indices[e]][5];
                        best = std::max(best, table[index]);
                    }
    return best;
}

Score best_hand(const std::vector<Card>& cards) {
    return decode_score(best_hand_encoded(cards));
}

std::vector<int> public_priority(const std::vector<Card>& cards) {
    if (cards.empty()) return {-1};
    if (cards.size() >= 5) {
        const auto score = best_hand(cards);
        return std::vector<int>(score.begin(), score.end());
    }
    std::array<int, 15> counts{};
    std::vector<int> values;
    for (const Card& card : cards) {
        ++counts[card.rank];
        values.push_back(card.rank);
    }
    std::sort(values.begin(), values.end(), std::greater<int>());
    std::vector<std::pair<int, int>> groups;
    for (int rank = 2; rank <= 14; ++rank) {
        if (counts[rank]) groups.emplace_back(counts[rank], rank);
    }
    std::sort(groups.begin(), groups.end(), std::greater<>());
    if (groups[0].first == 4) return {7, groups[0].second};
    if (groups[0].first == 3) {
        std::vector<int> out = {3, groups[0].second};
        for (int value : values) if (value != groups[0].second) out.push_back(value);
        return out;
    }
    if (groups[0].first == 2 && groups.size() > 1 && groups[1].first == 2) {
        const int high = std::max(groups[0].second, groups[1].second);
        const int low = std::min(groups[0].second, groups[1].second);
        std::vector<int> out = {2, high, low};
        for (int value : values) if (value != high && value != low) out.push_back(value);
        return out;
    }
    if (groups[0].first == 2) {
        std::vector<int> out = {1, groups[0].second};
        for (int value : values) if (value != groups[0].second) out.push_back(value);
        return out;
    }
    std::vector<int> out = {0};
    out.insert(out.end(), values.begin(), values.end());
    return out;
}

struct Player {
    std::vector<Card> hidden;
    std::vector<Card> shown;
    Card discarded{};
    bool has_discard = false;
    int stack_cap = 0;
    int invested = 0;
    int round_bet = 0;
    bool folded = false;
    bool all_in = false;
};

struct Event {
    uint8_t street = 5;
    uint8_t actor = 0;
    Action action = CHECK;
};

struct State {
    std::array<Player, 2> players;
    std::vector<Event> history;
    std::vector<Card> simulation_deck;
    int ante = 1;
    int effective_stack = kEffectiveStackAnte;
    int pot = 0;
    int highest_bet = 0;
    int raise_count = 0;
    int street = 5;
    int actor = 0;
    bool terminal = false;
};

int stack_cap(const State& state, int seat) {
    return state.players[seat].stack_cap > 0
        ? state.players[seat].stack_cap
        : state.effective_stack;
}

int street_cap(int street) {
    if (street == 5) return 1;
    if (street == 6) return 2;
    if (street == 7) return 3;
    return 0;
}

bool checked_this_street(const State& state, int seat) {
    return std::any_of(state.history.begin(), state.history.end(), [&](const Event& event) {
        return event.street == state.street && event.actor == seat && event.action == CHECK;
    });
}

int bets_this_street(const State& state, int seat) {
    return static_cast<int>(std::count_if(
        state.history.begin(), state.history.end(), [&](const Event& event) {
            return event.street == state.street && event.actor == seat && aggressive(event.action);
        }));
}

uint8_t valid_mask(const State& state, int seat) {
    const Player& player = state.players[seat];
    if (state.terminal || player.folded || player.all_in) return 0;
    const int call_amount = std::max(0, state.highest_bet - player.round_bet);
    uint8_t mask = static_cast<uint8_t>(1u << FOLD);
    mask |= static_cast<uint8_t>(1u << (call_amount == 0 ? CHECK : CALL));
    const int remaining = stack_cap(state, seat) - player.invested;
    const bool can_raise =
        !checked_this_street(state, seat) &&
        bets_this_street(state, seat) < street_cap(state.street);
    if (can_raise && state.highest_bet == 0 && remaining > 0) {
        mask |= static_cast<uint8_t>(1u << BBING);
    }
    if (can_raise && state.pot > 0 && remaining > call_amount) {
        if (state.highest_bet > 0) mask |= static_cast<uint8_t>(1u << DDADANG);
        mask |= static_cast<uint8_t>((1u << QUARTER) | (1u << HALF));
    }
    return mask;
}

std::vector<Action> actions_from_mask(uint8_t mask) {
    std::vector<Action> actions;
    for (int action = 0; action < kActionCount; ++action) {
        if (mask & (1u << action)) actions.push_back(static_cast<Action>(action));
    }
    return actions;
}

int raise_amount(const State& state, Action action, int call_amount) {
    const int pot_after_call = state.pot + call_amount;
    if (action == BBING) return state.ante;
    if (action == DDADANG) return std::max(1, state.highest_bet);
    if (action == QUARTER) return std::max(1, (pot_after_call + 3) / 4);
    if (action == HALF) return std::max(1, (pot_after_call + 1) / 2);
    return 0;
}

enum class ActionResult { Continue, Raise, RoundEnd, FoldEnd };

ActionResult apply_action(State& state, int seat, Action action) {
    if (!(valid_mask(state, seat) & (1u << action))) {
        throw std::runtime_error("invalid action");
    }
    Player& player = state.players[seat];
    Action previous = FULL;
    bool has_previous = false;
    for (auto it = state.history.rbegin(); it != state.history.rend(); ++it) {
        if (it->street != state.street) break;
        previous = it->action;
        has_previous = true;
        break;
    }
    const int old_highest = state.highest_bet;
    const int call_amount = std::max(0, old_highest - player.round_bet);

    if (action == FOLD) {
        player.folded = true;
        state.history.push_back(Event{
            static_cast<uint8_t>(state.street), static_cast<uint8_t>(seat), action});
        state.terminal = true;
        return ActionResult::FoldEnd;
    }
    if (action != CHECK) {
        const int requested = call_amount + raise_amount(state, action, call_amount);
        const int paid = std::min(requested, stack_cap(state, seat) - player.invested);
        player.round_bet += paid;
        player.invested += paid;
        player.all_in = player.invested >= stack_cap(state, seat);
        state.pot += paid;
        state.highest_bet = std::max(state.highest_bet, player.round_bet);
        if (aggressive(action)) ++state.raise_count;
    }
    state.history.push_back(Event{
        static_cast<uint8_t>(state.street), static_cast<uint8_t>(seat), action});
    if (state.highest_bet > old_highest) return ActionResult::Raise;
    if (action == CHECK && (!has_previous || previous != CHECK)) {
        return ActionResult::Continue;
    }
    return ActionResult::RoundEnd;
}

std::vector<Card> all_cards(const Player& player) {
    std::vector<Card> cards = player.hidden;
    cards.insert(cards.end(), player.shown.begin(), player.shown.end());
    return cards;
}

double terminal_net_search(const State& state, int seat) {
    const Player& player = state.players[seat];
    const Player& opponent = state.players[1 - seat];
    int award = 0;
    if (player.folded) {
        award = 0;
    } else if (opponent.folded) {
        award = state.pot;
    } else {
        const Score own = best_hand(all_cards(player));
        const Score other = best_hand(all_cards(opponent));
        if (own > other) award = state.pot;
        else if (own == other) award = state.pot / 2 + ((state.pot % 2 && seat == 0) ? 1 : 0);
    }
    return static_cast<double>(award - player.invested) / state.ante;
}

