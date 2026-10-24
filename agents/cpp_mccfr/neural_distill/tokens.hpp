// Observer-visible encodings of a C++ 7-stud State for the distilled student networks.
// Include after the engine (State, Card, valid_mask, stack_cap, kActionCount). See README.md.
// Reads only the viewer's own hidden/discarded cards, both players' shown cards, the public
// betting history and public chip counts. Never reads the opponent's hidden cards/discard or
// simulation_deck, so the existing abstraction_lbr privacy self-test applies unchanged.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace neural_distill {

// Token fields: type, rank, suit, slot, street, actor, action, misc. Index 0 always means "none".
constexpr int kTokenFields = 8;
constexpr int kMaxTokens = 48;
constexpr int kMaxEvents = 32;
constexpr int kNumerics = 20;
constexpr int kFlatCardSlots = 12;
constexpr int kFlatCardWidth = 18;   // rank 13 + suit 4 + present 1
constexpr int kFlatEventWidth = 14;  // street 3 + relative actor 2 + action 8 + present 1
constexpr int kFlatSize = 2 + kFlatCardSlots * kFlatCardWidth + kMaxEvents * kFlatEventWidth + kNumerics;
// Teacher abstraction key (MCCFR power-memory16 InfoKey fields) for the 'mlp_key' attribution control:
// street-5, power_cluster (two bytes), rank_bucket, pot_odds, stack_pot, own_bets, opponent_bets, checked,
// last_action_class, betting_goal, legal_mask. One-hot layout below, mirrored by train.py::key_flat.
constexpr int kKeyFields = 12;
using KeyFeatures = std::array<uint8_t, 16>;
constexpr std::array<int, 11> kKeyWidths = {3, 512, 16, 3, 4, 4, 4, 2, 6, 3, 8};
constexpr int kKeyFlatSize = 3 + 512 + 16 + 3 + 4 + 4 + 4 + 2 + 6 + 3 + 8;
// Embedding table sizes (index 0 = none), shared with train.py.
constexpr std::array<int, kTokenFields> kVocab = {8, 14, 5, 32, 4, 3, 9, 3};

enum TokenType : int8_t {
    kContext = 1,
    kOwnHidden = 2,
    kOwnShown = 3,
    kOpponentShown = 4,
    kOwnDiscard = 5,
    kEvent = 6,
    kQuery = 7,
};

using Token = std::array<int8_t, kTokenFields>;

struct Encoded {
    std::vector<Token> tokens;
    std::array<float, kNumerics> numerics{};
    uint8_t mask = 0;
};

inline Token card_token(TokenType type, const Card& card, int slot) {
    Token token{};
    token[0] = type;
    token[1] = static_cast<int8_t>(card.rank - 1);  // 2..14 -> 1..13
    token[2] = static_cast<int8_t>(card.suit + 1);  // 0..3 -> 1..4
    token[3] = static_cast<int8_t>(slot);
    return token;
}

inline Encoded encode(const State& state, int viewer) {
    Encoded out;
    const Player& own = state.players[viewer];
    const Player& other = state.players[1 - viewer];
    Token context{};
    context[0] = kContext;
    context[4] = static_cast<int8_t>(state.street - 4);
    context[7] = static_cast<int8_t>(viewer + 1);
    out.tokens.push_back(context);
    const auto add_cards = [&](TokenType type, const std::vector<Card>& cards, int capacity) {
        if (static_cast<int>(cards.size()) > capacity) {
            throw std::runtime_error("neural_distill card capacity exceeded");
        }
        for (int slot = 0; slot < static_cast<int>(cards.size()); ++slot) {
            out.tokens.push_back(card_token(type, cards[slot], slot));
        }
    };
    add_cards(kOwnHidden, own.hidden, 3);
    add_cards(kOwnShown, own.shown, 4);
    add_cards(kOpponentShown, other.shown, 4);
    if (own.has_discard) {
        out.tokens.push_back(card_token(kOwnDiscard, own.discarded, 0));
    }
    if (static_cast<int>(state.history.size()) > kMaxEvents) {
        throw std::runtime_error("neural_distill history capacity exceeded");
    }
    for (int index = 0; index < static_cast<int>(state.history.size()); ++index) {
        const Event& event = state.history[index];
        Token token{};
        token[0] = kEvent;
        token[3] = static_cast<int8_t>(index);
        token[4] = static_cast<int8_t>(event.street - 4);
        token[5] = static_cast<int8_t>(event.actor == viewer ? 1 : 2);
        token[6] = static_cast<int8_t>(event.action + 1);
        out.tokens.push_back(token);
    }
    Token query{};
    query[0] = kQuery;
    query[4] = static_cast<int8_t>(state.street - 4);
    query[7] = static_cast<int8_t>(viewer + 1);
    out.tokens.push_back(query);
    if (static_cast<int>(out.tokens.size()) > kMaxTokens) {
        throw std::runtime_error("neural_distill token capacity exceeded");
    }

    const double ante = std::max(1, state.ante);
    const auto chips = [&](double value) {
        return static_cast<float>(std::log1p(std::max(0.0, value) / ante));
    };
    const int call = std::max(0, state.highest_bet - own.round_bet);
    out.mask = valid_mask(state, viewer);
    auto& n = out.numerics;
    n[0] = chips(state.pot);
    n[1] = chips(stack_cap(state, viewer) - own.invested);
    n[2] = chips(stack_cap(state, 1 - viewer) - other.invested);
    n[3] = chips(own.invested);
    n[4] = chips(other.invested);
    n[5] = static_cast<float>(call / static_cast<double>(std::max(1, state.pot + call)));
    n[6] = chips(state.highest_bet);
    n[7] = chips(call);
    n[8] = static_cast<float>(state.raise_count / 3.0);
    n[9 + (state.street - 5)] = 1.0f;
    for (int action = 0; action < kActionCount; ++action) {
        n[12 + action] = out.mask & (1u << action) ? 1.0f : 0.0f;
    }
    return out;
}

// Same information as the token list, laid out in fixed slots for the MLP student.
// train.py builds the identical vector from stored tokens (flat_from_tokens).
inline std::vector<float> flat_from_tokens(const std::vector<Token>& tokens,
                                           const std::array<float, kNumerics>& numerics) {
    std::vector<float> x(kFlatSize, 0.0f);
    const int card_base = 2;
    const int event_base = card_base + kFlatCardSlots * kFlatCardWidth;
    const int numeric_base = event_base + kMaxEvents * kFlatEventWidth;
    for (const Token& token : tokens) {
        const int type = token[0];
        if (type == kContext) {
            x[token[7] - 1] = 1.0f;
        } else if (type >= kOwnHidden && type <= kOwnDiscard) {
            static const std::array<int, 4> first_slot = {0, 3, 7, 11};
            const int slot = first_slot[type - kOwnHidden] + token[3];
            const int base = card_base + slot * kFlatCardWidth;
            x[base + token[1] - 1] = 1.0f;
            x[base + 13 + token[2] - 1] = 1.0f;
            x[base + 17] = 1.0f;
        } else if (type == kEvent) {
            const int base = event_base + token[3] * kFlatEventWidth;
            x[base + token[4] - 1] = 1.0f;
            x[base + 3 + token[5] - 1] = 1.0f;
            x[base + 5 + token[6] - 1] = 1.0f;
            x[base + 13] = 1.0f;
        }
    }
    for (int i = 0; i < kNumerics; ++i) {
        x[numeric_base + i] = numerics[i];
    }
    return x;
}

inline std::vector<float> key_flat(const KeyFeatures& k) {
    std::vector<float> x(kKeyFlatSize, 0.0f);
    const int power = k[1] | (k[2] << 8);
    const std::array<int, 10> values = {k[0], power, k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10]};
    int base = 0;
    for (int i = 0; i < 10; ++i) {
        if (values[i] < 0 || values[i] >= kKeyWidths[i]) {
            throw std::runtime_error("key feature out of range");
        }
        x[base + values[i]] = 1.0f;
        base += kKeyWidths[i];
    }
    for (int bit = 0; bit < 8; ++bit) {
        x[base + bit] = (k[11] >> bit) & 1 ? 1.0f : 0.0f;
    }
    return x;
}

} // namespace neural_distill
