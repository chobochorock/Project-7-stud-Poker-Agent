#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "local_split_helpers.hpp"

namespace {

constexpr int kActionCount = 8;
constexpr int kEffectiveStackAnte = 1000;
constexpr int kPowerDimensions = 18;
constexpr int kDeepCfrCardSlots = 12;
constexpr int kDeepCfrHistorySlots = 24;
constexpr int kDeepCfrTensorSize =
    2 + 3 + kDeepCfrCardSlots * 53 +
    kDeepCfrHistorySlots * 49 + 7 + kActionCount;
constexpr std::array<double, 5> kTemperatureCandidates = {
    0.25, 0.5, 1.0, 2.0, 4.0};
constexpr uint64_t kSoftTemperatureStride = 16;
constexpr size_t kResidualPointCap = 64;
constexpr double kChildMassFraction = 0.1;

void write_progress(const std::string& line) {
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cerr << line << "\n";
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        escaped.push_back(character);
    }
    return escaped;
}

class ProgressHeartbeat {
public:
    ProgressHeartbeat(
        std::string solver,
        std::string phase,
        std::string unit,
        uint64_t total,
        double interval_seconds)
        : solver_(std::move(solver)),
          phase_(std::move(phase)),
          unit_(std::move(unit)),
          total_(total),
          interval_seconds_(interval_seconds),
          started_(std::chrono::steady_clock::now()) {
        if (interval_seconds_ > 0) {
            worker_ = std::thread([this] { run(); });
        }
    }

    ~ProgressHeartbeat() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        wake_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void update(uint64_t completed) {
        completed_.store(completed, std::memory_order_relaxed);
    }

private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!wake_.wait_for(
            lock,
            std::chrono::duration<double>(interval_seconds_),
            [this] { return stopped_; })) {
            lock.unlock();
            const uint64_t completed =
                completed_.load(std::memory_order_relaxed);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_).count();
            const double rate = completed / std::max(1e-9, elapsed);
            std::ostringstream output;
            output << "{\"solver\":\"" << solver_
                   << "\",\"phase\":\"" << phase_ << "-heartbeat\""
                   << ",\"" << unit_ << "_completed\":" << completed
                   << ",\"" << unit_ << "_total\":" << total_
                   << ",\"progress_percent\":"
                   << (total_ ? 100.0 * completed / total_ : 0.0)
                   << ",\"" << unit_ << "_per_second\":" << rate
                   << ",\"elapsed_seconds\":" << elapsed
                   << ",\"eta_seconds\":";
            if (completed) {
                output << (total_ - std::min(total_, completed)) /
                    std::max(1e-9, rate);
            } else {
                output << "null";
            }
            output << "}";
            write_progress(output.str());
            lock.lock();
        }
    }

    std::string solver_;
    std::string phase_;
    std::string unit_;
    uint64_t total_ = 0;
    double interval_seconds_ = 0.0;
    std::chrono::steady_clock::time_point started_;
    std::atomic<uint64_t> completed_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::thread worker_;
};

enum Action : uint8_t {
    CHECK = 0,
    BBING = 1,
    DDADANG = 2,
    QUARTER = 3,
    HALF = 4,
    FULL = 5,
    CALL = 6,
    FOLD = 7,
};

enum class SoftGrowthMode : uint8_t {
    None,
    Fixed,
    Mix,
    Simple,
    Point,
    Residual,
};

enum class InitialRegretPolicy : uint8_t {
    Fold,
    MadeCall,
};

enum class HeuristicVariant : uint8_t {
    Baseline,
    BluffDefense,
    OverBluff,
    HiddenBluff,
    SlowPlay,
};

constexpr std::array<HeuristicVariant, 5> kHeuristicPool = {
    HeuristicVariant::Baseline,
    HeuristicVariant::BluffDefense,
    HeuristicVariant::OverBluff,
    HeuristicVariant::HiddenBluff,
    HeuristicVariant::SlowPlay,
};

constexpr std::array<const char*, kActionCount> kActionNames = {
    "CHECK", "BBING", "DDADANG", "QUARTER",
    "HALF", "FULL", "CALL", "FOLD",
};

bool aggressive(Action action) {
    return action >= BBING && action <= FULL;
}

#include "stud_rules.hpp"

using DeepCfrTensor = std::array<float, kDeepCfrTensorSize>;

DeepCfrTensor deep_cfr_tensor(const State& state, int viewer) {
    DeepCfrTensor tensor{};
    size_t offset = 0;
    tensor[offset + viewer] = 1.0f;
    offset += 2;
    tensor[offset + state.street - 5] = 1.0f;
    offset += 3;

    const auto cards = [&](const std::vector<Card>& values, int slots) {
        if (static_cast<int>(values.size()) > slots) {
            throw std::runtime_error("Deep CFR card tensor capacity exceeded");
        }
        for (int slot = 0; slot < slots; ++slot) {
            const int token = slot < static_cast<int>(values.size())
                ? 1 + values[slot].suit * 13 + values[slot].rank - 2
                : 0;
            tensor[offset + slot * 53 + token] = 1.0f;
        }
        offset += slots * 53;
    };
    cards(state.players[viewer].hidden, 3);
    cards(state.players[viewer].shown, 4);
    cards(state.players[1 - viewer].shown, 4);
    const Card& discarded = state.players[viewer].discarded;
    const int discard_token = state.players[viewer].has_discard
        ? 1 + discarded.suit * 13 + discarded.rank - 2
        : 0;
    tensor[offset + discard_token] = 1.0f;
    offset += 53;

    if (state.history.size() > kDeepCfrHistorySlots) {
        throw std::runtime_error("Deep CFR history tensor capacity exceeded");
    }
    for (int slot = 0; slot < kDeepCfrHistorySlots; ++slot) {
        int token = 0;
        if (slot < static_cast<int>(state.history.size())) {
            const Event& event = state.history[slot];
            const int relative_actor = event.actor == viewer ? 0 : 1;
            token = 1 + (event.street - 5) * 16 +
                relative_actor * kActionCount + event.action;
        }
        tensor[offset + slot * 49 + token] = 1.0f;
    }
    offset += kDeepCfrHistorySlots * 49;

    const double ante = std::max(1, state.ante);
    const auto normalized = [&](double chips) {
        return static_cast<float>(std::log1p(std::max(0.0, chips) / ante));
    };
    const int call = std::max(
        0, state.highest_bet - state.players[viewer].round_bet);
    tensor[offset++] = normalized(state.pot);
    tensor[offset++] = normalized(
        stack_cap(state, viewer) - state.players[viewer].invested);
    tensor[offset++] = normalized(
        stack_cap(state, 1 - viewer) - state.players[1 - viewer].invested);
    tensor[offset++] = normalized(state.players[viewer].invested);
    tensor[offset++] = normalized(state.players[1 - viewer].invested);
    tensor[offset++] = static_cast<float>(
        call / static_cast<double>(std::max(1, state.pot + call)));
    tensor[offset++] = normalized(state.highest_bet);
    const uint8_t mask = valid_mask(state, viewer);
    for (int action = 0; action < kActionCount; ++action) {
        tensor[offset++] = mask & (1u << action) ? 1.0f : 0.0f;
    }
    assert(offset == tensor.size());
    return tensor;
}

double normalized_split_audit_terminal(
    const State& state,
    int seat,
    double chip_scale) {
    return terminal_net_search(state, seat) * state.ante / chip_scale;
}

double keep_value(const Card& card, const std::vector<Card>& cards) {
    int same_rank = -1;
    int same_suit = -1;
    int neighbors = 0;
    for (const Card& other : cards) {
        same_rank += other.rank == card.rank;
        same_suit += other.suit == card.suit;
        const int difference = std::abs(static_cast<int>(other.rank) - card.rank);
        if (!(other == card) && difference >= 1 && difference <= 4) ++neighbors;
    }
    return card.rank / 14.0 + same_rank * 1.25 + same_suit * 0.20 + neighbors * 0.08;
}

double reveal_value(const Card& card, const std::vector<Card>& cards) {
    int same_rank = -1;
    for (const Card& other : cards) same_rank += other.rank == card.rank;
    return card.rank + same_rank * 5.0;
}

std::pair<int, int> discard_reveal(const std::vector<Card>& cards) {
    int discard = 0;
    for (int i = 1; i < static_cast<int>(cards.size()); ++i) {
        const auto left = std::make_pair(keep_value(cards[i], cards), static_cast<int>(cards[i].rank));
        const auto right = std::make_pair(
            keep_value(cards[discard], cards), static_cast<int>(cards[discard].rank));
        if (left < right) discard = i;
    }
    int reveal = discard == 0 ? 1 : 0;
    for (int i = 0; i < static_cast<int>(cards.size()); ++i) {
        if (i != discard && reveal_value(cards[i], cards) > reveal_value(cards[reveal], cards)) {
            reveal = i;
        }
    }
    return {discard, reveal};
}

double made_strength(const Score& score) {
    const int category = score[0];
    if (category == 8) return 0.99;
    if (category == 7) return 0.96;
    if (category == 6) return 0.92;
    if (category == 5) return 0.86;
    if (category == 4) return 0.80;
    if (category == 3) return 0.72 + std::min(0.08, score[1] / 200.0);
    if (category == 2) return 0.60 + std::min(0.08, score[1] / 200.0);
    if (category == 1) return 0.38 + std::min(0.18, score[1] / 80.0);
    return 0.18 + std::min(0.14, score[1] / 100.0);
}

double partial_strength(const std::vector<Card>& cards) {
    std::array<int, 15> counts{};
    int high = 0;
    for (const Card& card : cards) {
        ++counts[card.rank];
        high = std::max(high, static_cast<int>(card.rank));
    }
    int max_count = 0;
    int group_rank = 0;
    int pair_count = 0;
    for (int rank = 2; rank <= 14; ++rank) {
        if (counts[rank] == 2) ++pair_count;
        if (std::make_pair(counts[rank], rank) > std::make_pair(max_count, group_rank)) {
            max_count = counts[rank];
            group_rank = rank;
        }
    }
    if (max_count == 4) return 0.92;
    if (max_count == 3) return 0.70 + std::min(0.08, group_rank / 200.0);
    if (pair_count >= 2) return 0.58 + std::min(0.06, high / 240.0);
    if (max_count == 2) return 0.36 + std::min(0.20, group_rank / 70.0);
    return 0.12 + std::min(0.18, high / 80.0);
}

double draw_bonus(const std::vector<Card>& cards) {
    std::array<int, 4> suits{};
    std::array<bool, 15> values{};
    for (const Card& card : cards) {
        ++suits[card.suit];
        values[card.rank] = true;
    }
    const int max_suit = *std::max_element(suits.begin(), suits.end());
    const double suit_bonus = max_suit >= 4 ? 0.12 : max_suit == 3 ? 0.06 : 0.0;
    if (values[14]) values[1] = true;
    int best = 0;
    for (int start = 1; start <= 10; ++start) {
        int count = 0;
        for (int rank = start; rank < start + 5; ++rank) count += values[rank];
        best = std::max(best, count);
    }
    return suit_bonus + (best >= 4 ? 0.10 : best == 3 ? 0.04 : 0.0);
}

double heuristic_strength(const State& state, int seat) {
    const std::vector<Card> own = all_cards(state.players[seat]);
    double strength = own.size() >= 5 ? made_strength(best_hand(own)) : partial_strength(own);
    strength += draw_bonus(own);

    const auto priority = public_priority(state.players[1 - seat].shown);
    const int category = priority[0];
    if (category >= 7) strength -= 0.24;
    else if (category == 3) strength -= 0.16;
    else if (category == 2) strength -= 0.12;
    else if (category == 1) strength -= 0.07;
    else if (priority.size() > 1 && priority[1] >= 13) strength -= 0.03;

    int recent_raises = 0;
    int checked = 0;
    for (auto it = state.history.rbegin(); it != state.history.rend() && checked < 6; ++it, ++checked) {
        if (it->actor != seat && aggressive(it->action)) ++recent_raises;
    }
    strength -= std::min(0.12, recent_raises * 0.04);

    const int call = std::max(0, state.highest_bet - state.players[seat].round_bet);
    if (call > 0) {
        const double pot_pressure = call / static_cast<double>(std::max(1, state.pot + call));
        const double chips = std::max(1, stack_cap(state, seat) - state.players[seat].invested);
        const double stack_pressure = call / chips;
        strength -= std::clamp(0.65 * pot_pressure + 0.35 * stack_pressure, 0.0, 1.0) * 0.08;
    }
    return std::clamp(strength, 0.0, 1.0);
}

Action first_valid(uint8_t mask, std::initializer_list<Action> preferences) {
    for (Action action : preferences) if (mask & (1u << action)) return action;
    return actions_from_mask(mask).front();
}

Action heuristic_call_or_fold(const State& state, int seat, double strength) {
    const uint8_t mask = valid_mask(state, seat);
    const int call = std::max(0, state.highest_bet - state.players[seat].round_bet);
    bool should_call = call <= 0;
    if (call > 0 && call <= state.ante) should_call = strength >= 0.22;
    else if (call <= state.ante * 2) should_call = strength >= 0.30;
    else if (call > 0) {
        const double odds = call / static_cast<double>(std::max(1, state.pot + call));
        should_call = strength >= std::min(0.72, std::max(0.28, odds + 0.08));
    }
    return should_call
        ? first_valid(mask, {CALL, CHECK, FOLD})
        : first_valid(mask, {FOLD, CHECK, CALL});
}

double opponent_public_completion_strength(
    const State& state,
    int viewer,
    int samples = 64) {
    const std::vector<Card>& shown = state.players[1 - viewer].shown;
    const int needed = std::max(0, 7 - static_cast<int>(shown.size()));
    if (!needed) return made_strength(best_hand(shown));

    uint64_t known = 0;
    auto add_known = [&](const std::vector<Card>& cards) {
        for (const Card& card : cards) {
            known |= 1ull << (card.suit * 13 + card.rank - 2);
        }
    };
    add_known(all_cards(state.players[viewer]));
    add_known(state.players[viewer].shown);
    add_known(shown);
    if (state.players[viewer].has_discard) {
        const Card card = state.players[viewer].discarded;
        known |= 1ull << (card.suit * 13 + card.rank - 2);
    }

    std::vector<Card> unseen;
    for (const Card& card : fresh_deck()) {
        const uint64_t bit = 1ull << (card.suit * 13 + card.rank - 2);
        if (!(known & bit)) unseen.push_back(card);
    }
    uint64_t seed = known ^ (static_cast<uint64_t>(state.street) << 57);
    for (const Card& card : shown) {
        seed ^= (card.suit * 13 + card.rank) * 0x9e3779b97f4a7c15ull;
    }
    std::mt19937_64 rng(seed);
    double total = 0.0;
    std::vector<Card> draw = unseen;
    std::vector<Card> completed;
    completed.reserve(7);
    for (int sample = 0; sample < samples; ++sample) {
        draw = unseen;
        for (int index = 0; index < needed; ++index) {
            const int selected = std::uniform_int_distribution<int>(
                index, static_cast<int>(draw.size()) - 1)(rng);
            std::swap(draw[index], draw[selected]);
        }
        completed = shown;
        completed.insert(completed.end(), draw.begin(), draw.begin() + needed);
        total += made_strength(best_hand(completed));
    }
    return total / samples;
}

Action heuristic_action(const State& state, int seat) {
    const uint8_t mask = valid_mask(state, seat);
    const double strength = heuristic_strength(state, seat);
    if (mask & (1u << CHECK)) {
        if (strength >= 0.90) return first_valid(mask, {FULL, HALF, QUARTER, BBING, CHECK});
        if (strength >= 0.76) return first_valid(mask, {HALF, QUARTER, BBING, CHECK});
        if (strength >= 0.58) return first_valid(mask, {QUARTER, BBING, CHECK});
        return CHECK;
    }
    if (strength >= 0.90) return first_valid(mask, {HALF, QUARTER, DDADANG, CALL, FOLD});

    const int call = std::max(0, state.highest_bet - state.players[seat].round_bet);
    const double pot_pressure = call / static_cast<double>(std::max(1, state.pot + call));
    const double chips = std::max(1, stack_cap(state, seat) - state.players[seat].invested);
    const double pressure = std::clamp(0.65 * pot_pressure + 0.35 * call / chips, 0.0, 1.0);
    if (strength >= 0.76 && pressure < 0.55) {
        return first_valid(mask, {QUARTER, DDADANG, CALL, FOLD});
    }
    return heuristic_call_or_fold(state, seat, strength);
}

bool bluff_defense_call(const State& state, int seat) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&](uint64_t value) {
        hash = (hash ^ value) * 1099511628211ull;
    };
    mix(seat);
    mix(state.street);
    mix(state.pot);
    mix(state.highest_bet);
    mix(state.players[seat].round_bet);
    for (const Card& card : all_cards(state.players[seat])) {
        mix(card.suit * 13 + card.rank);
    }
    for (const Card& card : state.players[1 - seat].shown) {
        mix(card.suit * 13 + card.rank);
    }
    for (const Event& event : state.history) {
        mix(event.street * 32 + event.actor * 8 + event.action);
    }
    return (hash >> 32) & 1u;
}

Action heuristic_variant_action(
    const State& state,
    int seat,
    HeuristicVariant variant) {
    const uint8_t mask = valid_mask(state, seat);
    const Action baseline = heuristic_action(state, seat);
    const bool facing_bet =
        state.highest_bet > state.players[seat].round_bet;

    if (variant == HeuristicVariant::BluffDefense) {
        return baseline == FOLD && (mask & (1u << CALL)) &&
                bluff_defense_call(state, seat)
            ? CALL
            : baseline;
    }
    if (variant == HeuristicVariant::SlowPlay) {
        const double strength = heuristic_strength(state, seat);
        if (state.street < 7 && strength >= 0.58) {
            return facing_bet
                ? first_valid(mask, {CALL, FOLD})
                : first_valid(mask, {CHECK, CALL, FOLD});
        }
        return baseline;
    }
    const bool bluffing_street =
        variant == HeuristicVariant::OverBluff ||
        (variant == HeuristicVariant::HiddenBluff && state.street == 7);
    if (!bluffing_street) return baseline;
    if (facing_bet) {
        return heuristic_call_or_fold(
            state, seat, heuristic_strength(state, seat));
    }
    if (opponent_public_completion_strength(state, seat) < 0.55) {
        return first_valid(mask, {HALF, QUARTER, BBING, CHECK, FOLD});
    }
    return baseline;
}

struct InfoKey {
    uint8_t abstraction = 0;
    uint8_t street = 7;
    uint16_t power_cluster = 0;
    uint8_t category = 0;
    uint8_t rank_bucket = 0;
    uint8_t own_public_category = 0;
    uint8_t own_public_suit = 0;
    uint8_t own_public_connected = 0;
    uint8_t opponent_public_category = 0;
    uint8_t opponent_public_suit = 0;
    uint8_t opponent_public_connected = 0;
    uint8_t pot_odds_bucket = 0;
    uint8_t stack_pot_bucket = 0;
    uint8_t own_bet_count = 0;
    uint8_t opponent_bet_count = 0;
    uint8_t checked = 0;
    uint8_t last_action_class = 0;
    uint8_t betting_goal = 0;
    uint8_t legal_mask = 0;
    uint8_t history_length = 0;
    uint64_t history_code = 0;

    bool operator==(const InfoKey& other) const {
        return abstraction == other.abstraction &&
            street == other.street &&
            power_cluster == other.power_cluster &&
            category == other.category &&
            rank_bucket == other.rank_bucket &&
            own_public_category == other.own_public_category &&
            own_public_suit == other.own_public_suit &&
            own_public_connected == other.own_public_connected &&
            opponent_public_category == other.opponent_public_category &&
            opponent_public_suit == other.opponent_public_suit &&
            opponent_public_connected == other.opponent_public_connected &&
            pot_odds_bucket == other.pot_odds_bucket &&
            stack_pot_bucket == other.stack_pot_bucket &&
            own_bet_count == other.own_bet_count &&
            opponent_bet_count == other.opponent_bet_count &&
            checked == other.checked &&
            last_action_class == other.last_action_class &&
            betting_goal == other.betting_goal &&
            legal_mask == other.legal_mask &&
            history_length == other.history_length &&
            history_code == other.history_code;
    }
};

struct InfoKeyHash {
    size_t operator()(const InfoKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * 1099511628211ull;
        };
        mix(key.abstraction);
        mix(key.street);
        mix(key.power_cluster);
        mix(key.category);
        mix(key.rank_bucket);
        mix(key.own_public_category);
        mix(key.own_public_suit);
        mix(key.own_public_connected);
        mix(key.opponent_public_category);
        mix(key.opponent_public_suit);
        mix(key.opponent_public_connected);
        mix(key.pot_odds_bucket);
        mix(key.stack_pot_bucket);
        mix(key.own_bet_count);
        mix(key.opponent_bet_count);
        mix(key.checked);
        mix(key.last_action_class);
        mix(key.betting_goal);
        mix(key.legal_mask);
        mix(key.history_length);
        mix(key.history_code);
        return hash;
    }
};

std::array<uint8_t, 3> public_features(const std::vector<Card>& cards) {
    if (cards.empty()) return {255, 0, 0};
    const int category = public_priority(cards)[0];
    std::array<int, 4> suits{};
    std::array<bool, 15> values{};
    for (const Card& card : cards) {
        ++suits[card.suit];
        values[card.rank] = true;
    }
    if (values[14]) values[1] = true;
    int connected = 0;
    for (int start = 1; start <= 10; ++start) {
        int count = 0;
        for (int rank = start; rank < start + 5; ++rank) count += values[rank];
        connected = std::max(connected, count);
    }
    return {
        static_cast<uint8_t>(category),
        static_cast<uint8_t>(*std::max_element(suits.begin(), suits.end())),
        static_cast<uint8_t>(connected),
    };
}

uint8_t ratio_bucket(double value, std::initializer_list<double> thresholds) {
    uint8_t bucket = 0;
    for (double threshold : thresholds) bucket += value > threshold;
    return bucket;
}

InfoKey make_key(const State& state, int viewer) {
    InfoKey key{};
    key.street = static_cast<uint8_t>(state.street);
    const Score score = best_hand(all_cards(state.players[viewer]));
    key.category = static_cast<uint8_t>(score[0]);
    key.rank_bucket = static_cast<uint8_t>(
        std::min(3, std::max(0, (score[1] - 2) / 4)));
    const auto own = public_features(state.players[viewer].shown);
    const auto opponent = public_features(state.players[1 - viewer].shown);
    key.own_public_category = own[0];
    key.own_public_suit = own[1];
    key.own_public_connected = own[2];
    key.opponent_public_category = opponent[0];
    key.opponent_public_suit = opponent[1];
    key.opponent_public_connected = opponent[2];
    const double call = std::max(0, state.highest_bet - state.players[viewer].round_bet);
    const double pot = std::max(1, state.pot);
    key.pot_odds_bucket = ratio_bucket(call / std::max(1.0, pot + call), {0.1, 0.2, 0.33, 0.5});
    const double chips = std::max(0, stack_cap(state, viewer) - state.players[viewer].invested);
    key.stack_pot_bucket = ratio_bucket(chips / pot, {0.5, 1.0, 2.0});
    key.own_bet_count = static_cast<uint8_t>(std::min(3, bets_this_street(state, viewer)));
    key.legal_mask = valid_mask(state, viewer);
    for (const Event& event : state.history) {
        if (event.street != state.street) continue;
        const uint8_t relative_actor = event.actor == viewer ? 0 : 1;
        const uint8_t token = static_cast<uint8_t>(relative_actor * 8 + event.action + 1);
        key.history_code = (key.history_code << 5) | token;
        ++key.history_length;
    }
    return key;
}

using PowerVector = std::array<double, kPowerDimensions>;

uint64_t card_mask(const std::vector<Card>& cards) {
    uint64_t mask = 0;
    for (const Card& card : cards) {
        mask |= 1ull << (card.suit * 13 + card.rank - 2);
    }
    return mask;
}

struct PowerObservationKey {
    uint64_t own = 0;
    uint64_t own_public = 0;
    uint64_t discarded = 0;
    uint64_t opponent_public = 0;
    uint8_t street = 0;

    bool operator==(const PowerObservationKey& other) const {
        return own == other.own &&
            own_public == other.own_public &&
            discarded == other.discarded &&
            opponent_public == other.opponent_public &&
            street == other.street;
    }
};

struct PowerObservationHash {
    size_t operator()(const PowerObservationKey& key) const {
        size_t hash = std::hash<uint64_t>{}(key.own);
        hash ^= std::hash<uint64_t>{}(key.own_public) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint64_t>{}(key.discarded) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint64_t>{}(key.opponent_public) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= key.street + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct PowerCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t resets = 0;
    size_t entries = 0;
    size_t limit = 0;
};

PowerObservationKey power_observation_key(const State& state, int viewer) {
    return {
        card_mask(all_cards(state.players[viewer])),
        card_mask(state.players[viewer].shown),
        state.players[viewer].has_discard
            ? 1ull << (state.players[viewer].discarded.suit * 13 +
                       state.players[viewer].discarded.rank - 2)
            : 0,
        card_mask(state.players[1 - viewer].shown),
        static_cast<uint8_t>(state.street),
    };
}

uint32_t best_after_one_card(
    const std::vector<Card>& current,
    const Card& added,
    uint32_t best) {
    for (size_t a = 0; a + 3 < current.size(); ++a)
        for (size_t b = a + 1; b + 2 < current.size(); ++b)
            for (size_t c = b + 1; c + 1 < current.size(); ++c)
                for (size_t d = c + 1; d < current.size(); ++d) {
                    best = std::max(best, score_five_encoded({
                        current[a], current[b], current[c], current[d], added}));
                }
    return best;
}

uint32_t best_after_two_cards(
    const std::vector<Card>& current,
    const Card& first,
    const Card& second,
    uint32_t best) {
    for (size_t a = 0; a + 2 < current.size(); ++a)
        for (size_t b = a + 1; b + 1 < current.size(); ++b)
            for (size_t c = b + 1; c < current.size(); ++c) {
                best = std::max(best, score_five_encoded({
                    current[a], current[b], current[c], first, second}));
            }
    return best;
}

PowerVector power_vector(const State& state, int viewer, int sample_limit) {
    const Player& player = state.players[viewer];
    std::vector<Card> current = all_cards(player);
    const int draws_needed = std::max(0, 7 - static_cast<int>(current.size()));
    if (draws_needed > 2) throw std::runtime_error("power vector requires 5th street or later");

    const PowerObservationKey observation = power_observation_key(state, viewer);
    const uint64_t excluded = observation.own | observation.discarded | observation.opponent_public;
    std::vector<Card> unseen;
    for (const Card& card : fresh_deck()) {
        const uint64_t bit = 1ull << (card.suit * 13 + card.rank - 2);
        if (!(excluded & bit)) unseen.push_back(card);
    }

    std::vector<std::array<int, 2>> completions;
    if (draws_needed == 1) {
        completions.reserve(unseen.size());
    } else if (draws_needed == 2) {
        completions.reserve(unseen.size() * (unseen.size() - 1) / 2);
    }
    if (draws_needed == 0) {
        completions.push_back({-1, -1});
    } else if (draws_needed == 1) {
        for (int i = 0; i < static_cast<int>(unseen.size()); ++i) completions.push_back({i, -1});
    } else {
        for (int i = 0; i + 1 < static_cast<int>(unseen.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(unseen.size()); ++j) {
                completions.push_back({i, j});
            }
        }
    }
    if (sample_limit > 0 && static_cast<int>(completions.size()) > sample_limit) {
        const uint64_t seed =
            observation.own ^ (observation.own_public << 3) ^ (observation.discarded << 1) ^
            (observation.opponent_public << 7) ^ (observation.street * 0x9e3779b97f4a7c15ull);
        std::mt19937_64 rng(seed);
        std::shuffle(completions.begin(), completions.end(), rng);
        completions.resize(sample_limit);
    }

    std::array<double, 9> categories{};
    double primary_rank = 0.0;
    const uint32_t current_best = best_hand_encoded(current);
    std::vector<uint32_t> one_card_best;
    if (draws_needed > 0) {
        one_card_best.reserve(unseen.size());
        for (const Card& card : unseen) {
            one_card_best.push_back(
                best_after_one_card(current, card, current_best));
        }
    }
    for (const auto& completion : completions) {
        uint32_t encoded = current_best;
        if (completion[0] >= 0) encoded = one_card_best[completion[0]];
        if (completion[1] >= 0) {
            encoded = std::max(encoded, one_card_best[completion[1]]);
            encoded = best_after_two_cards(
                current,
                unseen[completion[0]],
                unseen[completion[1]],
                encoded);
        }
        const Score score = decode_score(encoded);
        ++categories[score[0]];
        primary_rank += score[1] / 15.0;
    }

    PowerVector result{};
    const double total = static_cast<double>(completions.size());
    for (int category = 0; category < 9; ++category) {
        result[category] = std::sqrt(categories[category] / total);
    }
    result[9] = primary_rank / total;
    const auto opponent_features = public_features(state.players[1 - viewer].shown);
    int opponent_high = 0;
    for (const Card& card : state.players[1 - viewer].shown) {
        opponent_high = std::max(opponent_high, static_cast<int>(card.rank));
    }
    result[10] = opponent_features[0] == 255 ? 0.0 : opponent_features[0] / 8.0;
    result[11] = opponent_high / 14.0;
    result[12] = opponent_features[1] / 4.0;
    result[13] = opponent_features[2] / 5.0;
    const auto own_features = public_features(state.players[viewer].shown);
    int own_high = 0;
    for (const Card& card : state.players[viewer].shown) {
        own_high = std::max(own_high, static_cast<int>(card.rank));
    }
    result[14] = own_features[0] == 255 ? 0.0 : own_features[0] / 8.0;
    result[15] = own_high / 14.0;
    result[16] = own_features[1] / 4.0;
    result[17] = own_features[2] / 5.0;
    return result;
}

double squared_distance(const PowerVector& left, const PowerVector& right) {
    double distance = 0.0;
    for (int i = 0; i < kPowerDimensions; ++i) {
        const double difference = left[i] - right[i];
        distance += difference * difference;
    }
    return distance;
}

struct PowerSplitResult {
    uint16_t parent_leaf = 0;
    uint16_t child_leaf = 0;
    uint16_t depth = 0;
    uint64_t child0_touches = 0;
    uint64_t child1_touches = 0;
};

struct RouteNode {
    std::array<uint32_t, 2> child{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};
    std::array<PowerVector, 2> child_center{};
    uint16_t leaf_policy_id = 0;
    uint16_t depth = 0;
    uint64_t touches_since_split = 0;
    bool is_leaf = true;
};

class PowerAtlas {
public:
    explicit PowerAtlas(int sample_limit = 128) : sample_limit_(sample_limit) {}
    virtual ~PowerAtlas() = default;

    void fit(
        const std::array<std::vector<PowerVector>, 3>& samples,
        int requested_clusters,
        uint64_t seed) {
        fit(
            samples,
            std::array<int, 3>{
                requested_clusters,
                requested_clusters,
                requested_clusters},
            seed);
    }

    void fit(
        const std::array<std::vector<PowerVector>, 3>& samples,
        const std::array<int, 3>& requested_clusters,
        uint64_t seed) {
        gaussian_hard_ = false;
        tree_enabled_ = false;
        generation_id_ = 0;
        for (int street = 0; street < 3; ++street) {
            route_roots_[street].clear();
            route_nodes_[street].clear();
            leaf_routes_[street].clear();
            next_leaf_id_[street] = 0;
        }
        for (int street_index = 0; street_index < 3; ++street_index) {
            const auto& data = samples[street_index];
            if (data.empty()) throw std::runtime_error("no power samples for a street");
            const int cluster_count = std::min<int>(
                requested_clusters[street_index], data.size());
            auto& centers = centers_[street_index];
            centers.clear();
            centers.reserve(cluster_count);
            centers.push_back(data[seed % data.size()]);
            std::vector<double> nearest(data.size(), std::numeric_limits<double>::infinity());
            while (static_cast<int>(centers.size()) < cluster_count) {
                size_t farthest = 0;
                for (size_t i = 0; i < data.size(); ++i) {
                    nearest[i] = std::min(nearest[i], squared_distance(data[i], centers.back()));
                    if (nearest[i] > nearest[farthest]) farthest = i;
                }
                centers.push_back(data[farthest]);
            }

            std::vector<int> assignments(data.size());
            for (int iteration = 0; iteration < 20; ++iteration) {
                std::vector<PowerVector> sums(cluster_count);
                std::vector<int> counts(cluster_count);
                bool changed = false;
                for (size_t i = 0; i < data.size(); ++i) {
                    int best = 0;
                    double best_distance = squared_distance(data[i], centers[0]);
                    for (int cluster = 1; cluster < cluster_count; ++cluster) {
                        const double distance = squared_distance(data[i], centers[cluster]);
                        if (distance < best_distance) {
                            best = cluster;
                            best_distance = distance;
                        }
                    }
                    changed |= iteration == 0 || assignments[i] != best;
                    assignments[i] = best;
                    ++counts[best];
                    for (int dimension = 0; dimension < kPowerDimensions; ++dimension) {
                        sums[best][dimension] += data[i][dimension];
                    }
                }
                for (int cluster = 0; cluster < cluster_count; ++cluster) {
                    if (!counts[cluster]) continue;
                    for (int dimension = 0; dimension < kPowerDimensions; ++dimension) {
                        centers[cluster][dimension] =
                            sums[cluster][dimension] / counts[cluster];
                    }
                }
                if (!changed) break;
            }
            auto& masses = masses_[street_index];
            masses.assign(cluster_count, 1.0);
            for (const PowerVector& point : data) {
                int best = 0;
                double best_distance = squared_distance(point, centers[0]);
                for (int cluster = 1; cluster < cluster_count; ++cluster) {
                    const double distance =
                        squared_distance(point, centers[cluster]);
                    if (distance < best_distance) {
                        best = cluster;
                        best_distance = distance;
                    }
                }
                masses[best] += 1.0;
            }
            const double mass_total =
                std::accumulate(masses.begin(), masses.end(), 0.0);
            for (double& mass : masses) mass /= mass_total;
        }
        assignment_cache_.clear();
        rebuild_local_variances();
    }

    void fit_hard_em(
        const std::array<std::vector<PowerVector>, 3>& samples,
        int requested_clusters,
        uint64_t seed) {
        fit_hard_em(
            samples,
            std::array<int, 3>{
                requested_clusters,
                requested_clusters,
                requested_clusters},
            seed);
    }

    void fit_hard_em(
        const std::array<std::vector<PowerVector>, 3>& samples,
        const std::array<int, 3>& requested_clusters,
        uint64_t seed) {
        fit(samples, requested_clusters, seed);
        gaussian_hard_ = true;
        constexpr double kVarianceFloor = 1e-4;
        for (int street = 0; street < 3; ++street) {
            const auto& data = samples[street];
            auto& centers = centers_[street];
            auto& variances = diagonal_variances_[street];
            auto& masses = masses_[street];
            const int clusters = static_cast<int>(centers.size());
            variances.assign(clusters, PowerVector{});
            std::vector<int> assignments(data.size(), -1);

            for (int iteration = 0; iteration < 20; ++iteration) {
                std::vector<PowerVector> sums(clusters);
                std::vector<PowerVector> squared_sums(clusters);
                std::vector<size_t> counts(clusters);
                bool changed = false;
                for (size_t i = 0; i < data.size(); ++i) {
                    int best = 0;
                    double best_score = iteration == 0
                        ? -squared_distance(data[i], centers[0])
                        : gaussian_score(data[i], street, 0);
                    for (int cluster = 1; cluster < clusters; ++cluster) {
                        const double score = iteration == 0
                            ? -squared_distance(data[i], centers[cluster])
                            : gaussian_score(data[i], street, cluster);
                        if (score > best_score) {
                            best = cluster;
                            best_score = score;
                        }
                    }
                    changed |= assignments[i] != best;
                    assignments[i] = best;
                    ++counts[best];
                    for (int dimension = 0; dimension < kPowerDimensions; ++dimension) {
                        sums[best][dimension] += data[i][dimension];
                        squared_sums[best][dimension] +=
                            data[i][dimension] * data[i][dimension];
                    }
                }
                for (int cluster = 0; cluster < clusters; ++cluster) {
                    if (counts[cluster]) {
                        for (int dimension = 0;
                             dimension < kPowerDimensions;
                             ++dimension) {
                            const double mean =
                                sums[cluster][dimension] / counts[cluster];
                            centers[cluster][dimension] = mean;
                            variances[cluster][dimension] = std::max(
                                kVarianceFloor,
                                squared_sums[cluster][dimension] / counts[cluster] -
                                    mean * mean);
                        }
                    } else if (iteration == 0) {
                        variances[cluster].fill(1.0);
                    }
                    masses[cluster] =
                        (counts[cluster] + 1.0) /
                        (data.size() + static_cast<double>(clusters));
                }
                rebuild_gaussian_cache();
                if (!changed) break;
            }
        }
        assignment_cache_.clear();
        rebuild_local_variances();
    }

    virtual uint16_t assign(const State& state, int viewer) const {
        const PowerObservationKey observation = power_observation_key(state, viewer);
        auto found = assignment_cache_.find(observation);
        if (found != assignment_cache_.end()) {
            ++assignment_cache_hits_;
            return found->second;
        }
        ++assignment_cache_misses_;
        const PowerVector vector = power_vector(state, viewer, sample_limit_);
        const uint16_t assigned = assign_vector(state.street, vector);
        if (assignment_cache_limit_ > 0) {
            if (assignment_cache_.size() >= assignment_cache_limit_) {
                assignment_cache_.clear();
                ++assignment_cache_resets_;
            }
            assignment_cache_.emplace(observation, assigned);
        }
        return assigned;
    }

    void set_assignment_cache_limit(size_t limit) {
        assignment_cache_limit_ = limit;
        if (limit == 0 || assignment_cache_.size() > limit) {
            assignment_cache_.clear();
        }
    }

    PowerCacheStats assignment_cache_stats() const {
        return PowerCacheStats{
            assignment_cache_hits_,
            assignment_cache_misses_,
            assignment_cache_resets_,
            assignment_cache_.size(),
            assignment_cache_limit_};
    }

    virtual uint16_t assign_vector(int street, const PowerVector& vector) const {
        const int index = street - 5;
        if (index < 0 || index >= 3 || centers_[index].empty()) {
            throw std::runtime_error("power atlas is not fitted for this street");
        }
        int best = 0;
        double best_distance = gaussian_hard_
            ? -gaussian_score(vector, index, 0)
            : squared_distance(vector, centers_[index][0]);
        for (int cluster = 1; cluster < static_cast<int>(centers_[index].size()); ++cluster) {
            const double distance = gaussian_hard_
                ? -gaussian_score(vector, index, cluster)
                : squared_distance(vector, centers_[index][cluster]);
            if (distance < best_distance) {
                best = cluster;
                best_distance = distance;
            }
        }
        const uint16_t base = static_cast<uint16_t>(best);
        if (!tree_enabled_) return base;
        uint32_t route = route_roots_[index].at(base);
        for (;;) {
            const RouteNode& node = route_nodes_[index].at(route);
            if (node.is_leaf) return node.leaf_policy_id;
            const double left = squared_distance(vector, node.child_center[0]);
            const double right = squared_distance(vector, node.child_center[1]);
            route = node.child[right < left ? 1 : 0];
        }
    }

    void enable_tree() {
        if (tree_enabled_) return;
        initialize_route_roots();
        tree_enabled_ = true;
        rebuild_tree_generation();
        assignment_cache_.clear();
    }

    bool tree_enabled() const { return tree_enabled_; }
    uint64_t generation_id() const { return generation_id_; }

    size_t base_clusters(int street) const {
        return centers_.at(street - 5).size();
    }

    uint16_t leaf_depth(int street, uint16_t leaf) const {
        if (!tree_enabled_) {
            throw std::runtime_error("leaf depth requires a tree atlas");
        }
        const int index = street - 5;
        if (index < 0 || index >= 3) {
            throw std::runtime_error("invalid leaf-depth street");
        }
        for (const RouteNode& node : route_nodes_[index]) {
            if (node.is_leaf && node.leaf_policy_id == leaf) {
                return node.depth;
            }
        }
        throw std::runtime_error("power leaf not found");
    }

    uint64_t record_leaf_touch(int street, uint16_t leaf) {
        if (!tree_enabled_) {
            throw std::runtime_error("leaf touch requires a tree atlas");
        }
        const int index = street - 5;
        if (index < 0 || index >= 3 ||
            leaf >= leaf_routes_[index].size()) {
            throw std::runtime_error("invalid power leaf touch");
        }
        const uint32_t route = leaf_routes_[index][leaf];
        if (route >= route_nodes_[index].size() ||
            !route_nodes_[index][route].is_leaf) {
            throw std::runtime_error("power leaf route is invalid");
        }
        return ++route_nodes_[index][route].touches_since_split;
    }

    uint64_t leaf_touches_since_split(int street, uint16_t leaf) const {
        const int index = street - 5;
        if (!tree_enabled_ || index < 0 || index >= 3 ||
            leaf >= leaf_routes_[index].size()) {
            throw std::runtime_error("invalid power leaf touch query");
        }
        return route_nodes_[index][leaf_routes_[index][leaf]]
            .touches_since_split;
    }

    const PowerVector& base_center(int street, size_t cluster) const {
        return centers_.at(street - 5).at(cluster);
    }

    PowerSplitResult split_leaf(
        int street,
        uint16_t parent_leaf,
        const PowerVector& child0,
        const PowerVector& child1,
        double child1_mass) {
        if (!tree_enabled_) {
            throw std::runtime_error("local split requires a tree atlas");
        }
        const int index = street - 5;
        if (index < 0 || index >= 3) {
            throw std::runtime_error("invalid split street");
        }
        if (next_leaf_id_[index] > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("power atlas leaf limit reached");
        }
        uint32_t parent_route = std::numeric_limits<uint32_t>::max();
        for (uint32_t route = 0; route < route_nodes_[index].size(); ++route) {
            const RouteNode& node = route_nodes_[index][route];
            if (node.is_leaf && node.leaf_policy_id == parent_leaf) {
                parent_route = route;
                break;
            }
        }
        if (parent_route == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("parent power leaf not found");
        }
        const uint16_t depth = route_nodes_[index][parent_route].depth;
        const uint64_t parent_touches =
            route_nodes_[index][parent_route].touches_since_split;
        if (depth == std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("power route depth limit reached");
        }
        child1_mass = std::clamp(child1_mass, 0.0, 1.0);
        const uint64_t child1_touches = static_cast<uint64_t>(std::llround(
            static_cast<long double>(parent_touches) * child1_mass));
        const uint16_t child_leaf =
            static_cast<uint16_t>(next_leaf_id_[index]++);
        RouteNode left;
        left.leaf_policy_id = parent_leaf;
        left.depth = static_cast<uint16_t>(depth + 1);
        left.touches_since_split = parent_touches - child1_touches;
        RouteNode right;
        right.leaf_policy_id = child_leaf;
        right.depth = left.depth;
        right.touches_since_split = child1_touches;
        const uint32_t left_route = static_cast<uint32_t>(route_nodes_[index].size());
        route_nodes_[index].push_back(left);
        const uint32_t right_route = static_cast<uint32_t>(route_nodes_[index].size());
        route_nodes_[index].push_back(right);
        RouteNode& parent = route_nodes_[index][parent_route];
        parent.is_leaf = false;
        parent.touches_since_split = 0;
        parent.child = {left_route, right_route};
        parent.child_center = {child0, child1};
        rebuild_leaf_routes(index);
        assignment_cache_.clear();
        rebuild_tree_generation();
        return PowerSplitResult{
            parent_leaf,
            child_leaf,
            left.depth,
            left.touches_since_split,
            right.touches_since_split};
    }

    std::vector<std::pair<uint16_t, double>> weights(
        const State& state,
        int viewer,
        int top_k,
        double top_p,
        double temperature,
        bool local_bandwidth,
        const std::vector<double>* cluster_scales = nullptr,
        int override_cluster = -1,
        double override_scale = 1.0) const {
        if (tree_enabled_) {
            throw std::runtime_error("soft weights are disabled for tree atlases");
        }
        const int index = state.street - 5;
        if (index < 0 || index >= 3 || centers_[index].empty()) {
            throw std::runtime_error("power atlas is not fitted for this street");
        }
        const PowerVector vector = power_vector(state, viewer, sample_limit_);
        std::vector<std::pair<uint16_t, double>> scores;
        scores.reserve(centers_[index].size());
        for (int cluster = 0;
             cluster < static_cast<int>(centers_[index].size());
             ++cluster) {
            const double distance = squared_distance(vector, centers_[index][cluster]);
            double score =
                std::log(std::max(1e-12, masses_[index][cluster])) -
                distance / (2.0 * temperature);
            if (local_bandwidth) {
                double scale =
                    cluster_scales &&
                    cluster < static_cast<int>(cluster_scales->size())
                    ? (*cluster_scales)[cluster]
                    : 1.0;
                if (cluster == override_cluster) scale *= override_scale;
                const double variance = std::max(
                    1e-9,
                    temperature * scale * local_variances_[index][cluster]);
                score = std::log(
                            std::max(1e-12, masses_[index][cluster]))
                    - distance / (2.0 * variance)
                    - 0.5 * kPowerDimensions * std::log(variance);
            }
            scores.emplace_back(static_cast<uint16_t>(cluster), score);
        }
        std::sort(
            scores.begin(),
            scores.end(),
            [](const auto& left, const auto& right) {
                return left.second > right.second;
            });
        const double maximum = scores.front().second;
        double total = 0.0;
        for (auto& [cluster, score] : scores) {
            score = std::exp(score - maximum);
            total += score;
        }
        for (auto& [cluster, weight] : scores) weight /= total;
        if (top_p > 0.0) {
            double cumulative = 0.0;
            size_t retained = 0;
            do {
                cumulative += scores[retained++].second;
            } while (retained < scores.size() && cumulative < top_p);
            scores.resize(retained);
        } else {
            scores.resize(std::min<int>(top_k, scores.size()));
        }
        total = 0.0;
        for (const auto& [cluster, weight] : scores) total += weight;
        for (auto& [cluster, weight] : scores) weight /= total;
        return scores;
    }

    uint16_t append(const State& state, int viewer) {
        const uint16_t parent = assign(state, viewer);
        return append(
            power_vector(state, viewer, sample_limit_),
            state.street,
            parent,
            0.1);
    }

    uint16_t append(
        const PowerVector& center,
        int street,
        uint16_t parent,
        double child_fraction) {
        if (tree_enabled_) {
            throw std::runtime_error(
                "global append is disabled for local route-tree atlases");
        }
        const int index = street - 5;
        if (index < 0 || index >= 3 || centers_[index].empty()) {
            throw std::runtime_error("power atlas is not fitted for this street");
        }
        if (parent >= centers_[index].size() ||
            child_fraction <= 0.0 ||
            child_fraction >= 1.0) {
            throw std::runtime_error("invalid cluster split");
        }
        if (centers_[index].size() >=
            static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
            throw std::runtime_error("power atlas cluster limit reached");
        }
        const double child_mass = masses_[index][parent] * child_fraction;
        masses_[index][parent] -= child_mass;
        masses_[index].push_back(child_mass);
        centers_[index].push_back(center);
        if (gaussian_hard_) {
            diagonal_variances_[index].push_back(
                diagonal_variances_[index][parent]);
            rebuild_gaussian_cache();
        }
        assignment_cache_.clear();
        rebuild_local_variances();
        return static_cast<uint16_t>(centers_[index].size() - 1);
    }

    PowerVector vector(const State& state, int viewer) const {
        return power_vector(state, viewer, sample_limit_);
    }

    void save(const std::string& path) const {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write atlas: " + path);
        output.write(
            tree_enabled_ ? "POWERAT5" : gaussian_hard_ ? "POWERAT3" : "POWERAT2",
            8);
        output.write(reinterpret_cast<const char*>(&sample_limit_), sizeof(sample_limit_));
        if (tree_enabled_) {
            const uint8_t gaussian = gaussian_hard_ ? 1 : 0;
            output.write(reinterpret_cast<const char*>(&gaussian), sizeof(gaussian));
            output.write(
                reinterpret_cast<const char*>(&generation_id_),
                sizeof(generation_id_));
        }
        for (int index = 0; index < 3; ++index) {
            const auto& street = centers_[index];
            const uint32_t count = static_cast<uint32_t>(street.size());
            output.write(reinterpret_cast<const char*>(&count), sizeof(count));
            for (const PowerVector& center : street) {
                output.write(reinterpret_cast<const char*>(center.data()), sizeof(double) * center.size());
            }
            output.write(
                reinterpret_cast<const char*>(masses_[index].data()),
                sizeof(double) * masses_[index].size());
            if (gaussian_hard_) {
                for (const PowerVector& variance : diagonal_variances_[index]) {
                    output.write(
                        reinterpret_cast<const char*>(variance.data()),
                        sizeof(double) * variance.size());
                }
            }
            if (tree_enabled_) {
                const uint32_t root_count =
                    static_cast<uint32_t>(route_roots_[index].size());
                const uint32_t route_count =
                    static_cast<uint32_t>(route_nodes_[index].size());
                output.write(
                    reinterpret_cast<const char*>(&root_count), sizeof(root_count));
                for (uint32_t root : route_roots_[index]) {
                    output.write(reinterpret_cast<const char*>(&root), sizeof(root));
                }
                output.write(
                    reinterpret_cast<const char*>(&route_count), sizeof(route_count));
                for (const RouteNode& node : route_nodes_[index]) {
                    const uint8_t leaf = node.is_leaf ? 1 : 0;
                    output.write(reinterpret_cast<const char*>(&leaf), sizeof(leaf));
                    output.write(
                        reinterpret_cast<const char*>(&node.leaf_policy_id),
                        sizeof(node.leaf_policy_id));
                    output.write(
                        reinterpret_cast<const char*>(&node.depth), sizeof(node.depth));
                    output.write(
                        reinterpret_cast<const char*>(&node.touches_since_split),
                        sizeof(node.touches_since_split));
                    for (uint32_t child : node.child) {
                        output.write(
                            reinterpret_cast<const char*>(&child), sizeof(child));
                    }
                    for (const PowerVector& center : node.child_center) {
                        output.write(
                            reinterpret_cast<const char*>(center.data()),
                            sizeof(double) * center.size());
                    }
                }
                output.write(
                    reinterpret_cast<const char*>(&next_leaf_id_[index]),
                    sizeof(next_leaf_id_[index]));
            }
        }
        if (!output) throw std::runtime_error("failed to write atlas: " + path);
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read atlas: " + path);
        char magic[8]{};
        input.read(magic, sizeof(magic));
        const bool has_route_touches =
            std::memcmp(magic, "POWERAT5", 8) == 0;
        const bool has_route_tree = has_route_touches ||
            std::memcmp(magic, "POWERAT4", 8) == 0;
        const bool has_diagonal_variances =
            std::memcmp(magic, "POWERAT3", 8) == 0;
        const bool has_masses = has_diagonal_variances ||
            has_route_tree || std::memcmp(magic, "POWERAT2", 8) == 0;
        if (!has_masses && std::memcmp(magic, "POWERAT1", 8) != 0) {
            throw std::runtime_error("invalid atlas");
        }
        input.read(reinterpret_cast<char*>(&sample_limit_), sizeof(sample_limit_));
        uint64_t stored_generation = 0;
        bool route_gaussian = false;
        if (has_route_tree) {
            uint8_t gaussian = 0;
            input.read(reinterpret_cast<char*>(&gaussian), sizeof(gaussian));
            input.read(
                reinterpret_cast<char*>(&stored_generation),
                sizeof(stored_generation));
            route_gaussian = gaussian != 0;
        }
        for (int index = 0; index < 3; ++index) {
            auto& street = centers_[index];
            uint32_t count = 0;
            input.read(reinterpret_cast<char*>(&count), sizeof(count));
            street.resize(count);
            for (PowerVector& center : street) {
                input.read(reinterpret_cast<char*>(center.data()), sizeof(double) * center.size());
            }
            auto& masses = masses_[index];
            masses.assign(count, count ? 1.0 / count : 0.0);
            if (has_masses) {
                input.read(
                    reinterpret_cast<char*>(masses.data()),
                    sizeof(double) * masses.size());
            }
            auto& variances = diagonal_variances_[index];
            variances.clear();
            if (has_diagonal_variances || (has_route_tree && route_gaussian)) {
                variances.resize(count);
                for (PowerVector& variance : variances) {
                    input.read(
                        reinterpret_cast<char*>(variance.data()),
                        sizeof(double) * variance.size());
                }
            }
            route_roots_[index].clear();
            route_nodes_[index].clear();
            leaf_routes_[index].clear();
            next_leaf_id_[index] = 0;
            if (has_route_tree) {
                uint32_t root_count = 0;
                uint32_t route_count = 0;
                input.read(
                    reinterpret_cast<char*>(&root_count), sizeof(root_count));
                route_roots_[index].resize(root_count);
                for (uint32_t& root : route_roots_[index]) {
                    input.read(reinterpret_cast<char*>(&root), sizeof(root));
                }
                input.read(
                    reinterpret_cast<char*>(&route_count), sizeof(route_count));
                route_nodes_[index].resize(route_count);
                for (RouteNode& node : route_nodes_[index]) {
                    uint8_t leaf = 0;
                    input.read(reinterpret_cast<char*>(&leaf), sizeof(leaf));
                    node.is_leaf = leaf != 0;
                    input.read(
                        reinterpret_cast<char*>(&node.leaf_policy_id),
                        sizeof(node.leaf_policy_id));
                    input.read(
                        reinterpret_cast<char*>(&node.depth), sizeof(node.depth));
                    if (has_route_touches) {
                        input.read(
                            reinterpret_cast<char*>(&node.touches_since_split),
                            sizeof(node.touches_since_split));
                    }
                    for (uint32_t& child : node.child) {
                        input.read(reinterpret_cast<char*>(&child), sizeof(child));
                    }
                    for (PowerVector& center : node.child_center) {
                        input.read(
                            reinterpret_cast<char*>(center.data()),
                            sizeof(double) * center.size());
                    }
                }
                input.read(
                    reinterpret_cast<char*>(&next_leaf_id_[index]),
                    sizeof(next_leaf_id_[index]));
            }
        }
        if (!input) throw std::runtime_error("truncated atlas");
        assignment_cache_.clear();
        gaussian_hard_ = has_diagonal_variances || route_gaussian;
        tree_enabled_ = has_route_tree;
        rebuild_gaussian_cache();
        rebuild_local_variances();
        if (tree_enabled_) {
            validate_route_tree();
            rebuild_leaf_routes();
            rebuild_tree_generation();
            if (stored_generation != generation_id_) {
                throw std::runtime_error("tree atlas generation checksum mismatch");
            }
        } else {
            generation_id_ = 0;
            for (auto& routes : leaf_routes_) routes.clear();
        }
    }

    virtual size_t clusters(int street) const {
        return tree_enabled_
            ? next_leaf_id_.at(street - 5)
            : centers_.at(street - 5).size();
    }

private:
    void initialize_route_roots() {
        for (int street = 0; street < 3; ++street) {
            if (centers_[street].size() >
                static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
                throw std::runtime_error("too many base clusters for route tree");
            }
            route_roots_[street].clear();
            route_nodes_[street].clear();
            route_roots_[street].reserve(centers_[street].size());
            route_nodes_[street].reserve(centers_[street].size());
            for (size_t cluster = 0; cluster < centers_[street].size(); ++cluster) {
                route_roots_[street].push_back(
                    static_cast<uint32_t>(route_nodes_[street].size()));
                RouteNode node;
                node.leaf_policy_id = static_cast<uint16_t>(cluster);
                route_nodes_[street].push_back(node);
            }
            next_leaf_id_[street] =
                static_cast<uint32_t>(centers_[street].size());
        }
        rebuild_leaf_routes();
    }

    void rebuild_leaf_routes(int only_street = -1) {
        const uint32_t missing = std::numeric_limits<uint32_t>::max();
        for (int street = 0; street < 3; ++street) {
            if (only_street >= 0 && street != only_street) continue;
            auto& routes = leaf_routes_[street];
            routes.assign(next_leaf_id_[street], missing);
            for (uint32_t route = 0;
                 route < route_nodes_[street].size();
                 ++route) {
                const RouteNode& node = route_nodes_[street][route];
                if (node.is_leaf) routes.at(node.leaf_policy_id) = route;
            }
            if (std::find(routes.begin(), routes.end(), missing) !=
                routes.end()) {
                throw std::runtime_error("power leaf route index is incomplete");
            }
        }
    }

    void validate_route_tree() const {
        for (int street = 0; street < 3; ++street) {
            if (route_roots_[street].size() != centers_[street].size()) {
                throw std::runtime_error("tree atlas base-root count mismatch");
            }
            std::vector<uint8_t> leaf_ids(next_leaf_id_[street], 0);
            for (uint32_t root : route_roots_[street]) {
                if (root >= route_nodes_[street].size()) {
                    throw std::runtime_error("tree atlas root out of range");
                }
            }
            for (const RouteNode& node : route_nodes_[street]) {
                if (node.is_leaf) {
                    if (node.leaf_policy_id >= next_leaf_id_[street] ||
                        leaf_ids[node.leaf_policy_id]++) {
                        throw std::runtime_error("invalid or duplicate tree leaf ID");
                    }
                } else if (
                    node.child[0] >= route_nodes_[street].size() ||
                    node.child[1] >= route_nodes_[street].size()) {
                    throw std::runtime_error("tree atlas child out of range");
                }
            }
            if (std::count(leaf_ids.begin(), leaf_ids.end(), uint8_t{1}) !=
                static_cast<int>(next_leaf_id_[street])) {
                throw std::runtime_error("tree atlas leaf IDs are not contiguous");
            }
        }
    }

    void rebuild_tree_generation() {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * 1099511628211ull;
        };
        mix(static_cast<uint64_t>(sample_limit_));
        mix(gaussian_hard_ ? 1 : 0);
        for (int street = 0; street < 3; ++street) {
            mix(centers_[street].size());
            for (const PowerVector& center : centers_[street]) {
                for (double value : center) {
                    uint64_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    mix(bits);
                }
            }
            for (double value : masses_[street]) {
                uint64_t bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                mix(bits);
            }
            for (const PowerVector& variance : diagonal_variances_[street]) {
                for (double value : variance) {
                    uint64_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    mix(bits);
                }
            }
            mix(route_roots_[street].size());
            for (uint32_t root : route_roots_[street]) mix(root);
            mix(route_nodes_[street].size());
            for (const RouteNode& node : route_nodes_[street]) {
                mix(node.is_leaf ? 1 : 0);
                mix(node.leaf_policy_id);
                mix(node.depth);
                for (uint32_t child : node.child) mix(child);
                for (const PowerVector& center : node.child_center) {
                    for (double value : center) {
                        uint64_t bits = 0;
                        std::memcpy(&bits, &value, sizeof(bits));
                        mix(bits);
                    }
                }
            }
            mix(next_leaf_id_[street]);
        }
        generation_id_ = hash;
    }

    double gaussian_score(
        const PowerVector& point,
        int street,
        int cluster) const {
        double score = gaussian_offsets_[street][cluster];
        for (int dimension = 0; dimension < kPowerDimensions; ++dimension) {
            const double difference =
                point[dimension] - centers_[street][cluster][dimension];
            score -= 0.5 * difference * difference *
                diagonal_precisions_[street][cluster][dimension];
        }
        return score;
    }

    void rebuild_gaussian_cache() {
        if (!gaussian_hard_) return;
        for (int street = 0; street < 3; ++street) {
            const size_t count = diagonal_variances_[street].size();
            diagonal_precisions_[street].resize(count);
            gaussian_offsets_[street].resize(count);
            for (size_t cluster = 0; cluster < count; ++cluster) {
                double offset = std::log(std::max(1e-12, masses_[street][cluster]));
                for (int dimension = 0; dimension < kPowerDimensions; ++dimension) {
                    const double variance = std::max(
                        1e-9,
                        diagonal_variances_[street][cluster][dimension]);
                    diagonal_precisions_[street][cluster][dimension] = 1.0 / variance;
                    offset -= 0.5 * std::log(variance);
                }
                gaussian_offsets_[street][cluster] = offset;
            }
        }
    }

    void rebuild_local_variances() {
        for (int street = 0; street < 3; ++street) {
            const auto& centers = centers_[street];
            auto& variances = local_variances_[street];
            variances.assign(centers.size(), 1.0);
            if (centers.size() < 2) continue;
            for (size_t cluster = 0; cluster < centers.size(); ++cluster) {
                double nearest = std::numeric_limits<double>::infinity();
                for (size_t other = 0; other < centers.size(); ++other) {
                    if (other == cluster) continue;
                    nearest = std::min(
                        nearest,
                        squared_distance(centers[cluster], centers[other]));
                }
                variances[cluster] = std::max(1e-9, nearest / 4.0);
            }
        }
    }

    int sample_limit_;
    std::array<std::vector<PowerVector>, 3> centers_;
    std::array<std::vector<PowerVector>, 3> diagonal_variances_;
    std::array<std::vector<PowerVector>, 3> diagonal_precisions_;
    std::array<std::vector<double>, 3> gaussian_offsets_;
    std::array<std::vector<double>, 3> local_variances_;
    std::array<std::vector<double>, 3> masses_;
    std::array<std::vector<uint32_t>, 3> route_roots_;
    std::array<std::vector<RouteNode>, 3> route_nodes_;
    std::array<std::vector<uint32_t>, 3> leaf_routes_;
    std::array<uint32_t, 3> next_leaf_id_{};
    bool gaussian_hard_ = false;
    bool tree_enabled_ = false;
    uint64_t generation_id_ = 0;
    mutable std::unordered_map<
        PowerObservationKey, uint16_t, PowerObservationHash> assignment_cache_;
    size_t assignment_cache_limit_ = 1000000;
    mutable uint64_t assignment_cache_hits_ = 0;
    mutable uint64_t assignment_cache_misses_ = 0;
    mutable uint64_t assignment_cache_resets_ = 0;
};

class ActionRangeModel {
public:
    using Distribution = std::array<double, 9>;

    void observe(const State& state, int actor, Action action, int sample_limit) {
        const PowerVector power = power_vector(state, actor, sample_limit);
        const int street = state.street - 5;
        for (int category = 0; category < 9; ++category) {
            const double probability = power[category] * power[category];
            prior_counts_[category] += probability;
            action_counts_[street][action][category] += probability;
        }
    }

    void finalize() {
        double prior_total = std::accumulate(
            prior_counts_.begin(), prior_counts_.end(), 0.0);
        for (int category = 0; category < 9; ++category) {
            prior_[category] =
                (prior_counts_[category] + 1.0) / (prior_total + 9.0);
        }
        for (int street = 0; street < 3; ++street) {
            for (int category = 0; category < 9; ++category) {
                double total = 0.0;
                for (int action = 0; action < kActionCount; ++action) {
                    total += action_counts_[street][action][category];
                }
                for (int action = 0; action < kActionCount; ++action) {
                    likelihood_[street][action][category] =
                        (action_counts_[street][action][category] + 1.0) /
                        (total + kActionCount);
                }
            }
        }
        ready_ = true;
    }

    uint8_t bucket(const State& state, int viewer) const {
        if (!ready_) throw std::runtime_error("range model is not fitted");
        Distribution posterior = prior_;
        for (const Event& event : state.history) {
            if (event.actor == viewer || event.street < 5 || event.street > 7) {
                continue;
            }
            double total = 0.0;
            for (int category = 0; category < 9; ++category) {
                posterior[category] *=
                    likelihood_[event.street - 5][event.action][category];
                total += posterior[category];
            }
            if (total <= 0.0) posterior = prior_;
            else for (double& value : posterior) value /= total;
        }

        double expected = 0.0;
        double entropy = 0.0;
        double strong = 0.0;
        for (int category = 0; category < 9; ++category) {
            expected += category * posterior[category] / 8.0;
            if (posterior[category] > 0.0) {
                entropy -= posterior[category] * std::log(posterior[category]);
            }
            if (category >= 3) strong += posterior[category];
        }
        const int expected_bin = std::min(7, static_cast<int>(expected * 8.0));
        const int entropy_bin = std::min(
            3, static_cast<int>(entropy / std::log(9.0) * 4.0));
        return static_cast<uint8_t>(
            expected_bin + 8 * entropy_bin + 32 * (strong >= 0.20));
    }

    void save(const std::string& path) const {
        if (!ready_) throw std::runtime_error("range model is not fitted");
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write range model: " + path);
        output.write("RANGEV1", 8);
        output.write(
            reinterpret_cast<const char*>(prior_.data()),
            sizeof(double) * prior_.size());
        output.write(
            reinterpret_cast<const char*>(likelihood_.data()),
            sizeof(likelihood_));
        if (!output) throw std::runtime_error("failed to write range model: " + path);
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read range model: " + path);
        char magic[8]{};
        input.read(magic, sizeof(magic));
        input.read(
            reinterpret_cast<char*>(prior_.data()),
            sizeof(double) * prior_.size());
        input.read(reinterpret_cast<char*>(likelihood_.data()), sizeof(likelihood_));
        if (std::memcmp(magic, "RANGEV1", 7) != 0 || !input) {
            throw std::runtime_error("invalid range model");
        }
        ready_ = true;
    }

private:
    Distribution prior_counts_{};
    std::array<std::array<Distribution, kActionCount>, 3> action_counts_{};
    Distribution prior_{};
    std::array<std::array<Distribution, kActionCount>, 3> likelihood_{};
    bool ready_ = false;
};

uint8_t action_class(Action action) {
    if (action == CHECK) return 1;
    if (action == CALL) return 2;
    if (action == BBING || action == QUARTER) return 3;
    if (action == DDADANG || action == HALF) return 4;
    if (action == FOLD) return 5;
    return 0;
}

uint8_t street_aggression(const State& state, int viewer, int street) {
    uint8_t result = 0;
    for (const Event& event : state.history) {
        if (event.street != street || !aggressive(event.action)) {
            continue;
        }
        result |= event.actor == viewer ? 1 : 2;
    }
    return result;
}

uint8_t previous_street_aggression(const State& state, int viewer) {
    return state.street <= 5
        ? 0
        : street_aggression(state, viewer, state.street - 1);
}

uint8_t cumulative_street_aggression(const State& state, int viewer) {
    uint8_t code = 0;
    uint8_t scale = 1;
    for (int street = 5; street < state.street; ++street) {
        code += scale * street_aggression(state, viewer, street);
        scale *= 4;
    }
    return code;
}

uint8_t street_aggression_strength(
    const State& state,
    int viewer,
    int street) {
    uint8_t own = 0;
    uint8_t opponent = 0;
    for (const Event& event : state.history) {
        if (event.street != street || !aggressive(event.action)) continue;
        const uint8_t strength =
            event.action == DDADANG || event.action == HALF ? 2 : 1;
        uint8_t& target = event.actor == viewer ? own : opponent;
        target = std::max(target, strength);
    }
    return static_cast<uint8_t>(own + 3 * opponent);
}

uint8_t cumulative_street_aggression_strength(
    const State& state,
    int viewer) {
    uint8_t code = 0;
    uint8_t scale = 1;
    for (int street = 5; street < state.street; ++street) {
        code += scale * street_aggression_strength(state, viewer, street);
        scale *= 9;
    }
    return code;
}

InfoKey make_power_key(
    const State& state,
    int viewer,
    const PowerAtlas& atlas,
    bool preserve_recall = false,
    const ActionRangeModel* range_model = nullptr,
    uint8_t betting_memory = 0) {
    InfoKey key{};
    key.abstraction = atlas.tree_enabled() ? 8
        : range_model ? 4
        : preserve_recall ? 3
        : betting_memory == 3 ? 7
        : betting_memory == 2 ? 6
        : betting_memory == 1 ? 5
        : 1;
    key.street = static_cast<uint8_t>(state.street);
    key.power_cluster = atlas.assign(state, viewer);
    if (range_model) key.category = range_model->bucket(state, viewer);
    if (betting_memory) {
        key.rank_bucket = betting_memory == 3
            ? cumulative_street_aggression_strength(state, viewer)
            : betting_memory == 2
                ? cumulative_street_aggression(state, viewer)
                : previous_street_aggression(state, viewer);
    }
    const double call = std::max(0, state.highest_bet - state.players[viewer].round_bet);
    const double pot = std::max(1, state.pot);
    key.pot_odds_bucket = ratio_bucket(call / std::max(1.0, pot + call), {0.1, 0.2, 0.33, 0.5});
    const double chips = std::max(0, stack_cap(state, viewer) - state.players[viewer].invested);
    key.stack_pot_bucket = ratio_bucket(chips / pot, {0.5, 1.0, 2.0});
    key.own_bet_count = static_cast<uint8_t>(std::min(3, bets_this_street(state, viewer)));
    key.opponent_bet_count = static_cast<uint8_t>(
        std::min(3, bets_this_street(state, 1 - viewer)));
    key.checked = checked_this_street(state, viewer);
    for (auto it = state.history.rbegin(); it != state.history.rend(); ++it) {
        if (it->street != state.street) break;
        if (!key.last_action_class) key.last_action_class = action_class(it->action);
        if (it->actor == viewer && aggressive(it->action)) {
            key.betting_goal =
                (it->action == DDADANG || it->action == HALF) ? 2 : 1;
            break;
        }
    }
    key.legal_mask = valid_mask(state, viewer);
    if (preserve_recall) {
        for (const Event& event : state.history) {
            if (key.history_length >= 16) {
                throw std::runtime_error("betting history exceeds recall key capacity");
            }
            const uint8_t relative_actor = event.actor == viewer ? 0 : 1;
            const uint8_t token = static_cast<uint8_t>(
                relative_actor * kActionCount + event.action);
            key.history_code = (key.history_code << 4) | token;
            ++key.history_length;
        }
    }
    return key;
}

struct RegretNode {
    std::array<double, kActionCount> regrets{};
    std::array<double, kActionCount> raw_regrets{};
    std::array<double, kActionCount> strategy_sum{};
    uint64_t touches = 0;
};

std::array<RegretNode, 2> partition_regret_node(
    const RegretNode& parent,
    double child1_mass) {
    if (!std::isfinite(child1_mass) ||
        child1_mass <= 0.0 || child1_mass >= 1.0) {
        throw std::runtime_error("child regret mass must be in (0, 1)");
    }
    const std::array<double, 2> mass = {1.0 - child1_mass, child1_mass};
    std::array<RegretNode, 2> children{};
    for (int child = 0; child < 2; ++child) {
        for (int action = 0; action < kActionCount; ++action) {
            children[child].regrets[action] =
                mass[child] * parent.regrets[action];
            children[child].raw_regrets[action] =
                mass[child] * parent.raw_regrets[action];
            children[child].strategy_sum[action] =
                mass[child] * parent.strategy_sum[action];
        }
    }
    children[1].touches = std::min<uint64_t>(
        parent.touches,
        static_cast<uint64_t>(std::llround(
            child1_mass * static_cast<double>(parent.touches))));
    children[0].touches = parent.touches - children[1].touches;
    return children;
}

struct ConflictNode {
    std::array<double, kActionCount> absolute_regret{};
    std::array<double, kActionCount> signed_regret{};
    uint64_t samples = 0;
};

void initialize_regrets(
    RegretNode& node,
    uint8_t legal_mask,
    Action preferred_action,
    double preferred_regret,
    double other_regret) {
    for (int action = 0; action < kActionCount; ++action) {
        if (!(legal_mask & (1u << action))) continue;
        const double regret = action == preferred_action
            ? preferred_regret
            : other_regret;
        node.regrets[action] = regret;
        node.raw_regrets[action] = regret;
    }
}

constexpr int kH4ActionCount = 12;

struct H4View {
    uint32_t key = 0;
    std::array<int, 4> original_index{};
};

H4View make_h4_view(const std::vector<Card>& cards) {
    if (cards.size() != 4) throw std::runtime_error("H4 needs four cards");
    std::array<int, 4> suit_permutation = {0, 1, 2, 3};
    H4View best;
    best.key = std::numeric_limits<uint32_t>::max();
    do {
        std::array<std::pair<int, int>, 4> encoded{};
        for (int index = 0; index < 4; ++index) {
            encoded[index] = {
                (cards[index].rank - 2) * 4 +
                    suit_permutation[cards[index].suit],
                index};
        }
        std::sort(encoded.begin(), encoded.end());
        uint32_t key = 0;
        for (const auto& [card, _] : encoded) {
            key = (key << 6) | static_cast<uint32_t>(card + 1);
        }
        if (key < best.key) {
            best.key = key;
            for (int index = 0; index < 4; ++index) {
                best.original_index[index] = encoded[index].second;
            }
        }
    } while (std::next_permutation(
        suit_permutation.begin(), suit_permutation.end()));
    return best;
}

std::pair<int, int> h4_positions(int action) {
    const int discard = action / 3;
    int reveal = action % 3;
    if (reveal >= discard) ++reveal;
    return {discard, reveal};
}

struct H4Node {
    std::array<double, kH4ActionCount> regrets{};
    std::array<double, kH4ActionCount> strategy_sum{};
    uint64_t touches = 0;
};

class H4Policy {
public:
    virtual ~H4Policy() = default;
    virtual std::pair<int, int> choose(const std::vector<Card>& cards) = 0;
    virtual size_t buckets() const = 0;
};

std::array<double, kH4ActionCount> h4_regret_strategy(
    const H4Node& node) {
    std::array<double, kH4ActionCount> strategy{};
    double total = 0.0;
    for (int action = 0; action < kH4ActionCount; ++action) {
        strategy[action] = std::max(0.0, node.regrets[action]);
        total += strategy[action];
    }
    if (total <= 0.0) {
        strategy.fill(1.0 / kH4ActionCount);
    } else {
        for (double& probability : strategy) probability /= total;
    }
    return strategy;
}

class H4CFR final : public H4Policy {
public:
    explicit H4CFR(uint64_t seed) : rng_(seed) {}

    std::array<double, kH4ActionCount> current_policy(
        const std::vector<Card>& cards) const {
        const auto found = nodes_.find(make_h4_view(cards).key);
        return found == nodes_.end()
            ? std::array<double, kH4ActionCount>{
                1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12,
                1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12,
                1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12}
            : h4_regret_strategy(found->second);
    }

    std::array<double, kH4ActionCount> average_policy(
        const std::vector<Card>& cards,
        bool* found = nullptr) const {
        const auto entry = nodes_.find(make_h4_view(cards).key);
        if (found) *found = entry != nodes_.end();
        if (entry == nodes_.end()) return current_policy(cards);
        auto strategy = entry->second.strategy_sum;
        const double total =
            std::accumulate(strategy.begin(), strategy.end(), 0.0);
        if (total <= 0.0) return h4_regret_strategy(entry->second);
        for (double& probability : strategy) probability /= total;
        return strategy;
    }

    int sample_current(const std::vector<Card>& cards) {
        return sample(current_policy(cards));
    }

    std::pair<int, int> choose(const std::vector<Card>& cards) override {
        const auto entry = nodes_.find(make_h4_view(cards).key);
        if (entry == nodes_.end() || entry->second.touches < min_touches_) {
            return discard_reveal(cards);
        }
        bool found = false;
        const auto strategy = average_policy(cards, &found);
        const auto [discard, reveal] = h4_positions(sample(strategy));
        const H4View view = make_h4_view(cards);
        return {
            view.original_index[discard],
            view.original_index[reveal]};
    }

    void update(
        const std::vector<Card>& cards,
        const std::array<double, kH4ActionCount>& action_values) {
        H4Node& node = nodes_[make_h4_view(cards).key];
        const auto strategy = h4_regret_strategy(node);
        double value = 0.0;
        for (int action = 0; action < kH4ActionCount; ++action) {
            value += strategy[action] * action_values[action];
            node.strategy_sum[action] += strategy[action];
        }
        for (int action = 0; action < kH4ActionCount; ++action) {
            node.regrets[action] = std::max(
                0.0,
                node.regrets[action] + action_values[action] - value);
        }
        ++node.touches;
    }

    void save(const std::string& path) const {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write H4 model: " + path);
        const char magic[8] = {'H','4','C','F','R','V','1','\0'};
        const uint64_t count = nodes_.size();
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [key, node] : nodes_) {
            output.write(reinterpret_cast<const char*>(&key), sizeof(key));
            output.write(reinterpret_cast<const char*>(&node), sizeof(node));
        }
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open H4 model: " + path);
        char magic[8]{};
        uint64_t count = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (std::memcmp(magic, "H4CFRV1", 7) != 0) {
            throw std::runtime_error("incompatible H4 model");
        }
        nodes_.clear();
        nodes_.reserve(static_cast<size_t>(count * 1.3) + 1);
        for (uint64_t index = 0; index < count; ++index) {
            uint32_t key = 0;
            H4Node node;
            input.read(reinterpret_cast<char*>(&key), sizeof(key));
            input.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes_.emplace(key, node);
        }
        if (!input) throw std::runtime_error("truncated H4 model");
    }

    size_t buckets() const override { return nodes_.size(); }
    void set_min_touches(uint64_t value) { min_touches_ = value; }

private:
    int sample(const std::array<double, kH4ActionCount>& strategy) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double threshold = uniform(rng_);
        double cumulative = 0.0;
        for (int action = 0; action < kH4ActionCount; ++action) {
            cumulative += strategy[action];
            if (threshold <= cumulative) return action;
        }
        return kH4ActionCount - 1;
    }

    std::mt19937_64 rng_;
    std::unordered_map<uint32_t, H4Node> nodes_;
    uint64_t min_touches_ = 1;
};

struct H4QNode {
    std::array<double, kH4ActionCount> sums{};
    std::array<double, kH4ActionCount> squared_sums{};
    uint64_t samples = 0;
};

struct H4QStats {
    uint64_t samples = 0;
    uint64_t min_samples = 0;
    uint64_t max_samples = 0;
};

class H4QPolicy final : public H4Policy {
public:
    void set_min_samples(uint64_t value) { min_samples_ = value; }
    void set_lcb_beta(double value) { lcb_beta_ = value; }

    std::pair<int, int> choose(const std::vector<Card>& cards) override {
        const H4View view = make_h4_view(cards);
        const auto found = nodes_.find(view.key);
        if (found == nodes_.end() || found->second.samples < min_samples_) {
            return discard_reveal(cards);
        }
        const H4QNode& node = found->second;
        int best_action = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int action = 0; action < kH4ActionCount; ++action) {
            const double count = static_cast<double>(node.samples);
            const double mean = node.sums[action] / count;
            double standard_error = 0.0;
            if (node.samples > 1) {
                const double centered = std::max(
                    0.0,
                    node.squared_sums[action] -
                        node.sums[action] * node.sums[action] / count);
                standard_error = std::sqrt(
                    centered / (count - 1.0) / count);
            }
            const double score = mean - lcb_beta_ * standard_error;
            if (score > best_score) {
                best_score = score;
                best_action = action;
            }
        }
        const auto [discard, reveal] = h4_positions(best_action);
        return {
            view.original_index[discard],
            view.original_index[reveal]};
    }

    void update(
        const std::vector<Card>& cards,
        const std::array<double, kH4ActionCount>& action_values) {
        H4QNode& node = nodes_[make_h4_view(cards).key];
        for (int action = 0; action < kH4ActionCount; ++action) {
            node.sums[action] += action_values[action];
            node.squared_sums[action] +=
                action_values[action] * action_values[action];
        }
        ++node.samples;
    }

    void save(const std::string& path) const {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write H4 Q model: " + path);
        const char magic[8] = {'H','4','Q','E','V','V','1','\0'};
        const uint64_t count = nodes_.size();
        output.write(magic, sizeof(magic));
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [key, node] : nodes_) {
            output.write(reinterpret_cast<const char*>(&key), sizeof(key));
            output.write(reinterpret_cast<const char*>(&node), sizeof(node));
        }
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open H4 Q model: " + path);
        char magic[8]{};
        uint64_t count = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        if (std::memcmp(magic, "H4QEVV1", 7) != 0) {
            throw std::runtime_error("incompatible H4 Q model");
        }
        nodes_.clear();
        nodes_.reserve(static_cast<size_t>(count * 1.3) + 1);
        for (uint64_t index = 0; index < count; ++index) {
            uint32_t key = 0;
            H4QNode node;
            input.read(reinterpret_cast<char*>(&key), sizeof(key));
            input.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes_.emplace(key, node);
        }
        if (!input) throw std::runtime_error("truncated H4 Q model");
    }

    size_t buckets() const override { return nodes_.size(); }

    H4QStats stats() const {
        H4QStats result;
        result.min_samples = nodes_.empty()
            ? 0
            : std::numeric_limits<uint64_t>::max();
        for (const auto& [_, node] : nodes_) {
            result.samples += node.samples;
            result.min_samples = std::min(result.min_samples, node.samples);
            result.max_samples = std::max(result.max_samples, node.samples);
        }
        return result;
    }

private:
    std::unordered_map<uint32_t, H4QNode> nodes_;
    uint64_t min_samples_ = 32;
    double lcb_beta_ = 1.96;
};

struct ClusterGrowthCandidate {
    bool found = false;
    State state{};
    int actor = 0;
    double regret = 0.0;
    double best_value = 0.0;
    std::array<double, kActionCount> action_values{};
};

struct ResidualPoint {
    PowerVector vector{};
    double regret = 0.0;
};

struct SplitAuditConfig {
    bool enabled = false;
    double probability = 0.10;
    int rollouts = 12;
    double gap_threshold = 0.04;
    double confidence_z = 1.645;
    size_t min_strong = 2;
    double reward_gain = 0.20;
    size_t reservoir = 32;
    uint16_t max_depth = 8;
    size_t max_leaves = 512;
    uint64_t min_touches = 0;
    double gap_threshold_min = 0.0;
    uint64_t gap_touch_reference = 100000;
};

struct SplitAuditStats {
    uint64_t audits = 0;
    uint64_t strong_audits = 0;
    uint64_t audit_node_visits = 0;
    uint64_t queued_states = 0;
    uint64_t accepted_splits = 0;
    uint64_t rejected_splits = 0;
    uint64_t touch_deferred = 0;
    uint64_t capacity_deferred = 0;
    uint64_t depth_deferred = 0;
    uint64_t evidence_rejected = 0;
    uint64_t reward_rejected = 0;
    std::array<uint64_t, 3> accepted_by_street{};
    uint64_t inherited_touches = 0;
    std::array<uint64_t, 2> last_child_touches{};
    uint64_t threshold_evaluations = 0;
    double threshold_sum = 0.0;
    double threshold_min = std::numeric_limits<double>::infinity();
    double threshold_max = 0.0;
};

struct SplitAuditKey {
    uint8_t street = 0;
    uint8_t legal_mask = 0;
    uint16_t leaf = 0;

    bool operator==(const SplitAuditKey& other) const {
        return street == other.street &&
            legal_mask == other.legal_mask && leaf == other.leaf;
    }
};

struct SplitAuditKeyHash {
    size_t operator()(const SplitAuditKey& key) const {
        return static_cast<size_t>(key.street) << 24 |
            static_cast<size_t>(key.legal_mask) << 16 | key.leaf;
    }
};

using LocalSplitSample =
    local_split::Sample<kPowerDimensions, kActionCount>;

struct SplitAuditEvidence {
    std::vector<LocalSplitSample> samples;
    std::vector<uint8_t> strong;
    bool queued = false;
};

struct PendingSplitAudit {
    State state;
    int viewer = 0;
    SplitAuditKey key{};
};

std::array<double, kActionCount> current_strategy(const RegretNode& node, uint8_t mask) {
    std::array<double, kActionCount> strategy{};
    double total = 0.0;
    int count = 0;
    for (int action = 0; action < kActionCount; ++action) {
        if (!(mask & (1u << action))) continue;
        strategy[action] = std::max(0.0, node.regrets[action]);
        total += strategy[action];
        ++count;
    }
    if (total == 0.0) {
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) strategy[action] = 1.0 / count;
        }
    } else {
        for (double& probability : strategy) probability /= total;
    }
    return strategy;
}

std::array<double, kActionCount> average_strategy(const RegretNode& node, uint8_t mask) {
    std::array<double, kActionCount> strategy{};
    double total = 0.0;
    for (int action = 0; action < kActionCount; ++action) {
        if (mask & (1u << action)) total += node.strategy_sum[action];
    }
    if (total == 0.0) return current_strategy(node, mask);
    for (int action = 0; action < kActionCount; ++action) {
        if (mask & (1u << action)) strategy[action] = node.strategy_sum[action] / total;
    }
    return strategy;
}

std::array<double, kActionCount> uniform_strategy(uint8_t mask) {
    std::array<double, kActionCount> strategy{};
    const int count = static_cast<int>(actions_from_mask(mask).size());
    if (!count) return strategy;
    for (int action = 0; action < kActionCount; ++action) {
        if (mask & (1u << action)) strategy[action] = 1.0 / count;
    }
    return strategy;
}

std::array<double, kActionCount> projected_strategy(
    const std::array<double, kActionCount>& strategy,
    const std::array<double, kActionCount>& gradient,
    uint8_t mask,
    double step) {
    std::array<double, kActionCount> result{};
    std::vector<double> values;
    values.reserve(kActionCount);
    for (int action = 0; action < kActionCount; ++action) {
        if (mask & (1u << action)) {
            values.push_back(strategy[action] + step * gradient[action]);
        }
    }
    if (values.empty()) return result;
    std::sort(values.begin(), values.end(), std::greater<double>());
    double prefix = 0.0;
    double theta = 0.0;
    for (size_t index = 0; index < values.size(); ++index) {
        prefix += values[index];
        const double candidate = (prefix - 1.0) / (index + 1);
        if (index + 1 == values.size() || values[index + 1] <= candidate) {
            theta = candidate;
            break;
        }
    }
    double total = 0.0;
    for (int action = 0; action < kActionCount; ++action) {
        if (!(mask & (1u << action))) continue;
        result[action] = std::max(0.0, strategy[action] + step * gradient[action] - theta);
        total += result[action];
    }
    if (total <= 0.0) return uniform_strategy(mask);
    for (double& probability : result) probability /= total;
    return result;
}

int first_bettor(const State& state);

class MCCFR {
public:
    MCCFR(
        bool regret_plus,
        uint64_t seed,
        int start_street = 7,
        PowerAtlas* atlas = nullptr,
        uint64_t prune_after = 0,
        double prune_threshold = 50.0,
        uint64_t prune_refresh = 64,
        int soft_top_k = 1,
        double soft_top_p = 0.0,
        double soft_temperature = 0.1,
        bool soft_local_bandwidth = false,
        bool preserve_recall = false,
        SoftGrowthMode soft_growth_mode = SoftGrowthMode::None,
        double initial_fold_regret = 0.0,
        double initial_other_regret = 0.0,
        InitialRegretPolicy initial_regret_policy = InitialRegretPolicy::Fold,
        int initial_made_min_category = 1)
        : regret_plus_(regret_plus),
          start_street_(start_street),
          atlas_(atlas),
          prune_after_(prune_after),
          prune_threshold_(prune_threshold),
          prune_refresh_(prune_refresh),
          soft_top_k_(soft_top_k),
          soft_top_p_(soft_top_p),
          soft_temperature_(soft_temperature),
          soft_local_bandwidth_(soft_local_bandwidth),
          preserve_recall_(preserve_recall),
          soft_growth_mode_(soft_growth_mode),
          initial_fold_regret_(initial_fold_regret),
          initial_other_regret_(initial_other_regret),
          initial_regret_policy_(initial_regret_policy),
          initial_made_min_category_(initial_made_min_category),
          rng_(seed),
          audit_rng_(seed ^ 0x6a09e667f3bcc909ull) {
        if (atlas_ && atlas_->tree_enabled() &&
            (soft_top_k_ != 1 || soft_top_p_ > 0.0 ||
             soft_local_bandwidth_ ||
             soft_growth_mode_ != SoftGrowthMode::None)) {
            throw std::runtime_error("tree MCCFR is hard-only");
        }
        nodes_.reserve(1 << 20);
        if (atlas_) {
            for (int street = 5; street <= 7; ++street) {
                temperature_scales_[street - 5].assign(
                    atlas_->clusters(street), 1.0);
                temperature_losses_[street - 5].resize(
                    atlas_->clusters(street));
                temperature_samples_[street - 5].assign(
                    atlas_->clusters(street), 0);
                temperature_selected_losses_[street - 5].assign(
                    atlas_->clusters(street), 0.0);
                growth_candidates_[street - 5].resize(
                    atlas_->clusters(street));
                growth_regret_sums_[street - 5].assign(
                    atlas_->clusters(street), 0.0);
                growth_samples_[street - 5].assign(
                    atlas_->clusters(street), 0);
                residual_points_[street - 5].resize(
                    atlas_->clusters(street));
                residual_min_regrets_[street - 5].assign(
                    atlas_->clusters(street),
                    -std::numeric_limits<double>::infinity());
            }
        }
    }

    void use_range_model(const ActionRangeModel* model) {
        if (model && preserve_recall_) {
            throw std::runtime_error(
                "range posterior and exact recall are separate abstractions");
        }
        range_model_ = model;
    }

    void use_previous_street_summary(bool enabled) {
        if (enabled) betting_memory_ = 1;
        else if (betting_memory_ == 1) betting_memory_ = 0;
    }

    void use_cumulative_street_summary(bool enabled) {
        if (enabled) betting_memory_ = 2;
        else if (betting_memory_ == 2) betting_memory_ = 0;
    }

    void use_cumulative_street_strength_summary(bool enabled) {
        if (enabled) betting_memory_ = 3;
        else if (betting_memory_ == 3) betting_memory_ = 0;
    }

    Action choose(State const& actual, int viewer, int iterations) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            State simulation = determinize(actual, viewer);
            const int traverser = iteration % 2 == 0 ? viewer : 1 - viewer;
            traverse(std::move(simulation), traverser, {1.0, 1.0});
            ++traversals_;
        }
        const uint8_t mask = valid_mask(actual, viewer);
        return sample(policy(actual, viewer), mask);
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        const uint8_t mask = valid_mask(state, viewer);
        if (soft_enabled()) {
            std::array<double, kActionCount> mixed{};
            double total = 0.0;
            InfoKey key =
                make_power_key(state, viewer, *atlas_, preserve_recall_);
            const PowerObservationKey observation =
                power_observation_key(state, viewer);
            auto cached = soft_weight_cache_.find(observation);
            if (cached == soft_weight_cache_.end()) {
                cached = soft_weight_cache_.emplace(
                    observation,
                    atlas_->weights(
                        state,
                        viewer,
                        soft_top_k_,
                        soft_top_p_,
                        soft_temperature_,
                        soft_local_bandwidth_,
                        &temperature_scales_[state.street - 5])).first;
            }
            const auto add_neighbors = [&](const auto& weights) {
                for (const auto& [cluster, weight] : weights) {
                    key.power_cluster = cluster;
                    const auto it = nodes_.find(key);
                    if (it == nodes_.end()) continue;
                    const auto local = average_strategy(it->second, mask);
                    for (int action = 0; action < kActionCount; ++action) {
                        mixed[action] += weight * local[action];
                    }
                    total += weight;
                }
            };
            add_neighbors(cached->second);
            if (total <= 0.0 && soft_top_p_ > 0.0) {
                add_neighbors(atlas_->weights(
                    state,
                    viewer,
                    soft_top_k_,
                    1.0,
                    soft_temperature_,
                    soft_local_bandwidth_,
                    &temperature_scales_[state.street - 5]));
            }
            if (found) *found = total > 0.0;
            if (total <= 0.0) return uniform_strategy(mask);
            for (double& probability : mixed) probability /= total;
            return mixed;
        }
        const auto it = nodes_.find(bucket_key(state, viewer));
        if (found) *found = it != nodes_.end();
        return it == nodes_.end()
            ? uniform_strategy(mask)
            : average_strategy(it->second, mask);
    }

    void train_root(State state, int traverser) {
        if (soft_growth_mode_ == SoftGrowthMode::None) {
            traverse(std::move(state), traverser, {1.0, 1.0});
        } else {
            soft_traverse(std::move(state), traverser);
        }
        ++traversals_;
    }

    // Read-only current-policy view for matched training diagnostics.
    size_t bucket_count() const { return nodes_.size(); }

    std::array<double, kActionCount> instantaneous_policy(
        const State& state, int viewer, bool* found = nullptr) const {
        if (soft_enabled()) {
            throw std::runtime_error("instantaneous_policy requires hard buckets");
        }
        const auto it = nodes_.find(bucket_key(state, viewer));
        if (found) *found = it != nodes_.end();
        const uint8_t mask = valid_mask(state, viewer);
        return it == nodes_.end()
            ? uniform_strategy(mask) : current_strategy(it->second, mask);
    }

    void configure_split_audit(const SplitAuditConfig& config) {
        if (config.enabled && (!atlas_ || !atlas_->tree_enabled())) {
            throw std::runtime_error(
                "split audit requires a power-tree atlas");
        }
        split_audit_config_ = config;
        pending_split_audits_.clear();
        pending_split_audits_seen_ = 0;
        split_audit_evidence_.clear();
        split_candidates_.clear();
        split_audit_stats_ = {};
    }

    const SplitAuditStats& split_audit_stats() const {
        return split_audit_stats_;
    }

    PowerCacheStats power_cache_stats() const {
        return atlas_ ? atlas_->assignment_cache_stats() : PowerCacheStats{};
    }

    std::array<size_t, 3> split_leaf_counts() const {
        if (!atlas_ || !atlas_->tree_enabled()) return {};
        return {
            atlas_->clusters(5),
            atlas_->clusters(6),
            atlas_->clusters(7)};
    }

    bool process_split_audits() {
        if (!split_audit_config_.enabled) return false;
        for (const PendingSplitAudit& pending : pending_split_audits_) {
            evaluate_split_audit(pending);
        }
        pending_split_audits_.clear();
        pending_split_audits_seen_ = 0;

        while (!split_candidates_.empty()) {
            const SplitAuditKey key = split_candidates_.front();
            split_candidates_.erase(split_candidates_.begin());
            auto evidence = split_audit_evidence_.find(key);
            if (evidence == split_audit_evidence_.end()) continue;
            evidence->second.queued = false;

            const size_t strong_count = static_cast<size_t>(std::count(
                evidence->second.strong.begin(),
                evidence->second.strong.end(),
                static_cast<uint8_t>(1)));
            if (atlas_->clusters(key.street) >=
                split_audit_config_.max_leaves) {
                ++split_audit_stats_.capacity_deferred;
                clear_split_evidence(key.street, key.leaf);
                continue;
            }
            if (atlas_->leaf_depth(key.street, key.leaf) >=
                split_audit_config_.max_depth) {
                ++split_audit_stats_.depth_deferred;
                clear_split_evidence(key.street, key.leaf);
                continue;
            }
            if (strong_count < split_audit_config_.min_strong) {
                ++split_audit_stats_.rejected_splits;
                ++split_audit_stats_.evidence_rejected;
                clear_split_evidence(key.street, key.leaf);
                continue;
            }

            const auto split = local_split::reward_aware_two_means(
                evidence->second.samples);
            if (!split ||
                !local_split::passes_reward_split(
                    *split, split_audit_config_.reward_gain)) {
                ++split_audit_stats_.rejected_splits;
                ++split_audit_stats_.reward_rejected;
                clear_split_evidence(key.street, key.leaf);
                continue;
            }

            const PowerSplitResult atlas_split = atlas_->split_leaf(
                key.street,
                key.leaf,
                split->centroids[0],
                split->centroids[1],
                split->child_mass_ratio[1]);
            partition_regret_nodes(
                key.street,
                atlas_split.parent_leaf,
                atlas_split.child_leaf,
                split->child_mass_ratio[1]);
            split_audit_stats_.inherited_touches +=
                atlas_split.child0_touches + atlas_split.child1_touches;
            split_audit_stats_.last_child_touches = {
                atlas_split.child0_touches,
                atlas_split.child1_touches};
            ++split_audit_stats_.accepted_splits;
            ++split_audit_stats_.accepted_by_street.at(key.street - 5);
            clear_split_evidence(key.street, key.leaf);
            return true;
        }
        return false;
    }

    size_t partition_regret_nodes(
        int street,
        uint16_t parent_leaf,
        uint16_t child_leaf,
        double child1_mass) {
        if (!atlas_ || !atlas_->tree_enabled()) {
            throw std::runtime_error("regret partition requires a tree atlas");
        }
        if (street < 5 || street > 7 || parent_leaf == child_leaf ||
            parent_leaf >= atlas_->clusters(street) ||
            child_leaf >= atlas_->clusters(street)) {
            throw std::runtime_error("invalid tree regret partition IDs");
        }
        if (!imitation_priors_.empty() || !imitation_counts_.empty()) {
            throw std::runtime_error(
                "tree regret partition does not support imitation state");
        }
        struct Pending {
            InfoKey parent_key{};
            InfoKey child_key{};
            std::array<RegretNode, 2> children{};
        };
        std::vector<Pending> pending;
        for (const auto& [key, node] : nodes_) {
            if (key.abstraction != 8 || key.street != street ||
                key.power_cluster != parent_leaf) {
                continue;
            }
            InfoKey child_key = key;
            child_key.power_cluster = child_leaf;
            if (nodes_.find(child_key) != nodes_.end()) {
                throw std::runtime_error("new tree leaf already has regret state");
            }
            pending.push_back(Pending{
                key,
                child_key,
                partition_regret_node(node, child1_mass)});
        }
        nodes_.reserve(nodes_.size() + pending.size());
        for (const Pending& item : pending) {
            nodes_.find(item.parent_key)->second = item.children[0];
        }
        for (const Pending& item : pending) {
            const bool inserted =
                nodes_.emplace(item.child_key, item.children[1]).second;
            if (!inserted) {
                throw std::runtime_error("failed to create child regret node");
            }
        }
        return pending.size();
    }

    void use_decaying_imitation_prior(bool enabled) {
        decaying_imitation_prior_ = enabled;
    }

    void scale_imitation_prior(double scale) {
        for (auto& [key, prior] : imitation_priors_) {
            const double old_strength = prior.strength;
            const double new_strength = old_strength * scale;
            const double removed = old_strength - new_strength;
            auto node = nodes_.find(key);
            if (node != nodes_.end()) {
                for (int action = 0; action < kActionCount; ++action) {
                    if (!(key.legal_mask & (1u << action))) continue;
                    node->second.strategy_sum[action] = std::max(
                        0.0,
                        node->second.strategy_sum[action] -
                            removed * prior.policy[action]);
                }
            }
            prior.strength = new_strength;
        }
    }

    void imitate_policy(
        const State& state,
        int viewer,
        const std::array<double, kActionCount>& target,
        double strength) {
        const InfoKey key = bucket_key(state, viewer);
        auto [it, inserted] = nodes_.try_emplace(key);
        if (inserted) ++misses_;
        else ++hits_;
        uint64_t& count = imitation_counts_[key];
        if (decaying_imitation_prior_) {
            auto& prior = imitation_priors_[key];
            for (int action = 0; action < kActionCount; ++action) {
                if (!(key.legal_mask & (1u << action))) continue;
                prior.policy[action] =
                    (prior.policy[action] * count + target[action]) /
                    (count + 1.0);
                it->second.strategy_sum[action] =
                    strength * prior.policy[action];
            }
            prior.strength = strength;
            ++count;
            return;
        }
        for (int action = 0; action < kActionCount; ++action) {
            if (!(key.legal_mask & (1u << action))) continue;
            const double previous = count
                ? it->second.regrets[action] / strength
                : 0.0;
            const double average =
                (previous * count + target[action]) / (count + 1.0);
            it->second.regrets[action] = strength * average;
            it->second.raw_regrets[action] = strength * average;
            it->second.strategy_sum[action] = strength * average;
        }
        ++count;
    }

    void advance_imitation_state(State& state) {
        advance_street(state);
    }

    void save(const std::string& path) const {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot open model for writing: " + path);
        const bool tree_model = atlas_ && atlas_->tree_enabled();
        const char magic5[8] = {'M','C','C','F','R','V','5','\0'};
        const char magic6[8] = {'M','C','C','F','R','V','6','\0'};
        const char* magic = tree_model ? magic6 : magic5;
        output.write(magic, sizeof(magic5));
        const uint8_t plus = regret_plus_ ? 1 : 0;
        const uint8_t abstraction = abstraction_id();
        const uint8_t start_street = static_cast<uint8_t>(start_street_);
        const uint64_t count = nodes_.size();
        output.write(reinterpret_cast<const char*>(&plus), sizeof(plus));
        output.write(reinterpret_cast<const char*>(&abstraction), sizeof(abstraction));
        output.write(reinterpret_cast<const char*>(&start_street), sizeof(start_street));
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        if (tree_model) {
            const uint64_t generation = atlas_->generation_id();
            output.write(
                reinterpret_cast<const char*>(&generation), sizeof(generation));
            for (int street = 5; street <= 7; ++street) {
                const uint32_t bases =
                    static_cast<uint32_t>(atlas_->base_clusters(street));
                const uint32_t leaves =
                    static_cast<uint32_t>(atlas_->clusters(street));
                output.write(reinterpret_cast<const char*>(&bases), sizeof(bases));
                output.write(reinterpret_cast<const char*>(&leaves), sizeof(leaves));
            }
        }
        for (const auto& [key, node] : nodes_) {
            output.write(reinterpret_cast<const char*>(&key), sizeof(key));
            output.write(reinterpret_cast<const char*>(&node), sizeof(node));
        }
        const uint8_t has_prior =
            decaying_imitation_prior_ && !imitation_priors_.empty();
        const uint64_t prior_count = has_prior ? imitation_priors_.size() : 0;
        output.write(
            reinterpret_cast<const char*>(&has_prior), sizeof(has_prior));
        output.write(
            reinterpret_cast<const char*>(&prior_count), sizeof(prior_count));
        if (has_prior) {
            for (const auto& [key, prior] : imitation_priors_) {
                output.write(reinterpret_cast<const char*>(&key), sizeof(key));
                output.write(
                    reinterpret_cast<const char*>(&prior), sizeof(prior));
            }
        }
        if (!output) throw std::runtime_error("failed to write model: " + path);
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open model: " + path);
        char magic[8]{};
        uint8_t plus = 0;
        uint8_t abstraction = 0;
        uint8_t start_street = 0;
        uint64_t count = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&plus), sizeof(plus));
        input.read(reinterpret_cast<char*>(&abstraction), sizeof(abstraction));
        input.read(reinterpret_cast<char*>(&start_street), sizeof(start_street));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        const bool version4 = std::memcmp(magic, "MCCFRV4", 7) == 0;
        const bool version5 = std::memcmp(magic, "MCCFRV5", 7) == 0;
        const bool version6 = std::memcmp(magic, "MCCFRV6", 7) == 0;
        if ((!version4 && !version5 && !version6) ||
            plus != static_cast<uint8_t>(regret_plus_) ||
            abstraction != abstraction_id() ||
            start_street != start_street_) {
            throw std::runtime_error("incompatible model");
        }
        if (version6) {
            if (!atlas_ || !atlas_->tree_enabled()) {
                throw std::runtime_error("tree model requires a tree atlas");
            }
            uint64_t generation = 0;
            input.read(
                reinterpret_cast<char*>(&generation), sizeof(generation));
            if (generation != atlas_->generation_id()) {
                throw std::runtime_error("model and tree atlas generation mismatch");
            }
            for (int street = 5; street <= 7; ++street) {
                uint32_t bases = 0;
                uint32_t leaves = 0;
                input.read(reinterpret_cast<char*>(&bases), sizeof(bases));
                input.read(reinterpret_cast<char*>(&leaves), sizeof(leaves));
                if (bases != atlas_->base_clusters(street) ||
                    leaves != atlas_->clusters(street)) {
                    throw std::runtime_error("model and tree atlas shape mismatch");
                }
            }
        } else if (atlas_ && atlas_->tree_enabled()) {
            throw std::runtime_error("legacy model cannot be loaded as a tree model");
        }
        nodes_.clear();
        nodes_.reserve(static_cast<size_t>(count * 1.3) + 1);
        for (uint64_t i = 0; i < count; ++i) {
            InfoKey key{};
            RegretNode node{};
            input.read(reinterpret_cast<char*>(&key), sizeof(key));
            input.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes_.emplace(key, node);
        }
        imitation_priors_.clear();
        decaying_imitation_prior_ = false;
        if (version5 || version6) {
            uint8_t has_prior = 0;
            uint64_t prior_count = 0;
            input.read(
                reinterpret_cast<char*>(&has_prior), sizeof(has_prior));
            input.read(
                reinterpret_cast<char*>(&prior_count), sizeof(prior_count));
            imitation_priors_.reserve(
                static_cast<size_t>(prior_count * 1.3) + 1);
            for (uint64_t i = 0; i < prior_count; ++i) {
                InfoKey key{};
                ImitationPrior prior{};
                input.read(reinterpret_cast<char*>(&key), sizeof(key));
                input.read(reinterpret_cast<char*>(&prior), sizeof(prior));
                imitation_priors_.emplace(key, prior);
            }
            decaying_imitation_prior_ = has_prior && prior_count > 0;
        }
        if (!input) throw std::runtime_error("truncated model");
    }

    uint64_t merge_worker_delta(const MCCFR& base, const MCCFR& worker) {
        if (regret_plus_ || base.regret_plus_ || worker.regret_plus_) {
            throw std::runtime_error("parallel delta merge only supports --algorithm mccfr");
        }
        if (start_street_ != base.start_street_ ||
            start_street_ != worker.start_street_ ||
            (atlas_ != nullptr) != (base.atlas_ != nullptr) ||
            (atlas_ != nullptr) != (worker.atlas_ != nullptr)) {
            throw std::runtime_error("incompatible models for parallel merge");
        }

        uint64_t added_touches = 0;
        for (const auto& [key, worker_node] : worker.nodes_) {
            const auto base_it = base.nodes_.find(key);
            const RegretNode empty{};
            const RegretNode& base_node =
                base_it == base.nodes_.end() ? empty : base_it->second;
            if (worker_node.touches < base_node.touches) {
                throw std::runtime_error("worker model predates merge base");
            }

            RegretNode& merged = nodes_[key];
            for (int action = 0; action < kActionCount; ++action) {
                merged.regrets[action] +=
                    worker_node.regrets[action] - base_node.regrets[action];
                merged.raw_regrets[action] +=
                    worker_node.raw_regrets[action] - base_node.raw_regrets[action];
                merged.strategy_sum[action] +=
                    worker_node.strategy_sum[action] - base_node.strategy_sum[action];
            }
            const uint64_t touch_delta = worker_node.touches - base_node.touches;
            merged.touches += touch_delta;
            added_touches += touch_delta;
        }
        return added_touches;
    }

    struct Stats {
        uint64_t buckets = 0;
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t singletons = 0;
        uint64_t max_touches = 0;
        uint64_t pruned_branches = 0;
        double average_touches = 0.0;
        double median_touches = 0.0;
        double p95_touches = 0.0;
        double hit_rate = 0.0;
        double singleton_ratio = 0.0;
        double new_buckets_per_traversal = 0.0;
        std::array<uint64_t, 9> by_hand_category{};
        std::array<uint64_t, 64> by_range_bucket{};
        std::array<uint64_t, 4> by_previous_street_aggression{};
        std::array<uint64_t, 16> by_cumulative_street_aggression{};
        std::array<uint64_t, 81> by_cumulative_street_strength{};
        std::array<uint64_t, 16> by_history_length{};
        std::array<uint64_t, 3> by_street{};
        std::array<std::vector<uint64_t>, 3> by_power_cluster{};
        std::array<uint64_t, 4> by_own_bet_count{};
        std::array<uint64_t, 4> by_opponent_bet_count{};
        std::array<uint64_t, 2> by_checked{};
        std::array<uint64_t, 6> by_last_action_class{};
        std::array<uint64_t, 3> by_betting_goal{};
        std::array<uint64_t, 9> by_legal_action_count{};
    };

    Stats stats() const {
        Stats result;
        if (atlas_) {
            for (int street = 5; street <= 7; ++street) {
                result.by_power_cluster[street - 5].resize(
                    atlas_->clusters(street));
            }
        }
        result.buckets = nodes_.size();
        result.lookups = hits_ + misses_;
        result.hits = hits_;
        result.misses = misses_;
        result.pruned_branches = pruned_branches_;
        result.hit_rate = result.lookups ? hits_ / static_cast<double>(result.lookups) : 0.0;
        result.new_buckets_per_traversal =
            traversals_ ? misses_ / static_cast<double>(traversals_) : 0.0;
        std::vector<uint64_t> touches;
        touches.reserve(nodes_.size());
        uint64_t total = 0;
        for (const auto& [_, node] : nodes_) {
            touches.push_back(node.touches);
            total += node.touches;
            result.singletons += node.touches == 1;
            result.max_touches = std::max(result.max_touches, node.touches);
        }
        for (const auto& [key, _] : nodes_) {
            if (key.street >= 5 && key.street <= 7) ++result.by_street[key.street - 5];
            if (key.abstraction != 0) {
                const int street = key.street - 5;
                if (street >= 0 && street < 3 &&
                    key.power_cluster < result.by_power_cluster[street].size()) {
                    ++result.by_power_cluster[street][key.power_cluster];
                }
                ++result.by_own_bet_count[
                    std::min<size_t>(3, key.own_bet_count)];
                ++result.by_opponent_bet_count[
                    std::min<size_t>(3, key.opponent_bet_count)];
                ++result.by_checked[std::min<size_t>(1, key.checked)];
                ++result.by_last_action_class[
                    std::min<size_t>(5, key.last_action_class)];
                ++result.by_betting_goal[
                    std::min<size_t>(2, key.betting_goal)];
                int legal_actions = 0;
                for (int action = 0; action < kActionCount; ++action) {
                    legal_actions += (key.legal_mask >> action) & 1u;
                }
                ++result.by_legal_action_count[legal_actions];
                if (key.abstraction == 3) {
                    ++result.by_history_length[
                        std::min<size_t>(15, key.history_length)];
                } else if (key.abstraction == 4) {
                    ++result.by_range_bucket[
                        std::min<size_t>(63, key.category)];
                } else if (key.abstraction == 5) {
                    ++result.by_previous_street_aggression[
                        std::min<size_t>(3, key.rank_bucket)];
                } else if (key.abstraction == 6) {
                    ++result.by_cumulative_street_aggression[
                        std::min<size_t>(15, key.rank_bucket)];
                } else if (key.abstraction == 7) {
                    ++result.by_cumulative_street_strength[
                        std::min<size_t>(80, key.rank_bucket)];
                }
            } else {
                ++result.by_hand_category[std::min<size_t>(8, key.category)];
                ++result.by_history_length[
                    std::min<size_t>(15, key.history_length)];
            }
        }
        if (!touches.empty()) {
            std::sort(touches.begin(), touches.end());
            result.average_touches = total / static_cast<double>(touches.size());
            result.median_touches = touches[touches.size() / 2];
            result.p95_touches = touches[static_cast<size_t>((touches.size() - 1) * 0.95)];
            result.singleton_ratio = result.singletons / static_cast<double>(touches.size());
        }
        return result;
    }

    void dump_bucket_touches(const std::string& path) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("cannot write bucket touches: " + path);

        std::vector<std::pair<const InfoKey*, const RegretNode*>> rows;
        rows.reserve(nodes_.size());
        for (const auto& [key, node] : nodes_) rows.emplace_back(&key, &node);
        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
            return left.second->touches > right.second->touches;
        });

        output
            << "abstraction,street,power_cluster,category,memory_code,"
               "pot_odds_bucket,stack_pot_bucket,own_bet_count,"
               "opponent_bet_count,checked,last_action_class,betting_goal,"
               "legal_mask,history_length,history_code,touches\n";
        for (const auto& [key, node] : rows) {
            output
                << static_cast<int>(key->abstraction) << ','
                << static_cast<int>(key->street) << ','
                << key->power_cluster << ','
                << static_cast<int>(key->category) << ','
                << static_cast<int>(key->rank_bucket) << ','
                << static_cast<int>(key->pot_odds_bucket) << ','
                << static_cast<int>(key->stack_pot_bucket) << ','
                << static_cast<int>(key->own_bet_count) << ','
                << static_cast<int>(key->opponent_bet_count) << ','
                << static_cast<int>(key->checked) << ','
                << static_cast<int>(key->last_action_class) << ','
                << static_cast<int>(key->betting_goal) << ','
                << static_cast<int>(key->legal_mask) << ','
                << static_cast<int>(key->history_length) << ','
                << key->history_code << ','
                << node->touches << '\n';
        }
    }

    uint64_t traversals() const { return traversals_; }
    uint64_t node_visits() const { return node_visits_; }
    size_t imitation_prior_count() const { return imitation_priors_.size(); }

    struct ImitationPriorStats {
        size_t buckets = 0;
        size_t lambda_above_half = 0;
        double lambda_mean = 0.0;
        double lambda_median = 0.0;
        double lambda_p95 = 0.0;
        double lambda_max = 0.0;
        double average_prior_fraction_mean = 0.0;
        double average_prior_fraction_median = 0.0;
        double average_prior_fraction_p95 = 0.0;
        double strength_mean = 0.0;
    };

    ImitationPriorStats imitation_prior_stats() const {
        ImitationPriorStats result;
        std::vector<double> lambdas;
        std::vector<double> average_prior_fractions;
        lambdas.reserve(imitation_priors_.size());
        average_prior_fractions.reserve(imitation_priors_.size());
        for (const auto& [key, prior] : imitation_priors_) {
            const auto node = nodes_.find(key);
            if (node == nodes_.end()) continue;
            const double lambda = prior.strength /
                (prior.strength + node->second.touches);
            lambdas.push_back(lambda);
            result.lambda_mean += lambda;
            result.strength_mean += prior.strength;
            result.lambda_above_half += lambda > 0.5;
            result.lambda_max = std::max(result.lambda_max, lambda);
            double strategy_mass = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (key.legal_mask & (1u << action)) {
                    strategy_mass += node->second.strategy_sum[action];
                }
            }
            average_prior_fractions.push_back(std::min(
                1.0, prior.strength / std::max(1e-12, strategy_mass)));
        }
        result.buckets = lambdas.size();
        if (lambdas.empty()) return result;
        std::sort(lambdas.begin(), lambdas.end());
        result.lambda_mean /= lambdas.size();
        result.strength_mean /= lambdas.size();
        result.lambda_median = lambdas[lambdas.size() / 2];
        result.lambda_p95 = lambdas[static_cast<size_t>(
            (lambdas.size() - 1) * 0.95)];
        std::sort(
            average_prior_fractions.begin(), average_prior_fractions.end());
        result.average_prior_fraction_mean = std::accumulate(
            average_prior_fractions.begin(), average_prior_fractions.end(), 0.0) /
            average_prior_fractions.size();
        result.average_prior_fraction_median =
            average_prior_fractions[average_prior_fractions.size() / 2];
        result.average_prior_fraction_p95 = average_prior_fractions[
            static_cast<size_t>((average_prior_fractions.size() - 1) * 0.95)];
        return result;
    }

    struct ScaleStats {
        double regret_l1_total = 0.0;
        double regret_l1_mean = 0.0;
        double regret_l1_median = 0.0;
        double regret_l1_p95 = 0.0;
        double regret_l1_max = 0.0;
        double strategy_mass_total = 0.0;
        double strategy_mass_mean = 0.0;
        double strategy_mass_median = 0.0;
        double strategy_mass_p95 = 0.0;
        double strategy_mass_max = 0.0;
    };

    ScaleStats scale_stats() const {
        ScaleStats result;
        std::vector<double> regrets;
        std::vector<double> masses;
        regrets.reserve(nodes_.size());
        masses.reserve(nodes_.size());
        for (const auto& [_, node] : nodes_) {
            double regret = 0.0;
            double mass = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                regret += std::abs(node.regrets[action]);
                mass += node.strategy_sum[action];
            }
            regrets.push_back(regret);
            masses.push_back(mass);
            result.regret_l1_total += regret;
            result.strategy_mass_total += mass;
        }
        if (regrets.empty()) return result;
        std::sort(regrets.begin(), regrets.end());
        std::sort(masses.begin(), masses.end());
        const auto p95 = [](const std::vector<double>& values) {
            return values[static_cast<size_t>((values.size() - 1) * 0.95)];
        };
        result.regret_l1_mean = result.regret_l1_total / regrets.size();
        result.regret_l1_median = regrets[regrets.size() / 2];
        result.regret_l1_p95 = p95(regrets);
        result.regret_l1_max = regrets.back();
        result.strategy_mass_mean = result.strategy_mass_total / masses.size();
        result.strategy_mass_median = masses[masses.size() / 2];
        result.strategy_mass_p95 = p95(masses);
        result.strategy_mass_max = masses.back();
        return result;
    }

    std::unordered_map<
        InfoKey,
        std::array<double, kActionCount>,
        InfoKeyHash> policy_snapshot() const {
        std::unordered_map<
            InfoKey,
            std::array<double, kActionCount>,
            InfoKeyHash> snapshot;
        snapshot.reserve(nodes_.size());
        for (const auto& [key, node] : nodes_) {
            snapshot.emplace(key, average_strategy(node, key.legal_mask));
        }
        return snapshot;
    }

    std::unordered_map<InfoKey, uint64_t, InfoKeyHash> touch_snapshot() const {
        std::unordered_map<InfoKey, uint64_t, InfoKeyHash> snapshot;
        snapshot.reserve(nodes_.size());
        for (const auto& [key, node] : nodes_) {
            snapshot.emplace(key, node.touches);
        }
        return snapshot;
    }

    struct TemperatureSummary {
        uint64_t calibrated_clusters = 0;
        uint64_t samples = 0;
        double baseline_loss = 0.0;
        double selected_loss = 0.0;
        std::array<uint64_t, kTemperatureCandidates.size()> choices{};
    };

    struct GrowthSummary {
        bool added = false;
        int street = 0;
        uint16_t parent_cluster = 0;
        uint16_t new_cluster = 0;
        uint64_t initialized_nodes = 0;
        double regret = 0.0;
        std::array<double, kActionCount> initial_strategy{};
    };

    struct AdaptiveSummary {
        GrowthSummary growth;
        TemperatureSummary temperature;
        double maximum_average_regret = 0.0;
    };

    void reset_temperature_calibration() {
        for (int street = 0; street < 3; ++street) {
            for (auto& losses : temperature_losses_[street]) losses.fill(0.0);
            std::fill(
                temperature_samples_[street].begin(),
                temperature_samples_[street].end(),
                0);
            growth_candidates_[street].assign(
                growth_candidates_[street].size(), ClusterGrowthCandidate{});
            std::fill(
                temperature_selected_losses_[street].begin(),
                temperature_selected_losses_[street].end(),
                0.0);
            std::fill(
                growth_regret_sums_[street].begin(),
                growth_regret_sums_[street].end(),
                0.0);
            std::fill(
                growth_samples_[street].begin(),
                growth_samples_[street].end(),
                0);
            for (auto& points : residual_points_[street]) points.clear();
            std::fill(
                residual_min_regrets_[street].begin(),
                residual_min_regrets_[street].end(),
                -std::numeric_limits<double>::infinity());
        }
    }

    void calibrate_temperature_root(State state, int traverser) {
        if (!atlas_ || !soft_enabled() || !soft_local_bandwidth_) {
            throw std::runtime_error(
                "temperature calibration requires local soft power buckets");
        }
        calibrate_temperature_traverse(std::move(state), traverser);
    }

    TemperatureSummary apply_temperature_calibration(
        uint64_t minimum_samples,
        int selected_street = -1) {
        TemperatureSummary summary;
        double baseline_total = 0.0;
        double selected_total = 0.0;
        for (int street = 0; street < 3; ++street) {
            if (selected_street >= 0 && street != selected_street) continue;
            for (size_t cluster = 0;
                 cluster < temperature_scales_[street].size();
                 ++cluster) {
                const uint64_t samples = temperature_samples_[street][cluster];
                if (samples < minimum_samples) continue;
                size_t best = 0;
                double best_loss =
                    temperature_losses_[street][cluster][0] / samples;
                for (size_t candidate = 1;
                     candidate < kTemperatureCandidates.size();
                     ++candidate) {
                    const double loss =
                        temperature_losses_[street][cluster][candidate] / samples;
                    if (loss < best_loss) {
                        best = candidate;
                        best_loss = loss;
                    }
                }
                temperature_scales_[street][cluster] = std::clamp(
                    temperature_scales_[street][cluster] *
                        kTemperatureCandidates[best],
                    0.125,
                    8.0);
                temperature_selected_losses_[street][cluster] = best_loss;
                ++summary.calibrated_clusters;
                summary.samples += samples;
                ++summary.choices[best];
                baseline_total +=
                    temperature_losses_[street][cluster][2];
                selected_total +=
                    temperature_losses_[street][cluster][best];
            }
        }
        if (summary.samples) {
            summary.baseline_loss = baseline_total / summary.samples;
            summary.selected_loss = selected_total / summary.samples;
        }
        soft_weight_cache_.clear();
        return summary;
    }

    GrowthSummary append_growth_cluster(
        double threshold,
        uint64_t minimum_samples) {
        if (!atlas_) throw std::runtime_error("cluster growth requires power buckets");

        GrowthSummary result;
        const ClusterGrowthCandidate* selected = nullptr;
        int selected_street = -1;
        uint16_t selected_parent = 0;
        for (int street = 0; street < 3; ++street) {
            for (size_t cluster = 0;
                 cluster < growth_candidates_[street].size();
                 ++cluster) {
                const auto& candidate = growth_candidates_[street][cluster];
                if (!candidate.found ||
                    temperature_samples_[street][cluster] < minimum_samples) {
                    continue;
                }
                const double cluster_loss =
                    temperature_selected_losses_[street][cluster];
                if (selected && cluster_loss <= result.regret) continue;
                const auto strategy =
                    policy(candidate.state, candidate.actor);
                result.regret = cluster_loss;
                selected = &candidate;
                selected_street = street;
                selected_parent = static_cast<uint16_t>(cluster);
                result.initial_strategy = strategy;
            }
        }
        if (!selected || result.regret <= threshold) return result;

        const auto inheritance_weights = atlas_->weights(
            selected->state,
            selected->actor,
            soft_top_k_,
            soft_top_p_,
            soft_temperature_,
            soft_local_bandwidth_,
            &temperature_scales_[selected_street]);
        const auto seed_node = [](const auto& strategy) {
            RegretNode node{};
            for (int action = 0; action < kActionCount; ++action) {
                node.regrets[action] = strategy[action];
                node.raw_regrets[action] = strategy[action];
                node.strategy_sum[action] = strategy[action];
            }
            node.touches = 1;
            return node;
        };
        InfoKey key = make_power_key(
            selected->state,
            selected->actor,
            *atlas_,
            preserve_recall_);
        const uint16_t child =
            atlas_->append(selected->state, selected->actor);
        key.power_cluster = child;

        std::unordered_map<InfoKey, RegretNode, InfoKeyHash> additions;
        for (const auto& [template_key, ignored] : nodes_) {
            if (template_key.street != selected->state.street) continue;
            InfoKey child_key = template_key;
            child_key.power_cluster = child;
            if (additions.find(child_key) != additions.end()) continue;

            std::array<double, kActionCount> mixed{};
            double total = 0.0;
            for (const auto& [neighbor, weight] : inheritance_weights) {
                InfoKey neighbor_key = template_key;
                neighbor_key.power_cluster = neighbor;
                const auto found = nodes_.find(neighbor_key);
                if (found == nodes_.end()) continue;
                const auto local =
                    average_strategy(found->second, child_key.legal_mask);
                for (int action = 0; action < kActionCount; ++action) {
                    mixed[action] += weight * local[action];
                }
                total += weight;
            }
            if (total <= 0.0) continue;
            for (double& probability : mixed) probability /= total;
            additions.emplace(child_key, seed_node(mixed));
        }
        additions.try_emplace(key, seed_node(result.initial_strategy));
        result.initialized_nodes = additions.size();
        for (auto& [child_key, node] : additions) {
            nodes_.emplace(child_key, std::move(node));
        }

        temperature_scales_[selected_street].push_back(1.0);
        temperature_losses_[selected_street].emplace_back();
        temperature_samples_[selected_street].push_back(0);
        temperature_selected_losses_[selected_street].push_back(0.0);
        growth_candidates_[selected_street].emplace_back();
        growth_regret_sums_[selected_street].push_back(0.0);
        growth_samples_[selected_street].push_back(0);
        residual_points_[selected_street].emplace_back();
        residual_min_regrets_[selected_street].push_back(
            -std::numeric_limits<double>::infinity());
        soft_weight_cache_.clear();

        result.added = true;
        result.street = selected_street + 5;
        result.parent_cluster = selected_parent;
        result.new_cluster = child;
        return result;
    }

    AdaptiveSummary adapt_soft_clusters(
        double threshold,
        uint64_t minimum_samples,
        int selected_street = -1) {
        if (soft_growth_mode_ == SoftGrowthMode::None || !atlas_) {
            throw std::runtime_error(
                "adaptive soft clustering requires a growth mode");
        }
        if (selected_street < -1 || selected_street >= 3) {
            throw std::runtime_error("invalid adaptive street");
        }

        AdaptiveSummary summary;
        const ClusterGrowthCandidate* selected = nullptr;
        int growth_street = -1;
        uint16_t selected_parent = 0;
        for (int street = 0; street < 3; ++street) {
            if (selected_street >= 0 && street != selected_street) continue;
            for (size_t cluster = 0;
                 cluster < growth_candidates_[street].size();
                 ++cluster) {
                const uint64_t samples = growth_samples_[street][cluster];
                if (samples < minimum_samples ||
                    !growth_candidates_[street][cluster].found) {
                    continue;
                }
                const double average_regret =
                    growth_regret_sums_[street][cluster] / samples;
                if (selected &&
                    average_regret <= summary.maximum_average_regret) {
                    continue;
                }
                selected = &growth_candidates_[street][cluster];
                growth_street = street;
                selected_parent = static_cast<uint16_t>(cluster);
                summary.maximum_average_regret = average_regret;
            }
        }

        if (!selected || summary.maximum_average_regret <= threshold) {
            if (soft_growth_mode_ == SoftGrowthMode::Point ||
                soft_growth_mode_ == SoftGrowthMode::Residual) {
                return summary;
            }
            summary.temperature =
                apply_temperature_calibration(
                    minimum_samples, selected_street);
            return summary;
        }

        const uint8_t mask = valid_mask(selected->state, selected->actor);
        const auto old_policy = policy(selected->state, selected->actor);
        double old_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                old_value +=
                    old_policy[action] * selected->action_values[action];
            }
        }

        std::array<double, kActionCount> target{};
        if (soft_growth_mode_ == SoftGrowthMode::Mix) {
            double total = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
                target[action] = std::max(
                    0.0, selected->action_values[action] - old_value);
                total += target[action];
            }
            if (total > 0.0) {
                for (double& probability : target) probability /= total;
            }
        }
        if (std::accumulate(target.begin(), target.end(), 0.0) <= 0.0) {
            int best = -1;
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
                if (best < 0 ||
                    selected->action_values[action] >
                        selected->action_values[best]) {
                    best = action;
                }
            }
            target[best] = 1.0;
        }

        const auto inheritance_weights = atlas_->weights(
            selected->state,
            selected->actor,
            soft_top_k_,
            soft_top_p_,
            soft_temperature_,
            soft_local_bandwidth_,
            &temperature_scales_[growth_street]);
        InfoKey selected_key = make_power_key(
            selected->state,
            selected->actor,
            *atlas_,
            preserve_recall_);
        PowerVector child_center =
            atlas_->vector(selected->state, selected->actor);
        if (soft_growth_mode_ == SoftGrowthMode::Residual) {
            const auto& points =
                residual_points_[growth_street][selected_parent];
            PowerVector weighted_center{};
            double total_weight = 0.0;
            for (const ResidualPoint& point : points) {
                const double weight =
                    std::max(0.0, point.regret - threshold);
                if (weight <= 0.0) continue;
                total_weight += weight;
                for (int dimension = 0;
                     dimension < kPowerDimensions;
                     ++dimension) {
                    weighted_center[dimension] +=
                        weight * point.vector[dimension];
                }
            }
            if (total_weight > 0.0) {
                for (double& value : weighted_center) {
                    value /= total_weight;
                }
                child_center = weighted_center;
            }
        }
        const uint16_t child =
            atlas_->append(
                child_center,
                selected->state.street,
                selected_parent,
                kChildMassFraction);
        selected_key.power_cluster = child;

        const auto seed_node = [](const auto& strategy) {
            RegretNode node{};
            for (int action = 0; action < kActionCount; ++action) {
                node.regrets[action] = strategy[action];
                node.raw_regrets[action] = strategy[action];
                node.strategy_sum[action] = strategy[action];
            }
            node.touches = 1;
            return node;
        };
        std::unordered_map<InfoKey, RegretNode, InfoKeyHash> additions;
        for (const auto& [template_key, ignored] : nodes_) {
            if (template_key.street != selected->state.street) continue;
            InfoKey child_key = template_key;
            child_key.power_cluster = child;
            if (additions.find(child_key) != additions.end()) continue;

            std::array<double, kActionCount> inherited{};
            double total = 0.0;
            for (const auto& [neighbor, weight] : inheritance_weights) {
                InfoKey neighbor_key = template_key;
                neighbor_key.power_cluster = neighbor;
                const auto found = nodes_.find(neighbor_key);
                if (found == nodes_.end()) continue;
                const auto local = average_strategy(
                    found->second, child_key.legal_mask);
                for (int action = 0; action < kActionCount; ++action) {
                    inherited[action] += weight * local[action];
                }
                total += weight;
            }
            if (total <= 0.0) continue;
            for (double& probability : inherited) probability /= total;
            additions.emplace(child_key, seed_node(inherited));
        }
        additions[selected_key] = seed_node(target);
        summary.growth.initialized_nodes = additions.size();
        for (auto& [child_key, node] : additions) {
            nodes_.insert_or_assign(child_key, std::move(node));
        }

        temperature_scales_[growth_street].push_back(1.0);
        temperature_losses_[growth_street].emplace_back();
        temperature_samples_[growth_street].push_back(0);
        temperature_selected_losses_[growth_street].push_back(0.0);
        growth_candidates_[growth_street].emplace_back();
        growth_regret_sums_[growth_street].push_back(0.0);
        growth_samples_[growth_street].push_back(0);
        residual_points_[growth_street].emplace_back();
        residual_min_regrets_[growth_street].push_back(
            -std::numeric_limits<double>::infinity());
        soft_weight_cache_.clear();

        summary.growth.added = true;
        summary.growth.street = growth_street + 5;
        summary.growth.parent_cluster = selected_parent;
        summary.growth.new_cluster = child;
        summary.growth.regret = summary.maximum_average_regret;
        summary.growth.initial_strategy = target;
        return summary;
    }

    void save_temperatures(const std::string& path) const {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("cannot write temperatures: " + path);
        }
        output << "TEMPERATURES1\n";
        for (const auto& street : temperature_scales_) {
            output << street.size();
            for (double scale : street) output << ' ' << scale;
            output << '\n';
        }
    }

    void load_temperatures(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot read temperatures: " + path);
        }
        std::string magic;
        input >> magic;
        if (magic != "TEMPERATURES1") {
            throw std::runtime_error("invalid temperature file");
        }
        for (int street = 0; street < 3; ++street) {
            size_t count = 0;
            input >> count;
            if (count != temperature_scales_[street].size()) {
                throw std::runtime_error(
                    "temperature file does not match power atlas");
            }
            for (double& scale : temperature_scales_[street]) {
                input >> scale;
                if (!(scale > 0.0) || !std::isfinite(scale)) {
                    throw std::runtime_error("invalid cluster temperature");
                }
            }
        }
        if (!input) throw std::runtime_error("truncated temperature file");
        soft_weight_cache_.clear();
    }

    InfoKey information_key(const State& state, int viewer) const {
        return bucket_key(state, viewer);
    }

    void diagnose_conflict_root(State state, int traverser) {
        diagnose_conflict_traverse(std::move(state), traverser);
        ++diagnostic_roots_;
    }

    size_t conflict_buckets() const {
        return conflict_nodes_.size();
    }

    size_t observed_seventh_recall_infosets() const {
        return seventh_recall_infosets_.size();
    }

    double average_fixed_hand_seventh_infosets() const {
        return seventh_subgame_samples_
            ? seventh_subgame_infoset_sum_ /
                static_cast<double>(seventh_subgame_samples_)
            : 0.0;
    }

    size_t maximum_fixed_hand_seventh_infosets() const {
        return seventh_subgame_infoset_max_;
    }

    void dump_conflict(
        const std::string& path,
        uint64_t minimum_samples) const {
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("cannot write conflict diagnostics: " + path);
        }
        output
            << "abstraction,street,power_cluster,category,memory_code,"
               "own_public_category,own_public_suit,own_public_connected,"
               "opponent_public_category,opponent_public_suit,"
               "opponent_public_connected,pot_odds_bucket,stack_pot_bucket,"
               "own_bet_count,opponent_bet_count,checked,last_action_class,"
               "betting_goal,legal_mask,history_length,history_code,"
               "samples,reach_proxy,max_conflict,max_cancellation";
        for (const char* action : kActionNames) {
            output << ",abs_" << action << ",signed_" << action
                   << ",conflict_" << action;
        }
        output << '\n';
        for (const auto& [key, node] : conflict_nodes_) {
            if (node.samples < minimum_samples) continue;
            double maximum_conflict = 0.0;
            double maximum_cancellation = 0.0;
            std::array<double, kActionCount> conflicts{};
            for (int action = 0; action < kActionCount; ++action) {
                if (!(key.legal_mask & (1u << action)) ||
                    node.absolute_regret[action] <= 0.0) {
                    continue;
                }
                const double absolute = node.absolute_regret[action];
                conflicts[action] = absolute /
                    (std::abs(node.signed_regret[action]) + 1e-12);
                maximum_conflict = std::max(
                    maximum_conflict, conflicts[action]);
                maximum_cancellation = std::max(
                    maximum_cancellation,
                    1.0 - std::abs(node.signed_regret[action]) / absolute);
            }
            output
                << static_cast<int>(key.abstraction) << ','
                << static_cast<int>(key.street) << ','
                << key.power_cluster << ','
                << static_cast<int>(key.category) << ','
                << static_cast<int>(key.rank_bucket) << ','
                << static_cast<int>(key.own_public_category) << ','
                << static_cast<int>(key.own_public_suit) << ','
                << static_cast<int>(key.own_public_connected) << ','
                << static_cast<int>(key.opponent_public_category) << ','
                << static_cast<int>(key.opponent_public_suit) << ','
                << static_cast<int>(key.opponent_public_connected) << ','
                << static_cast<int>(key.pot_odds_bucket) << ','
                << static_cast<int>(key.stack_pot_bucket) << ','
                << static_cast<int>(key.own_bet_count) << ','
                << static_cast<int>(key.opponent_bet_count) << ','
                << static_cast<int>(key.checked) << ','
                << static_cast<int>(key.last_action_class) << ','
                << static_cast<int>(key.betting_goal) << ','
                << static_cast<int>(key.legal_mask) << ','
                << static_cast<int>(key.history_length) << ','
                << key.history_code << ','
                << node.samples << ','
                << node.samples /
                    static_cast<double>(std::max<uint64_t>(1, diagnostic_roots_))
                << ',' << maximum_conflict
                << ',' << maximum_cancellation;
            for (int action = 0; action < kActionCount; ++action) {
                output << ',' << node.absolute_regret[action]
                       << ',' << node.signed_regret[action]
                       << ',' << conflicts[action];
            }
            output << '\n';
        }
    }

private:
    bool soft_enabled() const {
        return soft_top_p_ > 0.0 || soft_top_k_ > 1;
    }

    void enumerate_seventh_betting_infosets(
        State state,
        std::unordered_map<InfoKey, uint8_t, InfoKeyHash>& infosets) const {
        if (state.terminal) return;
        const int actor = state.actor;
        infosets.emplace(
            make_power_key(state, actor, *atlas_, true), 1);
        const uint8_t mask = valid_mask(state, actor);
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd ||
                result == ActionResult::FoldEnd) {
                continue;
            }
            child.actor = 1 - actor;
            enumerate_seventh_betting_infosets(std::move(child), infosets);
        }
    }

    bool is_seventh_round_root(const State& state) const {
        return state.street == 7 &&
            std::none_of(
                state.history.begin(),
                state.history.end(),
                [](const Event& event) { return event.street == 7; });
    }

    double diagnose_conflict_traverse(State state, int traverser) {
        if (state.terminal) return terminal_net_search(state, traverser);
        if (atlas_ && is_seventh_round_root(state) &&
            seventh_subgame_samples_ < 100) {
            std::unordered_map<InfoKey, uint8_t, InfoKeyHash> infosets;
            enumerate_seventh_betting_infosets(state, infosets);
            ++seventh_subgame_samples_;
            seventh_subgame_infoset_sum_ += infosets.size();
            seventh_subgame_infoset_max_ = std::max(
                seventh_subgame_infoset_max_, infosets.size());
        }
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = bucket_key(state, actor);
        if (state.street == 7 && atlas_) {
            ++seventh_recall_infosets_[
                make_power_key(state, actor, *atlas_, true)];
        }
        const auto strategy = policy(state, actor);
        if (actor != traverser) {
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return diagnose_conflict_traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] =
                diagnose_conflict_traverse(std::move(child), traverser);
        }
        double node_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                node_value += strategy[action] * action_values[action];
            }
        }
        ConflictNode& diagnostic = conflict_nodes_[key];
        ++diagnostic.samples;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            const double regret = action_values[action] - node_value;
            diagnostic.absolute_regret[action] += std::abs(regret);
            diagnostic.signed_regret[action] += regret;
        }
        return node_value;
    }

    State determinize(const State& actual, int viewer) {
        State state = actual;
        std::array<bool, 52> known{};
        auto mark = [&](const Card& card) {
            known[card.suit * 13 + card.rank - 2] = true;
        };
        for (const Card& card : actual.players[viewer].hidden) mark(card);
        for (const Card& card : actual.players[viewer].shown) mark(card);
        if (actual.players[viewer].has_discard) mark(actual.players[viewer].discarded);
        for (const Card& card : actual.players[1 - viewer].shown) mark(card);
        std::vector<Card> remaining;
        for (const Card& card : fresh_deck()) {
            if (!known[card.suit * 13 + card.rank - 2]) remaining.push_back(card);
        }
        std::shuffle(remaining.begin(), remaining.end(), rng_);
        Player& opponent = state.players[1 - viewer];
        opponent.hidden.clear();
        opponent.discarded = remaining.back();
        opponent.has_discard = true;
        remaining.pop_back();
        const int hidden_count = actual.street == 7 ? 3 : 2;
        for (int i = 0; i < hidden_count; ++i) {
            opponent.hidden.push_back(remaining.back());
            remaining.pop_back();
        }
        state.simulation_deck = std::move(remaining);
        state.actor = viewer;
        state.terminal = false;
        return state;
    }

    void advance_street(State& state) {
        if (state.street == 7) {
            state.terminal = true;
            return;
        }
        ++state.street;
        const bool public_card = state.street != 7;
        for (int seat = 0; seat < 2; ++seat) {
            if (state.simulation_deck.empty()) throw std::runtime_error("simulation deck exhausted");
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
    }

    std::array<double, kActionCount> policy_with_temperature_override(
        const State& state,
        int viewer,
        int cluster,
        double scale) const {
        const uint8_t mask = valid_mask(state, viewer);
        std::array<double, kActionCount> mixed{};
        double total = 0.0;
        InfoKey key =
            make_power_key(state, viewer, *atlas_, preserve_recall_);
        const auto weights = atlas_->weights(
            state,
            viewer,
            soft_top_k_,
            soft_top_p_,
            soft_temperature_,
            true,
            &temperature_scales_[state.street - 5],
            cluster,
            scale);
        for (const auto& [neighbor, weight] : weights) {
            key.power_cluster = neighbor;
            const auto it = nodes_.find(key);
            if (it == nodes_.end()) continue;
            const auto local = average_strategy(it->second, mask);
            for (int action = 0; action < kActionCount; ++action) {
                mixed[action] += weight * local[action];
            }
            total += weight;
        }
        if (total <= 0.0) return uniform_strategy(mask);
        for (double& probability : mixed) probability /= total;
        return mixed;
    }

    double calibrate_temperature_traverse(State state, int traverser) {
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const auto strategy = policy(state, actor);
        if (actor != traverser) {
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return calibrate_temperature_traverse(
                std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        double best_value = -std::numeric_limits<double>::infinity();
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] = calibrate_temperature_traverse(
                std::move(child), traverser);
            best_value = std::max(best_value, action_values[action]);
        }

        const int street = state.street - 5;
        const uint16_t cluster = atlas_->assign(state, actor);
        for (size_t candidate = 0;
             candidate < kTemperatureCandidates.size();
             ++candidate) {
            const auto candidate_policy = policy_with_temperature_override(
                state,
                actor,
                cluster,
                kTemperatureCandidates[candidate]);
            double value = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    value += candidate_policy[action] * action_values[action];
                }
            }
            temperature_losses_[street][cluster][candidate] +=
                best_value - value;
        }
        ++temperature_samples_[street][cluster];

        double node_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                node_value += strategy[action] * action_values[action];
            }
        }
        auto& growth = growth_candidates_[street][cluster];
        const double regret = best_value - node_value;
        if (!growth.found || regret > growth.regret) {
            growth.found = true;
            growth.state = state;
            growth.actor = actor;
            growth.regret = regret;
            growth.best_value = best_value;
            growth.action_values = action_values;
        }
        return node_value;
    }

    double soft_traverse(State state, int traverser) {
        ++node_visits_;
        if (state.terminal) return terminal_net_search(state, traverser);

        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const int street = state.street - 5;
        const auto weights = atlas_->weights(
            state,
            actor,
            soft_top_k_,
            soft_top_p_,
            soft_temperature_,
            soft_local_bandwidth_,
            &temperature_scales_[street]);
        InfoKey base =
            make_power_key(state, actor, *atlas_, preserve_recall_);
        std::vector<InfoKey> keys;
        std::vector<std::array<double, kActionCount>> local_strategies;
        keys.reserve(weights.size());
        local_strategies.reserve(weights.size());
        std::array<double, kActionCount> strategy{};
        const Action initial_action = initial_regret_action(state, actor, mask);
        for (const auto& [cluster, weight] : weights) {
            InfoKey key = base;
            key.power_cluster = cluster;
            const auto local = current_strategy(
                get_node(key, initial_action), mask);
            keys.push_back(key);
            local_strategies.push_back(local);
            for (int action = 0; action < kActionCount; ++action) {
                strategy[action] += weight * local[action];
            }
        }

        if (actor != traverser) {
            for (size_t index = 0; index < keys.size(); ++index) {
                RegretNode& node = nodes_.find(keys[index])->second;
                for (int action = 0; action < kActionCount; ++action) {
                    if (mask & (1u << action)) {
                        node.strategy_sum[action] +=
                            weights[index].second *
                            local_strategies[index][action];
                    }
                }
            }
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return soft_traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        double best_value = -std::numeric_limits<double>::infinity();
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] =
                soft_traverse(std::move(child), traverser);
            best_value = std::max(best_value, action_values[action]);
        }

        double node_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                node_value += strategy[action] * action_values[action];
            }
        }
        for (size_t index = 0; index < keys.size(); ++index) {
            double local_value = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    local_value += local_strategies[index][action] *
                        action_values[action];
                }
            }
            RegretNode& node = nodes_.find(keys[index])->second;
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
                const double increment =
                    weights[index].second *
                    (action_values[action] - local_value);
                node.raw_regrets[action] += increment;
                const double regret = node.regrets[action] + increment;
                node.regrets[action] =
                    regret_plus_ ? std::max(0.0, regret) : regret;
            }
        }

        if (soft_growth_mode_ == SoftGrowthMode::Fixed) {
            return node_value;
        }
        const uint16_t dominant_cluster = weights.front().first;
        ++growth_samples_[street][dominant_cluster];
        const bool tune_temperature =
            soft_growth_mode_ == SoftGrowthMode::Mix ||
            soft_growth_mode_ == SoftGrowthMode::Simple;
        if (tune_temperature &&
            growth_samples_[street][dominant_cluster] %
                kSoftTemperatureStride == 0) {
            for (size_t candidate = 0;
                 candidate < kTemperatureCandidates.size();
                 ++candidate) {
                const auto candidate_policy =
                    policy_with_temperature_override(
                        state,
                        actor,
                        dominant_cluster,
                        kTemperatureCandidates[candidate]);
                double value = 0.0;
                for (int action = 0; action < kActionCount; ++action) {
                    if (mask & (1u << action)) {
                        value += candidate_policy[action] *
                            action_values[action];
                    }
                }
                temperature_losses_[street][dominant_cluster][candidate] +=
                    best_value - value;
            }
            ++temperature_samples_[street][dominant_cluster];
        }
        const double regret = best_value - node_value;
        growth_regret_sums_[street][dominant_cluster] += regret;
        if (soft_growth_mode_ == SoftGrowthMode::Residual) {
            auto& points = residual_points_[street][dominant_cluster];
            if (points.size() < kResidualPointCap) {
                points.push_back(ResidualPoint{
                    atlas_->vector(state, actor),
                    regret,
                });
                if (points.size() == kResidualPointCap) {
                    residual_min_regrets_[street][dominant_cluster] =
                        std::min_element(
                            points.begin(),
                            points.end(),
                            [](const auto& left, const auto& right) {
                                return left.regret < right.regret;
                            })->regret;
                }
            } else if (
                regret >
                residual_min_regrets_[street][dominant_cluster]) {
                auto smallest = std::min_element(
                    points.begin(),
                    points.end(),
                    [](const auto& left, const auto& right) {
                        return left.regret < right.regret;
                    });
                if (regret > smallest->regret) {
                    *smallest = ResidualPoint{
                        atlas_->vector(state, actor),
                        regret,
                    };
                    residual_min_regrets_[street][dominant_cluster] =
                        std::min_element(
                            points.begin(),
                            points.end(),
                            [](const auto& left, const auto& right) {
                                return left.regret < right.regret;
                            })->regret;
                }
            }
        }
        auto& growth = growth_candidates_[street][dominant_cluster];
        if (!growth.found || regret > growth.regret) {
            growth.found = true;
            growth.state = state;
            growth.actor = actor;
            growth.regret = regret;
            growth.best_value = best_value;
            growth.action_values = action_values;
        }
        return node_value;
    }

    void maybe_queue_split_audit(
        const State& state,
        int viewer,
        const InfoKey& key) {
        if (!split_audit_config_.enabled || state.street < 5 ||
            state.street > 7) {
            return;
        }
        if (atlas_->clusters(state.street) >=
            split_audit_config_.max_leaves) {
            ++split_audit_stats_.capacity_deferred;
            return;
        }
        if (atlas_->leaf_depth(state.street, key.power_cluster) >=
            split_audit_config_.max_depth) {
            ++split_audit_stats_.depth_deferred;
            return;
        }
        if (atlas_->record_leaf_touch(state.street, key.power_cluster) <
            split_audit_config_.min_touches) {
            ++split_audit_stats_.touch_deferred;
            return;
        }
        if (std::uniform_real_distribution<double>(0.0, 1.0)(audit_rng_) >=
            split_audit_config_.probability) {
            return;
        }

        PendingSplitAudit pending{
            state,
            viewer,
            SplitAuditKey{
                static_cast<uint8_t>(state.street),
                key.legal_mask,
                key.power_cluster}};
        ++pending_split_audits_seen_;
        ++split_audit_stats_.queued_states;
        if (pending_split_audits_.size() < split_audit_config_.reservoir) {
            pending_split_audits_.push_back(std::move(pending));
            return;
        }
        const uint64_t selected = std::uniform_int_distribution<uint64_t>(
            0, pending_split_audits_seen_ - 1)(audit_rng_);
        if (selected < pending_split_audits_.size()) {
            pending_split_audits_[selected] = std::move(pending);
        }
    }

    Action sample_audit_policy(
        const std::array<double, kActionCount>& strategy,
        uint8_t mask) {
        const double threshold =
            std::uniform_real_distribution<double>(0.0, 1.0)(audit_rng_);
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

    double split_audit_rollout(
        State state,
        int viewer,
        double payoff_scale_chips) {
        ++node_visits_;
        ++split_audit_stats_.audit_node_visits;
        if (state.terminal) {
            return normalized_split_audit_terminal(
                state, viewer, payoff_scale_chips);
        }
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const Action action = sample_audit_policy(
            policy(state, actor), mask);
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd) advance_street(state);
        else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
        return split_audit_rollout(
            std::move(state), viewer, payoff_scale_chips);
    }

    void record_split_audit_sample(
        const SplitAuditKey& key,
        const LocalSplitSample& sample,
        bool strong) {
        SplitAuditEvidence& evidence = split_audit_evidence_[key];
        if (evidence.samples.size() == split_audit_config_.reservoir) {
            evidence.samples.erase(evidence.samples.begin());
            evidence.strong.erase(evidence.strong.begin());
        }
        evidence.samples.push_back(sample);
        evidence.strong.push_back(strong ? 1 : 0);
        ++split_audit_stats_.audits;
        split_audit_stats_.strong_audits += strong;
        const size_t strong_count = static_cast<size_t>(std::count(
            evidence.strong.begin(), evidence.strong.end(),
            static_cast<uint8_t>(1)));
        if (strong_count >= split_audit_config_.min_strong &&
            !evidence.queued) {
            evidence.queued = true;
            split_candidates_.push_back(key);
        }
    }

    void evaluate_split_audit(const PendingSplitAudit& pending) {
        const uint8_t mask = pending.key.legal_mask;
        const std::vector<Action> legal = actions_from_mask(mask);
        if (legal.size() < 2) return;

        constexpr double kIllegalReward = -1e6;
        std::vector<std::array<double, kActionCount>> returns(
            split_audit_config_.rollouts);
        for (auto& reward : returns) reward.fill(kIllegalReward);
        const double payoff_scale_chips = std::max(
            pending.state.ante, pending.state.effective_stack);

        for (int rollout = 0;
             rollout < split_audit_config_.rollouts;
             ++rollout) {
            State chance_state = pending.state;
            std::shuffle(
                chance_state.simulation_deck.begin(),
                chance_state.simulation_deck.end(),
                audit_rng_);
            for (Action action : legal) {
                State child = chance_state;
                const ActionResult result = apply_action(
                    child, pending.viewer, action);
                if (result == ActionResult::RoundEnd) advance_street(child);
                else if (result != ActionResult::FoldEnd) {
                    child.actor = 1 - pending.viewer;
                }
                returns[rollout][action] = split_audit_rollout(
                    std::move(child), pending.viewer, payoff_scale_chips);
            }
        }

        LocalSplitSample sample;
        sample.feature = atlas_->vector(pending.state, pending.viewer);
        sample.reward.fill(kIllegalReward);
        for (Action action : legal) {
            for (const auto& reward : returns) {
                sample.reward[action] += reward[action];
            }
            sample.reward[action] =
                (sample.reward[action] - kIllegalReward) /
                split_audit_config_.rollouts;
        }
        Action best = legal.front();
        for (Action action : legal) {
            if (sample.reward[action] > sample.reward[best]) best = action;
        }

        const auto current = policy(pending.state, pending.viewer);
        double baseline = 0.0;
        for (Action action : legal) {
            baseline += current[action] * sample.reward[action];
        }
        for (Action action : legal) {
            sample.reward[action] -= baseline;
        }
        std::vector<double> gaps;
        gaps.reserve(returns.size());
        double gap_mean = 0.0;
        for (const auto& reward : returns) {
            double current_return = 0.0;
            for (Action action : legal) {
                current_return += current[action] * reward[action];
            }
            const double gap = reward[best] - current_return;
            gaps.push_back(gap);
            gap_mean += gap;
        }
        gap_mean /= gaps.size();
        double squared = 0.0;
        for (double gap : gaps) {
            const double delta = gap - gap_mean;
            squared += delta * delta;
        }
        const double standard_deviation = gaps.size() > 1
            ? std::sqrt(squared / (gaps.size() - 1))
            : 0.0;
        const double lower_bound = gap_mean -
            split_audit_config_.confidence_z * standard_deviation /
                std::sqrt(static_cast<double>(gaps.size()));
        const uint64_t touches = atlas_->leaf_touches_since_split(
            pending.key.street, pending.key.leaf);
        const double reference = static_cast<double>(
            split_audit_config_.gap_touch_reference);
        const double dynamic_threshold = std::max(
            split_audit_config_.gap_threshold_min,
            split_audit_config_.gap_threshold * std::sqrt(
                reference /
                std::max(reference, static_cast<double>(touches))));
        ++split_audit_stats_.threshold_evaluations;
        split_audit_stats_.threshold_sum += dynamic_threshold;
        split_audit_stats_.threshold_min = std::min(
            split_audit_stats_.threshold_min, dynamic_threshold);
        split_audit_stats_.threshold_max = std::max(
            split_audit_stats_.threshold_max, dynamic_threshold);
        record_split_audit_sample(
            pending.key,
            sample,
            lower_bound > dynamic_threshold);
    }

    void clear_split_evidence(int street, uint16_t leaf) {
        for (auto it = split_audit_evidence_.begin();
             it != split_audit_evidence_.end();) {
            if (it->first.street == street && it->first.leaf == leaf) {
                it = split_audit_evidence_.erase(it);
            } else {
                ++it;
            }
        }
        split_candidates_.erase(
            std::remove_if(
                split_candidates_.begin(),
                split_candidates_.end(),
                [&](const SplitAuditKey& key) {
                    return key.street == street && key.leaf == leaf;
                }),
            split_candidates_.end());
    }

    double traverse(State state, int traverser, std::array<double, 2> reach) {
        ++node_visits_;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = bucket_key(state, actor);
        RegretNode& node = get_node(
            key, initial_regret_action(state, actor, mask));
        maybe_queue_split_audit(state, actor, key);
        auto strategy = current_strategy(node, mask);
        const auto prior = imitation_priors_.find(key);
        if (prior != imitation_priors_.end()) {
            const double weight = prior->second.strength /
                (prior->second.strength + node.touches);
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
                strategy[action] =
                    (1.0 - weight) * strategy[action] +
                    weight * prior->second.policy[action];
            }
        }

        if (actor != traverser) {
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    node.strategy_sum[action] += strategy[action];
                }
            }
            const Action action = sample(strategy, mask);
            reach[actor] *= strategy[action];
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return traverse(std::move(state), traverser, reach);
        }

        std::array<double, kActionCount> action_values{};
        std::array<bool, kActionCount> evaluated{};
        const bool refresh =
            prune_after_ == 0 ||
            node.touches < prune_after_ ||
            (prune_refresh_ > 0 && node.touches % prune_refresh_ == 0);
        for (Action action : actions_from_mask(mask)) {
            const bool prune =
                !refresh &&
                strategy[action] == 0.0 &&
                node.raw_regrets[action] <= -prune_threshold_;
            if (prune) {
                ++pruned_branches_;
                continue;
            }
            evaluated[action] = true;
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            auto next_reach = reach;
            next_reach[actor] *= strategy[action];
            action_values[action] = traverse(std::move(child), traverser, next_reach);
        }
        double node_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) node_value += strategy[action] * action_values[action];
        }
        RegretNode& updated = nodes_.find(key)->second;
        for (int action = 0; action < kActionCount; ++action) {
            if (!evaluated[action]) continue;
            const double increment = action_values[action] - node_value;
            updated.raw_regrets[action] += increment;
            const double regret = updated.regrets[action] + increment;
            updated.regrets[action] = regret_plus_ ? std::max(0.0, regret) : regret;
        }
        return node_value;
    }

    InfoKey bucket_key(const State& state, int viewer) const {
        return atlas_
            ? make_power_key(
                state, viewer, *atlas_, preserve_recall_, range_model_,
                betting_memory_)
            : make_key(state, viewer);
    }

    uint8_t abstraction_id() const {
        if (!atlas_) return 0;
        if (atlas_->tree_enabled()) return 8;
        if (range_model_) return 4;
        if (betting_memory_ == 3) return 7;
        if (betting_memory_ == 2) return 6;
        if (betting_memory_ == 1) return 5;
        return preserve_recall_ ? 3 : 1;
    }

    Action initial_regret_action(
        const State& state,
        int actor,
        uint8_t legal_mask) const {
        if (initial_regret_policy_ == InitialRegretPolicy::MadeCall &&
            best_hand(all_cards(state.players[actor]))[0] >=
                initial_made_min_category_) {
            if (legal_mask & (1u << CHECK)) return CHECK;
            if (legal_mask & (1u << CALL)) return CALL;
        }
        return FOLD;
    }

    RegretNode& get_node(const InfoKey& key, Action initial_action = FOLD) {
        auto [it, inserted] = nodes_.try_emplace(key);
        if (inserted) {
            ++misses_;
            initialize_regrets(
                it->second,
                key.legal_mask,
                initial_action,
                initial_fold_regret_,
                initial_other_regret_);
        }
        else ++hits_;
        ++it->second.touches;
        return it->second;
    }

    Action sample(const std::array<double, kActionCount>& strategy, uint8_t mask) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double threshold = uniform(rng_);
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

    bool regret_plus_;
    int start_street_;
    PowerAtlas* atlas_;
    uint64_t prune_after_;
    double prune_threshold_;
    uint64_t prune_refresh_;
    int soft_top_k_;
    double soft_top_p_;
    double soft_temperature_;
    bool soft_local_bandwidth_;
    bool preserve_recall_;
    uint8_t betting_memory_ = 0;
    const ActionRangeModel* range_model_ = nullptr;
    SoftGrowthMode soft_growth_mode_;
    double initial_fold_regret_;
    double initial_other_regret_;
    InitialRegretPolicy initial_regret_policy_;
    int initial_made_min_category_;
    std::mt19937_64 rng_;
    std::mt19937_64 audit_rng_;
    std::unordered_map<InfoKey, RegretNode, InfoKeyHash> nodes_;
    std::unordered_map<InfoKey, uint64_t, InfoKeyHash> imitation_counts_;
    struct ImitationPrior {
        std::array<double, kActionCount> policy{};
        double strength = 0.0;
    };
    bool decaying_imitation_prior_ = false;
    std::unordered_map<InfoKey, ImitationPrior, InfoKeyHash>
        imitation_priors_;
    mutable std::unordered_map<
        PowerObservationKey,
        std::vector<std::pair<uint16_t, double>>,
        PowerObservationHash> soft_weight_cache_;
    std::array<std::vector<double>, 3> temperature_scales_;
    std::array<
        std::vector<std::array<double, kTemperatureCandidates.size()>>,
        3> temperature_losses_;
    std::array<std::vector<uint64_t>, 3> temperature_samples_;
    std::array<std::vector<double>, 3> temperature_selected_losses_;
    std::array<std::vector<ClusterGrowthCandidate>, 3> growth_candidates_;
    std::array<std::vector<double>, 3> growth_regret_sums_;
    std::array<std::vector<uint64_t>, 3> growth_samples_;
    std::array<std::vector<std::vector<ResidualPoint>>, 3> residual_points_;
    std::array<std::vector<double>, 3> residual_min_regrets_;
    std::unordered_map<InfoKey, ConflictNode, InfoKeyHash> conflict_nodes_;
    std::unordered_map<InfoKey, uint64_t, InfoKeyHash>
        seventh_recall_infosets_;
    uint64_t diagnostic_roots_ = 0;
    uint64_t seventh_subgame_samples_ = 0;
    uint64_t seventh_subgame_infoset_sum_ = 0;
    size_t seventh_subgame_infoset_max_ = 0;
    uint64_t traversals_ = 0;
    uint64_t node_visits_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t pruned_branches_ = 0;
    SplitAuditConfig split_audit_config_{};
    SplitAuditStats split_audit_stats_{};
    std::vector<PendingSplitAudit> pending_split_audits_;
    uint64_t pending_split_audits_seen_ = 0;
    std::unordered_map<
        SplitAuditKey,
        SplitAuditEvidence,
        SplitAuditKeyHash> split_audit_evidence_;
    std::vector<SplitAuditKey> split_candidates_;

    friend void self_test();
};

constexpr size_t kExactSeventhHistorySlots = 32;

struct ExactSeventhKey {
    uint8_t viewer = 0;
    uint8_t legal_mask = 0;
    uint8_t own_hidden_count = 0;
    uint8_t own_shown_count = 0;
    uint8_t opponent_shown_count = 0;
    uint8_t history_length = 0;
    uint8_t discarded = 0;
    std::array<uint8_t, 3> own_hidden{};
    std::array<uint8_t, 4> own_shown{};
    std::array<uint8_t, 4> opponent_shown{};
    std::array<uint8_t, kExactSeventhHistorySlots> history{};

    bool operator==(const ExactSeventhKey& other) const {
        return viewer == other.viewer &&
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

struct ExactSeventhKeyHash {
    size_t operator()(const ExactSeventhKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash = (hash ^ value) * 1099511628211ull;
        };
        mix(key.viewer);
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

uint8_t card_token(const Card& card) {
    return static_cast<uint8_t>(1 + card.suit * 13 + card.rank - 2);
}

template <size_t Capacity>
void encode_exact_cards(
    const std::vector<Card>& cards,
    bool preserve_history,
    std::array<uint8_t, Capacity>& output,
    uint8_t& count) {
    if (cards.size() > Capacity) {
        throw std::runtime_error("exact seventh card history exceeds capacity");
    }
    count = static_cast<uint8_t>(cards.size());
    std::vector<uint8_t> encoded;
    encoded.reserve(cards.size());
    for (const Card& card : cards) encoded.push_back(card_token(card));
    if (!preserve_history) {
        std::sort(encoded.begin(), encoded.end());
    }
    std::copy(encoded.begin(), encoded.end(), output.begin());
}

ExactSeventhKey make_exact_seventh_key(
    const State& state,
    int viewer,
    bool preserve_hand_history) {
    if (state.street != 7) {
        throw std::runtime_error("exact seventh key requires 7th street");
    }
    ExactSeventhKey key;
    key.viewer = static_cast<uint8_t>(viewer);
    key.legal_mask = valid_mask(state, viewer);
    encode_exact_cards(
        state.players[viewer].hidden,
        preserve_hand_history,
        key.own_hidden,
        key.own_hidden_count);
    encode_exact_cards(
        state.players[viewer].shown,
        preserve_hand_history,
        key.own_shown,
        key.own_shown_count);
    encode_exact_cards(
        state.players[1 - viewer].shown,
        preserve_hand_history,
        key.opponent_shown,
        key.opponent_shown_count);
    if (state.players[viewer].has_discard) {
        key.discarded = card_token(state.players[viewer].discarded);
    }
    if (state.history.size() > key.history.size()) {
        throw std::runtime_error("exact seventh betting history exceeds capacity");
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

class ExactSeventhResolverPolicy {
public:
    struct Stats {
        uint64_t subgames = 0;
        uint64_t traversals = 0;
        uint64_t node_visits = 0;
        uint64_t nodes_created = 0;
        uint64_t blueprint_fallbacks = 0;
        uint64_t posterior_builds = 0;
        uint64_t posterior_proposals = 0;
        uint64_t posterior_policy_queries = 0;
        uint64_t posterior_policy_misses = 0;
        uint64_t posterior_fallbacks = 0;
        double posterior_effective_particles_sum = 0.0;
        size_t maximum_local_nodes = 0;
    };

    ExactSeventhResolverPolicy(
        MCCFR& blueprint,
        int iterations,
        double prior_strength,
        bool preserve_hand_history,
        uint64_t seed,
        int posterior_particles = 0)
        : blueprint_(blueprint),
          iterations_(iterations),
          prior_strength_(prior_strength),
          preserve_hand_history_(preserve_hand_history),
          posterior_particle_count_(posterior_particles),
          rng_(seed) {
        if (posterior_particles < 0) {
            throw std::runtime_error("posterior particle count must be nonnegative");
        }
        nodes_.reserve(4096);
        posterior_.reserve(posterior_particles);
    }

    void begin_subgame(const State& root, int target_seat) {
        if (root.street != 7 || root.terminal) return;
        nodes_.clear();
        build_posterior(root, target_seat);
        ++stats_.subgames;
        for (int iteration = 0; iteration < iterations_; ++iteration) {
            State simulation = determinize(root, target_seat);
            const int traverser = iteration % 2 == 0
                ? target_seat
                : 1 - target_seat;
            traverse(std::move(simulation), traverser);
            ++stats_.traversals;
        }
        stats_.maximum_local_nodes = std::max(
            stats_.maximum_local_nodes, nodes_.size());
    }

    Action choose(const State& state, int viewer, int) {
        if (state.street != 7) return blueprint_.choose(state, viewer, 0);
        // Stdio callers submit independent current states, so a previous hand's
        // local tree must never leak into this decision.
        begin_subgame(state, viewer);
        return sample(policy(state, viewer), valid_mask(state, viewer));
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        if (state.street == 7) {
            const auto node = nodes_.find(make_exact_seventh_key(
                state, viewer, preserve_hand_history_));
            if (node != nodes_.end()) {
                if (found) *found = true;
                return average_strategy(node->second, valid_mask(state, viewer));
            }
            ++stats_.blueprint_fallbacks;
        }
        return blueprint_.policy(state, viewer, found);
    }

    Stats stats() const { return stats_; }
    int iterations() const { return iterations_; }
    double prior_strength() const { return prior_strength_; }
    bool preserve_hand_history() const { return preserve_hand_history_; }
    int posterior_particles() const { return posterior_particle_count_; }

private:
    struct PosteriorParticle {
        std::array<Card, 3> hidden{};
        Card discarded{};
        double log_weight = 0.0;
        double cumulative_weight = 0.0;
    };

    RegretNode& node(const State& state, int actor) {
        const ExactSeventhKey key = make_exact_seventh_key(
            state, actor, preserve_hand_history_);
        auto [position, inserted] = nodes_.try_emplace(key);
        if (inserted) {
            const auto prior = blueprint_.policy(state, actor);
            for (int action = 0; action < kActionCount; ++action) {
                if (!(key.legal_mask & (1u << action))) continue;
                position->second.regrets[action] =
                    prior_strength_ * prior[action];
                position->second.raw_regrets[action] =
                    prior_strength_ * prior[action];
                position->second.strategy_sum[action] =
                    prior_strength_ * prior[action];
            }
            ++stats_.nodes_created;
        }
        ++position->second.touches;
        return position->second;
    }

    double traverse(State state, int traverser) {
        ++stats_.node_visits;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const ExactSeventhKey key = make_exact_seventh_key(
            state, actor, preserve_hand_history_);
        RegretNode& current = node(state, actor);
        const auto strategy = current_strategy(current, mask);
        if (actor != traverser) {
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    current.strategy_sum[action] += strategy[action];
                }
            }
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) state.terminal = true;
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) child.terminal = true;
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] = traverse(std::move(child), traverser);
        }
        double value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                value += strategy[action] * action_values[action];
            }
        }
        RegretNode& updated = nodes_.find(key)->second;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            const double increment = action_values[action] - value;
            updated.raw_regrets[action] += increment;
            updated.regrets[action] = std::max(
                0.0, updated.regrets[action] + increment);
        }
        return value;
    }

    double history_log_likelihood(
        const State& actual,
        int viewer,
        const PosteriorParticle& particle) {
        if (actual.players[viewer].hidden.size() < 3 ||
            actual.players[viewer].shown.size() < 4 ||
            actual.players[1 - viewer].shown.size() < 4) {
            return -std::numeric_limits<double>::infinity();
        }
        const int opponent = 1 - viewer;
        State model;
        model.ante = actual.ante;
        model.effective_stack = actual.effective_stack;
        model.pot = actual.ante * 2;
        model.street = 5;
        for (int seat = 0; seat < 2; ++seat) {
            model.players[seat].stack_cap = actual.players[seat].stack_cap;
            model.players[seat].invested = actual.ante;
            model.players[seat].shown.assign(
                actual.players[seat].shown.begin(),
                actual.players[seat].shown.begin() + 3);
        }
        model.players[viewer].hidden.assign(
            actual.players[viewer].hidden.begin(),
            actual.players[viewer].hidden.begin() + 2);
        model.players[viewer].discarded = actual.players[viewer].discarded;
        model.players[viewer].has_discard =
            actual.players[viewer].has_discard;
        model.players[opponent].hidden.assign(
            particle.hidden.begin(), particle.hidden.begin() + 2);
        model.players[opponent].discarded = particle.discarded;
        model.players[opponent].has_discard = true;
        model.actor = first_bettor(model);

        double result = 0.0;
        for (const Event& event : actual.history) {
            while (model.street < event.street) {
                ++model.street;
                model.highest_bet = 0;
                model.raise_count = 0;
                for (Player& player : model.players) player.round_bet = 0;
                if (model.street == 6) {
                    for (int seat = 0; seat < 2; ++seat) {
                        model.players[seat].shown.push_back(
                            actual.players[seat].shown[3]);
                    }
                } else if (model.street == 7) {
                    model.players[viewer].hidden.push_back(
                        actual.players[viewer].hidden[2]);
                    model.players[opponent].hidden.push_back(
                        particle.hidden[2]);
                }
                model.actor = first_bettor(model);
            }
            if (event.street != model.street || event.actor != model.actor) {
                return -std::numeric_limits<double>::infinity();
            }
            if (event.actor == opponent) {
                bool found = false;
                const auto strategy = blueprint_.policy(
                    model, opponent, &found);
                ++stats_.posterior_policy_queries;
                stats_.posterior_policy_misses += !found;
                result += std::log(std::max(
                    1e-12, strategy[event.action]));
            }
            const ActionResult action_result = apply_action(
                model, event.actor, event.action);
            if (action_result == ActionResult::FoldEnd) {
                return -std::numeric_limits<double>::infinity();
            }
            if (action_result != ActionResult::RoundEnd) {
                model.actor = 1 - event.actor;
            }
        }
        return result;
    }

    void build_posterior(const State& actual, int viewer) {
        posterior_.clear();
        posterior_total_weight_ = 0.0;
        if (posterior_particle_count_ == 0) return;
        ++stats_.posterior_builds;
        if (actual.players[1 - viewer].shown.empty()) {
            ++stats_.posterior_fallbacks;
            return;
        }

        std::array<bool, 52> known{};
        const auto mark = [&](const Card& card) {
            known[card.suit * 13 + card.rank - 2] = true;
        };
        for (const Card& card : actual.players[viewer].hidden) mark(card);
        for (const Card& card : actual.players[viewer].shown) mark(card);
        if (actual.players[viewer].has_discard) {
            mark(actual.players[viewer].discarded);
        }
        for (const Card& card : actual.players[1 - viewer].shown) mark(card);
        std::vector<Card> remaining;
        for (const Card& card : fresh_deck()) {
            if (!known[card.suit * 13 + card.rank - 2]) {
                remaining.push_back(card);
            }
        }
        if (remaining.size() < 4) {
            ++stats_.posterior_fallbacks;
            return;
        }

        const Card observed_reveal =
            actual.players[1 - viewer].shown.front();
        const int proposal_limit = std::max(
            posterior_particle_count_ * 64,
            posterior_particle_count_ + 64);
        std::uniform_int_distribution<int> pick(
            0, static_cast<int>(remaining.size()) - 1);
        for (int proposal = 0;
             proposal < proposal_limit &&
                 static_cast<int>(posterior_.size()) <
                     posterior_particle_count_;
             ++proposal) {
            ++stats_.posterior_proposals;
            std::array<int, 4> indices{};
            for (int index = 0; index < 4; ++index) {
                bool duplicate;
                do {
                    indices[index] = pick(rng_);
                    duplicate = false;
                    for (int previous = 0; previous < index; ++previous) {
                        duplicate |= indices[previous] == indices[index];
                    }
                } while (duplicate);
            }
            std::vector<Card> initial{
                observed_reveal,
                remaining[indices[0]],
                remaining[indices[1]],
                remaining[indices[2]]};
            std::shuffle(initial.begin(), initial.end(), rng_);
            const auto [discard, reveal] = discard_reveal(initial);
            if (!(initial[reveal] == observed_reveal)) continue;

            PosteriorParticle particle;
            particle.discarded = initial[discard];
            int hidden = 0;
            for (int index = 0; index < 4; ++index) {
                if (index != discard && index != reveal) {
                    particle.hidden[hidden++] = initial[index];
                }
            }
            particle.hidden[2] = remaining[indices[3]];
            particle.log_weight = history_log_likelihood(
                actual, viewer, particle);
            if (std::isfinite(particle.log_weight)) {
                posterior_.push_back(particle);
            }
        }
        if (posterior_.empty()) {
            ++stats_.posterior_fallbacks;
            return;
        }

        double maximum = -std::numeric_limits<double>::infinity();
        for (const PosteriorParticle& particle : posterior_) {
            maximum = std::max(maximum, particle.log_weight);
        }
        double squared_weight_sum = 0.0;
        for (PosteriorParticle& particle : posterior_) {
            const double weight = std::exp(particle.log_weight - maximum);
            posterior_total_weight_ += weight;
            squared_weight_sum += weight * weight;
            particle.cumulative_weight = posterior_total_weight_;
        }
        stats_.posterior_effective_particles_sum +=
            squared_weight_sum > 0.0
                ? posterior_total_weight_ * posterior_total_weight_ /
                    squared_weight_sum
                : 0.0;
    }

    State determinize(const State& actual, int viewer) {
        State state = actual;
        if (!posterior_.empty() && posterior_total_weight_ > 0.0) {
            const double value = std::uniform_real_distribution<double>(
                0.0, posterior_total_weight_)(rng_);
            const auto position = std::lower_bound(
                posterior_.begin(), posterior_.end(), value,
                [](const PosteriorParticle& particle, double threshold) {
                    return particle.cumulative_weight < threshold;
                });
            const PosteriorParticle& particle =
                position == posterior_.end() ? posterior_.back() : *position;
            Player& opponent = state.players[1 - viewer];
            opponent.hidden.assign(
                particle.hidden.begin(), particle.hidden.end());
            opponent.discarded = particle.discarded;
            opponent.has_discard = true;
            state.simulation_deck.clear();
            state.terminal = false;
            return state;
        }
        std::array<bool, 52> known{};
        const auto mark = [&](const Card& card) {
            known[card.suit * 13 + card.rank - 2] = true;
        };
        for (const Card& card : actual.players[viewer].hidden) mark(card);
        for (const Card& card : actual.players[viewer].shown) mark(card);
        if (actual.players[viewer].has_discard) {
            mark(actual.players[viewer].discarded);
        }
        for (const Card& card : actual.players[1 - viewer].shown) mark(card);
        std::vector<Card> remaining;
        for (const Card& card : fresh_deck()) {
            if (!known[card.suit * 13 + card.rank - 2]) {
                remaining.push_back(card);
            }
        }
        std::shuffle(remaining.begin(), remaining.end(), rng_);
        Player& opponent = state.players[1 - viewer];
        opponent.hidden.assign(remaining.end() - 3, remaining.end());
        remaining.resize(remaining.size() - 3);
        opponent.discarded = remaining.back();
        opponent.has_discard = true;
        state.simulation_deck.clear();
        state.terminal = false;
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

    MCCFR& blueprint_;
    int iterations_;
    double prior_strength_;
    bool preserve_hand_history_;
    int posterior_particle_count_ = 0;
    std::vector<PosteriorParticle> posterior_;
    double posterior_total_weight_ = 0.0;
    std::mt19937_64 rng_;
    std::unordered_map<
        ExactSeventhKey,
        RegretNode,
        ExactSeventhKeyHash> nodes_;
    mutable Stats stats_;
};

struct BehaviorNode {
    std::array<double, kActionCount> strategy{};
    uint64_t touches = 0;
};

struct BehaviorGradient {
    std::array<double, kActionCount> sum{};
    uint64_t samples = 0;
    uint8_t legal_mask = 0;
};

class BucketAsymP {
    using Policy = std::unordered_map<InfoKey, BehaviorNode, InfoKeyHash>;
    using GradientPolicy =
        std::unordered_map<InfoKey, BehaviorGradient, InfoKeyHash>;
    using GradientRuns = std::array<std::array<GradientPolicy, 2>, 2>;

public:
    BucketAsymP(
        uint64_t seed,
        int start_street,
        const PowerAtlas* atlas,
        double step,
        double perturbation,
        bool preserve_recall = false)
        : start_street_(start_street),
          atlas_(atlas),
          step_(step),
          perturbation_(perturbation),
          preserve_recall_(preserve_recall),
          rng_(seed) {
        if (step <= 0.0 || perturbation < 0.0) {
            throw std::runtime_error("AsymP step must be positive and perturbation nonnegative");
        }
    }

    void initialize(const MCCFR& base) {
        const auto snapshot = base.policy_snapshot();
        for (auto& run : runs_) {
            for (auto& policy : run) {
                policy.clear();
                policy.reserve(snapshot.size());
                for (const auto& [key, strategy] : snapshot) {
                    policy.emplace(key, BehaviorNode{strategy, 0});
                }
            }
        }
    }

    Action choose(const State& state, int viewer, int) {
        const uint8_t mask = valid_mask(state, viewer);
        const InfoKey key = bucket_key(state, viewer);
        const BehaviorNode& node = get_node(viewer, viewer, key);
        return sample(node.strategy, mask);
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        const uint8_t mask = valid_mask(state, viewer);
        const InfoKey key = bucket_key(state, viewer);
        const auto node = runs_[viewer][viewer].find(key);
        if (found) *found = node != runs_[viewer][viewer].end();
        return node == runs_[viewer][viewer].end()
            ? uniform_strategy(mask)
            : node->second.strategy;
    }

    void train_root(const State& root) {
        train_roots(std::vector<State>{root});
    }

    void train_roots(const std::vector<State>& roots) {
        if (roots.empty()) return;
        for (int perturbed = 0; perturbed < 2; ++perturbed) {
            GradientRuns gradients;
            for (const State& root : roots) {
                traverse(root, perturbed, perturbed, &gradients);
                ++traversals_;
            }
            apply_gradients(
                perturbed,
                perturbed,
                gradients[perturbed][perturbed]);
            gradients[perturbed][perturbed].clear();
            for (const State& root : roots) {
                traverse(root, perturbed, 1 - perturbed, &gradients);
                ++traversals_;
            }
            apply_gradients(
                perturbed,
                1 - perturbed,
                gradients[perturbed][1 - perturbed]);
        }
    }

    void apply_gradients(
        int run,
        int actor,
        const GradientPolicy& gradients) {
        for (const auto& [key, accumulated] : gradients) {
            std::array<double, kActionCount> gradient{};
            for (int action = 0; action < kActionCount; ++action) {
                gradient[action] =
                    accumulated.sum[action] / accumulated.samples;
            }
            apply_gradient(
                run,
                actor,
                key,
                gradient,
                accumulated.legal_mask);
        }
    }

    void save(const std::string& path) const {
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write AsymP model: " + path);
        output.write("ASYMPV1", 8);
        const uint8_t abstraction = atlas_
            ? static_cast<uint8_t>(preserve_recall_ ? 3 : 1)
            : 0;
        const uint8_t street = static_cast<uint8_t>(start_street_);
        output.write(reinterpret_cast<const char*>(&abstraction), sizeof(abstraction));
        output.write(reinterpret_cast<const char*>(&street), sizeof(street));
        output.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        output.write(reinterpret_cast<const char*>(&perturbation_), sizeof(perturbation_));
        for (const auto& run : runs_) {
            for (const auto& policy : run) {
                const uint64_t count = policy.size();
                output.write(reinterpret_cast<const char*>(&count), sizeof(count));
                for (const auto& [key, node] : policy) {
                    output.write(reinterpret_cast<const char*>(&key), sizeof(key));
                    output.write(reinterpret_cast<const char*>(&node), sizeof(node));
                }
            }
        }
    }

    void load(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read AsymP model: " + path);
        char magic[8]{};
        uint8_t abstraction = 0;
        uint8_t street = 0;
        double saved_step = 0.0;
        double saved_perturbation = 0.0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&abstraction), sizeof(abstraction));
        input.read(reinterpret_cast<char*>(&street), sizeof(street));
        input.read(reinterpret_cast<char*>(&saved_step), sizeof(saved_step));
        input.read(reinterpret_cast<char*>(&saved_perturbation), sizeof(saved_perturbation));
        if (std::memcmp(magic, "ASYMPV1", 7) != 0 ||
            abstraction != (atlas_
                ? static_cast<uint8_t>(preserve_recall_ ? 3 : 1)
                : 0) ||
            street != start_street_) {
            throw std::runtime_error("incompatible AsymP model");
        }
        step_ = saved_step;
        perturbation_ = saved_perturbation;
        for (auto& run : runs_) {
            for (auto& policy : run) {
                uint64_t count = 0;
                input.read(reinterpret_cast<char*>(&count), sizeof(count));
                policy.clear();
                policy.reserve(static_cast<size_t>(count * 1.3) + 1);
                for (uint64_t index = 0; index < count; ++index) {
                    InfoKey key{};
                    BehaviorNode node{};
                    input.read(reinterpret_cast<char*>(&key), sizeof(key));
                    input.read(reinterpret_cast<char*>(&node), sizeof(node));
                    policy.emplace(key, node);
                }
            }
        }
        if (!input) throw std::runtime_error("truncated AsymP model");
    }

    uint64_t traversals() const { return traversals_; }
    uint64_t node_visits() const { return node_visits_; }
    uint64_t buckets() const {
        return runs_[0][0].size() + runs_[1][1].size();
    }
    double step() const { return step_; }
    double perturbation() const { return perturbation_; }

private:
    double traverse(
        State state,
        int run,
        int traverser,
        GradientRuns* gradients) {
        ++node_visits_;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = bucket_key(state, actor);
        const auto strategy = get_node(run, actor, key).strategy;

        if (actor != traverser) {
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return traverse(std::move(state), run, traverser, gradients);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_street(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] =
                traverse(std::move(child), run, traverser, gradients);
        }
        double node_value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                node_value += strategy[action] * action_values[action];
            }
        }
        std::array<double, kActionCount> gradient{};
        double squared_norm = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) squared_norm += strategy[action] * strategy[action];
        }
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            gradient[action] = action_values[action] - node_value;
            if (traverser == run) {
                gradient[action] -= perturbation_ * (strategy[action] - squared_norm);
            }
        }
        BehaviorGradient& accumulated = (*gradients)[run][actor][key];
        for (int action = 0; action < kActionCount; ++action) {
            accumulated.sum[action] += gradient[action];
        }
        ++accumulated.samples;
        accumulated.legal_mask = mask;
        return node_value;
    }

    void apply_gradient(
        int run,
        int actor,
        const InfoKey& key,
        const std::array<double, kActionCount>& gradient,
        uint8_t mask) {
        BehaviorNode& updated = get_node(run, actor, key);
        ++updated.touches;
        const double local_step =
            step_ / std::sqrt(static_cast<double>(updated.touches));
        updated.strategy = projected_strategy(
            updated.strategy, gradient, mask, local_step);
    }

    InfoKey bucket_key(const State& state, int viewer) const {
        return atlas_
            ? make_power_key(state, viewer, *atlas_, preserve_recall_)
            : make_key(state, viewer);
    }

    BehaviorNode& get_node(int run, int seat, const InfoKey& key) {
        Policy& policy = runs_[run][seat];
        auto [it, inserted] = policy.try_emplace(key);
        if (inserted) it->second.strategy = uniform_strategy(key.legal_mask);
        ++it->second.touches;
        return it->second;
    }

    void advance_street(State& state) {
        if (state.street == 7) {
            state.terminal = true;
            return;
        }
        ++state.street;
        const bool public_card = state.street != 7;
        for (int seat = 0; seat < 2; ++seat) {
            if (state.simulation_deck.empty()) throw std::runtime_error("simulation deck exhausted");
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
    }

    Action sample(const std::array<double, kActionCount>& strategy, uint8_t mask) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double threshold = uniform(rng_);
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

    int start_street_;
    const PowerAtlas* atlas_;
    double step_;
    double perturbation_;
    bool preserve_recall_ = false;
    std::mt19937_64 rng_;
    std::array<std::array<Policy, 2>, 2> runs_;
    uint64_t traversals_ = 0;
    uint64_t node_visits_ = 0;
};

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        const size_t end = value.find(delimiter, start);
        parts.push_back(value.substr(start, end - start));
        if (end == std::string::npos) return parts;
        start = end + 1;
    }
}

Card parse_card(const std::string& value) {
    if (value.size() < 2) throw std::runtime_error("invalid card: " + value);
    const std::string suits = "shdc";
    const size_t suit = suits.find(value[0]);
    if (suit == std::string::npos) throw std::runtime_error("invalid card suit: " + value);
    const std::string rank_text = value.substr(1);
    int rank = 0;
    if (rank_text == "A") rank = 14;
    else if (rank_text == "K") rank = 13;
    else if (rank_text == "Q") rank = 12;
    else if (rank_text == "J") rank = 11;
    else if (rank_text == "T" || rank_text == "10") rank = 10;
    else rank = std::stoi(rank_text);
    if (rank < 2 || rank > 14) throw std::runtime_error("invalid card rank: " + value);
    return Card{static_cast<uint8_t>(rank), static_cast<uint8_t>(suit)};
}

std::vector<Card> parse_cards(const std::string& value) {
    std::vector<Card> cards;
    if (value.empty() || value == "-") return cards;
    for (const std::string& card : split(value, ',')) cards.push_back(parse_card(card));
    return cards;
}

Action parse_action(const std::string& value) {
    for (int action = 0; action < kActionCount; ++action) {
        if (value == kActionNames[action]) return static_cast<Action>(action);
    }
    throw std::runtime_error("invalid action: " + value);
}

int parse_street(const std::string& value) {
    if (value == "5th") return 5;
    if (value == "6th") return 6;
    if (value == "7th_hidden") return 7;
    throw std::runtime_error("invalid street: " + value);
}

struct AgentRequest {
    State state;
    uint8_t expected_mask = 0;
};

AgentRequest parse_agent_request(const std::string& line) {
    const auto fields = split(line, '\t');
    if (fields.size() != 16 || fields[0] != "ACT") {
        throw std::runtime_error("ACT requires 15 tab-separated fields");
    }
    AgentRequest request;
    State& state = request.state;
    state.street = parse_street(fields[1]);
    state.ante = std::stoi(fields[2]);
    state.pot = std::stoi(fields[3]);
    state.highest_bet = std::stoi(fields[4]);
    const auto stack_caps = split(fields[5], ',');
    if (stack_caps.empty() || stack_caps.size() > 2) {
        throw std::runtime_error("effective stack must be own or own,opponent");
    }
    state.effective_stack = std::stoi(stack_caps[0]);
    state.players[0].stack_cap = state.effective_stack;
    state.players[1].stack_cap = stack_caps.size() == 2
        ? std::stoi(stack_caps[1])
        : state.effective_stack;
    state.players[0].invested = std::stoi(fields[6]);
    state.players[0].round_bet = std::stoi(fields[7]);
    state.players[1].invested = std::stoi(fields[8]);
    state.players[1].round_bet = std::stoi(fields[9]);
    state.players[0].hidden = parse_cards(fields[10]);
    state.players[0].shown = parse_cards(fields[11]);
    if (fields[12] != "-") {
        state.players[0].discarded = parse_card(fields[12]);
        state.players[0].has_discard = true;
    }
    state.players[1].shown = parse_cards(fields[13]);
    if (fields[14] != "-") {
        for (const std::string& encoded : split(fields[14], ';')) {
            const auto event = split(encoded, ':');
            if (event.size() != 3) throw std::runtime_error("invalid history event");
            state.history.push_back(Event{
                static_cast<uint8_t>(parse_street(event[0])),
                static_cast<uint8_t>(std::stoi(event[1])),
                parse_action(event[2]),
            });
        }
    }
    request.expected_mask = static_cast<uint8_t>(std::stoul(fields[15]));
    state.players[0].all_in =
        state.players[0].invested >= stack_cap(state, 0);
    state.players[1].all_in =
        state.players[1].invested >= stack_cap(state, 1);
    state.actor = 0;
    if (state.ante <= 0 ||
        stack_cap(state, 0) <= 0 ||
        stack_cap(state, 1) <= 0) {
        throw std::runtime_error("ante and effective stacks must be positive");
    }
    const uint8_t actual_mask = valid_mask(state, 0);
    if (actual_mask != request.expected_mask) {
        throw std::runtime_error(
            "legal action mask mismatch: python=" +
            std::to_string(request.expected_mask) +
            ", cpp=" + std::to_string(actual_mask));
    }
    return request;
}

template <typename Solver>
void run_agent_stdio(Solver& solver, H4Policy* h4_policy = nullptr) {
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            if (line == "QUIT") return;
            if (line.rfind("DISCARD\t", 0) == 0) {
                const auto cards = parse_cards(line.substr(8));
                if (cards.size() != 4) throw std::runtime_error("DISCARD requires four cards");
                const auto [discard, reveal] = h4_policy
                    ? h4_policy->choose(cards)
                    : discard_reveal(cards);
                std::cout << "DISCARD\t" << discard << "\t" << reveal << "\n";
            } else {
                AgentRequest request = parse_agent_request(line);
                const Action action = solver.choose(request.state, 0, 0);
                std::cout << "ACTION\t" << kActionNames[action] << "\n";
            }
        } catch (const std::exception& error) {
            std::cout << "ERROR\t" << error.what() << "\n";
        }
        std::cout << std::flush;
    }
}

class BeliefBR {
public:
    BeliefBR(int particles, double sigma, double margin, uint64_t seed)
        : particles_(particles), sigma_(sigma), margin_(margin), rng_(seed) {
        if (particles <= 0 || sigma <= 0) throw std::runtime_error("invalid belief-BR config");
    }

    Action choose(const State& state, int seat) {
        const Belief belief = estimate(state, seat);
        const uint8_t mask = valid_mask(state, seat);
        std::array<double, kActionCount> evs{};
        for (Action action : actions_from_mask(mask)) {
            evs[action] = action_ev(action, state, seat, belief);
        }
        Action best = FOLD;
        double best_ev = -std::numeric_limits<double>::infinity();
        for (Action action : actions_from_mask(mask)) {
            if (aggressive(action)) continue;
            if (evs[action] > best_ev) {
                best = action;
                best_ev = evs[action];
            }
        }
        Action best_raise = FOLD;
        double best_raise_ev = -std::numeric_limits<double>::infinity();
        for (Action action : actions_from_mask(mask)) {
            if (!aggressive(action)) continue;
            if (evs[action] > best_raise_ev) {
                best_raise = action;
                best_raise_ev = evs[action];
            }
        }
        const double call = std::max(
            0, state.highest_bet - state.players[seat].round_bet);
        const double required_margin = margin_ * (state.pot + call + 1.0);
        if (best_raise_ev >= best_ev + required_margin) best = best_raise;
        return best;
    }

    std::pair<int, int> choose_discard(const std::vector<Card>& cards) {
        int discard = 0;
        double best = -1.0;
        for (int index = 0; index < 4; ++index) {
            std::vector<Card> retained;
            for (int i = 0; i < 4; ++i) if (i != index) retained.push_back(cards[i]);
            const double equity = retained_equity(retained, cards);
            if (equity > best) {
                best = equity;
                discard = index;
            }
        }
        int reveal = discard == 0 ? 1 : 0;
        for (int index = 0; index < 4; ++index) {
            if (index == discard) continue;
            if (reveal_value(cards[index], cards) > reveal_value(cards[reveal], cards)) {
                reveal = index;
            }
        }
        return {discard, reveal};
    }

private:
    struct Particle {
        double strength = 0.0;
        double weight = 1.0;
        double win = 0.0;
    };

    struct Belief {
        double equity = 0.5;
        double uniform_equity = 0.5;
        std::vector<Particle> particles;
    };

    std::vector<Card> draw_without_replacement(
        const std::vector<Card>& deck,
        int count) {
        std::vector<Card> draw = deck;
        for (int i = 0; i < count; ++i) {
            std::uniform_int_distribution<int> pick(i, static_cast<int>(draw.size()) - 1);
            std::swap(draw[i], draw[pick(rng_)]);
        }
        draw.resize(count);
        return draw;
    }

    double observed_strength(const State& state, int seat) const {
        double signal = -1.0;
        for (const Event& event : state.history) {
            if (event.actor == seat) continue;
            double value = -1.0;
            if (event.action == HALF) value = 0.82;
            else if (event.action == DDADANG) value = 0.80;
            else if (event.action == QUARTER) value = 0.63;
            else if (event.action == BBING) value = 0.52;
            else if (event.action == CALL) value = 0.33;
            else if (event.action == CHECK) value = 0.18;
            signal = std::max(signal, value);
        }
        return signal;
    }

    double sampled_opponent_strength(
        const State& actual,
        int viewer,
        const std::vector<Card>& hidden,
        const std::vector<Card>& shown) const {
        State model;
        model.ante = actual.ante;
        model.effective_stack = actual.effective_stack;
        model.pot = actual.pot;
        model.players[0].hidden = hidden;
        model.players[0].shown = shown;
        model.players[1].shown = actual.players[viewer].shown;
        return heuristic_strength(model, 0);
    }

    Belief estimate(const State& state, int seat) {
        Belief belief;
        const std::vector<Card> own = all_cards(state.players[seat]);
        const std::vector<Card>& opponent_public = state.players[1 - seat].shown;
        uint64_t known = card_mask(own) | card_mask(opponent_public);
        if (state.players[seat].has_discard) {
            known |= 1ull << (
                state.players[seat].discarded.suit * 13 +
                state.players[seat].discarded.rank - 2);
        }
        std::vector<Card> deck;
        for (const Card& card : fresh_deck()) {
            if (!(known & (1ull << (card.suit * 13 + card.rank - 2)))) {
                deck.push_back(card);
            }
        }
        const int hidden_count = state.street == 7 ? 3 : 2;
        const int opponent_future =
            std::max(0, 7 - static_cast<int>(opponent_public.size()) - hidden_count);
        const int own_future = std::max(0, 7 - static_cast<int>(own.size()));
        const int sample_size = 1 + own_future + hidden_count + opponent_future;
        if (sample_size > static_cast<int>(deck.size())) return belief;

        const double action_signal = observed_strength(state, seat);
        double weighted_win = 0.0;
        double weight_sum = 0.0;
        double uniform_win = 0.0;
        belief.particles.reserve(particles_);
        for (int iteration = 0; iteration < particles_; ++iteration) {
            const std::vector<Card> draw = draw_without_replacement(deck, sample_size);
            int cursor = 1;  // Opponent's unknown discarded card.
            std::vector<Card> own_final = own;
            own_final.insert(
                own_final.end(), draw.begin() + cursor, draw.begin() + cursor + own_future);
            cursor += own_future;
            std::vector<Card> hidden(
                draw.begin() + cursor, draw.begin() + cursor + hidden_count);
            cursor += hidden_count;
            std::vector<Card> opponent_final = opponent_public;
            opponent_final.insert(opponent_final.end(), hidden.begin(), hidden.end());
            opponent_final.insert(
                opponent_final.end(),
                draw.begin() + cursor,
                draw.begin() + cursor + opponent_future);

            const Score own_score = best_hand(own_final);
            const Score opponent_score = best_hand(opponent_final);
            const double win = own_score > opponent_score
                ? 1.0 : own_score == opponent_score ? 0.5 : 0.0;
            const double strength = sampled_opponent_strength(
                state, seat, hidden, opponent_public);
            const double weight = action_signal < 0
                ? 1.0
                : std::exp(
                    -std::pow(strength - action_signal, 2) /
                    (2 * sigma_ * sigma_));
            weighted_win += weight * win;
            weight_sum += weight;
            uniform_win += win;
            belief.particles.push_back({strength, weight, win});
        }
        belief.uniform_equity = uniform_win / particles_;
        belief.equity = weight_sum > 0
            ? weighted_win / weight_sum
            : belief.uniform_equity;
        return belief;
    }

    bool opponent_calls(
        double strength,
        double raise,
        const State& state,
        int seat) const {
        if (strength >= 0.90) return true;
        if (raise <= state.ante) return strength >= 0.22;
        if (raise <= state.ante * 2) return strength >= 0.30;
        const double pot = state.pot +
            std::max(0, state.highest_bet - state.players[seat].round_bet) +
            raise;
        const double odds = raise / std::max(1.0, pot + raise);
        return strength >= std::min(0.72, std::max(0.28, odds + 0.08));
    }

    double action_ev(
        Action action,
        const State& state,
        int seat,
        const Belief& belief) const {
        const double pot = state.pot;
        const double call = std::max(
            0, state.highest_bet - state.players[seat].round_bet);
        const double invested = state.players[seat].invested;
        if (action == FOLD) return -invested;
        if (action == CHECK) return belief.equity * pot - invested;
        if (action == CALL) {
            return belief.equity * (pot + call) - (invested + call);
        }

        const double raise = raise_amount(state, action, static_cast<int>(call));
        const double fold_branch = pot - invested;
        const double final_pot = pot + call + 2 * raise;
        const double call_cost = invested + call + raise;
        double total = 0.0;
        double ev = 0.0;
        for (const Particle& particle : belief.particles) {
            total += particle.weight;
            if (opponent_calls(particle.strength, raise, state, seat)) {
                ev += particle.weight * (particle.win * final_pot - call_cost);
            } else {
                ev += particle.weight * fold_branch;
            }
        }
        return total > 0 ? ev / total : fold_branch;
    }

    double retained_equity(
        const std::vector<Card>& retained,
        const std::vector<Card>& initial) {
        const uint64_t known = card_mask(initial);
        std::vector<Card> deck;
        for (const Card& card : fresh_deck()) {
            if (!(known & (1ull << (card.suit * 13 + card.rank - 2)))) {
                deck.push_back(card);
            }
        }
        const int simulations = std::max(32, particles_ / 4);
        double total = 0.0;
        for (int iteration = 0; iteration < simulations; ++iteration) {
            const std::vector<Card> draw = draw_without_replacement(deck, 12);
            std::vector<Card> own = retained;
            own.insert(own.end(), draw.begin() + 1, draw.begin() + 5);
            std::vector<Card> opponent(draw.begin() + 5, draw.begin() + 12);
            const Score own_score = best_hand(own);
            const Score opponent_score = best_hand(opponent);
            total += own_score > opponent_score
                ? 1.0 : own_score == opponent_score ? 0.5 : 0.0;
        }
        return total / simulations;
    }

    static double reveal_value(const Card& card, const std::vector<Card>& cards) {
        int same_rank = -1;
        for (const Card& other : cards) same_rank += other.rank == card.rank;
        return card.rank + same_rank * 5.0;
    }

    int particles_;
    double sigma_;
    double margin_;
    std::mt19937_64 rng_;
};

template <typename TargetPolicy>
class PolicyLBR {
public:
    using AttributionCallback =
        std::function<void(const State&, int, double)>;

    struct Stats {
        uint64_t decisions = 0;
        uint64_t policy_queries = 0;
        uint64_t policy_misses = 0;
        uint64_t h4_particle_proposals = 0;
        uint64_t h4_particle_rejections = 0;
        double average_effective_particles = 0.0;
        std::array<uint64_t, kActionCount> actions{};
    };

    PolicyLBR(const TargetPolicy& target, int particles, uint64_t seed)
        : target_(target), particles_(particles), rng_(seed) {
        if (particles <= 0) throw std::runtime_error("invalid policy-LBR particle count");
    }

    void reset() {
        observations_.clear();
    }

    void set_attribution_callback(AttributionCallback callback) {
        attribution_callback_ = std::move(callback);
    }

    void observe(const State& state, int target_seat, Action action) {
        State public_state = state;
        for (Player& player : public_state.players) {
            player.hidden.clear();
            player.has_discard = false;
        }
        observations_.push_back({std::move(public_state), target_seat, action});
    }

    Action choose(const State& state, int seat) {
        const Belief belief = estimate(state, seat);
        const uint8_t mask = valid_mask(state, seat);
        Action best = actions_from_mask(mask).front();
        double best_ev = -std::numeric_limits<double>::infinity();
        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            const double ev = action_ev(action, state, seat, belief);
            action_values[action] = ev;
            if (ev > best_ev) {
                best = action;
                best_ev = ev;
            }
        }
        if (attribution_callback_) {
            const auto baseline = target_policy(state, seat);
            double baseline_ev = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    baseline_ev += baseline[action] * action_values[action];
                }
            }
            attribution_callback_(
                state,
                seat,
                std::max(0.0, best_ev - baseline_ev) /
                    std::max(1, state.ante));
        }
        ++decisions_;
        effective_particles_sum_ += belief.effective_particles;
        ++action_counts_[best];
        return best;
    }

    Stats stats() const {
        Stats result;
        result.decisions = decisions_;
        result.policy_queries = policy_queries_;
        result.policy_misses = policy_misses_;
        result.h4_particle_proposals = h4_particle_proposals_;
        result.h4_particle_rejections = h4_particle_rejections_;
        result.average_effective_particles =
            decisions_ ? effective_particles_sum_ / decisions_ : 0.0;
        result.actions = action_counts_;
        return result;
    }

private:
    struct Observation {
        State state;
        int target_seat = 0;
        Action action = CHECK;
    };

    struct Particle {
        std::array<Card, 3> target_hidden{};
        Card target_discarded{};
        double weight = 1.0;
        double win = 0.0;
    };

    struct Belief {
        std::vector<Particle> particles;
        double effective_particles = 0.0;
    };

    std::array<Card, 7> draw_without_replacement(
        const std::vector<Card>& deck,
        int count) {
        std::array<Card, 7> draw{};
        std::array<int, 7> indices{};
        for (int i = 0; i < count; ++i) {
            std::uniform_int_distribution<int> pick(
                0, static_cast<int>(deck.size()) - 1);
            int index;
            bool duplicate;
            do {
                index = pick(rng_);
                duplicate = false;
                for (int previous = 0; previous < i; ++previous) {
                    if (indices[previous] == index) {
                        duplicate = true;
                        break;
                    }
                }
            } while (duplicate);
            indices[i] = index;
            draw[i] = deck[index];
        }
        return draw;
    }

    void set_target_cards(
        State& model,
        const Particle& particle,
        int target_seat) const {
        const int hidden_count = model.street == 7 ? 3 : 2;
        model.players[target_seat].hidden.assign(
            particle.target_hidden.begin(),
            particle.target_hidden.begin() + hidden_count);
        model.players[target_seat].discarded = particle.target_discarded;
        model.players[target_seat].has_discard = true;
        model.players[1 - target_seat].hidden.clear();
        model.players[1 - target_seat].has_discard = false;
    }

    std::array<double, kActionCount> target_policy(
        const State& state,
        int target_seat) {
        bool found = false;
        const auto strategy = target_.policy(state, target_seat, &found);
        ++policy_queries_;
        policy_misses_ += !found;
        return strategy;
    }

    static int h4_multiplicity(
        const std::array<Card, 2>& hidden,
        const Card& shown,
        const Card& discarded) {
        const std::array<Card, 4> cards{
            hidden[0], hidden[1], shown, discarded};
        std::array<int, 4> order{0, 1, 2, 3};
        int matches = 0;
        do {
            std::vector<Card> hand;
            hand.reserve(4);
            for (int index : order) hand.push_back(cards[index]);
            const auto [discard_index, reveal_index] = discard_reveal(hand);
            matches += hand[discard_index] == discarded &&
                hand[reveal_index] == shown;
        } while (std::next_permutation(order.begin(), order.end()));
        return matches;
    }

    Belief estimate(const State& state, int seat) {
        Belief belief;
        const int target_seat = 1 - seat;
        const std::vector<Card> own = all_cards(state.players[seat]);
        const std::vector<Card>& target_public =
            state.players[target_seat].shown;
        if (target_public.empty()) return belief;
        uint64_t known = card_mask(own) | card_mask(target_public);
        if (state.players[seat].has_discard) {
            known |= 1ull << (
                state.players[seat].discarded.suit * 13 +
                state.players[seat].discarded.rank - 2);
        }
        std::vector<Card> deck;
        for (const Card& card : fresh_deck()) {
            const uint64_t bit =
                1ull << (card.suit * 13 + card.rank - 2);
            if (!(known & bit)) deck.push_back(card);
        }

        const int target_hidden_count = state.street == 7 ? 3 : 2;
        const int own_future =
            std::max(0, 7 - static_cast<int>(own.size()));
        const int target_future = std::max(
            0,
            7 - static_cast<int>(target_public.size()) -
                target_hidden_count);
        const int sample_size =
            1 + target_hidden_count + own_future + target_future;
        if (sample_size > static_cast<int>(deck.size())) return belief;

        belief.particles.reserve(particles_);
        std::vector<Card> own_final;
        std::vector<Card> target_final;
        own_final.reserve(7);
        target_final.reserve(7);
        const int proposal_limit = std::max(particles_ * 64, particles_ + 64);
        for (int proposal = 0;
             proposal < proposal_limit &&
                 static_cast<int>(belief.particles.size()) < particles_;
             ++proposal) {
            ++h4_particle_proposals_;
            const auto draw = draw_without_replacement(deck, sample_size);
            int cursor = 0;
            Particle particle;
            particle.target_discarded = draw[cursor++];
            for (int hidden = 0; hidden < target_hidden_count; ++hidden) {
                particle.target_hidden[hidden] = draw[cursor++];
            }
            const int h4_ways = h4_multiplicity(
                {particle.target_hidden[0], particle.target_hidden[1]},
                target_public.front(),
                particle.target_discarded);
            if (!h4_ways) {
                ++h4_particle_rejections_;
                continue;
            }

            own_final.assign(own.begin(), own.end());
            own_final.insert(
                own_final.end(),
                draw.begin() + cursor,
                draw.begin() + cursor + own_future);
            cursor += own_future;
            target_final.assign(target_public.begin(), target_public.end());
            target_final.insert(
                target_final.end(),
                particle.target_hidden.begin(),
                particle.target_hidden.begin() + target_hidden_count);
            target_final.insert(
                target_final.end(),
                draw.begin() + cursor,
                draw.begin() + cursor + target_future);

            const Score own_score = best_hand(own_final);
            const Score target_score = best_hand(target_final);
            particle.win = own_score > target_score
                ? 1.0 : own_score == target_score ? 0.5 : 0.0;
            particle.weight = std::log(static_cast<double>(h4_ways));
            belief.particles.push_back(std::move(particle));
        }

        for (const Observation& observation : observations_) {
            State hypothesis = observation.state;
            for (Particle& particle : belief.particles) {
                set_target_cards(hypothesis, particle, observation.target_seat);
                const auto strategy =
                    target_policy(hypothesis, observation.target_seat);
                particle.weight += std::log(std::max(
                    1e-12, strategy[observation.action]));
            }
        }

        double max_log_weight = -std::numeric_limits<double>::infinity();
        for (const Particle& particle : belief.particles) {
            max_log_weight = std::max(max_log_weight, particle.weight);
        }
        double weight_sum = 0.0;
        double squared_weight_sum = 0.0;
        for (Particle& particle : belief.particles) {
            const double weight = std::exp(particle.weight - max_log_weight);
            particle.weight = weight;
            weight_sum += weight;
            squared_weight_sum += weight * weight;
        }
        belief.effective_particles = squared_weight_sum > 0.0
            ? weight_sum * weight_sum / squared_weight_sum
            : 0.0;
        return belief;
    }

    double action_ev(
        Action action,
        const State& state,
        int seat,
        const Belief& belief) {
        const double pot = state.pot;
        const double call = std::max(
            0, state.highest_bet - state.players[seat].round_bet);
        const double invested = state.players[seat].invested;
        if (action == FOLD) return -invested;

        double weight_sum = 0.0;
        double weighted_win = 0.0;
        for (const Particle& particle : belief.particles) {
            weight_sum += particle.weight;
            weighted_win += particle.weight * particle.win;
        }
        const double equity =
            weight_sum > 0.0 ? weighted_win / weight_sum : 0.5;
        if (action == CHECK) return equity * pot - invested;
        if (action == CALL) {
            return equity * (pot + call) - (invested + call);
        }

        const double raise =
            raise_amount(state, action, static_cast<int>(call));
        const double fold_branch = pot - invested;
        const double final_pot = pot + call + 2 * raise;
        const double call_cost = invested + call + raise;
        double ev = 0.0;
        State response = state;
        const ActionResult result = apply_action(response, seat, action);
        if (result == ActionResult::Raise) response.actor = 1 - seat;
        for (const Particle& particle : belief.particles) {
            double fold_probability = 0.0;
            if (result == ActionResult::Raise) {
                set_target_cards(response, particle, 1 - seat);
                const auto strategy =
                    target_policy(response, 1 - seat);
                fold_probability = strategy[FOLD];
            }
            const double called =
                particle.win * final_pot - call_cost;
            ev += particle.weight * (
                fold_probability * fold_branch +
                (1.0 - fold_probability) * called);
        }
        return weight_sum > 0.0 ? ev / weight_sum : fold_branch;
    }

    const TargetPolicy& target_;
    int particles_;
    std::mt19937_64 rng_;
    std::vector<Observation> observations_;
    uint64_t decisions_ = 0;
    uint64_t policy_queries_ = 0;
    uint64_t policy_misses_ = 0;
    uint64_t h4_particle_proposals_ = 0;
    uint64_t h4_particle_rejections_ = 0;
    double effective_particles_sum_ = 0.0;
    std::array<uint64_t, kActionCount> action_counts_{};
    AttributionCallback attribution_callback_;
};

void apply_h4_choices(
    State& state,
    const std::array<std::pair<int, int>, 2>& choices) {
    for (int seat = 0; seat < 2; ++seat) {
        Player& player = state.players[seat];
        const auto [discard, reveal] = choices[seat];
        if (discard < 0 || discard >= 4 || reveal < 0 || reveal >= 4 ||
            discard == reveal) {
            throw std::runtime_error("invalid H4 action");
        }
        player.discarded = player.hidden[discard];
        player.has_discard = true;
        player.shown.push_back(player.hidden[reveal]);
        std::vector<Card> kept;
        for (int index = 0; index < 4; ++index) {
            if (index != discard && index != reveal) kept.push_back(player.hidden[index]);
        }
        player.hidden = std::move(kept);
    }
}

void prepare_initial_cards(
    State& state,
    const std::vector<Card>& deck,
    size_t& cursor,
    int belief_seat = -1,
    BeliefBR* belief_agent = nullptr,
    H4Policy* h4_policy = nullptr,
    int h4_seat = -1,
    bool h4_both = false) {
    for (int round = 0; round < 4; ++round) {
        for (int seat = 0; seat < 2; ++seat) {
            state.players[seat].hidden.push_back(deck[cursor++]);
        }
    }
    std::array<std::pair<int, int>, 2> choices;
    for (int seat = 0; seat < 2; ++seat) {
        const Player& player = state.players[seat];
        if (h4_policy && (h4_both || seat == h4_seat)) {
            choices[seat] = h4_policy->choose(player.hidden);
        } else if (seat == belief_seat && belief_agent) {
            choices[seat] = belief_agent->choose_discard(player.hidden);
        } else {
            choices[seat] = discard_reveal(player.hidden);
        }
    }
    apply_h4_choices(state, choices);
}

int first_bettor(const State& state) {
    const auto left = public_priority(state.players[0].shown);
    const auto right = public_priority(state.players[1].shown);
    return right > left ? 1 : 0;
}

struct HeuristicPolicy {
    explicit HeuristicPolicy(
        HeuristicVariant variant = HeuristicVariant::Baseline)
        : variant(variant) {}

    Action choose(const State& state, int seat, int) const {
        return heuristic_variant_action(state, seat, variant);
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int seat,
        bool* found = nullptr) const {
        if (found) *found = true;
        std::array<double, kActionCount> result{};
        result[choose(state, seat, 0)] = 1.0;
        return result;
    }

    HeuristicVariant variant;
};

HeuristicVariant heuristic_variant_from_name(const std::string& name) {
    if (name == "bluff-defense") return HeuristicVariant::BluffDefense;
    if (name == "over-bluff") return HeuristicVariant::OverBluff;
    if (name == "hidden-bluff") return HeuristicVariant::HiddenBluff;
    if (name == "slow-play") return HeuristicVariant::SlowPlay;
    return HeuristicVariant::Baseline;
}

bool heuristic_opponent_name(const std::string& name) {
    return name == "heuristic" || name == "bluff-defense" ||
        name == "over-bluff" || name == "hidden-bluff" ||
        name == "slow-play";
}

struct HalfPolicy {
    Action choose(const State& state, int seat, int) const {
        return first_valid(valid_mask(state, seat), {HALF, CALL, CHECK, FOLD});
    }
};

class ConditionalParticipationPolicy {
public:
    ConditionalParticipationPolicy(std::string mode, int min_category)
        : mode_(std::move(mode)), min_category_(min_category) {}

    Action choose(const State& state, int seat, int) const {
        const uint8_t mask = valid_mask(state, seat);
        if (mode_ == "fold") return FOLD;

        const bool made = best_hand(all_cards(state.players[seat]))[0] >=
            min_category_;
        if (mask & (1u << CHECK)) {
            if (mode_ == "made-bet" && made) {
                return first_valid(mask, {BBING, QUARTER, HALF, CHECK});
            }
            return CHECK;
        }
        return made && (mask & (1u << CALL)) ? CALL : FOLD;
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int seat,
        bool* found = nullptr) const {
        if (found) *found = true;
        std::array<double, kActionCount> result{};
        result[choose(state, seat, 0)] = 1.0;
        return result;
    }

private:
    std::string mode_;
    int min_category_;
};

void reset_round(State& state, int street) {
    state.street = street;
    state.highest_bet = 0;
    state.raise_count = 0;
    for (Player& player : state.players) player.round_bet = 0;
    state.actor = first_bettor(state);
}

State sample_fifth_street_root(
    const std::vector<Card>& deck,
    int ante,
    int stack_ante = kEffectiveStackAnte,
    H4Policy* h4_policy = nullptr) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * stack_ante;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    prepare_initial_cards(
        state, deck, cursor, -1, nullptr, h4_policy, -1, true);
    for (int seat = 0; seat < 2; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);
    }
    for (int seat = 0; seat < 2; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);
    }
    state.simulation_deck.assign(deck.begin() + cursor, deck.end());
    reset_round(state, 5);
    return state;
}

struct ImitationResult {
    uint64_t roots = 0;
    uint64_t decisions = 0;
    uint64_t covered_buckets = 0;
    uint64_t reference_buckets = 0;
};

Action imitation_action(
    const std::array<double, kActionCount>& target,
    uint8_t mask,
    double exploration,
    std::mt19937_64& rng) {
    int legal_actions = 0;
    for (int action = 0; action < kActionCount; ++action) {
        legal_actions += !!(mask & (1u << action));
    }
    const bool skip_fold = legal_actions > 1 && (mask & (1u << FOLD));
    const int exploration_actions = legal_actions - (skip_fold ? 1 : 0);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double threshold = uniform(rng);
    double cumulative = 0.0;
    Action selected = FOLD;
    for (int action = 0; action < kActionCount; ++action) {
        if (!(mask & (1u << action))) continue;
        selected = static_cast<Action>(action);
        cumulative +=
            (1.0 - exploration) * target[action] +
            exploration *
                ((!skip_fold || action != FOLD)
                    ? 1.0 / exploration_actions
                    : 0.0);
        if (threshold <= cumulative) break;
    }
    return selected;
}

void imitate_teacher_external(
    MCCFR& student,
    MCCFR& teacher,
    State state,
    int traverser,
    double strength,
    double exploration,
    std::mt19937_64& rng,
    const MCCFR* coverage_reference,
    ImitationResult& summary) {
    if (state.terminal) return;
    const int actor = state.actor;
    bool student_found = false;
    bool reference_found = false;
    student.policy(state, actor, &student_found);
    if (coverage_reference) {
        coverage_reference->policy(state, actor, &reference_found);
    }
    if (!student_found && reference_found) ++summary.covered_buckets;
    const auto target = teacher.policy(state, actor);
    student.imitate_policy(state, actor, target, strength);
    ++summary.decisions;

    const uint8_t mask = valid_mask(state, actor);
    if (actor != traverser) {
        const Action action = imitation_action(
            target, mask, exploration, rng);
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd) {
            student.advance_imitation_state(state);
        } else if (result != ActionResult::FoldEnd) {
            state.actor = 1 - actor;
        }
        imitate_teacher_external(
            student, teacher, std::move(state), traverser, strength,
            exploration, rng, coverage_reference, summary);
        return;
    }

    for (Action action : actions_from_mask(mask)) {
        if (action == FOLD) continue;
        State child = state;
        const ActionResult result = apply_action(child, actor, action);
        if (result == ActionResult::RoundEnd) {
            student.advance_imitation_state(child);
        } else if (result != ActionResult::FoldEnd) {
            child.actor = 1 - actor;
        }
        imitate_teacher_external(
            student, teacher, std::move(child), traverser, strength,
            exploration, rng, coverage_reference, summary);
    }
}

ImitationResult imitate_teacher_roots(
    MCCFR& student,
    MCCFR& teacher,
    uint64_t roots,
    double strength,
    double exploration,
    int ante,
    int stack_ante,
    uint64_t seed,
    const MCCFR* coverage_reference = nullptr,
    uint64_t report_every = 0,
    bool external_sampling = false) {
    std::mt19937_64 rng(seed);
    ImitationResult summary;
    if (coverage_reference) {
        summary.reference_buckets = coverage_reference->stats().buckets;
    }
    for (uint64_t root = 0; root < roots; ++root) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        State state = sample_fifth_street_root(deck, ante, stack_ante);
        if (external_sampling) {
            imitate_teacher_external(
                student, teacher, state, 0, strength, exploration, rng,
                coverage_reference, summary);
            imitate_teacher_external(
                student, teacher, std::move(state), 1, strength,
                exploration, rng, coverage_reference, summary);
        }
        else {
        while (!state.terminal) {
            const int actor = state.actor;
            bool student_found = false;
            bool reference_found = false;
            student.policy(state, actor, &student_found);
            if (coverage_reference) {
                coverage_reference->policy(state, actor, &reference_found);
            }
            if (!student_found && reference_found) ++summary.covered_buckets;
            const auto target = teacher.policy(state, actor);
            student.imitate_policy(
                state,
                actor,
                target,
                strength);
            const uint8_t mask = valid_mask(state, actor);
            const Action action = imitation_action(
                target, mask, exploration, rng);
            const ActionResult result = apply_action(state, actor, action);
            ++summary.decisions;
            if (result == ActionResult::RoundEnd) {
                student.advance_imitation_state(state);
            }
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
        }
        }
        summary.roots = root + 1;
        if (report_every && summary.roots % report_every == 0) {
            std::cerr
                << "{\"imitation_root\":" << summary.roots
                << ",\"imitation_decisions\":" << summary.decisions
                << ",\"buckets\":" << student.stats().buckets;
            if (summary.reference_buckets) {
                std::cerr
                    << ",\"covered_buckets\":" << summary.covered_buckets
                    << ",\"reference_buckets\":" << summary.reference_buckets
                    << ",\"coverage\":"
                    << summary.covered_buckets /
                        static_cast<double>(summary.reference_buckets);
            }
            std::cerr << "}\n";
        }
        if (summary.reference_buckets &&
            summary.covered_buckets == summary.reference_buckets) break;
    }
    return summary;
}

template <typename Solver>
void play_round(
    State& state,
    int mccfr_seat,
    Solver& solver,
    int iterations,
    int start_street,
    int belief_seat = -1,
    BeliefBR* belief_agent = nullptr,
    HeuristicVariant heuristic_variant = HeuristicVariant::Baseline) {
    if (state.players[0].all_in || state.players[1].all_in || state.terminal) return;
    int actor = state.actor;
    while (!state.terminal) {
        Action action;
        if (actor == mccfr_seat && state.street >= start_street) {
            action = solver.choose(state, actor, iterations);
        } else if (actor == belief_seat && belief_agent) {
            action = belief_agent->choose(state, actor);
        } else {
            action = heuristic_variant_action(state, actor, heuristic_variant);
        }
        const ActionResult result = apply_action(state, actor, action);
        if (result == ActionResult::RoundEnd || result == ActionResult::FoldEnd) break;
        actor = 1 - actor;
        state.actor = actor;
    }
}

double settle(const State& state, int target_seat) {
    std::array<int, 2> award{};
    if (state.players[0].folded) {
        award[1] = state.pot;
    } else if (state.players[1].folded) {
        award[0] = state.pot;
    } else {
        const int matched = std::min(state.players[0].invested, state.players[1].invested);
        award[0] = state.players[0].invested - matched;
        award[1] = state.players[1].invested - matched;
        const int contested = 2 * matched;
        const Score left = best_hand(all_cards(state.players[0]));
        const Score right = best_hand(all_cards(state.players[1]));
        if (left > right) award[0] += contested;
        else if (right > left) award[1] += contested;
        else {
            award[0] += contested / 2 + contested % 2;
            award[1] += contested / 2;
        }
    }
    return (award[target_seat] - state.players[target_seat].invested) /
        static_cast<double>(state.ante);
}

template <typename Solver>
double play_hand(
    const std::vector<Card>& deck,
    int mccfr_seat,
    Solver& solver,
    int iterations,
    int ante,
    int start_street,
    int belief_seat = -1,
    BeliefBR* belief_agent = nullptr,
    int stack_ante = kEffectiveStackAnte,
    H4Policy* h4_policy = nullptr,
    int h4_seat = -1,
    HeuristicVariant heuristic_variant = HeuristicVariant::Baseline) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * stack_ante;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    prepare_initial_cards(
        state,
        deck,
        cursor,
        belief_seat,
        belief_agent,
        h4_policy,
        h4_seat);

    for (int seat = 0; seat < 2; ++seat) state.players[seat].shown.push_back(deck[cursor++]);
    for (int street : {5, 6, 7}) {
        for (int seat = 0; seat < 2; ++seat) {
            if (street == 7) state.players[seat].hidden.push_back(deck[cursor++]);
            else state.players[seat].shown.push_back(deck[cursor++]);
        }
        reset_round(state, street);
        play_round(
            state,
            mccfr_seat,
            solver,
            iterations,
            start_street,
            belief_seat,
            belief_agent,
            heuristic_variant);
        if (state.terminal) break;
    }
    return settle(state, mccfr_seat);
}

template <typename SolverA, typename SolverB>
double play_hand_match(
    const std::vector<Card>& deck,
    int seat_a,
    SolverA& solver_a,
    SolverB& solver_b,
    int ante,
    int start_street,
    int stack_ante = kEffectiveStackAnte) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * stack_ante;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    prepare_initial_cards(state, deck, cursor);
    for (int seat = 0; seat < 2; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);
    }
    for (int street : {5, 6, 7}) {
        for (int seat = 0; seat < 2; ++seat) {
            if (street == 7) state.players[seat].hidden.push_back(deck[cursor++]);
            else state.players[seat].shown.push_back(deck[cursor++]);
        }
        reset_round(state, street);
        if (!state.players[0].all_in && !state.players[1].all_in) {
            int actor = state.actor;
            while (!state.terminal) {
                const Action action = actor == seat_a
                    ? solver_a.choose(state, actor, 0)
                    : solver_b.choose(state, actor, 0);
                const ActionResult result = apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd || result == ActionResult::FoldEnd) break;
                actor = 1 - actor;
                state.actor = actor;
            }
        }
        if (state.terminal) break;
    }
    return settle(state, seat_a);
}

std::pair<int, int> h4_choice(
    const std::vector<Card>& cards,
    int action) {
    const H4View view = make_h4_view(cards);
    const auto [discard, reveal] = h4_positions(action);
    return {
        view.original_index[discard],
        view.original_index[reveal]};
}

double h4_continuation_value(
    const std::vector<Card>& deck,
    const std::array<std::pair<int, int>, 2>& choices,
    int target_seat,
    MCCFR& continuation,
    int ante,
    int stack_ante) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * stack_ante;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    for (int round = 0; round < 4; ++round) {
        for (int seat = 0; seat < 2; ++seat) {
            state.players[seat].hidden.push_back(deck[cursor++]);
        }
    }
    apply_h4_choices(state, choices);
    for (int seat = 0; seat < 2; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);
    }
    for (int street : {5, 6, 7}) {
        for (int seat = 0; seat < 2; ++seat) {
            const Card card = deck[cursor++];
            if (street == 7) state.players[seat].hidden.push_back(card);
            else state.players[seat].shown.push_back(card);
        }
        reset_round(state, street);
        if (!state.players[0].all_in && !state.players[1].all_in) {
            while (!state.terminal) {
                const int actor = state.actor;
                const Action action = continuation.choose(state, actor, 0);
                const ActionResult result = apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd ||
                    result == ActionResult::FoldEnd) break;
                state.actor = 1 - actor;
            }
        }
        if (state.terminal) break;
    }
    return settle(state, target_seat);
}

void train_h4_root(
    H4CFR& h4,
    MCCFR& continuation,
    const std::vector<Card>& deck,
    int ante,
    int stack_ante) {
    std::array<std::vector<Card>, 2> cards;
    size_t cursor = 0;
    for (int round = 0; round < 4; ++round) {
        for (int seat = 0; seat < 2; ++seat) {
            cards[seat].push_back(deck[cursor++]);
        }
    }
    for (int traverser = 0; traverser < 2; ++traverser) {
        const int opponent = 1 - traverser;
        const int opponent_action = h4.sample_current(cards[opponent]);
        std::array<double, kH4ActionCount> values{};
        for (int action = 0; action < kH4ActionCount; ++action) {
            std::array<std::pair<int, int>, 2> choices;
            choices[traverser] = h4_choice(cards[traverser], action);
            choices[opponent] = h4_choice(
                cards[opponent], opponent_action);
            values[action] = h4_continuation_value(
                deck,
                choices,
                traverser,
                continuation,
                ante,
                stack_ante);
        }
        h4.update(cards[traverser], values);
    }
}

void train_h4_q_root(
    H4QPolicy& h4,
    MCCFR& continuation,
    const std::vector<Card>& deck,
    int ante,
    int stack_ante) {
    std::array<std::vector<Card>, 2> cards;
    size_t cursor = 0;
    for (int round = 0; round < 4; ++round) {
        for (int seat = 0; seat < 2; ++seat) {
            cards[seat].push_back(deck[cursor++]);
        }
    }
    for (int traverser = 0; traverser < 2; ++traverser) {
        const int opponent = 1 - traverser;
        const auto opponent_choice = discard_reveal(cards[opponent]);
        std::array<double, kH4ActionCount> values{};
        for (int action = 0; action < kH4ActionCount; ++action) {
            std::array<std::pair<int, int>, 2> choices;
            choices[traverser] = h4_choice(cards[traverser], action);
            choices[opponent] = opponent_choice;
            values[action] = h4_continuation_value(
                deck,
                choices,
                traverser,
                continuation,
                ante,
                stack_ante);
        }
        h4.update(cards[traverser], values);
    }
}

template <typename TargetPolicy>
void begin_target_subgame(TargetPolicy&, const State&, int) {}

void begin_target_subgame(
    ExactSeventhResolverPolicy& target,
    const State& state,
    int target_seat) {
    target.begin_subgame(state, target_seat);
}

template <typename TargetPolicy>
double play_hand_policy_lbr(
    const std::vector<Card>& deck,
    int lbr_seat,
    TargetPolicy& target,
    PolicyLBR<TargetPolicy>& lbr,
    int ante,
    int stack_ante = kEffectiveStackAnte,
    H4Policy* h4_policy = nullptr,
    std::array<std::vector<PowerVector>, 3>* power_samples = nullptr,
    int power_sample_limit = 128,
    size_t power_sample_cap = 0) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * stack_ante;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    const int target_seat = 1 - lbr_seat;
    prepare_initial_cards(
        state, deck, cursor, -1, nullptr, h4_policy, target_seat);
    for (int seat = 0; seat < 2; ++seat) {
        state.players[seat].shown.push_back(deck[cursor++]);
    }
    lbr.reset();
    for (int street : {5, 6, 7}) {
        for (int seat = 0; seat < 2; ++seat) {
            if (street == 7) {
                state.players[seat].hidden.push_back(deck[cursor++]);
            } else {
                state.players[seat].shown.push_back(deck[cursor++]);
            }
        }
        reset_round(state, street);
        if (street == 7) {
            begin_target_subgame(target, state, target_seat);
        }
        if (power_samples) {
            auto& street_samples = (*power_samples)[street - 5];
            for (int seat = 0;
                 seat < 2 && street_samples.size() < power_sample_cap;
                 ++seat) {
                street_samples.push_back(
                    power_vector(state, seat, power_sample_limit));
            }
        }
        if (!state.players[0].all_in && !state.players[1].all_in) {
            int actor = state.actor;
            while (!state.terminal) {
                Action action;
                if (actor == target_seat) {
                    action = target.choose(state, actor, 0);
                    lbr.observe(state, target_seat, action);
                } else {
                    action = lbr.choose(state, lbr_seat);
                }
                const ActionResult result =
                    apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd ||
                    result == ActionResult::FoldEnd) {
                    break;
                }
                actor = 1 - actor;
                state.actor = actor;
            }
        }
        if (state.terminal) break;
    }
    return settle(state, lbr_seat);
}

std::array<std::vector<PowerVector>, 3> collect_power_samples(
    int hands,
    uint64_t seed,
    int ante,
    int sample_limit,
    size_t sample_cap,
    int stack_ante = kEffectiveStackAnte,
    MCCFR* rollout_policy = nullptr,
    double policy_epsilon = 0.0,
    bool one_sample_per_player_street = false,
    bool heuristic_pool = false) {
    std::array<std::vector<PowerVector>, 3> samples;
    std::mt19937_64 rng(seed);
    for (int hand = 0; hand < hands; ++hand) {
        if (std::all_of(samples.begin(), samples.end(), [&](const auto& street) {
                return street.size() >= sample_cap;
            })) {
            break;
        }
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        std::array<HeuristicVariant, 2> variants{
            HeuristicVariant::Baseline,
            HeuristicVariant::Baseline};
        if (heuristic_pool) {
            for (HeuristicVariant& variant : variants) {
                variant = kHeuristicPool[
                    std::uniform_int_distribution<size_t>(
                        0, kHeuristicPool.size() - 1)(rng)];
            }
        }
        State state;
        state.ante = ante;
        state.effective_stack = ante * stack_ante;
        state.pot = ante * 2;
        for (Player& player : state.players) player.invested = ante;
        size_t cursor = 0;
        prepare_initial_cards(state, deck, cursor);
        for (int seat = 0; seat < 2; ++seat) state.players[seat].shown.push_back(deck[cursor++]);

        for (int street : {5, 6, 7}) {
            for (int seat = 0; seat < 2; ++seat) {
                if (street == 7) state.players[seat].hidden.push_back(deck[cursor++]);
                else state.players[seat].shown.push_back(deck[cursor++]);
            }
            reset_round(state, street);
            if (one_sample_per_player_street) {
                auto& street_samples = samples[street - 5];
                for (int seat = 0;
                     seat < 2 && street_samples.size() < sample_cap;
                     ++seat) {
                    street_samples.push_back(
                        power_vector(state, seat, sample_limit));
                }
            }
            if (state.players[0].all_in || state.players[1].all_in) continue;
            int actor = state.actor;
            while (!state.terminal) {
                auto& street_samples = samples[street - 5];
                if (!one_sample_per_player_street &&
                    street_samples.size() < sample_cap) {
                    street_samples.push_back(power_vector(state, actor, sample_limit));
                }
                Action action = rollout_policy
                    ? rollout_policy->choose(state, actor, 0)
                    : heuristic_variant_action(state, actor, variants[actor]);
                if (policy_epsilon > 0.0 &&
                    std::bernoulli_distribution(policy_epsilon)(rng)) {
                    const uint8_t mask = valid_mask(state, actor);
                    std::array<Action, kActionCount> legal{};
                    int count = 0;
                    for (int candidate = 0; candidate < kActionCount; ++candidate) {
                        if (mask & (1u << candidate)) {
                            legal[count++] = static_cast<Action>(candidate);
                        }
                    }
                    action = legal[std::uniform_int_distribution<int>(0, count - 1)(rng)];
                }
                const ActionResult result = apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd || result == ActionResult::FoldEnd) break;
                actor = 1 - actor;
                state.actor = actor;
            }
            if (state.terminal) break;
        }
    }
    return samples;
}

template <typename TargetPolicy>
std::array<std::vector<PowerVector>, 3> collect_lbr_power_samples(
    int max_hands,
    uint64_t seed,
    int ante,
    int stack_ante,
    int sample_limit,
    size_t sample_cap,
    int particles,
    TargetPolicy& target) {
    std::array<std::vector<PowerVector>, 3> samples;
    PolicyLBR<TargetPolicy> lbr(
        target, particles, seed ^ 0x94d049bb133111ebull);
    std::mt19937_64 rng(seed);
    for (int hand = 0; hand < max_hands; ++hand) {
        if (std::all_of(samples.begin(), samples.end(), [&](const auto& street) {
                return street.size() >= sample_cap;
            })) {
            break;
        }
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        play_hand_policy_lbr(
            deck,
            hand & 1,
            target,
            lbr,
            ante,
            stack_ante,
            nullptr,
            &samples,
            sample_limit,
            sample_cap);
    }
    return samples;
}

ActionRangeModel fit_action_range_model(
    int hands,
    uint64_t seed,
    int ante,
    int stack_ante,
    int sample_limit,
    MCCFR& teacher) {
    ActionRangeModel model;
    std::mt19937_64 rng(seed);
    for (int hand = 0; hand < hands; ++hand) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        State state = sample_fifth_street_root(deck, ante, stack_ante);
        while (!state.terminal) {
            const int actor = state.actor;
            const Action action = teacher.choose(state, actor, 0);
            model.observe(state, actor, action, sample_limit);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) {
                teacher.advance_imitation_state(state);
            } else if (result != ActionResult::FoldEnd) {
                state.actor = 1 - actor;
            }
        }
    }
    model.finalize();
    return model;
}

struct Options {
    int hands = 200;
    int iterations = 16;
    int ante = 1;
    int stack_ante = kEffectiveStackAnte;
    int report_every = 0;
    int start_street = 7;
    int clusters = 64;
    std::array<int, 3> clusters_by_street{0, 0, 0};
    int fit_hands = 0;
    int fit_range_hands = 0;
    int power_samples = 128;
    int fit_sample_cap = 50000;
    int fit_lbr_sample_cap = 0;
    int fit_heuristic_sample_cap = 0;
    int fit_lbr_particles = 64;
    int belief_particles = 240;
    int seventh_resolve_iterations = 0;
    int seventh_posterior_particles = 0;
    int split_rollouts = 12;
    int split_min_strong = 2;
    int split_reservoir = 32;
    int split_max_depth = 8;
    int split_max_leaves = 512;
    uint64_t split_min_touches = 0;
    uint64_t split_gap_touch_reference = 100000;
    uint64_t power_cache_max = 1000000;
    int soft_top_k = 1;
    int asymp_batch_roots = 1;
    int temperature_min_samples = 100;
    int cluster_growth_steps = 0;
    uint64_t root_iterations = 0;
    uint64_t root_node_budget = 0;
    uint64_t root_report_every = 0;
    uint64_t soft_adapt_every = 0;
    uint64_t temperature_calibration_roots = 0;
    uint64_t imitation_roots = 0;
    uint64_t imitation_report_every = 0;
    uint64_t h4_train_roots = 0;
    uint64_t h4_q_train_roots = 0;
    uint64_t h4_report_every = 0;
    uint64_t h4_min_touches = 8;
    uint64_t h4_q_min_samples = 32;
    uint64_t diagnostic_roots = 0;
    uint64_t diagnostic_min_samples = 8;
    uint64_t seed = 7;
    uint64_t prune_after = 0;
    uint64_t prune_refresh = 64;
    double prune_threshold = 50.0;
    double belief_sigma = 0.18;
    double belief_margin = 0.40;
    double asymp_step = 0.0005;
    double asymp_mu = 0.01;
    double progress_seconds = 30.0;
    double cluster_growth_threshold = 0.0;
    double cluster_growth_threshold_decay = 1.0;
    double cluster_growth_threshold_min = 0.0;
    double soft_top_p = 0.0;
    double soft_temperature = 0.1;
    double initial_fold_regret = 0.0;
    double initial_other_regret = 0.0;
    double seventh_resolve_prior = 100.0;
    double split_audit_probability = 0.10;
    double split_gap_threshold = 0.04;
    double split_gap_threshold_min = 0.0;
    double split_confidence_z = 1.645;
    double split_reward_gain = 0.20;
    double fit_policy_epsilon = 0.0;
    double fit_heuristic_epsilon = 0.0;
    double imitation_strength = 10.0;
    double imitation_prior_scale = 1.0;
    double imitation_exploration = 0.0;
    double h4_q_lcb_beta = 1.96;
    bool regret_plus = true;
    bool asymp = false;
    bool baseline_match = false;
    bool self_test = false;
    bool agent_stdio = false;
    bool imitation_external_sampling = false;
    bool soft_local_bandwidth = false;
    bool soft_adapt_round_robin = false;
    bool inspect_model = false;
    bool fit_heuristic_pool = false;
    bool seventh_hand_history = false;
    std::string bucket = "legacy";
    std::string split_audit = "none";
    std::string fit_clustering = "kmeans";
    std::string fit_lbr_target = "model";
    std::string soft_growth = "none";
    std::string opponent = "heuristic";
    std::string lbr_target = "model";
    int made_min_category = 1;
    std::string opponent_model;
    std::string imitation_mode = "regret";
    std::string initial_regret_policy = "fold";
    std::string imitation_model_path;
    std::string imitation_atlas_path;
    std::string imitation_cover_model_path;
    std::string init_from;
    std::string load_path;
    std::string save_path;
    std::string load_atlas_path;
    std::string save_atlas_path;
    std::string load_range_path;
    std::string save_range_path;
    std::string fit_policy_model_path;
    std::string fit_policy_atlas_path;
    std::string fit_lbr_model_path;
    std::string fit_lbr_atlas_path;
    std::string load_temperatures_path;
    std::string save_temperatures_path;
    std::string load_h4_path;
    std::string save_h4_path;
    std::string load_h4_q_path;
    std::string save_h4_q_path;
    std::string dump_bucket_touches_path;
    std::string dump_conflict_path;
    std::string dump_lbr_attribution_path;
    std::vector<std::string> merge_paths;
};

SoftGrowthMode soft_growth_mode(const std::string& value) {
    if (value == "fixed") return SoftGrowthMode::Fixed;
    if (value == "mix") return SoftGrowthMode::Mix;
    if (value == "simple") return SoftGrowthMode::Simple;
    if (value == "point") return SoftGrowthMode::Point;
    if (value == "residual") return SoftGrowthMode::Residual;
    return SoftGrowthMode::None;
}

InitialRegretPolicy initial_regret_policy(const std::string& value) {
    return value == "made-call"
        ? InitialRegretPolicy::MadeCall
        : InitialRegretPolicy::Fold;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto need_value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hands") options.hands = std::stoi(need_value(i));
        else if (arg == "--iterations") options.iterations = std::stoi(need_value(i));
        else if (arg == "--ante") options.ante = std::stoi(need_value(i));
        else if (arg == "--stack-ante") options.stack_ante = std::stoi(need_value(i));
        else if (arg == "--report-every") options.report_every = std::stoi(need_value(i));
        else if (arg == "--start-street") options.start_street = std::stoi(need_value(i));
        else if (arg == "--clusters") options.clusters = std::stoi(need_value(i));
        else if (arg == "--clusters-by-street") {
            const std::string value = need_value(i);
            std::stringstream stream(value);
            std::string part;
            for (int street = 0; street < 3; ++street) {
                if (!std::getline(stream, part, ',') || part.empty()) {
                    throw std::runtime_error(
                        "--clusters-by-street requires K5,K6,K7");
                }
                options.clusters_by_street[street] = std::stoi(part);
            }
            if (std::getline(stream, part, ',')) {
                throw std::runtime_error(
                    "--clusters-by-street requires exactly three values");
            }
        }
        else if (arg == "--fit-hands") options.fit_hands = std::stoi(need_value(i));
        else if (arg == "--fit-range-hands") {
            options.fit_range_hands = std::stoi(need_value(i));
        }
        else if (arg == "--power-samples") options.power_samples = std::stoi(need_value(i));
        else if (arg == "--fit-sample-cap") options.fit_sample_cap = std::stoi(need_value(i));
        else if (arg == "--fit-lbr-sample-cap") {
            options.fit_lbr_sample_cap = std::stoi(need_value(i));
        }
        else if (arg == "--fit-heuristic-sample-cap") {
            options.fit_heuristic_sample_cap = std::stoi(need_value(i));
        }
        else if (arg == "--fit-lbr-particles") {
            options.fit_lbr_particles = std::stoi(need_value(i));
        }
        else if (arg == "--belief-particles") options.belief_particles = std::stoi(need_value(i));
        else if (arg == "--seventh-resolve-iterations") {
            options.seventh_resolve_iterations = std::stoi(need_value(i));
        }
        else if (arg == "--seventh-posterior-particles") {
            options.seventh_posterior_particles = std::stoi(need_value(i));
        }
        else if (arg == "--split-audit") {
            options.split_audit = need_value(i);
        }
        else if (arg == "--split-audit-probability") {
            options.split_audit_probability = std::stod(need_value(i));
        }
        else if (arg == "--split-rollouts") {
            options.split_rollouts = std::stoi(need_value(i));
        }
        else if (arg == "--split-gap-threshold") {
            options.split_gap_threshold = std::stod(need_value(i));
        }
        else if (arg == "--split-gap-threshold-min") {
            options.split_gap_threshold_min = std::stod(need_value(i));
        }
        else if (arg == "--split-gap-touch-reference") {
            options.split_gap_touch_reference = std::stoull(need_value(i));
        }
        else if (arg == "--split-confidence-z") {
            options.split_confidence_z = std::stod(need_value(i));
        }
        else if (arg == "--split-min-strong") {
            options.split_min_strong = std::stoi(need_value(i));
        }
        else if (arg == "--split-reward-gain") {
            options.split_reward_gain = std::stod(need_value(i));
        }
        else if (arg == "--split-reservoir") {
            options.split_reservoir = std::stoi(need_value(i));
        }
        else if (arg == "--split-max-depth") {
            options.split_max_depth = std::stoi(need_value(i));
        }
        else if (arg == "--split-max-leaves") {
            options.split_max_leaves = std::stoi(need_value(i));
        }
        else if (arg == "--split-min-touches") {
            options.split_min_touches = std::stoull(need_value(i));
        }
        else if (arg == "--power-cache-max") {
            options.power_cache_max = std::stoull(need_value(i));
        }
        else if (arg == "--soft-top-k") options.soft_top_k = std::stoi(need_value(i));
        else if (arg == "--asymp-batch-roots") {
            options.asymp_batch_roots = std::stoi(need_value(i));
        }
        else if (arg == "--soft-top-p") options.soft_top_p = std::stod(need_value(i));
        else if (arg == "--initial-fold-regret") {
            options.initial_fold_regret = std::stod(need_value(i));
        }
        else if (arg == "--initial-other-regret") {
            options.initial_other_regret = std::stod(need_value(i));
        }
        else if (arg == "--seventh-resolve-prior") {
            options.seventh_resolve_prior = std::stod(need_value(i));
        }
        else if (arg == "--fit-policy-epsilon") {
            options.fit_policy_epsilon = std::stod(need_value(i));
        }
        else if (arg == "--fit-heuristic-epsilon") {
            options.fit_heuristic_epsilon = std::stod(need_value(i));
        }
        else if (arg == "--initial-regret-policy") {
            options.initial_regret_policy = need_value(i);
        }
        else if (arg == "--imitation-strength") {
            options.imitation_strength = std::stod(need_value(i));
        }
        else if (arg == "--imitation-prior-scale") {
            options.imitation_prior_scale = std::stod(need_value(i));
        }
        else if (arg == "--imitation-exploration") {
            options.imitation_exploration = std::stod(need_value(i));
        }
        else if (arg == "--h4-q-lcb-beta") {
            options.h4_q_lcb_beta = std::stod(need_value(i));
        }
        else if (arg == "--temperature-min-samples") {
            options.temperature_min_samples = std::stoi(need_value(i));
        }
        else if (arg == "--cluster-growth-steps") {
            options.cluster_growth_steps = std::stoi(need_value(i));
        }
        else if (arg == "--root-iterations") options.root_iterations = std::stoull(need_value(i));
        else if (arg == "--root-node-budget") {
            options.root_node_budget = std::stoull(need_value(i));
        }
        else if (arg == "--root-report-every") options.root_report_every = std::stoull(need_value(i));
        else if (arg == "--soft-adapt-every") {
            options.soft_adapt_every = std::stoull(need_value(i));
        }
        else if (arg == "--temperature-calibration-roots") {
            options.temperature_calibration_roots = std::stoull(need_value(i));
        }
        else if (arg == "--imitation-roots") {
            options.imitation_roots = std::stoull(need_value(i));
        }
        else if (arg == "--imitation-report-every") {
            options.imitation_report_every = std::stoull(need_value(i));
        }
        else if (arg == "--h4-train-roots") {
            options.h4_train_roots = std::stoull(need_value(i));
        }
        else if (arg == "--h4-q-train-roots") {
            options.h4_q_train_roots = std::stoull(need_value(i));
        }
        else if (arg == "--h4-report-every") {
            options.h4_report_every = std::stoull(need_value(i));
        }
        else if (arg == "--h4-min-touches") {
            options.h4_min_touches = std::stoull(need_value(i));
        }
        else if (arg == "--h4-q-min-samples") {
            options.h4_q_min_samples = std::stoull(need_value(i));
        }
        else if (arg == "--diagnostic-roots") {
            options.diagnostic_roots = std::stoull(need_value(i));
        }
        else if (arg == "--diagnostic-min-samples") {
            options.diagnostic_min_samples = std::stoull(need_value(i));
        }
        else if (arg == "--seed") options.seed = std::stoull(need_value(i));
        else if (arg == "--prune-after") options.prune_after = std::stoull(need_value(i));
        else if (arg == "--prune-threshold") options.prune_threshold = std::stod(need_value(i));
        else if (arg == "--prune-refresh") options.prune_refresh = std::stoull(need_value(i));
        else if (arg == "--bucket") options.bucket = need_value(i);
        else if (arg == "--fit-clustering") {
            options.fit_clustering = need_value(i);
        }
        else if (arg == "--fit-lbr-target") {
            options.fit_lbr_target = need_value(i);
        }
        else if (arg == "--soft-growth") options.soft_growth = need_value(i);
        else if (arg == "--opponent") options.opponent = need_value(i);
        else if (arg == "--lbr-target") options.lbr_target = need_value(i);
        else if (arg == "--made-min-category") {
            options.made_min_category = std::stoi(need_value(i));
        }
        else if (arg == "--opponent-model") options.opponent_model = need_value(i);
        else if (arg == "--imitation-mode") options.imitation_mode = need_value(i);
        else if (arg == "--imitate-from") options.imitation_model_path = need_value(i);
        else if (arg == "--imitate-atlas") options.imitation_atlas_path = need_value(i);
        else if (arg == "--imitation-cover-model") {
            options.imitation_cover_model_path = need_value(i);
        }
        else if (arg == "--belief-sigma") options.belief_sigma = std::stod(need_value(i));
        else if (arg == "--belief-margin") options.belief_margin = std::stod(need_value(i));
        else if (arg == "--asymp-step") options.asymp_step = std::stod(need_value(i));
        else if (arg == "--asymp-mu") options.asymp_mu = std::stod(need_value(i));
        else if (arg == "--progress-seconds") {
            options.progress_seconds = std::stod(need_value(i));
        }
        else if (arg == "--cluster-growth-threshold") {
            options.cluster_growth_threshold = std::stod(need_value(i));
        }
        else if (arg == "--cluster-growth-threshold-decay") {
            options.cluster_growth_threshold_decay =
                std::stod(need_value(i));
        }
        else if (arg == "--cluster-growth-threshold-min") {
            options.cluster_growth_threshold_min =
                std::stod(need_value(i));
        }
        else if (arg == "--soft-temperature") options.soft_temperature = std::stod(need_value(i));
        else if (arg == "--algorithm") {
            const std::string value = need_value(i);
            if (value == "mccfr-plus") {
                options.regret_plus = true;
                options.asymp = false;
            } else if (value == "mccfr") {
                options.regret_plus = false;
                options.asymp = false;
            } else if (value == "asymp") {
                options.regret_plus = false;
                options.asymp = true;
            } else {
                throw std::runtime_error("--algorithm must be mccfr, mccfr-plus, or asymp");
            }
        } else if (arg == "--load") options.load_path = need_value(i);
        else if (arg == "--init-from") options.init_from = need_value(i);
        else if (arg == "--save") options.save_path = need_value(i);
        else if (arg == "--merge") options.merge_paths.push_back(need_value(i));
        else if (arg == "--load-atlas") options.load_atlas_path = need_value(i);
        else if (arg == "--save-atlas") options.save_atlas_path = need_value(i);
        else if (arg == "--load-range") options.load_range_path = need_value(i);
        else if (arg == "--save-range") options.save_range_path = need_value(i);
        else if (arg == "--fit-policy-model") {
            options.fit_policy_model_path = need_value(i);
        }
        else if (arg == "--fit-policy-atlas") {
            options.fit_policy_atlas_path = need_value(i);
        }
        else if (arg == "--fit-lbr-model") {
            options.fit_lbr_model_path = need_value(i);
        }
        else if (arg == "--fit-lbr-atlas") {
            options.fit_lbr_atlas_path = need_value(i);
        }
        else if (arg == "--load-temperatures") {
            options.load_temperatures_path = need_value(i);
        }
        else if (arg == "--save-temperatures") {
            options.save_temperatures_path = need_value(i);
        }
        else if (arg == "--load-h4") options.load_h4_path = need_value(i);
        else if (arg == "--save-h4") options.save_h4_path = need_value(i);
        else if (arg == "--load-h4-q") options.load_h4_q_path = need_value(i);
        else if (arg == "--save-h4-q") options.save_h4_q_path = need_value(i);
        else if (arg == "--dump-bucket-touches") {
            options.dump_bucket_touches_path = need_value(i);
        }
        else if (arg == "--dump-conflict") {
            options.dump_conflict_path = need_value(i);
        }
        else if (arg == "--dump-lbr-attribution") {
            options.dump_lbr_attribution_path = need_value(i);
        }
        else if (arg == "--agent-stdio") options.agent_stdio = true;
        else if (arg == "--imitation-external-sampling") {
            options.imitation_external_sampling = true;
        }
        else if (arg == "--baseline-match") options.baseline_match = true;
        else if (arg == "--soft-local-bandwidth") options.soft_local_bandwidth = true;
        else if (arg == "--soft-adapt-round-robin") {
            options.soft_adapt_round_robin = true;
        }
        else if (arg == "--self-test") options.self_test = true;
        else if (arg == "--inspect-model") options.inspect_model = true;
        else if (arg == "--fit-heuristic-pool") {
            options.fit_heuristic_pool = true;
        }
        else if (arg == "--seventh-hand-history") {
            options.seventh_hand_history = true;
        }
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (options.hands <= 0 || options.hands % 2 != 0) {
        throw std::runtime_error("--hands must be a positive even number");
    }
    if (options.iterations < 0 || options.ante <= 0 || options.stack_ante <= 0) {
        throw std::runtime_error(
            "--iterations must be nonnegative; --ante and --stack-ante positive");
    }
    if (options.seventh_resolve_iterations < 0 ||
        options.seventh_posterior_particles < 0 ||
        options.seventh_resolve_prior < 0.0) {
        throw std::runtime_error(
            "seventh resolver iterations/prior/particles must be nonnegative");
    }
    if (options.seventh_resolve_iterations > 0 &&
        options.opponent != "policy-lbr" &&
        !options.agent_stdio) {
        throw std::runtime_error(
            "seventh resolver requires --opponent policy-lbr or --agent-stdio");
    }
    if (options.split_audit != "none" &&
        options.split_audit != "random") {
        throw std::runtime_error("--split-audit must be random");
    }
    if (options.split_audit == "random" &&
        (options.bucket != "power-tree" || options.asymp ||
         (!options.root_iterations && !options.root_node_budget))) {
        throw std::runtime_error(
            "split audit requires --bucket power-tree and MCCFR root training");
    }
    if (options.split_audit == "random" &&
        (options.save_path.empty() || options.save_atlas_path.empty())) {
        throw std::runtime_error(
            "split audit requires both --save and --save-atlas");
    }
    if (!(options.split_audit_probability > 0.0 &&
          options.split_audit_probability <= 1.0) ||
        options.split_rollouts <= 0 ||
        options.split_gap_threshold < 0.0 ||
        options.split_gap_threshold_min < 0.0 ||
        options.split_gap_threshold_min > options.split_gap_threshold ||
        options.split_gap_touch_reference == 0 ||
        options.split_confidence_z < 0.0 ||
        options.split_min_strong <= 0 ||
        options.split_reservoir < 2 ||
        options.split_min_strong > options.split_reservoir ||
        options.split_reward_gain < 0.0 ||
        options.split_reward_gain > 1.0 ||
        options.split_max_depth <= 0 ||
        options.split_max_depth > std::numeric_limits<uint16_t>::max() ||
        options.split_max_leaves <= 0 ||
        options.split_max_leaves > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error("invalid split-audit parameter");
    }
    if (options.start_street < 5 || options.start_street > 7) {
        throw std::runtime_error("--start-street must be 5, 6, or 7");
    }
    if ((options.root_iterations || options.root_node_budget) &&
        options.start_street != 5) {
        throw std::runtime_error(
            "root training requires --start-street 5");
    }
    if (options.root_iterations && options.root_node_budget) {
        throw std::runtime_error(
            "use either --root-iterations or --root-node-budget");
    }
    if (options.diagnostic_roots &&
        (options.start_street != 5 || options.dump_conflict_path.empty())) {
        throw std::runtime_error(
            "conflict diagnostics require --start-street 5 and --dump-conflict");
    }
    if (!options.dump_lbr_attribution_path.empty() &&
        options.opponent != "policy-lbr") {
        throw std::runtime_error(
            "--dump-lbr-attribution requires --opponent policy-lbr");
    }
    if (options.soft_growth != "none" &&
        options.soft_growth != "fixed" &&
        options.soft_growth != "mix" &&
        options.soft_growth != "simple" &&
        options.soft_growth != "point" &&
        options.soft_growth != "residual") {
        throw std::runtime_error(
            "--soft-growth must be none, fixed, mix, simple, point, or residual");
    }
    if (options.soft_growth == "fixed" &&
        (options.bucket != "power" ||
         options.soft_top_p <= 0.0 ||
         !options.soft_local_bandwidth ||
         (!options.root_iterations && !options.root_node_budget) ||
         options.asymp)) {
        throw std::runtime_error(
            "fixed soft training requires power buckets, --soft-top-p, "
            "--soft-local-bandwidth, and root training");
    }
    if ((options.soft_growth == "mix" ||
         options.soft_growth == "simple") &&
        (options.bucket != "power" ||
         options.soft_top_p <= 0.0 ||
         !options.soft_local_bandwidth ||
         options.soft_adapt_every == 0 ||
         (!options.root_iterations && !options.root_node_budget) ||
         options.asymp ||
         options.save_path.empty() ||
         options.save_atlas_path.empty() ||
         options.save_temperatures_path.empty())) {
        throw std::runtime_error(
            "adaptive soft growth requires power buckets, --soft-top-p, "
            "--soft-local-bandwidth, root training, --soft-adapt-every, "
            "and all three save paths");
    }
    if ((options.soft_growth == "point" ||
         options.soft_growth == "residual") &&
        (options.bucket != "power" ||
         options.soft_top_p <= 0.0 ||
         !options.soft_local_bandwidth ||
         options.soft_adapt_every == 0 ||
         (!options.root_iterations && !options.root_node_budget) ||
         options.asymp ||
         options.save_path.empty() ||
         options.save_atlas_path.empty())) {
        throw std::runtime_error(
            "mass growth requires power buckets, --soft-top-p, "
            "--soft-local-bandwidth, root training, --soft-adapt-every, "
            "--save, and --save-atlas");
    }
    if (options.bucket != "legacy" &&
        options.bucket != "power" &&
        options.bucket != "power-memory" &&
        options.bucket != "power-memory16" &&
        options.bucket != "power-memory81" &&
        options.bucket != "power-recall" &&
        options.bucket != "power-range" &&
        options.bucket != "power-tree") {
        throw std::runtime_error(
            "--bucket must be legacy, power, power-memory, power-memory16, "
            "power-memory81, power-recall, power-range, or power-tree");
    }
    if (options.bucket == "power-tree" &&
        (options.soft_top_k != 1 || options.soft_top_p > 0.0 ||
         options.soft_local_bandwidth || options.soft_growth != "none" ||
         options.temperature_calibration_roots != 0 ||
         !options.load_temperatures_path.empty() ||
         !options.save_temperatures_path.empty())) {
        throw std::runtime_error(
            "power-tree is hard-only and does not support soft or temperature options");
    }
    if (options.bucket == "power-tree" &&
        (!options.merge_paths.empty() || options.imitation_roots != 0)) {
        throw std::runtime_error(
            "power-tree does not yet support worker merge or imitation");
    }
    if (!heuristic_opponent_name(options.opponent) &&
        options.opponent != "half" &&
        options.opponent != "belief-br" &&
        options.opponent != "policy-lbr") {
        throw std::runtime_error(
            "--opponent must be heuristic, bluff-defense, over-bluff, "
            "hidden-bluff, slow-play, half, belief-br, or policy-lbr");
    }
    if (options.lbr_target != "model" &&
        options.lbr_target != "fold" &&
        options.lbr_target != "made-call" &&
        options.lbr_target != "made-bet") {
        throw std::runtime_error(
            "--lbr-target must be model, fold, made-call, or made-bet");
    }
    if (options.made_min_category < 0 || options.made_min_category > 8) {
        throw std::runtime_error("--made-min-category must be in [0,8]");
    }
    if (options.lbr_target != "model" &&
        options.opponent != "policy-lbr") {
        throw std::runtime_error(
            "non-model --lbr-target requires --opponent policy-lbr");
    }
    if (options.clusters <= 0 ||
        std::any_of(
            options.clusters_by_street.begin(),
            options.clusters_by_street.end(),
            [](int count) { return count < 0; }) ||
        (options.clusters_by_street != std::array<int, 3>{0, 0, 0} &&
         std::any_of(
             options.clusters_by_street.begin(),
             options.clusters_by_street.end(),
             [](int count) { return count == 0; })) ||
        options.fit_hands < 0 ||
        options.fit_range_hands < 0 ||
        options.power_samples <= 0 || options.fit_sample_cap <= 0 ||
        options.fit_lbr_sample_cap < 0 ||
        options.fit_heuristic_sample_cap < 0 ||
        options.fit_lbr_particles <= 0 ||
        options.belief_particles <= 0 || options.belief_sigma <= 0) {
        throw std::runtime_error("power-atlas counts must be positive");
    }
    if (options.prune_threshold < 0) {
        throw std::runtime_error("--prune-threshold must be nonnegative");
    }
    if (options.initial_fold_regret < 0.0) {
        throw std::runtime_error("--initial-fold-regret must be nonnegative");
    }
    if (options.initial_other_regret < 0.0) {
        throw std::runtime_error("--initial-other-regret must be nonnegative");
    }
    if (options.fit_policy_epsilon < 0.0 ||
        options.fit_policy_epsilon > 1.0) {
        throw std::runtime_error("--fit-policy-epsilon must be in [0,1]");
    }
    if (options.fit_heuristic_epsilon < 0.0 ||
        options.fit_heuristic_epsilon > 1.0) {
        throw std::runtime_error("--fit-heuristic-epsilon must be in [0,1]");
    }
    if (options.initial_regret_policy != "fold" &&
        options.initial_regret_policy != "made-call") {
        throw std::runtime_error(
            "--initial-regret-policy must be fold or made-call");
    }
    if (options.fit_clustering != "kmeans" &&
        options.fit_clustering != "hard-em") {
        throw std::runtime_error(
            "--fit-clustering must be kmeans or hard-em");
    }
    if (options.fit_lbr_target != "model" &&
        options.fit_lbr_target != "heuristic") {
        throw std::runtime_error(
            "--fit-lbr-target must be model or heuristic");
    }
    if (options.imitation_strength <= 0.0) {
        throw std::runtime_error("--imitation-strength must be positive");
    }
    if (options.imitation_prior_scale < 0.0) {
        throw std::runtime_error("--imitation-prior-scale must be nonnegative");
    }
    if (options.imitation_mode != "regret" &&
        options.imitation_mode != "prior") {
        throw std::runtime_error("--imitation-mode must be regret or prior");
    }
    if (options.imitation_exploration < 0.0 ||
        options.imitation_exploration > 1.0) {
        throw std::runtime_error("--imitation-exploration must be in [0,1]");
    }
    if (options.fit_policy_model_path.empty() !=
        options.fit_policy_atlas_path.empty()) {
        throw std::runtime_error(
            "--fit-policy-model and --fit-policy-atlas must be used together");
    }
    if (options.fit_heuristic_pool &&
        !options.fit_policy_model_path.empty()) {
        throw std::runtime_error(
            "--fit-heuristic-pool cannot be combined with --fit-policy-model");
    }
    if (options.fit_lbr_model_path.empty() !=
        options.fit_lbr_atlas_path.empty()) {
        throw std::runtime_error(
            "--fit-lbr-model and --fit-lbr-atlas must be used together");
    }
    if (options.fit_lbr_sample_cap > 0 &&
        options.fit_lbr_target == "model" &&
        options.fit_lbr_model_path.empty()) {
        throw std::runtime_error(
            "model --fit-lbr-target requires --fit-lbr-model and --fit-lbr-atlas");
    }
    if (options.bucket == "power-range") {
        if (options.asymp || options.start_street != 5) {
            throw std::runtime_error(
                "power-range currently requires 5th-street MCCFR");
        }
        if (options.load_range_path.empty() && options.fit_range_hands == 0) {
            throw std::runtime_error(
                "power-range requires --load-range or --fit-range-hands");
        }
        if (options.fit_range_hands > 0 &&
            (options.imitation_model_path.empty() ||
             options.imitation_atlas_path.empty() ||
             options.save_range_path.empty())) {
            throw std::runtime_error(
                "range fitting requires --imitate-from, --imitate-atlas, "
                "and --save-range");
        }
    } else if (!options.load_range_path.empty() ||
               !options.save_range_path.empty() ||
               options.fit_range_hands > 0) {
        throw std::runtime_error(
            "range options require --bucket power-range");
    }
    if (!options.imitation_cover_model_path.empty() &&
        !options.imitation_roots) {
        throw std::runtime_error(
            "--imitation-cover-model requires --imitation-roots");
    }
    if (options.imitation_roots &&
        (options.imitation_model_path.empty() ||
         options.imitation_atlas_path.empty() ||
         !options.load_path.empty() ||
         (options.bucket != "power" &&
          options.bucket != "power-memory" &&
          options.bucket != "power-memory16" &&
          options.bucket != "power-memory81" &&
          options.bucket != "power-recall" &&
          options.bucket != "power-range") ||
         options.start_street != 5 ||
         options.asymp)) {
        throw std::runtime_error(
            "imitation requires --imitate-from, --imitate-atlas, a fresh "
            "5th-street hard power, power-memory, power-memory16, power-memory81, power-recall, or power-range model, "
            "and non-AsymP training");
    }
    if (options.h4_train_roots &&
        (options.load_path.empty() ||
         options.save_h4_path.empty() ||
         options.start_street != 5 ||
         options.asymp)) {
        throw std::runtime_error(
            "H4 training requires --load, --save-h4, --start-street 5, "
            "and non-AsymP training");
    }
    if (!options.load_h4_path.empty() && options.asymp) {
        throw std::runtime_error("H4 policy is not implemented for AsymP");
    }
    if (options.h4_q_min_samples == 0 || options.h4_q_lcb_beta < 0.0) {
        throw std::runtime_error(
            "--h4-q-min-samples must be positive and LCB beta nonnegative");
    }
    if ((!options.load_h4_path.empty() && !options.load_h4_q_path.empty()) ||
        (options.h4_train_roots && options.h4_q_train_roots)) {
        throw std::runtime_error("choose either H4 CFR or H4 Q, not both");
    }
    if (options.h4_q_train_roots &&
        (options.load_path.empty() ||
         options.save_h4_q_path.empty() ||
         options.start_street != 5 ||
         options.asymp)) {
        throw std::runtime_error(
            "H4 Q training requires --load, --save-h4-q, --start-street 5, "
            "and non-AsymP training");
    }
    if (!options.load_h4_q_path.empty() && options.asymp) {
        throw std::runtime_error("H4 Q policy is not implemented for AsymP");
    }
    if (options.soft_top_k <= 0 ||
        options.soft_top_p < 0.0 ||
        options.soft_top_p > 1.0 ||
        options.soft_temperature <= 0.0) {
        throw std::runtime_error(
            "--soft-top-k/temperature must be positive and --soft-top-p in [0,1]");
    }
    if (options.temperature_min_samples <= 0) {
        throw std::runtime_error(
            "--temperature-min-samples must be positive");
    }
    if (options.cluster_growth_steps < 0 ||
        options.cluster_growth_threshold < 0.0) {
        throw std::runtime_error(
            "--cluster-growth-steps/threshold must be nonnegative");
    }
    if (options.cluster_growth_threshold_decay <= 0.0 ||
        options.cluster_growth_threshold_decay > 1.0 ||
        options.cluster_growth_threshold_min < 0.0 ||
        options.cluster_growth_threshold_min >
            options.cluster_growth_threshold) {
        throw std::runtime_error(
            "growth threshold decay must be in (0,1] and minimum in "
            "[0, threshold]");
    }
    if (options.cluster_growth_steps > 0 &&
        options.temperature_calibration_roots == 0) {
        throw std::runtime_error(
            "--cluster-growth-steps requires --temperature-calibration-roots");
    }
    if (options.cluster_growth_steps > 0 &&
        (options.save_path.empty() ||
         options.save_atlas_path.empty() ||
         options.save_temperatures_path.empty())) {
        throw std::runtime_error(
            "cluster growth requires --save, --save-atlas, and --save-temperatures");
    }
    if ((options.soft_top_k > 1 || options.soft_top_p > 0.0) &&
        options.bucket != "power" &&
        options.bucket != "power-recall") {
        throw std::runtime_error(
            "--soft-top-k > 1 requires --bucket power or power-recall");
    }
    if (options.asymp &&
        (options.soft_top_k > 1 || options.soft_top_p > 0.0)) {
        throw std::runtime_error("soft cluster policy is not implemented for AsymP");
    }
    if (options.asymp &&
        (options.bucket == "power-memory" ||
         options.bucket == "power-memory16" ||
         options.bucket == "power-memory81")) {
        throw std::runtime_error("power-memory modes are not implemented for AsymP");
    }
    if (options.temperature_calibration_roots &&
        (options.load_path.empty() ||
         options.bucket != "power" ||
         (options.soft_top_k <= 1 && options.soft_top_p <= 0.0) ||
         !options.soft_local_bandwidth ||
         options.start_street != 5 ||
         options.asymp)) {
        throw std::runtime_error(
            "temperature calibration requires a loaded, 5th-street, "
            "local-soft power MCCFR model");
    }
    if ((!options.load_temperatures_path.empty() ||
         !options.save_temperatures_path.empty()) &&
        options.bucket != "power") {
        throw std::runtime_error(
            "temperature files currently require --bucket power");
    }
    if (options.asymp && (options.asymp_step <= 0 || options.asymp_mu < 0)) {
        throw std::runtime_error("--asymp-step must be positive and --asymp-mu nonnegative");
    }
    if (options.asymp_batch_roots <= 0) {
        throw std::runtime_error("--asymp-batch-roots must be positive");
    }
    if (options.progress_seconds < 0) {
        throw std::runtime_error("--progress-seconds must be nonnegative");
    }
    if (options.asymp && options.load_path.empty() && options.init_from.empty()) {
        throw std::runtime_error("--algorithm asymp requires --load or --init-from");
    }
    if (!options.asymp && !options.init_from.empty()) {
        throw std::runtime_error("--init-from requires --algorithm asymp");
    }
    if (!options.asymp && !options.opponent_model.empty()) {
        throw std::runtime_error("--opponent-model currently requires --algorithm asymp");
    }
    if (options.asymp && options.iterations != 0) {
        throw std::runtime_error("--algorithm asymp currently requires --iterations 0");
    }
    if (options.asymp && options.root_node_budget != 0) {
        throw std::runtime_error(
            "--root-node-budget is not implemented for AsymP");
    }
    if (options.agent_stdio && options.load_path.empty()) {
        throw std::runtime_error("--agent-stdio requires --load");
    }
    if (options.opponent == "policy-lbr" &&
        ((options.lbr_target == "model" &&
          options.load_path.empty() && options.init_from.empty()) ||
         options.start_street != 5 ||
         options.iterations != 0)) {
        throw std::runtime_error(
            "policy-lbr requires --start-street 5, --iterations 0, and "
            "--load when --lbr-target=model");
    }
    if (!options.merge_paths.empty() &&
        (options.load_path.empty() || options.save_path.empty() ||
         options.root_iterations != 0 || options.root_node_budget != 0 ||
         options.regret_plus || options.asymp)) {
        throw std::runtime_error(
            "--merge requires --algorithm mccfr, --load, --save, and no root training");
    }
    return options;
}

void self_test() {
    const std::array<Card, 5> royal = {
        Card{14, 0}, Card{13, 0}, Card{12, 0}, Card{11, 0}, Card{10, 0}};
    assert(evaluate_five(royal)[0] == 8);

    const std::vector<Card> five = {
        Card{14, 0}, Card{14, 1}, Card{9, 2}, Card{7, 3}, Card{2, 0}};
    const Card sixth{9, 1};
    const Card seventh{9, 3};
    std::vector<Card> six = five;
    six.push_back(sixth);
    std::vector<Card> seven = six;
    seven.push_back(seventh);
    assert(best_hand(seven) == best_hand_direct(seven));
    assert(best_after_one_card(five, sixth, best_hand_encoded(five)) ==
        best_hand_encoded(six));
    const uint32_t one_sixth =
        best_after_one_card(five, sixth, best_hand_encoded(five));
    const uint32_t one_seventh =
        best_after_one_card(five, seventh, best_hand_encoded(five));
    assert(best_after_two_cards(
        five, sixth, seventh, std::max(one_sixth, one_seventh)) ==
        best_hand_encoded(seven));
    assert(best_after_one_card(six, seventh, best_hand_encoded(six)) ==
        best_hand_encoded(seven));

    State state;
    state.ante = 1;
    state.effective_stack = 1000;
    state.pot = 2;
    state.street = 5;
    const uint8_t initial = valid_mask(state, 0);
    assert(initial & (1u << CHECK));
    assert(initial & (1u << BBING));
    assert(initial & (1u << QUARTER));
    assert(apply_action(state, 0, CHECK) == ActionResult::Continue);
    state.actor = 1;
    const uint8_t after_check = valid_mask(state, 0);
    assert(!(after_check & (1u << BBING)));

    State audit_scale_state;
    audit_scale_state.ante = 1000;
    audit_scale_state.effective_stack = 1000000;
    audit_scale_state.pot = 2000;
    audit_scale_state.players[0].invested = 1000;
    audit_scale_state.players[1].invested = 1000;
    audit_scale_state.players[1].folded = true;
    audit_scale_state.terminal = true;
    assert(std::abs(terminal_net_search(audit_scale_state, 0) - 1.0) < 1e-12);
    assert(std::abs(normalized_split_audit_terminal(
        audit_scale_state, 0, 1000000.0) - 0.001) < 1e-12);

    State made_state;
    made_state.ante = 1;
    made_state.effective_stack = 1000;
    made_state.pot = 2;
    made_state.street = 5;
    made_state.players[0].invested = 1;
    made_state.players[1].invested = 1;
    made_state.players[0].hidden = {Card{14, 0}, Card{14, 1}};
    made_state.players[0].shown = {
        Card{9, 2}, Card{7, 3}, Card{2, 0}};
    ConditionalParticipationPolicy fold_policy("fold", 1);
    ConditionalParticipationPolicy call_policy("made-call", 1);
    ConditionalParticipationPolicy bet_policy("made-bet", 1);
    HalfPolicy half_policy;
    assert(fold_policy.choose(made_state, 0, 0) == FOLD);
    assert(call_policy.choose(made_state, 0, 0) == CHECK);
    assert(bet_policy.choose(made_state, 0, 0) == BBING);
    assert(half_policy.choose(made_state, 0, 0) == HALF);
    made_state.highest_bet = 1;
    made_state.players[1].round_bet = 1;
    assert(call_policy.choose(made_state, 0, 0) == CALL);

    State bluff_state = made_state;
    bluff_state.highest_bet = 0;
    bluff_state.players[1].round_bet = 0;
    bluff_state.players[0].hidden = {Card{2, 0}, Card{5, 1}};
    bluff_state.players[0].shown = {
        Card{7, 2}, Card{9, 3}, Card{11, 0}};
    bluff_state.players[1].shown = {
        Card{2, 1}, Card{6, 2}, Card{10, 3}};
    const double public_strength =
        opponent_public_completion_strength(bluff_state, 0);
    assert(public_strength >= 0.0 && public_strength <= 1.0);
    assert(public_strength == opponent_public_completion_strength(bluff_state, 0));
    assert(heuristic_variant_action(
        bluff_state, 0, HeuristicVariant::OverBluff) == HALF);
    assert(heuristic_variant_action(
        bluff_state, 0, HeuristicVariant::HiddenBluff) ==
        heuristic_action(bluff_state, 0));
    bluff_state.street = 7;
    assert(heuristic_variant_action(
        bluff_state, 0, HeuristicVariant::HiddenBluff) == HALF);
    bluff_state.highest_bet = 4;
    bluff_state.players[1].round_bet = 4;
    bluff_state.pot = 6;
    assert(!aggressive(heuristic_variant_action(
        bluff_state, 0, HeuristicVariant::OverBluff)));

    State slow_state = made_state;
    slow_state.street = 6;
    slow_state.highest_bet = 0;
    slow_state.players[1].round_bet = 0;
    slow_state.players[0].shown[0] = Card{14, 2};
    assert(aggressive(heuristic_action(slow_state, 0)));
    assert(heuristic_variant_action(
        slow_state, 0, HeuristicVariant::SlowPlay) == CHECK);

    State defense_state = bluff_state;
    defense_state.street = 5;
    defense_state.players[1].shown = {
        Card{14, 0}, Card{14, 1}, Card{13, 2}};
    assert(heuristic_action(defense_state, 0) == FOLD);
    assert(heuristic_variant_action(
        defense_state, 0, HeuristicVariant::BluffDefense) ==
        (bluff_defense_call(defense_state, 0) ? CALL : FOLD));

    RegretNode seeded_node;
    const uint8_t seeded_mask =
        (1u << FOLD) | (1u << CHECK) | (1u << HALF);
    initialize_regrets(seeded_node, seeded_mask, FOLD, 100.0, 1.0);
    assert(seeded_node.regrets[FOLD] == 100.0);
    assert(seeded_node.raw_regrets[CHECK] == 1.0);
    assert(seeded_node.regrets[HALF] == 1.0);
    assert(seeded_node.regrets[CALL] == 0.0);

    RegretNode averaging_node;
    averaging_node.regrets[CHECK] = 100.0;
    averaging_node.strategy_sum[CHECK] = 1.0;
    averaging_node.strategy_sum[FOLD] = 9.0;
    const uint8_t averaging_mask = (1u << CHECK) | (1u << FOLD);
    assert(current_strategy(averaging_node, averaging_mask)[CHECK] == 1.0);
    const auto averaged = average_strategy(averaging_node, averaging_mask);
    assert(std::abs(averaged[CHECK] - 0.1) < 1e-12);
    assert(std::abs(averaged[FOLD] - 0.9) < 1e-12);
    averaging_node.raw_regrets[CHECK] = -7.0;
    averaging_node.raw_regrets[FOLD] = 3.0;
    averaging_node.touches = 11;
    const auto partitioned = partition_regret_node(averaging_node, 0.25);
    for (int action = 0; action < kActionCount; ++action) {
        assert(std::abs(
            partitioned[0].regrets[action] +
            partitioned[1].regrets[action] -
            averaging_node.regrets[action]) < 1e-12);
        assert(std::abs(
            partitioned[0].raw_regrets[action] +
            partitioned[1].raw_regrets[action] -
            averaging_node.raw_regrets[action]) < 1e-12);
        assert(std::abs(
            partitioned[0].strategy_sum[action] +
            partitioned[1].strategy_sum[action] -
            averaging_node.strategy_sum[action]) < 1e-12);
    }
    assert(partitioned[0].touches + partitioned[1].touches == 11);
    for (const RegretNode& child : partitioned) {
        const auto child_current = current_strategy(child, averaging_mask);
        const auto child_average = average_strategy(child, averaging_mask);
        assert(std::abs(child_current[CHECK] - 1.0) < 1e-12);
        assert(std::abs(child_average[CHECK] - 0.1) < 1e-12);
        assert(std::abs(child_average[FOLD] - 0.9) < 1e-12);
    }

    MCCFR solver(true, 3, 5);
    std::mt19937_64 rng(3);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    const double result = play_hand(deck, 0, solver, 2, 1, 5);
    assert(std::isfinite(result));
    assert(solver.traversals() > 0);

    const auto samples = collect_power_samples(100, 19, 1, 32, 1000);
    const auto exploratory_samples =
        collect_power_samples(10, 20, 1, 8, 100, kEffectiveStackAnte, nullptr, 1.0);
    assert(!exploratory_samples[0].empty());
    const auto pool_samples = collect_power_samples(
        10, 24, 1, 8, 100, kEffectiveStackAnte,
        nullptr, 0.0, false, true);
    assert(!pool_samples[0].empty());
    PowerAtlas atlas(32);
    atlas.fit(samples, 4, 19);
    assert(atlas.clusters(5) == 4);
    PowerAtlas street_atlas(32);
    street_atlas.fit(samples, std::array<int, 3>{2, 3, 4}, 19);
    assert(street_atlas.clusters(5) == 2);
    assert(street_atlas.clusters(6) == 3);
    assert(street_atlas.clusters(7) == 4);
    PowerAtlas hard_em_atlas(32);
    hard_em_atlas.fit_hard_em(samples, 4, 19);
    assert(hard_em_atlas.clusters(5) == 4);
    MCCFR power_solver(true, 23, 7, &atlas);
    std::shuffle(deck.begin(), deck.end(), rng);
    assert(std::isfinite(play_hand(deck, 0, power_solver, 2, 1, 7)));
    MCCFR root_solver(false, 27, 5, &atlas);
    const auto one_per_samples = collect_power_samples(
        10, 21, 1, 8, 4, kEffectiveStackAnte, nullptr, 0.2, true);
    assert(one_per_samples[0].size() == 4);
    const auto lbr_samples = collect_lbr_power_samples(
        2, 22, 1, kEffectiveStackAnte, 8, 2, 2, root_solver);
    assert(lbr_samples[0].size() == 2);
    MCCFR recall_solver(
        false, 28, 5, &atlas, 0, 50.0, 64, 1, 0.0, 0.1, false, true);
    MCCFR temperature_solver(
        false, 29, 5, &atlas, 0, 50.0, 64, 4, 0.0, 1.0, true);
    std::shuffle(deck.begin(), deck.end(), rng);
    const State root = sample_fifth_street_root(deck, 1);
    PowerAtlas tree_atlas = atlas;
    tree_atlas.enable_tree();
    assert(tree_atlas.tree_enabled());
    assert(tree_atlas.clusters(5) == tree_atlas.base_clusters(5));
    bool append_rejected = false;
    try {
        tree_atlas.append(root, root.actor);
    } catch (const std::runtime_error&) {
        append_rejected = true;
    }
    assert(append_rejected);
    bool soft_rejected = false;
    try {
        tree_atlas.weights(root, root.actor, 2, 0.0, 0.1, false);
    } catch (const std::runtime_error&) {
        soft_rejected = true;
    }
    assert(soft_rejected);
    const PowerVector base0 = tree_atlas.base_center(5, 0);
    const PowerVector base1 = tree_atlas.base_center(5, 1);
    const uint16_t base1_before = tree_atlas.assign_vector(5, base1);
    PowerVector split_left = base0;
    PowerVector split_right = base0;
    split_left[0] -= 1e-6;
    split_right[0] += 1e-6;
    const auto first_split = tree_atlas.split_leaf(
        5, tree_atlas.assign_vector(5, base0), split_left, split_right, 0.5);
    assert(first_split.parent_leaf == 0);
    assert(first_split.child_leaf == 4);
    assert(tree_atlas.assign_vector(5, split_left) == first_split.parent_leaf);
    assert(tree_atlas.assign_vector(5, split_right) == first_split.child_leaf);
    assert(tree_atlas.assign_vector(5, base1) == base1_before);
    assert(tree_atlas.record_leaf_touch(5, first_split.child_leaf) == 1);
    for (int touch = 0; touch < 7; ++touch) {
        tree_atlas.record_leaf_touch(5, first_split.child_leaf);
    }
    PowerVector nested_left = split_right;
    PowerVector nested_right = split_right;
    nested_left[1] -= 1e-7;
    nested_right[1] += 1e-7;
    const auto second_split = tree_atlas.split_leaf(
        5, first_split.child_leaf, nested_left, nested_right, 0.25);
    assert(second_split.parent_leaf == first_split.child_leaf);
    assert(second_split.child_leaf == 5);
    assert(second_split.child0_touches == 6);
    assert(second_split.child1_touches == 2);
    assert(tree_atlas.assign_vector(5, split_left) == first_split.parent_leaf);
    assert(tree_atlas.assign_vector(5, nested_left) == second_split.parent_leaf);
    assert(tree_atlas.assign_vector(5, nested_right) == second_split.child_leaf);
    assert(tree_atlas.assign_vector(5, base1) == base1_before);
    assert(tree_atlas.leaf_touches_since_split(
        5, second_split.parent_leaf) == 6);
    assert(tree_atlas.leaf_touches_since_split(
        5, second_split.child_leaf) == 2);
    assert(tree_atlas.record_leaf_touch(5, second_split.child_leaf) == 3);

    const std::string legacy_atlas_path = "power_tree_self_test_legacy.bin";
    const std::string tree_atlas_path = "power_tree_self_test_v4.bin";
    atlas.save(legacy_atlas_path);
    PowerAtlas promoted_atlas(32);
    promoted_atlas.load(legacy_atlas_path);
    promoted_atlas.enable_tree();
    assert(promoted_atlas.tree_enabled());
    assert(promoted_atlas.assign_vector(5, base1) == base1_before);
    tree_atlas.save(tree_atlas_path);
    PowerAtlas loaded_tree_atlas(32);
    loaded_tree_atlas.load(tree_atlas_path);
    assert(loaded_tree_atlas.tree_enabled());
    assert(loaded_tree_atlas.generation_id() == tree_atlas.generation_id());
    assert(loaded_tree_atlas.assign_vector(5, split_left) ==
           first_split.parent_leaf);
    assert(loaded_tree_atlas.assign_vector(5, nested_right) ==
           second_split.child_leaf);
    assert(loaded_tree_atlas.leaf_touches_since_split(
        5, second_split.child_leaf) == 3);

    PowerAtlas touch_gate_atlas = atlas;
    touch_gate_atlas.enable_tree();
    MCCFR touch_gate_solver(false, 43, 5, &touch_gate_atlas);
    touch_gate_solver.configure_split_audit(SplitAuditConfig{
        true, 1.0, 1, 0.0, 0.0, 1, 0.0, 2, 8, 64,
        std::numeric_limits<uint64_t>::max()});
    touch_gate_solver.train_root(root, 0);
    touch_gate_solver.train_root(root, 1);
    touch_gate_solver.process_split_audits();
    assert(touch_gate_solver.split_audit_stats().audits == 0);
    assert(touch_gate_solver.split_audit_stats().touch_deferred > 0);

    PowerAtlas deferred_atlas = atlas;
    deferred_atlas.enable_tree();
    MCCFR deferred_solver(false, 44, 5, &deferred_atlas);
    deferred_solver.configure_split_audit(SplitAuditConfig{
        true, 1.0, 1, 1000.0, 0.0, 1, 0.0, 2, 8, 64});
    const uint64_t deferred_generation = deferred_atlas.generation_id();
    deferred_solver.train_root(root, 0);
    assert(deferred_atlas.generation_id() == deferred_generation);
    assert(deferred_solver.pending_split_audits_.size() <= 2);
    deferred_solver.train_root(root, 1);
    assert(deferred_atlas.generation_id() == deferred_generation);
    deferred_solver.process_split_audits();
    assert(deferred_solver.split_audit_stats().audits > 0);
    assert(deferred_solver.split_audit_stats().audit_node_visits > 0);
    assert(deferred_solver.node_visits() >=
           deferred_solver.split_audit_stats().audit_node_visits);

    PowerAtlas partition_atlas = atlas;
    partition_atlas.enable_tree();
    MCCFR partition_solver(false, 45, 5, &partition_atlas);
    partition_solver.train_root(root, 0);
    partition_solver.train_root(root, 1);
    const int root_actor = root.actor;
    const auto policy_before_partition =
        partition_solver.policy(root, root_actor);
    const uint16_t partition_parent =
        partition_atlas.assign(root, root_actor);
    PowerVector partition_left = partition_atlas.vector(root, root_actor);
    PowerVector partition_right = partition_left;
    partition_right[0] += 1e-7;
    const auto partition_split = partition_atlas.split_leaf(
        5,
        partition_parent,
        partition_left,
        partition_right,
        0.25);
    const size_t buckets_before_partition = partition_solver.stats().buckets;
    const size_t partitioned_nodes = partition_solver.partition_regret_nodes(
        5,
        partition_split.parent_leaf,
        partition_split.child_leaf,
        0.25);
    assert(partitioned_nodes > 0);
    assert(partition_solver.stats().buckets ==
           buckets_before_partition + partitioned_nodes);
    const auto policy_after_partition =
        partition_solver.policy(root, root_actor);
    for (int action = 0; action < kActionCount; ++action) {
        assert(std::abs(
            policy_before_partition[action] -
            policy_after_partition[action]) < 1e-12);
    }
    const std::string partition_atlas_path =
        "power_tree_self_test_partition.atlas";
    const std::string partition_model_path =
        "power_tree_self_test_partition.bin";
    partition_atlas.save(partition_atlas_path);
    partition_solver.save(partition_model_path);
    PowerAtlas loaded_partition_atlas(32);
    loaded_partition_atlas.load(partition_atlas_path);
    MCCFR loaded_partition_solver(false, 46, 5, &loaded_partition_atlas);
    loaded_partition_solver.load(partition_model_path);
    bool generation_mismatch_rejected = false;
    try {
        MCCFR wrong_generation_solver(false, 47, 5, &promoted_atlas);
        wrong_generation_solver.load(partition_model_path);
    } catch (const std::runtime_error&) {
        generation_mismatch_rejected = true;
    }
    assert(generation_mismatch_rejected);
    std::remove(legacy_atlas_path.c_str());
    std::remove(tree_atlas_path.c_str());
    std::remove(partition_atlas_path.c_str());
    std::remove(partition_model_path.c_str());

    PowerAtlas scheduled_atlas = atlas;
    scheduled_atlas.enable_tree();
    MCCFR scheduled_solver(false, 48, 5, &scheduled_atlas);
    scheduled_solver.train_root(root, 0);
    scheduled_solver.train_root(root, 1);
    scheduled_solver.configure_split_audit(SplitAuditConfig{
        true, 1.0, 1, 0.0, 0.0, 2, 0.20, 4, 8, 64});
    const int scheduled_actor = root.actor;
    const uint8_t scheduled_mask = valid_mask(root, scheduled_actor);
    const std::vector<Action> scheduled_actions =
        actions_from_mask(scheduled_mask);
    assert(scheduled_actions.size() >= 2);
    const uint16_t scheduled_parent =
        scheduled_atlas.assign(root, scheduled_actor);
    PowerVector scheduled_left =
        scheduled_atlas.vector(root, scheduled_actor);
    PowerVector scheduled_right = scheduled_left;
    scheduled_left[0] -= 1e-3;
    scheduled_right[0] += 1e-3;
    const SplitAuditKey scheduled_key{
        5, scheduled_mask, scheduled_parent};
    for (int index = 0; index < 4; ++index) {
        LocalSplitSample sample;
        sample.feature = index < 2 ? scheduled_left : scheduled_right;
        sample.reward.fill(-1e6);
        sample.reward[scheduled_actions[0]] = index < 2 ? 1.0 : 0.0;
        sample.reward[scheduled_actions[1]] = index < 2 ? 0.0 : 1.0;
        scheduled_solver.record_split_audit_sample(
            scheduled_key, sample, true);
    }
    auto regret_mass = [&](uint16_t first, uint16_t second) {
        RegretNode total;
        for (const auto& [key, node] : scheduled_solver.nodes_) {
            if (key.abstraction != 8 || key.street != 5 ||
                (key.power_cluster != first &&
                 key.power_cluster != second)) {
                continue;
            }
            for (int action = 0; action < kActionCount; ++action) {
                total.regrets[action] += node.regrets[action];
                total.raw_regrets[action] += node.raw_regrets[action];
                total.strategy_sum[action] += node.strategy_sum[action];
            }
            total.touches += node.touches;
        }
        return total;
    };
    const RegretNode scheduled_mass_before =
        regret_mass(scheduled_parent, scheduled_parent);
    const uint16_t scheduled_child = static_cast<uint16_t>(
        scheduled_atlas.clusters(5));
    assert(scheduled_solver.process_split_audits());
    assert(scheduled_solver.split_audit_stats().accepted_splits == 1);
    const RegretNode scheduled_mass_after =
        regret_mass(scheduled_parent, scheduled_child);
    for (int action = 0; action < kActionCount; ++action) {
        assert(std::abs(
            scheduled_mass_before.regrets[action] -
            scheduled_mass_after.regrets[action]) < 1e-10);
        assert(std::abs(
            scheduled_mass_before.raw_regrets[action] -
            scheduled_mass_after.raw_regrets[action]) < 1e-10);
        assert(std::abs(
            scheduled_mass_before.strategy_sum[action] -
            scheduled_mass_after.strategy_sum[action]) < 1e-10);
    }
    assert(scheduled_mass_before.touches == scheduled_mass_after.touches);
    assert(scheduled_solver.split_audit_evidence_.find(scheduled_key) ==
           scheduled_solver.split_audit_evidence_.end());

    State seventh_root = root;
    for (int seat = 0; seat < 2; ++seat) {
        seventh_root.players[seat].shown.push_back(
            seventh_root.simulation_deck.back());
        seventh_root.simulation_deck.pop_back();
    }
    for (int seat = 0; seat < 2; ++seat) {
        seventh_root.players[seat].hidden.push_back(
            seventh_root.simulation_deck.back());
        seventh_root.simulation_deck.pop_back();
    }
    seventh_root.history.push_back(Event{5, 0, CHECK});
    reset_round(seventh_root, 7);
    State opponent_hidden_change = seventh_root;
    std::swap(
        opponent_hidden_change.players[1].hidden[0],
        opponent_hidden_change.players[1].hidden[1]);
    assert(make_exact_seventh_key(seventh_root, 0, true) ==
           make_exact_seventh_key(opponent_hidden_change, 0, true));
    State own_hidden_order = seventh_root;
    std::swap(
        own_hidden_order.players[0].hidden[0],
        own_hidden_order.players[0].hidden[1]);
    assert(make_exact_seventh_key(seventh_root, 0, false) ==
           make_exact_seventh_key(own_hidden_order, 0, false));
    assert(!(make_exact_seventh_key(seventh_root, 0, true) ==
             make_exact_seventh_key(own_hidden_order, 0, true)));
    State betting_history_change = seventh_root;
    betting_history_change.history[0].action = BBING;
    assert(!(make_exact_seventh_key(seventh_root, 0, true) ==
             make_exact_seventh_key(betting_history_change, 0, true)));
    ExactSeventhResolverPolicy exact_resolver(
        root_solver, 4, 10.0, true, 43);
    exact_resolver.begin_subgame(seventh_root, 0);
    const auto exact_policy = exact_resolver.policy(
        seventh_root, seventh_root.actor);
    assert(std::abs(std::accumulate(
        exact_policy.begin(), exact_policy.end(), 0.0) - 1.0) < 1e-12);
    assert(exact_resolver.stats().subgames == 1);
    assert(exact_resolver.stats().node_visits > 0);
    const auto root_tensor = deep_cfr_tensor(root, root.actor);
    State private_change = root;
    private_change.players[1 - root.actor].hidden[0].rank =
        private_change.players[1 - root.actor].hidden[0].rank == 14 ? 13 : 14;
    assert(root_tensor == deep_cfr_tensor(private_change, root.actor));
    State visible_change = root;
    visible_change.history.push_back(Event{
        5, static_cast<uint8_t>(root.actor), CHECK});
    assert(root_tensor != deep_cfr_tensor(visible_change, root.actor));
    assert(sample_fifth_street_root(deck, 1, 20).effective_stack == 20);
    MCCFR imitation_solver(false, 38, 5, &atlas);
    std::array<double, kActionCount> imitation_target{};
    double imitation_total = 0.0;
    const uint8_t imitation_mask = valid_mask(root, root.actor);
    for (int action = 0; action < kActionCount; ++action) {
        if (!(imitation_mask & (1u << action))) continue;
        imitation_target[action] = action + 1.0;
        imitation_total += imitation_target[action];
    }
    for (double& probability : imitation_target) probability /= imitation_total;
    imitation_solver.imitate_policy(root, root.actor, imitation_target, 100.0);
    const auto imitated = imitation_solver.policy(root, root.actor);
    for (int action = 0; action < kActionCount; ++action) {
        assert(std::abs(imitated[action] - imitation_target[action]) < 1e-12);
    }
    MCCFR prior_imitation_solver(false, 42, 5, &atlas);
    prior_imitation_solver.use_decaying_imitation_prior(true);
    prior_imitation_solver.imitate_policy(
        root, root.actor, imitation_target, 100.0);
    assert(prior_imitation_solver.scale_stats().regret_l1_total == 0.0);
    prior_imitation_solver.train_root(root, 0);
    const auto prior_imitated =
        prior_imitation_solver.policy(root, root.actor);
    assert(std::abs(std::accumulate(
        prior_imitated.begin(), prior_imitated.end(), 0.0) - 1.0) < 1e-12);
    State original_history = root;
    State alternate_history = root;
    original_history.history.push_back(Event{
        5, static_cast<uint8_t>(root.actor), BBING});
    alternate_history.history.push_back(Event{
        5, static_cast<uint8_t>(root.actor), QUARTER});
    assert(make_power_key(original_history, root.actor, atlas) ==
           make_power_key(alternate_history, root.actor, atlas));
    assert(!(make_power_key(original_history, root.actor, atlas, true) ==
             make_power_key(alternate_history, root.actor, atlas, true)));
    State previous_street = root;
    previous_street.street = 6;
    previous_street.history.clear();
    previous_street.history.push_back(Event{
        5, static_cast<uint8_t>(root.actor), BBING});
    assert(previous_street_aggression(previous_street, root.actor) == 1);
    previous_street.history.push_back(Event{
        5, static_cast<uint8_t>(1 - root.actor), HALF});
    assert(previous_street_aggression(previous_street, root.actor) == 3);
    const InfoKey memory_key = make_power_key(
        previous_street, root.actor, atlas, false, nullptr, true);
    assert(memory_key.abstraction == 5 && memory_key.rank_bucket == 3);
    previous_street.street = 7;
    previous_street.history.push_back(Event{
        6, static_cast<uint8_t>(root.actor), QUARTER});
    assert(cumulative_street_aggression(previous_street, root.actor) == 7);
    const InfoKey cumulative_memory_key = make_power_key(
        previous_street, root.actor, atlas, false, nullptr, 2);
    assert(cumulative_memory_key.abstraction == 6 &&
           cumulative_memory_key.rank_bucket == 7);
    assert(cumulative_street_aggression_strength(
               previous_street, root.actor) == 16);
    const InfoKey strength_memory_key = make_power_key(
        previous_street, root.actor, atlas, false, nullptr, 3);
    assert(strength_memory_key.abstraction == 7 &&
           strength_memory_key.rank_bucket == 16);
    const InfoKey recall_key =
        make_power_key(original_history, root.actor, atlas, true);
    assert(recall_key.abstraction == 3);
    assert(recall_key.category == 0 && recall_key.rank_bucket == 0);
    assert(recall_key.own_public_category == 0 &&
           recall_key.own_public_suit == 0);
    ActionRangeModel range_model;
    range_model.observe(root, root.actor, CHECK, 32);
    range_model.finalize();
    const InfoKey range_key =
        make_power_key(root, root.actor, atlas, false, &range_model);
    assert(range_key.abstraction == 4 && range_key.category < 64);
    const auto soft_weights =
        atlas.weights(root, root.actor, 4, 0.0, 0.1, false);
    assert(soft_weights.size() == 4);
    assert(std::abs(std::accumulate(
        soft_weights.begin(),
        soft_weights.end(),
        0.0,
        [](double total, const auto& entry) {
            return total + entry.second;
        }) - 1.0) < 1e-12);
    const auto local_weights =
        atlas.weights(root, root.actor, 4, 0.0, 1.0, true);
    assert(std::abs(std::accumulate(
        local_weights.begin(),
        local_weights.end(),
        0.0,
        [](double total, const auto& entry) {
            return total + entry.second;
        }) - 1.0) < 1e-12);
    const auto nucleus_weights =
        atlas.weights(root, root.actor, 4, 0.01, 1.0, true);
    assert(nucleus_weights.size() == 1);
    root_solver.train_root(root, 0);
    root_solver.train_root(root, 1);
    assert(root_solver.node_visits() > 0);
    const std::vector<Card> h4_cards = {
        Card{14, 0}, Card{14, 1}, Card{9, 2}, Card{4, 0}};
    const std::vector<Card> h4_suit_permutation = {
        Card{14, 2}, Card{14, 0}, Card{9, 3}, Card{4, 2}};
    assert(make_h4_view(h4_cards).key ==
           make_h4_view(h4_suit_permutation).key);
    H4CFR h4_solver(41);
    std::shuffle(deck.begin(), deck.end(), rng);
    train_h4_root(h4_solver, root_solver, deck, 1, 1000);
    assert(h4_solver.buckets() > 0);
    const auto h4_policy = h4_solver.average_policy(h4_cards);
    assert(std::abs(std::accumulate(
        h4_policy.begin(), h4_policy.end(), 0.0) - 1.0) < 1e-12);
    H4QPolicy h4_q_solver;
    h4_q_solver.set_min_samples(1);
    h4_q_solver.set_lcb_beta(0.0);
    std::array<double, kH4ActionCount> h4_q_values{};
    h4_q_values[5] = 1.0;
    h4_q_solver.update(h4_cards, h4_q_values);
    assert(h4_q_solver.choose(h4_cards) == h4_choice(h4_cards, 5));
    MCCFR external_imitation_solver(false, 39, 5, &atlas);
    const auto external_imitation = imitate_teacher_roots(
        external_imitation_solver,
        root_solver,
        1,
        10.0,
        1.0,
        1,
        1000,
        39,
        &root_solver,
        0,
        true);
    assert(external_imitation.roots == 1);
    assert(external_imitation.decisions > 0);
    assert(external_imitation.covered_buckets > 0);
    recall_solver.train_root(root, 0);
    assert(recall_solver.node_visits() > 0);
    temperature_solver.train_root(root, 0);
    temperature_solver.train_root(root, 1);
    temperature_solver.calibrate_temperature_root(root, 0);
    const auto temperature_summary =
        temperature_solver.apply_temperature_calibration(1);
    assert(temperature_summary.calibrated_clusters > 0);
    assert(temperature_summary.selected_loss <=
           temperature_summary.baseline_loss + 1e-12);
    PowerAtlas growth_atlas = atlas;
    MCCFR growth_solver(
        false, 30, 5, &growth_atlas, 0, 50.0, 64, 1, 0.99, 1.0, true);
    growth_solver.train_root(root, 0);
    growth_solver.train_root(root, 1);
    growth_solver.reset_temperature_calibration();
    growth_solver.calibrate_temperature_root(root, 0);
    growth_solver.calibrate_temperature_root(root, 1);
    growth_solver.apply_temperature_calibration(1);
    const size_t growth_clusters_before =
        growth_atlas.clusters(5) +
        growth_atlas.clusters(6) +
        growth_atlas.clusters(7);
    const auto growth = growth_solver.append_growth_cluster(-1.0, 1);
    assert(growth.added);
    assert(growth.initialized_nodes > 0);
    assert(
        growth_atlas.clusters(5) +
        growth_atlas.clusters(6) +
        growth_atlas.clusters(7) ==
        growth_clusters_before + 1);
    assert(std::abs(std::accumulate(
        growth.initial_strategy.begin(),
        growth.initial_strategy.end(),
        0.0) - 1.0) < 1e-12);
    PowerAtlas mix_atlas = atlas;
    MCCFR mix_solver(
        false, 34, 5, &mix_atlas, 0, 50.0, 64,
        1, 0.99, 1.0, true, false, SoftGrowthMode::Mix);
    mix_solver.train_root(root, 0);
    mix_solver.train_root(root, 1);
    const auto mix_adaptive = mix_solver.adapt_soft_clusters(-1.0, 1, 0);
    assert(mix_adaptive.growth.added);
    assert(mix_adaptive.growth.street == 5);
    assert(std::abs(std::accumulate(
        mix_adaptive.growth.initial_strategy.begin(),
        mix_adaptive.growth.initial_strategy.end(),
        0.0) - 1.0) < 1e-12);
    MCCFR fixed_solver(
        false, 36, 5, &atlas, 0, 50.0, 64,
        1, 0.99, 1.0, true, false, SoftGrowthMode::Fixed);
    fixed_solver.train_root(root, 0);
    assert(fixed_solver.node_visits() > 0);
    PowerAtlas simple_atlas = atlas;
    MCCFR simple_solver(
        false, 35, 5, &simple_atlas, 0, 50.0, 64,
        1, 0.99, 1.0, true, false, SoftGrowthMode::Simple);
    simple_solver.train_root(root, 0);
    simple_solver.train_root(root, 1);
    const auto simple_adaptive =
        simple_solver.adapt_soft_clusters(-1.0, 1);
    assert(simple_adaptive.growth.added);
    assert(std::count_if(
        simple_adaptive.growth.initial_strategy.begin(),
        simple_adaptive.growth.initial_strategy.end(),
        [](double probability) { return probability > 0.0; }) == 1);
    PowerAtlas residual_atlas = atlas;
    MCCFR residual_solver(
        false, 37, 5, &residual_atlas, 0, 50.0, 64,
        1, 0.99, 1.0, true, false, SoftGrowthMode::Residual);
    residual_solver.train_root(root, 0);
    residual_solver.train_root(root, 1);
    const auto residual_adaptive =
        residual_solver.adapt_soft_clusters(-1.0, 1, 0);
    assert(residual_adaptive.growth.added);
    assert(residual_adaptive.growth.street == 5);
    const uint8_t projection_mask =
        static_cast<uint8_t>((1u << CHECK) | (1u << BBING) | (1u << FOLD));
    const auto projection_start = uniform_strategy(projection_mask);
    std::array<double, kActionCount> projection_gradient{};
    projection_gradient[CHECK] = 1.0;
    const auto projected = projected_strategy(
        projection_start, projection_gradient, projection_mask, 0.1);
    assert(projected[CHECK] > projection_start[CHECK]);
    assert(std::abs(std::accumulate(projected.begin(), projected.end(), 0.0) - 1.0) < 1e-12);
    BucketAsymP asymp_solver(31, 5, &atlas, 0.0005, 0.01);
    asymp_solver.initialize(root_solver);
    asymp_solver.train_root(root);
    assert(asymp_solver.traversals() == 4);
    assert(asymp_solver.node_visits() > 0);
    BucketAsymP batched_asymp_solver(31, 5, &atlas, 0.0005, 0.01);
    batched_asymp_solver.initialize(root_solver);
    batched_asymp_solver.train_roots({root, root});
    assert(batched_asymp_solver.traversals() == 8);
    assert(batched_asymp_solver.node_visits() > asymp_solver.node_visits());
    MCCFR merge_worker = root_solver;
    merge_worker.train_root(root, 0);
    MCCFR merged_solver = root_solver;
    assert(merged_solver.merge_worker_delta(root_solver, merge_worker) > 0);
    assert(merged_solver.stats().buckets >= root_solver.stats().buckets);
    bool policy_found = false;
    const auto root_policy = root_solver.policy(root, root.actor, &policy_found);
    assert(policy_found);
    assert(std::abs(
        std::accumulate(root_policy.begin(), root_policy.end(), 0.0) - 1.0) < 1e-12);
    PolicyLBR<MCCFR> policy_lbr(root_solver, 8, 33);
    PolicyLBR<MCCFR> hidden_check(root_solver, 8, 33);
    State changed_hidden = root;
    const int hidden_target = 1 - root.actor;
    std::swap(
        changed_hidden.players[hidden_target].hidden[0],
        changed_hidden.players[hidden_target].discarded);
    const Action visible_action = policy_lbr.choose(root, root.actor);
    assert(visible_action ==
        hidden_check.choose(changed_hidden, root.actor));
    assert(valid_mask(root, root.actor) & (1u << visible_action));
    std::shuffle(deck.begin(), deck.end(), rng);
    assert(std::isfinite(play_hand_policy_lbr(
        deck, 0, root_solver, policy_lbr, 1)));
    assert(policy_lbr.stats().decisions > 0);
    const AgentRequest request = parse_agent_request(
        "ACT\t5th\t1\t2\t0\t1000\t1\t0\t1\t0"
        "\tsA,hK\tdQ,cJ,s10\th2\tc9,d8,s7\t-\t155");
    assert(request.state.street == 5);
    assert(request.state.players[0].hidden.size() == 2);
    assert(valid_mask(request.state, 0) == 155);
    const AgentRequest unequal_request = parse_agent_request(
        "ACT\t5th\t1\t2\t0\t20,50\t1\t0\t1\t0"
        "\tsA,hK\tdQ,cJ,s10\th2\tc9,d8,s7\t-\t155");
    assert(stack_cap(unequal_request.state, 0) == 20);
    assert(stack_cap(unequal_request.state, 1) == 50);
    BeliefBR belief_agent(16, 0.18, 0.40, 29);
    std::shuffle(deck.begin(), deck.end(), rng);
    assert(std::isfinite(play_hand(
        deck, 0, power_solver, 0, 1, 7, 1, &belief_agent)));
    std::cout << "{\"self_test\":\"ok\",\"buckets\":"
              << solver.stats().buckets
              << ",\"power_buckets\":" << power_solver.stats().buckets << "}\n";
}

void print_result(
    const Options& options,
    const MCCFR& solver,
    const std::vector<double>& paired,
    int wins,
    int ties,
    int losses,
    double elapsed) {
    const double mean = std::accumulate(paired.begin(), paired.end(), 0.0) / paired.size();
    double squared = 0.0;
    for (double value : paired) squared += (value - mean) * (value - mean);
    const double sample_variance = paired.size() > 1 ? squared / (paired.size() - 1) : 0.0;
    const double standard_error = std::sqrt(sample_variance / paired.size());
    const auto stats = solver.stats();
    const auto split_stats = solver.split_audit_stats();
    const auto split_leaves = solver.split_leaf_counts();
    const auto power_cache = solver.power_cache_stats();
    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"agent_a\": \""
              << (options.regret_plus ? "cpp-mccfr-plus" : "cpp-mccfr")
              << (options.iterations ? "-online" : "-table") << "\",\n"
              << "  \"agent_b\": \"" << options.opponent << "\",\n"
              << "  \"betting_rules_version\": 3,\n"
              << "  \"effective_stack_ante\": " << options.stack_ante << ",\n"
              << "  \"bucket\": \"" << options.bucket << "\",\n"
              << "  \"soft_top_k\": " << options.soft_top_k << ",\n"
              << "  \"soft_top_p\": " << options.soft_top_p << ",\n"
              << "  \"soft_temperature\": " << options.soft_temperature << ",\n"
              << "  \"soft_local_bandwidth\": "
              << (options.soft_local_bandwidth ? "true" : "false") << ",\n"
              << "  \"soft_growth\": \"" << options.soft_growth << "\",\n"
              << "  \"start_street\": " << options.start_street << ",\n"
              << "  \"prune_after\": " << options.prune_after << ",\n"
              << "  \"hands\": " << options.hands << ",\n"
              << "  \"iterations_per_decision\": " << options.iterations << ",\n"
              << "  \"root_training_iterations\": " << options.root_iterations << ",\n"
              << "  \"root_training_node_budget\": " << options.root_node_budget << ",\n"
              << "  \"average_profit_ante_for_a\": " << mean << ",\n"
              << "  \"paired_standard_error_ante\": " << standard_error << ",\n"
              << "  \"ci95_ante_for_a\": [" << mean - 1.96 * standard_error
              << ", " << mean + 1.96 * standard_error << "],\n"
              << "  \"wins_for_a\": " << wins << ",\n"
              << "  \"ties\": " << ties << ",\n"
              << "  \"losses_for_a\": " << losses << ",\n"
              << "  \"elapsed_seconds\": " << elapsed << ",\n"
              << "  \"hands_per_second\": " << options.hands / elapsed << ",\n"
              << "  \"traversals\": " << solver.traversals() << ",\n"
              << "  \"training_node_visits\": " << solver.node_visits() << ",\n"
              << "  \"power_assignment_cache\": {"
              << "\"entries\":" << power_cache.entries
              << ",\"limit\":" << power_cache.limit
              << ",\"hits\":" << power_cache.hits
              << ",\"misses\":" << power_cache.misses
              << ",\"resets\":" << power_cache.resets << "},\n"
              << "  \"split_audit\": {\n"
              << "    \"mode\": \"" << options.split_audit << "\",\n"
              << "    \"audits\": " << split_stats.audits << ",\n"
              << "    \"strong_audits\": " << split_stats.strong_audits << ",\n"
              << "    \"audit_node_visits\": "
              << split_stats.audit_node_visits << ",\n"
              << "    \"queued_states\": " << split_stats.queued_states << ",\n"
              << "    \"touch_deferred\": "
              << split_stats.touch_deferred << ",\n"
              << "    \"capacity_deferred\": "
              << split_stats.capacity_deferred << ",\n"
              << "    \"depth_deferred\": "
              << split_stats.depth_deferred << ",\n"
              << "    \"accepted_splits\": "
              << split_stats.accepted_splits << ",\n"
              << "    \"accepted_by_street\": ["
              << split_stats.accepted_by_street[0] << ", "
              << split_stats.accepted_by_street[1] << ", "
              << split_stats.accepted_by_street[2] << "],\n"
              << "    \"inherited_touches\": "
              << split_stats.inherited_touches << ",\n"
              << "    \"last_child_touches\": ["
              << split_stats.last_child_touches[0] << ", "
              << split_stats.last_child_touches[1] << "],\n"
              << "    \"rejected_splits\": "
              << split_stats.rejected_splits << ",\n"
              << "    \"evidence_rejected\": "
              << split_stats.evidence_rejected << ",\n"
              << "    \"reward_rejected\": "
              << split_stats.reward_rejected << ",\n"
              << "    \"regret_rejected\": "
              << split_stats.reward_rejected << ",\n"
              << "    \"gap_threshold_mean\": "
              << (split_stats.threshold_evaluations
                      ? split_stats.threshold_sum /
                          split_stats.threshold_evaluations
                      : 0.0)
              << ",\n"
              << "    \"gap_threshold_min_seen\": "
              << (split_stats.threshold_evaluations
                      ? split_stats.threshold_min
                      : 0.0)
              << ",\n"
              << "    \"gap_threshold_max_seen\": "
              << split_stats.threshold_max << ",\n"
              << "    \"leaves_by_street\": ["
              << split_leaves[0] << ", " << split_leaves[1] << ", "
              << split_leaves[2] << "]\n"
              << "  },\n"
              << "  \"bucket_growth\": {\n"
              << "    \"buckets\": " << stats.buckets << ",\n"
              << "    \"lookups\": " << stats.lookups << ",\n"
              << "    \"hits\": " << stats.hits << ",\n"
              << "    \"misses\": " << stats.misses << ",\n"
              << "    \"hit_rate\": " << stats.hit_rate << ",\n"
              << "    \"single_touch_buckets\": " << stats.singletons << ",\n"
              << "    \"single_touch_ratio\": " << stats.singleton_ratio << ",\n"
              << "    \"average_touches\": " << stats.average_touches << ",\n"
              << "    \"median_touches\": " << stats.median_touches << ",\n"
              << "    \"p95_touches\": " << stats.p95_touches << ",\n"
              << "    \"max_touches\": " << stats.max_touches << ",\n"
              << "    \"pruned_branches\": " << stats.pruned_branches << ",\n"
              << "    \"new_buckets_per_traversal\": "
              << stats.new_buckets_per_traversal << ",\n";
    auto print_counts = [](const auto& values) {
        std::cout << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << values[i];
        }
        std::cout << "]";
    };
    if (options.bucket == "power" ||
        options.bucket == "power-memory" ||
        options.bucket == "power-memory16" ||
        options.bucket == "power-memory81" ||
        options.bucket == "power-recall" ||
        options.bucket == "power-range" ||
        options.bucket == "power-tree") {
        std::cout
            << "    \"buckets_by_hand_category\": null,\n"
            << "    \"buckets_by_history_length\": ";
        if (options.bucket == "power-recall") print_counts(stats.by_history_length);
        else std::cout << "null";
        std::cout << ",\n    \"buckets_by_range_bucket\": ";
        if (options.bucket == "power-range") print_counts(stats.by_range_bucket);
        else std::cout << "null";
        std::cout << ",\n    \"buckets_by_previous_street_aggression\": ";
        if (options.bucket == "power-memory") {
            print_counts(stats.by_previous_street_aggression);
        } else {
            std::cout << "null";
        }
        std::cout << ",\n    \"buckets_by_cumulative_street_aggression\": ";
        if (options.bucket == "power-memory16") {
            print_counts(stats.by_cumulative_street_aggression);
        } else {
            std::cout << "null";
        }
        std::cout << ",\n    \"buckets_by_cumulative_street_strength\": ";
        if (options.bucket == "power-memory81") {
            print_counts(stats.by_cumulative_street_strength);
        } else {
            std::cout << "null";
        }
        std::cout << ",\n"
            << "    \"buckets_by_power_cluster_by_street\": [";
        for (size_t street = 0; street < stats.by_power_cluster.size(); ++street) {
            if (street) std::cout << ", ";
            print_counts(stats.by_power_cluster[street]);
        }
        std::cout << "],\n"
                  << "    \"buckets_by_own_bet_count\": ";
        print_counts(stats.by_own_bet_count);
        std::cout << ",\n    \"buckets_by_opponent_bet_count\": ";
        print_counts(stats.by_opponent_bet_count);
        std::cout << ",\n    \"buckets_by_checked\": ";
        print_counts(stats.by_checked);
        std::cout << ",\n    \"buckets_by_last_action_class\": ";
        print_counts(stats.by_last_action_class);
        std::cout << ",\n    \"buckets_by_betting_goal\": ";
        print_counts(stats.by_betting_goal);
        std::cout << ",\n    \"buckets_by_legal_action_count\": ";
        print_counts(stats.by_legal_action_count);
        std::cout << ",\n";
    } else {
        std::cout << "    \"buckets_by_hand_category\": ";
        print_counts(stats.by_hand_category);
        std::cout << ",\n    \"buckets_by_history_length\": ";
        print_counts(stats.by_history_length);
        std::cout << ",\n";
    }
    std::cout << "    \"buckets_by_street\": ";
    print_counts(stats.by_street);
    std::cout << "\n"
              << "  }\n"
              << "}\n";
}

int run_baseline_match(const Options& options) {
    HeuristicPolicy heuristic;
    BeliefBR belief(
        options.belief_particles,
        options.belief_sigma,
        options.belief_margin,
        options.seed ^ 0xd1b54a32d192ed03ull);
    std::mt19937_64 deal_rng(options.seed);
    std::vector<double> paired;
    paired.reserve(options.hands / 2);
    int wins = 0;
    int ties = 0;
    int losses = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int hand = 0; hand < options.hands; hand += 2) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const double first = play_hand(
            deck, 0, heuristic, 0, options.ante, 5, 1, &belief,
            options.stack_ante);
        const double second = play_hand(
            deck, 1, heuristic, 0, options.ante, 5, 0, &belief,
            options.stack_ante);
        paired.push_back((first + second) / 2.0);
        for (double value : {first, second}) {
            if (value > 0) ++wins;
            else if (value < 0) ++losses;
            else ++ties;
        }
        if (options.report_every > 0 &&
            (hand + 2) % options.report_every == 0) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::cerr << (hand + 2) << "/" << options.hands
                      << " hands, " << (hand + 2) / seconds << " hands/s\n";
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double mean =
        std::accumulate(paired.begin(), paired.end(), 0.0) / paired.size();
    double squared = 0.0;
    for (double value : paired) squared += (value - mean) * (value - mean);
    const double sample_variance =
        paired.size() > 1 ? squared / (paired.size() - 1) : 0.0;
    const double standard_error =
        std::sqrt(sample_variance / paired.size());
    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"agent_a\": \"heuristic\",\n"
              << "  \"agent_b\": \"belief-br\",\n"
              << "  \"betting_rules_version\": 3,\n"
              << "  \"hands\": " << options.hands << ",\n"
              << "  \"belief_particles\": " << options.belief_particles << ",\n"
              << "  \"belief_sigma\": " << options.belief_sigma << ",\n"
              << "  \"belief_margin\": " << options.belief_margin << ",\n"
              << "  \"average_profit_ante_for_a\": " << mean << ",\n"
              << "  \"paired_standard_error_ante\": " << standard_error << ",\n"
              << "  \"ci95_ante_for_a\": [" << mean - 1.96 * standard_error
              << ", " << mean + 1.96 * standard_error << "],\n"
              << "  \"wins_for_a\": " << wins << ",\n"
              << "  \"ties\": " << ties << ",\n"
              << "  \"losses_for_a\": " << losses << ",\n"
              << "  \"elapsed_seconds\": " << elapsed << ",\n"
              << "  \"hands_per_second\": " << options.hands / elapsed << "\n"
              << "}\n";
    return 0;
}

template <typename TargetPolicy>
int run_policy_lbr(
    const Options& options,
    TargetPolicy& target,
    const char* target_name,
    H4Policy* h4_policy = nullptr) {
    PolicyLBR<TargetPolicy> lbr(
        target,
        options.belief_particles,
        options.seed ^ 0x94d049bb133111ebull);
    struct Attribution {
        uint64_t decisions = 0;
        double gain = 0.0;
        double maximum_gain = 0.0;
    };
    std::unordered_map<InfoKey, Attribution, InfoKeyHash> attribution;
    if (!options.dump_lbr_attribution_path.empty()) {
        if constexpr (std::is_same_v<TargetPolicy, MCCFR>) {
            lbr.set_attribution_callback(
                [&](const State& state, int seat, double gain) {
                    Attribution& row = attribution[
                        target.information_key(state, seat)];
                    ++row.decisions;
                    row.gain += gain;
                    row.maximum_gain = std::max(row.maximum_gain, gain);
                });
        } else {
            throw std::runtime_error(
                "LBR attribution requires an MCCFR target model");
        }
    }
    std::mt19937_64 deal_rng(options.seed);
    std::vector<double> paired;
    paired.reserve(options.hands / 2);
    int wins = 0;
    int ties = 0;
    int losses = 0;
    const auto started = std::chrono::steady_clock::now();
    ProgressHeartbeat heartbeat(
        "policy-lbr", "evaluation", "hands",
        options.hands, options.progress_seconds);
    for (int hand = 0; hand < options.hands; hand += 2) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const double first = play_hand_policy_lbr(
            deck, 0, target, lbr, options.ante, options.stack_ante,
            h4_policy);
        const double second = play_hand_policy_lbr(
            deck, 1, target, lbr, options.ante, options.stack_ante,
            h4_policy);
        paired.push_back((first + second) / 2.0);
        for (double value : {first, second}) {
            if (value > 0) ++wins;
            else if (value < 0) ++losses;
            else ++ties;
        }
        const int completed = hand + 2;
        heartbeat.update(completed);
        if (options.report_every > 0 &&
            completed % options.report_every == 0) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const double rate = completed / std::max(1e-9, seconds);
            std::ostringstream output;
            output << "{\"solver\":\"policy-lbr\""
                   << ",\"phase\":\"evaluation\""
                   << ",\"hands_completed\":" << completed
                   << ",\"hands_total\":" << options.hands
                   << ",\"progress_percent\":"
                   << 100.0 * completed / options.hands
                   << ",\"hands_per_second\":" << rate
                   << ",\"elapsed_seconds\":" << seconds
                   << ",\"eta_seconds\":"
                   << (options.hands - completed) /
                       std::max(1e-9, rate)
                   << "}";
            write_progress(output.str());
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const double mean =
        std::accumulate(paired.begin(), paired.end(), 0.0) / paired.size();
    double squared = 0.0;
    for (double value : paired) squared += (value - mean) * (value - mean);
    const double sample_variance =
        paired.size() > 1 ? squared / (paired.size() - 1) : 0.0;
    const double standard_error =
        std::sqrt(sample_variance / paired.size());
    const auto stats = lbr.stats();
    double attribution_gain = 0.0;
    double attribution_top5_share = 0.0;
    double attribution_top10_share = 0.0;
    std::array<double, 3> attribution_by_street{};
    if (!options.dump_lbr_attribution_path.empty()) {
        std::vector<std::pair<const InfoKey*, const Attribution*>> rows;
        rows.reserve(attribution.size());
        for (const auto& [key, value] : attribution) {
            rows.emplace_back(&key, &value);
            attribution_gain += value.gain;
            if (key.street >= 5 && key.street <= 7) {
                attribution_by_street[key.street - 5] += value.gain;
            }
        }
        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
            return left.second->gain > right.second->gain;
        });
        const auto top_share = [&](double fraction) {
            if (rows.empty() || attribution_gain <= 0.0) return 0.0;
            const size_t count = std::max<size_t>(
                1, static_cast<size_t>(std::ceil(rows.size() * fraction)));
            double gain = 0.0;
            for (size_t index = 0; index < count; ++index) {
                gain += rows[index].second->gain;
            }
            return gain / attribution_gain;
        };
        attribution_top5_share = top_share(0.05);
        attribution_top10_share = top_share(0.10);

        std::ofstream output(options.dump_lbr_attribution_path);
        if (!output) {
            throw std::runtime_error(
                "cannot write LBR attribution: " +
                options.dump_lbr_attribution_path);
        }
        output
            << "abstraction,street,power_cluster,category,memory_code,"
               "own_public_category,own_public_suit,own_public_connected,"
               "opponent_public_category,opponent_public_suit,"
               "opponent_public_connected,pot_odds_bucket,stack_pot_bucket,"
               "own_bet_count,opponent_bet_count,checked,last_action_class,"
               "betting_goal,legal_mask,history_length,history_code,"
               "decisions,gain_ante,mean_gain_ante,max_gain_ante\n";
        for (const auto& [key, value] : rows) {
            output
                << static_cast<int>(key->abstraction) << ','
                << static_cast<int>(key->street) << ','
                << key->power_cluster << ','
                << static_cast<int>(key->category) << ','
                << static_cast<int>(key->rank_bucket) << ','
                << static_cast<int>(key->own_public_category) << ','
                << static_cast<int>(key->own_public_suit) << ','
                << static_cast<int>(key->own_public_connected) << ','
                << static_cast<int>(key->opponent_public_category) << ','
                << static_cast<int>(key->opponent_public_suit) << ','
                << static_cast<int>(key->opponent_public_connected) << ','
                << static_cast<int>(key->pot_odds_bucket) << ','
                << static_cast<int>(key->stack_pot_bucket) << ','
                << static_cast<int>(key->own_bet_count) << ','
                << static_cast<int>(key->opponent_bet_count) << ','
                << static_cast<int>(key->checked) << ','
                << static_cast<int>(key->last_action_class) << ','
                << static_cast<int>(key->betting_goal) << ','
                << static_cast<int>(key->legal_mask) << ','
                << static_cast<int>(key->history_length) << ','
                << key->history_code << ','
                << value->decisions << ','
                << value->gain << ','
                << value->gain / std::max<uint64_t>(1, value->decisions)
                << ',' << value->maximum_gain << '\n';
        }
    }
    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"agent_a\": \"policy-lbr\",\n"
              << "  \"agent_b\": \"" << target_name << "\",\n"
              << "  \"model_path\": \""
              << json_escape(options.load_path) << "\",\n"
              << "  \"atlas_path\": \""
              << json_escape(options.load_atlas_path) << "\",\n"
              << "  \"evaluation\": \"approximate exploitability lower bound\",\n"
              << "  \"scope\": \"5th+; "
              << (h4_policy
                    ? "learned target H4; heuristic LBR H4"
                    : "heuristic H4")
              << "; particle range\",\n"
              << "  \"betting_rules_version\": 3,\n"
              << "  \"hands\": " << options.hands << ",\n"
              << "  \"particles\": " << options.belief_particles << ",\n"
              << "  \"soft_top_k\": " << options.soft_top_k << ",\n"
              << "  \"soft_top_p\": " << options.soft_top_p << ",\n"
              << "  \"soft_temperature\": " << options.soft_temperature << ",\n"
              << "  \"soft_local_bandwidth\": "
              << (options.soft_local_bandwidth ? "true" : "false") << ",\n"
              << "  \"seventh_resolve_iterations\": "
              << options.seventh_resolve_iterations << ",\n"
              << "  \"seventh_posterior_particles\": "
              << options.seventh_posterior_particles << ",\n"
              << "  \"seventh_hand_history\": "
              << (options.seventh_hand_history ? "true" : "false") << ",\n"
              << "  \"average_profit_ante_for_lbr\": " << mean << ",\n"
              << "  \"approx_exploitability_lower_bound_ante\": "
              << std::max(0.0, mean) << ",\n"
              << "  \"paired_standard_error_ante\": " << standard_error << ",\n"
              << "  \"ci95_ante_for_lbr\": [" << mean - 1.96 * standard_error
              << ", " << mean + 1.96 * standard_error << "],\n"
              << "  \"wins_for_lbr\": " << wins << ",\n"
              << "  \"ties\": " << ties << ",\n"
              << "  \"losses_for_lbr\": " << losses << ",\n"
              << "  \"decisions\": " << stats.decisions << ",\n"
              << "  \"policy_queries\": " << stats.policy_queries << ",\n"
              << "  \"policy_misses\": " << stats.policy_misses << ",\n"
              << "  \"h4_particle_proposals\": "
              << stats.h4_particle_proposals << ",\n"
              << "  \"h4_particle_rejections\": "
              << stats.h4_particle_rejections << ",\n"
              << "  \"average_effective_particles\": "
              << stats.average_effective_particles << ",\n"
              << "  \"action_counts\": {";
    bool first_action = true;
    for (int action = 0; action < kActionCount; ++action) {
        if (!stats.actions[action]) continue;
        if (!first_action) std::cout << ", ";
        first_action = false;
        std::cout << "\"" << kActionNames[action] << "\": "
                  << stats.actions[action];
    }
    std::cout << "},\n";
    if constexpr (std::is_same_v<TargetPolicy, ExactSeventhResolverPolicy>) {
        const auto resolver = target.stats();
        std::cout
              << "  \"seventh_resolver\": {"
              << "\"subgames\":" << resolver.subgames
              << ",\"traversals\":" << resolver.traversals
              << ",\"node_visits\":" << resolver.node_visits
              << ",\"nodes_created\":" << resolver.nodes_created
              << ",\"maximum_local_nodes\":"
              << resolver.maximum_local_nodes
              << ",\"blueprint_fallbacks\":"
              << resolver.blueprint_fallbacks
              << ",\"posterior_builds\":" << resolver.posterior_builds
              << ",\"posterior_proposals\":"
              << resolver.posterior_proposals
              << ",\"posterior_policy_queries\":"
              << resolver.posterior_policy_queries
              << ",\"posterior_policy_misses\":"
              << resolver.posterior_policy_misses
              << ",\"posterior_fallbacks\":"
              << resolver.posterior_fallbacks
              << ",\"average_effective_posterior_particles\":"
              << (resolver.posterior_builds
                    ? resolver.posterior_effective_particles_sum /
                        resolver.posterior_builds
                    : 0.0)
              << "},\n";
    }
    std::cout
              << "  \"lbr_attribution_buckets\": "
              << attribution.size() << ",\n"
              << "  \"lbr_local_gain_total_ante\": "
              << attribution_gain << ",\n"
              << "  \"lbr_local_gain_top5pct_bucket_share\": "
              << attribution_top5_share << ",\n"
              << "  \"lbr_local_gain_top10pct_bucket_share\": "
              << attribution_top10_share << ",\n"
              << "  \"lbr_local_gain_by_street\": ["
              << attribution_by_street[0] << ", "
              << attribution_by_street[1] << ", "
              << attribution_by_street[2] << "],\n"
              << "  \"elapsed_seconds\": " << elapsed << ",\n"
              << "  \"hands_per_second\": " << options.hands / elapsed << "\n"
              << "}\n";
    return 0;
}

void print_asymp_result(
    const Options& options,
    const BucketAsymP& solver,
    const std::vector<double>& paired,
    int wins,
    int ties,
    int losses,
    double elapsed) {
    const double mean =
        std::accumulate(paired.begin(), paired.end(), 0.0) / paired.size();
    double squared = 0.0;
    for (double value : paired) squared += (value - mean) * (value - mean);
    const double sample_variance =
        paired.size() > 1 ? squared / (paired.size() - 1) : 0.0;
    const double standard_error =
        std::sqrt(sample_variance / paired.size());
    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"agent_a\": \"cpp-bucket-asymp\",\n"
              << "  \"agent_b\": \""
              << (options.opponent_model.empty() ? options.opponent : "cpp-mccfr-table")
              << "\",\n"
              << "  \"betting_rules_version\": 3,\n"
              << "  \"bucket\": \"" << options.bucket << "\",\n"
              << "  \"start_street\": " << options.start_street << ",\n"
              << "  \"hands\": " << options.hands << ",\n"
              << "  \"root_training_iterations\": " << options.root_iterations << ",\n"
              << "  \"root_training_node_budget\": " << options.root_node_budget << ",\n"
              << "  \"asymp_batch_roots\": " << options.asymp_batch_roots << ",\n"
              << "  \"asymp_step\": " << solver.step() << ",\n"
              << "  \"asymp_mu\": " << solver.perturbation() << ",\n"
              << "  \"average_profit_ante_for_a\": " << mean << ",\n"
              << "  \"paired_standard_error_ante\": " << standard_error << ",\n"
              << "  \"ci95_ante_for_a\": [" << mean - 1.96 * standard_error
              << ", " << mean + 1.96 * standard_error << "],\n"
              << "  \"wins_for_a\": " << wins << ",\n"
              << "  \"ties\": " << ties << ",\n"
              << "  \"losses_for_a\": " << losses << ",\n"
              << "  \"elapsed_seconds\": " << elapsed << ",\n"
              << "  \"hands_per_second\": " << options.hands / elapsed << ",\n"
              << "  \"traversals\": " << solver.traversals() << ",\n"
              << "  \"training_node_visits\": " << solver.node_visits() << ",\n"
              << "  \"output_policy_buckets\": " << solver.buckets() << "\n"
              << "}\n";
}

int run_asymp(
    const Options& options,
    PowerAtlas* atlas_pointer) {
    BucketAsymP solver(
        options.seed ^ 0x8ebc6af09c88c6e3ull,
        options.start_street,
        atlas_pointer,
        options.asymp_step,
        options.asymp_mu,
        options.bucket == "power-recall");
    {
        ProgressHeartbeat heartbeat(
            "bucket-asymp", "initialization", "steps", 1,
            options.progress_seconds);
        if (!options.load_path.empty()) {
            solver.load(options.load_path);
        } else {
            MCCFR base(
                false,
                options.seed,
                options.start_street,
                atlas_pointer,
                0,
                50.0,
                64,
                1,
                0.0,
                0.1,
                false,
                options.bucket == "power-recall");
            base.load(options.init_from);
            solver.initialize(base);
        }
        heartbeat.update(1);
    }
    if (options.agent_stdio) {
        run_agent_stdio(solver);
        return 0;
    }
    if (options.root_iterations) {
        std::mt19937_64 root_rng(
            options.seed ^ 0xa0761d6478bd642full);
        const auto root_started = std::chrono::steady_clock::now();
        ProgressHeartbeat heartbeat(
            "bucket-asymp", "training", "roots",
            options.root_iterations, options.progress_seconds);
        uint64_t iteration = 0;
        uint64_t next_report = options.root_report_every;
        while (iteration < options.root_iterations) {
            const uint64_t batch_size = std::min<uint64_t>(
                options.asymp_batch_roots,
                options.root_iterations - iteration);
            std::vector<State> roots;
            roots.reserve(batch_size);
            for (uint64_t index = 0; index < batch_size; ++index) {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), root_rng);
                roots.push_back(sample_fifth_street_root(
                    deck, options.ante, options.stack_ante));
            }
            solver.train_roots(roots);
            iteration += batch_size;
            heartbeat.update(iteration);
            const auto now = std::chrono::steady_clock::now();
            const bool count_due = options.root_report_every &&
                iteration >= next_report;
            if (count_due || iteration == options.root_iterations) {
                while (next_report && next_report <= iteration) {
                    next_report += options.root_report_every;
                }
                const double seconds = std::chrono::duration<double>(
                    now - root_started).count();
                const double rate = iteration / std::max(1e-9, seconds);
                const double eta =
                    (options.root_iterations - iteration) /
                    std::max(1e-9, rate);
                std::ostringstream output;
                output << "{\"algorithm\":\"bucket-asymp\""
                       << ",\"phase\":\"training\""
                       << ",\"root_iteration\":" << iteration
                       << ",\"root_iterations_total\":"
                       << options.root_iterations
                       << ",\"batch_roots\":"
                       << options.asymp_batch_roots
                       << ",\"progress_percent\":"
                       << 100.0 * iteration / options.root_iterations
                       << ",\"traversals\":" << solver.traversals()
                       << ",\"training_node_visits\":"
                       << solver.node_visits()
                       << ",\"output_policy_buckets\":" << solver.buckets()
                       << ",\"roots_per_second\":" << rate
                       << ",\"elapsed_seconds\":" << seconds
                       << ",\"eta_seconds\":" << eta << "}";
                write_progress(output.str());
            }
        }
    }
    if (options.opponent == "policy-lbr") {
        if (!options.save_path.empty()) solver.save(options.save_path);
        return run_policy_lbr(
            options, solver, "cpp-bucket-asymp");
    }

    BeliefBR belief_agent(
        options.belief_particles,
        options.belief_sigma,
        options.belief_margin,
        options.seed ^ 0xd1b54a32d192ed03ull);
    BeliefBR* belief_pointer =
        options.opponent == "belief-br" ? &belief_agent : nullptr;
    const HeuristicVariant opponent_variant =
        heuristic_variant_from_name(options.opponent);
    MCCFR model_opponent(
        false,
        options.seed ^ 0x243f6a8885a308d3ull,
        options.start_street,
        atlas_pointer);
    if (!options.opponent_model.empty()) {
        model_opponent.load(options.opponent_model);
        belief_pointer = nullptr;
    }
    std::mt19937_64 deal_rng(options.seed);
    std::vector<double> paired;
    paired.reserve(options.hands / 2);
    int wins = 0;
    int ties = 0;
    int losses = 0;
    const auto started = std::chrono::steady_clock::now();
    ProgressHeartbeat heartbeat(
        "bucket-asymp", "evaluation", "hands",
        options.hands, options.progress_seconds);
    for (int hand = 0; hand < options.hands; hand += 2) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const double first = options.opponent_model.empty()
            ? play_hand(
                deck,
                0,
                solver,
                0,
                options.ante,
                options.start_street,
                belief_pointer ? 1 : -1,
                belief_pointer,
                options.stack_ante,
                nullptr,
                -1,
                opponent_variant)
            : play_hand_match(
                deck, 0, solver, model_opponent, options.ante,
                options.start_street, options.stack_ante);
        const double second = options.opponent_model.empty()
            ? play_hand(
                deck,
                1,
                solver,
                0,
                options.ante,
                options.start_street,
                belief_pointer ? 0 : -1,
                belief_pointer,
                options.stack_ante,
                nullptr,
                -1,
                opponent_variant)
            : play_hand_match(
                deck, 1, solver, model_opponent, options.ante,
                options.start_street, options.stack_ante);
        paired.push_back((first + second) / 2.0);
        for (double value : {first, second}) {
            if (value > 0) ++wins;
            else if (value < 0) ++losses;
            else ++ties;
        }
        const int completed = hand + 2;
        heartbeat.update(completed);
        const auto now = std::chrono::steady_clock::now();
        const bool count_due = options.report_every > 0 &&
            completed % options.report_every == 0;
        if (count_due || completed == options.hands) {
            const double seconds =
                std::chrono::duration<double>(now - started).count();
            const double rate = completed / std::max(1e-9, seconds);
            const double eta =
                (options.hands - completed) / std::max(1e-9, rate);
            std::ostringstream output;
            output << "{\"algorithm\":\"bucket-asymp\""
                   << ",\"phase\":\"evaluation\""
                   << ",\"hands_completed\":" << completed
                   << ",\"hands_total\":" << options.hands
                   << ",\"progress_percent\":"
                   << 100.0 * completed / options.hands
                   << ",\"hands_per_second\":" << rate
                   << ",\"elapsed_seconds\":" << seconds
                   << ",\"eta_seconds\":" << eta << "}";
            write_progress(output.str());
        }
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (!options.save_path.empty()) solver.save(options.save_path);
    print_asymp_result(
        options,
        solver,
        paired,
        wins,
        ties,
        losses,
        elapsed);
    return 0;
}

}  // namespace

// Extensibility bridge: a sibling translation unit can reuse this entire engine
// (State, rules, MCCFR, BeliefBR, PolicyLBR, atlases, ...) by defining
// STUD_MCCFR_NO_MAIN before `#include "stud_mccfr.cpp"` and supplying its own
// main -- e.g. a new probe or agent -- without editing this file. Everything is
// in an anonymous namespace, so each including TU gets its own internal copy and
// there is no ODR conflict. See REFACTORING.md for the module map.
#ifndef STUD_MCCFR_NO_MAIN
int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.self_test) {
            self_test();
            return 0;
        }
        if (options.baseline_match) return run_baseline_match(options);
        if (options.lbr_target != "model") {
            ConditionalParticipationPolicy target(
                options.lbr_target, options.made_min_category);
            const std::string target_name = options.lbr_target == "fold"
                ? "fold"
                : options.lbr_target + "-category-" +
                    std::to_string(options.made_min_category);
            return run_policy_lbr(
                options, target, target_name.c_str());
        }

        PowerAtlas atlas(options.power_samples);
        PowerAtlas* atlas_pointer = nullptr;
        if (options.bucket == "power" ||
            options.bucket == "power-memory" ||
            options.bucket == "power-memory16" ||
            options.bucket == "power-memory81" ||
            options.bucket == "power-recall" ||
            options.bucket == "power-range" ||
            options.bucket == "power-tree") {
            if (!options.load_atlas_path.empty()) {
                atlas.load(options.load_atlas_path);
            } else {
                if (options.fit_hands <= 0) {
                    throw std::runtime_error(
                        "power buckets require --load-atlas or positive --fit-hands");
                }
                std::cerr << "collecting " << options.fit_hands
                          << (options.fit_heuristic_pool
                                ? " random heuristic-pool hands"
                                : options.fit_policy_model_path.empty()
                                    ? " heuristic hands"
                                    : " source-policy self-play hands")
                          << " for the frozen power atlas...\n";
                std::unique_ptr<PowerAtlas> source_atlas;
                std::unique_ptr<MCCFR> source_policy;
                if (!options.fit_policy_model_path.empty()) {
                    source_atlas = std::make_unique<PowerAtlas>(
                        options.power_samples);
                    source_atlas->load(options.fit_policy_atlas_path);
                    source_policy = std::make_unique<MCCFR>(
                        false,
                        options.seed ^ 0x6a09e667f3bcc909ull,
                        5,
                        source_atlas.get());
                    source_policy->load(options.fit_policy_model_path);
                }
                const bool mixed_fit =
                    options.fit_lbr_sample_cap > 0 ||
                    options.fit_heuristic_sample_cap > 0;
                auto samples = collect_power_samples(
                    options.fit_hands,
                    options.seed ^ 0x517cc1b727220a95ull,
                    options.ante,
                    options.power_samples,
                    static_cast<size_t>(options.fit_sample_cap),
                    options.stack_ante,
                    source_policy.get(),
                    options.fit_policy_epsilon,
                    mixed_fit,
                    options.fit_heuristic_pool);
                auto append_samples = [&](auto&& extra) {
                    for (int street = 0; street < 3; ++street) {
                        samples[street].insert(
                            samples[street].end(),
                            extra[street].begin(),
                            extra[street].end());
                    }
                };
                if (options.fit_lbr_sample_cap > 0) {
                    if (options.fit_lbr_target == "heuristic") {
                        HeuristicPolicy target_policy;
                        append_samples(collect_lbr_power_samples(
                            options.fit_hands,
                            options.seed ^ 0x3c6ef372fe94f82bull,
                            options.ante,
                            options.stack_ante,
                            options.power_samples,
                            static_cast<size_t>(options.fit_lbr_sample_cap),
                            options.fit_lbr_particles,
                            target_policy));
                    } else {
                        PowerAtlas target_atlas(options.power_samples);
                        target_atlas.load(options.fit_lbr_atlas_path);
                        MCCFR target_policy(
                            false,
                            options.seed ^ 0xbb67ae8584caa73bull,
                            5,
                            &target_atlas);
                        target_policy.load(options.fit_lbr_model_path);
                        append_samples(collect_lbr_power_samples(
                            options.fit_hands,
                            options.seed ^ 0x3c6ef372fe94f82bull,
                            options.ante,
                            options.stack_ante,
                            options.power_samples,
                            static_cast<size_t>(options.fit_lbr_sample_cap),
                            options.fit_lbr_particles,
                            target_policy));
                    }
                }
                if (options.fit_heuristic_sample_cap > 0) {
                    append_samples(collect_power_samples(
                        options.fit_hands,
                        options.seed ^ 0xa54ff53a5f1d36f1ull,
                        options.ante,
                        options.power_samples,
                        static_cast<size_t>(options.fit_heuristic_sample_cap),
                        options.stack_ante,
                        nullptr,
                        options.fit_heuristic_epsilon,
                        true,
                        options.fit_heuristic_pool));
                }
                std::cerr << "fitting street centroids from "
                          << samples[0].size() << "/"
                          << samples[1].size() << "/"
                          << samples[2].size() << " samples...\n";
                const std::array<int, 3> cluster_counts =
                    options.clusters_by_street == std::array<int, 3>{0, 0, 0}
                    ? std::array<int, 3>{
                        options.clusters,
                        options.clusters,
                        options.clusters}
                    : options.clusters_by_street;
                std::cerr << "street clusters: "
                          << cluster_counts[0] << "/"
                          << cluster_counts[1] << "/"
                          << cluster_counts[2] << "\n";
                if (options.fit_clustering == "hard-em") {
                    atlas.fit_hard_em(samples, cluster_counts, options.seed);
                } else {
                    atlas.fit(samples, cluster_counts, options.seed);
                }
            }
            if (options.bucket == "power-tree") {
                atlas.enable_tree();
            } else if (atlas.tree_enabled()) {
                throw std::runtime_error(
                    "POWERAT4 atlas requires --bucket power-tree");
            }
            atlas.set_assignment_cache_limit(
                static_cast<size_t>(options.power_cache_max));
            if (!options.save_atlas_path.empty()) atlas.save(options.save_atlas_path);
            atlas_pointer = &atlas;
        } else if (!options.load_atlas_path.empty() || !options.save_atlas_path.empty()) {
            throw std::runtime_error("atlas options require a power bucket mode");
        }

        ActionRangeModel range_model;
        const ActionRangeModel* range_pointer = nullptr;
        if (options.bucket == "power-range") {
            if (!options.load_range_path.empty()) {
                range_model.load(options.load_range_path);
            } else {
                PowerAtlas teacher_atlas(options.power_samples);
                teacher_atlas.load(options.imitation_atlas_path);
                MCCFR teacher(
                    false,
                    options.seed ^ 0x3c6ef372fe94f82bull,
                    5,
                    &teacher_atlas);
                teacher.load(options.imitation_model_path);
                std::cerr << "fitting action range model from "
                          << options.fit_range_hands
                          << " teacher self-play hands...\n";
                range_model = fit_action_range_model(
                    options.fit_range_hands,
                    options.seed ^ 0xbb67ae8584caa73bull,
                    options.ante,
                    options.stack_ante,
                    options.power_samples,
                    teacher);
                range_model.save(options.save_range_path);
            }
            range_pointer = &range_model;
        }

        if (options.asymp) {
            return run_asymp(options, atlas_pointer);
        }

        MCCFR solver(
            options.regret_plus,
            options.seed ^ 0x9e3779b97f4a7c15ull,
            options.start_street,
            atlas_pointer,
            options.prune_after,
            options.prune_threshold,
            options.prune_refresh,
            options.soft_top_k,
            options.soft_top_p,
            options.soft_temperature,
            options.soft_local_bandwidth,
            options.bucket == "power-recall",
            soft_growth_mode(options.soft_growth),
            options.initial_fold_regret,
            options.initial_other_regret,
            initial_regret_policy(options.initial_regret_policy),
            options.made_min_category);
        solver.use_previous_street_summary(
            options.bucket == "power-memory");
        solver.use_cumulative_street_summary(
            options.bucket == "power-memory16");
        solver.use_cumulative_street_strength_summary(
            options.bucket == "power-memory81");
        solver.use_range_model(range_pointer);
        solver.use_decaying_imitation_prior(
            options.imitation_mode == "prior");
        if (!options.load_path.empty()) solver.load(options.load_path);
        solver.configure_split_audit(SplitAuditConfig{
            options.split_audit == "random",
            options.split_audit_probability,
            options.split_rollouts,
            options.split_gap_threshold,
            options.split_confidence_z,
            static_cast<size_t>(options.split_min_strong),
            options.split_reward_gain,
            static_cast<size_t>(options.split_reservoir),
            static_cast<uint16_t>(options.split_max_depth),
            static_cast<size_t>(options.split_max_leaves),
            options.split_min_touches,
            options.split_gap_threshold_min,
            options.split_gap_touch_reference});
        if (options.imitation_prior_scale != 1.0) {
            solver.scale_imitation_prior(options.imitation_prior_scale);
        }
        if (options.imitation_roots) {
            const auto started = std::chrono::steady_clock::now();
            PowerAtlas teacher_atlas(options.power_samples);
            teacher_atlas.load(options.imitation_atlas_path);
            MCCFR teacher(
                false,
                options.seed ^ 0xd6e8feb86659fd93ull,
                options.start_street,
                &teacher_atlas);
            teacher.load(options.imitation_model_path);
            MCCFR coverage_reference(
                false,
                options.seed ^ 0x94d049bb133111ebull,
                options.start_street,
                atlas_pointer);
            coverage_reference.use_previous_street_summary(
                options.bucket == "power-memory");
            coverage_reference.use_cumulative_street_summary(
                options.bucket == "power-memory16");
            coverage_reference.use_cumulative_street_strength_summary(
                options.bucket == "power-memory81");
            coverage_reference.use_range_model(range_pointer);
            const MCCFR* coverage_pointer = nullptr;
            if (!options.imitation_cover_model_path.empty()) {
                coverage_reference.load(options.imitation_cover_model_path);
                coverage_pointer = &coverage_reference;
            }
            const ImitationResult imitation = imitate_teacher_roots(
                solver,
                teacher,
                options.imitation_roots,
                options.imitation_strength,
                options.imitation_exploration,
                options.ante,
                options.stack_ante,
                options.seed ^ 0xa0761d6478bd642full,
                coverage_pointer,
                options.imitation_report_every,
                options.imitation_external_sampling);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::cerr
                << "{\"imitation_roots\":" << imitation.roots
                << ",\"imitation_decisions\":" << imitation.decisions
                << ",\"imitation_strength\":" << options.imitation_strength
                << ",\"imitation_mode\":\"" << options.imitation_mode
                << "\""
                << ",\"imitation_exploration\":"
                << options.imitation_exploration
                << ",\"imitation_external_sampling\":"
                << (options.imitation_external_sampling ? "true" : "false")
                << ",\"buckets\":" << solver.stats().buckets
                << ",\"covered_buckets\":" << imitation.covered_buckets
                << ",\"reference_buckets\":" << imitation.reference_buckets
                << ",\"coverage\":"
                << (imitation.reference_buckets
                    ? imitation.covered_buckets /
                        static_cast<double>(imitation.reference_buckets)
                    : 0.0)
                << ",\"elapsed_seconds\":" << seconds << "}\n";
        }
        H4CFR h4(options.seed ^ 0x8cb92baa5f0b7d31ull);
        h4.set_min_touches(options.h4_min_touches);
        H4QPolicy h4_q;
        h4_q.set_min_samples(options.h4_q_min_samples);
        h4_q.set_lcb_beta(options.h4_q_lcb_beta);
        H4Policy* h4_pointer = nullptr;
        H4QPolicy* h4_q_pointer = nullptr;
        if (!options.load_h4_path.empty()) {
            h4.load(options.load_h4_path);
            h4_pointer = &h4;
        }
        if (!options.load_h4_q_path.empty()) {
            h4_q.load(options.load_h4_q_path);
            h4_pointer = &h4_q;
            h4_q_pointer = &h4_q;
        }
        if (options.h4_train_roots) {
            std::mt19937_64 h4_rng(
                options.seed ^ 0x6a09e667f3bcc909ull);
            const auto h4_started = std::chrono::steady_clock::now();
            for (uint64_t root = 1; root <= options.h4_train_roots; ++root) {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), h4_rng);
                train_h4_root(
                    h4, solver, deck, options.ante, options.stack_ante);
                if (options.h4_report_every &&
                    (root % options.h4_report_every == 0 ||
                     root == options.h4_train_roots)) {
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - h4_started).count();
                    std::cerr
                        << "{\"h4_root\":" << root
                        << ",\"h4_buckets\":" << h4.buckets()
                        << ",\"elapsed_seconds\":" << seconds << "}\n";
                }
            }
            h4.save(options.save_h4_path);
            h4_pointer = &h4;
        }
        if (options.h4_q_train_roots) {
            std::mt19937_64 h4_q_rng(
                options.seed ^ 0xbb67ae8584caa73bull);
            const auto h4_q_started = std::chrono::steady_clock::now();
            for (uint64_t root = 1; root <= options.h4_q_train_roots; ++root) {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), h4_q_rng);
                train_h4_q_root(
                    h4_q, solver, deck, options.ante, options.stack_ante);
                if (options.h4_report_every &&
                    (root % options.h4_report_every == 0 ||
                     root == options.h4_q_train_roots)) {
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - h4_q_started).count();
                    const auto stats = h4_q.stats();
                    std::cerr
                        << "{\"h4_q_root\":" << root
                        << ",\"h4_q_buckets\":" << h4_q.buckets()
                        << ",\"h4_q_samples\":" << stats.samples
                        << ",\"elapsed_seconds\":" << seconds << "}\n";
                }
            }
            h4_q.save(options.save_h4_q_path);
            h4_pointer = &h4_q;
            h4_q_pointer = &h4_q;
        }
        if (!options.dump_bucket_touches_path.empty()) {
            solver.dump_bucket_touches(options.dump_bucket_touches_path);
        }
        if (options.inspect_model) {
            const auto scale = solver.scale_stats();
            const auto prior = solver.imitation_prior_stats();
            const H4QStats h4_q_stats = h4_q_pointer
                ? h4_q_pointer->stats()
                : H4QStats{};
            std::cout << std::fixed << std::setprecision(8)
                << "{\"buckets\":" << solver.stats().buckets
                << ",\"regret_l1_total\":" << scale.regret_l1_total
                << ",\"regret_l1_mean\":" << scale.regret_l1_mean
                << ",\"regret_l1_median\":" << scale.regret_l1_median
                << ",\"regret_l1_p95\":" << scale.regret_l1_p95
                << ",\"regret_l1_max\":" << scale.regret_l1_max
                << ",\"strategy_mass_total\":"
                << scale.strategy_mass_total
                << ",\"strategy_mass_mean\":" << scale.strategy_mass_mean
                << ",\"strategy_mass_median\":"
                << scale.strategy_mass_median
                << ",\"strategy_mass_p95\":" << scale.strategy_mass_p95
                << ",\"strategy_mass_max\":" << scale.strategy_mass_max
                << ",\"h4_buckets\":"
                << (h4_pointer ? h4_pointer->buckets() : 0)
                << ",\"h4_q_samples\":" << h4_q_stats.samples
                << ",\"h4_q_min_samples_seen\":"
                << h4_q_stats.min_samples
                << ",\"h4_q_max_samples_seen\":"
                << h4_q_stats.max_samples
                << ",\"imitation_prior_buckets\":"
                << solver.imitation_prior_count()
                << ",\"imitation_prior_strength_mean\":"
                << prior.strength_mean
                << ",\"imitation_lambda_mean\":" << prior.lambda_mean
                << ",\"imitation_lambda_median\":" << prior.lambda_median
                << ",\"imitation_lambda_p95\":" << prior.lambda_p95
                << ",\"imitation_lambda_max\":" << prior.lambda_max
                << ",\"imitation_lambda_above_half\":"
                << prior.lambda_above_half
                << ",\"average_policy_prior_fraction_mean\":"
                << prior.average_prior_fraction_mean
                << ",\"average_policy_prior_fraction_median\":"
                << prior.average_prior_fraction_median
                << ",\"average_policy_prior_fraction_p95\":"
                << prior.average_prior_fraction_p95
                << "}\n";
            return 0;
        }
        if (options.diagnostic_roots) {
            std::mt19937_64 diagnostic_rng(
                options.seed ^ 0x243f6a8885a308d3ull);
            const auto started = std::chrono::steady_clock::now();
            for (uint64_t root = 1;
                 root <= options.diagnostic_roots;
                 ++root) {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), diagnostic_rng);
                const State state = sample_fifth_street_root(
                    deck, options.ante, options.stack_ante, h4_pointer);
                solver.diagnose_conflict_root(state, 0);
                solver.diagnose_conflict_root(state, 1);
                if (options.root_report_every &&
                    (root % options.root_report_every == 0 ||
                     root == options.diagnostic_roots)) {
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                    std::cerr
                        << "{\"diagnostic_root\":" << root
                        << ",\"conflict_buckets\":"
                        << solver.conflict_buckets()
                        << ",\"observed_seventh_recall_infosets\":"
                        << solver.observed_seventh_recall_infosets()
                        << ",\"average_fixed_hand_seventh_infosets\":"
                        << solver.average_fixed_hand_seventh_infosets()
                        << ",\"maximum_fixed_hand_seventh_infosets\":"
                        << solver.maximum_fixed_hand_seventh_infosets()
                        << ",\"elapsed_seconds\":" << seconds << "}\n";
                }
            }
            solver.dump_conflict(
                options.dump_conflict_path,
                options.diagnostic_min_samples);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::cout
                << "{\"diagnostic_roots\":" << options.diagnostic_roots
                << ",\"diagnostic_traversals\":"
                << options.diagnostic_roots * 2
                << ",\"conflict_buckets\":"
                << solver.conflict_buckets()
                << ",\"observed_seventh_recall_infosets\":"
                << solver.observed_seventh_recall_infosets()
                << ",\"average_fixed_hand_seventh_infosets\":"
                << solver.average_fixed_hand_seventh_infosets()
                << ",\"maximum_fixed_hand_seventh_infosets\":"
                << solver.maximum_fixed_hand_seventh_infosets()
                << ",\"output\":\""
                << json_escape(options.dump_conflict_path)
                << "\",\"elapsed_seconds\":" << seconds << "}\n";
            return 0;
        }
        if (!options.load_temperatures_path.empty()) {
            solver.load_temperatures(options.load_temperatures_path);
        }
        if (options.temperature_calibration_roots) {
            std::mt19937_64 calibration_rng(
                options.seed ^ 0xe7037ed1a0b428dbull);
            const auto calibration_started =
                std::chrono::steady_clock::now();
            const int calibration_cycles =
                options.cluster_growth_steps > 0
                ? options.cluster_growth_steps + 1
                : 1;
            for (int cycle = 0; cycle < calibration_cycles; ++cycle) {
                solver.reset_temperature_calibration();
                for (uint64_t iteration = 1;
                     iteration <= options.temperature_calibration_roots;
                     ++iteration) {
                    auto deck = fresh_deck();
                    std::shuffle(deck.begin(), deck.end(), calibration_rng);
                    const State root =
                        sample_fifth_street_root(
                            deck, options.ante, options.stack_ante);
                    solver.calibrate_temperature_root(root, 0);
                    solver.calibrate_temperature_root(root, 1);
                    if (options.root_report_every &&
                        (iteration % options.root_report_every == 0 ||
                         iteration == options.temperature_calibration_roots)) {
                        const double seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            calibration_started).count();
                        std::cerr
                            << "{\"temperature_calibration_cycle\":"
                            << cycle
                            << ",\"temperature_calibration_root\":"
                            << iteration
                            << ",\"elapsed_seconds\":" << seconds << "}\n";
                    }
                }
                const auto summary = solver.apply_temperature_calibration(
                    static_cast<uint64_t>(options.temperature_min_samples));
                std::cerr
                    << "{\"temperature_calibration_cycle\":" << cycle
                    << ",\"temperature_calibrated_clusters\":"
                    << summary.calibrated_clusters
                    << ",\"temperature_samples\":" << summary.samples
                    << ",\"baseline_surrogate_loss\":"
                    << summary.baseline_loss
                    << ",\"selected_surrogate_loss\":"
                    << summary.selected_loss
                    << ",\"candidate_choices\":[";
                for (size_t index = 0; index < summary.choices.size(); ++index) {
                    if (index) std::cerr << ",";
                    std::cerr << summary.choices[index];
                }
                std::cerr << "]}\n";

                if (cycle >= options.cluster_growth_steps) continue;
                const auto growth = solver.append_growth_cluster(
                    options.cluster_growth_threshold,
                    static_cast<uint64_t>(options.temperature_min_samples));
                std::cerr
                    << "{\"cluster_growth_cycle\":" << cycle
                    << ",\"added\":" << (growth.added ? "true" : "false")
                    << ",\"regret_ante\":" << growth.regret;
                if (growth.added) {
                    std::cerr
                        << ",\"street\":" << growth.street
                        << ",\"parent_cluster\":" << growth.parent_cluster
                        << ",\"new_cluster\":" << growth.new_cluster
                        << ",\"initialized_nodes\":"
                        << growth.initialized_nodes
                        << ",\"initial_strategy\":[";
                    for (int action = 0; action < kActionCount; ++action) {
                        if (action) std::cerr << ",";
                        std::cerr << growth.initial_strategy[action];
                    }
                    std::cerr << "]";
                }
                std::cerr << "}\n";
                if (!growth.added) break;
            }
            if (!options.save_atlas_path.empty()) {
                atlas.save(options.save_atlas_path);
            }
        }
        if (!options.save_temperatures_path.empty()) {
            solver.save_temperatures(options.save_temperatures_path);
        }
        if (!options.merge_paths.empty()) {
            MCCFR base(
                false,
                options.seed,
                options.start_street,
                atlas_pointer,
                options.prune_after,
                options.prune_threshold,
                options.prune_refresh,
                options.soft_top_k,
                options.soft_top_p,
                options.soft_temperature,
                options.soft_local_bandwidth,
                options.bucket == "power-recall");
            base.use_previous_street_summary(
                options.bucket == "power-memory");
            base.use_cumulative_street_summary(
                options.bucket == "power-memory16");
            base.use_cumulative_street_strength_summary(
                options.bucket == "power-memory81");
            base.use_range_model(range_pointer);
            base.load(options.load_path);
            uint64_t added_touches = 0;
            for (const std::string& path : options.merge_paths) {
                MCCFR worker(
                    false,
                    options.seed,
                    options.start_street,
                    atlas_pointer,
                    options.prune_after,
                    options.prune_threshold,
                    options.prune_refresh,
                    options.soft_top_k,
                    options.soft_top_p,
                    options.soft_temperature,
                    options.soft_local_bandwidth,
                    options.bucket == "power-recall");
                worker.use_previous_street_summary(
                    options.bucket == "power-memory");
                worker.use_cumulative_street_summary(
                    options.bucket == "power-memory16");
                worker.use_cumulative_street_strength_summary(
                    options.bucket == "power-memory81");
                worker.use_range_model(range_pointer);
                worker.load(path);
                added_touches += solver.merge_worker_delta(base, worker);
            }
            solver.save(options.save_path);
            std::cout << "{\"merged_workers\":" << options.merge_paths.size()
                      << ",\"added_touches\":" << added_touches
                      << ",\"buckets\":" << solver.stats().buckets << "}\n";
            return 0;
        }
        if (options.agent_stdio) {
            if (options.seventh_resolve_iterations > 0) {
                ExactSeventhResolverPolicy resolved(
                    solver,
                    options.seventh_resolve_iterations,
                    options.seventh_resolve_prior,
                    options.seventh_hand_history,
                    options.seed ^ 0x8cb92baa3f3d8dd7ull,
                    options.seventh_posterior_particles);
                run_agent_stdio(resolved, h4_pointer);
            } else {
                run_agent_stdio(solver, h4_pointer);
            }
            return 0;
        }
        if (options.root_iterations || options.root_node_budget) {
            std::mt19937_64 root_rng(
                options.seed ^ 0xa0761d6478bd642full);
            const auto root_started = std::chrono::steady_clock::now();
            const bool adaptive =
                options.soft_growth == "mix" ||
                options.soft_growth == "simple" ||
                options.soft_growth == "point" ||
                options.soft_growth == "residual";
            if (adaptive) {
                solver.reset_temperature_calibration();
            }
            const uint64_t initial_node_visits = solver.node_visits();
            uint64_t iteration = 0;
            uint64_t adaptation = 0;
            while (
                options.root_iterations
                    ? iteration < options.root_iterations
                    : solver.node_visits() - initial_node_visits <
                        options.root_node_budget) {
                ++iteration;
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), root_rng);
                const State root =
                    sample_fifth_street_root(
                        deck, options.ante, options.stack_ante, h4_pointer);
                solver.train_root(root, 0);
                solver.train_root(root, 1);
                solver.process_split_audits();
                const bool training_complete =
                    options.root_iterations
                        ? iteration >= options.root_iterations
                        : solver.node_visits() - initial_node_visits >=
                            options.root_node_budget;
                if (adaptive &&
                    (iteration % options.soft_adapt_every == 0 ||
                     training_complete)) {
                    const double growth_threshold = std::max(
                        options.cluster_growth_threshold_min,
                        options.cluster_growth_threshold *
                            std::pow(
                                options.cluster_growth_threshold_decay,
                                static_cast<double>(adaptation)));
                    const int adaptive_street =
                        options.soft_adapt_round_robin
                            ? static_cast<int>(adaptation % 3)
                            : -1;
                    const auto adaptive = solver.adapt_soft_clusters(
                        growth_threshold,
                        static_cast<uint64_t>(
                            options.temperature_min_samples),
                        adaptive_street);
                    std::cerr
                        << "{\"soft_adapt_root\":" << iteration
                        << ",\"mode\":\"" << options.soft_growth
                        << "\",\"adaptation\":" << adaptation
                        << ",\"considered_street\":"
                        << (adaptive_street < 0 ? 0 : adaptive_street + 5)
                        << ",\"growth_threshold\":" << growth_threshold
                        << ",\"maximum_average_regret\":"
                        << adaptive.maximum_average_regret
                        << ",\"cluster_added\":"
                        << (adaptive.growth.added ? "true" : "false");
                    if (adaptive.growth.added) {
                        std::cerr
                            << ",\"street\":" << adaptive.growth.street
                            << ",\"parent_cluster\":"
                            << adaptive.growth.parent_cluster
                            << ",\"new_cluster\":"
                            << adaptive.growth.new_cluster
                            << ",\"initialized_nodes\":"
                            << adaptive.growth.initialized_nodes
                            << ",\"initial_strategy\":[";
                        for (int action = 0;
                             action < kActionCount;
                             ++action) {
                            if (action) std::cerr << ",";
                            std::cerr
                                << adaptive.growth.initial_strategy[action];
                        }
                        std::cerr << "]";
                    } else {
                        std::cerr
                            << ",\"temperature_clusters\":"
                            << adaptive.temperature.calibrated_clusters
                            << ",\"baseline_surrogate_loss\":"
                            << adaptive.temperature.baseline_loss
                            << ",\"selected_surrogate_loss\":"
                            << adaptive.temperature.selected_loss;
                    }
                    std::cerr << "}\n";
                    solver.reset_temperature_calibration();
                    solver.save(options.save_path);
                    atlas.save(options.save_atlas_path);
                    if (!options.save_temperatures_path.empty()) {
                        solver.save_temperatures(
                            options.save_temperatures_path);
                    }
                    ++adaptation;
                }
                if (options.root_report_every &&
                    (iteration % options.root_report_every == 0 ||
                     training_complete)) {
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - root_started).count();
                    std::cerr
                        << "{\"root_iteration\":" << iteration
                        << ",\"traversals\":" << solver.traversals()
                        << ",\"training_node_visits\":" << solver.node_visits()
                        << ",\"split_audits\":"
                        << solver.split_audit_stats().audits
                        << ",\"split_audit_node_visits\":"
                        << solver.split_audit_stats().audit_node_visits
                        << ",\"split_accepted\":"
                        << solver.split_audit_stats().accepted_splits
                        << ",\"split_rejected\":"
                        << solver.split_audit_stats().rejected_splits
                        << ",\"split_touch_deferred\":"
                        << solver.split_audit_stats().touch_deferred
                        << ",\"split_capacity_deferred\":"
                        << solver.split_audit_stats().capacity_deferred
                        << ",\"split_depth_deferred\":"
                        << solver.split_audit_stats().depth_deferred
                        << ",\"split_reward_rejected\":"
                        << solver.split_audit_stats().reward_rejected
                        << ",\"split_regret_rejected\":"
                        << solver.split_audit_stats().reward_rejected
                        << ",\"split_gap_threshold_mean\":"
                        << (solver.split_audit_stats().threshold_evaluations
                                ? solver.split_audit_stats().threshold_sum /
                                    solver.split_audit_stats().threshold_evaluations
                                : 0.0)
                        << ",\"split_gap_threshold_min_seen\":"
                        << (solver.split_audit_stats().threshold_evaluations
                                ? solver.split_audit_stats().threshold_min
                                : 0.0)
                        << ",\"split_gap_threshold_max_seen\":"
                        << solver.split_audit_stats().threshold_max
                        << ",\"split_accepted_by_street\":["
                        << solver.split_audit_stats().accepted_by_street[0]
                        << ","
                        << solver.split_audit_stats().accepted_by_street[1]
                        << ","
                        << solver.split_audit_stats().accepted_by_street[2]
                        << "]"
                        << ",\"split_inherited_touches\":"
                        << solver.split_audit_stats().inherited_touches
                        << ",\"split_last_child_touches\":["
                        << solver.split_audit_stats().last_child_touches[0]
                        << ","
                        << solver.split_audit_stats().last_child_touches[1]
                        << "]"
                        << ",\"split_leaves_by_street\":["
                        << solver.split_leaf_counts()[0] << ","
                        << solver.split_leaf_counts()[1] << ","
                        << solver.split_leaf_counts()[2] << "]"
                        << ",\"power_cache_entries\":"
                        << solver.power_cache_stats().entries
                        << ",\"power_cache_limit\":"
                        << solver.power_cache_stats().limit
                        << ",\"power_cache_hits\":"
                        << solver.power_cache_stats().hits
                        << ",\"power_cache_misses\":"
                        << solver.power_cache_stats().misses
                        << ",\"power_cache_resets\":"
                        << solver.power_cache_stats().resets
                        << ",\"requested_node_budget\":"
                        << options.root_node_budget
                        << ",\"buckets\":" << solver.stats().buckets
                        << ",\"elapsed_seconds\":" << seconds << "}\n";
                }
            }
            if (options.split_audit == "random") {
                if (!options.save_path.empty()) solver.save(options.save_path);
                if (!options.save_atlas_path.empty()) {
                    atlas.save(options.save_atlas_path);
                }
            }
        }
        if (options.opponent == "policy-lbr") {
            if (!options.save_path.empty()) solver.save(options.save_path);
            if (options.seventh_resolve_iterations > 0) {
                ExactSeventhResolverPolicy resolved(
                    solver,
                    options.seventh_resolve_iterations,
                    options.seventh_resolve_prior,
                    options.seventh_hand_history,
                    options.seed ^ 0x8cb92baa3f3d8dd7ull,
                    options.seventh_posterior_particles);
                return run_policy_lbr(
                    options,
                    resolved,
                    "cpp-mccfr-7th-exact-resolver",
                    h4_pointer);
            }
            return run_policy_lbr(
                options, solver, "cpp-mccfr-table", h4_pointer);
        }
        BeliefBR belief_agent(
            options.belief_particles,
            options.belief_sigma,
            options.belief_margin,
            options.seed ^ 0xd1b54a32d192ed03ull);
        BeliefBR* belief_pointer =
            options.opponent == "belief-br" ? &belief_agent : nullptr;
        const HeuristicVariant opponent_variant =
            heuristic_variant_from_name(options.opponent);
        HalfPolicy half_agent;
        std::mt19937_64 deal_rng(options.seed);
        std::vector<double> paired;
        paired.reserve(options.hands / 2);
        int wins = 0;
        int ties = 0;
        int losses = 0;
        const auto started = std::chrono::steady_clock::now();
        for (int hand = 0; hand < options.hands; hand += 2) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            const bool against_half = options.opponent == "half";
            const double first = against_half
                ? play_hand_match(
                    deck, 0, solver, half_agent, options.ante,
                    options.start_street, options.stack_ante)
                : play_hand(
                    deck, 0, solver, options.iterations, options.ante,
                    options.start_street, belief_pointer ? 1 : -1,
                    belief_pointer, options.stack_ante, h4_pointer, 0,
                    opponent_variant);
            const double second = against_half
                ? play_hand_match(
                    deck, 1, solver, half_agent, options.ante,
                    options.start_street, options.stack_ante)
                : play_hand(
                    deck, 1, solver, options.iterations, options.ante,
                    options.start_street, belief_pointer ? 0 : -1,
                    belief_pointer, options.stack_ante, h4_pointer, 1,
                    opponent_variant);
            paired.push_back((first + second) / 2.0);
            for (double value : {first, second}) {
                if (value > 0) ++wins;
                else if (value < 0) ++losses;
                else ++ties;
            }
            if (options.report_every > 0 && (hand + 2) % options.report_every == 0) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << (hand + 2) << "/" << options.hands
                          << " hands, " << solver.stats().buckets << " buckets, "
                          << (hand + 2) / seconds << " hands/s\n";
            }
        }
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (!options.save_path.empty()) solver.save(options.save_path);
        print_result(options, solver, paired, wins, ties, losses, elapsed);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
#endif  // STUD_MCCFR_NO_MAIN
