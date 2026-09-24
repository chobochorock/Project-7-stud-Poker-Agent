#define STUD_MCCFR_NO_MAIN
#include "stud_mccfr.cpp"

#include <filesystem>

namespace {

constexpr size_t kExactHistorySlots = 32;

struct ExactInfoKey {
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
    std::array<uint8_t, kExactHistorySlots> history{};

    bool operator==(const ExactInfoKey& other) const {
        return viewer == other.viewer && street == other.street &&
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

static_assert(std::is_trivially_copyable_v<ExactInfoKey>);

struct ExactInfoKeyHash {
    size_t operator()(const ExactInfoKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
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

ExactInfoKey exact_key(const State& state, int viewer) {
    if (state.history.size() > kExactHistorySlots) {
        throw std::runtime_error("exact betting history exceeds 32 actions");
    }
    ExactInfoKey key;
    key.viewer = static_cast<uint8_t>(viewer);
    key.street = static_cast<uint8_t>(state.street);
    key.legal_mask = valid_mask(state, viewer);
    std::vector<Card> canonical_hidden = state.players[viewer].hidden;
    if (canonical_hidden.size() >= 2) {
        std::sort(
            canonical_hidden.begin(), canonical_hidden.begin() + 2,
            [](const Card& left, const Card& right) {
                return card_index(left) < card_index(right);
            });
    }
    encode_exact_cards(
        canonical_hidden, true, key.own_hidden, key.own_hidden_count);
    encode_exact_cards(
        state.players[viewer].shown, true,
        key.own_shown, key.own_shown_count);
    encode_exact_cards(
        state.players[1 - viewer].shown, true,
        key.opponent_shown, key.opponent_shown_count);
    if (state.players[viewer].has_discard) {
        key.discarded = card_token(state.players[viewer].discarded);
    }
    key.history_length = static_cast<uint8_t>(state.history.size());
    for (size_t index = 0; index < state.history.size(); ++index) {
        const Event& event = state.history[index];
        const int relative_actor = event.actor == viewer ? 0 : 1;
        key.history[index] = static_cast<uint8_t>(
            1 + (event.street - 5) * 2 * kActionCount +
            relative_actor * kActionCount + event.action);
    }
    return key;
}

Card indexed_card(int index) {
    return Card{
        static_cast<uint8_t>(2 + index % 13),
        static_cast<uint8_t>(index / 13)};
}

uint64_t card_bit(const Card& card) {
    return uint64_t{1} << card_index(card);
}

int h4_outcome_multiplicity(
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

struct PrivateType {
    std::array<uint8_t, 2> hidden{};
    uint8_t discarded = 0;
    uint64_t mask = 0;
    double reach = 0.0;
};

struct RangeStats {
    size_t support = 0;
    double effective_support = 0.0;
    double entropy = 0.0;
};

class PrivateRange {
public:
    void build(const State& public_root, int seat) {
        entries_.clear();
        uint64_t public_mask = 0;
        for (const Player& player : public_root.players) {
            for (const Card& card : player.shown) public_mask |= card_bit(card);
        }
        if (public_root.players[seat].shown.empty()) {
            throw std::runtime_error("H4 public card is missing");
        }
        const Card reveal = public_root.players[seat].shown.front();
        for (int discarded = 0; discarded < 52; ++discarded) {
            if (public_mask & (uint64_t{1} << discarded)) continue;
            for (int first = 0; first < 52; ++first) {
                if (first == discarded ||
                    (public_mask & (uint64_t{1} << first))) continue;
                for (int second = first + 1; second < 52; ++second) {
                    if (second == discarded ||
                        (public_mask & (uint64_t{1} << second))) continue;
                    const int ways = h4_outcome_multiplicity(
                        {indexed_card(first), indexed_card(second)},
                        reveal, indexed_card(discarded));
                    if (!ways) continue;
                    entries_.push_back(PrivateType{
                        {static_cast<uint8_t>(first),
                         static_cast<uint8_t>(second)},
                        static_cast<uint8_t>(discarded),
                        (uint64_t{1} << first) |
                            (uint64_t{1} << second) |
                            (uint64_t{1} << discarded),
                        static_cast<double>(ways)});
                }
            }
        }
        rebuild();
    }

    void remove_public_cards(const std::vector<Card>& cards) {
        uint64_t mask = 0;
        for (const Card& card : cards) mask |= card_bit(card);
        for (PrivateType& entry : entries_) {
            if (entry.mask & mask) entry.reach = 0.0;
        }
        rebuild();
    }

    template <class Policy>
    double observe_action(
        const State& public_state,
        int seat,
        Action action,
        Policy&& policy,
        uint64_t& found,
        uint64_t& queries) {
        const double old_total = total_;
        for (PrivateType& entry : entries_) {
            if (entry.reach <= 0.0) continue;
            State state = public_state;
            set_private(state, seat, entry);
            bool present = false;
            const auto strategy = policy(state, seat, &present);
            ++queries;
            found += present;
            entry.reach *= std::max(0.0, strategy[action]);
        }
        rebuild();
        return old_total > 0.0 ? total_ / old_total : 0.0;
    }

    const PrivateType& sample(std::mt19937_64& rng) const {
        if (total_ <= 0.0) throw std::runtime_error("empty private range");
        std::uniform_real_distribution<double> uniform(0.0, total_);
        const double value = uniform(rng);
        const auto it = std::lower_bound(
            cumulative_.begin(), cumulative_.end(), value);
        const size_t index = std::min<size_t>(
            entries_.size() - 1,
            static_cast<size_t>(it - cumulative_.begin()));
        return entries_[index];
    }

    RangeStats stats() const {
        RangeStats result;
        double squared = 0.0;
        for (const PrivateType& entry : entries_) {
            if (entry.reach <= 0.0) continue;
            ++result.support;
            const double probability = entry.reach / total_;
            squared += probability * probability;
            result.entropy -= probability * std::log(probability);
        }
        result.effective_support = squared > 0.0 ? 1.0 / squared : 0.0;
        return result;
    }

    double normalized_reach(
        const std::vector<Card>& hidden,
        const Card& discarded) const {
        if (hidden.size() < 2 || total_ <= 0.0) return 0.0;
        std::array<int, 2> target{
            card_index(hidden[0]), card_index(hidden[1])};
        std::sort(target.begin(), target.end());
        for (const PrivateType& entry : entries_) {
            if (entry.hidden[0] == target[0] &&
                entry.hidden[1] == target[1] &&
                entry.discarded == card_index(discarded)) {
                return entry.reach / total_;
            }
        }
        return 0.0;
    }

    static void set_private(State& state, int seat, const PrivateType& type) {
        Player& player = state.players[seat];
        player.hidden = {
            indexed_card(type.hidden[0]), indexed_card(type.hidden[1])};
        player.discarded = indexed_card(type.discarded);
        player.has_discard = true;
    }

private:
    void rebuild() {
        cumulative_.resize(entries_.size());
        total_ = 0.0;
        for (size_t index = 0; index < entries_.size(); ++index) {
            total_ += entries_[index].reach;
            cumulative_[index] = total_;
        }
        if (total_ <= 0.0) {
            throw std::runtime_error("public observation emptied a private range");
        }
    }

    std::vector<PrivateType> entries_;
    std::vector<double> cumulative_;
    double total_ = 0.0;
};

class FactorizedPBS {
public:
    explicit FactorizedPBS(const State& public_root) {
        ranges_[0].build(public_root, 0);
        ranges_[1].build(public_root, 1);
    }

    State sample_world(const State& public_root, std::mt19937_64& rng) const {
        const PrivateType* left = nullptr;
        const PrivateType* right = nullptr;
        for (;;) {
            ++proposals_;
            left = &ranges_[0].sample(rng);
            right = &ranges_[1].sample(rng);
            if (!(left->mask & right->mask)) break;
            ++collisions_;
        }
        State state = public_root;
        PrivateRange::set_private(state, 0, *left);
        PrivateRange::set_private(state, 1, *right);
        uint64_t known = left->mask | right->mask;
        for (const Player& player : state.players) {
            for (const Card& card : player.shown) known |= card_bit(card);
        }
        state.simulation_deck.clear();
        for (int index = 0; index < 52; ++index) {
            if (!(known & (uint64_t{1} << index))) {
                state.simulation_deck.push_back(indexed_card(index));
            }
        }
        std::shuffle(
            state.simulation_deck.begin(), state.simulation_deck.end(), rng);
        return state;
    }

    template <class Policy>
    double observe_action(
        const State& public_state,
        int actor,
        Action action,
        Policy&& policy,
        uint64_t& found,
        uint64_t& queries) {
        return ranges_[actor].observe_action(
            public_state, actor, action,
            std::forward<Policy>(policy), found, queries);
    }

    void reveal_public_cards(const std::vector<Card>& cards) {
        ranges_[0].remove_public_cards(cards);
        ranges_[1].remove_public_cards(cards);
    }

    const PrivateRange& range(int seat) const { return ranges_[seat]; }
    RangeStats stats(int seat) const { return ranges_[seat].stats(); }
    uint64_t proposals() const { return proposals_; }
    uint64_t collisions() const { return collisions_; }

private:
    std::array<PrivateRange, 2> ranges_;
    mutable uint64_t proposals_ = 0;
    mutable uint64_t collisions_ = 0;
};

void advance_search_street(State& state) {
    if (state.street == 7) {
        state.terminal = true;
        return;
    }
    ++state.street;
    const bool public_card = state.street != 7;
    for (int seat = 0; seat < 2; ++seat) {
        if (state.simulation_deck.empty()) {
            throw std::runtime_error("simulation deck exhausted");
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
        advance_search_street(state);
        return;
    }
    state.actor = first_bettor(state);
}

Action sample_action(
    const std::array<double, kActionCount>& strategy,
    uint8_t mask,
    std::mt19937_64& rng) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double threshold = uniform(rng);
    double cumulative = 0.0;
    Action last = FOLD;
    for (int action = 0; action < kActionCount; ++action) {
        if (!(mask & (1u << action))) continue;
        last = static_cast<Action>(action);
        cumulative += strategy[action];
        if (threshold <= cumulative) break;
    }
    return last;
}

struct SolverMetrics {
    uint64_t iterations = 0;
    uint64_t traversals = 0;
    uint64_t node_visits = 0;
    uint64_t infosets = 0;
    std::array<uint64_t, 3> infosets_by_street{};
    double cumulative_positive_regret_mean = 0.0;
    double average_positive_regret_mean = 0.0;
    double regret_bound_proxy = 0.0;
    double touch_normalized_regret = 0.0;
    double average_touches = 0.0;
    double revisited_infoset_fraction = 0.0;
    double average_policy_entropy = 0.0;
};

class ExactPbsCfr {
public:
    ExactPbsCfr(uint64_t seed, MCCFR* blueprint, double prior_strength)
        : blueprint_(blueprint), prior_strength_(prior_strength), rng_(seed) {
        nodes_.reserve(1 << 16);
    }

    template <class Reporter>
    void train(
        const State& public_root,
        const FactorizedPBS& pbs,
        uint64_t target_iterations,
        uint64_t report_every,
        Reporter&& report) {
        while (iterations_ < target_iterations) {
            for (int traverser = 0; traverser < 2; ++traverser) {
                State state = pbs.sample_world(public_root, rng_);
                traverse(std::move(state), traverser);
                ++traversals_;
            }
            ++iterations_;
            if (report_every &&
                (iterations_ % report_every == 0 ||
                 iterations_ == target_iterations)) {
                report(metrics());
            }
        }
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        const auto it = nodes_.find(exact_key(state, viewer));
        if (it != nodes_.end()) {
            if (found) *found = true;
            return average_strategy(it->second, valid_mask(state, viewer));
        }
        if (blueprint_) return blueprint_->policy(state, viewer, found);
        if (found) *found = false;
        return uniform_strategy(valid_mask(state, viewer));
    }

    SolverMetrics metrics() const {
        SolverMetrics result;
        result.iterations = iterations_;
        result.traversals = traversals_;
        result.node_visits = node_visits_;
        result.infosets = nodes_.size();
        std::array<double, 2> maximum_regret_sum{};
        double positive_sum = 0.0;
        uint64_t legal_count = 0;
        uint64_t touch_count = 0;
        uint64_t revisited = 0;
        double entropy_sum = 0.0;
        for (const auto& [key, node] : nodes_) {
            if (key.street >= 5 && key.street <= 7) {
                ++result.infosets_by_street[key.street - 5];
            }
            double maximum = 0.0;
            for (int action = 0; action < kActionCount; ++action) {
                if (!(key.legal_mask & (1u << action))) continue;
                const double positive = std::max(0.0, node.regrets[action]);
                positive_sum += positive;
                maximum = std::max(maximum, positive);
                ++legal_count;
            }
            maximum_regret_sum[key.viewer] += maximum;
            touch_count += node.touches;
            revisited += node.touches > 1;
            const auto strategy = average_strategy(node, key.legal_mask);
            for (double probability : strategy) {
                if (probability > 0.0) {
                    entropy_sum -= probability * std::log(probability);
                }
            }
        }
        const double denominator = std::max<uint64_t>(1, iterations_);
        result.cumulative_positive_regret_mean = legal_count
            ? positive_sum / legal_count
            : 0.0;
        result.average_positive_regret_mean = legal_count
            ? positive_sum / legal_count / denominator
            : 0.0;
        result.regret_bound_proxy =
            0.5 * (maximum_regret_sum[0] + maximum_regret_sum[1]) /
            denominator;
        result.touch_normalized_regret = touch_count
            ? (maximum_regret_sum[0] + maximum_regret_sum[1]) / touch_count
            : 0.0;
        result.average_touches = nodes_.empty()
            ? 0.0
            : touch_count / static_cast<double>(nodes_.size());
        result.revisited_infoset_fraction = nodes_.empty()
            ? 0.0
            : revisited / static_cast<double>(nodes_.size());
        result.average_policy_entropy = nodes_.empty()
            ? 0.0
            : entropy_sum / nodes_.size();
        return result;
    }

    void save(const std::string& path, uint64_t root_fingerprint) const {
        const std::string temporary = path + ".tmp";
        std::ofstream output(temporary, std::ios::binary);
        if (!output) throw std::runtime_error("cannot write checkpoint");
        const char magic[8] = {'R','B','L','P','B','S','1','\0'};
        output.write(magic, sizeof(magic));
        output.write(
            reinterpret_cast<const char*>(&root_fingerprint),
            sizeof(root_fingerprint));
        output.write(reinterpret_cast<const char*>(&iterations_), sizeof(iterations_));
        output.write(reinterpret_cast<const char*>(&traversals_), sizeof(traversals_));
        output.write(reinterpret_cast<const char*>(&node_visits_), sizeof(node_visits_));
        const uint64_t count = nodes_.size();
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        std::ostringstream rng_state;
        rng_state << rng_;
        const std::string serialized_rng = rng_state.str();
        const uint64_t rng_size = serialized_rng.size();
        output.write(reinterpret_cast<const char*>(&rng_size), sizeof(rng_size));
        output.write(serialized_rng.data(), serialized_rng.size());
        for (const auto& [key, node] : nodes_) {
            output.write(reinterpret_cast<const char*>(&key), sizeof(key));
            output.write(reinterpret_cast<const char*>(&node), sizeof(node));
        }
        output.close();
        if (!output) throw std::runtime_error("failed to write checkpoint");
        std::filesystem::remove(path);
        std::filesystem::rename(temporary, path);
    }

    void load(const std::string& path, uint64_t root_fingerprint) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read checkpoint");
        char magic[8]{};
        uint64_t stored_fingerprint = 0;
        uint64_t count = 0;
        uint64_t rng_size = 0;
        input.read(magic, sizeof(magic));
        input.read(
            reinterpret_cast<char*>(&stored_fingerprint),
            sizeof(stored_fingerprint));
        input.read(reinterpret_cast<char*>(&iterations_), sizeof(iterations_));
        input.read(reinterpret_cast<char*>(&traversals_), sizeof(traversals_));
        input.read(reinterpret_cast<char*>(&node_visits_), sizeof(node_visits_));
        input.read(reinterpret_cast<char*>(&count), sizeof(count));
        input.read(reinterpret_cast<char*>(&rng_size), sizeof(rng_size));
        if (std::memcmp(magic, "RBLPBS1", 7) != 0 ||
            stored_fingerprint != root_fingerprint || rng_size > (1u << 20)) {
            throw std::runtime_error("incompatible exact-PBS checkpoint");
        }
        std::string serialized_rng(rng_size, '\0');
        input.read(serialized_rng.data(), serialized_rng.size());
        std::istringstream rng_state(serialized_rng);
        rng_state >> rng_;
        nodes_.clear();
        nodes_.reserve(static_cast<size_t>(count * 1.3) + 1);
        for (uint64_t index = 0; index < count; ++index) {
            ExactInfoKey key{};
            RegretNode node{};
            input.read(reinterpret_cast<char*>(&key), sizeof(key));
            input.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes_.emplace(key, node);
        }
        if (!input || !rng_state) {
            throw std::runtime_error("truncated exact-PBS checkpoint");
        }
    }

private:
    RegretNode& node(const State& state, int actor) {
        const ExactInfoKey key = exact_key(state, actor);
        auto [it, inserted] = nodes_.try_emplace(key);
        if (inserted && blueprint_ && prior_strength_ > 0.0) {
            bool found = false;
            const auto prior = blueprint_->policy(state, actor, &found);
            for (int action = 0; action < kActionCount; ++action) {
                if (!(key.legal_mask & (1u << action))) continue;
                it->second.regrets[action] = prior_strength_ * prior[action];
                it->second.raw_regrets[action] = prior_strength_ * prior[action];
                it->second.strategy_sum[action] = prior_strength_ * prior[action];
            }
        }
        ++it->second.touches;
        return it->second;
    }

    double traverse(State state, int traverser) {
        ++node_visits_;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const ExactInfoKey key = exact_key(state, actor);
        RegretNode& current = node(state, actor);
        const auto strategy = current_strategy(current, mask);
        if (actor != traverser) {
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    current.strategy_sum[action] += strategy[action];
                }
            }
            const Action action = sample_action(strategy, mask, rng_);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_search_street(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_search_street(child);
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

    MCCFR* blueprint_ = nullptr;
    double prior_strength_ = 0.0;
    std::mt19937_64 rng_;
    std::unordered_map<ExactInfoKey, RegretNode, ExactInfoKeyHash> nodes_;
    uint64_t iterations_ = 0;
    uint64_t traversals_ = 0;
    uint64_t node_visits_ = 0;
};

State public_copy(const State& actual) {
    State result = actual;
    result.simulation_deck.clear();
    for (Player& player : result.players) {
        player.hidden.clear();
        player.discarded = {};
        player.has_discard = false;
    }
    return result;
}

uint64_t public_fingerprint(const State& state) {
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&](uint64_t value) {
        hash = (hash ^ value) * 1099511628211ull;
    };
    mix(state.ante);
    mix(state.effective_stack);
    mix(state.street);
    mix(state.actor);
    for (const Player& player : state.players) {
        for (const Card& card : player.shown) mix(card_token(card));
        mix(player.invested);
    }
    for (const Event& event : state.history) {
        mix(event.street);
        mix(event.actor);
        mix(event.action);
    }
    return hash;
}

struct ExactPbsOptions {
    uint64_t iterations = 100;
    uint64_t report_every = 10;
    uint64_t seed = 81001;
    int ante = 1000;
    int stack_ante = 1000;
    std::string metrics = "cpp_mccfr/rebel_exact_pbs_metrics.csv";
    std::string checkpoint;
    std::string resume;
    std::string blueprint_model;
    std::string blueprint_atlas;
    std::string blueprint_bucket = "power-memory16";
    std::string blueprint_algorithm = "mccfr";
    double blueprint_prior = 0.0;
    bool self_test = false;
};

ExactPbsOptions parse_exact_pbs_options(int argc, char** argv) {
    ExactPbsOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            return argv[index];
        };
        if (argument == "--iterations") options.iterations = std::stoull(value());
        else if (argument == "--report-every") options.report_every = std::stoull(value());
        else if (argument == "--seed") options.seed = std::stoull(value());
        else if (argument == "--ante") options.ante = std::stoi(value());
        else if (argument == "--stack-ante") options.stack_ante = std::stoi(value());
        else if (argument == "--metrics") options.metrics = value();
        else if (argument == "--checkpoint") options.checkpoint = value();
        else if (argument == "--resume") options.resume = value();
        else if (argument == "--blueprint-model") options.blueprint_model = value();
        else if (argument == "--blueprint-atlas") options.blueprint_atlas = value();
        else if (argument == "--blueprint-bucket") options.blueprint_bucket = value();
        else if (argument == "--blueprint-algorithm") options.blueprint_algorithm = value();
        else if (argument == "--blueprint-prior") options.blueprint_prior = std::stod(value());
        else if (argument == "--self-test") options.self_test = true;
        else throw std::runtime_error("unknown option: " + argument);
    }
    if (!options.report_every) options.report_every = options.iterations;
    return options;
}

void append_metrics(
    const std::string& path,
    const SolverMetrics& metrics,
    const FactorizedPBS& pbs,
    double elapsed_seconds) {
    const bool header = !std::filesystem::exists(path) ||
        std::filesystem::file_size(path) == 0;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot write metrics: " + path);
    if (header) {
        output << "iterations,traversals,node_visits,infosets,infosets_5th,"
            "infosets_6th,infosets_7th,cumulative_positive_regret_mean,"
            "average_positive_regret_mean,regret_bound_proxy,"
            "touch_normalized_regret,average_touches,"
            "revisited_infoset_fraction,average_policy_entropy,"
            "range0_support,range1_support,"
            "range0_effective_support,range1_effective_support,"
            "compatibility_acceptance,elapsed_seconds\n";
    }
    const RangeStats left = pbs.stats(0);
    const RangeStats right = pbs.stats(1);
    const double acceptance = pbs.proposals()
        ? 1.0 - pbs.collisions() / static_cast<double>(pbs.proposals())
        : 1.0;
    output << metrics.iterations << ',' << metrics.traversals << ','
        << metrics.node_visits << ',' << metrics.infosets << ','
        << metrics.infosets_by_street[0] << ','
        << metrics.infosets_by_street[1] << ','
        << metrics.infosets_by_street[2] << ','
        << std::setprecision(12)
        << metrics.cumulative_positive_regret_mean << ','
        << metrics.average_positive_regret_mean << ','
        << metrics.regret_bound_proxy << ','
        << metrics.touch_normalized_regret << ','
        << metrics.average_touches << ','
        << metrics.revisited_infoset_fraction << ','
        << metrics.average_policy_entropy << ','
        << left.support << ',' << right.support << ','
        << left.effective_support << ',' << right.effective_support << ','
        << acceptance << ',' << elapsed_seconds << '\n';
}

void configure_blueprint_memory(MCCFR& blueprint, const std::string& bucket) {
    if (bucket == "power-memory4") {
        blueprint.use_previous_street_summary(true);
    } else if (bucket == "power-memory16") {
        blueprint.use_cumulative_street_summary(true);
    } else if (bucket == "power-memory81") {
        blueprint.use_cumulative_street_strength_summary(true);
    } else if (bucket != "power") {
        throw std::runtime_error("unsupported blueprint bucket: " + bucket);
    }
}

void exact_pbs_self_test() {
    std::mt19937_64 rng(7);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    const State actual = sample_fifth_street_root(deck, 1000, 1000);
    const State root = public_copy(actual);
    FactorizedPBS pbs(root);
    assert(pbs.stats(0).support > 0 && pbs.stats(1).support > 0);
    assert(pbs.range(0).normalized_reach(
        actual.players[0].hidden,
        actual.players[0].discarded) > 0.0);
    for (int sample = 0; sample < 100; ++sample) {
        const State world = pbs.sample_world(root, rng);
        uint64_t mask = 0;
        for (const Player& player : world.players) {
            for (const Card& card : player.shown) {
                assert(!(mask & card_bit(card)));
                mask |= card_bit(card);
            }
            for (const Card& card : player.hidden) {
                assert(!(mask & card_bit(card)));
                mask |= card_bit(card);
            }
            assert(!(mask & card_bit(player.discarded)));
            mask |= card_bit(player.discarded);
        }
    }
    ExactPbsCfr solver(8, nullptr, 0.0);
    solver.train(root, pbs, 2, 0, [](const SolverMetrics&) {});
    assert(solver.metrics().iterations == 2);
    assert(solver.metrics().infosets > 0);
    const auto path = (
        std::filesystem::temp_directory_path() /
        "stud_rebel_exact_pbs_test.bin").string();
    solver.save(path, public_fingerprint(root));
    ExactPbsCfr loaded(9, nullptr, 0.0);
    loaded.load(path, public_fingerprint(root));
    assert(loaded.metrics().iterations == 2);
    assert(loaded.metrics().infosets == solver.metrics().infosets);
    std::filesystem::remove(path);
    std::cout << "{\"self_test\":\"ok\",\"range_support\":["
              << pbs.stats(0).support << ',' << pbs.stats(1).support
              << "],\"infosets\":" << solver.metrics().infosets << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ExactPbsOptions options = parse_exact_pbs_options(argc, argv);
        if (options.self_test) {
            exact_pbs_self_test();
            return 0;
        }
        if (options.blueprint_model.empty() != options.blueprint_atlas.empty()) {
            throw std::runtime_error(
                "--blueprint-model and --blueprint-atlas must be used together");
        }

        std::mt19937_64 deal_rng(options.seed);
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const State actual = sample_fifth_street_root(
            deck, options.ante, options.stack_ante);
        const State root = public_copy(actual);
        FactorizedPBS pbs(root);
        const uint64_t fingerprint = public_fingerprint(root);

        std::unique_ptr<PowerAtlas> atlas;
        std::unique_ptr<MCCFR> blueprint;
        if (!options.blueprint_model.empty()) {
            atlas = std::make_unique<PowerAtlas>(128);
            atlas->load(options.blueprint_atlas);
            const bool plus = options.blueprint_algorithm == "mccfr-plus";
            if (!plus && options.blueprint_algorithm != "mccfr") {
                throw std::runtime_error("invalid --blueprint-algorithm");
            }
            blueprint = std::make_unique<MCCFR>(
                plus, options.seed ^ 0x9e3779b97f4a7c15ull,
                5, atlas.get());
            configure_blueprint_memory(*blueprint, options.blueprint_bucket);
            blueprint->load(options.blueprint_model);
        }

        ExactPbsCfr solver(
            options.seed ^ 0xd1b54a32d192ed03ull,
            blueprint.get(), options.blueprint_prior);
        if (!options.resume.empty()) solver.load(options.resume, fingerprint);
        const auto started = std::chrono::steady_clock::now();
        solver.train(
            root, pbs, options.iterations, options.report_every,
            [&](const SolverMetrics& metrics) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                append_metrics(options.metrics, metrics, pbs, elapsed);
                if (!options.checkpoint.empty()) {
                    solver.save(options.checkpoint, fingerprint);
                }
                std::cout << "{\"iteration\":" << metrics.iterations
                    << ",\"infosets\":" << metrics.infosets
                    << ",\"regret_bound_proxy\":"
                    << std::setprecision(10) << metrics.regret_bound_proxy
                    << ",\"average_positive_regret\":"
                    << metrics.average_positive_regret_mean
                    << ",\"elapsed_seconds\":" << elapsed << "}\n";
            });

        const int actor = actual.actor;
        bool actual_found = false;
        const auto actual_policy = solver.policy(actual, actor, &actual_found);
        const Action observed = static_cast<Action>(std::distance(
            actual_policy.begin(),
            std::max_element(actual_policy.begin(), actual_policy.end())));
        const double before = pbs.range(actor).normalized_reach(
            actual.players[actor].hidden,
            actual.players[actor].discarded);
        uint64_t belief_found = 0;
        uint64_t belief_queries = 0;
        const double action_likelihood = pbs.observe_action(
            root, actor, observed,
            [&](const State& state, int seat, bool* found) {
                return solver.policy(state, seat, found);
            },
            belief_found, belief_queries);
        const double after = pbs.range(actor).normalized_reach(
            actual.players[actor].hidden,
            actual.players[actor].discarded);
        const SolverMetrics final = solver.metrics();
        std::cout << "{\n"
            << "  \"solver\": \"rebel-exact-factorized-pbs-reference\",\n"
            << "  \"root_fingerprint\": " << fingerprint << ",\n"
            << "  \"iterations\": " << final.iterations << ",\n"
            << "  \"traversals\": " << final.traversals << ",\n"
            << "  \"node_visits\": " << final.node_visits << ",\n"
            << "  \"infosets\": " << final.infosets << ",\n"
            << "  \"range_support\": [" << pbs.stats(0).support << ','
            << pbs.stats(1).support << "],\n"
            << "  \"regret_bound_proxy\": "
            << final.regret_bound_proxy << ",\n"
            << "  \"belief_update\": {\"actor\":" << actor
            << ",\"action\":\"" << kActionNames[observed]
            << "\",\"local_policy_found\":"
            << (actual_found ? "true" : "false")
            << ",\"action_likelihood\":" << action_likelihood
            << ",\"policy_coverage\":"
            << (belief_queries
                ? belief_found / static_cast<double>(belief_queries)
                : 0.0)
            << ",\"true_type_weight_before\":" << before
            << ",\"true_type_weight_after\":" << after << "},\n"
            << "  \"metrics\": \"" << options.metrics << "\",\n"
            << "  \"checkpoint\": \"" << options.checkpoint << "\"\n"
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
