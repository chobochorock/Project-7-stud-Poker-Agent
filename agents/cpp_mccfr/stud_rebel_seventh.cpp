#define STUD_MCCFR_NO_MAIN
#include "stud_mccfr.cpp"

namespace {

struct RebelPrivateDeal {
    std::array<std::array<Card, 3>, 2> hidden{};
    std::array<Card, 2> discarded{};
};

struct RebelParticle {
    State root;
    double weight = 0.0;
};

constexpr int kRebelBeliefCards = 52;
constexpr int kRebelBeliefSeatSize = kRebelBeliefCards * 2;
constexpr int kRebelBeliefSize = kRebelBeliefSeatSize * 2;
constexpr int kRebelV7FeatureSize = kDeepCfrTensorSize + kRebelBeliefSize;
using RebelBeliefSummary = std::array<float, kRebelBeliefSize>;
using RebelV7Features = std::array<float, kRebelV7FeatureSize>;

RebelBeliefSummary rebel_belief_summary(
    const std::vector<RebelParticle>& particles,
    int viewer) {
    if (particles.empty() || (viewer != 0 && viewer != 1)) {
        throw std::runtime_error("invalid ReBeL belief summary request");
    }
    RebelBeliefSummary summary{};
    double total = 0.0;
    for (const RebelParticle& particle : particles) {
        total += particle.weight;
        for (int relative_seat = 0; relative_seat < 2; ++relative_seat) {
            const int seat = relative_seat ? 1 - viewer : viewer;
            const Player& player = particle.root.players[seat];
            const int offset = relative_seat * kRebelBeliefSeatSize;
            for (const Card& card : player.hidden) {
                summary[offset + card_index(card)] +=
                    static_cast<float>(particle.weight);
            }
            if (player.has_discard) {
                summary[offset + kRebelBeliefCards +
                        card_index(player.discarded)] +=
                    static_cast<float>(particle.weight);
            }
        }
    }
    if (!(total > 0.0)) throw std::runtime_error("empty ReBeL belief mass");
    for (float& value : summary) value /= static_cast<float>(total);
    return summary;
}

RebelV7Features rebel_v7_features(
    const State& state,
    int viewer,
    const std::vector<RebelParticle>& particles) {
    RebelV7Features features{};
    const DeepCfrTensor tensor = deep_cfr_tensor(state, viewer);
    const RebelBeliefSummary belief = rebel_belief_summary(particles, viewer);
    std::copy(tensor.begin(), tensor.end(), features.begin());
    std::copy(
        belief.begin(), belief.end(),
        features.begin() + kDeepCfrTensorSize);
    return features;
}

class RebelSeventhResolverPolicy {
public:
    struct ValueSample {
        RebelV7Features features{};
        float value = 0.0f;
    };

    struct Stats {
        uint64_t subgames = 0;
        uint64_t proposals = 0;
        uint64_t h4_rejected = 0;
        uint64_t replay_rejected = 0;
        uint64_t particles = 0;
        uint64_t traversals = 0;
        uint64_t node_visits = 0;
        uint64_t nodes_created = 0;
        uint64_t policy_queries = 0;
        uint64_t blueprint_fallbacks = 0;
        uint64_t blueprint_history_misses = 0;
        double effective_particles_sum = 0.0;
        double posterior_entropy_sum = 0.0;
        size_t maximum_local_nodes = 0;
        size_t maximum_particles = 0;
    };

    RebelSeventhResolverPolicy(
        MCCFR& blueprint,
        PowerAtlas& atlas,
        int iterations,
        int particle_count,
        double prior_strength,
        uint64_t seed,
        double belief_epsilon = 0.0)
        : blueprint_(blueprint),
          atlas_(atlas),
          iterations_(iterations),
          particle_count_(particle_count),
          prior_strength_(prior_strength),
          belief_epsilon_(belief_epsilon),
          rng_(seed) {
        if (iterations <= 0 || particle_count <= 0 || prior_strength < 0.0 ||
            belief_epsilon < 0.0 || belief_epsilon > 1.0) {
            throw std::runtime_error("invalid ReBeL seventh configuration");
        }
        nodes_.reserve(4096);
        particles_.reserve(particle_count);
    }

    Action choose(const State& state, int viewer, int) {
        if (state.street != 7) return blueprint_.choose(state, viewer, 0);
        ensure_subgame(state);
        return sample(policy(state, viewer), valid_mask(state, viewer));
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        if (state.street != 7) return blueprint_.policy(state, viewer, found);
        ensure_subgame(state);
        ++stats_.policy_queries;
        const InfoKey key = make_power_key(state, viewer, atlas_, true);
        const auto position = nodes_.find(key);
        if (position != nodes_.end()) {
            if (found) *found = true;
            return average_strategy(position->second, valid_mask(state, viewer));
        }
        if (found) *found = false;
        ++stats_.blueprint_fallbacks;
        return blueprint_.policy(state, viewer);
    }

    Stats stats() const { return stats_; }

    const std::vector<RebelParticle>& public_belief(
        const State& public_root) const {
        if (public_root.street != 6 && public_root.street != 7) {
            throw std::runtime_error("public belief supports only 6th/7th");
        }
        build_public_belief(public_root);
        return particles_;
    }

    std::vector<ValueSample> value_samples(
        const State& public_root,
        int particle_samples) const {
        if (public_root.street != 7 || particle_samples <= 0) {
            throw std::runtime_error("invalid seventh value-sample request");
        }
        ensure_subgame(public_root);
        std::vector<double> weights;
        weights.reserve(particles_.size());
        for (const RebelParticle& particle : particles_) {
            weights.push_back(particle.weight);
        }
        std::discrete_distribution<size_t> chance(
            weights.begin(), weights.end());
        std::vector<ValueSample> result;
        result.reserve(static_cast<size_t>(particle_samples) * 2);
        for (int sample = 0; sample < particle_samples; ++sample) {
            const State& root = particles_[chance(rng_)].root;
            for (int player = 0; player < 2; ++player) {
                result.push_back({
                    rebel_v7_features(root, player, particles_),
                    static_cast<float>(evaluate_fixed_deal(root, player))});
            }
        }
        return result;
    }

private:
    static std::string public_root_id(const State& state) {
        std::ostringstream output;
        output << state.ante << ':' << state.effective_stack;
        for (int seat = 0; seat < 2; ++seat) {
            output << '|' << stack_cap(state, seat) << ':';
            for (const Card& card : state.players[seat].shown) {
                output << static_cast<int>(card_token(card)) << '.';
            }
        }
        output << '|';
        for (const Event& event : state.history) {
            if (event.street >= 7) continue;
            output << static_cast<int>(event.street) << '.'
                   << static_cast<int>(event.actor) << '.'
                   << static_cast<int>(event.action) << ';';
        }
        return output.str();
    }

    static int h4_outcome_multiplicity(
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

    static void advance_known_street(
        State& state,
        const State& public_state,
        const RebelPrivateDeal& deal) {
        if (state.street == 5) {
            state.players[0].shown.push_back(public_state.players[0].shown[3]);
            state.players[1].shown.push_back(public_state.players[1].shown[3]);
            reset_round(state, 6);
            return;
        }
        if (state.street == 6) {
            state.players[0].hidden.push_back(deal.hidden[0][2]);
            state.players[1].hidden.push_back(deal.hidden[1][2]);
            reset_round(state, 7);
            return;
        }
        throw std::runtime_error("cannot advance beyond seventh street");
    }

    bool reconstruct_root(
        const State& public_state,
        const State& expected_root,
        const RebelPrivateDeal& deal,
        State& result,
        double& log_likelihood) const {
        result = State{};
        result.ante = public_state.ante;
        result.effective_stack = public_state.effective_stack;
        result.pot = 2 * public_state.ante;
        for (int seat = 0; seat < 2; ++seat) {
            if (public_state.players[seat].shown.size() != 4) return false;
            Player& player = result.players[seat];
            player.stack_cap = public_state.players[seat].stack_cap;
            player.invested = public_state.ante;
            player.hidden.assign(
                deal.hidden[seat].begin(), deal.hidden[seat].begin() + 2);
            player.shown.assign(
                public_state.players[seat].shown.begin(),
                public_state.players[seat].shown.begin() + 3);
            player.discarded = deal.discarded[seat];
            player.has_discard = true;
        }
        reset_round(result, 5);
        log_likelihood = 0.0;

        for (const Event& event : public_state.history) {
            if (event.street >= 7) continue;
            while (result.street < event.street) {
                advance_known_street(result, public_state, deal);
            }
            if (result.terminal || result.actor != event.actor) return false;
            const uint8_t mask = valid_mask(result, event.actor);
            if (!(mask & (1u << event.action))) return false;
            bool found = false;
            const auto strategy = blueprint_.policy(result, event.actor, &found);
            if (!found) ++stats_.blueprint_history_misses;
            const double action_probability =
                (1.0 - belief_epsilon_) * strategy[event.action] +
                belief_epsilon_ / actions_from_mask(mask).size();
            log_likelihood += std::log(std::max(
                1e-12, action_probability));
            const ActionResult action_result =
                apply_action(result, event.actor, event.action);
            if (action_result == ActionResult::FoldEnd) return false;
            if (action_result != ActionResult::RoundEnd) {
                result.actor = 1 - event.actor;
            }
        }
        while (result.street < public_state.street) {
            advance_known_street(result, public_state, deal);
        }
        if (result.street == 6) {
            result.simulation_deck = {
                deal.hidden[1][2], deal.hidden[0][2]};
        } else {
            result.simulation_deck.clear();
        }
        return !result.terminal &&
            result.pot == expected_root.pot &&
            result.players[0].invested ==
                expected_root.players[0].invested &&
            result.players[1].invested ==
                expected_root.players[1].invested;
    }

    static State replay_public_amounts(const State& public_state) {
        State replay;
        replay.ante = public_state.ante;
        replay.effective_stack = public_state.effective_stack;
        replay.pot = 2 * public_state.ante;
        for (int seat = 0; seat < 2; ++seat) {
            replay.players[seat].stack_cap = public_state.players[seat].stack_cap;
            replay.players[seat].invested = public_state.ante;
            replay.players[seat].shown.assign(
                public_state.players[seat].shown.begin(),
                public_state.players[seat].shown.begin() + 3);
        }
        reset_round(replay, 5);
        for (const Event& event : public_state.history) {
            if (event.street >= 7) continue;
            while (replay.street < event.street) {
                if (replay.street == 5) {
                    replay.players[0].shown.push_back(
                        public_state.players[0].shown[3]);
                    replay.players[1].shown.push_back(
                        public_state.players[1].shown[3]);
                    reset_round(replay, 6);
                } else {
                    reset_round(replay, 7);
                }
            }
            if (!(valid_mask(replay, event.actor) & (1u << event.action))) {
                throw std::runtime_error("invalid public history replay");
            }
            const ActionResult action_result =
                apply_action(replay, event.actor, event.action);
            if (action_result != ActionResult::RoundEnd &&
                action_result != ActionResult::FoldEnd) {
                replay.actor = 1 - event.actor;
            }
        }
        while (replay.street < public_state.street) {
            if (replay.street == 5) {
                replay.players[0].shown.push_back(
                    public_state.players[0].shown[3]);
                replay.players[1].shown.push_back(
                    public_state.players[1].shown[3]);
                reset_round(replay, 6);
            } else if (replay.street == 6) {
                reset_round(replay, 7);
            } else {
                throw std::runtime_error("invalid public replay street");
            }
        }
        return replay;
    }

    void build_public_belief(const State& state) const {
        particles_.clear();
        std::array<bool, 52> known{};
        for (const Player& player : state.players) {
            for (const Card& card : player.shown) {
                const int index = card_index(card);
                if (known[index]) {
                    throw std::runtime_error("duplicate public card");
                }
                known[index] = true;
            }
        }
        std::vector<Card> remaining;
        for (const Card& card : fresh_deck()) {
            if (!known[card_index(card)]) remaining.push_back(card);
        }

        std::vector<double> log_weights;
        log_weights.reserve(particle_count_);
        const State expected_root = replay_public_amounts(state);
        const uint64_t maximum_attempts =
            static_cast<uint64_t>(particle_count_) * 10000;
        for (uint64_t attempt = 0;
             particles_.size() < static_cast<size_t>(particle_count_) &&
             attempt < maximum_attempts;
             ++attempt) {
            ++stats_.proposals;
            std::shuffle(remaining.begin(), remaining.end(), rng_);
            RebelPrivateDeal deal;
            size_t cursor = 0;
            for (int seat = 0; seat < 2; ++seat) {
                deal.hidden[seat][0] = remaining[cursor++];
                deal.hidden[seat][1] = remaining[cursor++];
                deal.discarded[seat] = remaining[cursor++];
                deal.hidden[seat][2] = remaining[cursor++];
            }
            const int ways0 = h4_outcome_multiplicity(
                {deal.hidden[0][0], deal.hidden[0][1]},
                state.players[0].shown[0], deal.discarded[0]);
            const int ways1 = h4_outcome_multiplicity(
                {deal.hidden[1][0], deal.hidden[1][1]},
                state.players[1].shown[0], deal.discarded[1]);
            if (!ways0 || !ways1) {
                ++stats_.h4_rejected;
                continue;
            }
            State particle_root;
            double action_log_likelihood = 0.0;
            if (!reconstruct_root(
                    state, expected_root, deal, particle_root,
                    action_log_likelihood)) {
                ++stats_.replay_rejected;
                continue;
            }
            log_weights.push_back(
                std::log(static_cast<double>(ways0 * ways1)) +
                action_log_likelihood);
            particles_.push_back({std::move(particle_root), 0.0});
        }
        if (particles_.empty()) {
            throw std::runtime_error("failed to construct seventh public belief");
        }

        const double maximum =
            *std::max_element(log_weights.begin(), log_weights.end());
        double total = 0.0;
        for (size_t index = 0; index < particles_.size(); ++index) {
            particles_[index].weight = std::exp(log_weights[index] - maximum);
            total += particles_[index].weight;
        }
        double squared = 0.0;
        double entropy = 0.0;
        for (RebelParticle& particle : particles_) {
            particle.weight /= total;
            squared += particle.weight * particle.weight;
            if (particle.weight > 0.0) {
                entropy -= particle.weight * std::log(particle.weight);
            }
        }
        stats_.particles += particles_.size();
        stats_.maximum_particles = std::max(
            stats_.maximum_particles, particles_.size());
        stats_.effective_particles_sum += 1.0 / squared;
        stats_.posterior_entropy_sum += entropy;
    }

    void ensure_subgame(const State& state) const {
        const std::string id = public_root_id(state);
        if (id == current_root_id_) return;
        current_root_id_ = id;
        nodes_.clear();
        build_public_belief(state);
        ++stats_.subgames;
        std::vector<double> weights;
        weights.reserve(particles_.size());
        for (const RebelParticle& particle : particles_) {
            weights.push_back(particle.weight);
        }
        std::discrete_distribution<size_t> chance(weights.begin(), weights.end());
        for (int iteration = 0; iteration < iterations_; ++iteration) {
            for (int traverser = 0; traverser < 2; ++traverser) {
                State simulation = particles_[chance(rng_)].root;
                traverse(std::move(simulation), traverser);
                ++stats_.traversals;
            }
        }
        stats_.maximum_local_nodes = std::max(
            stats_.maximum_local_nodes, nodes_.size());
    }

    RegretNode& node(const State& state, int actor) const {
        const InfoKey key = make_power_key(state, actor, atlas_, true);
        auto [position, inserted] = nodes_.try_emplace(key);
        if (inserted) {
            const auto prior = blueprint_.policy(state, actor);
            const uint8_t mask = valid_mask(state, actor);
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
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

    double traverse(State state, int traverser) const {
        ++stats_.node_visits;
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = make_power_key(state, actor, atlas_, true);
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
            const double regret = action_values[action] - value;
            updated.raw_regrets[action] += regret;
            updated.regrets[action] = std::max(
                0.0, updated.regrets[action] + regret);
        }
        return value;
    }

    double evaluate_fixed_deal(State state, int player) const {
        if (state.terminal) return terminal_net_search(state, player);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = make_power_key(state, actor, atlas_, true);
        const auto position = nodes_.find(key);
        const auto strategy = position == nodes_.end()
            ? blueprint_.policy(state, actor)
            : average_strategy(position->second, mask);
        double value = 0.0;
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) child.terminal = true;
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            value += strategy[action] *
                evaluate_fixed_deal(std::move(child), player);
        }
        return value;
    }

    Action sample(
        const std::array<double, kActionCount>& strategy,
        uint8_t mask) const {
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
    PowerAtlas& atlas_;
    int iterations_;
    int particle_count_;
    double prior_strength_;
    double belief_epsilon_ = 0.0;
    mutable std::mt19937_64 rng_;
    mutable std::string current_root_id_;
    mutable std::vector<RebelParticle> particles_;
    mutable std::unordered_map<InfoKey, RegretNode, InfoKeyHash> nodes_;
    mutable Stats stats_;
};

void print_rebel_stats(
    const RebelSeventhResolverPolicy::Stats& stats,
    const PowerCacheStats& cache) {
    const double subgames = std::max<uint64_t>(1, stats.subgames);
    std::cerr << std::fixed << std::setprecision(8)
              << "{\"rebel_seventh\":{"
              << "\"subgames\":" << stats.subgames
              << ",\"proposals\":" << stats.proposals
              << ",\"h4_rejected\":" << stats.h4_rejected
              << ",\"replay_rejected\":" << stats.replay_rejected
              << ",\"particles\":" << stats.particles
              << ",\"average_effective_particles\":"
              << stats.effective_particles_sum / subgames
              << ",\"average_posterior_entropy\":"
              << stats.posterior_entropy_sum / subgames
              << ",\"traversals\":" << stats.traversals
              << ",\"node_visits\":" << stats.node_visits
              << ",\"nodes_created\":" << stats.nodes_created
              << ",\"policy_queries\":" << stats.policy_queries
              << ",\"counterfactual_policy_consistent\":true"
              << ",\"policy_definition\":"
                 "\"joint-public-root-with-blueprint-fallback\""
              << ",\"local_policy_coverage\":"
              << (stats.policy_queries
                    ? 1.0 - stats.blueprint_fallbacks /
                        static_cast<double>(stats.policy_queries)
                    : 1.0)
              << ",\"maximum_local_nodes\":" << stats.maximum_local_nodes
              << ",\"maximum_particles\":" << stats.maximum_particles
              << ",\"blueprint_fallbacks\":" << stats.blueprint_fallbacks
              << ",\"blueprint_history_misses\":"
              << stats.blueprint_history_misses
              << ",\"power_cache_entries\":" << cache.entries
              << ",\"power_cache_limit\":" << cache.limit
              << ",\"power_cache_resets\":" << cache.resets
              << "}}\n";
}

struct RebelOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string bucket = "power-memory16";
    int hands = 1000;
    int lbr_particles = 64;
    int rebel_particles = 64;
    int rebel_iterations = 100;
    double prior = 100.0;
    int ante = 1000;
    int stack_ante = 1000;
    int progress_seconds = 30;
    uint64_t power_cache_max = 250000;
    uint64_t seed = 71001;
    bool self_test = false;
};

RebelOptions parse_rebel_options(int argc, char** argv) {
    RebelOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") options.model = value(index);
        else if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--bucket") options.bucket = value(index);
        else if (argument == "--hands") options.hands = std::stoi(value(index));
        else if (argument == "--lbr-particles") {
            options.lbr_particles = std::stoi(value(index));
        } else if (argument == "--rebel-particles") {
            options.rebel_particles = std::stoi(value(index));
        } else if (argument == "--rebel-iterations") {
            options.rebel_iterations = std::stoi(value(index));
        } else if (argument == "--prior") {
            options.prior = std::stod(value(index));
        } else if (argument == "--ante") {
            options.ante = std::stoi(value(index));
        } else if (argument == "--stack-ante") {
            options.stack_ante = std::stoi(value(index));
        } else if (argument == "--progress-seconds") {
            options.progress_seconds = std::stoi(value(index));
        } else if (argument == "--power-cache-max") {
            options.power_cache_max = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.hands <= 0 || options.hands % 2 ||
        options.lbr_particles <= 0 || options.rebel_particles <= 0 ||
        options.rebel_iterations <= 0 || options.prior < 0.0 ||
        options.ante <= 0 || options.stack_ante <= 0) {
        throw std::runtime_error("invalid ReBeL CLI configuration");
    }
    if (options.bucket != "power" &&
        options.bucket != "power-tree" &&
        options.bucket != "power-memory16") {
        throw std::runtime_error("unsupported ReBeL bucket mode");
    }
    return options;
}

}  // namespace

#ifndef STUD_REBEL_SEVENTH_NO_MAIN
int main(int argc, char** argv) {
    try {
        RebelOptions rebel = parse_rebel_options(argc, argv);
        PowerAtlas atlas;
        atlas.load(rebel.atlas);
        atlas.set_assignment_cache_limit(rebel.power_cache_max);
        MCCFR blueprint(false, rebel.seed, 5, &atlas);
        blueprint.use_cumulative_street_summary(
            rebel.bucket == "power-memory16");
        blueprint.load(rebel.model);
        RebelSeventhResolverPolicy resolver(
            blueprint,
            atlas,
            rebel.rebel_iterations,
            rebel.rebel_particles,
            rebel.prior,
            rebel.seed ^ 0x6a09e667f3bcc909ull);

        Options options;
        options.load_path = rebel.model;
        options.load_atlas_path = rebel.atlas;
        options.hands = rebel.self_test ? 10 : rebel.hands;
        options.belief_particles = rebel.self_test ? 16 : rebel.lbr_particles;
        options.ante = rebel.ante;
        options.stack_ante = rebel.stack_ante;
        options.seed = rebel.seed;
        options.start_street = 5;
        options.iterations = 0;
        options.progress_seconds = rebel.progress_seconds;
        options.seventh_resolve_iterations = rebel.rebel_iterations;
        const int result = run_policy_lbr(
            options, resolver, "cpp-rebel-7th-pbs");
        std::cout.flush();
        const auto stats = resolver.stats();
        print_rebel_stats(stats, atlas.assignment_cache_stats());
        if (rebel.self_test) {
            if (!stats.subgames || !stats.particles || !stats.nodes_created ||
                stats.maximum_particles >
                    static_cast<size_t>(rebel.rebel_particles)) {
                throw std::runtime_error("ReBeL seventh self-test failed");
            }
            std::cerr << "{\"self_test\":\"ok\"}\n";
        }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#endif
