#define STUD5_NO_MAIN
#include "stud5_mccfr.cpp"

namespace {

constexpr uint32_t kTeamModelVersion = 2;

struct TeamInfoKey {
    FiveInfoKey base;
    uint32_t public_tail = 0;
    uint8_t public_tail_length = 0;

    bool operator==(const TeamInfoKey& other) const {
        return base == other.base &&
            public_tail == other.public_tail &&
            public_tail_length == other.public_tail_length;
    }
};

struct TeamInfoHash {
    size_t operator()(const TeamInfoKey& key) const {
        size_t hash = FiveInfoHash{}(key.base);
        hash ^= key.public_tail + 0x9e3779b97f4a7c15ull +
            (hash << 6) + (hash >> 2);
        hash ^= key.public_tail_length + 0x9e3779b97f4a7c15ull +
            (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct TeamNode {
    std::array<double, kActionCount> regret{};
    std::array<double, kActionCount> strategy_sum{};
    uint64_t touches = 0;
};

int team_role(int actor, int target) {
    const int relative = (actor - target + kSeats5) % kSeats5;
    if (relative == 0) throw std::runtime_error("target has no team role");
    return relative - 1;
}

class AlternatingTeamLBR {
public:
    AlternatingTeamLBR(FiveMCCFR& target, uint64_t seed)
        : target_(target), rng_(seed) {}

    TeamInfoKey make_key(
        const FiveState& state,
        int actor) {
        TeamInfoKey key;
        key.base = target_.make_key(state, actor);
        const size_t count = std::min<size_t>(1, state.history.size());
        key.public_tail_length = static_cast<uint8_t>(count);
        const size_t start = state.history.size() - count;
        for (size_t index = start; index < state.history.size(); ++index) {
            const auto& event = state.history[index];
            const int relative =
                (static_cast<int>(event.actor) - actor + kSeats5) % kSeats5;
            const uint32_t token = static_cast<uint32_t>(
                1 + relative * kActionCount + event.action);
            key.public_tail = (key.public_tail << 6) | token;
        }
        return key;
    }

    double train_root(
        const FiveState& root,
        int target_seat,
        int optimizing_role) {
        ++traversals_;
        return traverse(root, target_seat, optimizing_role);
    }

    Action choose(
        const FiveState& state,
        int target_seat,
        int actor,
        std::array<uint64_t, kActionCount>* counts = nullptr) {
        const uint8_t mask = five_valid_mask(state, actor);
        const int role = team_role(actor, target_seat);
        const TeamInfoKey key = make_key(state, actor);
        const auto it = nodes_[role].find(key);
        const auto strategy = it == nodes_[role].end()
            ? fallback_strategy(state, actor, mask)
            : average_strategy(it->second, state, actor, mask);
        if (it == nodes_[role].end()) ++policy_misses_;
        const Action action = sample(strategy, mask);
        if (counts) ++(*counts)[action];
        return action;
    }

    size_t buckets() const {
        size_t total = 0;
        for (const auto& role : nodes_) total += role.size();
        return total;
    }

    std::array<size_t, 4> buckets_by_role() const {
        std::array<size_t, 4> result{};
        for (int role = 0; role < 4; ++role) {
            result[role] = nodes_[role].size();
        }
        return result;
    }

    uint64_t node_visits() const { return node_visits_; }
    uint64_t traversals() const { return traversals_; }
    uint64_t policy_misses() const { return policy_misses_; }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot save team model");
        const char magic[8] = {'S','5','T','L','B','R','0','2'};
        out.write(magic, sizeof(magic));
        out.write(
            reinterpret_cast<const char*>(&kTeamModelVersion),
            sizeof(kTeamModelVersion));
        for (const auto& role : nodes_) {
            const uint64_t count = role.size();
            out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            for (const auto& [key, node] : role) {
                out.write(reinterpret_cast<const char*>(&key), sizeof(key));
                out.write(reinterpret_cast<const char*>(&node), sizeof(node));
            }
        }
        if (!out) throw std::runtime_error("failed to save team model");
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot load team model");
        char magic[8]{};
        uint32_t version = 0;
        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (std::string(magic, sizeof(magic)) != "S5TLBR02" ||
            version != kTeamModelVersion) {
            throw std::runtime_error("invalid team model");
        }
        for (auto& role : nodes_) {
            uint64_t count = 0;
            in.read(reinterpret_cast<char*>(&count), sizeof(count));
            role.clear();
            for (uint64_t index = 0; index < count; ++index) {
                TeamInfoKey key;
                TeamNode node;
                in.read(reinterpret_cast<char*>(&key), sizeof(key));
                in.read(reinterpret_cast<char*>(&node), sizeof(node));
                role.emplace(key, node);
            }
        }
        if (!in) throw std::runtime_error("truncated team model");
    }

private:
    using RoleNodes =
        std::unordered_map<TeamInfoKey, TeamNode, TeamInfoHash>;

    std::array<double, kActionCount> fallback_strategy(
        const FiveState& state,
        int actor,
        uint8_t mask) const {
        std::array<double, kActionCount> strategy{};
        if (mask) strategy[five_heuristic(state, actor)] = 1.0;
        return strategy;
    }

    std::array<double, kActionCount> current_strategy(
        const TeamNode& node,
        const FiveState& state,
        int actor,
        uint8_t mask) const {
        std::array<double, kActionCount> strategy{};
        double total = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = std::max(0.0, node.regret[action]);
            total += strategy[action];
        }
        if (total == 0.0) return fallback_strategy(state, actor, mask);
        for (double& probability : strategy) probability /= total;
        return strategy;
    }

    std::array<double, kActionCount> average_strategy(
        const TeamNode& node,
        const FiveState& state,
        int actor,
        uint8_t mask) const {
        std::array<double, kActionCount> strategy{};
        double total = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            strategy[action] = std::max(0.0, node.strategy_sum[action]);
            total += strategy[action];
        }
        if (total == 0.0) return fallback_strategy(state, actor, mask);
        for (double& probability : strategy) probability /= total;
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

    double traverse(
        FiveState state,
        int target_seat,
        int optimizing_role) {
        ++node_visits_;
        if (state.terminal) return -five_payoffs(state)[target_seat];
        const int actor = state.actor;
        const uint8_t mask = five_valid_mask(state, actor);
        if (!mask) {
            state.pending &= static_cast<uint8_t>(~(1u << actor));
            finish_if_needed(state, actor);
            return traverse(std::move(state), target_seat, optimizing_role);
        }

        if (actor == target_seat) {
            const Action action = target_.choose(state, actor);
            five_apply(state, action);
            return traverse(std::move(state), target_seat, optimizing_role);
        }

        const int role = team_role(actor, target_seat);
        if (role != optimizing_role) {
            const Action action = choose(state, target_seat, actor);
            five_apply(state, action);
            return traverse(std::move(state), target_seat, optimizing_role);
        }

        const TeamInfoKey key = make_key(state, actor);
        RoleNodes& role_nodes = nodes_[role];
        auto [it, inserted] = role_nodes.try_emplace(key);
        TeamNode& node = it->second;
        ++node.touches;
        const auto strategy = current_strategy(node, state, actor, mask);
        for (int action = 0; action < kActionCount; ++action) {
            node.strategy_sum[action] += strategy[action];
        }

        std::array<double, kActionCount> values{};
        double expected = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            FiveState child = state;
            five_apply(child, static_cast<Action>(action));
            values[action] =
                traverse(std::move(child), target_seat, optimizing_role);
            expected += strategy[action] * values[action];
        }

        TeamNode& updated = role_nodes.find(key)->second;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            updated.regret[action] = std::max(
                0.0,
                updated.regret[action] + values[action] - expected);
        }
        return expected;
    }

    FiveMCCFR& target_;
    std::mt19937_64 rng_;
    std::array<RoleNodes, 4> nodes_;
    uint64_t node_visits_ = 0;
    uint64_t traversals_ = 0;
    uint64_t policy_misses_ = 0;
};

struct TeamOptions {
    std::string target_model;
    std::string load_team;
    std::string save_team;
    uint64_t seed = 31001;
    int cycles = 1;
    int roots_per_role = 1000;
    int eval_deals = 1000;
    int mc_samples = 16;
    int stack_min = 50;
    int stack_max = 1000;
    double progress_seconds = 30.0;
    bool self_test = false;
};

TeamOptions parse_team_options(int argc, char** argv) {
    TeamOptions options;
    auto value = [&](int& index) {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return std::string(argv[index]);
    };
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--target-model") {
            options.target_model = value(index);
        } else if (arg == "--load-team") {
            options.load_team = value(index);
        } else if (arg == "--save-team") {
            options.save_team = value(index);
        } else if (arg == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (arg == "--cycles") {
            options.cycles = std::stoi(value(index));
        } else if (arg == "--roots-per-role") {
            options.roots_per_role = std::stoi(value(index));
        } else if (arg == "--eval-deals") {
            options.eval_deals = std::stoi(value(index));
        } else if (arg == "--mc-samples") {
            options.mc_samples = std::stoi(value(index));
        } else if (arg == "--stack-min") {
            options.stack_min = std::stoi(value(index));
        } else if (arg == "--stack-max") {
            options.stack_max = std::stoi(value(index));
        } else if (arg == "--progress-seconds") {
            options.progress_seconds = std::stod(value(index));
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown team option: " + arg);
        }
    }
    if (!options.self_test && options.target_model.empty()) {
        throw std::runtime_error("--target-model is required");
    }
    if (options.cycles < 0 ||
        options.roots_per_role < 0 ||
        options.eval_deals <= 0 ||
        options.mc_samples <= 0 ||
        options.stack_min <= 0 ||
        options.stack_min > options.stack_max ||
        options.progress_seconds < 0) {
        throw std::runtime_error("invalid team options");
    }
    return options;
}

struct TeamEvaluation {
    std::vector<double> learned;
    std::vector<double> heuristic;
    std::vector<double> improvement;
    std::array<uint64_t, kActionCount> team_actions{};
};

void play_team_hand(
    FiveState& state,
    int target_seat,
    FiveMCCFR& target,
    AlternatingTeamLBR* team,
    std::array<uint64_t, kActionCount>* counts) {
    while (!state.terminal) {
        const int actor = state.actor;
        const Action action = actor == target_seat
            ? target.choose(state, actor)
            : team
                ? team->choose(state, target_seat, actor, counts)
                : five_heuristic(state, actor);
        five_apply(state, action);
    }
}

TeamEvaluation evaluate_team(
    AlternatingTeamLBR& team,
    FiveMCCFR& target_learned,
    FiveMCCFR& target_heuristic,
    int deals,
    uint64_t seed,
    int stack_min,
    int stack_max,
    double progress_seconds) {
    TeamEvaluation result;
    result.learned.reserve(deals);
    result.heuristic.reserve(deals);
    result.improvement.reserve(deals);
    std::mt19937_64 rng(seed);
    const auto started = std::chrono::steady_clock::now();
    ProgressHeartbeat heartbeat(
        "alternating-team-lbr", "evaluation", "base_deals",
        deals, progress_seconds);
    for (int deal = 0; deal < deals; ++deal) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        const auto stacks =
            sample_five_stacks(rng, stack_min, stack_max);
        double learned = 0.0;
        double heuristic = 0.0;
        for (int target_seat = 0; target_seat < kSeats5; ++target_seat) {
            const FiveState root =
                make_five_root(deck, stacks, 1).seventh;
            FiveState learned_state = root;
            play_team_hand(
                learned_state,
                target_seat,
                target_learned,
                &team,
                &result.team_actions);
            learned += -five_payoffs(learned_state)[target_seat];

            FiveState heuristic_state = root;
            play_team_hand(
                heuristic_state,
                target_seat,
                target_heuristic,
                nullptr,
                nullptr);
            heuristic += -five_payoffs(heuristic_state)[target_seat];
        }
        learned /= kSeats5;
        heuristic /= kSeats5;
        result.learned.push_back(learned);
        result.heuristic.push_back(heuristic);
        result.improvement.push_back(learned - heuristic);
        const int completed = deal + 1;
        heartbeat.update(completed);
        if (completed == deals) {
            const auto now = std::chrono::steady_clock::now();
            const double seconds =
                std::chrono::duration<double>(now - started).count();
            const double rate = completed / std::max(1e-9, seconds);
            const double eta =
                (deals - completed) / std::max(1e-9, rate);
            std::ostringstream output;
            output << "{\"solver\":\"alternating-team-lbr\""
                   << ",\"phase\":\"evaluation\""
                   << ",\"base_deals_completed\":" << completed
                   << ",\"base_deals_total\":" << deals
                   << ",\"hands_completed\":" << completed * kSeats5
                   << ",\"progress_percent\":"
                   << 100.0 * completed / deals
                   << ",\"base_deals_per_second\":" << rate
                   << ",\"elapsed_seconds\":" << seconds
                   << ",\"eta_seconds\":" << eta << "}";
            write_progress(output.str());
        }
    }
    return result;
}

void team_self_test() {
    const auto samples =
        collect_five_atlas_samples(8, 4, 41, 50, 1000);
    FiveAtlas atlas;
    atlas.fit(samples.power, samples.stacks, 4, 2, 43);
    FiveMCCFR target(std::move(atlas), 4, 47);
    AlternatingTeamLBR team(target, 53);

    std::mt19937_64 rng(59);
    FiveState root;
    for (int attempt = 0; attempt < 100; ++attempt) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        root = make_five_root(
            deck, std::array<int, kSeats5>{100, 200, 300, 400, 500}, 1)
            .seventh;
        if (!root.terminal) break;
    }
    assert(!root.terminal);
    const int target_seat = (root.actor + 1) % kSeats5;
    const int actor = root.actor;
    const TeamInfoKey original = team.make_key(root, actor);
    FiveState hidden_changed = root;
    const int other = (actor + 2) % kSeats5;
    hidden_changed.players[other].hidden[0].rank =
        hidden_changed.players[other].hidden[0].rank == 14
        ? 2
        : hidden_changed.players[other].hidden[0].rank + 1;
    assert(team.make_key(hidden_changed, actor) == original);
    FiveState public_changed = root;
    public_changed.history.push_back(
        FiveEvent{static_cast<uint8_t>(target_seat), CHECK});
    assert(!(team.make_key(public_changed, actor) == original));

    const double value = team.train_root(
        root, target_seat, team_role(actor, target_seat));
    assert(std::isfinite(value));
    assert(team.buckets() > 0);
    std::cout << "{\"self_test\":\"ok\",\"hidden_information_shared\":false,"
              << "\"buckets\":" << team.buckets() << "}\n";
}

void print_team_result(
    const TeamOptions& options,
    const AlternatingTeamLBR& team,
    const TeamEvaluation& evaluation,
    uint64_t training_roots,
    double elapsed_seconds,
    uint64_t target_policy_misses,
    uint64_t training_team_policy_misses,
    uint64_t evaluation_team_policy_misses) {
    const NumericSummary learned = summarize_five(evaluation.learned);
    const NumericSummary heuristic = summarize_five(evaluation.heuristic);
    const NumericSummary improvement = summarize_five(evaluation.improvement);
    const auto role_buckets = team.buckets_by_role();
    auto interval = [](const NumericSummary& value) {
        return std::array<double, 2>{
            value.mean - 1.96 * value.standard_error,
            value.mean + 1.96 * value.standard_error};
    };
    const auto learned_ci = interval(learned);
    const auto heuristic_ci = interval(heuristic);
    const auto improvement_ci = interval(improvement);

    std::cout << std::fixed << std::setprecision(8)
              << "{\n"
              << "  \"solver\": \"alternating-team-lbr\",\n"
              << "  \"target_model\": \"" << options.target_model << "\",\n"
              << "  \"trained_street\": \"7th\",\n"
              << "  \"objective\": \"minus-target-chip-payoff\",\n"
              << "  \"training_cycles\": " << options.cycles << ",\n"
              << "  \"roots_per_role_per_cycle\": "
              << options.roots_per_role << ",\n"
              << "  \"training_roots\": " << training_roots << ",\n"
              << "  \"training_traversals\": " << team.traversals() << ",\n"
              << "  \"training_node_visits\": " << team.node_visits() << ",\n"
              << "  \"team_buckets\": " << team.buckets() << ",\n"
              << "  \"buckets_by_role\": ["
              << role_buckets[0] << "," << role_buckets[1] << ","
              << role_buckets[2] << "," << role_buckets[3] << "],\n"
              << "  \"evaluation\": {\n"
              << "    \"meaning\": \"achieved lower bound on B4(target);"
              << " not exact exploitability without v_C_star\",\n"
              << "    \"base_deals\": " << options.eval_deals << ",\n"
              << "    \"hands\": " << options.eval_deals * kSeats5 << ",\n"
              << "    \"learned_team_profit_ante\": {\"mean\":"
              << learned.mean << ",\"standard_error\":"
              << learned.standard_error << ",\"ci95\":["
              << learned_ci[0] << "," << learned_ci[1] << "]},\n"
              << "    \"heuristic_team_profit_ante\": {\"mean\":"
              << heuristic.mean << ",\"standard_error\":"
              << heuristic.standard_error << ",\"ci95\":["
              << heuristic_ci[0] << "," << heuristic_ci[1] << "]},\n"
              << "    \"gain_over_heuristic_team_ante\": {\"mean\":"
              << improvement.mean << ",\"standard_error\":"
              << improvement.standard_error << ",\"ci95\":["
              << improvement_ci[0] << "," << improvement_ci[1] << "]},\n"
              << "    \"target_policy_misses\": "
              << target_policy_misses << ",\n"
              << "    \"training_team_policy_misses\": "
              << training_team_policy_misses << ",\n"
              << "    \"evaluation_team_policy_misses\": "
              << evaluation_team_policy_misses << ",\n"
              << "    \"team_action_counts\": {";
    bool first = true;
    for (int action = 0; action < kActionCount; ++action) {
        if (!evaluation.team_actions[action]) continue;
        if (!first) std::cout << ",";
        first = false;
        std::cout << "\"" << kActionNames[action] << "\":"
                  << evaluation.team_actions[action];
    }
    std::cout << "}\n"
              << "  },\n"
              << "  \"elapsed_seconds\": " << elapsed_seconds << "\n"
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const TeamOptions options = parse_team_options(argc, argv);
        if (options.self_test) {
            team_self_test();
            return 0;
        }

        const auto started = std::chrono::steady_clock::now();
        FiveMCCFR target(FiveAtlas{}, options.mc_samples, options.seed + 1);
        target.load(options.target_model);
        AlternatingTeamLBR team(target, options.seed + 2);
        if (!options.load_team.empty()) team.load(options.load_team);

        std::mt19937_64 root_rng(options.seed + 3);
        uint64_t roots = 0;
        const uint64_t total_roots =
            static_cast<uint64_t>(options.cycles) * 4 *
            options.roots_per_role;
        const double total_roots_denominator =
            static_cast<double>(std::max<uint64_t>(1, total_roots));
        {
            const auto training_started =
                std::chrono::steady_clock::now();
            ProgressHeartbeat heartbeat(
                "alternating-team-lbr", "training", "roots",
                total_roots, options.progress_seconds);
            for (int cycle = 0; cycle < options.cycles; ++cycle) {
                for (int role = 0; role < 4; ++role) {
                    for (int index = 0;
                         index < options.roots_per_role;
                         ++index) {
                        auto deck = fresh_deck();
                        std::shuffle(deck.begin(), deck.end(), root_rng);
                        const auto stacks = sample_five_stacks(
                            root_rng, options.stack_min, options.stack_max);
                        const int target_seat =
                            static_cast<int>(roots % kSeats5);
                        const FiveState root =
                            make_five_root(deck, stacks, 1).seventh;
                        team.train_root(root, target_seat, role);
                        ++roots;
                        heartbeat.update(roots);
                    }
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        training_started).count();
                    const double rate = roots / std::max(1e-9, seconds);
                    std::ostringstream output;
                    output << "{\"solver\":\"alternating-team-lbr\""
                           << ",\"cycle\":" << cycle + 1
                           << ",\"phase\":\"training\""
                           << ",\"optimized_role\":" << role
                           << ",\"training_roots\":" << roots
                           << ",\"training_roots_total\":" << total_roots
                           << ",\"progress_percent\":"
                           << 100.0 * roots / total_roots_denominator
                           << ",\"team_buckets\":" << team.buckets()
                           << ",\"node_visits\":" << team.node_visits()
                           << ",\"roots_per_second\":" << rate
                           << ",\"elapsed_seconds\":" << seconds
                           << ",\"eta_seconds\":"
                           << (total_roots - roots) /
                               std::max(1e-9, rate)
                           << "}";
                    write_progress(output.str());
                }
            }
        }
        if (!options.save_team.empty()) team.save(options.save_team);
        const uint64_t training_team_policy_misses = team.policy_misses();

        FiveMCCFR target_learned(
            FiveAtlas{}, options.mc_samples, options.seed + 100);
        FiveMCCFR target_heuristic(
            FiveAtlas{}, options.mc_samples, options.seed + 100);
        target_learned.load(options.target_model);
        target_heuristic.load(options.target_model);
        const TeamEvaluation evaluation = evaluate_team(
            team,
            target_learned,
            target_heuristic,
            options.eval_deals,
            options.seed + 101,
            options.stack_min,
            options.stack_max,
            options.progress_seconds);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        print_team_result(
            options,
            team,
            evaluation,
            roots,
            elapsed,
            target_learned.policy_misses() +
                target_heuristic.policy_misses(),
            training_team_policy_misses,
            team.policy_misses() - training_team_policy_misses);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
