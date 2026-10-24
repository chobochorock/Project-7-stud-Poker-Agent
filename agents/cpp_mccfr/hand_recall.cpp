// Frozen hand k-means: abstract-observation recall versus current-hand only.
// Usage/build: python -m agents.cpp_mccfr.hand_recall --help; see HAND_RECALL.md.
// Both arms retain exact public betting history and use signed external MCCFR.
// Reuses the frozen v3 rules, power features and conditional-root gap evaluator.
#define main epoch_training_entrypoint
#include "../lightgbm_regret_ensemble/stud_epoch_ensemble.cpp"
#undef main

namespace hand_recall {
namespace fs = std::filesystem;
using epoch7::Values;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void number(std::string& key, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        key.push_back(static_cast<char>((value >> shift) & 255));
    }
}

State card_prefix(const State& state, int street) {
    require(street >= 5 && street <= state.street, "invalid card prefix");
    State result = state;
    result.street = street;
    result.simulation_deck.clear();
    for (auto& player : result.players) {
        require(player.shown.size() >= size_t(street == 5 ? 3 : 4), "short public hand");
        player.shown.resize(street == 5 ? 3 : 4);
        // The seventh card is private. Never let it enter a fifth/sixth bucket.
        if (street < 7 && player.hidden.size() > 2) { player.hidden.resize(2); }
    }
    return result;
}

std::string information_key(const State& state, int viewer,
                            const PowerAtlas& atlas, bool recall) {
    require(viewer == state.actor && !state.terminal, "key requires acting player");
    std::string key;
    key.reserve(110);
    key.push_back(static_cast<char>(viewer));
    key.push_back(static_cast<char>(state.street));
    key.push_back(static_cast<char>(valid_mask(state, viewer)));
    number(key, state.ante);
    for (int seat = 0; seat < 2; ++seat) { number(key, stack_cap(state, seat)); }
    // Keep every observed street's opening actor in BOTH arms. With exact
    // actions, ante and caps, all past public chip states are reconstructible.
    for (int street = 5; street <= state.street; ++street) {
        const State prefix = card_prefix(state, street);
        key.push_back(static_cast<char>(first_bettor(prefix)));
        if (recall || street == state.street) {
            const auto bucket = atlas.assign(prefix, viewer);
            key.push_back(static_cast<char>(bucket & 255));
            key.push_back(static_cast<char>(bucket >> 8));
        }
    }
    number(key, static_cast<uint32_t>(state.history.size()));
    for (const auto& event : state.history) {
        key.push_back(event.street);
        key.push_back(event.actor);
        key.push_back(event.action);
    }
    // No ratio bins or truncated histories. Amounts paid are deterministic
    // from the exact action prefix under the existing integer-chip rules.
    for (int value : {state.pot, state.highest_bet, state.raise_count}) {
        number(key, value);
    }
    for (const auto& player : state.players) {
        number(key, player.invested);
        number(key, player.round_bet);
        key.push_back(player.folded);
        key.push_back(player.all_in);
    }
    return key;
}

struct Solver {
    const PowerAtlas& atlas;
    bool recall;
    std::mt19937_64 rng;
    std::unordered_map<std::string, RegretNode> table;
    uint64_t nodes = 0;
    uint64_t key_bytes = 0;

    Solver(const PowerAtlas& source, bool memory, uint64_t seed)
        : atlas(source), recall(memory), rng(seed) {}

    double traverse(State state, int traverser) {
        ++nodes;
        if (state.terminal) { return terminal_net_search(state, traverser); }
        const int actor = state.actor;
        const auto mask = valid_mask(state, actor);
        const std::string key = information_key(state, actor, atlas, recall);
        auto inserted = table.try_emplace(key);
        if (inserted.second) { key_bytes += key.size(); }
        RegretNode& node = inserted.first->second;
        ++node.touches;
        const auto strategy = current_strategy(node, mask);
        if (actor != traverser) {
            // Existing external-sampling SIMPLE average: accumulate only at
            // sampled opponent nodes. No extra own-reach or linear multiplier.
            for (Action action : actions_from_mask(mask)) {
                node.strategy_sum[action] += strategy[action];
            }
            const Action action = epoch7::sample(strategy, mask, rng);
            return traverse(epoch7::child(std::move(state), action), traverser);
        }
        Values action_values{};
        double value = 0;
        for (Action action : actions_from_mask(mask)) {
            action_values[action] = traverse(epoch7::child(state, action), traverser);
            value += strategy[action] * action_values[action];
        }
        auto& updated = table.at(key);
        for (Action action : actions_from_mask(mask)) {
            updated.regrets[action] += action_values[action] - value;
        }
        return value;
    }
};

struct Target {
    const Solver& solver;
    bool average;
    uint64_t queries = 0;
    uint64_t missing = 0;
    Values policy(const State& state, int viewer) {
        ++queries;
        const auto key = information_key(state, viewer, solver.atlas, solver.recall);
        const auto found = solver.table.find(key);
        if (found == solver.table.end()) { ++missing; }
        const RegretNode empty{};
        const auto& node = found == solver.table.end() ? empty : found->second;
        return average ? average_strategy(node, valid_mask(state, viewer))
            : current_strategy(node, valid_mask(state, viewer));
    }
};

void write_text(std::ostream& out, const std::string& text) {
    epoch7::write(out, uint32_t(text.size()));
    out.write(text.data(), text.size());
    require(bool(out), "text write failed");
}

std::string read_text(std::istream& in, uint32_t limit = 100000) {
    const auto size = epoch7::read<uint32_t>(in);
    require(size <= limit, "oversized checkpoint string");
    std::string text(size, '\0');
    in.read(text.data(), size);
    require(bool(in), "truncated checkpoint string");
    return text;
}

void save_solver(std::ostream& out, const Solver& solver) {
    epoch7::write(out, uint8_t(solver.recall));
    epoch7::write(out, solver.nodes);
    std::ostringstream random_state;
    random_state << solver.rng;
    write_text(out, random_state.str());
    epoch7::write(out, uint64_t(solver.table.size()));
    for (const auto& entry : solver.table) {
        write_text(out, entry.first);
        for (double value : entry.second.regrets) { epoch7::write(out, value); }
        for (double value : entry.second.strategy_sum) { epoch7::write(out, value); }
        epoch7::write(out, entry.second.touches);
    }
}

void load_solver(std::istream& in, Solver& solver) {
    require(epoch7::read<uint8_t>(in) == solver.recall, "checkpoint arm mismatch");
    solver.nodes = epoch7::read<uint64_t>(in);
    std::istringstream random_state(read_text(in));
    random_state >> solver.rng;
    require(bool(random_state), "invalid RNG state");
    const auto count = epoch7::read<uint64_t>(in);
    require(count <= 100000000, "invalid table size");
    solver.table.clear();
    solver.key_bytes = 0;
    for (uint64_t index = 0; index < count; ++index) {
        const auto key = read_text(in, 10000);
        RegretNode node;
        for (auto& value : node.regrets) {
            value = epoch7::read<double>(in);
            require(std::isfinite(value), "invalid regret");
        }
        for (auto& value : node.strategy_sum) {
            value = epoch7::read<double>(in);
            require(std::isfinite(value) && value >= 0, "invalid strategy mass");
        }
        node.touches = epoch7::read<uint64_t>(in);
        require(solver.table.emplace(key, node).second, "duplicate information key");
        solver.key_bytes += key.size();
    }
}

struct Progress {
    uint64_t hands = 0;
    double train_seconds = 0;
    double eval_seconds = 0;
    uint64_t trace_bytes = 0;
    uint64_t panel_bytes = 0;
};

void checkpoint(const fs::path& directory, const std::string& identity,
                Progress progress, const Solver& recall, const Solver& baseline) {
    progress.trace_bytes = fs::file_size(directory / "per_hand.csv");
    progress.panel_bytes = fs::file_size(directory / "panel.csv");
    const auto path = directory / "checkpoint.bin";
    const auto temp = directory / "checkpoint.tmp";
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    write_text(out, "HANDRECALL1");
    write_text(out, identity);
    epoch7::write(out, progress.hands);
    epoch7::write(out, progress.train_seconds);
    epoch7::write(out, progress.eval_seconds);
    epoch7::write(out, progress.trace_bytes);
    epoch7::write(out, progress.panel_bytes);
    save_solver(out, recall);
    save_solver(out, baseline);
    out.close();
    require(bool(out), "checkpoint close failed");
    require(MoveFileExW(temp.c_str(), path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "checkpoint replace failed");
}

Progress restore(const fs::path& directory, const std::string& identity,
                 Solver& recall, Solver& baseline) {
    std::ifstream in(directory / "checkpoint.bin", std::ios::binary);
    require(read_text(in) == "HANDRECALL1", "invalid checkpoint format");
    require(read_text(in) == identity, "checkpoint manifest mismatch");
    Progress progress;
    progress.hands = epoch7::read<uint64_t>(in);
    progress.train_seconds = epoch7::read<double>(in);
    progress.eval_seconds = epoch7::read<double>(in);
    progress.trace_bytes = epoch7::read<uint64_t>(in);
    progress.panel_bytes = epoch7::read<uint64_t>(in);
    load_solver(in, recall);
    load_solver(in, baseline);
    require(in.peek() == std::char_traits<char>::eof(), "trailing checkpoint bytes");
    for (auto entry : {std::make_pair("per_hand.csv", progress.trace_bytes),
                       std::make_pair("panel.csv", progress.panel_bytes)}) {
        require(fs::file_size(directory / entry.first) >= entry.second,
            "trace shorter than checkpoint; cannot safely resume");
        // Discard only the uncommitted suffix of this run, not older results.
        fs::resize_file(directory / entry.first, entry.second);
    }
    return progress;
}

void header(std::ostream& out) {
    out << "hands,recall_current,baseline_current,recall_average,baseline_average"
        ",root_seed,recall_current_queries,recall_current_missing"
        ",baseline_current_queries,baseline_current_missing"
        ",recall_average_queries,recall_average_missing"
        ",baseline_average_queries,baseline_average_missing"
        ",recall_states,baseline_states,recall_nodes,baseline_nodes"
        ",recall_payload_bytes,baseline_payload_bytes,peak_rss_bytes\n";
}

void evaluate(std::ostream& out, uint64_t hand, uint64_t seed, int particle_count,
              const Solver& recall, const Solver& baseline) {
    const State root = epoch7::root_for(seed);
    std::mt19937_64 rng(seed ^ 0x123456789ull);
    const auto particles = epoch7::root_particles(root, particle_count, rng);
    std::array<Target, 4> policies{{{recall, false}, {baseline, false},
                                  {recall, true}, {baseline, true}}};
    out << hand;
    for (auto& policy : policies) {
        out << ',' << epoch7::local_gap(root, particles, policy, seed ^ 0xabcdefull);
    }
    out << ',' << seed;
    for (const auto& policy : policies) {
        out << ',' << policy.queries << ',' << policy.missing;
    }
    out << ',' << recall.table.size() << ',' << baseline.table.size()
        << ',' << recall.nodes << ',' << baseline.nodes
        << ',' << recall.key_bytes + recall.table.size() * sizeof(RegretNode)
        << ',' << baseline.key_bytes + baseline.table.size() * sizeof(RegretNode)
        << ',' << epoch7::peak_rss() << '\n';
    require(bool(out), "metric write failed");
}

// No training or rollout in these deterministic schema/observation checks.
void self_test() {
    struct FixtureAtlas : PowerAtlas {
        uint16_t assign(const State& state, int viewer) const override {
            return state.street == 5 ? card_token(state.players[viewer].shown[2]) : 1;
        }
    } atlas;
    State root = epoch7::root_for(5);
    State sixth = epoch7::child(epoch7::child(root, CHECK), CHECK);
    require(sixth.street == 6, "fixture did not reach sixth street");
    for (int viewer = 0; viewer < 2; ++viewer) {
        const auto prefix = card_prefix(sixth, 5);
        require(power_observation_key(root, viewer) == power_observation_key(prefix, viewer),
            "future card leaked into past observation");
    }
    State seventh = epoch7::child(epoch7::child(sixth, CHECK), CHECK);
    require(seventh.street == 7, "fixture did not reach seventh street");
    for (int street : {5, 6}) {
        const auto prefix = card_prefix(seventh, street);
        const auto& original = street == 5 ? root : sixth;
        for (int viewer = 0; viewer < 2; ++viewer) {
            require(power_observation_key(original, viewer) == power_observation_key(prefix, viewer),
                "private seventh card leaked into past observation");
        }
    }
    // Public card order affects the remembered fifth bucket but not the
    // sixth bucket or public strength. It MUST separate the recall arm only.
    State reordered = sixth;
    const int viewer = sixth.actor;
    std::swap(reordered.players[viewer].shown[0], reordered.players[viewer].shown[2]);
    require(information_key(sixth, viewer, atlas, false) ==
        information_key(reordered, viewer, atlas, false), "baseline unexpectedly recalls buckets");
    require(information_key(sixth, viewer, atlas, true) !=
        information_key(reordered, viewer, atlas, true), "past hand bucket was forgotten");
    State masked = seventh;
    masked.players[1 - seventh.actor].hidden.clear();
    masked.players[1 - seventh.actor].has_discard = false;
    masked.simulation_deck.clear();
    require(information_key(seventh, seventh.actor, atlas, true) ==
        information_key(masked, seventh.actor, atlas, true), "hidden state leaked into key");
    State changed = seventh;
    changed.history[0].action = BBING;
    require(information_key(seventh, seventh.actor, atlas, true) !=
        information_key(changed, seventh.actor, atlas, true), "action history was forgotten");
    Solver original(atlas, true, 71), restored(atlas, true, 9);
    const auto key = information_key(root, root.actor, atlas, true);
    auto& node = original.table[key];
    node.regrets[CHECK] = -3;
    node.regrets[BBING] = 4;
    node.strategy_sum[CHECK] = 3;
    node.strategy_sum[BBING] = 7;
    original.key_bytes = key.size();
    original.rng();
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    save_solver(stream, original);
    load_solver(stream, restored);
    require(original.rng() == restored.rng(), "RNG round trip failed");
    require(original.table.at(key).regrets == restored.table.at(key).regrets,
        "signed regret round trip failed");
    Target current{restored, false}, average{restored, true};
    require(current.policy(root, root.actor)[BBING] == 1, "regret matching failed");
    require(std::abs(average.policy(root, root.actor)[CHECK] - 0.3) < 1e-12,
        "average policy round trip failed");
    std::cout << "SELF_TEST passed: prefixes, memory, hidden invariance, signed regret, RNG, average\n";
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        self_test();
        return 0;
    }
    if (argc != 13) {
        std::cout << "Use python -m agents.cpp_mccfr.hand_recall --help\n"
            << "Internal: atlas directory hands seed eval_seed particles panel_every"
            << " panel_roots checkpoint_every max_states resume manifest_identity\n";
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    const fs::path directory(argv[2]);
    const uint64_t hands = std::stoull(argv[3]);
    const uint64_t seed = std::stoull(argv[4]), eval_seed = std::stoull(argv[5]);
    const int particles = std::stoi(argv[6]);
    const uint64_t panel_every = std::stoull(argv[7]);
    const int panel_roots = std::stoi(argv[8]);
    const uint64_t checkpoint_every = std::stoull(argv[9]);
    const uint64_t max_states = std::stoull(argv[10]);
    const bool resume = std::stoi(argv[11]) != 0;
    const std::string identity(argv[12]);
    require(hands > 0 && particles >= 2 && panel_every > 0 && panel_roots > 0 &&
        checkpoint_every > 0 && max_states > 0, "invalid run parameters");
    PowerAtlas atlas;
    atlas.load(argv[1]);
    atlas.set_assignment_cache_limit(100000);
    for (int street : {5, 6, 7}) {
        require(atlas.clusters(street) == 256, "requires frozen hard k-means256 atlas");
    }
    Solver recall(atlas, true, seed), baseline(atlas, false, seed);
    Progress progress;
    if (resume) { progress = restore(directory, identity, recall, baseline); }
    require(progress.hands <= hands, "target smaller than checkpoint");
    std::ofstream trace(directory / "per_hand.csv", resume ? std::ios::app : std::ios::trunc);
    std::ofstream panel(directory / "panel.csv", resume ? std::ios::app : std::ios::trunc);
    trace << std::setprecision(17);
    panel << std::setprecision(17);
    const auto timed_panel = [&]() {
        const auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < panel_roots; ++index) {
            const uint64_t root_seed = eval_seed ^ ((index + 1) * 0xd1b54a32d192ed03ull);
            evaluate(panel, progress.hands, root_seed, particles, recall, baseline);
        }
        progress.eval_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    };
    if (!resume) {
        header(trace);
        header(panel);
        timed_panel();
        trace.flush();
        panel.flush();
        checkpoint(directory, identity, progress, recall, baseline);
    }
    for (uint64_t hand = progress.hands + 1; hand <= hands; ++hand) {
        auto start = std::chrono::steady_clock::now();
        const State root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        for (auto* solver : {&recall, &baseline}) {
            solver->traverse(root, 0);
            solver->traverse(root, 1);
        }
        progress.hands = hand;
        progress.train_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        evaluate(trace, hand, eval_seed ^ (hand * 0x94d049bb133111ebull),
            particles, recall, baseline);
        progress.eval_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (hand % panel_every == 0) { timed_panel(); }
        const bool capped = recall.table.size() + baseline.table.size() >= max_states;
        if (hand % 1000 == 0 || capped || hand == hands) {
            trace.flush();
            panel.flush();
            std::cout << "PROGRESS hands=" << hand
                << " recall_states=" << recall.table.size()
                << " baseline_states=" << baseline.table.size()
                << " recall_nodes=" << recall.nodes << " baseline_nodes=" << baseline.nodes
                << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0
                << " train_seconds=" << progress.train_seconds
                << " eval_seconds=" << progress.eval_seconds << std::endl;
        }
        if (hand % checkpoint_every == 0 || hand == hands || capped) {
            trace.flush();
            panel.flush();
            checkpoint(directory, identity, progress, recall, baseline);
        }
        if (capped) {
            std::cout << "STOPPED max_states=" << max_states
                << "; checkpoint saved. Raise --max-states and --resume to continue.\n";
            return 2;
        }
    }
    return 0;
}
}  // namespace hand_recall

#ifndef HAND_RECALL_NO_MAIN
int main(int argc, char** argv) {
    try {
        return hand_recall::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
#endif
