// Collect pre-action observations with the existing C++ 7-stud v3 rules.
// Build: g++ -O2 -std=c++17 collect.cpp -o bin/collect.exe
// Usage: collect.exe OUTPUT.bin HANDS SEED COMPLETIONS; or --self-test.
// Fold probabilities are PER LEGAL DECISION, not per hand or street.
#define STUD_MCCFR_NO_MAIN
#include "../cpp_mccfr/stud_mccfr.cpp"
#include <filesystem>

namespace neural_data {
constexpr std::array<double, 3> kFold = {0.02625, 0.06125, 0.0875};
using Policy = std::array<double, kActionCount>;

Policy behavior(int street, uint8_t mask) {
    if (street < 5 || street > 7 || !mask) {
        throw std::runtime_error("invalid behavior-policy query");
    }
    const auto legal = actions_from_mask(mask);
    Policy policy{};
    const bool fold = mask & (1u << FOLD);
    if (legal.size() == 1) {
        policy[legal.front()] = 1;
    } else {
        const double probability = fold ? kFold[street - 5] : 0;
        for (Action action : legal) {
            policy[action] = action == FOLD ? probability
                : (1 - probability) / (legal.size() - int(fold));
        }
    }
    return policy;
}

Action draw_action(const Policy& policy, std::mt19937_64& rng) {
    return static_cast<Action>(std::discrete_distribution<int>(
        policy.begin(), policy.end())(rng));
}

void advance(State& state) {
    if (state.street == 7) {
        state.terminal = true;
        return;
    }
    ++state.street;
    for (auto& player : state.players) {
        if (state.simulation_deck.empty()) throw std::runtime_error("empty deck");
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (state.street == 7) player.hidden.push_back(card);
        else player.shown.push_back(card);
    }
    reset_round(state, state.street);
    if (state.players[0].all_in || state.players[1].all_in) advance(state);
}

struct Row {
    uint32_t hand;
    uint8_t actor, street, mask, action;
    std::array<float, 18> power;
    std::array<uint8_t, 12> cards;
    std::array<uint8_t, 24> history;
    std::array<float, 7> scalars;
    float outcome;
};

Row observe(const State& state, uint32_t hand, int completions) {
    Row row{};
    row.hand = hand;
    row.actor = state.actor;
    row.street = state.street;
    row.mask = valid_mask(state, state.actor);
    const auto power = power_vector(state, state.actor, completions);
    std::copy(power.begin(), power.end(), row.power.begin());
    // Reuse the existing ordered card/history encoding; never truncate silently.
    const auto tensor = deep_cfr_tensor(state, state.actor);
    size_t offset = 5;
    for (auto& token : row.cards) {
        token = std::max_element(tensor.begin() + offset,
            tensor.begin() + offset + 53) - (tensor.begin() + offset);
        offset += 53;
    }
    for (auto& token : row.history) {
        token = std::max_element(tensor.begin() + offset,
            tensor.begin() + offset + 49) - (tensor.begin() + offset);
        offset += 49;
    }
    std::copy_n(tensor.begin() + offset, 7, row.scalars.begin());
    return row;
}

template<class T> void write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("write failed");
}

void write_row(std::ostream& out, const Row& row) {
    write(out, row.hand);
    for (uint8_t value : {row.actor, row.street, row.mask, row.action}) write(out, value);
    write(out, row.power);
    write(out, row.cards);
    write(out, row.history);
    write(out, row.scalars);
    write(out, row.outcome);
}

void self_test() {
    for (int street = 5; street <= 7; ++street) {
        for (int mask = 1; mask < 256; ++mask) {
            const auto policy = behavior(street, mask);
            assert(std::abs(std::accumulate(policy.begin(), policy.end(), 0.) - 1) < 1e-12);
            double nonfold = -1;
            for (int a = 0; a < 8; ++a) {
                if (!(mask & (1 << a))) assert(policy[a] == 0);
                else if (a != FOLD) {
                    if (nonfold >= 0) assert(std::abs(policy[a] - nonfold) < 1e-12);
                    nonfold = policy[a];
                }
            }
            if ((mask & (1 << FOLD)) && nonfold >= 0) {
                assert(std::abs(policy[FOLD] - kFold[street - 5]) < 1e-12);
            }
        }
    }
    std::mt19937_64 rng(13);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    State state = sample_fifth_street_root(deck, 1000);
    const Row before = observe(state, 0, 128);
    State changed = state;
    auto& opponent = changed.players[1 - state.actor];
    std::swap(opponent.hidden[0], changed.simulation_deck[0]);
    std::swap(opponent.discarded, changed.simulation_deck[1]);
    const Row after = observe(changed, 0, 128);
    assert(before.power == after.power && before.cards == after.cards);
    assert(before.history == after.history && before.scalars == after.scalars);
    changed = state;
    changed.history.push_back({5, uint8_t(state.actor), CHECK});
    assert(observe(changed, 0, 128).history != before.history);
    changed.history.resize(25);
    bool rejected = false;
    try { deep_cfr_tensor(changed, changed.actor); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::cout << "PASS policy_masks=765 hidden_invariance=1 history_capacity=1\n";
}
}

int main(int argc, char** argv) {
    try {
        using namespace neural_data;
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            neural_data::self_test();
            return 0;
        }
        if (argc != 5) throw std::runtime_error("usage: collect OUTPUT HANDS SEED COMPLETIONS");
        const int hands = std::stoi(argv[2]);
        const uint64_t seed = std::stoull(argv[3]);
        const int completions = std::stoi(argv[4]);
        if (hands < 30 || completions < 1) throw std::runtime_error("invalid budget");
        if (std::filesystem::exists(argv[1])) throw std::runtime_error("output exists");
        std::ofstream out(argv[1], std::ios::binary);
        out.write("NAE7D01\0", 8);
        std::mt19937_64 rng(seed);
        std::array<uint64_t, 3> decisions{}, folds{};
        for (int hand = 0; hand < hands; ++hand) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), rng);
            State state = sample_fifth_street_root(deck, 1000);
            std::vector<Row> rows;
            while (!state.terminal) {
                const int actor = state.actor;
                Row row = observe(state, hand, completions);
                const Action action = draw_action(behavior(state.street, row.mask), rng);
                row.action = action;
                rows.push_back(row);
                ++decisions[state.street - 5];
                folds[state.street - 5] += action == FOLD;
                const ActionResult result = apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd) advance(state);
                else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            }
            assert(std::abs(terminal_net_search(state, 0) + terminal_net_search(state, 1)) < 1e-8);
            for (auto& row : rows) {
                row.outcome = float(terminal_net_search(state, row.actor));
                write_row(out, row);
            }
            if ((hand + 1) % 1000 == 0) {
                std::cout << "PROGRESS hands=" << hand + 1 << " rows="
                    << std::accumulate(decisions.begin(), decisions.end(), uint64_t(0))
                    << std::endl;
            }
        }
        for (int street = 0; street < 3; ++street) {
            std::cout << "POLICY street=" << street + 5 << " decisions=" << decisions[street]
                << " folds=" << folds[street] << " target=" << kFold[street] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
