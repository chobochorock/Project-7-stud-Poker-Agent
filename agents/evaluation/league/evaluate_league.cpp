// Frozen-policy round robin using the existing 7-stud v3 match engine.
// Build/run through run_league.py; no policy training or online resolving.
// Usage: exe ROSTER_TSV NEW_OUTPUT_DIR HANDS_PER_MATCH SEED
// Test: exe --self-test ROSTER_TSV
// TSV fields: id, kind, mode, model/run path, atlas path, hands, epoch_hands.
#define main epoch_training_entrypoint
#include "../../lightgbm_regret_ensemble/stud_epoch_ensemble.cpp"
#undef main
#include <functional>
#include <map>
#include <memory>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct PlayerPolicy {
    std::string id;
    std::function<epoch7::Values(const State&, int, bool*)> query;
    std::mt19937_64 rng{0};
    uint64_t decisions = 0, misses = 0;

    epoch7::Values policy(const State& state, int seat, bool* found = nullptr) const {
        const auto result = query(state, seat, found);
        const auto mask = valid_mask(state, seat);
        double total = 0;
        for (int a = 0; a < kActionCount; ++a) {
            require(std::isfinite(result[a]) && result[a] >= 0 &&
                ((mask & (1u << a)) || result[a] == 0), "illegal policy: " + id);
            total += result[a];
        }
        require(std::abs(total - 1) < 1e-8, "unnormalized policy: " + id);
        return result;
    }

    Action choose(const State& state, int seat, int iterations) {
        require(iterations == 0, "league must not train");
        bool found = false;
        const auto probabilities = policy(state, seat, &found);
        ++decisions; misses += !found;
        return epoch7::sample(probabilities, valid_mask(state, seat), rng);
    }
};

struct Roster {
    std::map<std::string, std::unique_ptr<PowerAtlas>> atlases;
    std::map<std::string, std::unique_ptr<MCCFR>> tables;
    std::vector<std::unique_ptr<epoch7::FeatureCache>> features;
    std::vector<std::unique_ptr<epoch7::Ensemble>> ensembles;
    std::vector<PlayerPolicy> players;

    PowerAtlas* atlas(const std::string& path) {
        if (path == "-") return nullptr;
        auto& value = atlases[path];
        if (!value) {
            value = std::make_unique<PowerAtlas>();
            value->load(path);
            value->set_assignment_cache_limit(100000);
        }
        return value.get();
    }

    MCCFR* table(const std::string& path, const std::string& atlas_path) {
        auto& value = tables[path + "\t" + atlas_path];
        if (value) return value.get();
        std::ifstream input(path, std::ios::binary);
        char magic[8]{};
        uint8_t plus = 0, abstraction = 0, street = 0;
        input.read(magic, 8);
        input.read(reinterpret_cast<char*>(&plus), 1);
        input.read(reinterpret_cast<char*>(&abstraction), 1);
        input.read(reinterpret_cast<char*>(&street), 1);
        require(input.good() && street == 5 && plus <= 1 &&
            (std::memcmp(magic, "MCCFRV4", 7) == 0 ||
             std::memcmp(magic, "MCCFRV5", 7) == 0 ||
             std::memcmp(magic, "MCCFRV6", 7) == 0), "unsupported model: " + path);
        require(abstraction == 0 || abstraction == 1 || abstraction == 5 ||
            abstraction == 6 || abstraction == 7 || abstraction == 8,
            "unsupported abstraction (no silent flag guessing): " + path);
        auto* power = atlas(atlas_path);
        require((abstraction == 0) == (power == nullptr), "atlas/model mismatch: " + path);
        value = std::make_unique<MCCFR>(plus != 0, 7, 5, power);
        if (abstraction == 5) value->use_previous_street_summary(true);
        if (abstraction == 6) value->use_cumulative_street_summary(true);
        if (abstraction == 7) value->use_cumulative_street_strength_summary(true);
        value->load(path);
        std::cout << "MODEL path=" << path << " abstraction=" << int(abstraction)
            << " buckets=" << value->bucket_count() << '\n' << std::flush;
        return value.get();
    }

    explicit Roster(const std::string& path) {
        std::ifstream input(path);
        require(input.good(), "cannot open roster");
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::istringstream stream(line);
            std::vector<std::string> fields;
            std::string field;
            while (std::getline(stream, field, '\t')) fields.push_back(field);
            require(fields.size() == 7, "expected seven roster fields");
            const auto& id = fields[0];
            require(!id.empty() && id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") ==
                std::string::npos, "invalid player id");
            for (const auto& player : players) require(player.id != id, "duplicate player id");
            PlayerPolicy player;
            player.id = id;
            if (fields[1] == "ensemble") {
                require(fields[2] == "current", "ensemble average is not exported");
                const auto hands = std::stoull(fields[5]), epoch_hands = std::stoull(fields[6]);
                auto* base = table((std::filesystem::path(fields[3]) /
                    ("hard256_" + std::to_string(hands) + ".bin")).string(), fields[4]);
                features.push_back(std::make_unique<epoch7::FeatureCache>(
                    epoch7::FeatureCache{*atlas(fields[4])}));
                ensembles.push_back(std::make_unique<epoch7::Ensemble>(*features.back(), 7));
                auto* ensemble = ensembles.back().get();
                epoch7::load_frozen_checkpoint(fields[3], *ensemble, *base, hands, epoch_hands);
                player.query = [ensemble](const State& s, int seat, bool* found) {
                    return ensemble->policy(s, seat, found);
                };
            } else if (fields[1] == "mccfr") {
                auto* model = table(fields[3], fields[4]);
                const bool current = fields[2] == "current";
                require(current || fields[2] == "average", "invalid policy mode");
                player.query = [model, current](const State& s, int seat, bool* found) {
                    return current ? model->instantaneous_policy(s, seat, found)
                                   : model->policy(s, seat, found);
                };
            } else if (fields[1] == "heuristic") {
                player.query = [](const State& s, int seat, bool* found) {
                    return HeuristicPolicy{}.policy(s, seat, found);
                };
            } else if (fields[1] == "uniform") {
                player.query = [](const State& s, int seat, bool* found) {
                    if (found) *found = true;
                    return uniform_strategy(valid_mask(s, seat));
                };
            } else throw std::runtime_error("unknown player kind");
            players.push_back(std::move(player));
        }
        require(players.size() >= 2, "at least two players required");
    }
};

void self_test(Roster& roster) {
    for (auto& player : roster.players) {
        auto state = epoch7::root_for(381);
        while (!state.terminal) {
            const int seat = state.actor;
            auto masked = state;
            masked.players[1 - seat].hidden.clear();
            masked.players[1 - seat].has_discard = false;
            masked.simulation_deck.clear();
            require(player.policy(state, seat) == player.policy(masked, seat),
                "unobserved-card leak: " + player.id);
            auto copy = player;
            player.rng.seed(17); copy.rng.seed(17);
            require(player.choose(state, seat, 0) == copy.choose(masked, seat, 0),
                "sampling not reproducible");
            const auto mask = valid_mask(state, seat);
            state = epoch7::child(state, mask & (1u << CALL) ? CALL : CHECK);
        }
        player.decisions = player.misses = 0;
    }
    ConditionalParticipationPolicy folds("fold", 1), calls("made-call", 0);
    for (int seat = 0; seat < 2; ++seat) {
        require(play_hand_match(fresh_deck(), seat, folds, calls, 1000, 5, 1000) == -1,
            "always-fold seat/payoff sanity check failed");
    }
    auto a = roster.players.front(), b = roster.players.back();
    a.rng.seed(3); b.rng.seed(4);
    const auto first = play_hand_match(fresh_deck(), 0, a, b, 1000, 5, 1000);
    a.rng.seed(3); b.rng.seed(4);
    const auto swapped = play_hand_match(fresh_deck(), 1, b, a, 1000, 5, 1000);
    require(std::abs(first + swapped) < 1e-9, "viewpoint/zero-sum check failed");
    std::cout << "SELF_TEST players=" << roster.players.size()
        << " legal_policy=ok privacy=ok reproducibility=ok zero_sum=ok ante_units=ok\n" << std::flush;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool testing = argc == 3 && std::string(argv[1]) == "--self-test";
        require(testing || argc == 5, "usage: exe ROSTER NEW_OUT HANDS_PER_MATCH SEED; or --self-test ROSTER");
        const std::filesystem::path output(testing ? "" : argv[2]);
        const int hands = testing ? 2 : std::stoi(argv[3]);
        const uint64_t seed = testing ? 17 : std::stoull(argv[4]);
        require((hands >= 4 && hands % 2 == 0) || testing, "even HANDS_PER_MATCH >= 4 required");
        if (!testing) require(!std::filesystem::exists(output), "output already exists");
        Roster roster(argv[testing ? 2 : 1]);
        self_test(roster);
        if (testing) return 0;
        std::map<const MCCFR*, size_t> original_sizes;
        for (const auto& [key, value] : roster.tables) original_sizes[value.get()] = value->bucket_count();
        std::filesystem::create_directories(output);
        std::ofstream csv(output / "paired_payoffs.csv"), matches(output / "matches.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        matches.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "match,pair,deal_seed,a,b,a_seat0_ante,a_seat1_ante\n" << std::setprecision(12);
        matches << "match,a,b,hands,mean_ante_a,seconds,a_decisions,a_misses,b_decisions,b_misses\n"
            << std::setprecision(12);
        const int pairs = hands / 2;
        int match = 0;
        auto& players = roster.players;
        for (size_t i = 0; i < players.size(); ++i) for (size_t j = i + 1; j < players.size(); ++j) {
            auto& a = players[i]; auto& b = players[j];
            a.decisions = a.misses = b.decisions = b.misses = 0;
            const auto start = std::chrono::steady_clock::now();
            double sum = 0;
            for (int pair = 0; pair < pairs; ++pair) {
                const uint64_t deal_seed = seed ^ (uint64_t(pair + 1) * 0xd1b54a32d192ed03ull);
                std::mt19937_64 deal_rng(deal_seed);
                auto deck = fresh_deck(); std::shuffle(deck.begin(), deck.end(), deal_rng);
                csv << match << ',' << pair << ',' << deal_seed << ',' << a.id << ',' << b.id;
                for (int seat = 0; seat < 2; ++seat) {
                    a.rng.seed(deal_seed ^ (uint64_t(i + 1) * 0xa0761d6478bd642full) ^
                        (uint64_t(seat + 1) * 0x94d049bb133111ebull));
                    b.rng.seed(deal_seed ^ (uint64_t(j + 1) * 0xa0761d6478bd642full) ^
                        (uint64_t(2 - seat) * 0x94d049bb133111ebull));
                    const double value = play_hand_match(deck, seat, a, b, 1000, 5, 1000);
                    require(std::isfinite(value) && std::abs(value) <= 1000, "invalid terminal payoff");
                    sum += value / 2;
                    csv << ',' << value;
                }
                csv << '\n';
                if ((pair + 1) % 500 == 0 || pair + 1 == pairs) {
                    csv.flush();
                    std::cout << "MATCH index=" << match + 1 << " a=" << a.id << " b=" << b.id
                        << " hands=" << 2 * (pair + 1) << '/' << hands << " mean_ante_a="
                        << sum / (pair + 1) << " seconds=" << std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - start).count() << '\n' << std::flush;
                }
            }
            matches << match << ',' << a.id << ',' << b.id << ',' << hands << ',' << sum / pairs << ','
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << ','
                << a.decisions << ',' << a.misses << ',' << b.decisions << ',' << b.misses << '\n';
            matches.flush(); ++match;
        }
        for (const auto& [model, size] : original_sizes) require(model->bucket_count() == size, "policy mutated");
        std::cout << "DONE players=" << players.size() << " matches=" << match
            << " total_hands=" << uint64_t(match) * hands << " peak_cpp_rss_bytes=" << epoch7::peak_rss()
            << '\n' << std::flush;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
