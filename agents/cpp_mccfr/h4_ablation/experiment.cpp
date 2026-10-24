// H4-only self-play and posterior-aware LBR. Run through run.py; see README.md.
#define VALIDATED_NO_MAIN
#include "../dynamic_tree_validated/solver.cpp"
#include "../abstraction_lbr/evaluate.hpp"
#include "h4_generated.hpp"

namespace h4_ablation {
using Values = std::array<double, 12>;
using Betting = abstraction_lbr::Target<FrozenModel>;
constexpr uint64_t kMix = 0xd1b54a32d192ed03ull;

struct Canonical {
    uint32_t key;
    std::array<Card, 4> ordered;
};

const Canonical& canonical(const std::vector<Card>& cards) {
    static std::unordered_map<uint64_t, Canonical> cache;
    const auto mask = card_mask(cards);
    auto found = cache.find(mask);
    if (found != cache.end()) { return found->second; }
    const auto view = make_h4_view(cards);
    Canonical value{view.key, {}};
    for (int i = 0; i < 4; ++i) { value.ordered[i] = cards[view.original_index[i]]; }
    return cache.emplace(mask, value).first->second;
}

std::pair<int, int> choice(const std::vector<Card>& cards, int action) {
    const auto& view = canonical(cards);
    const auto [discard, reveal] = h4_positions(action);
    return {int(std::find(cards.begin(), cards.end(), view.ordered[discard]) - cards.begin()),
            int(std::find(cards.begin(), cards.end(), view.ordered[reveal]) - cards.begin())};
}

struct Learner {
    std::map<uint64_t, H4Node> nodes;
    uint64_t key(const std::vector<Card>& cards, int seat) const {
        return uint64_t(canonical(cards).key) * 2 + seat;
    }
    Values policy(const std::vector<Card>& cards, int seat, bool average) const {
        const auto it = nodes.find(key(cards, seat));
        Values result;
        result.fill(1.0 / 12);
        if (it == nodes.end()) { return result; }
        if (!average) { return h4_regret_strategy(it->second); }
        result = it->second.strategy_sum;
        const double sum = std::accumulate(result.begin(), result.end(), 0.0);
        hand_recall::require(sum > 0, "empty average node");
        for (double& p : result) { p /= sum; }
        return result;
    }
    int sample(const std::vector<Card>& cards, int seat, bool average, std::mt19937_64& rng) const {
        const auto p = policy(cards, seat, average);
        return std::discrete_distribution<int>(p.begin(), p.end())(rng);
    }
    void update(const std::vector<Card>& cards, int seat, const Values& q) {
        const auto p = policy(cards, seat, false);
        const double value = std::inner_product(p.begin(), p.end(), q.begin(), 0.0);
        auto& node = nodes[key(cards, seat)];
        for (int a = 0; a < 12; ++a) {
            node.regrets[a] += q[a] - value;  // Signed, not CFR+ clipping.
            node.strategy_sum[a] += p[a];
        }
        ++node.touches;
    }
    double weight(const std::array<Card, 2>& hidden, Card shown, Card discarded, int seat) const {
        const std::vector<Card> cards{hidden[0], hidden[1], shown, discarded};
        const auto p = policy(cards, seat, true);
        const auto& ordered = canonical(cards).ordered;
        for (int a = 0; a < 12; ++a) {
            const auto [d, r] = h4_positions(a);
            if (ordered[d] == discarded && ordered[r] == shown) { return 24 * p[a]; }
        }
        throw std::runtime_error("H4 likelihood mapping failed");
    }
};

struct Target : Betting {
    const Learner* h4;
    Target(FrozenModel& model, int arm, const Learner* learner)
        : Betting{model, arm + 2}, h4(learner) {}
    bool learned_h4() const { return h4 != nullptr; }
    double h4_weight(const std::array<Card, 2>& hidden, Card shown, Card discarded, int seat) const {
        return h4->weight(hidden, shown, discarded, seat);
    }
};

struct InitialPolicy : H4Policy {
    const Learner& learner;
    int seat;
    std::mt19937_64 rng;
    InitialPolicy(const Learner& value, int player, uint64_t seed)
        : learner(value), seat(player), rng(seed) {}
    std::pair<int, int> choose(const std::vector<Card>& cards) override {
        return choice(cards, learner.sample(cards, seat, true, rng));
    }
    size_t buckets() const override { return learner.nodes.size(); }
};

std::array<std::vector<Card>, 2> initial_cards(const std::vector<Card>& deck) {
    std::array<std::vector<Card>, 2> cards;
    for (int i = 0; i < 8; ++i) { cards[i % 2].push_back(deck[i]); }
    return cards;
}

Values action_values(const Learner& h4, Betting& betting, const std::vector<Card>& deck,
                     int seat, bool average, uint64_t seed) {
    const auto cards = initial_cards(deck);
    std::mt19937_64 rng(seed);
    const auto other = choice(cards[1-seat], h4.sample(cards[1-seat], 1-seat, average, rng));
    Values q{};
    for (int a = 0; a < 12; ++a) {
        std::array<std::pair<int, int>, 2> choices;
        choices[seat] = choice(cards[seat], a);
        choices[1-seat] = other;
        betting.rng.seed(seed ^ 0x94d049bb133111ebull);
        q[a] = h4_frozen_value(deck, choices, seat, betting, 1000, 1000);
        hand_recall::require(std::isfinite(q[a]) && std::abs(q[a]) <= 1000, "bad H4 value");
    }
    return q;
}

void save(const std::filesystem::path& path, const std::array<Learner, 2>& learners,
          uint64_t hands, uint64_t seed) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    hand_recall::write_text(out, "H4_FROZEN_SIGNED1");
    epoch7::write(out, hands);
    epoch7::write(out, seed);
    for (const auto& learner : learners) {
        epoch7::write(out, uint64_t(learner.nodes.size()));
        for (const auto& [key, node] : learner.nodes) {
            epoch7::write(out, key);
            for (double r : node.regrets) { epoch7::write(out, r); }
            for (double p : node.strategy_sum) { epoch7::write(out, p); }
            epoch7::write(out, node.touches);
        }
    }
    out.close();
    hand_recall::require(MoveFileExA(temporary.c_str(), path.string().c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH), "checkpoint replace failed");
}

uint64_t load(const std::filesystem::path& path, std::array<Learner, 2>& learners, uint64_t& seed) {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    hand_recall::require(hand_recall::read_text(in) == "H4_FROZEN_SIGNED1", "H4 schema mismatch");
    const auto hands = epoch7::read<uint64_t>(in);
    seed = epoch7::read<uint64_t>(in);
    for (auto& learner : learners) {
        learner.nodes.clear();
        const auto count = epoch7::read<uint64_t>(in);
        hand_recall::require(count <= 600000, "invalid H4 node count");
        for (uint64_t i = 0; i < count; ++i) {
            const auto key = epoch7::read<uint64_t>(in);
            H4Node node;
            for (double& r : node.regrets) { r = epoch7::read<double>(in); }
            for (double& p : node.strategy_sum) { p = epoch7::read<double>(in); }
            node.touches = epoch7::read<uint64_t>(in);
            learner.nodes.emplace(key, node);
        }
    }
    in.exceptions(std::ios::badbit);
    hand_recall::require(in.peek() == std::char_traits<char>::eof(), "H4 trailing data");
    return hands;
}

void train_hand(std::array<Learner, 2>& learners, FrozenModel& model, uint64_t hand, uint64_t seed) {
    const uint64_t deal_seed = seed ^ (hand * kMix);
    std::mt19937_64 rng(deal_seed);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    const auto cards = initial_cards(deck);
    for (int arm = 0; arm < 2; ++arm) {
        Betting betting{model, arm + 2};
        std::array<Values,2> values;
        for (int seat = 0; seat < 2; ++seat) {
            values[seat] = action_values(learners[arm], betting, deck, seat, false,
                                        deal_seed ^ (uint64_t(seat+1) * 0xabcdef012345ull));
        }
        // Freeze both root policies until both sampled player updates are ready.
        for (int seat = 0; seat < 2; ++seat) {
            learners[arm].update(cards[seat], seat, values[seat]);
        }
    }
}

void panel(const std::filesystem::path& output, const std::array<Learner, 2>& learners,
           FrozenModel& model, uint64_t hands, int roots, int particles) {
    hand_recall::require(particles >= 2 && particles % 2 == 0, "panel needs even particles >=2");
    const auto path = output / "panel.csv";
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream csv(path, std::ios::app);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    if (fresh) { csv << "hands,arm,roots,particles,gap,crossfit_gap,mean_action_se,km_ratio\n"; }
    std::array<double, 2> gaps{}, crossfit{}, uncertainty{};
    for (int arm = 0; arm < 2; ++arm) {
        Betting betting{model, arm + 2};
        for (int root = 0; root < roots; ++root) {
            const int seat = root % 2;
            std::mt19937_64 root_rng(2026100861ull ^ (uint64_t(root+1) * kMix));
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), root_rng);
            const auto own = initial_cards(deck)[seat];
            std::vector<Card> unseen;
            for (auto c : fresh_deck()) {
                if (std::find(own.begin(), own.end(), c) == own.end()) { unseen.push_back(c); }
            }
            Values sum{}, sum2{};
            std::array<Values, 2> halves{};
            for (int p = 0; p < particles; ++p) {
                auto sampled = unseen;
                const uint64_t sample_seed = root_rng();
                std::mt19937_64 rng(sample_seed);
                std::shuffle(sampled.begin(), sampled.end(), rng);
                size_t cursor = 0;
                for (int i = 0; i < 52; ++i) {
                    deck[i] = i < 8 && i % 2 == seat ? own[i/2] : sampled[cursor++];
                }
                const auto q = action_values(learners[arm], betting, deck, seat, true, sample_seed);
                for (int a = 0; a < 12; ++a) {
                    sum[a] += q[a];
                    sum2[a] += q[a] * q[a];
                    halves[p % 2][a] += q[a] / (particles/2);
                }
            }
            const auto policy = learners[arm].policy(own, seat, true);
            Values q{};
            for (int a = 0; a < 12; ++a) {
                q[a] = sum[a] / particles;
                uncertainty[arm] += std::sqrt(std::max(0.0,
                    (sum2[a] - sum[a]*sum[a]/particles) / (particles-1) / particles)) / (12*roots);
            }
            gaps[arm] += (*std::max_element(q.begin(), q.end()) -
                std::inner_product(policy.begin(), policy.end(), q.begin(), 0.0)) / roots;
            for (int half = 0; half < 2; ++half) {
                const int best = int(std::max_element(halves[half].begin(), halves[half].end()) - halves[half].begin());
                crossfit[arm] += (halves[1-half][best] - std::inner_product(
                    policy.begin(), policy.end(), halves[1-half].begin(), 0.0)) / (2*roots);
            }
        }
    }
    for (int arm = 0; arm < 2; ++arm) {
        csv << std::setprecision(17) << hands << ',' << arm << ',' << roots << ',' << particles
            << ',' << gaps[arm] << ',' << crossfit[arm] << ',' << uncertainty[arm]
            << ',' << (gaps[1] > 0 ? gaps[arm]/gaps[1] : 0) << '\n';
    }
    std::cout << "PANEL hands=" << hands << " h4_average_dt_gap=" << gaps[0]
              << " h4_average_km_gap=" << gaps[1] << " dt_crossfit=" << crossfit[0]
              << " km_crossfit=" << crossfit[1] << std::endl;
}

void self_test(FrozenModel& model, const std::filesystem::path& output) {
    abstraction_lbr::self_test(model);
    Learner uniform;
    const std::vector<Card> cards{{2,0}, {7,1}, {11,2}, {14,3}};
    std::set<std::pair<int,int>> actions;
    for (int a = 0; a < 12; ++a) { actions.insert(choice(cards, a)); }
    hand_recall::require(actions.size() == 12, "H4 actions not bijective");
    hand_recall::require(std::abs(uniform.weight({cards[0], cards[1]}, cards[2], cards[3], 0)-2) < 1e-12,
                         "uniform H4 posterior is not uniform");
    Values q{};
    q[0] = 12;
    uniform.update(cards, 0, q);
    hand_recall::require(uniform.nodes.begin()->second.regrets[1] == -1, "negative regret clipped");
    hand_recall::require(uniform.policy(cards, 1, false)[0] == 1.0/12, "seat tables merged");
    uniform.update(cards, 0, q);
    double total = 0;
    for (int a = 0; a < 12; ++a) {
        const auto [d, r] = choice(cards, a);
        std::array<Card,2> hidden{};
        int i = 0;
        for (int k = 0; k < 4; ++k) { if (k != d && k != r) { hidden[i++] = cards[k]; } }
        const auto weight = uniform.weight(hidden, cards[r], cards[d], 0);
        hand_recall::require(std::abs(weight/24 - uniform.policy(cards,0,true)[a]) < 1e-12,
                             "learned H4 posterior does not match sampling policy");
        total += weight/24;
    }
    hand_recall::require(std::abs(total-1) < 1e-12, "H4 prior normalization");
    auto permuted = cards;
    std::reverse(permuted.begin(), permuted.end());
    for (int a = 0; a < 12; ++a) {
        const auto x = choice(cards,a), y = choice(permuted,a);
        hand_recall::require(cards[x.first] == permuted[y.first] && cards[x.second] == permuted[y.second],
                             "H4 depends on input order");
    }
    for (int example = 0; example < 20; ++example) {
        auto deck = fresh_deck();
        std::mt19937_64 rng(example+9);
        std::shuffle(deck.begin(),deck.end(),rng);
        const auto input = initial_cards(deck);
        auto reordered = input[0];
        std::reverse(reordered.begin(),reordered.end());
        const auto x = make_h4_view(input[0]), y = make_h4_view(reordered);
        for (int i = 0; i < 4; ++i) {
            hand_recall::require(input[0][x.original_index[i]] == reordered[y.original_index[i]],
                                 "uncached H4 mapping depends on order");
        }
        Betting betting{model,2};
        const std::array<std::pair<int,int>,2> selected{choice(input[0],example%12),choice(input[1],(example+3)%12)};
        betting.rng.seed(91);
        const double left = h4_frozen_value(deck,selected,0,betting,1000,1000);
        betting.rng.seed(91);
        const double right = h4_frozen_value(deck,selected,1,betting,1000,1000);
        hand_recall::require(std::abs(left+right) < 1e-9, "H4 continuation not zero-sum");
    }
    for (int arm = 0; arm < 2; ++arm) {
        Target target(model, arm, nullptr);
        for (int sample = 0; sample < 8; ++sample) {
            auto deck = fresh_deck();
            std::mt19937_64 rng(100 + sample);
            std::shuffle(deck.begin(), deck.end(), rng);
            for (int seat = 0; seat < 2; ++seat) {
                PolicyLBR<Target> old(target, 8, 61+sample);
                H4PolicyLBR<Target> adapted(target, 8, 61+sample);
                target.rng.seed(99);
                const auto value = play_hand_policy_lbr(deck, seat, target, old, 1000, 1000);
                target.rng.seed(99);
                hand_recall::require(value == play_hand_h4_lbr(deck, seat, target, adapted, 1000, 1000),
                                     "heuristic LBR parity failed");
                hand_recall::require(old.stats().actions == adapted.stats().actions &&
                    old.stats().policy_queries == adapted.stats().policy_queries,
                    "heuristic LBR decisions changed");
            }
        }
        target.h4 = &uniform;
        auto state = epoch7::root_for(381);
        auto masked = state;
        masked.players[1-state.actor].hidden.clear();
        masked.players[1-state.actor].has_discard = false;
        masked.simulation_deck.clear();
        H4PolicyLBR<Target> x(target, 16, 93), y(target, 16, 93);
        hand_recall::require(x.choose(state,state.actor) == y.choose(masked,state.actor),
                             "learned-prior LBR leaks hidden state");
    }
    std::array<Learner,2> one, resumed;
    train_hand(one, model, 1, 123);
    save(output / "test_checkpoint.bin", one, 1, 123);
    uint64_t seed = 0;
    hand_recall::require(load(output / "test_checkpoint.bin", resumed, seed) == 1 && seed == 123, "bad resume header");
    train_hand(one, model, 2, 123);
    train_hand(resumed, model, 2, 123);
    save(output / "test_full.bin", one, 2, 123);
    save(output / "test_resumed.bin", resumed, 2, 123);
    std::ifstream a(output / "test_full.bin",std::ios::binary), b(output / "test_resumed.bin",std::ios::binary);
    hand_recall::require(std::string(std::istreambuf_iterator<char>(a),{}) ==
        std::string(std::istreambuf_iterator<char>(b),{}), "resume is not bit-identical");
    std::cout << "SELF_TEST H4 mapping, signed regrets, seats, posterior likelihood, privacy, old-LBR parity, resume: PASS\n";
}

void evaluate(FrozenModel& model, const std::array<Learner,2>& learners,
              const std::filesystem::path& output, int pairs, int particles, uint64_t seed) {
    const std::array<const char*,4> names{"dt_heuristic","dt_learned","km_heuristic","km_learned"};
    const auto path = output / ("lbr_" + std::to_string(seed) + ".csv");
    hand_recall::require(!std::filesystem::exists(path), "existing evaluation");
    std::ofstream csv(path);
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    csv << std::setprecision(17) << "pair,deal_seed";
    for (auto name : names) { csv << ',' << name << "_seat0," << name << "_seat1"; }
    csv << '\n';
    std::array<double,4> sums{}, ess{};
    std::array<uint64_t,4> misses{}, queries{}, decisions{}, h4_queries{}, h4_unseen{};
    for (int index = 0; index < pairs; ++index) {
        const uint64_t deal_seed = seed ^ (uint64_t(index+1)*kMix);
        std::mt19937_64 rng(deal_seed);
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        csv << index << ',' << deal_seed;
        for (int kind = 0; kind < 4; ++kind) {
            const int arm = kind / 2;
            const bool learned = kind % 2;
            Target target(model, arm, learned ? &learners[arm] : nullptr);
            for (int seat = 0; seat < 2; ++seat) {
                const uint64_t action_seed = deal_seed ^ (uint64_t(seat+1)*0x94d049bb133111ebull);
                target.rng.seed(action_seed);
                InitialPolicy h4(learners[arm],1-seat,action_seed ^ 0x31415926ull);
                if (learned) {
                    ++h4_queries[kind];
                    const auto cards = initial_cards(deck)[1-seat];
                    h4_unseen[kind] += !learners[arm].nodes.count(learners[arm].key(cards,1-seat));
                }
                H4PolicyLBR<Target> lbr(target, particles, action_seed ^ 0xabcdef012345ull);
                const double value = play_hand_h4_lbr(deck,seat,target,lbr,1000,1000,learned ? &h4 : nullptr);
                hand_recall::require(std::isfinite(value) && std::abs(value) <= 1000, "invalid LBR payoff");
                csv << ',' << value;
                sums[kind] += value/2;
                const auto stats = lbr.stats();
                misses[kind] += stats.policy_misses;
                queries[kind] += stats.policy_queries;
                decisions[kind] += stats.decisions;
                ess[kind] += stats.average_effective_particles * stats.decisions;
            }
        }
        csv << '\n';
        if ((index+1) % 25 == 0 || index+1 == pairs) {
            csv.flush();
            std::cout << "LBR pairs=" << index+1 << " target_pairs=" << pairs;
            for (int kind = 0; kind < 4; ++kind) { std::cout << ' ' << names[kind] << "_ante=" << sums[kind]/(index+1); }
            std::cout << std::endl;
        }
    }
    std::ofstream meta(output / ("lbr_" + std::to_string(seed) + ".json"));
    meta << "{\"policies\":{";
    for (int kind = 0; kind < 4; ++kind) {
        if (kind) { meta << ','; }
        meta << '"' << names[kind] << "\":{\"misses\":" << misses[kind] << ",\"queries\":" << queries[kind]
             << ",\"h4_queries\":" << h4_queries[kind] << ",\"h4_unseen\":" << h4_unseen[kind]
             << ",\"decisions\":" << decisions[kind] << ",\"mean_ess\":" << ess[kind]/std::max(uint64_t(1),decisions[kind]) << '}';
    }
    meta << "},\"peak_rss_bytes\":" << epoch7::peak_rss() << "}\n";
}
} // namespace h4_ablation

int main(int argc, char** argv) {
    try {
        using namespace h4_ablation;
        hand_recall::require(argc >= 5, "Use python -m agents.cpp_mccfr.h4_ablation.run --help");
        FrozenModel model(argv[2], argv[3]);
        const std::filesystem::path output(argv[4]);
        const std::string mode = argv[1];
        if (mode == "test") { self_test(model,output); return 0; }
        std::array<Learner,2> learners;
        uint64_t seed = 0, hands = 0;
        const auto checkpoint = output / "h4_checkpoint.bin";
        if (std::filesystem::exists(checkpoint)) { hands = load(checkpoint,learners,seed); }
        if (mode == "train") {
            hand_recall::require(argc == 10, "train arguments");
            const uint64_t target = std::stoull(argv[5]), requested_seed = std::stoull(argv[6]);
            const int every = std::stoi(argv[7]), roots = std::stoi(argv[8]), particles = std::stoi(argv[9]);
            hand_recall::require(!hands || seed == requested_seed, "seed mismatch");
            seed = requested_seed;
            const auto start = std::chrono::steady_clock::now();
            if (!hands) { panel(output,learners,model,0,roots,particles); }
            for (uint64_t hand = hands+1; hand <= target; ++hand) {
                train_hand(learners,model,hand,seed);
                if (hand % every == 0 || hand == target) {
                    save(checkpoint,learners,hand,seed);
                    panel(output,learners,model,hand,roots,particles);
                }
                if (hand % 100 == 0 || hand == target) {
                    std::cout << "TRAIN hands=" << hand << " target_hands=" << target
                              << " dt_h4_states=" << learners[0].nodes.size() << " km_h4_states=" << learners[1].nodes.size()
                              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                              << " peak_rss_mib=" << epoch7::peak_rss()/1048576.0 << std::endl;
                }
            }
        } else if (mode == "lbr") {
            hand_recall::require(argc == 8 && hands > 0, "missing trained H4 checkpoint");
            evaluate(model,learners,output,std::stoi(argv[5]),std::stoi(argv[6]),std::stoull(argv[7]));
        } else { throw std::runtime_error("unknown mode"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
}
