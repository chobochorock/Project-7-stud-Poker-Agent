// ASWISS on heads-up seven-card stud (rules v3, 5th to 7th street, heuristic H4 discard/reveal).
//
// Build: g++ -O3 -std=c++17 aswiss_stud.cpp -o aswiss_stud
// Train: ./aswiss_stud train --partition aswiss|atom|context|history [--atlas ATLAS] --roots N [--epoch0 10000]
//                            [--z 2] [--inherit 0] [--seed 1] [--power-cache 1000000] [--checkpoint-every 1000000]
//                            [--snapshot-every 0] [--curve-every 10000] [--h4 heuristic|learn]
//                            [--label window|avg|window+avg] [--avg-frac 0.25] [--history-cards street|every]
//                            [--label-cap 0] [--backward 0] [--backward-min-visits 32] [--backward-max-subkeys 4000000]
//                            [--backward-test pair|labels] [--backward-z Z] [--gate-eps 0] [--gate-stat avg|iter]
//                            --out DIR
//                            (rerun the same command to resume)
// Eval:  ./aswiss_stud eval [--atlas ATLAS] --snapshot FILE --policy average|current [--pairs 2000]
//                           [--particles 64] [--gap-roots 4000] [--seed 2026102701]
// Test:  ./aswiss_stud self-test --atlas ATLAS
//
// Reuses the engine in ../../../cpp_mccfr/stud_mccfr.cpp (rules, power atlas, PolicyLBR) unchanged.
// Atom: the hard power InfoKey of a frozen atlas (power cluster x betting summary x legal mask), i.e.
// exactly the key of the engine's `--bucket power` k-means model. Context: the atom key with the power
// cluster cleared (no card information). Each context partitions its clusters into buckets; clusters
// never assigned explicitly fall in the context's rest bucket. Policy tables live on buckets.
//   context: one bucket per context, never split (the time-step analogue; no atlas needed)
//   atom:    one bucket per atom, never split (the atlas trained directly = hard k-means)
//   aswiss:  starts as context; at every doubling epoch end each atom visited in the epoch gets label
//            argmax_a R(a) when R(a) > z sqrt(sum r(a)^2) over the epoch's sampled regrets, else none.
//            Unvisited members (for a rest bucket: unseen clusters) count as none. A bucket carrying two
//            or more labels is split by label; children restart their tables (--inherit 1: each child starts
//            from a copy of the parent's regret and average tables); no merges.
//            --label avg (method B, ASWISS_CONVERGENCE_KO.md section 9): the labels come instead from the regrets of
//            the bucket average policy, estimated at each epoch end by a separate external-sampling pass of
//            --avg-frac x (epoch length) roots in which both seats play the frozen average policy (own RNG seeded
//            by seed and epoch, so resuming stays exact); same z test. --label window+avg splits by the pair of
//            both labels (the common refinement). Both need --partition aswiss and the heuristic H4.
//   history: bucket-history keys instead of atoms (see "bucket-history keys" below): the label units are keys made of
//            the actor's previous bucket and action, the opponent's actions since, and the power cluster when new cards
//            were observed (--history-cards street: first decision of each street; every: every decision); buckets start
//            as one per (street, legal mask) and split by the same z test. --label window|avg|window+avg as for aswiss
//            (the average-policy pass keeps a trail as training does; with avg alone no window is kept, so the log's
//            L/C/H columns are 0). Heuristic H4 only.
// --label-cap N: the labels of an epoch use at most N roots (window: the epoch's last N roots; average-policy pass:
//            at most N roots). History keys do not saturate (their number grows about 1.85x per epoch), so this bounds
//            the memory of long runs; it also stops the detection threshold from shrinking once epochs exceed N.
// --backward 1 (history keys, --label avg): after each forward split, the backward split of ASWISS_HISTORY_PROOF_KO.md
//            section 8 in its sampled one-step form (Model::hbackward); backward.csv logs it per epoch. The sub-keys come
//            from a replay of the average-policy pass, kept only for keys visited >= --backward-min-visits (default 32),
//            at most --backward-max-subkeys (default 4M, about 0.8 GB) of them; later new sub-keys are dropped.
//            --backward-test pair (default): a key is inconsistent only through a pair of sub-keys that each prefer their
//            own label significantly, with the threshold raised for the number of sub-keys (see hbackward); labels:
//            any two different significant labels (the first version). --backward-z Z: the base of the pair-test
//            threshold (default --z); larger is less sensitive.
// --gate-eps E (history keys, --label avg): judge (split) at an epoch end only when the bucket regret total of the
//            average policy, rho_P/2 per root, estimated from the average-policy pass (cross-fitted between its two
//            halves), is at most E; otherwise the epoch ends without a split. gate.csv logs every check.
//            --gate-stat iter gates instead on the training iterates' regret since the last judgment, sum_B max(0,
//            max_a R_B(a)) / roots / 2 from the sampled regrets MCCFR adds (no extra pass); both are logged.
// Training: external-sampling MCCFR+ (RM+) from a sampled 5th-street root, traversers 0 then 1 on the
// same deal; bucket regrets are applied after each traversal; linear averaging at opponent nodes.
// --h4 learn: the 4th-street discard/reveal (12 actions, exact suit-canonical four-card keys per seat) is learned
// jointly: the opponent's choice is sampled, the traverser enumerates its 12 choices, each followed by the betting
// traversal on the same cards; the H4 table uses the same partition rule (context = one bucket per seat, atom =
// exact keys, aswiss = split by labels). Evaluation then uses H4PolicyLBR (h4_lbr_generated.hpp), whose hidden-card
// posterior uses the learned H4 likelihood; the LBR attacker keeps the heuristic H4.
// Each epoch end writes snapshot_k.bin (partition + tables used during that epoch, before any split);
// --snapshot-every adds snapshot_rN.bin; snapshots.csv indexes both. curve.csv records the sampled BR-gap proxy of
// the current policy on the training deals (free by-product of the traversal) every --curve-every roots.
// Resumable: checkpoint.bin (full training state, RNGs, output lengths) is written every --checkpoint-every roots,
// at epoch ends, on Ctrl+C/SIGTERM and when OUT/STOP appears; rerunning the same command resumes bit-exactly.
// Eval: paired Policy-LBR (the engine's approximate exploitability lower bound, ante/hand; same deals,
// LBR seed and particles for every snapshot) and the sampled BR-gap proxy on fixed deals (sum over the
// traverser's visited information states of max_a Q - sum_a pi Q under external sampling, mean of both
// seats, ante/hand). Neither is exact exploitability.
#define STUD_MCCFR_NO_MAIN
#include "../../../cpp_mccfr/stud_mccfr.cpp"

#include <csignal>
#include <filesystem>
#include <map>
#include <set>
#include <unordered_set>

namespace {

#include "h4_lbr_generated.hpp"  // H4PolicyLBR, play_hand_h4_lbr (make_h4_lbr.py)

constexpr int A = kActionCount;
constexpr uint32_t kNone = 0xffffffffu;
constexpr int kAnte = 1000;

struct NullAtlas final : PowerAtlas {
    uint16_t assign(const State&, int) const override { return 0; }
};

enum class Partition { Context, Atom, Aswiss, History };

struct Bucket {
    std::array<double, A> q{}, s{}, pending{};
    bool dirty = false;
};
struct AtomStats {
    std::array<double, A> r{}, sq{};
    double visits = 0;
};
using AtomMap = std::unordered_map<InfoKey, AtomStats, InfoKeyHash>;
struct Context {
    std::unordered_map<uint16_t, uint32_t> member;  // explicit cluster -> bucket
    uint32_t rest = kNone;
};

InfoKey context_of(InfoKey key) { key.power_cluster = 0; return key; }

std::array<double, A> rm_plus(const std::array<double, A>& q, uint8_t mask) {
    std::array<double, A> s{};
    double total = 0; int count = 0;
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) { s[a] = std::max(0.0, q[a]); total += s[a]; ++count; }
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) s[a] = total > 0 ? s[a] / total : 1.0 / count;
    return s;
}
std::array<double, A> normalized(const std::array<double, A>& num, uint8_t mask) {
    double total = 0;
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) total += num[a];
    if (total <= 0) return uniform_strategy(mask);
    std::array<double, A> s{};
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) s[a] = num[a] / total;
    return s;
}
Action sample_action(const std::array<double, A>& s, uint8_t mask, std::mt19937_64& rng) {
    const double x = std::uniform_real_distribution<double>(0, 1)(rng);
    double acc = 0; Action last = FOLD;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        last = static_cast<Action>(a); acc += s[a];
        if (x < acc) return last;
    }
    return last;
}
void advance(State& state) {  // same as MCCFR::advance_street (draws from the sampled deck)
    if (state.street == 7) { state.terminal = true; return; }
    ++state.street;
    for (int seat = 0; seat < 2; ++seat) {
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (state.street != 7) state.players[seat].shown.push_back(card);
        else state.players[seat].hidden.push_back(card);
        state.players[seat].round_bet = 0;
    }
    state.highest_bet = 0; state.raise_count = 0;
    if (state.players[0].all_in || state.players[1].all_in) { advance(state); return; }
    state.actor = first_bettor(state);
}
void step(State& state, int actor, Action action) {
    const ActionResult result = apply_action(state, actor, action);
    if (result == ActionResult::RoundEnd) advance(state);
    else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
}

template <class T> void put(std::ofstream& out, const T& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void get(std::ifstream& in, T& v) { in.read(reinterpret_cast<char*>(&v), sizeof(T)); }
void put_key(std::ofstream& out, const InfoKey& k) {
    for (uint8_t v : {k.abstraction, k.street, k.category, k.rank_bucket, k.own_public_category, k.own_public_suit,
                      k.own_public_connected, k.opponent_public_category, k.opponent_public_suit,
                      k.opponent_public_connected, k.pot_odds_bucket, k.stack_pot_bucket, k.own_bet_count,
                      k.opponent_bet_count, k.checked, k.last_action_class, k.betting_goal, k.legal_mask,
                      k.history_length}) put(out, v);
    put(out, k.power_cluster); put(out, k.history_code);
}
InfoKey get_key(std::ifstream& in) {
    InfoKey k{};
    for (uint8_t* v : {&k.abstraction, &k.street, &k.category, &k.rank_bucket, &k.own_public_category,
                       &k.own_public_suit, &k.own_public_connected, &k.opponent_public_category,
                       &k.opponent_public_suit, &k.opponent_public_connected, &k.pot_odds_bucket,
                       &k.stack_pot_bucket, &k.own_bet_count, &k.opponent_bet_count, &k.checked,
                       &k.last_action_class, &k.betting_goal, &k.legal_mask, &k.history_length}) get(in, *v);
    get(in, k.power_cluster); get(in, k.history_code);
    return k;
}

// ---- bucket-history keys (--partition history) ---------------------------------------------------------------------
// The key of a decision is (structural context = street and legal mask; the actor's bucket at its previous decision of
// this hand and the action taken there; the opponent's actions since then; the power cluster of the cards when new
// cards were observed, i.e. at the actor's first decision of each street, or at every decision with
// --history-cards every). The past is remembered only through the bucket it fell into. Buckets partition the keys of
// a structural context and split by labels exactly as the atom version; a key whose previous bucket was split later
// resolves through the split genealogy (parent_) to the entry of its ancestor key, so keys refine as earlier buckets
// split. Card information still comes from the power cluster (the heuristic part); betting information comes only from
// the bucket history (no hand-made betting summary).
constexpr uint32_t kUnseen = 0xfffffffeu;  // the previous decision fell in no known bucket (evaluation only)
constexpr uint16_t kNoCards = 0xffffu;     // no new card observation at this decision
struct HKey {  // member key within a structural context
    uint32_t prev = kNone;        // own bucket at the previous own decision of this hand (kNone: first decision)
    uint8_t action = 0xff;        // own action there
    uint16_t cluster = kNoCards;  // power cluster when new cards were observed
    uint64_t opp = 0;             // opponent actions since then, 5 bits each: (street - 5) * 8 + action + 1
    bool operator==(const HKey& o) const { return prev == o.prev && action == o.action && cluster == o.cluster && opp == o.opp; }
    bool operator<(const HKey& o) const {
        return std::tie(prev, action, cluster, opp) < std::tie(o.prev, o.action, o.cluster, o.opp);
    }
};
struct HKeyHash {
    size_t operator()(const HKey& k) const {
        uint64_t h = (uint64_t(k.prev) << 32) ^ (uint64_t(k.action) << 16) ^ k.cluster;
        h ^= k.opp + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};
struct HFull {  // structural context (street << 8 | legal mask) + member key
    uint16_t ctx = 0;
    HKey m;
    bool operator==(const HFull& o) const { return ctx == o.ctx && m == o.m; }
};
struct HFullHash {
    size_t operator()(const HFull& k) const { return HKeyHash()(k.m) * 31 + k.ctx; }
};
using HMap = std::unordered_map<HFull, struct AtomStats, HFullHash>;
struct HFullLess {
    bool operator()(const HFull& a, const HFull& b) const { return a.ctx != b.ctx ? a.ctx < b.ctx : a.m < b.m; }
};
// Average-policy statistics of (key, key of the same player's previous decision) for the backward split.
struct SubKey {
    HFull k;
    uint32_t p;  // id of the previous key in SubMap::prev
    bool operator==(const SubKey& o) const { return p == o.p && k == o.k; }
};
struct SubKeyHash {
    size_t operator()(const SubKey& s) const { return HFullHash()(s.k) * 1000003u ^ s.p; }
};
struct SubMap {
    std::unordered_map<HFull, uint32_t, HFullHash> ids;
    std::vector<HFull> prev;
    std::unordered_map<SubKey, struct AtomStats, SubKeyHash> stats;
    size_t cap = SIZE_MAX;  // no new sub-key once this many are held (frequent ones appear early); their visits are dropped
    uint64_t dropped = 0;
    uint32_t id(const HFull& k) {
        const auto [it, fresh] = ids.try_emplace(k, static_cast<uint32_t>(prev.size()));
        if (fresh) prev.push_back(k);
        return it->second;
    }
};
struct HContext {
    std::unordered_map<HKey, uint32_t, HKeyHash> member;  // explicit key -> bucket
    uint32_t rest = kNone;
};
struct Trail {  // per seat: bucket and action of its last decision, history length after it, street of it
    std::array<uint32_t, 2> prev{kNone, kNone};
    std::array<uint8_t, 2> action{0xff, 0xff};
    std::array<uint16_t, 2> since{0, 0};
    std::array<uint8_t, 2> street{0, 0};
};
void trail_step(Trail& t, const State& before, int actor, Action a, uint32_t bucket) {
    t.prev[actor] = bucket; t.action[actor] = a;
    t.since[actor] = static_cast<uint16_t>(before.history.size() + 1); t.street[actor] = static_cast<uint8_t>(before.street);
}
// The hidden cards of `s` as held at r.street. Cards absent from `s` stay absent (the LBR passes states in which
// the attacker's hidden cards are cleared).
void sync_hidden(State& r, const State& s) {
    for (int seat = 0; seat < 2; ++seat) {
        const auto& h = s.players[seat].hidden;
        r.players[seat].hidden.assign(h.begin(), h.begin() + std::min<size_t>(r.street == 7 ? 3 : 2, h.size()));
    }
}
// The 5th-street root of `s`, with the later cards queued so that replaying s.history reproduces s (a missing 7th-street
// hidden card is queued as a placeholder; sync_hidden removes it again).
State root_of(const State& s) {
    State r;
    r.ante = s.ante; r.effective_stack = s.effective_stack; r.pot = 2 * s.ante;
    for (int seat = 0; seat < 2; ++seat) {
        const Player& p = s.players[seat];
        Player& q = r.players[seat];
        q.hidden.assign(p.hidden.begin(), p.hidden.begin() + std::min<size_t>(2, p.hidden.size()));
        q.shown.assign(p.shown.begin(), p.shown.begin() + std::min<size_t>(3, p.shown.size()));
        q.discarded = p.discarded; q.has_discard = p.has_discard; q.stack_cap = p.stack_cap; q.invested = s.ante;
    }
    for (int seat = 1; seat >= 0; --seat)
        if (s.street == 7) r.simulation_deck.push_back(s.players[seat].hidden.size() > 2 ? s.players[seat].hidden[2] : Card{});
    for (int seat = 1; seat >= 0; --seat) if (s.players[seat].shown.size() > 3) r.simulation_deck.push_back(s.players[seat].shown[3]);
    reset_round(r, 5);
    return r;
}
void put_hkey(std::ofstream& out, const HKey& m) { put(out, m.prev); put(out, m.action); put(out, m.cluster); put(out, m.opp); }
HKey get_hkey(std::ifstream& in) { HKey m; get(in, m.prev); get(in, m.action); get(in, m.cluster); get(in, m.opp); return m; }

// ---- 4th-street discard/reveal decision (12 actions), learned when --h4 learn -------------------------------------
// Information state = (seat, own four cards up to suit relabeling) = (seat, make_h4_view(cards).key); the 12 actions
// are the engine's canonical (discard, reveal) positions (h4_positions / h4_choice). These keys are exact (no card
// abstraction), so H4 atoms are the original information sets. Partition per seat, as for betting contexts:
// context = one bucket per seat, atom = one bucket per key, aswiss = split by labels from one bucket per seat.
constexpr int H = kH4ActionCount;
struct H4Bucket {
    std::array<double, H> q{}, s{}, pending{};
    bool dirty = false;
};
struct H4Atom {
    std::array<double, H> r{}, sq{};
    double visits = 0;
};
struct H4Context {
    std::unordered_map<uint32_t, uint32_t> member;  // explicit key -> bucket
    uint32_t rest = kNone;
};
std::array<double, H> rm_plus_h4(const std::array<double, H>& q) {
    std::array<double, H> s{};
    double total = 0;
    for (int a = 0; a < H; ++a) { s[a] = std::max(0.0, q[a]); total += s[a]; }
    for (double& p : s) p = total > 0 ? p / total : 1.0 / H;
    return s;
}
std::array<double, H> normalized_h4(const std::array<double, H>& num) {
    const double total = std::accumulate(num.begin(), num.end(), 0.0);
    std::array<double, H> s{};
    for (int a = 0; a < H; ++a) s[a] = total > 0 ? num[a] / total : 1.0 / H;
    return s;
}
int sample_h4(const std::array<double, H>& s, std::mt19937_64& rng) {
    const double x = std::uniform_real_distribution<double>(0, 1)(rng);
    double acc = 0;
    for (int a = 0; a < H - 1; ++a) { acc += s[a]; if (x < acc) return a; }
    return H - 1;
}
uint32_t h4_key(const std::vector<Card>& cards) { return make_h4_view(cards).key; }
std::array<std::vector<Card>, 2> initial_cards(const std::vector<Card>& deck) {  // dealing order of prepare_initial_cards
    std::array<std::vector<Card>, 2> cards;
    for (int i = 0; i < 8; ++i) cards[i % 2].push_back(deck[i]);
    return cards;
}
// 5th-street root for given discard/reveal choices; identical to sample_fifth_street_root when the choices are the
// heuristic's (checked in self-test)
State h4_root(const std::vector<Card>& deck, const std::array<std::pair<int, int>, 2>& choices, int ante) {
    State state;
    state.ante = ante;
    state.effective_stack = ante * kEffectiveStackAnte;
    state.pot = ante * 2;
    for (Player& player : state.players) player.invested = ante;
    size_t cursor = 0;
    for (int round = 0; round < 4; ++round)
        for (int seat = 0; seat < 2; ++seat) state.players[seat].hidden.push_back(deck[cursor++]);
    apply_h4_choices(state, choices);
    for (int street = 4; street <= 5; ++street)
        for (int seat = 0; seat < 2; ++seat) state.players[seat].shown.push_back(deck[cursor++]);
    state.simulation_deck.assign(deck.begin() + cursor, deck.end());
    reset_round(state, 5);
    return state;
}

class H4Table {
public:
    explicit H4Table(Partition partition) : partition_(partition) {}
    uint32_t lookup(int seat, uint32_t key) const {
        const H4Context& c = ctx_[seat];
        const auto m = c.member.find(key);
        if (m != c.member.end()) return m->second;
        return partition_ == Partition::Atom ? kNone : c.rest;
    }
    uint32_t lookup_or_create(int seat, uint32_t key) {
        H4Context& c = ctx_[seat];
        if (partition_ == Partition::Atom) {
            auto [m, fresh] = c.member.try_emplace(key, 0u);
            if (fresh) m->second = new_bucket();
            return m->second;
        }
        if (c.rest == kNone) c.rest = new_bucket();
        const auto m = c.member.find(key);
        return m != c.member.end() ? m->second : c.rest;
    }
    std::array<double, H> strategy(int seat, uint32_t key, bool average, bool* found = nullptr) const {
        const uint32_t b = lookup(seat, key);
        if (found) *found = b != kNone;
        if (b == kNone) { std::array<double, H> u; u.fill(1.0 / H); return u; }
        return average ? normalized_h4(buckets_[b].s) : rm_plus_h4(buckets_[b].q);
    }
    std::array<double, H> current(uint32_t b) const { return rm_plus_h4(buckets_[b].q); }
    void add_average(uint32_t b, double weight, const std::array<double, H>& sigma) {
        for (int a = 0; a < H; ++a) buckets_[b].s[a] += weight * sigma[a];
    }
    void add_regret(uint32_t b, int seat, uint32_t key, const std::array<double, H>& v, double node, bool track) {
        H4Bucket& bucket = buckets_[b];
        if (!bucket.dirty) { bucket.dirty = true; touched_.push_back(b); }
        H4Atom* st = track ? &atoms_[(uint64_t(seat) << 32) | key] : nullptr;
        for (int a = 0; a < H; ++a) {
            const double r = v[a] - node;
            bucket.pending[a] += r;
            if (st) { st->r[a] += r; st->sq[a] += r * r; }
        }
        if (st) st->visits += 1;
    }
    void apply() {
        for (uint32_t b : touched_) {
            H4Bucket& bucket = buckets_[b];
            for (int a = 0; a < H; ++a) { bucket.q[a] = std::max(0.0, bucket.q[a] + bucket.pending[a]); bucket.pending[a] = 0; }
            bucket.dirty = false;
        }
        touched_.clear();
    }
    // same label rule as the betting partition (z test on the window sums; unvisited members and unseen keys = none)
    int split(double z, bool inherit) {
        int splits = 0;
        for (int seat = 0; seat < 2; ++seat) {
            H4Context& c = ctx_[seat];
            std::map<uint32_t, int> label_of;
            for (const auto& [packed, st] : atoms_) {
                if (int(packed >> 32) != seat) continue;
                const int best = int(std::max_element(st.r.begin(), st.r.end()) - st.r.begin());
                label_of[uint32_t(packed)] = st.visits > 0 && st.r[best] > z * std::sqrt(st.sq[best]) ? best : -1;
            }
            std::map<uint32_t, std::set<int>> labels;
            if (c.rest != kNone) labels[c.rest].insert(-1);
            for (const auto& [key, b] : c.member) {
                const auto it = label_of.find(key);
                labels[b].insert(it == label_of.end() ? -1 : it->second);
            }
            for (const auto& [key, label] : label_of)
                if (!c.member.count(key) && c.rest != kNone) labels[c.rest].insert(label);
            std::map<uint32_t, std::map<int, uint32_t>> child;
            for (const auto& [b, set] : labels) {
                if (set.size() < 2) continue;
                ++splits;
                for (int label : set) {
                    const uint32_t nb = new_bucket();
                    if (inherit) { buckets_[nb].q = buckets_[b].q; buckets_[nb].s = buckets_[b].s; }
                    child[b][label] = nb;
                }
            }
            if (child.empty()) continue;
            std::vector<std::pair<uint32_t, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            for (const auto& [key, b] : members) {
                const auto it = child.find(b);
                if (it == child.end()) continue;
                const auto l = label_of.find(key);
                c.member[key] = it->second.at(l == label_of.end() ? -1 : l->second);
            }
            const auto rest = child.find(c.rest);
            if (rest != child.end()) {
                for (const auto& [key, label] : label_of)
                    if (!c.member.count(key) && label >= 0) c.member[key] = rest->second.at(label);
                c.rest = rest->second.at(-1);
            }
        }
        return splits;
    }
    void clear_window() { atoms_.clear(); }
    size_t live() const {
        std::set<uint32_t> s;
        for (const auto& c : ctx_) {
            if (c.rest != kNone) s.insert(c.rest);
            for (const auto& [key, b] : c.member) s.insert(b);
        }
        return s.size();
    }
    size_t atoms() const { return atoms_.size(); }
    // snapshot: partition + tables of live buckets (compact ids)
    void save(std::ofstream& out) const {
        std::map<uint32_t, uint32_t> remap;
        auto id = [&](uint32_t b) { return b == kNone ? kNone : remap.try_emplace(b, (uint32_t)remap.size()).first->second; };
        for (const auto& c : ctx_) {
            put(out, id(c.rest));
            std::vector<std::pair<uint32_t, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            put(out, static_cast<uint32_t>(members.size()));
            for (const auto& [key, b] : members) { put(out, key); put(out, id(b)); }
        }
        std::vector<uint32_t> original(remap.size());
        for (const auto& [b, compact] : remap) original[compact] = b;
        put(out, static_cast<uint64_t>(original.size()));
        for (uint32_t b : original) { put(out, buckets_[b].q); put(out, buckets_[b].s); }
    }
    void load(std::ifstream& in) { load_partition(in); load_buckets(in); atoms_.clear(); }
    // checkpoint: every bucket record (ids unchanged) and the atom window
    void save_state(std::ofstream& out) const {
        for (const auto& c : ctx_) {
            put(out, c.rest);
            put(out, static_cast<uint32_t>(c.member.size()));
            for (const auto& [key, b] : c.member) { put(out, key); put(out, b); }
        }
        put(out, static_cast<uint64_t>(buckets_.size()));
        for (const auto& b : buckets_) { put(out, b.q); put(out, b.s); }
        put(out, static_cast<uint64_t>(atoms_.size()));
        for (const auto& [packed, st] : atoms_) { put(out, packed); put(out, st.r); put(out, st.sq); put(out, st.visits); }
    }
    void load_state(std::ifstream& in) {
        load_partition(in); load_buckets(in);
        uint64_t n; get(in, n);
        atoms_.clear();
        for (uint64_t i = 0; i < n; ++i) { uint64_t packed; H4Atom st; get(in, packed); get(in, st.r); get(in, st.sq); get(in, st.visits); atoms_[packed] = st; }
        touched_.clear();
    }

private:
    void load_partition(std::ifstream& in) {
        for (auto& c : ctx_) {
            c = H4Context{};
            get(in, c.rest);
            uint32_t m; get(in, m);
            for (uint32_t j = 0; j < m; ++j) { uint32_t key, b; get(in, key); get(in, b); c.member[key] = b; }
        }
    }
    void load_buckets(std::ifstream& in) {
        uint64_t n; get(in, n);
        buckets_.assign(n, H4Bucket{});
        for (auto& b : buckets_) { get(in, b.q); get(in, b.s); }
    }
    uint32_t new_bucket() { buckets_.emplace_back(); return static_cast<uint32_t>(buckets_.size() - 1); }
    Partition partition_;
    std::array<H4Context, 2> ctx_;
    std::vector<H4Bucket> buckets_;
    std::vector<uint32_t> touched_;
    std::unordered_map<uint64_t, H4Atom> atoms_;
};

class Model {
public:
    Model(Partition partition, const PowerAtlas* atlas, bool learn_h4 = false)
        : partition_(partition), atlas_(atlas), h4_(learn_h4 ? std::make_unique<H4Table>(partition) : nullptr) {
        if (partition != Partition::Context && !atlas) throw std::runtime_error("atom/aswiss partitions need --atlas");
    }
    bool has_h4() const { return h4_ != nullptr; }
    const H4Table* h4() const { return h4_.get(); }
    // --partition history: cards at every decision (true) or only at the first decision of each street (false)
    void set_history_cards(bool every) { hcards_every_ = every; }
    // Regret sums of the training iterates since the last judgment, per bucket, before the RM+ clipping (the sampled
    // counterfactual regrets MCCFR adds; unbiased for the iterates' regret). Kept only for --gate-eps, and checkpointed.
    void track_iterate_regret(bool on) { track_rwin_ = on; }
    void count_root() { if (track_rwin_) ++rwin_roots_; }
    // sum_B max(0, max_a R_B(a)) / roots / 2 over the window (ante per root), then the window restarts if `reset`
    double iterate_regret_half(bool reset) {
        double total = 0;
        for (const auto& r : rwin_) total += std::max(0.0, *std::max_element(r.begin(), r.end()));
        const double v = rwin_roots_ ? total / static_cast<double>(rwin_roots_) / 2 : 0;
        if (reset) { rwin_.assign(rwin_.size(), {}); rwin_roots_ = 0; }
        return v;
    }
    bool history_cards_every() const { return hcards_every_; }
    std::array<double, H> h4_strategy(int seat, const std::vector<Card>& cards, bool average, bool* found = nullptr) const {
        return h4_->strategy(seat, h4_key(cards), average, found);
    }

    // One training traversal of `traverser` from a dealt deck. Heuristic H4: the engine's sample_fifth_street_root.
    // Learned H4: the opponent's discard/reveal is sampled from its current H4 strategy (its average is accumulated
    // here, linear weight); the traverser enumerates its 12 actions, each followed by the betting traversal on the same
    // cards; the H4 regrets are applied with the betting regrets after the traversal (apply_regrets).
    double traverse_root(const std::vector<Card>& deck, int traverser, double weight, std::mt19937_64& rng, bool track) {
        if (partition_ == Partition::History)
            return traverse_h(sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), Trail{}, traverser, weight, rng, track);
        if (!h4_) return traverse(sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), traverser, weight, rng, track);
        const auto cards = initial_cards(deck);
        const int opponent = 1 - traverser;
        const uint32_t key_t = h4_key(cards[traverser]), key_o = h4_key(cards[opponent]);
        const uint32_t b_t = h4_->lookup_or_create(traverser, key_t), b_o = h4_->lookup_or_create(opponent, key_o);
        const auto sigma_t = h4_->current(b_t), sigma_o = h4_->current(b_o);
        h4_->add_average(b_o, weight, sigma_o);
        std::array<std::pair<int, int>, 2> choices;
        choices[opponent] = h4_choice(cards[opponent], sample_h4(sigma_o, rng));
        std::array<double, H> v{};
        double node = 0;
        for (int a = 0; a < H; ++a) {
            choices[traverser] = h4_choice(cards[traverser], a);
            v[a] = traverse(h4_root(deck, choices, kAnte), traverser, weight, rng, track);
            node += sigma_t[a] * v[a];
        }
        h4_->add_regret(b_t, traverser, key_t, v, node, track);
        traversal_gap_ += *std::max_element(v.begin(), v.end()) - node;
        return node;
    }

    // atom key of the actor's information state (context partition: the context key itself)
    InfoKey atom_key(const State& state, int viewer) const {
        return partition_ == Partition::Context ? make_power_key(state, viewer, null_atlas_)
                                                : make_power_key(state, viewer, *atlas_);
    }
    uint32_t lookup(const InfoKey& atom) const {
        const auto it = contexts_.find(context_of(atom));
        if (it == contexts_.end()) return kNone;
        const auto m = it->second.member.find(atom.power_cluster);
        if (m != it->second.member.end()) return m->second;
        return partition_ == Partition::Atom ? kNone : it->second.rest;
    }
    uint32_t lookup_or_create(const InfoKey& atom) {
        auto [it, inserted] = contexts_.try_emplace(context_of(atom));
        Context& c = it->second;
        if (partition_ == Partition::Atom) {
            auto [m, fresh] = c.member.try_emplace(atom.power_cluster, 0u);
            if (fresh) m->second = new_bucket();
            return m->second;
        }
        if (inserted) c.rest = new_bucket();
        const auto m = c.member.find(atom.power_cluster);
        return m != c.member.end() ? m->second : c.rest;
    }
    std::array<double, A> strategy(const State& state, int viewer, bool average, bool* found = nullptr) const {
        const uint8_t mask = valid_mask(state, viewer);
        const uint32_t b = partition_ == Partition::History ? hlookup(hkey_replay(state, viewer)) : lookup(atom_key(state, viewer));
        if (found) *found = b != kNone;
        if (b == kNone) return uniform_strategy(mask);
        return average ? normalized(buckets_[b].s, mask) : rm_plus(buckets_[b].q, mask);
    }

    // one external-sampling traversal for `traverser`; `track` accumulates per-atom window regrets
    double traverse(State state, int traverser, double weight, std::mt19937_64& rng, bool track) {
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey atom = atom_key(state, actor);
        const uint32_t b = lookup_or_create(atom);
        const auto sigma = rm_plus(buckets_[b].q, mask);
        if (actor != traverser) {
            for (int a = 0; a < A; ++a) if (mask & (1u << a)) buckets_[b].s[a] += weight * sigma[a];
            step(state, actor, sample_action(sigma, mask, rng));
            return traverse(std::move(state), traverser, weight, rng, track);
        }
        std::array<double, A> v{};
        double node = 0;
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            State child = state;
            step(child, actor, static_cast<Action>(a));
            v[a] = traverse(std::move(child), traverser, weight, rng, track);
            node += sigma[a] * v[a];
        }
        Bucket& bucket = buckets_[b];  // re-read: the recursion may have grown buckets_
        if (!bucket.dirty) { bucket.dirty = true; touched_.push_back(b); }
        AtomStats* st = track ? &atoms_[atom] : nullptr;
        double best = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            const double r = v[a] - node;
            bucket.pending[a] += r;
            best = std::max(best, r);
            if (st) { st->r[a] += r; st->sq[a] += r * r; }
        }
        if (st) st->visits += 1;
        traversal_gap_ += best;  // diagnostic only: sampled BR-gap proxy of the current policy on training deals
        return node;
    }
    // bucket-history key of the actor's decision, given the trail of the hand so far
    HFull hkey(const State& state, int viewer, const Trail& t) const {
        HFull k;
        k.ctx = static_cast<uint16_t>(state.street << 8 | valid_mask(state, viewer));
        k.m.prev = t.prev[viewer]; k.m.action = t.action[viewer];
        if (hcards_every_ || t.street[viewer] != state.street) k.m.cluster = atlas_->assign(state, viewer);
        int tokens = 0;
        for (size_t i = t.since[viewer]; i < state.history.size(); ++i) {
            const Event& e = state.history[i];
            if (++tokens > 12) throw std::runtime_error("opponent action run exceeds the history key capacity");
            k.m.opp = k.m.opp << 5 | static_cast<uint64_t>((e.street - 5) * 8 + e.action + 1);
        }
        return k;
    }
    // the same key computed from the state alone, by replaying the hand from its 5th-street root (evaluation)
    HFull hkey_replay(const State& state, int viewer) const {
        State r = root_of(state);
        Trail t;
        for (const Event& e : state.history) {
            if (e.actor == viewer) {
                const uint32_t b = hlookup(hkey(r, viewer, t));
                trail_step(t, r, viewer, e.action, b == kNone ? kUnseen : b);
            }
            step(r, e.actor, e.action);
            sync_hidden(r, state);
        }
        return hkey(r, viewer, t);
    }
    uint32_t hresolve(const HContext& c, HKey m) const {
        while (true) {
            const auto it = c.member.find(m);
            if (it != c.member.end()) return it->second;
            if (m.prev == kNone || m.prev == kUnseen || parent_[m.prev] == kNone) return c.rest;
            m.prev = parent_[m.prev];
        }
    }
    // policy of bucket b (uniform for kNone)
    std::array<double, A> bucket_policy(uint32_t b, uint8_t mask, bool average) const {
        if (b == kNone) return uniform_strategy(mask);
        return average ? normalized(buckets_[b].s, mask) : rm_plus(buckets_[b].q, mask);
    }
    uint32_t hlookup(const HFull& k) const {
        const auto it = hctx_.find(k.ctx);
        return it == hctx_.end() ? kNone : hresolve(it->second, k.m);
    }
    uint32_t hlookup_or_create(const HFull& k) {
        auto [it, inserted] = hctx_.try_emplace(k.ctx);
        if (inserted) it->second.rest = new_bucket();
        return hresolve(it->second, k.m);
    }
    // external-sampling traversal with bucket-history keys (the trail carries each seat's last bucket and action)
    double traverse_h(State state, Trail trail, int traverser, double weight, std::mt19937_64& rng, bool track) {
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const HFull key = hkey(state, actor, trail);
        const uint32_t b = hlookup_or_create(key);
        const auto sigma = rm_plus(buckets_[b].q, mask);
        if (actor != traverser) {
            for (int a = 0; a < A; ++a) if (mask & (1u << a)) buckets_[b].s[a] += weight * sigma[a];
            const Action a = sample_action(sigma, mask, rng);
            trail_step(trail, state, actor, a, b);
            step(state, actor, a);
            return traverse_h(std::move(state), trail, traverser, weight, rng, track);
        }
        std::array<double, A> v{};
        double node = 0;
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            State child = state;
            Trail child_trail = trail;
            trail_step(child_trail, state, actor, static_cast<Action>(a), b);
            step(child, actor, static_cast<Action>(a));
            v[a] = traverse_h(std::move(child), child_trail, traverser, weight, rng, track);
            node += sigma[a] * v[a];
        }
        Bucket& bucket = buckets_[b];  // re-read: the recursion may have grown buckets_
        if (!bucket.dirty) { bucket.dirty = true; touched_.push_back(b); }
        AtomStats* st = track ? &hatoms_[key] : nullptr;
        double best = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            const double r = v[a] - node;
            bucket.pending[a] += r;
            best = std::max(best, r);
            if (st) { st->r[a] += r; st->sq[a] += r * r; }
        }
        if (st) st->visits += 1;
        traversal_gap_ += best;
        return node;
    }
    // History split: split() with history keys as the label units. Explicit entries not visited this epoch (in
    // particular entries whose previous bucket has since been split) count as none, like the rest bucket's unseen keys.
    // Labels from the window sums, from the average-policy regrets `avg` (avg without window), or from both (the
    // common refinement), as in split().
    static int hlabel(const HFull& k, const AtomStats& st, double z) {
        const uint8_t mask = k.ctx & 0xff;
        int best = -1;
        for (int a = 0; a < A; ++a) if ((mask & (1u << a)) && (best < 0 || st.r[a] > st.r[best])) best = a;
        return st.visits > 0 && st.r[best] > z * std::sqrt(st.sq[best]) ? best : -1;
    }
    int hsplit(double z, const HMap* avg = nullptr, bool use_window = true) {
        auto label = [z](const HFull& k, const AtomStats& st) { return hlabel(k, st, z); };
        struct Labels { int window = -1, average = -1; };
        std::unordered_map<HFull, Labels, HFullHash> labels_of;
        if (use_window) for (const auto& [k, st] : hatoms_) labels_of[k].window = label(k, st);
        if (avg) for (const auto& [k, st] : *avg) labels_of[k].average = label(k, st);
        std::map<uint16_t, std::vector<std::pair<HKey, int>>> by_ctx;  // ordered: reproducible bucket ids
        for (const auto& [k, l] : labels_of)
            by_ctx[k.ctx].push_back({k.m, !avg ? l.window : !use_window ? l.average
                                          : l.window < 0 && l.average < 0 ? -1 : (l.window + 1) * 16 + (l.average + 1)});
        int splits = 0;
        for (auto& [ctx, visited] : by_ctx) {
            const auto found = hctx_.find(ctx);
            if (found == hctx_.end()) continue;
            HContext& c = found->second;
            std::sort(visited.begin(), visited.end());
            std::unordered_map<HKey, int, HKeyHash> label_of(visited.begin(), visited.end());
            std::vector<std::pair<HKey, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            std::map<uint32_t, std::set<int>> labels;  // bucket -> labels of its keys
            labels[c.rest].insert(-1);                 // unseen keys
            for (const auto& [m, b] : members) {
                const auto it = label_of.find(m);
                labels[b].insert(it == label_of.end() ? -1 : it->second);
            }
            std::vector<std::pair<HKey, uint32_t>> implicit;  // visited keys without an entry, with their bucket
            for (const auto& [m, label] : visited) {
                if (c.member.count(m)) continue;
                const uint32_t b = hresolve(c, m);
                implicit.push_back({m, b});
                labels[b].insert(label);
            }
            std::map<uint32_t, std::map<int, uint32_t>> child;
            for (const auto& [b, set] : labels) {
                if (set.size() < 2) continue;
                ++splits;
                for (int label : set) child[b][label] = new_bucket(b);
            }
            if (child.empty()) continue;
            for (const auto& [m, b] : members) {
                const auto it = child.find(b);
                if (it == child.end()) continue;
                const auto l = label_of.find(m);
                c.member[m] = it->second.at(l == label_of.end() ? -1 : l->second);
            }
            for (const auto& [m, b] : implicit) {  // unlabelled ones keep following their ancestor entry or the rest
                const auto it = child.find(b);
                const int label = label_of.at(m);
                if (it != child.end() && label >= 0) c.member[m] = it->second.at(label);
            }
            const auto rest = child.find(c.rest);
            if (rest != child.end()) c.rest = rest->second.at(-1);
        }
        return splits;
    }
    // Backward split (ASWISS_HISTORY_PROOF_KO.md section 8), sampled one-step version. Information sets do not recur
    // here, so the members of a key are told apart by the key of the same player's previous decision (sub-keys of the
    // average-policy pass, `sub`), whose labels use the same z test. A key whose sub-keys carry two or more labels hides
    // cancellation the forward rule cannot see; its sub-keys share the previous bucket, where the difference was
    // merged. Each such previous key is marked with the labels its sub-keys carry, and every bucket holding marked keys
    // (after hsplit) is split by the marks: one child per set of labels, one for the unmarked keys and the rest. A pair
    // of sub-keys of one key with different labels that the sets leave together is separated by marks naming the key.
    // Children copy the tables of the bucket they split from: they share its label, only later keys change.
    struct Backward { int splits = 0; size_t subkeys = 0, inconsistent = 0, pairs = 0, two_labels = 0; };
    // pair_test: a key counts as inconsistent only through a pair of sub-keys i, j with different labels that each
    // prefer their own label to the other's: R_i(l_i) - R_i(l_j) > z_m (sqrt(Q_i(l_i)) + sqrt(Q_i(l_j))) and the same
    // for j (sqrt(sum (x-y)^2) <= sqrt(sum x^2) + sqrt(sum y^2), so the margin is conservative), where
    // z_m = sqrt(zb^2 + 2 ln m) (zb: --backward-z, default z) for the m labelled sub-keys of the key (about the same false-positive rate per key
    // however many sub-keys it has). Without pair_test any two different labels count.
    Backward hbackward(double z, const SubMap& sub, bool pair_test = false, double zb = -1) {
        if (zb < 0) zb = z;
        Backward res;
        res.subkeys = sub.stats.size();
        struct Labelled { uint32_t p; int l; const AtomStats* st; };
        std::map<HFull, std::vector<Labelled>, HFullLess> by_key;
        for (const auto& [sk, st] : sub.stats) {
            const int l = hlabel(sk.k, st, z);
            if (l >= 0) by_key[sk.k].push_back({sk.p, l, &st});
        }
        auto prefers = [](const AtomStats& st, int a, int b, double zt) {
            return st.r[a] - st.r[b] > zt * (std::sqrt(st.sq[a]) + std::sqrt(st.sq[b]));
        };
        std::map<uint32_t, uint32_t> coarse;                          // previous key id -> label bits
        std::map<uint32_t, std::vector<std::pair<uint32_t, int>>> fine;  // previous key id -> (key index, label)
        struct Pair { uint32_t lo, lo0, kid; int lam, lam0; };
        std::vector<Pair> pairs;
        uint32_t kid = 0;
        for (auto& [k, v] : by_key) {
            ++kid;
            std::sort(v.begin(), v.end(), [&](const auto& x, const auto& y) { return HFullLess()(sub.prev[x.p], sub.prev[y.p]); });
            if (std::all_of(v.begin(), v.end(), [&](const auto& e) { return e.l == v[0].l; })) continue;
            ++res.two_labels;
            if (!pair_test) {
                ++res.inconsistent;
                for (const auto& e : v) {
                    coarse[e.p] |= 1u << e.l;
                    if (e.l != v[0].l) pairs.push_back({e.p, v[0].p, kid, e.l, v[0].l});
                }
                continue;
            }
            const double zm = std::sqrt(zb * zb + 2 * std::log(static_cast<double>(v.size())));
            std::vector<char> confirmed(v.size(), 0);
            for (size_t i = 0; i < v.size(); ++i)
                for (size_t j = i + 1; j < v.size(); ++j)
                    if (v[i].l != v[j].l && prefers(*v[i].st, v[i].l, v[j].l, zm) && prefers(*v[j].st, v[j].l, v[i].l, zm)) {
                        confirmed[i] = confirmed[j] = 1;
                        pairs.push_back({v[i].p, v[j].p, kid, v[i].l, v[j].l});
                    }
            if (std::find(confirmed.begin(), confirmed.end(), 1) == confirmed.end()) continue;
            ++res.inconsistent;
            for (size_t i = 0; i < v.size(); ++i) if (confirmed[i]) coarse[v[i].p] |= 1u << v[i].l;
        }
        res.pairs = pairs.size();
        for (const Pair& q : pairs)
            if (coarse[q.lo] == coarse[q.lo0] && hlookup(sub.prev[q.lo]) == hlookup(sub.prev[q.lo0])) {
                fine[q.lo].push_back({q.kid, q.lam}); fine[q.lo0].push_back({q.kid, q.lam0});
            }
        // context -> bucket (after hsplit) -> signature -> marked keys; ordered, so bucket ids are reproducible
        std::map<uint16_t, std::map<uint32_t, std::map<std::vector<uint32_t>, std::vector<HKey>>>> groups;
        for (const auto& [p, bits] : coarse) {
            const HFull& k = sub.prev[p];
            const uint32_t b = hlookup(k);
            if (b == kNone) continue;
            std::vector<uint32_t> sig{bits};
            if (const auto f = fine.find(p); f != fine.end()) {
                auto v = f->second;
                std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end());
                for (const auto& [kk, l] : v) { sig.push_back(kk); sig.push_back(static_cast<uint32_t>(l)); }
            }
            groups[k.ctx][b][sig].push_back(k.m);
        }
        for (auto& [ctx, by_bucket] : groups) {
            HContext& c = hctx_.at(ctx);
            std::map<uint32_t, std::vector<HKey>> explicit_of;
            for (const auto& [m, b] : c.member) if (by_bucket.count(b)) explicit_of[b].push_back(m);
            for (auto& [b, sigs] : by_bucket) {
                std::unordered_set<HKey, HKeyHash> marked;
                for (const auto& [sig, keys] : sigs) marked.insert(keys.begin(), keys.end());
                auto& ex = explicit_of[b];
                std::sort(ex.begin(), ex.end());
                const bool others = c.rest == b || std::any_of(ex.begin(), ex.end(), [&](const HKey& m) { return !marked.count(m); });
                if (sigs.size() + (others ? 1 : 0) < 2) continue;
                ++res.splits;
                auto child = [&] {
                    const uint32_t nb = new_bucket(b);
                    buckets_[nb].q = buckets_[b].q; buckets_[nb].s = buckets_[b].s;
                    return nb;
                };
                for (const auto& [sig, keys] : sigs) {
                    const uint32_t nb = child();
                    for (const HKey& m : keys) c.member[m] = nb;
                }
                if (others) {
                    const uint32_t nb = child();
                    for (const HKey& m : ex) if (!marked.count(m)) c.member[m] = nb;
                    if (c.rest == b) c.rest = nb;
                }
            }
        }
        return res;
    }
    // sum of max_a Q - E_sigma Q over the traverser's nodes since the last call (ante units)
    double take_traversal_gap() { const double g = traversal_gap_; traversal_gap_ = 0; return g; }
    // Untouched buckets have no pending regret, so only touched ones are updated (RM+).
    void apply_regrets() {
        if (track_rwin_ && rwin_.size() < buckets_.size()) rwin_.resize(buckets_.size());
        for (uint32_t b : touched_) {
            Bucket& bucket = buckets_[b];
            if (track_rwin_) for (int a = 0; a < A; ++a) rwin_[b][a] += bucket.pending[a];
            for (int a = 0; a < A; ++a) { bucket.q[a] = std::max(0.0, bucket.q[a] + bucket.pending[a]); bucket.pending[a] = 0; }
            bucket.dirty = false;
        }
        touched_.clear();
        if (h4_) h4_->apply();
    }

    struct Window { double L = 0, C = 0; size_t atoms = 0; };
    // sampled atom-level L and bucket-level C of the epoch's window sums (diagnostic)
    Window window() const {
        if (partition_ == Partition::History) {
            Window w; w.atoms = hatoms_.size();
            std::unordered_map<uint32_t, std::array<double, A>> sums;
            std::unordered_map<uint32_t, uint8_t> masks;
            for (const auto& [k, st] : hatoms_) {
                w.L += psi(st.r, k.ctx & 0xff);
                const uint32_t b = hlookup(k);
                auto& s = sums[b];
                for (int a = 0; a < A; ++a) s[a] += st.r[a];
                masks[b] = k.ctx & 0xff;
            }
            for (const auto& [b, s] : sums) w.C += psi(s, masks[b]);
            return w;
        }
        Window w; w.atoms = atoms_.size();
        std::unordered_map<uint32_t, std::array<double, A>> sums;
        std::unordered_map<uint32_t, uint8_t> masks;
        for (const auto& [atom, st] : atoms_) {
            w.L += psi(st.r, atom.legal_mask);
            const uint32_t b = lookup(atom);
            auto& s = sums[b];
            for (int a = 0; a < A; ++a) s[a] += st.r[a];
            masks[b] = atom.legal_mask;
        }
        for (const auto& [b, s] : sums) w.C += psi(s, masks[b]);
        return w;
    }
    // label of one atom: argmax_a R(a) when R(a) > z sqrt(sum r(a)^2), else -1 (none)
    static int atom_label(const InfoKey& atom, const AtomStats& st, double z) {
        int best = -1;
        for (int a = 0; a < A; ++a) if ((atom.legal_mask & (1u << a)) && (best < 0 || st.r[a] > st.r[best])) best = a;
        return st.visits > 0 && st.r[best] > z * std::sqrt(st.sq[best]) ? best : -1;
    }
    // ASWISS refinement; returns the number of buckets split. Labels come from the window sums, from the
    // average-policy regrets `avg` (avg without window), or from the pair of both (the common refinement).
    int split(double z, bool inherit = false, const AtomMap* avg = nullptr, bool use_window = true) {
        if (partition_ == Partition::History) return hsplit(z);
        struct Labels { int window = -1, average = -1; };
        std::unordered_map<InfoKey, Labels, InfoKeyHash> labels_of;
        if (use_window) for (const auto& [atom, st] : atoms_) labels_of[atom].window = atom_label(atom, st, z);
        if (avg) for (const auto& [atom, st] : *avg) labels_of[atom].average = atom_label(atom, st, z);
        std::map<InfoKey, std::vector<std::pair<uint16_t, int>>, KeyLess> by_context;  // ordered: reproducible ids
        for (const auto& [atom, l] : labels_of) {
            const int label = !avg ? l.window : !use_window ? l.average
                            : l.window < 0 && l.average < 0 ? -1 : (l.window + 1) * 16 + (l.average + 1);
            by_context[context_of(atom)].push_back({atom.power_cluster, label});
        }
        int splits = 0;
        for (auto& [ctx, visited] : by_context) {
            const auto found = contexts_.find(ctx);
            if (found == contexts_.end()) continue;  // a context only the average-policy pass reached
            Context& c = found->second;
            std::sort(visited.begin(), visited.end());
            std::unordered_map<uint16_t, int> label_of(visited.begin(), visited.end());
            std::map<uint32_t, std::set<int>> labels;  // bucket -> labels of its members
            labels[c.rest].insert(-1);                 // unseen clusters
            for (const auto& [cluster, b] : c.member) {
                const auto it = label_of.find(cluster);
                labels[b].insert(it == label_of.end() ? -1 : it->second);
            }
            for (const auto& [cluster, label] : visited)
                if (!c.member.count(cluster)) labels[c.rest].insert(label);
            std::map<uint32_t, std::map<int, uint32_t>> child;  // splitting bucket -> label -> new bucket
            for (const auto& [b, set] : labels) {
                if (set.size() < 2) continue;
                ++splits;
                for (int label : set) {
                    const uint32_t c = new_bucket();
                    if (inherit) { buckets_[c].q = buckets_[b].q; buckets_[c].s = buckets_[b].s; }
                    child[b][label] = c;
                }
            }
            if (child.empty()) continue;
            std::vector<std::pair<uint16_t, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            for (const auto& [cluster, b] : members) {
                const auto it = child.find(b);
                if (it == child.end()) continue;
                const auto l = label_of.find(cluster);
                c.member[cluster] = it->second.at(l == label_of.end() ? -1 : l->second);
            }
            const auto rest = child.find(c.rest);
            if (rest != child.end()) {
                for (const auto& [cluster, label] : visited)
                    if (!c.member.count(cluster) && label >= 0) c.member[cluster] = rest->second.at(label);
                c.rest = rest->second.at(-1);
            }
        }
        last_h4_splits_ = h4_ ? h4_->split(z, inherit) : 0;
        return splits;
    }
    void clear_window() { atoms_.clear(); hatoms_.clear(); if (h4_) h4_->clear_window(); }
    int last_h4_splits() const { return last_h4_splits_; }
    size_t h4_live() const { return h4_ ? h4_->live() : 0; }

    // Full training state for checkpoints: every bucket record (ids unchanged), the partition and the atom window.
    // Pending regrets are always zero between roots, so they are not stored.
    void save_state(std::ofstream& out) const {
        put(out, static_cast<uint8_t>(partition_));
        if (partition_ == Partition::History) {
            put(out, static_cast<uint8_t>(hcards_every_));
            put_history(out);
            put(out, static_cast<uint64_t>(hatoms_.size()));
            for (const auto& [k, st] : hatoms_) { put(out, k.ctx); put_hkey(out, k.m); put(out, st.r); put(out, st.sq); put(out, st.visits); }
            if (track_rwin_) {
                put(out, rwin_roots_); put(out, static_cast<uint64_t>(rwin_.size()));
                for (const auto& r : rwin_) put(out, r);
            }
            return;
        }
        put(out, static_cast<uint8_t>(h4_ ? 1 : 0));
        if (h4_) h4_->save_state(out);
        put(out, static_cast<uint64_t>(contexts_.size()));
        for (const auto& [ctx, c] : contexts_) {
            put_key(out, ctx); put(out, c.rest);
            put(out, static_cast<uint32_t>(c.member.size()));
            for (const auto& [cluster, b] : c.member) { put(out, cluster); put(out, b); }
        }
        put(out, static_cast<uint64_t>(buckets_.size()));
        for (const auto& b : buckets_) { put(out, b.q); put(out, b.s); }
        put(out, static_cast<uint64_t>(atoms_.size()));
        for (const auto& [atom, st] : atoms_) { put_key(out, atom); put(out, st.r); put(out, st.sq); put(out, st.visits); }
    }
    void load_state(std::ifstream& in) {
        uint8_t partition; get(in, partition);
        if (partition != static_cast<uint8_t>(partition_)) throw std::runtime_error("checkpoint partition differs");
        if (partition_ == Partition::History) {
            uint8_t every; get(in, every);
            if (bool(every) != hcards_every_) throw std::runtime_error("checkpoint --history-cards differs");
            get_history(in);
            uint64_t n; get(in, n);
            hatoms_.clear();
            for (uint64_t i = 0; i < n; ++i) {
                HFull k; get(in, k.ctx); k.m = get_hkey(in);
                AtomStats st; get(in, st.r); get(in, st.sq); get(in, st.visits);
                hatoms_[k] = st;
            }
            if (track_rwin_) {
                uint64_t m; get(in, rwin_roots_); get(in, m);
                rwin_.assign(m, {});
                for (auto& r : rwin_) get(in, r);
            }
            touched_.clear();
            return;
        }
        uint8_t with_h4; get(in, with_h4);
        if (bool(with_h4) != bool(h4_)) throw std::runtime_error("checkpoint H4 mode differs");
        if (h4_) h4_->load_state(in);
        uint64_t n; get(in, n);
        contexts_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey ctx = get_key(in);
            Context c; get(in, c.rest);
            uint32_t m; get(in, m);
            for (uint32_t j = 0; j < m; ++j) { uint16_t cluster; uint32_t b; get(in, cluster); get(in, b); c.member[cluster] = b; }
            contexts_[ctx] = std::move(c);
        }
        get(in, n);
        buckets_.assign(n, Bucket{});
        for (auto& b : buckets_) { get(in, b.q); get(in, b.s); }
        get(in, n);
        atoms_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey atom = get_key(in);
            AtomStats st; get(in, st.r); get(in, st.sq); get(in, st.visits);
            atoms_[atom] = st;
        }
        touched_.clear();
    }

    // live buckets (referenced by a partition) per street 5/6/7
    std::array<size_t, 3> live_buckets() const {
        std::array<std::set<uint32_t>, 3> live;
        if (partition_ == Partition::History) {  // referenced by a rest or an explicit entry (stale entries included)
            for (const auto& [ctx, c] : hctx_) {
                auto& s = live[(ctx >> 8) - 5];
                s.insert(c.rest);
                for (const auto& [m, b] : c.member) s.insert(b);
            }
            return {live[0].size(), live[1].size(), live[2].size()};
        }
        for (const auto& [ctx, c] : contexts_) {
            auto& s = live[ctx.street - 5];
            if (c.rest != kNone) s.insert(c.rest);
            for (const auto& [cluster, b] : c.member) s.insert(b);
        }
        return {live[0].size(), live[1].size(), live[2].size()};
    }
    size_t contexts() const { return partition_ == Partition::History ? hctx_.size() : contexts_.size(); }
    size_t bucket_records() const { return buckets_.size(); }  // live + retired (split parents are never freed)
    // approximate bytes of the training tables: bucket records, the atom window and the partition maps
    std::array<double, 3> table_megabytes() const {
        if (partition_ == Partition::History) {
            size_t members = 0;
            for (const auto& [ctx, c] : hctx_) members += c.member.size();
            constexpr double node = 2 * sizeof(void*);
            return {buckets_.capacity() * sizeof(Bucket) / 1e6, hatoms_.size() * (sizeof(HFull) + sizeof(AtomStats) + node) / 1e6,
                    (members * (sizeof(HKey) + sizeof(uint32_t) + node) + parent_.capacity() * sizeof(uint32_t)) / 1e6};
        }
        size_t members = 0;
        for (const auto& [ctx, c] : contexts_) members += c.member.size();
        constexpr double node = 2 * sizeof(void*);  // hash-node link and cached hash
        return {buckets_.capacity() * sizeof(Bucket) / 1e6,
                atoms_.size() * (sizeof(InfoKey) + sizeof(AtomStats) + node) / 1e6,
                (contexts_.size() * (sizeof(InfoKey) + sizeof(Context) + node) +
                 members * (sizeof(uint16_t) + sizeof(uint32_t) + node)) / 1e6};
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        out.write("ASWSTUD2", 8);
        put(out, static_cast<uint8_t>(partition_));
        put(out, static_cast<uint8_t>(h4_ ? 1 : 0));
        if (partition_ == Partition::History) {  // every bucket record is kept: keys refer to retired buckets by id
            put(out, static_cast<uint8_t>(hcards_every_));
            put_history(out);
            return;
        }
        if (h4_) h4_->save(out);
        std::map<uint32_t, uint32_t> remap;  // live bucket -> compact id
        auto id = [&](uint32_t b) { return b == kNone ? kNone : remap.try_emplace(b, (uint32_t)remap.size()).first->second; };
        std::vector<std::pair<InfoKey, const Context*>> ordered;
        for (const auto& [ctx, c] : contexts_) ordered.push_back({ctx, &c});
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return KeyLess()(x.first, y.first); });
        put(out, static_cast<uint64_t>(ordered.size()));
        for (const auto& [ctx, c] : ordered) {
            put_key(out, ctx); put(out, id(c->rest));
            std::vector<std::pair<uint16_t, uint32_t>> members(c->member.begin(), c->member.end());
            std::sort(members.begin(), members.end());
            put(out, static_cast<uint32_t>(members.size()));
            for (const auto& [cluster, b] : members) { put(out, cluster); put(out, id(b)); }
        }
        std::vector<uint32_t> original(remap.size());
        for (const auto& [b, compact] : remap) original[compact] = b;
        put(out, static_cast<uint64_t>(original.size()));
        for (uint32_t b : original) { put(out, buckets_[b].q); put(out, buckets_[b].s); }
    }
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char magic[8];
        in.read(magic, 8);
        const std::string version(magic, 8);
        if (!in || (version != "ASWSTUD1" && version != "ASWSTUD2")) throw std::runtime_error("not an ASWISS stud snapshot: " + path);
        uint8_t partition; get(in, partition);
        if (partition != static_cast<uint8_t>(partition_)) throw std::runtime_error("snapshot partition differs from --partition");
        uint8_t with_h4 = 0;
        if (version == "ASWSTUD2") get(in, with_h4);
        h4_.reset();
        if (with_h4) { h4_ = std::make_unique<H4Table>(partition_); h4_->load(in); }
        if (partition_ == Partition::History) {
            uint8_t every; get(in, every);
            hcards_every_ = every;
            get_history(in);
            if (!in) throw std::runtime_error("truncated snapshot: " + path);
            return;
        }
        uint64_t n; get(in, n);
        contexts_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey ctx = get_key(in);
            Context c; get(in, c.rest);
            uint32_t m; get(in, m);
            for (uint32_t j = 0; j < m; ++j) { uint16_t cluster; uint32_t b; get(in, cluster); get(in, b); c.member[cluster] = b; }
            contexts_[ctx] = std::move(c);
        }
        get(in, n);
        buckets_.assign(n, Bucket{});
        for (auto& b : buckets_) { get(in, b.q); get(in, b.s); }
        if (!in) throw std::runtime_error("truncated snapshot: " + path);
    }

private:
    struct KeyLess {
        bool operator()(const InfoKey& x, const InfoKey& y) const {
            auto t = [](const InfoKey& k) {
                return std::make_tuple(k.street, k.legal_mask, k.pot_odds_bucket, k.stack_pot_bucket, k.own_bet_count,
                                       k.opponent_bet_count, k.checked, k.last_action_class, k.betting_goal, k.abstraction,
                                       k.category, k.rank_bucket, k.history_length, k.history_code, k.power_cluster);
            };
            return t(x) < t(y);
        }
    };
    static double psi(const std::array<double, A>& x, uint8_t mask) {
        double m = 0;
        for (int a = 0; a < A; ++a) if (mask & (1u << a)) m = std::max(m, x[a]);
        return m;
    }
    uint32_t new_bucket(uint32_t parent = kNone) {
        buckets_.emplace_back();
        if (partition_ == Partition::History) parent_.push_back(parent);
        return static_cast<uint32_t>(buckets_.size() - 1);
    }
    // history partition, genealogy and every bucket record (snapshots and checkpoints)
    void put_history(std::ofstream& out) const {
        std::vector<uint16_t> ids;
        for (const auto& [ctx, c] : hctx_) ids.push_back(ctx);
        std::sort(ids.begin(), ids.end());
        put(out, static_cast<uint64_t>(ids.size()));
        for (uint16_t ctx : ids) {
            const HContext& c = hctx_.at(ctx);
            put(out, ctx); put(out, c.rest);
            std::vector<std::pair<HKey, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            put(out, static_cast<uint64_t>(members.size()));
            for (const auto& [m, b] : members) { put_hkey(out, m); put(out, b); }
        }
        put(out, static_cast<uint64_t>(buckets_.size()));
        for (size_t b = 0; b < buckets_.size(); ++b) { put(out, buckets_[b].q); put(out, buckets_[b].s); put(out, parent_[b]); }
    }
    void get_history(std::ifstream& in) {
        uint64_t n; get(in, n);
        hctx_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            uint16_t ctx; get(in, ctx);
            HContext c; get(in, c.rest);
            uint64_t m; get(in, m);
            for (uint64_t j = 0; j < m; ++j) { const HKey key = get_hkey(in); uint32_t b; get(in, b); c.member[key] = b; }
            hctx_[ctx] = std::move(c);
        }
        get(in, n);
        buckets_.assign(n, Bucket{});
        parent_.assign(n, kNone);
        for (uint64_t b = 0; b < n; ++b) { get(in, buckets_[b].q); get(in, buckets_[b].s); get(in, parent_[b]); }
    }

    Partition partition_;
    const PowerAtlas* atlas_;
    std::unique_ptr<H4Table> h4_;
    int last_h4_splits_ = 0;
    NullAtlas null_atlas_;
    std::unordered_map<InfoKey, Context, InfoKeyHash> contexts_;
    std::vector<Bucket> buckets_;
    std::vector<uint32_t> touched_;
    std::unordered_map<InfoKey, AtomStats, InfoKeyHash> atoms_;
    double traversal_gap_ = 0;
    // --partition history
    bool hcards_every_ = false;
    std::unordered_map<uint16_t, HContext> hctx_;
    std::vector<uint32_t> parent_;  // bucket -> the bucket it was split from (kNone: a context's first bucket)
    bool track_rwin_ = false;
    std::vector<std::array<double, A>> rwin_;  // see track_iterate_regret
    uint64_t rwin_roots_ = 0;
    HMap hatoms_;
};

// frozen snapshot as a PolicyLBR target; with a learned H4 it also supplies the H4 likelihood used by H4PolicyLBR
struct SnapshotPolicy {
    const Model& model;
    bool average;
    mutable std::mt19937_64 rng;
    std::array<double, A> policy(const State& state, int seat, bool* found = nullptr) const {
        return model.strategy(state, seat, average, found);
    }
    Action choose(const State& state, int seat, int) {
        return sample_action(policy(state, seat), valid_mask(state, seat), rng);
    }
    bool learned_h4() const { return model.has_h4(); }
    // weight of a hidden-card particle: 24 x P(this discard/reveal | four cards), on the scale of h4_multiplicity
    double h4_weight(const std::array<Card, 2>& hidden, Card shown, Card discarded, int seat) const {
        const std::vector<Card> cards{hidden[0], hidden[1], shown, discarded};
        const auto p = model.h4_strategy(seat, cards, average);
        for (int a = 0; a < H; ++a) {
            const auto [d, r] = h4_choice(cards, a);
            if (cards[d] == discarded && cards[r] == shown) return 24 * p[a];
        }
        throw std::runtime_error("H4 likelihood mapping failed");
    }
};
// the target's own discard/reveal in LBR hands, sampled from the snapshot's H4 policy for `seat`
struct LearnedH4 final : H4Policy {
    const Model& model;
    bool average;
    int seat = 0;
    std::mt19937_64 rng;
    LearnedH4(const Model& m, bool avg, uint64_t seed) : model(m), average(avg), rng(seed) {}
    std::pair<int, int> choose(const std::vector<Card>& cards) override {
        return h4_choice(cards, sample_h4(model.h4_strategy(seat, cards, average), rng));
    }
    size_t buckets() const override { return model.h4_live(); }
};

// sampled BR-gap proxy of one traverser on one deal (ante units)
double gap_traverse(const Model& model, bool average, State state, int traverser, std::mt19937_64& rng, double& gap) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const auto sigma = model.strategy(state, actor, average);
    if (actor != traverser) {
        step(state, actor, sample_action(sigma, mask, rng));
        return gap_traverse(model, average, std::move(state), traverser, rng, gap);
    }
    double node = 0, best = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        const double v = gap_traverse(model, average, std::move(child), traverser, rng, gap);
        node += sigma[a] * v; best = std::max(best, v);
    }
    gap += best - node;
    return node;
}

// One external-sampling traversal under the frozen average policy (method B labels): adds the traverser's sampled
// regrets v(a) - sum_b avg(b) v(b) to `stats` per atom. The model is not changed.
double avg_traverse(const Model& model, State state, int traverser, std::mt19937_64& rng, AtomMap& stats) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const auto sigma = model.strategy(state, actor, true);
    if (actor != traverser) {
        step(state, actor, sample_action(sigma, mask, rng));
        return avg_traverse(model, std::move(state), traverser, rng, stats);
    }
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        v[a] = avg_traverse(model, std::move(child), traverser, rng, stats);
        node += sigma[a] * v[a];
    }
    AtomStats& st = stats[model.atom_key(state, actor)];
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        const double r = v[a] - node;
        st.r[a] += r; st.sq[a] += r * r;
    }
    st.visits += 1;
    return node;
}

// avg_traverse with bucket-history keys: the trail carries each seat's last bucket under the frozen partition. With
// `sub`, the traverser's regrets at the keys in `only` are also summed per (key, key of its previous decision); `pk`
// carries each seat's previous key (ctx 0: none yet). `stats` may be null (a replay that only fills `sub`).
using HSet = std::unordered_set<HFull, HFullHash>;
using BucketHalves = std::unordered_map<uint32_t, std::array<double, 2 * A>>;  // bucket -> regret sums of halves 0 and 1
double avg_traverse_h(const Model& model, State state, Trail trail, int traverser, std::mt19937_64& rng, HMap* stats,
                      SubMap* sub = nullptr, const HSet* only = nullptr, BucketHalves* halves = nullptr, int half = 0,
                      std::array<HFull, 2> pk = {}) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const HFull key = model.hkey(state, actor, trail);
    const uint32_t b = model.hlookup(key), seen = b == kNone ? kUnseen : b;
    const auto sigma = model.bucket_policy(b, mask, true);
    const HFull prev_key = pk[actor];
    if (sub) pk[actor] = key;
    if (actor != traverser) {
        const Action a = sample_action(sigma, mask, rng);
        trail_step(trail, state, actor, a, seen);
        step(state, actor, a);
        return avg_traverse_h(model, std::move(state), trail, traverser, rng, stats, sub, only, halves, half, pk);
    }
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        Trail child_trail = trail;
        trail_step(child_trail, state, actor, static_cast<Action>(a), seen);
        step(child, actor, static_cast<Action>(a));
        v[a] = avg_traverse_h(model, std::move(child), child_trail, traverser, rng, stats, sub, only, halves, half, pk);
        node += sigma[a] * v[a];
    }
    AtomStats* st = stats ? &(*stats)[key] : nullptr;
    AtomStats* ss = nullptr;
    if (sub && prev_key.ctx != 0 && (!only || only->count(key))) {
        const SubKey sk{key, sub->id(prev_key)};
        if (const auto it = sub->stats.find(sk); it != sub->stats.end()) ss = &it->second;
        else if (sub->stats.size() < sub->cap) ss = &sub->stats[sk];
        else ++sub->dropped;
    }
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        const double r = v[a] - node;
        if (st) { st->r[a] += r; st->sq[a] += r * r; }
        if (ss) { ss->r[a] += r; ss->sq[a] += r * r; }
        if (halves && b != kNone) (*halves)[b][half * A + a] += r;
    }
    if (st) st->visits += 1;
    if (ss) ss->visits += 1;
    return node;
}

std::map<std::string, std::string> parse(int argc, char** argv, std::map<std::string, std::string> opt) {
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!opt.count(argv[i])) throw std::runtime_error(std::string("unknown option ") + argv[i]);
        opt[argv[i]] = argv[i + 1];
    }
    return opt;
}
Partition parse_partition(const std::string& s) {
    if (s == "context") return Partition::Context;
    if (s == "atom") return Partition::Atom;
    if (s == "aswiss") return Partition::Aswiss;
    if (s == "history") return Partition::History;
    throw std::runtime_error("--partition must be context, atom, aswiss or history");
}
std::unique_ptr<PowerAtlas> load_atlas(const std::string& path, size_t cache) {
    if (path.empty()) return nullptr;
    auto atlas = std::make_unique<PowerAtlas>(128);
    atlas->load(path);
    atlas->set_assignment_cache_limit(cache);
    return atlas;
}

double peak_rss_mb() {  // VmHWM of this process (Linux); -1 elsewhere
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
    return -1;
}

volatile std::sig_atomic_t g_stop = 0;
void request_stop(int) { g_stop = 1; }

uint64_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}
void put_string(std::ofstream& out, const std::string& s) { put(out, static_cast<uint64_t>(s.size())); out.write(s.data(), s.size()); }
std::string get_string(std::ifstream& in) { uint64_t n; get(in, n); std::string s(n, '\0'); in.read(s.data(), n); return s; }
template <class R> std::string rng_text(const R& r) { std::ostringstream o; o << r; return o.str(); }
template <class R> void rng_load(R& r, const std::string& text) { std::istringstream i(text); i >> r; }

// Loop counters, RNG states and output-file lengths stored with every checkpoint.
struct LoopState {
    uint64_t t = 0, k = 0, L = 0, epoch_start = 0, curve_roots = 0;
    double seconds = 0, curve_sum = 0, curve_sq = 0;
    std::string deal_rng, rng;
    uint64_t log_bytes = 0, index_bytes = 0, curve_bytes = 0;
};

// Checkpoint = config line + loop state + model state, written to checkpoint.tmp and renamed over checkpoint.bin.
// On load, checkpoint.bin is preferred; a complete checkpoint.tmp left by an interrupted rename is also accepted.
void write_checkpoint(const std::string& out, const std::string& config, const LoopState& s, const Model& model) {
    const std::string tmp = out + "/checkpoint.tmp", path = out + "/checkpoint.bin";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write("ASWCKPT1", 8);
        put_string(f, config);
        for (uint64_t v : {s.t, s.k, s.L, s.epoch_start, s.curve_roots, s.log_bytes, s.index_bytes, s.curve_bytes}) put(f, v);
        for (double v : {s.seconds, s.curve_sum, s.curve_sq}) put(f, v);
        put_string(f, s.deal_rng); put_string(f, s.rng);
        model.save_state(f);
        f.write("ASWEND01", 8);
        if (!f) throw std::runtime_error("failed to write " + tmp);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // Windows cannot rename over an existing file
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path);
    }
}
bool read_checkpoint(const std::string& file, const std::string& config, LoopState& s, Model& model) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (!f || std::string(magic, 8) != "ASWCKPT1") return false;
    const std::string stored = get_string(f);
    if (stored != config)
        throw std::runtime_error("checkpoint was written with different settings:\n  " + stored + "\nnow:\n  " + config);
    for (uint64_t* v : {&s.t, &s.k, &s.L, &s.epoch_start, &s.curve_roots, &s.log_bytes, &s.index_bytes, &s.curve_bytes}) get(f, *v);
    for (double* v : {&s.seconds, &s.curve_sum, &s.curve_sq}) get(f, *v);
    s.deal_rng = get_string(f); s.rng = get_string(f);
    model.load_state(f);
    f.read(magic, 8);
    return f && std::string(magic, 8) == "ASWEND01";
}

int train(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "aswiss"}, {"--atlas", ""}, {"--roots", "1000000"}, {"--epoch0", "10000"},
                                {"--z", "2"}, {"--seed", "1"}, {"--power-cache", "1000000"}, {"--out", ""},
                                {"--inherit", "0"}, {"--checkpoint-every", "1000000"}, {"--snapshot-every", "0"},
                                {"--curve-every", "10000"}, {"--h4", "heuristic"}, {"--label", "window"},
                                {"--avg-frac", "0.25"}, {"--history-cards", "street"}, {"--label-cap", "0"},
                                {"--backward", "0"}, {"--backward-min-visits", "32"},
                                {"--backward-max-subkeys", "4000000"}, {"--backward-test", "pair"},
                                {"--backward-z", ""}, {"--gate-eps", "0"}, {"--gate-stat", "avg"}});
    const Partition partition = parse_partition(o["--partition"]);
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    if (o["--h4"] != "heuristic" && o["--h4"] != "learn") throw std::runtime_error("--h4 must be heuristic or learn");
    const bool learn_h4 = o["--h4"] == "learn";
    const std::string label_rule = o["--label"];
    if (label_rule != "window" && label_rule != "avg" && label_rule != "window+avg")
        throw std::runtime_error("--label must be window, avg or window+avg");
    if (label_rule != "window" && ((partition != Partition::Aswiss && partition != Partition::History) || learn_h4))
        throw std::runtime_error("--label avg and window+avg need --partition aswiss or history and the heuristic H4");
    const uint64_t label_cap = std::stoull(o["--label-cap"]);  // 0: no cap
    const double avg_frac = std::stod(o["--avg-frac"]);
    if (o["--history-cards"] != "street" && o["--history-cards"] != "every")
        throw std::runtime_error("--history-cards must be street or every");
    if (partition == Partition::History && (learn_h4 || o["--inherit"] == "1"))
        throw std::runtime_error("--partition history supports neither --h4 learn nor --inherit 1");
    const bool backward = o["--backward"] == "1";
    if (o["--backward"] != "0" && o["--backward"] != "1") throw std::runtime_error("--backward must be 0 or 1");
    if (backward && (partition != Partition::History || label_rule != "avg"))
        throw std::runtime_error("--backward 1 needs --partition history --label avg");
    const double backward_min_visits = std::stod(o["--backward-min-visits"]);
    const size_t backward_max_subkeys = std::stoull(o["--backward-max-subkeys"]);
    if (o["--backward-test"] != "pair" && o["--backward-test"] != "labels") throw std::runtime_error("--backward-test must be pair or labels");
    const bool backward_pair_test = o["--backward-test"] == "pair";
    const double backward_z = o["--backward-z"].empty() ? -1.0 : std::stod(o["--backward-z"]);
    const double gate_eps = std::stod(o["--gate-eps"]);
    const bool gated = gate_eps > 0;
    if (o["--gate-stat"] != "avg" && o["--gate-stat"] != "iter") throw std::runtime_error("--gate-stat must be avg or iter");
    const bool gate_iter = o["--gate-stat"] == "iter";
    if (gated && (partition != Partition::History || label_rule != "avg"))
        throw std::runtime_error("--gate-eps needs --partition history --label avg");
    Model model(partition, atlas.get(), learn_h4);
    model.set_history_cards(o["--history-cards"] == "every");
    model.track_iterate_regret(std::stod(o["--gate-eps"]) > 0);
    const std::string out = o.at("--out");
    if (out.empty()) throw std::runtime_error("--out is required");
    std::filesystem::create_directories(out);
    const uint64_t T = std::stoull(o["--roots"]), epoch0 = std::stoull(o["--epoch0"]);
    const uint64_t checkpoint_every = std::stoull(o["--checkpoint-every"]), snapshot_every = std::stoull(o["--snapshot-every"]);
    const uint64_t curve_every = std::stoull(o["--curve-every"]);
    const double z = std::stod(o["--z"]);
    const bool inherit = o["--inherit"] == "1";
    // settings that must match on resume (--roots may grow; output cadences may change)
    std::ostringstream cfg;
    cfg << "partition=" << o["--partition"] << ";atlas=" << std::filesystem::path(o["--atlas"]).filename().string() << ":"
        << file_size_or_zero(o["--atlas"]) << ";epoch0=" << epoch0 << ";z=" << o["--z"] << ";inherit=" << o["--inherit"]
        << ";seed=" << o["--seed"] << (learn_h4 ? ";h4=learn" : "")
        << (label_rule != "window" ? ";label=" + label_rule + ";avg_frac=" + o["--avg-frac"] : "")
        << (partition == Partition::History ? ";history_cards=" + o["--history-cards"] : "")
        << (label_cap ? ";label_cap=" + o["--label-cap"] : "") << (backward ? ";backward=1;backward_min_visits=" + o["--backward-min-visits"] + ";backward_max_subkeys=" +
                          o["--backward-max-subkeys"] + ";backward_test=" + o["--backward-test"] +
                          (o["--backward-z"].empty() ? "" : ";backward_z=" + o["--backward-z"]) : "")
        << (gated ? ";gate_eps=" + o["--gate-eps"] + ";gate_log=2" + (gate_iter ? ";gate_stat=iter" : "") : "");
    const std::string config = cfg.str();

    std::mt19937_64 deal_rng(std::stoull(o["--seed"])), rng(std::stoull(o["--seed"]) ^ 0x9e3779b97f4a7c15ull);
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    const std::string backward_path = out + "/backward.csv", gate_path = out + "/gate.csv";
    LoopState s;
    s.L = epoch0;
    const bool resumed = read_checkpoint(out + "/checkpoint.bin", config, s, model) ||
                         read_checkpoint(out + "/checkpoint.tmp", config, s, model);
    if (resumed) {
        rng_load(deal_rng, s.deal_rng); rng_load(rng, s.rng);
        // drop rows written after the checkpoint; they are reproduced exactly by the replay
        std::filesystem::resize_file(log_path, s.log_bytes);
        std::filesystem::resize_file(index_path, s.index_bytes);
        std::filesystem::resize_file(curve_path, s.curve_bytes);
        for (const auto& [on, path] : {std::pair{backward, backward_path}, std::pair{gated, gate_path}}) {
            if (!on) continue;  // keep the rows of the epochs before the checkpoint; the replay writes the later ones again
            std::ifstream in(path);
            std::string line, kept;
            while (std::getline(in, line))
                if (line.rfind("epoch", 0) == 0 || std::stoull(line.substr(0, line.find(','))) < s.k) kept += line + "\n";
            in.close();
            std::ofstream(path) << kept;
        }
        std::cerr << "resumed at root " << s.t << " (epoch " << s.k << ")\n";
    } else {
        if (std::filesystem::exists(log_path))
            throw std::runtime_error(out + " already has log.csv but no usable checkpoint; use a new --out");
        std::ofstream(log_path) << "epoch,roots,contexts,buckets_after,buckets5,buckets6,buckets7,splits,atoms,L_per_root,"
                                   "C_per_root,H_per_root,seconds,power_cache_hits,power_cache_misses,bucket_records,"
                                   "bucket_mb,atom_window_mb,partition_mb,peak_rss_mb,h4_buckets_after,h4_splits\n";
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets,peak_rss_mb\n";
        if (gated) std::ofstream(gate_path) << "epoch,roots,buckets,rho_plain_half,rho_cross_half,iterate_regret_half,eps,judged\n";
        if (backward) std::ofstream(backward_path) << "epoch,roots,keys,keys_diagnosed,subkeys,subkey_visits_dropped,keys_two_labels,keys_inconsistent,pairs,"
                                                       "backward_splits,peak_rss_mb\n";
    }
    std::ofstream log(log_path, std::ios::app), index(index_path, std::ios::app), curve(curve_path, std::ios::app);
    // graceful stop: Ctrl+C / SIGTERM, or creating OUT/STOP (works for background processes on Windows too)
    const std::string stop_file = out + "/STOP";
    std::filesystem::remove(stop_file);
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef SIGBREAK
    std::signal(SIGBREAK, request_stop);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const double base_seconds = s.seconds;
    auto elapsed = [&] { return base_seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto live_total = [&] { const auto l = model.live_buckets(); return l[0] + l[1] + l[2]; };
    auto checkpoint = [&] {
        log.flush(); index.flush(); curve.flush();
        s.seconds = elapsed(); s.deal_rng = rng_text(deal_rng); s.rng = rng_text(rng);
        s.log_bytes = file_size_or_zero(log_path); s.index_bytes = file_size_or_zero(index_path);
        s.curve_bytes = file_size_or_zero(curve_path);
        write_checkpoint(out, config, s, model);
    };
    const bool track = partition == Partition::Aswiss || partition == Partition::History;
    uint64_t epoch_end = std::min(T, s.epoch_start + s.L);
    while (s.t < T) {
        ++s.t;
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        // --label-cap: window sums only over the epoch's last label_cap roots (bounds the window's memory)
        // history keys with --label avg: no window at all (it would only feed the L/C diagnostic and doubles the memory)
        const bool in_window = track && (!label_cap || epoch_end - s.t < label_cap) &&
                               !(partition == Partition::History && label_rule == "avg");
        for (int p = 0; p < 2; ++p) { model.traverse_root(deck, p, static_cast<double>(s.t), rng, in_window); model.apply_regrets(); }
        model.count_root();
        const double gap = model.take_traversal_gap() / 2;  // mean of the two seats, this root
        s.curve_sum += gap; s.curve_sq += gap * gap; ++s.curve_roots;
        if (curve_every && s.t % curve_every == 0) {
            const double n = s.curve_roots, mean = s.curve_sum / n;
            const double se = n > 1 ? std::sqrt(std::max(0.0, (s.curve_sq - n * mean * mean) / (n - 1) / n)) : 0.0;
            curve << s.t << ',' << elapsed() << ',' << s.curve_roots << ',' << mean << ',' << se << ',' << live_total() << ','
                  << peak_rss_mb() << '\n';
            s.curve_sum = s.curve_sq = 0; s.curve_roots = 0;
        }
        const bool epoch_done = s.t == epoch_end;
        if (snapshot_every && s.t % snapshot_every == 0 && !epoch_done) {
            const std::string file = "snapshot_r" + std::to_string(s.t) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",periodic," << s.k << ',' << live_total() << ',' << elapsed() << '\n';
        }
        if (epoch_done) {
            const std::string file = "snapshot_" + std::to_string(s.k) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",epoch," << s.k << ',' << live_total() << ',' << elapsed() << '\n';
            const auto w = model.window();
            const auto mb = model.table_megabytes();  // measured before the window is cleared
            int splits = 0;
            if (track && label_rule == "window") splits = model.split(z, inherit);
            else if (track && partition == Partition::History) {  // method B with bucket-history keys
                HMap avg;
                BucketHalves halves;
                const uint64_t eval_seed = std::stoull(o["--seed"]) * 1000003ull + s.k;
                std::mt19937_64 eval_rng(eval_seed);
                uint64_t n = std::max<uint64_t>(1000, static_cast<uint64_t>(avg_frac * static_cast<double>(s.t - s.epoch_start)));
                if (label_cap) n = std::min(n, label_cap);
                for (uint64_t i = 0; i < n; ++i) {
                    auto eval_deck = fresh_deck();
                    std::shuffle(eval_deck.begin(), eval_deck.end(), eval_rng);
                    for (int p = 0; p < 2; ++p)
                        avg_traverse_h(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), Trail{}, p, eval_rng, &avg,
                                       nullptr, nullptr, gated ? &halves : nullptr, static_cast<int>(i & 1));
                }
                // --gate-eps: judge only when the bucket regret total of the average policy, rho_P/2 per root, is at most
                // eps. rho_P = sum_B max(0, max_a r(B, a)) estimated from this pass: the plain estimate is biased upward
                // by the noise of tens of thousands of bucket sums, so the gate uses the cross-fitted one (each bucket's
                // best action chosen on one half of the roots, its regret measured on the other half, both ways).
                bool judge = true;
                if (gated) {
                    double plain = 0, cross = 0;
                    for (const auto& [b, h] : halves) {
                        double best = 0, m0 = 0, m1 = 0;
                        int a0 = -1, a1 = -1;
                        for (int a = 0; a < A; ++a) {
                            best = std::max(best, h[a] + h[A + a]);
                            if (h[a] > m0) { m0 = h[a]; a0 = a; }
                            if (h[A + a] > m1) { m1 = h[A + a]; a1 = a; }
                        }
                        plain += best;
                        if (a0 >= 0) cross += h[A + a0];
                        if (a1 >= 0) cross += h[a1];
                    }
                    plain /= 2.0 * static_cast<double>(n); cross /= 2.0 * static_cast<double>(n);
                    const double iter = model.iterate_regret_half(false);
                    judge = (gate_iter ? iter : cross) <= gate_eps;
                    if (judge) model.iterate_regret_half(true);  // the window restarts at every judgment
                    std::ofstream(gate_path, std::ios::app) << s.k << ',' << s.t << ',' << live_total() << ',' << plain << ',' << cross
                                                            << ',' << iter << ',' << gate_eps << ',' << judge << '\n';
                    std::cerr << "gate: rho_P/2 per root " << cross << " (plain " << plain << "), iterates' regret/2 " << iter
                              << " vs eps " << gate_eps << (judge ? ": judge\n" : ": no split this epoch\n");
                }
                // backward split: replay the same pass (same deals and samples, same frozen partition) and sum the
                // sub-keys of the keys visited at least --backward-min-visits times (rarer keys can hardly show two
                // significant sub-key labels), up to --backward-max-subkeys sub-keys
                SubMap sub;
                sub.cap = backward_max_subkeys;
                HSet frequent;
                if (backward && judge) {
                    for (const auto& [k, st] : avg) if (st.visits >= backward_min_visits) frequent.insert(k);
                    std::mt19937_64 replay_rng(eval_seed);
                    for (uint64_t i = 0; i < n; ++i) {
                        auto eval_deck = fresh_deck();
                        std::shuffle(eval_deck.begin(), eval_deck.end(), replay_rng);
                        for (int p = 0; p < 2; ++p)
                            avg_traverse_h(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), Trail{}, p, replay_rng,
                                           nullptr, &sub, &frequent);
                    }
                }
                if (judge) splits = model.hsplit(z, &avg, label_rule == "window+avg");
                std::cerr << "average-policy pass: " << n << " roots, " << avg.size() << " keys\n";
                if (backward && judge) {
                    const double rss = peak_rss_mb();  // includes the sub-key table, still alive here
                    const auto br = model.hbackward(z, sub, backward_pair_test, backward_z);
                    splits += br.splits;
                    std::ofstream(backward_path, std::ios::app) << s.k << ',' << s.t << ',' << avg.size() << ',' << frequent.size() << ','
                        << br.subkeys << ',' << sub.dropped << ',' << br.two_labels << ',' << br.inconsistent << ',' << br.pairs << ','
                        << br.splits << ',' << rss << '\n';
                    std::cerr << "backward split: " << br.subkeys << " sub-keys, " << br.inconsistent << " keys with two labels, "
                              << br.splits << " buckets split\n";
                }
            }
            else if (partition == Partition::Aswiss) {
                // method B: regrets of the frozen average policy from a separate sampled pass (deterministic per epoch)
                AtomMap avg;
                std::mt19937_64 eval_rng(std::stoull(o["--seed"]) * 1000003ull + s.k);
                uint64_t n = std::max<uint64_t>(1000, static_cast<uint64_t>(avg_frac * static_cast<double>(s.t - s.epoch_start)));
                if (label_cap) n = std::min(n, label_cap);
                for (uint64_t i = 0; i < n; ++i) {
                    auto eval_deck = fresh_deck();
                    std::shuffle(eval_deck.begin(), eval_deck.end(), eval_rng);
                    for (int p = 0; p < 2; ++p)
                        avg_traverse(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), p, eval_rng, avg);
                }
                splits = model.split(z, inherit, &avg, label_rule == "window+avg");
                std::cerr << "average-policy pass: " << n << " roots, " << avg.size() << " atoms\n";
            }
            model.clear_window();
            const auto live = model.live_buckets();
            const double len = std::min<double>(label_cap ? label_cap : s.t - s.epoch_start, s.t - s.epoch_start), secs = elapsed();
            const auto cache = atlas ? atlas->assignment_cache_stats() : PowerCacheStats{};
            log << s.k << ',' << s.t << ',' << model.contexts() << ',' << live[0] + live[1] + live[2] << ',' << live[0] << ','
                << live[1] << ',' << live[2] << ',' << splits << ',' << w.atoms << ',' << w.L / len << ',' << w.C / len << ','
                << (w.L - w.C) / len << ',' << secs << ',' << cache.hits << ',' << cache.misses << ','
                << model.bucket_records() << ',' << mb[0] << ',' << mb[1] << ',' << mb[2] << ',' << peak_rss_mb() << ','
                << model.h4_live() << ',' << model.last_h4_splits() << '\n';
            std::cerr << "epoch " << s.k << " roots=" << s.t << " buckets_after=" << live[0] + live[1] + live[2]
                      << " splits=" << splits << " h4_buckets=" << model.h4_live() << " atoms=" << w.atoms << " H/root=" << (w.L - w.C) / len << " (" << secs << "s)\n";
            s.epoch_start = s.t; ++s.k; s.L *= 2; epoch_end = std::min(T, s.t + s.L);
        }
        if (epoch_done || (checkpoint_every && s.t % checkpoint_every == 0) || s.t == T) checkpoint();
        if (s.t % 1000 == 0 && std::filesystem::exists(stop_file)) g_stop = 1;
        if (g_stop) {
            checkpoint();
            std::cerr << "stopped at root " << s.t << "; checkpoint saved, rerun the same command to resume\n";
            return 130;
        }
    }
    return 0;
}

int eval(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "aswiss"}, {"--atlas", ""}, {"--snapshot", ""}, {"--policy", "average"},
                                {"--pairs", "2000"}, {"--particles", "64"}, {"--gap-roots", "4000"},
                                {"--seed", "2026102701"}, {"--power-cache", "1000000"}, {"--pairs-csv", ""}});
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    Model model(parse_partition(o["--partition"]), atlas.get());
    model.load(o.at("--snapshot"));
    const bool average = o["--policy"] == "average";
    if (!average && o["--policy"] != "current") throw std::runtime_error("--policy must be average or current");
    const uint64_t seed = std::stoull(o["--seed"]);
    SnapshotPolicy target{model, average, std::mt19937_64(seed ^ 0x5851f42d4c957f2dull)};
    std::mt19937_64 deal_rng(seed);
    const int pairs = std::stoi(o["--pairs"]);
    std::vector<double> paired;
    const auto t0 = std::chrono::steady_clock::now();
    PolicyLBR<SnapshotPolicy> lbr(target, std::stoi(o["--particles"]), seed ^ 0x94d049bb133111ebull);
    // learned H4: the target plays its own discard/reveal and the LBR posterior uses its H4 likelihood; the LBR
    // attacker keeps the heuristic H4 (not a best response over its 12 initial actions), as in cpp_mccfr/h4_ablation
    H4PolicyLBR<SnapshotPolicy> h4_lbr(target, std::stoi(o["--particles"]), seed ^ 0x94d049bb133111ebull);
    LearnedH4 target_h4(model, average, seed ^ 0x6a09e667f3bcc909ull);
    for (int i = 0; i < pairs; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        std::array<double, 2> value{};
        for (int lbr_seat = 0; lbr_seat < 2; ++lbr_seat) {
            if (model.has_h4()) {
                target_h4.seat = 1 - lbr_seat;
                value[lbr_seat] = play_hand_h4_lbr(deck, lbr_seat, target, h4_lbr, kAnte, kEffectiveStackAnte, &target_h4);
            } else {
                value[lbr_seat] = play_hand_policy_lbr(deck, lbr_seat, target, lbr, kAnte, kEffectiveStackAnte);
            }
        }
        paired.push_back((value[0] + value[1]) / 2.0);
    }
    const double lbr_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto mean_se = [](const std::vector<double>& x) {
        const double m = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double v = 0;
        for (double y : x) v += (y - m) * (y - m);
        return std::make_pair(m, x.size() > 1 ? std::sqrt(v / (x.size() - 1) / x.size()) : 0.0);
    };
    if (!o["--pairs-csv"].empty()) {
        std::ofstream csv(o["--pairs-csv"]);
        csv << "pair,lbr_ante\n";
        for (size_t i = 0; i < paired.size(); ++i) csv << i << ',' << paired[i] << '\n';
    }
    // sampled BR-gap proxy on fixed deals (independent of the LBR deals)
    std::mt19937_64 gap_deals(seed ^ 0xd1b54a32d192ed03ull), gap_rng(seed ^ 0x2545f4914f6cdd1dull);
    std::vector<double> gaps;
    const int gap_roots = std::stoi(o["--gap-roots"]);
    for (int i = 0; i < gap_roots; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), gap_deals);
        State root;
        if (model.has_h4()) {  // both discard/reveal choices sampled from the evaluated H4 policy
            const auto cards = initial_cards(deck);
            std::array<std::pair<int, int>, 2> choices;
            for (int seat = 0; seat < 2; ++seat)
                choices[seat] = h4_choice(cards[seat], sample_h4(model.h4_strategy(seat, cards[seat], average), gap_rng));
            root = h4_root(deck, choices, kAnte);
        } else {
            root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        }
        double gap0 = 0, gap1 = 0;
        gap_traverse(model, average, root, 0, gap_rng, gap0);
        gap_traverse(model, average, root, 1, gap_rng, gap1);
        gaps.push_back((gap0 + gap1) / 2.0);
    }
    const auto [lm, lse] = mean_se(paired);
    const auto [gm, gse] = mean_se(gaps);
    uint64_t policy_queries = 0, policy_misses = 0;
    if (model.has_h4()) { const auto st = h4_lbr.stats(); policy_queries = st.policy_queries; policy_misses = st.policy_misses; }
    else { const auto st = lbr.stats(); policy_queries = st.policy_queries; policy_misses = st.policy_misses; }
    std::cout << std::setprecision(9) << "{\"snapshot\":\"" << json_escape(o["--snapshot"]) << "\",\"policy\":\"" << o["--policy"]
              << "\",\"pairs\":" << pairs << ",\"particles\":" << o["--particles"] << ",\"seed\":" << seed
              << ",\"lbr_ante\":" << lm << ",\"lbr_se\":" << lse << ",\"gap_roots\":" << gap_roots
              << ",\"gap_proxy_ante\":" << gm << ",\"gap_proxy_se\":" << gse << ",\"policy_queries\":" << policy_queries
              << ",\"policy_misses\":" << policy_misses << ",\"lbr_seconds\":" << lbr_seconds
              << ",\"h4\":\"" << (model.has_h4() ? "learned" : "heuristic") << "\",\"h4_buckets\":" << model.h4_live() << "}\n";
    return 0;
}

int self_test(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--atlas", ""}});
    const auto atlas = load_atlas(o.at("--atlas"), 1000000);
    std::mt19937_64 deal_rng(7), rng(8);
    Model aswiss(Partition::Aswiss, atlas.get()), atom(Partition::Atom, atlas.get()), context(Partition::Context, nullptr);
    NullAtlas null_atlas;
    for (uint64_t t = 1; t <= 3000; ++t) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const State root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        // atom keys are the engine's hard power keys; context keys equal them with the cluster cleared
        State s = root;
        while (!s.terminal) {
            const InfoKey engine = make_power_key(s, s.actor, *atlas);
            if (!(aswiss.atom_key(s, s.actor) == engine)) throw std::runtime_error("atom key differs from engine key");
            if (!(context.atom_key(s, s.actor) == context_of(engine))) throw std::runtime_error("context key mismatch");
            step(s, s.actor, sample_action(uniform_strategy(valid_mask(s, s.actor)), valid_mask(s, s.actor), rng));
        }
        for (Model* m : {&aswiss, &atom, &context})
            for (int p = 0; p < 2; ++p) { m->traverse(root, p, static_cast<double>(t), rng, m == &aswiss); m->apply_regrets(); }
        if (t % 1000 == 0) {
            const auto before = aswiss.live_buckets();
            aswiss.split(1.0);
            aswiss.clear_window();
            const auto after = aswiss.live_buckets();
            std::cerr << "split at " << t << ": " << before[0] + before[1] + before[2] << " -> " << after[0] + after[1] + after[2] << "\n";
        }
    }
    // snapshot round trip: identical policies on fresh states
    for (Model* m : {&aswiss, &atom, &context}) {
        const Partition part = m == &aswiss ? Partition::Aswiss : m == &atom ? Partition::Atom : Partition::Context;
        m->save("aswiss_stud_self_test.bin");
        Model loaded(part, part == Partition::Context ? nullptr : atlas.get());
        loaded.load("aswiss_stud_self_test.bin");
        for (int i = 0; i < 300; ++i) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            State s = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
            while (!s.terminal) {
                for (bool average : {true, false}) {
                    bool f1 = false, f2 = false;
                    if (m->strategy(s, s.actor, average, &f1) != loaded.strategy(s, s.actor, average, &f2) || f1 != f2)
                        throw std::runtime_error("snapshot round trip changed a policy");
                }
                const auto sigma = m->strategy(s, s.actor, true);
                step(s, s.actor, sample_action(sigma, valid_mask(s, s.actor), rng));
            }
        }
    }
    // bucket-history keys, after splits: the key replayed from the state alone equals the key carried by the
    // traversal trail; the root replay reproduces the state; snapshot round trip keeps every policy
    for (bool every : {false, true}) {
        Model hist(Partition::History, atlas.get());
        hist.set_history_cards(every);
        for (uint64_t t = 1; t <= 3000; ++t) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            for (int p = 0; p < 2; ++p) { hist.traverse_root(deck, p, static_cast<double>(t), rng, true); hist.apply_regrets(); }
            if (t % 500 == 0) { hist.split(1.0); hist.clear_window(); }
        }
        hist.save("aswiss_stud_self_test.bin");
        Model loaded(Partition::History, atlas.get());
        loaded.load("aswiss_stud_self_test.bin");
        if (loaded.history_cards_every() != every) throw std::runtime_error("history card mode lost in snapshot");
        int decisions = 0, misses = 0;
        for (int i = 0; i < 500; ++i) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            State s = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
            Trail trail;
            while (!s.terminal) {
                const int actor = s.actor;
                const HFull key = hist.hkey(s, actor, trail);
                if (!(hist.hkey_replay(s, actor) == key)) throw std::runtime_error("replayed history key differs from the trail key");
                State lbr_view = s;  // as PolicyLBR asks the target: the other seat's hidden cards cleared
                lbr_view.players[1 - actor].hidden.clear();
                if (!(hist.hkey_replay(lbr_view, actor) == key)) throw std::runtime_error("history key depends on hidden cards of the other seat");
                State r = root_of(s);
                for (const Event& e : s.history) { step(r, e.actor, e.action); sync_hidden(r, s); }
                for (int seat = 0; seat < 2; ++seat)
                    if (r.players[seat].hidden != s.players[seat].hidden || r.players[seat].shown != s.players[seat].shown ||
                        r.players[seat].invested != s.players[seat].invested || r.players[seat].round_bet != s.players[seat].round_bet)
                        throw std::runtime_error("root replay changed a player");
                if (r.pot != s.pot || r.street != s.street || r.actor != s.actor || r.highest_bet != s.highest_bet ||
                    r.raise_count != s.raise_count || r.history.size() != s.history.size())
                    throw std::runtime_error("root replay changed the betting state");
                for (bool average : {true, false}) {
                    bool f1 = false, f2 = false;
                    if (hist.strategy(s, actor, average, &f1) != loaded.strategy(s, actor, average, &f2) || f1 != f2)
                        throw std::runtime_error("history snapshot round trip changed a policy");
                    misses += !f1;
                }
                const uint8_t mask = valid_mask(s, actor);
                const Action a = sample_action(hist.strategy(s, actor, false), mask, rng);
                const uint32_t b = hist.hlookup(key);
                trail_step(trail, s, actor, a, b == kNone ? kUnseen : b);
                step(s, actor, a);
                ++decisions;
            }
        }
        const auto live = hist.live_buckets();
        std::cerr << "history (" << (every ? "cards every decision" : "cards per street") << "): " << decisions
                  << " decisions checked, " << misses << " unseen policy lookups, buckets " << live[0] + live[1] + live[2]
                  << ", contexts " << hist.contexts() << "\n";
    }
    // H4: the learned-H4 root with the heuristic's choices is the engine's 5th-street root
    auto same_state =[](const State& x, const State& y) {
        for (int seat = 0; seat < 2; ++seat) {
            const Player &a = x.players[seat], &b = y.players[seat];
            if (a.hidden != b.hidden || a.shown != b.shown || !(a.discarded == b.discarded) || a.invested != b.invested) return false;
        }
        return x.simulation_deck == y.simulation_deck && x.actor == y.actor && x.pot == y.pot && x.street == y.street;
    };
    for (int i = 0; i < 2000; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const auto cards = initial_cards(deck);
        const std::array<std::pair<int, int>, 2> heuristic{discard_reveal(cards[0]), discard_reveal(cards[1])};
        if (!same_state(h4_root(deck, heuristic, kAnte), sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte)))
            throw std::runtime_error("h4_root differs from sample_fifth_street_root");
        for (int a = 0; a < H; ++a) {  // every canonical action is a distinct legal (discard, reveal) pair
            const auto [d, r] = h4_choice(cards[0], a);
            if (d == r || d < 0 || d > 3 || r < 0 || r > 3) throw std::runtime_error("bad h4_choice");
        }
    }
    // H4 learning: snapshot round trip keeps H4 and betting policies; untrained H4 likelihood equals 24/12
    for (Partition part : {Partition::Aswiss, Partition::Atom}) {
        Model learner(part, atlas.get(), true);
        for (uint64_t t = 1; t <= 400; ++t) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            for (int p = 0; p < 2; ++p) { learner.traverse_root(deck, p, double(t), rng, part == Partition::Aswiss); learner.apply_regrets(); }
            if (t % 200 == 0 && part == Partition::Aswiss) { learner.split(1.0); learner.clear_window(); }
        }
        learner.save("aswiss_stud_self_test.bin");
        Model loaded(part, atlas.get());
        loaded.load("aswiss_stud_self_test.bin");
        if (!loaded.has_h4()) throw std::runtime_error("H4 section lost in snapshot");
        for (int i = 0; i < 300; ++i) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            const auto cards = initial_cards(deck);
            for (int seat = 0; seat < 2; ++seat)
                for (bool average : {true, false})
                    if (learner.h4_strategy(seat, cards[seat], average) != loaded.h4_strategy(seat, cards[seat], average))
                        throw std::runtime_error("H4 snapshot round trip changed a policy");
        }
        std::cerr << "H4 " << (part == Partition::Aswiss ? "aswiss" : "atom") << " buckets after 400 roots: " << learner.h4_live() << "\n";
    }
    {
        Model fresh_h4(Partition::Atom, atlas.get(), true);
        SnapshotPolicy probe{fresh_h4, true, std::mt19937_64(1)};
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const auto cards = initial_cards(deck);
        const auto [d, r] = h4_choice(cards[0], 5);
        std::array<Card, 2> hidden{};
        int k = 0;
        for (int i = 0; i < 4; ++i) if (i != d && i != r) hidden[k++] = cards[0][i];
        if (std::abs(probe.h4_weight(hidden, cards[0][r], cards[0][d], 0) - 2.0) > 1e-12)
            throw std::runtime_error("uniform H4 likelihood is not 24/12");
    }
    std::remove("aswiss_stud_self_test.bin");
    std::cerr << "self-test passed; buckets aswiss/atom/context = " << aswiss.live_buckets()[0] + aswiss.live_buckets()[1] + aswiss.live_buckets()[2]
              << "/" << atom.live_buckets()[0] + atom.live_buckets()[1] + atom.live_buckets()[2] << "/"
              << context.live_buckets()[0] + context.live_buckets()[1] + context.live_buckets()[2] << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "train") return train(argc, argv);
        if (mode == "eval") return eval(argc, argv);
        if (mode == "self-test") return self_test(argc, argv);
        throw std::runtime_error("usage: aswiss_stud train|eval|self-test [options]");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
