// ASWISS on heads-up seven-card stud (rules v3, 5th to 7th street, heuristic H4 discard/reveal).
//
// Build: g++ -O3 -std=c++17 aswiss_stud.cpp -o aswiss_stud
// Train: ./aswiss_stud train --partition aswiss|atom|context|history [--atlas ATLAS] --roots N [--epoch0 10000]
//                            [--z 2] [--inherit 0] [--seed 1] [--power-cache 1000000] [--checkpoint-every 1000000]
//                            [--snapshot-every 0] [--curve-every 10000] [--h4 heuristic|learn]
//                            [--label window|avg|window+avg] [--avg-frac 0.25] [--history-cards street|every]
//                            [--label-cap 0] [--backward 0] [--backward-min-visits 32] [--backward-max-subkeys 4000000]
//                            [--backward-test pair|labels] [--backward-z Z] [--gate-eps 0] [--gate-stat avg|iter]
//                            [--avg-fill none|plain] [--avg-fill-reach 1e-4] [--threads 1]
//                            --out DIR
//                            (rerun the same command to resume)
// Eval:  ./aswiss_stud eval [--atlas ATLAS] --snapshot FILE --policy average|current [--pairs 2000] [--rho-roots 0]
//                           [--particles 64] [--gap-roots 4000] [--seed 2026102701]
// Test:  ./aswiss_stud self-test --atlas ATLAS
// League: ./aswiss_stud league --agents FILE [--deals 20000] [--seed 2026110901] --out DIR  (duplicate head-to-head; see league())
// Bucket regret: ./aswiss_stud bucket-regret [--partition history|atom|aswiss|context] --atlas ATLAS --snapshot FILE [--roots 2000000]
//                (per-bucket regret of the average policy: web confidence, comparisons; see bucket_regret())
// H profile:     ./aswiss_stud h-profile --atlas ATLAS --snapshot FILE [--roots 300000]  (posterior vs card cancellation; see h_profile())
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
// --avg-fill plain (history keys, --label avg): CFR's average adds weight x own reach x strategy, so where the average
//            policy stopped going it stays at early iterations. The average-policy pass (labels, gate, backward sub-keys)
//            then uses, at the traverser's nodes whose own reach under the average policy along the path is below
//            --avg-fill-reach (1e-4), the plain average of that bucket's current strategies (linear in t, not weighted
//            by reach), kept lazily without an extra pass. The policy itself (snapshots, evaluation) is unchanged: those
//            nodes carry no realization weight. fill.csv logs the pass's traverser nodes and how many were filled.
// --threads K (--label avg|window+avg): the epoch-end average-policy pass and the backward replay run on K threads
//            (training itself stays serial). K > 1 draws each root from its own generator and adds 1024-root chunks
//            in root order, so every K > 1 gives the same bits; K = 1 is the serial pass (other deals). A run resumes
//            only with the same choice of 1 or > 1 (see chunked_pass).
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
#ifdef _WIN32  // process memory on Windows, declared by hand: <windows.h> macros would collide with the engine's names
#include <cstddef>
extern "C" {
struct AswMemoryCounters {  // layout of PROCESS_MEMORY_COUNTERS (DWORD = unsigned long, SIZE_T = size_t)
    unsigned long cb, page_faults;
    size_t peak_working_set, working_set, quota_peak_paged, quota_paged, quota_peak_nonpaged, quota_nonpaged, pagefile, peak_pagefile;
};
__declspec(dllimport) void* __stdcall GetCurrentProcess(void);
__declspec(dllimport) int __stdcall K32GetProcessMemoryInfo(void* process, AswMemoryCounters* counters, unsigned long cb);
struct AswFileTime { unsigned long low, high; };  // FILETIME
__declspec(dllimport) int __stdcall GetProcessTimes(void* process, AswFileTime* creation, AswFileTime* exit, AswFileTime* kernel,
                                                    AswFileTime* user);
}
#endif
#define STUD_MCCFR_NO_MAIN
#include "../../../cpp_mccfr/stud_mccfr.cpp"

#include <csignal>
#include <deque>
#include <memory>
#include <thread>
#include <filesystem>
#include <map>
#include <set>
#include <unordered_set>
#include <numeric>
#include <queue>

namespace {

#include "h4_lbr_generated.hpp"  // H4PolicyLBR, play_hand_h4_lbr (make_h4_lbr.py)

constexpr int A = kActionCount;
constexpr uint32_t kNone = 0xffffffffu;
constexpr int kAnte = 1000;
#include "online_lbr.hpp"  // --online-lbr-*: test-then-train Policy-LBR during training (every trainer)

struct NullAtlas final : PowerAtlas {
    uint16_t assign(const State&, int) const override { return 0; }
};

enum class Partition { Context, Atom, Aswiss, History, PP };

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

#include "pp_obs.hpp"   // --partition pp: observation streams, features, compact samples
#include "pp_tree.hpp"  // --partition pp: prior tree, posterior genealogy, prior fit

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
    bool ordered = false;     // keep `order` (a chunk of the parallel replay, merged in order of first appearance)
    std::vector<SubKey> order;  // sub-keys in order of first insertion
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
        if (partition != Partition::Context && partition != Partition::PP && !atlas)
            throw std::runtime_error("atom/aswiss/history partitions need --atlas");
        if (partition == Partition::PP) {
            if (learn_h4) throw std::runtime_error("--partition pp needs the heuristic H4");
            pp_ = std::make_unique<pp::Data>();
        }
    }
    // ---- --partition pp (pp_model.hpp) ----
    void pp_setup(const pp::Opts& o, uint64_t seed);
    const pp::Data* ppd() const { return pp_.get(); }
    pp::Data* ppd() { return pp_.get(); }
    uint32_t pp_locate(const State& s, int viewer, int32_t* leaf_out) const;
    std::array<double, A> pp_policy_at(const State& s, int seat, int32_t level) const;
    pp::Dir pp_direction(const pp::Sample& x, int32_t level, uint64_t seed) const;
    int pp_label(const std::array<double, A>& r, const std::array<double, A>& se, uint8_t mask) const;
    std::vector<pp::Dir> pp_directions(const std::vector<pp::Sample>& xs, int32_t level, uint64_t salt) const;
    static State pp_view_state(const pp::Sample& x);
    uint32_t pp_route_sum(uint32_t b, const std::vector<const pp::Dir*>& ds) const;
    uint32_t pp_route_root(const pp::Cell& cell, const std::vector<const pp::Sample*>& xs, const std::vector<const pp::Dir*>& now,
                           int32_t cur_level, uint64_t salt, uint64_t& steps);
    uint32_t pp_train_bucket(const State& s, int v, bool traverser);
    int pp_resolve(int32_t l, const pp::Rows& rows, std::vector<int>& idx, int32_t cur_level, uint64_t salt, size_t cap,
                   size_t ratio_target, uint64_t& steps, std::ostream* plog, uint64_t epoch);
    void pp_finalize_fill(int32_t l, bool at_judgment);
    bool pp_online_due(uint64_t t) const;
    double traverse_pp(State state, int traverser, double weight, std::mt19937_64& rng);
    template <class V> void pp_pass(State s, int traverser, std::mt19937_64& rng, V& visit) const;
    template <class KeyOf> std::map<uint64_t, pp::Reservoir> pp_collect(uint64_t n, uint64_t seed, size_t cap, int shard, int shards,
                                                                      KeyOf key_of) const;
    int pp_judge(uint64_t seed, uint64_t epoch, uint64_t roots, uint64_t epoch_len, uint64_t fill_steps, std::ostream& jlog,
                 std::ostream& plog);
    void pp_save(std::ofstream& out, bool full) const;
    void pp_load(std::ifstream& in, bool full);
    uint64_t pp_take_fill_steps() { const uint64_t v = pp_fill_steps_; pp_fill_steps_ = 0; return v; }
    bool has_h4() const { return h4_ != nullptr; }
    const H4Table* h4() const { return h4_.get(); }
    // --partition history: cards at every decision (true) or only at the first decision of each street (false)
    void set_history_cards(bool every) { hcards_every_ = every; }
    // Regret sums of the training iterates since the last judgment, per bucket, before the RM+ clipping (the sampled
    // counterfactual regrets MCCFR adds; unbiased for the iterates' regret). Kept only for --gate-eps, and checkpointed.
    void track_iterate_regret(bool on) { track_rwin_ = on; }
    // --avg-fill plain: plain sums of each bucket's current strategies (see the header)
    void enable_avg_fill(double reach) {
        avg_fill_ = true; fill_reach_ = reach;
        plain_.assign(buckets_.size(), {}); plain_last_.assign(buckets_.size(), plain_clock_);
    }
    void plain_tick(double w) { if (avg_fill_) plain_clock_ += w; }
    void plain_flush_all() { if (avg_fill_) for (uint32_t b = 0; b < plain_.size(); ++b) plain_flush(b); }
    bool fills(double own, uint32_t b) const { return avg_fill_ && b != kNone && own < fill_reach_; }
    std::array<double, A> plain_policy(uint32_t b, uint8_t mask) const { return normalized(plain_[b], mask); }
    // counted per thread (--threads: each label-pass thread takes its own counts)
    static void note_avg_node(bool filled) { ++avg_nodes_; if (filled) ++filled_nodes_; }
    static std::pair<uint64_t, uint64_t> take_fill_counts() {
        const auto r = std::make_pair(avg_nodes_, filled_nodes_); avg_nodes_ = filled_nodes_ = 0; return r;
    }
    void count_root() { if (track_rwin_) ++rwin_roots_; }
    // --log-splits: rows of the next hsplit / hbackward go to `log`, tagged with this epoch and root count
    void set_split_log(std::ostream* log, uint64_t epoch, uint64_t roots) { split_log_ = log; split_epoch_ = epoch; split_roots_ = roots; }
    uint64_t iterate_window_roots() const { return rwin_roots_; }
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
        if (partition_ == Partition::PP) return traverse_pp(sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), traverser, weight, rng);
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
        const uint32_t b = partition_ == Partition::PP ? pp_locate(state, viewer, nullptr)
                         : partition_ == Partition::History ? hlookup(hkey_replay(state, viewer)) : lookup(atom_key(state, viewer));
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
    // the same key computed from the state alone, by replaying the hand from its 5th-street root (evaluation);
    // `replayed` (serve) receives the replayed state, to check that it reproduces `state`
    HFull hkey_replay(const State& state, int viewer, State* replayed = nullptr) const {
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
        if (replayed) *replayed = r;
        return hkey(r, viewer, t);
    }
    // serve EXPLAIN (history partition): the key of `state` and how it resolves (the same walk as hresolve): the key
    // itself (member, hops 0), the key with its previous bucket replaced by an ancestor (member, hops > 0), or the
    // context's rest bucket (member false); no bucket when the structural context never occurred in training
    struct Resolution {
        HFull key;
        uint32_t bucket = kNone;
        int hops = 0;
        bool context = false, member = false;
        State replayed;
    };
    Resolution resolve_history(const State& state, int viewer) const {
        Resolution res;
        res.key = hkey_replay(state, viewer, &res.replayed);
        const auto it = hctx_.find(res.key.ctx);
        if (it == hctx_.end()) return res;
        res.context = true;
        HKey m = res.key.m;
        while (true) {
            const auto f = it->second.member.find(m);
            if (f != it->second.member.end()) { res.bucket = f->second; res.member = true; return res; }
            if (m.prev == kNone || m.prev == kUnseen || parent_[m.prev] == kNone) { res.bucket = it->second.rest; return res; }
            m.prev = parent_[m.prev]; ++res.hops;
        }
    }
    Partition partition() const { return partition_; }
    size_t bucket_count() const { return buckets_.size(); }
    size_t history_context_count() const { return hctx_.size(); }
    uint32_t parent_of(uint32_t b) const { return partition_ == Partition::History ? parent_[b] : kNone; }
    const Bucket& bucket_at(uint32_t b) const { return buckets_[b]; }
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
        struct Labels { int window = -1, average = -1; const AtomStats* wst = nullptr; const AtomStats* ast = nullptr; };
        std::unordered_map<HFull, Labels, HFullHash> labels_of;
        if (use_window) for (const auto& [k, st] : hatoms_) { auto& l = labels_of[k]; l.window = label(k, st); l.wst = &st; }
        if (avg) for (const auto& [k, st] : *avg) { auto& l = labels_of[k]; l.average = label(k, st); l.ast = &st; }
        // --log-splits: per child, the keys moved into it and the statistics behind their label (the average-policy
        // label's when there is one, else the window's): visits, sum of the label action's regret R and of its squares Q
        struct ChildLog { uint64_t keys = 0; double visits = 0, r = 0, q = 0; int window = -1, average = -1; };
        std::map<uint32_t, ChildLog> child_log;
        auto note_key = [&](uint32_t nb, uint16_t ctx, const HKey& m) {
            if (!split_log_) return;
            ChildLog& cl = child_log[nb];
            ++cl.keys;
            const auto it = labels_of.find(HFull{ctx, m});
            if (it == labels_of.end()) return;
            const Labels& l = it->second;
            cl.window = l.window; cl.average = l.average;
            const int a = l.average >= 0 ? l.average : l.window;
            const AtomStats* st = l.average >= 0 ? l.ast : l.wst ? l.wst : l.ast;
            if (!st) return;
            cl.visits += st->visits;
            if (a >= 0) { cl.r += st->r[a]; cl.q += st->sq[a]; }
        };
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
                note_key(c.member[m], ctx, m);
            }
            for (const auto& [m, b] : implicit) {  // unlabelled ones keep following their ancestor entry or the rest
                const auto it = child.find(b);
                const int label = label_of.at(m);
                if (it != child.end() && label >= 0) { c.member[m] = it->second.at(label); note_key(c.member[m], ctx, m); }
            }
            const auto rest = child.find(c.rest);
            if (rest != child.end()) c.rest = rest->second.at(-1);
            if (split_log_)
                for (const auto& [b, by_label] : child)
                    for (const auto& [code, nb] : by_label) {
                        const ChildLog cl = child_log.count(nb) ? child_log[nb] : ChildLog{};
                        const int w = !avg ? code : use_window && code >= 0 ? code / 16 - 1 : (avg && !use_window) ? -1 : cl.window;
                        const int g = !avg ? -1 : !use_window ? code : code >= 0 ? code % 16 - 1 : -1;
                        *split_log_ << split_epoch_ << ',' << split_roots_ << ",forward," << (ctx >> 8) << ',' << (ctx & 0xff) << ','
                                    << b << ',' << nb << ',' << w << ',' << g << ",-1," << cl.keys << ',' << cl.visits << ',' << cl.r << ','
                                    << cl.q << ',' << (cl.q > 0 ? cl.r / std::sqrt(cl.q) : 0.0) << '\n';
                    }
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
                    if (avg_fill_) { plain_[nb] = plain_[b]; plain_last_[nb] = plain_last_[b]; }
                    return nb;
                };
                for (const auto& [sig, keys] : sigs) {
                    const uint32_t nb = child();
                    for (const HKey& m : keys) c.member[m] = nb;
                    if (split_log_)  // label_bits: the labels the sub-keys of these previous keys carried
                        *split_log_ << split_epoch_ << ',' << split_roots_ << ",backward," << (ctx >> 8) << ',' << (ctx & 0xff) << ','
                                    << b << ',' << nb << ",-1,-1," << sig[0] << ',' << keys.size() << ",0,0,0,0\n";
                }
                if (others) {
                    const uint32_t nb = child();
                    uint64_t moved = 0;
                    for (const HKey& m : ex) if (!marked.count(m)) { c.member[m] = nb; ++moved; }
                    if (c.rest == b) c.rest = nb;
                    if (split_log_)
                        *split_log_ << split_epoch_ << ',' << split_roots_ << ",backward," << (ctx >> 8) << ',' << (ctx & 0xff) << ','
                                    << b << ',' << nb << ",-1,-1,0," << moved << ",0,0,0,0\n";
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
            if (avg_fill_) plain_flush(b);  // the strategy that held since its last change, before it changes
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
        if (partition_ == Partition::PP) { pp_save(out, true); return; }
        if (partition_ == Partition::History) {
            put(out, static_cast<uint8_t>(hcards_every_));
            put_history(out);
            put(out, static_cast<uint64_t>(hatoms_.size()));
            for (const auto& [k, st] : hatoms_) { put(out, k.ctx); put_hkey(out, k.m); put(out, st.r); put(out, st.sq); put(out, st.visits); }
            if (track_rwin_) {
                put(out, rwin_roots_); put(out, static_cast<uint64_t>(rwin_.size()));
                for (const auto& r : rwin_) put(out, r);
            }
            if (avg_fill_) {
                put(out, plain_clock_); put(out, static_cast<uint64_t>(plain_.size()));
                for (size_t b = 0; b < plain_.size(); ++b) { put(out, plain_[b]); put(out, plain_last_[b]); }
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
        if (partition_ == Partition::PP) { pp_load(in, true); touched_.clear(); return; }
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
            if (avg_fill_) {
                uint64_t m; get(in, plain_clock_); get(in, m);
                plain_.assign(m, {}); plain_last_.assign(m, 0.0);
                for (size_t b = 0; b < m; ++b) { get(in, plain_[b]); get(in, plain_last_[b]); }
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
        if (partition_ == Partition::PP) {  // the buckets the live leaves point to (provisional pointers of pending leaves)
            for (const auto& L : pp_->leaves) if (L.node >= 0) live[(L.ctx >> 8) - 5].insert(pp_->current(static_cast<int32_t>(&L - pp_->leaves.data())));
            return {live[0].size(), live[1].size(), live[2].size()};
        }
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
    size_t contexts() const { return partition_ == Partition::PP ? pp_->cells.size() : partition_ == Partition::History ? hctx_.size() : contexts_.size(); }
    size_t bucket_records() const { return buckets_.size(); }  // live + retired (split parents are never freed)
    // approximate bytes of the training tables: bucket records, the atom window and the partition maps
    std::array<double, 3> table_megabytes() const {
        if (partition_ == Partition::PP) {  // bucket tables; prior tree, leaves and genealogy; stored average tables
            const pp::Data& D = *pp_;
            double tables = 0;
            for (const auto& [lv, t] : D.tables) tables += static_cast<double>(t.capacity() * sizeof(t[0]));
            double fill = 0;
            for (const auto& L : D.leaves) fill += static_cast<double>(L.fill.capacity() * sizeof(pp::FillRow));
            constexpr double node = 2 * sizeof(void*);
            return {buckets_.capacity() * sizeof(Bucket) / 1e6,
                    (D.nodes.capacity() * sizeof(pp::TNode) + D.leaves.capacity() * sizeof(pp::Leaf) + fill +
                     D.splits.size() * (sizeof(uint32_t) + sizeof(pp::Split) + node) + (D.born.capacity() + parent_.capacity()) * 4.0 +
                     D.cells.capacity() * sizeof(pp::Cell)) / 1e6,
                    tables / 1e6};
        }
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
        if (partition_ == Partition::PP) { pp_save(out, false); return; }
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
        if (partition_ == Partition::PP) {
            pp_load(in, false);
            if (!in) throw std::runtime_error("truncated snapshot: " + path);
            return;
        }
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
        if (avg_fill_) { plain_.emplace_back(); plain_last_.push_back(plain_clock_); }
        if (partition_ == Partition::History) parent_.push_back(parent);
        if (partition_ == Partition::PP) { parent_.push_back(parent); pp_->born.push_back(pp_->level + (parent != kNone ? 1 : 0)); }
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
    std::ostream* split_log_ = nullptr;       // --log-splits (history partition): splits.csv
    uint64_t split_epoch_ = 0, split_roots_ = 0;
    bool avg_fill_ = false;
    double fill_reach_ = 1e-4, plain_clock_ = 0;
    std::vector<std::array<double, A>> plain_;
    std::vector<double> plain_last_;
    static inline thread_local uint64_t avg_nodes_ = 0, filled_nodes_ = 0;
    void plain_flush(uint32_t b) {
        const double dw = plain_clock_ - plain_last_[b];
        if (dw <= 0) return;
        const auto sigma = rm_plus(buckets_[b].q, static_cast<uint8_t>((1u << A) - 1));  // unused actions keep q = 0
        for (int a = 0; a < A; ++a) plain_[b][a] += dw * sigma[a];
        plain_last_[b] = plain_clock_;
    }
    std::vector<std::array<double, A>> rwin_;  // see track_iterate_regret
    uint64_t rwin_roots_ = 0;
    HMap hatoms_;
    // --partition pp
    std::unique_ptr<pp::Data> pp_;
    uint64_t pp_seed_ = 0, pp_root_now_ = 0, pp_epoch_ = 0, pp_fill_steps_ = 0;
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
// bucket -> regret sums of halves 0 and 1 (entries 0..2A-1), then the traverser visits of each half (2A, 2A+1)
using BucketHalves = std::unordered_map<uint32_t, std::array<double, 2 * A + 2>>;
// rho_P/2 per root from n roots split into two halves: {plain, cross-fitted}. plain = sum_B max(0, max_a R_B(a)) is
// biased upward by sampling noise; cross-fitted takes each bucket's best action on one half and its regret on the
// other, both ways (no selection bias; slightly low).
std::pair<double, double> rho_estimates(const BucketHalves& halves, uint64_t n) {
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
    return {plain / (2.0 * static_cast<double>(n)), cross / (2.0 * static_cast<double>(n))};
}
double avg_traverse_h(const Model& model, State state, Trail trail, int traverser, std::mt19937_64& rng, HMap* stats,
                      SubMap* sub = nullptr, const HSet* only = nullptr, BucketHalves* halves = nullptr, int half = 0,
                      std::array<HFull, 2> pk = {}, double own = 1.0) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const HFull key = model.hkey(state, actor, trail);
    const uint32_t b = model.hlookup(key), seen = b == kNone ? kUnseen : b;
    const auto avg = model.bucket_policy(b, mask, true);
    const HFull prev_key = pk[actor];
    if (sub) pk[actor] = key;
    if (actor != traverser) {
        const Action a = sample_action(avg, mask, rng);
        trail_step(trail, state, actor, a, seen);
        step(state, actor, a);
        return avg_traverse_h(model, std::move(state), trail, traverser, rng, stats, sub, only, halves, half, pk, own);
    }
    // own: the traverser's own reach of this node under the average policy (--avg-fill: below the threshold, its
    // average stayed at early iterations, so the bucket's plain average of current strategies values this node)
    const bool filled = model.fills(own, b);
    model.note_avg_node(filled);
    const auto sigma = filled ? model.plain_policy(b, mask) : avg;
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        Trail child_trail = trail;
        trail_step(child_trail, state, actor, static_cast<Action>(a), seen);
        step(child, actor, static_cast<Action>(a));
        v[a] = avg_traverse_h(model, std::move(child), child_trail, traverser, rng, stats, sub, only, halves, half, pk, own * avg[a]);
        node += sigma[a] * v[a];
    }
    AtomStats* st = stats ? &(*stats)[key] : nullptr;
    AtomStats* ss = nullptr;
    if (sub && prev_key.ctx != 0 && (!only || only->count(key))) {
        const SubKey sk{key, sub->id(prev_key)};
        if (const auto it = sub->stats.find(sk); it != sub->stats.end()) ss = &it->second;
        else if (sub->stats.size() < sub->cap) {
            ss = &sub->stats[sk];
            if (sub->ordered) sub->order.push_back(sk);
        }
        else ++sub->dropped;
    }
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        const double r = v[a] - node;
        if (st) { st->r[a] += r; st->sq[a] += r * r; }
        if (ss) { ss->r[a] += r; ss->sq[a] += r * r; }
        if (halves && b != kNone) (*halves)[b][half * A + a] += r;
    }
    if (halves && b != kNone) (*halves)[b][2 * A + half] += 1;
    if (st) st->visits += 1;
    if (ss) ss->visits += 1;
    return node;
}

// --threads K > 1: the average-policy pass and its backward replay run in chunks of kLabelChunk roots, K chunks at a
// time. Root i draws its deal and samples from its own generator (label_root_rng), each chunk sums into fresh tables
// of its own, and after every round the chunk tables are added to the totals in root order (per-key totals on K
// threads, each owning every K-th of kLabelShards key shards). The totals, and so the splits, are therefore the same
// bits for every K > 1. --threads 1 keeps the serial pass, whose single generator draws other deals. Meanwhile the
// atlas's shared cache is only read; each thread memoizes its misses in its own overlay (PowerAtlas::overlay), and
// the overlays join the shared cache after the pass. The cache only saves time, but its hit and miss counts in
// log.csv can differ with K (which entries a full cache drops depends on the order they join).
constexpr uint64_t kLabelChunk = 1024;
constexpr size_t kLabelShards = 64;

std::mt19937_64 label_root_rng(uint64_t eval_seed, uint64_t i) {
    std::seed_seq seq{static_cast<uint32_t>(eval_seed), static_cast<uint32_t>(eval_seed >> 32), static_cast<uint32_t>(i),
                      static_cast<uint32_t>(i >> 32)};
    return std::mt19937_64(seq);
}
// the deal of root i of a chunked pass, with the generator that then samples its traversals
State label_root(uint64_t eval_seed, uint64_t i, std::mt19937_64& rng) {
    rng = label_root_rng(eval_seed, i);
    auto deck = fresh_deck();
    std::shuffle(deck.begin(), deck.end(), rng);
    return sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
}

// f(t) for t < m on m threads (t = 0 on the calling thread); rethrows a worker's exception after all have joined
void run_threads(int m, const std::function<void(int)>& f) {
    std::exception_ptr error;
    std::mutex lock;
    const auto guarded = [&](int t) {
        try { f(t); }
        catch (...) { std::lock_guard<std::mutex> g(lock); if (!error) error = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < m; ++t) pool.emplace_back(guarded, t);
    guarded(0);
    for (auto& th : pool) th.join();
    if (error) std::rethrow_exception(error);
}

// Runs chunk(c, t) for the chunks c of an n-root pass, `threads` at a time with t the slot of c in its round, each
// thread with its own atlas overlay; merge(c0, m) follows every round of the m chunks from c0.
void chunked_pass(uint64_t n, int threads, PowerAtlas* atlas, const std::function<void(uint64_t, int)>& chunk,
                  const std::function<void(uint64_t, int)>& merge) {
    std::vector<PowerAtlas::AssignOverlay> overlay(threads);
    if (atlas)  // together no larger than the shared cache (--power-cache)
        for (auto& o : overlay) o.limit = std::max<size_t>(1, atlas->assignment_cache_stats().limit / threads);
    const uint64_t chunks = (n + kLabelChunk - 1) / kLabelChunk;
    for (uint64_t c0 = 0; c0 < chunks; c0 += threads) {
        const int m = static_cast<int>(std::min<uint64_t>(threads, chunks - c0));
        run_threads(m, [&](int t) {
            struct Use {  // the overlay is set only while this thread runs its chunk
                explicit Use(PowerAtlas::AssignOverlay* o) { PowerAtlas::overlay = o; }
                ~Use() { PowerAtlas::overlay = nullptr; }
            } use(&overlay[t]);
            chunk(c0 + t, t);
        });
        merge(c0, m);
    }
    // the pass's assignments join the shared cache, as a serial pass's would (the backward replay reuses them)
    if (atlas) for (auto& o : overlay) atlas->absorb(o);
}

void add_stats(AtomStats& to, const AtomStats& from) {
    for (int a = 0; a < A; ++a) { to.r[a] += from.r[a]; to.sq[a] += from.sq[a]; }
    to.visits += from.visits;
}
template <class Map>
size_t label_shard(const typename Map::key_type& k) {
    return static_cast<size_t>((static_cast<uint64_t>(typename Map::hasher()(k)) * 0x9e3779b97f4a7c15ull) >> 58) % kLabelShards;
}
// adds the chunk tables (in root order) to the sharded per-key totals on `threads` threads
template <class Map>
void merge_chunk_stats(std::vector<Map>& shards, const std::vector<Map>& chunks, int m, int threads) {
    run_threads(threads, [&](int t) {
        for (int c = 0; c < m; ++c)
            for (const auto& [k, st] : chunks[c]) {
                const size_t s = label_shard<Map>(k);
                if (static_cast<int>(s % threads) == t) add_stats(shards[s][k], st);
            }
    });
}
template <class Map>
Map join_shards(std::vector<Map>& shards) {
    size_t total = 0;
    for (const auto& s : shards) total += s.size();
    Map out;
    out.reserve(total);
    for (auto& s : shards) { out.merge(s); Map().swap(s); }  // keys are disjoint across shards: every node moves
    return out;
}
// Adds a chunk of the parallel backward replay to the totals as the serial replay would have: previous keys take ids
// in order of first appearance; new sub-keys enter in order of first appearance while fewer than `cap` are held, and
// the visits of the others count as dropped (a sub-key the chunk dropped was dropped by the serial replay too).
void merge_sub(SubMap& to, const SubMap& from) {
    std::vector<uint32_t> id(from.prev.size());
    for (size_t j = 0; j < from.prev.size(); ++j) id[j] = to.id(from.prev[j]);
    for (const SubKey& sk : from.order) {
        const AtomStats& st = from.stats.at(sk);
        const SubKey g{sk.k, id[sk.p]};
        if (const auto it = to.stats.find(g); it != to.stats.end()) add_stats(it->second, st);
        else if (to.stats.size() < to.cap) to.stats.emplace(g, st);
        else to.dropped += static_cast<uint64_t>(st.visits);
    }
    to.dropped += from.dropped;
}
void merge_halves(BucketHalves& to, const BucketHalves& from) {
    for (const auto& [b, h] : from) {
        auto& t = to[b];
        for (int j = 0; j < 2 * A + 2; ++j) t[j] += h[j];
    }
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
    if (s == "pp") return Partition::PP;
    throw std::runtime_error("--partition must be context, atom, aswiss, history or pp");
}
std::unique_ptr<PowerAtlas> load_atlas(const std::string& path, size_t cache) {
    if (path.empty()) return nullptr;
    auto atlas = std::make_unique<PowerAtlas>(128);
    atlas->load(path);
    atlas->set_assignment_cache_limit(cache);
    return atlas;
}

// Process memory in MB (-1 if unknown): peak and current resident memory (Linux VmHWM / VmRSS, Windows peak / current
// working set) and committed private memory (Linux VmData, Windows private bytes): memory_mb(0|1|2).
double memory_mb(int which) {
#ifdef _WIN32
    AswMemoryCounters c{};
    c.cb = sizeof(c);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c))) return -1;
    const size_t v = which == 0 ? c.peak_working_set : which == 1 ? c.working_set : c.pagefile;
    return static_cast<double>(v) / (1024.0 * 1024.0);
#else
    static const char* const field[3] = {"VmHWM:", "VmRSS:", "VmData:"};
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind(field[which], 0) == 0) return std::stod(line.substr(std::strlen(field[which]))) / 1024.0;
    return -1;
#endif
}
double peak_rss_mb() { return memory_mb(0); }
// CPU seconds of this process (all threads, user + kernel), -1 if unknown
double cpu_seconds() {
#ifdef _WIN32
    AswFileTime c{}, e{}, k{}, u{};
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return -1;
    auto ticks = [](const AswFileTime& f) { return static_cast<double>((static_cast<uint64_t>(f.high) << 32) | f.low); };
    return (ticks(k) + ticks(u)) * 1e-7;
#else
    timespec ts{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return -1;
    return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
#endif
}
// the resource columns of curve.csv (kResourceHeader): memory (see memory_mb), process CPU seconds and the wall-clock
// time as Unix seconds (shows sleep / stalls and when other runs started or ended)
constexpr const char* kResourceHeader = "peak_rss_mb,rss_mb,commit_mb,cpu_seconds,unix_time";
std::string memory_columns() {
    std::ostringstream m;
    m << memory_mb(0) << ',' << memory_mb(1) << ',' << memory_mb(2) << ',' << cpu_seconds() << ','
      << std::fixed << std::setprecision(3)
      << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    return m.str();
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
void write_checkpoint(const std::string& out, const std::string& config, const LoopState& s, const Model& model,
                      OnlineLbr* online = nullptr, const uint64_t* regret_bytes = nullptr) {
    const std::string tmp = out + "/checkpoint.tmp", path = out + "/checkpoint.bin";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write("ASWCKPT1", 8);
        put_string(f, config);
        for (uint64_t v : {s.t, s.k, s.L, s.epoch_start, s.curve_roots, s.log_bytes, s.index_bytes, s.curve_bytes}) put(f, v);
        for (double v : {s.seconds, s.curve_sum, s.curve_sq}) put(f, v);
        put_string(f, s.deal_rng); put_string(f, s.rng);
        model.save_state(f);
        if (online && online->on()) online->save(f);  // only with --online-lbr-every (part of the config line)
        if (regret_bytes) put(f, *regret_bytes);       // only with --log-regret (part of the config line)
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
bool read_checkpoint(const std::string& file, const std::string& config, LoopState& s, Model& model, OnlineLbr* online = nullptr,
                     uint64_t* online_bytes = nullptr, uint64_t* regret_bytes = nullptr) {
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
    if (online && online->on()) *online_bytes = online->load(f);
    if (regret_bytes) get(f, *regret_bytes);
    f.read(magic, 8);
    return f && std::string(magic, 8) == "ASWEND01";
}

#include "pp_model.hpp"  // --partition pp: the Model members (routing, particle directions, judgment, files)

namespace sota {  // SOTA neural baselines (sota/stud7_net_main.hpp, included before main)
int train_net(int argc, char** argv);
int eval_net(std::map<std::string, std::string>& o);
int train_rebel(int argc, char** argv);  // sota/rebel_main.hpp
int eval_rebel(std::map<std::string, std::string>& o);
}

int train(int argc, char** argv) {
    for (int i = 2; i + 1 < argc; i += 2)  // --partition net: the neural baselines have their own trainer
        if (std::string(argv[i]) == "--partition" && std::string(argv[i + 1]) == "net") return sota::train_net(argc, argv);
    for (int i = 2; i + 1 < argc; i += 2)  // --partition rebel: scaled-down ReBeL (search + value networks)
        if (std::string(argv[i]) == "--partition" && std::string(argv[i + 1]) == "rebel") return sota::train_rebel(argc, argv);
    auto o = parse(argc, argv, {{"--partition", "aswiss"}, {"--atlas", ""}, {"--roots", "1000000"}, {"--epoch0", ""},
                                {"--z", "2"}, {"--seed", "1"}, {"--power-cache", "1000000"}, {"--out", ""},
                                {"--inherit", "0"}, {"--checkpoint-every", "1000000"}, {"--snapshot-every", "0"},
                                {"--curve-every", "10000"}, {"--h4", "heuristic"}, {"--label", "window"},
                                {"--avg-frac", "0.25"}, {"--history-cards", "street"}, {"--label-cap", "0"},
                                {"--backward", "0"}, {"--backward-min-visits", "32"},
                                {"--backward-max-subkeys", "4000000"}, {"--backward-test", "pair"},
                                {"--backward-z", ""}, {"--gate-eps", "0"}, {"--gate-stat", "avg"},
                                {"--avg-fill", "none"}, {"--avg-fill-reach", "1e-4"}, {"--threads", "1"},
                                {"--online-lbr-every", "0"}, {"--online-lbr-particles", "16"}, {"--online-lbr-policy", ""},
                                {"--log-regret", "0"}, {"--log-splits", "0"}, {"--budget-cpu-seconds", "0"}, {"--freeze-splits-at", "0"},
                                {"--judge-at", "0"},
                                // --partition pp (PP_STUD7_DESIGN_KO.md; see pp_model.hpp)
                                {"--pp-assign", "now"}, {"--leaf-start", "root"}, {"--pp-particles", "64"},
                                {"--pp-samples", "64"}, {"--pp-min-samples", "8"}, {"--pp-m", "16"}, {"--pp-rows", "64"},
                                {"--pp-pass-frac", "0.125"}, {"--pp-budget", "0.25"}, {"--pp-budget-min-steps", "2000000"},
                                {"--judge-mem-mb", "256"}, {"--prior-ratio", "4"}, {"--prior-criterion", "label"},
                                {"--crossfit", "0"}, {"--min-leaf", "1"}, {"--features", "general"}, {"--mem-budget-mb", "0"},
                                {"--online-lbr-schedule", "fixed"}, {"--inherit-regret", "0"}, {"--inherit-avg", "0"}});
    const Partition partition = parse_partition(o["--partition"]);
    if (o["--epoch0"].empty()) o["--epoch0"] = partition == Partition::PP ? "1000" : "10000";  // the spec's 1000 * 2^k for pp
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
    if (o["--avg-fill"] != "none" && o["--avg-fill"] != "plain") throw std::runtime_error("--avg-fill must be none or plain");
    const bool avg_fill = o["--avg-fill"] == "plain";
    if (avg_fill && (partition != Partition::History || label_rule != "avg"))
        throw std::runtime_error("--avg-fill plain needs --partition history --label avg");
    const int threads = std::stoi(o["--threads"]);
    if (threads < 1 || threads > 256) throw std::runtime_error("--threads must be 1 to 256");
    const bool label_pass = label_rule != "window";  // the epoch-end average-policy pass, the only part --threads runs in parallel
    if (threads > 1 && !label_pass) std::cerr << "note: --threads only affects --label avg|window+avg; this run is serial\n";
    Model model(partition, atlas.get(), learn_h4);
    model.set_history_cards(o["--history-cards"] == "every");
    // --partition pp: settings (all part of the checkpoint configuration below)
    pp::Opts ppo;
    const bool pp_mode = partition == Partition::PP;
    if (pp_mode) {
        auto choice = [&](const char* k, std::initializer_list<const char*> ok) {
            for (const char* v : ok) if (o[k] == v) return;
            throw std::runtime_error(std::string(k) + " has an unknown value: " + o[k]);
        };
        choice("--pp-assign", {"now", "fill"}); choice("--leaf-start", {"parent", "root"});
        choice("--prior-criterion", {"label", "regret"}); choice("--crossfit", {"0", "1"}); choice("--features", {"general", "poker"});
        choice("--online-lbr-schedule", {"fixed", "split"});
        if (label_rule != "window" || o["--inherit"] != "0") throw std::runtime_error("--partition pp takes neither --label nor --inherit");
        ppo.fill = o["--pp-assign"] == "fill"; ppo.root_start = o["--leaf-start"] == "root";
        ppo.poker = o["--features"] == "poker"; ppo.label_criterion = o["--prior-criterion"] == "label"; ppo.crossfit = o["--crossfit"] == "1";
        ppo.particles = std::stoi(o["--pp-particles"]); ppo.n0 = std::stoi(o["--pp-samples"]); ppo.n_min = std::stoi(o["--pp-min-samples"]);
        ppo.m = std::stoi(o["--pp-m"]); ppo.rho = std::stoi(o["--pp-rows"]); ppo.min_leaf = std::stoi(o["--min-leaf"]);
        ppo.z = std::stod(o["--z"]); ppo.ratio = std::stod(o["--prior-ratio"]); ppo.pass_frac = std::stod(o["--pp-pass-frac"]);
        ppo.budget = std::stod(o["--pp-budget"]); ppo.budget_min_steps = std::stod(o["--pp-budget-min-steps"]);
        ppo.judge_mem_mb = std::stod(o["--judge-mem-mb"]); ppo.mem_budget_mb = std::stod(o["--mem-budget-mb"]);
        ppo.threads = threads; ppo.online_split = o["--online-lbr-schedule"] == "split";
        ppo.inherit_regret = std::stod(o["--inherit-regret"]); ppo.inherit_avg = std::stod(o["--inherit-avg"]);
        if (ppo.inherit_regret < 0 || ppo.inherit_regret > 1 || ppo.inherit_avg < 0 || ppo.inherit_avg > 1)
            throw std::runtime_error("--inherit-regret and --inherit-avg must be in [0, 1] (spec: a child's start may not exceed its parent)");
        if (ppo.particles < 1 || ppo.n0 < 2 || ppo.n_min < 2 || ppo.m < 1 || ppo.rho < ppo.m || ppo.min_leaf < 1)
            throw std::runtime_error("--pp-particles >= 1, --pp-samples >= 2, --pp-min-samples >= 2, --pp-rows >= --pp-m >= 1, --min-leaf >= 1");
        model.pp_setup(ppo, std::stoull(o["--seed"]));
    } else if (o["--online-lbr-schedule"] != "fixed") throw std::runtime_error("--online-lbr-schedule split needs --partition pp");
    if (!pp_mode && (o["--inherit-regret"] != "0" || o["--inherit-avg"] != "0"))
        throw std::runtime_error("--inherit-regret / --inherit-avg are --partition pp options (other partitions: --inherit 0|1)");
    // --log-regret 1: regret.csv, the bucket regret total sum_B max(0, max_a R_B(a)) per root of the current epoch's window
    // (R_B: the sampled counterfactual regrets of the training iterates, before RM+ clipping), at every curve row and
    // at every epoch end before the split; the window restarts at each epoch end (with --gate-eps: at each judgment).
    // It only reads the regrets the training adds: the training is unchanged.
    if (o["--log-regret"] != "0" && o["--log-regret"] != "1") throw std::runtime_error("--log-regret must be 0 or 1");
    const bool log_regret = o["--log-regret"] == "1";
    // --log-splits 1 (history partition): splits.csv, one row per child bucket of every forward and backward split
    // (see Model::hsplit / hbackward): parent, child, labels, keys moved, the label's visits, R, Q and R / sqrt(Q).
    // Logging only; rows of epochs after a checkpoint are rewritten on resume.
    if (o["--log-splits"] != "0" && o["--log-splits"] != "1") throw std::runtime_error("--log-splits must be 0 or 1");
    const bool log_splits = o["--log-splits"] == "1" && partition == Partition::History;
    model.track_iterate_regret(std::stod(o["--gate-eps"]) > 0 || log_regret);
    if (avg_fill) model.enable_avg_fill(std::stod(o["--avg-fill-reach"]));
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
        << (gated ? ";gate_eps=" + o["--gate-eps"] + ";gate_log=2" + (gate_iter ? ";gate_stat=iter" : "") : "")
        << (avg_fill ? ";avg_fill=plain;avg_fill_reach=" + o["--avg-fill-reach"] : "")
        << (label_pass && threads > 1 ? ";label_pass=chunked" : "")  // any --threads > 1 gives the same bits
        << (pp_mode ? ";pp_assign=" + o["--pp-assign"] + ";leaf_start=" + o["--leaf-start"] + ";pp_particles=" + o["--pp-particles"] +
                          ";pp_samples=" + o["--pp-samples"] + ";pp_min_samples=" + o["--pp-min-samples"] + ";pp_m=" + o["--pp-m"] +
                          ";pp_rows=" + o["--pp-rows"] + ";pp_pass_frac=" + o["--pp-pass-frac"] + ";pp_budget=" + o["--pp-budget"] +
                          ";pp_budget_min_steps=" + o["--pp-budget-min-steps"] + ";judge_mem_mb=" + o["--judge-mem-mb"] +
                          ";prior_ratio=" + o["--prior-ratio"] + ";prior_criterion=" + o["--prior-criterion"] + ";crossfit=" + o["--crossfit"] +
                          ";min_leaf=" + o["--min-leaf"] + ";features=" + o["--features"] + ";mem_budget_mb=" + o["--mem-budget-mb"] +
                          ";online_schedule=" + o["--online-lbr-schedule"] +
                          (ppo.inherit_regret > 0 || ppo.inherit_avg > 0 ? ";inherit_regret=" + o["--inherit-regret"] + ";inherit_avg=" + o["--inherit-avg"]
                                                                          : std::string()) : std::string());
    OnlineLbr online;
    online.setup(o, std::stoull(o["--seed"]), true, true);
    if (online.on() && learn_h4) throw std::runtime_error("--online-lbr-every needs the heuristic H4");
    cfg << online.config() << (log_regret ? ";log_regret=1" : "");
    const std::string config = cfg.str();

    std::mt19937_64 deal_rng(std::stoull(o["--seed"])), rng(std::stoull(o["--seed"]) ^ 0x9e3779b97f4a7c15ull);
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    const std::string backward_path = out + "/backward.csv", gate_path = out + "/gate.csv", fill_path = out + "/fill.csv";
    const std::string regret_path = out + "/regret.csv", splits_path = out + "/splits.csv";
    const std::string ppj_path = out + "/pp_judge.csv", ppp_path = out + "/prior_splits.csv";
    LoopState s;
    s.L = epoch0;
    uint64_t online_bytes = 0, regret_bytes = 0;
    uint64_t* regret_slot = log_regret ? &regret_bytes : nullptr;
    const bool resumed = read_checkpoint(out + "/checkpoint.bin", config, s, model, &online, &online_bytes, regret_slot) ||
                         read_checkpoint(out + "/checkpoint.tmp", config, s, model, &online, &online_bytes, regret_slot);
    online.open(out, resumed, online_bytes);
    if (resumed && log_regret) std::filesystem::resize_file(regret_path, regret_bytes);
    if (resumed) {
        rng_load(deal_rng, s.deal_rng); rng_load(rng, s.rng);
        // drop rows written after the checkpoint; they are reproduced exactly by the replay
        std::filesystem::resize_file(log_path, s.log_bytes);
        std::filesystem::resize_file(index_path, s.index_bytes);
        std::filesystem::resize_file(curve_path, s.curve_bytes);
        for (const auto& [on, path] : {std::pair{backward, backward_path}, std::pair{gated, gate_path}, std::pair{avg_fill, fill_path},
                                       std::pair{log_splits, splits_path}, std::pair{pp_mode, ppj_path}, std::pair{pp_mode, ppp_path}}) {
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
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds,train_cpu_seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets," << kResourceHeader
                                  << ",train_cpu_seconds\n";
        if (log_regret) std::ofstream(regret_path) << "roots,seconds,epoch,kind,window_roots,buckets,bucket_records,regret_sum_per_root\n";
        if (log_splits) std::ofstream(splits_path) << "epoch,roots,kind,street,legal_mask,parent,child,label_window,label_average,"
                                                     "label_bits,keys,visits,regret,regret_sq,z\n";
        if (gated) std::ofstream(gate_path) << "epoch,roots,buckets,rho_plain_half,rho_cross_half,iterate_regret_half,eps,judged\n";
        if (avg_fill) std::ofstream(fill_path) << "epoch,roots,buckets,traverser_nodes,filled_nodes\n";
        if (pp_mode) {
            std::ofstream(ppj_path) << "epoch,roots,level,buckets_before,judged_buckets,samples_per_bucket,samples,splits,children,"
                                       "pending_leaves,rows,leaves_fixed_by_rows,leaves_fixed_by_none,fill_leaves_fixed,signals,prior_splits,"
                                       "leaves_after,buckets_after,steps_posterior,steps_prior,steps_root_route,budget_steps,steps_per_sample,pass_roots,"
                                       "shards_posterior,shards_prior,pass_seconds,estimate_seconds,judge_seconds,bucket_mb,tree_mb,"
                                       "stored_tables_mb,rss_mb,peak_rss_mb,fill_steps,frozen\n";
            std::ofstream(ppp_path) << "epoch,roots,leaf,street,legal_mask,depth,feature,threshold,gain_a,gain_b,rows,signal,left_leaf,"
                                       "right_leaf,left_bucket,right_bucket\n";
        }
        if (backward) std::ofstream(backward_path) << "epoch,roots,keys,keys_diagnosed,subkeys,subkey_visits_dropped,keys_two_labels,keys_inconsistent,pairs,"
                                                       "backward_splits,peak_rss_mb\n";
    }
    // Training CPU time (--budget-cpu-seconds, column train_cpu_seconds): process CPU time since this process started
    // training, minus the online LBR's time, plus that of earlier processes of the run (OUT/train_cpu.txt, written
    // after every checkpoint; not part of the checkpoint, so runs started by older executables resume unchanged: they
    // start from their training wall seconds and keep their CSV columns). Reading clocks does not change the training.
    // --budget-cpu-seconds S (0 = none; not a setting of the run): when it reaches S (checked every 1,000 roots), a
    // budget snapshot snapshot_b<roots>.bin (kind "budget"), a checkpoint, and a stop; rerun with 0 to continue to --roots.
    const double budget = std::stod(o["--budget-cpu-seconds"]);
    if (budget < 0) throw std::runtime_error("--budget-cpu-seconds must be >= 0");
    const std::string cpu_path = out + "/train_cpu.txt";
    double cpu_base = 0;
    if (resumed) {
        std::ifstream in(cpu_path);
        if (!(in >> cpu_base)) {
            cpu_base = s.seconds;
            std::cerr << "no " << cpu_path << ": training CPU time starts from the training wall seconds " << s.seconds << "\n";
        }
    }
    auto header_has = [](const std::string& path, const std::string& col) {
        std::ifstream in(path); std::string h; std::getline(in, h);
        return h.find(col) != std::string::npos;
    };
    const bool cpu_curve = header_has(curve_path, "train_cpu_seconds"), cpu_index = header_has(index_path, "train_cpu_seconds");
    const double cpu_start = cpu_seconds(), online_cpu0 = online.seconds;
    auto train_cpu = [&] { return cpu_base + (cpu_seconds() - cpu_start) - (online.seconds - online_cpu0); };
    if (budget > 0 && cpu_base >= budget) {
        std::cerr << "CPU budget " << budget << " s already reached (training CPU " << cpu_base << " s at root " << s.t
                  << "); rerun with --budget-cpu-seconds 0 or a larger budget to continue towards --roots\n";
        return 0;
    }
    std::ofstream log(log_path, std::ios::app), index(index_path, std::ios::app), curve(curve_path, std::ios::app);
    std::ofstream regret, splits_log, ppj_log, ppp_log;
    if (pp_mode) { ppj_log.open(ppj_path, std::ios::app); ppp_log.open(ppp_path, std::ios::app); }
    if (log_regret) regret.open(regret_path, std::ios::app);
    if (log_splits) splits_log.open(splits_path, std::ios::app);
    // graceful stop: Ctrl+C / SIGTERM, or creating OUT/STOP (works for background processes on Windows too)
    const std::string stop_file = out + "/STOP";
    std::filesystem::remove(stop_file);
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef SIGBREAK
    std::signal(SIGBREAK, request_stop);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const double base_seconds = s.seconds, online0 = online.seconds;
    // training seconds: the online LBR's own time is excluded (it is in online_lbr.csv)
    auto elapsed = [&] { return base_seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - (online.seconds - online0); };
    auto live_total = [&] { const auto l = model.live_buckets(); return l[0] + l[1] + l[2]; };
    auto regret_row = [&](const char* kind) {  // --log-regret
        regret << s.t << ',' << elapsed() << ',' << s.k << ',' << kind << ',' << model.iterate_window_roots() << ',' << live_total()
               << ',' << model.bucket_records() << ',' << 2 * model.iterate_regret_half(false) << '\n';
    };
    auto checkpoint = [&] {
        log.flush(); index.flush(); curve.flush();
        s.seconds = elapsed(); s.deal_rng = rng_text(deal_rng); s.rng = rng_text(rng);
        s.log_bytes = file_size_or_zero(log_path); s.index_bytes = file_size_or_zero(index_path);
        s.curve_bytes = file_size_or_zero(curve_path);
        if (log_regret) { regret.flush(); regret_bytes = file_size_or_zero(regret_path); }
        if (log_splits) splits_log.flush();
        if (pp_mode) { ppj_log.flush(); ppp_log.flush(); }
        write_checkpoint(out, config, s, model, &online, regret_slot);
        { std::ofstream(cpu_path + ".tmp") << std::setprecision(17) << train_cpu() << ' ' << s.t << '\n'; }
        std::filesystem::rename(cpu_path + ".tmp", cpu_path);
    };
    const bool track = partition == Partition::Aswiss || partition == Partition::History || partition == Partition::PP;
    // --freeze-splits-at R (R > 0): an epoch ending at root R or later is not judged (no label pass, no split, no
    // window): training goes on over the partition reached, with snapshots and epochs as before. It is not part of the
    // checkpoint configuration, so a run can be frozen when it is resumed (the run is the same up to the freeze).
    const uint64_t freeze_at = std::stoull(o["--freeze-splits-at"]);
    // --judge-at R (R > 0, a root still ahead): the epoch containing R ends at R and is judged there, frozen or not
    // (snapshot snapshot_K.bin, regret row, label pass, splits, as at any epoch end). The schedule then goes on from R
    // as from any epoch end (the next epoch is twice as long). Not part of the checkpoint configuration either: a
    // root already passed is ignored, so the same command can be rerun after the judgment.
    const uint64_t judge_at = std::stoull(o["--judge-at"]);
    auto judging = [&](uint64_t end) { return track && (freeze_at == 0 || end < freeze_at || (judge_at && end == judge_at)); };
    auto schedule_end = [&](uint64_t end) { return judge_at > s.t && judge_at < end ? judge_at : end; };
    if (judge_at && !track) throw std::runtime_error("--judge-at needs a splitting partition (history or aswiss)");
    uint64_t epoch_end = schedule_end(std::min(T, s.epoch_start + s.L));
    if (judge_at > s.t) std::cerr << "judgment forced at root " << judge_at << (judge_at > T ? " (beyond --roots: no effect)" : "") << "\n";
    while (s.t < T) {
        ++s.t;
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        if (ppo.online_split ? online.on() && model.pp_online_due(s.t) : online.due(static_cast<long long>(s.t)))  // test, then train
            for (const auto& policy : online.policies) {
                SnapshotPolicy target{model, policy == "average", std::mt19937_64(online.target_seed(static_cast<long long>(s.t), policy))};
                online.evaluate(static_cast<long long>(s.t), deck, target, policy, elapsed());
            }
        // --label-cap: window sums only over the epoch's last label_cap roots (bounds the window's memory)
        // history keys with --label avg: no window at all (it would only feed the L/C diagnostic and doubles the memory)
        const bool in_window = judging(epoch_end) && (!label_cap || epoch_end - s.t < label_cap) &&
                               !(partition == Partition::History && label_rule == "avg") && !pp_mode;
        for (int p = 0; p < 2; ++p) {
            model.traverse_root(deck, p, static_cast<double>(s.t), rng, in_window);
            model.plain_tick(0.5 * static_cast<double>(s.t));
            model.apply_regrets();
        }
        model.count_root();
        const double gap = model.take_traversal_gap() / 2;  // mean of the two seats, this root
        s.curve_sum += gap; s.curve_sq += gap * gap; ++s.curve_roots;
        if (curve_every && s.t % curve_every == 0) {
            const double n = s.curve_roots, mean = s.curve_sum / n;
            const double se = n > 1 ? std::sqrt(std::max(0.0, (s.curve_sq - n * mean * mean) / (n - 1) / n)) : 0.0;
            curve << s.t << ',' << elapsed() << ',' << s.curve_roots << ',' << mean << ',' << se << ',' << live_total() << ','
                  << memory_columns();
            if (cpu_curve) curve << ',' << train_cpu();
            curve << '\n';
            s.curve_sum = s.curve_sq = 0; s.curve_roots = 0;
            if (log_regret) regret_row("running");
        }
        const bool epoch_done = s.t == epoch_end;
        if (snapshot_every && s.t % snapshot_every == 0 && !epoch_done) {
            const std::string file = "snapshot_r" + std::to_string(s.t) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",periodic," << s.k << ',' << live_total() << ',' << elapsed();
            if (cpu_index) index << ',' << train_cpu();
            index << '\n';
        }
        if (epoch_done) {
            const std::string file = "snapshot_" + std::to_string(s.k) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",epoch," << s.k << ',' << live_total() << ',' << elapsed();
            if (cpu_index) index << ',' << train_cpu();
            index << '\n';
            if (log_regret) regret_row("epoch");  // the epoch's whole window, before its split
            const auto w = model.window();
            const auto mb = model.table_megabytes();  // measured before the window is cleared
            int splits = 0;
            if (pp_mode) {
                // --mem-budget-mb: once the resident memory plus a judgment's transient memory reaches the budget, no more
                // judgments (the partition reached so far is trained to the end)
                pp::Data& D = *model.ppd();
                if (ppo.mem_budget_mb > 0 && !D.frozen && memory_mb(1) + ppo.judge_mem_mb >= ppo.mem_budget_mb) {
                    D.frozen = true;
                    std::cerr << "memory budget " << ppo.mem_budget_mb << " MB reached: splits frozen from root " << s.t << "\n";
                }
                const uint64_t fill_steps = model.pp_take_fill_steps();
                if (judging(s.t) && !D.frozen) {
                    splits = model.pp_judge(std::stoull(o["--seed"]), s.k, s.t, s.t - s.epoch_start, fill_steps, ppj_log, ppp_log);
                } else {
                    ppj_log << s.k << ',' << s.t << ',' << D.level << std::string(30, ',') << memory_mb(1) << ',' << peak_rss_mb() << ','
                            << fill_steps << ',' << D.frozen << '\n';
                }
            }
            else if (judging(s.t) && label_rule == "window") splits = model.split(z, inherit);
            else if (judging(s.t) && partition == Partition::History) {  // method B with bucket-history keys
                HMap avg;
                BucketHalves halves;
                const uint64_t eval_seed = std::stoull(o["--seed"]) * 1000003ull + s.k;
                uint64_t n = std::max<uint64_t>(1000, static_cast<uint64_t>(avg_frac * static_cast<double>(s.t - s.epoch_start)));
                if (label_cap) n = std::min(n, label_cap);
                model.plain_flush_all();
                model.take_fill_counts();
                uint64_t fill_nodes = 0, fill_filled = 0;
                const double pass_start = elapsed();
                if (threads == 1) {
                    std::mt19937_64 eval_rng(eval_seed);
                    for (uint64_t i = 0; i < n; ++i) {
                        auto eval_deck = fresh_deck();
                        std::shuffle(eval_deck.begin(), eval_deck.end(), eval_rng);
                        for (int p = 0; p < 2; ++p)
                            avg_traverse_h(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), Trail{}, p, eval_rng, &avg,
                                           nullptr, nullptr, gated ? &halves : nullptr, static_cast<int>(i & 1));
                    }
                    std::tie(fill_nodes, fill_filled) = model.take_fill_counts();
                } else {
                    // one fresh table per chunk of a round (its iteration order then depends on that chunk alone)
                    std::vector<HMap> shards(kLabelShards), part(threads);
                    std::vector<BucketHalves> part_halves(threads);
                    std::vector<std::pair<uint64_t, uint64_t>> part_fill(threads);
                    chunked_pass(n, threads, atlas.get(), [&](uint64_t c, int t) {
                        Model::take_fill_counts();
                        std::mt19937_64 rng;
                        for (uint64_t i = c * kLabelChunk; i < std::min(n, (c + 1) * kLabelChunk); ++i) {
                            const State root = label_root(eval_seed, i, rng);
                            for (int p = 0; p < 2; ++p)
                                avg_traverse_h(model, root, Trail{}, p, rng, &part[t], nullptr, nullptr, gated ? &part_halves[t] : nullptr,
                                               static_cast<int>(i & 1));
                        }
                        part_fill[t] = Model::take_fill_counts();
                    }, [&](uint64_t, int m) {
                        merge_chunk_stats(shards, part, m, threads);
                        for (int t = 0; t < m; ++t) {
                            if (gated) merge_halves(halves, part_halves[t]);
                            fill_nodes += part_fill[t].first; fill_filled += part_fill[t].second;
                            part[t] = HMap(); part_halves[t] = BucketHalves();
                        }
                    });
                    avg = join_shards(shards);
                }
                const double pass_secs = elapsed() - pass_start;
                // --gate-eps: judge only when the bucket regret total of the average policy, rho_P/2 per root, is at most
                // eps. rho_P = sum_B max(0, max_a r(B, a)) estimated from this pass: the plain estimate is biased upward
                // by the noise of tens of thousands of bucket sums, so the gate uses the cross-fitted one (each bucket's
                // best action chosen on one half of the roots, its regret measured on the other half, both ways).
                if (avg_fill) {
                    std::ofstream(fill_path, std::ios::app) << s.k << ',' << s.t << ',' << live_total() << ',' << fill_nodes << ',' << fill_filled << '\n';
                    std::cerr << "average-policy pass: " << fill_filled << " of " << fill_nodes << " traverser nodes valued with the plain average\n";
                }
                bool judge = true;
                if (gated) {
                    const auto [plain, cross] = rho_estimates(halves, n);
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
                double replay_secs = 0;
                if (backward && judge) {
                    for (const auto& [k, st] : avg) if (st.visits >= backward_min_visits) frequent.insert(k);
                    const double replay_start = elapsed();
                    if (threads == 1) {
                        std::mt19937_64 replay_rng(eval_seed);
                        for (uint64_t i = 0; i < n; ++i) {
                            auto eval_deck = fresh_deck();
                            std::shuffle(eval_deck.begin(), eval_deck.end(), replay_rng);
                            for (int p = 0; p < 2; ++p)
                                avg_traverse_h(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), Trail{}, p, replay_rng,
                                               nullptr, &sub, &frequent);
                        }
                    } else {
                        std::vector<SubMap> part(threads);
                        const auto fresh = [&](SubMap& m) { m = SubMap(); m.cap = backward_max_subkeys; m.ordered = true; };
                        for (auto& m : part) fresh(m);
                        chunked_pass(n, threads, atlas.get(), [&](uint64_t c, int t) {
                            std::mt19937_64 rng;
                            for (uint64_t i = c * kLabelChunk; i < std::min(n, (c + 1) * kLabelChunk); ++i) {
                                const State root = label_root(eval_seed, i, rng);
                                for (int p = 0; p < 2; ++p) avg_traverse_h(model, root, Trail{}, p, rng, nullptr, &part[t], &frequent);
                            }
                        }, [&](uint64_t, int m) {
                            for (int t = 0; t < m; ++t) { merge_sub(sub, part[t]); fresh(part[t]); }
                        });
                    }
                    replay_secs = elapsed() - replay_start;
                }
                if (log_splits) model.set_split_log(&splits_log, s.k, s.t);
                if (judge) splits = model.hsplit(z, &avg, label_rule == "window+avg");
                std::cerr << "average-policy pass: " << n << " roots, " << avg.size() << " keys (" << pass_secs << " s"
                          << (backward && judge ? ", replay " + std::to_string(replay_secs) + " s" : "") << ", " << threads
                          << (threads == 1 ? " thread)\n" : " threads)\n");
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
            else if (judging(s.t) && partition == Partition::Aswiss) {
                // method B: regrets of the frozen average policy from a separate sampled pass (deterministic per epoch)
                AtomMap avg;
                const uint64_t eval_seed = std::stoull(o["--seed"]) * 1000003ull + s.k;
                uint64_t n = std::max<uint64_t>(1000, static_cast<uint64_t>(avg_frac * static_cast<double>(s.t - s.epoch_start)));
                if (label_cap) n = std::min(n, label_cap);
                const double pass_start = elapsed();
                if (threads == 1) {
                    std::mt19937_64 eval_rng(eval_seed);
                    for (uint64_t i = 0; i < n; ++i) {
                        auto eval_deck = fresh_deck();
                        std::shuffle(eval_deck.begin(), eval_deck.end(), eval_rng);
                        for (int p = 0; p < 2; ++p)
                            avg_traverse(model, sample_fifth_street_root(eval_deck, kAnte, kEffectiveStackAnte), p, eval_rng, avg);
                    }
                } else {
                    std::vector<AtomMap> shards(kLabelShards), part(threads);
                    chunked_pass(n, threads, atlas.get(), [&](uint64_t c, int t) {
                        std::mt19937_64 rng;
                        for (uint64_t i = c * kLabelChunk; i < std::min(n, (c + 1) * kLabelChunk); ++i) {
                            const State root = label_root(eval_seed, i, rng);
                            for (int p = 0; p < 2; ++p) avg_traverse(model, root, p, rng, part[t]);
                        }
                    }, [&](uint64_t, int m) {
                        merge_chunk_stats(shards, part, m, threads);
                        for (int t = 0; t < m; ++t) part[t] = AtomMap();
                    });
                    avg = join_shards(shards);
                }
                const double pass_secs = elapsed() - pass_start;
                splits = model.split(z, inherit, &avg, label_rule == "window+avg");
                std::cerr << "average-policy pass: " << n << " roots, " << avg.size() << " atoms (" << pass_secs << " s, " << threads
                          << (threads == 1 ? " thread)\n" : " threads)\n");
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
                      << " splits=" << splits << (track && !judging(s.t) ? " (splits frozen)" : "") << " h4_buckets=" << model.h4_live() << " atoms=" << w.atoms << " H/root=" << (w.L - w.C) / len << " (" << secs << "s)\n";
            if (log_regret && !gated) model.iterate_regret_half(true);  // the next epoch's window (gated: reset at judgments)
            s.epoch_start = s.t; ++s.k; s.L *= 2; epoch_end = schedule_end(std::min(T, s.t + s.L));
        }
        if (epoch_done || (checkpoint_every && s.t % checkpoint_every == 0) || s.t == T) checkpoint();
        if (s.t % 1000 == 0 && std::filesystem::exists(stop_file)) g_stop = 1;
        if (budget > 0 && s.t % 1000 == 0 && s.t < T && train_cpu() >= budget) {  // CPU budget: snapshot, checkpoint, stop
            const std::string file = "snapshot_b" + std::to_string(s.t) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",budget," << s.k << ',' << live_total() << ',' << elapsed();
            if (cpu_index) index << ',' << train_cpu();
            index << '\n';
            checkpoint();
            std::cerr << "CPU budget " << budget << " s reached at root " << s.t << " (training CPU " << train_cpu() << " s); snapshot "
                      << file << " and checkpoint saved. Rerun with --budget-cpu-seconds 0 (or larger) to continue\n";
            return 0;
        }
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
                                {"--seed", "2026102701"}, {"--power-cache", "1000000"}, {"--pairs-csv", ""},
                                {"--rho-roots", "0"}, {"--threads", "1"},
                                {"--rebel-iters56", ""}, {"--rebel-iters7", ""}, {"--rebel-buckets", ""}, {"--rebel-play", ""}});
    if (o["--partition"] == "net") return sota::eval_net(o);
    if (o["--partition"] == "rebel") return sota::eval_rebel(o);  // --threads: parallel LBR hands (ReBeL only)
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
        if (x.empty()) return std::make_pair(0.0, 0.0);
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
    // --rho-roots N (history partition, average policy): the bucket regret total rho_P/2 of the evaluated policy from N
    // sampled roots (own deals and seed), the statistic the --gate-eps gate uses; -1 when not computed
    const uint64_t rho_roots = std::stoull(o["--rho-roots"]);
    double rho_plain = -1, rho_cross = -1;
    if (rho_roots && average && parse_partition(o["--partition"]) == Partition::History && !model.has_h4()) {
        BucketHalves halves;
        std::mt19937_64 rho_rng(seed ^ 0x3c6ef372fe94f82bull);
        for (uint64_t i = 0; i < rho_roots; ++i) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), rho_rng);
            for (int p = 0; p < 2; ++p)
                avg_traverse_h(model, sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), Trail{}, p, rho_rng, nullptr,
                               nullptr, nullptr, &halves, static_cast<int>(i & 1));
        }
        std::tie(rho_plain, rho_cross) = rho_estimates(halves, rho_roots);
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
              << ",\"h4\":\"" << (model.has_h4() ? "learned" : "heuristic") << "\",\"h4_buckets\":" << model.h4_live() << ",\"rho_roots\":" << rho_roots << ",\"rho_plain_half\":" << rho_plain
              << ",\"rho_cross_half\":" << rho_cross << "}\n";
    return 0;
}

// ---- league: duplicate head-to-head matches between saved policies and the engine's heuristics ----------------------
// ./aswiss_stud league --agents FILE [--deals 20000] [--seed 2026110901] [--power-cache 1000000] --out DIR
// FILE is a CSV with the header name,kind,partition,atlas,snapshot,policy; one agent per row:
//   snapshot rows: kind snapshot, partition context|atom|aswiss|history, the run's atlas (empty for context), the
//                  snapshot file, policy average|current (what the run plays: average)
//   heuristic rows: kind heuristic, partition = the engine's variant (baseline, bluff-defense, over-bluff,
//                  hidden-bluff, slow-play); the other fields empty
// Every pair plays the same deals, each deal twice with the seats swapped (duplicate poker), so the cards cancel within
// a deal and every pair sees the same cards. Discard/reveal is the game's heuristic H4 for both seats (snapshots with a
// learned H4 are refused). Mixed policies sample their actions from a generator seeded by (seed, deal, pair, seat
// order), so a pair's result does not depend on which other agents are in the league. Values are ante per hand for
// the row agent: league_pairs.csv (every pair, mean and 95% interval), league_agents.csv (each agent's mean over all
// opponents, interval over deals), league_matrix.md (both tables), league_deals.csv (per-deal values, --deals-csv 1).
struct LeagueAgent {
    std::string name, kind, partition, atlas_path, snapshot, policy;
    HeuristicVariant variant = HeuristicVariant::Baseline;
    std::unique_ptr<Model> model;
    const PowerAtlas* atlas = nullptr;
    bool average = true;
    Action choose(const State& state, int seat, std::mt19937_64& rng) const {
        if (!model) return heuristic_variant_action(state, seat, variant);
        return sample_action(model->strategy(state, seat, average), valid_mask(state, seat), rng);
    }
};
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out(1);
    for (char c : line) {
        if (c == ',') out.emplace_back();
        else if (c != '\r') out.back() += c;
    }
    for (auto& f : out) {  // trim spaces
        const auto a = f.find_first_not_of(' '), b = f.find_last_not_of(' ');
        f = a == std::string::npos ? "" : f.substr(a, b - a + 1);
    }
    return out;
}
// seat 0 is agents[0]; returns seat 0's result in ante (heuristic H4 for both seats)
double play_league_hand(const std::vector<Card>& deck, const std::array<const LeagueAgent*, 2>& agents,
                        std::array<std::mt19937_64, 2>& rng) {
    State state;
    state.ante = kAnte;
    state.effective_stack = kAnte * kEffectiveStackAnte;
    state.pot = kAnte * 2;
    for (Player& player : state.players) player.invested = kAnte;
    size_t cursor = 0;
    prepare_initial_cards(state, deck, cursor, -1, nullptr, nullptr, -1);
    for (int seat = 0; seat < 2; ++seat) state.players[seat].shown.push_back(deck[cursor++]);
    for (int street : {5, 6, 7}) {
        for (int seat = 0; seat < 2; ++seat) {
            if (street == 7) state.players[seat].hidden.push_back(deck[cursor++]);
            else state.players[seat].shown.push_back(deck[cursor++]);
        }
        reset_round(state, street);
        if (!state.players[0].all_in && !state.players[1].all_in) {
            int actor = state.actor;
            while (!state.terminal) {
                const Action action = agents[actor]->choose(state, actor, rng[actor]);
                const ActionResult result = apply_action(state, actor, action);
                if (result == ActionResult::RoundEnd || result == ActionResult::FoldEnd) break;
                actor = 1 - actor;
                state.actor = actor;
            }
        }
        if (state.terminal) break;
    }
    return settle(state, 0);
}
int league(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--agents", ""}, {"--deals", "20000"}, {"--seed", "2026110901"}, {"--power-cache", "1000000"},
                                {"--out", ""}, {"--deals-csv", "0"}});
    std::ifstream in(o.at("--agents"));
    if (!in) throw std::runtime_error("league: cannot read --agents " + o["--agents"]);
    std::string line;
    std::getline(in, line);
    const auto header = split_csv_line(line);
    const std::vector<std::string> want{"name", "kind", "partition", "atlas", "snapshot", "policy"};
    if (header != want) throw std::runtime_error("league: the agents file header must be name,kind,partition,atlas,snapshot,policy");
    std::vector<LeagueAgent> agents;
    std::map<std::string, std::unique_ptr<PowerAtlas>> atlases;
    const size_t cache = std::stoull(o["--power-cache"]);
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line.find_first_not_of(" ,\r") == std::string::npos) continue;
        auto f = split_csv_line(line);
        f.resize(6);
        LeagueAgent a;
        a.name = f[0]; a.kind = f[1]; a.partition = f[2]; a.atlas_path = f[3]; a.snapshot = f[4]; a.policy = f[5].empty() ? "average" : f[5];
        if (a.name.empty()) throw std::runtime_error("league: an agent without a name");
        for (const auto& b : agents) if (b.name == a.name) throw std::runtime_error("league: duplicate agent name " + a.name);
        if (a.kind == "heuristic") {
            const std::string v = a.partition.empty() ? "baseline" : a.partition;
            a.variant = heuristic_variant_from_name(v);
            if (v != "baseline" && a.variant == HeuristicVariant::Baseline) throw std::runtime_error("league: unknown heuristic variant " + v);
        } else if (a.kind == "snapshot") {
            if (a.policy != "average" && a.policy != "current") throw std::runtime_error("league: policy must be average or current (" + a.name + ")");
            a.average = a.policy == "average";
            if (!a.atlas_path.empty()) {
                auto& slot = atlases[a.atlas_path];
                if (!slot) slot = load_atlas(a.atlas_path, cache);
                a.atlas = slot.get();
            }
            a.model = std::make_unique<Model>(parse_partition(a.partition), a.atlas);
            a.model->load(a.snapshot);
            if (a.model->has_h4()) throw std::runtime_error("league: " + a.name + " has a learned H4; the league plays the heuristic H4");
            std::cerr << "loaded " << a.name << " (" << a.partition << ", " << a.snapshot << ", " << a.policy << ")\n";
        } else {
            throw std::runtime_error("league: kind must be snapshot or heuristic (" + a.name + ")");
        }
        agents.push_back(std::move(a));
    }
    const int n = static_cast<int>(agents.size());
    if (n < 2) throw std::runtime_error("league: at least two agents");
    const int deals = std::stoi(o["--deals"]);
    const uint64_t seed = std::stoull(o["--seed"]);
    const std::string out = o.at("--out");
    std::filesystem::create_directories(out);
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) pairs.emplace_back(i, j);
    // value[p][d]: agent pairs[p].first's result on deal d, averaged over the two seatings
    std::vector<std::vector<double>> value(pairs.size(), std::vector<double>(deals));
    auto mix = [](uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
                                x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); };
    std::hash<std::string> hs;
    std::mt19937_64 deal_rng(seed);
    const auto t0 = std::chrono::steady_clock::now();
    for (int d = 0; d < deals; ++d) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        for (size_t p = 0; p < pairs.size(); ++p) {
            const LeagueAgent &a = agents[pairs[p].first], &b = agents[pairs[p].second];
            const uint64_t key = mix(seed ^ mix(static_cast<uint64_t>(d) ^ mix(hs(a.name) ^ mix(hs(b.name)))));
            double sum = 0;
            for (int order = 0; order < 2; ++order) {  // order 0: a in seat 0; order 1: b in seat 0
                std::array<std::mt19937_64, 2> rng{std::mt19937_64(mix(key ^ (4ull * order))), std::mt19937_64(mix(key ^ (4ull * order + 1)))};
                const double r0 = order == 0 ? play_league_hand(deck, {&a, &b}, rng) : play_league_hand(deck, {&b, &a}, rng);
                sum += order == 0 ? r0 : -r0;
            }
            value[p][d] = sum / 2;
        }
        if ((d + 1) % 1000 == 0 || d + 1 == deals) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cerr << "deal " << d + 1 << "/" << deals << " (" << std::fixed << std::setprecision(1) << sec << " s, about "
                      << sec / (d + 1) * (deals - d - 1) << " s left)\n" << std::defaultfloat;
        }
    }
    auto mean_ci = [](const std::vector<double>& x) {
        double m = 0, v = 0;
        for (double y : x) m += y;
        m /= x.size();
        for (double y : x) v += (y - m) * (y - m);
        const double se = x.size() > 1 ? std::sqrt(v / (x.size() - 1) / x.size()) : 0.0;
        return std::make_pair(m, 1.96 * se);
    };
    std::ofstream pc(out + "/league_pairs.csv");
    pc << "agent,opponent,deals,ante_per_hand,ci95\n" << std::setprecision(6);
    std::vector<std::vector<std::pair<double, double>>> cell(n, std::vector<std::pair<double, double>>(n, {0.0, 0.0}));
    for (size_t p = 0; p < pairs.size(); ++p) {
        const auto [m, h] = mean_ci(value[p]);
        const auto [i, j] = pairs[p];
        cell[i][j] = {m, h}; cell[j][i] = {-m, h};
        pc << agents[i].name << ',' << agents[j].name << ',' << deals << ',' << m << ',' << h << '\n';
        pc << agents[j].name << ',' << agents[i].name << ',' << deals << ',' << -m << ',' << h << '\n';
    }
    // each agent's mean result against the field: per deal, the average over its opponents, then the interval over deals
    std::vector<std::pair<double, double>> field(n);
    std::ofstream ac(out + "/league_agents.csv");
    ac << "agent,kind,partition,snapshot,policy,opponents,deals,mean_vs_field,ci95\n" << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        std::vector<double> per(deals, 0.0);
        for (size_t p = 0; p < pairs.size(); ++p) {
            if (pairs[p].first == i) for (int d = 0; d < deals; ++d) per[d] += value[p][d] / (n - 1);
            if (pairs[p].second == i) for (int d = 0; d < deals; ++d) per[d] -= value[p][d] / (n - 1);
        }
        field[i] = mean_ci(per);
        ac << agents[i].name << ',' << agents[i].kind << ',' << agents[i].partition << ',' << agents[i].snapshot << ','
           << agents[i].policy << ',' << n - 1 << ',' << deals << ',' << field[i].first << ',' << field[i].second << '\n';
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return field[x].first > field[y].first; });
    std::ostringstream md;
    md << std::fixed << std::setprecision(3);
    md << "League: " << deals << " deals, each played twice with the seats swapped (duplicate), seed " << seed
       << ". Values: ante per hand for the row agent, 95% interval over deals.\n\n";
    md << "| rank | agent | mean against the field |\n|---:|---|---:|\n";
    for (int r = 0; r < n; ++r) md << "| " << r + 1 << " | " << agents[order[r]].name << " | " << field[order[r]].first << " ± " << field[order[r]].second << " |\n";
    md << "\n| row vs column |";
    for (int c : order) md << ' ' << agents[c].name << " |";
    md << "\n|---|";
    for (int c = 0; c < n; ++c) md << "---:|";
    md << '\n';
    for (int r : order) {
        md << "| " << agents[r].name << " |";
        for (int c : order) {
            if (r == c) md << " - |";
            else md << ' ' << cell[r][c].first << " ± " << cell[r][c].second << " |";
        }
        md << '\n';
    }
    std::ofstream(out + "/league_matrix.md") << md.str();
    if (o["--deals-csv"] == "1") {
        std::ofstream dc(out + "/league_deals.csv");
        dc << "deal";
        for (const auto& [i, j] : pairs) dc << ',' << agents[i].name << " vs " << agents[j].name;
        dc << '\n' << std::setprecision(6);
        for (int d = 0; d < deals; ++d) {
            dc << d;
            for (size_t p = 0; p < pairs.size(); ++p) dc << ',' << value[p][d];
            dc << '\n';
        }
    }
    std::cout << md.str();
    return 0;
}

// ---- bucket-regret: each bucket's regret under a snapshot's average policy (confidence in the web analysis) ----------
// ./aswiss_stud bucket-regret [--partition history] --atlas ATLAS --snapshot FILE [--roots 2000000] [--seed 2026111001]
//                             [--min-visits 20] [--r0 0] [--power-cache 1000000] [--out FILE (default SNAPSHOT.regret.csv)]
// The average-policy pass of method B: both seats play the snapshot's average policy, the traverser
// expands its own actions, the other seat and chance are sampled; roots split into two halves by parity. Per bucket:
//   visits            traverser nodes of the bucket in the pass
//   plain_per_visit   max(0, max_a R(a)) / visits (R: the sampled counterfactual regret sums; biased up by the noise)
//   cross_per_visit   (R_1(a_0) + R_0(a_1)) / visits, a_h = argmax of half h's sums when positive: one half picks
//                     the action, the other values it, so the noise does not pick its own maximum (as rho_P/2)
//   regret_per_visit  max(0, cross_per_visit): ante a visit would gain on average by switching to the best fixed
//                     action instead of the average policy (the bucket's term of rho_P, per visit)
//   confidence        r0 / (r0 + regret_per_visit) in (0, 1]: 1 at no regret, 1/2 at r0, about r0 / regret for large
//                     regrets (inversely proportional); r0 = the median regret_per_visit of the buckets with at least
//                     --min-visits visits, unless --r0 > 0. Fewer visits: no confidence (empty), too few samples.
// The first line is a comment with the settings and r0; `serve --bucket-regret FILE` reports the row of the bucket.
// Printed summary: rho_cross_half / rho_plain_half (the bucket regret total per root, rho_P/2, cross-fitted / plain),
// scored_visit_share (share of the traverser visits in buckets with --min-visits or more) and regret_per_visit_scored
// (their visit-weighted regret per visit): totals grow with the number of buckets, the per-visit figure compares
// abstractions of different sizes; the cross-fitted values are conservative where a bucket has few visits.
// --partition atom|aswiss|context (k-means and atom-label runs): the same pass with the atom key's bucket (no
// --avg-fill there), so a fixed k-means snapshot and an ASWISS snapshot get the same statistic on the same deals.
double avg_traverse_atom(const Model& model, State state, int traverser, std::mt19937_64& rng, BucketHalves& halves, int half) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const uint32_t b = model.lookup(model.atom_key(state, actor));
    const auto avg = model.bucket_policy(b, mask, true);
    if (actor != traverser) {
        step(state, actor, sample_action(avg, mask, rng));
        return avg_traverse_atom(model, std::move(state), traverser, rng, halves, half);
    }
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        v[a] = avg_traverse_atom(model, std::move(child), traverser, rng, halves, half);
        node += avg[a] * v[a];
    }
    if (b != kNone) {
        auto& h = halves[b];
        for (int a = 0; a < A; ++a) if (mask & (1u << a)) h[half * A + a] += v[a] - node;
        h[2 * A + half] += 1;
    }
    return node;
}
int bucket_regret(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "history"}, {"--atlas", ""}, {"--snapshot", ""}, {"--roots", "2000000"},
                                {"--seed", "2026111001"}, {"--min-visits", "20"}, {"--r0", "0"}, {"--power-cache", "1000000"},
                                {"--out", ""}});
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    const Partition partition = parse_partition(o["--partition"]);
    Model model(partition, atlas.get());
    model.load(o.at("--snapshot"));
    if (model.has_h4()) throw std::runtime_error("bucket-regret: snapshots with a learned H4 are not supported");
    const uint64_t roots = std::stoull(o["--roots"]), seed = std::stoull(o["--seed"]);
    const double min_visits = std::stod(o["--min-visits"]);
    const std::string out = o["--out"].empty() ? o["--snapshot"] + ".regret.csv" : o["--out"];
    BucketHalves halves;
    std::mt19937_64 rng(seed);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < roots; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        for (int p = 0; p < 2; ++p) {
            if (partition == Partition::History)
                avg_traverse_h(model, sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), Trail{}, p, rng, nullptr,
                               nullptr, nullptr, &halves, static_cast<int>(i & 1));
            else
                avg_traverse_atom(model, sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), p, rng, halves, static_cast<int>(i & 1));
        }
        if ((i + 1) % 100000 == 0 || i + 1 == roots) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cerr << "root " << i + 1 << "/" << roots << " (" << std::fixed << std::setprecision(0) << sec << " s, about "
                      << sec / (i + 1) * (roots - i - 1) << " s left), buckets seen " << halves.size() << "\n" << std::defaultfloat;
        }
    }
    struct Row { uint32_t b; double visits, plain, cross; int best; };
    std::vector<Row> rows;
    rows.reserve(halves.size());
    for (const auto& [b, h] : halves) {
        double best = 0, m0 = 0, m1 = 0;
        int a0 = -1, a1 = -1, ab = -1;
        for (int a = 0; a < A; ++a) {
            if (h[a] + h[A + a] > best) { best = h[a] + h[A + a]; ab = a; }
            if (h[a] > m0) { m0 = h[a]; a0 = a; }
            if (h[A + a] > m1) { m1 = h[A + a]; a1 = a; }
        }
        const double cross = (a0 >= 0 ? h[A + a0] : 0.0) + (a1 >= 0 ? h[a1] : 0.0);
        const double visits = h[2 * A] + h[2 * A + 1];
        if (visits > 0) rows.push_back({b, visits, best / visits, cross / visits, ab});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.b < y.b; });
    double r0 = std::stod(o["--r0"]);
    std::vector<double> enough;
    for (const auto& r : rows) if (r.visits >= min_visits) enough.push_back(std::max(0.0, r.cross));
    if (r0 <= 0) {
        if (enough.empty()) throw std::runtime_error("bucket-regret: no bucket has --min-visits visits; raise --roots");
        std::nth_element(enough.begin(), enough.begin() + enough.size() / 2, enough.end());
        r0 = std::max(enough[enough.size() / 2], 1e-12);
    }
    std::ofstream csv(out);
    if (!csv) throw std::runtime_error("bucket-regret: cannot write " + out);
    csv << "# snapshot=" << o["--snapshot"] << ",partition=" << o["--partition"] << ",roots=" << roots << ",seed=" << seed << ",min_visits=" << min_visits
        << ",r0=" << std::setprecision(9) << r0 << "\n";
    csv << "bucket,visits,regret_per_visit,cross_per_visit,plain_per_visit,best_action,confidence\n";
    std::array<int, 5> band{};  // confidence >= 0.8, 0.6, 0.4, 0.2, below
    double total_cross = 0, total_plain = 0, visits_all = 0, visits_scored = 0, regret_scored = 0;
    for (const auto& r : rows) {
        const double reg = std::max(0.0, r.cross);
        csv << r.b << ',' << r.visits << ',' << reg << ',' << r.cross << ',' << r.plain << ','
            << (r.best >= 0 ? kActionNames[r.best] : "none") << ',';
        if (r.visits >= min_visits) {
            const double c = r0 / (r0 + reg);
            csv << c;
            ++band[c >= 0.8 ? 0 : c >= 0.6 ? 1 : c >= 0.4 ? 2 : c >= 0.2 ? 3 : 4];
        }
        csv << '\n';
        total_cross += r.cross * r.visits; total_plain += r.plain * r.visits;
        visits_all += r.visits;
        if (r.visits >= min_visits) { visits_scored += r.visits; regret_scored += reg * r.visits; }
    }
    std::cout << std::setprecision(6) << "{\"out\":\"" << json_escape(out) << "\",\"partition\":\"" << o["--partition"] << "\",\"roots\":" << roots << ",\"buckets_seen\":" << rows.size()
              << ",\"buckets_scored\":" << enough.size() << ",\"model_buckets\":" << model.bucket_count() << ",\"r0\":" << r0
              << ",\"rho_cross_half\":" << total_cross / (2.0 * roots) << ",\"rho_plain_half\":" << total_plain / (2.0 * roots)
              << ",\"visits\":" << visits_all << ",\"scored_visit_share\":" << (visits_all > 0 ? visits_scored / visits_all : 0.0)
              << ",\"regret_per_visit_scored\":" << (visits_scored > 0 ? regret_scored / visits_scored : 0.0)
              << ",\"confidence_bands\":{\">=0.8\":" << band[0] << ",\"0.6-0.8\":" << band[1] << ",\"0.4-0.6\":" << band[2]
              << ",\"0.2-0.4\":" << band[3] << ",\"<0.2\":" << band[4] << "}}\n";
    return 0;
}

// ---- h-profile: where the cancellation of a history-key snapshot sits (posterior keys vs card resolution) -----------
// ./aswiss_stud h-profile --atlas ATLAS --snapshot FILE [--roots 300000] [--seed 2026111002] [--max-units 20000000]
// The average-policy pass of bucket-regret (both seats play the snapshot's average policy; the traverser expands its
// actions; roots split into two halves by parity). Every traverser node adds its sampled counterfactual regrets to one
// unit at each of five levels, all refinements of the bucket:
//   L0 bucket                      the posterior bucket (what the policy is stored on)
//   L1 key                         the bucket-history key (the label unit: street, legal mask, previous own bucket and
//                                  action, opponent actions since, power cluster at a street's first decision)
//   L2 bucket x cluster            the bucket and the power cluster (codebook cell) of the actor's cards at this decision
//   L3 bucket x cluster x made     plus the category and primary rank of the actor's best hand (card information finer
//                                  than the codebook cell)
//   L4 key x cluster               both refinements together
// For each level: S_L = sum over units of psi(R_u) per root and per seat (rho/2 at that resolution), plain and
// cross-fitted (one half picks a unit's action, the other values it). Every level refines the bucket, so S_L >= S_0
// (subadditivity of psi) and the differences are the cancellation inside buckets that the refinement exposes:
//   S_1 - S_0 = H_post        between history keys (what further posterior splits could remove)
//   S_2 - S_0 = H_cluster     between codebook cells
//   S_3 - S_2 = H_beyond      inside codebook cells, from finer card information (what splitting the codebook would add)
// In expectation the plain value is at least the true S_L (noise lifts the maximum) and the cross-fitted value at most
// it (a unit's action is picked on one half and valued on the other; a unit seen in one half only adds 0). So the
// plain difference between two levels is the optimistic cancellation, and the cross-fitted difference is what a
// refinement chosen on one half actually gains on the other: with few visits per fine unit it can be negative,
// meaning that at this sample size the finer labels are too noisy to help. `both_halves_share` is the share of visits
// in units seen in both halves. Per-street values follow. The model is not changed.
template <class V> struct ProfileLevel {
    std::unordered_map<uint64_t, std::array<V, 2 * A + 2>> units;
    std::array<std::unordered_map<uint64_t, uint8_t>, 1> street_of;  // unit -> street (5..7)
    double dropped = 0;
};
inline uint64_t pmix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 29;
    return h;
}
struct ProfileState {
    ProfileLevel<double> l0, l1, l2;
    ProfileLevel<float> l3, l4;  // the finest levels: float sums halve their memory
    size_t cap = 20000000;
};
template <class V>
void profile_add(ProfileLevel<V>& lv, uint64_t id, int street, const std::array<double, A>& r, uint8_t mask, int half, size_t cap) {
    auto it = lv.units.find(id);
    if (it == lv.units.end()) {
        if (lv.units.size() >= cap) { lv.dropped += 1; return; }
        it = lv.units.emplace(id, std::array<V, 2 * A + 2>{}).first;
        lv.street_of[0][id] = static_cast<uint8_t>(street);
    }
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) it->second[half * A + a] += static_cast<V>(r[a]);
    it->second[2 * A + half] += 1;
}
double hprof_traverse(const Model& model, const PowerAtlas& atlas, State state, Trail trail, int traverser, std::mt19937_64& rng,
                      ProfileState& ps, int half) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const HFull key = model.hkey(state, actor, trail);
    const uint32_t b = model.hlookup(key), seen = b == kNone ? kUnseen : b;
    const auto avg = model.bucket_policy(b, mask, true);
    if (actor != traverser) {
        const Action a = sample_action(avg, mask, rng);
        trail_step(trail, state, actor, a, seen);
        step(state, actor, a);
        return hprof_traverse(model, atlas, std::move(state), trail, traverser, rng, ps, half);
    }
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        Trail child_trail = trail;
        trail_step(child_trail, state, actor, static_cast<Action>(a), seen);
        step(child, actor, static_cast<Action>(a));
        v[a] = hprof_traverse(model, atlas, std::move(child), child_trail, traverser, rng, ps, half);
        node += avg[a] * v[a];
    }
    std::array<double, A> r{};
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) r[a] = v[a] - node;
    const int street = state.street;
    // unit ids of the five levels (the street is part of every bucket and key, so of every unit)
    const uint64_t id0 = b == kNone ? 0xffffffffffffull : b;
    const uint64_t id1 = pmix(pmix(pmix(pmix(pmix(1, key.ctx), key.m.prev), key.m.action), key.m.cluster), key.m.opp);
    const uint64_t cluster = 1 + atlas.assign(state, actor);
    const uint64_t id2 = pmix(pmix(2, id0), cluster);
    const Score best = best_hand(all_cards(state.players[actor]));
    const uint64_t id3 = pmix(pmix(id2, 100 + best[0]), 200 + best[1]);
    const uint64_t id4 = pmix(pmix(4, id1), cluster);
    profile_add(ps.l0, id0, street, r, mask, half, ps.cap);
    profile_add(ps.l1, id1, street, r, mask, half, ps.cap);
    profile_add(ps.l2, id2, street, r, mask, half, ps.cap);
    profile_add(ps.l3, id3, street, r, mask, half, ps.cap);
    profile_add(ps.l4, id4, street, r, mask, half, ps.cap);
    return node;
}
// per street (index 0: all, 1..3: streets 5..7): plain and cross-fitted psi totals, visits, visits in units seen in both halves
template <class V> void profile_totals(const ProfileLevel<V>& lv, std::array<double, 4>& plain, std::array<double, 4>& cross,
                                       std::array<double, 4>& visits, std::array<double, 4>& both) {
    plain.fill(0); cross.fill(0); visits.fill(0); both.fill(0);
    for (const auto& [id, h] : lv.units) {
        double best = 0, m0 = 0, m1 = 0;
        int a0 = -1, a1 = -1;
        for (int a = 0; a < A; ++a) {
            best = std::max(best, static_cast<double>(h[a]) + h[A + a]);
            if (h[a] > m0) { m0 = h[a]; a0 = a; }
            if (h[A + a] > m1) { m1 = h[A + a]; a1 = a; }
        }
        const double c = (a0 >= 0 ? static_cast<double>(h[A + a0]) : 0.0) + (a1 >= 0 ? static_cast<double>(h[a1]) : 0.0);
        const double vis = static_cast<double>(h[2 * A]) + h[2 * A + 1];
        const bool two = h[2 * A] > 0 && h[2 * A + 1] > 0;
        const int st = lv.street_of[0].at(id);
        for (int k : {0, st - 4}) {
            plain[k] += best; cross[k] += c; visits[k] += vis;
            if (two) both[k] += vis;
        }
    }
}
int h_profile(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--atlas", ""}, {"--snapshot", ""}, {"--roots", "300000"}, {"--seed", "2026111002"},
                                {"--max-units", "20000000"}, {"--power-cache", "1000000"}});
    const auto atlas = load_atlas(o.at("--atlas"), std::stoull(o["--power-cache"]));
    if (!atlas) throw std::runtime_error("h-profile needs --atlas (the run's)");
    Model model(Partition::History, atlas.get());
    model.load(o.at("--snapshot"));
    if (model.has_h4()) throw std::runtime_error("h-profile: snapshots with a learned H4 are not supported");
    const uint64_t roots = std::stoull(o["--roots"]), seed = std::stoull(o["--seed"]);
    ProfileState ps;
    ps.cap = std::stoull(o["--max-units"]);
    std::mt19937_64 rng(seed);
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < roots; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        for (int p = 0; p < 2; ++p)
            hprof_traverse(model, *atlas, sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte), Trail{}, p, rng, ps, static_cast<int>(i & 1));
        if ((i + 1) % 50000 == 0 || i + 1 == roots) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cerr << "root " << i + 1 << "/" << roots << " (" << std::fixed << std::setprecision(0) << sec << " s, about "
                      << sec / (i + 1) * (roots - i - 1) << " s left), units L0-L4 " << ps.l0.units.size() << " " << ps.l1.units.size() << " "
                      << ps.l2.units.size() << " " << ps.l3.units.size() << " " << ps.l4.units.size() << "\n" << std::defaultfloat;
        }
    }
    const double per = 2.0 * static_cast<double>(roots);  // per root and seat (rho/2)
    std::ostringstream js;
    js << std::setprecision(6) << "{\"snapshot\":\"" << json_escape(o["--snapshot"]) << "\",\"roots\":" << roots << ",\"seed\":" << seed
       << ",\"model_buckets\":" << model.bucket_count() << ",\"levels\":[";
    const char* names[5] = {"bucket", "key", "bucket_x_cluster", "bucket_x_cluster_x_made_hand", "key_x_cluster"};
    std::array<std::array<double, 4>, 5> P{}, C{}, Vv{}, Bt{};
    std::array<size_t, 5> units{};
    std::array<double, 5> dropped{};
    profile_totals(ps.l0, P[0], C[0], Vv[0], Bt[0]); units[0] = ps.l0.units.size(); dropped[0] = ps.l0.dropped;
    profile_totals(ps.l1, P[1], C[1], Vv[1], Bt[1]); units[1] = ps.l1.units.size(); dropped[1] = ps.l1.dropped;
    profile_totals(ps.l2, P[2], C[2], Vv[2], Bt[2]); units[2] = ps.l2.units.size(); dropped[2] = ps.l2.dropped;
    profile_totals(ps.l3, P[3], C[3], Vv[3], Bt[3]); units[3] = ps.l3.units.size(); dropped[3] = ps.l3.dropped;
    profile_totals(ps.l4, P[4], C[4], Vv[4], Bt[4]); units[4] = ps.l4.units.size(); dropped[4] = ps.l4.dropped;
    for (int l = 0; l < 5; ++l) {
        js << (l ? "," : "") << "{\"level\":\"" << names[l] << "\",\"units\":" << units[l] << ",\"visits\":" << Vv[l][0]
           << ",\"dropped_visits\":" << dropped[l] << ",\"both_halves_share\":" << (Vv[l][0] > 0 ? Bt[l][0] / Vv[l][0] : 0.0)
           << ",\"plain\":" << P[l][0] / per << ",\"cross\":" << C[l][0] / per << ",\"by_street\":{";
        for (int k = 1; k <= 3; ++k)
            js << (k > 1 ? "," : "") << "\"" << k + 4 << "\":{\"plain\":" << P[l][k] / per << ",\"cross\":" << C[l][k] / per
               << ",\"both_halves_share\":" << (Vv[l][k] > 0 ? Bt[l][k] / Vv[l][k] : 0.0) << "}";
        js << "}}";
    }
    js << "],\"H_post\":{\"plain\":" << (P[1][0] - P[0][0]) / per << ",\"cross\":" << (C[1][0] - C[0][0]) / per << "}"
       << ",\"H_cluster\":{\"plain\":" << (P[2][0] - P[0][0]) / per << ",\"cross\":" << (C[2][0] - C[0][0]) / per << "}"
       << ",\"H_beyond_codebook\":{\"plain\":" << (P[3][0] - P[2][0]) / per << ",\"cross\":" << (C[3][0] - C[2][0]) / per << "}"
       << ",\"H_key_and_cluster\":{\"plain\":" << (P[4][0] - P[0][0]) / per << ",\"cross\":" << (C[4][0] - C[0][0]) / per << "}"
       << ",\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "}\n";
    std::cout << js.str();
    return 0;
}

#include "sota/stud7_net_main.hpp"
#include "sota/rebel_main.hpp"

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

// ---- serve: a frozen snapshot as the web game's agent (agent/aswiss_agent.py) -------------------------------------
// The line protocol of stud_mccfr --agent-stdio: READY, then per line DISCARD <4 cards> (heuristic H4), ACT <the same 15
// tab-separated fields> or QUIT. Additions:
//  - an optional 17th field: the viewer's position among the two players in table order (0 first, 1 second). The
//    viewer then takes that seat, so a tie in the public-card priority goes to the same player as in the web game
//    (poker_env: the lower seat; the trainer: seat 0).
//  - EXPLAIN (same fields): one JSON line with the policy, the action sampled from it, and (history partition) the key,
//    the bucket it resolves to and the bucket's split genealogy.
//  - INFO: the loaded snapshot.
// History keys are computed by replaying the hand from its 5th-street root (as eval does); a request whose replay does
// not reproduce the given state (street, pot, bets, actor) is refused with ERROR rather than answered from the key of
// a different hand.
State seat_view(State s, int seat) {
    if (seat == 0) return s;
    std::swap(s.players[0], s.players[1]);
    for (Event& e : s.history) e.actor = static_cast<uint8_t>(1 - e.actor);
    s.actor = 1;
    return s;
}
std::string street_name(int street) { return street == 5 ? "5th" : street == 6 ? "6th" : "7th_hidden"; }
std::string policy_json(const std::array<double, A>& p, uint8_t mask) {
    std::ostringstream o; o << '{';
    bool first = true;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        o << (first ? "" : ",") << '"' << kActionNames[a] << "\":" << std::setprecision(6) << p[a];
        first = false;
    }
    o << '}';
    return o.str();
}
int serve(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "history"}, {"--atlas", ""}, {"--snapshot", ""}, {"--policy", "average"},
                                {"--seed", "7"}, {"--power-cache", "1000000"}, {"--bucket-regret", ""}});
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    Model model(parse_partition(o["--partition"]), atlas.get());
    model.load(o.at("--snapshot"));
    const bool average = o["--policy"] == "average";
    if (!average && o["--policy"] != "current") throw std::runtime_error("--policy must be average or current");
    const bool history = model.partition() == Partition::History;
    std::mt19937_64 rng(std::stoull(o["--seed"]));
    // --bucket-regret FILE (bucket-regret output): each bucket's regret per visit and confidence, reported by EXPLAIN
    struct RegretRow { double visits, regret, cross, plain; std::string best, confidence; };
    std::unordered_map<uint32_t, RegretRow> regret_rows;
    std::string regret_meta;
    const bool with_regret = !o["--bucket-regret"].empty();
    if (with_regret) {
        std::ifstream in(o["--bucket-regret"]);
        if (!in) throw std::runtime_error("cannot read --bucket-regret " + o["--bucket-regret"]);
        std::string row;
        while (std::getline(in, row)) {
            if (!row.empty() && row.back() == '\r') row.pop_back();
            if (row.empty()) continue;
            if (row[0] == '#') { regret_meta = row.substr(1); continue; }
            if (row.rfind("bucket,", 0) == 0) continue;
            auto f = split(row, ',');
            f.resize(7);
            regret_rows[static_cast<uint32_t>(std::stoul(f[0]))] = {std::stod(f[1]), std::stod(f[2]), std::stod(f[3]), std::stod(f[4]), f[5], f[6]};
        }
    }
    auto meta_value = [&](const std::string& name) {  // r0, roots, ... from the comment line
        const auto at = regret_meta.find(name + "=");
        if (at == std::string::npos) return std::string();
        const auto end = regret_meta.find(',', at);
        return regret_meta.substr(at + name.size() + 1, end == std::string::npos ? std::string::npos : end - at - name.size() - 1);
    };
    std::cout << "READY\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        try {
            if (line == "QUIT") return 0;
            if (line == "INFO") {
                std::cout << "INFO\t{\"snapshot\":\"" << o["--snapshot"] << "\",\"partition\":\"" << o["--partition"]
                          << "\",\"policy\":\"" << o["--policy"] << "\",\"buckets\":" << model.bucket_count()
                          << ",\"history_contexts\":" << model.history_context_count() << ",\"history_cards\":\""
                          << (model.history_cards_every() ? "every" : "street") << "\",\"training_ante\":" << kAnte
                          << ",\"training_stack_ante\":" << kEffectiveStackAnte << "}\n";
            } else if (line.rfind("DISCARD\t", 0) == 0) {
                const auto cards = parse_cards(line.substr(8));
                if (cards.size() != 4) throw std::runtime_error("DISCARD requires four cards");
                const auto [discard, reveal] = discard_reveal(cards);
                std::cout << "DISCARD\t" << discard << "\t" << reveal << "\n";
            } else {
                auto fields = split(line, '\t');
                if (fields.empty() || (fields[0] != "ACT" && fields[0] != "EXPLAIN"))
                    throw std::runtime_error("unknown request: " + fields[0]);
                const bool explain = fields[0] == "EXPLAIN";
                int seat = 0;
                if (fields.size() == 17) { seat = std::stoi(fields.back()); fields.pop_back(); }
                if (seat != 0 && seat != 1) throw std::runtime_error("viewer position must be 0 or 1");
                fields[0] = "ACT";
                std::string request;
                for (size_t i = 0; i < fields.size(); ++i) request += (i ? "\t" : "") + fields[i];
                const AgentRequest parsed = parse_agent_request(request);  // checks the legal mask (viewer at seat 0)
                const State state = seat_view(parsed.state, seat);
                const uint8_t mask = valid_mask(state, seat);
                if (mask != parsed.expected_mask) throw std::runtime_error("legal action mask mismatch after the seat swap");
                Model::Resolution res;
                std::array<double, A> policy;
                if (history) {
                    res = model.resolve_history(state, seat);
                    const State& r = res.replayed;
                    std::ostringstream diff;
                    if (r.terminal) diff << " replay ended the hand;";
                    if (r.street != state.street) diff << " street " << r.street << " vs " << state.street << ';';
                    if (r.actor != seat) diff << " actor " << r.actor << " vs viewer " << seat << ';';
                    if (r.pot != state.pot) diff << " pot " << r.pot << " vs " << state.pot << ';';
                    if (r.highest_bet != state.highest_bet) diff << " highest bet " << r.highest_bet << " vs " << state.highest_bet << ';';
                    for (int i = 0; i < 2; ++i)
                        if (r.players[i].invested != state.players[i].invested || r.players[i].round_bet != state.players[i].round_bet)
                            diff << " seat " << i << " invested/round " << r.players[i].invested << '/' << r.players[i].round_bet
                                 << " vs " << state.players[i].invested << '/' << state.players[i].round_bet << ';';
                    if (!diff.str().empty()) throw std::runtime_error("replay does not reproduce the state:" + diff.str());
                    policy = model.bucket_policy(res.bucket, mask, average);
                } else {
                    policy = model.strategy(state, seat, average);
                }
                const Action action = sample_action(policy, mask, rng);
                if (!explain) { std::cout << "ACTION\t" << kActionNames[action] << "\n"; std::cout << std::flush; continue; }
                std::ostringstream js;
                js << "{\"action\":\"" << kActionNames[action] << "\",\"street\":\"" << street_name(state.street)
                   << "\",\"seat\":" << seat << ",\"policy\":\"" << o["--policy"] << "\",\"probabilities\":" << policy_json(policy, mask)
                   << ",\"stack_matches_training\":" << (state.effective_stack == state.ante * kEffectiveStackAnte ? "true" : "false");
                if (history) {
                    const uint32_t b = res.bucket;
                    js << ",\"resolved\":\"" << (!res.context ? "unseen_context" : !res.member ? "rest" : res.hops ? "ancestor" : "exact")
                       << "\",\"hops\":" << res.hops << ",\"bucket\":";
                    if (b == kNone) js << "null"; else js << b;
                    const HKey& m = res.key.m;
                    js << ",\"key\":{\"street\":" << (res.key.ctx >> 8) << ",\"legal_mask\":" << (res.key.ctx & 0xff) << ",\"prev_bucket\":";
                    if (m.prev == kNone) js << "null"; else if (m.prev == kUnseen) js << "\"unseen\""; else js << m.prev;
                    js << ",\"prev_action\":";
                    if (m.action == 0xff) js << "null"; else js << '"' << kActionNames[m.action] << '"';
                    js << ",\"cluster\":";
                    if (m.cluster == kNoCards) js << "null"; else js << m.cluster;
                    std::vector<std::string> opp;
                    for (uint64_t x = m.opp; x; x >>= 5) {
                        const int tok = static_cast<int>(x & 31) - 1;
                        opp.insert(opp.begin(), street_name(5 + tok / 8) + ":" + kActionNames[tok % 8]);
                    }
                    js << ",\"opponent_actions\":[";
                    for (size_t i = 0; i < opp.size(); ++i) js << (i ? "," : "") << '"' << opp[i] << '"';
                    js << "]}";
                    if (b != kNone) {
                        const Bucket& bk = model.bucket_at(b);
                        double weight = 0;
                        for (int a = 0; a < A; ++a) weight += bk.s[a];
                        js << ",\"average\":" << policy_json(model.bucket_policy(b, mask, true), mask)
                           << ",\"current\":" << policy_json(model.bucket_policy(b, mask, false), mask)
                           << ",\"average_weight\":" << std::setprecision(6) << weight << ",\"genealogy\":[";
                        int depth = 0;
                        for (uint32_t x = b; x != kNone; x = model.parent_of(x)) js << (depth++ ? "," : "") << x;
                        js << "]";
                    }
                    if (with_regret) {
                        const auto it = b == kNone ? regret_rows.end() : regret_rows.find(b);
                        js << ",\"regret\":";
                        if (it == regret_rows.end()) {
                            js << "{\"visits\":0,\"confidence\":null,\"r0\":" << (meta_value("r0").empty() ? "null" : meta_value("r0"))
                               << ",\"pass_roots\":" << (meta_value("roots").empty() ? "null" : meta_value("roots")) << "}";
                        } else {
                            const RegretRow& r = it->second;
                            js << std::setprecision(6) << "{\"visits\":" << r.visits << ",\"per_visit\":" << r.regret
                               << ",\"cross_per_visit\":" << r.cross << ",\"plain_per_visit\":" << r.plain << ",\"best_action\":\""
                               << r.best << "\",\"confidence\":" << (r.confidence.empty() ? "null" : r.confidence) << ",\"r0\":"
                               << (meta_value("r0").empty() ? "null" : meta_value("r0")) << ",\"min_visits\":"
                               << (meta_value("min_visits").empty() ? "null" : meta_value("min_visits")) << ",\"pass_roots\":"
                               << (meta_value("roots").empty() ? "null" : meta_value("roots")) << "}";
                        }
                    }
                }
                js << "}";
                std::cout << "EXPLAIN\t" << js.str() << "\n";
            }
        } catch (const std::exception& error) {
            std::cout << "ERROR\t" << error.what() << "\n";
        }
        std::cout << std::flush;
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "train") return train(argc, argv);
        if (mode == "eval") return eval(argc, argv);
        if (mode == "self-test") return self_test(argc, argv);
        if (mode == "serve") return serve(argc, argv);
        if (mode == "league") return league(argc, argv);
        if (mode == "bucket-regret") return bucket_regret(argc, argv);
        if (mode == "h-profile") return h_profile(argc, argv);
        if (mode == "pp-test") return pp_test(argc, argv);
        throw std::runtime_error("usage: aswiss_stud train|eval|self-test|serve [options]");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
