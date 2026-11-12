// Prior-posterior ASWISS on 7-stud (PP_STUD7_DESIGN_KO.md sections 3-5): observation streams with canonical suits, the
// general features F1-F4 (+ the poker option), compact samples of information states, and the particle estimate of an
// information state's direction. Included inside the anonymous namespace of aswiss_stud.cpp, after the engine and A.
//
// Observation stream of the viewer v (all "mine" = v's), as (channel, round, order, symbol):
//   round 0 (4th-street cards): PC my four initial cards (sorted), HO the opponent's four (?), AM my discard,
//                               PM my revealed card, PO the opponent's revealed card
//   round 1 (5th street): UM my two new up cards, UO the opponent's two, then the 5th-street bets
//   round 2 (6th street): UM / UO one up card each, then the bets
//   round 3 (7th street): PC my hidden card, HO the opponent's (?), then the bets
//   bets: PM if I acted, PO if the opponent did, symbol = the action. Order counts chance observations and action
//   observations of the round separately (the two players' bets share one counter).
// Symbols: card = 4 (rank - 2) + canonical suit (0..51), action = 52 + action (52..59), unknown = 60. Suits are an
// exchangeable attribute: the stream uses, of the 24 suit permutations, the one that makes the sequence of card symbols
// (round-0 private cards sorted) lexicographically smallest.
#pragma once

namespace pp {

enum Channel : int { PC = 0, UM, UO, UC, HO, AM, PM, PO };
inline bool chance_ch(int c) { return c <= HO; }
constexpr int kUnknown = 60, kAct0 = 52;
inline bool is_card_sym(int s) { return s < 52; }
inline int sym_rank(int s) { return s / 4 + 2; }
inline int sym_suit(int s) { return s % 4; }
inline uint32_t pack(int ch, int rnd, int ord, int sym) { return uint32_t(ch) << 15 | uint32_t(rnd) << 13 | uint32_t(ord) << 7 | uint32_t(sym); }
inline int o_ch(uint32_t o) { return o >> 15 & 7; }
inline int o_rnd(uint32_t o) { return o >> 13 & 3; }
inline int o_ord(uint32_t o) { return o >> 7 & 63; }
inline int o_sym(uint32_t o) { return o & 127; }

struct Stream {
    std::array<uint32_t, 64> o{};
    int n = 0, seat = 0, pot = 0;  // pot in antes (poker option only)
};

const std::array<std::array<int, 4>, 24>& suit_perms() {
    static const auto perms = [] {
        std::array<std::array<int, 4>, 24> p{};
        std::array<int, 4> x{0, 1, 2, 3};
        int i = 0;
        do { p[i++] = x; } while (std::next_permutation(x.begin(), x.end()));
        return p;
    }();
    return perms;
}

// the viewer's observation stream (canonical suits)
Stream make_stream(const State& s, int v) {
    const Player& me = s.players[v];
    const Player& op = s.players[1 - v];
    if (!me.has_discard || me.hidden.size() < 2 || me.shown.size() < 3 || op.shown.size() < 3)
        throw std::runtime_error("pp stream: a decision before the 5th-street deal");
    struct Item { int ch, rnd, ord; const Card* card; int sym; };  // card == nullptr: symbol given
    std::array<Item, 64> it{};
    int n = 0, cc = 0, ac = 0, rnd = 0;
    auto chance = [&](int ch, const Card* c, int sym = 0) { it[n++] = {ch, rnd, cc++, c, sym}; };
    auto action = [&](int ch, const Card* c, int sym = 0) { it[n++] = {ch, rnd, ac++, c, sym}; };
    std::array<const Card*, 4> init{&me.hidden[0], &me.hidden[1], &me.discarded, &me.shown[0]};
    const int init_at = n;
    for (const Card* c : init) chance(PC, c);
    for (int k = 0; k < 4; ++k) chance(HO, nullptr, kUnknown);
    action(AM, &me.discarded); action(PM, &me.shown[0]); action(PO, &op.shown[0]);
    auto bets = [&](int street) {
        for (const Event& e : s.history)
            if (e.street == street) action(e.actor == v ? PM : PO, nullptr, kAct0 + e.action);
    };
    rnd = 1; cc = ac = 0;
    chance(UM, &me.shown[1]); chance(UM, &me.shown[2]); chance(UO, &op.shown[1]); chance(UO, &op.shown[2]);
    bets(5);
    if (s.street >= 6) {
        rnd = 2; cc = ac = 0;
        chance(UM, &me.shown[3]); chance(UO, &op.shown[3]);
        bets(6);
    }
    if (s.street >= 7) {
        rnd = 3; cc = ac = 0;
        chance(PC, &me.hidden[2]); chance(HO, nullptr, kUnknown);
        bets(7);
    }
    // canonical suit permutation: smallest card-symbol sequence, the four initial private cards sorted (descending).
    // It depends on the viewer's cards only, so it is memoized per card configuration (thread-local direct-mapped cache;
    // a cache hit returns exactly what the search would).
    std::array<uint8_t, 16> ckey;
    ckey.fill(0xff);
    for (int i = 0, k = 0; i < n; ++i) if (it[i].card) ckey[k++] = static_cast<uint8_t>(it[i].card->suit * 13 + it[i].card->rank - 2);
    struct PermEntry { std::array<uint8_t, 16> key; std::array<int8_t, 4> perm; bool used = false; };
    static thread_local std::vector<PermEntry> cache(1 << 14);
    uint64_t ch = 1469598103934665603ull;
    for (uint8_t b : ckey) { ch ^= b; ch *= 1099511628211ull; }
    PermEntry& slot = cache[(ch ^ (ch >> 29)) & ((1 << 14) - 1)];
    std::array<int, 4> best_perm{0, 1, 2, 3};
    if (slot.used && slot.key == ckey) {
        for (int k = 0; k < 4; ++k) best_perm[k] = slot.perm[k];
    } else {
    std::array<int, 16> best_key{}, key{};
    int nk = 0;
    bool first = true;
    for (const auto& perm : suit_perms()) {
        std::array<int, 4> init_sym;
        for (int k = 0; k < 4; ++k) init_sym[k] = 4 * (init[k]->rank - 2) + perm[init[k]->suit];
        std::sort(init_sym.begin(), init_sym.end(), std::greater<int>());
        nk = 0;
        for (int k = 0; k < 4; ++k) key[nk++] = init_sym[k];
        for (int i = 0; i < n; ++i)
            if (it[i].card && !(i >= init_at && i < init_at + 4)) key[nk++] = 4 * (it[i].card->rank - 2) + perm[it[i].card->suit];
        if (first || std::lexicographical_compare(key.begin(), key.begin() + nk, best_key.begin(), best_key.begin() + nk)) {
            best_key = key; best_perm = perm; first = false;
        }
    }
    slot.used = true; slot.key = ckey;
    for (int k = 0; k < 4; ++k) slot.perm[k] = static_cast<int8_t>(best_perm[k]);
    }
    Stream st;
    st.seat = v; st.pot = s.pot / std::max(1, s.ante);
    std::array<int, 4> init_sym;
    for (int k = 0; k < 4; ++k) init_sym[k] = 4 * (init[k]->rank - 2) + best_perm[init[k]->suit];
    std::sort(init_sym.begin(), init_sym.end(), std::greater<int>());
    for (int i = 0; i < n; ++i) {
        int sym = it[i].sym;
        if (i >= init_at && i < init_at + 4) sym = init_sym[i - init_at];
        else if (it[i].card) sym = 4 * (it[i].card->rank - 2) + best_perm[it[i].card->suit];
        st.o[st.n++] = pack(it[i].ch, it[i].rnd, it[i].ord, sym);
    }
    return st;
}

// ---- features (PRIOR_FEATURES_KO.md, with card attributes: 0 = rank, 1 = suit) ----
enum Kind : int8_t {
    F1 = 1,   // slot (ch, rnd, ord) holds symbol a
    F1A,      // slot (ch, rnd, ord) holds a card whose attribute a is b
    F2,       // count of symbol a in channel ch
    F2A,      // count of cards with attribute a = b in channel ch
    F2ALL,    // count of symbol a in the chance channels
    F2ALLA,   // count of cards with attribute a = b in the chance channels
    F2SORT,   // suits (exchangeable): the a-th largest per-suit card count in channel ch (-1: chance channels)
    F3CNT,    // round rnd: count of action a in channel ch (PM / PO)
    F3TOT,    // round rnd: number of action observations (AM, PM, PO)
    F3LAST,   // the last observation of channel ch (PM / PO) is symbol a
    F4SEAT, F4TOT, F4CH, F4RND, F4CUR,
    PHAND,    // poker: count of rank a in my hand (PC - AM + UM)
    PPUB,     // poker: count of rank a among my up cards (UM + revealed)
    PMAX, PPAIRS, PPOT
};
struct Desc {
    int8_t kind = 0, ch = -1, rnd = -1, ord = -1;
    int8_t a = -1, b = -1;
    auto tie() const { return std::make_tuple(kind, ch, rnd, ord, a, b); }
    bool operator<(const Desc& o) const { return tie() < o.tie(); }
    bool operator==(const Desc& o) const { return tie() == o.tie(); }
};
std::string sym_name(int s) {
    static const char* rk = "23456789TJQKA";
    static const char* act[] = {"check", "bbing", "ddadang", "quarter", "half", "full", "call", "fold"};
    if (s == kUnknown) return "?";
    if (s >= kAct0) return act[s - kAct0];
    return std::string(1, rk[sym_rank(s) - 2]) + "s" + std::to_string(sym_suit(s));
}
std::string desc_name(const Desc& d) {
    static const char* chn[] = {"PC", "UM", "UO", "UC", "HO", "AM", "PM", "PO"};
    static const char* rk = "23456789TJQKA";
    auto ch = [&](int c) { return c >= 0 && c < 8 ? std::string(chn[c]) : std::string("chance"); };
    auto attr = [&](int a, int b) { return a == 0 ? std::string("rank=") + rk[b - 2] : "suit=" + std::to_string(b); };
    std::ostringstream o;
    switch (d.kind) {
        case F1: o << "F1(" << ch(d.ch) << "," << int(d.rnd) << "," << int(d.ord) << ")=" << sym_name(d.a); break;
        case F1A: o << "F1(" << ch(d.ch) << "," << int(d.rnd) << "," << int(d.ord) << ")." << attr(d.a, d.b); break;
        case F2: o << "F2(" << ch(d.ch) << "," << sym_name(d.a) << ")"; break;
        case F2A: o << "F2(" << ch(d.ch) << "," << attr(d.a, d.b) << ")"; break;
        case F2ALL: o << "F2(chance," << sym_name(d.a) << ")"; break;
        case F2ALLA: o << "F2(chance," << attr(d.a, d.b) << ")"; break;
        case F2SORT: o << "F2sorted(" << ch(d.ch) << ",suit," << int(d.a) << ")"; break;
        case F3CNT: o << "F3(" << int(d.rnd) << "," << ch(d.ch) << "," << sym_name(d.a) << ")"; break;
        case F3TOT: o << "F3total(" << int(d.rnd) << ")"; break;
        case F3LAST: o << "F3last(" << ch(d.ch) << ")=" << sym_name(d.a); break;
        case F4SEAT: o << "F4seat"; break;
        case F4TOT: o << "F4observations"; break;
        case F4CH: o << "F4chance"; break;
        case F4RND: o << "F4round"; break;
        case F4CUR: o << "F4round_actions"; break;
        case PHAND: o << "Phand(rank=" << rk[d.a - 2] << ")"; break;
        case PPUB: o << "Ppublic(rank=" << rk[d.a - 2] << ")"; break;
        case PMAX: o << "Pmax_same"; break;
        case PPAIRS: o << "Ppairs"; break;
        case PPOT: o << "Ppot"; break;
    }
    return o.str();
}
inline int attr_of(int sym, int a) { return a == 0 ? sym_rank(sym) : sym_suit(sym); }
std::array<int, 4> suit_counts(const Stream& s, int ch) {
    std::array<int, 4> c{};
    for (int k = 0; k < s.n; ++k) {
        const int x = o_ch(s.o[k]), y = o_sym(s.o[k]);
        if (is_card_sym(y) && (ch >= 0 ? x == ch : chance_ch(x))) ++c[sym_suit(y)];
    }
    std::sort(c.begin(), c.end(), std::greater<int>());
    return c;
}
std::array<int, 15> hand_ranks(const Stream& s) {
    std::array<int, 15> h{};
    for (int k = 0; k < s.n; ++k) {
        const int x = o_ch(s.o[k]), y = o_sym(s.o[k]);
        if (!is_card_sym(y)) continue;
        if (x == PC || x == UM) ++h[sym_rank(y)];
        else if (x == AM) --h[sym_rank(y)];
    }
    return h;
}
// the value of feature d on a stream (every feature is 0 when what it counts is absent)
double feat(const Stream& s, const Desc& d) {
    int c = 0;
    switch (d.kind) {
        case F1:
            for (int k = 0; k < s.n; ++k)
                if (o_ch(s.o[k]) == d.ch && o_rnd(s.o[k]) == d.rnd && o_ord(s.o[k]) == d.ord) return o_sym(s.o[k]) == d.a;
            return 0;
        case F1A:
            for (int k = 0; k < s.n; ++k)
                if (o_ch(s.o[k]) == d.ch && o_rnd(s.o[k]) == d.rnd && o_ord(s.o[k]) == d.ord)
                    return is_card_sym(o_sym(s.o[k])) && attr_of(o_sym(s.o[k]), d.a) == d.b;
            return 0;
        case F2: for (int k = 0; k < s.n; ++k) c += o_ch(s.o[k]) == d.ch && o_sym(s.o[k]) == d.a; return c;
        case F2A:
            for (int k = 0; k < s.n; ++k) c += o_ch(s.o[k]) == d.ch && is_card_sym(o_sym(s.o[k])) && attr_of(o_sym(s.o[k]), d.a) == d.b;
            return c;
        case F2ALL: for (int k = 0; k < s.n; ++k) c += chance_ch(o_ch(s.o[k])) && o_sym(s.o[k]) == d.a; return c;
        case F2ALLA:
            for (int k = 0; k < s.n; ++k) c += chance_ch(o_ch(s.o[k])) && is_card_sym(o_sym(s.o[k])) && attr_of(o_sym(s.o[k]), d.a) == d.b;
            return c;
        case F2SORT: return suit_counts(s, d.ch)[d.a];
        case F3CNT: for (int k = 0; k < s.n; ++k) c += o_rnd(s.o[k]) == d.rnd && o_ch(s.o[k]) == d.ch && o_sym(s.o[k]) == d.a; return c;
        case F3TOT: for (int k = 0; k < s.n; ++k) c += o_rnd(s.o[k]) == d.rnd && !chance_ch(o_ch(s.o[k])); return c;
        case F3LAST: {
            int best = -1, key = -1;
            for (int k = 0; k < s.n; ++k)
                if (o_ch(s.o[k]) == d.ch) { const int kk = o_rnd(s.o[k]) * 64 + o_ord(s.o[k]); if (kk > key) { key = kk; best = o_sym(s.o[k]); } }
            return best == d.a;
        }
        case F4SEAT: return s.seat;
        case F4TOT: return s.n;
        case F4CH: for (int k = 0; k < s.n; ++k) c += chance_ch(o_ch(s.o[k])); return c;
        case F4RND: for (int k = 0; k < s.n; ++k) c = std::max(c, o_rnd(s.o[k])); return c;
        case F4CUR: {
            int r = 0;
            for (int k = 0; k < s.n; ++k) r = std::max(r, o_rnd(s.o[k]));
            for (int k = 0; k < s.n; ++k) c += (o_ch(s.o[k]) == PM || o_ch(s.o[k]) == PO) && o_rnd(s.o[k]) == r;
            return c;
        }
        case PHAND: return hand_ranks(s)[d.a];
        case PPUB:
            for (int k = 0; k < s.n; ++k)
                c += is_card_sym(o_sym(s.o[k])) && (o_ch(s.o[k]) == UM || (o_ch(s.o[k]) == PM && o_rnd(s.o[k]) == 0)) && sym_rank(o_sym(s.o[k])) == d.a;
            return c;
        case PMAX: { const auto h = hand_ranks(s); return *std::max_element(h.begin(), h.end()); }
        case PPAIRS: { const auto h = hand_ranks(s); for (int r = 2; r <= 14; ++r) c += h[r] >= 2; return c; }
        case PPOT: return s.pot;
    }
    return 0;
}
// the features a stream can have a nonzero value on (plus those that are always defined): the candidate set of a fit
void features_of(const Stream& s, bool poker, std::vector<Desc>& out) {
    auto add = [&](int kind, int ch, int rnd, int ord, int a, int b) {
        Desc d; d.kind = int8_t(kind); d.ch = int8_t(ch); d.rnd = int8_t(rnd); d.ord = int8_t(ord); d.a = int8_t(a); d.b = int8_t(b);
        out.push_back(d);
    };
    std::array<bool, 8> card_ch{};
    for (int k = 0; k < s.n; ++k) {
        const int ch = o_ch(s.o[k]), rnd = o_rnd(s.o[k]), ord = o_ord(s.o[k]), sym = o_sym(s.o[k]);
        add(F1, ch, rnd, ord, sym, -1);
        add(F2, ch, -1, -1, sym, -1);
        if (chance_ch(ch)) add(F2ALL, -1, -1, -1, sym, -1);
        if (is_card_sym(sym)) {
            card_ch[ch] = true;
            for (int a = 0; a < 2; ++a) {
                add(F1A, ch, rnd, ord, a, attr_of(sym, a));
                add(F2A, ch, -1, -1, a, attr_of(sym, a));
                if (chance_ch(ch)) add(F2ALLA, -1, -1, -1, a, attr_of(sym, a));
            }
        }
        if (ch == PM || ch == PO) { add(F3CNT, ch, rnd, -1, sym, -1); add(F3LAST, ch, -1, -1, sym, -1); }
        if (!chance_ch(ch)) add(F3TOT, -1, rnd, -1, -1, -1);
        if (poker && is_card_sym(sym)) { add(PHAND, -1, -1, -1, sym_rank(sym), -1); add(PPUB, -1, -1, -1, sym_rank(sym), -1); }
    }
    for (int ch = 0; ch < 8; ++ch) if (card_ch[ch]) for (int k = 0; k < 4; ++k) add(F2SORT, ch, -1, -1, k, -1);
    for (int k = 0; k < 4; ++k) add(F2SORT, -1, -1, -1, k, -1);
    for (int kind : {F4SEAT, F4TOT, F4CH, F4RND, F4CUR}) add(kind, -1, -1, -1, -1, -1);
    if (poker) for (int kind : {PMAX, PPAIRS, PPOT}) add(kind, -1, -1, -1, -1, -1);
}

// ---- compact information-state samples (32 bytes) ----
// own: h0 h1 (kept initial hidden cards), discard, shown0 (revealed), shown1, shown2, shown3, hidden2; opponent: shown0-3.
// A card byte is suit * 13 + rank - 2 (0xff: not dealt yet). Actions: 4 bits each, in history order.
struct Sample {
    uint8_t seat = 0, nact = 0;
    std::array<uint8_t, 12> c{};
    std::array<uint8_t, 18> act{};
    bool operator==(const Sample& o) const { return seat == o.seat && nact == o.nact && c == o.c && act == o.act; }
};
inline uint8_t card_byte(const Card& c) { return static_cast<uint8_t>(c.suit * 13 + c.rank - 2); }
inline Card byte_card(uint8_t b) { return Card{static_cast<uint8_t>(b % 13 + 2), static_cast<uint8_t>(b / 13)}; }
Sample make_sample(const State& s, int v) {
    const Player& me = s.players[v];
    const Player& op = s.players[1 - v];
    Sample x;
    x.seat = static_cast<uint8_t>(v);
    x.c.fill(0xff);
    x.c[0] = card_byte(me.hidden[0]); x.c[1] = card_byte(me.hidden[1]); x.c[2] = card_byte(me.discarded);
    for (size_t k = 0; k < me.shown.size() && k < 4; ++k) x.c[3 + k] = card_byte(me.shown[k]);
    if (me.hidden.size() > 2) x.c[7] = card_byte(me.hidden[2]);
    for (size_t k = 0; k < op.shown.size() && k < 4; ++k) x.c[8 + k] = card_byte(op.shown[k]);
    if (s.history.size() > 36) throw std::runtime_error("pp sample: more than 36 actions");
    x.nact = static_cast<uint8_t>(s.history.size());
    for (size_t i = 0; i < s.history.size(); ++i) x.act[i / 2] |= static_cast<uint8_t>(s.history[i].action << (4 * (i % 2)));
    return x;
}
inline Action sample_action_at(const Sample& x, int i) { return static_cast<Action>(x.act[i / 2] >> (4 * (i % 2)) & 15); }
uint64_t sample_hash(const Sample& x) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t(x.seat) << 8 | x.nact);
    auto mix = [&](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 31; };
    for (uint8_t b : x.c) mix(b);
    for (uint8_t b : x.act) mix(b);
    return h;
}
int sample_street(const Sample& x) { return x.c[7] != 0xff ? 7 : x.c[6] != 0xff ? 6 : 5; }

// The state of sample x with the opponent's unseen cards of a particle: discard, two kept hidden cards, and the cards
// dealt later (own6/own7/opp6/opp7: used only where the sample has not seen them; opp7 is the opponent's 7th-street
// hidden card). The hand is replayed from its 5th-street root; `on_opponent(state, action)` sees every opponent decision.
template <class F>
State rebuild(const Sample& x, Card opp_disc, Card opp_h0, Card opp_h1, Card own6, Card own7, Card opp6, Card opp7, F&& on_opponent) {
    const int v = x.seat, o = 1 - v;
    State s;
    s.ante = kAnte;
    s.effective_stack = kAnte * kEffectiveStackAnte;
    s.pot = 2 * kAnte;
    for (Player& p : s.players) p.invested = kAnte;
    Player& me = s.players[v];
    Player& op = s.players[o];
    me.hidden = {byte_card(x.c[0]), byte_card(x.c[1])};
    me.discarded = byte_card(x.c[2]); me.has_discard = true;
    me.shown = {byte_card(x.c[3]), byte_card(x.c[4]), byte_card(x.c[5])};
    op.hidden = {opp_h0, opp_h1};
    op.discarded = opp_disc; op.has_discard = true;
    op.shown = {byte_card(x.c[8]), byte_card(x.c[9]), byte_card(x.c[10])};
    std::array<Card, 2> c6, c7;  // per seat
    c6[v] = x.c[6] != 0xff ? byte_card(x.c[6]) : own6;
    c7[v] = x.c[7] != 0xff ? byte_card(x.c[7]) : own7;
    c6[o] = x.c[11] != 0xff ? byte_card(x.c[11]) : opp6;
    c7[o] = opp7;
    s.simulation_deck = {c7[1], c7[0], c6[1], c6[0]};  // advance() deals from the back, seat 0 first
    reset_round(s, 5);
    for (int i = 0; i < x.nact; ++i) {
        const Action a = sample_action_at(x, i);
        const int actor = s.actor;
        if (actor == o) on_opponent(static_cast<const State&>(s), a);
        step(s, actor, a);
    }
    if (s.terminal || s.actor != v) throw std::runtime_error("pp sample: replay does not reach the sampled decision");
    return s;
}

}  // namespace pp
