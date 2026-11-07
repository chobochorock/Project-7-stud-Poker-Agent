// ReBeL (Brown, Bakhtin, Lerer, Gong; NeurIPS 2020) scaled down to the 7-stud v3 game (heads-up, from 5th street,
// heuristic H4) and to one CPU core. Pieces kept from ReBeL: public belief states (PBS: both players' ranges over their
// private cards, updated by Bayes with the search policy), depth-limited subgames (one betting round), value networks
// that map a PBS to a value for every private hand of both players, self-play data generation with exploration, and the
// same search at test time.
//
// What is scaled down or approximated (the README lists the same items):
//  - Private type = the two 3rd-street hole cards (a pair, 1,326 slots). The H4 discard is marginalised (the opponent's
//    discard is treated as an unknown dead card; a player's own discard is not part of its search information).
//    The H4 heuristic's likelihood of the revealed card gives the initial ranges (as in ../../../../cpp_mccfr/
//    stud_rebel_exact_pbs.cpp, summed over discards).
//  - 5th and 6th street: exact vectorised CFR over all pairs (DCFR, simultaneous updates), value network at every
//    end-of-round leaf (CFR-D: the leaf PBS uses the current iterate's ranges). The value network predicts the value
//    before the next street's cards; its training target is the next street's root value after one sampled deal.
//  - 7th street: solved to the showdown, but each player's (pair, 7th card) types are grouped into K buckets by
//    showdown-strength quantile of its range (exact within a bucket game whose chance matrix is built from the exact
//    card-compatible ranges). Values of a type = its bucket's value (card-removal differences inside a bucket ignored).
//  - One search per street (the street's root PBS); no re-solving after the opponent's in-street actions; no policy
//    network warm start; few CFR iterations; small networks; FIFO replay buffers.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "nn.hpp"

namespace sota {
namespace rebel {

constexpr int NP = 1326, NT3 = 22100;
constexpr int IN = 110 + 2 * NP, OUT = 2 * NP;

struct PairTable {
    std::array<int, NP> a{}, b{};
    std::array<uint64_t, NP> mask{};
    std::array<std::array<int16_t, 52>, 52> idx{};
    PairTable() {
        int p = 0;
        for (int y = 1; y < 52; ++y)
            for (int x = 0; x < y; ++x) {
                a[p] = x; b[p] = y; mask[p] = (1ull << x) | (1ull << y);
                idx[x][y] = idx[y][x] = static_cast<int16_t>(p); ++p;
            }
        for (int x = 0; x < 52; ++x) idx[x][x] = -1;
    }
};
inline const PairTable& PT() { static const PairTable t; return t; }
inline int tri_of(int x, int y, int z) {
    if (x > y) std::swap(x, y);
    if (y > z) std::swap(y, z);
    if (x > y) std::swap(x, y);
    return z * (z - 1) * (z - 2) / 6 + y * (y - 1) / 2 + x;
}
inline Card card_at(int i) { return Card{static_cast<uint8_t>(i % 13 + 2), static_cast<uint8_t>(i / 13)}; }
inline uint64_t cards_mask(const std::vector<Card>& v) {
    uint64_t m = 0;
    for (const Card& c : v) m |= 1ull << card_index(c);
    return m;
}
inline uint64_t public_mask(const State& s) { return cards_mask(s.players[0].shown) | cards_mask(s.players[1].shown); }

// ---------------------------------------------------------------- private types of one street (same set for both seats)
struct Types {
    bool seventh = false;
    int n = 0;
    std::vector<int16_t> pair;                 // pair index of the hole cards
    std::vector<int8_t> c7;                    // 7th-street hole card (-1 before 7th)
    std::vector<int> of_pair;                  // 5th/6th: type of each pair (-1: blocked by a public card)
    std::vector<int> first;                    // 7th: types of pair p are first[p] .. first[p + 1] - 1
    void build(uint64_t pub, bool seventh_) {
        seventh = seventh_; pair.clear(); c7.clear(); of_pair.assign(NP, -1); first.assign(NP + 1, 0);
        const auto& P = PT();
        for (int p = 0; p < NP; ++p) {
            first[p] = static_cast<int>(pair.size());
            if (P.mask[p] & pub) continue;
            if (!seventh) { of_pair[p] = static_cast<int>(pair.size()); pair.push_back(static_cast<int16_t>(p)); c7.push_back(-1); continue; }
            for (int c = 0; c < 52; ++c) {
                if ((pub >> c & 1) || (P.mask[p] >> c & 1)) continue;
                pair.push_back(static_cast<int16_t>(p)); c7.push_back(static_cast<int8_t>(c));
            }
        }
        first[NP] = static_cast<int>(pair.size());
        n = static_cast<int>(pair.size());
    }
};

// M[t] = sum of q[t'] over the types t' that share no card with t (inclusion-exclusion over shared cards)
struct Compat {
    std::vector<double> m1, m2, m3;
    void mass(const Types& T, const float* q, float* M) {
        const auto& P = PT();
        m1.assign(52, 0.0); m2.assign(NP, 0.0);
        double tot = 0;
        if (!T.seventh) {
            for (int t = 0; t < T.n; ++t) {
                const double v = q[t];
                if (v == 0) continue;
                const int p = T.pair[t];
                tot += v; m1[P.a[p]] += v; m1[P.b[p]] += v; m2[p] += v;
            }
            for (int t = 0; t < T.n; ++t) {
                const int p = T.pair[t];
                M[t] = static_cast<float>(tot - m1[P.a[p]] - m1[P.b[p]] + m2[p]);
            }
            return;
        }
        m3.assign(NT3, 0.0);
        for (int t = 0; t < T.n; ++t) {
            const double v = q[t];
            if (v == 0) continue;
            const int p = T.pair[t], x = P.a[p], y = P.b[p], z = T.c7[t];
            tot += v; m1[x] += v; m1[y] += v; m1[z] += v;
            m2[p] += v; m2[P.idx[x][z]] += v; m2[P.idx[y][z]] += v; m3[tri_of(x, y, z)] += v;
        }
        for (int t = 0; t < T.n; ++t) {
            const int p = T.pair[t], x = P.a[p], y = P.b[p], z = T.c7[t];
            M[t] = static_cast<float>(tot - m1[x] - m1[y] - m1[z] + m2[p] + m2[P.idx[x][z]] + m2[P.idx[y][z]] - m3[tri_of(x, y, z)]);
        }
    }
};

// ---------------------------------------------------------------- one betting round as a tree
struct Node {
    int8_t kind = 0;   // 0 decision, 1 fold (actor = the folder), 2 end of round (net leaf before 7th; showdown at 7th)
    int8_t actor = -1;
    uint8_t mask = 0;
    int row = 0;       // decision: first row of the actor's tables (one row per legal action, in action order)
    int pot = 0, inv[2] = {0, 0};
    int leaf = -1;     // end of round: index into Tree::leaf_state
    std::array<int, kActionCount> child{};
};
struct Tree {
    std::vector<Node> nodes;    // preorder: a parent comes before its children
    std::vector<State> leaf_state;
    int rows[2] = {0, 0};
    void build(const State& root) { nodes.clear(); leaf_state.clear(); rows[0] = rows[1] = 0; add(root); }
    int add(const State& s) {
        const int id = static_cast<int>(nodes.size());
        nodes.emplace_back();
        Node nd; nd.pot = s.pot; nd.inv[0] = s.players[0].invested; nd.inv[1] = s.players[1].invested;
        nd.child.fill(-1);
        const int actor = s.actor;
        const uint8_t mask = s.terminal ? 0 : valid_mask(s, actor);
        if (!mask) { nd.kind = 2; nd.leaf = static_cast<int>(leaf_state.size()); leaf_state.push_back(s); nodes[id] = nd; return id; }
        nd.kind = 0; nd.actor = static_cast<int8_t>(actor); nd.mask = mask; nd.row = rows[actor];
        rows[actor] += __builtin_popcount(mask);
        nodes[id] = nd;
        for (int a = 0; a < kActionCount; ++a) {
            if (!(mask >> a & 1)) continue;
            State c = s;
            const ActionResult r = apply_action(c, actor, static_cast<Action>(a));
            int cid;
            if (r == ActionResult::FoldEnd) {
                cid = static_cast<int>(nodes.size());
                Node f; f.kind = 1; f.actor = static_cast<int8_t>(actor); f.pot = c.pot; f.inv[0] = c.players[0].invested;
                f.inv[1] = c.players[1].invested; f.child.fill(-1);
                nodes.push_back(f);
            } else if (r == ActionResult::RoundEnd) {
                cid = static_cast<int>(nodes.size());
                Node l; l.kind = 2; l.pot = c.pot; l.inv[0] = c.players[0].invested; l.inv[1] = c.players[1].invested;
                l.leaf = static_cast<int>(leaf_state.size()); l.child.fill(-1);
                leaf_state.push_back(c);
                nodes.push_back(l);
            } else {
                c.actor = 1 - actor;
                cid = add(c);
            }
            nodes[id].child[a] = cid;
        }
        return id;
    }
    // the node reached from the root by this round's actions (-1 if they leave the tree)
    int walk(const std::vector<Action>& actions) const {
        int v = 0;
        for (Action a : actions) {
            if (v < 0 || nodes[v].kind != 0 || !(nodes[v].mask >> a & 1)) return -1;
            v = nodes[v].child[a];
        }
        return v;
    }
};

// payoff (ante units) at a fold node or a showdown, for seat i, given the share of the contested pot i wins
inline double fold_payoff(const Node& z, int i, int ante) {
    return (z.actor == i ? -z.inv[i] : z.pot - z.inv[i]) / static_cast<double>(ante);
}
inline void showdown_terms(const Node& z, int i, int ante, double& fixed, double& contested) {
    const int matched = std::min(z.inv[0], z.inv[1]);
    fixed = -matched / static_cast<double>(ante);  // (refund of the unmatched part) - investment
    contested = 2.0 * matched / ante;
}

// ---------------------------------------------------------------- value network input / output
// x: shown cards of both seats, pot features, both normalised ranges over pairs. y: value / pot for every pair of each seat.
inline void encode(const State& s, const double* q0, const double* q1, Sparse& x) {
    x.clear();
    for (int seat = 0; seat < 2; ++seat)
        for (const Card& c : s.players[seat].shown) x.add(52 * seat + card_index(c), 1.0f);
    const double pot = std::max(1, s.pot);
    x.add(104, static_cast<float>(std::log(pot / s.ante) / 4.0));
    x.add(105, static_cast<float>(s.players[0].invested / pot));
    x.add(106, static_cast<float>(s.players[1].invested / pot));
    x.add(107, (s.players[0].all_in || s.players[1].all_in) ? 1.0f : 0.0f);
    x.add(108, 1.0f);
    for (int p = 0; p < NP; ++p) if (q0[p] > 0) x.add(110 + p, static_cast<float>(q0[p] * 250.0));
    for (int p = 0; p < NP; ++p) if (q1[p] > 0) x.add(110 + NP + p, static_cast<float>(q1[p] * 250.0));
}

struct Config {
    int iters56 = 32, iters7 = 64, buckets = 64;
    int hidden = 128;
    float lr = 3e-4f;
    int batch = 32, train_steps = 4, capacity = 4096;
    // exploration (paper p.15, 18): with probability explore one seat per hand (explore_seats 1, chosen at random) takes a
    // uniform random action at its decisions; explore_seats 2: both seats
    double explore = 0.25, floor = 1e-6;
    int explore_seats = 1;
    // random_iter: at 5th / 6th street the hand is played, and the beliefs passed on, with the policy of one CFR iteration
    // t drawn with probability proportional to its averaging weight t^2 (ReBeL's Algorithm 1 and Theorem 3; the
    // average strategy is the paper's "unsafe search"); 7th street (no leaf values) plays its average strategy
    bool random_iter = true;
    std::string loss = "huber";  // value-network loss: huber (paper p.8, pointwise, delta 1) | mse
    uint64_t seed = 1;
};

// t in 1..iters with probability t^2 / sum (the DCFR averaging weight of solve_pairs)
inline int sample_iteration(int iters, std::mt19937_64& r) {
    double z = 0;
    for (int t = 1; t <= iters; ++t) z += static_cast<double>(t) * t;
    double u = std::uniform_real_distribution<double>(0, z)(r);
    for (int t = 1; t <= iters; ++t) { u -= static_cast<double>(t) * t; if (u <= 0) return t; }
    return iters;
}

// ---------------------------------------------------------------- 5th / 6th street solver (exact pair types)
struct PairSolve {
    Tree tree; Types T;
    std::vector<float> avg[2];     // rows x n, normalised average strategy
    std::vector<float> iter[2];    // rows x n, the policy of iteration `sampled` (when sampled > 0)
    std::vector<float> value[2];   // conditional root value of each type (ante units)
    int sampled = 0;
    double seconds = 0; long long net_calls = 0;
    // probability of action a at a decision node for one type: average strategy
    double prob(int node, int type, int a) const { return at(avg, node, type, a); }
    // the policy the hand is played with: the sampled iteration's when there is one, else the average
    double play_prob(int node, int type, int a) const { return at(sampled > 0 ? iter : avg, node, type, a); }
    double at(const std::vector<float> (&P)[2], int node, int type, int a) const {
        const Node& nd = tree.nodes[node];
        int k = 0;
        for (int b = 0; b < a; ++b) k += nd.mask >> b & 1;
        return P[nd.actor][static_cast<size_t>(nd.row + k) * T.n + type];
    }
};

// r[seat]: root reach of each pair (any scale; blocked pairs ignored). net: value network of this street's leaves.
// sample_iter > 0: also keep the current strategy of that iteration (PairSolve::iter, play_prob).
inline PairSolve solve_pairs(const State& root, const std::vector<double> r[2], const Mlp& net, int iters, int sample_iter = 0) {
    const auto t0 = std::chrono::steady_clock::now();
    PairSolve S;
    S.tree.build(root);
    S.T.build(public_mask(root), false);
    const Tree& G = S.tree;
    const Types& T = S.T;
    const int n = T.n, N = static_cast<int>(G.nodes.size());
    std::vector<float> R[2], cur[2], ssum[2];
    for (int i = 0; i < 2; ++i) {
        R[i].assign(static_cast<size_t>(G.rows[i]) * n, 0.0f); cur[i].assign(R[i].size(), 0.0f); ssum[i].assign(R[i].size(), 0.0f);
    }
    std::vector<float> reach(static_cast<size_t>(N) * 2 * n), cfv(static_cast<size_t>(N) * 2 * n), M(static_cast<size_t>(2) * n);
    std::vector<double> rootsum[2] = {std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
    auto RE = [&](int v, int i) { return &reach[(static_cast<size_t>(v) * 2 + i) * n]; };
    auto CF = [&](int v, int i) { return &cfv[(static_cast<size_t>(v) * 2 + i) * n]; };
    for (int i = 0; i < 2; ++i) for (int t = 0; t < n; ++t) RE(0, i)[t] = static_cast<float>(r[i][T.pair[t]]);
    Compat cp;
    std::vector<float> Mroot[2] = {std::vector<float>(n), std::vector<float>(n)};
    cp.mass(T, RE(0, 1), Mroot[0].data()); cp.mass(T, RE(0, 0), Mroot[1].data());
    std::vector<double> q0(NP), q1(NP);
    Sparse x; Mlp::Cache c;
    double wsum = 0;
    for (int it = 1; it <= iters; ++it) {
        // current strategy (regret matching on the positive part)
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor, k = __builtin_popcount(nd.mask);
            float* Rr = &R[i][static_cast<size_t>(nd.row) * n];
            float* Cr = &cur[i][static_cast<size_t>(nd.row) * n];
            for (int t = 0; t < n; ++t) {
                float z = 0;
                for (int j = 0; j < k; ++j) z += std::max(0.0f, Rr[static_cast<size_t>(j) * n + t]);
                for (int j = 0; j < k; ++j) Cr[static_cast<size_t>(j) * n + t] = z > 0 ? std::max(0.0f, Rr[static_cast<size_t>(j) * n + t]) / z : 1.0f / k;
            }
        }
        if (it == sample_iter) { S.iter[0] = cur[0]; S.iter[1] = cur[1]; S.sampled = it; }
        // reach (preorder)
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor;
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                const float* pr = &cur[i][static_cast<size_t>(nd.row + j) * n];
                std::memcpy(RE(ch, 1 - i), RE(v, 1 - i), sizeof(float) * n);
                float* dst = RE(ch, i); const float* src = RE(v, i);
                for (int t = 0; t < n; ++t) dst[t] = src[t] * pr[t];
                ++j;
            }
        }
        // values at fold nodes and leaves
        for (int v = 0; v < N; ++v) {
            const Node& z = G.nodes[v];
            if (z.kind == 0) continue;
            cp.mass(T, RE(v, 1), &M[0]); cp.mass(T, RE(v, 0), &M[n]);
            if (z.kind == 1) {
                for (int i = 0; i < 2; ++i) {
                    const float u = static_cast<float>(fold_payoff(z, i, root.ante));
                    float* d = CF(v, i); const float* m = &M[static_cast<size_t>(i) * n];
                    for (int t = 0; t < n; ++t) d[t] = u * m[t];
                }
                continue;
            }
            std::fill(q0.begin(), q0.end(), 0.0); std::fill(q1.begin(), q1.end(), 0.0);
            double z0 = 0, z1 = 0;
            for (int t = 0; t < n; ++t) { z0 += RE(v, 0)[t]; z1 += RE(v, 1)[t]; }
            for (int t = 0; t < n; ++t) {
                if (z0 > 0) q0[T.pair[t]] = RE(v, 0)[t] / z0;
                if (z1 > 0) q1[T.pair[t]] = RE(v, 1)[t] / z1;
            }
            encode(G.leaf_state[z.leaf], q0.data(), q1.data(), x);
            net.forward(x, c); ++S.net_calls;
            const float scale = static_cast<float>(static_cast<double>(z.pot) / root.ante);
            for (int i = 0; i < 2; ++i) {
                float* d = CF(v, i); const float* m = &M[static_cast<size_t>(i) * n];
                for (int t = 0; t < n; ++t) d[t] = c.y[static_cast<size_t>(i) * NP + T.pair[t]] * scale * m[t];
            }
        }
        // values and regrets (reverse preorder)
        const double pos = std::pow(it, 1.5) / (std::pow(it, 1.5) + 1.0);
        for (int v = N - 1; v >= 0; --v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor, o = 1 - i;
            float* own = CF(v, i); float* opp = CF(v, o);
            std::fill(own, own + n, 0.0f); std::fill(opp, opp + n, 0.0f);
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                const float* pr = &cur[i][static_cast<size_t>(nd.row + j) * n];
                const float* ci = CF(ch, i); const float* co = CF(ch, o);
                for (int t = 0; t < n; ++t) { own[t] += pr[t] * ci[t]; opp[t] += co[t]; }
                ++j;
            }
            j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const float* ci = CF(nd.child[a], i);
                float* Rr = &R[i][static_cast<size_t>(nd.row + j) * n];
                for (int t = 0; t < n; ++t) {
                    float g = Rr[t] + ci[t] - own[t];
                    Rr[t] = g > 0 ? static_cast<float>(g * pos) : 0.5f * g;
                }
                ++j;
            }
        }
        // averages with weight t^2 (DCFR gamma = 2)
        const double w = static_cast<double>(it) * it;
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor, k = __builtin_popcount(nd.mask);
            const float* rr = RE(v, i);
            for (int j = 0; j < k; ++j) {
                float* s = &ssum[i][static_cast<size_t>(nd.row + j) * n];
                const float* pr = &cur[i][static_cast<size_t>(nd.row + j) * n];
                for (int t = 0; t < n; ++t) s[t] += static_cast<float>(w) * rr[t] * pr[t];
            }
        }
        for (int i = 0; i < 2; ++i) { const float* cv = CF(0, i); for (int t = 0; t < n; ++t) rootsum[i][t] += w * cv[t]; }
        wsum += w;
    }
    for (int i = 0; i < 2; ++i) {
        S.avg[i].assign(ssum[i].size(), 0.0f);
        S.value[i].assign(n, 0.0f);
        for (int t = 0; t < n; ++t) S.value[i][t] = Mroot[i][t] > 0 ? static_cast<float>(rootsum[i][t] / wsum / Mroot[i][t]) : 0.0f;
    }
    for (int v = 0; v < N; ++v) {
        const Node& nd = G.nodes[v];
        if (nd.kind != 0) continue;
        const int i = nd.actor, k = __builtin_popcount(nd.mask);
        for (int t = 0; t < n; ++t) {
            double z = 0;
            for (int j = 0; j < k; ++j) z += ssum[i][static_cast<size_t>(nd.row + j) * n + t];
            for (int j = 0; j < k; ++j)
                S.avg[i][static_cast<size_t>(nd.row + j) * n + t] = z > 0 ? static_cast<float>(ssum[i][static_cast<size_t>(nd.row + j) * n + t] / z) : 1.0f / k;
        }
    }
    S.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

// ---------------------------------------------------------------- 7th street solver (strength buckets, to the showdown)
struct SeventhSolve {
    Tree tree; Types T;
    int K = 0;
    std::vector<uint32_t> strength[2];   // per type
    std::vector<uint32_t> edges[2];      // K-1 strength thresholds
    std::vector<int> bucket[2];          // per type
    std::vector<float> avg[2];           // rows x K
    std::vector<float> bvalue[2];        // conditional value of each bucket (ante units)
    std::vector<float> type_mass[2];     // compatible opponent mass of each type at the root (for pair aggregation)
    std::vector<double> A, W;            // bucket-game chance matrices (K x K, seat 0 rows)
    double seconds = 0, exploit_gap = 0; // bucket-game NashConv of the average strategy (ante units)
    int bucket_of(int seat, uint32_t s) const {
        return static_cast<int>(std::upper_bound(edges[seat].begin(), edges[seat].end(), s) - edges[seat].begin());
    }
    double prob(int node, int b, int a) const {
        const Node& nd = tree.nodes[node];
        int k = 0;
        for (int c = 0; c < a; ++c) k += nd.mask >> c & 1;
        return avg[nd.actor][static_cast<size_t>(nd.row + k) * K + b];
    }
    // value of each pair (ante units) for seat i: types of the pair averaged with their compatible opponent mass
    void pair_values(int i, std::vector<float>& out) const {
        out.assign(NP, 0.0f);
        for (int p = 0; p < NP; ++p) {
            double num = 0, den = 0;
            for (int t = T.first[p]; t < T.first[p + 1]; ++t) { num += type_mass[i][t] * bvalue[i][bucket[i][t]]; den += type_mass[i][t]; }
            out[p] = den > 0 ? static_cast<float>(num / den) : 0.0f;
        }
    }
};

inline uint32_t type_strength(const Types& T, int t, const std::vector<Card>& shown) {
    const auto& P = PT();
    std::vector<Card> cards = shown;
    cards.push_back(card_at(P.a[T.pair[t]])); cards.push_back(card_at(P.b[T.pair[t]])); cards.push_back(card_at(T.c7[t]));
    return best_hand_encoded(cards);
}

inline SeventhSolve solve_seventh(const State& root, const std::vector<double> r[2], int iters, int K) {
    const auto t0 = std::chrono::steady_clock::now();
    SeventhSolve S;
    S.tree.build(root);
    S.T.build(public_mask(root), true);
    S.K = K;
    const Types& T = S.T;
    const int n = T.n;
    std::vector<float> rr[2];
    for (int i = 0; i < 2; ++i) {
        rr[i].resize(n);
        S.strength[i].resize(n);
        for (int t = 0; t < n; ++t) { rr[i][t] = static_cast<float>(r[i][T.pair[t]]); S.strength[i][t] = type_strength(T, t, root.players[i].shown); }
    }
    // buckets: strength quantiles of the range mass
    std::vector<int> order[2];
    for (int i = 0; i < 2; ++i) {
        order[i].resize(n);
        std::iota(order[i].begin(), order[i].end(), 0);
        std::sort(order[i].begin(), order[i].end(), [&](int a, int b) { return S.strength[i][a] < S.strength[i][b]; });
        double tot = 0;
        for (int t = 0; t < n; ++t) tot += rr[i][t];
        S.edges[i].clear();
        double acc = 0; int next = 1;
        for (int k = 0; k < n && next < K; ++k) {
            acc += rr[i][order[i][k]];
            if (acc >= tot * next / K) {
                const uint32_t s = S.strength[i][order[i][k]];
                if (S.edges[i].empty() || S.edges[i].back() < s) S.edges[i].push_back(s);  // ties stay in one bucket
                while (next < K && acc >= tot * next / K) ++next;
            }
        }
        S.bucket[i].resize(n);
        for (int t = 0; t < n; ++t) S.bucket[i][t] = S.bucket_of(i, S.strength[i][t]);
    }
    // chance matrices of the bucket game: A[b0][b1] = sum r0 r1 compat, W[b0][b1] = same times seat 0's showdown share
    std::vector<double> A(static_cast<size_t>(K) * K, 0.0), W(A.size(), 0.0);
    Compat cp;
    std::vector<float> q(n), Mt(n);
    for (int b1 = 0; b1 < K; ++b1) {
        for (int t = 0; t < n; ++t) q[t] = S.bucket[1][t] == b1 ? rr[1][t] : 0.0f;
        cp.mass(T, q.data(), Mt.data());
        for (int t = 0; t < n; ++t) if (rr[0][t] > 0) A[static_cast<size_t>(S.bucket[0][t]) * K + b1] += static_cast<double>(rr[0][t]) * Mt[t];
        // showdown share: sweep seat 0's types in strength order against seat 1's types of bucket b1
        const auto& P = PT();
        for (int pass = 0; pass < 2; ++pass) {  // pass 0: strictly weaker opponents, pass 1: weaker or equal
            std::vector<double> m1(52, 0.0), m2(NP, 0.0), m3(NT3, 0.0);
            double tot = 0; int j = 0;
            for (int k = 0; k < n; ++k) {
                const int t = order[0][k];
                const uint32_t s = S.strength[0][t];
                while (j < n) {
                    const int u = order[1][j];
                    const uint32_t so = S.strength[1][u];
                    if (pass == 0 ? so >= s : so > s) break;
                    ++j;
                    const double v = q[u];
                    if (v == 0) continue;
                    const int p = T.pair[u], x = P.a[p], y = P.b[p], zc = T.c7[u];
                    tot += v; m1[x] += v; m1[y] += v; m1[zc] += v;
                    m2[p] += v; m2[P.idx[x][zc]] += v; m2[P.idx[y][zc]] += v; m3[tri_of(x, y, zc)] += v;
                }
                if (rr[0][t] <= 0) continue;
                const int p = T.pair[t], x = P.a[p], y = P.b[p], zc = T.c7[t];
                const double L = tot - m1[x] - m1[y] - m1[zc] + m2[p] + m2[P.idx[x][zc]] + m2[P.idx[y][zc]] - m3[tri_of(x, y, zc)];
                W[static_cast<size_t>(S.bucket[0][t]) * K + b1] += 0.5 * rr[0][t] * L;  // (L_less + L_lesseq) / 2
            }
        }
    }
    for (int i = 0; i < 2; ++i) { S.type_mass[i].resize(n); cp.mass(T, rr[1 - i].data(), S.type_mass[i].data()); }
    S.A = A; S.W = W;
    // CFR on the bucket game
    const Tree& G = S.tree;
    const int N = static_cast<int>(G.nodes.size());
    std::vector<double> R[2], cur[2], ssum[2];
    for (int i = 0; i < 2; ++i) { R[i].assign(static_cast<size_t>(G.rows[i]) * K, 0.0); cur[i] = R[i]; ssum[i] = R[i]; }
    std::vector<double> reach(static_cast<size_t>(N) * 2 * K), cfv(reach.size());
    auto RE = [&](int v, int i) { return &reach[(static_cast<size_t>(v) * 2 + i) * K]; };
    auto CF = [&](int v, int i) { return &cfv[(static_cast<size_t>(v) * 2 + i) * K]; };
    std::vector<double> rootsum[2] = {std::vector<double>(K, 0.0), std::vector<double>(K, 0.0)};
    double wsum = 0;
    // seat 0 / seat 1 counterfactual values at a terminal from the opponent's bucket reach
    auto terminal = [&](const Node& z, const double* r0, const double* r1, double* c0, double* c1) {
        for (int b = 0; b < K; ++b) { c0[b] = 0; c1[b] = 0; }
        if (z.kind == 1) {
            const double u0 = fold_payoff(z, 0, root.ante), u1 = fold_payoff(z, 1, root.ante);
            for (int b0 = 0; b0 < K; ++b0) {
                const double* a = &A[static_cast<size_t>(b0) * K];
                double s0 = 0;
                for (int b1 = 0; b1 < K; ++b1) { s0 += a[b1] * r1[b1]; c1[b1] += a[b1] * r0[b0]; }
                c0[b0] = u0 * s0;
            }
            for (int b1 = 0; b1 < K; ++b1) c1[b1] *= u1;
            return;
        }
        double f0, k0, f1, k1;
        showdown_terms(z, 0, root.ante, f0, k0); showdown_terms(z, 1, root.ante, f1, k1);
        for (int b0 = 0; b0 < K; ++b0) {
            const double* a = &A[static_cast<size_t>(b0) * K];
            const double* w = &W[static_cast<size_t>(b0) * K];
            double s0 = 0;
            for (int b1 = 0; b1 < K; ++b1) {
                s0 += (f0 * a[b1] + k0 * w[b1]) * r1[b1];
                c1[b1] += (f1 * a[b1] + k1 * (a[b1] - w[b1])) * r0[b0];
            }
            c0[b0] = s0;
        }
    };
    for (int i = 0; i < 2; ++i) std::fill(RE(0, i), RE(0, i) + K, 1.0);
    for (int it = 1; it <= iters; ++it) {
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor, k = __builtin_popcount(nd.mask);
            for (int b = 0; b < K; ++b) {
                double z = 0;
                for (int j = 0; j < k; ++j) z += std::max(0.0, R[i][static_cast<size_t>(nd.row + j) * K + b]);
                for (int j = 0; j < k; ++j) cur[i][static_cast<size_t>(nd.row + j) * K + b] = z > 0 ? std::max(0.0, R[i][static_cast<size_t>(nd.row + j) * K + b]) / z : 1.0 / k;
            }
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                std::copy(RE(v, 1 - i), RE(v, 1 - i) + K, RE(ch, 1 - i));
                for (int b = 0; b < K; ++b) RE(ch, i)[b] = RE(v, i)[b] * cur[i][static_cast<size_t>(nd.row + j) * K + b];
                ++j;
            }
        }
        const double pos = std::pow(it, 1.5) / (std::pow(it, 1.5) + 1.0);
        for (int v = N - 1; v >= 0; --v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) { terminal(nd, RE(v, 0), RE(v, 1), CF(v, 0), CF(v, 1)); continue; }
            const int i = nd.actor, o = 1 - i;
            double* own = CF(v, i); double* opp = CF(v, o);
            std::fill(own, own + K, 0.0); std::fill(opp, opp + K, 0.0);
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                for (int b = 0; b < K; ++b) { own[b] += cur[i][static_cast<size_t>(nd.row + j) * K + b] * CF(ch, i)[b]; opp[b] += CF(ch, o)[b]; }
                ++j;
            }
            j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                for (int b = 0; b < K; ++b) {
                    double& g = R[i][static_cast<size_t>(nd.row + j) * K + b];
                    g += CF(nd.child[a], i)[b] - own[b];
                    g = g > 0 ? g * pos : 0.5 * g;
                }
                ++j;
            }
        }
        const double w = static_cast<double>(it) * it;
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor, k = __builtin_popcount(nd.mask);
            for (int j = 0; j < k; ++j)
                for (int b = 0; b < K; ++b)
                    ssum[i][static_cast<size_t>(nd.row + j) * K + b] += w * RE(v, i)[b] * cur[i][static_cast<size_t>(nd.row + j) * K + b];
        }
        for (int i = 0; i < 2; ++i) for (int b = 0; b < K; ++b) rootsum[i][b] += w * CF(0, i)[b];
        wsum += w;
    }
    for (int i = 0; i < 2; ++i) {
        S.avg[i].assign(ssum[i].size(), 0.0f);
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0 || nd.actor != i) continue;
            const int k = __builtin_popcount(nd.mask);
            for (int b = 0; b < K; ++b) {
                double z = 0;
                for (int j = 0; j < k; ++j) z += ssum[i][static_cast<size_t>(nd.row + j) * K + b];
                for (int j = 0; j < k; ++j)
                    S.avg[i][static_cast<size_t>(nd.row + j) * K + b] = z > 0 ? static_cast<float>(ssum[i][static_cast<size_t>(nd.row + j) * K + b] / z) : 1.0f / k;
            }
        }
    }
    // bucket values; buckets without mass take the nearest bucket's value
    for (int i = 0; i < 2; ++i) {
        std::vector<double> mass(K, 0.0);
        for (int b0 = 0; b0 < K; ++b0) for (int b1 = 0; b1 < K; ++b1) mass[i == 0 ? b0 : b1] += A[static_cast<size_t>(b0) * K + b1];
        S.bvalue[i].assign(K, 0.0f);
        std::vector<char> ok(K, 0);
        for (int b = 0; b < K; ++b) if (mass[b] > 0) { S.bvalue[i][b] = static_cast<float>(rootsum[i][b] / wsum / mass[b]); ok[b] = 1; }
        for (int b = 0; b < K; ++b) {
            if (ok[b]) continue;
            for (int d = 1; d < K; ++d) {
                if (b - d >= 0 && ok[b - d]) { S.bvalue[i][b] = S.bvalue[i][b - d]; break; }
                if (b + d < K && ok[b + d]) { S.bvalue[i][b] = S.bvalue[i][b + d]; break; }
            }
        }
    }
    // NashConv of the average strategy in the bucket game (best response by backward induction over buckets)
    {
        std::vector<double> ravg(reach.size());
        auto RA = [&](int v, int i) { return &ravg[(static_cast<size_t>(v) * 2 + i) * K]; };
        for (int i = 0; i < 2; ++i) std::fill(RA(0, i), RA(0, i) + K, 1.0);
        for (int v = 0; v < N; ++v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) continue;
            const int i = nd.actor;
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                std::copy(RA(v, 1 - i), RA(v, 1 - i) + K, RA(ch, 1 - i));
                for (int b = 0; b < K; ++b) RA(ch, i)[b] = RA(v, i)[b] * S.avg[i][static_cast<size_t>(nd.row + j) * K + b];
                ++j;
            }
        }
        std::vector<double> br(reach.size()), tmp(2 * static_cast<size_t>(K));
        auto BR = [&](int v, int i) { return &br[(static_cast<size_t>(v) * 2 + i) * K]; };
        for (int v = N - 1; v >= 0; --v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) { terminal(nd, RA(v, 0), RA(v, 1), BR(v, 0), BR(v, 1)); continue; }
            const int i = nd.actor, o = 1 - i;
            double* own = BR(v, i); double* opp = BR(v, o);
            std::fill(own, own + K, -1e300); std::fill(opp, opp + K, 0.0);
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                for (int b = 0; b < K; ++b) { own[b] = std::max(own[b], BR(ch, i)[b]); opp[b] += BR(ch, o)[b]; }
            }
        }
        // BR(0, i) = best-response counterfactual values of seat i against the other's average strategy; game value from i's
        // average strategy: sum of the average-strategy root values. NashConv = sum_i (BR_i - value_i).
        std::vector<double> val(reach.size());
        auto VA = [&](int v, int i) { return &val[(static_cast<size_t>(v) * 2 + i) * K]; };
        for (int v = N - 1; v >= 0; --v) {
            const Node& nd = G.nodes[v];
            if (nd.kind != 0) { terminal(nd, RA(v, 0), RA(v, 1), VA(v, 0), VA(v, 1)); continue; }
            const int i = nd.actor, o = 1 - i;
            double* own = VA(v, i); double* opp = VA(v, o);
            std::fill(own, own + K, 0.0); std::fill(opp, opp + K, 0.0);
            int j = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (!(nd.mask >> a & 1)) continue;
                const int ch = nd.child[a];
                for (int b = 0; b < K; ++b) { own[b] += S.avg[i][static_cast<size_t>(nd.row + j) * K + b] * VA(ch, i)[b]; opp[b] += VA(ch, o)[b]; }
                ++j;
            }
        }
        double tot = 0;
        for (size_t e = 0; e < A.size(); ++e) tot += A[e];
        double gap = 0;
        for (int i = 0; i < 2; ++i) for (int b = 0; b < K; ++b) gap += BR(0, i)[b] - VA(0, i)[b];
        S.exploit_gap = tot > 0 ? gap / tot : 0.0;
    }
    S.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
}

// ---------------------------------------------------------------- ranges
// Initial range of a seat at the 5th-street root: the number of (deal order, discard) combinations for which the
// heuristic H4 would reveal the seat's first shown card, for every pair of hole cards (discard marginalised).
inline std::vector<double> h4_range(const State& root, int seat) {
    const uint64_t pub = public_mask(root);
    const Card shown = root.players[seat].shown.front();
    const auto& P = PT();
    std::vector<int> free;
    for (int c = 0; c < 52; ++c) if (!(pub >> c & 1)) free.push_back(c);
    std::vector<double> r(NP, 0.0);
    // every 4-card set {shown, x, y, z}: count the deal orders in which H4 reveals `shown`; the discard's partners
    // are the hole pair
    std::vector<Card> hand(4);
    for (size_t i = 0; i < free.size(); ++i)
        for (size_t j = i + 1; j < free.size(); ++j)
            for (size_t k = j + 1; k < free.size(); ++k) {
                const int ids[4] = {-1, free[i], free[j], free[k]};
                std::array<int, 4> order{0, 1, 2, 3};
                do {
                    for (int q = 0; q < 4; ++q) hand[q] = order[q] == 0 ? shown : card_at(ids[order[q]]);
                    const auto [di, ri] = discard_reveal(hand);
                    if (order[ri] != 0 || order[di] == 0) continue;
                    int pair[2], m = 0;
                    for (int q = 1; q < 4; ++q) if (q != order[di]) pair[m++] = ids[q];
                    r[P.idx[pair[0]][pair[1]]] += 1.0;
                } while (std::next_permutation(order.begin(), order.end()));
            }
    return r;
}
inline void normalise(std::vector<double>& r) {
    const double z = std::accumulate(r.begin(), r.end(), 0.0);
    if (z > 0) for (double& v : r) v /= z;
}
inline void block(std::vector<double>& r, uint64_t pub) {
    const auto& P = PT();
    for (int p = 0; p < NP; ++p) if (P.mask[p] & pub) r[p] = 0.0;
}

// ---------------------------------------------------------------- self-play training
struct Example {
    Sparse x;
    std::vector<std::pair<int16_t, float>> y;  // (output index, target value / pot)
};

struct Trainer {
    Config cfg;
    Mlp v6, v7;          // value networks of the 5th-street leaves (before 6th) and the 6th-street leaves (before 7th)
    Adam o6, o7;
    std::deque<Example> buf6, buf7;
    std::mt19937_64 rng;
    long long episodes = 0, examples6 = 0, examples7 = 0, steps = 0;
    double loss6 = 0, loss7 = 0, solve_s[3] = {0, 0, 0}, train_s = 0, gap7 = 0;
    long long solves[3] = {0, 0, 0};
    int explorer = -1;  // the exploring seat of the current hand (-1: both)

    explicit Trainer(const Config& c) : cfg(c), rng(c.seed) {
        v6 = Mlp(IN, c.hidden, c.hidden, OUT, c.seed * 11 + 1);
        v7 = Mlp(IN, c.hidden, c.hidden, OUT, c.seed * 11 + 2);
        o6.lr = o7.lr = c.lr; o6.init(v6); o7.init(v7);
    }

    int sample_index(const std::vector<double>& w) {
        double z = std::accumulate(w.begin(), w.end(), 0.0);
        double u = std::uniform_real_distribution<double>(0, z)(rng);
        for (size_t i = 0; i < w.size(); ++i) { u -= w[i]; if (u <= 0 && w[i] > 0) return static_cast<int>(i); }
        for (size_t i = w.size(); i-- > 0;) if (w[i] > 0) return static_cast<int>(i);
        return -1;
    }
    // a world (one pair per seat, no shared card) from the joint PBS
    std::array<int, 2> sample_world(const std::vector<double> r[2]) {
        const auto& P = PT();
        for (int tries = 0; tries < 100000; ++tries) {
            const int a = sample_index(r[0]), b = sample_index(r[1]);
            if (a < 0 || b < 0) break;
            if (!(P.mask[a] & P.mask[b])) return {a, b};
        }
        throw std::runtime_error("rebel: cannot sample a compatible world");
    }
    int pick_action(uint8_t mask, double explore_u, const std::array<double, kActionCount>& p) {
        if (std::uniform_real_distribution<double>(0, 1)(rng) < explore_u) {
            const int n = __builtin_popcount(mask);
            int k = std::uniform_int_distribution<int>(0, n - 1)(rng);
            for (int a = 0; a < kActionCount; ++a) if (mask >> a & 1) { if (k == 0) return a; --k; }
        }
        double u = std::uniform_real_distribution<double>(0, 1)(rng), acc = 0; int last = -1;
        for (int a = 0; a < kActionCount; ++a) if (mask >> a & 1) { acc += p[a]; last = a; if (u < acc) return a; }
        return last;
    }

    // one self-play hand from a random 5th-street root; adds up to one example per value network
    void episode() {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        State s = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        for (auto& pl : s.players) { pl.hidden.clear(); pl.has_discard = false; }
        std::vector<double> r[2] = {h4_range(s, 0), h4_range(s, 1)};
        normalise(r[0]); normalise(r[1]);
        Sparse pending_x; bool pending = false;
        explorer = cfg.explore_seats == 2 ? -1 : std::uniform_int_distribution<int>(0, 1)(rng);
        for (int street = 5; street <= 7; ++street) {
            const auto& P = PT();
            std::array<int, 2> world = sample_world(r);
            int leaf_node = -1;
            State leaf;
            const auto ts = std::chrono::steady_clock::now();
            if (street < 7) {
                const int tstar = cfg.random_iter ? sample_iteration(cfg.iters56, rng) : 0;
                const PairSolve S = solve_pairs(s, r, street == 5 ? v6 : v7, cfg.iters56, tstar);
                solve_s[street - 5] += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count(); ++solves[street - 5];
                if (pending) {  // the previous street's leaf PBS -> this street's root values
                    Example e; e.x = pending_x;
                    const double pot = static_cast<double>(s.pot) / s.ante;
                    for (int i = 0; i < 2; ++i)
                        for (int t = 0; t < S.T.n; ++t) e.y.emplace_back(static_cast<int16_t>(i * NP + S.T.pair[t]), static_cast<float>(S.value[i][t] / pot));
                    push(buf6, std::move(e)); ++examples6;
                }
                leaf_node = play_street(S.tree, [&](int v, int a) {
                    const int t = S.T.of_pair[world[S.tree.nodes[v].actor]];
                    return S.play_prob(v, t, a);
                }, [&](int v, int a, int seat) {
                    for (int p = 0; p < NP; ++p) {
                        if (r[seat][p] <= 0) continue;
                        const int t = S.T.of_pair[p];
                        r[seat][p] *= std::max(cfg.floor, t >= 0 ? S.play_prob(v, t, a) : 0.0);
                    }
                });
                if (leaf_node < 0) return;  // fold
                leaf = S.tree.leaf_state[S.tree.nodes[leaf_node].leaf];
            } else {
                const SeventhSolve S = solve_seventh(s, r, cfg.iters7, cfg.buckets);
                solve_s[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count(); ++solves[2];
                gap7 += S.exploit_gap;
                if (pending) {
                    Example e; e.x = pending_x;
                    const double pot = static_cast<double>(s.pot) / s.ante;
                    for (int i = 0; i < 2; ++i) {
                        std::vector<float> pvv; S.pair_values(i, pvv);
                        for (int p = 0; p < NP; ++p) if (!(P.mask[p] & public_mask(s))) e.y.emplace_back(static_cast<int16_t>(i * NP + p), static_cast<float>(pvv[p] / pot));
                    }
                    push(buf7, std::move(e)); ++examples7;
                }
                return;  // nothing to learn after the 7th-street root
            }
            normalise(r[0]); normalise(r[1]);
            // the leaf PBS (before the next cards): input of the next example
            {
                std::vector<double> q0(NP, 0.0), q1(NP, 0.0);
                for (int p = 0; p < NP; ++p) { q0[p] = r[0][p]; q1[p] = r[1][p]; }
                encode(leaf, q0.data(), q1.data(), pending_x);
                pending = true;
            }
            // deal the next street to the sampled world (6th: one shown card each; 7th: one hole card each)
            uint64_t used = public_mask(leaf) | P.mask[world[0]] | P.mask[world[1]];
            std::vector<int> deckleft;
            for (int c = 0; c < 52; ++c) if (!(used >> c & 1)) deckleft.push_back(c);
            std::shuffle(deckleft.begin(), deckleft.end(), rng);
            s = leaf;
            if (street == 5) {
                for (int seat = 0; seat < 2; ++seat) s.players[seat].shown.push_back(card_at(deckleft[seat]));
                block(r[0], public_mask(s)); block(r[1], public_mask(s));
                normalise(r[0]); normalise(r[1]);
            }
            for (auto& pl : s.players) pl.round_bet = 0;
            reset_round(s, street + 1);
        }
    }

    template <class Prob, class Update>
    int play_street(const Tree& G, Prob prob, Update update) {
        int v = 0;
        while (G.nodes[v].kind == 0) {
            const Node& nd = G.nodes[v];
            std::array<double, kActionCount> p{};
            for (int a = 0; a < kActionCount; ++a) if (nd.mask >> a & 1) p[a] = prob(v, a);
            const int a = pick_action(nd.mask, explorer < 0 || nd.actor == explorer ? cfg.explore : 0.0, p);
            update(v, a, nd.actor);
            v = nd.child[a];
        }
        return G.nodes[v].kind == 1 ? -1 : v;
    }

    void push(std::deque<Example>& b, Example&& e) {
        b.push_back(std::move(e));
        while (static_cast<int>(b.size()) > cfg.capacity) b.pop_front();
    }

    double train_net(Mlp& net, Adam& opt, std::deque<Example>& b) {
        if (static_cast<int>(b.size()) < cfg.batch) return 0;
        Mlp::Grad g; g.init(net);
        std::uniform_int_distribution<size_t> pick(0, b.size() - 1);
        std::vector<float> dy(OUT, 0.0f);
        double loss = 0; long long cnt = 0;
        for (int k = 0; k < cfg.batch; ++k) {
            const Example& e = b[pick(rng)];
            Mlp::Cache c; net.forward(e.x, c);
            std::fill(dy.begin(), dy.end(), 0.0f);
            const float scale = 1.0f / std::max<size_t>(1, e.y.size());
            for (const auto& [o, t] : e.y) {
                const float d = c.y[o] - t;
                // Huber (delta 1): gradient d inside [-1, 1], sign(d) outside; the logged loss stays the squared error
                dy[o] = (cfg.loss == "huber" ? std::max(-1.0f, std::min(1.0f, d)) : 2.0f * d) * scale;
                loss += d * d; ++cnt;
            }
            net.backward(e.x, c, dy.data(), g);
        }
        opt.step(net, g, 1.0f / cfg.batch);
        return cnt ? loss / cnt : 0;
    }

    void train() {
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < cfg.train_steps; ++k) {
            const double a = train_net(v6, o6, buf6), b = train_net(v7, o7, buf7);
            if (a > 0) loss6 = a;
            if (b > 0) loss7 = b;
            ++steps;
        }
        train_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

}  // namespace rebel
}  // namespace sota
