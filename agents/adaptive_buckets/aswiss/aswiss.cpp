// Refinement-only abstraction splitting by labels of cumulative (windowed) counterfactual regret, eq. (6).
//
// Build: g++ -O3 -std=c++17 aswiss.cpp -o aswiss
// Run:   ./aswiss --tree DIR --buckets FILE --mode full|es --iters T --epoch0 L0 --split cum|none
//                  [--tol 1e-3] [--z 2] [--seed 1] --out OUTDIR
// Input: arrays written by export.py (preorder CSR tree, player-0 utility at terminals) and an initial
// infoset -> bucket map. Output: OUTDIR/log.csv (one row per epoch end) and, per epoch end k,
// OUTDIR/policy_{bucket,true,current}_k.bin (double[infosets * width]) for the Python exact evaluator.
//
// full: simultaneous full-tree CFR+ (RM+ on bucket regret sums, linear averaging).
// es:   external-sampling MCCFR+ (alternating traversers, RM+ on sampled bucket regret sums applied after
//       each traversal, linear "simple" averaging at opponent nodes).
// Bucket tables: RM+ regrets Q and average numerators S. Per original infoset: the true average
// numerator (partition independent) and the window regret sums used only for the split rule.
// Split rule at each epoch end: label(I) = argmax_a R_I(a) when the evidence is positive (full: per-reach
// regret > tol; es: R_I(a) - z * sqrt(sum of squared sampled regrets) > 0), else "none". A bucket whose
// members carry two or more labels is split by label; children restart Q and S. No merges.
// --label selects the labels the split rule uses (ASWISS_CONVERGENCE_KO.md section 9); every rule except `window`
// uses noise-tolerant label sets (Lemma 9): the labels within 2*eta of the best estimate, the "none" label being 0. A
// bucket is split when its members' sets have no common label; each member then joins the child of the label of its
// set shared by the most members of the bucket.
//   window      the rule above (window sums of the sampled regrets, argmax + test)
//   window_tol  the same sums with tolerant sets, eta = z * sqrt(max_a sum of squared sampled regrets)
//   replay      method A: exact regrets of the current strategy at random iterations (each with probability
//               --replay-p, own RNG), summed with weight 1/p (unbiased for the window sum); eta = z * sqrt(max_a of
//               the unbiased variance estimate (1-p)/p^2 * sum r^2)
//   avg         method B: exact regrets r^avg(I) of the bucket average policy at the epoch end (eta = 0)
//   window+avg  the common refinement of `window` and `avg` (a bucket splits when either rule splits it)
// --unit history labels bucket-history keys instead of information sets (export_history.py): the key of I is (time
// step of I, the buckets and actions of its player's earlier decisions, the observations since the last one), so the
// past is remembered only through the buckets it fell into. Regret sums are pooled over each key and every member
// gets the key's label set, so a split never separates two information sets with the same key. Keys refine as the
// buckets of earlier decisions split; with one bucket per information set they identify information sets.
// Every epoch end also logs, for the bucket average policy avg, rho(avg)/2 (an upper bound on its exploitability,
// Lemma 7), rho_P(avg)/2, H(avg)/2 and the fraction of buckets whose members share a label under avg.
// With --label avg every judgment goes through judge_avg and is logged in judgments.csv (rho/2 split into rho_P/2,
// sum_B G_B/2 and the within-key cancellation sum_K H_K/2, ASWISS_HISTORY_PROOF_KO.md Theorem H1).
// --backward 1 (with --unit history --label avg): also the backward split of section 8 (see judge_avg).
// --gate-eps E: judge only when the abstract game is solved to E: every --gate-every iterations (default --epoch0)
//            compute rho_P(avg)/2 exactly and judge when it is <= the threshold; epoch ends then only checkpoint and
//            log (their `splits` counts the splits since the previous row). Threshold: E (--gate-decay none),
//            E * sqrt(epoch0 / t) (sqrt_iters, t = iterations so far) or E / sqrt(1 + splitting judgments so far)
//            (sqrt_rounds). Every check goes to gate.csv; a judgment that splits saves the policies it judged as
//            policy_presplit_J_{avg,cur}.bin (eval_judgments.py evaluates them).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

template <class T> vector<T> load(const string& path) {
    ifstream in(path, ios::binary | ios::ate);
    if (!in) throw runtime_error("cannot open " + path);
    size_t bytes = in.tellg(); in.seekg(0);
    vector<T> data(bytes / sizeof(T));
    in.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}
template <class T> void save(const string& path, const vector<T>& data) {
    ofstream out(path, ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

constexpr int W = 4, TERMINAL = -2, CHANCE = -1;
vector<int32_t> players, info, degree, edges, counts, owner, bucket;
vector<int64_t> start;
vector<double> prob, util, delta_range;
int nI, B;
vector<double> Q, S, Strue, Rw, Sq, mass, Rtmp, ExW, ExC;
vector<double> Rp, Rp2, Rit;                   // method A: replayed window sums, squared sums, one-iteration buffer
mt19937_64 replay_rng;
vector<int32_t> prev_info, prev_action, inc_code, timestep_of;  // --unit history inputs
vector<int> history_order;                                      // information sets, earlier decisions first

// Bucket-history key of every information set under the current partition; returns the number of keys.
int history_keys(vector<int>& key) {
    map<vector<int>, int> ids;
    vector<vector<int>> past(nI);  // (bucket, action) of each earlier decision of the acting player
    key.assign(nI, -1);
    for (int I : history_order) {
        if (prev_info[I] >= 0) {
            past[I] = past[prev_info[I]];
            past[I].push_back(bucket[prev_info[I]]); past[I].push_back(prev_action[I]);
        }
        vector<int> k = past[I];
        k.push_back(timestep_of[I]); k.push_back(inc_code[I]);
        key[I] = ids.emplace(k, (int)ids.size()).first->second;
    }
    return (int)ids.size();
}
// Replace every information set's row of `x` (width w) by the sum over its key.
void pool(vector<double>& x, const vector<int>& key, int keys, int w) {
    vector<double> sum((size_t)keys * w, 0.0);
    for (int I = 0; I < nI; ++I) for (int a = 0; a < w; ++a) sum[(size_t)key[I] * w + a] += x[(size_t)I * w + a];
    for (int I = 0; I < nI; ++I) for (int a = 0; a < w; ++a) x[(size_t)I * w + a] = sum[(size_t)key[I] * w + a];
}
vector<int> touched;                           // buckets with pending regrets in Rtmp (apply_regrets visits only these)
vector<char> dirty;
double ex_weight = 0;
mt19937_64 rng;

inline void strategy(int b, int n, double* s) {
    double total = 0;
    for (int a = 0; a < n; ++a) total += max(Q[b * W + a], 0.0);
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? max(Q[b * W + a], 0.0) / total : 1.0 / n;
}
inline void touch(int b) { if (!dirty[b]) { dirty[b] = 1; touched.push_back(b); } }

// Simultaneous full-tree CFR+: returns player-0 value; p0/p1 own reaches, pc chance reach.
double full(int h, double p0, double p1, double pc, double weight) {
    int pl = players[h];
    if (pl == TERMINAL) return util[h];
    int64_t e = start[h]; int d = degree[h];
    if (pl == CHANCE) {
        double v = 0;
        for (int k = 0; k < d; ++k) v += prob[e + k] * full(edges[e + k], p0, p1, pc * prob[e + k], weight);
        return v;
    }
    int I = info[h], b = bucket[I];
    double s[W], v[W], node = 0;
    strategy(b, d, s);
    for (int a = 0; a < d; ++a) {
        v[a] = pl == 0 ? full(edges[e + a], p0 * s[a], p1, pc, weight) : full(edges[e + a], p0, p1 * s[a], pc, weight);
        node += s[a] * v[a];
    }
    double sign = pl == 0 ? 1.0 : -1.0, cf = (pl == 0 ? p1 : p0) * pc, own = pl == 0 ? p0 : p1;
    touch(b);
    for (int a = 0; a < d; ++a) {
        double r = sign * (v[a] - node) * cf;
        Rtmp[b * W + a] += r; Rw[I * W + a] += r;
        S[b * W + a] += weight * own * s[a]; Strue[I * W + a] += weight * own * s[a];
    }
    mass[I] += cf;
    return node;
}

// Diagnostic only (never used by the split rule): exact instantaneous counterfactual regrets of every original
// infoset under the current lifted strategy, added to the window sums ExW (factor) and to the cumulative
// linear-weighted sums ExC (wfactor). Called every --exact-every iterations, so window sums are time-subsampled.
double exact(int h, double p0, double p1, double pc, double factor, double wfactor) {
    int pl = players[h];
    if (pl == TERMINAL) return util[h];
    int64_t e = start[h]; int d = degree[h];
    if (pl == CHANCE) {
        double v = 0;
        for (int k = 0; k < d; ++k) v += prob[e + k] * exact(edges[e + k], p0, p1, pc * prob[e + k], factor, wfactor);
        return v;
    }
    int I = info[h];
    double s[W], v[W], node = 0;
    strategy(bucket[I], d, s);
    for (int a = 0; a < d; ++a) {
        v[a] = pl == 0 ? exact(edges[e + a], p0 * s[a], p1, pc, factor, wfactor) : exact(edges[e + a], p0, p1 * s[a], pc, factor, wfactor);
        node += s[a] * v[a];
    }
    double sign = pl == 0 ? 1.0 : -1.0, cf = (pl == 0 ? p1 : p0) * pc;
    for (int a = 0; a < d; ++a) {
        double r = sign * (v[a] - node) * cf;
        ExW[I * W + a] += factor * r; ExC[I * W + a] += wfactor * r;
    }
    return node;
}

// Exact counterfactual regrets of every original infoset under a lifted strategy, added to out: the current RM+
// strategy of the bucket tables when pol is null, otherwise pol[I * W + a].
double exact_into(int h, double p0, double p1, double pc, const double* pol, vector<double>& out) {
    int pl = players[h];
    if (pl == TERMINAL) return util[h];
    int64_t e = start[h]; int d = degree[h];
    if (pl == CHANCE) {
        double v = 0;
        for (int k = 0; k < d; ++k) v += prob[e + k] * exact_into(edges[e + k], p0, p1, pc * prob[e + k], pol, out);
        return v;
    }
    int I = info[h];
    double s[W], v[W], node = 0;
    if (pol) for (int a = 0; a < d; ++a) s[a] = pol[(size_t)I * W + a];
    else strategy(bucket[I], d, s);
    for (int a = 0; a < d; ++a) {
        v[a] = pl == 0 ? exact_into(edges[e + a], p0 * s[a], p1, pc, pol, out) : exact_into(edges[e + a], p0, p1 * s[a], pc, pol, out);
        node += s[a] * v[a];
    }
    double sign = pl == 0 ? 1.0 : -1.0, cf = (pl == 0 ? p1 : p0) * pc;
    for (int a = 0; a < d; ++a) out[(size_t)I * W + a] += sign * (v[a] - node) * cf;
    return node;
}

// Tolerant label set: bit 0 = "none" (value 0), bit a + 1 = action a; labels within 2 * eta of the best value.
uint32_t label_set(const double* x, int n, double eta) {
    double best = 0;
    for (int a = 0; a < n; ++a) best = max(best, x[a]);
    double floor = best - 2 * eta - 1e-12;
    uint32_t m = 0 >= floor ? 1u : 0u;
    for (int a = 0; a < n; ++a) if (x[a] >= floor) m |= 1u << (a + 1);
    return m;
}

// External sampling for traverser p: returns the sampled value for p.
double es(int h, int p, double weight) {
    int pl = players[h];
    if (pl == TERMINAL) return p == 0 ? util[h] : -util[h];
    int64_t e = start[h]; int d = degree[h];
    if (pl == CHANCE) {
        double x = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
        int k = 0;
        for (; k < d - 1; ++k) { acc += prob[e + k]; if (x < acc) break; }
        return es(edges[e + k], p, weight);
    }
    int I = info[h], b = bucket[I];
    double s[W];
    strategy(b, d, s);
    if (pl == p) {
        double v[W], node = 0;
        for (int a = 0; a < d; ++a) { v[a] = es(edges[e + a], p, weight); node += s[a] * v[a]; }
        touch(b);
        for (int a = 0; a < d; ++a) {
            double r = v[a] - node;
            Rtmp[b * W + a] += r; Rw[I * W + a] += r; Sq[I * W + a] += r * r;
        }
        mass[I] += 1;
        return node;
    }
    for (int a = 0; a < d; ++a) { S[b * W + a] += weight * s[a]; Strue[I * W + a] += weight * s[a]; }
    double x = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
    int k = 0;
    for (; k < d - 1; ++k) { acc += s[k]; if (x < acc) break; }
    return es(edges[e + k], p, weight);
}

// Untouched buckets have Rtmp = 0 and Q >= 0, so skipping them leaves every table bit-identical.
void apply_regrets() {
    for (int b : touched) {
        for (int i = b * W; i < (b + 1) * W; ++i) { Q[i] = max(Q[i] + Rtmp[i], 0.0); Rtmp[i] = 0; }
        dirty[b] = 0;
    }
    touched.clear();
}

vector<double> policy(const vector<double>& num, bool lifted) {
    vector<double> out((size_t)nI * W, 0.0);
    for (int I = 0; I < nI; ++I) {
        int n = counts[I]; const double* row = lifted ? &num[(size_t)bucket[I] * W] : &num[(size_t)I * W];
        double total = 0;
        for (int a = 0; a < n; ++a) total += row[a];
        for (int a = 0; a < n; ++a) out[(size_t)I * W + a] = total > 0 ? row[a] / total : 1.0 / n;
    }
    return out;
}
vector<double> current_policy() {
    vector<double> out((size_t)nI * W, 0.0);
    for (int I = 0; I < nI; ++I) strategy(bucket[I], counts[I], &out[(size_t)I * W]);
    return out;
}
inline double psi(const double* x, int n) {
    double m = 0;
    for (int a = 0; a < n; ++a) m = max(m, x[a]);
    return m;
}

// Method B judgment on the exact regrets R of the bucket average policy (one row per information set, unpooled).
// Forward split: every unit (information set, or bucket-history key with history) gets the label set of its regret sum;
// a bucket whose units share no label is split by label, children restart their tables (as the epoch-end rule above).
// Backward split (ASWISS_HISTORY_PROOF_KO.md section 8, needs history): a key whose members (information sets) share no
// label hides cancellation H_K. For each member whose representative label (lowest of its set) differs from that of the
// key's first member, walk both back to the closest earlier decision where their keys differ (they share every earlier
// bucket, so the information was merged there) and mark the two keys of that decision with the members' labels. Each
// marked bucket is split by the label sets of its keys (unmarked keys stay together); a pair the label sets leave
// together is separated by marks naming the key as well, so every judged pair ends in different children. Children of
// a backward-only split copy the parent's tables: they share the parent's label, only the memory of later keys changes.
struct Judgment { int units = 0, forward = 0, backward = 0, splits = 0, inconsistent = 0; double rho = 0, keysum = 0, rhoP = 0; };
Judgment judge_avg(vector<double> R, bool history, bool backward) {
    Judgment j;
    vector<int> key(nI);
    if (history) j.units = history_keys(key);
    else { j.units = nI; for (int I = 0; I < nI; ++I) key[I] = I; }
    vector<uint32_t> raw(nI);
    vector<double> bsum((size_t)B * W, 0.0);
    vector<int> bcount(B, 0);
    for (int I = 0; I < nI; ++I) {
        raw[I] = label_set(&R[(size_t)I * W], counts[I], 0.0);
        j.rho += psi(&R[(size_t)I * W], counts[I]);
        for (int a = 0; a < counts[I]; ++a) bsum[(size_t)bucket[I] * W + a] += R[(size_t)I * W + a];
        bcount[bucket[I]] = counts[I];
    }
    for (int b = 0; b < B; ++b) j.rhoP += psi(&bsum[(size_t)b * W], bcount[b]);
    if (history) pool(R, key, j.units, W);
    vector<int> first(j.units, -1);
    for (int I = 0; I < nI; ++I) if (first[key[I]] < 0) { first[key[I]] = I; j.keysum += psi(&R[(size_t)I * W], counts[I]); }
    // forward labels: the set of each unit's (pooled) regret, the member's child = its label shared by most members
    vector<uint32_t> m1(nI), c1(B, ~0u);
    vector<array<int, W + 1>> f1(B);
    for (auto& f : f1) f.fill(0);
    for (int I = 0; I < nI; ++I) {
        m1[I] = label_set(&R[(size_t)I * W], counts[I], 0.0);
        c1[bucket[I]] &= m1[I];
        for (int l = 0; l <= W; ++l) f1[bucket[I]][l] += (m1[I] >> l) & 1;
    }
    auto pick = [](uint32_t m, const array<int, W + 1>& f) {
        int best = -1;
        for (int l = 0; l <= W; ++l) if (((m >> l) & 1) && (best < 0 || f[l] > f[best])) best = l;
        return best;
    };
    // backward marks
    vector<int> sig(nI, -1);
    if (backward && history) {
        vector<uint32_t> common(j.units, ~0u);
        for (int I = 0; I < nI; ++I) common[key[I]] &= raw[I];
        for (int K = 0; K < j.units; ++K) j.inconsistent += common[K] == 0;
        auto rep = [](uint32_t m) { return __builtin_ctz(m); };
        map<int, uint32_t> coarse;              // marked key -> labels wanted below it
        map<int, vector<pair<int, int>>> fine;  // marked key -> (key below, label) for pairs coarse leaves together
        struct Pair { int K, lo, lo0, lam, lam0; };
        vector<Pair> pairs;
        for (int I = 0; I < nI; ++I) {
            const int K = key[I], I0 = first[K];
            if (common[K] != 0 || I == I0) continue;
            const int lam = rep(raw[I]), lam0 = rep(raw[I0]);
            if (lam == lam0) continue;
            int J = prev_info[I], J0 = prev_info[I0];
            while (J >= 0 && key[J] == key[J0]) { J = prev_info[J]; J0 = prev_info[J0]; }
            if (J < 0) throw runtime_error("two information sets with one key and no merge point");  // Lemma H3
            coarse[key[J]] |= 1u << lam; coarse[key[J0]] |= 1u << lam0;
            pairs.push_back({K, key[J], key[J0], lam, lam0});
        }
        for (const Pair& p : pairs)
            if (coarse[p.lo] == coarse[p.lo0]) { fine[p.lo].push_back({p.K, p.lam}); fine[p.lo0].push_back({p.K, p.lam0}); }
        vector<char> target(B, 0);
        for (const auto& [k, m] : coarse) target[bucket[first[k]]] = 1;
        map<vector<int>, int> ids;
        vector<int> key_sig(j.units, -1);
        for (int I = 0; I < nI; ++I) {
            if (!target[bucket[I]]) continue;
            const int k = key[I];
            if (key_sig[k] < 0) {
                vector<int> s{coarse.count(k) ? (int)coarse[k] : 0};
                if (fine.count(k)) {
                    auto& v = fine[k];
                    sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
                    for (auto [K, l] : v) { s.push_back(K); s.push_back(l); }
                }
                key_sig[k] = ids.emplace(s, (int)ids.size()).first->second;
            }
            sig[I] = key_sig[k];
        }
    }
    // children: (parent, forward label or -2 when the parent is not forward split, backward signature or -1)
    map<tuple<int, int, int>, int> child;
    vector<int> next(nI), keep, parent_of, children(B, 0);
    int B2 = 0;
    for (int I = 0; I < nI; ++I) {
        int b = bucket[I];
        int lab = c1[b] != 0 ? -2 : pick(m1[I], f1[b]) * 8;
        auto ck = make_tuple(b, lab, sig[I]);
        auto it = child.find(ck);
        if (it == child.end()) { it = child.emplace(ck, B2++).first; keep.push_back(lab == -2 ? b : -1); ++children[b]; }
        next[I] = it->second;
    }
    vector<set<int>> sigs(B);
    for (int I = 0; I < nI; ++I) if (sig[I] >= 0) sigs[bucket[I]].insert(sig[I]);
    for (int b = 0; b < B; ++b) {
        j.forward += c1[b] == 0;
        j.backward += sigs[b].size() > 1;
        j.splits += children[b] > 1;
    }
    vector<double> Q2((size_t)B2 * W, 0), S2((size_t)B2 * W, 0);
    for (int nb = 0; nb < B2; ++nb)
        if (keep[nb] >= 0)
            for (int a = 0; a < W; ++a) { Q2[nb * W + a] = Q[keep[nb] * W + a]; S2[nb * W + a] = S[keep[nb] * W + a]; }
    bucket = next; B = B2; Q = Q2; S = S2; Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
    return j;
}

int main(int argc, char** argv) {
    map<string, string> opt{{"--mode", "es"}, {"--iters", "100000"}, {"--epoch0", "1000"}, {"--split", "cum"},
                            {"--tol", "1e-3"}, {"--z", "2"}, {"--seed", "1"}, {"--exact-every", "0"},
                            {"--test", "normal"}, {"--delta", "0.05"}, {"--label", "window"},
                            {"--replay-p", "0.01"}, {"--replay-seed", "777"}};
    for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
    string dir = opt.at("--tree"), out = opt.at("--out"), mode = opt["--mode"];
    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    owner = load<int32_t>(dir + "/owner.bin"); bucket = load<int32_t>(opt.at("--buckets"));
    nI = counts.size(); B = 0;
    for (int b : bucket) B = max(B, b + 1);
    rng.seed(stoull(opt["--seed"]));
    long long T = stoll(opt["--iters"]), L = stoll(opt["--epoch0"]);
    double tol = stod(opt["--tol"]), z = stod(opt["--z"]);
    bool split = opt["--split"] == "cum";
    long long exact_every = stoll(opt["--exact-every"]);
    string test = opt["--test"];
    double delta0 = stod(opt["--delta"]);
    string label_rule = opt["--label"];
    if (label_rule != "window" && label_rule != "window_tol" && label_rule != "replay" && label_rule != "avg" &&
        label_rule != "window+avg") throw runtime_error("unknown --label " + label_rule);
    double replay_p = stod(opt["--replay-p"]);
    replay_rng.seed(stoull(opt["--replay-seed"]));
    const bool history_unit = opt.count("--unit") && opt["--unit"] == "history";
    if (opt.count("--unit") && opt["--unit"] != "history" && opt["--unit"] != "infoset") throw runtime_error("unknown --unit");
    // replay sums every member exactly in the same iteration, so pooled squared sums would not be the key's variance;
    // the sampled window sums visit at most one member of a key per traversal (the key fixes the player's actions)
    if (history_unit && label_rule == "replay") throw runtime_error("--unit history does not support --label replay");
    const bool backward = opt.count("--backward") && opt["--backward"] == "1";
    const double gate_eps = opt.count("--gate-eps") ? stod(opt["--gate-eps"]) : 0.0, L0 = (double)L;
    const long long gate_every = opt.count("--gate-every") ? stoll(opt["--gate-every"]) : L;
    const string gate_decay = opt.count("--gate-decay") ? opt["--gate-decay"] : "none";
    const bool gated = gate_eps > 0;
    if (backward && !(history_unit && label_rule == "avg" && split)) throw runtime_error("--backward 1 needs --unit history --label avg");
    if (gated && !(label_rule == "avg" && split)) throw runtime_error("--gate-eps needs --label avg --split cum");
    if (gate_decay != "none" && gate_decay != "sqrt_iters" && gate_decay != "sqrt_rounds") throw runtime_error("unknown --gate-decay");
    if (history_unit) {
        prev_info = load<int32_t>(dir + "/prev.bin"); prev_action = load<int32_t>(dir + "/prev_action.bin");
        inc_code = load<int32_t>(dir + "/inc.bin"); timestep_of = load<int32_t>(dir + "/bucket_timestep.bin");
        vector<int> depth(nI, 0);
        for (int I = 0; I < nI; ++I) for (int J = prev_info[I]; J >= 0; J = prev_info[J]) ++depth[I];
        history_order.resize(nI);
        for (int I = 0; I < nI; ++I) history_order[I] = I;
        stable_sort(history_order.begin(), history_order.end(), [&](int a, int b) { return depth[a] < depth[b]; });
    }
    if (test != "normal") delta_range = load<double>(dir + "/delta.bin");
    Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0); Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
    Strue.assign((size_t)nI * W, 0); ExW.assign((size_t)nI * W, 0); ExC.assign((size_t)nI * W, 0); Rw.assign((size_t)nI * W, 0); Sq.assign((size_t)nI * W, 0); mass.assign(nI, 0);
    if (label_rule == "replay") { Rp.assign((size_t)nI * W, 0); Rp2.assign((size_t)nI * W, 0); Rit.assign((size_t)nI * W, 0); }
    FILE* log = fopen((out + "/log.csv").c_str(), "w");
    FILE* judgment_log = nullptr; FILE* gate_log = nullptr;
    if (split && label_rule == "avg") {
        judgment_log = fopen((out + "/judgments.csv").c_str(), "w");
        fprintf(judgment_log, "judgment,iters,trigger,threshold,rho_half,HK_half,G_half,rhoP_half,buckets_before,buckets_after,"
                              "splits,forward_splits,backward_splits,inconsistent_keys,units,presplit\n");
    }
    if (gated) { gate_log = fopen((out + "/gate.csv").c_str(), "w"); fprintf(gate_log, "iters,buckets,rhoP_half,threshold,passed\n"); }
    int judgments = 0, rounds = 0, splits_since = 0;
    // one row per judgment: exact rho/2 = sum_I psi(r_I)/2 = (rho_P + sum_B G_B + sum_K H_K)/2 of the policy judged
    auto write_judgment = [&](const char* trigger, long long it, double thr, const Judgment& jd, int before, const string& stem) {
        fprintf(judgment_log, "%d,%lld,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%d,%d,%d,%d,%d,%s\n", judgments++, it, trigger, thr,
                jd.rho / 2, (jd.rho - jd.keysum) / 2, (jd.keysum - jd.rhoP) / 2, jd.rhoP / 2, before, B, jd.splits, jd.forward,
                jd.backward, jd.inconsistent, jd.units, stem.c_str());
        fflush(judgment_log);
    };
    fprintf(log, "epoch,iters,buckets_after,splits,L_per_iter,C_per_iter,H_per_iter,seconds,exact_L_per_iter,exact_C_per_iter,exact_H_per_iter,exact_bound_expl,avg_rho_half,avg_rhoP_half,avg_H_half,avg_consistent,units\n");
    auto t0 = chrono::steady_clock::now();
    long long t = 0, epoch_end = L, epoch_start = 0; int k = 0;
    while (t < T) {
        ++t;
        double weight = (double)t;  // linear averaging
        if (exact_every > 0 && t % exact_every == 0) {
            exact(0, 1, 1, 1, (double)exact_every, (double)exact_every * weight);
            ex_weight += (double)exact_every * weight;
        }
        if (label_rule == "replay" && uniform_real_distribution<double>(0, 1)(replay_rng) < replay_p) {
            fill(Rit.begin(), Rit.end(), 0.0);
            exact_into(0, 1, 1, 1, nullptr, Rit);   // regrets of the strategy used in this iteration
            for (size_t i = 0; i < Rit.size(); ++i) { Rp[i] += Rit[i]; Rp2[i] += Rit[i] * Rit[i]; }
        }
        if (mode == "full") { full(0, 1, 1, 1, weight); apply_regrets(); }
        else for (int p = 0; p < 2; ++p) { es(0, p, weight); apply_regrets(); }
        const bool sched = t == epoch_end || t == T;
        if (sched) {
            // window diagnostics from the per-infoset window sums (sampled in es mode)
            vector<double> bsum((size_t)B * W, 0.0);
            double Lw = 0, Cw = 0;
            for (int I = 0; I < nI; ++I) {
                Lw += psi(&Rw[(size_t)I * W], counts[I]);
                for (int a = 0; a < counts[I]; ++a) bsum[(size_t)bucket[I] * W + a] += Rw[(size_t)I * W + a];
            }
            vector<int> bcount(B, 0);
            for (int I = 0; I < nI; ++I) bcount[bucket[I]] = counts[I];
            for (int b = 0; b < B; ++b) Cw += psi(&bsum[(size_t)b * W], bcount[b]);
            // exact (time-subsampled) window L/C/H on the partition used during this epoch, and the cumulative
            // bound exploitability(linear average) <= sum_I psi(sum_t t r_t(I)) / (2 sum_t t)
            double eL = 0, eC = 0, eCum = 0;
            if (exact_every > 0) {
                vector<double> esum((size_t)B * W, 0.0);
                for (int I = 0; I < nI; ++I) {
                    eL += psi(&ExW[(size_t)I * W], counts[I]);
                    eCum += psi(&ExC[(size_t)I * W], counts[I]);
                    for (int a = 0; a < counts[I]; ++a) esum[(size_t)bucket[I] * W + a] += ExW[(size_t)I * W + a];
                }
                for (int b = 0; b < B; ++b) eC += psi(&esum[(size_t)b * W], bcount[b]);
                fill(ExW.begin(), ExW.end(), 0);
            }
            // regrets of the bucket average policy (the checkpoint policy_bucket_k): certificate diagnostics, method B labels
            vector<double> avg_pol = policy(S, true), Ravg((size_t)nI * W, 0.0);
            exact_into(0, 1, 1, 1, avg_pol.data(), Ravg);
            double rho = 0, rhoP = 0;
            int consistent = 0;
            {
                vector<double> asum((size_t)B * W, 0.0);
                vector<uint32_t> common(B, ~0u);
                for (int I = 0; I < nI; ++I) {
                    rho += psi(&Ravg[(size_t)I * W], counts[I]);
                    for (int a = 0; a < counts[I]; ++a) asum[(size_t)bucket[I] * W + a] += Ravg[(size_t)I * W + a];
                    common[bucket[I]] &= label_set(&Ravg[(size_t)I * W], counts[I], 0.0);
                }
                for (int b = 0; b < B; ++b) { rhoP += psi(&asum[(size_t)b * W], bcount[b]); consistent += common[b] != 0; }
            }
            const int Bprev = B;  // partition of the epoch just finished
            int splits = 0, units = nI;
            if (split && label_rule == "avg") {
                // checkpoint policies BEFORE any split (they describe the epoch just finished)
                save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
                save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
                if (!gated) {
                    Judgment jd = judge_avg(Ravg, history_unit, backward);
                    splits = jd.splits; units = jd.units;
                    write_judgment("epoch", t, -1.0, jd, Bprev, "bucket_" + to_string(k));
                } else {  // splits happen only at gate checks; this row reports those since the previous row
                    splits = splits_since; splits_since = 0;
                    if (history_unit) { vector<int> key; units = history_keys(key); }
                }
            } else if (split) {
                if (history_unit) {  // pool every label input over the bucket-history key
                    vector<int> key;
                    units = history_keys(key);
                    pool(Rw, key, units, W); pool(Sq, key, units, W); pool(mass, key, units, 1); pool(Ravg, key, units, W);
                    if (label_rule == "replay") { pool(Rp, key, units, W); pool(Rp2, key, units, W); }
                }
                vector<int> label(nI, -1);
                for (int I = 0; I < nI; ++I) {
                    if (mass[I] <= 0) continue;
                    const double* r = &Rw[(size_t)I * W]; int n = counts[I], best = 0;
                    for (int a = 1; a < n; ++a) if (r[a] > r[best]) best = a;
                    // es label tests on the window sum of sampled regrets for action `best` (n = visits this epoch):
                    //   normal:    R > z * sqrt(sum r^2)                                  (CLT approximation)
                    //   hoeffding: R > Delta_I * sqrt(2 n ln(1/d_k))                      (Azuma, increments in [-Delta_I, Delta_I])
                    //   bernstein: R > sqrt(2 n V ln(2/d_k)) + 7 w n ln(2/d_k) / (3(n-1)) (empirical Bernstein, w = 2 Delta_I)
                    // with d_k = delta / (k+1)^2, summable over epochs.
                    bool positive;
                    if (mode == "full") positive = r[best] / mass[I] > tol;
                    else if (test == "normal") positive = r[best] - z * sqrt(Sq[(size_t)I * W + best]) > 0;
                    else {
                        double n = mass[I], dk = delta0 / ((k + 1.0) * (k + 1.0)), thr;
                        if (test == "hoeffding") thr = delta_range[I] * sqrt(2 * n * std::log(1 / dk));
                        else {
                            if (n < 2) continue;
                            double mean = r[best] / n, var = max(0.0, (Sq[(size_t)I * W + best] - n * mean * mean) / (n - 1));
                            thr = sqrt(2 * n * var * std::log(2 / dk)) + 7 * (2 * delta_range[I]) * n * std::log(2 / dk) / (3 * (n - 1));
                        }
                        positive = r[best] > thr;
                    }
                    if (positive) label[I] = best;
                }
                // label sets: m1 (window rules, replay, avg) and m2 (avg, for window+avg); window = one label per set
                bool two = label_rule == "window+avg";
                vector<uint32_t> m1(nI), m2(two ? nI : 0);
                for (int I = 0; I < nI; ++I) {
                    const size_t o = (size_t)I * W; const int n = counts[I];
                    if (label_rule == "window" || two) m1[I] = 1u << (label[I] + 1);
                    else if (label_rule == "window_tol") {
                        double v = 0;
                        for (int a = 0; a < n; ++a) v = max(v, Sq[o + a]);
                        m1[I] = label_set(&Rw[o], n, mass[I] > 0 ? z * sqrt(v) : 0.0);
                    } else if (label_rule == "replay") {
                        double x[W], v = 0;
                        for (int a = 0; a < n; ++a) { x[a] = Rp[o + a] / replay_p; v = max(v, (1 - replay_p) / (replay_p * replay_p) * Rp2[o + a]); }
                        m1[I] = label_set(x, n, z * sqrt(v));
                    } else m1[I] = label_set(&Ravg[o], n, 0.0);
                    if (two) m2[I] = label_set(&Ravg[o], n, 0.0);
                }
                // per bucket: common label of each set family, and how many members carry each label
                vector<uint32_t> c1(B, ~0u), c2(B, ~0u);
                vector<array<int, W + 1>> f1(B), f2(two ? B : 0);
                for (auto& f : f1) f.fill(0);
                for (auto& f : f2) f.fill(0);
                for (int I = 0; I < nI; ++I) {
                    int b = bucket[I];
                    c1[b] &= m1[I];
                    for (int l = 0; l <= W; ++l) f1[b][l] += (m1[I] >> l) & 1;
                    if (two) { c2[b] &= m2[I]; for (int l = 0; l <= W; ++l) f2[b][l] += (m2[I] >> l) & 1; }
                }
                auto pick = [](uint32_t m, const array<int, W + 1>& f) {  // the member's label shared by most members
                    int best = -1;
                    for (int l = 0; l <= W; ++l) if (((m >> l) & 1) && (best < 0 || f[l] > f[best])) best = l;
                    return best;
                };
                map<pair<int, int>, int> child;
                vector<int> next(nI);
                int B2 = 0;
                vector<int> keep;  // parent of each unsplit new id, -1 for children
                for (int I = 0; I < nI; ++I) {
                    int b = bucket[I];
                    bool s1 = c1[b] == 0, s2 = two && c2[b] == 0;
                    int lab = !s1 && !s2 ? -2 : (s1 ? pick(m1[I], f1[b]) : 0) * 8 + (s2 ? pick(m2[I], f2[b]) : 0);
                    auto key = make_pair(b, lab);
                    auto it = child.find(key);
                    if (it == child.end()) { it = child.emplace(key, B2++).first; keep.push_back(lab == -2 ? b : -1); }
                    next[I] = it->second;
                }
                for (int b = 0; b < B; ++b) splits += c1[b] == 0 || (two && c2[b] == 0);
                vector<double> Q2((size_t)B2 * W, 0), S2((size_t)B2 * W, 0);
                for (int nb = 0; nb < B2; ++nb)
                    if (keep[nb] >= 0)
                        for (int a = 0; a < W; ++a) { Q2[nb * W + a] = Q[keep[nb] * W + a]; S2[nb * W + a] = S[keep[nb] * W + a]; }
                // checkpoint policies BEFORE applying the split (they describe the epoch just finished)
                save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
                save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
                bucket = next; B = B2; Q = Q2; S = S2; Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
            } else {
                save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
                save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
            }
            save(out + "/policy_true_" + to_string(k) + ".bin", policy(Strue, false));
            double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            double len = (double)(t - epoch_start);
            fprintf(log, "%d,%lld,%d,%d,%.9g,%.9g,%.9g,%.3f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6f,%d\n", k, t, B, splits, Lw / len, Cw / len, (Lw - Cw) / len, secs,
                    eL / len, eC / len, (eL - eC) / len, ex_weight > 0 ? eCum / (2 * ex_weight) : -1.0,
                    rho / 2, rhoP / 2, (rho - rhoP) / 2, (double)consistent / Bprev, units);
            fflush(log);
            fprintf(stderr, "epoch %d iters=%lld buckets_after=%d splits=%d L/len=%.5f C/len=%.5f H/len=%.5f (%.1fs)\n",
                    k, t, B, splits, Lw / len, Cw / len, (Lw - Cw) / len, secs);
            epoch_start = t;
            fill(Rw.begin(), Rw.end(), 0); fill(Sq.begin(), Sq.end(), 0); fill(mass.begin(), mass.end(), 0);
            if (label_rule == "replay") { fill(Rp.begin(), Rp.end(), 0); fill(Rp2.begin(), Rp2.end(), 0); }
            ++k; L *= 2; epoch_end = t + L;
            if (epoch_end > T) epoch_end = T;
        }
        if (gated && t % gate_every == 0 && t != T) {  // gate check: judge only when rho_P(avg)/2 <= threshold
            vector<double> gpol = policy(S, true), gR((size_t)nI * W, 0.0);
            exact_into(0, 1, 1, 1, gpol.data(), gR);
            vector<double> gsum((size_t)B * W, 0.0);
            vector<int> gcount(B, 0);
            for (int I = 0; I < nI; ++I) {
                for (int a = 0; a < counts[I]; ++a) gsum[(size_t)bucket[I] * W + a] += gR[(size_t)I * W + a];
                gcount[bucket[I]] = counts[I];
            }
            double grhoP = 0;
            for (int b = 0; b < B; ++b) grhoP += psi(&gsum[(size_t)b * W], gcount[b]);
            const double thr = gate_eps * (gate_decay == "sqrt_iters" ? sqrt(L0 / (double)t)
                                           : gate_decay == "sqrt_rounds" ? 1 / sqrt(rounds + 1.0) : 1.0);
            const bool pass = grhoP / 2 <= thr;
            fprintf(gate_log, "%lld,%d,%.9g,%.9g,%d\n", t, B, grhoP / 2, thr, (int)pass);
            fflush(gate_log);
            if (pass) {
                const vector<double> gcur = current_policy();
                const int Bb = B;
                Judgment jd = judge_avg(gR, history_unit, backward);
                string stem = "-";
                if (jd.splits > 0) {  // the policies the judgment saw, for the pre-split evaluation
                    stem = "presplit_" + to_string(judgments);
                    save(out + "/policy_" + stem + "_avg.bin", gpol); save(out + "/policy_" + stem + "_cur.bin", gcur);
                    splits_since += jd.splits; ++rounds;
                }
                write_judgment("gate", t, thr, jd, Bb, stem);
            }
        }
    }
    save(out + "/final_bucket_map.bin", bucket);
    fclose(log);
    if (judgment_log) fclose(judgment_log);
    if (gate_log) fclose(gate_log);
}
