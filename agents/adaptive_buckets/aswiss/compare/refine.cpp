// Exact (full-traversal) comparison of abstraction-refinement rules and CFR variants on an exported game tree
// (../export.py, ../small/export_small.py). Writes checkpoints in the format ../eval.py evaluates.
//
// Build: g++ -O3 -std=c++17 refine.cpp -o refine
// Run:   ./refine --tree DIR --out OUTDIR [--buckets FILE | --init ira] [--iters 81910] [--epoch0 10]
//                 [--update alt|sim] [--solver cfr+|dcfr|pcfr+] [--delay 0]
//                 [--refine none|label|ira] [--tol 1e-3] [--kb 10] [--kh 90] [--lscale 100] [--seed 1]
//
// Iterations: one iteration updates both players once. alt (default): player 0's half-step, then player 1's, each a
// full traversal with the current strategies (player 1 already sees player 0's update). sim: one traversal, both
// players updated from the same profile (the setting of ASWISS_CONVERGENCE_KO.md and ../aswiss.cpp --mode full).
// Solvers (bucket tables; with an identity --buckets they are the plain algorithms):
//   cfr+   RM+ (regrets clipped at 0), average weighted by max(0, t - delay)            (Tammelin 2014)
//   dcfr   DCFR(1.5, 0, 2): positive regrets x t^1.5/(t^1.5+1), negative x 1/2, average weight t^2
//                                                                                        (Brown & Sandholm 2019)
//   pcfr+  predictive RM+ (strategy from [Q + last instantaneous regret]^+), average weight t^2 (Farina et al. 2021)
// Refinement (buckets only split, never merge; a split resets the tables of every bucket it creates or changes):
//   none   fixed abstraction
//   label  ASWISS theorem-1 rule (ASWISS_CONVERGENCE_KO.md): at the end of each doubling epoch (t = L0, 3L0, 7L0, ...)
//          label(I) = argmax_a R_I(a) if R_I(best) / (counterfactual reach summed over the epoch) > tol, else none,
//          from the window sums R_I of the exact regrets of every original information set; a bucket whose members
//          carry two or more labels is split by label (unsplit buckets keep their tables; same as ../aswiss.cpp)
//   ira    CFR+IRA (Cermak, Lisy, Bosansky, arXiv 1803.05392, Algorithms 6-10), alt + cfr+ only. Half-step counter t
//          (one player's update = one t, as in the paper):
//          heuristic: every half-step, kh information sets of the acting player in abstracted buckets (Algorithm 8,
//            buckets drawn with probability proportional to size) record their regret of that half-step; in each
//            bucket the sampled members are grouped by gamma = {a : r(a) >= max r - 1/(5 sqrt t)}; with two or more
//            groups the largest group keeps the unsampled members and every group becomes a bucket. (With one group
//            nothing changes: the paper's pseudocode would re-create, and so reset, the bucket; we read that as
//            unintended.)
//          regret bound: at t = T_j (T_{j+1} - T_j = 2^j) kb information sets of both players are drawn the same
//            way and their regrets are summed from T_j; at every later half-step a monitored I with
//            max_a R(a) / (t - T_j) > Delta_I sqrt|A(I)| / (lscale sqrt t) is cut out as a bucket of its own
//            (section 5.1 uses lscale = 100).
//          --init ira starts from the paper's initial abstraction (section 4.1.1): per player, one bucket per
//          (length of the player's own action sequence, number of actions).
// Checkpoints: at the epoch ends and at --iters: policy_{bucket,true,current}_k.bin (bucket average lifted, the
// partition-independent sequence-form average of each information set, current strategy) and a log.csv row; for
// label, the policies are those of the epoch just finished (before its split), as in ../aswiss.cpp.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
vector<int32_t> players, info, degree, edges, counts, owner;
vector<int64_t> start;
vector<double> prob, util, delta_range;
int nI = 0;

enum class Solver { CfrPlus, Dcfr, PcfrPlus };
Solver solver = Solver::CfrPlus;
long long delay = 0;

// abstraction
vector<int> bucket;                 // information set -> bucket
vector<vector<int>> members;        // bucket -> information sets
vector<int> bowner, bcount;         // bucket -> player, number of actions
vector<double> Q, S, M, R;          // per bucket x W: regrets, average numerator, prediction, pending regrets
// per information set
vector<double> Strue, Rw, mass;     // sequence-form average numerator; label window sums and reach
vector<int> hslot, bslot;           // index into rh / rb, or -1 (CFR+IRA monitored sets)
vector<double> rh, rb;
bool track_window = false;
double avg_weight = 0;              // weight of the current half-step's average contribution

inline void strategy(int b, double* s) {
    const int n = bcount[b];
    const double* q = &Q[(size_t)b * W];
    const double* m = &M[(size_t)b * W];
    double x[W], total = 0;
    for (int a = 0; a < n; ++a) { x[a] = max(solver == Solver::PcfrPlus ? q[a] + m[a] : q[a], 0.0); total += x[a]; }
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? x[a] / total : 1.0 / n;
}

// Full traversal: returns player-0 value. upd = player whose regrets/average are updated, or -1 for both (sim).
double walk(int h, double p0, double p1, double pc, int upd) {
    const int pl = players[h];
    if (pl == TERMINAL) return util[h];
    const int64_t e = start[h]; const int d = degree[h];
    if (pl == CHANCE) {
        double v = 0;
        for (int k = 0; k < d; ++k) v += prob[e + k] * walk(edges[e + k], p0, p1, pc * prob[e + k], upd);
        return v;
    }
    const int I = info[h], b = bucket[I];
    double s[W], v[W], node = 0;
    strategy(b, s);
    for (int a = 0; a < d; ++a) {
        v[a] = pl == 0 ? walk(edges[e + a], p0 * s[a], p1, pc, upd) : walk(edges[e + a], p0, p1 * s[a], pc, upd);
        node += s[a] * v[a];
    }
    if (upd >= 0 && upd != pl) return node;
    const double sign = pl == 0 ? 1.0 : -1.0, cf = (pl == 0 ? p1 : p0) * pc, own = pl == 0 ? p0 : p1;
    const size_t bo = (size_t)b * W, io = (size_t)I * W;
    const int hs = hslot[I], bs = bslot[I];
    for (int a = 0; a < d; ++a) {
        const double r = sign * (v[a] - node) * cf;
        R[bo + a] += r;
        if (track_window) Rw[io + a] += r;
        if (hs >= 0) rh[(size_t)hs * W + a] += r;
        if (bs >= 0) rb[(size_t)bs * W + a] += r;
        if (avg_weight > 0) { S[bo + a] += avg_weight * own * s[a]; Strue[io + a] += avg_weight * own * s[a]; }
    }
    if (track_window) mass[I] += cf;
    return node;
}

// regrets of the updated player(s) enter the tables; n = 1-based iteration count of that player
void apply(int upd, long long n) {
    const double pos = pow((double)n, 1.5) / (pow((double)n, 1.5) + 1.0), neg = 0.5;
    const int B = (int)bowner.size();
    for (int b = 0; b < B; ++b) {
        if (upd >= 0 && bowner[b] != upd) continue;
        for (size_t i = (size_t)b * W; i < (size_t)(b + 1) * W; ++i) {
            const double r = R[i];
            if (solver == Solver::Dcfr) { const double q = Q[i] + r; Q[i] = q * (q > 0 ? pos : neg); }
            else Q[i] = max(Q[i] + r, 0.0);
            if (solver == Solver::PcfrPlus) M[i] = r;
            R[i] = 0;
        }
    }
}
double weight_of(long long n) {  // average weight of iteration n (1-based)
    if (solver != Solver::CfrPlus) return (double)n * (double)n;
    return (double)max(0LL, n - delay);
}

int new_bucket(int pl, int n) {
    bowner.push_back(pl); bcount.push_back(n); members.emplace_back();
    for (auto* x : {&Q, &S, &M, &R}) x->insert(x->end(), W, 0.0);
    return (int)bowner.size() - 1;
}
void reset_bucket(int b) {
    for (auto* x : {&Q, &S, &M, &R}) fill(x->begin() + (size_t)b * W, x->begin() + (size_t)(b + 1) * W, 0.0);
}
// Replace bucket b by the given groups (a partition of its members, at least two): the first keeps id b; all reset.
void split_bucket(int b, const vector<vector<int>>& groups) {
    const int pl = bowner[b], n = bcount[b];
    members[b] = groups[0];
    reset_bucket(b);
    for (size_t g = 1; g < groups.size(); ++g) {
        const int nb = new_bucket(pl, n);
        members[nb] = groups[g];
        for (int I : groups[g]) bucket[I] = nb;
    }
}

// Algorithm 8: up to k information sets of the given player (-1: both) in buckets with two or more members; buckets
// are drawn without replacement with probability proportional to their size (Efraimidis-Spirakis keys).
vector<int> sample_sets(int pl, int k, mt19937_64& rng) {
    vector<pair<double, int>> keyed;
    uniform_real_distribution<double> u(0.0, 1.0);
    for (int b = 0; b < (int)bowner.size(); ++b)
        if ((pl < 0 || bowner[b] == pl) && members[b].size() >= 2)
            keyed.push_back({-log(1.0 - u(rng)) / (double)members[b].size(), b});
    sort(keyed.begin(), keyed.end());
    vector<int> out;
    for (const auto& [key, b] : keyed) {
        if ((int)out.size() >= k) break;
        const int room = k - (int)out.size();
        if ((int)members[b].size() <= room) out.insert(out.end(), members[b].begin(), members[b].end());
        else {
            vector<int> m = members[b];
            shuffle(m.begin(), m.end(), rng);
            out.insert(out.end(), m.begin(), m.begin() + room);
        }
    }
    return out;
}

vector<double> lifted_average() {
    vector<double> out((size_t)nI * W, 0.0);
    for (int I = 0; I < nI; ++I) {
        const int n = counts[I]; const double* row = &S[(size_t)bucket[I] * W];
        double total = 0;
        for (int a = 0; a < n; ++a) total += row[a];
        for (int a = 0; a < n; ++a) out[(size_t)I * W + a] = total > 0 ? row[a] / total : 1.0 / n;
    }
    return out;
}
vector<double> true_average() {
    vector<double> out((size_t)nI * W, 0.0);
    for (int I = 0; I < nI; ++I) {
        const int n = counts[I]; const double* row = &Strue[(size_t)I * W];
        double total = 0;
        for (int a = 0; a < n; ++a) total += row[a];
        for (int a = 0; a < n; ++a) out[(size_t)I * W + a] = total > 0 ? row[a] / total : 1.0 / n;
    }
    return out;
}
vector<double> current() {
    vector<double> out((size_t)nI * W, 0.0);
    for (int I = 0; I < nI; ++I) strategy(bucket[I], &out[(size_t)I * W]);
    return out;
}

// paper's initial abstraction: per player, (length of own action sequence, number of actions)
vector<int> ira_init() {
    vector<int> len(nI, -1);
    vector<array<int, 3>> stack{{0, 0, 0}};  // node, own-sequence lengths of players 0 and 1
    while (!stack.empty()) {
        auto [h, l0, l1] = stack.back(); stack.pop_back();
        const int pl = players[h];
        if (pl == TERMINAL) continue;
        const int64_t e = start[h];
        if (pl >= 0) {
            const int I = info[h], l = pl == 0 ? l0 : l1;
            if (len[I] >= 0 && len[I] != l) throw runtime_error("own sequence length differs within an information set");
            len[I] = l;
        }
        for (int k = 0; k < degree[h]; ++k) stack.push_back({edges[e + k], l0 + (pl == 0), l1 + (pl == 1)});
    }
    map<array<int, 3>, int> ids;
    vector<int> b(nI);
    for (int I = 0; I < nI; ++I) b[I] = ids.emplace(array<int, 3>{owner[I], len[I], counts[I]}, (int)ids.size()).first->second;
    return b;
}

int main(int argc, char** argv) {
    map<string, string> o{{"--iters", "81910"}, {"--epoch0", "10"}, {"--update", "alt"}, {"--solver", "cfr+"},
                          {"--delay", "0"}, {"--refine", "none"}, {"--tol", "1e-3"}, {"--kb", "10"}, {"--kh", "90"},
                          {"--lscale", "100"}, {"--seed", "1"}};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (string(argv[i]) != "--tree" && string(argv[i]) != "--out" && string(argv[i]) != "--buckets" &&
            string(argv[i]) != "--init" && !o.count(argv[i]))
            throw runtime_error(string("unknown option ") + argv[i]);
        o[argv[i]] = argv[i + 1];
    }
    const string dir = o.at("--tree"), out = o.at("--out");
    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    owner = load<int32_t>(dir + "/owner.bin"); delta_range = load<double>(dir + "/delta.bin");
    nI = (int)counts.size();
    const string s = o["--solver"], up = o["--update"], ref = o["--refine"];
    solver = s == "cfr+" ? Solver::CfrPlus : s == "dcfr" ? Solver::Dcfr : s == "pcfr+" ? Solver::PcfrPlus
             : throw runtime_error("--solver must be cfr+, dcfr or pcfr+");
    if (up != "alt" && up != "sim") throw runtime_error("--update must be alt or sim");
    if (ref != "none" && ref != "label" && ref != "ira") throw runtime_error("--refine must be none, label or ira");
    if (ref == "ira" && (up != "alt" || solver != Solver::CfrPlus)) throw runtime_error("--refine ira needs --update alt --solver cfr+");
    delay = stoll(o["--delay"]);
    const long long T = stoll(o["--iters"]), L0 = stoll(o["--epoch0"]);
    const double tol = stod(o["--tol"]), lscale = stod(o["--lscale"]);
    const int kb = stoi(o["--kb"]), kh = stoi(o["--kh"]);
    mt19937_64 rng(stoull(o["--seed"]));
    if (o.count("--buckets") == o.count("--init")) throw runtime_error("give exactly one of --buckets FILE and --init ira");
    if (o.count("--init") && o["--init"] != "ira") throw runtime_error("--init must be ira");
    const vector<int> b0 = o.count("--buckets") ? [&] { auto v = load<int32_t>(o["--buckets"]); return vector<int>(v.begin(), v.end()); }() : ira_init();
    if ((int)b0.size() != nI) throw runtime_error("bucket map size differs from the number of information sets");
    {   // renumber densely; members of a bucket must share player and number of actions
        map<int, int> id;
        bucket.assign(nI, -1);
        for (int I = 0; I < nI; ++I) {
            auto [it, fresh] = id.emplace(b0[I], (int)id.size());
            if (fresh) new_bucket(owner[I], counts[I]);
            const int b = it->second;
            if (bowner[b] != owner[I] || bcount[b] != counts[I]) throw runtime_error("a bucket mixes players or action counts");
            bucket[I] = b; members[b].push_back(I);
        }
    }
    Strue.assign((size_t)nI * W, 0); hslot.assign(nI, -1); bslot.assign(nI, -1);
    track_window = ref == "label";
    if (track_window) { Rw.assign((size_t)nI * W, 0); mass.assign(nI, 0); }

    FILE* log = fopen((out + "/log.csv").c_str(), "w");
    if (!log) throw runtime_error("cannot write " + out + "/log.csv (does the directory exist?)");
    fprintf(log, "epoch,iters,buckets_after,splits,splits_heuristic,splits_bound,tracked_sets,seconds\n");
    const auto t0 = chrono::steady_clock::now();
    long long epoch_end = L0, L = L0, half = 0;  // half: CFR+IRA's t (half-steps so far)
    int k = 0, splits_h = 0, splits_b = 0, splits_label = 0;
    long long T_last = 0, T_next = 1; int jexp = 0;  // regret-bound update schedule (Algorithm 6 lines 9-14)
    vector<int> monitored;                            // Î_b
    for (long long it = 1; it <= T; ++it) {
        if (up == "sim") {
            avg_weight = weight_of(it);
            walk(0, 1, 1, 1, -1);
            apply(-1, it);
        } else {
            for (int pl = 0; pl < 2; ++pl) {
                ++half;
                vector<int> hsets;
                if (ref == "ira") {
                    if (half == T_next) {  // new monitored set Î_b
                        for (int I : monitored) bslot[I] = -1;
                        monitored = sample_sets(-1, kb, rng);
                        rb.assign(monitored.size() * W, 0.0);
                        for (size_t j = 0; j < monitored.size(); ++j) bslot[monitored[j]] = (int)j;
                        T_last = half; T_next = half + (1LL << jexp); ++jexp;
                    }
                    hsets = sample_sets(pl, kh, rng);
                    rh.assign(hsets.size() * W, 0.0);
                    for (size_t j = 0; j < hsets.size(); ++j) hslot[hsets[j]] = (int)j;
                }
                avg_weight = weight_of(it);
                walk(0, 1, 1, 1, pl);
                apply(pl, it);
                if (ref != "ira") continue;
                // heuristic update (Algorithm 10)
                const double tol_h = 1.0 / (5.0 * sqrt((double)half));
                map<int, map<uint32_t, vector<int>>> groups;  // bucket -> gamma -> sampled members
                for (size_t j = 0; j < hsets.size(); ++j) {
                    const int I = hsets[j], n = counts[I];
                    const double* r = &rh[j * W];
                    double best = r[0];
                    for (int a = 1; a < n; ++a) best = max(best, r[a]);
                    uint32_t gamma = 0;
                    for (int a = 0; a < n; ++a) if (r[a] >= best - tol_h) gamma |= 1u << a;
                    groups[bucket[I]][gamma].push_back(I);
                }
                for (int I : hsets) hslot[I] = -1;
                for (auto& [b, byg] : groups) {
                    if (byg.size() < 2) continue;
                    vector<vector<int>> parts;
                    for (auto& [g, v] : byg) parts.push_back(v);
                    stable_sort(parts.begin(), parts.end(), [](const vector<int>& x, const vector<int>& y) { return x.size() > y.size(); });
                    vector<int> rest;  // unsampled members join the largest group
                    {
                        vector<int> in;
                        for (auto& p : parts) in.insert(in.end(), p.begin(), p.end());
                        sort(in.begin(), in.end());
                        for (int I : members[b]) if (!binary_search(in.begin(), in.end(), I)) rest.push_back(I);
                    }
                    parts[0].insert(parts[0].end(), rest.begin(), rest.end());
                    split_bucket(b, parts);
                    ++splits_h;
                }
                // regret-bound update (Algorithm 9)
                if (half != T_last) {
                    const double span = (double)(half - T_last);
                    vector<int> keep;
                    for (size_t j = 0; j < monitored.size(); ++j) {
                        const int I = monitored[j], b = bucket[I], n = counts[I];
                        if (members[b].size() < 2) { bslot[I] = -1; continue; }  // already alone
                        const double* r = &rb[(size_t)bslot[I] * W];
                        double best = r[0];
                        for (int a = 1; a < n; ++a) best = max(best, r[a]);
                        const double bound = delta_range[I] * sqrt((double)n) / (lscale * sqrt((double)half));
                        if (best / span > bound) {
                            vector<int> rest;
                            for (int x : members[b]) if (x != I) rest.push_back(x);
                            split_bucket(b, {rest, {I}});
                            ++splits_b; bslot[I] = -1;
                        } else keep.push_back(I);
                    }
                    if (keep.size() != monitored.size()) {  // keep slots of the remaining ones
                        vector<double> rb2(keep.size() * W);
                        for (size_t j = 0; j < keep.size(); ++j) {
                            const int I = keep[j];
                            copy(rb.begin() + (size_t)bslot[I] * W, rb.begin() + (size_t)(bslot[I] + 1) * W, rb2.begin() + j * W);
                            bslot[I] = (int)j;
                        }
                        monitored = keep; rb = rb2;
                    }
                }
            }
        }
        if (it != epoch_end && it != T) continue;
        // checkpoint
        save(out + "/policy_bucket_" + to_string(k) + ".bin", lifted_average());
        save(out + "/policy_current_" + to_string(k) + ".bin", current());
        save(out + "/policy_true_" + to_string(k) + ".bin", true_average());
        int splits = 0;
        if (ref == "label") {
            vector<int> label(nI, -1);
            for (int I = 0; I < nI; ++I) {
                if (mass[I] <= 0) continue;
                const double* r = &Rw[(size_t)I * W];
                int best = 0;
                for (int a = 1; a < counts[I]; ++a) if (r[a] > r[best]) best = a;
                if (r[best] / mass[I] > tol) label[I] = best;
            }
            const int B = (int)bowner.size();
            for (int b = 0; b < B; ++b) {
                map<int, vector<int>> byl;
                for (int I : members[b]) byl[label[I]].push_back(I);
                if (byl.size() < 2) continue;
                // children in order of first member, like ../aswiss.cpp; all of them restart their tables
                vector<vector<int>> parts;
                vector<int> seen;
                for (int I : members[b]) {
                    if (find(seen.begin(), seen.end(), label[I]) != seen.end()) continue;
                    seen.push_back(label[I]); parts.push_back(byl[label[I]]);
                }
                split_bucket(b, parts);
                ++splits;
            }
            splits_label += splits;
            fill(Rw.begin(), Rw.end(), 0.0); fill(mass.begin(), mass.end(), 0.0);
        }
        const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        const int tracked = ref == "label" ? nI : ref == "ira" ? kb + kh : 0;
        fprintf(log, "%d,%lld,%zu,%d,%d,%d,%d,%.3f\n", k, it, bowner.size(), ref == "label" ? splits : splits_h + splits_b,
                splits_h, splits_b, tracked, secs);
        fflush(log);
        fprintf(stderr, "checkpoint %d iters=%lld buckets=%zu (%.1fs)\n", k, it, bowner.size(), secs);
        if (ref == "ira") splits_h = splits_b = 0;
        ++k; L *= 2; epoch_end = it + L;
    }
    vector<int32_t> final_map(bucket.begin(), bucket.end());
    save(out + "/final_bucket_map.bin", final_map);
    fclose(log);
}
