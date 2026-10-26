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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
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
double ex_weight = 0;
mt19937_64 rng;

inline void strategy(int b, int n, double* s) {
    double total = 0;
    for (int a = 0; a < n; ++a) total += max(Q[b * W + a], 0.0);
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? max(Q[b * W + a], 0.0) / total : 1.0 / n;
}

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

void apply_regrets() {
    for (int i = 0; i < B * W; ++i) { Q[i] = max(Q[i] + Rtmp[i], 0.0); Rtmp[i] = 0; }
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

int main(int argc, char** argv) {
    map<string, string> opt{{"--mode", "es"}, {"--iters", "100000"}, {"--epoch0", "1000"}, {"--split", "cum"},
                            {"--tol", "1e-3"}, {"--z", "2"}, {"--seed", "1"}, {"--exact-every", "0"},
                            {"--test", "normal"}, {"--delta", "0.05"}};
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
    if (test != "normal") delta_range = load<double>(dir + "/delta.bin");
    Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0); Rtmp.assign((size_t)B * W, 0);
    Strue.assign((size_t)nI * W, 0); ExW.assign((size_t)nI * W, 0); ExC.assign((size_t)nI * W, 0); Rw.assign((size_t)nI * W, 0); Sq.assign((size_t)nI * W, 0); mass.assign(nI, 0);
    FILE* log = fopen((out + "/log.csv").c_str(), "w");
    fprintf(log, "epoch,iters,buckets_after,splits,L_per_iter,C_per_iter,H_per_iter,seconds,exact_L_per_iter,exact_C_per_iter,exact_H_per_iter,exact_bound_expl\n");
    auto t0 = chrono::steady_clock::now();
    long long t = 0, epoch_end = L, epoch_start = 0; int k = 0;
    while (t < T) {
        ++t;
        double weight = (double)t;  // linear averaging
        if (exact_every > 0 && t % exact_every == 0) {
            exact(0, 1, 1, 1, (double)exact_every, (double)exact_every * weight);
            ex_weight += (double)exact_every * weight;
        }
        if (mode == "full") { full(0, 1, 1, 1, weight); apply_regrets(); }
        else for (int p = 0; p < 2; ++p) { es(0, p, weight); apply_regrets(); }
        if (t != epoch_end && t != T) continue;
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
        int splits = 0;
        if (split) {
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
            vector<map<int, int>> seen(B);
            for (int I = 0; I < nI; ++I) seen[bucket[I]][label[I]] = 0;
            map<pair<int, int>, int> child;
            vector<int> next(nI);
            int B2 = 0;
            vector<int> keep;  // parent of each unsplit new id, -1 for children
            for (int I = 0; I < nI; ++I) {
                int b = bucket[I], lab = seen[b].size() > 1 ? label[I] : -2;
                auto key = make_pair(b, lab);
                auto it = child.find(key);
                if (it == child.end()) { it = child.emplace(key, B2++).first; keep.push_back(lab == -2 ? b : -1); }
                next[I] = it->second;
            }
            for (int b = 0; b < B; ++b) splits += seen[b].size() > 1;
            vector<double> Q2((size_t)B2 * W, 0), S2((size_t)B2 * W, 0);
            for (int nb = 0; nb < B2; ++nb)
                if (keep[nb] >= 0)
                    for (int a = 0; a < W; ++a) { Q2[nb * W + a] = Q[keep[nb] * W + a]; S2[nb * W + a] = S[keep[nb] * W + a]; }
            // checkpoint policies BEFORE applying the split (they describe the epoch just finished)
            save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
            save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
            bucket = next; B = B2; Q = Q2; S = S2; Rtmp.assign((size_t)B * W, 0);
        } else {
            save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
            save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
        }
        save(out + "/policy_true_" + to_string(k) + ".bin", policy(Strue, false));
        double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        double len = (double)(t - epoch_start);
        fprintf(log, "%d,%lld,%d,%d,%.9g,%.9g,%.9g,%.3f,%.9g,%.9g,%.9g,%.9g\n", k, t, B, splits, Lw / len, Cw / len, (Lw - Cw) / len, secs,
                eL / len, eC / len, (eL - eC) / len, ex_weight > 0 ? eCum / (2 * ex_weight) : -1.0);
        fflush(log);
        fprintf(stderr, "epoch %d iters=%lld buckets_after=%d splits=%d L/len=%.5f C/len=%.5f H/len=%.5f (%.1fs)\n",
                k, t, B, splits, Lw / len, Cw / len, (Lw - Cw) / len, secs);
        epoch_start = t;
        fill(Rw.begin(), Rw.end(), 0); fill(Sq.begin(), Sq.end(), 0); fill(mass.begin(), mass.end(), 0);
        ++k; L *= 2; epoch_end = t + L;
        if (epoch_end > T) epoch_end = T;
    }
    save(out + "/final_bucket_map.bin", bucket);
    fclose(log);
}
