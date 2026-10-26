// ASWISS with convergence-gated splitting: split decisions are made only once the abstract game has
// (approximately) converged since the last split, measured by the sampled bucket-level average regret
//   C~/len = sum_B max(0, max_a sum_{t in window} r~_t(B,a)) / len   <=  eps.
//
// Build: g++ -O3 -std=c++17 gated.cpp -o gated
// Run:   ./gated --tree DIR --buckets FILE --iters T --eps 0.1 [--test normal|bernstein|hoeffding] [--z 2]
//                [--delta 0.05] [--gate-check 1000] [--gate-min 10000] [--inherit 0] [--exact-every 100]
//                [--checkpoint0 10000] [--seed 1] --out OUTDIR
// Training: external-sampling MCCFR+ on bucket tables (as aswiss.cpp --mode es). The window starts at the last
// iteration that changed the partition. Every gate-check iterations (once the window has gate-min iterations)
// the gate compares C~/len with eps; when it passes, every information set gets a label from its window sum of
// sampled regrets (normal: R > z sqrt(sum r^2); bernstein/hoeffding as in aswiss.cpp with level
// delta/(d+1)^2 for the d-th decision) and buckets with two or more labels are split. A decision that splits
// nothing keeps the window running. Children restart their tables unless --inherit 1.
// Outputs: log.csv and policy_{bucket,true,current}_k.bin at fixed checkpoints (10k, 30k, 70k, ... doubling,
// the same budgets as aswiss.cpp), and decisions.csv with one row per gate pass. The exact columns come from a
// diagnostic full traversal every exact-every iterations and never influence training.
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
vector<int32_t> players, info, degree, edges, counts, bucket;
vector<int64_t> start;
vector<double> prob, util, delta_range;
int nI, B;
vector<double> Q, S, Strue, Rtmp, Bw;          // bucket tables, per-traversal bucket regrets, window bucket sums
vector<double> Rw, Sq, mass;                   // per-infoset window sums of sampled regrets (labels)
vector<double> ExK, ExG, ExC;                  // exact diagnostics: since checkpoint, since window start, cumulative
double ex_weight = 0;
mt19937_64 rng;
uniform_real_distribution<double> U(0, 1);

inline void strategy(int b, int n, double* s) {
    double total = 0;
    for (int a = 0; a < n; ++a) total += max(Q[b * W + a], 0.0);
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? max(Q[b * W + a], 0.0) / total : 1.0 / n;
}
inline int sample(const double* p, int d) {
    double x = U(rng), acc = 0;
    int k = 0;
    for (; k < d - 1; ++k) { acc += p[k]; if (x < acc) break; }
    return k;
}
double es(int h, int p, double weight) {
    int pl = players[h];
    if (pl == TERMINAL) return p == 0 ? util[h] : -util[h];
    int64_t e = start[h]; int d = degree[h];
    if (pl == CHANCE) return es(edges[e + sample(&prob[e], d)], p, weight);
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
    return es(edges[e + sample(s, d)], p, weight);
}
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
        ExK[I * W + a] += factor * r; ExG[I * W + a] += factor * r; ExC[I * W + a] += wfactor * r;
    }
    return node;
}
void apply_regrets() {
    for (int i = 0; i < B * W; ++i) { Bw[i] += Rtmp[i]; Q[i] = max(Q[i] + Rtmp[i], 0.0); Rtmp[i] = 0; }
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
inline double psi(const double* x, int n) { double m = 0; for (int a = 0; a < n; ++a) m = max(m, x[a]); return m; }
// exact L and C of per-infoset sums X on the current partition
pair<double, double> exact_LC(const vector<double>& X) {
    vector<double> bs((size_t)B * W, 0.0); vector<int> bc(B, 0);
    double L = 0, C = 0;
    for (int I = 0; I < nI; ++I) {
        L += psi(&X[(size_t)I * W], counts[I]); bc[bucket[I]] = counts[I];
        for (int a = 0; a < counts[I]; ++a) bs[(size_t)bucket[I] * W + a] += X[(size_t)I * W + a];
    }
    for (int b = 0; b < B; ++b) C += psi(&bs[(size_t)b * W], bc[b]);
    return {L, C};
}

int main(int argc, char** argv) {
    map<string, string> opt{{"--iters", "2560000"}, {"--eps", "0.1"}, {"--test", "normal"}, {"--z", "2"}, {"--delta", "0.05"},
                            {"--gate-check", "1000"}, {"--gate-min", "10000"}, {"--inherit", "0"}, {"--exact-every", "100"},
                            {"--checkpoint0", "10000"}, {"--seed", "1"}};
    for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
    string dir = opt.at("--tree"), out = opt.at("--out"), test = opt["--test"];
    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    delta_range = load<double>(dir + "/delta.bin"); bucket = load<int32_t>(opt.at("--buckets"));
    nI = counts.size(); B = 0;
    for (int b : bucket) B = max(B, b + 1);
    rng.seed(stoull(opt["--seed"]));
    long long T = stoll(opt["--iters"]), gate_check = stoll(opt["--gate-check"]), gate_min = stoll(opt["--gate-min"]);
    long long exact_every = stoll(opt["--exact-every"]), next_ckpt = stoll(opt["--checkpoint0"]), ckpt_len = next_ckpt;
    double eps = stod(opt["--eps"]), z = stod(opt["--z"]), delta0 = stod(opt["--delta"]);
    bool inherit = opt["--inherit"] == "1";
    Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0); Rtmp.assign((size_t)B * W, 0); Bw.assign((size_t)B * W, 0);
    Strue.assign((size_t)nI * W, 0); Rw.assign((size_t)nI * W, 0); Sq.assign((size_t)nI * W, 0); mass.assign(nI, 0);
    ExK.assign((size_t)nI * W, 0); ExG.assign((size_t)nI * W, 0); ExC.assign((size_t)nI * W, 0);
    FILE* logf = fopen((out + "/log.csv").c_str(), "w");
    fprintf(logf, "epoch,iters,buckets_after,splits,decisions,seconds,exact_L_per_iter,exact_C_per_iter,exact_H_per_iter,exact_bound_expl\n");
    FILE* decf = fopen((out + "/decisions.csv").c_str(), "w");
    fprintf(decf, "iters,window_len,sampled_C_per_iter,exact_C_per_iter,exact_H_per_iter,buckets_before,splits,buckets_after\n");
    auto t0 = chrono::steady_clock::now();
    long long t = 0, window_start = 0, last_ckpt = 0; int k = 0, decisions = 0, splits_total = 0;
    while (t < T) {
        ++t;
        double weight = (double)t;
        if (exact_every > 0 && t % exact_every == 0) {
            exact(0, 1, 1, 1, (double)exact_every, (double)exact_every * weight);
            ex_weight += (double)exact_every * weight;
        }
        for (int p = 0; p < 2; ++p) { es(0, p, weight); apply_regrets(); }
        // ---- convergence gate
        long long len = t - window_start;
        if (len >= gate_min && t % gate_check == 0) {
            vector<int> bc(B, 0);
            for (int I = 0; I < nI; ++I) bc[bucket[I]] = counts[I];
            double Cs = 0;
            for (int b = 0; b < B; ++b) Cs += psi(&Bw[(size_t)b * W], bc[b]);
            if (Cs / len <= eps) {
                double dk = delta0 / ((decisions + 1.0) * (decisions + 1.0));
                ++decisions;
                vector<int> label(nI, -1);
                for (int I = 0; I < nI; ++I) {
                    if (mass[I] <= 0) continue;
                    const double* r = &Rw[(size_t)I * W]; int na = counts[I], best = 0; double n = mass[I];
                    for (int a = 1; a < na; ++a) if (r[a] > r[best]) best = a;
                    double thr;
                    if (test == "normal") thr = z * sqrt(Sq[(size_t)I * W + best]);
                    else if (test == "hoeffding") thr = delta_range[I] * sqrt(2 * n * std::log(1 / dk));
                    else {
                        if (n < 2) continue;
                        double mean = r[best] / n, var = max(0.0, (Sq[(size_t)I * W + best] - n * mean * mean) / (n - 1));
                        thr = sqrt(2 * n * var * std::log(2 / dk)) + 7 * (2 * delta_range[I]) * n * std::log(2 / dk) / (3 * (n - 1));
                    }
                    if (r[best] > thr) label[I] = best;
                }
                vector<map<int, int>> seen(B);
                for (int I = 0; I < nI; ++I) seen[bucket[I]][label[I]] = 0;
                int splits = 0;
                for (int b = 0; b < B; ++b) splits += seen[b].size() > 1;
                auto lc = exact_LC(ExG);
                int Bbefore = B;
                if (splits > 0) {
                    map<pair<int, int>, int> child; vector<int> next(nI), keep, parent;
                    int B2 = 0;
                    for (int I = 0; I < nI; ++I) {
                        int b = bucket[I], lab = seen[b].size() > 1 ? label[I] : -2;
                        auto key = make_pair(b, lab);
                        auto it = child.find(key);
                        if (it == child.end()) { it = child.emplace(key, B2++).first; keep.push_back(lab == -2 ? b : -1); parent.push_back(b); }
                        next[I] = it->second;
                    }
                    vector<double> Q2((size_t)B2 * W, 0), S2((size_t)B2 * W, 0);
                    for (int nb = 0; nb < B2; ++nb)
                        if (keep[nb] >= 0 || inherit)
                            for (int a = 0; a < W; ++a) { Q2[nb * W + a] = Q[parent[nb] * W + a]; S2[nb * W + a] = S[parent[nb] * W + a]; }
                    bucket = next; B = B2; Q = Q2; S = S2; Rtmp.assign((size_t)B * W, 0); Bw.assign((size_t)B * W, 0);
                    fill(Rw.begin(), Rw.end(), 0); fill(Sq.begin(), Sq.end(), 0); fill(mass.begin(), mass.end(), 0);
                    fill(ExG.begin(), ExG.end(), 0);
                    window_start = t; splits_total += splits;
                }
                fprintf(decf, "%lld,%lld,%.9g,%.9g,%.9g,%d,%d,%d\n", t, len, Cs / len, lc.second / len, (lc.first - lc.second) / len,
                        Bbefore, splits, B);
                fflush(decf);
            }
        }
        // ---- fixed-budget checkpoints for evaluation
        if (t == next_ckpt || t == T) {
            auto lc = exact_LC(ExK);
            double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count(), clen = (double)(t - last_ckpt);
            double eCum = 0;
            for (int I = 0; I < nI; ++I) eCum += psi(&ExC[(size_t)I * W], counts[I]);
            save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
            save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
            save(out + "/policy_true_" + to_string(k) + ".bin", policy(Strue, false));
            fprintf(logf, "%d,%lld,%d,%d,%d,%.3f,%.9g,%.9g,%.9g,%.9g\n", k, t, B, splits_total, decisions, secs, lc.first / clen,
                    lc.second / clen, (lc.first - lc.second) / clen, ex_weight > 0 ? eCum / (2 * ex_weight) : -1.0);
            fflush(logf);
            fprintf(stderr, "ckpt %d iters=%lld buckets=%d splits=%d decisions=%d exactC/len=%.4f exactH/len=%.4f (%.1fs)\n",
                    k, t, B, splits_total, decisions, lc.second / clen, (lc.first - lc.second) / clen, secs);
            fill(ExK.begin(), ExK.end(), 0);
            last_ckpt = t; ++k; ckpt_len *= 2; next_ckpt = min(T, t + ckpt_len);
        }
    }
    save(out + "/final_bucket_map.bin", bucket);
    fclose(logf); fclose(decf);
}
