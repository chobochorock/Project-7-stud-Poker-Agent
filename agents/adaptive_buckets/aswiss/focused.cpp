// Focused probing variant of ASWISS: detect suspicious buckets with a few on-policy rollout probes, then
// accumulate per-infoset regret statistics only for those buckets during the next epoch and split them.
//
// Build: g++ -O3 -std=c++17 focused.cpp -o focused
// Run:   ./focused --tree DIR --buckets FILE --iters T --epoch0 L0 --test normal|hoeffding|bernstein
//                  [--z 2] [--delta 0.05] [--probe-prob 0.02] [--probe-m 4] [--reservoir 32] [--min-support 2]
//                  [--audit 0.05] [--bottom-up 1] [--exact-every 100] [--seed 1] --out OUTDIR
// Input/output formats are those of aswiss.cpp (export.py arrays; log.csv and policy_{bucket,true,current}_k.bin).
//
// Training is external-sampling MCCFR+ on bucket tables, exactly as in aswiss.cpp. Per bucket status:
//  normal: when the traverser visits a member infoset I, with probability probe-prob run a probe: for each legal
//          action a, probe-m on-policy rollouts from h.a (chance, both players sampled from the current strategy)
//          give v(a); the probe regret is v(a) - sum_b sigma(b) v(b). (I, probe regret) goes into the bucket's
//          fixed-size reservoir. At the epoch end the bucket is flagged when its reservoir holds two different
//          labels (argmax with positive regret, else "none") each supported by >= min-support probes, or at random
//          with probability audit (keeps every bucket checked infinitely often).
//          bottom-up: reservoir flags are kept only in the deepest street that has any (audits are always kept).
//  decide: every visit of a member infoset runs a probe and adds it to that infoset's statistics (sum, sum of
//          squares, count). At the epoch end each tracked infoset gets a label by the chosen test, the bucket is
//          split by label when two or more labels occur, and the bucket returns to normal.
// Probes only feed these statistics; regret updates use the ordinary external-sampling values.
// Memory for decisions = tracked infosets (members of decide buckets that were visited) + reservoirs.
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
vector<int32_t> players, info, degree, edges, counts, bucket, street_of;
vector<int64_t> start;
vector<double> prob, util, delta_range;
int nI, B;
vector<double> Q, S, Strue, Rtmp, ExW, ExC;
double ex_weight = 0;
mt19937_64 rng;
uniform_real_distribution<double> U(0, 1);

// focused-probing state
vector<int> status;                          // 0 normal, 1 decide (per bucket)
struct Probe { int I; double r[W]; };
vector<vector<Probe>> reservoir;             // per bucket
vector<long long> reservoir_seen;
vector<double> Ps, Pq; vector<double> Pn;    // per-infoset probe statistics (only decide-bucket members)
vector<int> tracked;                         // infosets with Pn > 0 this epoch
double probe_prob = 0.02; int probe_m = 4, reservoir_cap = 32;
long long probe_nodes = 0;

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
// One on-policy rollout from node h; returns the utility for player p.
double rollout(int h, int p) {
    while (players[h] != TERMINAL) {
        ++probe_nodes;
        int64_t e = start[h]; int d = degree[h];
        if (players[h] == CHANCE) { h = edges[e + sample(&prob[e], d)]; continue; }
        double s[W]; strategy(bucket[info[h]], d, s);
        h = edges[e + sample(s, d)];
    }
    return p == 0 ? util[h] : -util[h];
}
void probe(int h, int p, const double* s, double* r) {
    int64_t e = start[h]; int d = degree[h];
    double v[W], node = 0;
    for (int a = 0; a < d; ++a) {
        double sum = 0;
        for (int m = 0; m < probe_m; ++m) sum += rollout(edges[e + a], p);
        v[a] = sum / probe_m; node += s[a] * v[a];
    }
    for (int a = 0; a < d; ++a) r[a] = v[a] - node;
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
        for (int a = 0; a < d; ++a) Rtmp[b * W + a] += v[a] - node;
        if (status[b] == 1) {
            double r[W]; probe(h, p, s, r);
            if (Pn[I] == 0) tracked.push_back(I);
            Pn[I] += 1;
            for (int a = 0; a < d; ++a) { Ps[I * W + a] += r[a]; Pq[I * W + a] += r[a] * r[a]; }
        } else if (U(rng) < probe_prob) {
            Probe pr{I, {0, 0, 0, 0}}; probe(h, p, s, pr.r);
            long long seen = ++reservoir_seen[b];
            if ((int)reservoir[b].size() < reservoir_cap) reservoir[b].push_back(pr);
            else { long long j = (long long)(U(rng) * seen); if (j < reservoir_cap) reservoir[b][j] = pr; }
        }
        return node;
    }
    for (int a = 0; a < d; ++a) { S[b * W + a] += weight * s[a]; Strue[I * W + a] += weight * s[a]; }
    return es(edges[e + sample(s, d)], p, weight);
}

// Diagnostic only: exact instantaneous counterfactual regrets (see aswiss.cpp).
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
    for (int a = 0; a < d; ++a) { double r = sign * (v[a] - node) * cf; ExW[I * W + a] += factor * r; ExC[I * W + a] += wfactor * r; }
    return node;
}

void apply_regrets() { for (int i = 0; i < B * W; ++i) { Q[i] = max(Q[i] + Rtmp[i], 0.0); Rtmp[i] = 0; } }
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

int main(int argc, char** argv) {
    map<string, string> opt{{"--iters", "2560000"}, {"--epoch0", "10000"}, {"--test", "normal"}, {"--z", "2"},
                            {"--delta", "0.05"}, {"--probe-prob", "0.02"}, {"--probe-m", "4"}, {"--reservoir", "32"},
                            {"--min-support", "2"}, {"--audit", "0.05"}, {"--bottom-up", "1"}, {"--exact-every", "100"},
                            {"--seed", "1"}};
    for (int i = 1; i + 1 < argc; i += 2) opt[argv[i]] = argv[i + 1];
    string dir = opt.at("--tree"), out = opt.at("--out"), test = opt["--test"];
    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    delta_range = load<double>(dir + "/delta.bin"); street_of = load<int32_t>(dir + "/street.bin");
    bucket = load<int32_t>(opt.at("--buckets"));
    nI = counts.size(); B = 0;
    for (int b : bucket) B = max(B, b + 1);
    rng.seed(stoull(opt["--seed"]));
    long long T = stoll(opt["--iters"]), L = stoll(opt["--epoch0"]), exact_every = stoll(opt["--exact-every"]);
    double z = stod(opt["--z"]), delta0 = stod(opt["--delta"]), audit = stod(opt["--audit"]);
    int min_support = stoi(opt["--min-support"]); bool bottom_up = opt["--bottom-up"] == "1";
    probe_prob = stod(opt["--probe-prob"]); probe_m = stoi(opt["--probe-m"]); reservoir_cap = stoi(opt["--reservoir"]);
    Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0); Rtmp.assign((size_t)B * W, 0);
    Strue.assign((size_t)nI * W, 0); ExW.assign((size_t)nI * W, 0); ExC.assign((size_t)nI * W, 0);
    Ps.assign((size_t)nI * W, 0); Pq.assign((size_t)nI * W, 0); Pn.assign(nI, 0);
    status.assign(B, 0); reservoir.assign(B, {}); reservoir_seen.assign(B, 0);
    FILE* logf = fopen((out + "/log.csv").c_str(), "w");
    fprintf(logf, "epoch,iters,buckets_after,splits,decide_buckets,flags_next,audits_next,tracked_infosets,reservoir_entries,"
                  "probe_nodes,seconds,exact_L_per_iter,exact_C_per_iter,exact_H_per_iter,exact_bound_expl\n");
    auto t0 = chrono::steady_clock::now();
    long long t = 0, epoch_end = L, epoch_start = 0; int k = 0;
    while (t < T) {
        ++t;
        double weight = (double)t;
        if (exact_every > 0 && t % exact_every == 0) {
            exact(0, 1, 1, 1, (double)exact_every, (double)exact_every * weight);
            ex_weight += (double)exact_every * weight;
        }
        for (int p = 0; p < 2; ++p) { es(0, p, weight); apply_regrets(); }
        if (t != epoch_end && t != T) continue;
        // exact diagnostics on the partition used this epoch
        vector<int> bcount(B, 0);
        for (int I = 0; I < nI; ++I) bcount[bucket[I]] = counts[I];
        double eL = 0, eC = 0, eCum = 0;
        vector<double> esum((size_t)B * W, 0.0);
        for (int I = 0; I < nI; ++I) {
            eL += psi(&ExW[(size_t)I * W], counts[I]); eCum += psi(&ExC[(size_t)I * W], counts[I]);
            for (int a = 0; a < counts[I]; ++a) esum[(size_t)bucket[I] * W + a] += ExW[(size_t)I * W + a];
        }
        for (int b = 0; b < B; ++b) eC += psi(&esum[(size_t)b * W], bcount[b]);
        fill(ExW.begin(), ExW.end(), 0);
        save(out + "/policy_bucket_" + to_string(k) + ".bin", policy(S, true));
        save(out + "/policy_current_" + to_string(k) + ".bin", current_policy());
        save(out + "/policy_true_" + to_string(k) + ".bin", policy(Strue, false));
        // 1) decide buckets: label tracked infosets with the chosen test
        double dk = delta0 / ((k + 1.0) * (k + 1.0));
        vector<int> label(nI, -1);
        int decide_buckets = 0;
        for (int b = 0; b < B; ++b) decide_buckets += status[b] == 1;
        long long tracked_count = tracked.size();
        for (int I : tracked) {
            const double* r = &Ps[(size_t)I * W]; int na = counts[I], best = 0; double n = Pn[I];
            for (int a = 1; a < na; ++a) if (r[a] > r[best]) best = a;
            double thr;
            if (test == "normal") thr = z * sqrt(Pq[(size_t)I * W + best]);
            else if (test == "hoeffding") thr = delta_range[I] * sqrt(2 * n * std::log(1 / dk));
            else {
                if (n < 2) continue;
                double mean = r[best] / n, var = max(0.0, (Pq[(size_t)I * W + best] - n * mean * mean) / (n - 1));
                thr = sqrt(2 * n * var * std::log(2 / dk)) + 7 * (2 * delta_range[I]) * n * std::log(2 / dk) / (3 * (n - 1));
            }
            if (r[best] > thr) label[I] = best;
        }
        vector<map<int, int>> seen(B);
        for (int I = 0; I < nI; ++I) if (status[bucket[I]] == 1) seen[bucket[I]][label[I]] = 0;
        // 2) normal buckets: reservoir flags and random audits (decided on the pre-split ids)
        vector<int> flag(B, 0), aud(B, 0);
        int deepest = -1;
        for (int b = 0; b < B; ++b) {
            if (status[b] == 1) continue;
            map<int, int> support;
            for (const Probe& pr : reservoir[b]) {
                int na = counts[pr.I], best = 0;
                for (int a = 1; a < na; ++a) if (pr.r[a] > pr.r[best]) best = a;
                ++support[pr.r[best] > 0 ? best : -1];
            }
            int strong = 0;
            for (auto& kv : support) strong += kv.second >= min_support;
            if (strong >= 2) flag[b] = 1;
            if (U(rng) < audit) aud[b] = 1;
        }
        vector<int> bstreet(B, -1);
        for (int I = 0; I < nI; ++I) bstreet[bucket[I]] = street_of[I];
        if (bottom_up) {
            for (int b = 0; b < B; ++b) if (flag[b]) deepest = max(deepest, bstreet[b]);
            for (int b = 0; b < B; ++b) if (flag[b] && bstreet[b] != deepest) flag[b] = 0;
        }
        // 3) split decide buckets by label; build new ids and statuses
        map<pair<int, int>, int> child; vector<int> next(nI), keep, nstatus;
        int B2 = 0, splits = 0;
        for (int I = 0; I < nI; ++I) {
            int b = bucket[I], lab = (status[b] == 1 && seen[b].size() > 1) ? label[I] : -2;
            auto key = make_pair(b, lab);
            auto it = child.find(key);
            if (it == child.end()) {
                it = child.emplace(key, B2++).first;
                keep.push_back(lab == -2 ? b : -1);
                nstatus.push_back(lab == -2 && status[b] == 0 && (flag[b] || aud[b]) ? 1 : 0);
            }
            next[I] = it->second;
        }
        for (int b = 0; b < B; ++b) splits += status[b] == 1 && seen[b].size() > 1;
        int flags_next = 0, audits_next = 0;
        for (int b = 0; b < B; ++b) { flags_next += status[b] == 0 && flag[b]; audits_next += status[b] == 0 && aud[b] && !flag[b]; }
        long long reservoir_entries = 0;
        for (auto& r : reservoir) reservoir_entries += r.size();
        vector<double> Q2((size_t)B2 * W, 0), S2((size_t)B2 * W, 0);
        for (int nb = 0; nb < B2; ++nb)
            if (keep[nb] >= 0)
                for (int a = 0; a < W; ++a) { Q2[nb * W + a] = Q[keep[nb] * W + a]; S2[nb * W + a] = S[keep[nb] * W + a]; }
        bucket = next; B = B2; Q = Q2; S = S2; Rtmp.assign((size_t)B * W, 0);
        status = nstatus; reservoir.assign(B, {}); reservoir_seen.assign(B, 0);
        for (int I : tracked) { Pn[I] = 0; for (int a = 0; a < W; ++a) { Ps[(size_t)I * W + a] = 0; Pq[(size_t)I * W + a] = 0; } }
        tracked.clear();
        double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count(), len = (double)(t - epoch_start);
        fprintf(logf, "%d,%lld,%d,%d,%d,%d,%d,%lld,%lld,%lld,%.3f,%.9g,%.9g,%.9g,%.9g\n", k, t, B, splits, decide_buckets,
                flags_next, audits_next, tracked_count, reservoir_entries, probe_nodes, secs, eL / len, eC / len, (eL - eC) / len,
                ex_weight > 0 ? eCum / (2 * ex_weight) : -1.0);
        fflush(logf);
        fprintf(stderr, "epoch %d iters=%lld buckets=%d splits=%d decide=%d flags=%d audits=%d tracked=%lld H/len=%.4f (%.1fs)\n",
                k, t, B, splits, decide_buckets, flags_next, audits_next, tracked_count, (eL - eC) / len, secs);
        epoch_start = t; ++k; L *= 2; epoch_end = min(T, t + L);
    }
    save(out + "/final_bucket_map.bin", bucket);
    fclose(logf);
}
