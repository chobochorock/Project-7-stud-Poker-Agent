// Pure ASWISS on a small exported game tree (Kuhn, Leduc, mini-stud; export.py): the original rule with no
// per-information-set state.
//
// Build:  g++ -O3 -std=c++17 pure_aswiss.cpp -o pure_aswiss
// Run:    ./pure_aswiss --tree DIR --buckets DIR/bucket_timestep.bin --out RUN [--seconds 86400] [--iters N]
//                       [--route-impl lazy|table] [--memo none|iter|C] [--k0 8] [--kmax 64] [--members 16]
//                       [--epoch0 1000] [--tol 1e-3] [--seed 1] [--checkpoint-minutes 20]
//         (rerun the same command to resume from RUN/state.bin)
//
// The algorithm (thesis draft, chapter 4):
// - Start from P0 (--buckets: the time-step partition, game structure only). Train every bucket with pure external
//   sampling MCCFR (regret matching on cumulative regrets, uniform averaging, alternating traversers).
// - The abstraction is the starting partition plus the list of split events. An information set's bucket is found
//   every time it is needed by walking the events from P0 (route): at each event its bucket split, it moves to the
//   child whose label is its update direction at that event. Nothing about information sets is stored.
// - Update direction at an event = the label (action, or "none" when no action's regret per unit of counterfactual
//   reach exceeds --tol) maximizing the summed counterfactual regret under the epoch's instantaneous strategies,
//   sum_j r^{sigma_j}(I). The epoch's iterates cannot all be kept, so K of them are drawn uniformly at random among
//   the epoch's iterations and stored (bucket strategy tables): an unbiased estimate of the sum (Theorem A of
//   ASWISS_CONVERGENCE_KO.md). K_k = min(--kmax, --k0 x (k + 1)) for epoch k (it grows, as Theorem A needs).
//   Computing a direction exactly needs the stored strategies at the opponent's earlier decisions (counterfactual
//   reach) and at every later decision (values), i.e. those decisions' buckets at that event, found the same way.
// - Split decision at an epoch end (doubling epochs from --epoch0): each bucket keeps a reservoir of --members
//   information sets drawn from the traverser's visits in the epoch (memory per bucket, not per information set).
//   The members' directions under the epoch's stored iterates are computed; a bucket whose members show two or more
//   labels splits into one child per label; children restart their tables. No merges.
// Implementations of route (same partition, same arithmetic, so the same run; only the time differs):
//   --route-impl lazy --memo none   every route is recomputed from P0, nothing kept (strict)
//   --route-impl lazy --memo iter   routes computed within one iteration are reused in it, then forgotten
//   --route-impl lazy --memo C      at most C (information set, level) results kept, emptied when full
//   --route-impl table              after each split every information set's bucket is computed once and stored
//                                   (the per-information-set table: a reference for the cost of routing)
// Outputs (RUN): log.csv (one row per epoch end and at the budget end), route.csv (routing work), policy_bucket_K.bin
// and policy_current_K.bin (every information set's average / current policy at that row, for eval.py; computed
// outside the training clock), state.bin (checkpoint). train_seconds excludes checkpoints and policy files.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
template <class T> void save_vec(const string& path, const vector<T>& data) {
    ofstream out(path, ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

constexpr int W = 4, TERMINAL = -2, CHANCE = -1, KMAX = 64;
vector<int32_t> players, info, degree, edges, counts, p0_bucket;
vector<int64_t> start;
vector<double> prob, util;
int nI = 0, B = 0;
vector<int32_t> parent_node, parent_slot;
vector<vector<int32_t>> nodes_of;  // information set -> its histories (game structure, ascending = preorder)

// ---- training tables (per bucket) ----
vector<double> Q, S, Rtmp;
vector<int32_t> bucket_n;  // number of actions of each bucket
vector<int> touched; vector<char> dirty;
inline void strategy(int b, int n, double* s) {
    double total = 0;
    for (int a = 0; a < n; ++a) total += max(Q[(size_t)b * W + a], 0.0);
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? max(Q[(size_t)b * W + a], 0.0) / total : 1.0 / n;
}
vector<double> strategy_table() {
    vector<double> t((size_t)B * W, 0.0);
    for (int b = 0; b < B; ++b) strategy(b, bucket_n[b], &t[(size_t)b * W]);
    return t;
}

// ---- split events and routing ----
struct Event {
    vector<int32_t> unsplit;           // bucket before -> its id after, -1 when it split
    vector<map<int, int>> kids;        // split bucket -> (label: -1 none, or action) -> child id
    int K = 0;                         // stored iterates
    vector<double> tables;             // K x B_before x W strategy tables
};
vector<Event> events;
int level_now = 0;
struct Memo { unordered_map<uint64_t, int32_t> m; long long cap = 0; };  // cap: -1 unlimited, 0 none, > 0 bounded
Memo memo_train;
bool memo_iter = false, table_mode = false;
vector<int32_t> bucket_now;  // table mode: every information set's bucket at level_now
struct Count { long long routes = 0, hits = 0, assigns = 0, nodes = 0; } cnt;
bool counting = true;
double tol = 1e-3;

int route(int I, int level, Memo& memo);
// player-0 values of the subtree of h under each stored strategy table (level: the partition the tables are over)
void sub_values(int h, const double* tables, int K, int Bl, int level, Memo& memo, double* out) {
    if (counting) ++cnt.nodes;
    const int pl = players[h];
    if (pl == TERMINAL) { for (int j = 0; j < K; ++j) out[j] = util[h]; return; }
    const int64_t s0 = start[h]; const int d = degree[h];
    double child[KMAX];
    for (int j = 0; j < K; ++j) out[j] = 0;
    if (pl == CHANCE) {
        for (int k = 0; k < d; ++k) {
            sub_values(edges[s0 + k], tables, K, Bl, level, memo, child);
            for (int j = 0; j < K; ++j) out[j] += prob[s0 + k] * child[j];
        }
        return;
    }
    const size_t bx = (size_t)route(info[h], level, memo);
    for (int a = 0; a < d; ++a) {
        sub_values(edges[s0 + a], tables, K, Bl, level, memo, child);
        for (int j = 0; j < K; ++j) out[j] += tables[((size_t)j * Bl + bx) * W + a] * child[j];
    }
}
// summed counterfactual regret R(a) = sum_j r^{sigma_j}(I, a) and summed counterfactual reach M of information set I
// (in bucket b of the partition at `level`) under the K strategy tables over that partition
void direction(int I, int b, const double* tables, int K, int Bl, int level, Memo& memo, double* R, double& M) {
    if (counting) ++cnt.assigns;
    const int n = counts[I], pl = players[nodes_of[I][0]];
    for (int a = 0; a < W; ++a) R[a] = 0;
    M = 0;
    double rj[KMAX][W] = {}, mj[KMAX] = {};
    vector<int> path;
    for (int h : nodes_of[I]) {
        path.clear();
        for (int x = h; parent_node[x] >= 0; x = parent_node[x]) path.push_back(x);
        double popp[KMAX], pc = 1;
        for (int j = 0; j < K; ++j) popp[j] = 1;
        for (size_t i = path.size(); i-- > 0;) {
            const int x = path[i], p = parent_node[x], slot = parent_slot[x];
            if (counting) ++cnt.nodes;
            if (players[p] == CHANCE) pc *= prob[start[p] + slot];
            else if (players[p] != pl) {
                const size_t bx = (size_t)route(info[p], level, memo);
                for (int j = 0; j < K; ++j) popp[j] *= tables[((size_t)j * Bl + bx) * W + slot];
            }
        }
        double v[W][KMAX];
        for (int a = 0; a < n; ++a) sub_values(edges[start[h] + a], tables, K, Bl, level, memo, v[a]);
        const double sign = pl == 0 ? 1.0 : -1.0;
        for (int j = 0; j < K; ++j) {
            const double* s = &tables[((size_t)j * Bl + b) * W];
            double node = 0;
            for (int a = 0; a < n; ++a) node += s[a] * v[a][j];
            const double cf = popp[j] * pc;
            for (int a = 0; a < n; ++a) rj[j][a] += sign * (v[a][j] - node) * cf;
            mj[j] += cf;
        }
    }
    for (int j = 0; j < K; ++j) { for (int a = 0; a < n; ++a) R[a] += rj[j][a]; M += mj[j]; }
}
// the label among the children's labels with the largest regret per unit of reach ("none" scores --tol)
int choose_kid(const double* R, double M, const map<int, int>& kids) {
    int best = 0; double bv = 0; bool first = true;
    for (const auto& [lab, id] : kids) {
        const double v = lab < 0 ? tol : (M > 0 ? R[lab] / M : 0.0);
        if (first || v > bv) { bv = v; best = lab; first = false; }
    }
    return best;
}
// the update direction among all labels (split decision)
int full_label(const double* R, double M, int n) {
    int best = -1; double bv = tol;
    if (M <= 0) return -1;
    for (int a = 0; a < n; ++a) if (R[a] / M > bv) { bv = R[a] / M; best = a; }
    return best;
}
int route(int I, int level, Memo& memo) {
    if (counting) ++cnt.routes;
    if (level == 0) return p0_bucket[I];
    if (table_mode && level == level_now && !bucket_now.empty()) return bucket_now[I];
    const uint64_t key = (uint64_t)I << 8 | (uint64_t)level;
    if (memo.cap != 0) {
        const auto it = memo.m.find(key);
        if (it != memo.m.end()) { if (counting) ++cnt.hits; return it->second; }
    }
    const int b = route(I, level - 1, memo);
    const Event& ev = events[level - 1];
    int nb;
    if (ev.unsplit[b] >= 0) nb = ev.unsplit[b];
    else {
        double R[W], M;
        direction(I, b, ev.tables.data(), ev.K, (int)ev.unsplit.size(), level - 1, memo, R, M);
        nb = ev.kids[b].at(choose_kid(R, M, ev.kids[b]));
    }
    if (memo.cap != 0) {
        if (memo.cap > 0 && (long long)memo.m.size() >= memo.cap) memo.m.clear();
        memo.m.emplace(key, nb);
    }
    return nb;
}
inline int bucket_of(int I) { return table_mode ? bucket_now[I] : route(I, level_now, memo_train); }  // table: set at start

// ---- external sampling (pure MCCFR) with per-bucket member reservoirs ----
mt19937_64 rng, res_rng, iter_rng;
int members_cap = 16;
vector<vector<int32_t>> members; vector<long long> seen;
inline void touch(int b) { if (!dirty[b]) { dirty[b] = 1; touched.push_back(b); } }
void apply_regrets() {
    for (int b : touched) { for (int a = 0; a < W; ++a) { Q[(size_t)b * W + a] += Rtmp[(size_t)b * W + a]; Rtmp[(size_t)b * W + a] = 0; } dirty[b] = 0; }
    touched.clear();
}
double es(int h, int p) {
    const int pl = players[h];
    if (pl == TERMINAL) return p == 0 ? util[h] : -util[h];
    const int64_t e = start[h]; const int d = degree[h];
    if (pl == CHANCE) {
        double x = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
        int k = 0;
        for (; k < d - 1; ++k) { acc += prob[e + k]; if (x < acc) break; }
        return es(edges[e + k], p);
    }
    const int I = info[h], b = bucket_of(I);
    double s[W];
    strategy(b, d, s);
    if (pl == p) {
        double v[W], node = 0;
        for (int a = 0; a < d; ++a) { v[a] = es(edges[e + a], p); node += s[a] * v[a]; }
        touch(b);
        for (int a = 0; a < d; ++a) Rtmp[(size_t)b * W + a] += v[a] - node;
        const long long c = ++seen[b];
        auto& mb = members[b];
        if ((int)mb.size() < members_cap) mb.push_back(I);
        else { const long long j = uniform_int_distribution<long long>(0, c - 1)(res_rng); if (j < members_cap) mb[j] = I; }
        return node;
    }
    for (int a = 0; a < d; ++a) S[(size_t)b * W + a] += s[a];
    double x = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
    int k = 0;
    for (; k < d - 1; ++k) { acc += s[k]; if (x < acc) break; }
    return es(edges[e + k], p);
}

// ---- checkpoint ----
template <class T> void put(ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void get(istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof(T)); }
template <class T> void putv(ostream& o, const vector<T>& v) { uint64_t n = v.size(); put(o, n); o.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T)); }
template <class T> void getv(istream& i, vector<T>& v) { uint64_t n; get(i, n); v.resize(n); i.read(reinterpret_cast<char*>(v.data()), n * sizeof(T)); }
void puts_(ostream& o, const string& s) { uint64_t n = s.size(); put(o, n); o.write(s.data(), n); }
void gets_(istream& i, string& s) { uint64_t n; get(i, n); s.assign(n, '\0'); i.read(s.data(), n); }
string rng_text(const mt19937_64& r) { ostringstream o; o << r; return o.str(); }
void rng_load(mt19937_64& r, const string& s) { istringstream i(s); i >> r; }

struct Run {
    long long t = 0, epoch_start = 0, epoch_end = 0, L = 0; int k = 0, rows = 0;
    vector<long long> samples; size_t next_sample = 0; vector<double> cur_tables; int curK = 0;
    double train_before = 0;  // training seconds before this process started
};

int main(int argc, char** argv) try {
    map<string, string> opt{{"--iters", "1000000000000"}, {"--seconds", "0"}, {"--epoch0", "1000"}, {"--seed", "1"},
                            {"--route-impl", "lazy"}, {"--memo", "iter"}, {"--k0", "8"}, {"--kmax", "64"},
                            {"--members", "16"}, {"--tol", "1e-3"}, {"--checkpoint-minutes", "20"}};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!opt.count(argv[i]) && string(argv[i]) != "--tree" && string(argv[i]) != "--buckets" && string(argv[i]) != "--out")
            throw runtime_error(string("unknown option ") + argv[i]);
        opt[argv[i]] = argv[i + 1];
    }
    const string dir = opt.at("--tree"), out = opt.at("--out");
    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    p0_bucket = load<int32_t>(opt.at("--buckets"));
    nI = (int)counts.size();
    if ((int)p0_bucket.size() != nI) throw runtime_error("--buckets size differs from the information sets");
    tol = stod(opt["--tol"]);
    members_cap = stoi(opt["--members"]);
    const int k0 = stoi(opt["--k0"]), kmax = min(KMAX, stoi(opt["--kmax"]));
    table_mode = opt["--route-impl"] == "table";
    if (!table_mode && opt["--route-impl"] != "lazy") throw runtime_error("--route-impl must be lazy or table");
    if (opt["--memo"] == "none") memo_train.cap = 0;
    else if (opt["--memo"] == "iter") { memo_train.cap = -1; memo_iter = true; }
    else memo_train.cap = stoll(opt["--memo"]);
    const double budget = stod(opt["--seconds"]), ckpt_every = 60.0 * stod(opt["--checkpoint-minutes"]);
    long long T = stoll(opt["--iters"]);
    const long long epoch0 = stoll(opt["--epoch0"]);
    const int N = (int)players.size();
    parent_node.assign(N, -1); parent_slot.assign(N, -1); nodes_of.assign(nI, {});
    for (int h = 0; h < N; ++h) {
        if (players[h] == TERMINAL) continue;
        for (int k = 0; k < degree[h]; ++k) { parent_node[edges[start[h] + k]] = h; parent_slot[edges[start[h] + k]] = k; }
        if (players[h] != CHANCE) nodes_of[info[h]].push_back(h);
    }
    for (int b : p0_bucket) B = max(B, b + 1);
    bucket_n.assign(B, 0);
    for (int I = 0; I < nI; ++I) bucket_n[p0_bucket[I]] = counts[I];
    // the run's identity (a resume must match it; the routing implementation and memo may differ: same run)
    ostringstream cfg;
    cfg << "pure_aswiss v1;tree=" << dir << ";buckets=" << opt["--buckets"] << ";epoch0=" << epoch0 << ";seed=" << opt["--seed"]
        << ";k0=" << k0 << ";kmax=" << kmax << ";members=" << members_cap << ";tol=" << opt["--tol"];
    filesystem::create_directories(out);
    const string state_path = out + "/state.bin", log_path = out + "/log.csv", route_path = out + "/route.csv";
    Run run;
    double excluded = 0;
    const auto t0 = chrono::steady_clock::now();
    auto train_secs = [&] { return run.train_before + chrono::duration<double>(chrono::steady_clock::now() - t0).count() - excluded; };
    auto excl = [&](auto&& f) { const auto x0 = chrono::steady_clock::now(); f(); excluded += chrono::duration<double>(chrono::steady_clock::now() - x0).count(); };
    auto draw_samples = [&] {  // K iterations uniformly among the epoch's (distinct, sorted), from iter_rng
        const long long Lk = run.epoch_end - run.epoch_start;
        const int K = (int)min<long long>(Lk, min(kmax, k0 * (run.k + 1)));
        set<long long> pick;
        while ((int)pick.size() < K) pick.insert(run.epoch_start + 1 + uniform_int_distribution<long long>(0, Lk - 1)(iter_rng));
        run.samples.assign(pick.begin(), pick.end()); run.next_sample = 0; run.curK = 0; run.cur_tables.clear();
    };
    uint64_t log_bytes = 0, route_bytes = 0;
    auto save_state = [&] {
        const string tmp = state_path + ".tmp";
        ofstream o(tmp, ios::binary);
        puts_(o, cfg.str());
        put(o, run.t); put(o, run.epoch_start); put(o, run.epoch_end); put(o, run.L); put(o, run.k); put(o, run.rows);
        putv(o, run.samples); put(o, run.next_sample); putv(o, run.cur_tables); put(o, run.curK);
        const double ts = train_secs(); put(o, ts);
        put(o, B); put(o, level_now); putv(o, Q); putv(o, S); putv(o, bucket_n);
        uint64_t ne = events.size(); put(o, ne);
        for (const Event& ev : events) {
            putv(o, ev.unsplit); put(o, ev.K); putv(o, ev.tables);
            uint64_t nk = ev.kids.size(); put(o, nk);
            for (const auto& m : ev.kids) { uint64_t sz = m.size(); put(o, sz); for (const auto& [l, id] : m) { put(o, l); put(o, id); } }
        }
        uint64_t nm = members.size(); put(o, nm);
        for (const auto& m : members) putv(o, m);
        putv(o, seen);
        puts_(o, rng_text(rng)); puts_(o, rng_text(res_rng)); puts_(o, rng_text(iter_rng));
        put(o, cnt);
        log_bytes = filesystem::exists(log_path) ? filesystem::file_size(log_path) : 0;
        route_bytes = filesystem::exists(route_path) ? filesystem::file_size(route_path) : 0;
        put(o, log_bytes); put(o, route_bytes);
        o.close();
        filesystem::rename(tmp, state_path);
    };
    bool resumed = false;
    if (filesystem::exists(state_path)) {
        ifstream in(state_path, ios::binary);
        string c; gets_(in, c);
        if (c != cfg.str()) throw runtime_error("state.bin is another run's (" + c + ")");
        get(in, run.t); get(in, run.epoch_start); get(in, run.epoch_end); get(in, run.L); get(in, run.k); get(in, run.rows);
        getv(in, run.samples); get(in, run.next_sample); getv(in, run.cur_tables); get(in, run.curK);
        get(in, run.train_before);
        get(in, B); get(in, level_now); getv(in, Q); getv(in, S); getv(in, bucket_n);
        uint64_t ne; get(in, ne); events.resize(ne);
        for (Event& ev : events) {
            getv(in, ev.unsplit); get(in, ev.K); getv(in, ev.tables);
            uint64_t nk; get(in, nk); ev.kids.assign(nk, {});
            for (auto& m : ev.kids) { uint64_t sz; get(in, sz); for (uint64_t i = 0; i < sz; ++i) { int l, id; get(in, l); get(in, id); m[l] = id; } }
        }
        uint64_t nm; get(in, nm); members.resize(nm);
        for (auto& m : members) getv(in, m);
        getv(in, seen);
        string r1, r2, r3; gets_(in, r1); gets_(in, r2); gets_(in, r3);
        rng_load(rng, r1); rng_load(res_rng, r2); rng_load(iter_rng, r3);
        get(in, cnt); get(in, log_bytes); get(in, route_bytes);
        if (!in) throw runtime_error("truncated state.bin");
        filesystem::resize_file(log_path, log_bytes); filesystem::resize_file(route_path, route_bytes);
        resumed = true;
        cerr << "resumed at iteration " << run.t << ", level " << level_now << ", buckets " << B << ", training " << run.train_before << " s\n";
    } else {
        const uint64_t seed = stoull(opt["--seed"]);
        rng.seed(seed); res_rng.seed(seed ^ 0x9e3779b97f4a7c15ull); iter_rng.seed(seed ^ 0xbf58476d1ce4e5b9ull);
        Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0);
        members.assign(B, {}); seen.assign(B, 0);
        run.L = epoch0; run.epoch_start = 0; run.epoch_end = epoch0;
        draw_samples();
        ofstream(log_path) << "epoch,iters,buckets,buckets_after,splits,K,level,train_seconds\n";
        ofstream(route_path) << "epoch,iters,level,buckets,memo_entries,routes,memo_hits,assigns,nodes,train_seconds\n";
    }
    Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
    if (table_mode) bucket_now = p0_bucket;
    if (table_mode && level_now > 0) {  // the stored table at level_now (recomputed after a resume)
        Memo m; m.cap = -1; bucket_now.clear(); vector<int32_t> bn(nI);
        for (int I = 0; I < nI; ++I) bn[I] = route(I, level_now, m);
        bucket_now = move(bn);
    }
    (void)resumed;
    FILE* logf = fopen(log_path.c_str(), "a");
    FILE* routef = fopen(route_path.c_str(), "a");
    auto write_policies = [&](int row) {  // every information set's policies (outside the clock): eval.py input
        excl([&] {
            const bool was = counting; counting = false;
            Memo m; m.cap = -1;
            vector<double> avg((size_t)nI * W, 0.0), cur((size_t)nI * W, 0.0);
            for (int I = 0; I < nI; ++I) {
                const int b = table_mode ? bucket_now[I] : route(I, level_now, m), n = counts[I];
                double tot = 0;
                for (int a = 0; a < n; ++a) tot += S[(size_t)b * W + a];
                for (int a = 0; a < n; ++a) avg[(size_t)I * W + a] = tot > 0 ? S[(size_t)b * W + a] / tot : 1.0 / n;
                strategy(b, n, &cur[(size_t)I * W]);
            }
            save_vec(out + "/policy_bucket_" + to_string(row) + ".bin", avg);
            save_vec(out + "/policy_current_" + to_string(row) + ".bin", cur);
            counting = was;
        });
    };
    auto last_ckpt = chrono::steady_clock::now();
    while (run.t < T) {
        ++run.t;
        if (memo_iter) memo_train.m.clear();
        if (run.next_sample < run.samples.size() && run.samples[run.next_sample] == run.t) {  // store this iterate
            const auto tab = strategy_table();
            run.cur_tables.insert(run.cur_tables.end(), tab.begin(), tab.end());
            ++run.curK; ++run.next_sample;
        }
        for (int p = 0; p < 2; ++p) { es(0, p); apply_regrets(); }
        const bool budget_end = budget > 0 && train_secs() >= budget;
        if (budget_end) T = run.t;
        if (run.t == run.epoch_end || run.t == T) {
            const int before = B;
            write_policies(run.rows);
            int splits = 0;
            const bool judge = run.t == run.epoch_end && run.curK > 0;  // a truncated last epoch is not judged
            if (judge) {
                Event ev;
                ev.K = run.curK; ev.tables = move(run.cur_tables); ev.unsplit.assign(B, -1); ev.kids.assign(B, {});
                Memo dm; dm.cap = memo_train.cap; if (memo_iter) dm.cap = -1;  // the decision's own routing memo
                int B2 = 0;
                vector<int> keep, n2;
                for (int b = 0; b < B; ++b) {
                    set<int> labels;
                    vector<int32_t> mem = members[b];
                    sort(mem.begin(), mem.end()); mem.erase(unique(mem.begin(), mem.end()), mem.end());
                    if (mem.size() >= 2)
                        for (int I : mem) {
                            double R[W], M;
                            direction(I, b, ev.tables.data(), ev.K, B, level_now, table_mode ? memo_train : dm, R, M);
                            labels.insert(full_label(R, M, counts[I]));
                        }
                    if (labels.size() >= 2) {
                        ++splits;
                        for (int l : labels) { ev.kids[b][l] = B2++; keep.push_back(-1); n2.push_back(bucket_n[b]); }
                    } else { ev.unsplit[b] = B2++; keep.push_back(b); n2.push_back(bucket_n[b]); }
                }
                events.push_back(move(ev));
                ++level_now;
                vector<double> Q2((size_t)B2 * W, 0.0), S2((size_t)B2 * W, 0.0);
                for (int nb = 0; nb < B2; ++nb)
                    if (keep[nb] >= 0)
                        for (int a = 0; a < W; ++a) { Q2[(size_t)nb * W + a] = Q[(size_t)keep[nb] * W + a]; S2[(size_t)nb * W + a] = S[(size_t)keep[nb] * W + a]; }
                B = B2; Q = move(Q2); S = move(S2); bucket_n = n2;
                Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
                members.assign(B, {}); seen.assign(B, 0);
                memo_train.m.clear();
                if (table_mode) {  // the per-information-set table of the new partition (training time)
                    Memo m; m.cap = -1; bucket_now.clear(); vector<int32_t> bn(nI);
                    for (int I = 0; I < nI; ++I) bn[I] = route(I, level_now, m);
                    bucket_now = move(bn);
                }
                ++run.k; run.epoch_start = run.t; run.L *= 2; run.epoch_end = run.t + run.L;
                draw_samples();
            }
            fprintf(logf, "%d,%lld,%d,%d,%d,%d,%d,%.3f\n", run.rows, run.t, before, B, splits, judge ? events.back().K : run.curK, level_now, train_secs());
            fflush(logf);
            fprintf(routef, "%d,%lld,%d,%d,%zu,%lld,%lld,%lld,%lld,%.3f\n", run.rows, run.t, level_now, B, memo_train.m.size(), cnt.routes,
                    cnt.hits, cnt.assigns, cnt.nodes, train_secs());
            fflush(routef);
            cerr << "row " << run.rows << " iters " << run.t << " level " << level_now << " buckets " << before << " -> " << B
                 << " splits " << splits << " nodes " << cnt.nodes << " (" << train_secs() << " s)\n";
            ++run.rows;
            excl(save_state);
            last_ckpt = chrono::steady_clock::now();
        } else if (chrono::duration<double>(chrono::steady_clock::now() - last_ckpt).count() >= ckpt_every) {
            excl(save_state);
            last_ckpt = chrono::steady_clock::now();
            cerr << "checkpoint at iteration " << run.t << " level " << level_now << " buckets " << B << " nodes " << cnt.nodes
                 << " (" << train_secs() << " s)\n";
        }
    }
    fclose(logf); fclose(routef);
    ofstream(out + "/DONE") << run.t << "\n";
    return 0;
} catch (const exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
}
