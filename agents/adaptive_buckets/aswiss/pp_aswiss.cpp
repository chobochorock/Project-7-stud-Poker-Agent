// Prior-posterior ASWISS and pure ASWISS on a small exported game tree (mini-stud, Kuhn, Leduc).
// Spec: agents/adaptive_buckets/ASWISS_SPEC_V2_KO.md; design: PRIOR_POSTERIOR_DESIGN_KO.md; features: PRIOR_FEATURES_KO.md.
//
// Build:  g++ -O3 -std=c++17 pp_aswiss.cpp -o pp_aswiss
// Inputs: TREE (export.py or small/export_small.py, then export_obs.py), the ESD table (gen_esd_table.py).
// Run:    ./pp_aswiss --tree DIR --out RUN [--mode pp|pure] [--seconds S] [--iters T] [--seed 1] [options below]
//         (rerun the same command to resume from RUN/state.bin)
//
// Both modes: pure external-sampling MCCFR (regret matching on cumulative regrets, uniform averaging at the opponent's
// nodes, alternating traversers; one iteration = one traversal per seat) on posterior buckets. The posterior starts
// from one bucket per (player, legal action set) (constraint.bin). At an epoch end every bucket is judged:
//   - its members (at most --members distinct information sets visited by the traverser in the epoch, a uniform
//     bottom-k sample by hash) get their direction r(I,a) = sum over the epoch's stored iterates sigma of the exact
//     counterfactual regret r^sigma(I,a) (K_k = 8(k+1) iterates drawn uniformly in epoch k; --kmax caps K);
//   - per action, a one-sided outlier test on the members' r(I,a): generalized ESD (--outlier esd, critical values
//     from --esd-table, alpha = 0.05/|A|, r = n/2) or z-sigma (--outlier z --z 2);
//   - the test becomes a threshold rule: mu*_a, sd*_a of the non-outliers, c_a = the smallest z of the outliers;
//     label(I) = argmax_a z_a(I) among actions with a child and z_a >= c_a, else none (-1). The bucket splits when its
//     members carry two or more labels: one child per label plus the none child (the parent's continuation).
//   - children restart their tables (--inherit-regret L --inherit-avg M copy L x and M x the parent's).
// Epoch ends: --schedule static (iterations 1000 * 2^k) or eps (sum over buckets of max(0, max_a Q_b(a)) / visits of b
// since its birth < eps, eps = that sum at iteration 1000; buckets with < 25 visits use their parent's value; checked
// every --eps-check iterations, at least 1000 iterations after a split).
// --mode pure: an information set's bucket is found by walking the split events from the start (route); the routing
//   memo (--memo 1) and the per-information-set table of the current level (--table 1) are caches of the same exact
//   function (test T2).
// --mode pp: information set -> prior tree leaf (tests on game-general features of its observation stream) -> pointer
//   to a posterior bucket. A leaf whose pointer is unknown is pending: its visited information sets are routed one by
//   one through the posterior genealogy from the leaf's start node, and the first --m traverser visits are kept; then
//   the leaf's pointer is the route of their mean direction. If the individual labels of those samples disagree at
//   some event, the prior is short there (signal): --prior-split immediate splits that leaf right away (one binary
//   test, total leaves <= 2 x the leaves at the last posterior split, otherwise held to the next one); --prior-split
//   epoch splits the signalled leaves at the next epoch end, best first, up to 2 x the leaves. The split data are all
//   the leaf's information sets with their per-visit direction vectors at the signal's event (weights: counterfactual
//   reach), the criterion the weighted squared error reduction, chosen on one random half and accepted only when it
//   also reduces the other half's (--min-leaf rows per side in the first half). New leaves are pending from the
//   start of the genealogy, with the parent's pointer as provisional pointer.
//   Inside a direction computation, another information set's bucket at an old level j is the level-j ancestor of its
//   current bucket (its leaf's pointer; a pending leaf's provisional pointer).
//   --prior-start equal | length | identity (test only: one leaf per information set; with --m 1 --eager 1 the run
//   is identical to --mode pure, test T4). --features general | poker. --leaf-cache / --pend-cache: caches (T2).
// --prior-ratio R (pp): after every posterior judgment the prior grows (best first) to at least R x the posterior
// buckets, and immediate splits may go up to that size; --crossfit 0: prior tests chosen and accepted on all rows.
// --resolve-now 1 (pp): new leaves get m samples drawn from their information sets (by counterfactual reach) at the
// split and are finalized at once. --prior-criterion label: prior tests reduce the Gini impurity of the posterior
// child each information set picks at the event (instead of the regret vectors' squared error).
// --epoch0 E (static schedule: judgments at E * 2^k, default 1000) and --eps-at A (eps schedule: the first judgment and
// eps at iteration A, default 1000).
// --direction avg: one stored strategy per split event, the split-time average strategy (rule (a) then gates labels by
// per-visit regret > --tol); iterates (default) is the K-iterate sum. --leaf-start parent (pp): new prior leaves keep
// their parent's pointer and no route starts before the newest event, so older events' tables are freed; root
// (default) routes new leaves from the start of the genealogy (rule (alpha)).
// --mode tree (single tree, PP_STUD7_DESIGN_KO.md sections 16 and 18): every prior leaf is its own bucket (no posterior
//   genealogy, no pointers). Each leaf keeps --acc-candidates K candidate tests (a feature and a threshold half-way between
//   the values of two different information sets of a reservoir of --acc-pool of its traverser visits); every traverser
//   visit adds its sampled regrets v(a) - sum sigma v to the side of each candidate it falls on (sums, sums of squares,
//   visits). At an epoch end a candidate passes when both sides have --acc-min-visits visits and the squared distance of
//   the sides' mean regret vectors, standardised under equal means, is >= --acc-z; with --acc-metric dir the sides'
//   unit mean regret vectors (0 for a side whose mean is not significantly non-zero) must also be >= --acc-dist apart
//   (|u_L - u_R|^2 in 0..4). The leaf splits by its passing candidate of largest squared-error drop (dir: largest
//   distance); the two new buckets start from --inherit-regret / --inherit-avg x the leaf's tables; other leaves replace
//   their failed and one-sided candidates. --prior-start equal, --schedule static.
// --mode ppacc (prior-posterior with accumulated regrets, PP_STUD7_DESIGN_KO.md section 19): prior leaves point to
//   posterior buckets (the strategy tables). Every prior leaf keeps its own regret sums since its pointer was set. Epoch
//   ends alternate between a prior phase (even epochs) and a posterior phase (odd epochs).
//   Prior phase: a leaf that uses its bucket alone, or has >= 50% of the bucket's visits since the pointers were set,
//   carries the --acc-candidates tests of --mode tree; it splits by its best passing candidate (the same test and
//   --acc-metric); both new leaves keep the pointer and start from their side's sums. Other leaves drop their candidates.
//   Posterior phase: in a bucket used by two or more leaves, a leaf diverges when the candidate test of "this leaf"
//   against "the bucket's other leaves" passes. A bucket with a diverging leaf is split: each leaf whose mean regret is
//   significantly non-zero (the one-side test of --acc-metric dir) is labelled with its largest-regret action, the others
//   stay; every label group gets a new bucket starting from --inherit-regret x / --inherit-avg x the bucket's tables, and
//   the bucket's leaves restart their sums and candidates. Unused buckets are kept (counted as not used).
// --analyze N (pp, after a run): N information sets of resolved leaves, own route vs pointer -> RUN/analysis.json.
// Outputs (RUN): log.csv (rows on the grid 1000 * 2^(j/--rows-per-octave) and at epoch ends), policy_bucket_K.bin and
// policy_current_K.bin per row (for eval.py), events.csv, final_bucket_map.bin, prior_splits.csv, prior_tree.json (pp), and with
// --measure 1 the regret sums of every row and depth_spread.csv (pp). train_seconds is process CPU time without
// rows, measurement and checkpoints; assign_seconds (included in it) is the routing of pending leaves, their
// finalization, prior splits and the pure mode's table rebuilds.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
double cpu_now() { timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// ======================= game =======================
constexpr int W = 4, TERMINAL = -2, CHANCE = -1, KMAX = 256;
vector<int32_t> players, info, degree, edges, counts, constraint_of, seat_of, obs;
vector<int64_t> start, obs_start;
vector<double> prob, util;
int nI = 0, N = 0, n_constraints = 0;
vector<int32_t> parent_node, parent_slot;
vector<vector<int32_t>> nodes_of;
vector<int> card_syms;
vector<string> sym_names;
inline int o_ch(int32_t c) { return c >> 24; }
inline int o_rnd(int32_t c) { return (c >> 16) & 255; }
inline int o_ord(int32_t c) { return (c >> 8) & 255; }
inline int o_sym(int32_t c) { return c & 255; }
inline bool chance_ch(int ch) { return ch <= 4; }
enum { PC = 0, UM, UO, UC, HO, AM, PM, PO };

// ======================= options =======================
bool pp_mode = true, split_on = true, measure = false, eager = false, leaf_cache_on = true, pend_cache_on = true, memo_on = true, table_on = true;
int members_cap = 64, m_samples = 16, min_leaf = 1, kmax = KMAX, rows_per_octave = 2, esd_min_n = 5;
double zthr = 2.0, inherit_r = 0, inherit_s = 0, prior_ratio = 0;
bool crossfit = true, resolve_now = false, direction_avg = false, leaf_start_parent = false;
// --mode tree, --mode ppacc
bool tree_mode = false, ppacc_mode = false, acc_dir = false;
int acc_k = 8, acc_min = 100, acc_pool = 16;
double acc_z = 3, acc_dist = 0.5;
uint64_t tree_salt = 0;
double tol = 1e-3;
string prior_criterion = "regret";
vector<int> resolve_q;  // --resolve-now: new leaves with samples drawn at the split, finalized at once
string outlier = "esd", scale = "counterfactual", schedule = "static", prior_split = "immediate", prior_start = "equal", features = "general";
long long eps_check = 1000;

// ======================= ESD table =======================
map<pair<int, int>, vector<double>> esd_lambda;  // (n, |A|) -> lambda_1..lambda_r
int esd_nmax = 0;
void load_esd(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("cannot open the ESD table " + path + " (gen_esd_table.py)");
    string line; getline(in, line);
    while (getline(in, line)) {
        stringstream ss(line); string f; vector<string> v;
        while (getline(ss, f, ',')) v.push_back(f);
        const int n = stoi(v[0]), A = stoi(v[1]), i = stoi(v[2]);
        auto& lam = esd_lambda[{n, A}];
        if ((int)lam.size() < i) lam.resize(i);
        lam[i - 1] = stod(v[3]);
        esd_nmax = max(esd_nmax, n);
    }
}

// ======================= posterior =======================
int B = 0, level_now = 0;
vector<double> Q, S, Rtmp;
vector<int32_t> bucket_n;
vector<long long> vis;          // traverser visits since the bucket's birth (eps rule)
vector<double> parent_eps;      // the parent's eps value at the split
vector<int> touched; vector<char> dirty;
struct Member { uint64_t pri; int32_t I; };
vector<vector<Member>> members;
uint64_t member_salt = 0;
inline void strategy(int b, int n, double* s) {
    double total = 0;
    for (int a = 0; a < n; ++a) total += max(Q[(size_t)b * W + a], 0.0);
    for (int a = 0; a < n; ++a) s[a] = total > 0 ? max(Q[(size_t)b * W + a], 0.0) / total : 1.0 / n;
}
vector<double> avg_table() {  // every bucket's average strategy (uniform before any average mass)
    vector<double> t((size_t)B * W, 0.0);
    for (int b = 0; b < B; ++b) {
        const int n = bucket_n[b];
        double tot = 0;
        for (int a = 0; a < n; ++a) tot += S[(size_t)b * W + a];
        for (int a = 0; a < n; ++a) t[(size_t)b * W + a] = tot > 0 ? S[(size_t)b * W + a] / tot : 1.0 / n;
    }
    return t;
}
vector<double> strategy_table() {
    vector<double> t((size_t)B * W, 0.0);
    for (int b = 0; b < B; ++b) strategy(b, bucket_n[b], &t[(size_t)b * W]);
    return t;
}
inline double psi(const double* x, int n) { double m = 0; for (int a = 0; a < n; ++a) m = max(m, x[a]); return m; }

struct Split { int b = -1; double mu[W] = {}, sd[W] = {}, c[W] = {}; map<int, int> kids; int none_kid = -1; int outliers[W] = {}; int n = 0; double gain = 0; };
long long n_gain_rejected = 0;
struct Event {
    vector<int32_t> unsplit;    // bucket before -> id after, -1 when it split
    vector<int32_t> parent;     // bucket after -> bucket before
    vector<int32_t> split_idx;  // bucket before -> index into splits, or -1
    vector<Split> splits;
    int K = 0, Bl = 0;          // stored iterates, buckets before
    vector<double> tables;      // K x Bl x W
    long long t = 0;
};
vector<Event> events;
struct TabView { const double* t; int K, Bl; };

// label of a direction R under a split's threshold rule (restrict: only labels that have a child)
// --scale counterfactual: the test is on R(I,a); fact-counterfact: on R(I,a) / M(I) (per visit), M = counterfactual reach
// --outlier label (rule (a) of the spec): label = argmax_a R(I,a) among actions with R(I,a) > z sqrt(sum_k r_k(I,a)^2)
// over the K stored iterates (R2 = that sum of squares), else none; no bucket statistics
int label_of(const Split& sp, const double* R, const double* R2, double M, int n, bool restrict_kids) {
    if (outlier == "label") {
        // --direction avg has one stored strategy, so the z test over iterates is undefined: the gate is then the
        // per-visit regret R(I,a) / M(I) > --tol (the original rule of pure_aswiss.cpp)
        int best = -1; double bv = -1e300;
        for (int a = 0; a < n; ++a) {
            if (restrict_kids && !sp.kids.count(a)) continue;
            const bool pass = direction_avg ? (M > 0 && R[a] / M > tol) : R[a] > zthr * sqrt(max(0.0, R2[a]));
            if (pass && R[a] > bv) { bv = R[a]; best = a; }
        }
        return best;
    }
    int best = -1; double bz = -1e300;
    const bool pv = scale == "fact-counterfact";
    if (pv && !(M > 0)) return -1;
    for (int a = 0; a < n; ++a) {
        if (restrict_kids && !sp.kids.count(a)) continue;
        if (!(sp.c[a] < 1e299)) continue;
        const double z = ((pv ? R[a] / M : R[a]) - sp.mu[a]) / sp.sd[a];
        if (z >= sp.c[a] && z > bz) { bz = z; best = a; }
    }
    return best;
}
int kid_of_label(const Split& sp, int lab) { auto it = sp.kids.find(lab); return it == sp.kids.end() ? sp.none_kid : it->second; }

// ======================= prior =======================
struct Desc { int kind = 0, ch = -1, rnd = -1, ord = -1, sym = -1; };
enum { F1 = 1, F2, F2ALL, F3CNT, F3TOT, F3LAST, F4SEAT, F4TOT, F4CH, F4RND, F4CUR, PHAND, PPUB, PMAX, PPAIRS };
bool operator<(const Desc& a, const Desc& b) { return tie(a.kind, a.ch, a.rnd, a.ord, a.sym) < tie(b.kind, b.ch, b.rnd, b.ord, b.sym); }
string desc_name(const Desc& d) {
    static const char* chn[] = {"PC", "UM", "UO", "UC", "HO", "AM", "PM", "PO"};
    ostringstream o;
    auto ch = [&](int c) { return c >= 0 && c < 8 ? chn[c] : "?"; };
    auto sy = [&](int s) { return s >= 0 && s < (int)sym_names.size() ? sym_names[s] : to_string(s); };
    switch (d.kind) {
        case F1: o << "F1(" << ch(d.ch) << "," << d.rnd << "," << d.ord << ")=" << sy(d.sym); break;
        case F2: o << "F2(" << ch(d.ch) << "," << sy(d.sym) << ")"; break;
        case F2ALL: o << "F2(chance," << sy(d.sym) << ")"; break;
        case F3CNT: o << "F3(" << d.rnd << "," << ch(d.ch) << "," << sy(d.sym) << ")"; break;
        case F3TOT: o << "F3total(" << d.rnd << ")"; break;
        case F3LAST: o << "F3last(" << ch(d.ch) << ")=" << sy(d.sym); break;
        case F4SEAT: o << "F4seat"; break;
        case F4TOT: o << "F4observations"; break;
        case F4CH: o << "F4chance"; break;
        case F4RND: o << "F4round"; break;
        case F4CUR: o << "F4round_actions"; break;
        case PHAND: o << "Phand(" << sy(d.sym) << ")"; break;
        case PPUB: o << "Ppublic(" << sy(d.sym) << ")"; break;
        case PMAX: o << "Pmax_same"; break;
        case PPAIRS: o << "Ppairs"; break;
    }
    return o.str();
}
double feat(int I, const Desc& d) {
    const int32_t* b = &obs[obs_start[I]];
    const int n = (int)(obs_start[I + 1] - obs_start[I]);
    auto cnt = [&](int ch, int sym) { int c = 0; for (int k = 0; k < n; ++k) if (o_ch(b[k]) == ch && o_sym(b[k]) == sym) ++c; return c; };
    auto hand = [&](int sym) { return cnt(PC, sym) - cnt(AM, sym) + cnt(UM, sym) + cnt(UC, sym); };
    switch (d.kind) {
        case F1: for (int k = 0; k < n; ++k) if (o_ch(b[k]) == d.ch && o_rnd(b[k]) == d.rnd && o_ord(b[k]) == d.ord) return o_sym(b[k]) == d.sym; return 0;
        case F2: return cnt(d.ch, d.sym);
        case F2ALL: { int c = 0; for (int k = 0; k < n; ++k) if (chance_ch(o_ch(b[k])) && o_sym(b[k]) == d.sym) ++c; return c; }
        case F3CNT: { int c = 0; for (int k = 0; k < n; ++k) if (o_ch(b[k]) == d.ch && o_rnd(b[k]) == d.rnd && o_sym(b[k]) == d.sym) ++c; return c; }
        case F3TOT: { int c = 0; for (int k = 0; k < n; ++k) if (!chance_ch(o_ch(b[k])) && o_rnd(b[k]) == d.rnd) ++c; return c; }
        case F3LAST: {
            int best = -1, key = -1;
            for (int k = 0; k < n; ++k) if (o_ch(b[k]) == d.ch) { const int kk = o_rnd(b[k]) * 256 + o_ord(b[k]); if (kk > key) { key = kk; best = o_sym(b[k]); } }
            return best == d.sym;
        }
        case F4SEAT: return seat_of[I];
        case F4TOT: return n;
        case F4CH: { int c = 0; for (int k = 0; k < n; ++k) if (chance_ch(o_ch(b[k]))) ++c; return c; }
        case F4RND: { int r = 0; for (int k = 0; k < n; ++k) r = max(r, o_rnd(b[k])); return r; }
        case F4CUR: {
            int r = 0; for (int k = 0; k < n; ++k) r = max(r, o_rnd(b[k]));
            int c = 0; for (int k = 0; k < n; ++k) if ((o_ch(b[k]) == PM || o_ch(b[k]) == PO) && o_rnd(b[k]) == r) ++c; return c;
        }
        case PHAND: return hand(d.sym);
        case PPUB: return cnt(UM, d.sym) + cnt(PM, d.sym);
        case PMAX: { int m = 0; for (int s : card_syms) m = max(m, hand(s)); return m; }
        case PPAIRS: { int c = 0; for (int s : card_syms) c += hand(s) >= 2; return c; }
    }
    return 0;
}
struct TNode { Desc d; double thr = 0; int left = -1, right = -1, leaf = -1; };
struct Leaf {
    int node = -1, cell = -1, depth = 0;
    bool pending = false;
    int ptr = -1, prov = -1, start_level = 0, start_b = -1;
    vector<int32_t> samples;
    long long born = 0;
    int sig_level = -1, sig_bucket = -1;  // a signal waiting (held or epoch mode)
};
vector<TNode> tnodes;
vector<int> cell_root, cell_constraint, cell_ver;
vector<vector<int32_t>> cell_members;
vector<Leaf> leaves;
vector<int32_t> cell_of_I, leaf_cache, leaf_cache_ver;
long long assign_ver = 1;
vector<int32_t> pend_cache, pend_cache_b; vector<long long> pend_cache_ver;
vector<int> waiting;           // leaves with a signal waiting (held in immediate mode, flagged in epoch mode)
vector<int> to_finalize;       // leaves that reached m samples in this iteration
int leaves_at_event = 0;
long long n_finalize = 0, n_signals = 0, n_prior_splits = 0, n_prior_rejected = 0;
FILE* prior_log = nullptr;
FILE* evf_global = nullptr;  // events.csv (--mode ppacc writes its posterior splits there)

int leaf_of(int I) {
    const int c = cell_of_I[I];
    if (leaf_cache_on && leaf_cache_ver[I] == cell_ver[c]) return leaf_cache[I];
    int x = cell_root[c];
    while (tnodes[x].leaf < 0) x = feat(I, tnodes[x].d) <= tnodes[x].thr ? tnodes[x].left : tnodes[x].right;
    if (leaf_cache_on) { leaf_cache[I] = tnodes[x].leaf; leaf_cache_ver[I] = cell_ver[c]; }
    return tnodes[x].leaf;
}

// ======================= --mode tree: accumulated regrets on candidate tests =======================
struct AccCand { Desc d; double thr = 0; double s[2][W] = {}, q[2][W] = {}, n[2] = {}; };
struct LeafAcc { vector<AccCand> cand; vector<int32_t> pool; uint64_t seen = 0; uint32_t gen = 0; };
vector<LeafAcc> lacc;
long long n_acc_tested = 0;
struct LeafOwn { double s[W] = {}, q[W] = {}, n = 0; };   // --mode ppacc: a leaf's regret sums since its pointer was set
vector<LeafOwn> lown;
long long n_post_splits = 0, n_diverging = 0;
void acc_visit(int l, int I, const double* r, int n) {
    if ((int)lacc.size() <= l) lacc.resize(leaves.size());
    LeafAcc& L = lacc[l];
    for (AccCand& c : L.cand) {
        const int side = feat(I, c.d) <= c.thr ? 0 : 1;
        c.n[side] += 1;
        for (int a = 0; a < n; ++a) { c.s[side][a] += r[a]; c.q[side][a] += r[a] * r[a]; }
    }
    if (find(L.pool.begin(), L.pool.end(), I) != L.pool.end()) return;  // the reservoir holds distinct information sets
    ++L.seen;
    if ((int)L.pool.size() < acc_pool) L.pool.push_back(I);
    else { const uint64_t j = mix64(tree_salt ^ ((uint64_t)l << 32) ^ L.seen) % L.seen; if (j < (uint64_t)acc_pool) L.pool[j] = I; }
}
struct AccTest { bool testable = false, pass = false; double d2 = 0, z = 0, gain = 0, dist = 0; };
AccTest acc_test(const AccCand& c, int n) {
    AccTest t;
    const double nl = c.n[0], nr = c.n[1];
    if (nl < acc_min || nr < acc_min) return t;
    t.testable = true;
    const double h = 1 / nl + 1 / nr;
    double mu = 0, var = 0;
    for (int a = 0; a < n; ++a) {
        const double ml = c.s[0][a] / nl, mr = c.s[1][a] / nr;
        const double within = max(0.0, c.q[0][a] - c.s[0][a] * ml) + max(0.0, c.q[1][a] - c.s[1][a] * mr);
        const double v = within / max(1.0, nl + nr - 2);
        t.d2 += (ml - mr) * (ml - mr); mu += v * h; var += 2 * (v * h) * (v * h);
    }
    t.z = var > 0 ? (t.d2 - mu) / sqrt(var) : (t.d2 > 0 ? 1e300 : 0);
    t.gain = nl * nr / (nl + nr) * t.d2;
    t.pass = t.z >= acc_z;
    if (acc_dir) {
        double u[2][W] = {};
        for (int side = 0; side < 2; ++side) {
            const double m_n = c.n[side];
            double m2 = 0, mu1 = 0, var1 = 0;
            for (int a = 0; a < n; ++a) {
                const double m = c.s[side][a] / m_n;
                const double v = max(0.0, c.q[side][a] - c.s[side][a] * m) / max(1.0, m_n - 1);
                m2 += m * m; mu1 += v / m_n; var1 += 2 * (v / m_n) * (v / m_n);
            }
            const double zs = var1 > 0 ? (m2 - mu1) / sqrt(var1) : (m2 > 0 ? 1e300 : 0);
            if (zs >= acc_z && m2 > 0) for (int a = 0; a < n; ++a) u[side][a] = c.s[side][a] / m_n / sqrt(m2);
        }
        for (int a = 0; a < n; ++a) t.dist += (u[0][a] - u[1][a]) * (u[0][a] - u[1][a]);
        t.pass = t.pass && t.dist >= acc_dist;
    }
    return t;
}
vector<Desc> candidates_of(int I);
void acc_fill(int l) {
    if ((int)lacc.size() <= l) lacc.resize(leaves.size());
    LeafAcc& L = lacc[l];
    if ((int)L.cand.size() >= acc_k || L.pool.size() < 2) return;
    mt19937_64 r(mix64(tree_salt ^ 0xacc1ull ^ ((uint64_t)l << 24) ^ (uint64_t)L.gen++));
    for (int tries = 0; (int)L.cand.size() < acc_k && tries < 64 * acc_k; ++tries) {
        const size_t i = r() % L.pool.size(), j = r() % L.pool.size();
        if (i == j) continue;
        const vector<Desc> fs = candidates_of(L.pool[i]);
        if (fs.empty()) continue;
        const Desc d = fs[r() % fs.size()];
        const double vi = feat(L.pool[i], d), vj = feat(L.pool[j], d);
        if (vi == vj) continue;
        const double thr = 0.5 * (vi + vj);
        bool dup = false;
        for (const AccCand& c : L.cand) if (!(c.d < d) && !(d < c.d) && c.thr == thr) { dup = true; break; }
        if (dup) continue;
        AccCand c; c.d = d; c.thr = thr;
        L.cand.push_back(c);
    }
}

// ======================= routing and directions =======================
vector<int32_t> p0_bucket, bucket_now;
unordered_map<uint64_t, int32_t> memo;
int route(int I, int level);
int route_from(int I, int level0, int b0);
int ancestor(int b, int level) { for (int e = level_now - 1; e >= level; --e) b = events[e].parent[b]; return b; }
inline int eval_bucket(int I) {  // pp: the current bucket used by other information sets' computations
    const Leaf& L = leaves[leaf_of(I)];
    return L.pending ? L.prov : L.ptr;
}
inline int bucket_at(int I, int level) { return pp_mode ? ancestor(eval_bucket(I), level) : route(I, level); }
struct Count { long long directions = 0, nodes = 0, routes = 0; } cnt;

void sub_values(int h, const TabView& tv, int level, double* out) {
    ++cnt.nodes;
    const int pl = players[h];
    if (pl == TERMINAL) { for (int j = 0; j < tv.K; ++j) out[j] = util[h]; return; }
    const int64_t s0 = start[h]; const int d = degree[h];
    double child[KMAX];
    for (int j = 0; j < tv.K; ++j) out[j] = 0;
    if (pl == CHANCE) {
        for (int k = 0; k < d; ++k) {
            sub_values(edges[s0 + k], tv, level, child);
            for (int j = 0; j < tv.K; ++j) out[j] += prob[s0 + k] * child[j];
        }
        return;
    }
    const size_t bx = (size_t)bucket_at(info[h], level);
    for (int a = 0; a < d; ++a) {
        sub_values(edges[s0 + a], tv, level, child);
        for (int j = 0; j < tv.K; ++j) out[j] += tv.t[((size_t)j * tv.Bl + bx) * W + a] * child[j];
    }
}
// summed counterfactual regret R(a) = sum_j r^{sigma_j}(I, a) and summed counterfactual reach M of information set I in
// bucket b of the partition at `level`, under the K strategy tables over that partition
void direction(int I, int b, const TabView& tv, int level, double* R, double& M, double* R2 = nullptr, vector<double>* per = nullptr) {
    ++cnt.directions;
    const int n = counts[I], pl = players[nodes_of[I][0]], K = tv.K;
    for (int a = 0; a < W; ++a) R[a] = 0;
    M = 0;
    // local buffers: in pure mode a direction computation routes other information sets, which can compute their
    // directions (recursion); shared buffers would be overwritten by the inner call (found by test T2)
    vector<double> rj((size_t)K * W, 0.0), mj(K, 0.0), popp(K, 1.0), v((size_t)W * KMAX, 0.0);
    vector<int> path;
    for (int h : nodes_of[I]) {
        path.clear();
        for (int x = h; parent_node[x] >= 0; x = parent_node[x]) path.push_back(x);
        fill(popp.begin(), popp.end(), 1.0);
        double pc = 1;
        for (size_t i = path.size(); i-- > 0;) {
            const int x = path[i], p = parent_node[x], slot = parent_slot[x];
            if (players[p] == CHANCE) pc *= prob[start[p] + slot];
            else if (players[p] != pl) {
                const size_t bx = (size_t)bucket_at(info[p], level);
                for (int j = 0; j < K; ++j) popp[j] *= tv.t[((size_t)j * tv.Bl + bx) * W + slot];
            }
        }
        for (int a = 0; a < n; ++a) sub_values(edges[start[h] + a], tv, level, &v[(size_t)a * KMAX]);
        const double sign = pl == 0 ? 1.0 : -1.0;
        for (int j = 0; j < K; ++j) {
            const double* s = &tv.t[((size_t)j * tv.Bl + b) * W];
            double node = 0;
            for (int a = 0; a < n; ++a) node += s[a] * v[(size_t)a * KMAX + j];
            const double cf = popp[j] * pc;
            for (int a = 0; a < n; ++a) rj[(size_t)j * W + a] += sign * (v[(size_t)a * KMAX + j] - node) * cf;
            mj[j] += cf;
        }
    }
    for (int j = 0; j < K; ++j) { for (int a = 0; a < n; ++a) R[a] += rj[(size_t)j * W + a]; M += mj[j]; }
    if (R2) { for (int a = 0; a < W; ++a) R2[a] = 0; for (int j = 0; j < K; ++j) for (int a = 0; a < n; ++a) R2[a] += rj[(size_t)j * W + a] * rj[(size_t)j * W + a]; }
    if (per) *per = rj;
}
inline TabView view(const Event& ev) { return {ev.tables.data(), ev.K, ev.Bl}; }
int kid_for(const Event& ev, int b, int I, int level) {
    const Split& sp = ev.splits[ev.split_idx[b]];
    double R[W], M, R2[W];
    direction(I, b, view(ev), level, R, M, R2);
    return kid_of_label(sp, label_of(sp, R, R2, M, counts[I], true));
}
int route(int I, int level) {
    ++cnt.routes;
    if (level == 0) return p0_bucket[I];
    if (table_on && level == level_now && !bucket_now.empty()) return bucket_now[I];
    const uint64_t key = (uint64_t)I << 8 | (uint64_t)level;
    if (memo_on) { auto it = memo.find(key); if (it != memo.end()) return it->second; }
    const int b = route(I, level - 1);
    const Event& ev = events[level - 1];
    const int nb = ev.unsplit[b] >= 0 ? ev.unsplit[b] : kid_for(ev, b, I, level - 1);
    if (memo_on) memo.emplace(key, nb);
    return nb;
}
int route_from(int I, int level0, int b0) {  // pp: one information set through the genealogy from (level0, b0)
    int b = b0;
    for (int lv = level0; lv < level_now; ++lv) {
        const Event& ev = events[lv];
        b = ev.unsplit[b] >= 0 ? ev.unsplit[b] : kid_for(ev, b, I, lv);
    }
    return b;
}
double assign_cpu = 0;

int train_bucket(int I) {
    if (tree_mode || ppacc_mode) return leaves[leaf_of(I)].ptr;
    if (!pp_mode) return table_on ? bucket_now[I] : route(I, level_now);
    const int l = leaf_of(I);
    const Leaf& L = leaves[l];
    if (!L.pending) return L.ptr;
    if (pend_cache_on && pend_cache[I] >= 0 && pend_cache_ver[I] == assign_ver && pend_cache_b[I] == l) return pend_cache[I];
    double t0 = cpu_now();
    const int b = route_from(I, L.start_level, L.start_b);
    assign_cpu += cpu_now() - t0;
    if (pend_cache_on) { pend_cache[I] = b; pend_cache_b[I] = l; pend_cache_ver[I] = assign_ver; }
    return b;
}

// ======================= MCCFR =======================
mt19937_64 rng, iter_rng, split_rng;
vector<double> Rmeas, Rwin; vector<long long> vmeas, vwin;  // --measure: per information set, cumulative and epoch window
inline void touch(int b) { if (!dirty[b]) { dirty[b] = 1; touched.push_back(b); } }
void apply_regrets() {
    for (int b : touched) { for (int a = 0; a < W; ++a) { Q[(size_t)b * W + a] += Rtmp[(size_t)b * W + a]; Rtmp[(size_t)b * W + a] = 0; } dirty[b] = 0; }
    touched.clear();
}
void add_member(int b, int I) {
    auto& mb = members[b];
    const uint64_t pri = mix64(member_salt ^ (uint64_t)I * 0x9E3779B97F4A7C15ull);
    if ((int)mb.size() >= members_cap) {
        auto mx = max_element(mb.begin(), mb.end(), [](const Member& x, const Member& y) { return x.pri < y.pri; });
        if (pri >= mx->pri) return;
        for (const Member& m : mb) if (m.I == I) return;
        *mx = {pri, I};
        return;
    }
    for (const Member& m : mb) if (m.I == I) return;
    mb.push_back({pri, I});
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
    const int I = info[h], b = train_bucket(I);
    double s[W];
    strategy(b, d, s);
    if (pl == p) {
        double v[W], node = 0;
        for (int a = 0; a < d; ++a) { v[a] = es(edges[e + a], p); node += s[a] * v[a]; }
        touch(b);
        for (int a = 0; a < d; ++a) Rtmp[(size_t)b * W + a] += v[a] - node;
        if (measure) {
            for (int a = 0; a < d; ++a) { Rmeas[(size_t)I * W + a] += v[a] - node; Rwin[(size_t)I * W + a] += v[a] - node; }
            ++vmeas[I]; ++vwin[I];
        }
        ++vis[b];
        add_member(b, I);
        if (tree_mode || ppacc_mode) {
            double r[W]; for (int a = 0; a < d; ++a) r[a] = v[a] - node;
            const int l = leaf_of(I);
            acc_visit(l, I, r, d);
            if (ppacc_mode) {
                if ((int)lown.size() <= l) lown.resize(leaves.size());
                LeafOwn& o = lown[l];
                for (int a = 0; a < d; ++a) { o.s[a] += r[a]; o.q[a] += r[a] * r[a]; }
                o.n += 1;
            }
        }
        if (pp_mode) {
            const int l = leaf_of(I);
            Leaf& L = leaves[l];
            if (L.pending && (int)L.samples.size() < m_samples) {
                L.samples.push_back(I);
                if ((int)L.samples.size() == m_samples) to_finalize.push_back(l);
            }
        }
        return node;
    }
    for (int a = 0; a < d; ++a) S[(size_t)b * W + a] += s[a];
    double x = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
    int k = 0;
    for (; k < d - 1; ++k) { acc += s[k]; if (x < acc) break; }
    return es(edges[e + k], p);
}

// ======================= prior splits =======================
struct Row { int I; double w; double x[W]; int lab = -1; };  // lab: the posterior child its direction picks at the event
struct Candidate { Desc d; double thr = 0, gainA = 0, gainB = 0; bool ok = false; };
// weighted squared error of vectors x (n dims) with weights w: sum w|x|^2 - |sum w x|^2 / sum w
struct Acc { double w = 0, wx[W] = {}, wxx = 0; void add(const Row& r, int n) { w += r.w; for (int a = 0; a < n; ++a) { wx[a] += r.w * r.x[a]; wxx += r.w * r.x[a] * r.x[a]; } } };
double sse(const Acc& c, int n) { if (c.w <= 0) return 0; double q = 0; for (int a = 0; a < n; ++a) q += c.wx[a] * c.wx[a]; return c.wxx - q / c.w; }
vector<Desc> candidates(const vector<Row>& rows) {
    set<Desc> ds;
    for (const Row& r : rows) {
        for (int64_t k = obs_start[r.I]; k < obs_start[r.I + 1]; ++k) {
            const int32_t c = obs[k];
            const int ch = o_ch(c), rnd = o_rnd(c), ord = o_ord(c), sym = o_sym(c);
            ds.insert({F1, ch, rnd, ord, sym});
            ds.insert({F2, ch, -1, -1, sym});
            if (chance_ch(ch)) ds.insert({F2ALL, -1, -1, -1, sym});
            if (ch == PM || ch == PO) { ds.insert({F3CNT, ch, rnd, -1, sym}); ds.insert({F3LAST, ch, -1, -1, sym}); }
            if (!chance_ch(ch)) ds.insert({F3TOT, -1, rnd, -1, -1});
        }
    }
    for (int k : {F4SEAT, F4TOT, F4CH, F4RND, F4CUR}) ds.insert({k, -1, -1, -1, -1});
    if (features == "poker") {
        for (int s : card_syms) { ds.insert({PHAND, -1, -1, -1, s}); ds.insert({PPUB, -1, -1, -1, s}); }
        ds.insert({PMAX, -1, -1, -1, -1}); ds.insert({PPAIRS, -1, -1, -1, -1});
    }
    return vector<Desc>(ds.begin(), ds.end());
}
vector<Desc> candidates_of(int I) { Row r; r.I = I; r.w = 1; return candidates({r}); }
// --prior-criterion label: the reduction of the reach-weighted Gini impurity of the rows' labels (the posterior child
// each information set's own direction picks at the event), w (1 - sum p^2) = w - sum c^2 / w
Candidate best_split_label(const vector<Row>& rows, const vector<char>& half) {
    Candidate best;
    if (rows.size() < 2) return best;
    map<int, int> li;
    for (const Row& r : rows) li.emplace(r.lab, 0);
    int C = 0; for (auto& [l, i] : li) i = C++;
    if (C < 2) return best;
    auto imp = [&](const vector<double>& c, double w) { if (w <= 0) return 0.0; double q = 0; for (double x : c) q += x * x; return w - q / w; };
    vector<double> totA(C, 0.0); double wA = 0; int nA = 0;
    for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) { totA[li[rows[i].lab]] += rows[i].w; wA += rows[i].w; ++nA; }
    const double baseA = imp(totA, wA);
    const vector<Desc> ds = candidates(rows);
    vector<pair<double, int>> va;
    vector<double> left(C), right(C);
    for (const Desc& d : ds) {
        va.clear();
        for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) va.push_back({feat(rows[i].I, d), (int)i});
        sort(va.begin(), va.end());
        fill(left.begin(), left.end(), 0.0); double wl = 0;
        for (int k = 0; k + 1 < (int)va.size(); ++k) {
            const Row& r = rows[va[k].second];
            left[li[r.lab]] += r.w; wl += r.w;
            if (va[k].first == va[k + 1].first) continue;
            const int nl = k + 1, nr = nA - nl;
            if (nl < min_leaf || nr < min_leaf) continue;
            for (int c = 0; c < C; ++c) right[c] = totA[c] - left[c];
            const double g = baseA - imp(left, wl) - imp(right, wA - wl);
            if (g > best.gainA * (1 + 1e-12) + 1e-300) { best.gainA = g; best.d = d; best.thr = 0.5 * (va[k].first + va[k + 1].first); best.ok = true; }
        }
    }
    if (!best.ok) return best;
    if (nA == (int)rows.size()) { best.gainB = best.gainA; best.ok = best.gainA > 1e-12 * wA; return best; }
    vector<double> pB(C, 0.0), lB(C, 0.0), rB(C, 0.0); double wp = 0, wl = 0, wr = 0;
    for (size_t i = 0; i < rows.size(); ++i) if (half[i]) {
        const int c = li[rows[i].lab]; pB[c] += rows[i].w; wp += rows[i].w;
        if (feat(rows[i].I, best.d) <= best.thr) { lB[c] += rows[i].w; wl += rows[i].w; } else { rB[c] += rows[i].w; wr += rows[i].w; }
    }
    best.gainB = imp(pB, wp) - imp(lB, wl) - imp(rB, wr);
    best.ok = best.gainB > 1e-12 * max(wp, 1e-300) && wl > 0 && wr > 0;
    return best;
}
// the best binary test on the rows: chosen on half A (rows with half[i] == 0), checked on half B
Candidate best_split(const vector<Row>& rows, const vector<char>& half, int n) {
    Candidate best;
    if (prior_criterion == "label") return best_split_label(rows, half);
    if (rows.size() < 2) return best;
    const vector<Desc> ds = candidates(rows);
    vector<pair<double, int>> va;
    Acc totA; for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) totA.add(rows[i], n);
    const double baseA = sse(totA, n);
    int nA = 0; for (char h : half) nA += !h;
    for (const Desc& d : ds) {
        va.clear();
        for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) va.push_back({feat(rows[i].I, d), (int)i});
        sort(va.begin(), va.end());
        Acc left;
        for (int k = 0; k + 1 < (int)va.size(); ++k) {
            left.add(rows[va[k].second], n);
            if (va[k].first == va[k + 1].first) continue;
            const int nl = k + 1, nr = nA - nl;
            if (nl < min_leaf || nr < min_leaf) continue;
            Acc right; right.w = totA.w - left.w; right.wxx = totA.wxx - left.wxx;
            for (int a = 0; a < n; ++a) right.wx[a] = totA.wx[a] - left.wx[a];
            const double g = baseA - sse(left, n) - sse(right, n);
            if (g > best.gainA * (1 + 1e-12) + 1e-300) { best.gainA = g; best.d = d; best.thr = 0.5 * (va[k].first + va[k + 1].first); best.ok = true; }
        }
    }
    if (!best.ok) return best;
    if (nA == (int)rows.size()) {  // --crossfit 0: chosen and accepted on all rows (exact small games have no noise)
        best.gainB = best.gainA;
        best.ok = best.gainA > 1e-12 * max(1.0, baseA);
        return best;
    }
    Acc pB, lB, rB;
    for (size_t i = 0; i < rows.size(); ++i) if (half[i]) { pB.add(rows[i], n); (feat(rows[i].I, best.d) <= best.thr ? lB : rB).add(rows[i], n); }
    best.gainB = sse(pB, n) - sse(lB, n) - sse(rB, n);
    best.ok = best.gainB > 1e-12 * max(1.0, sse(pB, n)) && lB.w > 0 && rB.w > 0;
    return best;
}
// one row: I's per-visit direction at event lv under bucket bs, its counterfactual reach, and its label there
bool fill_row(Row& r, int I, int bs, int lv) {
    const Event& ev = events[lv];
    double R[W], M, R2[W];
    direction(I, bs, view(ev), lv, R, M, R2);
    if (!(M > 0)) return false;
    r.I = I; r.w = M;
    for (int a = 0; a < W; ++a) r.x[a] = a < counts[I] ? R[a] / M : 0;
    if (ev.split_idx[bs] >= 0) { const Split& sp = ev.splits[ev.split_idx[bs]]; r.lab = kid_of_label(sp, label_of(sp, R, R2, M, counts[I], true)); }
    else r.lab = -1;
    return true;
}
// rows of a leaf: its information sets with per-visit directions at the signal event (bucket bs at level lv)
vector<Row> leaf_rows(int l, int lv, int bs) {
    vector<Row> rows;
    for (int I : cell_members[leaves[l].cell]) {
        if (leaf_of(I) != l) continue;
        Row r;
        if (fill_row(r, I, bs, lv)) rows.push_back(r);
    }
    return rows;
}
long long cur_t = 0;
void finalize(int l);
// split leaf l by candidate cd: l's node becomes internal, two new pending leaves (start of the genealogy)
void commit_split(int l, const Candidate& cd, int lv, const vector<Row>& rows) {
    const size_t nrows = rows.size();
    const int node = leaves[l].node, cell = leaves[l].cell, depth = leaves[l].depth;
    const int prov = leaves[l].pending ? leaves[l].prov : leaves[l].ptr;
    tnodes[node].d = cd.d; tnodes[node].thr = cd.thr; tnodes[node].leaf = -1;
    int kidn[2];
    for (int s = 0; s < 2; ++s) {
        TNode t; t.leaf = (int)leaves.size();
        kidn[s] = (int)tnodes.size(); tnodes.push_back(t);
        Leaf L; L.node = kidn[s]; L.cell = cell; L.depth = depth + 1; L.pending = true; L.prov = prov;
        L.start_level = 0; L.start_b = cell_constraint[cell]; L.born = cur_t;
        if (leaf_start_parent) {  // --leaf-start parent: the new leaf keeps its parent's place (no route from the start)
            const Leaf& P = leaves[l];
            if (P.pending) { L.start_level = P.start_level; L.start_b = P.start_b; }
            else { L.pending = false; L.ptr = P.ptr; }
        }
        leaves.push_back(L);
    }
    tnodes[node].left = kidn[0]; tnodes[node].right = kidn[1];
    leaves[l].node = -1; leaves[l].pending = false; leaves[l].samples.clear(); leaves[l].sig_level = -1;
    ++cell_ver[cell]; ++assign_ver; ++n_prior_splits;
    fprintf(prior_log, "%lld,%d,%d,%d,%s,%.6g,%.6g,%.6g,%zu,%d,%d,%d\n", cur_t, l, cell, depth, desc_name(cd.d).c_str(), cd.thr,
            cd.gainA, cd.gainB, nrows, lv, (int)leaves.size() - 2, (int)leaves.size() - 1);
    if (resolve_now && !leaf_start_parent) {  // m samples per new leaf drawn from its rows by counterfactual reach, finalized right away
        for (int s = 0; s < 2; ++s) {
            const int nl = (int)leaves.size() - 2 + s;
            vector<const Row*> part; double tw = 0;
            for (const Row& r : rows) if ((feat(r.I, cd.d) <= cd.thr) == (s == 0)) { part.push_back(&r); tw += r.w; }
            if (part.empty() || !(tw > 0)) continue;
            vector<int32_t> smp;
            for (int k = 0; k < m_samples; ++k) {
                double x = uniform_real_distribution<double>(0, tw)(split_rng), acc = 0;
                size_t j = 0;
                for (; j + 1 < part.size(); ++j) { acc += part[j]->w; if (x < acc) break; }
                smp.push_back(part[j]->I);
            }
            leaves[nl].samples = move(smp);
            resolve_q.push_back(nl);
        }
    }
}
void drain_resolve() {  // finalize the leaves queued by --resolve-now (a finalize may split again and queue more)
    while (!resolve_q.empty()) { const int l = resolve_q.back(); resolve_q.pop_back(); if (leaves[l].node >= 0) finalize(l); }
}
int n_leaves_alive() { int c = 0; for (const Leaf& L : leaves) c += L.node >= 0; return c; }
// --mode tree: the judgment at an epoch end (one split per leaf; new leaves are not judged until the next epoch end)
int tree_epoch_end() {
    lacc.resize(leaves.size());
    const int nl0 = (int)leaves.size();
    int splits = 0;
    for (int l = 0; l < nl0; ++l) {
        if (leaves[l].node < 0) continue;
        const int n = counts[cell_members[leaves[l].cell][0]];
        int best = -1;
        AccTest bt;
        vector<AccTest> ts(lacc[l].cand.size());
        for (size_t k = 0; k < ts.size(); ++k) {
            ts[k] = acc_test(lacc[l].cand[k], n);
            n_acc_tested += ts[k].testable;
            const bool better = acc_dir ? (ts[k].dist > bt.dist || (ts[k].dist == bt.dist && ts[k].gain > bt.gain)) : ts[k].gain > bt.gain;
            if (ts[k].pass && (best < 0 || better)) { best = (int)k; bt = ts[k]; }
        }
        if (best < 0) {  // drop the tested-and-failed and the one-sided candidates, refill
            vector<AccCand> keep;
            for (size_t k = 0; k < ts.size(); ++k) {
                const AccCand& c = lacc[l].cand[k];
                const bool one_sided = c.n[0] + c.n[1] >= 4.0 * acc_min && min(c.n[0], c.n[1]) < acc_min;
                if (!(ts[k].testable && !ts[k].pass) && !one_sided) keep.push_back(c);
            }
            lacc[l].cand.swap(keep);
            acc_fill(l);
            continue;
        }
        const AccCand c = lacc[l].cand[best];
        const vector<int32_t> pool = lacc[l].pool;
        const int node = leaves[l].node, cell = leaves[l].cell, depth = leaves[l].depth, pb = leaves[l].ptr;
        tnodes[node].d = c.d; tnodes[node].thr = c.thr; tnodes[node].leaf = -1;
        // buckets: the left leaf keeps the parent's id, the right one gets a new id; both start from
        // inherit_r x / inherit_s x the parent's tables
        const int nb = B++;
        Q.resize((size_t)B * W, 0); S.resize((size_t)B * W, 0); Rtmp.resize((size_t)B * W, 0);
        bucket_n.push_back(bucket_n[pb]); vis.push_back(0); parent_eps.push_back(-1); members.emplace_back(); dirty.push_back(0);
        for (int a = 0; a < W; ++a) {
            Q[(size_t)nb * W + a] = inherit_r * Q[(size_t)pb * W + a]; S[(size_t)nb * W + a] = inherit_s * S[(size_t)pb * W + a];
            Q[(size_t)pb * W + a] *= inherit_r; S[(size_t)pb * W + a] *= inherit_s;
        }
        int kid[2];
        for (int side = 0; side < 2; ++side) {
            TNode t; t.leaf = (int)leaves.size();
            kid[side] = (int)tnodes.size(); tnodes.push_back(t);
            Leaf L; L.node = kid[side]; L.cell = cell; L.depth = depth + 1; L.pending = false; L.ptr = side == 0 ? pb : nb; L.born = cur_t;
            leaves.push_back(L);
            LeafAcc K;
            for (int32_t I : pool) if ((feat(I, c.d) <= c.thr) == (side == 0)) K.pool.push_back(I);
            K.seen = K.pool.size();
            lacc.push_back(K);
            acc_fill((int)leaves.size() - 1);
        }
        tnodes[node].left = kid[0]; tnodes[node].right = kid[1];
        leaves[l].node = -1; lacc[l] = LeafAcc{};
        ++cell_ver[cell]; ++n_prior_splits; ++splits;
        fprintf(prior_log, "%lld,%d,%d,%d,%s,%.6g,%.6g,%.6g,%.0f,%d,%d,%d\n", cur_t, l, cell, depth, desc_name(c.d).c_str(), c.thr,
                acc_dir ? bt.dist : bt.gain, bt.z, c.n[0] + c.n[1], level_now, (int)leaves.size() - 2, (int)leaves.size() - 1);
    }
    if (splits) ++level_now;
    return splits;
}

// --mode ppacc helpers: is a leaf's mean regret significantly non-zero (the one-side test of --acc-metric dir), and
// its largest-regret action
bool own_significant(const LeafOwn& o, int na, int& argmax) {
    argmax = -1;
    if (o.n < acc_min) return false;
    double m2 = 0, mu1 = 0, var1 = 0, best = -1e300;
    for (int a = 0; a < na; ++a) {
        const double m = o.s[a] / o.n;
        const double v = max(0.0, o.q[a] - o.s[a] * m) / max(1.0, o.n - 1);
        m2 += m * m; mu1 += v / o.n; var1 += 2 * (v / o.n) * (v / o.n);
        if (m > best) { best = m; argmax = a; }
    }
    const double z = var1 > 0 ? (m2 - mu1) / sqrt(var1) : (m2 > 0 ? 1e300 : 0);
    return z >= acc_z && m2 > 0;
}
// --mode ppacc: the judgment at an epoch end, a prior phase (even epochs) or a posterior phase (odd epochs)
int ppacc_epoch_end(int k) {
    lacc.resize(leaves.size()); lown.resize(leaves.size());
    const int nl0 = (int)leaves.size();
    vector<vector<int>> users(B);
    vector<double> bvis(B, 0.0);
    for (int l = 0; l < nl0; ++l) if (leaves[l].node >= 0) { users[leaves[l].ptr].push_back(l); bvis[leaves[l].ptr] += lown[l].n; }
    int splits = 0;
    if (k % 2 == 0) {  // ---- prior phase ----
        for (int l = 0; l < nl0; ++l) {
            if (leaves[l].node < 0) continue;
            const int b = leaves[l].ptr;
            const bool eligible = users[b].size() == 1 || (bvis[b] > 0 && lown[l].n >= 0.5 * bvis[b]);
            if (!eligible) { lacc[l].cand.clear(); continue; }
            const int n = counts[cell_members[leaves[l].cell][0]];
            int best = -1; AccTest bt;
            vector<AccTest> ts(lacc[l].cand.size());
            for (size_t c = 0; c < ts.size(); ++c) {
                ts[c] = acc_test(lacc[l].cand[c], n);
                n_acc_tested += ts[c].testable;
                const bool better = acc_dir ? (ts[c].dist > bt.dist || (ts[c].dist == bt.dist && ts[c].gain > bt.gain)) : ts[c].gain > bt.gain;
                if (ts[c].pass && (best < 0 || better)) { best = (int)c; bt = ts[c]; }
            }
            if (best < 0) {
                vector<AccCand> keep;
                for (size_t c = 0; c < ts.size(); ++c) {
                    const AccCand& x = lacc[l].cand[c];
                    const bool one_sided = x.n[0] + x.n[1] >= 4.0 * acc_min && min(x.n[0], x.n[1]) < acc_min;
                    if (!(ts[c].testable && !ts[c].pass) && !one_sided) keep.push_back(x);
                }
                lacc[l].cand.swap(keep);
                acc_fill(l);
                continue;
            }
            const AccCand c = lacc[l].cand[best];
            const vector<int32_t> pool = lacc[l].pool;
            const int node = leaves[l].node, cell = leaves[l].cell, depth = leaves[l].depth;
            tnodes[node].d = c.d; tnodes[node].thr = c.thr; tnodes[node].leaf = -1;
            int kid[2];
            for (int side = 0; side < 2; ++side) {
                TNode t; t.leaf = (int)leaves.size();
                kid[side] = (int)tnodes.size(); tnodes.push_back(t);
                Leaf L; L.node = kid[side]; L.cell = cell; L.depth = depth + 1; L.pending = false; L.ptr = b; L.born = cur_t;
                leaves.push_back(L);
                LeafAcc K;
                for (int32_t I : pool) if ((feat(I, c.d) <= c.thr) == (side == 0)) K.pool.push_back(I);
                K.seen = K.pool.size();
                lacc.push_back(K);
                LeafOwn o; for (int a = 0; a < W; ++a) { o.s[a] = c.s[side][a]; o.q[a] = c.q[side][a]; } o.n = c.n[side];
                lown.push_back(o);
                acc_fill((int)leaves.size() - 1);
            }
            tnodes[node].left = kid[0]; tnodes[node].right = kid[1];
            leaves[l].node = -1; lacc[l] = LeafAcc{}; lown[l] = LeafOwn{};
            ++cell_ver[cell]; ++n_prior_splits; ++splits;
            fprintf(prior_log, "%lld,%d,%d,%d,%s,%.6g,%.6g,%.6g,%.0f,%d,%d,%d\n", cur_t, l, cell, depth, desc_name(c.d).c_str(), c.thr,
                    acc_dir ? bt.dist : bt.gain, bt.z, c.n[0] + c.n[1], level_now, (int)leaves.size() - 2, (int)leaves.size() - 1);
        }
        return splits;
    }
    // ---- posterior phase ----
    const int B0 = B;
    for (int b = 0; b < B0; ++b) {
        if (users[b].size() < 2) continue;
        const int n = bucket_n[b];
        double tot_s[W] = {}, tot_q[W] = {}, tot_n = 0;
        for (int l : users[b]) { for (int a = 0; a < n; ++a) { tot_s[a] += lown[l].s[a]; tot_q[a] += lown[l].q[a]; } tot_n += lown[l].n; }
        bool diverging = false;
        for (int l : users[b]) {
            AccCand x;  // side 0: this leaf, side 1: the bucket's other leaves
            for (int a = 0; a < n; ++a) {
                x.s[0][a] = lown[l].s[a]; x.q[0][a] = lown[l].q[a];
                x.s[1][a] = tot_s[a] - lown[l].s[a]; x.q[1][a] = tot_q[a] - lown[l].q[a];
            }
            x.n[0] = lown[l].n; x.n[1] = tot_n - lown[l].n;
            if (acc_test(x, n).pass) { diverging = true; ++n_diverging; }
        }
        if (!diverging) continue;
        map<int, vector<int>> groups;  // label (-1: stays) -> leaves
        for (int l : users[b]) { int am = -1; const bool sig = own_significant(lown[l], n, am); groups[sig ? am : -1].push_back(l); }
        if (groups.size() < 2) continue;  // every leaf wants the same action: nothing to separate
        string labs, kids;
        bool first = true;
        for (auto& [lab, ls] : groups) {
            int nb = b;
            if (!first) {  // the first group (the staying one when there is one) keeps the bucket's id
                nb = B++;
                Q.resize((size_t)B * W, 0); S.resize((size_t)B * W, 0); Rtmp.resize((size_t)B * W, 0);
                bucket_n.push_back(bucket_n[b]); vis.push_back(0); parent_eps.push_back(-1); members.emplace_back(); dirty.push_back(0);
                for (int a = 0; a < W; ++a) { Q[(size_t)nb * W + a] = inherit_r * Q[(size_t)b * W + a]; S[(size_t)nb * W + a] = inherit_s * S[(size_t)b * W + a]; }
            }
            for (int l : ls) leaves[l].ptr = nb;
            labs += (first ? "" : " ") + to_string(lab); kids += (first ? "" : " ") + to_string(nb);
            first = false;
        }
        for (int a = 0; a < W; ++a) { Q[(size_t)b * W + a] *= inherit_r; S[(size_t)b * W + a] *= inherit_s; }
        for (int l : users[b]) { lown[l] = LeafOwn{}; lacc[l].cand.clear(); acc_fill(l); }
        fprintf(evf_global, "%d,%lld,%d,%zu,,%s,%s\n", level_now, cur_t, b, users[b].size(), labs.c_str(), kids.c_str());
        ++splits; ++n_post_splits;
    }
    if (splits) { ++level_now; fflush(evf_global); }
    return splits;
}
bool try_split_now(int l, int lv, int bs) {  // immediate mode: one binary test
    vector<Row> rows = leaf_rows(l, lv, bs);
    vector<char> half(rows.size(), 0);
    if (crossfit) for (auto& h : half) h = (char)(split_rng() & 1);
    const Candidate cd = best_split(rows, half, counts[cell_members[leaves[l].cell][0]]);
    if (!cd.ok) { ++n_prior_rejected; return false; }
    commit_split(l, cd, lv, rows);
    return true;
}
// epoch mode (and held signals at an epoch end): best first over the waiting leaves and their descendants
void process_waiting(bool best_first) {
    if (waiting.empty()) return;
    const int cap = max(2 * n_leaves_alive(), (int)min<double>(nI, ceil(prior_ratio * B)));
    if (!best_first) {  // immediate mode's held signals: one binary test each, in order, up to the cap
        vector<int> w = move(waiting); waiting.clear();
        for (int l : w) {
            if (leaves[l].node < 0 || leaves[l].sig_level < 0) continue;
            if (n_leaves_alive() + 1 > cap) { waiting.push_back(l); continue; }
            const int lv = leaves[l].sig_level, bs = leaves[l].sig_bucket;
            leaves[l].sig_level = -1;
            try_split_now(l, lv, bs);
        }
        return;
    }
    struct Item { double gain; int l, lv, bs; vector<Row> rows; vector<char> half; Candidate cd; };
    vector<Item> items;
    auto make = [&](int l, int lv, int bs, vector<Row> rows) {
        Item it; it.l = l; it.lv = lv; it.bs = bs; it.rows = move(rows);
        it.half.assign(it.rows.size(), 0); if (crossfit) for (auto& h : it.half) h = (char)(split_rng() & 1);
        it.cd = best_split(it.rows, it.half, counts[cell_members[leaves[l].cell][0]]);
        it.gain = it.cd.ok ? it.cd.gainB : -1;
        if (!it.cd.ok) ++n_prior_rejected;
        else items.push_back(move(it));
    };
    vector<int> w = move(waiting); waiting.clear();
    for (int l : w) {
        if (leaves[l].node < 0 || leaves[l].sig_level < 0) continue;
        const int lv = leaves[l].sig_level, bs = leaves[l].sig_bucket;
        leaves[l].sig_level = -1;
        make(l, lv, bs, leaf_rows(l, lv, bs));
    }
    while (!items.empty() && n_leaves_alive() + 1 <= cap) {
        size_t k = 0;
        for (size_t i = 1; i < items.size(); ++i) if (items[i].gain > items[k].gain) k = i;
        Item it = move(items[k]); items.erase(items.begin() + k);
        commit_split(it.l, it.cd, it.lv, it.rows);
        const int a = (int)leaves.size() - 2, b = (int)leaves.size() - 1;
        vector<Row> ra, rb;
        for (const Row& r : it.rows) (feat(r.I, it.cd.d) <= it.cd.thr ? ra : rb).push_back(r);
        make(a, it.lv, it.bs, move(ra));
        make(b, it.lv, it.bs, move(rb));
    }
}
// --prior-ratio R: after a posterior judgment, grow the prior best first until it has at least R x (posterior buckets)
// leaves (at most one per information set). Rows: every leaf's information sets with their per-visit directions at
// the judgment just made, under the leaf's bucket of that level; the same criterion as a signalled split.
void grow_to_ratio() {
    if (prior_ratio <= 0 || events.empty()) return;
    const int target = (int)min<double>(nI, ceil(prior_ratio * B));
    if (n_leaves_alive() >= target) return;
    const int lv = level_now - 1;
    vector<vector<int>> by_leaf(leaves.size());
    for (int I = 0; I < nI; ++I) by_leaf[leaf_of(I)].push_back(I);
    struct Item { double gain; int l; vector<Row> rows; Candidate cd; };
    vector<Item> items;
    auto make = [&](int l, vector<Row> rows) {
        Item it; it.l = l; it.rows = move(rows);
        vector<char> half(it.rows.size(), 0);
        if (crossfit) for (auto& h : half) h = (char)(split_rng() & 1);
        it.cd = best_split(it.rows, half, counts[cell_members[leaves[l].cell][0]]);
        it.gain = it.cd.gainB;
        if (it.cd.ok) items.push_back(move(it));
    };
    for (size_t l = 0; l < leaves.size(); ++l) {
        if (leaves[l].node < 0 || by_leaf[l].size() < 2) continue;
        const int bs = ancestor(leaves[l].pending ? leaves[l].prov : leaves[l].ptr, lv);
        vector<Row> rows;
        for (int I : by_leaf[l]) { Row r; if (fill_row(r, I, bs, lv)) rows.push_back(r); }
        make((int)l, move(rows));
    }
    while (!items.empty() && n_leaves_alive() < target) {
        size_t k = 0;
        for (size_t i = 1; i < items.size(); ++i) if (items[i].gain > items[k].gain) k = i;
        Item it = move(items[k]); items.erase(items.begin() + k);
        commit_split(it.l, it.cd, lv, it.rows);
        const int a = (int)leaves.size() - 2, b = (int)leaves.size() - 1;
        vector<Row> ra, rb;
        for (const Row& r : it.rows) (feat(r.I, it.cd.d) <= it.cd.thr ? ra : rb).push_back(r);
        make(a, move(ra));
        make(b, move(rb));
    }
}
// a pending leaf with m samples: its pointer = the route of the samples' mean direction; signal when the samples'
// own labels disagree at some event
void finalize(int l) {
    Leaf& L0 = leaves[l];
    if (!L0.pending || (int)L0.samples.size() < m_samples) return;
    ++n_finalize;
    map<int, int> mult;
    for (int I : L0.samples) ++mult[I];
    int b = L0.start_b, sig_lv = -1, sig_b = -1;
    for (int lv = L0.start_level; lv < level_now; ++lv) {
        const Event& ev = events[lv];
        if (ev.unsplit[b] >= 0) { b = ev.unsplit[b]; continue; }
        const Split& sp = ev.splits[ev.split_idx[b]];
        double mean[W] = {}, meanM = 0, mean2[W] = {};
        vector<double> per, meanper((size_t)ev.K * W, 0.0);
        set<int> labs;
        int n = 0;
        for (const auto& [I, k] : mult) {
            double R[W], M, R2[W];
            direction(I, b, view(ev), lv, R, M, R2, &per);
            n = counts[I];
            for (int a = 0; a < W; ++a) mean[a] += k * R[a] / m_samples;
            meanM += k * M / m_samples;
            for (size_t q = 0; q < meanper.size(); ++q) meanper[q] += k * per[q] / m_samples;
            labs.insert(kid_of_label(sp, label_of(sp, R, R2, M, counts[I], true)));
        }
        for (int j = 0; j < ev.K; ++j) for (int a = 0; a < W; ++a) mean2[a] += meanper[(size_t)j * W + a] * meanper[(size_t)j * W + a];
        if (labs.size() >= 2 && sig_lv < 0) { sig_lv = lv; sig_b = b; }
        b = kid_of_label(sp, label_of(sp, mean, mean2, meanM, n, true));
    }
    Leaf& L = leaves[l];
    L.pending = false; L.ptr = b; L.samples.clear();
    ++assign_ver;
    if (sig_lv >= 0) {
        ++n_signals;
        L.sig_level = sig_lv; L.sig_bucket = sig_b;
        if (prior_split == "immediate" && n_leaves_alive() + 1 <= max(2 * leaves_at_event, (int)min<double>(nI, ceil(prior_ratio * B)))) {
            L.sig_level = -1;
            try_split_now(l, sig_lv, sig_b);
        } else waiting.push_back(l);
    }
}

// ======================= ESD judgment =======================
// one-sided generalized ESD on x: the number of outliers (they are the largest values)
int esd_count(vector<double> x, int nact) {
    const int n = (int)x.size();
    if (n < esd_min_n || n < 5) return 0;
    const int nn = min(n, esd_nmax);
    auto it = esd_lambda.find({nn, max(1, min(4, nact))});
    if (it == esd_lambda.end()) throw runtime_error("ESD table has no row for n=" + to_string(nn));
    const vector<double>& lam = it->second;
    const int r = min((int)lam.size(), n / 2);
    sort(x.begin(), x.end());
    int found = 0;
    for (int i = 1; i <= r; ++i) {
        const int m = n - i + 1;  // remaining before removing the i-th
        double mean = 0; for (int k = 0; k < m; ++k) mean += x[k]; mean /= m;
        double var = 0; for (int k = 0; k < m; ++k) var += (x[k] - mean) * (x[k] - mean); var /= (m - 1);
        if (!(var > 0)) break;
        const double Ri = (x[m - 1] - mean) / sqrt(var);
        if (Ri > lam[i - 1]) found = i;
    }
    return found;
}
// judge bucket b from its members' directions; returns the split (kids empty when it does not split)
Split judge(int b, const vector<int>& mem, const TabView& tv, map<int, vector<double>>& dirs) {
    Split sp; sp.b = b;
    const int n = bucket_n[b], cntm = (int)mem.size();
    sp.n = cntm;
    if (cntm < 2) return sp;
    vector<array<double, W>> R(cntm), R2(cntm);
    vector<double> Mv(cntm);
    for (int i = 0; i < cntm; ++i) {
        direction(mem[i], b, tv, level_now, R[i].data(), Mv[i], R2[i].data());
        dirs[mem[i]] = vector<double>(R[i].begin(), R[i].end());
    }
    if (outlier == "label") {
        for (int a = 0; a < W; ++a) { sp.mu[a] = 0; sp.sd[a] = 1; sp.c[a] = 1e300; }
        set<int> labs;
        for (int i = 0; i < cntm; ++i) { const int l = label_of(sp, R[i].data(), R2[i].data(), Mv[i], n, false); labs.insert(l); if (l >= 0) ++sp.outliers[l]; }
        if (labs.size() < 2) return sp;
        for (int lab : labs) if (lab >= 0) sp.kids[lab] = -1;
        return sp;
    }
    const bool pv = scale == "fact-counterfact";
    vector<int> tested;  // fact-counterfact: members with positive reach are tested (per-visit values)
    for (int i = 0; i < cntm; ++i) if (!pv || Mv[i] > 0) tested.push_back(i);
    for (int a = 0; a < W; ++a) { sp.mu[a] = 0; sp.sd[a] = 1; sp.c[a] = 1e300; }
    if ((int)tested.size() < 2) return sp;
    const int cntt = (int)tested.size();
    for (int a = 0; a < n; ++a) {
        vector<double> x(cntt);
        for (int t = 0; t < cntt; ++t) { const int i = tested[t]; x[t] = pv ? R[i][a] / Mv[i] : R[i][a]; }
        if (outlier == "z") {
            double mean = 0; for (double v : x) mean += v; mean /= cntt;
            double var = 0; for (double v : x) var += (v - mean) * (v - mean); var /= max(1, cntt - 1);
            sp.mu[a] = mean; sp.sd[a] = max(sqrt(var), 1e-12 * max(1.0, fabs(mean)));
            sp.c[a] = var > 0 ? zthr : 1e300;
            int o = 0; for (double v : x) o += (v - mean) / sp.sd[a] >= zthr && var > 0; sp.outliers[a] = o;
            continue;
        }
        const int k = esd_count(x, n);
        sp.outliers[a] = k;
        vector<double> sx = x; sort(sx.begin(), sx.end());
        const int m = cntt - k;
        double mean = 0; for (int i = 0; i < m; ++i) mean += sx[i]; mean /= m;
        double var = 0; for (int i = 0; i < m; ++i) var += (sx[i] - mean) * (sx[i] - mean); var /= max(1, m - 1);
        sp.mu[a] = mean; sp.sd[a] = max(sqrt(var), 1e-12 * max(1.0, fabs(mean)));
        if (k > 0) sp.c[a] = (sx[m] - mean) / sp.sd[a];  // the smallest outlier's z
    }
    set<int> labs;
    vector<int> lab(cntm);
    for (int i = 0; i < cntm; ++i) labs.insert(lab[i] = label_of(sp, R[i].data(), R2[i].data(), Mv[i], n, false));
    if (labs.size() < 2) return sp;
    if (pv) {  // the counterfactual check: the split must reduce the cancellation among the members
        map<int, array<double, W>> g; array<double, W> tot{};
        double each = 0;
        for (int i = 0; i < cntm; ++i) {
            auto& gg = g[lab[i]];
            for (int a = 0; a < n; ++a) { gg[a] += R[i][a]; tot[a] += R[i][a]; }
            each += psi(R[i].data(), n);
        }
        double gain = -psi(tot.data(), n);
        for (auto& [l, v] : g) gain += psi(v.data(), n);
        sp.gain = gain;
        if (!(gain > 1e-9 * max(each, 1e-300))) { ++n_gain_rejected; return sp; }
    }
    for (int lab : labs) if (lab >= 0) sp.kids[lab] = -1;  // ids assigned by the caller
    return sp;
}

// ======================= checkpoint =======================
template <class T> void put(ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void get(istream& i, T& v) { i.read(reinterpret_cast<char*>(&v), sizeof(T)); }
template <class T> void putv(ostream& o, const vector<T>& v) { uint64_t n = v.size(); put(o, n); o.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T)); }
template <class T> void getv(istream& i, vector<T>& v) { uint64_t n; get(i, n); v.resize(n); i.read(reinterpret_cast<char*>(v.data()), n * sizeof(T)); }
void puts_(ostream& o, const string& s) { uint64_t n = s.size(); put(o, n); o.write(s.data(), n); }
void gets_(istream& i, string& s) { uint64_t n; get(i, n); s.assign(n, '\0'); i.read(s.data(), n); }
string rng_text(const mt19937_64& r) { ostringstream o; o << r; return o.str(); }
void rng_load(mt19937_64& r, const string& s) { istringstream i(s); i >> r; }

struct Run {
    long long t = 0, epoch_start = 0, next_row_j = 0; int k = 0, rows = 0;
    vector<long long> samples; size_t next_sample = 0; vector<double> cur_tables; int curK = 0, Kk = 0;
    long long res_count = 0;   // eps schedule: iterations offered to the iterate reservoir
    double eps = -1, cpu_before = 0, assign_before = 0;
};

int main(int argc, char** argv) try {
    map<string, string> opt{{"--mode", "pp"}, {"--iters", "1000000000000"}, {"--seconds", "0"}, {"--seed", "1"},
                            {"--members", "64"}, {"--m", "16"}, {"--min-leaf", "1"}, {"--kmax", to_string(KMAX)},
                            {"--outlier", "esd"}, {"--scale", "counterfactual"}, {"--z", "2"}, {"--esd-min-n", "5"}, {"--schedule", "static"},
                            {"--eps-check", "1000"}, {"--prior-split", "immediate"}, {"--prior-start", "equal"},
                            {"--features", "general"}, {"--inherit-regret", "0"}, {"--inherit-avg", "0"}, {"--measure", "0"},
                            {"--eager", "0"}, {"--leaf-cache", "1"}, {"--pend-cache", "1"}, {"--memo", "1"}, {"--table", "1"},
                            {"--rows-per-octave", "2"}, {"--checkpoint-minutes", "10"}, {"--stop-after-rows", "0"},
                            {"--esd-table", (filesystem::path(__FILE__).parent_path() / "esd_table.csv").string()},
                            {"--p0", ""}, {"--split", "1"}, {"--analyze", "0"}, {"--prior-ratio", "0"}, {"--crossfit", "1"}, {"--resolve-now", "0"}, {"--prior-criterion", "regret"},
                            {"--epoch0", "1000"}, {"--eps-at", "1000"},
                            {"--direction", "iterates"}, {"--leaf-start", "root"}, {"--tol", "1e-3"},
                            {"--acc-candidates", "8"}, {"--acc-min-visits", "100"}, {"--acc-z", "3"}, {"--acc-pool", "16"},
                            {"--acc-metric", "mse"}, {"--acc-dist", "0.5"}};
    for (int i = 1; i + 1 < argc; i += 2) {
        const string k = argv[i];
        if (!opt.count(k) && k != "--tree" && k != "--out") throw runtime_error("unknown option " + k);
        opt[k] = argv[i + 1];
    }
    const string dir = opt.at("--tree"), out = opt.at("--out");
    pp_mode = opt["--mode"] == "pp";
    tree_mode = opt["--mode"] == "tree";
    ppacc_mode = opt["--mode"] == "ppacc";
    acc_k = stoi(opt["--acc-candidates"]); acc_min = stoi(opt["--acc-min-visits"]); acc_z = stod(opt["--acc-z"]);
    acc_pool = stoi(opt["--acc-pool"]); acc_dir = opt["--acc-metric"] == "dir"; acc_dist = stod(opt["--acc-dist"]);
    if (opt["--acc-metric"] != "mse" && !acc_dir) throw runtime_error("--acc-metric must be mse or dir");
    if (acc_k < 1 || acc_min < 2 || acc_pool < 2) throw runtime_error("--acc-candidates >= 1, --acc-min-visits >= 2, --acc-pool >= 2");
    split_on = opt["--split"] == "1";
    prior_ratio = stod(opt["--prior-ratio"]); crossfit = opt["--crossfit"] == "1";
    resolve_now = opt["--resolve-now"] == "1"; prior_criterion = opt["--prior-criterion"];
    direction_avg = opt["--direction"] == "avg"; leaf_start_parent = opt["--leaf-start"] == "parent"; tol = stod(opt["--tol"]);
    if (opt["--direction"] != "iterates" && !direction_avg) throw runtime_error("--direction must be iterates or avg");
    if (opt["--leaf-start"] != "root" && !leaf_start_parent) throw runtime_error("--leaf-start must be root or parent");
    if (leaf_start_parent && !pp_mode) throw runtime_error("--leaf-start parent is for --mode pp");
    if (prior_criterion != "regret" && prior_criterion != "label") throw runtime_error("--prior-criterion must be regret or label");
    if (!pp_mode && !tree_mode && !ppacc_mode && opt["--mode"] != "pure") throw runtime_error("--mode must be pp, pure, tree or ppacc");
    members_cap = stoi(opt["--members"]); m_samples = stoi(opt["--m"]); min_leaf = stoi(opt["--min-leaf"]);
    kmax = min(KMAX, stoi(opt["--kmax"])); outlier = opt["--outlier"]; scale = opt["--scale"]; zthr = stod(opt["--z"]); esd_min_n = stoi(opt["--esd-min-n"]);
    schedule = opt["--schedule"]; eps_check = stoll(opt["--eps-check"]); prior_split = opt["--prior-split"];
    prior_start = opt["--prior-start"]; features = opt["--features"];
    inherit_r = stod(opt["--inherit-regret"]); inherit_s = stod(opt["--inherit-avg"]);
    measure = opt["--measure"] == "1"; eager = opt["--eager"] == "1";
    leaf_cache_on = opt["--leaf-cache"] == "1"; pend_cache_on = opt["--pend-cache"] == "1";
    memo_on = opt["--memo"] == "1"; table_on = opt["--table"] == "1"; rows_per_octave = stoi(opt["--rows-per-octave"]);
    if (outlier != "esd" && outlier != "z" && outlier != "label") throw runtime_error("--outlier must be esd, z or label");
    if ((tree_mode || ppacc_mode) && (opt["--prior-start"] != "equal" || opt["--schedule"] != "static" || !opt["--p0"].empty()))
        throw runtime_error("--mode tree and ppacc need --prior-start equal, --schedule static and no --p0");
    if (scale != "counterfactual" && scale != "fact-counterfact") throw runtime_error("--scale must be counterfactual or fact-counterfact");
    if (schedule != "static" && schedule != "eps") throw runtime_error("--schedule must be static or eps");
    if (prior_split != "immediate" && prior_split != "epoch") throw runtime_error("--prior-split must be immediate or epoch");
    if (prior_start != "equal" && prior_start != "length" && prior_start != "identity") throw runtime_error("bad --prior-start");
    if (features != "general" && features != "poker") throw runtime_error("--features must be general or poker");
    if (outlier == "esd" && !tree_mode && !ppacc_mode) {  // the given path, else next to the source, the executable, or in the working directory
        string path = opt["--esd-table"];
        for (const string& c : {path, (filesystem::path(__FILE__).parent_path() / "esd_table.csv").string(),
                                (filesystem::path(argv[0]).parent_path() / "esd_table.csv").string(), string("esd_table.csv")})
            if (!c.empty() && filesystem::exists(c)) { path = c; break; }
        load_esd(path);
    }
    const double budget = stod(opt["--seconds"]), ckpt_every = 60.0 * stod(opt["--checkpoint-minutes"]);
    long long T = stoll(opt["--iters"]);
    const int stop_after_rows = stoi(opt["--stop-after-rows"]);

    players = load<int32_t>(dir + "/players.bin"); info = load<int32_t>(dir + "/info.bin");
    start = load<int64_t>(dir + "/start.bin"); degree = load<int32_t>(dir + "/degree.bin");
    edges = load<int32_t>(dir + "/edges.bin"); prob = load<double>(dir + "/prob.bin");
    util = load<double>(dir + "/util.bin"); counts = load<int32_t>(dir + "/counts.bin");
    constraint_of = load<int32_t>(dir + "/constraint.bin"); seat_of = load<int32_t>(dir + "/seat.bin");
    obs = load<int32_t>(dir + "/obs.bin"); obs_start = load<int64_t>(dir + "/obs_start.bin");
    nI = (int)counts.size(); N = (int)players.size();
    if ((int)constraint_of.size() != nI || (int)obs_start.size() != nI + 1) throw runtime_error("export_obs.py files do not match the tree");
    {   // card symbols from obs_meta.json (a list of ints after "card_symbols")
        ifstream mf(dir + "/obs_meta.json"); string s((istreambuf_iterator<char>(mf)), {});
        const size_t p = s.find("\"card_symbols\"");
        if (p != string::npos) { size_t a = s.find('[', p), b = s.find(']', a); stringstream ss(s.substr(a + 1, b - a - 1)); string f; while (getline(ss, f, ',')) card_syms.push_back(stoi(f)); }
        const size_t q = s.find("\"symbols\"");
        if (q != string::npos) {
            size_t a = s.find('[', q), b = s.find(']', a);
            for (size_t x = s.find('"', a); x != string::npos && x < b; x = s.find('"', x + 1)) { const size_t y = s.find('"', x + 1); sym_names.push_back(s.substr(x + 1, y - x - 1)); x = y; }
        }
    }
    for (int c : constraint_of) n_constraints = max(n_constraints, c + 1);
    parent_node.assign(N, -1); parent_slot.assign(N, -1); nodes_of.assign(nI, {});
    for (int h = 0; h < N; ++h) {
        if (players[h] == TERMINAL) continue;
        for (int k = 0; k < degree[h]; ++k) { parent_node[edges[start[h] + k]] = h; parent_slot[edges[start[h] + k]] = k; }
        if (players[h] != CHANCE) nodes_of[info[h]].push_back(h);
    }
    p0_bucket = constraint_of;
    if (!opt["--p0"].empty()) {  // a fixed starting partition (with --split 0: plain MCCFR on it, e.g. the k-means baseline)
        if (pp_mode) throw runtime_error("--p0 is for --mode pure");
        p0_bucket = load<int32_t>(opt["--p0"]);
        if ((int)p0_bucket.size() != nI) throw runtime_error("--p0 size differs from the information sets");
    }
    // prior cells
    cell_of_I.assign(nI, -1);
    {
        map<pair<int, int>, int> ids;
        for (int I = 0; I < nI; ++I) {
            int key2 = 0;
            if (prior_start == "length") { for (int64_t k = obs_start[I]; k < obs_start[I + 1]; ++k) key2 += chance_ch(o_ch(obs[k])); }
            if (prior_start == "identity") key2 = I;
            auto it = ids.find({constraint_of[I], key2});
            if (it == ids.end()) { it = ids.emplace(make_pair(constraint_of[I], key2), (int)ids.size()).first; cell_constraint.push_back(constraint_of[I]); }
            cell_of_I[I] = it->second;
        }
        cell_members.assign(cell_constraint.size(), {});
        for (int I = 0; I < nI; ++I) cell_members[cell_of_I[I]].push_back(I);
    }
    leaf_cache.assign(nI, -1); leaf_cache_ver.assign(nI, -1);
    pend_cache.assign(nI, -1); pend_cache_b.assign(nI, -1); pend_cache_ver.assign(nI, 0);

    ostringstream cfg;
    cfg << "pp_aswiss v1;tree=" << dir;
    for (const auto& [k, v] : opt)
        if (k != "--seconds" && k != "--iters" && k != "--checkpoint-minutes" && k != "--leaf-cache" && k != "--pend-cache" &&
            k != "--memo" && k != "--table" && k != "--stop-after-rows" && k != "--esd-table" && k != "--analyze" &&
            !(k == "--prior-ratio" && v == "0") && !(k == "--crossfit" && v == "1") && !(k == "--resolve-now" && v == "0") &&
            !(k == "--prior-criterion" && v == "regret") && !(k == "--epoch0" && v == "1000") && !(k == "--eps-at" && v == "1000") &&
            !(k == "--direction" && v == "iterates") && !(k == "--leaf-start" && v == "root") && !(k == "--tol" && v == "1e-3") &&
            !(k == "--acc-candidates" && v == "8") && !(k == "--acc-min-visits" && v == "100") && !(k == "--acc-z" && v == "3") &&
            !(k == "--acc-pool" && v == "16") && !(k == "--acc-metric" && v == "mse") && !(k == "--acc-dist" && v == "0.5"))  // defaults of later options: old runs resume
            cfg << ";" << k << "=" << v;
    filesystem::create_directories(out);
    const string state_path = out + "/state.bin", log_path = out + "/log.csv", ev_path = out + "/events.csv",
                 pr_path = out + "/prior_splits.csv", ds_path = out + "/depth_spread.csv";
    Run run;
    double excluded = 0;
    const double cpu0 = cpu_now();
    auto train_secs = [&] { return run.cpu_before + (cpu_now() - cpu0) - excluded; };
    auto excl = [&](auto&& f) { const double x0 = cpu_now(); f(); excluded += cpu_now() - x0; };
    // --epoch0 E: static judgments at E * 2^k; --eps-at A: the eps rule's first judgment (and eps) at iteration A
    const long long epoch0 = stoll(opt["--epoch0"]), eps_at = stoll(opt["--eps-at"]);
    if (epoch0 < 1 || eps_at < 1) throw runtime_error("--epoch0 and --eps-at must be positive");
    auto epoch_end_static = [&](int k) { return epoch0 << k; };
    auto Kfor = [&](int k) { return min(kmax, 8 * (k + 1)); };
    auto draw_samples = [&] {  // static schedule: K iterations uniformly among the epoch's (distinct, sorted)
        run.Kk = Kfor(run.k); run.curK = 0; run.cur_tables.clear(); run.next_sample = 0; run.samples.clear(); run.res_count = 0;
        if (schedule != "static") return;
        const long long a = run.epoch_start, b = epoch_end_static(run.k), Lk = b - a;
        const int K = (int)min<long long>(Lk, run.Kk);
        set<long long> pick;
        while ((int)pick.size() < K) pick.insert(a + 1 + uniform_int_distribution<long long>(0, Lk - 1)(iter_rng));
        run.samples.assign(pick.begin(), pick.end());
    };
    uint64_t log_bytes = 0, ev_bytes = 0, pr_bytes = 0, ds_bytes = 0;
    auto fsize = [](const string& p) -> uint64_t { return filesystem::exists(p) ? filesystem::file_size(p) : 0; };
    auto save_state = [&] {
        const string tmp = state_path + ".tmp";
        ofstream o(tmp, ios::binary);
        puts_(o, cfg.str());
        put(o, run.t); put(o, run.epoch_start); put(o, run.next_row_j); put(o, run.k); put(o, run.rows);
        putv(o, run.samples); put(o, run.next_sample); putv(o, run.cur_tables); put(o, run.curK); put(o, run.Kk); put(o, run.res_count);
        put(o, run.eps);
        const double ts = train_secs(), as = assign_cpu; put(o, ts); put(o, as);
        put(o, B); put(o, level_now); putv(o, Q); putv(o, S); putv(o, bucket_n); putv(o, vis); putv(o, parent_eps);
        uint64_t ne = events.size(); put(o, ne);
        for (const Event& ev : events) {
            putv(o, ev.unsplit); putv(o, ev.parent); putv(o, ev.split_idx); put(o, ev.K); put(o, ev.Bl); putv(o, ev.tables); put(o, ev.t);
            uint64_t ns = ev.splits.size(); put(o, ns);
            for (const Split& sp : ev.splits) {
                put(o, sp.b); put(o, sp.mu); put(o, sp.sd); put(o, sp.c); put(o, sp.none_kid); put(o, sp.outliers); put(o, sp.n);
                uint64_t nk = sp.kids.size(); put(o, nk); for (const auto& [l, id] : sp.kids) { put(o, l); put(o, id); }
            }
        }
        uint64_t nm = members.size(); put(o, nm); for (const auto& m : members) putv(o, m);
        put(o, member_salt);
        putv(o, tnodes); putv(o, cell_root); putv(o, cell_ver);
        uint64_t nl = leaves.size(); put(o, nl);
        for (const Leaf& L : leaves) {
            put(o, L.node); put(o, L.cell); put(o, L.depth); put(o, L.pending); put(o, L.ptr); put(o, L.prov);
            put(o, L.start_level); put(o, L.start_b); putv(o, L.samples); put(o, L.born); put(o, L.sig_level); put(o, L.sig_bucket);
        }
        putv(o, waiting); put(o, leaves_at_event); put(o, assign_ver);
        put(o, n_finalize); put(o, n_signals); put(o, n_prior_splits); put(o, n_prior_rejected); put(o, n_gain_rejected);
        putv(o, Rmeas); putv(o, Rwin); putv(o, vmeas); putv(o, vwin);
        puts_(o, rng_text(rng)); puts_(o, rng_text(iter_rng)); puts_(o, rng_text(split_rng));
        put(o, cnt);
        if (tree_mode || ppacc_mode) {
            put(o, n_acc_tested);
            uint64_t na = lacc.size(); put(o, na);
            for (const LeafAcc& L : lacc) { putv(o, L.cand); putv(o, L.pool); put(o, L.seen); put(o, L.gen); }
        }
        if (ppacc_mode) { putv(o, lown); put(o, n_post_splits); put(o, n_diverging); }
        fflush(prior_log);
        log_bytes = fsize(log_path); ev_bytes = fsize(ev_path); pr_bytes = fsize(pr_path); ds_bytes = fsize(ds_path);
        put(o, log_bytes); put(o, ev_bytes); put(o, pr_bytes); put(o, ds_bytes);
        o.close();
        filesystem::rename(tmp, state_path);
    };
    tree_salt = mix64(stoull(opt["--seed"]) ^ 0x7ee5a17c0ffeeull);
    bool resumed = false;
    if (filesystem::exists(state_path)) {
        ifstream in(state_path, ios::binary);
        string c; gets_(in, c);
        if (c != cfg.str()) throw runtime_error("state.bin is another run's (" + c + ")");
        get(in, run.t); get(in, run.epoch_start); get(in, run.next_row_j); get(in, run.k); get(in, run.rows);
        getv(in, run.samples); get(in, run.next_sample); getv(in, run.cur_tables); get(in, run.curK); get(in, run.Kk); get(in, run.res_count);
        get(in, run.eps);
        get(in, run.cpu_before); get(in, run.assign_before); assign_cpu = run.assign_before;
        get(in, B); get(in, level_now); getv(in, Q); getv(in, S); getv(in, bucket_n); getv(in, vis); getv(in, parent_eps);
        uint64_t ne; get(in, ne); events.resize(ne);
        for (Event& ev : events) {
            getv(in, ev.unsplit); getv(in, ev.parent); getv(in, ev.split_idx); get(in, ev.K); get(in, ev.Bl); getv(in, ev.tables); get(in, ev.t);
            uint64_t ns; get(in, ns); ev.splits.resize(ns);
            for (Split& sp : ev.splits) {
                get(in, sp.b); get(in, sp.mu); get(in, sp.sd); get(in, sp.c); get(in, sp.none_kid); get(in, sp.outliers); get(in, sp.n);
                uint64_t nk; get(in, nk); for (uint64_t i = 0; i < nk; ++i) { int l, id; get(in, l); get(in, id); sp.kids[l] = id; }
            }
        }
        uint64_t nm; get(in, nm); members.resize(nm); for (auto& m : members) getv(in, m);
        get(in, member_salt);
        getv(in, tnodes); getv(in, cell_root); getv(in, cell_ver);
        uint64_t nl; get(in, nl); leaves.resize(nl);
        for (Leaf& L : leaves) {
            get(in, L.node); get(in, L.cell); get(in, L.depth); get(in, L.pending); get(in, L.ptr); get(in, L.prov);
            get(in, L.start_level); get(in, L.start_b); getv(in, L.samples); get(in, L.born); get(in, L.sig_level); get(in, L.sig_bucket);
        }
        getv(in, waiting); get(in, leaves_at_event); get(in, assign_ver);
        get(in, n_finalize); get(in, n_signals); get(in, n_prior_splits); get(in, n_prior_rejected); get(in, n_gain_rejected);
        getv(in, Rmeas); getv(in, Rwin); getv(in, vmeas); getv(in, vwin);
        string r1, r2, r3; gets_(in, r1); gets_(in, r2); gets_(in, r3);
        rng_load(rng, r1); rng_load(iter_rng, r2); rng_load(split_rng, r3);
        get(in, cnt);
        if (tree_mode || ppacc_mode) {
            get(in, n_acc_tested);
            uint64_t na; get(in, na); lacc.assign(na, {});
            for (LeafAcc& L : lacc) { getv(in, L.cand); getv(in, L.pool); get(in, L.seen); get(in, L.gen); }
        }
        if (ppacc_mode) { getv(in, lown); get(in, n_post_splits); get(in, n_diverging); }
        get(in, log_bytes); get(in, ev_bytes); get(in, pr_bytes); get(in, ds_bytes);
        if (!in) throw runtime_error("truncated state.bin");
        filesystem::resize_file(log_path, log_bytes); filesystem::resize_file(ev_path, ev_bytes);
        filesystem::resize_file(pr_path, pr_bytes);
        if (filesystem::exists(ds_path)) filesystem::resize_file(ds_path, ds_bytes);
        resumed = true;
        if (stoi(opt["--analyze"]) > 0) {
            // --analyze N (pp): does the prior send information sets to the bucket their own direction would choose?
            // For N information sets drawn uniformly among those whose leaf is resolved: own = the route of I alone
            // through the genealogy from its starting bucket (the pending-leaf rule applied to I), pointer = its
            // leaf's pointer. Writes analysis.json; nothing is trained and no output of the run changes.
            if (!pp_mode) throw runtime_error("--analyze is for --mode pp");
            if (leaf_start_parent) throw runtime_error("--analyze routes from the start of the genealogy, whose tables --leaf-start parent frees");
            const int nsamp = stoi(opt["--analyze"]);
            vector<int> pool;
            for (int I = 0; I < nI; ++I) if (!leaves[leaf_of(I)].pending) pool.push_back(I);
            mt19937_64 ar(12345);
            shuffle(pool.begin(), pool.end(), ar);
            if ((int)pool.size() > nsamp) pool.resize(nsamp);
            long long mis = 0; double wv = 0, wmis = 0; map<int, pair<long long, long long>> by_depth; map<int, long long> first_level;
            const double t0 = cpu_now();
            for (int I : pool) {
                const Leaf& L = leaves[leaf_of(I)];
                // own route level by level, compared with the pointer's ancestors
                int b = p0_bucket[I], diverge = -1;
                for (int lv = 0; lv < level_now; ++lv) {
                    if (diverge < 0 && b != ancestor(L.ptr, lv)) diverge = lv;
                    const Event& ev = events[lv];
                    b = ev.unsplit[b] >= 0 ? ev.unsplit[b] : kid_for(ev, b, I, lv);
                }
                if (diverge < 0 && b != L.ptr) diverge = level_now;
                const bool wrong = b != L.ptr;
                const double w = vmeas.empty() ? 1.0 : (double)vmeas[I];
                mis += wrong; wv += w; wmis += wrong ? w : 0;
                auto& d = by_depth[L.depth]; ++d.first; d.second += wrong;
                if (wrong) ++first_level[diverge];
            }
            ofstream j(out + "/analysis.json");
            j << "{\"iters\":" << run.t << ",\"level\":" << level_now << ",\"buckets\":" << B << ",\"sampled\":" << pool.size()
              << ",\"misassigned\":" << mis << ",\"share\":" << (pool.empty() ? 0.0 : (double)mis / pool.size())
              << ",\"visit_weighted_share\":" << (wv > 0 ? wmis / wv : NAN) << ",\"by_leaf_depth\":{";
            bool first = true;
            for (const auto& [dd, c] : by_depth) { j << (first ? "" : ",") << "\"" << dd << "\":[" << c.first << "," << c.second << "]"; first = false; }
            j << "},\"first_divergent_level\":{";
            first = true;
            for (const auto& [lv, c] : first_level) { j << (first ? "" : ",") << "\"" << lv << "\":" << c; first = false; }
            j << "},\"seconds\":" << cpu_now() - t0 << "}\n";
            cerr << "analysis: " << mis << " of " << pool.size() << " sampled information sets are not where their own route goes ("
                 << (pool.empty() ? 0.0 : 100.0 * mis / pool.size()) << "%, visit-weighted " << (wv > 0 ? 100.0 * wmis / wv : NAN) << "%)\n";
            return 0;
        }
        cerr << "resumed at iteration " << run.t << ", level " << level_now << ", buckets " << B << ", CPU " << run.cpu_before << " s\n";
    } else {
        const uint64_t seed = stoull(opt["--seed"]);
        rng.seed(seed); iter_rng.seed(seed ^ 0xbf58476d1ce4e5b9ull); split_rng.seed(seed ^ 0x94d049bb133111ebull);
        member_salt = mix64(seed ^ 0x1234567ull);
        B = 0; for (int b : p0_bucket) B = max(B, b + 1);
        Q.assign((size_t)B * W, 0); S.assign((size_t)B * W, 0); bucket_n.assign(B, 0);
        for (int I = 0; I < nI; ++I) bucket_n[p0_bucket[I]] = counts[I];
        vis.assign(B, 0); parent_eps.assign(B, -1); members.assign(B, {});
        cell_ver.assign(cell_constraint.size(), 0);
        for (size_t c = 0; c < cell_constraint.size(); ++c) {
            TNode t; t.leaf = (int)leaves.size(); cell_root.push_back((int)tnodes.size()); tnodes.push_back(t);
            Leaf L; L.node = cell_root.back(); L.cell = (int)c; L.ptr = cell_constraint[c]; leaves.push_back(L);
        }
        leaves_at_event = (int)leaves.size();
        if (measure) { Rmeas.assign((size_t)nI * W, 0); Rwin.assign((size_t)nI * W, 0); vmeas.assign(nI, 0); vwin.assign(nI, 0); }
        run.epoch_start = 0; run.k = 0; run.next_row_j = -3LL * rows_per_octave;
        draw_samples();
        ofstream(log_path) << "row,iters,epoch,level,buckets,buckets_after,splits,prior_leaves,pending_leaves,prior_splits,"
                              "prior_rejected,finalized,signals,waiting,K,eps_stat,eps,infoset_regret_sum,bucket_regret_sum,"
                              "regret_sum_gap,table_regret_sum,train_seconds,assign_seconds,directions,gain_rejected,buckets_used,routing_mb,peak_rss_mb\n";
        ofstream(ev_path) << "event,iters,bucket,members,outliers,labels,children\n";
        ofstream(pr_path) << "iters,leaf,cell,depth,test,threshold,gain_choose_half,gain_check_half,rows,signal_level,left,right\n";
        if (measure && pp_mode) ofstream(ds_path) << "row,iters,leaf,cell,depth,infosets,visited,visits,rms,max_dev,pending\n";
    }
    prior_log = fopen(pr_path.c_str(), "a");
    Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
    auto rebuild_table = [&] {  // pure: every information set's bucket at level_now (a cache of route)
        if (pp_mode || tree_mode || ppacc_mode || !table_on) return;
        const double a0 = cpu_now();
        bucket_now.clear();
        vector<int32_t> bn(nI);
        for (int I = 0; I < nI; ++I) bn[I] = route(I, level_now);
        bucket_now = move(bn);
        assign_cpu += cpu_now() - a0;
    };
    rebuild_table();
    (void)resumed;
    FILE* logf = fopen(log_path.c_str(), "a");
    FILE* evf = fopen(ev_path.c_str(), "a");
    evf_global = evf;
    FILE* dsf = measure && pp_mode ? fopen(ds_path.c_str(), "a") : nullptr;
    auto current_bucket = [&](int I) {  // measurement and policies: where I is trained now (pending: its own route)
        if (tree_mode || ppacc_mode) return leaves[leaf_of(I)].ptr;
        if (!pp_mode) return table_on ? bucket_now[I] : route(I, level_now);
        const Leaf& L = leaves[leaf_of(I)];
        return L.pending ? route_from(I, L.start_level, L.start_b) : L.ptr;
    };
    auto eps_stat = [&] {
        double s = 0;
        for (int b = 0; b < B; ++b) s += vis[b] >= 25 || parent_eps[b] < 0 ? psi(&Q[(size_t)b * W], bucket_n[b]) / max<long long>(1, vis[b]) : parent_eps[b];
        return s;
    };
    struct RowStats { double irs = NAN, brs = NAN, trs = 0; int used = 0; };
    auto routing_mb = [&] { size_t b = run.cur_tables.size(); for (const Event& e : events) b += e.tables.size(); return b * 8.0 / 1e6; };
    auto peak_rss_mb = [] { ifstream f("/proc/self/status"); string k; while (f >> k) { if (k == "VmHWM:") { double v; f >> v; return v / 1024.0; } } return (double)NAN; };  // used: buckets some information set is in now
    auto snapshot = [&](int before) {
        RowStats st;
        excl([&] {
            vector<int> bk(nI);
            for (int I = 0; I < nI; ++I) bk[I] = current_bucket(I);
            { vector<char> u(B, 0); for (int b : bk) u[b] = 1; for (char c : u) st.used += c; }
            vector<double> avg((size_t)nI * W, 0.0), cur((size_t)nI * W, 0.0);
            for (int I = 0; I < nI; ++I) {
                const int b = bk[I], n = counts[I];
                double tot = 0;
                for (int a = 0; a < n; ++a) tot += S[(size_t)b * W + a];
                for (int a = 0; a < n; ++a) avg[(size_t)I * W + a] = tot > 0 ? S[(size_t)b * W + a] / tot : 1.0 / n;
                strategy(b, n, &cur[(size_t)I * W]);
            }
            save_vec(out + "/policy_bucket_" + to_string(run.rows) + ".bin", avg);
            save_vec(out + "/policy_current_" + to_string(run.rows) + ".bin", cur);
            double& irs = st.irs; double& brs = st.brs; double& trs = st.trs;
            for (int b = 0; b < before; ++b) trs += psi(&Q[(size_t)b * W], bucket_n[b]);
            trs /= max<long long>(1, run.t);
            if (measure) {
                irs = 0; brs = 0;
                vector<double> sumB((size_t)B * W, 0.0);
                for (int I = 0; I < nI; ++I) {
                    irs += psi(&Rmeas[(size_t)I * W], counts[I]);
                    for (int a = 0; a < counts[I]; ++a) sumB[(size_t)bk[I] * W + a] += Rmeas[(size_t)I * W + a];
                }
                for (int b = 0; b < B; ++b) brs += psi(&sumB[(size_t)b * W], bucket_n[b]);
                irs /= max<long long>(1, run.t); brs /= max<long long>(1, run.t);
                if (dsf) {
                    vector<vector<int>> byleaf(leaves.size());
                    for (int I = 0; I < nI; ++I) byleaf[leaf_of(I)].push_back(I);
                    for (size_t l = 0; l < leaves.size(); ++l) {
                        if (leaves[l].node < 0) continue;
                        double w = 0, mx[W] = {}; long long nv = 0; int nvis = 0;
                        for (int I : byleaf[l]) if (vwin[I] > 0) { ++nvis; nv += vwin[I]; w += vwin[I]; for (int a = 0; a < counts[I]; ++a) mx[a] += Rwin[(size_t)I * W + a]; }
                        double rms = 0, md = 0;
                        if (w > 0) {
                            for (int a = 0; a < W; ++a) mx[a] /= w;
                            for (int I : byleaf[l]) if (vwin[I] > 0) {
                                double d2 = 0;
                                for (int a = 0; a < counts[I]; ++a) { const double dv = Rwin[(size_t)I * W + a] / vwin[I] - mx[a]; d2 += dv * dv; md = max(md, fabs(dv)); }
                                rms += vwin[I] * d2;
                            }
                            rms = sqrt(rms / w);
                        }
                        fprintf(dsf, "%d,%lld,%zu,%d,%d,%zu,%d,%lld,%.6g,%.6g,%d\n", run.rows, run.t, l, leaves[l].cell, leaves[l].depth,
                                byleaf[l].size(), nvis, nv, rms, md, (int)leaves[l].pending);
                    }
                    fflush(dsf);
                }
            }
        });
        return st;
    };
    auto emit = [&](const RowStats& st, int before, int splits) {
        excl([&] {
            const double irs = st.irs, brs = st.brs, trs = st.trs;
            int pend = 0, alive = 0;
            for (const Leaf& L : leaves) if (L.node >= 0) { ++alive; pend += L.pending; }
            fprintf(logf, "%d,%lld,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%zu,%d,%.6g,%.6g,%.8g,%.8g,%.8g,%.8g,%.3f,%.3f,%lld,%lld,%d,%.3f,%.1f\n",
                    run.rows, run.t, run.k, level_now, before, B, splits, pp_mode || tree_mode || ppacc_mode ? alive : 0, pp_mode ? pend : 0, n_prior_splits,
                    n_prior_rejected, n_finalize, n_signals, waiting.size(), run.curK, schedule == "eps" ? eps_stat() : NAN,
                    run.eps, irs, brs, irs - brs, trs, train_secs(), assign_cpu, cnt.directions, n_gain_rejected, st.used, routing_mb(), peak_rss_mb());
            fflush(logf);
            cerr << "row " << run.rows << " iters " << run.t << " level " << level_now << " buckets " << before << " -> " << B;
            if (pp_mode || tree_mode || ppacc_mode) cerr << " leaves " << alive << " pending " << pend << " prior splits " << n_prior_splits;
            if (ppacc_mode) cerr << " posterior splits " << n_post_splits << " used buckets " << st.used;
            cerr << " (" << train_secs() << " s, assign " << assign_cpu << " s)\n";
            ++run.rows;
        });
    };
    // the posterior judgment at an epoch end
    auto epoch_end = [&]() -> int {
        if (pp_mode && !waiting.empty()) { const double a0 = cpu_now(); process_waiting(prior_split == "epoch"); drain_resolve(); assign_cpu += cpu_now() - a0; }
        Event ev;
        ev.K = run.curK; ev.tables = move(run.cur_tables); ev.Bl = B; ev.t = run.t;
        ev.unsplit.assign(B, -1); ev.split_idx.assign(B, -1);
        const TabView tv{ev.tables.data(), ev.K, ev.Bl};
        int B2 = 0, splits = 0;
        vector<int> keep;  // new id -> old id
        vector<char> is_split_child;
        for (int b = 0; b < B; ++b) {
            vector<Member> mm = members[b];
            sort(mm.begin(), mm.end(), [](const Member& x, const Member& y) { return x.pri < y.pri; });
            vector<int> mem; for (const Member& m : mm) mem.push_back(m.I);
            map<int, vector<double>> dirs;
            Split sp = ev.K > 0 ? judge(b, mem, tv, dirs) : Split{};
            if (sp.kids.empty()) { ev.unsplit[b] = B2++; keep.push_back(b); is_split_child.push_back(0); continue; }
            ++splits;
            sp.none_kid = B2++; keep.push_back(b); is_split_child.push_back(1);
            for (auto& [lab, id] : sp.kids) { id = B2++; keep.push_back(b); is_split_child.push_back(1); }
            ev.split_idx[b] = (int)ev.splits.size();
            string outs, labs, kids;
            for (int a = 0; a < bucket_n[b]; ++a) outs += (a ? " " : "") + to_string(sp.outliers[a]);
            labs = "none"; kids = to_string(sp.none_kid);
            for (const auto& [lab, id] : sp.kids) { labs += " " + to_string(lab); kids += " " + to_string(id); }
            fprintf(evf, "%d,%lld,%d,%d,%s,%s,%s\n", level_now, run.t, b, sp.n, outs.c_str(), labs.c_str(), kids.c_str());
            ev.splits.push_back(sp);
        }
        fflush(evf);
        ev.parent = keep;
        vector<double> Q2((size_t)B2 * W, 0.0), S2((size_t)B2 * W, 0.0), pe(B2, -1);
        vector<int32_t> n2(B2); vector<long long> v2(B2, 0);
        for (int nb = 0; nb < B2; ++nb) {
            const int ob = keep[nb];
            n2[nb] = bucket_n[ob];
            if (!is_split_child[nb]) {
                for (int a = 0; a < W; ++a) { Q2[(size_t)nb * W + a] = Q[(size_t)ob * W + a]; S2[(size_t)nb * W + a] = S[(size_t)ob * W + a]; }
                v2[nb] = vis[ob]; pe[nb] = parent_eps[ob];
            } else {
                for (int a = 0; a < W; ++a) { Q2[(size_t)nb * W + a] = inherit_r * Q[(size_t)ob * W + a]; S2[(size_t)nb * W + a] = inherit_s * S[(size_t)ob * W + a]; }
                pe[nb] = vis[ob] >= 25 || parent_eps[ob] < 0 ? psi(&Q[(size_t)ob * W], bucket_n[ob]) / max<long long>(1, vis[ob]) : parent_eps[ob];
            }
        }
        events.push_back(move(ev));
        ++level_now;
        B = B2; Q = move(Q2); S = move(S2); bucket_n = n2; vis = v2; parent_eps = pe;
        Rtmp.assign((size_t)B * W, 0); dirty.assign(B, 0);
        members.assign(B, {});
        member_salt = mix64(member_salt);
        if (measure) { fill(Rwin.begin(), Rwin.end(), 0.0); fill(vwin.begin(), vwin.end(), 0); }
        const double a0 = cpu_now();
        if (pp_mode) {
            const Event& e = events.back();
            for (Leaf& L : leaves) {
                if (L.node < 0) continue;
                if (!L.pending) {
                    if (e.unsplit[L.ptr] >= 0) L.ptr = e.unsplit[L.ptr];
                    else { const int ob = L.ptr; L.pending = true; L.start_level = level_now - 1; L.start_b = ob; L.prov = e.splits[e.split_idx[ob]].none_kid; L.samples.clear(); }
                } else L.prov = e.unsplit[L.prov] >= 0 ? e.unsplit[L.prov] : e.splits[e.split_idx[L.prov]].none_kid;
            }
            ++assign_ver;
            if (eager) {  // test T4: every pending leaf of one information set is resolved now (as pure's table rebuild)
                for (size_t l = 0; l < leaves.size(); ++l) {
                    Leaf& L = leaves[l];
                    if (L.node < 0 || !L.pending) continue;
                    if (cell_members[L.cell].size() != 1) throw runtime_error("--eager needs --prior-start identity");
                    L.ptr = route_from(cell_members[L.cell][0], L.start_level, L.start_b); L.pending = false; L.samples.clear();
                }
                ++assign_ver;
            }
            if (leaf_start_parent) {
                // no route ever starts before the newest event: pending leaves restart from their provisional
                // pointer's place before it, and the strategy tables of older events are freed
                for (Leaf& L : leaves)
                    if (L.node >= 0 && L.pending && L.start_level < level_now - 1) { L.start_level = level_now - 1; L.start_b = ancestor(L.prov, level_now - 1); }
                for (int e2 = 0; e2 + 1 < level_now; ++e2) { vector<double>().swap(events[e2].tables); }
                vector<int> keep;
                for (int l : waiting) { if (leaves[l].sig_level >= level_now - 1) keep.push_back(l); else leaves[l].sig_level = -1; }
                waiting = keep;
                ++assign_ver;
            }
            grow_to_ratio();
            drain_resolve();
            leaves_at_event = n_leaves_alive();
        }
        assign_cpu += cpu_now() - a0;
        rebuild_table();
        return splits;
    };
    auto next_row_t = [&] { return (long long)llround(1000.0 * pow(2.0, (double)run.next_row_j / rows_per_octave)); };
    double last_ckpt = cpu_now();
    while (run.t < T) {
        ++run.t; cur_t = run.t;
        // store this iterate (static: drawn iterations; eps: reservoir over the epoch's iterations)
        if (!split_on || direction_avg || tree_mode || ppacc_mode) {  // --direction avg, --mode tree: nothing stored during the epoch
        } else if (schedule == "static") {
            if (run.next_sample < run.samples.size() && run.samples[run.next_sample] == run.t) {
                const auto tab = strategy_table();
                run.cur_tables.insert(run.cur_tables.end(), tab.begin(), tab.end());
                ++run.curK; ++run.next_sample;
            }
        } else {
            const long long c = ++run.res_count;
            if (run.curK < run.Kk) { const auto tab = strategy_table(); run.cur_tables.insert(run.cur_tables.end(), tab.begin(), tab.end()); ++run.curK; }
            else {
                const long long j = uniform_int_distribution<long long>(0, c - 1)(iter_rng);
                if (j < run.Kk) { const auto tab = strategy_table(); copy(tab.begin(), tab.end(), run.cur_tables.begin() + (size_t)j * B * W); }
            }
        }
        for (int p = 0; p < 2; ++p) { es(0, p); apply_regrets(); }
        if (pp_mode && !to_finalize.empty()) {
            const double a0 = cpu_now();
            vector<int> f = move(to_finalize); to_finalize.clear();
            for (int l : f) finalize(l);
            drain_resolve();
            assign_cpu += cpu_now() - a0;
        }
        const bool budget_end = budget > 0 && train_secs() >= budget;
        if (budget_end) T = run.t;
        bool is_end = false;
        if (schedule == "static") is_end = run.t == epoch_end_static(run.k);
        else if (run.k == 0) is_end = run.t == eps_at;
        else is_end = run.t - run.epoch_start >= 1000 && run.t % eps_check == 0 && eps_stat() < run.eps;
        if (is_end && direction_avg && split_on) { run.cur_tables = avg_table(); run.curK = 1; }  // the split-time average strategy
        const bool grid_row = run.t == next_row_t();
        if (grid_row) ++run.next_row_j;
        if (is_end || grid_row || run.t == T) {
            const int before = B;
            int splits = 0;
            if (is_end && run.curK > 0 && schedule == "eps" && run.k == 0) run.eps = eps_stat();
            const RowStats st = snapshot(before);
            if (is_end && (run.curK > 0 || tree_mode || ppacc_mode) && split_on) {
                splits = tree_mode ? tree_epoch_end() : ppacc_mode ? ppacc_epoch_end(run.k) : epoch_end();
                ++run.k; run.epoch_start = run.t;
                draw_samples();
            }
            emit(st, before, splits);
            excl(save_state);
            last_ckpt = cpu_now();
            if (stop_after_rows > 0 && run.rows >= stop_after_rows) break;
        } else if (cpu_now() - last_ckpt >= ckpt_every) {
            excl(save_state);
            last_ckpt = cpu_now();
        }
    }
    excl([&] {  // every information set's bucket at the end (the k-means baseline matches these counts)
        vector<int32_t> fm(nI);
        for (int I = 0; I < nI; ++I) fm[I] = current_bucket(I);
        save_vec(out + "/final_bucket_map.bin", fm);
    });
    if (pp_mode || tree_mode || ppacc_mode) {
        excl([&] {
            ofstream j(out + "/prior_tree.json");
            j << "{\"cells\":" << cell_root.size() << ",\"nodes\":[";
            for (size_t i = 0; i < tnodes.size(); ++i) {
                const TNode& t = tnodes[i];
                j << (i ? "," : "") << "{\"id\":" << i;
                if (t.leaf >= 0) { const Leaf& L = leaves[t.leaf]; j << ",\"leaf\":" << t.leaf << ",\"pointer\":" << (L.pending ? L.prov : L.ptr) << ",\"pending\":" << (int)L.pending << ",\"depth\":" << L.depth; }
                else j << ",\"test\":\"" << desc_name(t.d) << "\",\"threshold\":" << t.thr << ",\"left\":" << t.left << ",\"right\":" << t.right;
                j << "}";
            }
            j << "],\"roots\":[";
            for (size_t c = 0; c < cell_root.size(); ++c) j << (c ? "," : "") << cell_root[c];
            j << "]}\n";
        });
    }
    fclose(logf); fclose(evf); if (dsf) fclose(dsf); fclose(prior_log);
    if (run.t >= T) ofstream(out + "/DONE") << run.t << "\n";
    return 0;
} catch (const exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
}
