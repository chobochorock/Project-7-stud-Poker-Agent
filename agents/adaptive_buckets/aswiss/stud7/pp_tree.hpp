// Prior-posterior ASWISS on 7-stud: the state of the partition (start cells, prior tree, leaves with pointers, the
// posterior genealogy and its split events) and the prior-tree fit (PP_STUD7_DESIGN_KO.md sections 4-6). Included
// inside the anonymous namespace of aswiss_stud.cpp after pp_obs.hpp.
#pragma once

namespace pp {

struct Opts {
    bool fill = false;           // --pp-assign fill: pending leaves fill m training visits (spec "assign while learning")
    bool root_start = false;     // --leaf-start root: new prior leaves route from the start of the genealogy
    bool poker = false;          // --features poker
    bool label_criterion = true; // --prior-criterion label (Gini of the rows' posterior children) | regret (TARA)
    bool crossfit = false;       // --crossfit 1: tests chosen on one half of the rows, checked on the other
    int particles = 64, n0 = 64, n_min = 8, m = 16, rho = 64, min_leaf = 1;
    double z = 2, ratio = 4, pass_frac = 0.125, budget = 0.25, budget_min_steps = 2e6, judge_mem_mb = 256, mem_budget_mb = 0;
    double inherit_regret = 0, inherit_avg = 0;  // weak inheritance: a child starts from lambda x / mu x its parent's tables
    bool inherit_share = false;  // --inherit-share 1 (tree): times the child's share of the source leaf's judged samples
    int threads = 1;
    bool online_split = false;   // --online-lbr-schedule split
    uint64_t epoch_growth = 2;   // --epoch-growth G: the next epoch is G times as long (the split schedule's interval is G^i)
    // --tree-split accum (tree only): leaves split by the regret accumulated during training on the two sides of K
    // candidate tests (PP_STUD7_DESIGN_KO.md section 18) instead of the particle labels of a judgment
    bool acc = false;
    int acc_k = 8, acc_min = 100, acc_pool = 16;
    double acc_z = 3;
    bool acc_dir = false;        // --acc-metric dir: the sides must also differ in the direction of their regret vectors
    double acc_dist = 0.5;       // --acc-dist: least |u_L - u_R|^2 of the unit mean regret vectors (0..4) for dir
    bool tree = false;           // --pp-structure tree: every prior leaf is its own bucket (no posterior layer, no pointers)
};

struct TNode { Desc d; float thr = 0; int32_t left = -1, right = -1, leaf = -1; };
struct Split { int32_t level = 0; std::array<uint32_t, A + 1> kid{}; };  // kid[a]: child of label a, kid[A]: the none child
struct FillRow { Sample x; std::array<float, A> r{}, se{}; int8_t label = -1; };
struct Leaf {
    int32_t node = -1;
    uint16_t ctx = 0, depth = 0;
    bool pending = false;
    uint32_t ptr = 0xffffffffu, prov = 0xffffffffu, start_b = 0xffffffffu;
    int32_t start_level = 0;
    uint64_t born = 0;
    std::vector<FillRow> fill;   // --pp-assign fill: the samples collected while pending
};
struct Cell { uint16_t ctx = 0; int32_t root = -1; uint32_t bucket = 0xffffffffu; };
// a particle estimate of one sample's direction (r: per-visit regret, se: its standard error over the particles)
struct Dir { std::array<double, A> r{}, se{}; uint8_t mask = 0; int label = -1; double neff = 0; uint64_t steps = 0; };
struct Rows { std::vector<Sample> x; std::vector<Dir> d; };              // a judgment's rows with their directions
struct Reservoir { uint64_t visits = 0; std::vector<Sample> x; std::vector<uint64_t> h; };
// --tree-split accum: a candidate test of a leaf with the training regrets of its two sides (0: feat <= thr, 1: above):
// per legal action the sum and the sum of squares of the per-visit regret v(a) - sum_b sigma(b) v(b), and the visits
struct AccCand {
    Desc d; float thr = 0;
    std::array<std::array<double, A>, 2> s{}, q{};
    std::array<double, 2> n{};
};
struct LeafAcc {
    std::vector<AccCand> cand;
    std::vector<Sample> pool;    // reservoir of the leaf's traverser visits (candidates are drawn from it)
    uint64_t seen = 0;           // traverser visits offered to the reservoir
    uint32_t gen = 0;            // candidate draws so far (seeds the next draw)
};
// the test of one candidate: D2 = |mean_L - mean_R|^2 over the legal actions, standardised by its mean and standard
// deviation when both sides have the same mean (pooled within-side variance, normal approximation); gain = the drop of
// the squared error of the per-visit regret vectors, n_L n_R / (n_L + n_R) * D2
// --acc-metric dir: u_s = the side's mean regret vector scaled to unit length when it is significantly non-zero (the same
// standardisation, one side), else 0 (that side has nothing it wants to change); dist = |u_L - u_R|^2 in [0, 4]: 0 when
// both sides want the same direction (whatever their sizes), 1 when one side is satisfied and the other is not, 4 when
// they want opposite directions. The candidate passes when z >= z_min (the sides differ) and dist >= dist_min.
struct AccTest { bool testable = false, pass = false; double d2 = 0, z = 0, gain = 0, dist = 0; };
AccTest acc_test(const AccCand& c, uint8_t mask, double min_visits, double z_min, bool dir = false, double dist_min = 0) {
    AccTest t;
    const double nl = c.n[0], nr = c.n[1];
    if (nl < min_visits || nr < min_visits) return t;
    t.testable = true;
    const double h = 1 / nl + 1 / nr;
    double mu = 0, var = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        const double ml = c.s[0][a] / nl, mr = c.s[1][a] / nr;
        const double within = std::max(0.0, c.q[0][a] - c.s[0][a] * ml) + std::max(0.0, c.q[1][a] - c.s[1][a] * mr);
        const double v = within / std::max(1.0, nl + nr - 2);
        t.d2 += (ml - mr) * (ml - mr);
        mu += v * h; var += 2 * (v * h) * (v * h);
    }
    t.z = var > 0 ? (t.d2 - mu) / std::sqrt(var) : (t.d2 > 0 ? 1e300 : 0);
    t.gain = nl * nr / (nl + nr) * t.d2;
    t.pass = t.z >= z_min;
    if (dir) {
        std::array<std::array<double, A>, 2> u{};
        for (int side = 0; side < 2; ++side) {
            const double n = c.n[side];
            double m2 = 0, mu1 = 0, var1 = 0;
            for (int a = 0; a < A; ++a) {
                if (!(mask & (1u << a))) continue;
                const double m = c.s[side][a] / n;
                const double v = std::max(0.0, c.q[side][a] - c.s[side][a] * m) / std::max(1.0, n - 1);
                m2 += m * m; mu1 += v / n; var1 += 2 * (v / n) * (v / n);
            }
            const double zs = var1 > 0 ? (m2 - mu1) / std::sqrt(var1) : (m2 > 0 ? 1e300 : 0);
            if (zs >= z_min && m2 > 0)
                for (int a = 0; a < A; ++a) if (mask & (1u << a)) u[side][a] = c.s[side][a] / n / std::sqrt(m2);
        }
        for (int a = 0; a < A; ++a) t.dist += (u[0][a] - u[1][a]) * (u[0][a] - u[1][a]);
        t.pass = t.pass && t.dist >= dist_min;
    }
    return t;
}

struct Data {
    Opts opt;
    std::unordered_map<uint16_t, int32_t> cell_of;
    std::vector<Cell> cells;
    std::vector<TNode> nodes;
    std::vector<Leaf> leaves;
    std::vector<int32_t> born;                     // per bucket record: the split level it was created at
    std::unordered_map<uint32_t, Split> splits;    // bucket -> its split event
    int32_t level = 0;                             // split events so far (a judgment that split nothing is not one)
    std::map<int32_t, std::vector<std::array<float, A>>> tables;  // stored average tables per level (root / fill)
    // counters (checkpointed)
    uint64_t epoch_nodes = 0;                      // training decision nodes since the last judgment (the budget's unit)
    double steps_per_sample = 4000;                // policy lookups per particle estimate (from the last judgment)
    uint64_t last_split_root = 0, fill_estimates = 0, prior_splits = 0, signals = 0;
    bool frozen = false;                           // --mem-budget-mb reached: no more judgments
    int32_t leaves_at_judgment = 0;
    std::vector<std::string> split_rows;           // prior_splits.csv rows not yet written (fill mode, during training)
    std::vector<LeafAcc> acc;                      // --tree-split accum: per leaf (index = leaf), grown on demand

    uint32_t ancestor(uint32_t b, int32_t lv, const std::vector<uint32_t>& parent) const {
        while (b != 0xffffffffu && born[b] > lv) b = parent[b];
        return b;
    }
    uint32_t kid_of(uint32_t b, int label) const {
        const Split& sp = splits.at(b);
        return label >= 0 && sp.kid[label] != 0xffffffffu ? sp.kid[label] : sp.kid[A];
    }
    int32_t leaf_of(int32_t root, const Stream& s) const {
        int32_t x = root;
        while (nodes[x].leaf < 0) x = feat(s, nodes[x].d) <= nodes[x].thr ? nodes[x].left : nodes[x].right;
        return nodes[x].leaf;
    }
    uint32_t current(int32_t l) const { return leaves[l].pending ? leaves[l].prov : leaves[l].ptr; }
    size_t live_leaves() const { size_t n = 0; for (const Leaf& L : leaves) n += L.node >= 0; return n; }
};

// ---- prior-tree fit on the rows of one leaf ----
struct Row {
    Stream s;
    std::array<float, A> x{};  // per-visit direction (regret criterion)
    uint32_t lab = 0;          // the posterior child the row's own direction picks (label criterion)
    double w = 1;
};
struct Candidate { Desc d; double thr = 0, gainA = 0, gainB = 0; bool ok = false; };

std::vector<Desc> candidate_features(const std::vector<Row>& rows, bool poker) {
    std::vector<Desc> all;
    for (const Row& r : rows) features_of(r.s, poker, all);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}
// The best binary test: chosen on half A (half[i] == 0), checked on half B (all rows when crossfit is off). Criterion:
// label: reduction of the weighted Gini impurity of the rows' children, w - sum c^2 / w;
// regret: reduction of the weighted squared error of the direction vectors, sum w|x|^2 - |sum w x|^2 / sum w.
Candidate best_split(const std::vector<Row>& rows, const std::vector<char>& half, const Opts& opt) {
    Candidate best;
    if (rows.size() < 2) return best;
    int nA = 0; for (char h : half) nA += !h;
    const std::vector<Desc> ds = candidate_features(rows, opt.poker);
    // label indices
    std::map<uint32_t, int> li;
    for (const Row& r : rows) li.emplace(r.lab, 0);
    int C = 0; for (auto& [l, i] : li) i = C++;
    if (opt.label_criterion && C < 2) return best;
    struct Acc {
        double w = 0, wxx = 0;
        std::array<double, A> wx{};
        std::vector<double> c;
    };
    auto add = [&](Acc& a, const Row& r) {
        a.w += r.w;
        if (opt.label_criterion) a.c[li[r.lab]] += r.w;
        else for (int k = 0; k < A; ++k) { a.wx[k] += r.w * r.x[k]; a.wxx += r.w * r.x[k] * r.x[k]; }
    };
    auto fresh = [&] { Acc a; a.c.assign(C, 0.0); return a; };
    auto imp = [&](const Acc& a) {
        if (a.w <= 0) return 0.0;
        double q = 0;
        if (opt.label_criterion) { for (double x : a.c) q += x * x; return a.w - q / a.w; }
        for (double x : a.wx) q += x * x;
        return a.wxx - q / a.w;
    };
    auto sub = [&](const Acc& t, const Acc& l) {
        Acc r = fresh(); r.w = t.w - l.w; r.wxx = t.wxx - l.wxx;
        for (int k = 0; k < A; ++k) r.wx[k] = t.wx[k] - l.wx[k];
        for (int c = 0; c < C; ++c) r.c[c] = t.c[c] - l.c[c];
        return r;
    };
    Acc totA = fresh();
    for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) add(totA, rows[i]);
    const double baseA = imp(totA);
    std::vector<std::pair<double, int>> va;
    for (const Desc& d : ds) {
        va.clear();
        for (size_t i = 0; i < rows.size(); ++i) if (!half[i]) va.push_back({feat(rows[i].s, d), static_cast<int>(i)});
        std::sort(va.begin(), va.end());
        Acc left = fresh();
        for (int k = 0; k + 1 < static_cast<int>(va.size()); ++k) {
            add(left, rows[va[k].second]);
            if (va[k].first == va[k + 1].first) continue;
            const int nl = k + 1, nr = nA - nl;
            if (nl < opt.min_leaf || nr < opt.min_leaf) continue;
            const double g = baseA - imp(left) - imp(sub(totA, left));
            if (g > best.gainA * (1 + 1e-12) + 1e-300) {
                best.gainA = g; best.d = d; best.thr = static_cast<float>(0.5 * (va[k].first + va[k + 1].first)); best.ok = true;
            }
        }
    }
    if (!best.ok) return best;
    if (nA == static_cast<int>(rows.size())) { best.gainB = best.gainA; best.ok = best.gainA > 1e-12 * std::max(1.0, baseA); return best; }
    Acc pB = fresh(), lB = fresh(), rB = fresh();
    for (size_t i = 0; i < rows.size(); ++i)
        if (half[i]) { add(pB, rows[i]); add(feat(rows[i].s, best.d) <= best.thr ? lB : rB, rows[i]); }
    best.gainB = imp(pB) - imp(lB) - imp(rB);
    best.ok = best.gainB > 1e-12 * std::max(1.0, imp(pB)) && lB.w > 0 && rB.w > 0;
    return best;
}

// A test that separates the rows at all, the most balanced one (rows on the smaller side; ties: the first feature in
// their order). Used by --pp-structure tree when the rows' labels differ but no test has a positive gain (for example
// labels in an XOR pattern), so that a leaf whose samples disagree is always split.
Candidate balanced_split(const std::vector<Row>& rows, const Opts& opt) {
    Candidate best;
    int best_min = 0;
    const std::vector<Desc> ds = candidate_features(rows, opt.poker);
    std::vector<double> v(rows.size());
    for (const Desc& d : ds) {
        for (size_t i = 0; i < rows.size(); ++i) v[i] = feat(rows[i].s, d);
        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        for (size_t k = 0; k + 1 < u.size(); ++k) {
            if (u[k] == u[k + 1]) continue;
            const int m = static_cast<int>(std::min(k + 1, u.size() - k - 1));
            if (m > best_min) { best_min = m; best.d = d; best.thr = static_cast<float>(0.5 * (u[k] + u[k + 1])); best.ok = true; }
        }
    }
    return best;
}

}  // namespace pp
