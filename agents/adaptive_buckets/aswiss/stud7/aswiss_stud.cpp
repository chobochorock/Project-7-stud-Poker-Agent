// ASWISS on heads-up seven-card stud (rules v3, 5th to 7th street, heuristic H4 discard/reveal).
//
// Build: g++ -O3 -std=c++17 aswiss_stud.cpp -o aswiss_stud
// Train: ./aswiss_stud train --partition aswiss|atom|context [--atlas ATLAS] --roots N [--epoch0 10000]
//                            [--z 2] [--inherit 0] [--seed 1] [--power-cache 1000000] [--checkpoint-every 1000000]
//                            [--snapshot-every 0] [--curve-every 10000] --out DIR      (rerun to resume)
// Eval:  ./aswiss_stud eval [--atlas ATLAS] --snapshot FILE --policy average|current [--pairs 2000]
//                           [--particles 64] [--gap-roots 4000] [--seed 2026102701]
// Test:  ./aswiss_stud self-test --atlas ATLAS
//
// Reuses the engine in ../../../cpp_mccfr/stud_mccfr.cpp (rules, power atlas, PolicyLBR) unchanged.
// Atom: the hard power InfoKey of a frozen atlas (power cluster x betting summary x legal mask), i.e.
// exactly the key of the engine's `--bucket power` k-means model. Context: the atom key with the power
// cluster cleared (no card information). Each context partitions its clusters into buckets; clusters
// never assigned explicitly fall in the context's rest bucket. Policy tables live on buckets.
//   context: one bucket per context, never split (the time-step analogue; no atlas needed)
//   atom:    one bucket per atom, never split (the atlas trained directly = hard k-means)
//   aswiss:  starts as context; at every doubling epoch end each atom visited in the epoch gets label
//            argmax_a R(a) when R(a) > z sqrt(sum r(a)^2) over the epoch's sampled regrets, else none.
//            Unvisited members (for a rest bucket: unseen clusters) count as none. A bucket carrying two
//            or more labels is split by label; children restart their tables (--inherit 1: each child starts
//            from a copy of the parent's regret and average tables); no merges.
// Training: external-sampling MCCFR+ (RM+) from a sampled 5th-street root, traversers 0 then 1 on the
// same deal; bucket regrets are applied after each traversal; linear averaging at opponent nodes.
// Each epoch end writes snapshot_k.bin (partition + tables used during that epoch, before any split);
// --snapshot-every adds snapshot_rN.bin; snapshots.csv indexes both. curve.csv records the sampled BR-gap proxy of
// the current policy on the training deals (free by-product of the traversal) every --curve-every roots.
// Resumable: checkpoint.bin (full training state, RNGs, output lengths) is written every --checkpoint-every roots,
// at epoch ends, on Ctrl+C/SIGTERM and when OUT/STOP appears; rerunning the same command resumes bit-exactly.
// Eval: paired Policy-LBR (the engine's approximate exploitability lower bound, ante/hand; same deals,
// LBR seed and particles for every snapshot) and the sampled BR-gap proxy on fixed deals (sum over the
// traverser's visited information states of max_a Q - sum_a pi Q under external sampling, mean of both
// seats, ante/hand). Neither is exact exploitability.
#define STUD_MCCFR_NO_MAIN
#include "../../../cpp_mccfr/stud_mccfr.cpp"

#include <csignal>
#include <filesystem>
#include <map>
#include <set>

namespace {

constexpr int A = kActionCount;
constexpr uint32_t kNone = 0xffffffffu;

struct NullAtlas final : PowerAtlas {
    uint16_t assign(const State&, int) const override { return 0; }
};

enum class Partition { Context, Atom, Aswiss };

struct Bucket {
    std::array<double, A> q{}, s{}, pending{};
    bool dirty = false;
};
struct AtomStats {
    std::array<double, A> r{}, sq{};
    double visits = 0;
};
struct Context {
    std::unordered_map<uint16_t, uint32_t> member;  // explicit cluster -> bucket
    uint32_t rest = kNone;
};

InfoKey context_of(InfoKey key) { key.power_cluster = 0; return key; }

std::array<double, A> rm_plus(const std::array<double, A>& q, uint8_t mask) {
    std::array<double, A> s{};
    double total = 0; int count = 0;
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) { s[a] = std::max(0.0, q[a]); total += s[a]; ++count; }
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) s[a] = total > 0 ? s[a] / total : 1.0 / count;
    return s;
}
std::array<double, A> normalized(const std::array<double, A>& num, uint8_t mask) {
    double total = 0;
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) total += num[a];
    if (total <= 0) return uniform_strategy(mask);
    std::array<double, A> s{};
    for (int a = 0; a < A; ++a) if (mask & (1u << a)) s[a] = num[a] / total;
    return s;
}
Action sample_action(const std::array<double, A>& s, uint8_t mask, std::mt19937_64& rng) {
    const double x = std::uniform_real_distribution<double>(0, 1)(rng);
    double acc = 0; Action last = FOLD;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        last = static_cast<Action>(a); acc += s[a];
        if (x < acc) return last;
    }
    return last;
}
void advance(State& state) {  // same as MCCFR::advance_street (draws from the sampled deck)
    if (state.street == 7) { state.terminal = true; return; }
    ++state.street;
    for (int seat = 0; seat < 2; ++seat) {
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (state.street != 7) state.players[seat].shown.push_back(card);
        else state.players[seat].hidden.push_back(card);
        state.players[seat].round_bet = 0;
    }
    state.highest_bet = 0; state.raise_count = 0;
    if (state.players[0].all_in || state.players[1].all_in) { advance(state); return; }
    state.actor = first_bettor(state);
}
void step(State& state, int actor, Action action) {
    const ActionResult result = apply_action(state, actor, action);
    if (result == ActionResult::RoundEnd) advance(state);
    else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
}

template <class T> void put(std::ofstream& out, const T& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void get(std::ifstream& in, T& v) { in.read(reinterpret_cast<char*>(&v), sizeof(T)); }
void put_key(std::ofstream& out, const InfoKey& k) {
    for (uint8_t v : {k.abstraction, k.street, k.category, k.rank_bucket, k.own_public_category, k.own_public_suit,
                      k.own_public_connected, k.opponent_public_category, k.opponent_public_suit,
                      k.opponent_public_connected, k.pot_odds_bucket, k.stack_pot_bucket, k.own_bet_count,
                      k.opponent_bet_count, k.checked, k.last_action_class, k.betting_goal, k.legal_mask,
                      k.history_length}) put(out, v);
    put(out, k.power_cluster); put(out, k.history_code);
}
InfoKey get_key(std::ifstream& in) {
    InfoKey k{};
    for (uint8_t* v : {&k.abstraction, &k.street, &k.category, &k.rank_bucket, &k.own_public_category,
                       &k.own_public_suit, &k.own_public_connected, &k.opponent_public_category,
                       &k.opponent_public_suit, &k.opponent_public_connected, &k.pot_odds_bucket,
                       &k.stack_pot_bucket, &k.own_bet_count, &k.opponent_bet_count, &k.checked,
                       &k.last_action_class, &k.betting_goal, &k.legal_mask, &k.history_length}) get(in, *v);
    get(in, k.power_cluster); get(in, k.history_code);
    return k;
}

class Model {
public:
    Model(Partition partition, const PowerAtlas* atlas) : partition_(partition), atlas_(atlas) {
        if (partition != Partition::Context && !atlas) throw std::runtime_error("atom/aswiss partitions need --atlas");
    }

    // atom key of the actor's information state (context partition: the context key itself)
    InfoKey atom_key(const State& state, int viewer) const {
        return partition_ == Partition::Context ? make_power_key(state, viewer, null_atlas_)
                                                : make_power_key(state, viewer, *atlas_);
    }
    uint32_t lookup(const InfoKey& atom) const {
        const auto it = contexts_.find(context_of(atom));
        if (it == contexts_.end()) return kNone;
        const auto m = it->second.member.find(atom.power_cluster);
        if (m != it->second.member.end()) return m->second;
        return partition_ == Partition::Atom ? kNone : it->second.rest;
    }
    uint32_t lookup_or_create(const InfoKey& atom) {
        auto [it, inserted] = contexts_.try_emplace(context_of(atom));
        Context& c = it->second;
        if (partition_ == Partition::Atom) {
            auto [m, fresh] = c.member.try_emplace(atom.power_cluster, 0u);
            if (fresh) m->second = new_bucket();
            return m->second;
        }
        if (inserted) c.rest = new_bucket();
        const auto m = c.member.find(atom.power_cluster);
        return m != c.member.end() ? m->second : c.rest;
    }
    std::array<double, A> strategy(const State& state, int viewer, bool average, bool* found = nullptr) const {
        const uint8_t mask = valid_mask(state, viewer);
        const uint32_t b = lookup(atom_key(state, viewer));
        if (found) *found = b != kNone;
        if (b == kNone) return uniform_strategy(mask);
        return average ? normalized(buckets_[b].s, mask) : rm_plus(buckets_[b].q, mask);
    }

    // one external-sampling traversal for `traverser`; `track` accumulates per-atom window regrets
    double traverse(State state, int traverser, double weight, std::mt19937_64& rng, bool track) {
        if (state.terminal) return terminal_net_search(state, traverser);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey atom = atom_key(state, actor);
        const uint32_t b = lookup_or_create(atom);
        const auto sigma = rm_plus(buckets_[b].q, mask);
        if (actor != traverser) {
            for (int a = 0; a < A; ++a) if (mask & (1u << a)) buckets_[b].s[a] += weight * sigma[a];
            step(state, actor, sample_action(sigma, mask, rng));
            return traverse(std::move(state), traverser, weight, rng, track);
        }
        std::array<double, A> v{};
        double node = 0;
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            State child = state;
            step(child, actor, static_cast<Action>(a));
            v[a] = traverse(std::move(child), traverser, weight, rng, track);
            node += sigma[a] * v[a];
        }
        Bucket& bucket = buckets_[b];  // re-read: the recursion may have grown buckets_
        if (!bucket.dirty) { bucket.dirty = true; touched_.push_back(b); }
        AtomStats* st = track ? &atoms_[atom] : nullptr;
        double best = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < A; ++a) {
            if (!(mask & (1u << a))) continue;
            const double r = v[a] - node;
            bucket.pending[a] += r;
            best = std::max(best, r);
            if (st) { st->r[a] += r; st->sq[a] += r * r; }
        }
        if (st) st->visits += 1;
        traversal_gap_ += best;  // diagnostic only: sampled BR-gap proxy of the current policy on training deals
        return node;
    }
    // sum of max_a Q - E_sigma Q over the traverser's nodes since the last call (ante units)
    double take_traversal_gap() { const double g = traversal_gap_; traversal_gap_ = 0; return g; }
    // Untouched buckets have no pending regret, so only touched ones are updated (RM+).
    void apply_regrets() {
        for (uint32_t b : touched_) {
            Bucket& bucket = buckets_[b];
            for (int a = 0; a < A; ++a) { bucket.q[a] = std::max(0.0, bucket.q[a] + bucket.pending[a]); bucket.pending[a] = 0; }
            bucket.dirty = false;
        }
        touched_.clear();
    }

    struct Window { double L = 0, C = 0; size_t atoms = 0; };
    // sampled atom-level L and bucket-level C of the epoch's window sums (diagnostic)
    Window window() const {
        Window w; w.atoms = atoms_.size();
        std::unordered_map<uint32_t, std::array<double, A>> sums;
        std::unordered_map<uint32_t, uint8_t> masks;
        for (const auto& [atom, st] : atoms_) {
            w.L += psi(st.r, atom.legal_mask);
            const uint32_t b = lookup(atom);
            auto& s = sums[b];
            for (int a = 0; a < A; ++a) s[a] += st.r[a];
            masks[b] = atom.legal_mask;
        }
        for (const auto& [b, s] : sums) w.C += psi(s, masks[b]);
        return w;
    }
    // ASWISS refinement from the window's atom labels; returns the number of buckets split
    int split(double z, bool inherit = false) {
        std::map<InfoKey, std::vector<std::pair<uint16_t, int>>, KeyLess> by_context;  // ordered: reproducible ids
        for (const auto& [atom, st] : atoms_) {
            int best = -1;
            for (int a = 0; a < A; ++a) if ((atom.legal_mask & (1u << a)) && (best < 0 || st.r[a] > st.r[best])) best = a;
            const int label = st.visits > 0 && st.r[best] > z * std::sqrt(st.sq[best]) ? best : -1;
            by_context[context_of(atom)].push_back({atom.power_cluster, label});
        }
        int splits = 0;
        for (auto& [ctx, visited] : by_context) {
            Context& c = contexts_.at(ctx);
            std::sort(visited.begin(), visited.end());
            std::unordered_map<uint16_t, int> label_of(visited.begin(), visited.end());
            std::map<uint32_t, std::set<int>> labels;  // bucket -> labels of its members
            labels[c.rest].insert(-1);                 // unseen clusters
            for (const auto& [cluster, b] : c.member) {
                const auto it = label_of.find(cluster);
                labels[b].insert(it == label_of.end() ? -1 : it->second);
            }
            for (const auto& [cluster, label] : visited)
                if (!c.member.count(cluster)) labels[c.rest].insert(label);
            std::map<uint32_t, std::map<int, uint32_t>> child;  // splitting bucket -> label -> new bucket
            for (const auto& [b, set] : labels) {
                if (set.size() < 2) continue;
                ++splits;
                for (int label : set) {
                    const uint32_t c = new_bucket();
                    if (inherit) { buckets_[c].q = buckets_[b].q; buckets_[c].s = buckets_[b].s; }
                    child[b][label] = c;
                }
            }
            if (child.empty()) continue;
            std::vector<std::pair<uint16_t, uint32_t>> members(c.member.begin(), c.member.end());
            std::sort(members.begin(), members.end());
            for (const auto& [cluster, b] : members) {
                const auto it = child.find(b);
                if (it == child.end()) continue;
                const auto l = label_of.find(cluster);
                c.member[cluster] = it->second.at(l == label_of.end() ? -1 : l->second);
            }
            const auto rest = child.find(c.rest);
            if (rest != child.end()) {
                for (const auto& [cluster, label] : visited)
                    if (!c.member.count(cluster) && label >= 0) c.member[cluster] = rest->second.at(label);
                c.rest = rest->second.at(-1);
            }
        }
        return splits;
    }
    void clear_window() { atoms_.clear(); }

    // Full training state for checkpoints: every bucket record (ids unchanged), the partition and the atom window.
    // Pending regrets are always zero between roots, so they are not stored.
    void save_state(std::ofstream& out) const {
        put(out, static_cast<uint8_t>(partition_));
        put(out, static_cast<uint64_t>(contexts_.size()));
        for (const auto& [ctx, c] : contexts_) {
            put_key(out, ctx); put(out, c.rest);
            put(out, static_cast<uint32_t>(c.member.size()));
            for (const auto& [cluster, b] : c.member) { put(out, cluster); put(out, b); }
        }
        put(out, static_cast<uint64_t>(buckets_.size()));
        for (const auto& b : buckets_) { put(out, b.q); put(out, b.s); }
        put(out, static_cast<uint64_t>(atoms_.size()));
        for (const auto& [atom, st] : atoms_) { put_key(out, atom); put(out, st.r); put(out, st.sq); put(out, st.visits); }
    }
    void load_state(std::ifstream& in) {
        uint8_t partition; get(in, partition);
        if (partition != static_cast<uint8_t>(partition_)) throw std::runtime_error("checkpoint partition differs");
        uint64_t n; get(in, n);
        contexts_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey ctx = get_key(in);
            Context c; get(in, c.rest);
            uint32_t m; get(in, m);
            for (uint32_t j = 0; j < m; ++j) { uint16_t cluster; uint32_t b; get(in, cluster); get(in, b); c.member[cluster] = b; }
            contexts_[ctx] = std::move(c);
        }
        get(in, n);
        buckets_.assign(n, Bucket{});
        for (auto& b : buckets_) { get(in, b.q); get(in, b.s); }
        get(in, n);
        atoms_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey atom = get_key(in);
            AtomStats st; get(in, st.r); get(in, st.sq); get(in, st.visits);
            atoms_[atom] = st;
        }
        touched_.clear();
    }

    // live buckets (referenced by a partition) per street 5/6/7
    std::array<size_t, 3> live_buckets() const {
        std::array<std::set<uint32_t>, 3> live;
        for (const auto& [ctx, c] : contexts_) {
            auto& s = live[ctx.street - 5];
            if (c.rest != kNone) s.insert(c.rest);
            for (const auto& [cluster, b] : c.member) s.insert(b);
        }
        return {live[0].size(), live[1].size(), live[2].size()};
    }
    size_t contexts() const { return contexts_.size(); }
    size_t bucket_records() const { return buckets_.size(); }  // live + retired (split parents are never freed)
    // approximate bytes of the training tables: bucket records, the atom window and the partition maps
    std::array<double, 3> table_megabytes() const {
        size_t members = 0;
        for (const auto& [ctx, c] : contexts_) members += c.member.size();
        constexpr double node = 2 * sizeof(void*);  // hash-node link and cached hash
        return {buckets_.capacity() * sizeof(Bucket) / 1e6,
                atoms_.size() * (sizeof(InfoKey) + sizeof(AtomStats) + node) / 1e6,
                (contexts_.size() * (sizeof(InfoKey) + sizeof(Context) + node) +
                 members * (sizeof(uint16_t) + sizeof(uint32_t) + node)) / 1e6};
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        out.write("ASWSTUD1", 8);
        put(out, static_cast<uint8_t>(partition_));
        std::map<uint32_t, uint32_t> remap;  // live bucket -> compact id
        auto id = [&](uint32_t b) { return b == kNone ? kNone : remap.try_emplace(b, (uint32_t)remap.size()).first->second; };
        std::vector<std::pair<InfoKey, const Context*>> ordered;
        for (const auto& [ctx, c] : contexts_) ordered.push_back({ctx, &c});
        std::sort(ordered.begin(), ordered.end(), [](const auto& x, const auto& y) { return KeyLess()(x.first, y.first); });
        put(out, static_cast<uint64_t>(ordered.size()));
        for (const auto& [ctx, c] : ordered) {
            put_key(out, ctx); put(out, id(c->rest));
            std::vector<std::pair<uint16_t, uint32_t>> members(c->member.begin(), c->member.end());
            std::sort(members.begin(), members.end());
            put(out, static_cast<uint32_t>(members.size()));
            for (const auto& [cluster, b] : members) { put(out, cluster); put(out, id(b)); }
        }
        std::vector<uint32_t> original(remap.size());
        for (const auto& [b, compact] : remap) original[compact] = b;
        put(out, static_cast<uint64_t>(original.size()));
        for (uint32_t b : original) { put(out, buckets_[b].q); put(out, buckets_[b].s); }
    }
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        char magic[8];
        in.read(magic, 8);
        if (!in || std::string(magic, 8) != "ASWSTUD1") throw std::runtime_error("not an ASWISS stud snapshot: " + path);
        uint8_t partition; get(in, partition);
        if (partition != static_cast<uint8_t>(partition_)) throw std::runtime_error("snapshot partition differs from --partition");
        uint64_t n; get(in, n);
        contexts_.clear();
        for (uint64_t i = 0; i < n; ++i) {
            const InfoKey ctx = get_key(in);
            Context c; get(in, c.rest);
            uint32_t m; get(in, m);
            for (uint32_t j = 0; j < m; ++j) { uint16_t cluster; uint32_t b; get(in, cluster); get(in, b); c.member[cluster] = b; }
            contexts_[ctx] = std::move(c);
        }
        get(in, n);
        buckets_.assign(n, Bucket{});
        for (auto& b : buckets_) { get(in, b.q); get(in, b.s); }
        if (!in) throw std::runtime_error("truncated snapshot: " + path);
    }

private:
    struct KeyLess {
        bool operator()(const InfoKey& x, const InfoKey& y) const {
            auto t = [](const InfoKey& k) {
                return std::make_tuple(k.street, k.legal_mask, k.pot_odds_bucket, k.stack_pot_bucket, k.own_bet_count,
                                       k.opponent_bet_count, k.checked, k.last_action_class, k.betting_goal, k.abstraction,
                                       k.category, k.rank_bucket, k.history_length, k.history_code, k.power_cluster);
            };
            return t(x) < t(y);
        }
    };
    static double psi(const std::array<double, A>& x, uint8_t mask) {
        double m = 0;
        for (int a = 0; a < A; ++a) if (mask & (1u << a)) m = std::max(m, x[a]);
        return m;
    }
    uint32_t new_bucket() { buckets_.emplace_back(); return static_cast<uint32_t>(buckets_.size() - 1); }

    Partition partition_;
    const PowerAtlas* atlas_;
    NullAtlas null_atlas_;
    std::unordered_map<InfoKey, Context, InfoKeyHash> contexts_;
    std::vector<Bucket> buckets_;
    std::vector<uint32_t> touched_;
    std::unordered_map<InfoKey, AtomStats, InfoKeyHash> atoms_;
    double traversal_gap_ = 0;
};

// frozen snapshot as a PolicyLBR target
struct SnapshotPolicy {
    const Model& model;
    bool average;
    mutable std::mt19937_64 rng;
    std::array<double, A> policy(const State& state, int seat, bool* found = nullptr) const {
        return model.strategy(state, seat, average, found);
    }
    Action choose(const State& state, int seat, int) {
        return sample_action(policy(state, seat), valid_mask(state, seat), rng);
    }
};

// sampled BR-gap proxy of one traverser on one deal (ante units)
double gap_traverse(const Model& model, bool average, State state, int traverser, std::mt19937_64& rng, double& gap) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const auto sigma = model.strategy(state, actor, average);
    if (actor != traverser) {
        step(state, actor, sample_action(sigma, mask, rng));
        return gap_traverse(model, average, std::move(state), traverser, rng, gap);
    }
    double node = 0, best = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        const double v = gap_traverse(model, average, std::move(child), traverser, rng, gap);
        node += sigma[a] * v; best = std::max(best, v);
    }
    gap += best - node;
    return node;
}

std::map<std::string, std::string> parse(int argc, char** argv, std::map<std::string, std::string> opt) {
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!opt.count(argv[i])) throw std::runtime_error(std::string("unknown option ") + argv[i]);
        opt[argv[i]] = argv[i + 1];
    }
    return opt;
}
Partition parse_partition(const std::string& s) {
    if (s == "context") return Partition::Context;
    if (s == "atom") return Partition::Atom;
    if (s == "aswiss") return Partition::Aswiss;
    throw std::runtime_error("--partition must be context, atom or aswiss");
}
std::unique_ptr<PowerAtlas> load_atlas(const std::string& path, size_t cache) {
    if (path.empty()) return nullptr;
    auto atlas = std::make_unique<PowerAtlas>(128);
    atlas->load(path);
    atlas->set_assignment_cache_limit(cache);
    return atlas;
}
constexpr int kAnte = 1000;

double peak_rss_mb() {  // VmHWM of this process (Linux); -1 elsewhere
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::stod(line.substr(6)) / 1024.0;
    return -1;
}

volatile std::sig_atomic_t g_stop = 0;
void request_stop(int) { g_stop = 1; }

uint64_t file_size_or_zero(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}
void put_string(std::ofstream& out, const std::string& s) { put(out, static_cast<uint64_t>(s.size())); out.write(s.data(), s.size()); }
std::string get_string(std::ifstream& in) { uint64_t n; get(in, n); std::string s(n, '\0'); in.read(s.data(), n); return s; }
template <class R> std::string rng_text(const R& r) { std::ostringstream o; o << r; return o.str(); }
template <class R> void rng_load(R& r, const std::string& text) { std::istringstream i(text); i >> r; }

// Loop counters, RNG states and output-file lengths stored with every checkpoint.
struct LoopState {
    uint64_t t = 0, k = 0, L = 0, epoch_start = 0, curve_roots = 0;
    double seconds = 0, curve_sum = 0, curve_sq = 0;
    std::string deal_rng, rng;
    uint64_t log_bytes = 0, index_bytes = 0, curve_bytes = 0;
};

// Checkpoint = config line + loop state + model state, written to checkpoint.tmp and renamed over checkpoint.bin.
// On load, checkpoint.bin is preferred; a complete checkpoint.tmp left by an interrupted rename is also accepted.
void write_checkpoint(const std::string& out, const std::string& config, const LoopState& s, const Model& model) {
    const std::string tmp = out + "/checkpoint.tmp", path = out + "/checkpoint.bin";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write("ASWCKPT1", 8);
        put_string(f, config);
        for (uint64_t v : {s.t, s.k, s.L, s.epoch_start, s.curve_roots, s.log_bytes, s.index_bytes, s.curve_bytes}) put(f, v);
        for (double v : {s.seconds, s.curve_sum, s.curve_sq}) put(f, v);
        put_string(f, s.deal_rng); put_string(f, s.rng);
        model.save_state(f);
        f.write("ASWEND01", 8);
        if (!f) throw std::runtime_error("failed to write " + tmp);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // Windows cannot rename over an existing file
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path);
    }
}
bool read_checkpoint(const std::string& file, const std::string& config, LoopState& s, Model& model) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (!f || std::string(magic, 8) != "ASWCKPT1") return false;
    const std::string stored = get_string(f);
    if (stored != config)
        throw std::runtime_error("checkpoint was written with different settings:\n  " + stored + "\nnow:\n  " + config);
    for (uint64_t* v : {&s.t, &s.k, &s.L, &s.epoch_start, &s.curve_roots, &s.log_bytes, &s.index_bytes, &s.curve_bytes}) get(f, *v);
    for (double* v : {&s.seconds, &s.curve_sum, &s.curve_sq}) get(f, *v);
    s.deal_rng = get_string(f); s.rng = get_string(f);
    model.load_state(f);
    f.read(magic, 8);
    return f && std::string(magic, 8) == "ASWEND01";
}

int train(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "aswiss"}, {"--atlas", ""}, {"--roots", "1000000"}, {"--epoch0", "10000"},
                                {"--z", "2"}, {"--seed", "1"}, {"--power-cache", "1000000"}, {"--out", ""},
                                {"--inherit", "0"}, {"--checkpoint-every", "1000000"}, {"--snapshot-every", "0"},
                                {"--curve-every", "10000"}});
    const Partition partition = parse_partition(o["--partition"]);
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    Model model(partition, atlas.get());
    const std::string out = o.at("--out");
    if (out.empty()) throw std::runtime_error("--out is required");
    std::filesystem::create_directories(out);
    const uint64_t T = std::stoull(o["--roots"]), epoch0 = std::stoull(o["--epoch0"]);
    const uint64_t checkpoint_every = std::stoull(o["--checkpoint-every"]), snapshot_every = std::stoull(o["--snapshot-every"]);
    const uint64_t curve_every = std::stoull(o["--curve-every"]);
    const double z = std::stod(o["--z"]);
    const bool inherit = o["--inherit"] == "1";
    // settings that must match on resume (--roots may grow; output cadences may change)
    std::ostringstream cfg;
    cfg << "partition=" << o["--partition"] << ";atlas=" << std::filesystem::path(o["--atlas"]).filename().string() << ":"
        << file_size_or_zero(o["--atlas"]) << ";epoch0=" << epoch0 << ";z=" << o["--z"] << ";inherit=" << o["--inherit"]
        << ";seed=" << o["--seed"];
    const std::string config = cfg.str();

    std::mt19937_64 deal_rng(std::stoull(o["--seed"])), rng(std::stoull(o["--seed"]) ^ 0x9e3779b97f4a7c15ull);
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    LoopState s;
    s.L = epoch0;
    const bool resumed = read_checkpoint(out + "/checkpoint.bin", config, s, model) ||
                         read_checkpoint(out + "/checkpoint.tmp", config, s, model);
    if (resumed) {
        rng_load(deal_rng, s.deal_rng); rng_load(rng, s.rng);
        // drop rows written after the checkpoint; they are reproduced exactly by the replay
        std::filesystem::resize_file(log_path, s.log_bytes);
        std::filesystem::resize_file(index_path, s.index_bytes);
        std::filesystem::resize_file(curve_path, s.curve_bytes);
        std::cerr << "resumed at root " << s.t << " (epoch " << s.k << ")\n";
    } else {
        if (std::filesystem::exists(log_path))
            throw std::runtime_error(out + " already has log.csv but no usable checkpoint; use a new --out");
        std::ofstream(log_path) << "epoch,roots,contexts,buckets_after,buckets5,buckets6,buckets7,splits,atoms,L_per_root,"
                                   "C_per_root,H_per_root,seconds,power_cache_hits,power_cache_misses,bucket_records,"
                                   "bucket_mb,atom_window_mb,partition_mb,peak_rss_mb\n";
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets,peak_rss_mb\n";
    }
    std::ofstream log(log_path, std::ios::app), index(index_path, std::ios::app), curve(curve_path, std::ios::app);
    // graceful stop: Ctrl+C / SIGTERM, or creating OUT/STOP (works for background processes on Windows too)
    const std::string stop_file = out + "/STOP";
    std::filesystem::remove(stop_file);
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
#ifdef SIGBREAK
    std::signal(SIGBREAK, request_stop);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const double base_seconds = s.seconds;
    auto elapsed = [&] { return base_seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto live_total = [&] { const auto l = model.live_buckets(); return l[0] + l[1] + l[2]; };
    auto checkpoint = [&] {
        log.flush(); index.flush(); curve.flush();
        s.seconds = elapsed(); s.deal_rng = rng_text(deal_rng); s.rng = rng_text(rng);
        s.log_bytes = file_size_or_zero(log_path); s.index_bytes = file_size_or_zero(index_path);
        s.curve_bytes = file_size_or_zero(curve_path);
        write_checkpoint(out, config, s, model);
    };
    const bool track = partition == Partition::Aswiss;
    uint64_t epoch_end = std::min(T, s.epoch_start + s.L);
    while (s.t < T) {
        ++s.t;
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const State root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        for (int p = 0; p < 2; ++p) { model.traverse(root, p, static_cast<double>(s.t), rng, track); model.apply_regrets(); }
        const double gap = model.take_traversal_gap() / 2;  // mean of the two seats, this root
        s.curve_sum += gap; s.curve_sq += gap * gap; ++s.curve_roots;
        if (curve_every && s.t % curve_every == 0) {
            const double n = s.curve_roots, mean = s.curve_sum / n;
            const double se = n > 1 ? std::sqrt(std::max(0.0, (s.curve_sq - n * mean * mean) / (n - 1) / n)) : 0.0;
            curve << s.t << ',' << elapsed() << ',' << s.curve_roots << ',' << mean << ',' << se << ',' << live_total() << ','
                  << peak_rss_mb() << '\n';
            s.curve_sum = s.curve_sq = 0; s.curve_roots = 0;
        }
        const bool epoch_done = s.t == epoch_end;
        if (snapshot_every && s.t % snapshot_every == 0 && !epoch_done) {
            const std::string file = "snapshot_r" + std::to_string(s.t) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",periodic," << s.k << ',' << live_total() << ',' << elapsed() << '\n';
        }
        if (epoch_done) {
            const std::string file = "snapshot_" + std::to_string(s.k) + ".bin";
            model.save(out + "/" + file);
            index << file << ',' << s.t << ",epoch," << s.k << ',' << live_total() << ',' << elapsed() << '\n';
            const auto w = model.window();
            const auto mb = model.table_megabytes();  // measured before the window is cleared
            const int splits = partition == Partition::Aswiss ? model.split(z, inherit) : 0;
            model.clear_window();
            const auto live = model.live_buckets();
            const double len = static_cast<double>(s.t - s.epoch_start), secs = elapsed();
            const auto cache = atlas ? atlas->assignment_cache_stats() : PowerCacheStats{};
            log << s.k << ',' << s.t << ',' << model.contexts() << ',' << live[0] + live[1] + live[2] << ',' << live[0] << ','
                << live[1] << ',' << live[2] << ',' << splits << ',' << w.atoms << ',' << w.L / len << ',' << w.C / len << ','
                << (w.L - w.C) / len << ',' << secs << ',' << cache.hits << ',' << cache.misses << ','
                << model.bucket_records() << ',' << mb[0] << ',' << mb[1] << ',' << mb[2] << ',' << peak_rss_mb() << '\n';
            std::cerr << "epoch " << s.k << " roots=" << s.t << " buckets_after=" << live[0] + live[1] + live[2]
                      << " splits=" << splits << " atoms=" << w.atoms << " H/root=" << (w.L - w.C) / len << " (" << secs << "s)\n";
            s.epoch_start = s.t; ++s.k; s.L *= 2; epoch_end = std::min(T, s.t + s.L);
        }
        if (epoch_done || (checkpoint_every && s.t % checkpoint_every == 0) || s.t == T) checkpoint();
        if (s.t % 1000 == 0 && std::filesystem::exists(stop_file)) g_stop = 1;
        if (g_stop) {
            checkpoint();
            std::cerr << "stopped at root " << s.t << "; checkpoint saved, rerun the same command to resume\n";
            return 130;
        }
    }
    return 0;
}

int eval(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--partition", "aswiss"}, {"--atlas", ""}, {"--snapshot", ""}, {"--policy", "average"},
                                {"--pairs", "2000"}, {"--particles", "64"}, {"--gap-roots", "4000"},
                                {"--seed", "2026102701"}, {"--power-cache", "1000000"}, {"--pairs-csv", ""}});
    const auto atlas = load_atlas(o["--atlas"], std::stoull(o["--power-cache"]));
    Model model(parse_partition(o["--partition"]), atlas.get());
    model.load(o.at("--snapshot"));
    const bool average = o["--policy"] == "average";
    if (!average && o["--policy"] != "current") throw std::runtime_error("--policy must be average or current");
    const uint64_t seed = std::stoull(o["--seed"]);
    SnapshotPolicy target{model, average, std::mt19937_64(seed ^ 0x5851f42d4c957f2dull)};
    PolicyLBR<SnapshotPolicy> lbr(target, std::stoi(o["--particles"]), seed ^ 0x94d049bb133111ebull);
    std::mt19937_64 deal_rng(seed);
    const int pairs = std::stoi(o["--pairs"]);
    std::vector<double> paired;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < pairs; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const double first = play_hand_policy_lbr(deck, 0, target, lbr, kAnte, kEffectiveStackAnte);
        const double second = play_hand_policy_lbr(deck, 1, target, lbr, kAnte, kEffectiveStackAnte);
        paired.push_back((first + second) / 2.0);
    }
    const double lbr_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto mean_se = [](const std::vector<double>& x) {
        const double m = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double v = 0;
        for (double y : x) v += (y - m) * (y - m);
        return std::make_pair(m, x.size() > 1 ? std::sqrt(v / (x.size() - 1) / x.size()) : 0.0);
    };
    if (!o["--pairs-csv"].empty()) {
        std::ofstream csv(o["--pairs-csv"]);
        csv << "pair,lbr_ante\n";
        for (size_t i = 0; i < paired.size(); ++i) csv << i << ',' << paired[i] << '\n';
    }
    // sampled BR-gap proxy on fixed deals (independent of the LBR deals)
    std::mt19937_64 gap_deals(seed ^ 0xd1b54a32d192ed03ull), gap_rng(seed ^ 0x2545f4914f6cdd1dull);
    std::vector<double> gaps;
    const int gap_roots = std::stoi(o["--gap-roots"]);
    for (int i = 0; i < gap_roots; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), gap_deals);
        const State root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        double gap0 = 0, gap1 = 0;
        gap_traverse(model, average, root, 0, gap_rng, gap0);
        gap_traverse(model, average, root, 1, gap_rng, gap1);
        gaps.push_back((gap0 + gap1) / 2.0);
    }
    const auto [lm, lse] = mean_se(paired);
    const auto [gm, gse] = mean_se(gaps);
    const auto stats = lbr.stats();
    std::cout << std::setprecision(9) << "{\"snapshot\":\"" << o["--snapshot"] << "\",\"policy\":\"" << o["--policy"]
              << "\",\"pairs\":" << pairs << ",\"particles\":" << o["--particles"] << ",\"seed\":" << seed
              << ",\"lbr_ante\":" << lm << ",\"lbr_se\":" << lse << ",\"gap_roots\":" << gap_roots
              << ",\"gap_proxy_ante\":" << gm << ",\"gap_proxy_se\":" << gse << ",\"policy_queries\":" << stats.policy_queries
              << ",\"policy_misses\":" << stats.policy_misses << ",\"lbr_seconds\":" << lbr_seconds << "}\n";
    return 0;
}

int self_test(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--atlas", ""}});
    const auto atlas = load_atlas(o.at("--atlas"), 1000000);
    std::mt19937_64 deal_rng(7), rng(8);
    Model aswiss(Partition::Aswiss, atlas.get()), atom(Partition::Atom, atlas.get()), context(Partition::Context, nullptr);
    NullAtlas null_atlas;
    for (uint64_t t = 1; t <= 3000; ++t) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        const State root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        // atom keys are the engine's hard power keys; context keys equal them with the cluster cleared
        State s = root;
        while (!s.terminal) {
            const InfoKey engine = make_power_key(s, s.actor, *atlas);
            if (!(aswiss.atom_key(s, s.actor) == engine)) throw std::runtime_error("atom key differs from engine key");
            if (!(context.atom_key(s, s.actor) == context_of(engine))) throw std::runtime_error("context key mismatch");
            step(s, s.actor, sample_action(uniform_strategy(valid_mask(s, s.actor)), valid_mask(s, s.actor), rng));
        }
        for (Model* m : {&aswiss, &atom, &context})
            for (int p = 0; p < 2; ++p) { m->traverse(root, p, static_cast<double>(t), rng, m == &aswiss); m->apply_regrets(); }
        if (t % 1000 == 0) {
            const auto before = aswiss.live_buckets();
            aswiss.split(1.0);
            aswiss.clear_window();
            const auto after = aswiss.live_buckets();
            std::cerr << "split at " << t << ": " << before[0] + before[1] + before[2] << " -> " << after[0] + after[1] + after[2] << "\n";
        }
    }
    // snapshot round trip: identical policies on fresh states
    for (Model* m : {&aswiss, &atom, &context}) {
        const Partition part = m == &aswiss ? Partition::Aswiss : m == &atom ? Partition::Atom : Partition::Context;
        m->save("aswiss_stud_self_test.bin");
        Model loaded(part, part == Partition::Context ? nullptr : atlas.get());
        loaded.load("aswiss_stud_self_test.bin");
        for (int i = 0; i < 300; ++i) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            State s = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
            while (!s.terminal) {
                for (bool average : {true, false}) {
                    bool f1 = false, f2 = false;
                    if (m->strategy(s, s.actor, average, &f1) != loaded.strategy(s, s.actor, average, &f2) || f1 != f2)
                        throw std::runtime_error("snapshot round trip changed a policy");
                }
                const auto sigma = m->strategy(s, s.actor, true);
                step(s, s.actor, sample_action(sigma, valid_mask(s, s.actor), rng));
            }
        }
    }
    std::remove("aswiss_stud_self_test.bin");
    std::cerr << "self-test passed; buckets aswiss/atom/context = " << aswiss.live_buckets()[0] + aswiss.live_buckets()[1] + aswiss.live_buckets()[2]
              << "/" << atom.live_buckets()[0] + atom.live_buckets()[1] + atom.live_buckets()[2] << "/"
              << context.live_buckets()[0] + context.live_buckets()[1] + context.live_buckets()[2] << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "train") return train(argc, argv);
        if (mode == "eval") return eval(argc, argv);
        if (mode == "self-test") return self_test(argc, argv);
        throw std::runtime_error("usage: aswiss_stud train|eval|self-test [options]");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
