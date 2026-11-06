// Neural SOTA baselines on the 7-stud game of aswiss_stud.cpp (heads-up, rules v3, 5th-7th street, heuristic H4),
// included by aswiss_stud.cpp after its own definitions. `train --partition net --algo mmd` trains; `eval --partition
// net` evaluates a network snapshot with the same Policy-LBR deals, LBR seed, particles and gap deals as every other
// arm. Network input: the engine's Deep CFR information tensor (deep_cfr_tensor: own hidden/up cards, opponent up
// cards, own discard, betting history, pot features, legal mask), passed sparsely.
// Outputs follow the tabular trainer: log.csv (one row per snapshot), curve.csv (every --curve-every roots; the
// training-gap columns are nan), snapshots.csv, snapshot_K.net / snapshot_rN.net, checkpoint.net (resume by rerunning
// the same command), OUT/STOP to stop. One root = one self-play hand from a sampled 5th-street root.
#pragma once
#include "mmd.hpp"
#include "escher.hpp"

namespace sota {

struct Stud7Game {
    static constexpr int A = kActionCount;
    using State = ::State;
    State root(std::mt19937_64& rng) const {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        return sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
    }
    bool terminal(const State& s) const { return s.terminal; }
    double utility(const State& s, int seat) const { return terminal_net_search(s, seat); }
    int actor(const State& s) const { return s.actor; }
    void chance(State&, std::mt19937_64&) const { throw std::runtime_error("7-stud deals inside step"); }
    uint8_t mask(const State& s) const { return valid_mask(s, s.actor); }
    void step(State& s, int a) const { ::step(s, s.actor, static_cast<Action>(a)); }
    void features(const State& s, int seat, Sparse& x) const {
        const DeepCfrTensor t = deep_cfr_tensor(s, seat);
        x.clear();
        for (int i = 0; i < kDeepCfrTensorSize; ++i) if (t[i] != 0.0f) x.x.emplace_back(i, t[i]);
    }
    int input_dim() const { return kDeepCfrTensorSize; }
    // full history for ESCHER's value network: both seats' tensors side by side (both players' private cards)
    void history_features(const State& s, Sparse& x) const {
        x.clear();
        for (int seat = 0; seat < 2; ++seat) {
            const DeepCfrTensor t = deep_cfr_tensor(s, seat);
            for (int i = 0; i < kDeepCfrTensorSize; ++i) if (t[i] != 0.0f) x.x.emplace_back(seat * kDeepCfrTensorSize + i, t[i]);
        }
    }
    int history_dim() const { return 2 * kDeepCfrTensorSize; }
};

// PolicyLBR target backed by networks: a softmax policy network, or (regret[0] set) regret matching on per-seat
// regret networks
struct NetPolicy {
    const Mlp& net;
    mutable std::mt19937_64 rng;
    const Mlp* regret[2] = {nullptr, nullptr};
    std::array<double, kActionCount> policy(const State& state, int seat, bool* found = nullptr) const {
        if (found) *found = true;
        Sparse x;
        Stud7Game{}.features(state, seat, x);
        const uint8_t mask = valid_mask(state, seat);
        std::array<double, kActionCount> p{};
        if (regret[0]) {
            Mlp::Cache c; regret[seat]->forward(x, c);
            double z = 0; int best = -1;
            for (int a = 0; a < kActionCount; ++a) if (mask >> a & 1) {
                p[a] = std::max(0.0, static_cast<double>(c.y[a])); z += p[a];
                if (best < 0 || c.y[a] > c.y[best]) best = a;
            }
            // as Escher::current with the default fallback: the largest predicted regret when none is positive
            for (int a = 0; a < kActionCount; ++a) p[a] = z > 0 ? p[a] / z : (a == best ? 1.0 : 0.0);
            return p;
        }
        Mlp::Cache c; net.forward(x, c);
        masked_softmax(c.y.data(), mask, kActionCount, p.data());
        return p;
    }
    Action choose(const State& state, int seat, int) { return sample_action(policy(state, seat), valid_mask(state, seat), rng); }
    bool learned_h4() const { return false; }
    double h4_weight(const std::array<Card, 2>&, Card, Card, int) const { throw std::runtime_error("network policies use the heuristic H4"); }
};

inline double gap_traverse_net(const NetPolicy& pol, State state, int traverser, std::mt19937_64& rng, double& gap) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    const auto sigma = pol.policy(state, actor);
    if (actor != traverser) {
        step(state, actor, sample_action(sigma, mask, rng));
        return gap_traverse_net(pol, std::move(state), traverser, rng, gap);
    }
    double node = 0, best = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < kActionCount; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        const double v = gap_traverse_net(pol, std::move(child), traverser, rng, gap);
        node += sigma[a] * v; best = std::max(best, v);
    }
    gap += best - node;
    return node;
}

// --- persistence: snapshot = policy network; checkpoint = whole trainer state ---
inline void save_adam(std::ofstream& out, const Adam& o) {
    out.write(reinterpret_cast<const char*>(&o.t), sizeof(o.t));
    for (int i = 0; i < 6; ++i) {
        out.write(reinterpret_cast<const char*>(o.m[i].data()), static_cast<std::streamsize>(o.m[i].size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(o.v[i].data()), static_cast<std::streamsize>(o.v[i].size() * sizeof(float)));
    }
}
inline void load_adam(std::ifstream& in, Adam& o) {
    in.read(reinterpret_cast<char*>(&o.t), sizeof(o.t));
    for (int i = 0; i < 6; ++i) {
        in.read(reinterpret_cast<char*>(o.m[i].data()), static_cast<std::streamsize>(o.m[i].size() * sizeof(float)));
        in.read(reinterpret_cast<char*>(o.v[i].data()), static_cast<std::streamsize>(o.v[i].size() * sizeof(float)));
    }
}
constexpr char kNetMagic[8] = {'S', 'T', 'U', 'D', 'N', 'E', 'T', '1'};
inline void save_snapshot(const std::string& path, const Mlp& pol) {
    std::ofstream out(path, std::ios::binary);
    out.write(kNetMagic, 8);
    save_mlp(out, pol);
}
// ESCHER snapshot: average policy network, then the two regret networks
constexpr char kEscherMagic[8] = {'S', 'T', 'U', 'D', 'E', 'S', 'C', '1'};
inline void save_escher_snapshot(const std::string& path, const Mlp& avg, const Mlp& r0, const Mlp& r1) {
    std::ofstream out(path, std::ios::binary);
    out.write(kEscherMagic, 8);
    save_mlp(out, avg); save_mlp(out, r0); save_mlp(out, r1);
}
struct NetSnapshot { Mlp policy; bool escher = false; Mlp regret[2]; };
inline NetSnapshot load_snapshot(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    in.read(magic, 8);
    NetSnapshot s;
    if (in && std::string(magic, 8) == std::string(kNetMagic, 8)) { s.policy = load_mlp(in); return s; }
    if (in && std::string(magic, 8) == std::string(kEscherMagic, 8)) {
        s.escher = true; s.policy = load_mlp(in); s.regret[0] = load_mlp(in); s.regret[1] = load_mlp(in);
        return s;
    }
    throw std::runtime_error(path + " is not a network snapshot");
}

struct NetLoop {  // counters the checkpoint restores
    long long next_epoch_roots = 0, epoch_len = 0, next_snapshot = 0, next_curve = 0, next_checkpoint = 0;
    int k = 0;
    double seconds = 0;
    std::streamoff log_bytes = 0, index_bytes = 0, curve_bytes = 0;
    long long curve_roots0 = 0;
    double train_cpu = 0;    // training CPU seconds over all processes of the run (TrainCpuClock)
    double budget_mark = 0;  // the --budget-cpu-seconds at which the last budget stop happened (0: none)
};

// Training CPU clock of the network and ReBeL trainers (--budget-cpu-seconds, column train_cpu_seconds): CPU time of this
// process since training (re)started, minus the CPU time spent in the online LBR evaluation (which adds what it spends
// to `online`), plus the training CPU time of the earlier processes of the run (NetLoop::train_cpu). The trainers are
// single-threaded, so this is the CPU time the training itself used, whatever else ran on the machine.
struct TrainCpuClock {
    double base = 0, start = 0, online = 0;
    void begin(double saved) { base = saved; start = cpu_seconds(); online = 0; }
    double now() const { return base + (cpu_seconds() - start) - online; }
};

inline void save_checkpoint(const std::string& path, const std::string& config, const Mmd<Stud7Game>& t, const NetLoop& L,
                            OnlineLbr* online = nullptr) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const uint64_t n = config.size();
        out.write(reinterpret_cast<const char*>(&n), sizeof(n)); out.write(config.data(), static_cast<std::streamsize>(n));
        save_mlp(out, t.pol); save_mlp(out, t.val);
        const char has_magnet = t.magnet.in > 0;
        out.write(&has_magnet, 1);
        if (has_magnet) save_mlp(out, t.magnet);
        save_adam(out, t.opt_p); save_adam(out, t.opt_v);
        std::ostringstream rs; rs << t.rng;
        const std::string r = rs.str(); const uint64_t rn = r.size();
        out.write(reinterpret_cast<const char*>(&rn), sizeof(rn)); out.write(r.data(), static_cast<std::streamsize>(rn));
        for (long long v : {t.updates, t.episodes_done, t.samples_done}) out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        out.write(reinterpret_cast<const char*>(&L), sizeof(L));
        if (online && online->on()) online->save(out);
    }
    std::filesystem::rename(tmp, path);
}
inline bool load_checkpoint(const std::string& path, const std::string& config, Mmd<Stud7Game>& t, NetLoop& L,
                            OnlineLbr* online = nullptr, uint64_t* online_bytes = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::string stored(n, '\0');
    in.read(stored.data(), static_cast<std::streamsize>(n));
    if (stored != config) throw std::runtime_error("checkpoint was written with different settings:\n  " + stored + "\nnow:\n  " + config);
    t.pol = load_mlp(in); t.val = load_mlp(in);
    char has_magnet = 0;
    in.read(&has_magnet, 1);
    if (has_magnet) t.magnet = load_mlp(in);
    t.opt_p.init(t.pol); t.opt_v.init(t.val);
    load_adam(in, t.opt_p); load_adam(in, t.opt_v);
    uint64_t rn = 0;
    in.read(reinterpret_cast<char*>(&rn), sizeof(rn));
    std::string r(rn, '\0');
    in.read(r.data(), static_cast<std::streamsize>(rn));
    std::istringstream rs(r); rs >> t.rng;
    for (long long* v : {&t.updates, &t.episodes_done, &t.samples_done}) in.read(reinterpret_cast<char*>(v), sizeof(*v));
    in.read(reinterpret_cast<char*>(&L), sizeof(L));
    if (online && online->on()) *online_bytes = online->load(in);
    if (!in) throw std::runtime_error("truncated checkpoint " + path);
    return true;
}

// --- ESCHER checkpoint: networks, optimizers, reservoirs, RNG, counters ---
template <class T> void put_raw(std::ofstream& out, const T& v) { out.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void get_raw(std::ifstream& in, T& v) { in.read(reinterpret_cast<char*>(&v), sizeof(T)); }
inline void put_sparse(std::ofstream& out, const Sparse& x) {
    const uint32_t n = static_cast<uint32_t>(x.x.size());
    put_raw(out, n);
    for (const auto& [i, v] : x.x) { put_raw(out, static_cast<int32_t>(i)); put_raw(out, v); }
}
inline void get_sparse(std::ifstream& in, Sparse& x) {
    uint32_t n = 0; get_raw(in, n);
    x.x.resize(n);
    for (auto& [i, v] : x.x) { int32_t ii = 0; get_raw(in, ii); i = ii; get_raw(in, v); }
}
template <class S> void put_reservoir(std::ofstream& out, const Reservoir<S>& r) {
    put_raw(out, static_cast<uint64_t>(r.capacity)); put_raw(out, r.seen); put_raw(out, static_cast<uint64_t>(r.data.size()));
    for (const auto& s : r.data) {
        put_sparse(out, s.x); put_raw(out, s.mask); put_raw(out, s.w);
        out.write(reinterpret_cast<const char*>(s.v.data()), sizeof(s.v));
    }
}
template <class S> void get_reservoir(std::ifstream& in, Reservoir<S>& r) {
    uint64_t cap = 0, size = 0;
    get_raw(in, cap); get_raw(in, r.seen); get_raw(in, size);
    r.capacity = static_cast<size_t>(cap);
    r.data.resize(static_cast<size_t>(size));
    for (auto& s : r.data) {
        get_sparse(in, s.x); get_raw(in, s.mask); get_raw(in, s.w);
        in.read(reinterpret_cast<char*>(s.v.data()), sizeof(s.v));
    }
}
inline void put_rng(std::ofstream& out, const std::mt19937_64& rng) {
    std::ostringstream rs; rs << rng;
    const std::string r = rs.str(); const uint64_t rn = r.size();
    put_raw(out, rn); out.write(r.data(), static_cast<std::streamsize>(rn));
}
inline void get_rng(std::ifstream& in, std::mt19937_64& rng) {
    uint64_t rn = 0; get_raw(in, rn);
    std::string r(rn, '\0'); in.read(r.data(), static_cast<std::streamsize>(rn));
    std::istringstream rs(r); rs >> rng;
}
inline void save_checkpoint(const std::string& path, const std::string& config, const Escher<Stud7Game>& e, const NetLoop& L,
                            OnlineLbr* online = nullptr) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const uint64_t n = config.size();
        put_raw(out, n); out.write(config.data(), static_cast<std::streamsize>(n));
        save_mlp(out, e.value); save_mlp(out, e.regret[0]); save_mlp(out, e.regret[1]);
        save_adam(out, e.opt_v); save_adam(out, e.opt_r[0]); save_adam(out, e.opt_r[1]);
        put_reservoir(out, e.rmem[0]); put_reservoir(out, e.rmem[1]); put_reservoir(out, e.smem);
        put_rng(out, e.rng);
        for (long long v : {e.t, e.roots_done, e.regret_samples}) put_raw(out, v);
        put_raw(out, L);
        if (online && online->on()) online->save(out);
    }
    std::filesystem::rename(tmp, path);
}
inline bool load_checkpoint(const std::string& path, const std::string& config, Escher<Stud7Game>& e, NetLoop& L,
                            OnlineLbr* online = nullptr, uint64_t* online_bytes = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t n = 0; get_raw(in, n);
    std::string stored(n, '\0'); in.read(stored.data(), static_cast<std::streamsize>(n));
    if (stored != config) throw std::runtime_error("checkpoint was written with different settings:\n  " + stored + "\nnow:\n  " + config);
    e.value = load_mlp(in); e.regret[0] = load_mlp(in); e.regret[1] = load_mlp(in);
    e.opt_v.init(e.value); e.opt_r[0].init(e.regret[0]); e.opt_r[1].init(e.regret[1]);
    load_adam(in, e.opt_v); load_adam(in, e.opt_r[0]); load_adam(in, e.opt_r[1]);
    get_reservoir(in, e.rmem[0]); get_reservoir(in, e.rmem[1]); get_reservoir(in, e.smem);
    get_rng(in, e.rng);
    for (long long* v : {&e.t, &e.roots_done, &e.regret_samples}) get_raw(in, *v);
    get_raw(in, L);
    if (online && online->on()) *online_bytes = online->load(in);
    if (!in) throw std::runtime_error("truncated checkpoint " + path);
    return true;
}

}  // namespace sota
