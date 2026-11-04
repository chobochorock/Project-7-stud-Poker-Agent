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
};

// PolicyLBR target backed by a policy network
struct NetPolicy {
    const Mlp& net;
    mutable std::mt19937_64 rng;
    std::array<double, kActionCount> policy(const State& state, int seat, bool* found = nullptr) const {
        if (found) *found = true;
        Sparse x;
        Stud7Game{}.features(state, seat, x);
        Mlp::Cache c; net.forward(x, c);
        std::array<double, kActionCount> p{};
        masked_softmax(c.y.data(), valid_mask(state, seat), kActionCount, p.data());
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
inline Mlp load_snapshot(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    in.read(magic, 8);
    if (!in || std::string(magic, 8) != std::string(kNetMagic, 8)) throw std::runtime_error(path + " is not a network snapshot");
    return load_mlp(in);
}

struct NetLoop {  // counters the checkpoint restores
    long long next_epoch_roots = 0, epoch_len = 0, next_snapshot = 0, next_curve = 0, next_checkpoint = 0;
    int k = 0;
    double seconds = 0;
    std::streamoff log_bytes = 0, index_bytes = 0, curve_bytes = 0;
    long long curve_roots0 = 0;
};

inline void save_checkpoint(const std::string& path, const std::string& config, const Mmd<Stud7Game>& t, const NetLoop& L) {
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
    }
    std::filesystem::rename(tmp, path);
}
inline bool load_checkpoint(const std::string& path, const std::string& config, Mmd<Stud7Game>& t, NetLoop& L) {
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
    if (!in) throw std::runtime_error("truncated checkpoint " + path);
    return true;
}

}  // namespace sota
