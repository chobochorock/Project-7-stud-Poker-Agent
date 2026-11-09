// ESCHER (McAleer, Farina, Lanctot, Sandholm, ICLR 2023, arXiv 2206.04122), templated on the game like mmd.hpp.
//
// Iteration t:
//  1. history value network: `value_hands` hands with the current profile sigma_t for both seats, except that each
//     decision takes a uniform random action with probability value_explore; only the histories after a hand's last
//     such exploratory action are kept (their continuation followed sigma_t, so the return is an unbiased sample of
//     v_sigma_t there), so the network also learns the values after actions sigma_t rarely takes, which step 2 needs.
//     The network maps the full history (both players' private information) to player 0's return (/ reward_scale);
//     `value_steps` minibatch Adam steps on this iteration's data only, from a fresh initialization each iteration when
//     value_reinit (the official code's default; the paper's Algorithm 2 line 4 "retrain q on data from pi"), else
//     warm-started. The paper (p.18) and the official code mix 0.01 uniform into the policy for these hands; the code
//     corrects the targets with importance ratios, this keeps only the histories after the last exploratory action.
//  2. for each player i, `traversals` hands in which i acts from a fixed sampling policy (uniform over legal actions)
//     and the opponent from sigma_t. At each of i's information sets on the path, with history h:
//         r(a) = q(h,a) - sum_b sigma_t(b) q(h,b),   q(h,a) = V(h.a) (the exact utility when h.a is terminal),
//     stored with weight t in player i's regret reservoir. No importance weights: i's sampling policy is fixed, so
//     its reach multiplies every iteration's regret of an information set by the same constant, which regret
//     matching ignores. At the opponent's information sets sigma_t is stored with weight t in the strategy reservoir.
//  3. player i's regret network is retrained (from a fresh initialization by default, as Deep CFR) on its reservoir
//     with loss sum_t t * ||R(x) - r||^2 over legal actions; sigma_{t+1} = regret matching on R.
// The average policy network (softmax) is trained on the strategy reservoir (weighted cross-entropy) when a snapshot
// is taken, with its own RNG (taking a snapshot does not change the training); it is the policy the tabular guarantee
// is about. The current policy is regret matching on R; when no predicted regret is positive it plays the action with
// the largest one (fallback "argmax": Deep CFR and the official ESCHER code) or the uniform policy ("uniform").
//
// Additional game interface: void history_features(const State&, Sparse&) const; int history_dim() const.
#pragma once
#include <array>
#include <numeric>
#include "nn.hpp"
#include "remote_train.hpp"

namespace sota {

struct EscherConfig {
    int hidden1 = 128, hidden2 = 64;
    int value_hands = 1024, value_steps = 200, traversals = 1024, regret_steps = 400, avg_steps = 2000, batch = 256;
    size_t regret_capacity = 1000000, strategy_capacity = 1000000;
    float lr = 1e-3f, value_lr = 1e-3f;
    double reward_scale = 1.0, value_explore = 0.01;
    bool reinit = true, value_reinit = true;
    std::string fallback = "argmax";  // argmax | uniform
    uint64_t seed = 1;
};

template <class S>
struct Reservoir {  // uniform reservoir sampling (Vitter's algorithm R)
    std::vector<S> data;
    size_t capacity = 0;
    long long seen = 0;
    long long add(S s, std::mt19937_64& rng) {  // the slot written, -1 when the sample was not kept
        ++seen;
        if (data.size() < capacity) { data.push_back(std::move(s)); return static_cast<long long>(data.size()) - 1; }
        const long long j = std::uniform_int_distribution<long long>(0, seen - 1)(rng);
        if (j < static_cast<long long>(capacity)) { data[static_cast<size_t>(j)] = std::move(s); return j; }
        return -1;
    }
};

template <class Game>
struct Escher {
    static constexpr int A = Game::A;
    const Game& game;
    EscherConfig cfg;
    Mlp value, regret[2], avg;
    Adam opt_v, opt_r[2];
    std::mt19937_64 rng;
    long long t = 0, roots_done = 0, regret_samples = 0;
    double last_value_loss = 0, last_regret_loss[2] = {0, 0};
    struct RegretSample { Sparse x; uint8_t mask; std::array<float, A> v; float w; };  // v: sampled regrets
    struct StratSample { Sparse x; uint8_t mask; std::array<float, A> v; float w; };   // v: sigma_t
    struct ValueSample { Sparse h; float ret; };
    Reservoir<RegretSample> rmem[2];
    Reservoir<StratSample> smem;
    // --remote-train (remote_train.hpp): networks trained in another process; the reservoir slots written since the
    // last training request are sent before it (0, 1: regret reservoirs, 2: strategy reservoir)
    RemoteTrainer* remote = nullptr;
    mutable std::vector<uint32_t> dirty[3];
    mutable std::vector<uint8_t> is_dirty[3];
    void mark(int r, long long slot) const {
        if (!remote || slot < 0) return;
        auto& f = is_dirty[r];
        if (f.size() <= static_cast<size_t>(slot)) f.resize(static_cast<size_t>(slot) + 1, 0);
        if (!f[slot]) { f[slot] = 1; dirty[r].push_back(static_cast<uint32_t>(slot)); }
    }
    template <class Sample> void put_sample(const Sample& x) const {
        remote->put_sparse(x.x); remote->put<uint8_t>(x.mask);
        for (int a = 0; a < A; ++a) remote->put<float>(x.v[a]);
        remote->put<float>(x.w);
    }
    void sync(int r) const {  // buffer a RES_UPDATE of reservoir r's written slots (sent with the next request)
        if (dirty[r].empty()) return;
        remote->put<uint32_t>(RemoteTrainer::RES_UPDATE); remote->put<uint8_t>(static_cast<uint8_t>(r));
        remote->put<uint32_t>(static_cast<uint32_t>(dirty[r].size()));
        for (uint32_t slot : dirty[r]) {
            remote->put<uint32_t>(slot);
            if (r < 2) put_sample(rmem[r].data[slot]); else put_sample(smem.data[slot]);
            is_dirty[r][slot] = 0;
        }
        dirty[r].clear();
    }
    // connect a remote trainer (after a checkpoint, if any, is loaded): handshake, then every stored sample is dirty
    void attach(RemoteTrainer* r) {
        remote = r;
        remote->put<uint32_t>(RemoteTrainer::HELLO);
        for (int v : {game.input_dim(), game.history_dim(), cfg.hidden1, cfg.hidden2, A, cfg.batch})
            remote->put<uint32_t>(static_cast<uint32_t>(v));
        remote->put<uint32_t>(static_cast<uint32_t>(cfg.regret_capacity)); remote->put<uint32_t>(static_cast<uint32_t>(cfg.strategy_capacity));
        remote->put<uint64_t>(cfg.seed);
        remote->send();
        remote->expect(RemoteTrainer::HELLO);
        if (remote->get<uint32_t>() != RemoteTrainer::kVersion) throw std::runtime_error("remote trainer: protocol version mismatch");
        for (int i = 0; i < 2; ++i) for (size_t k = 0; k < rmem[i].data.size(); ++k) mark(i, static_cast<long long>(k));
        for (size_t k = 0; k < smem.data.size(); ++k) mark(2, static_cast<long long>(k));
    }
    void put_train_head(uint32_t steps, float lr) const {
        remote->put<uint32_t>(static_cast<uint32_t>(t)); remote->put<uint32_t>(steps);
        remote->put<uint32_t>(static_cast<uint32_t>(cfg.batch)); remote->put<float>(lr);
    }

    Escher(const Game& g, const EscherConfig& c) : game(g), cfg(c), rng(c.seed) {
        value = Mlp(g.history_dim(), c.hidden1, c.hidden2, 1, c.seed * 7 + 1);
        for (int i = 0; i < 2; ++i) {
            regret[i] = Mlp(g.input_dim(), c.hidden1, c.hidden2, A, c.seed * 7 + 2 + i);
            opt_r[i].lr = c.lr; opt_r[i].init(regret[i]);
            rmem[i].capacity = c.regret_capacity;
        }
        avg = Mlp(g.input_dim(), c.hidden1, c.hidden2, A, c.seed * 7 + 5);
        opt_v.lr = c.value_lr; opt_v.init(value);
        smem.capacity = c.strategy_capacity;
    }

    // regret matching on the predicted regrets (fallback when none is positive: see cfg.fallback)
    void current(int seat, const Sparse& x, uint8_t mask, double* p) const {
        Mlp::Cache c; regret[seat].forward(x, c);
        double z = 0;
        for (int a = 0; a < A; ++a) { p[a] = (mask >> a & 1) ? std::max(0.0, static_cast<double>(c.y[a])) : 0.0; z += p[a]; }
        if (z > 0) { for (int a = 0; a < A; ++a) p[a] /= z; return; }
        if (cfg.fallback == "argmax") {
            int best = -1;
            for (int a = 0; a < A; ++a) if ((mask >> a & 1) && (best < 0 || c.y[a] > c.y[best])) best = a;
            for (int a = 0; a < A; ++a) p[a] = a == best ? 1.0 : 0.0;
            return;
        }
        int n = 0;
        for (int a = 0; a < A; ++a) n += mask >> a & 1;
        for (int a = 0; a < A; ++a) p[a] = (mask >> a & 1) ? 1.0 / n : 0.0;
    }
    void average(const Sparse& x, uint8_t mask, double* p) const {
        Mlp::Cache c; avg.forward(x, c); masked_softmax(c.y.data(), mask, A, p);
    }
    int sample(const double* p, uint8_t mask) {
        double r = std::uniform_real_distribution<double>(0, 1)(rng), acc = 0; int last = -1;
        for (int a = 0; a < A; ++a) if (mask >> a & 1) { acc += p[a]; last = a; if (r < acc) return a; }
        return last;
    }
    double value_of(const typename Game::State& s, int traverser) const {  // V for the traverser
        if (game.terminal(s)) return game.utility(s, traverser);
        Sparse h; game.history_features(s, h);
        Mlp::Cache c; value.forward(h, c);
        const double v0 = c.y[0] * cfg.reward_scale;
        return traverser == 0 ? v0 : -v0;
    }

    void train_value() {
        std::vector<ValueSample> data;
        for (int e = 0; e < cfg.value_hands; ++e) {
            auto s = game.root(rng);
            size_t first = data.size();
            while (!game.terminal(s)) {
                const int pl = game.actor(s);
                if (pl < 0) { game.chance(s, rng); continue; }
                const uint8_t mask = game.mask(s);
                ValueSample v; game.history_features(s, v.h); data.push_back(std::move(v));
                if (std::uniform_real_distribution<double>(0, 1)(rng) < cfg.value_explore) {
                    int n = 0;
                    for (int b = 0; b < A; ++b) n += mask >> b & 1;
                    const int pickn = std::uniform_int_distribution<int>(0, n - 1)(rng);
                    int a = -1;
                    for (int b = 0, c = 0; b < A; ++b) if (mask >> b & 1) { if (c == pickn) { a = b; break; } ++c; }
                    data.resize(first);  // this history and the earlier ones are not followed by sigma_t
                    game.step(s, a);
                    continue;
                }
                Sparse x; game.features(s, pl, x);
                double p[A]; current(pl, x, mask, p);
                game.step(s, sample(p, mask));
            }
            const double u0 = game.utility(s, 0);
            for (size_t i = first; i < data.size(); ++i) data[i].ret = static_cast<float>(u0 / cfg.reward_scale);
        }
        roots_done += cfg.value_hands;
        if (cfg.value_reinit) {
            value = Mlp(game.history_dim(), cfg.hidden1, cfg.hidden2, 1, cfg.seed * 7 + 1 + 100 * static_cast<uint64_t>(t));
            opt_v.init(value);
        }
        if (data.empty()) return;
        if (remote) {
            remote->put<uint32_t>(RemoteTrainer::VALUE_TRAIN); put_train_head(static_cast<uint32_t>(cfg.value_steps), cfg.value_lr);
            remote->put<uint8_t>(cfg.value_reinit ? 1 : 0); remote->put_mlp(value);
            remote->put<uint32_t>(static_cast<uint32_t>(data.size()));
            for (const ValueSample& v : data) { remote->put_sparse(v.h); remote->put<float>(v.ret); }
            value = remote->train(RemoteTrainer::VALUE_TRAIN, last_value_loss);
            return;
        }
        Mlp::Grad g; g.init(value);
        double loss = 0; long long n = 0;
        std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
        for (int st = 0; st < cfg.value_steps; ++st) {
            for (int b = 0; b < cfg.batch; ++b) {
                const ValueSample& v = data[pick(rng)];
                Mlp::Cache c; value.forward(v.h, c);
                const float d = c.y[0] - v.ret, dy = 2.0f * d;
                value.backward(v.h, c, &dy, g);
                if (st == cfg.value_steps - 1) { loss += d * d; ++n; }
            }
            opt_v.step(value, g, 1.0f / static_cast<float>(cfg.batch));
        }
        last_value_loss = loss / std::max<long long>(1, n);
    }

    void traverse(int i) {
        const float w = static_cast<float>(t);
        for (int k = 0; k < cfg.traversals; ++k) {
            auto s = game.root(rng);
            while (!game.terminal(s)) {
                const int pl = game.actor(s);
                if (pl < 0) { game.chance(s, rng); continue; }
                const uint8_t mask = game.mask(s);
                Sparse x; game.features(s, pl, x);
                double sigma[A]; current(pl, x, mask, sigma);
                if (pl == i) {
                    double q[A] = {}, node = 0;
                    for (int a = 0; a < A; ++a) {
                        if (!(mask >> a & 1)) continue;
                        auto child = s; game.step(child, a);
                        q[a] = value_of(child, i); node += sigma[a] * q[a];
                    }
                    RegretSample rs; rs.x = x; rs.mask = mask; rs.w = w;
                    // in units of reward_scale, like the value network (regret matching ignores the scale)
                    for (int a = 0; a < A; ++a) rs.v[a] = (mask >> a & 1) ? static_cast<float>((q[a] - node) / cfg.reward_scale) : 0.0f;
                    mark(i, rmem[i].add(std::move(rs), rng)); ++regret_samples;
                    int n = 0;
                    for (int a = 0; a < A; ++a) n += mask >> a & 1;
                    const int pickn = std::uniform_int_distribution<int>(0, n - 1)(rng);  // uniform sampling policy
                    int a = -1;
                    for (int b = 0, c = 0; b < A; ++b) if (mask >> b & 1) { if (c == pickn) { a = b; break; } ++c; }
                    game.step(s, a);
                } else {
                    StratSample ss; ss.x = x; ss.mask = mask; ss.w = w;
                    for (int a = 0; a < A; ++a) ss.v[a] = static_cast<float>(sigma[a]);
                    mark(2, smem.add(std::move(ss), rng));
                    game.step(s, sample(sigma, mask));
                }
            }
        }
        roots_done += cfg.traversals;
    }

    void train_regret(int i) {
        if (cfg.reinit) { regret[i] = Mlp(game.input_dim(), cfg.hidden1, cfg.hidden2, A, cfg.seed * 7 + 2 + i + 100 * static_cast<uint64_t>(t)); opt_r[i].init(regret[i]); }
        auto& mem = rmem[i].data;
        if (mem.empty()) return;
        if (remote) {
            sync(i);
            remote->put<uint32_t>(RemoteTrainer::REGRET_TRAIN); remote->put<uint8_t>(static_cast<uint8_t>(i));
            put_train_head(static_cast<uint32_t>(cfg.regret_steps), cfg.lr); remote->put<uint8_t>(cfg.reinit ? 1 : 0);
            remote->put_mlp(regret[i]);
            regret[i] = remote->train(RemoteTrainer::REGRET_TRAIN, last_regret_loss[i]);
            return;
        }
        Mlp::Grad g; g.init(regret[i]);
        std::uniform_int_distribution<size_t> pick(0, mem.size() - 1);
        double loss = 0, wsum = 0;
        for (int st = 0; st < cfg.regret_steps; ++st) {
            double bw = 0;
            for (int b = 0; b < cfg.batch; ++b) {
                const RegretSample& rs = mem[pick(rng)];
                Mlp::Cache c; regret[i].forward(rs.x, c);
                float dy[A] = {};
                const float wn = rs.w / static_cast<float>(t);  // weights t'/t in (0, 1]
                for (int a = 0; a < A; ++a) if (rs.mask >> a & 1) {
                    const float d = c.y[a] - rs.v[a];
                    dy[a] = 2.0f * wn * d;
                    if (st == cfg.regret_steps - 1) { loss += wn * d * d; }
                }
                if (st == cfg.regret_steps - 1) wsum += wn;
                bw += wn;
                regret[i].backward(rs.x, c, dy, g);
            }
            opt_r[i].step(regret[i], g, 1.0f / static_cast<float>(cfg.batch));
            (void)bw;
        }
        last_regret_loss[i] = loss / std::max(1e-12, wsum);
    }

    void iterate() {
        ++t;
        train_value();
        for (int i = 0; i < 2; ++i) { traverse(i); train_regret(i); }
    }

    // The average policy for monitoring (online LBR) without touching the training: its own network, optimizer and RNG,
    // warm-started across calls, `steps` steps of the same weighted cross-entropy on the strategy reservoir (read only).
    void train_average_monitor(Mlp& net, Adam& opt, std::mt19937_64& r, int steps) const {
        const auto& mem = smem.data;
        if (mem.empty() || steps <= 0) return;
        if (remote) {  // the remote side keeps this network's optimizer across calls (opt and r are not used)
            sync(2);
            remote->put<uint32_t>(RemoteTrainer::AVG_TRAIN); remote->put<uint8_t>(1);
            put_train_head(static_cast<uint32_t>(steps), opt.lr); remote->put<uint64_t>(cfg.seed ^ 0x6a09e667f3bcc908ull);
            remote->put_mlp(net);
            double loss = 0;
            net = remote->train(RemoteTrainer::AVG_TRAIN, loss);
            return;
        }
        Mlp::Grad g; g.init(net);
        std::uniform_int_distribution<size_t> pick(0, mem.size() - 1);
        for (int st = 0; st < steps; ++st) {
            for (int b = 0; b < cfg.batch; ++b) {
                const StratSample& ss = mem[pick(r)];
                Mlp::Cache c; net.forward(ss.x, c);
                double p[A]; masked_softmax(c.y.data(), ss.mask, A, p);
                const float wn = ss.w / static_cast<float>(t);
                float dy[A] = {};
                for (int a = 0; a < A; ++a) if (ss.mask >> a & 1) dy[a] = wn * static_cast<float>(p[a] - ss.v[a]);
                net.backward(ss.x, c, dy, g);
            }
            opt.step(net, g, 1.0f / static_cast<float>(cfg.batch));
        }
    }

    // average policy network from the strategy reservoir (fresh, weighted cross-entropy, its own RNG: the training's
    // RNG is untouched, so snapshots at any roots leave the training unchanged)
    void train_average() {
        avg = Mlp(game.input_dim(), cfg.hidden1, cfg.hidden2, A, cfg.seed * 7 + 5 + 1000 * static_cast<uint64_t>(t));
        Adam opt; opt.lr = cfg.lr; opt.init(avg);
        auto& mem = smem.data;
        if (mem.empty()) return;
        if (remote) {
            sync(2);
            remote->put<uint32_t>(RemoteTrainer::AVG_TRAIN); remote->put<uint8_t>(0);
            put_train_head(static_cast<uint32_t>(cfg.avg_steps), cfg.lr);
            remote->put<uint64_t>(cfg.seed ^ 0x2b7e151628aed2a6ull ^ (0x9e3779b97f4a7c15ull * static_cast<uint64_t>(t + 1)));
            remote->put_mlp(avg);
            double loss = 0;
            avg = remote->train(RemoteTrainer::AVG_TRAIN, loss);
            return;
        }
        std::mt19937_64 r(cfg.seed ^ 0x2b7e151628aed2a6ull ^ (0x9e3779b97f4a7c15ull * static_cast<uint64_t>(t + 1)));
        Mlp::Grad g; g.init(avg);
        std::uniform_int_distribution<size_t> pick(0, mem.size() - 1);
        for (int st = 0; st < cfg.avg_steps; ++st) {
            for (int b = 0; b < cfg.batch; ++b) {
                const StratSample& ss = mem[pick(r)];
                Mlp::Cache c; avg.forward(ss.x, c);
                double p[A]; masked_softmax(c.y.data(), ss.mask, A, p);
                const float wn = ss.w / static_cast<float>(t);
                float dy[A] = {};
                for (int a = 0; a < A; ++a) if (ss.mask >> a & 1) dy[a] = wn * static_cast<float>(p[a] - ss.v[a]);  // d CE / d logits
                avg.backward(ss.x, c, dy, g);
            }
            opt.step(avg, g, 1.0f / static_cast<float>(cfg.batch));
        }
    }
};

}  // namespace sota
