// Magnetic mirror descent (MMD) self-play with a policy network and a value baseline, after Sokota et al. (arXiv
// 2206.05825) and the deep-RL setting of Rudolph et al. (ICLR 2026, arXiv 2502.08938), templated on the game.
//
// Each update collects `episodes` self-play hands with the current policy (both seats share the network; the seat is
// part of the input), then runs `epochs` passes of minibatch Adam on
//   L(theta) = E[-(pi_theta(a|x) / pi_old(a|x)) * A] + alpha * KL(pi_theta || uniform) + beta * KL(pi_theta || pi_old)
// where A is the Monte Carlo return of the acting seat minus the value network's estimate, normalized per batch, and
// both KL terms are exact over the legal actions of each sampled state (uniform = the magnet). The value network
// regresses returns / reward_scale. alpha follows alpha0 / sqrt(1 + update / alpha_halflife) (0: constant).
// magnet_every K > 0: the magnet is not uniform but a frozen copy of the policy network, replaced by the current one
// every K updates (as R-NaD moves its reference policy). With a fixed magnet the last iterate converges (tabular) to the
// regularized equilibrium of that magnet; moving it walks those equilibria toward a Nash equilibrium.
// The policy that MMD guarantees (tabular) is the last iterate, so evaluation uses the current network.
//
// Game interface (all members static or const):
//   static constexpr int A;  using State;
//   State root(std::mt19937_64&) const;  bool terminal(const State&) const;  double utility(const State&, int seat) const;
//   int actor(const State&) const (-1: chance);  void chance(State&, std::mt19937_64&) const;
//   uint8_t mask(const State&) const;  void step(State&, int action) const;
//   void features(const State&, int seat, Sparse&) const;  int input_dim() const;
#pragma once
#include <array>
#include <chrono>
#include <numeric>
#include "nn.hpp"

namespace sota {

struct MmdConfig {
    int hidden1 = 256, hidden2 = 128;
    int episodes = 2048, epochs = 4, minibatch = 512;
    float lr = 3e-4f, value_lr = 1e-3f;
    double alpha0 = 0.2, alpha_halflife = 0, beta = 1.0, reward_scale = 1.0;
    int magnet_every = 0;
    bool normalize = true;  // advantages divided by their batch standard deviation (then alpha is relative to it)
    uint64_t seed = 1;
};

template <class Game>
struct Mmd {
    static constexpr int A = Game::A;
    const Game& game;
    MmdConfig cfg;
    Mlp pol, val, magnet;  // magnet: empty (in == 0) for the uniform magnet
    Adam opt_p, opt_v;
    std::mt19937_64 rng;
    long long updates = 0, episodes_done = 0, samples_done = 0;
    double last_entropy = 0, last_kl = 0, last_value_loss = 0, last_return_abs = 0;

    struct Sample { Sparse x; uint8_t mask; int action; std::array<float, A> p_old, p_mag; int seat; float ret; float adv; };

    Mmd(const Game& g, const MmdConfig& c)
        : game(g), cfg(c), pol(g.input_dim(), c.hidden1, c.hidden2, A, c.seed * 2 + 1),
          val(g.input_dim(), c.hidden1, c.hidden2, 1, c.seed * 2 + 2), rng(c.seed) {
        opt_p.lr = c.lr; opt_v.lr = c.value_lr; opt_p.init(pol); opt_v.init(val);
    }

    // current policy at state s for the acting seat
    void policy(const Sparse& x, uint8_t mask, double* p) const {
        Mlp::Cache c; pol.forward(x, c); masked_softmax(c.y.data(), mask, A, p);
    }

    double alpha() const {
        return cfg.alpha_halflife > 0 ? cfg.alpha0 / std::sqrt(1.0 + static_cast<double>(updates) / cfg.alpha_halflife) : cfg.alpha0;
    }

    std::vector<Sample> collect() {
        std::vector<Sample> batch;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double abs_ret = 0;
        for (int e = 0; e < cfg.episodes; ++e) {
            auto s = game.root(rng);
            const size_t first = batch.size();
            while (!game.terminal(s)) {
                const int pl = game.actor(s);
                if (pl < 0) { game.chance(s, rng); continue; }
                Sample smp;
                game.features(s, pl, smp.x);
                smp.mask = game.mask(s); smp.seat = pl;
                double p[A];
                policy(smp.x, smp.mask, p);
                double r = u(rng), acc = 0; int a = -1;
                for (int k = 0; k < A; ++k) if (smp.mask >> k & 1) { acc += p[k]; a = k; if (r < acc) break; }
                smp.action = a;
                for (int k = 0; k < A; ++k) smp.p_old[k] = static_cast<float>(p[k]);
                if (magnet.in > 0) {
                    Mlp::Cache cm; magnet.forward(smp.x, cm);
                    double pm[A]; masked_softmax(cm.y.data(), smp.mask, A, pm);
                    for (int k = 0; k < A; ++k) smp.p_mag[k] = static_cast<float>(pm[k]);
                } else {
                    int n = 0;
                    for (int k = 0; k < A; ++k) n += smp.mask >> k & 1;
                    for (int k = 0; k < A; ++k) smp.p_mag[k] = (smp.mask >> k & 1) ? 1.0f / static_cast<float>(n) : 0.0f;
                }
                batch.push_back(std::move(smp));
                game.step(s, a);
            }
            const double u0 = game.utility(s, 0), u1 = game.utility(s, 1);
            abs_ret += std::fabs(u0);
            for (size_t i = first; i < batch.size(); ++i) batch[i].ret = static_cast<float>(batch[i].seat == 0 ? u0 : u1);
        }
        episodes_done += cfg.episodes; samples_done += static_cast<long long>(batch.size());
        last_return_abs = abs_ret / cfg.episodes;
        return batch;
    }

    void update(std::vector<Sample>& batch) {
        // advantages from the value network before this update, normalized over the batch
        double mean = 0, sq = 0;
        for (auto& smp : batch) {
            Mlp::Cache c; val.forward(smp.x, c);
            smp.adv = smp.ret - static_cast<float>(c.y[0] * cfg.reward_scale);
            mean += smp.adv;
        }
        mean /= static_cast<double>(batch.size());
        for (auto& smp : batch) sq += (smp.adv - mean) * (smp.adv - mean);
        const double sd = std::sqrt(sq / std::max<size_t>(1, batch.size() - 1)) + 1e-8;
        for (auto& smp : batch) smp.adv = cfg.normalize ? static_cast<float>((smp.adv - mean) / sd) : smp.adv;
        const double al = alpha();
        std::vector<size_t> order(batch.size());
        std::iota(order.begin(), order.end(), 0);
        Mlp::Grad gp, gv; gp.init(pol); gv.init(val);
        double ent = 0, kl = 0, vloss = 0; long long seen = 0;
        for (int ep = 0; ep < cfg.epochs; ++ep) {
            std::shuffle(order.begin(), order.end(), rng);
            for (size_t lo = 0; lo < order.size(); lo += cfg.minibatch) {
                const size_t hi = std::min(order.size(), lo + static_cast<size_t>(cfg.minibatch));
                for (size_t q = lo; q < hi; ++q) {
                    const Sample& smp = batch[order[q]];
                    Mlp::Cache c; pol.forward(smp.x, c);
                    double p[A]; masked_softmax(c.y.data(), smp.mask, A, p);
                    int n = 0;
                    for (int k = 0; k < A; ++k) n += smp.mask >> k & 1;
                    // gradient of the loss with respect to the logits (legal actions only)
                    double klu = 0, klo = 0;
                    for (int k = 0; k < A; ++k) if (smp.mask >> k & 1 && p[k] > 0) {
                        klu += p[k] * (std::log(p[k]) - std::log(std::max(1e-12, static_cast<double>(smp.p_mag[k]))));
                        klo += p[k] * (std::log(p[k]) - std::log(std::max(1e-12, static_cast<double>(smp.p_old[k]))));
                    }
                    const double ratio = p[smp.action] / std::max(1e-12, static_cast<double>(smp.p_old[smp.action]));
                    float dz[A] = {};
                    for (int k = 0; k < A; ++k) {
                        if (!(smp.mask >> k & 1)) continue;
                        const double lp = std::log(std::max(1e-300, p[k]));
                        const double gu = lp - std::log(std::max(1e-12, static_cast<double>(smp.p_mag[k]))), go = lp - std::log(std::max(1e-12, static_cast<double>(smp.p_old[k])));
                        double d = -smp.adv * ratio * ((k == smp.action) - p[k]);
                        d += al * p[k] * (gu - klu) + cfg.beta * p[k] * (go - klo);
                        dz[k] = static_cast<float>(d);
                    }
                    pol.backward(smp.x, c, dz, gp);
                    Mlp::Cache cv; val.forward(smp.x, cv);
                    const float target = smp.ret / static_cast<float>(cfg.reward_scale);
                    const float dv = 2.0f * (cv.y[0] - target);
                    val.backward(smp.x, cv, &dv, gv);
                    double h = 0;
                    for (int k = 0; k < A; ++k) if (p[k] > 0) h -= p[k] * std::log(p[k]);
                    if (ep == cfg.epochs - 1) { ent += h; kl += klo; vloss += (cv.y[0] - target) * (cv.y[0] - target); ++seen; }
                }
                const float scale = 1.0f / static_cast<float>(hi - lo);
                opt_p.step(pol, gp, scale);
                opt_v.step(val, gv, scale);
            }
        }
        last_entropy = ent / std::max<long long>(1, seen); last_kl = kl / std::max<long long>(1, seen);
        last_value_loss = vloss / std::max<long long>(1, seen);
        ++updates;
        if (cfg.magnet_every > 0 && updates % cfg.magnet_every == 0) magnet = pol;
    }
};

}  // namespace sota
