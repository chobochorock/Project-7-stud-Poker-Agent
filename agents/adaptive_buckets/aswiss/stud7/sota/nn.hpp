// Small MLP for the SOTA baselines (no external library): sparse first layer, two ReLU hidden layers, linear output,
// Adam with lazy updates of the first-layer rows a batch touched. Single precision; deterministic for a given seed and
// sample order (no threads inside).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sota {

// Input: a few active coordinates (one-hot tokens and small dense values).
struct Sparse {
    std::vector<std::pair<int, float>> x;
    void clear() { x.clear(); }
    void add(int i, float v) { if (v != 0.0f) x.emplace_back(i, v); }
};

struct Mlp {
    int in = 0, h1 = 0, h2 = 0, out = 0;
    std::vector<float> W1, b1, W2, b2, W3, b3;  // W1: in x h1 (row per input), W2: h1 x h2, W3: h2 x out

    Mlp() = default;
    Mlp(int in_, int h1_, int h2_, int out_, uint64_t seed) : in(in_), h1(h1_), h2(h2_), out(out_) {
        std::mt19937_64 rng(seed);
        auto init = [&](std::vector<float>& w, int fan_in, size_t n) {
            std::normal_distribution<float> g(0.0f, std::sqrt(2.0f / static_cast<float>(fan_in)));
            w.resize(n);
            for (auto& v : w) v = g(rng);
        };
        init(W1, 64, static_cast<size_t>(in) * h1);  // sparse input: about 64 active coordinates, not `in`
        init(W2, h1, static_cast<size_t>(h1) * h2);
        init(W3, h2, static_cast<size_t>(h2) * out);
        for (auto& v : W3) v *= 0.1f;  // start near uniform / zero
        b1.assign(h1, 0.0f); b2.assign(h2, 0.0f); b3.assign(out, 0.0f);
    }
    size_t parameters() const { return W1.size() + b1.size() + W2.size() + b2.size() + W3.size() + b3.size(); }

    struct Cache { std::vector<float> a1, a2, y; };
    // y = W3 relu(W2 relu(W1 x + b1) + b2) + b3
    void forward(const Sparse& s, Cache& c) const {
        c.a1.assign(b1.begin(), b1.end());
        for (const auto& [i, v] : s.x) {
            const float* w = &W1[static_cast<size_t>(i) * h1];
            for (int j = 0; j < h1; ++j) c.a1[j] += v * w[j];
        }
        for (auto& v : c.a1) v = std::max(v, 0.0f);
        c.a2.assign(b2.begin(), b2.end());
        for (int j = 0; j < h1; ++j) {
            const float a = c.a1[j];
            if (a == 0.0f) continue;
            const float* w = &W2[static_cast<size_t>(j) * h2];
            for (int k = 0; k < h2; ++k) c.a2[k] += a * w[k];
        }
        for (auto& v : c.a2) v = std::max(v, 0.0f);
        c.y.assign(b3.begin(), b3.end());
        for (int k = 0; k < h2; ++k) {
            const float a = c.a2[k];
            if (a == 0.0f) continue;
            const float* w = &W3[static_cast<size_t>(k) * out];
            for (int o = 0; o < out; ++o) c.y[o] += a * w[o];
        }
    }

    // Gradient accumulator for one minibatch.
    struct Grad {
        std::vector<float> gW1, gb1, gW2, gb2, gW3, gb3;
        std::vector<int> rows; std::vector<char> touched;
        void init(const Mlp& m) {
            gW1.assign(m.W1.size(), 0.0f); gb1.assign(m.h1, 0.0f); gW2.assign(m.W2.size(), 0.0f);
            gb2.assign(m.h2, 0.0f); gW3.assign(m.W3.size(), 0.0f); gb3.assign(m.out, 0.0f);
            touched.assign(m.in, 0); rows.clear();
        }
    };
    void backward(const Sparse& s, const Cache& c, const float* dy, Grad& g) const {
        std::vector<float> d2(h2, 0.0f), d1(h1, 0.0f);
        for (int k = 0; k < h2; ++k) {
            const float a = c.a2[k];
            float* gw = &g.gW3[static_cast<size_t>(k) * out];
            const float* w = &W3[static_cast<size_t>(k) * out];
            float acc = 0.0f;
            for (int o = 0; o < out; ++o) { gw[o] += a * dy[o]; acc += w[o] * dy[o]; }
            d2[k] = a > 0.0f ? acc : 0.0f;
        }
        for (int o = 0; o < out; ++o) g.gb3[o] += dy[o];
        for (int j = 0; j < h1; ++j) {
            const float a = c.a1[j];
            float* gw = &g.gW2[static_cast<size_t>(j) * h2];
            const float* w = &W2[static_cast<size_t>(j) * h2];
            float acc = 0.0f;
            for (int k = 0; k < h2; ++k) { gw[k] += a * d2[k]; acc += w[k] * d2[k]; }
            d1[j] = a > 0.0f ? acc : 0.0f;
        }
        for (int k = 0; k < h2; ++k) g.gb2[k] += d2[k];
        for (const auto& [i, v] : s.x) {
            if (!g.touched[i]) { g.touched[i] = 1; g.rows.push_back(i); }
            float* gw = &g.gW1[static_cast<size_t>(i) * h1];
            for (int j = 0; j < h1; ++j) gw[j] += v * d1[j];
        }
        for (int j = 0; j < h1; ++j) g.gb1[j] += d1[j];
    }
};

// Adam; the first-layer moments of rows a batch did not touch are left as they are (lazy Adam).
struct Adam {
    float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    long long t = 0;
    std::vector<float> m[6], v[6];
    void init(const Mlp& net) {
        const size_t n[6] = {net.W1.size(), net.b1.size(), net.W2.size(), net.b2.size(), net.W3.size(), net.b3.size()};
        for (int i = 0; i < 6; ++i) { m[i].assign(n[i], 0.0f); v[i].assign(n[i], 0.0f); }
        t = 0;
    }
    // scale: multiplies the gradients (1 / batch size)
    void step(Mlp& net, Mlp::Grad& g, float scale) {
        ++t;
        const float c1 = 1.0f - std::pow(beta1, static_cast<float>(t)), c2 = 1.0f - std::pow(beta2, static_cast<float>(t));
        auto upd = [&](std::vector<float>& w, std::vector<float>& gw, int k, size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) {
                const float gr = gw[i] * scale;
                m[k][i] = beta1 * m[k][i] + (1 - beta1) * gr;
                v[k][i] = beta2 * v[k][i] + (1 - beta2) * gr * gr;
                w[i] -= lr * (m[k][i] / c1) / (std::sqrt(v[k][i] / c2) + eps);
                gw[i] = 0.0f;
            }
        };
        for (int r : g.rows) {
            upd(net.W1, g.gW1, 0, static_cast<size_t>(r) * net.h1, static_cast<size_t>(r + 1) * net.h1);
            g.touched[r] = 0;
        }
        g.rows.clear();
        upd(net.b1, g.gb1, 1, 0, net.b1.size());
        upd(net.W2, g.gW2, 2, 0, net.W2.size()); upd(net.b2, g.gb2, 3, 0, net.b2.size());
        upd(net.W3, g.gW3, 4, 0, net.W3.size()); upd(net.b3, g.gb3, 5, 0, net.b3.size());
    }
};

inline void save_mlp(std::ofstream& out, const Mlp& n) {
    const int32_t dims[4] = {n.in, n.h1, n.h2, n.out};
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    for (const auto* w : {&n.W1, &n.b1, &n.W2, &n.b2, &n.W3, &n.b3})
        out.write(reinterpret_cast<const char*>(w->data()), static_cast<std::streamsize>(w->size() * sizeof(float)));
}
inline Mlp load_mlp(std::ifstream& in) {
    int32_t dims[4];
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!in) throw std::runtime_error("truncated network file");
    Mlp n; n.in = dims[0]; n.h1 = dims[1]; n.h2 = dims[2]; n.out = dims[3];
    n.W1.resize(static_cast<size_t>(n.in) * n.h1); n.b1.resize(n.h1); n.W2.resize(static_cast<size_t>(n.h1) * n.h2);
    n.b2.resize(n.h2); n.W3.resize(static_cast<size_t>(n.h2) * n.out); n.b3.resize(n.out);
    for (auto* w : {&n.W1, &n.b1, &n.W2, &n.b2, &n.W3, &n.b3})
        in.read(reinterpret_cast<char*>(w->data()), static_cast<std::streamsize>(w->size() * sizeof(float)));
    if (!in) throw std::runtime_error("truncated network file");
    return n;
}

// Masked softmax of logits over the legal actions (bit a of mask set).
inline void masked_softmax(const float* logits, uint8_t mask, int n, double* p) {
    double mx = -1e300;
    for (int a = 0; a < n; ++a) if (mask >> a & 1) mx = std::max(mx, static_cast<double>(logits[a]));
    double z = 0;
    for (int a = 0; a < n; ++a) { p[a] = (mask >> a & 1) ? std::exp(static_cast<double>(logits[a]) - mx) : 0.0; z += p[a]; }
    for (int a = 0; a < n; ++a) p[a] /= z;
}

}  // namespace sota
