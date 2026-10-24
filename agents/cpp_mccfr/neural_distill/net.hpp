// CPU forward pass of the student networks trained by train.py (same parameter layout and math).
// Include after tokens.hpp. Weights: "NDNET1" file written by train.py::export_weights.
// Transformer: summed field embeddings + linear numeric projection on the query token, pre-norm
// encoder blocks with full attention over the token list, final LayerNorm, heads read the query token.
// MLP: two GELU layers over flat_from_tokens(). LayerNorm eps 1e-5, exact (erf) GELU, float32.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace neural_distill {

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};

struct Outputs {
    std::array<float, 8> regret{};
    std::array<float, 8> average_logits{};
};

class Net {
public:
    std::string architecture;
    int width = 0, layers = 0, heads = 0, ffn = 0, hidden = 0;

    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

    explicit Net(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open weights " + path);
        }
        char magic[8];
        in.read(magic, 8);
        if (std::string(magic, 6) != "NDNET1") {
            throw std::runtime_error("invalid weights magic");
        }
        const auto read_u32 = [&]() {
            uint32_t value = 0;
            in.read(reinterpret_cast<char*>(&value), 4);
            if (!in) {
                throw std::runtime_error("truncated weights");
            }
            return value;
        };
        const auto read_string = [&]() {
            std::string text(read_u32(), '\0');
            in.read(text.data(), static_cast<std::streamsize>(text.size()));
            if (!in) {
                throw std::runtime_error("truncated weights");
            }
            return text;
        };
        architecture = read_string();
        width = static_cast<int>(read_u32());
        layers = static_cast<int>(read_u32());
        heads = static_cast<int>(read_u32());
        ffn = static_cast<int>(read_u32());
        hidden = static_cast<int>(read_u32());
        const uint32_t count = read_u32();
        for (uint32_t i = 0; i < count; ++i) {
            const std::string name = read_string();
            Tensor tensor;
            const uint32_t dims = read_u32();
            size_t size = 1;
            for (uint32_t d = 0; d < dims; ++d) {
                tensor.shape.push_back(static_cast<int>(read_u32()));
                size *= static_cast<size_t>(tensor.shape.back());
            }
            tensor.data.resize(size);
            in.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(size * 4));
            if (!in) {
                throw std::runtime_error("truncated tensor " + name);
            }
            params_.emplace(name, std::move(tensor));
        }
        if (in.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("trailing bytes in weights");
        }
        if (architecture != "transformer" && architecture != "mlp" && architecture != "mlp_key") {
            throw std::runtime_error("unknown architecture " + architecture);
        }
        validate();
        cache();
    }

    bool needs_key() const { return architecture == "mlp_key"; }

    Outputs forward(const std::vector<Token>& tokens, const std::array<float, kNumerics>& numerics,
                    const KeyFeatures* key = nullptr) const {
        if (architecture == "transformer") {
            return forward_transformer(tokens, numerics);
        }
        if (needs_key() && !key) {
            throw std::runtime_error("mlp_key needs teacher key features");
        }
        return forward_mlp(tokens, numerics, needs_key() ? key : nullptr);
    }

private:
    struct Block {
        const Tensor *ln1w, *ln1b, *qkvw, *qkvb, *outw, *outb, *ln2w, *ln2b, *ff1w, *ff1b, *ff2w, *ff2b;
    };
    std::map<std::string, Tensor> params_;
    std::array<const Tensor*, kTokenFields> embedding_{};
    const Tensor *query_w_ = nullptr, *query_b_ = nullptr, *final_w_ = nullptr, *final_b_ = nullptr;
    std::vector<Block> blocks_;

    void cache() {
        if (architecture != "transformer") {
            return;
        }
        for (int f = 0; f < kTokenFields; ++f) {
            embedding_[f] = &params_.at("embedding." + std::to_string(f) + ".weight");
        }
        query_w_ = &params_.at("query.weight");
        query_b_ = &params_.at("query.bias");
        final_w_ = &params_.at("final_norm.weight");
        final_b_ = &params_.at("final_norm.bias");
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";
            blocks_.push_back(Block{&params_.at(p + "ln1.weight"), &params_.at(p + "ln1.bias"),
                                    &params_.at(p + "qkv.weight"), &params_.at(p + "qkv.bias"),
                                    &params_.at(p + "out.weight"), &params_.at(p + "out.bias"),
                                    &params_.at(p + "ln2.weight"), &params_.at(p + "ln2.bias"),
                                    &params_.at(p + "ff1.weight"), &params_.at(p + "ff1.bias"),
                                    &params_.at(p + "ff2.weight"), &params_.at(p + "ff2.bias")});
        }
    }

    const Tensor& get(const std::string& name, std::initializer_list<int> shape) const {
        const auto it = params_.find(name);
        if (it == params_.end()) {
            throw std::runtime_error("missing parameter " + name);
        }
        if (it->second.shape != std::vector<int>(shape)) {
            throw std::runtime_error("shape mismatch for " + name);
        }
        return it->second;
    }

    void validate() const {
        size_t expected = 0;
        const auto need = [&](const std::string& name, std::initializer_list<int> shape) {
            get(name, shape);
            ++expected;
        };
        if (architecture == "mlp" || architecture == "mlp_key") {
            need("fc1.weight", {hidden, kFlatSize + (architecture == "mlp_key" ? kKeyFlatSize : 0)});
            need("fc1.bias", {hidden});
            need("fc2.weight", {hidden, hidden});
            need("fc2.bias", {hidden});
            need("regret.weight", {8, hidden});
            need("regret.bias", {8});
            need("average.weight", {8, hidden});
            need("average.bias", {8});
        } else {
            if (width % heads != 0) {
                throw std::runtime_error("width not divisible by heads");
            }
            for (int f = 0; f < kTokenFields; ++f) {
                need("embedding." + std::to_string(f) + ".weight", {kVocab[f], width});
            }
            need("query.weight", {width, kNumerics});
            need("query.bias", {width});
            for (int l = 0; l < layers; ++l) {
                const std::string p = "blocks." + std::to_string(l) + ".";
                need(p + "ln1.weight", {width});
                need(p + "ln1.bias", {width});
                need(p + "qkv.weight", {3 * width, width});
                need(p + "qkv.bias", {3 * width});
                need(p + "out.weight", {width, width});
                need(p + "out.bias", {width});
                need(p + "ln2.weight", {width});
                need(p + "ln2.bias", {width});
                need(p + "ff1.weight", {ffn, width});
                need(p + "ff1.bias", {ffn});
                need(p + "ff2.weight", {width, ffn});
                need(p + "ff2.bias", {width});
            }
            need("final_norm.weight", {width});
            need("final_norm.bias", {width});
            need("regret.weight", {8, width});
            need("regret.bias", {8});
            need("average.weight", {8, width});
            need("average.bias", {8});
        }
        if (expected != params_.size()) {
            throw std::runtime_error("unexpected parameter count in weights");
        }
    }

    // y[o] = b[o] + sum_i W[o, i] x[i]
    static void linear(const Tensor& weight, const Tensor& bias, const float* x, float* y) {
        const int out = weight.shape[0], in = weight.shape[1];
        for (int o = 0; o < out; ++o) {
            const float* row = weight.data.data() + static_cast<size_t>(o) * in;
            float sum = bias.data[o];
            for (int i = 0; i < in; ++i) {
                sum += row[i] * x[i];
            }
            y[o] = sum;
        }
    }

    static void layer_norm(const Tensor& gamma, const Tensor& beta, const float* x, float* y, int n) {
        double mean = 0;
        for (int i = 0; i < n; ++i) {
            mean += x[i];
        }
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; ++i) {
            var += (x[i] - mean) * (x[i] - mean);
        }
        var /= n;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < n; ++i) {
            y[i] = static_cast<float>((x[i] - mean) * inv) * gamma.data[i] + beta.data[i];
        }
    }

    static float gelu(float x) {
        return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }

    Outputs heads_from(const float* h, int n) const {
        Outputs out;
        const Tensor& rw = params_.at("regret.weight");
        const Tensor& rb = params_.at("regret.bias");
        const Tensor& aw = params_.at("average.weight");
        const Tensor& ab = params_.at("average.bias");
        if (rw.shape[1] != n) {
            throw std::runtime_error("head width mismatch");
        }
        linear(rw, rb, h, out.regret.data());
        linear(aw, ab, h, out.average_logits.data());
        return out;
    }

    Outputs forward_mlp(const std::vector<Token>& tokens, const std::array<float, kNumerics>& numerics,
                        const KeyFeatures* key) const {
        auto x = flat_from_tokens(tokens, numerics);
        if (key) {
            const auto extra = key_flat(*key);
            x.insert(x.end(), extra.begin(), extra.end());
        }
        std::vector<float> h1(hidden), h2(hidden);
        linear(params_.at("fc1.weight"), params_.at("fc1.bias"), x.data(), h1.data());
        for (float& v : h1) {
            v = gelu(v);
        }
        linear(params_.at("fc2.weight"), params_.at("fc2.bias"), h1.data(), h2.data());
        for (float& v : h2) {
            v = gelu(v);
        }
        return heads_from(h2.data(), hidden);
    }

    Outputs forward_transformer(const std::vector<Token>& tokens, const std::array<float, kNumerics>& numerics) const {
        const int n = static_cast<int>(tokens.size());
        const int d = width;
        if (n == 0 || tokens.back()[0] != kQuery) {
            throw std::runtime_error("last token must be the query");
        }
        std::vector<float> x(static_cast<size_t>(n) * d, 0.0f);
        std::vector<float> q(d);
        for (int t = 0; t < n; ++t) {
            float* row = x.data() + static_cast<size_t>(t) * d;
            for (int f = 0; f < kTokenFields; ++f) {
                const int index = tokens[t][f];
                if (index < 0 || index >= kVocab[f]) {
                    throw std::runtime_error("token field out of range");
                }
                const float* e = embedding_[f]->data.data() + static_cast<size_t>(index) * d;
                for (int i = 0; i < d; ++i) {
                    row[i] += e[i];
                }
            }
            if (tokens[t][0] == kQuery) {
                linear(*query_w_, *query_b_, numerics.data(), q.data());
                for (int i = 0; i < d; ++i) {
                    row[i] += q[i];
                }
            }
        }
        const int dh = d / heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        std::vector<float> normed(static_cast<size_t>(n) * d), qkv(static_cast<size_t>(n) * 3 * d);
        std::vector<float> attended(static_cast<size_t>(n) * d), projected(d), mid(ffn), scores(n);
        for (const Block& b : blocks_) {
            for (int t = 0; t < n; ++t) {
                layer_norm(*b.ln1w, *b.ln1b, x.data() + static_cast<size_t>(t) * d,
                           normed.data() + static_cast<size_t>(t) * d, d);
                linear(*b.qkvw, *b.qkvb, normed.data() + static_cast<size_t>(t) * d,
                       qkv.data() + static_cast<size_t>(t) * 3 * d);
            }
            for (int h = 0; h < heads; ++h) {
                for (int t = 0; t < n; ++t) {
                    const float* qh = qkv.data() + static_cast<size_t>(t) * 3 * d + h * dh;
                    float best = -1e30f;
                    for (int s = 0; s < n; ++s) {
                        const float* k = qkv.data() + static_cast<size_t>(s) * 3 * d + d + h * dh;
                        float dot = 0;
                        for (int i = 0; i < dh; ++i) {
                            dot += qh[i] * k[i];
                        }
                        scores[s] = dot * scale;
                        best = std::max(best, scores[s]);
                    }
                    double total = 0;
                    for (int s = 0; s < n; ++s) {
                        scores[s] = std::exp(scores[s] - best);
                        total += scores[s];
                    }
                    float* out = attended.data() + static_cast<size_t>(t) * d + h * dh;
                    for (int i = 0; i < dh; ++i) {
                        out[i] = 0.0f;
                    }
                    for (int s = 0; s < n; ++s) {
                        const float w = static_cast<float>(scores[s] / total);
                        const float* v = qkv.data() + static_cast<size_t>(s) * 3 * d + 2 * d + h * dh;
                        for (int i = 0; i < dh; ++i) {
                            out[i] += w * v[i];
                        }
                    }
                }
            }
            for (int t = 0; t < n; ++t) {
                float* row = x.data() + static_cast<size_t>(t) * d;
                linear(*b.outw, *b.outb, attended.data() + static_cast<size_t>(t) * d, projected.data());
                for (int i = 0; i < d; ++i) {
                    row[i] += projected[i];
                }
                layer_norm(*b.ln2w, *b.ln2b, row, normed.data(), d);
                linear(*b.ff1w, *b.ff1b, normed.data(), mid.data());
                for (float& v : mid) {
                    v = gelu(v);
                }
                linear(*b.ff2w, *b.ff2b, mid.data(), projected.data());
                for (int i = 0; i < d; ++i) {
                    row[i] += projected[i];
                }
            }
        }
        std::vector<float> last(d);
        layer_norm(*final_w_, *final_b_, x.data() + static_cast<size_t>(n - 1) * d, last.data(), d);
        return heads_from(last.data(), d);
    }
};

// Student policies from network outputs. Current: regret matching on the regret head (uniform if no
// positive legal value, as in current_strategy). Average: softmax of the average head over legal actions.
inline std::array<double, 8> regret_matching(const Outputs& out, uint8_t mask) {
    std::array<double, 8> p{};
    double total = 0;
    int count = 0;
    for (int a = 0; a < 8; ++a) {
        if (mask & (1u << a)) {
            p[a] = std::max(0.0, static_cast<double>(out.regret[a]));
            total += p[a];
            ++count;
        }
    }
    for (int a = 0; a < 8; ++a) {
        if (mask & (1u << a)) {
            p[a] = total > 0 ? p[a] / total : 1.0 / count;
        }
    }
    return p;
}

inline std::array<double, 8> softmax_policy(const Outputs& out, uint8_t mask) {
    std::array<double, 8> p{};
    double best = -1e300, total = 0;
    for (int a = 0; a < 8; ++a) {
        if (mask & (1u << a)) {
            best = std::max(best, static_cast<double>(out.average_logits[a]));
        }
    }
    for (int a = 0; a < 8; ++a) {
        if (mask & (1u << a)) {
            p[a] = std::exp(static_cast<double>(out.average_logits[a]) - best);
            total += p[a];
        }
    }
    for (double& v : p) {
        v /= total;
    }
    return p;
}

} // namespace neural_distill
