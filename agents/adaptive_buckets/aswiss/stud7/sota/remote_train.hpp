// Network training in another process for ESCHER (--remote-train 1): gpu_trainer.py starts the trainer with its stdin
// and stdout as pipes and trains on its device (CUDA). This side keeps the traversals, the reservoirs, the checkpoints
// and the loop; it sends each training phase's new data and the network's initial weights (the same initialization as
// the CPU version) and receives the trained weights. Inference during traversals stays on the CPU (a GPU answers one
// query at a time 2-4 times slower than a CPU thread; calib_gpu.py).
//
// Messages, little-endian and packed: u32 type, then the payload. Every training request is answered by its type,
// f32 loss of the last step, and the trained network (i32 in, h1, h2, out, then W1 b1 W2 b2 W3 b3 as f32 in Mlp's
// layout). A sparse input is u16 nnz, nnz x i32 index, nnz x f32 value.
//   HELLO          u32 input_dim, history_dim, hidden1, hidden2, A, batch, regret_capacity, strategy_capacity; u64 seed
//                  -> HELLO, u32 version
//   VALUE_TRAIN    u32 t, steps, batch; f32 lr; u8 reset_optimizer; network; u32 n; n x (sparse history, f32 return)
//   RES_UPDATE     u8 reservoir (0, 1: regret of that seat; 2: strategy); u32 n; n x (u32 slot, sparse, u8 mask,
//                  A x f32 target, f32 weight)                                            (no answer)
//   REGRET_TRAIN   u8 seat; u32 t, steps, batch; f32 lr; u8 reset_optimizer; network
//   AVG_TRAIN      u8 kind (0: snapshot, fresh optimizer; 1: monitor, optimizer kept across calls); u32 t, steps, batch;
//                  f32 lr; u64 seed; network
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "nn.hpp"

namespace sota {

struct RemoteTrainer {
    enum : uint32_t { HELLO = 1, VALUE_TRAIN = 2, RES_UPDATE = 3, REGRET_TRAIN = 4, AVG_TRAIN = 5 };
    static constexpr uint32_t kVersion = 1;
    std::vector<char> buf;
    double seconds = 0;  // wall time spent waiting for the remote side (training)

    void start() {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    }
    template <class T> void put(const T& v) {
        const char* p = reinterpret_cast<const char*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    void put_floats(const std::vector<float>& v) {
        const char* p = reinterpret_cast<const char*>(v.data());
        buf.insert(buf.end(), p, p + v.size() * sizeof(float));
    }
    void put_sparse(const Sparse& s) {
        if (s.x.size() > 65535) throw std::runtime_error("remote trainer: sparse input too long");
        put<uint16_t>(static_cast<uint16_t>(s.x.size()));
        for (const auto& [i, v] : s.x) put<int32_t>(i);
        for (const auto& [i, v] : s.x) put<float>(v);
    }
    void put_mlp(const Mlp& n) {
        put<int32_t>(n.in); put<int32_t>(n.h1); put<int32_t>(n.h2); put<int32_t>(n.out);
        for (const auto* w : {&n.W1, &n.b1, &n.W2, &n.b2, &n.W3, &n.b3}) put_floats(*w);
    }
    void send() {
        if (!buf.empty() && std::fwrite(buf.data(), 1, buf.size(), stdout) != buf.size())
            throw std::runtime_error("remote trainer: write failed");
        std::fflush(stdout);
        buf.clear();
    }
    void get_bytes(void* p, size_t n) {
        if (n && std::fread(p, 1, n, stdin) != n) throw std::runtime_error("remote trainer: the trainer process closed the pipe");
    }
    template <class T> T get() { T v; get_bytes(&v, sizeof(T)); return v; }
    void expect(uint32_t type) {
        const uint32_t got = get<uint32_t>();
        if (got != type) throw std::runtime_error("remote trainer: answer " + std::to_string(got) + " to request " + std::to_string(type));
    }
    Mlp get_mlp() {
        Mlp n;
        n.in = get<int32_t>(); n.h1 = get<int32_t>(); n.h2 = get<int32_t>(); n.out = get<int32_t>();
        n.W1.resize(static_cast<size_t>(n.in) * n.h1); n.b1.resize(n.h1); n.W2.resize(static_cast<size_t>(n.h1) * n.h2);
        n.b2.resize(n.h2); n.W3.resize(static_cast<size_t>(n.h2) * n.out); n.b3.resize(n.out);
        for (auto* w : {&n.W1, &n.b1, &n.W2, &n.b2, &n.W3, &n.b3}) get_bytes(w->data(), w->size() * sizeof(float));
        return n;
    }
    // send the buffered request and read the answer: loss and network
    Mlp train(uint32_t type, double& loss) {
        const auto t0 = std::chrono::steady_clock::now();
        send();
        expect(type);
        loss = get<float>();
        Mlp n = get_mlp();
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return n;
    }
};

}  // namespace sota
