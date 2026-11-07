// Build: g++ -O3 -std=c++17 bench_mlp_step.cpp -o bench_mlp_step;  run: ./bench_mlp_step IN H1 H2 OUT BATCH STEPS
// Time one Adam minibatch step of the baselines' MLP (stud7/sota/nn.hpp) with one-hot input, as in nn_tree.cpp.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "../../stud7/sota/nn.hpp"
int main(int argc, char** argv) {
    const int in = atoi(argv[1]), h1 = atoi(argv[2]), h2 = atoi(argv[3]), out = atoi(argv[4]), B = atoi(argv[5]), steps = atoi(argv[6]);
    sota::Mlp net(in, h1, h2, out, 1); sota::Adam opt; opt.init(net); sota::Mlp::Grad g; g.init(net);
    std::mt19937_64 rng(2); std::vector<sota::Sparse> xs(B); std::vector<float> dy(out);
    sota::Mlp::Cache c;
    auto run = [&](int n) {
        for (int s = 0; s < n; ++s) {
            for (int b = 0; b < B; ++b) { xs[b].clear(); xs[b].add(static_cast<int>(rng() % in), 1.0f); }
            for (int b = 0; b < B; ++b) {
                net.forward(xs[b], c);
                for (int o = 0; o < out; ++o) dy[o] = c.y[o] - 0.1f;  // squared-error gradient
                net.backward(xs[b], c, dy.data(), g);
            }
            opt.step(net, g, 1.0f / B);
        }
    };
    run(2);
    auto t0 = std::chrono::steady_clock::now(); run(steps);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("ours in=%d %dx%d out=%d batch=%d: %.3f ms per step\n", in, h1, h2, out, B, 1e3 * s / steps);
}
