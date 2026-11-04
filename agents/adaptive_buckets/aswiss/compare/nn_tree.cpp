// The SOTA neural baselines of ../stud7/sota on an exported game tree (Kuhn, Leduc, mini-stud), for exact
// exploitability with ../eval.py. Same algorithm code as the 7-stud trainer; only the game differs.
//
// Build: g++ -O3 -std=c++17 nn_tree.cpp -o nn_tree
// Run:   ./nn_tree --tree DIR --out OUTDIR --algo mmd [--episodes-total 2000000] [MMD options, see ../stud7/sota/mmd.hpp]
//   --input onehot (default: one-hot information set, so the network can represent any policy) | features (features.bin)
// Checkpoints after 1, 2, 4, 8, ... updates and the last one: policy_{bucket,true,current}_k.bin (the current network
// policy at every information set; MMD evaluates its last iterate) and a log.csv row (iters = episodes so far).
#include <chrono>
#include <cstdio>
#include <map>
#include "../stud7/sota/mmd.hpp"

using namespace std;

template <class T> vector<T> load(const string& path) {
    ifstream in(path, ios::binary | ios::ate);
    if (!in) throw runtime_error("cannot open " + path);
    size_t bytes = in.tellg(); in.seekg(0);
    vector<T> data(bytes / sizeof(T));
    in.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}
template <class T> void save(const string& path, const vector<T>& data) {
    ofstream out(path, ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

constexpr int W = 4, TERMINAL = -2, CHANCE = -1;

struct TreeGame {
    static constexpr int A = W;
    using State = int;
    vector<int32_t> players, info, degree, edges, counts;
    vector<int64_t> start;
    vector<double> prob, util, feats;
    int nI = 0, fdim = 0;
    bool onehot = true;
    State root(mt19937_64&) const { return 0; }
    bool terminal(const State& h) const { return players[h] == TERMINAL; }
    double utility(const State& h, int seat) const { return seat == 0 ? util[h] : -util[h]; }
    int actor(const State& h) const { return players[h] == CHANCE ? -1 : players[h]; }
    void chance(State& h, mt19937_64& rng) const {
        double r = uniform_real_distribution<double>(0, 1)(rng), acc = 0;
        const int64_t e = start[h];
        for (int k = 0; k < degree[h]; ++k) { acc += prob[e + k]; if (r < acc || k == degree[h] - 1) { h = edges[e + k]; return; } }
    }
    uint8_t mask(const State& h) const { return static_cast<uint8_t>((1u << degree[h]) - 1); }
    void step(State& h, int a) const { h = edges[start[h] + a]; }
    void infoset_features(int I, sota::Sparse& x) const {
        x.clear();
        if (onehot) { x.add(I, 1.0f); return; }
        for (int d = 0; d < fdim; ++d) x.add(d, static_cast<float>(feats[static_cast<size_t>(I) * fdim + d]));
    }
    void features(const State& h, int, sota::Sparse& x) const { infoset_features(info[h], x); }
    int input_dim() const { return onehot ? nI : fdim; }
};

int main(int argc, char** argv) {
    map<string, string> o{{"--algo", "mmd"}, {"--episodes-total", "2000000"}, {"--input", "onehot"}, {"--seed", "1"},
                          {"--episodes", "1024"}, {"--epochs", "4"}, {"--minibatch", "256"}, {"--lr", "3e-4"},
                          {"--value-lr", "1e-3"}, {"--alpha", "0.2"}, {"--alpha-halflife", "0"}, {"--beta", "1"},
                          {"--hidden1", "128"}, {"--hidden2", "64"}, {"--reward-scale", "1"},
                          {"--magnet-every", "0"}};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (string(argv[i]) != "--tree" && string(argv[i]) != "--out" && !o.count(argv[i])) throw runtime_error(string("unknown option ") + argv[i]);
        o[argv[i]] = argv[i + 1];
    }
    const string dir = o.at("--tree"), out = o.at("--out");
    TreeGame g;
    g.players = load<int32_t>(dir + "/players.bin"); g.info = load<int32_t>(dir + "/info.bin");
    g.start = load<int64_t>(dir + "/start.bin"); g.degree = load<int32_t>(dir + "/degree.bin");
    g.edges = load<int32_t>(dir + "/edges.bin"); g.prob = load<double>(dir + "/prob.bin");
    g.util = load<double>(dir + "/util.bin"); g.counts = load<int32_t>(dir + "/counts.bin");
    g.nI = static_cast<int>(g.counts.size());
    g.onehot = o["--input"] == "onehot";
    if (!g.onehot) { g.feats = load<double>(dir + "/features.bin"); g.fdim = static_cast<int>(g.feats.size() / g.nI); }
    if (o["--algo"] != "mmd") throw runtime_error("--algo must be mmd");
    sota::MmdConfig c;
    c.episodes = stoi(o["--episodes"]); c.epochs = stoi(o["--epochs"]); c.minibatch = stoi(o["--minibatch"]);
    c.lr = stof(o["--lr"]); c.value_lr = stof(o["--value-lr"]); c.alpha0 = stod(o["--alpha"]);
    c.alpha_halflife = stod(o["--alpha-halflife"]); c.beta = stod(o["--beta"]); c.hidden1 = stoi(o["--hidden1"]);
    c.hidden2 = stoi(o["--hidden2"]); c.magnet_every = stoi(o["--magnet-every"]); c.reward_scale = stod(o["--reward-scale"]); c.seed = stoull(o["--seed"]);
    sota::Mmd<TreeGame> mmd(g, c);
    const long long total = stoll(o["--episodes-total"]);
    FILE* log = fopen((out + "/log.csv").c_str(), "w");
    if (!log) throw runtime_error("cannot write " + out + "/log.csv");
    fprintf(log, "epoch,iters,buckets_after,updates,samples,alpha,entropy,kl_old,value_loss,seconds\n");
    const auto t0 = chrono::steady_clock::now();
    int k = 0; long long next = 1;
    while (mmd.episodes_done < total) {
        auto batch = mmd.collect();
        mmd.update(batch);
        if (mmd.updates != next && mmd.episodes_done < total) continue;
        next *= 2;
        vector<double> pol(static_cast<size_t>(g.nI) * W, 0.0);
        sota::Sparse x;
        for (int I = 0; I < g.nI; ++I) {
            g.infoset_features(I, x);
            double p[W]; mmd.policy(x, static_cast<uint8_t>((1u << g.counts[I]) - 1), p);
            for (int a = 0; a < W; ++a) pol[static_cast<size_t>(I) * W + a] = p[a];
        }
        for (const char* kind : {"bucket", "true", "current"}) save(out + "/policy_" + kind + "_" + to_string(k) + ".bin", pol);
        const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        fprintf(log, "%d,%lld,%d,%lld,%lld,%.6g,%.6g,%.6g,%.6g,%.3f\n", k, mmd.episodes_done, g.nI, mmd.updates, mmd.samples_done,
                mmd.alpha(), mmd.last_entropy, mmd.last_kl, mmd.last_value_loss, secs);
        fflush(log);
        fprintf(stderr, "checkpoint %d episodes=%lld updates=%lld entropy=%.3f kl=%.4f vloss=%.4f (%.1fs)\n", k, mmd.episodes_done,
                mmd.updates, mmd.last_entropy, mmd.last_kl, mmd.last_value_loss, secs);
        ++k;
    }
    fclose(log);
}
