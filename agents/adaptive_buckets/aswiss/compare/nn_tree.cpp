// The SOTA neural baselines of ../stud7/sota on an exported game tree (Kuhn, Leduc, mini-stud), for exact
// exploitability with ../eval.py. Same algorithm code as the 7-stud trainer; only the game differs.
//
// Build: g++ -O3 -std=c++17 nn_tree.cpp -o nn_tree
// Run:   ./nn_tree --tree DIR --out OUTDIR --algo mmd [--episodes-total 2000000] [MMD options, see ../stud7/sota/mmd.hpp]
//        ./nn_tree --tree DIR --out OUTDIR --algo escher [--iterations 256] [ESCHER options, see ../stud7/sota/escher.hpp]
//   ESCHER checkpoints: policy_bucket_k / policy_true_k = the average policy network (trained at the checkpoint),
//   policy_current_k = regret matching on the regret networks; iters = hands used so far (value + regret traversals).
//   --input onehot (default: one-hot information set, so the network can represent any policy) | features (features.bin)
// Checkpoints after 1, 2, 4, 8, ... updates and the last one: policy_{bucket,true,current}_k.bin (the current network
// policy at every information set; MMD evaluates its last iterate) and a log.csv row (iters = episodes so far).
#include <chrono>
#include <cstdio>
#include <map>
#include "../stud7/sota/mmd.hpp"
#include "../stud7/sota/escher.hpp"

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
    void history_features(const State& h, sota::Sparse& x) const { x.clear(); x.add(h, 1.0f); }  // one-hot history (node)
    int history_dim() const { return static_cast<int>(players.size()); }
};

int main(int argc, char** argv) {
    map<string, string> o{{"--algo", "mmd"}, {"--episodes-total", "2000000"}, {"--input", "onehot"}, {"--seed", "1"},
                          {"--episodes", "1024"}, {"--epochs", "4"}, {"--minibatch", "256"}, {"--lr", "3e-4"},
                          {"--value-lr", "1e-3"}, {"--alpha", "0.2"}, {"--alpha-halflife", "0"}, {"--beta", "1"},
                          {"--hidden1", "128"}, {"--hidden2", "64"}, {"--reward-scale", "1"},
                          {"--magnet-every", "0"}, {"--normalize", "1"}, {"--iterations", "256"}, {"--value-hands", "1024"},
                          {"--value-steps", "200"}, {"--traversals", "1024"}, {"--regret-steps", "400"}, {"--avg-steps", "2000"},
                          {"--batch", "256"}, {"--capacity", "1000000"}, {"--reinit", "1"}, {"--value-explore", "0.25"},
                          {"--schedule", "const"}, {"--eta", "0.05"}, {"--horizon", "10000000"}, {"--clip", "0"}, {"--gamma", "1"},
                          {"--value-reinit", "0"}, {"--fallback", "uniform"}, {"--seconds", "0"}};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (string(argv[i]) != "--tree" && string(argv[i]) != "--out" && !o.count(argv[i])) throw runtime_error(string("unknown option ") + argv[i]);
        o[argv[i]] = argv[i + 1];
    }
    const string dir = o.at("--tree"), out = o.at("--out");
    // --seconds S: stop at the first update / iteration after the training clock passes S seconds, with a last checkpoint.
    // Training clock (column train_seconds) = wall time minus the checkpoints (policy tables, ESCHER's average network).
    const double budget = stod(o["--seconds"]);
    double excluded = 0;
    TreeGame g;
    g.players = load<int32_t>(dir + "/players.bin"); g.info = load<int32_t>(dir + "/info.bin");
    g.start = load<int64_t>(dir + "/start.bin"); g.degree = load<int32_t>(dir + "/degree.bin");
    g.edges = load<int32_t>(dir + "/edges.bin"); g.prob = load<double>(dir + "/prob.bin");
    g.util = load<double>(dir + "/util.bin"); g.counts = load<int32_t>(dir + "/counts.bin");
    g.nI = static_cast<int>(g.counts.size());
    g.onehot = o["--input"] == "onehot";
    if (!g.onehot) { g.feats = load<double>(dir + "/features.bin"); g.fdim = static_cast<int>(g.feats.size() / g.nI); }
    if (o["--algo"] == "escher") {
        sota::EscherConfig e;
        e.hidden1 = stoi(o["--hidden1"]); e.hidden2 = stoi(o["--hidden2"]); e.value_hands = stoi(o["--value-hands"]);
        e.value_steps = stoi(o["--value-steps"]); e.traversals = stoi(o["--traversals"]); e.regret_steps = stoi(o["--regret-steps"]);
        e.avg_steps = stoi(o["--avg-steps"]); e.batch = stoi(o["--batch"]); e.regret_capacity = e.strategy_capacity = stoull(o["--capacity"]);
        e.lr = stof(o["--lr"]); e.value_lr = stof(o["--value-lr"]); e.reward_scale = stod(o["--reward-scale"]);
        e.reinit = o["--reinit"] == "1"; e.value_explore = stod(o["--value-explore"]); e.seed = stoull(o["--seed"]);
        e.value_reinit = o["--value-reinit"] == "1"; e.fallback = o["--fallback"];
        sota::Escher<TreeGame> es(g, e);
        FILE* log = fopen((out + "/log.csv").c_str(), "w");
        if (!log) throw runtime_error("cannot write " + out + "/log.csv");
        fprintf(log, "epoch,iters,buckets_after,iteration,regret_samples,value_loss,regret_loss0,regret_loss1,seconds,train_seconds\n");
        const auto t0 = chrono::steady_clock::now();
        auto train_secs = [&] { return chrono::duration<double>(chrono::steady_clock::now() - t0).count() - excluded; };
        long long T = stoll(o["--iterations"]);
        int k = 0; long long next = 1;
        for (long long it = 1; it <= T; ++it) {
            es.iterate();
            if (budget > 0 && train_secs() >= budget) T = it;  // time budget: last iteration, checkpoint below
            if (it != next && it != T) continue;
            next *= 2;
            const double train_now = train_secs();
            const auto c0 = chrono::steady_clock::now();
            es.train_average();
            vector<double> pa(static_cast<size_t>(g.nI) * W, 0.0), pc(pa.size(), 0.0);
            sota::Sparse x;
            for (int I = 0; I < g.nI; ++I) {
                g.infoset_features(I, x);
                const uint8_t mask = static_cast<uint8_t>((1u << g.counts[I]) - 1);
                es.average(x, mask, &pa[static_cast<size_t>(I) * W]);
            }
            // the current policy uses the regret network of the information set's owner
            {
                vector<int32_t> owner = load<int32_t>(dir + "/owner.bin");
                for (int I = 0; I < g.nI; ++I) {
                    g.infoset_features(I, x);
                    es.current(owner[I], x, static_cast<uint8_t>((1u << g.counts[I]) - 1), &pc[static_cast<size_t>(I) * W]);
                }
            }
            save(out + "/policy_bucket_" + to_string(k) + ".bin", pa); save(out + "/policy_true_" + to_string(k) + ".bin", pa);
            save(out + "/policy_current_" + to_string(k) + ".bin", pc);
            const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            excluded += chrono::duration<double>(chrono::steady_clock::now() - c0).count();
            fprintf(log, "%d,%lld,%d,%lld,%lld,%.6g,%.6g,%.6g,%.3f,%.3f\n", k, es.roots_done, g.nI, it, es.regret_samples, es.last_value_loss,
                    es.last_regret_loss[0], es.last_regret_loss[1], secs, train_now);
            fflush(log);
            fprintf(stderr, "checkpoint %d iteration=%lld hands=%lld value_loss=%.4f regret_loss=%.4f/%.4f (%.1fs)\n", k, it,
                    es.roots_done, es.last_value_loss, es.last_regret_loss[0], es.last_regret_loss[1], secs);
            ++k;
        }
        fclose(log);
        return 0;
    }
    if (o["--algo"] != "mmd") throw runtime_error("--algo must be mmd or escher");
    sota::MmdConfig c;
    c.episodes = stoi(o["--episodes"]); c.epochs = stoi(o["--epochs"]); c.minibatch = stoi(o["--minibatch"]);
    c.lr = stof(o["--lr"]); c.value_lr = stof(o["--value-lr"]); c.alpha0 = stod(o["--alpha"]);
    c.alpha_halflife = stod(o["--alpha-halflife"]); c.beta = stod(o["--beta"]); c.hidden1 = stoi(o["--hidden1"]);
    c.hidden2 = stoi(o["--hidden2"]); c.magnet_every = stoi(o["--magnet-every"]); c.normalize = o["--normalize"] == "1"; c.reward_scale = stod(o["--reward-scale"]); c.seed = stoull(o["--seed"]);
    c.schedule = o["--schedule"]; c.eta0 = stod(o["--eta"]); c.horizon = stod(o["--horizon"]); c.clip = stod(o["--clip"]);
    c.gamma = stod(o["--gamma"]);
    sota::Mmd<TreeGame> mmd(g, c);
    const long long total = stoll(o["--episodes-total"]);
    FILE* log = fopen((out + "/log.csv").c_str(), "w");
    if (!log) throw runtime_error("cannot write " + out + "/log.csv");
    fprintf(log, "epoch,iters,buckets_after,updates,samples,alpha,entropy,kl_old,value_loss,seconds,train_seconds\n");
    const auto t0 = chrono::steady_clock::now();
    auto train_secs = [&] { return chrono::duration<double>(chrono::steady_clock::now() - t0).count() - excluded; };
    int k = 0; long long next = 1;
    bool stop = false;
    while (!stop && mmd.episodes_done < total) {
        auto batch = mmd.collect();
        mmd.update(batch);
        if (budget > 0 && train_secs() >= budget) stop = true;  // time budget: last update, checkpoint below
        if (!stop && mmd.updates != next && mmd.episodes_done < total) continue;
        next *= 2;
        const double train_now = train_secs();
        const auto c0 = chrono::steady_clock::now();
        vector<double> pol(static_cast<size_t>(g.nI) * W, 0.0);
        sota::Sparse x;
        for (int I = 0; I < g.nI; ++I) {
            g.infoset_features(I, x);
            double p[W]; mmd.policy(x, static_cast<uint8_t>((1u << g.counts[I]) - 1), p);
            for (int a = 0; a < W; ++a) pol[static_cast<size_t>(I) * W + a] = p[a];
        }
        for (const char* kind : {"bucket", "true", "current"}) save(out + "/policy_" + kind + "_" + to_string(k) + ".bin", pol);
        const double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        excluded += chrono::duration<double>(chrono::steady_clock::now() - c0).count();
        fprintf(log, "%d,%lld,%d,%lld,%lld,%.6g,%.6g,%.6g,%.6g,%.3f,%.3f\n", k, mmd.episodes_done, g.nI, mmd.updates, mmd.samples_done,
                mmd.alpha(), mmd.last_entropy, mmd.last_kl, mmd.last_value_loss, secs, train_now);
        fflush(log);
        fprintf(stderr, "checkpoint %d episodes=%lld updates=%lld entropy=%.3f kl=%.4f vloss=%.4f (%.1fs)\n", k, mmd.episodes_done,
                mmd.updates, mmd.last_entropy, mmd.last_kl, mmd.last_value_loss, secs);
        ++k;
    }
    fclose(log);
}
