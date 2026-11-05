// `train --partition rebel` and `eval --partition rebel` of aswiss_stud.cpp (scaled-down ReBeL, rebel.hpp).
// Included after stud7_net_main.hpp (reuses its checkpoint helpers and NetLoop).
//
// Train: self-play hands ("roots" = hands; each hand solves up to three street subgames), value networks V6 / V7
// trained online from FIFO replay buffers. Snapshots (snapshot_K.net, magic STUDRBL1) hold the settings and both
// value networks; checkpoint.net holds everything (networks, Adam, buffers, RNG, counters) for bit-exact resume.
// Eval: the snapshot's networks + the same search at test time, one solve per street root (cached per public
// history), against the same Policy-LBR as every other arm (same deals; the LBR's particle RNG is seeded per pair so
// --threads does not change the result). The BR-gap proxy is not computed for ReBeL (it would need a search at every
// public state the gap traversal visits); gap_proxy_ante is NaN.
#pragma once
#include <thread>
#include "rebel.hpp"

namespace sota {

namespace rebel {
constexpr char kMagic[8] = {'S', 'T', 'U', 'D', 'R', 'B', 'L', '1'};

inline std::string config_string(const Config& c) {
    std::ostringstream s;
    s << "iters56=" << c.iters56 << ";iters7=" << c.iters7 << ";buckets=" << c.buckets << ";hidden=" << c.hidden << ";lr=" << c.lr
      << ";batch=" << c.batch << ";train_steps=" << c.train_steps << ";capacity=" << c.capacity << ";explore=" << c.explore
      << ";floor=" << c.floor;
    return s.str();
}
inline void save_snapshot(const std::string& path, const Config& c, const Mlp& v6, const Mlp& v7) {
    std::ofstream out(path, std::ios::binary);
    out.write(kMagic, 8);
    put_raw(out, c.iters56); put_raw(out, c.iters7); put_raw(out, c.buckets); put_raw(out, c.floor);
    save_mlp(out, v6); save_mlp(out, v7);
}
inline void load_snapshot(const std::string& path, Config& c, Mlp& v6, Mlp& v7) {
    std::ifstream in(path, std::ios::binary);
    char m[8];
    in.read(m, 8);
    if (!in || std::string(m, 8) != std::string(kMagic, 8)) throw std::runtime_error(path + " is not a ReBeL snapshot");
    get_raw(in, c.iters56); get_raw(in, c.iters7); get_raw(in, c.buckets); get_raw(in, c.floor);
    v6 = load_mlp(in); v7 = load_mlp(in);
    c.hidden = v6.h1;
}
inline void put_examples(std::ofstream& out, const std::deque<Example>& b) {
    put_raw(out, static_cast<uint64_t>(b.size()));
    for (const Example& e : b) {
        put_sparse(out, e.x);
        put_raw(out, static_cast<uint64_t>(e.y.size()));
        out.write(reinterpret_cast<const char*>(e.y.data()), static_cast<std::streamsize>(e.y.size() * sizeof(e.y[0])));
    }
}
inline void get_examples(std::ifstream& in, std::deque<Example>& b) {
    uint64_t n = 0; get_raw(in, n);
    b.clear();
    for (uint64_t k = 0; k < n; ++k) {
        Example e;
        get_sparse(in, e.x);
        uint64_t m = 0; get_raw(in, m);
        e.y.resize(static_cast<size_t>(m));
        in.read(reinterpret_cast<char*>(e.y.data()), static_cast<std::streamsize>(m * sizeof(e.y[0])));
        b.push_back(std::move(e));
    }
}
inline void save_checkpoint(const std::string& path, const std::string& config, const Trainer& t, const NetLoop& L,
                            OnlineLbr* online = nullptr) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const uint64_t n = config.size();
        put_raw(out, n); out.write(config.data(), static_cast<std::streamsize>(n));
        save_mlp(out, t.v6); save_mlp(out, t.v7);
        save_adam(out, t.o6); save_adam(out, t.o7);
        put_examples(out, t.buf6); put_examples(out, t.buf7);
        put_rng(out, t.rng);
        for (long long v : {t.episodes, t.examples6, t.examples7, t.steps, t.solves[0], t.solves[1], t.solves[2]}) put_raw(out, v);
        for (double v : {t.loss6, t.loss7, t.solve_s[0], t.solve_s[1], t.solve_s[2], t.train_s, t.gap7}) put_raw(out, v);
        put_raw(out, L);
        if (online && online->on()) online->save(out);
    }
    std::filesystem::rename(tmp, path);
}
inline bool load_checkpoint(const std::string& path, const std::string& config, Trainer& t, NetLoop& L,
                            OnlineLbr* online = nullptr, uint64_t* online_bytes = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t n = 0; get_raw(in, n);
    std::string stored(n, '\0'); in.read(stored.data(), static_cast<std::streamsize>(n));
    if (stored != config) throw std::runtime_error("checkpoint was written with different settings:\n  " + stored + "\nnow:\n  " + config);
    t.v6 = load_mlp(in); t.v7 = load_mlp(in);
    t.o6.init(t.v6); t.o7.init(t.v7);
    load_adam(in, t.o6); load_adam(in, t.o7);
    get_examples(in, t.buf6); get_examples(in, t.buf7);
    get_rng(in, t.rng);
    for (long long* v : {&t.episodes, &t.examples6, &t.examples7, &t.steps, &t.solves[0], &t.solves[1], &t.solves[2]}) get_raw(in, *v);
    for (double* v : {&t.loss6, &t.loss7, &t.solve_s[0], &t.solve_s[1], &t.solve_s[2], &t.train_s, &t.gap7}) get_raw(in, *v);
    get_raw(in, L);
    if (online && online->on()) *online_bytes = online->load(in);
    if (!in) throw std::runtime_error("truncated checkpoint " + path);
    return true;
}

// ---------------------------------------------------------------- test-time agent (TargetPolicy of the engine's LBR)
struct Agent {
    const Mlp* v6 = nullptr;
    const Mlp* v7 = nullptr;
    Config cfg;
    mutable std::mt19937_64 rng;
    struct Entry {
        std::string key;
        std::vector<double> r[2];   // root ranges (normalised)
        std::shared_ptr<PairSolve> ps;
        std::shared_ptr<SeventhSolve> ss;
    };
    mutable std::deque<Entry> cache;
    mutable long long solves[3] = {0, 0, 0}, queries = 0, misses = 0;
    mutable double solve_seconds = 0;

    static std::string key_of(const State& s) {
        std::string k(1, static_cast<char>(s.street));
        for (const Player& p : s.players) { for (const Card& c : p.shown) k += static_cast<char>(card_index(c)); k += '|'; }
        for (const Event& e : s.history) { k += static_cast<char>(e.street); k += static_cast<char>(e.actor); k += static_cast<char>(e.action); }
        return k;
    }
    // the solve at a street root (ranges r are used only when it is not cached)
    const Entry& get(const State& root, const std::vector<double>* r) const {
        const std::string k = key_of(root);
        for (auto it = cache.rbegin(); it != cache.rend(); ++it) if (it->key == k) return *it;
        Entry e; e.key = k;
        if (root.street == 5) { e.r[0] = h4_range(root, 0); e.r[1] = h4_range(root, 1); normalise(e.r[0]); normalise(e.r[1]); }
        else { e.r[0] = r[0]; e.r[1] = r[1]; }
        const auto t0 = std::chrono::steady_clock::now();
        if (root.street < 7) e.ps = std::make_shared<PairSolve>(solve_pairs(root, e.r, root.street == 5 ? *v6 : *v7, cfg.iters56));
        else e.ss = std::make_shared<SeventhSolve>(solve_seventh(root, e.r, cfg.iters7, cfg.buckets));
        solve_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++solves[root.street - 5];
        cache.push_back(std::move(e));
        while (cache.size() > 8) cache.pop_front();
        return cache.back();
    }
    static int pair_of_cards(const Card& a, const Card& b) { return PT().idx[card_index(a)][card_index(b)]; }

    std::array<double, kActionCount> policy(const State& st, int seat, bool* found = nullptr) const {
        ++queries;
        if (found) *found = true;
        std::array<double, kActionCount> out{};
        const uint8_t legal = valid_mask(st, seat);
        auto uniform = [&] {
            ++misses;
            if (found) *found = false;
            const int n = __builtin_popcount(legal);
            for (int a = 0; a < kActionCount; ++a) out[a] = (legal >> a & 1) ? 1.0 / n : 0.0;
            return out;
        };
        // replay the public history from the 5th-street root
        State cur;
        cur.ante = st.ante; cur.effective_stack = st.effective_stack; cur.pot = 2 * st.ante;
        for (int s = 0; s < 2; ++s) {
            cur.players[s].invested = st.ante;
            if (st.players[s].shown.size() < 3) return uniform();
            cur.players[s].shown.assign(st.players[s].shown.begin(), st.players[s].shown.begin() + 3);
        }
        reset_round(cur, 5);
        std::vector<double> r[2];
        size_t ev = 0;
        for (int street = 5; street <= 7; ++street) {
            const Entry& E = get(cur, r);
            std::vector<Action> acts;
            while (ev < st.history.size() && st.history[ev].street == street) acts.push_back(st.history[ev++].action);
            const Tree& G = E.ps ? E.ps->tree : E.ss->tree;
            if (street == st.street) {
                const int v = G.walk(acts);
                if (v < 0 || G.nodes[v].kind != 0 || G.nodes[v].actor != seat || G.nodes[v].mask != legal) return uniform();
                if (E.ps) {
                    const auto& h = st.players[seat].hidden;
                    if (h.size() < 2) return uniform();
                    const int t = E.ps->T.of_pair[pair_of_cards(h[0], h[1])];
                    if (t < 0) return uniform();
                    for (int a = 0; a < kActionCount; ++a) if (legal >> a & 1) out[a] = E.ps->prob(v, t, a);
                } else {
                    std::vector<Card> cards = st.players[seat].hidden;
                    if (cards.size() < 3) return uniform();
                    cards.insert(cards.end(), st.players[seat].shown.begin(), st.players[seat].shown.end());
                    const int b = E.ss->bucket_of(seat, best_hand_encoded(cards));
                    for (int a = 0; a < kActionCount; ++a) if (legal >> a & 1) out[a] = E.ss->prob(v, b, a);
                }
                return out;
            }
            if (street == 7 || !E.ps) return uniform();  // the history goes past the last street
            // finish this street: Bayes update of the actor's range with the search policy, then the next root
            r[0] = E.r[0]; r[1] = E.r[1];
            int v = 0;
            for (Action a : acts) {
                const Node& nd = G.nodes[v];
                if (nd.kind != 0 || !(nd.mask >> a & 1)) return uniform();
                for (int p = 0; p < NP; ++p) {
                    if (r[nd.actor][p] <= 0) continue;
                    const int t = E.ps->T.of_pair[p];
                    r[nd.actor][p] *= std::max(cfg.floor, t >= 0 ? E.ps->prob(v, t, a) : 0.0);
                }
                v = nd.child[a];
            }
            if (v < 0 || G.nodes[v].kind != 2) return uniform();
            cur = G.leaf_state[G.nodes[v].leaf];
            if (street == 5) {
                for (int s = 0; s < 2; ++s) {
                    if (st.players[s].shown.size() < 4) return uniform();
                    cur.players[s].shown.push_back(st.players[s].shown[3]);
                }
                block(r[0], public_mask(cur)); block(r[1], public_mask(cur));
            }
            normalise(r[0]); normalise(r[1]);
            reset_round(cur, street + 1);
        }
        return uniform();
    }
    Action choose(const State& state, int seat, int) {
        const auto p = policy(state, seat);
        return sample_action(p, valid_mask(state, seat), rng);
    }
    bool learned_h4() const { return false; }
    double h4_weight(const std::array<Card, 2>&, Card, Card, int) const { throw std::runtime_error("ReBeL uses the heuristic H4"); }
};
}  // namespace rebel

int train_rebel(int argc, char** argv) {
    // the options run_100m passes to every arm are accepted; --z, --atlas, --power-cache, --h4 and --threads have no effect
    auto o = parse(argc, argv, {{"--partition", "rebel"}, {"--atlas", ""}, {"--roots", "20000"}, {"--epoch0", "64"}, {"--z", "2"},
                                {"--seed", "1"}, {"--out", ""}, {"--power-cache", "1000000"}, {"--checkpoint-every", "500"},
                                {"--snapshot-every", "0"}, {"--curve-every", "100"}, {"--h4", "heuristic"}, {"--threads", "1"},
                                {"--rebel-iters56", "32"}, {"--rebel-iters7", "64"}, {"--rebel-buckets", "64"},
                                {"--rebel-hidden", "128"}, {"--rebel-lr", "3e-4"}, {"--rebel-batch", "32"},
                                {"--rebel-train-steps", "4"}, {"--rebel-capacity", "4096"}, {"--rebel-explore", "0.1"},
                                {"--online-lbr-every", "0"}, {"--online-lbr-particles", "16"}, {"--online-lbr-policy", ""}});
    if (o["--h4"] != "heuristic") throw std::runtime_error("--partition rebel uses the heuristic H4 (--h4 heuristic)");
    const std::string out = o.at("--out");
    if (out.empty()) throw std::runtime_error("--out is required");
    std::filesystem::create_directories(out);
    rebel::Config c;
    c.seed = std::stoull(o["--seed"]);
    c.iters56 = std::stoi(o["--rebel-iters56"]); c.iters7 = std::stoi(o["--rebel-iters7"]); c.buckets = std::stoi(o["--rebel-buckets"]);
    c.hidden = std::stoi(o["--rebel-hidden"]); c.lr = std::stof(o["--rebel-lr"]); c.batch = std::stoi(o["--rebel-batch"]);
    c.train_steps = std::stoi(o["--rebel-train-steps"]); c.capacity = std::stoi(o["--rebel-capacity"]);
    c.explore = std::stod(o["--rebel-explore"]);
    const long long T = std::stoll(o["--roots"]), epoch0 = std::stoll(o["--epoch0"]);
    const long long checkpoint_every = std::stoll(o["--checkpoint-every"]), snapshot_every = std::stoll(o["--snapshot-every"]);
    const long long curve_every = std::stoll(o["--curve-every"]);
    OnlineLbr online;  // ReBeL: its search policy, as "current"
    online.setup(o, c.seed, false, true);
    const std::string config = "rebel;seed=" + std::to_string(c.seed) + ";epoch0=" + std::to_string(epoch0) + ";" + rebel::config_string(c) +
                               online.config();
    rebel::Trainer tr(c);
    const long long params = static_cast<long long>(tr.v6.parameters() + tr.v7.parameters());
    NetLoop L;
    L.epoch_len = epoch0; L.next_epoch_roots = epoch0;
    L.next_snapshot = snapshot_every > 0 ? snapshot_every : std::numeric_limits<long long>::max();
    L.next_curve = curve_every > 0 ? curve_every : std::numeric_limits<long long>::max();
    L.next_checkpoint = checkpoint_every > 0 ? checkpoint_every : std::numeric_limits<long long>::max();
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    const std::string ck_path = out + "/checkpoint.net";
    uint64_t online_bytes = 0;
    const bool resumed = rebel::load_checkpoint(ck_path, config, tr, L, &online, &online_bytes);
    if (resumed) {
        std::filesystem::resize_file(log_path, L.log_bytes);
        std::filesystem::resize_file(index_path, L.index_bytes);
        std::filesystem::resize_file(curve_path, L.curve_bytes);
        std::cerr << "resumed at hand " << tr.episodes << "\n";
    } else {
        if (std::filesystem::exists(log_path)) throw std::runtime_error(out + " already has log.csv but no usable checkpoint; use a new --out");
        std::ofstream(log_path) << "epoch,roots,contexts,buckets_after,examples6,examples7,loss6,loss7,solves5,solves6,solves7,"
                                   "solve5_s,solve6_s,solve7_s,train_s,gap7_mean,seconds\n";
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets,peak_rss_mb\n";
        std::ofstream(out + "/net_kind.txt") << "rebel\n";
    }
    std::ofstream log(log_path, std::ios::app), index(index_path, std::ios::app), curve(curve_path, std::ios::app);
    online.open(out, resumed, online_bytes);
    const double base = L.seconds, online0 = online.seconds;
    const auto t0 = std::chrono::steady_clock::now();
    // training seconds: the online LBR's own time is excluded (it is in online_lbr.csv)
    auto elapsed = [&] { return base + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - (online.seconds - online0); };
    auto checkpoint = [&] {
        log.flush(); index.flush(); curve.flush();
        L.seconds = elapsed();
        L.log_bytes = static_cast<std::streamoff>(std::filesystem::file_size(log_path));
        L.index_bytes = static_cast<std::streamoff>(std::filesystem::file_size(index_path));
        L.curve_bytes = static_cast<std::streamoff>(std::filesystem::file_size(curve_path));
        rebel::save_checkpoint(ck_path, config, tr, L, &online);
    };
    auto stats = [&] {
        std::ostringstream s;
        s << tr.examples6 << ',' << tr.examples7 << ',' << tr.loss6 << ',' << tr.loss7 << ',' << tr.solves[0] << ',' << tr.solves[1] << ','
          << tr.solves[2] << ',' << tr.solve_s[0] << ',' << tr.solve_s[1] << ',' << tr.solve_s[2] << ',' << tr.train_s << ','
          << (tr.solves[2] ? tr.gap7 / tr.solves[2] : 0.0);
        return s.str();
    };
    std::cerr << "rebel: value networks " << rebel::IN << " -> " << c.hidden << " -> " << c.hidden << " -> " << rebel::OUT << ", "
              << params << " parameters\n";
    while (tr.episodes < T) {
        if (std::filesystem::exists(out + "/STOP")) {
            checkpoint();
            std::filesystem::remove(out + "/STOP");
            std::cerr << "stopped at hand " << tr.episodes << "; checkpoint saved, rerun the same command to resume\n";
            return 0;
        }
        if (online.due(tr.episodes + 1)) {  // test, then train
            const long long t = tr.episodes + 1;
            rebel::Agent agent;
            agent.v6 = &tr.v6; agent.v7 = &tr.v7; agent.cfg = c;
            agent.rng.seed(online.target_seed(t, "current"));
            online.evaluate(t, online.deck_at(t), agent, "current", elapsed());
        }
        tr.episode();
        tr.train();
        ++tr.episodes;
        const long long done = tr.episodes;
        const double secs = elapsed();
        while (done >= L.next_curve) {
            curve << done << ',' << secs << ',' << done - L.curve_roots0 << ",nan,nan," << params << ",-1\n";
            L.curve_roots0 = done; L.next_curve += curve_every;
        }
        bool saved = false;
        if (done >= L.next_epoch_roots || done >= T) {
            const std::string file = "snapshot_" + std::to_string(L.k) + ".net";
            rebel::save_snapshot(out + "/" + file, c, tr.v6, tr.v7);
            index << file << ',' << done << ",epoch," << L.k << ',' << params << ',' << secs << '\n';
            log << L.k << ',' << done << ',' << params << ',' << params << ',' << stats() << ',' << secs << '\n';
            std::cerr << "epoch " << L.k << " hands=" << done << " " << stats() << " (" << secs << "s)\n";
            ++L.k;
            while (L.next_epoch_roots <= done) { L.epoch_len *= 2; L.next_epoch_roots += L.epoch_len; }
            saved = true;
        }
        if (done >= L.next_snapshot && done < T) {
            const std::string file = "snapshot_r" + std::to_string(done) + ".net";
            rebel::save_snapshot(out + "/" + file, c, tr.v6, tr.v7);
            index << file << ',' << done << ",periodic," << L.k << ',' << params << ',' << secs << '\n';
            while (L.next_snapshot <= done) L.next_snapshot += snapshot_every;
        }
        if (done >= L.next_checkpoint || saved || done >= T) {
            while (L.next_checkpoint <= done) L.next_checkpoint += checkpoint_every;
            checkpoint();
        }
    }
    return 0;
}

// same LBR deals and particles as eval(); the LBR particle RNG is seeded per pair (seed, pair) so that --threads
// gives the same numbers. --policy is accepted for evaluate.py (ReBeL plays its search policy).
int eval_rebel(std::map<std::string, std::string>& o) {
    rebel::Config c;
    Mlp v6, v7;
    rebel::load_snapshot(o.at("--snapshot"), c, v6, v7);
    // diagnostics: search settings at test time other than the snapshot's (default: the snapshot's own)
    if (!o["--rebel-iters56"].empty()) c.iters56 = std::stoi(o["--rebel-iters56"]);
    if (!o["--rebel-iters7"].empty()) c.iters7 = std::stoi(o["--rebel-iters7"]);
    if (!o["--rebel-buckets"].empty()) c.buckets = std::stoi(o["--rebel-buckets"]);
    const uint64_t seed = std::stoull(o["--seed"]);
    const int pairs = std::stoi(o["--pairs"]), particles = std::stoi(o["--particles"]);
    const int threads = std::max(1, std::stoi(o.count("--threads") && !o["--threads"].empty() ? o["--threads"] : "1"));
    std::mt19937_64 deal_rng(seed);
    std::vector<std::vector<Card>> decks(pairs);
    for (auto& deck : decks) { deck = fresh_deck(); std::shuffle(deck.begin(), deck.end(), deal_rng); }
    std::vector<double> paired(pairs, 0.0);
    std::vector<long long> solves(3, 0), queries(threads, 0), misses(threads, 0);
    std::vector<double> solve_seconds(threads, 0.0);
    std::vector<std::array<long long, 3>> tsolves(threads, {0, 0, 0});
    std::atomic<int> next{0};
    const auto t0 = std::chrono::steady_clock::now();
    auto worker = [&](int w) {
        for (int i = next++; i < pairs; i = next++) {
            rebel::Agent agent;
            agent.v6 = &v6; agent.v7 = &v7; agent.cfg = c;
            agent.rng.seed(seed ^ 0x5851f42d4c957f2dull ^ (0x9e3779b97f4a7c15ull * static_cast<uint64_t>(i + 1)));
            PolicyLBR<rebel::Agent> lbr(agent, particles, seed ^ 0x94d049bb133111ebull ^ (0xbf58476d1ce4e5b9ull * static_cast<uint64_t>(i + 1)));
            std::array<double, 2> value{};
            for (int lbr_seat = 0; lbr_seat < 2; ++lbr_seat)
                value[lbr_seat] = play_hand_policy_lbr(decks[i], lbr_seat, agent, lbr, kAnte, kEffectiveStackAnte);
            paired[i] = (value[0] + value[1]) / 2.0;
            queries[w] += agent.queries; misses[w] += agent.misses; solve_seconds[w] += agent.solve_seconds;
            for (int k = 0; k < 3; ++k) tsolves[w][k] += agent.solves[k];
        }
    };
    std::vector<std::thread> pool;
    for (int w = 0; w < threads; ++w) pool.emplace_back(worker, w);
    for (auto& th : pool) th.join();
    const double lbr_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double m = 0, v = 0;
    for (double y : paired) m += y;
    m /= std::max(1, pairs);
    for (double y : paired) v += (y - m) * (y - m);
    const double se = pairs > 1 ? std::sqrt(v / (pairs - 1) / pairs) : 0.0;
    if (!o["--pairs-csv"].empty()) {
        std::ofstream csv(o["--pairs-csv"]);
        csv << "pair,lbr_ante\n";
        for (int i = 0; i < pairs; ++i) csv << i << ',' << paired[i] << '\n';
    }
    long long q = 0, mi = 0; double ss = 0;
    for (int w = 0; w < threads; ++w) { q += queries[w]; mi += misses[w]; ss += solve_seconds[w]; for (int k = 0; k < 3; ++k) solves[k] += tsolves[w][k]; }
    std::cout << std::setprecision(9) << "{\"snapshot\":\"" << json_escape(o["--snapshot"]) << "\",\"policy\":\"" << o["--policy"]
              << "\",\"pairs\":" << pairs << ",\"particles\":" << particles << ",\"seed\":" << seed << ",\"lbr_ante\":" << m
              << ",\"lbr_se\":" << se << ",\"gap_roots\":0,\"gap_proxy_ante\":NaN,\"gap_proxy_se\":NaN"
              << ",\"policy_queries\":" << q << ",\"policy_misses\":" << mi << ",\"lbr_seconds\":" << lbr_seconds
              << ",\"threads\":" << threads << ",\"search\":\"" << rebel::config_string(c) << "\",\"solves\":[" << solves[0] << ',' << solves[1] << ',' << solves[2] << "],\"solve_seconds\":" << ss
              << ",\"h4\":\"heuristic\",\"h4_buckets\":0,\"rho_roots\":0,\"rho_plain_half\":-1,\"rho_cross_half\":-1,\"parameters\":"
              << v6.parameters() + v7.parameters() << ",\"lbr_rng\":\"per_pair\"}\n";
    return 0;
}

}  // namespace sota
