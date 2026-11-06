// `train --partition net` and `eval --partition net` of aswiss_stud.cpp (see stud7_net.hpp). Included after parse(),
// json_escape() and the tabular eval, before main().
//   --algo mmd     MMD self-play (mmd.hpp); snapshots hold the policy network (one policy: the last iterate)
//   --algo escher  ESCHER (escher.hpp); snapshots hold the average policy network and both regret networks
// OUT/net_kind.txt names the algorithm, so evaluate.py knows which policies a snapshot has.
#pragma once
#include "stud7_net.hpp"

namespace sota {

int train_net(int argc, char** argv) {
    // the options run_100m passes to every arm are accepted; --z, --atlas, --power-cache and --threads have no effect here
    auto o = parse(argc, argv, {{"--partition", "net"}, {"--algo", "mmd"}, {"--atlas", ""}, {"--roots", "1000000"},
                                {"--epoch0", "10000"}, {"--z", "2"}, {"--seed", "1"}, {"--out", ""}, {"--power-cache", "1000000"},
                                {"--checkpoint-every", "1000000"}, {"--snapshot-every", "0"}, {"--curve-every", "10000"},
                                {"--h4", "heuristic"}, {"--threads", "1"},
                                {"--mmd-episodes", "2048"}, {"--mmd-epochs", "2"}, {"--mmd-minibatch", "512"}, {"--mmd-lr", "3e-4"},
                                {"--mmd-value-lr", "1e-3"}, {"--mmd-alpha", "0.2"}, {"--mmd-alpha-halflife", "0"}, {"--mmd-beta", "1"},
                                {"--mmd-magnet-every", "0"}, {"--mmd-hidden1", "128"}, {"--mmd-hidden2", "64"},
                                {"--mmd-reward-scale", "10"}, {"--mmd-normalize", "1"},
                                {"--escher-value-hands", "4096"}, {"--escher-value-steps", "200"}, {"--escher-traversals", "8192"},
                                {"--escher-regret-steps", "300"}, {"--escher-avg-steps", "2000"}, {"--escher-batch", "512"},
                                {"--escher-capacity", "500000"}, {"--escher-lr", "1e-3"}, {"--escher-value-lr", "1e-3"},
                                {"--escher-hidden1", "128"}, {"--escher-hidden2", "64"}, {"--escher-reward-scale", "10"},
                                {"--escher-reinit", "1"}, {"--escher-value-explore", "0.25"},
                                {"--online-lbr-every", "0"}, {"--online-lbr-particles", "16"}, {"--online-lbr-policy", ""},
                                {"--online-lbr-avg-steps", "500"}});
    const std::string algo = o["--algo"];
    if (algo != "mmd" && algo != "escher") throw std::runtime_error("--partition net needs --algo mmd or escher");
    if (o["--h4"] != "heuristic") throw std::runtime_error("--partition net uses the heuristic H4 (--h4 heuristic)");
    const std::string out = o.at("--out");
    if (out.empty()) throw std::runtime_error("--out is required");
    std::filesystem::create_directories(out);
    const uint64_t seed = std::stoull(o["--seed"]);
    const long long T = std::stoll(o["--roots"]), epoch0 = std::stoll(o["--epoch0"]);
    const long long checkpoint_every = std::stoll(o["--checkpoint-every"]), snapshot_every = std::stoll(o["--snapshot-every"]);
    const long long curve_every = std::stoll(o["--curve-every"]);
    std::ostringstream cfg;  // settings that must match on resume (--roots may grow; output cadences may change)
    cfg << "net;algo=" << algo << ";seed=" << seed << ";epoch0=" << epoch0;
    for (const auto& [k, v] : o) if (k.rfind("--" + algo + "-", 0) == 0) cfg << ";" << k.substr(2) << "=" << v;
    cfg << ";input=deep_cfr_tensor";
    // online LBR: MMD has only its last iterate; ESCHER's average network is trained only at snapshots, so its online
    // rows are the current policy (regret matching on the regret networks)
    // online LBR policies: MMD has only its last iterate (current). ESCHER: average (default) = a monitoring average
    // network trained, before every iteration with due evaluations, --online-lbr-avg-steps warm-started steps on the
    // strategy reservoir with its own RNG (the training is unchanged; its time counts as evaluation time); current =
    // regret matching on the regret networks.
    OnlineLbr online;
    online.setup(o, seed, algo == "escher", true);
    const int monitor_steps = std::stoi(o["--online-lbr-avg-steps"]);
    cfg << online.config() << (online.on() && algo == "escher" ? ";online_avg_steps=" + o["--online-lbr-avg-steps"] : "");
    const std::string config = cfg.str();
    Stud7Game game;
    std::unique_ptr<Mmd<Stud7Game>> mmd;
    std::unique_ptr<Escher<Stud7Game>> esc;
    if (algo == "mmd") {
        MmdConfig c;
        c.episodes = std::stoi(o["--mmd-episodes"]); c.epochs = std::stoi(o["--mmd-epochs"]); c.minibatch = std::stoi(o["--mmd-minibatch"]);
        c.lr = std::stof(o["--mmd-lr"]); c.value_lr = std::stof(o["--mmd-value-lr"]); c.alpha0 = std::stod(o["--mmd-alpha"]);
        c.alpha_halflife = std::stod(o["--mmd-alpha-halflife"]); c.beta = std::stod(o["--mmd-beta"]);
        c.magnet_every = std::stoi(o["--mmd-magnet-every"]); c.hidden1 = std::stoi(o["--mmd-hidden1"]);
        c.hidden2 = std::stoi(o["--mmd-hidden2"]); c.reward_scale = std::stod(o["--mmd-reward-scale"]);
        c.normalize = o["--mmd-normalize"] == "1"; c.seed = seed;
        mmd = std::make_unique<Mmd<Stud7Game>>(game, c);
    } else {
        EscherConfig c;
        c.value_hands = std::stoi(o["--escher-value-hands"]); c.value_steps = std::stoi(o["--escher-value-steps"]);
        c.traversals = std::stoi(o["--escher-traversals"]); c.regret_steps = std::stoi(o["--escher-regret-steps"]);
        c.avg_steps = std::stoi(o["--escher-avg-steps"]); c.batch = std::stoi(o["--escher-batch"]);
        c.regret_capacity = c.strategy_capacity = std::stoull(o["--escher-capacity"]);
        c.lr = std::stof(o["--escher-lr"]); c.value_lr = std::stof(o["--escher-value-lr"]);
        c.hidden1 = std::stoi(o["--escher-hidden1"]); c.hidden2 = std::stoi(o["--escher-hidden2"]);
        c.reward_scale = std::stod(o["--escher-reward-scale"]); c.reinit = o["--escher-reinit"] == "1";
        c.value_explore = std::stod(o["--escher-value-explore"]); c.seed = seed;
        esc = std::make_unique<Escher<Stud7Game>>(game, c);
    }
    const long long params = static_cast<long long>(mmd ? mmd->pol.parameters() : esc->avg.parameters());
    Mlp monitor; Adam monitor_opt; std::mt19937_64 monitor_rng(seed ^ 0x6a09e667f3bcc908ull);
    const bool monitor_on = esc && online.on() &&
                            std::find(online.policies.begin(), online.policies.end(), "average") != online.policies.end();
    if (monitor_on) {
        monitor = Mlp(game.input_dim(), esc->cfg.hidden1, esc->cfg.hidden2, Stud7Game::A, seed * 7 + 77);
        monitor_opt.lr = esc->cfg.lr; monitor_opt.init(monitor);
        online.extra_save = [&](std::ofstream& f) { save_mlp(f, monitor); save_adam(f, monitor_opt); put_rng(f, monitor_rng); };
        online.extra_load = [&](std::ifstream& f) {
            monitor = load_mlp(f); monitor_opt.init(monitor); load_adam(f, monitor_opt); get_rng(f, monitor_rng);
        };
    }
    auto roots_done = [&] { return mmd ? mmd->episodes_done : esc->roots_done; };
    NetLoop L;
    L.epoch_len = epoch0; L.next_epoch_roots = epoch0;
    L.next_snapshot = snapshot_every > 0 ? snapshot_every : std::numeric_limits<long long>::max();
    L.next_curve = curve_every > 0 ? curve_every : std::numeric_limits<long long>::max();
    L.next_checkpoint = checkpoint_every > 0 ? checkpoint_every : std::numeric_limits<long long>::max();
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    const std::string ck_path = out + "/checkpoint.net";
    uint64_t online_bytes = 0;
    const bool resumed = mmd ? load_checkpoint(ck_path, config, *mmd, L, &online, &online_bytes)
                             : load_checkpoint(ck_path, config, *esc, L, &online, &online_bytes);
    const std::string head = mmd ? "updates,samples,alpha,entropy,kl_old,value_loss,return_abs" : "iterations,regret_samples,value_loss,regret_loss0,regret_loss1";
    if (resumed) {
        std::filesystem::resize_file(log_path, L.log_bytes);
        std::filesystem::resize_file(index_path, L.index_bytes);
        std::filesystem::resize_file(curve_path, L.curve_bytes);
        std::cerr << "resumed at root " << roots_done() << "\n";
    } else {
        if (std::filesystem::exists(log_path)) throw std::runtime_error(out + " already has log.csv but no usable checkpoint; use a new --out");
        std::ofstream(log_path) << "epoch,roots,contexts,buckets_after," << head << ",seconds\n";
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets,peak_rss_mb,rss_mb,commit_mb\n";
        std::ofstream(out + "/net_kind.txt") << algo << "\n";
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
        if (mmd) save_checkpoint(ck_path, config, *mmd, L, &online); else save_checkpoint(ck_path, config, *esc, L, &online);
    };
    auto snapshot = [&](const std::string& file) {
        if (mmd) { save_snapshot(out + "/" + file, mmd->pol); return; }
        esc->train_average();
        save_escher_snapshot(out + "/" + file, esc->avg, esc->regret[0], esc->regret[1]);
    };
    auto stats = [&] {
        std::ostringstream s;
        if (mmd) s << mmd->updates << ',' << mmd->samples_done << ',' << mmd->alpha() << ',' << mmd->last_entropy << ','
                   << mmd->last_kl << ',' << mmd->last_value_loss << ',' << mmd->last_return_abs;
        else s << esc->t << ',' << esc->regret_samples << ',' << esc->last_value_loss << ',' << esc->last_regret_loss[0] << ','
               << esc->last_regret_loss[1];
        return s.str();
    };
    std::cerr << algo << ": input " << game.input_dim() << ", " << params << " policy parameters\n";
    while (roots_done() < T) {
        if (std::filesystem::exists(out + "/STOP")) {
            checkpoint();
            std::filesystem::remove(out + "/STOP");
            std::cerr << "stopped at root " << roots_done() << "; checkpoint saved, rerun the same command to resume\n";
            return 0;
        }
        if (online.on()) {  // test, then train: the roots of the next update / iteration, on the tabular deal stream
            const long long d0 = roots_done();
            const long long step = mmd ? mmd->cfg.episodes : esc->cfg.value_hands + 2LL * esc->cfg.traversals;
            const long long first = (d0 / online.every + 1) * online.every;
            if (monitor_on && first <= d0 + step) {  // refresh the monitoring average network once for this iteration
                const auto m0 = std::chrono::steady_clock::now();
                esc->train_average_monitor(monitor, monitor_opt, monitor_rng, monitor_steps);
                online.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - m0).count();
            }
            for (long long t = first; t <= d0 + step; t += online.every) {
                for (const auto& policy : online.policies) {
                    const bool average = policy == "average";
                    NetPolicy target{mmd ? mmd->pol : average ? monitor : esc->avg, std::mt19937_64(online.target_seed(t, policy))};
                    if (esc && !average) { target.regret[0] = &esc->regret[0]; target.regret[1] = &esc->regret[1]; }
                    online.evaluate(t, online.deck_at(t), target, policy, elapsed());
                }
            }
        }
        if (mmd) { auto batch = mmd->collect(); mmd->update(batch); }
        else esc->iterate();
        const long long done = roots_done();
        const double secs = elapsed();
        while (done >= L.next_curve) {
            curve << done << ',' << secs << ',' << done - L.curve_roots0 << ",nan,nan," << params << ',' << memory_columns() << '\n';
            L.curve_roots0 = done; L.next_curve += curve_every;
        }
        bool saved = false;
        if (done >= L.next_epoch_roots || done >= T) {
            const std::string file = "snapshot_" + std::to_string(L.k) + ".net";
            snapshot(file);
            const double s2 = elapsed();  // includes training the average network (ESCHER)
            index << file << ',' << done << ",epoch," << L.k << ',' << params << ',' << s2 << '\n';
            log << L.k << ',' << done << ',' << params << ',' << params << ',' << stats() << ',' << s2 << '\n';
            std::cerr << "epoch " << L.k << " roots=" << done << " " << stats() << " (" << s2 << "s)\n";
            ++L.k;
            while (L.next_epoch_roots <= done) { L.epoch_len *= 2; L.next_epoch_roots += L.epoch_len; }
            saved = true;
        }
        if (done >= L.next_snapshot && done < T) {
            const std::string file = "snapshot_r" + std::to_string(done) + ".net";
            snapshot(file);
            index << file << ',' << done << ",periodic," << L.k << ',' << params << ',' << elapsed() << '\n';
            while (L.next_snapshot <= done) L.next_snapshot += snapshot_every;
        }
        if (done >= L.next_checkpoint || saved || done >= T) {
            while (L.next_checkpoint <= done) L.next_checkpoint += checkpoint_every;
            checkpoint();
        }
    }
    return 0;
}

// same LBR deals / seed / particles and gap deals as eval(). MMD snapshots have one policy (--policy current);
// ESCHER snapshots have the average policy network (--policy average) and regret matching (--policy current).
int eval_net(std::map<std::string, std::string>& o) {
    const NetSnapshot snap = load_snapshot(o.at("--snapshot"));
    const bool average = o["--policy"] == "average";
    if (!average && o["--policy"] != "current") throw std::runtime_error("--policy must be average or current");
    if (average && !snap.escher) throw std::runtime_error("this network snapshot has only the current (last-iterate) policy: --policy current");
    const uint64_t seed = std::stoull(o["--seed"]);
    NetPolicy target{snap.policy, std::mt19937_64(seed ^ 0x5851f42d4c957f2dull)};
    if (snap.escher && !average) { target.regret[0] = &snap.regret[0]; target.regret[1] = &snap.regret[1]; }
    std::mt19937_64 deal_rng(seed);
    const int pairs = std::stoi(o["--pairs"]);
    std::vector<double> paired;
    const auto t0 = std::chrono::steady_clock::now();
    PolicyLBR<NetPolicy> lbr(target, std::stoi(o["--particles"]), seed ^ 0x94d049bb133111ebull);
    for (int i = 0; i < pairs; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), deal_rng);
        std::array<double, 2> value{};
        for (int lbr_seat = 0; lbr_seat < 2; ++lbr_seat)
            value[lbr_seat] = play_hand_policy_lbr(deck, lbr_seat, target, lbr, kAnte, kEffectiveStackAnte);
        paired.push_back((value[0] + value[1]) / 2.0);
    }
    const double lbr_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto mean_se = [](const std::vector<double>& x) {
        if (x.empty()) return std::make_pair(0.0, 0.0);
        const double m = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
        double v = 0;
        for (double y : x) v += (y - m) * (y - m);
        return std::make_pair(m, x.size() > 1 ? std::sqrt(v / (x.size() - 1) / x.size()) : 0.0);
    };
    if (!o["--pairs-csv"].empty()) {
        std::ofstream csv(o["--pairs-csv"]);
        csv << "pair,lbr_ante\n";
        for (size_t i = 0; i < paired.size(); ++i) csv << i << ',' << paired[i] << '\n';
    }
    std::mt19937_64 gap_deals(seed ^ 0xd1b54a32d192ed03ull), gap_rng(seed ^ 0x2545f4914f6cdd1dull);
    std::vector<double> gaps;
    const int gap_roots = std::stoi(o["--gap-roots"]);
    for (int i = 0; i < gap_roots; ++i) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), gap_deals);
        const State root = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        double gap0 = 0, gap1 = 0;
        gap_traverse_net(target, root, 0, gap_rng, gap0);
        gap_traverse_net(target, root, 1, gap_rng, gap1);
        gaps.push_back((gap0 + gap1) / 2.0);
    }
    const auto [lm, lse] = mean_se(paired);
    const auto [gm, gse] = mean_se(gaps);
    const auto st = lbr.stats();
    std::cout << std::setprecision(9) << "{\"snapshot\":\"" << json_escape(o["--snapshot"]) << "\",\"policy\":\"" << o["--policy"]
              << "\",\"pairs\":" << pairs << ",\"particles\":" << o["--particles"] << ",\"seed\":" << seed << ",\"lbr_ante\":" << lm
              << ",\"lbr_se\":" << lse << ",\"gap_roots\":" << gap_roots << ",\"gap_proxy_ante\":" << gm << ",\"gap_proxy_se\":" << gse
              << ",\"policy_queries\":" << st.policy_queries << ",\"policy_misses\":" << st.policy_misses
              << ",\"lbr_seconds\":" << lbr_seconds << ",\"h4\":\"heuristic\",\"h4_buckets\":0,\"rho_roots\":0,\"rho_plain_half\":-1,"
              << "\"rho_cross_half\":-1,\"parameters\":" << snap.policy.parameters() << "}\n";
    return 0;
}

}  // namespace sota
