// `train --partition net` and `eval --partition net` of aswiss_stud.cpp (see stud7_net.hpp). Included after parse(),
// json_escape() and the tabular eval, before main().
#pragma once
#include "stud7_net.hpp"

namespace sota {

int train_net(int argc, char** argv) {
    // the options run_100m passes to every arm are accepted; --z, --atlas, --power-cache and --threads have no effect here
    auto o = parse(argc, argv, {{"--partition", "net"}, {"--algo", "mmd"}, {"--atlas", ""}, {"--roots", "1000000"},
                                {"--epoch0", "10000"}, {"--z", "2"}, {"--seed", "1"}, {"--out", ""}, {"--power-cache", "1000000"},
                                {"--checkpoint-every", "1000000"}, {"--snapshot-every", "0"}, {"--curve-every", "10000"},
                                {"--h4", "heuristic"}, {"--threads", "1"}, {"--mmd-episodes", "2048"}, {"--mmd-epochs", "2"},
                                {"--mmd-minibatch", "512"}, {"--mmd-lr", "3e-4"}, {"--mmd-value-lr", "1e-3"},
                                {"--mmd-alpha", "0.2"}, {"--mmd-alpha-halflife", "0"}, {"--mmd-beta", "1"},
                                {"--mmd-magnet-every", "0"}, {"--mmd-hidden1", "128"}, {"--mmd-hidden2", "64"},
                                {"--mmd-reward-scale", "10"}});
    if (o["--algo"] != "mmd") throw std::runtime_error("--partition net supports --algo mmd");
    if (o["--h4"] != "heuristic") throw std::runtime_error("--partition net uses the heuristic H4 (--h4 heuristic)");
    const std::string out = o.at("--out");
    if (out.empty()) throw std::runtime_error("--out is required");
    std::filesystem::create_directories(out);
    MmdConfig c;
    c.episodes = std::stoi(o["--mmd-episodes"]); c.epochs = std::stoi(o["--mmd-epochs"]); c.minibatch = std::stoi(o["--mmd-minibatch"]);
    c.lr = std::stof(o["--mmd-lr"]); c.value_lr = std::stof(o["--mmd-value-lr"]); c.alpha0 = std::stod(o["--mmd-alpha"]);
    c.alpha_halflife = std::stod(o["--mmd-alpha-halflife"]); c.beta = std::stod(o["--mmd-beta"]);
    c.magnet_every = std::stoi(o["--mmd-magnet-every"]); c.hidden1 = std::stoi(o["--mmd-hidden1"]); c.hidden2 = std::stoi(o["--mmd-hidden2"]);
    c.reward_scale = std::stod(o["--mmd-reward-scale"]); c.seed = std::stoull(o["--seed"]);
    const long long T = std::stoll(o["--roots"]), epoch0 = std::stoll(o["--epoch0"]);
    const long long checkpoint_every = std::stoll(o["--checkpoint-every"]), snapshot_every = std::stoll(o["--snapshot-every"]);
    const long long curve_every = std::stoll(o["--curve-every"]);
    std::ostringstream cfg;  // settings that must match on resume (--roots may grow; output cadences may change)
    cfg << "net;algo=mmd;seed=" << c.seed << ";epoch0=" << epoch0 << ";episodes=" << c.episodes << ";epochs=" << c.epochs
        << ";minibatch=" << c.minibatch << ";lr=" << o["--mmd-lr"] << ";value_lr=" << o["--mmd-value-lr"] << ";alpha=" << o["--mmd-alpha"]
        << ";alpha_halflife=" << o["--mmd-alpha-halflife"] << ";beta=" << o["--mmd-beta"] << ";magnet_every=" << c.magnet_every
        << ";hidden=" << c.hidden1 << "x" << c.hidden2 << ";reward_scale=" << o["--mmd-reward-scale"] << ";input=deep_cfr_tensor";
    const std::string config = cfg.str();
    Stud7Game game;
    Mmd<Stud7Game> t(game, c);
    const long long params = static_cast<long long>(t.pol.parameters());
    NetLoop L;
    L.epoch_len = epoch0; L.next_epoch_roots = epoch0;
    L.next_snapshot = snapshot_every > 0 ? snapshot_every : std::numeric_limits<long long>::max();
    L.next_curve = curve_every > 0 ? curve_every : std::numeric_limits<long long>::max();
    L.next_checkpoint = checkpoint_every > 0 ? checkpoint_every : std::numeric_limits<long long>::max();
    const std::string log_path = out + "/log.csv", index_path = out + "/snapshots.csv", curve_path = out + "/curve.csv";
    const std::string ck_path = out + "/checkpoint.net";
    const bool resumed = load_checkpoint(ck_path, config, t, L);
    if (resumed) {
        std::filesystem::resize_file(log_path, L.log_bytes);
        std::filesystem::resize_file(index_path, L.index_bytes);
        std::filesystem::resize_file(curve_path, L.curve_bytes);
        std::cerr << "resumed at root " << t.episodes_done << " (update " << t.updates << ")\n";
    } else {
        if (std::filesystem::exists(log_path)) throw std::runtime_error(out + " already has log.csv but no usable checkpoint; use a new --out");
        std::ofstream(log_path) << "epoch,roots,contexts,buckets_after,updates,samples,alpha,entropy,kl_old,value_loss,return_abs,seconds\n";
        std::ofstream(index_path) << "file,roots,kind,epoch,buckets,seconds\n";
        std::ofstream(curve_path) << "roots,seconds,roots_in_chunk,train_gap_mean,train_gap_se,buckets,peak_rss_mb\n";
    }
    std::ofstream log(log_path, std::ios::app), index(index_path, std::ios::app), curve(curve_path, std::ios::app);
    const double base = L.seconds;
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return base + std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto checkpoint = [&] {
        log.flush(); index.flush(); curve.flush();
        L.seconds = elapsed();
        L.log_bytes = static_cast<std::streamoff>(std::filesystem::file_size(log_path));
        L.index_bytes = static_cast<std::streamoff>(std::filesystem::file_size(index_path));
        L.curve_bytes = static_cast<std::streamoff>(std::filesystem::file_size(curve_path));
        save_checkpoint(ck_path, config, t, L);
    };
    std::cerr << "network: input " << game.input_dim() << ", hidden " << c.hidden1 << "x" << c.hidden2 << ", " << params
              << " policy parameters; " << c.episodes << " hands per update\n";
    while (t.episodes_done < T) {
        if (std::filesystem::exists(out + "/STOP")) {
            checkpoint();
            std::filesystem::remove(out + "/STOP");
            std::cerr << "stopped at root " << t.episodes_done << "; checkpoint saved, rerun the same command to resume\n";
            return 0;
        }
        auto batch = t.collect();
        t.update(batch);
        const long long done = t.episodes_done;
        const double secs = elapsed();
        while (done >= L.next_curve) {
            curve << done << ',' << secs << ',' << done - L.curve_roots0 << ",nan,nan," << params << ",-1\n";
            L.curve_roots0 = done; L.next_curve += curve_every;
        }
        bool saved = false;
        if (done >= L.next_epoch_roots || done >= T) {
            const std::string file = "snapshot_" + std::to_string(L.k) + ".net";
            save_snapshot(out + "/" + file, t.pol);
            index << file << ',' << done << ",epoch," << L.k << ',' << params << ',' << secs << '\n';
            log << L.k << ',' << done << ',' << params << ',' << params << ',' << t.updates << ',' << t.samples_done << ',' << t.alpha()
                << ',' << t.last_entropy << ',' << t.last_kl << ',' << t.last_value_loss << ',' << t.last_return_abs << ',' << secs << '\n';
            std::cerr << "epoch " << L.k << " roots=" << done << " updates=" << t.updates << " entropy=" << t.last_entropy
                      << " kl=" << t.last_kl << " value_loss=" << t.last_value_loss << " (" << secs << "s)\n";
            ++L.k;
            while (L.next_epoch_roots <= done) { L.epoch_len *= 2; L.next_epoch_roots += L.epoch_len; }
            saved = true;
        }
        if (done >= L.next_snapshot && done < T) {
            const std::string file = "snapshot_r" + std::to_string(done) + ".net";
            save_snapshot(out + "/" + file, t.pol);
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

// same LBR deals / seed / particles and gap deals as eval(); a network has no separate average policy
int eval_net(std::map<std::string, std::string>& o) {
    if (o["--policy"] != "current") throw std::runtime_error("network snapshots have only the current (last-iterate) policy: --policy current");
    const Mlp net = load_snapshot(o.at("--snapshot"));
    const uint64_t seed = std::stoull(o["--seed"]);
    NetPolicy target{net, std::mt19937_64(seed ^ 0x5851f42d4c957f2dull)};
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
    std::cout << std::setprecision(9) << "{\"snapshot\":\"" << json_escape(o["--snapshot"]) << "\",\"policy\":\"current\",\"pairs\":"
              << pairs << ",\"particles\":" << o["--particles"] << ",\"seed\":" << seed << ",\"lbr_ante\":" << lm
              << ",\"lbr_se\":" << lse << ",\"gap_roots\":" << gap_roots << ",\"gap_proxy_ante\":" << gm << ",\"gap_proxy_se\":" << gse
              << ",\"policy_queries\":" << st.policy_queries << ",\"policy_misses\":" << st.policy_misses
              << ",\"lbr_seconds\":" << lbr_seconds << ",\"h4\":\"heuristic\",\"h4_buckets\":0,\"rho_roots\":0,\"rho_plain_half\":-1,"
              << "\"rho_cross_half\":-1,\"parameters\":" << net.parameters() << "}\n";
    return 0;
}

}  // namespace sota
