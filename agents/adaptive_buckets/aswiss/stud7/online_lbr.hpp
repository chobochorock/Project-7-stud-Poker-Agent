// Online (test-then-train) Policy-LBR, shared by every trainer of aswiss_stud.cpp (tabular, sota/ networks, sota/ ReBeL).
//
// Every --online-lbr-every roots, before root t is trained on, the policy as it stands is attacked by the engine's
// PolicyLBR (--online-lbr-particles particles) on the deal of root t, once from each seat; the pair mean is one row of
// OUT/online_lbr.csv. A moving average of the rows is the metric (plot_online_lbr.py: window and 90% interval).
// - The deal of root t is the t-th deck of the stream mt19937_64(--seed), one std::shuffle of fresh_deck() per root:
//   exactly the tabular trainer's training deal of root t, and the same deck for every arm with the same seed, so
//   arms evaluated at the same roots are paired by deal. Network / ReBeL trainers replay that stream (they do not
//   train on it).
// - Seeds of the LBR particles and of the target's own action sampling depend only on (seed, t, policy): rows do not
//   depend on resumption or on any other setting.
// - The training does not see any of it (separate RNGs, const policy queries): a run with online LBR trains exactly
//   like one without. Its time is kept out of the training seconds (curve.csv, snapshots.csv, log.csv) and reported
//   separately (column online_seconds = cumulative time spent here).
// - Few particles make a weaker attacker (1 particle cannot weigh the target's actions at all), so values are lower
//   than the offline evaluation's 64-particle LBR; compare runs at the same particle count only.
// Included inside the anonymous namespace of aswiss_stud.cpp after the engine (which brings <fstream>, <sstream>,
// <random>) and <filesystem>, and after kAnte.
#pragma once

struct OnlineLbr {
    long long every = 0;
    int particles = 16;
    uint64_t seed = 0;
    std::vector<std::string> policies;
    std::mt19937_64 deals;
    long long dealt = 0;  // decks drawn from `deals`
    std::vector<Card> deck;
    long long rows = 0;
    double seconds = 0;   // cumulative time spent in evaluations
    std::string path;
    std::ofstream out;

    bool on() const { return every > 0; }
    bool due(long long t) const { return on() && t > 0 && t % every == 0; }

    // options: --online-lbr-every N (0 = off), --online-lbr-particles P, --online-lbr-policy average|current|both
    void setup(std::map<std::string, std::string>& o, uint64_t seed_, bool has_average, bool has_current) {
        every = std::stoll(o["--online-lbr-every"]);
        if (every < 0) throw std::runtime_error("--online-lbr-every must be >= 0");
        if (!on()) return;
        particles = std::stoi(o["--online-lbr-particles"]);
        if (particles < 1) throw std::runtime_error("--online-lbr-particles must be >= 1");
        std::string p = o["--online-lbr-policy"];
        if (p.empty()) p = has_average ? "average" : "current";
        if (p == "both") policies = {"average", "current"};
        else if (p == "average" || p == "current") policies = {p};
        else throw std::runtime_error("--online-lbr-policy must be average, current or both");
        for (const auto& q : policies)
            if ((q == "average" && !has_average) || (q == "current" && !has_current))
                throw std::runtime_error("--online-lbr-policy " + q + " is not available for this trainer");
        seed = seed_;
        deals.seed(seed_);
    }
    std::string config() const {
        if (!on()) return "";
        std::ostringstream s;
        s << ";online_lbr=" << every << "/" << particles;
        for (const auto& p : policies) s << "/" << p;
        return s.str();
    }
    // new run: header; resumed: cut rows written after the checkpoint (they are written again)
    void open(const std::string& dir, bool resumed, uint64_t bytes) {
        if (!on()) return;
        path = dir + "/online_lbr.csv";
        if (resumed) std::filesystem::resize_file(path, bytes);
        else std::ofstream(path) << "roots,seconds,policy,lbr_seat0,lbr_seat1,lbr_pair,particles,online_seconds\n";
        out.open(path, std::ios::app);
    }
    // the deck of root t (t increasing) from the replayed deal stream
    const std::vector<Card>& deck_at(long long t) {
        if (t < dealt) throw std::runtime_error("online LBR: deal stream is past root " + std::to_string(t));
        while (dealt < t) { deck = fresh_deck(); std::shuffle(deck.begin(), deck.end(), deals); ++dealt; }
        return deck;
    }
    template <class Target>
    void evaluate(long long t, const std::vector<Card>& d, Target& target, const std::string& policy, double train_seconds) {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t salt = policy == "current" ? 0x632be59bd9b4e019ull : 0;
        const uint64_t k = 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(t);
        PolicyLBR<Target> lbr(target, particles, seed ^ 0x94d049bb133111ebull ^ k ^ salt);
        const double v0 = play_hand_policy_lbr(d, 0, target, lbr, kAnte, kEffectiveStackAnte);
        const double v1 = play_hand_policy_lbr(d, 1, target, lbr, kAnte, kEffectiveStackAnte);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++rows;
        out << t << ',' << train_seconds << ',' << policy << ',' << v0 << ',' << v1 << ',' << (v0 + v1) / 2 << ',' << particles << ','
            << seconds << '\n';
    }
    // seed of the target's own action sampling in the evaluation of root t
    uint64_t target_seed(long long t, const std::string& policy) const {
        return seed ^ 0x5851f42d4c957f2dull ^ (0xbf58476d1ce4e5b9ull * static_cast<uint64_t>(t)) ^ (policy == "current" ? 1ull : 0ull);
    }
    uint64_t bytes() { out.flush(); return on() ? static_cast<uint64_t>(std::filesystem::file_size(path)) : 0; }
    void save(std::ofstream& f) {
        std::ostringstream r; r << deals;
        const std::string text = r.str();
        const uint64_t n = text.size(), b = bytes();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n)); f.write(text.data(), static_cast<std::streamsize>(n));
        for (long long v : {dealt, rows}) f.write(reinterpret_cast<const char*>(&v), sizeof(v));
        f.write(reinterpret_cast<const char*>(&seconds), sizeof(seconds));
        f.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }
    // returns the online_lbr.csv length at the checkpoint
    uint64_t load(std::ifstream& f) {
        uint64_t n = 0, b = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        std::string text(n, '\0');
        f.read(text.data(), static_cast<std::streamsize>(n));
        std::istringstream r(text); r >> deals;
        for (long long* v : {&dealt, &rows}) f.read(reinterpret_cast<char*>(v), sizeof(*v));
        f.read(reinterpret_cast<char*>(&seconds), sizeof(seconds));
        f.read(reinterpret_cast<char*>(&b), sizeof(b));
        return b;  // later evaluations are at roots past `dealt`, so the last deck itself is not needed
    }
};

// the three options every trainer accepts
inline void add_online_lbr_options(std::map<std::string, std::string>& defaults) {
    defaults["--online-lbr-every"] = "0";
    defaults["--online-lbr-particles"] = "16";
    defaults["--online-lbr-policy"] = "";
}
