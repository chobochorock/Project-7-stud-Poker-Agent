// Read-only final held-out evaluation of saved 7-stud epoch models; see README.md.
// Build: g++ -O3 -std=c++17 -DNOMINMAX -static-libstdc++ -static-libgcc
//   evaluate_epoch_ensemble.cpp -o bin/evaluate_epoch_ensemble.exe -lpsapi
// Usage: exe RUN_DIR ATLAS SEED ROOTS PARTICLES HANDS EPOCH_HANDS
// Writes independent_audit.csv: root-only local gaps, not exploitability.
#define main epoch_training_entrypoint
#include "stud_epoch_ensemble.cpp"
#undef main

struct UniformEpochPolicy {
    epoch7::Values policy(const State& state, int viewer, bool* found = nullptr) {
        if (found) *found = true;
        return uniform_strategy(valid_mask(state, viewer));
    }
};

int main(int argc, char** argv) {
    try {
        using namespace epoch7;
        if (argc != 8) throw std::runtime_error("usage: exe RUN_DIR ATLAS SEED ROOTS PARTICLES HANDS EPOCH_HANDS");
        const std::filesystem::path dir(argv[1]);
        const uint64_t seed = std::stoull(argv[3]);
        const int roots = std::stoi(argv[4]), samples = std::stoi(argv[5]);
        const uint64_t hands = std::stoull(argv[6]), epoch_hands = std::stoull(argv[7]);
        if (roots < 1 || samples < 1) throw std::runtime_error("positive evaluation counts required");
        PowerAtlas atlas; atlas.load(argv[2]); atlas.set_assignment_cache_limit(100000);
        FeatureCache features{atlas}; Ensemble ensemble(features, seed);
        MCCFR baseline(false, seed, 5, &atlas);
        baseline.use_cumulative_street_summary(true);
        load_frozen_checkpoint(dir, ensemble, baseline, hands, epoch_hands);
        CurrentBaseline current{baseline}; UniformEpochPolicy uniform;
        AverageEnsemble average{ensemble};
        const bool has_average = std::all_of(ensemble.models.begin(), ensemble.models.end(),
            [](const EpochModel& model) { return model.leafwise; });
        const auto output = dir / "independent_audit.csv";
        if (std::filesystem::exists(output)) throw std::runtime_error("independent audit already exists");
        std::ofstream csv(output);
        csv << "root,seed,particles,ensemble_current,hard256_current,hard256_average,uniform,matched_epoch_models";
        if (has_average) csv << ",ensemble_average";
        csv << '\n';
        csv << std::setprecision(12);
        std::array<double, 4> totals{};
        double average_total = 0;
        for (int index = 0; index < roots; ++index) {
            const uint64_t eval_seed = seed ^ (uint64_t(index + 1) * 0xd1b54a32d192ed03ull);
            const auto root = root_for(eval_seed);
            std::mt19937_64 rng(eval_seed ^ 0xabcdef012345ull);
            const auto particles = root_particles(root, samples, rng);
            const std::array<double, 4> gaps{
                local_gap(root, particles, ensemble, eval_seed),
                local_gap(root, particles, current, eval_seed),
                local_gap(root, particles, baseline, eval_seed),
                local_gap(root, particles, uniform, eval_seed)};
            const auto f = features(root, root.actor);
            const auto group = group_id(root, root.actor);
            int matched = 0;
            for (const auto& model : ensemble.models) {
                bool found = false;
                model.query(group, f, &found);
                if (found) ++matched;
            }
            csv << index << ',' << seed << ',' << samples;
            for (size_t i = 0; i < gaps.size(); ++i) {
                if (!std::isfinite(gaps[i]) || gaps[i] < 0) throw std::runtime_error("bad audit value");
                totals[i] += gaps[i]; csv << ',' << gaps[i];
            }
            csv << ',' << matched;
            if (has_average) {
                const double gap = local_gap(root, particles, average, eval_seed);
                if (!std::isfinite(gap) || gap < 0) throw std::runtime_error("bad average audit value");
                average_total += gap; csv << ',' << gap;
            }
            csv << '\n';
            if ((index + 1) % 32 == 0) {
                csv.flush(); std::cout << "AUDIT_ROOTS completed=" << index + 1 << '\n' << std::flush;
            }
        }
        std::cout << std::setprecision(10) << "{\"roots\":" << roots << ",\"particles\":" << samples
            << ",\"ensemble_current\":" << totals[0] / roots
            << ",\"hard256_current\":" << totals[1] / roots
            << ",\"hard256_average\":" << totals[2] / roots
            << ",\"uniform\":" << totals[3] / roots;
        if (has_average) std::cout << ",\"ensemble_average\":" << average_total / roots;
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'; return 1;
    }
}
