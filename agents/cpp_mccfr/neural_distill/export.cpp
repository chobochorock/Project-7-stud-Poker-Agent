// Teacher-labelled decision states for neural_distill/train.py. Build/run through run.py; see README.md.
// Usage: export.exe TEACHER_SPEC NEW_DATASET HANDS SEED   (TEACHER_SPEC: dt=RUN_DIR | mccfr=ATLAS,MODEL,BUCKET,HANDS)
// Each hand: fresh fifth-street root (epoch7::root_for, heuristic H4), both seats act by the teacher's
// current or average policy (chosen per seat and hand) mixed with epsilon-uniform exploration,
// epsilon in {0, 0.1, 0.3}. Every decision is recorded from the actor's view with teacher labels.
#include "tree_engine.inc"
#include "tokens.hpp"
#include "teacher.hpp"

namespace {

int run(int argc, char** argv) {
    hand_recall::require(argc == 5 || (argc == 6 && std::string(argv[5]) == "--keys-only"),
                         "usage: export.exe TEACHER_SPEC NEW_DATASET HANDS SEED [--keys-only]");
    const bool keys_only = argc == 6;
    const std::filesystem::path output(argv[2]);
    const uint64_t hands = std::stoull(argv[3]), seed = std::stoull(argv[4]);
    hand_recall::require(!std::filesystem::exists(output), "dataset exists; use a new path");
    const auto teacher = neural_distill::make_teacher(argv[1]);
    std::cout << "TEACHER identity=" << teacher->identity << " hands=" << teacher->hands << std::endl;
    std::ofstream out(output, std::ios::binary);
    out.write(keys_only ? "NDKEYS1\0" : "NDDATA1\0", 8);
    uint64_t count = 0;
    out.write(reinterpret_cast<const char*>(&count), 8);
    static const std::array<double, 3> epsilons = {0.0, 0.1, 0.3};
    uint64_t found = 0;
    for (uint64_t hand = 0; hand < hands; ++hand) {
        std::mt19937_64 rng(seed ^ ((hand + 1) * 0x9e3779b97f4a7c15ull));
        State state = epoch7::root_for(rng());
        std::array<int, 2> behavior{}, epsilon{};
        for (int seat = 0; seat < 2; ++seat) {
            behavior[seat] = static_cast<int>(rng() % 2);
            epsilon[seat] = static_cast<int>(rng() % 3);
        }
        while (!state.terminal) {
            const int viewer = state.actor;
            const auto encoded = neural_distill::encode(state, viewer);
            bool is_found = false;
            const RegretNode& node = teacher->node(state, &is_found);
            neural_distill::Record r;
            r.hand = static_cast<uint32_t>(hand);
            r.viewer = static_cast<uint8_t>(viewer);
            r.street = static_cast<uint8_t>(state.street);
            r.found = is_found;
            r.mask = encoded.mask;
            r.ntok = static_cast<uint8_t>(encoded.tokens.size());
            r.behavior = static_cast<uint8_t>(behavior[viewer]);
            r.epsilon = static_cast<uint8_t>(epsilon[viewer]);
            std::copy(encoded.tokens.begin(), encoded.tokens.end(), r.tokens.begin());
            r.numerics = encoded.numerics;
            double scale = 0;
            for (int a = 0; a < kActionCount; ++a) {
                if (encoded.mask & (1u << a)) {
                    scale = std::max(scale, std::abs(node.regrets[a]));
                }
            }
            r.scale = static_cast<float>(scale);
            const auto current = current_strategy(node, encoded.mask);
            const auto average = average_strategy(node, encoded.mask);
            hand_recall::require(current == teacher->policy(state, viewer, false, nullptr) &&
                                 average == teacher->policy(state, viewer, true, nullptr),
                                 "teacher table and teacher policy disagree");
            for (int a = 0; a < kActionCount; ++a) {
                const bool legal = encoded.mask & (1u << a);
                r.regret[a] = legal && scale > 0 ? static_cast<float>(node.regrets[a] / scale) : 0.0f;
                r.current[a] = static_cast<float>(current[a]);
                r.average[a] = static_cast<float>(average[a]);
            }
            if (keys_only) {
                neural_distill::KeyFeatures key{};
                hand_recall::require(teacher->key_features(state, key), "teacher has no key features");
                out.write(reinterpret_cast<const char*>(key.data()), static_cast<std::streamsize>(key.size()));
            } else {
                neural_distill::write_record(out, r);
            }
            ++count;
            found += is_found;
            const double explore = epsilons[epsilon[viewer]];
            Action action;
            if (std::uniform_real_distribution<double>(0, 1)(rng) < explore) {
                action = epoch7::sample(uniform_strategy(encoded.mask), encoded.mask, rng);
            } else {
                action = epoch7::sample(behavior[viewer] ? average : current, encoded.mask, rng);
            }
            state = epoch7::child(state, action);
        }
        if ((hand + 1) % 50000 == 0 || hand + 1 == hands) {
            std::cout << "PROGRESS hands=" << hand + 1 << " records=" << count
                      << " found_share=" << double(found) / std::max<uint64_t>(1, count) << std::endl;
        }
    }
    out.seekp(8);
    out.write(reinterpret_cast<const char*>(&count), 8);
    out.close();
    hand_recall::require(bool(out), "dataset close failed");
    std::cout << "DONE records=" << count << " found=" << found << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
