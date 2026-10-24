// Policy-LBR evaluation of a distilled student next to its teacher; build/run through run.py.
// Usage:
//   evaluate.exe TEACHER_SPEC STUDENT NEW_OUTPUT PAIRS PARTICLES SEED  paired LBR matches
//   evaluate.exe --self-test TEACHER_SPEC STUDENT                      privacy/legality/reproducibility
//   evaluate.exe --parity WEIGHTS DATASET REFERENCE [KEYS]             C++ vs Torch forward on stored states
// TEACHER_SPEC: dt=RUN_DIR | mccfr=ATLAS,MODEL,BUCKET,HANDS (see teacher.hpp).
// STUDENT: an NDNET1 weights file, or "smooth:LAMBDA" = control policy (1-LAMBDA)*teacher + LAMBDA*uniform.
// Policy kinds reuse abstraction_lbr::evaluate: model_current/model_average = student (regret matching on
// the regret head / legal softmax of the average head), baseline_current/baseline_average = teacher,
// uniform. LBR unit: attacker ante/hand (higher = worse for the evaluated policy).
#include "tree_engine.inc"
#include "../abstraction_lbr/evaluate.hpp"
#include "tokens.hpp"
#include "net.hpp"
#include "teacher.hpp"
#include <unordered_map>

namespace {

// Agreement statistics between student and teacher at the states the matches actually query.
// "actual" = the real target decision (the opponent's hidden cards are present in the state);
// "hypothetical" = PolicyLBR particle states (likelihood and fold-response queries) where they are cleared.
struct Agreement {
    double tv = 0;
    uint64_t count = 0, agree = 0;
    void add(double value, bool same) {
        tv += value;
        ++count;
        agree += same;
    }
};

struct StudentModel {
    std::unique_ptr<neural_distill::Teacher> teacher;
    std::unique_ptr<neural_distill::Net> net;
    double smoothing = -1;
    std::string label;
    uint64_t hands = 0;
    // [policy 0=current,1=average][0=actual,1=hypothetical][street 5..7 -> 0..2]
    std::array<std::array<std::array<Agreement, 3>, 2>, 2> agreement{};
    uint64_t cache_hits = 0, cache_misses = 0;
    std::unordered_map<std::string, neural_distill::Outputs> cache;

    StudentModel(const std::string& spec, const std::string& student)
        : teacher(neural_distill::make_teacher(spec)), hands(teacher->hands) {
        if (student.rfind("smooth:", 0) == 0) {
            smoothing = std::stod(student.substr(7));
            hand_recall::require(smoothing >= 0 && smoothing <= 1, "smoothing must be in [0,1]");
            label = "smooth:" + student.substr(7);
        } else {
            net = std::make_unique<neural_distill::Net>(student);
            label = net->architecture;
        }
    }

    neural_distill::Outputs outputs(const neural_distill::Encoded& e) {
        std::string key(reinterpret_cast<const char*>(e.tokens.data()), e.tokens.size() * neural_distill::kTokenFields);
        key.append(reinterpret_cast<const char*>(e.numerics.data()), e.numerics.size() * sizeof(float));
        if (current_key_ok) {
            key.append(reinterpret_cast<const char*>(current_key.data()), current_key.size());
        }
        const auto it = cache.find(key);
        if (it != cache.end()) {
            ++cache_hits;
            return it->second;
        }
        ++cache_misses;
        if (cache.size() > 2000000) {
            cache.clear();
        }
        const auto out = net->forward(e.tokens, e.numerics, current_key_ok ? &current_key : nullptr);
        cache.emplace(std::move(key), out);
        return out;
    }
    neural_distill::KeyFeatures current_key{};
    bool current_key_ok = false;

    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        if (kind == 1 || kind == 3) {
            return teacher->policy(state, viewer, kind == 3, found);
        }
        hand_recall::require(viewer == state.actor, "student policy must be queried at the acting information set");
        const bool average = kind == 2;
        const auto reference = teacher->policy(state, viewer, average, nullptr);
        const uint8_t mask = valid_mask(state, viewer);
        std::array<double, 8> p{};
        if (net) {
            const auto encoded = neural_distill::encode(state, viewer);
            current_key_ok = net->needs_key() && teacher->key_features(state, current_key);
            hand_recall::require(!net->needs_key() || current_key_ok, "student needs teacher key features");
            const auto out = outputs(encoded);
            p = average ? neural_distill::softmax_policy(out, encoded.mask) : neural_distill::regret_matching(out, encoded.mask);
        } else {
            const auto uniform = uniform_strategy(mask);
            for (int a = 0; a < kActionCount; ++a) {
                p[a] = (1 - smoothing) * reference[a] + smoothing * uniform[a];
            }
        }
        if (found) {
            *found = true;
        }
        double tv = 0;
        int best = -1, best_reference = -1;
        for (int a = 0; a < kActionCount; ++a) {
            tv += std::abs(p[a] - reference[a]);
            if (mask & (1u << a)) {
                if (best < 0 || p[a] > p[best]) {
                    best = a;
                }
                if (best_reference < 0 || reference[a] > reference[best_reference]) {
                    best_reference = a;
                }
            }
        }
        const bool actual = !state.players[1 - viewer].hidden.empty();
        agreement[average][actual ? 0 : 1][state.street - 5].add(tv / 2, best == best_reference);
        epoch7::Values result{};
        for (int a = 0; a < kActionCount; ++a) {
            result[a] = p[a];
        }
        return result;
    }
};

void write_diagnostics(const StudentModel& model, const std::filesystem::path& path) {
    std::ofstream out(path);
    out << std::setprecision(12) << "{\"student\":\"" << model.label << "\""
        << ",\"teacher_identity\":\"" << model.teacher->identity << "\""
        << ",\"teacher_training_hands\":" << model.hands
        << ",\"forward_cache_hits\":" << model.cache_hits << ",\"forward_cache_misses\":" << model.cache_misses;
    const char* policies[2] = {"current", "average"};
    const char* types[2] = {"actual", "hypothetical"};
    for (int p = 0; p < 2; ++p) {
        for (int t = 0; t < 2; ++t) {
            Agreement total;
            out << ",\"" << policies[p] << '_' << types[t] << "_by_street\":{";
            for (int s = 0; s < 3; ++s) {
                const auto& a = model.agreement[p][t][s];
                total.tv += a.tv;
                total.count += a.count;
                total.agree += a.agree;
                out << (s ? "," : "") << '"' << s + 5 << "\":{\"queries\":" << a.count
                    << ",\"mean_tv\":" << (a.count ? a.tv / a.count : 0)
                    << ",\"best_action_agreement\":" << (a.count ? double(a.agree) / a.count : 0) << '}';
            }
            out << "},\"" << policies[p] << '_' << types[t] << "\":{\"queries\":" << total.count
                << ",\"mean_tv\":" << (total.count ? total.tv / total.count : 0)
                << ",\"best_action_agreement\":" << (total.count ? double(total.agree) / total.count : 0) << '}';
        }
    }
    out << ",\"note\":\"TV and best-action agreement to the teacher, unweighted over every student query; actual = real target decisions, hypothetical = PolicyLBR particle states\"}\n";
    hand_recall::require(bool(out), "diagnostics write failed");
}

int parity(const std::string& weights, const std::string& dataset, const std::string& reference,
           const std::string& keys_path = "") {
    neural_distill::Net net(weights);
    std::ifstream data(dataset, std::ios::binary), ref(reference, std::ios::binary);
    std::ifstream keys;
    if (!keys_path.empty()) {
        keys.open(keys_path, std::ios::binary);
        hand_recall::require(bool(keys), "cannot open keys sidecar");
    }
    hand_recall::require(bool(data) && bool(ref), "cannot open parity inputs");
    uint32_t count = 0;
    ref.read(reinterpret_cast<char*>(&count), 4);
    double worst = 0, worst_policy = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t index = 0;
        std::array<float, 8> regret{}, logits{};
        ref.read(reinterpret_cast<char*>(&index), 8);
        ref.read(reinterpret_cast<char*>(regret.data()), 32);
        ref.read(reinterpret_cast<char*>(logits.data()), 32);
        hand_recall::require(bool(ref), "truncated parity reference");
        const auto r = neural_distill::read_record(data, index);
        std::vector<neural_distill::Token> tokens(r.tokens.begin(), r.tokens.begin() + r.ntok);
        neural_distill::KeyFeatures key{};
        if (keys.is_open()) {
            keys.seekg(static_cast<std::streamoff>(16 + index * key.size()));
            keys.read(reinterpret_cast<char*>(key.data()), static_cast<std::streamsize>(key.size()));
            hand_recall::require(bool(keys), "truncated keys sidecar");
        }
        const auto out = net.forward(tokens, r.numerics, keys.is_open() ? &key : nullptr);
        neural_distill::Outputs torch_out;
        torch_out.regret = regret;
        torch_out.average_logits = logits;
        for (int a = 0; a < 8; ++a) {
            worst = std::max(worst, double(std::abs(out.regret[a] - regret[a])));
            worst = std::max(worst, double(std::abs(out.average_logits[a] - logits[a])));
        }
        const auto p1 = neural_distill::regret_matching(out, r.mask), q1 = neural_distill::regret_matching(torch_out, r.mask);
        const auto p2 = neural_distill::softmax_policy(out, r.mask), q2 = neural_distill::softmax_policy(torch_out, r.mask);
        for (int a = 0; a < 8; ++a) {
            worst_policy = std::max({worst_policy, std::abs(p1[a] - q1[a]), std::abs(p2[a] - q2[a])});
        }
    }
    std::cout << std::setprecision(9) << "PARITY states=" << count << " max_abs_output_diff=" << worst
              << " max_abs_policy_diff=" << worst_policy << std::endl;
    return worst < 1e-3 && worst_policy < 1e-3 ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if ((argc == 5 || argc == 6) && std::string(argv[1]) == "--parity") {
            return parity(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : "");
        }
        if (argc == 4 && std::string(argv[1]) == "--self-test") {
            StudentModel model(argv[2], argv[3]);
            abstraction_lbr::self_test(model);
            return 0;
        }
        hand_recall::require(argc == 7,
            "usage: evaluate.exe TEACHER_SPEC STUDENT NEW_OUTPUT PAIRS PARTICLES SEED | --self-test SPEC STUDENT | "
            "--parity WEIGHTS DATASET REFERENCE");
        StudentModel model(argv[1], argv[2]);
        std::cout << "LOADED teacher=" << model.teacher->identity << " student=" << model.label << std::endl;
        char* forwarded[] = {argv[0], argv[1], argv[3], argv[4], argv[5], argv[6]};
        const int status = abstraction_lbr::evaluate(model, 6, forwarded);
        write_diagnostics(model, std::filesystem::path(argv[3]) / "student_diagnostics.json");
        return status;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
