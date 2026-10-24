// Frozen, read-only teachers (see make_teacher for the spec strings).
// Also defines the 576-byte training record shared with train.py::RECORD.
#pragma once
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace neural_distill {

// Teacher interface. node(): the stored table entry (cumulative signed regrets, average sums) for the
// actor's information key, or an empty node (uniform policy) if unseen. policy(): the teacher's own
// current (regret matching) or average strategy for that state.
class Teacher {
public:
    uint64_t hands = 0;
    std::string identity;
    virtual ~Teacher() = default;
    virtual const RegretNode& node(const State& state, bool* found) const = 0;
    virtual epoch7::Values policy(const State& state, int viewer, bool average, bool* found) const = 0;
    // Teacher's own abstraction key for the acting player (attribution control); false if unsupported.
    virtual bool key_features(const State&, KeyFeatures&) const { return false; }
};

// "dt=RUN": matched hard256 betting-summary + seat k-means stored as the baseline of a dynamic-tree
// checkpoint (same loader as abstraction_lbr/tree.cpp).
class DtTeacher : public Teacher {
public:
    PowerAtlas atlas;
    std::unique_ptr<dynamic_tree::Solver> tree, baseline;

    explicit DtTeacher(const std::filesystem::path& run) {
        atlas.load((run / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        tree = std::make_unique<dynamic_tree::Solver>(atlas, true, 0, dynamic_tree::Config{});
        baseline = std::make_unique<dynamic_tree::Solver>(atlas, false, 0, dynamic_tree::Config{});
        std::ifstream in(run / "checkpoint.bin", std::ios::binary);
        hand_recall::require(hand_recall::read_text(in) == "DYNAMIC_TREE1", "invalid tree checkpoint");
        identity = "dt-baseline:" + hand_recall::read_text(in);
        hands = epoch7::read<uint64_t>(in);
        for (int i = 0; i < 3; ++i) {
            epoch7::read<uint64_t>(in);
        }
        tree->load(in);
        baseline->load(in);
        hand_recall::require(in.peek() == std::char_traits<char>::eof(), "trailing checkpoint bytes");
        for (auto& node : tree->tree) {
            std::vector<dynamic_tree::Sample>().swap(node.samples);
        }
    }

    const RegretNode& node(const State& state, bool* found) const override {
        static const RegretNode empty{};
        epoch7::Features x{};
        int id = -1;
        const auto key = baseline->key(state, x, id, false);
        const auto it = baseline->table.find(key);
        if (found) {
            *found = it != baseline->table.end();
        }
        return it == baseline->table.end() ? empty : it->second;
    }

    epoch7::Values policy(const State& state, int viewer, bool average, bool* found) const override {
        hand_recall::require(viewer == state.actor, "teacher policy must be queried at the acting information set");
        const auto& n = node(state, found);
        const auto mask = valid_mask(state, viewer);
        return average ? average_strategy(n, mask) : current_strategy(n, mask);
    }
};

// "mccfr=ATLAS,MODEL,BUCKET,HANDS": a stud_mccfr MCCFRV4/5 table (e.g. the K512 power-memory16 10M run),
// loaded exactly as autoencoder_abstraction/neural_mccfr.cpp evaluate_checkpoint does. MCCFR keeps its
// node table private, so the same file is parsed a second time (same record layout as MCCFR::load) to
// read the regrets behind instantaneous_policy(); export.cpp checks that both views agree.
class MccfrTeacher : public Teacher {
public:
    PowerAtlas atlas;
    std::unique_ptr<MCCFR> solver;
    std::unordered_map<InfoKey, RegretNode, InfoKeyHash> nodes;

    MccfrTeacher(const std::string& atlas_path, const std::string& model_path, const std::string& bucket, uint64_t trained) {
        hand_recall::require(bucket == "power" || bucket == "power-memory16", "unsupported mccfr bucket");
        atlas.load(atlas_path);
        atlas.set_assignment_cache_limit(100000);
        solver = std::make_unique<MCCFR>(false, 7, 5, &atlas);
        solver->use_cumulative_street_summary(bucket != "power");
        solver->load(model_path);
        std::ifstream in(model_path, std::ios::binary);
        char magic[8]{};
        uint8_t plus = 0, abstraction = 0, start_street = 0;
        uint64_t count = 0;
        in.read(magic, 8);
        in.read(reinterpret_cast<char*>(&plus), 1);
        in.read(reinterpret_cast<char*>(&abstraction), 1);
        in.read(reinterpret_cast<char*>(&start_street), 1);
        in.read(reinterpret_cast<char*>(&count), 8);
        hand_recall::require(std::memcmp(magic, "MCCFRV4", 7) == 0 || std::memcmp(magic, "MCCFRV5", 7) == 0,
                             "only MCCFRV4/5 tables are supported");
        nodes.reserve(static_cast<size_t>(count * 1.3) + 1);
        for (uint64_t i = 0; i < count; ++i) {
            InfoKey key{};
            RegretNode node{};
            in.read(reinterpret_cast<char*>(&key), sizeof(key));
            in.read(reinterpret_cast<char*>(&node), sizeof(node));
            nodes.emplace(key, node);
        }
        hand_recall::require(bool(in) && nodes.size() == solver->bucket_count(), "mccfr table parse mismatch");
        hands = trained;
        identity = "mccfr:" + std::filesystem::path(model_path).filename().string() + ":" + std::to_string(count);
    }

    const RegretNode& node(const State& state, bool* found) const override {
        static const RegretNode empty{};
        const auto it = nodes.find(solver->information_key(state, state.actor));
        if (found) {
            *found = it != nodes.end();
        }
        return it == nodes.end() ? empty : it->second;
    }

    epoch7::Values policy(const State& state, int viewer, bool average, bool* found) const override {
        hand_recall::require(viewer == state.actor, "teacher policy must be queried at the acting information set");
        return average ? solver->policy(state, viewer, found) : solver->instantaneous_policy(state, viewer, found);
    }

    bool key_features(const State& state, KeyFeatures& out) const override {
        const InfoKey k = solver->information_key(state, state.actor);
        out = KeyFeatures{};
        out[0] = static_cast<uint8_t>(k.street - 5);
        out[1] = static_cast<uint8_t>(k.power_cluster & 255);
        out[2] = static_cast<uint8_t>(k.power_cluster >> 8);
        out[3] = k.rank_bucket;
        out[4] = k.pot_odds_bucket;
        out[5] = k.stack_pot_bucket;
        out[6] = k.own_bet_count;
        out[7] = k.opponent_bet_count;
        out[8] = k.checked;
        out[9] = k.last_action_class;
        out[10] = k.betting_goal;
        out[11] = k.legal_mask;
        return true;
    }
};

// Spec: "dt=RUN_DIR" or "mccfr=ATLAS,MODEL,BUCKET,HANDS".
inline std::unique_ptr<Teacher> make_teacher(const std::string& spec) {
    if (spec.rfind("dt=", 0) == 0) {
        return std::make_unique<DtTeacher>(spec.substr(3));
    }
    if (spec.rfind("mccfr=", 0) == 0) {
        std::vector<std::string> parts;
        std::stringstream stream(spec.substr(6));
        std::string part;
        while (std::getline(stream, part, ',')) {
            parts.push_back(part);
        }
        hand_recall::require(parts.size() == 4, "mccfr teacher spec: ATLAS,MODEL,BUCKET,HANDS");
        return std::make_unique<MccfrTeacher>(parts[0], parts[1], parts[2], std::stoull(parts[3]));
    }
    throw std::runtime_error("unknown teacher spec " + spec);
}

struct Record {
    uint32_t hand = 0;
    uint8_t viewer = 0, street = 0, found = 0, mask = 0, ntok = 0, behavior = 0, epsilon = 0, reserved = 0;
    std::array<Token, kMaxTokens> tokens{};
    std::array<float, kNumerics> numerics{};
    std::array<float, 8> regret{}, current{}, average{};
    float scale = 0;
};

inline void write_record(std::ostream& out, const Record& r) {
    const auto put = [&](const void* data, size_t size) {
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    };
    put(&r.hand, 4);
    put(&r.viewer, 1);
    put(&r.street, 1);
    put(&r.found, 1);
    put(&r.mask, 1);
    put(&r.ntok, 1);
    put(&r.behavior, 1);
    put(&r.epsilon, 1);
    put(&r.reserved, 1);
    put(r.tokens.data(), kMaxTokens * kTokenFields);
    put(r.numerics.data(), kNumerics * 4);
    put(r.regret.data(), 32);
    put(r.current.data(), 32);
    put(r.average.data(), 32);
    put(&r.scale, 4);
    if (!out) {
        throw std::runtime_error("record write failed");
    }
}

constexpr size_t kRecordBytes = 576;
constexpr size_t kHeaderBytes = 16;

// Reads tokens/numerics/mask of record `index` from a dataset file (used by the parity check).
inline Record read_record(std::istream& in, uint64_t index) {
    Record r;
    in.seekg(static_cast<std::streamoff>(kHeaderBytes + index * kRecordBytes));
    const auto get = [&](void* data, size_t size) {
        in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
        if (!in) {
            throw std::runtime_error("record read failed");
        }
    };
    get(&r.hand, 4);
    get(&r.viewer, 1);
    get(&r.street, 1);
    get(&r.found, 1);
    get(&r.mask, 1);
    get(&r.ntok, 1);
    get(&r.behavior, 1);
    get(&r.epsilon, 1);
    get(&r.reserved, 1);
    get(r.tokens.data(), kMaxTokens * kTokenFields);
    get(r.numerics.data(), kNumerics * 4);
    get(r.regret.data(), 32);
    get(r.current.data(), 32);
    get(r.average.data(), 32);
    get(&r.scale, 4);
    return r;
}

} // namespace neural_distill
