#define main stud_mccfr_embedded_main
#include "stud_mccfr.cpp"
#undef main

#include <winsock2.h>
#include <ws2tcpip.h>

namespace {

constexpr uint32_t kIpcMagic = 0x52464344;
constexpr uint32_t kIpcVersion = 1;

struct IpcHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t columns;
};

void send_all(SOCKET socket, const void* data, size_t size) {
    const char* cursor = static_cast<const char*>(data);
    while (size) {
        const int sent = send(socket, cursor, static_cast<int>(size), 0);
        if (sent <= 0) throw std::runtime_error("Deep CFR IPC send failed");
        cursor += sent;
        size -= sent;
    }
}

void receive_all(SOCKET socket, void* data, size_t size) {
    char* cursor = static_cast<char*>(data);
    while (size) {
        const int received = recv(socket, cursor, static_cast<int>(size), 0);
        if (received <= 0) throw std::runtime_error("Deep CFR IPC receive failed");
        cursor += received;
        size -= received;
    }
}

std::string argument(
    int argc, char** argv, const std::string& name, const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

class AdvantageClient {
public:
    explicit AdvantageClient(int port, bool uniform = false) : uniform_(uniform) {
        if (uniform_) return;
        if (WSAStartup(MAKEWORD(2, 2), &winsock_) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
        socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        const BOOL no_delay = TRUE;
        setsockopt(
            socket_, IPPROTO_TCP, TCP_NODELAY,
            reinterpret_cast<const char*>(&no_delay), sizeof(no_delay));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<u_short>(port));
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        if (socket_ == INVALID_SOCKET ||
            connect(socket_, reinterpret_cast<sockaddr*>(&address),
                    sizeof(address)) == SOCKET_ERROR) {
            close();
            throw std::runtime_error("cannot connect to Deep CFR IPC server");
        }
    }

    ~AdvantageClient() { close(); }

    std::array<double, kActionCount> strategy(const State& state, int actor) {
        if (uniform_) {
            ++queries_;
            return uniform_strategy(valid_mask(state, actor));
        }
        const auto tensor = deep_cfr_tensor(state, actor);
        const IpcHeader request{
            kIpcMagic, kIpcVersion, 1,
            static_cast<uint32_t>(tensor.size())};
        send_all(socket_, &request, sizeof(request));
        send_all(socket_, tensor.data(), tensor.size() * sizeof(float));
        IpcHeader response{};
        std::array<float, kActionCount> output{};
        receive_all(socket_, &response, sizeof(response));
        if (response.magic != kIpcMagic || response.version != kIpcVersion ||
            response.rows != 1 || response.columns != kActionCount) {
            throw std::runtime_error("invalid Deep CFR IPC response");
        }
        receive_all(socket_, output.data(), output.size() * sizeof(float));
        ++queries_;

        const uint8_t mask = valid_mask(state, actor);
        std::array<double, kActionCount> result{};
        double total = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            result[action] = std::max(0.0, static_cast<double>(output[action]));
            total += result[action];
        }
        if (total <= 0.0) return uniform_strategy(mask);
        for (double& probability : result) probability /= total;
        return result;
    }

    uint64_t queries() const { return queries_; }

private:
    void close() {
        if (socket_ != INVALID_SOCKET) {
            shutdown(socket_, SD_BOTH);
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }
        WSACleanup();
    }

    WSADATA winsock_{};
    SOCKET socket_ = INVALID_SOCKET;
    uint64_t queries_ = 0;
    bool uniform_ = false;
};

class SampleWriter {
public:
    explicit SampleWriter(const std::string& path, bool compact)
        : output_(path, std::ios::binary), compact_(compact) {
        if (!output_) throw std::runtime_error("cannot write samples: " + path);
        output_.write(compact_ ? "DCFRC1\0" : "DCFRS1\0", 8);
        write_u32(1);
        write_u32(kDeepCfrTensorSize);
        write_u32(kActionCount);
        write_u64(0);
    }

    void add(
        const State& state,
        int actor,
        uint8_t kind,
        float iteration,
        const std::array<double, kActionCount>& target) {
        const auto tensor = deep_cfr_tensor(state, actor);
        const uint8_t mask = valid_mask(state, actor);
        const uint8_t player = static_cast<uint8_t>(actor);
        const uint8_t reserved = 0;
        output_.write(reinterpret_cast<const char*>(&iteration), sizeof(iteration));
        output_.write(reinterpret_cast<const char*>(&player), sizeof(player));
        output_.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
        output_.write(reinterpret_cast<const char*>(&mask), sizeof(mask));
        output_.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
        if (compact_) write_compact_state(state, actor);
        else output_.write(
            reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(float));
        for (double value : target) {
            const float converted = static_cast<float>(value);
            output_.write(reinterpret_cast<const char*>(&converted), sizeof(converted));
        }
        ++count_;
        if (kind == 0) ++advantages_;
        else ++strategies_;
    }

    void finish() {
        output_.seekp(20);
        write_u64(count_);
        output_.close();
    }

    uint64_t count() const { return count_; }
    uint64_t advantages() const { return advantages_; }
    uint64_t strategies() const { return strategies_; }

private:
    void write_u8(uint8_t value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    void write_f32(float value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    static uint8_t card_token(const Card& card) {
        return static_cast<uint8_t>(
            1 + card.suit * 13 + card.rank - 2);
    }

    void write_cards(const std::vector<Card>& cards, int slots) {
        if (static_cast<int>(cards.size()) > slots) {
            throw std::runtime_error("compact Deep CFR card capacity exceeded");
        }
        for (int slot = 0; slot < slots; ++slot) {
            write_u8(slot < static_cast<int>(cards.size())
                ? card_token(cards[slot]) : 0);
        }
    }

    void write_compact_state(const State& state, int viewer) {
        write_u8(static_cast<uint8_t>(state.street - 5));
        write_cards(state.players[viewer].hidden, 3);
        write_cards(state.players[viewer].shown, 4);
        write_cards(state.players[1 - viewer].shown, 4);
        write_u8(state.players[viewer].has_discard
            ? card_token(state.players[viewer].discarded) : 0);

        if (state.history.size() > kDeepCfrHistorySlots) {
            throw std::runtime_error("compact Deep CFR history capacity exceeded");
        }
        for (int slot = 0; slot < kDeepCfrHistorySlots; ++slot) {
            uint8_t token = 0;
            if (slot < static_cast<int>(state.history.size())) {
                const Event& event = state.history[slot];
                const int relative_actor = event.actor == viewer ? 0 : 1;
                token = static_cast<uint8_t>(
                    1 + (event.street - 5) * 16
                    + relative_actor * kActionCount + event.action);
            }
            write_u8(token);
        }

        const double ante = std::max(1, state.ante);
        const auto normalized = [&](double chips) {
            return static_cast<float>(
                std::log1p(std::max(0.0, chips) / ante));
        };
        const int call = std::max(
            0, state.highest_bet - state.players[viewer].round_bet);
        write_f32(normalized(state.pot));
        write_f32(normalized(
            stack_cap(state, viewer) - state.players[viewer].invested));
        write_f32(normalized(
            stack_cap(state, 1 - viewer)
            - state.players[1 - viewer].invested));
        write_f32(normalized(state.players[viewer].invested));
        write_f32(normalized(state.players[1 - viewer].invested));
        write_f32(static_cast<float>(
            call / static_cast<double>(std::max(1, state.pot + call))));
        write_f32(normalized(state.highest_bet));
    }

    void write_u32(uint32_t value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    void write_u64(uint64_t value) {
        output_.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    std::ofstream output_;
    bool compact_ = false;
    uint64_t count_ = 0;
    uint64_t advantages_ = 0;
    uint64_t strategies_ = 0;
};

void advance_street_local(State& state) {
    if (state.street == 7) {
        state.terminal = true;
        return;
    }
    ++state.street;
    const bool public_card = state.street != 7;
    for (int seat = 0; seat < 2; ++seat) {
        const Card card = state.simulation_deck.back();
        state.simulation_deck.pop_back();
        if (public_card) state.players[seat].shown.push_back(card);
        else state.players[seat].hidden.push_back(card);
        state.players[seat].round_bet = 0;
    }
    state.highest_bet = 0;
    state.raise_count = 0;
    state.actor = first_bettor(state);
}

void apply_and_advance(State& state, Action action) {
    const int actor = state.actor;
    const ActionResult result = apply_action(state, actor, action);
    if (result == ActionResult::RoundEnd) advance_street_local(state);
    else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
}

State sample_training_root(
    std::mt19937_64& rng, int ante, int stack_ante, int start_street) {
    for (;;) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        State state = sample_fifth_street_root(deck, ante, stack_ante);
        while (state.street < start_street && !state.terminal) {
            apply_and_advance(state, heuristic_action(state, state.actor));
        }
        if (!state.terminal && state.street == start_street) return state;
    }
}

class Traverser {
public:
    Traverser(
        AdvantageClient& client,
        SampleWriter& writer,
        std::mt19937_64& rng,
        int traverser,
        float iteration)
        : client_(client), writer_(writer), rng_(rng), traverser_(traverser),
          iteration_(iteration) {}

    double run_root(State state) {
        current_gap_ = 0.0;
        current_gap_by_street_.fill(0.0);
        const int ante = std::max(1, state.ante);
        const double value = run(std::move(state));
        const double normalized = current_gap_ / ante;
        ++roots_;
        gap_sum_ += normalized;
        gap_squared_sum_ += normalized * normalized;
        for (int street = 0; street < 3; ++street) {
            gap_by_street_sum_[street] +=
                current_gap_by_street_[street] / ante;
        }
        return value;
    }

    double run(State state) {
        ++nodes_;
        if (state.terminal) return terminal_net_search(state, traverser_);
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const auto strategy = client_.strategy(state, actor);
        if (actor != traverser_) {
            writer_.add(state, actor, 1, iteration_, strategy);
            const Action action = sample(strategy, mask);
            apply_and_advance(state, action);
            return run(std::move(state));
        }

        std::array<double, kActionCount> action_values{};
        double value = 0.0;
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            apply_and_advance(child, action);
            action_values[action] = run(std::move(child));
            value += strategy[action] * action_values[action];
        }
        std::array<double, kActionCount> regrets{};
        double best = -std::numeric_limits<double>::infinity();
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            regrets[action] = action_values[action] - value;
            best = std::max(best, action_values[action]);
        }
        const double gap = std::max(0.0, best - value);
        current_gap_ += gap;
        if (state.street >= 5 && state.street <= 7) {
            current_gap_by_street_[state.street - 5] += gap;
        }
        writer_.add(state, actor, 0, iteration_, regrets);
        return value;
    }

    uint64_t nodes() const { return nodes_; }
    double gap_mean() const { return gap_sum_ / std::max<uint64_t>(1, roots_); }
    double gap_standard_error() const {
        if (roots_ < 2) return 0.0;
        const double variance = std::max(
            0.0,
            (gap_squared_sum_ - gap_sum_ * gap_sum_ / roots_) /
                (roots_ - 1));
        return std::sqrt(variance / roots_);
    }
    std::array<double, 3> gap_by_street_mean() const {
        auto result = gap_by_street_sum_;
        for (double& value : result) {
            value /= std::max<uint64_t>(1, roots_);
        }
        return result;
    }

private:
    Action sample(const std::array<double, kActionCount>& strategy, uint8_t mask) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double threshold = uniform(rng_);
        double cumulative = 0.0;
        Action last = FOLD;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            last = static_cast<Action>(action);
            cumulative += strategy[action];
            if (threshold <= cumulative) return last;
        }
        return last;
    }

    AdvantageClient& client_;
    SampleWriter& writer_;
    std::mt19937_64& rng_;
    int traverser_;
    float iteration_;
    uint64_t nodes_ = 0;
    uint64_t roots_ = 0;
    double current_gap_ = 0.0;
    double gap_sum_ = 0.0;
    double gap_squared_sum_ = 0.0;
    std::array<double, 3> current_gap_by_street_{};
    std::array<double, 3> gap_by_street_sum_{};
};

}  // namespace

#ifndef DEEP_CFR_TRAVERSE_NO_MAIN
int main(int argc, char** argv) {
    try {
        const int port = std::stoi(argument(argc, argv, "--port", "28731"));
        const int traverser = std::stoi(argument(argc, argv, "--traverser", "0"));
        const int traversals = std::stoi(argument(argc, argv, "--traversals", "100"));
        const int iteration = std::stoi(argument(argc, argv, "--iteration", "1"));
        const int ante = std::stoi(argument(argc, argv, "--ante", "1000"));
        const int stack_ante = std::stoi(argument(argc, argv, "--stack-ante", "1000"));
        const int start_street = std::stoi(argument(argc, argv, "--start-street", "5"));
        const int report_every = std::stoi(argument(
            argc, argv, "--report-every", "0"));
        const bool uniform_policy = argument(
            argc, argv, "--uniform-policy", "0") == "1";
        const bool compact_output = argument(
            argc, argv, "--compact-output", "0") == "1";
        const uint64_t seed = std::stoull(argument(argc, argv, "--seed", "7"));
        const std::string output = argument(argc, argv, "--output");
        if (output.empty() || traverser < 0 || traverser > 1 || traversals <= 0 ||
            iteration <= 0 || ante <= 0 || stack_ante <= 0 ||
            start_street < 5 || start_street > 7 || report_every < 0) {
            throw std::runtime_error("invalid Deep CFR traversal arguments");
        }

        AdvantageClient client(port, uniform_policy);
        SampleWriter writer(output, compact_output);
        std::mt19937_64 rng(seed);
        Traverser engine(client, writer, rng, traverser, static_cast<float>(iteration));
        const auto started = std::chrono::steady_clock::now();
        for (int root = 0; root < traversals; ++root) {
            engine.run_root(sample_training_root(
                rng, ante, stack_ante, start_street));
            if (report_every > 0 && (root + 1) % report_every == 0) {
                const double progress_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                std::cerr << std::fixed << std::setprecision(6)
                          << "{\"generator\":\"deep-cfr-stud\""
                          << ",\"phase\":\"traversal\""
                          << ",\"traverser\":" << traverser
                          << ",\"roots_completed\":" << (root + 1)
                          << ",\"roots_total\":" << traversals
                          << ",\"nodes\":" << engine.nodes()
                          << ",\"trajectory_br_gap_mean_ante\":"
                          << engine.gap_mean()
                          << ",\"trajectory_br_gap_standard_error_ante\":"
                          << engine.gap_standard_error()
                          << ",\"elapsed_seconds\":" << progress_seconds
                          << "}\n";
            }
        }
        writer.finish();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"generator\":\"deep-cfr-stud\""
                  << ",\"start_street\":" << start_street
                  << ",\"traverser\":" << traverser
                  << ",\"traversals\":" << traversals
                  << ",\"nodes\":" << engine.nodes()
                  << ",\"network_queries\":" << client.queries()
                  << ",\"samples\":" << writer.count()
                  << ",\"advantage_samples\":" << writer.advantages()
                  << ",\"strategy_samples\":" << writer.strategies()
                  << ",\"sample_format\":\""
                  << (compact_output ? "lossless-token-v1" : "dense-float32-v1")
                  << "\""
                  << ",\"trajectory_br_gap_mean_ante\":"
                  << engine.gap_mean()
                  << ",\"trajectory_br_gap_standard_error_ante\":"
                  << engine.gap_standard_error()
                  << ",\"trajectory_br_gap_by_street_mean_ante\":[";
        const auto gap_by_street = engine.gap_by_street_mean();
        std::cout << gap_by_street[0] << ','
                  << gap_by_street[1] << ','
                  << gap_by_street[2] << ']'
                  << ",\"elapsed_seconds\":" << seconds << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
#endif
