#define STUD_REBEL_SEVENTH_NO_MAIN
#include "stud_rebel_seventh.cpp"

#include <winsock2.h>
#include <ws2tcpip.h>

namespace {

constexpr uint32_t kValueMagic = 0x4c564252;
constexpr uint32_t kValueVersion = 1;

struct ValueHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t columns;
};

void value_send_all(SOCKET socket, const void* data, size_t size) {
    const char* cursor = static_cast<const char*>(data);
    while (size) {
        const int sent = send(socket, cursor, static_cast<int>(size), 0);
        if (sent <= 0) throw std::runtime_error("ReBeL value IPC send failed");
        cursor += sent;
        size -= sent;
    }
}

void value_receive_all(SOCKET socket, void* data, size_t size) {
    char* cursor = static_cast<char*>(data);
    while (size) {
        const int received = recv(socket, cursor, static_cast<int>(size), 0);
        if (received <= 0) {
            throw std::runtime_error("ReBeL value IPC receive failed");
        }
        cursor += received;
        size -= received;
    }
}

class V7Client {
public:
    explicit V7Client(int port) {
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
            connect(
                socket_, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) == SOCKET_ERROR) {
            close();
            throw std::runtime_error("cannot connect to ReBeL value server");
        }
    }

    ~V7Client() { close(); }

    double value(const RebelV7Features& features) {
        const ValueHeader request{
            kValueMagic, kValueVersion, 1,
            static_cast<uint32_t>(features.size())};
        value_send_all(socket_, &request, sizeof(request));
        value_send_all(
            socket_, features.data(), features.size() * sizeof(float));
        ValueHeader response{};
        float output = 0.0f;
        value_receive_all(socket_, &response, sizeof(response));
        if (response.magic != kValueMagic ||
            response.version != kValueVersion ||
            response.rows != 1 || response.columns != 1) {
            throw std::runtime_error("invalid ReBeL value response");
        }
        value_receive_all(socket_, &output, sizeof(output));
        ++queries_;
        return output;
    }

    uint64_t queries() const { return queries_; }
    uint64_t cache_hits() const { return 0; }
    size_t cache_entries() const { return 0; }

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
};

class RecursiveSixthResolverPolicy {
public:
    struct Stats {
        uint64_t subgames = 0;
        uint64_t traversals = 0;
        uint64_t node_visits = 0;
        uint64_t nodes_created = 0;
        uint64_t leaf_queries = 0;
        uint64_t blueprint_fallbacks = 0;
        size_t maximum_local_nodes = 0;
        size_t maximum_particles = 0;
        double effective_particles_sum = 0.0;
    };

    RecursiveSixthResolverPolicy(
        MCCFR& blueprint,
        PowerAtlas& atlas,
        RebelSeventhResolverPolicy& seventh,
        V7Client& value_client,
        int iterations,
        int particles,
        double prior,
        uint64_t seed)
        : blueprint_(blueprint),
          atlas_(atlas),
          seventh_(seventh),
          value_client_(value_client),
          iterations_(iterations),
          particle_limit_(particles),
          prior_(prior),
          rng_(seed) {
        if (iterations <= 0 || particles <= 0 || prior < 0.0) {
            throw std::runtime_error("invalid recursive ReBeL configuration");
        }
        nodes_.reserve(4096);
        particles_.reserve(particles);
    }

    Action choose(const State& state, int viewer, int) {
        if (state.street < 6) return blueprint_.choose(state, viewer, 0);
        if (state.street == 7) return seventh_.choose(state, viewer, 0);
        ensure_subgame(state);
        return sample(policy(state, viewer), valid_mask(state, viewer));
    }

    std::array<double, kActionCount> policy(
        const State& state,
        int viewer,
        bool* found = nullptr) const {
        if (state.street < 6) return blueprint_.policy(state, viewer, found);
        if (state.street == 7) return seventh_.policy(state, viewer, found);
        ensure_subgame(state);
        const InfoKey key = make_power_key(state, viewer, atlas_, true);
        const auto position = nodes_.find(key);
        if (position != nodes_.end()) {
            if (found) *found = true;
            return average_strategy(
                position->second, valid_mask(state, viewer));
        }
        if (found) *found = false;
        ++stats_.blueprint_fallbacks;
        return blueprint_.policy(state, viewer);
    }

    Stats stats() const { return stats_; }

private:
    static std::string root_id(const State& state) {
        std::ostringstream output;
        output << state.ante << ':' << state.effective_stack;
        for (int seat = 0; seat < 2; ++seat) {
            output << '|' << stack_cap(state, seat) << ':';
            for (const Card& card : state.players[seat].shown) {
                output << static_cast<int>(card_token(card)) << '.';
            }
        }
        output << '|';
        for (const Event& event : state.history) {
            output << static_cast<int>(event.street) << '.'
                   << static_cast<int>(event.actor) << '.'
                   << static_cast<int>(event.action) << ';';
        }
        return output.str();
    }

    void ensure_subgame(const State& state) const {
        if (state.street != 6) {
            throw std::runtime_error("sixth resolver received wrong street");
        }
        const std::string id = root_id(state);
        if (id == current_root_id_) return;
        current_root_id_ = id;
        root_history_size_ = state.history.size();
        nodes_.clear();
        particles_.clear();
        const auto& belief = seventh_.public_belief(state);
        if (particle_limit_ >= static_cast<int>(belief.size())) {
            particles_ = belief;
        } else {
            std::vector<double> source_weights;
            source_weights.reserve(belief.size());
            for (const RebelParticle& particle : belief) {
                source_weights.push_back(particle.weight);
            }
            std::discrete_distribution<size_t> source(
                source_weights.begin(), source_weights.end());
            particles_.reserve(particle_limit_);
            for (int index = 0; index < particle_limit_; ++index) {
                RebelParticle sampled = belief[source(rng_)];
                sampled.weight = 1.0;
                particles_.push_back(std::move(sampled));
            }
        }
        if (particles_.empty()) {
            throw std::runtime_error("empty sixth public belief");
        }
        double squared = 0.0;
        double total = 0.0;
        for (const RebelParticle& particle : particles_) total += particle.weight;
        for (RebelParticle& particle : particles_) {
            particle.weight /= total;
            squared += particle.weight * particle.weight;
        }
        ++stats_.subgames;
        stats_.effective_particles_sum += 1.0 / squared;
        stats_.maximum_particles = std::max(
            stats_.maximum_particles, particles_.size());
        std::vector<double> weights;
        weights.reserve(particles_.size());
        for (const RebelParticle& particle : particles_) {
            weights.push_back(particle.weight);
        }
        std::discrete_distribution<size_t> chance(weights.begin(), weights.end());
        for (int iteration = 0; iteration < iterations_; ++iteration) {
            for (int traverser = 0; traverser < 2; ++traverser) {
                traverse(particles_[chance(rng_)].root, traverser);
                ++stats_.traversals;
            }
        }
        stats_.maximum_local_nodes = std::max(
            stats_.maximum_local_nodes, nodes_.size());
    }

    RegretNode& node(const State& state, int actor) const {
        const InfoKey key = make_power_key(state, actor, atlas_, true);
        auto [position, inserted] = nodes_.try_emplace(key);
        if (inserted) {
            const auto prior_policy = blueprint_.policy(state, actor);
            const uint8_t mask = valid_mask(state, actor);
            for (int action = 0; action < kActionCount; ++action) {
                if (!(mask & (1u << action))) continue;
                const double mass = prior_ * prior_policy[action];
                position->second.regrets[action] = mass;
                position->second.raw_regrets[action] = mass;
                position->second.strategy_sum[action] = mass;
            }
            ++stats_.nodes_created;
        }
        ++position->second.touches;
        return position->second;
    }

    static void advance_to_seventh(State& state) {
        if (state.street != 6 || state.simulation_deck.size() < 2) {
            throw std::runtime_error("invalid sixth leaf transition");
        }
        ++state.street;
        for (int seat = 0; seat < 2; ++seat) {
            const Card card = state.simulation_deck.back();
            state.simulation_deck.pop_back();
            state.players[seat].hidden.push_back(card);
        }
        reset_round(state, 7);
    }

    std::vector<RebelParticle> leaf_belief(const State& leaf) const {
        std::vector<RebelParticle> result;
        result.reserve(particles_.size());
        for (const RebelParticle& source : particles_) {
            RebelParticle particle = source;
            bool valid = true;
            for (size_t index = root_history_size_;
                 index < leaf.history.size(); ++index) {
                const Event& event = leaf.history[index];
                if (event.street != 6 || particle.root.street != 6 ||
                    particle.root.actor != event.actor) {
                    valid = false;
                    break;
                }
                const uint8_t mask = valid_mask(particle.root, event.actor);
                if (!(mask & (1u << event.action))) {
                    valid = false;
                    break;
                }
                const InfoKey key = make_power_key(
                    particle.root, event.actor, atlas_, true);
                const auto position = nodes_.find(key);
                const auto strategy = position == nodes_.end()
                    ? blueprint_.policy(particle.root, event.actor)
                    : current_strategy(position->second, mask);
                particle.weight *= std::max(
                    1e-12, strategy[event.action]);
                const ActionResult action_result = apply_action(
                    particle.root, event.actor, event.action);
                if (action_result == ActionResult::RoundEnd) {
                    advance_to_seventh(particle.root);
                } else if (action_result == ActionResult::FoldEnd) {
                    valid = false;
                    break;
                } else {
                    particle.root.actor = 1 - event.actor;
                }
            }
            if (valid && particle.weight > 0.0 && particle.root.street == 7) {
                result.push_back(std::move(particle));
            }
        }
        if (result.empty()) {
            throw std::runtime_error("sixth search produced empty leaf belief");
        }
        return result;
    }

    double traverse(State state, int traverser) const {
        ++stats_.node_visits;
        if (state.terminal) return terminal_net_search(state, traverser);
        if (state.street == 7) {
            ++stats_.leaf_queries;
            const auto belief = leaf_belief(state);
            return value_client_.value(
                rebel_v7_features(state, traverser, belief));
        }
        const int actor = state.actor;
        const uint8_t mask = valid_mask(state, actor);
        const InfoKey key = make_power_key(state, actor, atlas_, true);
        RegretNode& current = node(state, actor);
        const auto strategy = current_strategy(current, mask);
        if (actor != traverser) {
            for (int action = 0; action < kActionCount; ++action) {
                if (mask & (1u << action)) {
                    current.strategy_sum[action] += strategy[action];
                }
            }
            const Action action = sample(strategy, mask);
            const ActionResult result = apply_action(state, actor, action);
            if (result == ActionResult::RoundEnd) advance_to_seventh(state);
            else if (result != ActionResult::FoldEnd) state.actor = 1 - actor;
            return traverse(std::move(state), traverser);
        }

        std::array<double, kActionCount> action_values{};
        for (Action action : actions_from_mask(mask)) {
            State child = state;
            const ActionResult result = apply_action(child, actor, action);
            if (result == ActionResult::RoundEnd) advance_to_seventh(child);
            else if (result != ActionResult::FoldEnd) child.actor = 1 - actor;
            action_values[action] = traverse(std::move(child), traverser);
        }
        double value = 0.0;
        for (int action = 0; action < kActionCount; ++action) {
            if (mask & (1u << action)) {
                value += strategy[action] * action_values[action];
            }
        }
        RegretNode& updated = nodes_.find(key)->second;
        for (int action = 0; action < kActionCount; ++action) {
            if (!(mask & (1u << action))) continue;
            const double regret = action_values[action] - value;
            updated.raw_regrets[action] += regret;
            updated.regrets[action] = std::max(
                0.0, updated.regrets[action] + regret);
        }
        return value;
    }

    Action sample(
        const std::array<double, kActionCount>& strategy,
        uint8_t mask) const {
        const double threshold =
            std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
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

    MCCFR& blueprint_;
    PowerAtlas& atlas_;
    RebelSeventhResolverPolicy& seventh_;
    V7Client& value_client_;
    int iterations_;
    int particle_limit_;
    double prior_;
    mutable std::mt19937_64 rng_;
    mutable std::string current_root_id_;
    mutable size_t root_history_size_ = 0;
    mutable std::vector<RebelParticle> particles_;
    mutable std::unordered_map<InfoKey, RegretNode, InfoKeyHash> nodes_;
    mutable Stats stats_;
};

struct RecursiveOptions {
    std::string model =
        "cpp_mccfr\\made_call_r1000_k512_epsheur20_memory16_30m.bin";
    std::string atlas =
        "cpp_mccfr\\power512_epsheur20_memory16_v1.bin";
    std::string bucket = "power-memory16";
    int value_port = 28741;
    int hands = 1000;
    int lbr_particles = 64;
    int sixth_particles = 64;
    int sixth_iterations = 32;
    int seventh_particles = 64;
    int seventh_iterations = 100;
    int progress_seconds = 30;
    int ante = 1000;
    int stack_ante = 1000;
    double sixth_prior = 100.0;
    double seventh_prior = 100.0;
    uint64_t power_cache_max = 250000;
    uint64_t seed = 74001;
    bool self_test = false;
};

RecursiveOptions parse_recursive_options(int argc, char** argv) {
    RecursiveOptions options;
    const auto value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("missing option value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model") options.model = value(index);
        else if (argument == "--atlas") options.atlas = value(index);
        else if (argument == "--bucket") options.bucket = value(index);
        else if (argument == "--value-port") {
            options.value_port = std::stoi(value(index));
        } else if (argument == "--hands") {
            options.hands = std::stoi(value(index));
        } else if (argument == "--lbr-particles") {
            options.lbr_particles = std::stoi(value(index));
        } else if (argument == "--sixth-particles") {
            options.sixth_particles = std::stoi(value(index));
        } else if (argument == "--sixth-iterations") {
            options.sixth_iterations = std::stoi(value(index));
        } else if (argument == "--sixth-prior") {
            options.sixth_prior = std::stod(value(index));
        } else if (argument == "--seventh-particles") {
            options.seventh_particles = std::stoi(value(index));
        } else if (argument == "--seventh-iterations") {
            options.seventh_iterations = std::stoi(value(index));
        } else if (argument == "--seventh-prior") {
            options.seventh_prior = std::stod(value(index));
        } else if (argument == "--progress-seconds") {
            options.progress_seconds = std::stoi(value(index));
        } else if (argument == "--ante") {
            options.ante = std::stoi(value(index));
        } else if (argument == "--stack-ante") {
            options.stack_ante = std::stoi(value(index));
        } else if (argument == "--power-cache-max") {
            options.power_cache_max = std::stoull(value(index));
        } else if (argument == "--seed") {
            options.seed = std::stoull(value(index));
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.value_port <= 0 || options.value_port >= 65536 ||
        options.hands <= 0 || options.hands % 2 ||
        options.lbr_particles <= 0 || options.sixth_particles <= 0 ||
        options.sixth_iterations <= 0 || options.seventh_particles <= 0 ||
        options.seventh_iterations <= 0 || options.sixth_prior < 0.0 ||
        options.seventh_prior < 0.0 || options.ante <= 0 ||
        options.stack_ante <= 0) {
        throw std::runtime_error("invalid recursive ReBeL CLI configuration");
    }
    if (options.bucket != "power" &&
        options.bucket != "power-tree" &&
        options.bucket != "power-memory16") {
        throw std::runtime_error("unsupported recursive ReBeL bucket mode");
    }
    return options;
}

void print_recursive_stats(
    const RecursiveSixthResolverPolicy::Stats& sixth,
    const RebelSeventhResolverPolicy::Stats& seventh,
    const V7Client& value_client) {
    const double subgames = std::max<uint64_t>(1, sixth.subgames);
    std::cerr << std::fixed << std::setprecision(8)
              << "{\"rebel_recursive\":{"
              << "\"sixth_subgames\":" << sixth.subgames
              << ",\"sixth_traversals\":" << sixth.traversals
              << ",\"sixth_node_visits\":" << sixth.node_visits
              << ",\"sixth_nodes_created\":" << sixth.nodes_created
              << ",\"sixth_leaf_queries\":" << sixth.leaf_queries
              << ",\"sixth_average_effective_particles\":"
              << sixth.effective_particles_sum / subgames
              << ",\"sixth_maximum_local_nodes\":"
              << sixth.maximum_local_nodes
              << ",\"sixth_blueprint_fallbacks\":"
              << sixth.blueprint_fallbacks
              << ",\"seventh_subgames\":" << seventh.subgames
              << ",\"v7_network_queries\":" << value_client.queries()
              << ",\"v7_cache_hits\":" << value_client.cache_hits()
              << ",\"v7_cache_entries\":" << value_client.cache_entries()
              << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        RecursiveOptions recursive = parse_recursive_options(argc, argv);
        if (recursive.self_test) {
            recursive.hands = 10;
            recursive.lbr_particles = 16;
            recursive.sixth_particles = 16;
            recursive.sixth_iterations = 4;
            recursive.seventh_particles = 16;
            recursive.seventh_iterations = 4;
            recursive.progress_seconds = 0;
        }
        PowerAtlas atlas;
        atlas.load(recursive.atlas);
        atlas.set_assignment_cache_limit(recursive.power_cache_max);
        MCCFR blueprint(false, recursive.seed, 5, &atlas);
        blueprint.use_cumulative_street_summary(
            recursive.bucket == "power-memory16");
        blueprint.load(recursive.model);
        RebelSeventhResolverPolicy seventh(
            blueprint,
            atlas,
            recursive.seventh_iterations,
            recursive.seventh_particles,
            recursive.seventh_prior,
            recursive.seed ^ 0x6a09e667f3bcc909ull);
        V7Client value_client(recursive.value_port);
        RecursiveSixthResolverPolicy resolver(
            blueprint,
            atlas,
            seventh,
            value_client,
            recursive.sixth_iterations,
            recursive.sixth_particles,
            recursive.sixth_prior,
            recursive.seed ^ 0xbb67ae8584caa73bull);

        Options options;
        options.load_path = recursive.model;
        options.load_atlas_path = recursive.atlas;
        options.hands = recursive.hands;
        options.belief_particles = recursive.lbr_particles;
        options.ante = recursive.ante;
        options.stack_ante = recursive.stack_ante;
        options.seed = recursive.seed;
        options.start_street = 5;
        options.iterations = 0;
        options.progress_seconds = recursive.progress_seconds;
        const int result = run_policy_lbr(
            options, resolver, "cpp-rebel-6th-v7");
        const auto sixth_stats = resolver.stats();
        const auto seventh_stats = seventh.stats();
        print_recursive_stats(sixth_stats, seventh_stats, value_client);
        if (recursive.self_test) {
            if (!sixth_stats.subgames || !sixth_stats.leaf_queries ||
                !value_client.queries() || !seventh_stats.subgames) {
                throw std::runtime_error("recursive ReBeL self-test failed");
            }
            std::cerr << "{\"self_test\":\"ok\"}\n";
        }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
