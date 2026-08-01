#define DEEP_CFR_TRAVERSE_NO_MAIN
#include "deep_cfr_traverse.cpp"

namespace {

class NetworkPolicy {
public:
    NetworkPolicy(int port, uint64_t seed) : client_(port), rng_(seed) {}

    Action choose(const State& state, int actor, int) {
        const auto strategy = client_.strategy(state, actor);
        const uint8_t mask = valid_mask(state, actor);
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

    uint64_t queries() const { return client_.queries(); }

private:
    AdvantageClient client_;
    std::mt19937_64 rng_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const int port = std::stoi(argument(argc, argv, "--port", "28731"));
        const int hands = std::stoi(argument(argc, argv, "--hands", "10000"));
        const int ante = std::stoi(argument(argc, argv, "--ante", "1000"));
        const int stack_ante = std::stoi(argument(argc, argv, "--stack-ante", "1000"));
        const uint64_t seed = std::stoull(argument(argc, argv, "--seed", "7"));
        if (hands <= 0 || hands % 2 || ante <= 0 || stack_ante <= 0) {
            throw std::runtime_error("hands must be positive and even; chip settings positive");
        }
        NetworkPolicy policy(port, seed ^ 0x9e3779b97f4a7c15ull);
        std::mt19937_64 deal_rng(seed);
        std::vector<double> paired;
        paired.reserve(hands / 2);
        int wins = 0;
        int ties = 0;
        int losses = 0;
        const auto started = std::chrono::steady_clock::now();
        for (int hand = 0; hand < hands; hand += 2) {
            auto deck = fresh_deck();
            std::shuffle(deck.begin(), deck.end(), deal_rng);
            const double first = play_hand(
                deck, 0, policy, 0, ante, 7, -1, nullptr, stack_ante);
            const double second = play_hand(
                deck, 1, policy, 0, ante, 7, -1, nullptr, stack_ante);
            paired.push_back((first + second) / 2.0);
            for (double value : {first, second}) {
                if (value > 0) ++wins;
                else if (value < 0) ++losses;
                else ++ties;
            }
        }
        const double mean = std::accumulate(paired.begin(), paired.end(), 0.0) /
            paired.size();
        double variance = 0.0;
        for (double value : paired) variance += (value - mean) * (value - mean);
        variance /= std::max<size_t>(1, paired.size() - 1);
        const double error = std::sqrt(variance / paired.size());
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << std::fixed << std::setprecision(8)
                  << "{\n  \"agent_a\": \"deep-cfr-7th-policy\",\n"
                  << "  \"agent_b\": \"heuristic\",\n"
                  << "  \"hands\": " << hands << ",\n"
                  << "  \"average_profit_ante_for_a\": " << mean << ",\n"
                  << "  \"paired_standard_error_ante\": " << error << ",\n"
                  << "  \"ci95_ante_for_a\": [" << mean - 1.96 * error
                  << ", " << mean + 1.96 * error << "],\n"
                  << "  \"wins_for_a\": " << wins << ",\n"
                  << "  \"ties\": " << ties << ",\n"
                  << "  \"losses_for_a\": " << losses << ",\n"
                  << "  \"network_queries\": " << policy.queries() << ",\n"
                  << "  \"elapsed_seconds\": " << seconds << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
