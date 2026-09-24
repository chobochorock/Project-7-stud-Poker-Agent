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
        const int sent = send(socket, cursor, static_cast<int>(std::min(
            size, static_cast<size_t>(std::numeric_limits<int>::max()))), 0);
        if (sent <= 0) throw std::runtime_error("Deep CFR IPC send failed");
        cursor += sent;
        size -= sent;
    }
}

void receive_all(SOCKET socket, void* data, size_t size) {
    char* cursor = static_cast<char*>(data);
    while (size) {
        const int received = recv(socket, cursor, static_cast<int>(std::min(
            size, static_cast<size_t>(std::numeric_limits<int>::max()))), 0);
        if (received <= 0) throw std::runtime_error("Deep CFR IPC receive failed");
        cursor += received;
        size -= received;
    }
}

int argument(int argc, char** argv, const std::string& name, int fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return std::stoi(argv[index + 1]);
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int port = argument(argc, argv, "--port", 28731);
        const int batches = argument(argc, argv, "--batches", 100);
        const int batch_size = argument(argc, argv, "--batch-size", 256);
        const int ante = argument(argc, argv, "--ante", 1000);
        const int stack_ante = argument(argc, argv, "--stack-ante", 1000);
        const int seed = argument(argc, argv, "--seed", 7);
        if (port <= 0 || port >= 65536 || batches <= 0 || batch_size <= 0 ||
            ante <= 0 || stack_ante <= 0) {
            throw std::runtime_error("IPC probe arguments must be positive");
        }

        WSADATA winsock{};
        if (WSAStartup(MAKEWORD(2, 2), &winsock) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
        SOCKET socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (socket_handle == INVALID_SOCKET) {
            WSACleanup();
            throw std::runtime_error("socket creation failed");
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<u_short>(port));
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        if (connect(socket_handle, reinterpret_cast<sockaddr*>(&address),
                    sizeof(address)) == SOCKET_ERROR) {
            closesocket(socket_handle);
            WSACleanup();
            throw std::runtime_error("cannot connect to Deep CFR IPC server");
        }

        std::mt19937_64 rng(seed);
        std::vector<float> inputs(
            static_cast<size_t>(batch_size) * kDeepCfrTensorSize);
        std::vector<float> outputs(
            static_cast<size_t>(batch_size) * kActionCount);
        const auto started = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int batch = 0; batch < batches; ++batch) {
            for (int row = 0; row < batch_size; ++row) {
                auto deck = fresh_deck();
                std::shuffle(deck.begin(), deck.end(), rng);
                const State state = sample_fifth_street_root(
                    deck, ante, stack_ante);
                const auto tensor = deep_cfr_tensor(state, state.actor);
                std::copy(tensor.begin(), tensor.end(),
                          inputs.begin() + static_cast<size_t>(row) * tensor.size());
            }
            const IpcHeader request{
                kIpcMagic, kIpcVersion,
                static_cast<uint32_t>(batch_size),
                static_cast<uint32_t>(kDeepCfrTensorSize)};
            send_all(socket_handle, &request, sizeof(request));
            send_all(socket_handle, inputs.data(), inputs.size() * sizeof(float));
            IpcHeader response{};
            receive_all(socket_handle, &response, sizeof(response));
            if (response.magic != kIpcMagic || response.version != kIpcVersion ||
                response.rows != static_cast<uint32_t>(batch_size) ||
                response.columns != kActionCount) {
                throw std::runtime_error("invalid Deep CFR IPC response");
            }
            receive_all(socket_handle, outputs.data(), outputs.size() * sizeof(float));
            for (float value : outputs) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite network output");
                }
                checksum += value;
            }
        }
        shutdown(socket_handle, SD_BOTH);
        closesocket(socket_handle);
        WSACleanup();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const uint64_t rows = static_cast<uint64_t>(batches) * batch_size;
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"probe\":\"deep-cfr-ipc\""
                  << ",\"tensor_dimensions\":" << kDeepCfrTensorSize
                  << ",\"actions\":" << kActionCount
                  << ",\"batches\":" << batches
                  << ",\"batch_size\":" << batch_size
                  << ",\"rows\":" << rows
                  << ",\"rows_per_second\":" << rows / elapsed
                  << ",\"elapsed_seconds\":" << elapsed
                  << ",\"checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
