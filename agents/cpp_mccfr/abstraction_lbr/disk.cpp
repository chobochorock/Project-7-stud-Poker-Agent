// Read-only paused hand-recall adapter. Reuses the exact node codec and key.
#include "disk_engine.inc"
#include "evaluate.hpp"
#include <map>

struct DiskModel {
    recall_disk::WriterLock writer_lock;
    PowerAtlas atlas;
    sqlite3* db = nullptr;
    sqlite3_stmt* lookup = nullptr;
    uint64_t hands = 0;
    struct Entry { RegretNode node{}; bool found = false; };
    std::unordered_map<std::string, Entry> cache;
    void check(int code) {
        hand_recall::require(code == SQLITE_OK || code == SQLITE_ROW || code == SQLITE_DONE,
            db ? sqlite3_errmsg(db) : "SQLite open failed");
    }
    explicit DiskModel(const std::filesystem::path& run) : writer_lock(run) {
        atlas.load((run / "hand_atlas.bin").string());
        atlas.set_assignment_cache_limit(100000);
        check(sqlite3_open_v2((run / "model.sqlite").u8string().c_str(), &db,
            SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr));
        check(sqlite3_exec(db, "PRAGMA query_only=ON; PRAGMA cache_size=-131072; BEGIN;", nullptr, nullptr, nullptr));
        sqlite3_stmt* statement = nullptr;
        check(sqlite3_prepare_v2(db, "SELECT k,v FROM meta WHERE k!='state'", -1, &statement, nullptr));
        std::map<std::string, std::string> meta;
        int code;
        while ((code = sqlite3_step(statement)) == SQLITE_ROW) {
            meta[reinterpret_cast<const char*>(sqlite3_column_text(statement, 0))] = std::string(
                static_cast<const char*>(sqlite3_column_blob(statement, 1)), sqlite3_column_bytes(statement, 1));
        }
        check(code);
        sqlite3_finalize(statement);
        hand_recall::require(meta.at("status") == "ready", "database is not ready");
        hands = std::stoull(meta.at("hands"));
        check(sqlite3_prepare_v2(db, "SELECT v FROM nodes WHERE k=?1", -1, &lookup, nullptr));
        std::cout << "LOADED hands=" << hands << " identity=" << meta.at("identity") << std::endl;
    }
    ~DiskModel() {
        if (lookup) sqlite3_finalize(lookup);
        if (db) sqlite3_close(db);
    }
    epoch7::Values policy(const State& state, int viewer, int kind, bool* found) {
        const bool recall = kind % 2 == 0;
        const auto key = std::string(1, char(recall)) + hand_recall::information_key(state, viewer, atlas, recall);
        auto it = cache.find(key);
        if (it == cache.end()) {
            if (cache.size() >= 100000) cache.clear();
            Entry entry;
            check(sqlite3_reset(lookup));
            check(sqlite3_bind_blob(lookup, 1, key.data(), int(key.size()), SQLITE_TRANSIENT));
            const int code = sqlite3_step(lookup);
            check(code);
            if (code == SQLITE_ROW) {
                entry.node = recall_disk::unpack_node(sqlite3_column_blob(lookup, 0),
                    sqlite3_column_bytes(lookup, 0)).legacy();
                entry.found = true;
            }
            it = cache.emplace(key, entry).first;
        }
        if (found) *found = it->second.found;
        return kind >= 2 ? average_strategy(it->second.node, valid_mask(state, viewer))
                         : current_strategy(it->second.node, valid_mask(state, viewer));
    }
};

int main(int argc, char** argv) {
    try {
        hand_recall::require(argc == 6 || (argc == 3 && std::string(argv[1]) == "--self-test"),
            "usage: exe RUN NEW_OUTPUT PAIRS PARTICLES SEED; or --self-test RUN");
        DiskModel model(argv[argc == 3 ? 2 : 1]);
        return abstraction_lbr::evaluate(model, argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 1;
    }
}
