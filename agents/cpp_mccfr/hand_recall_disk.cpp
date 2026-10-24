// Bounded-memory, transactional storage for the frozen hand-recall experiment.
// Usage/build: python -m agents.cpp_mccfr.hand_recall_disk --help.
// HAND_RECALL_DISK.md documents migration, exact-parity checks and disk limits.
// The old algorithm is reused; its standalone entrypoint is disabled here.
#define HAND_RECALL_NO_MAIN
#include "hand_recall.cpp"
#undef HAND_RECALL_NO_MAIN
#include <sqlite3.h>
#include <csignal>
#include <cstring>

namespace recall_disk {
namespace fs = std::filesystem;
using hand_recall::require;
using epoch7::Values;

struct WriterLock {
    HANDLE handle;
    explicit WriterLock(const fs::path& directory) {
        handle = CreateFileW((directory / "engine.lock").c_str(), GENERIC_READ | GENERIC_WRITE,
            0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        require(handle != INVALID_HANDLE_VALUE, "another C++ writer is using this run");
    }
    ~WriterLock() { CloseHandle(handle); }
};

struct Node {
    Values regrets{}, strategy_sum{};
    uint64_t touches = 0;
    RegretNode legacy() const {
        RegretNode result;
        result.regrets = regrets;
        result.strategy_sum = strategy_sum;
        result.touches = touches;
        return result;
    }
};
static_assert(sizeof(Node) == 136, "unexpected node layout");

std::string pack_node(const Node& node) {
    uint16_t mask = 0;
    std::string result(2, '\0');
    result.append(reinterpret_cast<const char*>(&node.touches), sizeof(node.touches));
    for (int index = 0; index < 16; ++index) {
        const double value = index < 8 ? node.regrets[index] : node.strategy_sum[index - 8];
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        if (bits != 0) {
            mask |= uint16_t(1) << index;
            result.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
        }
    }
    std::memcpy(result.data(), &mask, sizeof(mask));
    return result;
}

Node unpack_node(const void* bytes, int length) {
    require(length >= 10, "truncated SQLite node");
    const auto* data = static_cast<const char*>(bytes);
    Node node;
    uint16_t mask;
    std::memcpy(&mask, data, sizeof(mask));
    std::memcpy(&node.touches, data + 2, sizeof(node.touches));
    int offset = 10;
    for (int index = 0; index < 16; ++index) {
        if ((mask & (uint16_t(1) << index)) == 0) { continue; }
        require(offset + 8 <= length, "truncated SQLite value");
        double& value = index < 8 ? node.regrets[index] : node.strategy_sum[index - 8];
        std::memcpy(&value, data + offset, sizeof(value));
        require(std::isfinite(value) && (index < 8 || value >= 0), "invalid SQLite value");
        offset += 8;
    }
    require(offset == length, "trailing SQLite node bytes");
    return node;
}

struct Database {
    sqlite3* handle = nullptr;
    sqlite3_stmt* lookup = nullptr;
    sqlite3_stmt* upsert = nullptr;
    struct Entry { Node node; bool found = false, dirty = false; };
    std::unordered_map<std::string, Entry> cache;
    size_t cache_limit;
    uint64_t reads = 0, writes = 0;

    void check(int code) const {
        if (code != SQLITE_OK && code != SQLITE_DONE && code != SQLITE_ROW) {
            throw std::runtime_error("SQLite " + std::to_string(code) + ": " +
                (handle ? sqlite3_errmsg(handle) : "open failed"));
        }
    }
    void exec(const std::string& sql) { check(sqlite3_exec(handle, sql.c_str(), nullptr, nullptr, nullptr)); }
    sqlite3_stmt* prepare(const char* sql) {
        sqlite3_stmt* statement = nullptr;
        check(sqlite3_prepare_v2(handle, sql, -1, &statement, nullptr));
        return statement;
    }
    Database(const fs::path& path, int cache_mib, size_t entries) : cache_limit(entries) {
        require(cache_mib >= 1 && entries > 0, "invalid cache limits");
        check(sqlite3_open_v2(path.u8string().c_str(), &handle,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr));
        sqlite3_busy_timeout(handle, 5000);
        exec("PRAGMA page_size=8192; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;"
             "PRAGMA temp_store=FILE; PRAGMA mmap_size=0; PRAGMA cache_spill=ON;"
             "PRAGMA journal_size_limit=67108864; PRAGMA wal_autocheckpoint=4096;");
        exec("PRAGMA cache_size=-" + std::to_string(int64_t(cache_mib) * 1024));
        exec("CREATE TABLE IF NOT EXISTS nodes(k BLOB PRIMARY KEY,v BLOB NOT NULL) WITHOUT ROWID;"
             "CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v BLOB NOT NULL) WITHOUT ROWID;"
             "CREATE TABLE IF NOT EXISTS metrics(kind INTEGER,hand INTEGER,root INTEGER,line TEXT NOT NULL,"
             "PRIMARY KEY(kind,hand,root)) WITHOUT ROWID;"
             "CREATE TABLE IF NOT EXISTS action_values(hand INTEGER,root INTEGER,policy INTEGER,v BLOB NOT NULL,"
             "PRIMARY KEY(hand,root,policy)) WITHOUT ROWID;");
        lookup = prepare("SELECT v FROM nodes WHERE k=?1");
        upsert = prepare("INSERT INTO nodes VALUES(?1,?2) ON CONFLICT(k) DO UPDATE SET v=excluded.v");
    }
    ~Database() {
        // Pending work is rolled back, never silently committed by a destructor.
        if (lookup) { sqlite3_finalize(lookup); }
        if (upsert) { sqlite3_finalize(upsert); }
        if (handle) { sqlite3_close_v2(handle); }
    }
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    void write_node(const std::string& key, const Node& node) {
        const auto packed = pack_node(node);
        check(sqlite3_reset(upsert));
        check(sqlite3_bind_blob(upsert, 1, key.data(), int(key.size()), SQLITE_TRANSIENT));
        check(sqlite3_bind_blob(upsert, 2, packed.data(), int(packed.size()), SQLITE_TRANSIENT));
        check(sqlite3_step(upsert));
        ++writes;
    }
    void flush_cache() {
        for (const auto& [key, entry] : cache) {
            if (entry.dirty) { write_node(key, entry.node); }
        }
        cache.clear();
    }
    // Value copies, not references: recursive traversal may evict any cache row.
    std::pair<Node, bool> get(const std::string& key) {
        auto found = cache.find(key);
        if (found == cache.end()) {
            if (cache.size() >= cache_limit) { flush_cache(); }
            Entry entry;
            check(sqlite3_reset(lookup));
            check(sqlite3_bind_blob(lookup, 1, key.data(), int(key.size()), SQLITE_TRANSIENT));
            const int code = sqlite3_step(lookup);
            check(code);
            if (code == SQLITE_ROW) {
                entry.node = unpack_node(sqlite3_column_blob(lookup, 0), sqlite3_column_bytes(lookup, 0));
                entry.found = true;
            }
            check(sqlite3_reset(lookup));
            ++reads;
            found = cache.emplace(key, entry).first;
        }
        return {found->second.node, found->second.found};
    }
    void put(const std::string& key, const Node& node) {
        if (!cache.count(key) && cache.size() >= cache_limit) { flush_cache(); }
        cache[key] = {node, true, true};
    }
    void set(const std::string& key, const std::string& value) {
        auto* statement = prepare("INSERT INTO meta VALUES(?1,?2) ON CONFLICT(k) DO UPDATE SET v=excluded.v");
        check(sqlite3_bind_text(statement, 1, key.c_str(), -1, SQLITE_TRANSIENT));
        check(sqlite3_bind_blob(statement, 2, value.data(), int(value.size()), SQLITE_TRANSIENT));
        const int code = sqlite3_step(statement);
        sqlite3_finalize(statement);
        check(code);
    }
    std::string get_meta(const std::string& key) {
        auto* statement = prepare("SELECT v FROM meta WHERE k=?1");
        check(sqlite3_bind_text(statement, 1, key.c_str(), -1, SQLITE_TRANSIENT));
        const int code = sqlite3_step(statement);
        std::string value;
        if (code == SQLITE_ROW) {
            value.assign(static_cast<const char*>(sqlite3_column_blob(statement, 0)),
                sqlite3_column_bytes(statement, 0));
        }
        sqlite3_finalize(statement);
        check(code);
        return value;
    }
    void metric(int kind, uint64_t hand, int root, const std::string& line) {
        auto* statement = prepare("INSERT INTO metrics VALUES(?1,?2,?3,?4)");
        check(sqlite3_bind_int(statement, 1, kind));
        check(sqlite3_bind_int64(statement, 2, hand));
        check(sqlite3_bind_int(statement, 3, root));
        check(sqlite3_bind_text(statement, 4, line.c_str(), -1, SQLITE_TRANSIENT));
        const int code = sqlite3_step(statement);
        sqlite3_finalize(statement);
        check(code);
    }
    void action_values(uint64_t hand, int root, int policy, const Values& q,
                       const Values& se, const Values& strategy) {
        std::array<Values, 3> values{q, se, strategy};
        auto* statement = prepare("INSERT INTO action_values VALUES(?1,?2,?3,?4)");
        check(sqlite3_bind_int64(statement, 1, hand));
        check(sqlite3_bind_int(statement, 2, root));
        check(sqlite3_bind_int(statement, 3, policy));
        check(sqlite3_bind_blob(statement, 4, values.data(), sizeof(values), SQLITE_TRANSIENT));
        const int code = sqlite3_step(statement);
        sqlite3_finalize(statement);
        check(code);
    }
    void commit() {
        flush_cache();
        exec("COMMIT");
        // No long-lived reader in this driver. A concurrent plot may delay
        // truncation, but must not make a committed training step fail.
        const int code = sqlite3_wal_checkpoint_v2(handle, nullptr, SQLITE_CHECKPOINT_TRUNCATE, nullptr, nullptr);
        if (code != SQLITE_BUSY) { check(code); }
    }
};

struct Solver {
    const PowerAtlas& atlas;
    Database& db;
    bool recall;
    std::mt19937_64 rng;
    uint64_t nodes = 0, states = 0, key_bytes = 0;
    Solver(const PowerAtlas& source, Database& store, bool memory, uint64_t seed)
        : atlas(source), db(store), recall(memory), rng(seed) {}
    std::string key(const State& state, int viewer) const {
        return std::string(1, char(recall)) + hand_recall::information_key(state, viewer, atlas, recall);
    }
    double traverse(State state, int traverser) {
        ++nodes;
        if (state.terminal) { return terminal_net_search(state, traverser); }
        const int actor = state.actor;
        const auto mask = valid_mask(state, actor);
        const auto identity = key(state, actor);
        auto [node, found] = db.get(identity);
        if (!found) { ++states; key_bytes += identity.size() - 1; }
        ++node.touches;
        const auto strategy = current_strategy(node.legacy(), mask);
        if (actor != traverser) {
            for (Action action : actions_from_mask(mask)) { node.strategy_sum[action] += strategy[action]; }
            db.put(identity, node);
            const Action action = epoch7::sample(strategy, mask, rng);
            return traverse(epoch7::child(std::move(state), action), traverser);
        }
        db.put(identity, node);
        Values values{};
        double value = 0;
        for (Action action : actions_from_mask(mask)) {
            values[action] = traverse(epoch7::child(state, action), traverser);
            value += strategy[action] * values[action];
        }
        node = db.get(identity).first;
        for (Action action : actions_from_mask(mask)) { node.regrets[action] += values[action] - value; }
        db.put(identity, node);
        return value;
    }
};

struct Target {
    Solver& solver;
    bool average;
    uint64_t queries = 0, missing = 0, uniform = 0;
    Values policy(const State& state, int viewer) {
        ++queries;
        const auto [node, found] = solver.db.get(solver.key(state, viewer));
        missing += !found;
        const auto mask = valid_mask(state, viewer);
        double regret_mass = 0, strategy_mass = 0;
        for (auto action : actions_from_mask(mask)) {
            regret_mass += std::max(0.0, node.regrets[action]);
            strategy_mass += node.strategy_sum[action];
        }
        uniform += regret_mass == 0 && (!average || strategy_mass == 0);
        return average ? average_strategy(node.legacy(), mask) : current_strategy(node.legacy(), mask);
    }
};

struct Gap {
    double gap = 0, crossfit = 0, mean_se = 0;
    Values q{}, se{}, strategy{};
};

Gap local_gap(const State& root, const std::vector<State>& particles, Target& policy, uint64_t seed) {
    Gap result;
    const int viewer = root.actor;
    const auto actions = actions_from_mask(valid_mask(root, viewer));
    result.strategy = policy.policy(root, viewer);
    std::array<Values, 2> half{};
    Values squares{}, sums{};
    const size_t middle = particles.size() / 2;
    for (size_t i = 0; i < particles.size(); ++i) {
        for (Action action : actions) {
            const double value = epoch7::rollout(epoch7::child(particles[i], action), viewer,
                policy, seed + i * 0x9e3779b97f4a7c15ull);
            // Identical summation order to the frozen epoch7 evaluator.
            result.q[action] += value / particles.size();
            half[i >= middle][action] += value / (i < middle ? middle : particles.size() - middle);
            sums[action] += value;
            squares[action] += value * value;
        }
    }
    double expected = 0, best = -std::numeric_limits<double>::infinity();
    for (Action action : actions) {
        best = std::max(best, result.q[action]);
        expected += result.strategy[action] * result.q[action];
        result.se[action] = std::sqrt(std::max(0.0,
            (squares[action] - sums[action] * sums[action] / particles.size()) /
            (particles.size() * (particles.size() - 1))));
        result.mean_se += result.se[action] / actions.size();
    }
    result.gap = std::max(0.0, best - expected);
    for (int side = 0; side < 2; ++side) {
        const auto best_action = *std::max_element(actions.begin(), actions.end(),
            [&](Action a, Action b) { return half[side][a] < half[side][b]; });
        double reference = 0;
        for (Action action : actions) { reference += result.strategy[action] * half[1 - side][action]; }
        result.crossfit += (half[1 - side][best_action] - reference) / 2;
    }
    return result;
}

void evaluate(Database& db, int kind, uint64_t hand, int root_index, uint64_t seed,
              int count, Solver& recall, Solver& baseline) {
    const State root = epoch7::root_for(seed);
    std::mt19937_64 rng(seed ^ 0x123456789ull);
    const auto particles = epoch7::root_particles(root, count, rng);
    std::array<Target, 4> targets{{{recall, false}, {baseline, false}, {recall, true}, {baseline, true}}};
    std::array<Gap, 4> gaps;
    std::ostringstream out;
    out << std::setprecision(17) << hand;
    for (size_t i = 0; i < targets.size(); ++i) {
        gaps[i] = local_gap(root, particles, targets[i], seed ^ 0xabcdefull);
        out << ',' << gaps[i].gap;
        if (kind == 1) { db.action_values(hand, root_index, int(i), gaps[i].q, gaps[i].se, gaps[i].strategy); }
    }
    out << ',' << seed;
    for (const auto& target : targets) { out << ',' << target.queries << ',' << target.missing; }
    out << ',' << recall.states << ',' << baseline.states << ',' << recall.nodes << ',' << baseline.nodes
        << ',' << recall.key_bytes + recall.states * sizeof(RegretNode)
        << ',' << baseline.key_bytes + baseline.states * sizeof(RegretNode)
        << ',' << epoch7::peak_rss();
    for (size_t i = 0; i < targets.size(); ++i) {
        out << ',' << gaps[i].crossfit << ',' << gaps[i].mean_se << ',' << targets[i].uniform;
    }
    db.metric(kind, hand, root_index, out.str());
}

std::string encode_state(const hand_recall::Progress& progress, const Solver& recall, const Solver& baseline) {
    std::ostringstream out(std::ios::binary);
    hand_recall::write_text(out, "RECALLDISK1");
    epoch7::write(out, progress.hands);
    epoch7::write(out, progress.train_seconds);
    epoch7::write(out, progress.eval_seconds);
    for (const auto* solver : {&recall, &baseline}) {
        epoch7::write(out, uint8_t(solver->recall));
        epoch7::write(out, solver->nodes);
        epoch7::write(out, solver->states);
        epoch7::write(out, solver->key_bytes);
        std::ostringstream rng;
        rng << solver->rng;
        hand_recall::write_text(out, rng.str());
    }
    return out.str();
}

hand_recall::Progress decode_state(const std::string& value, Solver& recall, Solver& baseline) {
    std::istringstream in(value, std::ios::binary);
    require(hand_recall::read_text(in) == "RECALLDISK1", "invalid disk checkpoint");
    hand_recall::Progress progress;
    progress.hands = epoch7::read<uint64_t>(in);
    progress.train_seconds = epoch7::read<double>(in);
    progress.eval_seconds = epoch7::read<double>(in);
    for (auto* solver : {&recall, &baseline}) {
        require(epoch7::read<uint8_t>(in) == solver->recall, "arm mismatch");
        solver->nodes = epoch7::read<uint64_t>(in);
        solver->states = epoch7::read<uint64_t>(in);
        solver->key_bytes = epoch7::read<uint64_t>(in);
        std::istringstream rng(hand_recall::read_text(in));
        rng >> solver->rng;
        require(bool(rng), "invalid sampler RNG");
    }
    require(in.peek() == std::char_traits<char>::eof(), "trailing disk checkpoint bytes");
    return progress;
}

void commit(Database& db, const hand_recall::Progress& progress, Solver& recall, Solver& baseline) {
    db.set("state", encode_state(progress, recall, baseline));
    db.set("hands", std::to_string(progress.hands));
    db.set("status", "ready");
    db.commit();
}

bool enough_disk(const fs::path& directory, uint64_t reserve) { return fs::space(directory).available >= reserve; }

void import_checkpoint(Database& db, const fs::path& source, const std::string& old_identity,
                       Solver& recall, Solver& baseline, uint64_t reserve, const fs::path& directory) {
    require(db.get_meta("status") != "ready", "refusing to overwrite trained database");
    std::ifstream in(source / "checkpoint.bin", std::ios::binary);
    require(hand_recall::read_text(in) == "HANDRECALL1", "invalid source checkpoint");
    require(hand_recall::read_text(in) == old_identity, "source checkpoint identity mismatch");
    hand_recall::Progress progress;
    progress.hands = epoch7::read<uint64_t>(in);
    progress.train_seconds = epoch7::read<double>(in);
    progress.eval_seconds = epoch7::read<double>(in);
    progress.trace_bytes = epoch7::read<uint64_t>(in);
    progress.panel_bytes = epoch7::read<uint64_t>(in);
    db.exec("BEGIN IMMEDIATE");
    db.set("status", "importing");
    db.commit();
    db.exec("BEGIN IMMEDIATE");
    // Idempotent re-import after interruption: same keys and exact payloads,
    // not increments. Training is forbidden until both arms and logs are ready.
    for (auto* solver : {&recall, &baseline}) {
        require(epoch7::read<uint8_t>(in) == solver->recall, "import arm mismatch");
        solver->nodes = epoch7::read<uint64_t>(in);
        std::istringstream random_state(hand_recall::read_text(in));
        random_state >> solver->rng;
        require(bool(random_state), "invalid imported RNG");
        solver->states = epoch7::read<uint64_t>(in);
        solver->key_bytes = 0;
        for (uint64_t row = 0; row < solver->states; ++row) {
            const std::string key = hand_recall::read_text(in, 10000);
            require(key.size() >= 3 && uint8_t(key[1]) >= 5 && uint8_t(key[1]) <= 7, "invalid information key");
            solver->key_bytes += key.size();
            Node node;
            in.read(reinterpret_cast<char*>(&node), sizeof(node));
            require(bool(in), "truncated imported node");
            for (double value : node.regrets) { require(std::isfinite(value), "invalid imported regret"); }
            for (double value : node.strategy_sum) { require(std::isfinite(value) && value >= 0, "invalid imported average"); }
            db.write_node(std::string(1, char(solver->recall)) + key, node);
            if ((row + 1) % 25000 == 0) {
                db.commit();
                if (!enough_disk(directory, reserve)) { throw std::runtime_error("import paused: low disk; rerun prepare after freeing space"); }
                db.exec("BEGIN IMMEDIATE");
            }
            if ((row + 1) % 250000 == 0) {
                std::cout << "IMPORT arm=" << (solver->recall ? "recall" : "baseline")
                    << " rows=" << row + 1 << '/' << solver->states
                    << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0 << std::endl;
            }
        }
    }
    require(in.peek() == std::char_traits<char>::eof(), "trailing source checkpoint bytes");
    db.exec("DELETE FROM metrics; DELETE FROM action_values;");
    for (int kind : {0, 1}) {
        const auto path = source / (kind ? "panel.csv" : "per_hand.csv");
        const auto bytes = kind ? progress.panel_bytes : progress.trace_bytes;
        require(fs::file_size(path) >= bytes, "source CSV shorter than checkpoint");
        std::ifstream csv(path, std::ios::binary);
        std::string line;
        std::getline(csv, line);
        int root = 0;
        uint64_t previous = std::numeric_limits<uint64_t>::max();
        while (uint64_t(csv.tellg()) < bytes && std::getline(csv, line)) {
            require(csv.tellg() >= 0 && uint64_t(csv.tellg()) <= bytes, "CSV checkpoint is not a line boundary");
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            // Only the writer's numeric prefix is parsed; CSV fields stay opaque.
            std::istringstream prefix(line);
            uint64_t hand;
            char comma;
            require(bool(prefix >> hand >> comma) && comma == ',' && hand <= progress.hands, "invalid source metric prefix");
            if (previous != hand) { root = 0; previous = hand; }
            db.metric(kind, hand, kind ? root++ : 0, line);
        }
        require(csv.tellg() >= 0 && uint64_t(csv.tellg()) == bytes, "incomplete committed CSV prefix");
    }
    commit(db, progress, recall, baseline);
    std::cout << "IMPORTED hands=" << progress.hands << " states=" << recall.states + baseline.states << std::endl;
}

volatile std::sig_atomic_t stop_requested = 0;
BOOL WINAPI console_control(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        stop_requested = 1;
        return TRUE;
    }
    return FALSE;
}

void assert_parity(hand_recall::Solver& memory, Solver& disk) {
    require(memory.nodes == disk.nodes && memory.table.size() == disk.states && memory.key_bytes == disk.key_bytes,
        "RAM/disk counters differ");
    require(memory.rng == disk.rng, "RAM/disk RNG differs");
    for (const auto& [key, node] : memory.table) {
        auto [value, found] = disk.db.get(std::string(1, char(disk.recall)) + key);
        require(found && value.regrets == node.regrets && value.strategy_sum == node.strategy_sum &&
            value.touches == node.touches, "RAM/disk node differs");
    }
}

void self_test(const fs::path& directory, const fs::path& atlas_path) {
    require(!fs::exists(directory / "model.sqlite"), "self-test requires empty directory");
    Node sparse;
    sparse.regrets[1] = -0.0;
    sparse.regrets[6] = -12.25;
    sparse.strategy_sum[4] = 0.125;
    sparse.touches = 99;
    const auto packed = pack_node(sparse);
    const auto unpacked = unpack_node(packed.data(), int(packed.size()));
    require(packed.size() == 34 && std::memcmp(&sparse, &unpacked, sizeof(Node)) == 0,
        "sparse codec changed double bits or touches");
    PowerAtlas atlas;
    atlas.load(atlas_path.string());
    hand_recall::Solver memory_r(atlas, true, 111), memory_b(atlas, false, 111);
    hand_recall::Progress progress;
    {
        Database db(directory / "model.sqlite", 1, 3);
        Solver recall(atlas, db, true, 111), baseline(atlas, db, false, 111);
        db.exec("BEGIN IMMEDIATE");
        for (int hand = 1; hand <= 6; ++hand) {
            const State root = epoch7::root_for(41 + hand % 3);
            for (int actor : {0, 1}) {
                const double first = memory_r.traverse(root, actor);
                require(first == recall.traverse(root, actor), "recall traversal value differs");
                const double second = memory_b.traverse(root, actor);
                require(second == baseline.traverse(root, actor), "baseline traversal value differs");
            }
            assert_parity(memory_r, recall);
            assert_parity(memory_b, baseline);
            progress.hands = hand;
        }
        std::mt19937_64 rng(9);
        const auto root = epoch7::root_for(77);
        const auto particles = epoch7::root_particles(root, 8, rng);
        for (bool average : {false, true}) {
            hand_recall::Target old{memory_r, average};
            Target target{recall, average};
            require(epoch7::local_gap(root, particles, old, 3) == local_gap(root, particles, target, 3).gap,
                "new diagnostic changed original gap");
            require(old.queries == target.queries && old.missing == target.missing, "evaluation coverage differs");
        }
        db.metric(0, 6, 0, "6,test");
        commit(db, progress, recall, baseline);
        db.exec("BEGIN IMMEDIATE");
        recall.traverse(epoch7::root_for(43), 0);
        db.metric(0, 7, 0, "7,uncommitted");
        db.flush_cache();
        // Simulate lost uncommitted work; destructor closes and rolls it back.
    }
    {
        Database db(directory / "model.sqlite", 1, 3);
        Solver recall(atlas, db, true, 0), baseline(atlas, db, false, 0);
        progress = decode_state(db.get_meta("state"), recall, baseline);
        require(progress.hands == 6, "checkpoint advanced across rollback");
        assert_parity(memory_r, recall);
        assert_parity(memory_b, baseline);
        db.exec("BEGIN IMMEDIATE");
        const auto root = epoch7::root_for(51);
        for (int actor : {0, 1}) {
            memory_r.traverse(root, actor);
            recall.traverse(root, actor);
            memory_b.traverse(root, actor);
            baseline.traverse(root, actor);
        }
        assert_parity(memory_r, recall);
        assert_parity(memory_b, baseline);
        progress.hands = 7;
        commit(db, progress, recall, baseline);
    }
    const auto source = directory / "legacy";
    const auto destination = directory / "imported";
    fs::create_directories(source);
    fs::create_directories(destination);
    for (const auto* filename : {"per_hand.csv", "panel.csv"}) {
        std::ofstream csv(source / filename);
        hand_recall::header(csv);
        hand_recall::evaluate(csv, 7, 77, 8, memory_r, memory_b);
    }
    hand_recall::checkpoint(source, "test_identity", progress, memory_r, memory_b);
    {
        Database db(destination / "model.sqlite", 1, 3);
        Solver recall(atlas, db, true, 0), baseline(atlas, db, false, 0);
        import_checkpoint(db, source, "test_identity", recall, baseline, 0, destination);
        assert_parity(memory_r, recall);
        assert_parity(memory_b, baseline);
        require(decode_state(db.get_meta("state"), recall, baseline).hands == 7, "import progress differs");
    }
    std::cout << "SELF_TEST exact RAM/disk parity, eviction, gap, rollback, RNG resume, legacy import: PASS\n";
}

int run(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--self-test") {
        self_test(argv[2], argv[3]);
        return 0;
    }
    if (argc != 17) {
        std::cout << "Use python -m agents.cpp_mccfr.hand_recall_disk --help\n";
        return argc == 2 ? 0 : 1;
    }
    const std::string mode = argv[1], identity = argv[3];
    const fs::path directory(argv[2]);
    const uint64_t hands = std::stoull(argv[4]), seed = std::stoull(argv[5]), eval_seed = std::stoull(argv[6]);
    const int particles = std::stoi(argv[7]), panel_roots = std::stoi(argv[9]);
    const uint64_t panel_every = std::stoull(argv[8]), commit_every = std::stoull(argv[10]);
    const int cache_mib = std::stoi(argv[11]);
    const uint64_t cache_entries = std::stoull(argv[12]), reserve = std::stoull(argv[13]);
    const fs::path source(argv[14]);
    const std::string old_identity = argv[15];
    const uint64_t report_every = std::stoull(argv[16]);
    require(hands && seed && eval_seed && particles >= 2 && particles % 2 == 0 && panel_roots > 0 &&
        panel_every && commit_every && report_every, "invalid runtime parameters");
    require(enough_disk(directory, reserve), "insufficient free disk before start");
    WriterLock writer_lock(directory);
    PowerAtlas atlas;
    atlas.load((directory / "hand_atlas.bin").string());
    atlas.set_assignment_cache_limit(100000);
    Database db(directory / "model.sqlite", cache_mib, cache_entries);
    if (db.get_meta("identity").empty()) { db.set("identity", identity); }
    require(db.get_meta("identity") == identity, "database/manifest identity mismatch");
    Solver recall(atlas, db, true, seed), baseline(atlas, db, false, seed);
    if (mode == "import") {
        import_checkpoint(db, source, old_identity, recall, baseline, reserve, directory);
        return 0;
    }
    require(mode == "train", "unknown disk driver mode");
    SetConsoleCtrlHandler(console_control, TRUE);
    hand_recall::Progress progress;
    const auto timed_panel = [&]() {
        const auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < panel_roots; ++index) {
            evaluate(db, 1, progress.hands, index, eval_seed ^ ((index + 1) * 0xd1b54a32d192ed03ull),
                particles, recall, baseline);
        }
        progress.eval_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    if (db.get_meta("status") == "ready") {
        progress = decode_state(db.get_meta("state"), recall, baseline);
    } else {
        require(db.get_meta("status").empty() && source.empty(), "finish checkpoint import before training");
        db.exec("BEGIN IMMEDIATE");
        timed_panel();
        commit(db, progress, recall, baseline);
    }
    require(hands >= progress.hands, "target below saved hand count");
    std::cout << "READY hands=" << progress.hands << " target=" << hands
        << " sqlite=" << sqlite3_libversion() << " db_cache_mib=" << cache_mib
        << " node_cache_entries=" << cache_entries << std::endl;
    uint64_t committed = progress.hands;
    db.exec("BEGIN IMMEDIATE");
    for (uint64_t hand = progress.hands + 1; hand <= hands; ++hand) {
        auto start = std::chrono::steady_clock::now();
        const auto root = epoch7::root_for(seed ^ (hand * 0xa0761d6478bd642full));
        for (auto* solver : {&recall, &baseline}) {
            solver->traverse(root, 0);
            solver->traverse(root, 1);
        }
        progress.hands = hand;
        progress.train_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        evaluate(db, 0, hand, 0, eval_seed ^ (hand * 0x94d049bb133111ebull), particles, recall, baseline);
        progress.eval_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (hand % panel_every == 0) { timed_panel(); }
        const bool low_disk = !enough_disk(directory, reserve);
        const bool save = hand - committed >= commit_every || hand == hands || stop_requested || low_disk;
        if (save) {
            commit(db, progress, recall, baseline);
            committed = hand;
        }
        if (hand % report_every == 0 || hand == hands || stop_requested || low_disk) {
            std::cout << "PROGRESS hands=" << hand << " committed_hands=" << committed
                << " target=" << hands << " recall_states=" << recall.states
                << " baseline_states=" << baseline.states << " peak_rss_mib=" << epoch7::peak_rss() / 1048576.0
                << " db_gib=" << fs::file_size(directory / "model.sqlite") / double(uint64_t(1) << 30)
                << " free_gib=" << fs::space(directory).available / double(uint64_t(1) << 30)
                << " db_reads=" << db.reads << " db_writes=" << db.writes
                << " train_seconds=" << progress.train_seconds << " eval_seconds=" << progress.eval_seconds << std::endl;
        }
        if (low_disk || stop_requested) {
            std::cout << "PAUSED reason=" << (low_disk ? "low_disk" : "user_stop")
                << " committed_hands=" << committed << "; rerun train to resume\n";
            return 2;
        }
        if (save && hand < hands) { db.exec("BEGIN IMMEDIATE"); }
    }
    if (sqlite3_get_autocommit(db.handle) == 0) { db.exec("ROLLBACK"); }
    std::cout << "COMPLETE hands=" << progress.hands << " target=" << hands << std::endl;
    return 0;
}
}  // namespace recall_disk

int main(int argc, char** argv) {
    try {
        return recall_disk::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << "; last committed DB checkpoint is preserved\n";
        return 1;
    }
}
