// Prior-posterior ASWISS on 7-stud: the Model members of --partition pp (PP_STUD7_DESIGN_KO.md sections 5-7).
// Included inside the anonymous namespace of aswiss_stud.cpp after the Model class and label_root().
//
// Routing: (street, legal mask) = the start cell, shared by both seats -> prior tree on the viewer's observation stream
// -> leaf -> its pointer (a posterior bucket); a pending leaf uses its provisional pointer (the none child).
// Direction of an information state at split level j: the particle estimate (Policy-LBR's belief) of the per-visit
// counterfactual regret of the average policy at level j, sigma_j(I) = the average table of the level-j ancestor of
// I's bucket (the live table during a judgment, a stored copy afterwards). Label (rule (a)): the action of largest
// regret among those whose regret exceeds z standard errors over the particles, else none (-1).
#pragma once

namespace {
inline uint64_t pp_mix(uint64_t a, uint64_t b) {
    uint64_t h = a * 0x9e3779b97f4a7c15ull ^ (b + 0x632be59bd9b4e019ull + (a << 6) + (a >> 2));
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 29;
    return h;
}
int pp_h4_multiplicity(const std::array<Card, 2>& hidden, const Card& shown, const Card& discarded) {  // as PolicyLBR's
    const std::array<Card, 4> cards{hidden[0], hidden[1], shown, discarded};
    std::array<int, 4> order{0, 1, 2, 3};
    int matches = 0;
    do {
        std::vector<Card> hand;
        hand.reserve(4);
        for (int index : order) hand.push_back(cards[index]);
        const auto [d, r] = discard_reveal(hand);
        matches += hand[d] == discarded && hand[r] == shown;
    } while (std::next_permutation(order.begin(), order.end()));
    return matches;
}
}  // namespace

void Model::pp_setup(const pp::Opts& o, uint64_t seed) {
    if (!pp_) pp_ = std::make_unique<pp::Data>();
    pp_->opt = o;
    pp_seed_ = seed;
}

uint32_t Model::pp_locate(const State& s, int viewer, int32_t* leaf_out) const {
    const uint16_t ctx = static_cast<uint16_t>(s.street << 8 | valid_mask(s, viewer));
    const auto it = pp_->cell_of.find(ctx);
    if (it == pp_->cell_of.end()) { if (leaf_out) *leaf_out = -1; return kNone; }
    const pp::Cell& c = pp_->cells[it->second];
    const int32_t l = pp_->nodes[c.root].leaf >= 0 ? pp_->nodes[c.root].leaf : pp_->leaf_of(c.root, pp::make_stream(s, viewer));
    if (leaf_out) *leaf_out = l;
    return pp_->current(l);
}

std::array<double, A> Model::pp_policy_at(const State& s, int seat, int32_t level) const {
    const uint8_t mask = valid_mask(s, seat);
    uint32_t b = pp_locate(s, seat, nullptr);
    if (b == kNone) return uniform_strategy(mask);
    b = pp_->ancestor(b, level, parent_);
    const auto t = pp_->tables.find(level);
    if (t == pp_->tables.end()) return normalized(buckets_[b].s, mask);
    std::array<double, A> x{};
    for (int a = 0; a < A; ++a) x[a] = t->second[b][a];
    return normalized(x, mask);
}

// Particle estimate of sample x's direction at split level `level` (see the header). The particles are drawn exactly as
// Policy-LBR's belief (same proposal, H4 multiplicity and the opponent's observed actions under the policy), then each
// legal action is rolled out once per particle with both seats on the level's average policy.
pp::Dir Model::pp_direction(const pp::Sample& x, int32_t level, uint64_t seed) const {
    pp::Dir out;
    std::mt19937_64 rng(seed);
    const int v = x.seat, street = pp::sample_street(x);
    std::vector<Card> own{pp::byte_card(x.c[0]), pp::byte_card(x.c[1])};
    if (x.c[7] != 0xff) own.push_back(pp::byte_card(x.c[7]));
    for (int k = 3; k < 7; ++k) if (x.c[k] != 0xff) own.push_back(pp::byte_card(x.c[k]));
    std::vector<Card> target_public;
    for (int k = 8; k < 12; ++k) if (x.c[k] != 0xff) target_public.push_back(pp::byte_card(x.c[k]));
    const Card own_disc = pp::byte_card(x.c[2]);
    const uint64_t known = card_mask(own) | card_mask(target_public) | 1ull << (own_disc.suit * 13 + own_disc.rank - 2);
    std::vector<Card> deck;
    for (const Card& c : fresh_deck()) if (!(known & 1ull << (c.suit * 13 + c.rank - 2))) deck.push_back(c);
    const int hc = street == 7 ? 3 : 2;
    const int own_future = std::max(0, 7 - static_cast<int>(own.size()));
    const int target_future = std::max(0, 7 - static_cast<int>(target_public.size()) - hc);
    const int sample_size = 1 + hc + own_future + target_future;
    struct Particle { Card disc, h[3], of[2], tf[2]; double logw = 0; };
    std::vector<Particle> ps;
    const int P = pp_->opt.particles;
    ps.reserve(P);
    const int limit = std::max(P * 64, P + 64);
    for (int proposal = 0; proposal < limit && static_cast<int>(ps.size()) < P; ++proposal) {
        std::array<Card, 8> draw{};
        std::array<int, 8> idx{};
        for (int i = 0; i < sample_size; ++i) {  // draw_without_replacement of PolicyLBR
            std::uniform_int_distribution<int> pick(0, static_cast<int>(deck.size()) - 1);
            int index;
            bool duplicate;
            do {
                index = pick(rng);
                duplicate = false;
                for (int prev = 0; prev < i; ++prev) if (idx[prev] == index) { duplicate = true; break; }
            } while (duplicate);
            idx[i] = index; draw[i] = deck[index];
        }
        int cur = 0;
        Particle p;
        p.disc = draw[cur++];
        for (int k = 0; k < hc; ++k) p.h[k] = draw[cur++];
        for (int k = 0; k < own_future; ++k) p.of[k] = draw[cur++];
        for (int k = 0; k < target_future; ++k) p.tf[k] = draw[cur++];
        const int ways = pp_h4_multiplicity({p.h[0], p.h[1]}, target_public.front(), p.disc);
        if (!ways) continue;
        p.logw = std::log(static_cast<double>(ways));
        ps.push_back(p);
    }
    if (ps.empty()) return out;
    // the particle's cards in the replay: own 6th/7th (if not seen), opponent 6th (if not seen) and 7th
    auto cards_of = [&](const Particle& p, Card& own6, Card& own7, Card& opp6, Card& opp7) {
        int oi = 0, ti = 0;
        own6 = street <= 5 ? p.of[oi++] : Card{};
        own7 = street <= 6 ? p.of[oi++] : Card{};
        opp6 = street <= 5 ? p.tf[ti++] : Card{};
        opp7 = street == 7 ? p.h[2] : p.tf[ti++];
    };
    std::vector<State> states;
    states.reserve(ps.size());
    uint64_t steps = 0;
    for (Particle& p : ps) {
        Card own6, own7, opp6, opp7;
        cards_of(p, own6, own7, opp6, opp7);
        states.push_back(pp::rebuild(x, p.disc, p.h[0], p.h[1], own6, own7, opp6, opp7, [&](const State& s, Action a) {
            const auto sigma = pp_policy_at(s, 1 - v, level);
            ++steps;
            p.logw += std::log(std::max(1e-12, sigma[a]));
        }));
    }
    double mx = -std::numeric_limits<double>::infinity();
    for (const Particle& p : ps) mx = std::max(mx, p.logw);
    std::vector<double> w(ps.size());
    double W = 0, W2 = 0;
    for (size_t i = 0; i < ps.size(); ++i) { w[i] = std::exp(ps[i].logw - mx); W += w[i]; W2 += w[i] * w[i]; }
    const State& s0 = states.front();
    out.mask = valid_mask(s0, v);
    const auto own_sigma = pp_policy_at(s0, v, level);
    ++steps;
    std::vector<std::array<double, A>> u(ps.size());
    for (size_t i = 0; i < ps.size(); ++i)
        for (int a = 0; a < A; ++a) {
            if (!(out.mask & (1u << a))) continue;
            State c = states[i];
            step(c, v, static_cast<Action>(a));
            while (!c.terminal) {
                const int actor = c.actor;
                const auto sigma = pp_policy_at(c, actor, level);
                ++steps;
                step(c, actor, sample_action(sigma, valid_mask(c, actor), rng));
            }
            u[i][a] = terminal_net_search(c, v);
        }
    out.neff = W2 > 0 ? W * W / W2 : 0;
    for (int a = 0; a < A; ++a) {
        if (!(out.mask & (1u << a))) continue;
        std::vector<double> d(ps.size());
        double m = 0;
        for (size_t i = 0; i < ps.size(); ++i) {
            double node = 0;
            for (int b = 0; b < A; ++b) if (out.mask & (1u << b)) node += own_sigma[b] * u[i][b];
            d[i] = u[i][a] - node;
            m += w[i] * d[i];
        }
        m /= W;
        double var = 0;
        for (size_t i = 0; i < ps.size(); ++i) var += w[i] * (d[i] - m) * (d[i] - m);
        var /= W;
        out.r[a] = m;
        out.se[a] = out.neff > 0 ? std::sqrt(var / out.neff) : std::numeric_limits<double>::infinity();
    }
    out.label = pp_label(out.r, out.se, out.mask);
    out.steps = steps;
    return out;
}

int Model::pp_label(const std::array<double, A>& r, const std::array<double, A>& se, uint8_t mask) const {
    int best = -1;
    for (int a = 0; a < A; ++a)
        if ((mask & (1u << a)) && r[a] > pp_->opt.z * se[a] && (best < 0 || r[a] > r[best])) best = a;
    return best;
}

// directions of many samples at one level, on --threads threads (each sample has its own generator: same bits for any
// thread count)
std::vector<pp::Dir> Model::pp_directions(const std::vector<pp::Sample>& xs, int32_t level, uint64_t salt) const {
    std::vector<pp::Dir> out(xs.size());
    const int T = std::max(1, std::min<int>(pp_->opt.threads, static_cast<int>(xs.size())));
    auto work = [&](int t) {
        for (size_t i = t; i < xs.size(); i += T) out[i] = pp_direction(xs[i], level, pp_mix(pp_mix(pp_seed_, salt), i));
    };
    if (T == 1) work(0);
    else {
        std::vector<std::thread> th;
        for (int t = 1; t < T; ++t) th.emplace_back(work, t);
        work(0);
        for (auto& x : th) x.join();
    }
    return out;
}

// a sample's information state with arbitrary unseen cards (for its observation stream)
State Model::pp_view_state(const pp::Sample& x) {
    uint64_t known = 0;
    for (uint8_t b : x.c) if (b != 0xff) known |= 1ull << b;
    std::vector<Card> spare;
    for (int b = 0; b < 52 && spare.size() < 7; ++b) if (!(known & 1ull << b)) spare.push_back(pp::byte_card(static_cast<uint8_t>(b)));
    return pp::rebuild(x, spare[0], spare[1], spare[2], spare[3], spare[4], spare[5], spare[6], [](const State&, Action) {});
}

uint32_t Model::pp_route_sum(uint32_t b, const std::vector<const pp::Dir*>& ds) const {
    std::array<double, A> r{}, se2{};
    uint8_t mask = 0;
    for (const pp::Dir* d : ds) {
        mask = d->mask;
        for (int a = 0; a < A; ++a) { r[a] += d->r[a]; se2[a] += d->se[a] * d->se[a]; }
    }
    std::array<double, A> se{};
    for (int a = 0; a < A; ++a) se[a] = std::sqrt(se2[a]);
    return pp_->kid_of(b, ds.empty() ? -1 : pp_label(r, se, mask));
}

// --leaf-start root: the bucket a new leaf with these samples reaches from its cell's first bucket, following every
// split event with the samples' directions at that event's level (`now`: the directions at the current level)
uint32_t Model::pp_route_root(const pp::Cell& cell, const std::vector<const pp::Sample*>& xs, const std::vector<const pp::Dir*>& now,
                              int32_t cur_level, uint64_t salt, uint64_t& steps) {
    uint32_t b = cell.bucket;
    while (true) {
        const auto it = pp_->splits.find(b);
        if (it == pp_->splits.end()) return b;
        const int32_t lv = it->second.level;
        if (lv == cur_level) { b = pp_route_sum(b, now); continue; }
        std::vector<pp::Sample> v;
        for (const pp::Sample* x : xs) v.push_back(*x);
        const auto ds = pp_directions(v, lv, pp_mix(salt, static_cast<uint64_t>(lv) * 7919 + b));
        std::vector<const pp::Dir*> p;
        for (const auto& d : ds) { p.push_back(&d); steps += d.steps; }
        b = pp_route_sum(b, p);
    }
}

// The training lookup: creates a start cell on its first visit. --pp-assign fill: a traverser visit to a pending leaf is
// routed by its own direction and kept as one of the leaf's m samples; the m-th fixes the pointer.
uint32_t Model::pp_train_bucket(const State& s, int v, bool traverser) {
    const uint16_t ctx = static_cast<uint16_t>(s.street << 8 | valid_mask(s, v));
    auto [it, inserted] = pp_->cell_of.try_emplace(ctx, static_cast<int32_t>(pp_->cells.size()));
    if (inserted) {
        pp::Cell c;
        c.ctx = ctx;
        c.bucket = new_bucket();
        c.root = static_cast<int32_t>(pp_->nodes.size());
        pp::TNode n; n.leaf = static_cast<int32_t>(pp_->leaves.size());
        pp_->nodes.push_back(n);
        pp::Leaf L; L.node = c.root; L.ctx = ctx; L.ptr = c.bucket; L.born = pp_root_now_;
        pp_->leaves.push_back(L);
        pp_->cells.push_back(c);
    }
    const pp::Cell& c = pp_->cells[it->second];
    const int32_t l = pp_->nodes[c.root].leaf >= 0 ? pp_->nodes[c.root].leaf : pp_->leaf_of(c.root, pp::make_stream(s, v));
    const pp::Leaf& L = pp_->leaves[l];
    if (!L.pending) return L.ptr;
    if (!(pp_->opt.fill && traverser)) return L.prov;
    const pp::Sample x = pp::make_sample(s, v);
    const pp::Dir d = pp_direction(x, L.start_level, pp_mix(pp_mix(pp_seed_, 0xf111), ++pp_->fill_estimates));
    pp_fill_steps_ += d.steps;
    const uint32_t b = pp_->kid_of(L.start_b, d.label);
    pp::FillRow row;
    row.x = x; row.label = static_cast<int8_t>(d.label);
    for (int a = 0; a < A; ++a) { row.r[a] = static_cast<float>(d.r[a]); row.se[a] = static_cast<float>(std::min(d.se[a], 1e30)); }
    pp_->leaves[l].fill.push_back(row);
    if (static_cast<int>(pp_->leaves[l].fill.size()) >= pp_->opt.m) pp_finalize_fill(l, false);
    return b;
}

// Fix the pointer of pending leaf l from its rows (the first m in sample order), and grow the prior tree from it: a leaf
// whose rows pick two or more posterior children (signal) is split by the best test on its rows, its two new leaves get
// their own pointers the same way, and so on while the leaf count stays under the cap. Returns the prior splits made.
int Model::pp_resolve(int32_t l, const pp::Rows& rows, std::vector<int>& idx, int32_t cur_level, uint64_t salt, size_t cap,
                      size_t ratio_target, uint64_t& steps, std::ostream* plog, uint64_t epoch) {
    struct Item { int32_t leaf; std::vector<int> idx; double gain; bool signal; pp::Candidate cand; };
    const uint32_t start_b = pp_->leaves[l].start_b;
    auto pointer_of = [&](int32_t leaf, const std::vector<int>& id, bool fresh_leaf) {
        std::vector<const pp::Dir*> ds;
        std::vector<const pp::Sample*> xs;
        for (size_t k = 0; k < id.size() && static_cast<int>(k) < pp_->opt.m; ++k) { ds.push_back(&rows.d[id[k]]); xs.push_back(&rows.x[id[k]]); }
        if (fresh_leaf && pp_->opt.root_start) {
            const auto& cell = pp_->cells[pp_->cell_of.at(pp_->leaves[leaf].ctx)];
            return pp_route_root(cell, xs, ds, cur_level, pp_mix(salt, static_cast<uint64_t>(leaf)), steps);
        }
        return pp_route_sum(start_b, ds);
    };
    auto kid_label = [&](int i) { return pp_->kid_of(start_b, rows.d[i].label); };
    std::unordered_map<int, pp::Stream> streams;  // row index -> its observation stream
    auto stream_of = [&](int i) -> const pp::Stream& {
        auto it = streams.find(i);
        if (it == streams.end()) it = streams.emplace(i, pp::make_stream(pp_view_state(rows.x[i]), rows.x[i].seat)).first;
        return it->second;
    };
    auto evaluate = [&](int32_t leaf, std::vector<int> id) {
        Item it{leaf, std::move(id), 0, false, {}};
        std::set<uint32_t> kids;
        for (size_t k = 0; k < it.idx.size() && static_cast<int>(k) < pp_->opt.m; ++k) kids.insert(kid_label(it.idx[k]));
        it.signal = kids.size() >= 2;
        if (it.idx.size() >= 2) {
            std::vector<pp::Row> rr(it.idx.size());
            for (size_t k = 0; k < it.idx.size(); ++k) {
                const int i = it.idx[k];
                rr[k].s = stream_of(i);
                for (int a = 0; a < A; ++a) rr[k].x[a] = (rows.d[i].mask & (1u << a)) ? static_cast<float>(rows.d[i].r[a]) : 0.f;
                rr[k].lab = kid_label(i);
            }
            std::vector<char> half(rr.size(), 0);
            if (pp_->opt.crossfit) {
                std::mt19937_64 hr(pp_mix(salt, 0xc0ffee + static_cast<uint64_t>(leaf)));
                for (auto& h : half) h = static_cast<char>(hr() & 1);
            }
            it.cand = pp::best_split(rr, half, pp_->opt);
            it.gain = it.cand.ok ? it.cand.gainB : 0;
        }
        return it;
    };
    auto cmp = [](const Item& a, const Item& b) {
        if (a.signal != b.signal) return !a.signal;
        if (a.gain != b.gain) return a.gain < b.gain;
        return a.leaf > b.leaf;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> q(cmp);
    // the pending leaf itself
    {
        pp::Leaf& L = pp_->leaves[l];
        L.ptr = pointer_of(l, idx, false);
        L.pending = false;
        L.fill.clear();
        Item it = evaluate(l, idx);
        if (it.signal) ++pp_->signals;
        q.push(std::move(it));
    }
    int made = 0;
    while (!q.empty()) {
        Item it = q.top(); q.pop();
        if (!it.cand.ok) continue;
        const size_t alive = pp_->live_leaves();
        if (alive >= cap) break;
        if (!it.signal && alive >= ratio_target) continue;
        // split leaf it.leaf by it.cand
        const int32_t node = pp_->leaves[it.leaf].node;
        const uint16_t ctx = pp_->leaves[it.leaf].ctx, depth = pp_->leaves[it.leaf].depth;
        pp_->nodes[node].d = it.cand.d; pp_->nodes[node].thr = static_cast<float>(it.cand.thr); pp_->nodes[node].leaf = -1;
        std::array<std::vector<int>, 2> part;
        for (int i : it.idx) part[pp::feat(stream_of(i), it.cand.d) <= it.cand.thr ? 0 : 1].push_back(i);
        std::array<int32_t, 2> nl{};
        for (int side = 0; side < 2; ++side) {
            pp::TNode n; n.leaf = static_cast<int32_t>(pp_->leaves.size());
            const int32_t nn = static_cast<int32_t>(pp_->nodes.size());
            pp_->nodes.push_back(n);
            (side == 0 ? pp_->nodes[node].left : pp_->nodes[node].right) = nn;
            pp::Leaf L; L.node = nn; L.ctx = ctx; L.depth = static_cast<uint16_t>(depth + 1); L.born = pp_root_now_;
            L.start_b = start_b; L.start_level = pp_->leaves[it.leaf].start_level;
            pp_->leaves.push_back(L);
            nl[side] = n.leaf;
        }
        pp_->leaves[it.leaf].node = -1;
        ++made; ++pp_->prior_splits;
        for (int side = 0; side < 2; ++side) {
            pp_->leaves[nl[side]].ptr = part[side].empty() ? pp_->leaves[it.leaf].ptr : pointer_of(nl[side], part[side], true);
            q.push(evaluate(nl[side], part[side]));
        }
        if (plog) *plog << epoch << ',' << pp_root_now_ << ',' << it.leaf << ',' << (ctx >> 8) << ',' << (ctx & 0xff) << ',' << depth << ','
                        << pp::desc_name(it.cand.d) << ',' << it.cand.thr << ',' << it.cand.gainA << ',' << it.cand.gainB << ','
                        << it.idx.size() << ',' << it.signal << ',' << nl[0] << ',' << nl[1] << ',' << pp_->leaves[nl[0]].ptr << ','
                        << pp_->leaves[nl[1]].ptr << '\n';
    }
    return made;
}

// --pp-assign fill: leaf l has its samples (m, or fewer at the next judgment); fix it from them
void Model::pp_finalize_fill(int32_t l, bool at_judgment) {
    pp::Rows rows;
    for (const auto& f : pp_->leaves[l].fill) {
        rows.x.push_back(f.x);
        pp::Dir d;
        d.mask = valid_mask(pp_view_state(f.x), f.x.seat);
        for (int a = 0; a < A; ++a) { d.r[a] = f.r[a]; d.se[a] = f.se[a]; }
        d.label = f.label;
        rows.d.push_back(d);
    }
    if (rows.x.empty()) { pp_->leaves[l].ptr = pp_->leaves[l].prov; pp_->leaves[l].pending = false; return; }
    std::vector<int> idx(rows.x.size());
    std::iota(idx.begin(), idx.end(), 0);
    const size_t cap = static_cast<size_t>(2 * std::max<int32_t>(1, pp_->leaves_at_judgment));
    uint64_t steps = 0;
    std::ostringstream log;
    (void)at_judgment;  // fill mode splits signalled leaves only (the spec's immediate prior split), at any time
    pp_resolve(l, rows, idx, pp_->leaves[l].start_level, pp_mix(pp_seed_, 0xf12 + pp_->fill_estimates), cap, 0, steps, &log,
               pp_epoch_);
    pp_fill_steps_ += steps;
    std::istringstream in(log.str());
    std::string line;
    while (std::getline(in, line)) pp_->split_rows.push_back(line);
}

bool Model::pp_online_due(uint64_t t) const {
    const int i = std::min<int>(pp_->level, 62);
    if (t <= pp_->last_split_root) return false;
    return (t - pp_->last_split_root) % (1ull << i) == 0;
}

double Model::traverse_pp(State state, int traverser, double weight, std::mt19937_64& rng) {
    if (state.terminal) return terminal_net_search(state, traverser);
    const int actor = state.actor;
    const uint8_t mask = valid_mask(state, actor);
    ++pp_->epoch_nodes;
    const uint32_t b = pp_train_bucket(state, actor, actor == traverser);
    const auto sigma = rm_plus(buckets_[b].q, mask);
    if (actor != traverser) {
        for (int a = 0; a < A; ++a) if (mask & (1u << a)) buckets_[b].s[a] += weight * sigma[a];
        step(state, actor, sample_action(sigma, mask, rng));
        return traverse_pp(std::move(state), traverser, weight, rng);
    }
    std::array<double, A> v{};
    double node = 0;
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State child = state;
        step(child, actor, static_cast<Action>(a));
        v[a] = traverse_pp(std::move(child), traverser, weight, rng);
        node += sigma[a] * v[a];
    }
    Bucket& bucket = buckets_[b];
    if (!bucket.dirty) { bucket.dirty = true; touched_.push_back(b); }
    double best = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        const double r = v[a] - node;
        bucket.pending[a] += r;
        best = std::max(best, r);
    }
    traversal_gap_ += best;
    return node;
}

// a pass of external-sampling traversals with both seats on the live average policy; visit(state, actor, bucket, leaf)
// at every traverser decision node
template <class V>
void Model::pp_pass(State s, int traverser, std::mt19937_64& rng, V& visit) const {
    if (s.terminal) return;
    const int actor = s.actor;
    const uint8_t mask = valid_mask(s, actor);
    int32_t leaf = -1;
    const uint32_t b = pp_locate(s, actor, &leaf);
    const auto sigma = bucket_policy(b, mask, true);
    if (actor != traverser) {
        step(s, actor, sample_action(sigma, mask, rng));
        pp_pass(std::move(s), traverser, rng, visit);
        return;
    }
    visit(s, actor, b, leaf);
    for (int a = 0; a < A; ++a) {
        if (!(mask & (1u << a))) continue;
        State c = s;
        step(c, actor, static_cast<Action>(a));
        pp_pass(std::move(c), traverser, rng, visit);
    }
}

// Reservoirs of distinct information states per key from n pass roots (keys of shard `shard` of `shards` only)
template <class KeyOf>
std::map<uint64_t, pp::Reservoir> Model::pp_collect(uint64_t n, uint64_t seed, size_t cap, int shard, int shards, KeyOf key_of) const {
    std::map<uint64_t, pp::Reservoir> res;
    std::mt19937_64 pick(pp_mix(seed, 0x7e5));
    auto visit = [&](const State& s, int actor, uint32_t b, int32_t leaf) {
        uint64_t key;
        if (!key_of(b, leaf, key)) return;
        if (shards > 1 && pp_mix(key, 0x5a4d) % static_cast<uint64_t>(shards) != static_cast<uint64_t>(shard)) return;
        pp::Reservoir& r = res[key];
        ++r.visits;
        const pp::Sample x = pp::make_sample(s, actor);
        const uint64_t h = pp::sample_hash(x);
        if (std::find(r.h.begin(), r.h.end(), h) != r.h.end()) return;
        if (r.x.size() < cap) { r.x.push_back(x); r.h.push_back(h); return; }
        const uint64_t j = std::uniform_int_distribution<uint64_t>(0, r.visits - 1)(pick);
        if (j < cap) { r.x[j] = x; r.h[j] = h; }
    };
    for (uint64_t i = 0; i < n; ++i) {
        std::mt19937_64 rng;
        const State root = label_root(seed, i, rng);
        for (int p = 0; p < 2; ++p) pp_pass(root, p, rng, visit);
    }
    return res;
}

// One judgment (PP_STUD7_DESIGN_KO.md section 6). Returns the posterior splits; writes one row of pp_judge.csv.
int Model::pp_judge(uint64_t seed, uint64_t epoch, uint64_t roots, uint64_t epoch_len, uint64_t fill_steps, std::ostream& jlog,
                    std::ostream& plog) {
    pp::Data& D = *pp_;
    const pp::Opts& O = D.opt;
    pp_root_now_ = roots; pp_epoch_ = epoch;
    const auto t_start = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    for (const auto& r : D.split_rows) plog << r << '\n';  // fill mode: prior splits made during training
    D.split_rows.clear();
    // fill mode: leaves still pending from the last judgment are fixed from the samples they have (or the none child)
    uint64_t fill_left = 0;
    for (int32_t l = 0; l < static_cast<int32_t>(D.leaves.size()); ++l)
        if (D.leaves[l].node >= 0 && D.leaves[l].pending) { pp_finalize_fill(l, true); ++fill_left; }
    for (const auto& r : D.split_rows) plog << r << '\n';
    D.split_rows.clear();
    if (!O.root_start) D.tables.clear();
    const int32_t e = D.level;
    const auto live0 = live_buckets();
    const size_t live_before = live0[0] + live0[1] + live0[2], leaves0 = D.live_leaves();
    D.leaves_at_judgment = static_cast<int32_t>(leaves0);
    const double budget_total = std::max(O.budget_min_steps, O.budget * static_cast<double>(D.epoch_nodes));
    const double budget_stage = budget_total / 2;
    const uint64_t n_pass = std::max<uint64_t>(1000, static_cast<uint64_t>(O.pass_frac * static_cast<double>(epoch_len)));
    const double sample_bytes = sizeof(pp::Sample) + 8 + 16;
    // ---- stage 1-2: pass S and the posterior splits ----
    const int shards_s = std::max(1, static_cast<int>(std::ceil(live_before * O.n0 * sample_bytes / (O.judge_mem_mb * 1e6))));
    uint64_t judged = 0, samples = 0, steps2 = 0, splits = 0, children = 0, n_used = O.n0;
    double t_pass = 0, t_est = 0;
    std::vector<uint32_t> split_buckets;
    for (int sh = 0; sh < shards_s; ++sh) {
        const double p0 = secs();
        auto res = pp_collect(n_pass, pp_mix(seed * 1000003ull + epoch, 0x51), static_cast<size_t>(O.n0), sh, shards_s,
                              [&](uint32_t b, int32_t, uint64_t& key) { if (b == kNone) return false; key = b; return true; });
        t_pass += secs() - p0;
        // plan: n samples per bucket within this shard's share of the budget (deterministic: from step counts)
        std::vector<std::pair<uint64_t, uint32_t>> order;  // (-visits, bucket)
        uint64_t full = 0;
        for (const auto& [k, r] : res) { order.push_back({~r.visits, static_cast<uint32_t>(k)}); full += std::min<uint64_t>(O.n0, r.x.size()); }
        std::sort(order.begin(), order.end());
        const double plan = budget_stage / shards_s / std::max(1.0, D.steps_per_sample);
        int n = O.n0;
        if (full > plan) n = std::max(O.n_min, static_cast<int>(std::floor(O.n0 * plan / std::max<double>(1, full))));
        n_used = std::min<uint64_t>(n_used, n);
        std::vector<pp::Sample> xs;
        std::vector<std::pair<uint32_t, size_t>> span;  // bucket, first sample index
        double used = 0;
        for (const auto& [nv, b] : order) {
            const auto& r = res[b];
            const size_t k = std::min<size_t>(n, r.x.size());
            if (k < 2) continue;
            if (used + k > plan && !span.empty()) break;
            span.push_back({b, xs.size()});
            for (size_t i = 0; i < k; ++i) xs.push_back(r.x[i]);
            used += k;
        }
        res.clear();
        const double e0 = secs();
        const auto ds = pp_directions(xs, e, pp_mix(pp_mix(seed, epoch), 0x200 + sh));
        t_est += secs() - e0;
        samples += xs.size();
        for (const auto& d : ds) steps2 += d.steps;
        for (size_t j = 0; j < span.size(); ++j) {
            const uint32_t b = span[j].first;
            const size_t lo = span[j].second, hi = j + 1 < span.size() ? span[j + 1].second : xs.size();
            ++judged;
            std::set<int> labels;
            for (size_t i = lo; i < hi; ++i) labels.insert(ds[i].label);
            if (labels.size() < 2) continue;
            pp::Split sp;
            sp.level = e;
            sp.kid.fill(kNone);
            for (int lab : labels) if (lab >= 0) sp.kid[lab] = new_bucket(b);
            sp.kid[A] = new_bucket(b);
            children += labels.count(-1) ? labels.size() : labels.size() + 1;
            D.splits[b] = sp;
            split_buckets.push_back(b);
            ++splits;
        }
    }
    if (splits) {
        ++D.level;
        D.last_split_root = roots;
    }
    // ---- stage 3: the leaves pointing to a split bucket become pending (provisional pointer: the none child) ----
    std::vector<int32_t> pending;
    for (int32_t l = 0; l < static_cast<int32_t>(D.leaves.size()); ++l) {
        pp::Leaf& L = D.leaves[l];
        if (L.node < 0 || !D.splits.count(L.ptr) || D.splits.at(L.ptr).level != e || !splits) continue;
        L.pending = true; L.start_b = L.ptr; L.start_level = e; L.prov = D.kid_of(L.ptr, -1);
        pending.push_back(l);
    }
    if ((O.root_start || O.fill) && splits) {  // keep this level's average tables (root start: for good; fill: until fixed)
        auto& t = D.tables[e];
        t.resize(buckets_.size());
        for (size_t b = 0; b < buckets_.size(); ++b) for (int a = 0; a < A; ++a) t[b][a] = static_cast<float>(buckets_[b].s[a]);
    }
    // ---- stages 4-6 (assign now): pass R, pointers, prior splits ----
    uint64_t rows_total = 0, resolved_rows = 0, resolved_prov = 0, steps4 = 0, prior_before = D.prior_splits, signals_before = D.signals;
    int shards_r = 0;
    if (!O.fill && !pending.empty()) {
        std::set<int32_t> pend(pending.begin(), pending.end());
        shards_r = std::max(1, static_cast<int>(std::ceil(pending.size() * O.rho * sample_bytes / (O.judge_mem_mb * 1e6))));
        const auto live1 = live_buckets();
        const size_t ratio_target = static_cast<size_t>(O.ratio * static_cast<double>(live1[0] + live1[1] + live1[2]));
        const size_t cap = 2 * std::max<size_t>(1, leaves0);
        for (int sh = 0; sh < shards_r; ++sh) {
            const double p0 = secs();
            auto res = pp_collect(n_pass, pp_mix(seed * 1000003ull + epoch, 0x52), static_cast<size_t>(O.rho), sh, shards_r,
                                  [&](uint32_t, int32_t leaf, uint64_t& key) {
                                      if (leaf < 0 || !pend.count(leaf)) return false;
                                      key = static_cast<uint64_t>(leaf); return true;
                                  });
            t_pass += secs() - p0;
            uint64_t full = 0;
            std::vector<std::pair<uint64_t, int32_t>> order;
            for (const auto& [k, r] : res) { order.push_back({~r.visits, static_cast<int32_t>(k)}); full += std::min<uint64_t>(O.rho, r.x.size()); }
            std::sort(order.begin(), order.end());
            const double plan = budget_stage / shards_r / std::max(1.0, D.steps_per_sample);
            int rho = O.rho;
            if (full > plan) rho = std::max(O.m, static_cast<int>(std::floor(O.rho * plan / std::max<double>(1, full))));
            pp::Rows rows;
            std::vector<std::pair<int32_t, std::vector<int>>> leaf_rows;
            double used = 0;
            for (const auto& [nv, l] : order) {
                const auto& r = res[static_cast<uint64_t>(l)];
                const size_t k = std::min<size_t>(rho, r.x.size());
                if (used + k > plan && !leaf_rows.empty()) break;
                std::vector<int> id;
                for (size_t i = 0; i < k; ++i) { id.push_back(static_cast<int>(rows.x.size())); rows.x.push_back(r.x[i]); }
                leaf_rows.push_back({l, std::move(id)});
                used += k;
            }
            res.clear();
            const double e0 = secs();
            rows.d = pp_directions(rows.x, e, pp_mix(pp_mix(seed, epoch), 0x400 + sh));
            t_est += secs() - e0;
            rows_total += rows.x.size();
            for (const auto& d : rows.d) steps4 += d.steps;
            for (auto& [l, id] : leaf_rows) {
                pp_resolve(l, rows, id, e, pp_mix(pp_mix(seed, epoch), 0x600 + static_cast<uint64_t>(l)), cap, ratio_target, steps4, &plog, epoch);
                ++resolved_rows;
            }
        }
        for (int32_t l : pending) if (D.leaves[l].node >= 0 && D.leaves[l].pending) {
            D.leaves[l].ptr = D.leaves[l].prov; D.leaves[l].pending = false; ++resolved_prov;
        }
    }
    const uint64_t estimates = samples + rows_total;
    if (estimates) D.steps_per_sample = static_cast<double>(steps2 + steps4) / static_cast<double>(estimates);
    D.epoch_nodes = 0;
    const auto live2 = live_buckets();
    const auto mb = table_megabytes();
    jlog << epoch << ',' << roots << ',' << e << ',' << live_before << ',' << judged << ',' << n_used << ',' << samples << ','
         << splits << ',' << children << ',' << pending.size() << ',' << rows_total << ',' << resolved_rows << ',' << resolved_prov << ','
         << fill_left << ',' << (D.signals - signals_before) << ',' << (D.prior_splits - prior_before) << ',' << D.live_leaves() << ','
         << live2[0] + live2[1] + live2[2] << ',' << steps2 << ',' << steps4 << ',' << budget_total << ',' << D.steps_per_sample << ','
         << n_pass << ',' << shards_s << ',' << shards_r << ',' << t_pass << ',' << t_est << ',' << secs() << ','
         << mb[0] << ',' << mb[1] << ',' << mb[2] << ',' << memory_mb(1) << ',' << memory_mb(0) << ',' << fill_steps << ','
         << D.frozen << '\n';
    return static_cast<int>(splits);
}

void Model::pp_save(std::ofstream& out, bool full) const {
    const pp::Data& D = *pp_;
    put(out, D.level);
    put(out, static_cast<uint64_t>(D.cells.size()));
    for (const auto& c : D.cells) { put(out, c.ctx); put(out, c.root); put(out, c.bucket); }
    put(out, static_cast<uint64_t>(D.nodes.size()));
    for (const auto& n : D.nodes) { put(out, n.d); put(out, n.thr); put(out, n.left); put(out, n.right); put(out, n.leaf); }
    put(out, static_cast<uint64_t>(D.leaves.size()));
    for (const auto& L : D.leaves) {
        put(out, L.node); put(out, L.ctx); put(out, L.depth); put(out, static_cast<uint8_t>(L.pending)); put(out, L.ptr); put(out, L.prov);
        put(out, L.start_b); put(out, L.start_level); put(out, L.born);
        if (full) { put(out, static_cast<uint32_t>(L.fill.size())); for (const auto& f : L.fill) put(out, f); }
    }
    std::vector<std::pair<uint32_t, pp::Split>> sp(D.splits.begin(), D.splits.end());
    std::sort(sp.begin(), sp.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    put(out, static_cast<uint64_t>(sp.size()));
    for (const auto& [b, s] : sp) { put(out, b); put(out, s); }
    put(out, static_cast<uint64_t>(buckets_.size()));
    for (size_t b = 0; b < buckets_.size(); ++b) { put(out, buckets_[b].q); put(out, buckets_[b].s); put(out, parent_[b]); put(out, D.born[b]); }
    if (!full) return;
    put(out, static_cast<uint64_t>(D.tables.size()));
    for (const auto& [lv, t] : D.tables) {
        put(out, lv); put(out, static_cast<uint64_t>(t.size()));
        out.write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(t[0])));
    }
    put(out, D.epoch_nodes); put(out, D.steps_per_sample); put(out, D.last_split_root); put(out, D.fill_estimates);
    put(out, D.prior_splits); put(out, D.signals); put(out, static_cast<uint8_t>(D.frozen)); put(out, D.leaves_at_judgment);
    put(out, static_cast<uint64_t>(D.split_rows.size()));
    for (const auto& r : D.split_rows) put_string(out, r);
}

void Model::pp_load(std::ifstream& in, bool full) {
    if (!pp_) pp_ = std::make_unique<pp::Data>();
    pp::Data& D = *pp_;
    get(in, D.level);
    uint64_t n; get(in, n);
    D.cells.assign(n, {}); D.cell_of.clear();
    for (uint64_t i = 0; i < n; ++i) { auto& c = D.cells[i]; get(in, c.ctx); get(in, c.root); get(in, c.bucket); D.cell_of[c.ctx] = static_cast<int32_t>(i); }
    get(in, n);
    D.nodes.assign(n, {});
    for (auto& x : D.nodes) { get(in, x.d); get(in, x.thr); get(in, x.left); get(in, x.right); get(in, x.leaf); }
    get(in, n);
    D.leaves.assign(n, {});
    for (auto& L : D.leaves) {
        uint8_t p;
        get(in, L.node); get(in, L.ctx); get(in, L.depth); get(in, p); L.pending = p; get(in, L.ptr); get(in, L.prov);
        get(in, L.start_b); get(in, L.start_level); get(in, L.born);
        if (full) { uint32_t m; get(in, m); L.fill.assign(m, {}); for (auto& f : L.fill) get(in, f); }
    }
    get(in, n);
    D.splits.clear();
    for (uint64_t i = 0; i < n; ++i) { uint32_t b; pp::Split s; get(in, b); get(in, s); D.splits[b] = s; }
    get(in, n);
    buckets_.assign(n, Bucket{}); parent_.assign(n, kNone); D.born.assign(n, 0);
    for (uint64_t b = 0; b < n; ++b) { get(in, buckets_[b].q); get(in, buckets_[b].s); get(in, parent_[b]); get(in, D.born[b]); }
    if (!full) return;
    get(in, n);
    D.tables.clear();
    for (uint64_t i = 0; i < n; ++i) {
        int32_t lv; uint64_t m; get(in, lv); get(in, m);
        auto& t = D.tables[lv];
        t.resize(m);
        in.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(m * sizeof(t[0])));
    }
    get(in, D.epoch_nodes); get(in, D.steps_per_sample); get(in, D.last_split_root); get(in, D.fill_estimates);
    get(in, D.prior_splits); get(in, D.signals);
    uint8_t f; get(in, f); D.frozen = f;
    get(in, D.leaves_at_judgment);
    get(in, n);
    D.split_rows.clear();
    for (uint64_t i = 0; i < n; ++i) D.split_rows.push_back(get_string(in));
}

// ---- pp-test: tests T1, T2, T3 and T7 of PP_STUD7_DESIGN_KO.md section 10 ----
//   aswiss_stud pp-test --snapshot FILE [--states 200000] [--seed 7] [--directions 20]
// T1  (seat, observation stream, legal mask) separates the information states of random decisions exactly up to suit
//     symmetry: equal streams <=> equal suit-canonical engine information (own cards, discard, reveal, up cards, bets)
// T2  a random suit permutation of a state leaves its stream unchanged
// T3  a compact sample replays to the same state (cards, pot, bets, actor) with the true unseen cards, and its stream
//     from arbitrary unseen cards equals the state's own stream
// T7  (a) the particle weights equal Policy-LBR's belief (same seed: same effective particle count); (b) the estimate
//     settles as the particles grow (P = 64, 256, 1024, 4096) on --directions sampled states (needs --snapshot)
int pp_test(int argc, char** argv) {
    auto o = parse(argc, argv, {{"--snapshot", ""}, {"--states", "200000"}, {"--seed", "7"}, {"--directions", "20"}});
    std::mt19937_64 rng(std::stoull(o["--seed"]));
    const uint64_t n_states = std::stoull(o["--states"]);
    // random decision states: random deals, uniformly random legal actions, a random stop
    std::vector<State> states;
    while (states.size() < n_states) {
        auto deck = fresh_deck();
        std::shuffle(deck.begin(), deck.end(), rng);
        State s = sample_fifth_street_root(deck, kAnte, kEffectiveStackAnte);
        while (!s.terminal) {
            states.push_back(s);
            const auto acts = actions_from_mask(valid_mask(s, s.actor));
            step(s, s.actor, acts[std::uniform_int_distribution<size_t>(0, acts.size() - 1)(rng)]);
        }
    }
    auto engine_key = [](const State& s, int v) {  // suit-canonical engine information of the viewer
        std::string best;
        for (const auto& perm : pp::suit_perms()) {
            auto c = [&](const Card& x) { return static_cast<char>('A' + 4 * (x.rank - 2) + perm[x.suit]); };
            const Player& me = s.players[v];
            const Player& op = s.players[1 - v];
            std::string k;
            std::array<char, 2> h{c(me.hidden[0]), c(me.hidden[1])};
            std::sort(h.begin(), h.end());
            k += h[0]; k += h[1]; k += c(me.discarded); k += '|';
            for (const Card& x : me.shown) k += c(x);
            k += '|';
            if (me.hidden.size() > 2) k += c(me.hidden[2]);
            k += '|';
            for (const Card& x : op.shown) k += c(x);
            k += '|';
            for (const Event& e : s.history) { k += static_cast<char>('0' + e.street); k += static_cast<char>('0' + (e.actor == v)); k += static_cast<char>('a' + e.action); }
            k += '|'; k += static_cast<char>('0' + v); k += static_cast<char>(valid_mask(s, v));
            if (best.empty() || k < best) best = k;
        }
        return best;
    };
    auto stream_key = [](const State& s, int v) {
        const pp::Stream st = pp::make_stream(s, v);
        std::string k(reinterpret_cast<const char*>(st.o.data()), st.n * sizeof(uint32_t));
        k += static_cast<char>(v); k += static_cast<char>(valid_mask(s, v));
        return k;
    };
    std::unordered_map<std::string, std::string> s2e, e2s;
    uint64_t t1_bad = 0, t2_bad = 0, t3_bad = 0, t3_stream_bad = 0;
    for (const State& s : states) {
        const int v = s.actor;
        const std::string sk = stream_key(s, v), ek = engine_key(s, v);
        auto [a, fa] = s2e.try_emplace(sk, ek);
        auto [b, fb] = e2s.try_emplace(ek, sk);
        t1_bad += a->second != ek || b->second != sk;
        // T2: permute suits
        State p = s;
        std::array<int, 4> perm{0, 1, 2, 3};
        std::shuffle(perm.begin(), perm.end(), rng);
        auto pc = [&](Card& x) { x.suit = static_cast<uint8_t>(perm[x.suit]); };
        for (Player& pl : p.players) { for (Card& x : pl.hidden) pc(x); for (Card& x : pl.shown) pc(x); pc(pl.discarded); }
        for (Card& x : p.simulation_deck) pc(x);
        t2_bad += stream_key(p, v) != sk;
        // T3: the sample's replay with the true unseen cards, and its stream with arbitrary ones
        const pp::Sample x = pp::make_sample(s, v);
        const Player& op = s.players[1 - v];
        const Player& me = s.players[v];
        const Card none{};
        const State r = pp::rebuild(x, op.discarded, op.hidden[0], op.hidden[1], me.shown.size() > 3 ? none : none, none,
                                    op.shown.size() > 3 ? op.shown[3] : none, op.hidden.size() > 2 ? op.hidden[2] : none, [](const State&, Action) {});
        bool same = r.pot == s.pot && r.street == s.street && r.actor == s.actor && r.history.size() == s.history.size() &&
                    r.highest_bet == s.highest_bet && r.raise_count == s.raise_count;
        for (int q = 0; q < 2 && same; ++q) {
            same = r.players[q].shown == s.players[q].shown && r.players[q].invested == s.players[q].invested &&
                   r.players[q].round_bet == s.players[q].round_bet && r.players[q].discarded == s.players[q].discarded &&
                   r.players[q].hidden.size() == s.players[q].hidden.size();
            for (size_t i = 0; same && i < r.players[q].hidden.size(); ++i) same = r.players[q].hidden[i] == s.players[q].hidden[i];
        }
        t3_bad += !same;
        t3_stream_bad += stream_key(Model::pp_view_state(x), v) != sk;
    }
    std::cout << "T1 information states: " << states.size() << " decisions, " << s2e.size() << " distinct streams, " << e2s.size()
              << " distinct canonical engine keys, mismatches " << t1_bad << (t1_bad || s2e.size() != e2s.size() ? "  FAIL\n" : "  pass\n");
    std::cout << "T2 suit permutation: " << t2_bad << " streams changed" << (t2_bad ? "  FAIL\n" : "  pass\n");
    std::cout << "T3 sample replay: " << t3_bad << " states differ, " << t3_stream_bad << " streams differ"
              << (t3_bad || t3_stream_bad ? "  FAIL\n" : "  pass\n");
    if (o["--snapshot"].empty()) return t1_bad || t2_bad || t3_bad || t3_stream_bad;
    // T7
    Model model(Partition::PP, nullptr);
    model.load(o["--snapshot"]);
    pp::Opts opt;
    model.pp_setup(opt, 1);
    const int level = model.ppd()->level;
    SnapshotPolicy target{model, true, std::mt19937_64(1)};
    int t7a_bad = 0, nd = std::stoi(o["--directions"]);
    std::cout << "T7 particle directions at level " << level << " (" << nd << " states; r(a) per particle count, max |change| vs 4096)\n";
    double worst64 = 0, worst1024 = 0;
    for (int i = 0; i < nd; ++i) {
        const State& s = states[std::uniform_int_distribution<size_t>(0, states.size() - 1)(rng)];
        const int v = s.actor;
        const pp::Sample x = pp::make_sample(s, v);
        // (a) Policy-LBR's belief on the same state and seed
        const uint64_t seed = 1000 + i;
        PolicyLBR<SnapshotPolicy> lbr(target, 64, seed);
        {
            const pp::Sample xx = x;
            State replay = pp::rebuild(xx, s.players[1 - v].discarded, s.players[1 - v].hidden[0], s.players[1 - v].hidden[1], Card{}, Card{},
                                       Card{}, Card{}, [&](const State& st, Action a) { lbr.observe(st, 1 - v, a); });
            (void)replay;
        }
        lbr.choose(s, v);
        model.ppd()->opt.particles = 64;
        const pp::Dir d64 = model.pp_direction(x, level, seed);
        const double lbr_eff = lbr.stats().average_effective_particles;
        const bool ok = std::abs(lbr_eff - d64.neff) <= 1e-9 * std::max(1.0, lbr_eff);
        t7a_bad += !ok;
        std::array<pp::Dir, 3> ds;
        int k = 0;
        for (int P : {256, 1024, 4096}) { model.ppd()->opt.particles = P; ds[k++] = model.pp_direction(x, level, seed + 77); }
        double c64 = 0, c1024 = 0;
        for (int a = 0; a < A; ++a) if (d64.mask & (1u << a)) {
            c64 = std::max(c64, std::abs(d64.r[a] - ds[2].r[a]));
            c1024 = std::max(c1024, std::abs(ds[1].r[a] - ds[2].r[a]));
        }
        worst64 = std::max(worst64, c64); worst1024 = std::max(worst1024, c1024);
        std::cout << "  state " << i << " street " << s.street << ": eff. particles LBR " << lbr_eff << " / ours " << d64.neff
                  << (ok ? "" : " MISMATCH") << "; labels P64/256/1024/4096 = " << d64.label << "/" << ds[0].label << "/" << ds[1].label
                  << "/" << ds[2].label << "; max|r64-r4096| " << c64 << ", max|r1024-r4096| " << c1024 << " (se64 max "
                  << *std::max_element(d64.se.begin(), d64.se.end(), [](double p, double q) { return (std::isinf(p) ? 0 : p) < (std::isinf(q) ? 0 : q); })
                  << ")\n";
    }
    std::cout << "T7a belief = Policy-LBR's: " << t7a_bad << " mismatches" << (t7a_bad ? "  FAIL\n" : "  pass\n");
    std::cout << "T7b settling: worst max|r64-r4096| " << worst64 << ", worst max|r1024-r4096| " << worst1024
              << (worst1024 < worst64 ? "  (shrinks: pass)\n" : "  (does not shrink: CHECK)\n");
    return t1_bad || t2_bad || t3_bad || t3_stream_bad || t7a_bad;
}
