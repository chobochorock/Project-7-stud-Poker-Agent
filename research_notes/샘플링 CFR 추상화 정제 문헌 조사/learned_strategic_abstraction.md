# Learned / Strategy-Aware Information Abstraction for Imperfect-Information Games (and regret/value clustering, bisimulation, VQ, Bregman/Jensen-gap clustering)

_Collected 2026-10-26. **How claims were checked:** WebFetch could not reach any host in this sandbox (DNS failures, and the egress proxy refused arxiv.org, semanticscholar.org, iclr.cc, aaai.org and others). So every claim was checked through search-engine retrieval of the primary page (its abstract, listing or text snippets). No full PDF was read end to end. Labels:_
- **[V-abs]**: checked against the primary source's abstract or official listing.
- **[V-snip]**: checked against text snippets from the primary paper body; the full text was not read.
- **[V-2nd]**: from a secondary summary (paper-note site, third-party review, or a later citing paper). Should be checked against the original.
- **[U]**: unverified. Comes from background knowledge, with no source retrieved this session.
- **[INF]**: my own derivation or inference.

---

## Q1. Potential-aware / EMD abstraction (Gilpin & Sandholm; Johanson et al. AAMAS 2013; Ganzfried & Sandholm AAAI 2014)

### Takeaway
The "strategic similarity" signals that are standard in poker are still card-outcome statistics: distributions of hand strength (equity histograms), with Earth Mover's Distance (EMD) for imperfect recall, and potential-aware trajectories over future rounds. These are clustered offline with k-means before solving. Nothing in this lineage uses regrets or values from the solver, and nothing gives a solution-quality guarantee. The evidence that they work is empirical (exploitability in limit hold'em, head-to-head play in no-limit).

### Cited Findings
- **Gilpin, Sandholm & Sørensen (AAAI 2007), "Potential-Aware Automated Abstraction of Sequential Games, and Holistic Equilibrium Analysis of Texas Hold'em Poker"** is the origin of "potential-aware" abstraction [V-abs listing] — [Semantic Scholar](https://www.semanticscholar.org/paper/Potential-Aware-Automated-Abstraction-of-Sequential-Gilpin-Sandholm/d755988c915426f21383d21a7df3661871c779e2). Method details [U]: the similarity signal is a histogram over next-round abstract buckets, the distance is L2, the clustering runs bottom-up with k-means per round, and the abstraction is built offline and has perfect recall.
- **Johanson, Burch, Valenzano & Bowling (AAMAS 2013), "Evaluating State-Space Abstractions in Extensive-Form Games"** evaluates abstraction quality in Texas hold'em independently of any particular abstract strategy, using a recent quality metric. It compares distribution-aware vs expectation-based abstractions and perfect- vs imperfect-recall abstractions, and reports that the metric predicts tournament performance better [V-abs] — [AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2013/docs/p271.pdf), [UAlberta PDF](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf). The poster lists "CFR and CFR-BR Exploitability" among its evaluations [V-snip] — [poster](https://johanson.ca/publications/poker/2013-aamas-kmeans-abstraction/2013-aamas-kmeans-abstraction-poster.pdf).
  - **Pre-river rounds:** k-means with EMD over hand-strength histograms. The paper notes that EMD measures how far probability mass is moved, unlike L2 or Kolmogorov–Smirnov [V-snip] — [AAMAS13 PDF](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf).
  - **River:** OCHS (Opponent Cluster Hand Strength). Opponent hands are split into 8 clusters (built by clustering first-round hands with EMD). Each hand becomes a vector of 8 hand-strength values, and these vectors are clustered with k-means under L2 [V-snip] — [AAMAS13 PDF](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf); [Gibson PhD thesis](https://poker.cs.ualberta.ca/publications/gibson.phd.pdf).
  - I did not retrieve the paper's numeric exploitability results (see Gaps).
- **Ganzfried & Sandholm (AAAI 2014), "Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance in Imperfect-Information Games"** (pp. 682–690) [V-abs] — [AAAI PDF](https://cdn.aaai.org/ojs/8816/8816-13-12344-1-2-20201228.pdf), [Semantic Scholar](https://www.semanticscholar.org/paper/Potential-Aware-Imperfect-Recall-Abstraction-with-Ganzfried-Sandholm/6e24f43ca186be95bc651d09121367b3905b5824), [mlanthology](https://mlanthology.org/aaai/2014/ganzfried2014aaai-potential/).
  - It is the first algorithm for potential-aware imperfect-recall abstraction with EMD. Instead of only the strength distribution at the final round (the prior leading method: imperfect-recall, distribution-aware k-means with EMD), it considers the trajectory of strength distributions over all future rounds [V-abs].
  - "Experiments on no-limit Texas Hold'em show that our algorithm improves performance over the previously best approach" [V-abs]. The head-to-head margins were not retrieved.
  - Mechanism [U]: histograms over next-round clusters, with ground distance equal to the EMD between next-round cluster centroids, computed with a fast approximate EMD.
  - Tartanian7 (Brown, Ganzfried & Sandholm, AAAI 2015) solved an abstraction computed with this algorithm, using external-sampling MCCFR [V-snip] — [Tartanian7, Semantic Scholar](https://www.semanticscholar.org/paper/Tartanian7:-A-Champion-Two-Player-No-Limit-Texas-Brown-Ganzfried/23499b01b70faf5b9215959e259d64bdb69e5d2a).
- **DeepStack** relies on a potential-aware card abstraction with 1000 buckets for its counterfactual-value-network inputs and outputs, and that many-to-one mapping can limit how well the network fits [V-snip of a citing paper] — [arXiv 1807.00900](https://arxiv.org/pdf/1807.00900). A secondary source says the 1000 buckets came from k-means and were never released [V-2nd; returned by search without a clear originating URL].
- **Kroer & Sandholm 2014** were the first to give solution-quality bounds for this kind of abstraction, and Kroer & Sandholm 2016 the first for imperfect-recall abstraction. The potential-aware EMD line has no such bounds [V-snip] — [EC'16 ACM](https://dl.acm.org/doi/10.1145/2940716.2940736), [EC'14 PDF](https://www.cs.cmu.edu/~sandholm/extensiveGameAbstraction.ec14.pdf).

### Inferences
- [INF] The baseline the requester calls "k-means" is presumably feature or equity k-means. In the literature the strongest outcome-based baselines are EMD over equity histograms (pre-river) plus OCHS (river), and the potential-aware EMD variant. A fair comparison should name which of these is being overlaid.
- [INF] All three methods are offline, solver-agnostic and card-outcome-only. None of them observes regrets or values, so they cannot target in-bucket regret cancellation directly.

### Gaps
- I could not retrieve the numeric exploitability tables from Johanson et al. 2013 or the mbb/h margins from Ganzfried & Sandholm 2014, because the full PDFs were unreachable.
- Gilpin–Sandholm 2007 mechanism details are [U]. So are the details of GameShrink, Gilpin & Sandholm's lossless "ordered game isomorphism" abstraction ([ResearchGate listing](https://www.researchgate.net/publication/220430980_Lossless_abstraction_of_imperfect_information_games); the title was found, but its content was not checked).

---

## Q2. EV / value-based and learned abstraction, 2024–2026 (EVPA, WEVA, FROI/"Beyond Outcome-Based", KrwEmd, Embedding CFR, Abstraction Agent, others)

### Takeaway
The closest prior art to "cluster information sets by solver-derived strategic quantities" is Li & Huang's line of work.
- **EVPA (ICLR 2025)** merges information sets by neural-network estimates of current and future expected values, online during subgame solving.
- **WEVA (arXiv 2605.10900, May 2026)** runs a short CFR warm-up on the unabstracted game, collects per-hand EV features at every decision node, builds depth-weighted vectors and clusters them with k-means++. It reports up to >80% lower exploitability than equity- and rank-based abstractions.

Both use expected values, not regret vectors. Neither is reported to have an abstraction-quality guarantee. The Fu et al. (CASIA) line (KrwEmd, FROI, Embedding CFR) keeps outcome or win-rate signals but adds history or continuous embeddings.

### Cited Findings
- **EVPA: Li & Huang, "Efficient Online Pruning and Abstraction for Imperfect Information Extensive-Form Games", ICLR 2025** (Tsinghua) [V-abs] — [ICLR poster](https://iclr.cc/virtual/2025/poster/29930), [proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c1b5863a6b0f925617b917bb2f55be0-Paper-Conference.pdf), [mlanthology](https://mlanthology.org/iclr/2025/li2025iclr-efficient-a/).
  - **Name:** EVPA stands for "Expected-Value Pruning and Abstraction", not "potential-aware". The assignment's expansion was wrong.
  - **Components:** (i) EV estimation from approximate Nash-equilibrium strategies; (ii) minimax pruning before CFR; (iii) dynamic online information abstraction that merges information sets by their current and future subgame expected values [V-abs].
  - **Networks:** M = 10 independent networks, trained with Huber loss, each mapping an information-set feature vector to a scalar EV estimate [V-snip, proceedings PDF].
  - **Training cost:** later papers by the same authors describe EVPA as training on CFR traversal data at the cost of billions of samples [V-2nd] — [WEVA arXiv 2605.10900](https://arxiv.org/abs/2605.10900), [Abstraction Agent arXiv 2609.04303](https://arxiv.org/abs/2609.04303). One summary says >10B samples [V-2nd] — [Lacuna](https://lacuna.tiptreesystems.com/work/efficient-online-pruning-and-abstraction-for-imperfect-information-extensive/wrk_55c9b3cc185043ebb71da6866f992ebf).
  - **Clustering step:** reportedly k-means++ on current and future EVs, running in under 1 s during play [V-2nd, same Lacuna summary; not confirmed in the paper text].
  - **Results:** on HUNL it beats DeepStack's replication and Slumbot by significant win-rate margins in several settings, and reaches an approximate NE in 1–2% of the DeepStack replication's solve time [V-abs]. Up to 100 ± 68 mbb/h vs Slumbot at 0.2–2 s per move [V-2nd, Lacuna; unconfirmed]. These are head-to-head win rates, not exploitability.
  - **Guarantee:** none found in any retrieved text.
- **WEVA: Li & Huang, "Effective, Efficient, and General Information Abstraction for Imperfect-Information Extensive-Form Games", arXiv 2605.10900 (11 May 2026, cs.GT)** [V-abs] — [arXiv abs](https://arxiv.org/abs/2605.10900), [HTML](https://arxiv.org/html/2605.10900).
  - **Name:** WEVA = "Warm-up Expected Value-based Abstraction".
  - **Signal:** run W CFR iterations on the full (unabstracted) game, collect per-hand EV features at each decision node, and combine them into a depth-weighted vector.
  - **Buckets:** k-means++.
  - **Timing:** computed before the abstract solve, but derived from a short full-game solve. The authors describe it as needing no domain knowledge and no pre-training, with little overhead [V-abs].
  - **Results:** across 3 games, several bucket counts and several CFR variants, it consistently beats equity- and rank-based abstractions, with up to >80% lower exploitability. W = 10 warm-up iterations already beat existing methods in most settings [V-abs]. The benchmarks are river endgames: "HUNL Endgame", "Double-Board HUNL", and a third I could not identify [V-snip].
  - **Caveat:** a third-party review flags that early CFR estimates may be too noisy for reliable clustering in general [V-2nd] — [Pith](https://pith.science/paper/2605.10900).
  - **Guarantee:** none found.
  - **Metric:** exploitability of the lifted strategy.
- **"Abstraction Agent": Li & Huang, arXiv 2609.04303 / Findings of EMNLP 2026** [V-abs] — [arXiv](https://arxiv.org/abs/2609.04303), [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1246/).
  - **Signal:** an LLM discovers continuous strategic features from natural-language rules and scores private states on them. There is no hand evaluator, rollout or tree traversal.
  - **Buckets:** correlation-based feature selection, then k-means.
  - **Results:** up to 62% lower lifted-strategy exploitability than an expected-hand-strength (EHS) baseline on HUNL turn endgames, and all K = 100 runs beat EHS. On "ROVER Trials", a novel game, it beats a scalar rank baseline at every granularity and a 2-D potential-aware baseline at most granularities. Results for PLO4 and Riichi Mahjong are qualitative [V-abs].
  - **Related-work note:** the paper says WEVA eliminates neural training but still needs CFR iterations on the unabstracted game [V-snip].
- **FROI: Fu, Yin, Liu, Xu & Huang, "Beyond Outcome-Based Imperfect-Recall: Higher-Resolution Abstractions for Imperfect-Information Games", arXiv 2510.15094 (16 Oct 2025)** [V-abs] — [arXiv](https://arxiv.org/abs/2510.15094v1), [HTML](https://arxiv.org/html/2510.15094). ResearchGate lists it as AAMAS 2026 [V-2nd] — [ResearchGate](https://www.researchgate.net/publication/405212856_Beyond_Outcome-Based_Imperfect-Recall_Higher-Resolution_Abstractions_for_Imperfect-Information_Games).
  - **Framework:** "signal observation ordered games" (SOOGs), plus a "resolution bound", an information-theoretic upper bound on achievable performance under a given signal abstraction. It formalizes the mainstream potential-aware outcome-based imperfect-recall methods as "potential-aware outcome isomorphism" (PAOI).
  - **Method:** "full-recall outcome isomorphism" (FROI), which keeps history. It consistently beats outcome-based imperfect-recall methods on hold'em-style benchmarks [V-abs].
  - **Scale:** in HULH, PAOI reduces river states to about 20,000 while FROI keeps over 638M [V-2nd; summary site, unconfirmed].
  - **Signal type:** outcome/equity-based, not value- or regret-based.
- **KrwEmd: Fu, Xu, Bai, Zhao & Huang, "Signal Observation Models and Historical Information Integration in Poker Hand Abstraction", arXiv 2403.11486** [V-abs] — [arXiv](https://arxiv.org/abs/2403.11486). Revised as "KrwEmd: Revising the Imperfect-Recall Abstraction from Forgetting Everything", arXiv 2511.12089 (Nov 2025) [V-abs] — [arXiv PDF](https://arxiv.org/pdf/2511.12089).
  - **Signal:** a k-recall win-rate feature, so information from past rounds is kept.
  - **Buckets:** clustering of signal-observation infosets under EMD.
  - **Timing:** offline.
  - **Results:** "significantly outperforms" existing algorithms in Numeral211 hold'em, and history-using algorithms reach a higher performance ceiling [V-abs]. The papers argue that PaEmd-style (potential-aware EMD) abstraction, used in DeepStack, Libratus and Pluribus, discards all history [V-snip].
- **Embedding CFR: Fu, Liu, Xu & Huang, "No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding", arXiv 2511.12083 / AAAI-26 (vol. 40 no. 20, pp. 16906–16913)** [V-abs] — [arXiv](https://arxiv.org/abs/2511.12083), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38736).
  - **Idea:** replace hard buckets with a continuous low-dimensional embedding. Regret accumulation and strategy updates run in embedding space.
  - **Guarantee:** theoretical analysis supporting reduced cumulative regret.
  - **Results:** at equal memory, exploitability falls faster than with clustering-based abstraction [V-abs].
  - **Details** [V-2nd] — [paper note](https://en.papernotes.org/AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd/): a pre-trained HandEbdNet maps each hand to an m-dimensional probability vector, supervised by hand strength (win-rate distributions). Each info-block has m "advisors". The strategy is mapped back through the embedding matrix. In effect this is soft clustering (each hand is a mixture over m prototypes) with an outcome-based training signal, not a regret-based one.
- **Word-embedding abstraction: "Domain-Independent Game Abstraction using Word Embedding Techniques", arXiv 2605.15543** encodes poker hands as text and embeds them. On Kuhn poker it beats random bucketing but is worse than poker-specific hand bucketing [V-snip] — [arXiv PDF](https://arxiv.org/pdf/2605.15543).
- **LAMIR: Kubíček et al., "Look-ahead Reasoning with a Learned Model in Imperfect Information Games", arXiv 2510.05048 / ICLR 2026** [V-abs] — [arXiv](https://arxiv.org/abs/2510.05048), [mlanthology](https://mlanthology.org/iclr/2026/kubicek2026iclr-lookahead/).
  - Learns an abstraction "by clustering information sets" so that subgames stay small enough for CFR+ look-ahead. It targets games without chance (Dark Chess, Stratego, Battleship, II-Goofspiel) [V-abs].
  - A "Tabular K-Means" appendix exists [V-snip]. A secondary summary says the clustering property is either the legal actions or the CFR strategies [V-2nd]. This is one of the few explicitly strategy-based clusterings found.

### Inferences
- [INF] **The design space, mapped to the requester's axes:**

| Method | Signal | Buckets | Timing | Guarantee |
|---|---|---|---|---|
| EVPA | NN-estimated current + future EV (scalars) | k-means++ (per secondary source) | online | none found |
| WEVA | depth-weighted per-node EV vectors from a W-iteration full-game CFR warm-up | k-means++ | pre-solve, solver-derived | none found |
| Abstraction Agent | LLM-scored features | k-means | offline | none |
| FROI / KrwEmd | outcome or win-rate with history | isomorphism / EMD k-means | offline | resolution bound (an information measure, not an exploitability bound) |
| Embedding CFR | pre-trained win-rate embedding | soft prototypes | offline pre-training | regret-reduction analysis |
| LAMIR | legal actions / CFR strategies | tabular k-means | learned | not found |

- [INF] **WEVA is the nearest published neighbour to the requester's proposal.** Both use solver-generated quantities from a short full-game run and both use k-means. WEVA clusters EV (value) vectors under squared Euclidean distance; the proposal clusters average-regret vectors under the cancellation (Jensen-gap) divergence.
  - One structural difference: per-action regret is r(a) = v(a) − v(σ), so a per-action EV vector carries the regret vector up to the scalar offset v(σ). WEVA's features are per-hand EVs at decision nodes; whether they are per-action I could not confirm.
  - A comparison against WEVA-style EV clustering is the most relevant baseline beyond equity/EMD k-means.
- [INF] **Evaluation metrics differ across these papers:**
  - lifted-strategy exploitability: WEVA, Abstraction Agent, Embedding CFR;
  - head-to-head mbb/h: EVPA;
  - resolution bound: FROI/KrwEmd.

  Per the project's AGENTS.md, these must not be conflated with each other or with the project's Local BR-gap.

### Gaps
- I could not confirm from primary text whether EVPA's merge rule is k-means++, threshold-based or something else, nor the exact EV feature definition (per-action or per-hand).
- I could not identify WEVA's third benchmark, the exact weighting scheme, or whether its features are per-action EVs.
- No retrieved paper gives a quality guarantee for EV-based clustering.
- I found no 2024–2026 paper that uses regret vectors (rather than EVs or win rates) as the clustering signal.

---

## Q3. Regret- or counterfactual-value-based clustering of information sets; strategy-based abstraction

### Takeaway
I found **no paper that clusters information sets by their regret vectors using a cancellation-exact divergence**. The relevant prior work is of four kinds:
1. Theory saying when merged information sets lose nothing: Lanctot et al.'s (skew) well-formed games, where utilities are proportional up to a "skew".
2. Bound-minimizing abstraction cast as clustering with a metric derived from the bound (Kroer & Sandholm EC'16).
3. Regret-driven function approximation that implicitly learns an abstraction (RCFR, Waugh et al. AAAI'15, whose regression tree partitions the feature space by predicted regret).
4. Solver-driven refinement of imperfect-recall abstractions (Čermák et al. FPIRA / AIJ 2020).

### Cited Findings
- **Lanctot, Gibson, Burch, Zinkevich & Bowling (ICML 2012), "No-Regret Learning in Extensive-Form Games with Imperfect Recall"** [V-abs] — [arXiv](https://arxiv.org/abs/1205.0622), [PDF](https://webdocs.cs.ualberta.ca/~bowling/papers/12icml-ir.pdf).
  - It gives the first CFR regret bound for a general class of imperfect-recall games, and CFR applied to any abstraction in that class bounds regret in the full game.
  - In "well-formed" games CFR still minimizes regret. In "skew well-formed" games regret is bounded according to how far the utilities are skewed (utilities and chance probabilities are proportional only up to an additive term) [V-snip].
  - In imperfect-recall games the sum of counterfactual regrets is no longer an upper bound on external regret [V-snip] — [HTML](https://arxiv.org/html/1205.0622).
  - Experiments: die-roll poker, phantom tic-tac-toe, Bluff [V-abs].
  - Kroer & Sandholm later relaxed this to "chance-relaxed skew well-formed" (CRSWF) games [V-snip] — [NeurIPS 2018 PDF](https://proceedings.neurips.cc/paper/2018/file/aa942ab2bfa6ebda4840e7360ce6e7ef-Paper.pdf).
- **Kroer & Sandholm (EC 2016), "Imperfect-Recall Abstractions with Bounds in Games"** [V-abs] — [ACM](https://dl.acm.org/doi/10.1145/2940716.2940736), [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302).
  - For games with certain properties, choosing a bound-minimizing abstraction for a single level becomes a **clustering problem whose distance function is the increase in the bound**.
  - When payoffs are on the same scale, the input is a metric space, which gives a 2-approximation. Otherwise it is not a metric [V-abs].
  - The distance between information sets (the error from merging them) is symmetric and satisfies the triangle inequality but not identity of indiscernibles. Zero-distance pairs can be merged losslessly as preprocessing [V-snip].
  - Its signal is payoff and chance differences between mapped leaves, not regrets.
- **Kroer & Sandholm (EC 2014)** gave the first solution-quality bounds for lossy extensive-form abstraction [V-snip] — [PDF](https://www.cs.cmu.edu/~sandholm/extensiveGameAbstraction.ec14.pdf).
- **Kroer & Sandholm (NeurIPS 2018), "A Unified Framework for Extensive-Form Game Abstraction with Bounds"** gives an exact decomposition of solution-quality error without earlier restrictive assumptions. It uses Zinkevich's counterfactual values of information sets in its analysis [V-snip] — [NeurIPS PDF](https://proceedings.neurips.cc/paper/2018/file/aa942ab2bfa6ebda4840e7360ce6e7ef-Paper.pdf). The exact form of its CFV-error terms was not retrieved.
- **Sandholm's (AAAI 2015) survey:** running CFR on their class of (possibly imperfect-recall) abstractions gives bounded regret in the full game [V-snip] — [PDF](https://cdn.aaai.org/ojs/9757/9757-13-13285-1-2-20201228.pdf).
- **RCFR: Waugh, Morrill, Bagnell & Bowling (AAAI 2015), "Solving Games with Functional Regret Estimation"** [V-abs] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9445), [arXiv 1411.7974](https://arxiv.org/pdf/1411.7974), [UAlberta PDF](https://poker.cs.ualberta.ca/publications/15aaai-rcfr.pdf).
  - A regressor, learned online, estimates each action's regret from domain features, and a no-regret learner uses the estimates.
  - The authors describe it as a generalization of abstraction: the regressor adapts the abstraction to the current policy and learns its structure from data.
  - Regret is bounded in terms of approximation quality, and self-play converges to NE when true regrets are realizable [V-abs].
  - The original regressor is a regression tree over hand-crafted features such as expected hand strength [V-snip].
  - It reportedly beats state-of-the-art abstraction techniques given the same resources [V-abs].
- **Brown & Sandholm (AAAI 2016), "Strategy-Based Warm Starting for Regret Minimization in Games":** bucketing can speed early convergence "due to all information sets in a bucket sharing their acquired regrets" and the smaller tree [V-snip] — [PDF](https://www.cs.cmu.edu/~sandholm/warmStart.aaai16.withAppendixAndTypoFix.pdf).
- **Brown & Sandholm (IJCAI 2015), "Simultaneous Abstraction and Equilibrium Finding in Games"** concerns *action* abstraction: actions are added during the computation where the current strategy suggests they matter. It is not card or information-set clustering [V-abs] — [ACM](https://dl.acm.org/doi/abs/10.5555/2832249.2832317), [Semantic Scholar](https://www.semanticscholar.org/paper/Simultaneous-Abstraction-and-Equilibrium-Finding-in-Brown-Sandholm/1dca537dca268059d333ffcc22a5684ce06dd17b).
- **Čermák, Bošanský & Lisý (IJCAI 2017), FPIRA** [V-snip] — [IJCAI PDF](https://www.ijcai.org/proceedings/2017/0130.pdf).
  - Starts from a coarse imperfect-recall abstraction, runs fictitious play, and refines the information-set structure only when an update could make average strategies more exploitable than the same update in the perfect-recall game.
  - Per a follow-up, the check is the worst-case utility difference between abstract and original strategies over opponent strategies [V-snip].
  - Results: a simplified poker game with 7·10^5 information sets is solved using 1.8% of the information sets; on Leduc it converges at the same rate as plain FP [V-snip].
- **Čermák, Lisý & Bošanský (AIJ 2020)**, the journal extension: domain-independent construction of bounded-loss imperfect-recall abstractions. It requires traversing the full unabstracted tree [V-abs] — [arXiv 1803.05392](https://arxiv.org/abs/1803.05392), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370220300126).
- **LAMIR** clusters information sets with tabular k-means, reportedly on legal actions or CFR strategies [V-2nd] — [arXiv 2510.05048](https://arxiv.org/abs/2510.05048). This is a strategy-based clustering.

### Inferences
- [INF] **Link to the requester's cancellation divergence.** Lanctot et al.'s well-formed condition (merged infosets' per-action counterfactual utilities proportional, so their regrets are proportional) is a sufficient condition for zero in-bucket regret cancellation at every iteration. The requester's D_psi (see Q6) measures the per-snapshot violation of a weaker condition: same argmax / same sign.
- [INF] **Kroer–Sandholm 2016 is the closest structural precedent** for "k-means/k-center in a divergence derived from a regret bound". Their distance is derived from a worst-case solution-quality bound over payoffs and chance. The requester's is derived from the realized cancellation of the solver's own regrets, which is data-dependent and strategy-aware.
- [INF] **RCFR is the closest precedent for learning feature-space dissimilarities from regrets.** A regression tree trained to predict regrets induces a partition whose leaves group information sets with similar predicted regret vectors. A regression tree over enum and numeric features with a multi-output regret target is a ready-made "regret-trained abstraction" baseline.

### Gaps
- No paper found that clusters by per-infoset (average) regret vectors, or that defines a "regret cancellation" divergence.
- The exact skew-well-formed theorem constants (Lanctot 2012) and the CFV-error decomposition (Kroer & Sandholm 2018) were not retrieved verbatim.
- I could not confirm the label "CFR+IRA" for the AIJ 2020 algorithm.

---

## Q4. Bisimulation metrics / state-abstraction metrics in games or POMDPs

### Takeaway
Bisimulation metrics are well developed for MDPs (Ferns, Panangaden & Precup) and have been learned with deep encoders (DBC, MICo). I found **no bisimulation metric defined for imperfect-information extensive-form games or used for poker abstraction**. The nearest game-theoretic analogues are:
- approximate state abstraction with Q*/minimax-value criteria for two-player zero-sum Markov games, with a duality-gap bound (arXiv 2412.15877);
- exact bisimulation quotients for pure-strategy transfer in logic/verification work.

Kroer & Sandholm's payoff/chance-error bounds are structurally an approximate-bisimulation notion, although they are not named as such.

### Cited Findings
- **Ferns, Panangaden & Precup (UAI 2004), "Metrics for Finite MDPs":** bisimulation-based metrics for discounted infinite-horizon MDPs [V-abs] — [arXiv 1207.4114](https://arxiv.org/pdf/1207.4114).
  - The SIAM J. Comput. 2011 version covers continuous MDPs and shows the optimal value function varies continuously with respect to the metric [V-snip] — [PDF](https://www.cs.mcgill.ca/~prakash/Pubs/siamFP11.pdf).
  - Ferns & Precup (UAI 2014) show the bisimulation metric is the optimal value function of an optimal coupling of the MDP with itself [V-abs] — [PDF](https://www.auai.org/uai2014/proceedings/individuals/67.pdf).
  - Computing it can cost more than dynamic programming on the original MDP [V-snip] — [AAAI 2015 representation discovery](https://www.cs.mcgill.ca/~prakash/Pubs/aaai15_paper.pdf).
- **DBC: Zhang, McAllister, Calandra, Gal & Levine (ICLR 2021 oral), "Learning Invariant Representations for RL without Reconstruction"** trains an encoder so that latent distances match bisimulation distances, and is robust to task-irrelevant distractors [V-abs] — [arXiv 2006.10742](https://arxiv.org/pdf/2006.10742), [ICLR](https://iclr.cc/virtual/2021/oral/3480).
- **MICo: Castro, Kastner, Panangaden & Rowland (NeurIPS 2021):** a sampling-based behavioural distance that is easier to learn at scale [V-abs] — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/fd06b8ea02fe5b1c2496fe1700e9d16c-Paper.pdf).
- **Castro (AAAI 2020)** introduced π-bisimulation, behavioural similarity under a fixed policy π [V-snip, via a citing paper] — [arXiv 2412.15877](https://arxiv.org/html/2412.15877v1).
- **Abel, Hershkowitz & Littman (ICML 2016), "Near Optimal Behavior via Approximate State Abstraction":** approximate abstractions that group states with similar Q, R and T functions, with polynomially bounded behaviour error [V-abs] — [PMLR](http://proceedings.mlr.press/v48/abel16.html), [arXiv 1701.04113](https://arxiv.org/abs/1701.04113).
- **"Approximate State Abstraction for Markov Games", arXiv 2412.15877** extends Abel et al. to two-player zero-sum Markov games with optimal-Q / minimax-value criteria and bounds the duality gap of the resulting equilibrium [V-abs] — [arXiv HTML](https://arxiv.org/html/2412.15877v1).
- **Optimistic Q-learning with aggregated states** has a regret bound with an ε·HK term, where ε is the maximum gap between optimal Q-values in a common aggregate. This is a value-similarity aggregation guarantee [V-snip] — [arXiv 1912.06366](https://arxiv.org/pdf/1912.06366).
- **Bisimulation in games (logic and verification):**
  - In regular imperfect-information games, a finite bisimulation quotient of the information tree transfers pure strategies, but randomized-strategy transfer can fail [V-snip] — [arXiv 2403.20133](https://arxiv.org/html/2403.20133).
  - "Nash Equilibrium and Bisimulation Invariance": players must act identically on bisimilar histories [V-snip] — [arXiv 1808.09213](https://arxiv.org/pdf/1808.09213).
- **Policy-behaviour metrics in Markov games:** a metric of reward difference plus Wasserstein-1 between transitions, under occupancy measures, used for policy diversity [V-snip] — [arXiv 2106.04958](https://arxiv.org/pdf/2106.04958).

### Inferences
- [INF] Bisimulation is the MDP analogue of Kroer–Sandholm-style bounded abstraction: reward error plus transition (chance) error, recursively. A poker-specific "bisimulation metric over infosets" would need to handle the opponent's belief/range (public-belief-state level). This is why the game literature uses bound-based metrics (Kroer–Sandholm) or outcome distributions (EMD) instead.
- [INF] **DBC is a direct template for the requester's learned-dissimilarity idea:** train an encoder (enum embeddings plus an MLP for numeric features) so that ‖φ(I) − φ(J)‖ regresses a target divergence. In DBC the target is a bisimulation distance; here it would be a regret-space divergence between infosets. This is the term to cite.

### Gaps
- No bisimulation metric for imperfect-information extensive-form games, and no use of DBC/MICo with CFR, was found.
- POMDP bisimulation (Castro, Panangaden & Precup, IJCAI 2009) was not retrieved [U].

---

## Q5. Vector-quantized / discrete-latent abstractions in games or RL, especially with CFR

### Takeaway
VQ codebooks are used as learned discrete abstractions for planning with MCTS (VQ planning; Stochastic MuZero chance codes), including stochastic "single-player chess". I found **no use of VQ-VAE codebooks as an information-set abstraction for CFR**. The nearest CFR-side analogue is Embedding CFR's continuous soft prototypes (Q2).

### Cited Findings
- **Ozair, Li, Razavi, Antonoglou, van den Oord & Vinyals (ICML 2021), "Vector Quantized Models for Planning"** [V-abs] — [arXiv 2106.04615](https://arxiv.org/abs/2106.04615), [PMLR PDF](http://proceedings.mlr.press/v139/ozair21a/ozair21a.pdf).
  - Discrete autoencoder codes represent the environment's stochastic responses, and stochastic MCTS searches over both agent actions and latent codes.
  - It substantially beats offline MuZero on "single-player chess", where opponent moves are folded into the dynamics, and scales to DeepMind Lab, which is partially observable.
- **Stochastic MuZero** uses a VQ-VAE-style chance-outcome encoder with a fixed one-hot codebook [V-2nd, via a citing paper] — [arXiv 2212.11087](https://arxiv.org/pdf/2212.11087).
- **Hierarchical imitation learning** with a VQ-VAE subgoal model, inspired by Ozair et al. [V-abs] — [arXiv 2301.12962](https://arxiv.org/pdf/2301.12962).
- A targeted search for VQ / VQ-VAE / codebook combined with CFR / imperfect-information abstraction returned nothing relevant [search result, 2026-10-26].

### Inferences
- [INF] Hard k-means abstraction is a VQ codebook (Linde–Buzo–Gray is a special case of Bregman hard clustering; see Q6). "Regret-space k-means" is therefore VQ of regret vectors with a non-Euclidean distortion. A VQ-VAE version would learn an encoder from infoset features to code indices whose codebook distortion is the cancellation divergence. Its straight-through/commitment losses would need a differentiable surrogate, because ψ is piecewise linear (Q6).

### Gaps
- No VQ+CFR or VQ+poker-abstraction work found.
- DreamerV2/V3 categorical latents were not checked this session [U].

---

## Q6. Bregman-divergence / Jensen-gap clustering applied to regrets, values or policies; what the requester's D_psi is

### Takeaway
The requester's identity, "in-bucket regret cancellation = mass-weighted Bregman divergence to the bucket mean", is exactly Banerjee et al.'s **Bregman information**, also called the **Jensen-gap information**, for the convex function ψ(x) = max(0, maxₐ xₐ). Bregman hard clustering (k-means with centroid = weighted mean) minimizes total Bregman information, so "k-means in regret space with D_ψ minimizes cancellation" follows from Banerjee et al. 2005. Because ψ is not differentiable, the right framework is generalized (subgradient) Bregman divergences:
- Telgarsky & Dasgupta extend Bregman hard and agglomerative clustering to non-differentiable convex functions.
- ψ is piecewise-linear and positively homogeneous, so D_ψ is a Lovász-Bregman-type divergence (Iyer & Bilmes): it depends on the centroid only through its argmax/sign "region".

I found **no prior use of Bregman or Jensen-gap clustering on regrets, values or policies in games**. A key consequence for design: the objective is **degenerate** (see Inferences).

### Cited Findings
- **Banerjee, Merugu, Dhillon & Ghosh (JMLR 6:1705–1749, 2005), "Clustering with Bregman Divergences"** [V-abs] — [JMLR](https://www.jmlr.org/papers/v6/banerjee05b.html), [PDF mirror](https://www.lix.polytechnique.fr/~nielsen/MEF/Banerjee_2005_JMLR.pdf).
  - Hard clustering is posed as minimizing the loss in **Bregman information**, and the iterative (k-means-type) algorithm monotonically decreases it. k-means, Linde–Buzo–Gray and information-theoretic (KL) clustering are special cases. There is a bijection between regular exponential families and regular Bregman divergences, which gives EM soft clustering [V-abs].
  - Bregman information is the expected Bregman divergence to the mean [V-snip] — [Ghosh slides](https://www.isical.ac.in/~scc/seminars/CCBD.pdf).
  - Banerjee's 2003 talk introduced Bregman information together with a "Jensen's equality" [V-abs] — [TTI-C announcement](https://mailman.cs.uchicago.edu/mailman/public/colloquium/2003-December/000222.html).
- **"Equivalence of Informations Characterizes Bregman Divergences", arXiv 2501.02104** defines **Jensen-gap information** as Σ wᵢ φ(xᵢ) − φ(Σ wᵢ xᵢ). It shows Bregman divergences are the unique class for which Jensen-gap information equals divergence-based information, and argues it is a natural loss for clustering and quantization [V-abs] — [arXiv PDF](https://arxiv.org/pdf/2501.02104).
- **Telgarsky & Dasgupta (ICML 2012), "Agglomerative Bregman Clustering"** extends Bregman hard clustering to **non-differentiable convex functions** and derives an agglomerative merge cost [V-snip] — [arXiv 1206.6446](https://arxiv.org/pdf/1206.6446).
- **Generalized Bregman divergences:** for non-differentiable φ they are defined with a subgradient map (or directional derivatives), and the divergence depends on which subgradient is chosen at kinks [V-snip] — [Iyer & Bilmes NeurIPS 2012](http://papers.neurips.cc/paper/4692-submodular-bregman-and-the-lovasz-bregman-divergences-with-applications.pdf), [arXiv 2112.09191](https://arxiv.org/pdf/2112.09191), [Wikipedia](https://en.wikipedia.org/wiki/Bregman_divergence).
- **Lovász–Bregman divergence (Iyer & Bilmes, UAI 2013):** a generalized Bregman divergence of the Lovász extension of a submodular function. It is a distortion between a "score" vector and an "ordering", was introduced for clustering ranked vectors, and has Kendall-τ-like special cases [V-abs] — [arXiv 1308.5275](https://arxiv.org/abs/1308.5275).
- **Bregman divergences in RL** appear as regularizers or analysis tools, not clustering losses:
  - O'Donoghue relates the Bregman divergence between the current and optimal policy to value suboptimality in regularized MDPs [V-abs] — [arXiv 2210.12160](https://arxiv.org/abs/2210.12160).
  - Policy mirror descent with general Bregman divergences [V-abs] — [arXiv 2105.11066](https://arxiv.org/html/2105.11066).
- Targeted searches for Bregman clustering applied to policies, values or regrets in games returned nothing relevant [search results, 2026-10-26].

### Inferences
These are [INF] derivations from the cited definitions, not results from any paper.

**1. Exact identity.** Write ρ̄ = Σ m_I ρ_I / M. For any subgradient g ∈ ∂ψ(ρ̄):

> Σ_I m_I ψ(ρ_I) − M ψ(ρ̄) = Σ_I m_I D_ψ^g(ρ_I, ρ̄)

This holds because Σ_I m_I (ρ_I − ρ̄) = 0. It is the Bregman information / Jensen gap, and it does not depend on the choice of subgradient.

**2. Degeneracy.** Because ψ is positively 1-homogeneous, ψ(y) = ⟨g, y⟩ for every g ∈ ∂ψ(y). Hence:
- If maxₐ yₐ > 0: D_ψ(x, y) = ψ(x) − x_{a*(y)}, where a*(y) is y's argmax action.
- If maxₐ yₐ < 0: D_ψ(x, y) = ψ(x).

So **D_ψ depends on the centroid only through which of the |A|+1 linear pieces it lies in**: its argmax action, or "all regrets nonpositive". This is the Lovász-Bregman "depends on the ordering" phenomenon in its top-1 form. Indeed maxₐ xₐ is the Lovász extension of the submodular function F(S) = 1[S ≠ ∅]; adding a constant 0 coordinate gives ψ.

Consequences:
- **(a) Zero condition.** A bucket's cancellation is zero iff all its members' ρ_I lie in one closed linear region of ψ (they share a nonnegative argmax, or all are ≤ 0).
- **(b) Trivial optimum.** With k ≥ |A|+1 buckets per partition cell, a snapshot-optimal clustering has zero cancellation: group by sign and argmax of the average regret. Magnitudes are ignored.
- **(c) Lloyd ties.** The Lloyd assignment step is tie-ridden: centroids with the same argmax are indistinguishable.

Practical algorithms should therefore use one of:
- agglomerative merges with exact Jensen-gap increase (Telgarsky–Dasgupta);
- Hartigan-style single-point moves on the true objective;
- a smoothed ψ (point 3);
- an objective aggregated over several regret snapshots or iterations, so that the partition is robust to the drift of ρ_I.

**3. Smooth surrogates.**
- **Log-sum-exp.** ψ_τ(x) = τ·log(1 + Σₐ exp(xₐ/τ)) → ψ as τ → 0. Its Bregman divergence is D_{ψ_τ}(x, y) = τ·KL(softmax_τ([0, y]) ‖ softmax_τ([0, x])), the standard identity that the Bregman divergence of log-sum-exp is the reversed KL between softmaxes. Regret-space clustering with ψ_τ is therefore Banerjee/Dhillon KL (information-theoretic) clustering of Hedge/Boltzmann policies derived from the regrets: a principled, non-degenerate relaxation.
- **Regret-matching potential.** Φ(x) = ½‖x₊‖² is the Blackwell / regret-matching potential (Hart–Mas-Colell); background, [U]. Its gradient x₊ is regret matching's unnormalized strategy. Its Bregman divergence is a natural regret-matching-specific alternative.

**4. Terminology.** Precise names for the requester's construction:
- "Bregman hard clustering (Banerjee et al. 2005) with a generalized (subgradient) Bregman divergence";
- "minimizing Jensen-gap / Bregman information";
- "a Lovász-Bregman-type divergence (Iyer & Bilmes) for the piecewise-linear potential ψ".

### Gaps
- No source applies Bregman or Jensen-gap clustering to regrets, CFVs or policies in games. This looks unoccupied, but absence of evidence from search snippets is not proof.
- The LB-divergence statement "depends only on the ordering of y" is my reading of Iyer & Bilmes's "score vs ordering" framing; the search did not show it verbatim.
- The Lovász-extension identity for max is my derivation, not cited.

---

## Q7. Metric learning for categorical features: what to call the "learned per-enum distance matrix plus MLP for numeric features" idea

### Takeaway
Learning a distance matrix between the values of each categorical attribute from a supervised target is the **Value Difference Metric (VDM)** family. **HVDM** (Wilson & Martinez, JAIR 1997) combines VDM on nominal attributes with normalized Euclidean distance on numeric ones. That is structurally the requester's "enum distance matrix + numeric distance". The unsupervised analogue is **DILCA** (context-based categorical dissimilarity learning). The neural version is **entity embeddings of categorical variables** (Guo & Berkhahn 2016), with embedding distances learned by supervised training. With regret vectors as the supervised target and a learned joint metric, the general names are **"supervised distance-metric learning"** or **"deep metric learning"**, and **DBC-style metric regression** (Q4) when distances are regressed onto a target divergence.

### Cited Findings
- **Wilson & Martinez (JAIR 6:1–34, 1997), "Improved Heterogeneous Distance Functions"** proposes HVDM, IVDM and WVDM [V-abs] — [JAIR](https://www.jair.org/index.php/jair/article/view/10182), [arXiv cs/9701101](https://arxiv.org/abs/cs/9701101).
  - HVDM uses Euclidean distance on linear attributes and VDM on nominal ones, with normalization.
  - VDM's nominal distance is based on class-conditional probabilities. The original VDM needed continuous attributes to be discretized.
  - IVDM and WVDM gave higher average accuracy than HVDM [V-abs].
  - VDM originates with Stanfill & Waltz 1986, and Cost & Salzberg's 1993 MVDM is a modified version [U; my searches for the originals failed].
- **DILCA: Ienco, Pensa & Meo (ACM TKDD 6(1), 2012), "From Context to Distance: Learning Dissimilarity for Categorical Data Clustering"** [V-abs] — [ACM](https://dl.acm.org/doi/10.1145/2133360.2133361).
  - The distance between two values of one attribute is the Euclidean distance between the conditional distributions of selected "context" attributes. Context attributes are chosen by symmetric uncertainty above a threshold. It scales to thousands of attributes [V-snip].
  - Other learned categorical dissimilarities exist, including an association-based measure that is Euclidean-compatible and extends to mixed data [V-snip] — [ResearchGate](https://www.researchgate.net/publication/222521333_An_association-based_dissimilarity_measure_for_categorical_data).
  - Survey: [arXiv 2408.17244](https://arxiv.org/html/2408.17244v3).
  - A 2026 paper on learnable weighting of intra-attribute distances for nominal and ordinal attributes [V-listing] — [arXiv 2607.05464](https://arxiv.org/pdf/2607.05464).
- **Guo & Berkhahn (2016), "Entity Embeddings of Categorical Variables":** each category value gets a learned vector, trained as part of supervised neural-network training. It helps most for high-cardinality and sparse features [V-abs] — [arXiv 1604.06737](https://arxiv.org/abs/1604.06737).

### Inferences
- [INF] **Closest classical analogue: a "regret-conditioned VDM".** For each enum value v of a feature, form the mass-weighted mean regret vector E[ρ | feature = v]. Define the distance between values u and v as a divergence between those conditional means, either L2 or D_ψ/ψ_τ from Q6. This is exactly VDM with a class-conditional distribution replaced by a regret-conditional mean.
- [INF] **Neural generalization:** entity embeddings for enums plus an MLP for numeric features, trained so that embedding distances regress a regret-space divergence (DBC-style) or so that a decoder predicts ρ (RCFR-style; Deep-CFR-style regret networks).
- [INF] **Distinguish the two goals.**
  - **Learning a metric, then clustering** (VDM/HVDM/DILCA, metric learning).
  - **Learning the partition directly against the objective**: RCFR regression trees, or Bregman hard clustering on predicted ρ.

  The second avoids the metric being only a proxy for cancellation.

### Gaps
- Primary sources for Stanfill & Waltz (VDM) and Cost & Salzberg (MVDM) were not retrieved.
- No work found that learns categorical-feature distances from regrets or other game-solver signals.
