# Online / Adaptive Abstraction Refinement During CFR/MCCFR Equilibrium Computation (literature notes, compiled 2026-10-26)

Access caveat, read first: in this session, full-text fetches were blocked. The egress proxy returned 403 for arxiv.org, export.arxiv.org, ar5iv.arxiv.org, ijcai.org, dspace.cvut.cz and api.semanticscholar.org, and WebFetch could not resolve any of these hosts. As a result, no paper was read end to end. Each claim below carries one of these tags:
- **[V-abs]**: the abstract or official listing text of the primary source, quoted or closely paraphrased in search results.
- **[V-snip]**: a sentence or formula from the primary paper's body text, as surfaced by the search engine. I did not read it in its surrounding context.
- **[S]**: a secondary source, such as an aggregator summary (Lacuna, Liner, emergentmind) or a citing paper.
- **[U]**: unverified, from background recollection. Items tagged [U] are confined to the Gaps and Inferences sections.

Semantic Scholar's "cited-by" API was blocked, so the citation-graph searches were done through web-search proxies for those lists.

## Q1. CFR+IRA and FPIRA (Čermák, Lisý, Bošanský): criterion, split mechanics, cost, theorems, sampling

### Takeaway
CFR+IRA runs CFR+ over the **full unabstracted tree every iteration**. It also **randomly samples a subset of k_b original infosets** and monitors their immediate regret over a window of iterations. It splits an abstract infoset when a monitored original infoset's regret decreases more slowly than the no-regret rate. Because the monitored subset is random, the guarantee is a **probabilistic (with-probability-≥ p_r) bound on average external regret in the original perfect-recall game**. A heuristic split based on the action with the largest regret is added on top; it cannot hurt the bound, but on its own it is insufficient. FPIRA is the fictitious-play counterpart. Its best responses are computed in the original game, and its guarantee is inherited from FP. Neither method samples trajectories (MCCFR-style). The authors list the full-tree-traversal requirement as the main limitation.

### Cited Findings
**Bibliography and venues**
- The journal version is Čermák, Lisý, Bošanský, "Automated construction of bounded-loss imperfect-recall abstractions in extensive-form games," *Artificial Intelligence* 282:103248 (2020). [V-abs] — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370220300126); [arXiv 1803.05392](https://arxiv.org/abs/1803.05392)
- The IJCAI 2020 item is an **Extended Abstract** of the AIJ paper (pp. 5030–5034), not a separate full conference paper. The requester's label "IJCAI 2020" should be read with this in mind. [V-abs] — [IJCAI 2020 #701](https://www.ijcai.org/proceedings/2020/701); [PDF](https://www.ijcai.org/Proceedings/2020/0701.pdf)
- The predecessor that introduced FPIRA is Čermák, Bošanský, Lisý, "An Algorithm for Constructing and Solving Imperfect Recall Abstractions of Large Extensive-Form Games," IJCAI 2017, pp. 936–942, DOI 10.24963/IJCAI.2017/130. [V-abs] — [IJCAI 2017 PDF](https://www.ijcai.org/proceedings/2017/0130.pdf); [mlanthology](https://mlanthology.org/ijcai/2017/cermak2017ijcai-algorithm/)
- A related source is Čermák's 2018 CTU PhD thesis, *Solving Extensive-Form Games Using Imperfect Recall Abstractions*. I could not read it (fetch blocked). — [dspace.cvut.cz](https://dspace.cvut.cz/bitstream/handle/10467/75527/F3-D-2018-Cermak-Jiri-cermak.pdf)

**What the algorithms do (abstract-level)**
- Both algorithms "can start from an arbitrary or the coarsest abstraction." They "simultaneously solve the abstracted game, detect the missing information causing problems and refine the abstraction to include this information," repeating "until provable convergence to the desired approximation of the Nash equilibrium of the original game." Scope: two-player zero-sum perfect-recall EFGs. [V-abs/V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392); [AIC CTU page](https://www.aic.fel.cvut.cz/research-results/automated-construction-of-abstractions-in-games)
- Empirical headline: the methods can "closely approximate Nash equilibrium of large games using abstraction with only 0.9% of information sets of the original game." [V-abs] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- IJCAI 2017 FPIRA headline: it solved "a simplified poker game with 7·10^5 information sets using an abstracted game with only 1.8% of information sets of the original game." [V-abs] — [IJCAI 2017](https://www.ijcai.org/proceedings/2017/0130.pdf)
- Stated limitation: "the algorithms require traversing the complete unabstracted game tree." The future-work discussion calls this a notable limitation. [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392); [AIC CTU page](https://www.aic.fel.cvut.cz/research-results/automated-construction-of-abstractions-in-games)

**FPIRA (fictitious play + IR abstraction)**
- "The best responses are computed directly in G" (the original perfect-recall game). Refinement detection is "based on the difference between the quality of the strategies expected from running FP directly on the original … EFG with perfect recall and the result obtained from applying it to the abstraction." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- Guarantee: "the guarantee of convergence of FP to the Nash equilibrium of the original two-player zero-sum EFG with perfect recall directly translates to the guarantee of convergence of FPIRA to the Nash equilibrium of this game." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- FPIRA's purpose is to reduce memory for storing average strategies, by refining "only where the imperfect recall might cause problems." [V-abs] — [IJCAI 2017](https://www.ijcai.org/proceedings/2017/0130.pdf)

**CFR+IRA (CFR+ + IR abstraction)**
- Motivation: "replaces FP by the CFR+ algorithm since CFR+ is known to have a significantly faster empirical convergence." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- Refinement trigger: "the abstraction is refined when the observed convergence is slower than the theoretical guarantee provided by CFR+ in the original two-player zero-sum EFG with perfect recall." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- **Sampled monitoring.** The following sentences are paraphrased, nearly verbatim, from a search extraction of the paper. CFR+IRA "samples a subset of k_b unabstracted information sets of the original game," "checks the immediate regret in these monitored information sets for a well-chosen number of iterations before sampling a new subset." "If the immediate regret in any monitored information set decreases slower than guaranteed by the no-regret algorithm, the abstraction is too coarse and must be refined," and the monitored set is split off from its abstract infoset. The game is still "solve[d] … using CFR+, traversing the whole unabstracted game tree in each iteration." [V-snip, via search extraction] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392); [IJCAI 2020 PDF](https://www.ijcai.org/Proceedings/2020/0701.pdf)
- Guarantee statement: "Since CFR+IRA randomly samples information sets during the abstraction update, we provide a probabilistic bound on the average external regret." The authors first derive the bound for the regret-bound update alone, then extend it to the combination with the heuristic update. They conclude "CFR+IRA is guaranteed to converge to the Nash equilibrium of the original two-player zero-sum EFG with perfect recall." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- **Bound form (Lemma 3, from search extraction; formula not checked against the PDF):** R_i^T / T ≤ B_i(T) = (1/T)·(Δ·|I_i|·√A_max·√T + Δ·|I_i|·T_lim), holding "with probability at least p_r(T_lim, B_i(T))." Here p_r "is computed assuming the worst-case number of splits and the maximum possible number of iterations to split each information set, so it is a lower bound on the actual probability." The first term is the standard CFR+ √T bound over the original infosets I_i. The second term charges linear regret Δ for up to T_lim iterations before the needed splits are done. [V-snip, unverified exact notation] — [arXiv 1803.05392 §4.3.3](https://arxiv.org/pdf/1803.05392)
- **Heuristic update** (the requester's "best-action-set" split). "When using the heuristic abstraction update in combination with the regret bound update, Theorem 2 still holds, since in the worst case the heuristic update does not perform any splits and all the information set splits need to be performed by the regret bound update. Adding the heuristic abstraction update can only reduce the number of iterations needed." The heuristic "significantly improves the empirical convergence." [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- **The heuristic alone is insufficient.** In the paper's counterexample, the refinements I′ and I″ of an abstract infoset share the same largest-regret action index, so the heuristic never splits. "CFR+IRA in I converges to a uniform strategy with the average expected value 2.75 in both I′ and I″ since the regret updates lead to equal regrets for both c and d, while CFR+ converges to a strategy with average expected value 5." This means the heuristic groups original infosets by the identity of their best or largest-regret action, and it can miss **regret cancellation** that leaves the argmax unchanged. [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- Cost profile: CFR+IRA "requires at least an order of magnitude less memory than DOEFG and FPIRA." The IRA methods "often require more CPU time than solving the full game with CFR+ due to the overhead of the refinement process." [S] — [Lacuna summary](https://lacuna.tiptreesystems.com/work/automated-construction-of-bounded-loss-imperfect-recall-abstractions-in/wrk_45cb7e3ca88da3e9451ceece10758fdf)
- Origin of the idea: Čermák's AAMAS doctoral-consortium abstract planned "an algorithm, which given a perfect recall game creates its coarse imperfect recall abstraction" and "simultaneously solve[s] the abstraction and refine[s] the information set structure." [V-abs] — [ACM DL 10.5555/3091125.3091452](https://dl.acm.org/doi/pdf/10.5555/3091125.3091452)

**Why per-bucket regret is not enough (background the IRA criterion relies on)**
- Under perfect recall, external regret ≤ the sum of positive parts of immediate counterfactual regrets (Zinkevich et al.). Under imperfect recall, "the sum of counterfactual regrets no longer forms an upper bound on the external regret," so driving bucket-level regret to zero can still leave nonzero external regret in the original game. [V-snip] — [Lanctot et al. ICML 2012](https://arxiv.org/abs/1205.0622); [Čermák et al., arXiv 1608.01510](https://arxiv.org/pdf/1608.01510)

### Inferences
- The requester's design (hashed random subset of tracked original infosets, epoch windows) is **structurally the same monitoring device as CFR+IRA's k_b-sampled monitored set with a "well-chosen number of iterations" before resampling**. The novelty therefore cannot be claimed as "sampled monitoring of original infosets." The genuinely different elements would be:
  - (a) trajectory sampling (external-sampling MCCFR), as opposed to a full CFR+ traversal;
  - (b) the aggregate cancellation statistic H = L − C, as opposed to a per-monitored-infoset test of the immediate-regret decay rate;
  - (c) an almost-sure (as opposed to with-probability-≥ p_r) convergence claim.
- CFR+IRA's guarantee is a finite-T high-probability bound with a T_lim term. An a.s. claim needs a different argument. One route: refinement-only plus a finite game implies finitely many splits; then each original infoset must be monitored infinitely often (Borel–Cantelli over epochs) with growing windows, so that a persistent violation is detected a.s. This is a proof sketch, not a published result.
- With full-traversal CFR+IRA, per-original-infoset counterfactual regrets are exact in every iteration. Under external sampling they become unbiased but noisy estimates. Since |R̃^T − R^T|/T → 0 a.s. under bounded importance weights, positive-part and max functionals of cumulative sampled regrets are consistent but **biased upward at finite T** (Jensen). Split thresholds would need noise-aware inflation, analogous to MCCFR's (1/δ)·√(2/p) factor, or else false-positive splits occur. False positives are harmless for correctness in a refinement-only scheme, since they only cost memory, but they matter for the memory claim.

### Gaps
- I could not retrieve the exact definitions in 1803.05392 of the regret-bound check (the requester's "L_I(n)" notation, k_b, the window length and how it grows, T_lim), nor the verbatim statements of Theorem 1, Theorem 2 or Lemma 3, because full text was blocked. The Lemma 3 formula above comes from a search-engine extraction, and its notation should be re-checked against the PDF.
- **Whether regrets and average strategies are reset, copied or kept on a split in CFR+IRA could not be verified.** No search snippet addressed it. [U] My unverified recollection is that the children inherit the parent bucket's accumulated regret/strategy (copy), but this must be checked in §4 of the arXiv PDF before relying on it.
- I also could not verify whether the heuristic split uses cumulative regrets or current-iteration values of each original infoset (which affects memory, since per-original-infoset storage would erase the savings), or how FPIRA precisely groups original infosets when splitting (by best-response action?).
- The CTU thesis (which may include a sampling discussion) was not readable.

## Q2. Brown & Sandholm: "Simultaneous Abstraction and Equilibrium Finding in Games" (IJCAI 2015) and "Regret Transfer and Parameter Optimization" (AAAI 2014)

### Takeaway
Both papers refine **action** abstraction, not information abstraction, while CFR runs. The 2015 paper adds actions at run time without restarting. It does this by treating each new subgame as if it had always existed but had never been reached, and it warm-starts the new subgame's regrets with a "recovery game" or regret transfer. Its abstract claims anytime convergence to a Nash equilibrium of the full game, even for infinite games. The 2014 paper warm-starts CFR across payoff-perturbed games, using discounted transferred regrets, to run gradient descent on action parameters such as bet sizes. It calls itself the first action-abstraction algorithm with convergence guarantees. No evidence was found that either method is sampled (MCCFR) or that either refines information abstraction.

### Cited Findings
- **2015 citation:** Brown & Sandholm, "Simultaneous Abstraction and Equilibrium Finding in Games," IJCAI 2015, pp. 489–496. [V-abs] — [IJCAI PDF](https://www.ijcai.org/Proceedings/15/Papers/075.pdf); [CMU copy](https://www.cs.cmu.edu/~sandholm/simultaneous.ijcai15.pdf); [abstract page](https://ijcai.org/Abstract/15/075)
- **2015 abstract claims:** the method lets an agent "begin learning with a coarse abstraction" and add actions "at run time." It "can quickly — in constant time — add actions to the abstraction while provably not having to restart the equilibrium finding." It "enables anytime convergence to a Nash equilibrium of the full game even in infinite games." Experimentally it "outperform[s] fixed abstractions at every stage of the run": early on it improves as fast as a coarse abstraction, and "later it converges to a better solution than does equilibrium finding in fine-grained abstractions." [V-abs] — [IJCAI abstract](https://ijcai.org/Abstract/15/075); [Semantic Scholar listing](https://www.semanticscholar.org/paper/Simultaneous-Abstraction-and-Equilibrium-Finding-in-Brown-Sandholm/1dca537dca268059d333ffcc22a5684ce06dd17b)
- **2015 mechanism (body snippets):**
  - Goal: "preserve, as much as possible, what we have learned in Γ, without weakening CFR's long-term convergence guarantee."
  - A new subgame is treated as if it had been present all along, with the actions leading into it never played.
  - A "recovery game" initializes the new subgame's regrets.
  - Regret transfer applies only in special cases, because mismatched action counts "would imply that illegal actions were taken, which would invalidate the theoretical guarantees."
  - The rule for when to add an action assumes "regrets for actions not in the abstraction grow at rate ∼T while all other regrets grow at rate ∼√T," and adds an action if doing so would make summed average regret fall faster.
  [V-snip, via search extraction from the IJCAI PDF] — [IJCAI PDF](https://www.ijcai.org/Proceedings/15/Papers/075.pdf)
- **2014 citation:** Brown & Sandholm, "Regret Transfer and Parameter Optimization," AAAI 2014, pp. 594–601, DOI 10.1609/aaai.v28i1.8832. [V-abs] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8832); [mlanthology](https://mlanthology.org/aaai/2014/brown2014aaai-regret/)
- **2014 content (abstract level):**
  - It warm-starts no-regret learning: regrets from one game seed the solver on a structurally identical game with different payoffs, with the prior regrets discounted. This is extended to CFR.
  - It uses this warm start inside gradient descent over a parameter vector (e.g., poker bet sizes).
  - It describes the result as "the first action abstraction algorithm with convergence guarantees for extensive-form games."
  - Experiments are on no-limit Leduc and Texas Hold'em.
  [V-abs] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8832)
- **Follow-up for continuous actions:** Martin & Sandholm (2024), "Simultaneous incremental support adjustment and metagame solving," incrementally adjusts strategy support in continuous-action games. [V-abs, details not read] — [arXiv 2406.08683](https://arxiv.org/abs/2406.08683)

### Inferences
- For the requester, the transferable idea from 2015 is **how to make a refinement look like it was always there**. A split is guarantee-preserving if the refined game's regret can be decomposed as (pre-split regret, bounded) + (post-split regret, sublinear). This is the same accounting shape as CFR+IRA's T_lim term.
- The ∼T vs ∼√T growth heuristic in the 2015 paper is a *rate-based* detection criterion, the same family as CFR+IRA's "decreases slower than guaranteed."

### Gaps
- I did not verify whether the 2015 algorithm needs full-game best-response computations or full traversals to decide where to add actions [U: I recall full-game best-response use], whether it was run with vanilla CFR or a sampled variant, or its exact theorem statement and assumptions (beyond the abstract's "anytime convergence … even in infinite games"). Full text was blocked.
- The 2014 transfer formula (how regrets are scaled or discounted as a function of the payoff change) and its exact theorem were not retrieved.

## Q3. Other methods that change an abstraction while an equilibrium-finding algorithm runs (and CFR+IRA follow-ups)

### Takeaway
Besides CFR+IRA/FPIRA and the Brown–Sandholm action-insertion line, published "change-the-abstraction-during-solving" methods fall into five groups:
- (i) action-abstraction learning inside the solve (Hawkin et al. 2011/2012);
- (ii) learned or soft abstractions via function approximation (RCFR 2015 → Deep CFR; Embedding CFR, AAAI 2026);
- (iii) online clustering of infosets by expected values during real-time solving (EVPA, ICLR 2025), plus warm-up-then-cluster pre-solve abstractions (WEVA, 2026);
- (iv) strategy-space growth with guarantees (double oracle / XDO; Zhang–Sandholm certificates, including a *sampled black-box* version);
- (v) runtime blueprint patching in poker AIs (Libratus self-improver; nested subgame solving).

None of these, apart from CFR+IRA/FPIRA, refines an *information* abstraction during solving with a guarantee to an NE of the original perfect-recall game. No follow-up extending CFR+IRA was found.

### Cited Findings
**(i) Action abstraction learned during solving**
- Hawkin, Holte, Szafron (AAAI 2011), "Automated Action Abstraction of Imperfect Information Extensive-Form Games." Bet sizes are chosen by added "bet-sizing players" in a transformed multiplayer game. The authors prove that the NE of the transformed game corresponds to the NE of the abstract game with dynamically chosen bet sizes, solve it with a CFR-based regret minimizer, and evaluate on no-limit Leduc. [V-abs] — [AAAI 2011 PDF](https://www.cs.ualberta.ca/~games/poker/publications/AAAI11.pdf); [mlanthology](https://mlanthology.org/aaai/2011/hawkin2011aaai-automated)
- Hawkin, Holte, Szafron (AAAI 2012), "Using Sliding Windows to Generate Action Abstractions," adjusts the bet-size range of each bet individually across iterations. [V-abs] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/8401)

**(ii) Abstraction learned jointly with the equilibrium (function approximation)**
- Waugh, Morrill, Bagnell, Bowling (AAAI 2015), Regression CFR. "Learns a function approximator online to estimate the regret," and "both the abstraction as well as the equilibrium are learned during self-play." It comes with "a bound relating the quality of the function approximation and regret," which implies convergence to NE "so long as the regrets are ultimately realizable by the function approximator." [V-abs] — [arXiv 1411.7974](https://arxiv.org/abs/1411.7974); [UAlberta PDF](https://webdocs.cs.ualberta.ca/~bowling/papers/15aaai-rcfr.pdf)
- Fu, Liu, Xu, Huang (AAAI 2026), "No-Regret Strategy Solving … via Pre-Trained Embedding" (Embedding CFR). It replaces hard clustering with a low-dimensional continuous infoset embedding, and regret accumulation and strategy updates happen in the embedding space. The embedding is **pre-trained**, not refined during solving. [V-abs] — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/38736); [arXiv 2511.12083](https://arxiv.org/abs/2511.12083)

**(iii) Online or EV-based information abstraction in real-time solving**
- Li & Huang (ICLR 2025 spotlight), EVPA. It combines expected-value estimation based on approximate NE strategies, minimax pruning before CFR, and "dynamic online information abstraction" that "merges information sets by their current and future expected values in subgames." Evaluated on HUNL; it reportedly needs 1–2% of the solving time of a DeepStack replication and beats it and Slumbot. No convergence guarantee to the original game was found in the abstract. [V-abs] — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c1b5863a6b0f925617b917bb2f55be0-Abstract-Conference.html); [mlanthology](https://mlanthology.org/iclr/2025/li2025iclr-efficient-a/)
- Li & Huang (arXiv 2605.10900, May 2026), WEVA (Warm-up Expected Value-based Abstraction). It runs "a small number of CFR iterations on the full game as a warm-up," then clusters EV features into buckets. The abstraction is fixed before the solve, not refined during it. It reports exploitability reductions of up to over 80% versus equity- and rank-based abstractions. [V-abs/S] — [arXiv 2605.10900](https://arxiv.org/abs/2605.10900)
- Li & Huang (EMNLP Findings 2026), "Abstraction Agent." This is offline LLM-guided feature discovery plus k-means. It needs no tree traversal and is not a during-solve refinement. The same paper characterizes EVPA and WEVA as "still traversing the game tree." [V-abs] — [arXiv 2609.04303](https://arxiv.org/abs/2609.04303); [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1246/)
- Fu et al. KROI (arXiv 2403.11486) and FROI (arXiv 2510.15094, AAMAS 2026) are static, history-aware hand abstractions. Neither is a during-solve method. [V-abs] — [arXiv 2510.15094](https://arxiv.org/abs/2510.15094); [arXiv 2403.11486](https://arxiv.org/abs/2403.11486)

**(iv) Strategy-space or game-tree growth with guarantees (not information abstraction)**
- McAleer et al. (NeurIPS 2021), XDO. An extensive-form double oracle that "mixes best responses at every infostate" and is "guaranteed to converge to an approximate Nash equilibrium linearly in the number of infostates." [V-abs] — [arXiv 2103.06426](https://arxiv.org/abs/2103.06426)
- Zhang & Sandholm (AAAI 2021), "Finding and Certifying (Near-)Optimal Strategies in Black-Box Extensive-Form Games." It obtains **high-probability** exploitability certificates using a black box "that can do nothing more than play through games," with only a regret minimizer as a subroutine. It also gives an Õ(1/√T) equilibrium-finding algorithm that "does not need sampling strategies with lower-bounded reach probabilities, which MCCFR requires," and it expands only a small fraction of the tree. This is the **closest published sampled-plus-guarantee incremental method**, but it grows the explored game tree; it does not split information buckets. [V-abs] — [arXiv 2009.07384](https://arxiv.org/abs/2009.07384)
- Its predecessor, Zhang & Sandholm (NeurIPS 2020), "Small Nash Equilibrium Certificates in Very Large Games," assumes node expansion plus exact LP solves. [V-abs] — [arXiv 2006.16387](https://arxiv.org/pdf/2006.16387)

**(v) Poker-AI practice**
- Libratus has three modules: an MCCFR-solved blueprint abstraction, nested subgame solving, and a self-improver. The self-improver "analyzes opponents' bet sizes to detect potential holes in Libratus' blueprint strategy" and "adds these missing decision branches, computes strategies for them, and adds them to the blueprint." This is action-abstraction patching between or during play, with no global convergence claim. [S] — [IJCAI 2017 demo](https://www.ijcai.org/proceedings/2017/0772.pdf); [HPCwire](https://www.hpcwire.com/off-the-wire/carnegie-mellon-reveals-inner-workings-victorious-ai/)
- Pluribus uses a **fixed** blueprint information abstraction solved with Linear MCCFR plus pruning. With pruning, many infosets mapped to an abstract infoset "are traversed only 5% as often, so the strategy for the abstract infoset can better focus on generalizing across infosets that are likely to actually be encountered." This shows that sampling changes the effective weighting within buckets, but no refinement takes place. [V-snip] — [Pluribus supplement](https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf)
- Tartanian7 (2014) ran MCCFR within public-card clusters under a hierarchical, fixed abstraction. [V-snip] — [Brown, Ganzfried, Sandholm 2014](http://www.cs.cmu.edu/~sandholm/2014champion.pdf)

**Fixed-abstraction bounds (the theory a refinement scheme would build on)**
- Lanctot, Gibson, Burch, Zinkevich, Bowling (ICML 2012). For well-formed imperfect-recall abstractions, CFR regret bounds carry over. For skew well-formed abstractions, "there is no guarantee that regret will be minimized by CFR … However, we can still bound regret," and the bound tends to Σ_I |P̆(I)|·δ_I. They report a regret increase with skew δ in Skew-DRP. MCCFR-specific forms were not retrieved. [V-snip] — [arXiv 1205.0622](https://arxiv.org/abs/1205.0622); [author PDF](https://mlanctot.info/files/papers/12icml-ir.pdf)
- Kroer & Sandholm (EC 2016; arXiv 1409.3302), chance-relaxed skew well-formed (CRSWF) abstractions. These give "the first general, algorithm-agnostic solution-quality guarantees for Nash equilibria … computed in imperfect-recall abstractions, evaluated in the original perfect-recall game." [V-abs] — [arXiv 1409.3302](https://arxiv.org/abs/1409.3302)
- Kroer & Sandholm (NeurIPS 2018), unified framework. Bounded-counterfactual-regret strategies in CRSWF abstractions have error growing linearly in tree height, versus linearly in the number of infosets for Lanctot et al. [S, NeurIPS metadata summary] — [NeurIPS 2018 PDF](https://papers.neurips.cc/paper_files/paper/2018/file/aa942ab2bfa6ebda4840e7360ce6e7ef-Paper.pdf)
- Johanson et al. (AAAI 2012), CFR-BR, finds optimal abstract strategies against a full-game best responder. The abstraction is fixed, and the method relates to "evaluate-in-the-original-game" monitoring. [V-abs, title-level only] — [AAAI12 CFR-BR PDF](https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf)

**Single-agent analog: sampled search with progressive abstraction refinement**
- Hostetler, Fern, Dietterich (UAI 2015; JAIR 2017), PARSS. It "refines an initially coarse abstraction during search in order to match the abstraction granularity to the sample budget." A secondary source says nodes are split in half so that the abstraction converges to the original states in the limit. This applies to MDPs/MCTS, not games. [V-abs/S] — [UAI 2015 PDF](https://web.engr.oregonstate.edu/~tgd/publications/hostetler-fern-dietterich-progressive-abstraction-refinement-for-sparse-sampling-uai2015.pdf); [JAIR 2017](https://www.jair.org/index.php/jair/article/view/11096)

**Search for CFR+IRA follow-ups**
- Searches combining "CFR+IRA"/"FPIRA" with "cited," "follow-up," "extend," "sampling" and "Monte Carlo" returned only the original papers, aggregator summaries, and CTU pages. No follow-up paper that extends CFR+IRA, in particular to sampling, was found. [search result] — [search hit list incl. AIJ](https://www.sciencedirect.com/science/article/pii/S0004370220300126)

### Inferences
- Relative to the requester's mechanism, the nearest relatives are:
  - **CFR+IRA**: the same split criterion family and the same sampled monitoring, but with full traversal.
  - **Brown–Sandholm 2015**: the same "insert refinement without restart" accounting, but for action abstraction.
  - **Zhang–Sandholm 2021**: sampled play-through with a high-probability guarantee, but for tree expansion rather than bucket splitting.
  - **RCFR**: a learned abstraction, with a guarantee conditional on realizability.
- EVPA is the only recent method that changes the *information* abstraction during solving, and it optimizes speed, not provable convergence to the original NE.

### Gaps
- A real "cited-by" enumeration of 1803.05392 / AIJ 2020 (Semantic Scholar or Google Scholar) could not be run. The proxy blocked api.semanticscholar.org and scholar pages. It is possible, though no search surfaced one, that a citing paper contains a sampled IRA variant as a minor contribution.
- Not searched or verified: Moravčík et al. (AAAI 2016) "Refining subgames in large imperfect information games," and Bošanský et al.'s double-oracle-for-EFG (JAIR 2014) details [U]. Both are strategy- or subgame-refinement methods, not information-abstraction refinement.
- I could not verify whether Hawkin et al.'s transformed-game approach converges to an NE of the *original* (unabstracted) game or only of the dynamically abstracted game. The abstract says the latter.
- I did not verify whether "Monte Carlo Tree Search with Iteratively Refining State Abstractions" (NeurIPS 2021) contains sampling-plus-refinement guarantees. Only its title was seen ([poster](https://nips.cc/virtual/2021/poster/27913)), and it is single-agent in any case.

## Q4. Does any published SAMPLED (Monte Carlo) refinement-with-guarantee to an NE of the ORIGINAL game exist?

### Takeaway
**No.** After the searches listed below, I found no published method that changes (splits or refines) an *information* abstraction during **trajectory-sampled** CFR (MCCFR: outcome or external sampling) and proves convergence (a.s. or with high probability) to a Nash equilibrium of the original perfect-recall game. The closest items each miss one ingredient:
- CFR+IRA samples *which original infosets to monitor* and gives a probabilistic bound, but it traverses the full tree with CFR+ every iteration.
- Zhang & Sandholm (AAAI 2021) is sampled with a high-probability guarantee, but it grows the game tree and is not a bucket-splitting method.
- Lanctot et al. (2012) covers MCCFR on a fixed imperfect-recall abstraction, with no refinement.
- Brown & Sandholm (2015) refines during solving with a full-game guarantee, but it inserts actions.

The authors of CFR+IRA themselves flag full-tree traversal as the key limitation.

### Cited Findings
- CFR+IRA and FPIRA "require traversing the complete unabstracted game tree," which the future-work section calls a notable limitation. [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392); [AIC CTU](https://www.aic.fel.cvut.cz/research-results/automated-construction-of-abstractions-in-games)
- CFR+IRA's only randomness is in *sampling the monitored infosets*, and hence its bound is probabilistic. [V-snip] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392)
- The closest sampled game-solving result with a guarantee is a high-probability certificate from play-throughs only, using a regret minimizer, without MCCFR's lower-bounded-reach requirement. It is tree expansion, not information-abstraction refinement. [V-abs] — [Zhang & Sandholm, arXiv 2009.07384](https://arxiv.org/abs/2009.07384)
- MCCFR background:
  - Outcome and external sampling have probabilistic regret bounds. [V-snip] — [Lanctot et al. NeurIPS 2009](http://papers.neurips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
  - Tighter MCCFR bounds were given by Gibson et al. (2012). [V-snip] — [NeurIPS 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf)
  - ESCHER gives a "with high probability" guarantee, not an a.s. one. [S] — [ESCHER ICLR 2023](https://www.mit.edu/~gfarina/2023/escher_iclr23/2206.04122.pdf)
- A search-engine synthesis for "Monte Carlo CFR abstraction refinement during training convergence guarantee sampled imperfect recall" explicitly returned no paper combining sampled MCCFR with abstraction refinement and a convergence proof. The closest matches it surfaced were CFR+IRA (CFR+, not MCCFR) and Lanctot et al. (2012; fixed abstraction). [search result] — [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392); [Lanctot 2012](https://arxiv.org/abs/1205.0622)

**Searches performed** (WebSearch, standard or extended mode, 2026-10-26). Full-text fetch was blocked for arxiv, ijcai, dspace.cvut.cz and semanticscholar.
1. "Čermák Lisý Bošanský imperfect recall abstraction refinement CFR+IRA FPIRA"
2. "CFR+IRA abstraction update regret bound information set split original game traversal"
3. "CFR+IRA monitored information sets immediate regret sampled subset probabilistic bound average external regret"
4. "FPIRA fictitious play imperfect recall abstraction best response refinement convergence Nash equilibrium original game"
5. "Monte Carlo CFR abstraction refinement during training convergence guarantee sampled imperfect recall"
6. "CFR+IRA OR FPIRA cited abstraction refinement imperfect-recall follow-up"
7. "CFR+IRA heuristic update OR regret bound update … merge split"
8. "CFR+IRA Theorem 2 probabilistic bound … split regrets initialized"
9. "dynamic information abstraction refine buckets during MCCFR training poker split infoset regret variance"
10. "adaptive abstraction refinement counterfactual regret minimization 2025 2026 arXiv online split information sets during solving"
11. "imperfect recall abstraction regrets cancel merged information sets bucket counterfactual regret sum positive part lost"
12. "Čermák thesis … future work sampling Monte Carlo CFR traversing complete game tree"
13. "MCCFR external sampling abstraction refinement split buckets convergence almost surely Nash equilibrium original game imperfect recall refinement-only"
14. Further queries on: Brown–Sandholm 2014 and 2015 (three queries); Li–Huang EVPA, WEVA and Abstraction Agent; Embedding CFR; FROI; Zhang–Sandholm certificates; Lanctot 2012; Kroer–Sandholm 2016; RCFR; Hawkin et al.; the Libratus self-improver; PARSS; XDO.

None of these surfaced a sampled refinement method with a guarantee.

### Inferences
- **Positioning for the requester:**
  - Full-traversal special case: their scheme is close to CFR+IRA.
  - Monitoring device: hashed or random subset of original infosets with epoch windows. This is also anticipated by CFR+IRA, which uses k_b sampled monitored infosets resampled after a chosen number of iterations.
  - What appears unpublished:
    - (1) estimating the refinement statistic from **external-sampling MCCFR trajectories** (importance-weighted sampled counterfactual regrets) instead of exact CFR+ values;
    - (2) using the **aggregate cancellation gap H = L − C** as the trigger, instead of CFR+IRA's per-infoset immediate-regret decay test or its largest-regret-action heuristic;
    - (3) an **almost-sure** convergence statement, as opposed to CFR+IRA's with-probability-≥ p_r bound with worst-case split accounting.
  - The CFR+IRA counterexample (equal regrets on c and d, so the argmax heuristic never splits even though the value is 2.75 vs 5) is a ready-made motivating example for why a cancellation-magnitude statistic like H is needed.
- **Proof obligations the literature suggests:**
  - (a) L (sum over original infosets of positive cumulative cf regret) upper-bounds original-game external regret under perfect recall. C does not under imperfect recall (Lanctot 2012; arXiv 1608.01510). The argument must therefore drive L/T → 0, not C/T → 0.
  - (b) Refinement-only in a finite game means finitely many splits. The CFR+IRA-style accounting (a √T term plus a bounded pre-split linear term, cf. the T_lim term) then applies after the last split, provided the final partition's in-bucket cancellation vanishes.
  - (c) For an a.s. claim with hashed subsets, every original infoset must be tracked in infinitely many epochs (or always tracked with positive probability), with growing epoch windows. MCCFR regret noise must be controlled by a martingale SLLN, which requires bounded importance weights, i.e., the lower-bounded sampling probabilities that external sampling has.
  - (d) Under RM+ (CFR+), regrets are thresholded at 0 each iteration. The "cumulative regret" entering L and C is then the RM+ Q-value, not the raw sum. CFR+IRA's bound uses CFR+'s √T guarantee, so check which quantity the CFR+IRA test monitors before claiming equivalence. Not verified.
- **Evaluation wording:** these are regret/exploitability statements about the original game. The project's Local BR-gap metric must not be conflated with them (per AGENTS.md).

### Gaps
- Absence-of-evidence caveat: Google Scholar and Semantic Scholar "cited-by" lists for 1803.05392 and AIJ 282:103248 could not be enumerated directly because of blocked hosts. A sampled IRA variant could exist inside a citing thesis or workshop paper that web search does not index well.
- Recommended manual check: open the Semantic Scholar citation list for arXiv 1803.05392 and the CTU thesis future-work chapter.
- Exact CFR+IRA parameters (k_b, the window schedule, T_lim, the definition of p_r) and the split-time regret handling remain unverified. They should be read from §4 of [arXiv 1803.05392](https://arxiv.org/pdf/1803.05392) before the requester's paper claims a difference from or an extension of CFR+IRA.
