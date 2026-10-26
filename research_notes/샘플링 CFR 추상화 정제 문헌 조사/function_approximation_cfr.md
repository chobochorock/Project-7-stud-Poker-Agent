# Function-approximation variants of CFR: what is approximated, assumptions, guarantees, and how novel a regret-space VQ/embedding abstraction is

> **Verification legend (applies to every bullet).**
> **[VD]** = verified directly: I read the primary artifact myself (official GitHub README).
> **[VS]** = verified from the primary abstract or listing page (arXiv, AAAI OJS, ERA, ACL Anthology) as quoted or paraphrased by the web-search tool. I could not open the page myself.
> **[3P]** = from a third-party summary (papernotes.org, Lacuna, Pith) found by search. This is weaker evidence.
> **[U]** = unverified in this session. It comes from my prior knowledge of the paper and should be checked against the PDF at the link.
> **Access caveat:** this session's egress policy blocked arxiv.org, export.arxiv.org, ojs.aaai.org, proceedings.mlr.press, openreview.net, ifaamas.org, ualberta.ca hosts, semanticscholar.org, huggingface.co and archive.org. Only GitHub was reachable. **No theorem statement below was checked against a PDF.** Exact constants marked [U] must be confirmed before they are quoted in a report.
> Search date: 2026-10-26.

## Q1. RCFR: Waugh, Morrill, Bagnell, Bowling, "Solving Games with Functional Regret Estimation" (AAAI 2015, arXiv 1411.7974)

### Takeaway
RCFR is the original "regress the cumulative regret, then regret-match on the estimate" algorithm. Its guarantee has a usual O(√T) regret term plus a term that grows with regression error, so it converges to Nash in self-play when the regrets are realizable by the approximator. The authors present it as a principled generalization of abstraction, in which the abstraction is learned during self-play. A hard-assignment codebook that stores a per-code regret is the special case of RCFR with one-hot (indicator) features.

### Cited Findings
- **Bibliographic [VS].** Kevin Waugh, Dustin Morrill, J. Andrew Bagnell, Michael Bowling. AAAI 2015, pp. 2138–2145, DOI 10.1609/aaai.v29i1.9445. — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9445); [arXiv 1411.7974](https://arxiv.org/abs/1411.7974); [mlanthology](https://mlanthology.org/aaai/2015/waugh2015aaai-solving/)
- **What is approximated [VS].** "The approach learns a function approximator online to estimate the regret for choosing a particular action." A no-regret algorithm (regret matching) uses these estimates in place of the true regrets to define the sequence of policies. — [arXiv abs](https://arxiv.org/abs/1411.7974); [Semantic Scholar listing](https://www.semanticscholar.org/paper/Solving-Games-with-Functional-Regret-Estimation-Waugh-Morrill/28386ad2b37548a480ff43b94fef33ac132f20b4)
- **Guarantee, qualitative [VS].** The authors "prove the approach sound by providing a bound relating the quality of the function approximation and regret of the algorithm". A corollary is convergence to a Nash equilibrium in self-play if the regrets can be realized by the approximator. The abstract's wording is "ultimately realizable" [U]. — [arXiv abs](https://arxiv.org/abs/1411.7974); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/9445)
- **Abstraction as a special case [VS].** The technique "can be understood as a principled generalization of existing work on abstraction in large games". In RCFR "both the abstraction as well as the equilibrium are learned during self-play". — [arXiv abs](https://arxiv.org/abs/1411.7974)
- **Empirical claim [U].** The abstract says RCFR gets higher-quality (less exploitable) strategies than state-of-the-art abstraction techniques given the same resources. — [arXiv abs](https://arxiv.org/abs/1411.7974)
- **Algorithm details [U].**
  - Each iteration computes exact counterfactual regrets with a full (vanilla CFR) tree walk, so there is no MCCFR sampling in the original paper.
  - The regressor's targets bootstrap the previous estimate: (previous predicted cumulative regret) + (new instantaneous regret) for each (I,a). Features are φ(I,a). The fit uses squared loss.
  - The policy is regret matching on the positive part of the predictions, with a uniform fallback.
  - Experiments use small poker games (Leduc-like variants) with a tree or linear regressor over hand features.
  - — [arXiv 1411.7974](https://arxiv.org/abs/1411.7974); [author PDF](https://webdocs.cs.ualberta.ca/~bowling/papers/15aaai-rcfr.pdf)
- **Bound form [U].** The bound is total regret ≤ (tabular CFR term, O(|I|·Δ·√|A|·√T)) + (a term that is linear in T and grows like the square root of the per-iteration regression error). As a result the average regret has a non-vanishing floor that grows with √(error). D'Orazio et al. 2020 state that their analysis "recover[s] a tighter bound for RCFR" [VS], so the original bound is not tight. I could not obtain the exact constants or the norm used for the error. — [arXiv 1411.7974](https://arxiv.org/abs/1411.7974); [f-RCFR arXiv 1912.02967](https://arxiv.org/abs/1912.02967)
- **Morrill MSc thesis, "Using Regret Estimation to Solve Games Compactly" (U Alberta, Spring 2016) [VS].**
  - It introduces and analyzes RCFR. "The regressor essentially becomes a tunable, compact, and dynamic abstraction of the game that is informed by and adapts to the particular solution being computed."
  - It combines RCFR with regret-matching+ (RCFR+).
  - Experiments use two small poker games. "RCFR variants tend to produce less exploitable strategies" than "traditional static abstractions of similar complexity."
  - — [thesis PDF](https://poker.cs.ualberta.ca/publications/Morrill_Dustin_R_201603_MSc.pdf); [ERA record](https://era.library.ualberta.ca/items/51264e8e-5123-4429-b91b-77efa5ea38ce)
- **Average strategy [U].** The original RCFR experiments keep the average strategy outside the regressor, accumulated exactly in the small games. Approximating the average policy is not covered by the RCFR bound. Later neural methods handle it explicitly (see Q3). — [arXiv 1411.7974](https://arxiv.org/abs/1411.7974)

### Inferences
- **Indicator features reduce RCFR to abstraction (my derivation, consistent with the [VS] "generalization of abstraction" claim).**
  - Take φ(I,a) = one-hot(bucket(I)) ⊗ one-hot(a) and fit with squared loss. The least-squares prediction for every I in bucket B is the sample-weighted mean of the members' targets.
  - That mean equals (bucket's summed regret)/(count). This is a positive scalar that is the same for every action of B, and regret matching ignores it.
  - So RCFR with indicator features plays exactly CFR (or MCCFR) on the imperfect-recall abstraction.
  - The requester's "code position = running mean of sampled regret with step 1/n" is this least-squares solution computed online. With a fixed hard assignment, a code's regret is therefore the bucket's MCCFR regret up to a per-code positive scale, as the requester claims.
- **Realizability becomes concrete under hard assignment.** The approximation error at iteration t is the deviation of each member's cumulative-regret vector from its code's (scaled) mean.
  - Bound-relevant error is zero only if all members of a code have proportional cumulative-regret vectors.
  - Behaviour-relevant error is zero if members share the same normalized positive-regret vector, because then regret matching gives the same policy.
  - The RCFR-style bound charges the first (scale-sensitive) error, so it can be loose for buckets whose members play identically.

### Gaps
- Exact statement of the RCFR theorem: constants, error norm (L2 on R vs R₊?), whether the error is per infoset or summed, and whether it is per iteration or averaged. The PDF hosts were blocked.
- Whether RCFR's experiments used regression trees or linear models. Exact features, games and bucket counts of the abstraction baselines.
- Whether the 2015 paper explicitly writes out "indicator features = abstraction" as a formal special case or only says "generalization" in prose.

## Q2. f-RCFR: D'Orazio, Morrill, Wright, Bowling, "Alternative Function Approximation Parameterizations for Solving Games: An Analysis of f-Regression Counterfactual Regret Minimization" (AAMAS 2020, arXiv 1912.02967)

### Takeaway
f-RCFR generalizes RCFR's "regret-match on approximated regrets" to any link function f, chosen through (Φ,f)-regret matching. It proves approximation-error-aware regret bounds that tighten the RCFR bound, and gives exploitability bounds for polynomial and exponential (softmax) links. Empirically, ReLU (normalized positive part) wins when approximation error is small and softmax can win when error is large. This matters for soft or distance-weighted code assignments, where the error can be large.

### Cited Findings
- **Bibliographic [VS].** AAMAS 2020. arXiv 1912.02967 was first submitted on 6 Dec 2019 and revised on 1 May 2020. — [arXiv abs](https://arxiv.org/abs/1912.02967); [AAMAS PDF p339](https://ifaamas.org/Proceedings/aamas2020/pdfs/p339.pdf)
- **Motivation [VS].** RCFR uses "normalized rectified linear unit (ReLU) parameterized policies". Softmax is standard in RL and in the tabular case "yields a regret bound with a better dependence on the number of actions". — [arXiv abs](https://arxiv.org/abs/1912.02967)
- **Theory [VS].**
  - They derive "approximation error-aware regret bounds for (Φ, f)-regret matching, which applies to a general class of link functions and regret objectives".
  - These bounds "recover a tighter bound for RCFR and provide a theoretical justification for RCFR implementations with alternative policy parameterizations (f-RCFR), including softmax".
  - They "provide exploitability bounds for f-RCFR with the polynomial and exponential link functions in zero-sum imperfect information games".
  - — [arXiv abs](https://arxiv.org/abs/1912.02967)
- **Empirics [VS].** "The previously studied ReLU parameterization performs better when the approximation error is small while the softmax parameterization can perform better when the approximation error is large." — [arXiv abs](https://arxiv.org/abs/1912.02967)
- **Bound details [U].**
  - The bounds sit in the paper's Section 2.3 [VS: search snippet referenced "bounds presented in Section 2.3"].
  - For the polynomial link (p-norm regret matching, with ReLU as the p=2 case), the error enters inside the square root as a sum over iterations of a norm of (true cumulative regret − approximation). This is additive under the √, not an additive linear-in-T term outside it.
  - For the exponential link, a temperature and log|A| dependence appears.
  - Exact norms and constants were not verified.
  - — [arXiv 1912.02967](https://arxiv.org/abs/1912.02967)
- **Setting [U].** Full-traversal CFR with the cumulative regret regressed, as in RCFR. Average strategy outside the approximator. Experiments on small poker and goofspiel-type games where approximation error was controlled by the feature set. — [arXiv 1912.02967](https://arxiv.org/abs/1912.02967)

### Inferences
- For the requester's soft or distance-weighted assignment, the policy link is a design axis separate from the code-regret estimator. f-RCFR suggests trying softmax on code-interpolated regrets when codebooks are small, which means large approximation error.
- Because f-RCFR bounds are stated for a generic regret estimate, they apply unchanged to a codebook estimator. The open task is measuring the estimator's error against the true per-infoset cumulative regret.

### Gaps
- Verbatim theorem statements and constants. Which norm (L1, Lp or L∞) the error is measured in. Whether the error must be bounded per infoset and per iteration or only on average.
- The exact games and approximators used in the experiments.

## Q3. Neural CFR family (Deep CFR, SD-CFR, DNCFR, DREAM, ESCHER) and 2024–2026 successors

### Takeaway
Every neural CFR descendant regresses some regret-like quantity: sampled instantaneous advantages averaged with linear weights (Deep CFR, SD-CFR, DREAM), cumulative regret directly (DNCFR), or IS-free regret estimates from a history value function (ESCHER). They differ mainly in:
- **Sampling scheme:** external sampling (Deep CFR, SD-CFR); robust sampling (DNCFR); outcome sampling with learned baselines (DREAM, VR-DeepDCFR+); fixed-policy sampling with no IS (ESCHER).
- **Average-strategy handling:** a separate network (Deep CFR, DNCFR, ESCHER) or exact reconstruction from stored per-iteration models (SD-CFR, DREAM).

The canonical guarantee (Deep CFR Thm 1) has an O(√T) term plus a term linear in T and proportional to √(excess regression loss). Average regret therefore plateaus at a level ∝ √ε, and the bound does not cover average-policy-network error.

### Cited Findings

#### Deep CFR: Brown, Lerer, Gross, Sandholm (ICML 2019, arXiv 1811.00164)
- **Motivation [VS].** Deep CFR uses neural networks to approximate CFR in the full game. It argues that abstraction is often manual, can miss strategic nuances, and has a chicken-and-egg problem. — [PMLR PDF](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf); [arXiv](https://arxiv.org/abs/1811.00164)
- **Abstraction baseline [VS].** The paper describes its abstraction baselines with: "The 40,000 cluster abstraction means that the more than 10^9 different decisions in the game were clustered into 40,000 abstract decisions, where situations in the same bucket are treated identically". The baseline buckets come from k-means on domain-specific features. — [PMLR PDF](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)
- **What is approximated, and the loss [U].**
  - The advantage network V(I,a|θ_p) is retrained from scratch each iteration on a reservoir buffer M_V of sampled instantaneous regrets r̃^{t'}(I,a).
  - Loss: E_{(I,t',r̃)∼M_V}[ t' · Σ_a (r̃^{t'}(a) − V(I,a|θ))² ], which is MSE with Linear-CFR weight t'.
  - The minimizer is proportional to the linear-weighted cumulative regret. Proportionality is enough for regret matching.
  - Policy: regret matching on V₊. If no V is positive, it plays the argmax action.
  - — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)
- **Sampling [U].** External-sampling MCCFR with K traversals per player per iteration. — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)
- **Average strategy [VD for the existence of the "AVG_NET" method; U for details].** A separate average-policy network is trained on a reservoir buffer of (I, t, σ^t(I)) with t-weighted MSE. — [DREAM repo README](https://github.com/EricSteinberger/DREAM) ("AVG_NET is the method used by Deep CFR"); [arXiv](https://arxiv.org/abs/1811.00164)
- **Guarantee [U; recalled verbatim, verify].** "Let L^t_V − L^t_{V*} ≤ ε_L", where ε_L is the gap between the network's average MSE on the buffer and the minimum achievable loss. "If the value memories are sufficiently large, then with probability 1−ρ", R^T_p ≤ (1 + √2/√(ρK))·Δ·|I_p|·√|A|·√T + 4T·|I_p|·√(|A|·Δ·ε_L). Corollary: average regret → at most 4|I_p|√(|A|Δε_L).
  - The error enters as the square root of the excess loss, multiplied by T. It is therefore a non-vanishing floor.
  - Assumptions: the buffers are representative ("sufficiently large" reservoir), the error is excess loss relative to the best function rather than absolute error, and the average-policy network is not covered.
  - — [arXiv 1811.00164](https://arxiv.org/abs/1811.00164)
- **Games [U].** Leduc for exploitability. Heads-up flop hold'em (FHP) and heads-up limit hold'em, head-to-head against NFSP and k-means abstraction baselines. — [arXiv](https://arxiv.org/abs/1811.00164)

#### Single Deep CFR (SD-CFR): Steinberger (arXiv 1901.07621, 2019)
- **No average network [VD].** The author's README says the "Single" averaging method "introduced in [SD-CFR]… is generally better", and that "SD-CFR is expected to perform better, is faster, and requires less memory". Experiments: exploitability in Leduc and BigLeduc, head-to-head against Deep CFR, and "the effect of reservoir sampling on B^M with various capacities". — [Deep-CFR repo README](https://github.com/EricSteinberger/Deep-CFR); [DREAM repo README](https://github.com/EricSteinberger/DREAM)
- **How averaging works [U].**
  - SD-CFR stores every iteration's advantage network in a buffer B^M.
  - It recovers the linear-average strategy exactly. For play, it samples one iteration's network with probability ∝ t at the start of each episode (trajectory-level sampling). For queries at I, it weights σ^t(I) by t·π^t_p(I).
  - This removes the average-network approximation error from Deep CFR's pipeline. The regret-side bound is inherited from Deep CFR.
  - — [arXiv 1901.07621](https://arxiv.org/abs/1901.07621)

#### Double Neural CFR (DNCFR): Li, Hu, Ge, Jiang, Qi, Song (ICLR 2020, arXiv 1812.10607)
- **Design [U].**
  - Two networks: a cumulative-regret network (RSN) and an average-strategy network (ASN). Both are recurrent (LSTM with attention) over the infoset's action sequence.
  - Sampling uses "robust sampling", where the traverser samples k actions per infoset. k=1 is outcome-like and k=|A| is external sampling. It runs inside mini-batch MCCFR, optionally with an MCCFR+-style clip.
  - Networks are warm-started from the previous iteration instead of retrained from scratch.
  - The theory shows unbiasedness and variance properties of robust sampling and mini-batch MCCFR in the tabular case. There is no function-approximation error bound comparable to Deep CFR's.
  - Games: Leduc and HUNL subgames.
  - — [arXiv 1812.10607](https://arxiv.org/abs/1812.10607)

#### DREAM: Steinberger, Lerer, Brown (arXiv 2006.10410, 2020)
- **Scope [VD].** DREAM ("Deep Regret minimization with Advantage baselines and Model-free learning") was evaluated on Flop Hold'em Poker (FHP) and Leduc. It uses the SD-CFR "Single" averaging method ("Single is the method introduced in [SD-CFR] and also used in DREAM"). — [DREAM repo README](https://github.com/EricSteinberger/DREAM)
- **Method [U].**
  - Model-free outcome sampling, with an exploration mixture for the traverser and on-policy play for the opponent. No resets to arbitrary histories are needed.
  - A learned history-action Q baseline (VR-MCCFR-style control variate) reduces outcome-sampling variance.
  - The advantage network is trained Deep CFR-style on baseline-corrected sampled advantages.
  - The paper claims convergence to Nash in two-player zero-sum games under its assumptions.
  - — [arXiv 2006.10410](https://arxiv.org/abs/2006.10410)

#### ESCHER: McAleer, Farina, Lanctot, Sandholm (ICLR 2023, arXiv 2206.04122)
- **Claim [VD].** "ESCHER is an unbiased model-free method for finding approximate Nash equilibria in large two-player zero-sum games that does not require any importance sampling. ESCHER is principled and is guaranteed to converge to an approximate Nash equilibrium with high probability in the tabular case." It "outperforms the prior state of the art—DREAM and neural fictitious self play (NFSP)—and the difference becomes dramatic as game size increases." — [ESCHER repo README](https://github.com/Sandholm-Lab/ESCHER); [arXiv](https://arxiv.org/abs/2206.04122); [ICLR 2023 listing](https://mlanthology.org/iclr/2023/mcaleer2023iclr-escher/)
- **Mechanism [VS].** It uses a history value function and a fixed sampling strategy for the updating player, which avoids importance sampling. The appendix proof starts from the decomposition of total regret into per-infoset counterfactual regrets, then adapts an argument from Farina et al. (2020) to bound how regret degrades when counterfactual regrets are estimated. — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122)
- **Why no IS is needed [U].** With a fixed (time-invariant) sampling policy for the traverser, the expected un-reweighted estimate q̂(h,a) − v̂(h), summed over h ∈ I, equals the counterfactual regret multiplied by the fixed policy's reach of I. That is a time-invariant positive scale per infoset, which regret matching ignores. A regret network regresses these estimates. A separate average-policy network is trained. — [arXiv 2206.04122](https://arxiv.org/abs/2206.04122)

#### 2024–2026 successors
- **VR-DeepDCFR+ / VR-DeepPDCFR+, "Deep (Predictive) Discounted CFR" [VS].**
  - Authors: Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng. arXiv 2511.08174 (Nov 2025), AAAI 2026.
  - These are model-free neural versions of DCFR+ and PDCFR+.
  - Each iteration collects K outcome-sampled episodes. It computes variance-reduced sampled advantages with a DREAM-inspired value-network baseline, then updates a **cumulative** advantage network by bootstrapping with discounting and clipping.
  - Reported results: faster convergence on typical IIGs and stronger results in a large poker game than model-free neural baselines.
  - — [arXiv abs](https://arxiv.org/abs/2511.08174); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/38780/42742); [papernotes, 3P](https://en.papernotes.org/AAAI2026/reinforcement_learning/deep_predictive_discounted_counterfactual_regret_minimization/)
- **Robust Deep MCCFR [VS].** Z. El Jaafari, arXiv 2509.00923 (Aug 2025). It analyzes the risks of neural MCCFR (non-stationary target distribution shift, action-support collapse, variance explosion, warm-starting bias) and how they scale with game size. — [arXiv html](https://arxiv.org/html/2509.00923v1)
- **RL-CFR (arXiv 2403.04344) [VS, title only].** Uses RL to improve **action** abstraction, not information abstraction. Peripheral here. — [arXiv PDF](https://arxiv.org/pdf/2403.04344)

### Inferences
- **The requester's 1/n running mean has a precedent in Deep CFR.** Deep CFR already relies on "mean ∝ sum is enough for regret matching". Its minimizer is a (weighted) mean of sampled regrets.
  - Under external sampling, the conditional mean of r̃ for infoset I in the buffer is Σ_t t·r^t(I) / Σ_t t·q^t(I), where q^t(I) is the sampling reach. This is a per-infoset positive scale.
  - The same argument shows that a hard code's running mean over visits is ∝ the bucket's summed MCCFR regret.
- **Error floors.** If a code's estimator is plugged into the Deep CFR / RCFR analysis, ε becomes the codebook's excess squared error at representing members' cumulative regrets. Average regret then plateaus at a level ∝ √ε, the same qualitative statement as for a k-means abstraction.
- **Average strategy in a codebook.** A codebook is tiny (k×|A|), so SD-CFR-style exact averaging is cheap: store a snapshot per iteration, or accumulate a reach-weighted strategy sum per code. Storing snapshots is the safe choice if codes or encoder move during training. A per-code strategy sum would otherwise mix strategies of infosets that later change bucket.
- **Sampling scale caveat.**
  - ESCHER-style fixed-policy sampling and outcome-sampling IS weights each create per-infoset scale factors. Inside a code they act as member weights, so the code regret becomes ∝ Σ_{I∈B} w(I)·R(I), not the plain bucket sum. This matters when comparing to "bucket MCCFR regret".
  - External sampling gives an unbiased plain sum.
  - Separately, soft assignment with per-code 1/n_k normalization breaks proportionality across codes. Σ_k w_k(I)·R_k/n_k is not ∝ Σ_k w_k(I)·R_k when the n_k differ, so the 1/n "mean" choice changes the soft policy.

### Gaps
- Verbatim theorem statements for Deep CFR, DNCFR, DREAM and ESCHER, especially ESCHER's bound in terms of history-value error. PDFs were blocked.
- Exact game lists and numbers for DNCFR and ESCHER.
- Unverified leads not checked this session:
  - Hierarchical Deep CFR (Chen, Lan, Aggarwal, NeurIPS 2023, believed arXiv 2305.17327).
  - ARMAC (Gruslys et al. 2020).
  - Neural ReCFR-B and other bootstrapped neural CFR variants.
  - Any 2026 neural-CFR paper after Nov 2025 beyond those listed.

## Q4. Embedding CFR: Fu, Liu, Xu, Huang, "No-Regret Strategy Solving in Imperfect-Information Games via Pre-Trained Embedding" (AAAI 2026, arXiv 2511.12083)

### Takeaway
Embedding CFR replaces hard k-means buckets with a **pre-trained, frozen** continuous embedding of information sets (hands), produced by a network called HandEbdNet. It then runs regret accumulation and strategy updates "in the embedding space" at the same memory budget as clustering.

Its embedding signal is **win-rate (equity) distribution supervision, not regret**. This is the key difference from the requester's encoder, which is trained into regret space.

Its guarantee is described only as a theoretical analysis showing the method reduces cumulative regret. I could not obtain the update rule, the theorem statement, the sampling scheme or the averaging method.

### Cited Findings
- **Bibliographic [VS].** Yanchang Fu, Shengda Liu, Pei Xu, Kaiqi Huang. arXiv 2511.12083, posted Nov 2025, v2 in Dec 2025. AAAI-26. — [arXiv abs](https://arxiv.org/abs/2511.12083); [AAAI OJS 38736](https://ojs.aaai.org/index.php/AAAI/article/view/38736)
- **Problem statement [VS].** Current methods use pre-trained discrete clustering. Hard assignment loses the fine-grained differences between information sets that strategy solving needs. — [arXiv abs](https://arxiv.org/abs/2511.12083)
- **Method, abstract level [VS].**
  - It "pre-trains information-set features and embeds them into a low-dimensional space".
  - It runs "regret accumulation and strategy updates in this embedding space".
  - A "theoretical analysis" shows its "ability to reduce cumulative regret".
  - "With the same spatial overhead", it achieves "significantly faster exploitability convergence" on poker than cluster-based abstraction.
  - It claims to be the first poker AI algorithm to pre-train information-set abstractions through low-dimensional embedding.
  - — [arXiv abs](https://arxiv.org/abs/2511.12083); [Lacuna summary, 3P](https://lacuna.tiptreesystems.com/work/no-regret-strategy-solving-in-imperfect-information-games-via-pre-trained/wrk_4ebed4cbd32e41e25a4a1a0e366855b6)
- **Repository [VD].**
  - The official repo README names the pre-trained embedding network **"HandEbdNet"** and the OpenReview id `bc0FkNOd2U`.
  - It states the repo "is currently empty and serves as the official placeholder". Code will be uploaded "after the AAAI-26 conference publication".
  - The shallow HEAD I cloned is commit 39c47c8, dated 2025-11-17, and contains only the README.
  - — [GitHub PhilEnchan/EmbeddingCFR](https://github.com/PhilEnchan/EmbeddingCFR); [OpenReview](https://openreview.net/forum?id=bc0FkNOd2U)
- **How embeddings are obtained [3P].**
  - HandEbdNet takes a hand encoding as input and outputs a **probability-distribution vector**, which serves as the embedding coordinates. It is trained with **win-rate distributions as supervision**.
  - It generates embeddings on the fly instead of storing large matrices.
  - Summaries conflict on one point: one says win/loss/draw distributions are the *input* rather than the target.
  - — [papernotes.org note](https://en.papernotes.org/AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd/); [Lacuna](https://lacuna.tiptreesystems.com/work/no-regret-strategy-solving-in-imperfect-information-games-via-pre-trained/wrk_4ebed4cbd32e41e25a4a1a0e366855b6)
- **Strategy mapping, evaluation and thesis [3P].**
  - The strategy is mapped back to the original information sets.
  - The paper is evaluated on **Numeral211 Hold'em** against clustering abstractions at equal memory.
  - The authors argue "the bottleneck lies in the discrete clustering paradigm itself, not in the specific clustering algorithm".
  - — [papernotes.org](https://en.papernotes.org/AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd/)
- **Related work by the same group [VS, title only].** KrwEmd (arXiv 2511.12089), "Revising the Imperfect-Recall Abstraction from Forgetting Everything", is a companion abstraction paper. — [arXiv PDF](https://arxiv.org/pdf/2511.12089)

### Inferences
- **Probable mechanism (hypothesis, unverified).** The memory budget is "same spatial overhead" as k clusters, and the embedding is a probability vector p(I) ∈ Δ^d. So the most plausible design is a regret table R[j][a] over d embedding coordinates.
  - The infoset's regret would be Σ_j p_j(I)·R[j][a].
  - The update would be R[j][a] += p_j(I)·r(I,a).
  - That is soft-assignment abstraction, which is linear RCFR with nonnegative, normalized features updated by accumulation rather than least squares. If true, it is essentially the requester's soft (distance-weighted) variant with a frozen, equity-trained encoder. **Confirm against the PDF before relying on this.**
- **Difference from the requester's design.** Embedding CFR's embedding is (per the [3P] sources) trained from **equity / win-rate distributions** (outcome-based features in the potential-aware/EMD tradition) and frozen before solving. The requester's encoder would be trained from **regret or advantage signals** and the codes would *be* the regret tables. That is a different training signal and coupling.

### Gaps
- Exact regret and strategy update equations in embedding space (kernel? per-dimension tables? interpolation among neighbors?).
- Theorem statement and assumptions, including how embedding approximation error enters.
- Sampling (full CFR or MCCFR) and average-strategy computation.
- Embedding dimension vs. number of k-means buckets. Exploitability numbers. Games other than Numeral211.
- The arXiv/AAAI/OpenReview full text was blocked and the code is not released, so none of these could be checked.

## Q5. Vector quantization, codebooks, learned discrete abstractions, or encoders trained from regret/advantage signals inside CFR, and the novelty of the requester's design

### Takeaway
I found **no published work** that trains a vector-quantized codebook or encoder **from regret/advantage signals** and uses the codes **as the regret tables** inside (MC)CFR. This is negative evidence from search, without full-text access.

The closest precedents are:
- **RCFR / Morrill:** the regressor is described as a "tunable, compact, and dynamic abstraction". This covers the hard-code and soft-code solver theoretically.
- **Embedding CFR:** a pre-trained continuous embedding, but trained on equity rather than regret.
- **WEVA (2026):** buckets come from k-means on expected-value features taken from a short full-game CFR warm-up. This is CFR-derived and value-space, but static and not regret-trained.
- **EVPA:** neural networks trained on CFR traversal data for per-hand features.
- **Deep CFR family:** regresses regret directly but uses no discrete codes.

So the solver part (code = running-mean MCCFR regret, soft = RCFR-like) is not novel. The learned regret-space encoder/codebook coupling appears to be unpublished.

### Cited Findings
- **No VQ/codebook + CFR hit [VS, negative].** An extended search for vector quantization or codebooks combined with CFR abstraction returned no matching paper. The nearest results were clustering-based abstraction and Deep CFR. — [search result: WEVA arXiv 2605.10900](https://arxiv.org/abs/2605.10900); [Deep CFR PDF](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)
- **WEVA [VS].** Boning Li, Longbo Huang. arXiv 2605.10900, submitted 11 May 2026. "Effective, Efficient, and General Information Abstraction for Imperfect-Information Extensive-Form Games."
  - It runs W warm-up CFR iterations on the **full game**, collects per-hand **expected-value features at each decision node**, and clusters them with k-means++.
  - It needs no domain knowledge and no pre-training.
  - It beats equity-based and rank-based abstractions across three games, bucket counts and CFR variants, with exploitability reductions "up to over 80%".
  - W = 10 warm-up iterations already beats existing methods in most settings.
  - — [arXiv abs](https://arxiv.org/abs/2605.10900); [Pith summary, 3P](https://pith.science/paper/2605.10900)
- **EVPA [VS for the summary claim; listing unconfirmed].** A search summary says Li and Huang's earlier EVPA "trained neural networks on CFR traversal data for per-hand features". A later paper notes that EVPA and WEVA both gather their abstraction signal by traversing the game tree. The ICLR 2025 paper "Efficient Online Pruning and Abstraction for Imperfect Information Extensive-Form Games" appeared in the same results and is plausibly EVPA. I did not verify the authors or link. — [WEVA arXiv](https://arxiv.org/abs/2605.10900); [ICLR 2025 listing](https://mlanthology.org/iclr/2025/li2025iclr-efficient-a/)
- **Abstraction Agent [VS].** Boning Li, Longbo Huang. arXiv 2609.04303; Findings of EMNLP 2026.
  - An LLM discovers continuous strategic features from a natural-language game description, scores private states on them, filters redundant features by correlation, and clusters the states with k-means.
  - It uses no hand evaluators, training data or tree traversal.
  - It was applied to poker, Omaha and Mahjong.
  - — [arXiv abs](https://arxiv.org/abs/2609.04303); [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1246/)
- **Kim & Sandholm word-embedding abstraction [VS].** arXiv 2605.15543 (15 May 2026).
  - Actions are treated as words and gameplay data as a corpus. The resulting embedding vectors are clustered with k-means into abstractions.
  - The method is "effective" but does not beat game-specific algorithms.
  - A reviewer flags that the clusters may reflect co-occurrence more than strategic value [3P].
  - — [arXiv abs](https://arxiv.org/abs/2605.15543); [Pith review](https://pith.science/paper/2605.15543)
- **Imperfect-recall abstraction with bounds [VS].** Kroer & Sandholm compute bound-minimizing abstractions with a MIP clustering formulation, then run CFR and measure full-game regret. — [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302)
- **Dynamic abstraction via regression [VS].** Morrill: the RCFR regressor is "a tunable, compact, and dynamic abstraction of the game that is informed by and adapts to the particular solution being computed". — [thesis PDF](https://poker.cs.ualberta.ca/publications/Morrill_Dustin_R_201603_MSc.pdf)
- **Pre-trained embedding instead of clustering [VS/3P].** Embedding CFR (see Q4). The encoder is trained beforehand and not adapted from regret similarity [3P]. — [papernotes.org](https://en.papernotes.org/AAAI2026/llm_pretraining/no-regret_strategy_solving_in_imperfect-information_games_via_pre-trained_embedd/)

### Cross-method comparison matrix (synthesis; tags refer to the sources cited above)

| Method | Approximated object | Traversal / sampling | Target & loss | Guarantee: how error enters | Average strategy | Games |
|---|---|---|---|---|---|---|
| RCFR 2015 | cumulative regret R(I,a) via regressor on φ(I,a) [VS] | full traversal [U] | prev. estimate + exact r^t; squared loss [U] | O(√T) + error term (√-type, non-vanishing); NE if realizable [VS qualitative / U form] | outside approximator [U] | small poker [U]; thesis: 2 small poker games [VS] |
| f-RCFR 2020 | same, with link f (ReLU/poly/exp) [VS] | full traversal [U] | as RCFR [U] | error-aware (Φ,f)-RM bounds; exploitability bounds for poly and exp links; tighter than RCFR [VS]; error inside √ [U] | outside [U] | small games [U] |
| Deep CFR 2019 | linear-weighted mean of sampled advantages (∝ cumulative regret) [U] | external sampling, K traversals/iter [U] | t-weighted MSE, retrain from scratch, reservoir [U] | w.p. 1−ρ: (1+√2/√(ρK))Δ\|I\|√\|A\|√T + 4T\|I\|√(\|A\|Δε_L); avg-regret floor ∝ √ε_L [U] | separate avg-policy net ("AVG_NET") [VD] | Leduc, FHP, HULH [U]; k-means 40k-bucket baselines [VS] |
| SD-CFR 2019 | same advantage nets [U] | external sampling [U] | same [U] | Deep CFR regret side; no avg-net error [U] | exact, from stored nets B^M ("Single") [VD] | Leduc, BigLeduc, h2h vs Deep CFR [VD] |
| DNCFR 2020 | cumulative regret (RSN) + avg strategy (ASN), RNNs [U] | robust sampling, mini-batch MCCFR(+) [U] | MSE, warm-start [U] | tabular unbiasedness/convergence only [U] | ASN net [U] | Leduc, HUNL subgames [U] |
| DREAM 2020 | advantages with learned Q baseline [U] | outcome sampling, model-free [U] | Deep CFR-style MSE [U] | NE convergence claim [U] | "Single" (SD-CFR) [VD] | Leduc, FHP [VD] |
| ESCHER 2023 | regret estimates from history value fn, no IS [VS/VD] | fixed sampling policy for traverser [VS] | regress q̂−v̂ (∝ cf. regret) [U] | approx. NE w.h.p. in tabular case [VD]; estimated-regret degradation via Farina et al. 2020 [VS] | avg-policy net [U] | beats DREAM/NFSP, gap grows with size [VD] |
| VR-Deep(P)DCFR+ 2025/26 | cumulative advantage (discounted, clipped) + value baseline [VS] | outcome sampling, K episodes [VS] | bootstrapped cumulative-advantage update [VS] | not found | not found | typical IIGs + large poker [VS] |
| Embedding CFR 2025/26 | regret and strategy in a pre-trained embedding space [VS]; HandEbdNet [VD] | not found | encoder: win-rate-distribution supervision [3P] | "reduces cumulative regret" [VS]; statement not found | not found | Numeral211 Hold'em [3P] |
| WEVA 2026 (abstraction, not FA) | buckets from k-means++ on warm-up CFR EV features [VS] | full-game CFR warm-up (W≈10) [VS] | none (clustering) | none found | standard (abstract game) | 3 games [VS] |

### Inferences
- **Where the requester's design sits.**
  - (a) A hard-assignment, frozen VQ codebook with a per-code running-mean MCCFR regret is MCCFR on an imperfect-recall abstraction, equivalently RCFR with one-hot features. It is not novel as a solver, and its theory comes from the RCFR bound and the imperfect-recall abstraction bounds.
  - (b) Soft distance-weighted assignment is RCFR with nonnegative normalized features (a kernel or soft-clustering regressor). It is probably very close to Embedding CFR's mechanics, though that is unverified (see Q4).
  - (c) The potentially novel part is the encoder trained so that distances reflect **regret** (or advantage) similarity, with a VQ codebook that is simultaneously the abstraction and the regret store, refined during sampled CFR. The closest published neighbours each differ:
    - WEVA uses value-space features from a CFR warm-up, then static k-means.
    - EVPA uses NN features from CFR traversal data.
    - Embedding CFR uses an equity-trained, frozen encoder.
    - Deep CFR, DREAM and ESCHER regress regret with no codes.
    - Morrill/RCFR call the regressor a "dynamic abstraction" but use features fixed in advance.
- **Theory that transfers.** RCFR, f-RCFR and Deep CFR bounds hold for *any* regret estimator, given a per-iteration error bound on cumulative regret. The requester can therefore report a bound by measuring within-code regret dispersion. For hard codes, the error is the deviation of members' cumulative regret from the scaled code mean, and the bound's average-regret floor is ∝ √(that error).
- **Theory that does not transfer.** If the encoder or code positions move while regret accumulates, a code's stored regret mixes contributions computed under different assignments. No cited bound covers this non-stationary case. Deep CFR sidesteps it by retraining from scratch on all stored samples each iteration. DNCFR's warm-start and Robust Deep MCCFR's "warm-starting bias" and "non-stationary target distribution shift" findings point at the same risk.
- **Circularity risk.** Codes defined by early-iteration regret may not separate infosets whose regrets diverge later. This mirrors the WEVA caveat that early CFR estimates may be noisy [3P, Pith].
- **Evaluation implication (aligns with the project's AGENTS.md).**
  - Novelty claims should be backed by an equal-memory comparison against a matched k-means abstraction baseline, such as the 40k-bucket style used in Deep CFR and the equal-overhead comparison in Embedding CFR.
  - Current and average policies should be reported separately, because the cited methods differ exactly there (AVG_NET vs Single).

### Gaps
- Full-text confirmation that none of WEVA, EVPA, Embedding CFR or KrwEmd trains its encoder on regret targets. Abstracts suggest EV, equity and win-rate signals, but bodies were not accessible.
- Unverified leads not checked this session (no links verified):
  - Brown & Sandholm, "Simultaneous Abstraction and Equilibrium Finding in Games" (IJCAI 2015), which refines abstraction during CFR.
  - Lanctot et al., "No-Regret Learning in Extensive-Form Games with Imperfect Recall" (ICML 2012), which gives CFR bounds on (skew) well-formed imperfect-recall abstractions with additive error terms.
  - Kroer & Sandholm, "A Unified Framework for Extensive-Form Game Abstraction with Bounds" (NeurIPS 2018).
  - VQ in game-playing but not CFR: Ozair et al., "Vector Quantized Models for Planning" (ICML 2021, believed arXiv 2106.04615), and Stochastic MuZero (Antonoglou et al., ICLR 2022), which use VQ codes for chance outcomes in MCTS rather than for regret.
- Whether any 2026 workshop paper (NeurIPS/ICLR workshops, Findings) combines VQ-VAE with Deep CFR. None was found, but search coverage of workshop papers is weak.
