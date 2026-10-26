# CFR / MCCFR on Imperfect-Recall Abstractions: Regret Guarantees, Average-Strategy Representation, and the Converse of Zinkevich's Theorem 3

How the evidence is labelled (read this first). On 2026-10-26 I could not read the full texts. The agent proxy refused connections (HTTP 403) to arxiv.org, webdocs.cs.ualberta.ca, www.cs.cmu.edu, icml.cc, ijcai.org, cdn.aaai.org, dl.acm.org, proceedings.neurips.cc, semanticscholar and others. WebFetch failed with DNS errors for every host. Only the search engine worked. The labels below mean:
- **[V-abs]**: I confirmed the claim from the paper's own abstract or landing page, as shown in search results.
- **[V-snip]**: I confirmed it from a search-engine excerpt of the primary PDF or HTML. The wording may be paraphrased, and I did not check the exact formula.
- **[V-2nd]**: I confirmed it only from a secondary source that cites the paper.
- **[UNVERIFIED-recall]**: The statement comes from my prior knowledge of the paper. I could not check it this session. Each one names the source URL to check, and these statements appear only under "Gaps".

Nothing in these notes is a direct quotation that I checked against a PDF. Exact constants in theorems must be checked before anyone cites them.

## Q1. Lanctot, Gibson, Burch, Zinkevich, Bowling (ICML 2012): well-formed / skew well-formed games, the bound, sampling coverage, average strategy

### Takeaway
Lanctot et al. (2012) prove the first CFR regret bound for a class of imperfect-recall games. The class is "well-formed" or "skew well-formed" games, each defined relative to a perfect-recall refinement Γ′. In this class, CFR run on the imperfect-recall game (or abstraction) Γ gives a regret bound in the full or refined game Γ′. The skew version adds a utility-slack term δ that does not go to zero. I confirmed the class definitions and the headline claims, but not the exact constants. I also could not confirm whether the theorem covers MCCFR, or how the paper handles the stored average strategy versus the Γ′ average.

### Cited Findings
- [V-abs] The abstract says: "We present the first regret bound for CFR when applied to a general class of games with imperfect recall"; "CFR applied to any abstraction belonging to our general class results in a regret bound not just for the abstract game, but for the full game as well"; and "for a subset of this class, CFR minimizes average regret in the extensive game." It also says CFR's existing regret bounds "depend on the requirement of perfect recall." — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622); [ACM DL entry](https://dl.acm.org/doi/10.5555/3042573.3042707)
- [V-snip] Venue: ICML 2012 (Proc. 29th ICML), pp. 65–72. A longer version is U. Alberta Tech Report TR12-04. — [dblp](https://dblp.org/rec/journals/corr/abs-1205-0622.html); [Bowling publications page](https://webdocs.cs.ualberta.ca/~bowling/publications/b2hd-12icml-ir-w-tr.html); [PDF](https://webdocs.cs.ualberta.ca/~bowling/papers/12icml-ir.pdf)
- [V-snip] Γ′ is a perfect-recall *refinement* of Γ, so the abstraction is a coarsening of Γ′. The refinement is defined in Section 5.1, and skew well-formedness is defined relative to a perfect-recall refinement in Section 5.2. — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622)
- [V-snip] Skew well-formed games are defined with a bijection and constants that satisfy several conditions between corresponding (refined) information sets. Search summaries said "four conditions", which I could not confirm. "The only difference between Definitions 2 and 3 is in condition (i)": utilities need only be proportional *up to* an additive constant δ, not exactly proportional. "Any well-formed game is skew well-formed by setting δ = 0." — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622); [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/1205.0622)
- [V-snip] The well-formed bound is "Theorem 1". "In the case when Γ has perfect recall, because Γ is well-formed with respect to itself, Theorem 1 is a direct generalization of the original CFR bound." — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622)
- [V-snip] The experiments use die-roll poker (DRP), phantom tic-tac-toe (PTTT) and Bluff, and show "how imperfect recall can be used to trade a small increase in regret for a significant reduction in memory." In DRP-IR, players forget their first die roll and know only the sum of their dice. "As the skew δ grows, so does our regret in Skew-DRP(δ) after a fixed number of iterations." — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622); [deepai mirror](https://deepai.org/publication/no-regret-learning-in-extensive-form-games-with-imperfect-recall)
- [V-snip] The paper applies the folk theorem: "the average of the strategy profiles generated is a 2ε-Nash equilibrium" under their bound. — [arXiv 1205.0622 (search excerpt)](https://arxiv.org/html/1205.0622)
- [V-2nd] Kroer's thesis contrasts the bound with Kroer & Sandholm's: the Lanctot et al. bound has "a linear dependence on the number of information sets", while the Kroer–Sandholm perfect-recall payoff-error bound has no dependence on game size. — [Kroer PhD thesis (CMU 2018)](http://www.columbia.edu/~ck2945/files/ckroer_computer_science_2018.pdf)
- [V-2nd] Later work restates the obstacle: "In imperfect recall games, the sum of counterfactual regrets no longer forms an upper bound on the external regret (Lanctot et al., 2012)." — [Čermák, Bošanský et al., "Towards Solving Imperfect Recall Games" (AAAI-17 WS)](https://cdn.aaai.org/ocs/ws/ws0384/15192-68362-1-PB.pdf)

### Inferences
- In Lanctot et al., the guarantee is on regret in **Γ′ (perfect recall) of the lifted iterates** σ′^t(I′) = σ^t(I) for I′ ⊆ I. Zinkevich's folk theorem (Q6) turns this into a 2ε-NE statement for the **Γ′ average**, which is the sequence-form or realization-plan average with weights π_i^{σ^t}(I′) at each I′. The folk theorem does not apply directly to a bucket-level average stored in Γ, unless members of the same abstract infoset have identical own-reach (see Q5). I inferred this from the structure of the result. I could not check whether the paper itself says so.
- Suppose condition (i) is exact utility proportionality, and the other conditions make opponent sequences identical in Γ′ and player-i's *future* abstract sequences identical in Γ (I recall this but did not verify it; see Gaps). Then for I′, I″ ∈ P(I) the counterfactual action values are positively proportional at every iteration with a constant c > 0 that does not depend on t. So member regret vectors lie on one ray. Because ψ(x) = max(0, max_a x_a) is positively homogeneous, the requester's Jensen/Bregman term H_B is then **exactly 0**. The requester's identity therefore reproduces the well-formed case of Lanctot's theorem as the zero-heterogeneity special case. Under skew well-formedness, the deviation from the ray is O(δ·π_{-i}) per iteration. This gives an H_B that grows linearly in T, i.e., a non-vanishing average-regret floor. That matches the verified empirical finding that regret rises with δ.
- The well-formed condition, as I recall it, constrains the **suffix** (player i's future abstract decisions below I′ and I″), not the **prefix**. DRP-IR merges histories with different first-roll buckets, so DRP-IR members can have different own-reach. This fits the requester's proposition (3): Lanctot's regret guarantee can hold while the bucket-level stored average still differs from the Γ′ average. I could not confirm whether the paper discusses this.

### Gaps
- **[UNVERIFIED-recall] Exact conditions of Definition 2 (well-formed).** For all i, I ∈ 𝓘_i, I′, I″ ∈ P(I), there is a bijection φ: Z_{I′} → Z_{I″} such that for all z ∈ Z_{I′}:
  - (i) u_i(z) = ℓ_{I,I′,I″} u_i(φ(z)) for a positive scaling constant;
  - (ii) chance reach is proportional, possibly folded into the same ℓ;
  - (iii) in Γ′, X_{−i}(z) = X_{−i}(φ(z)), i.e., opponents' (infoset, action) sequences are identical;
  - (iv) in Γ, X_i(z[I′], z) = X_i(φ(z)[I″], φ(z)), i.e., player i's own abstract sequence *from I′ to z* is identical.

  Check Section 5 of [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622).
- **[UNVERIFIED-recall] Theorem form.** Roughly: "If Γ is well-formed w.r.t. Γ′, then the average regret in Γ′ for player i of choosing strategies according to CFR in Γ is bounded by R_{i,full}^T/T ≤ Δ_i K √|A_i| |𝓘_i| / √T". Here K is a constant built from the ℓ scalings, possibly a max over I of Σ_{I′∈P(I)} of them. The skew theorem adds a T-independent term Σ_{I∈𝓘_i} Σ_{I′,I″∈P(I)} (opponent-reach-weighted) δ_{I,I′,I″}. I cannot confirm the exact constant, the weights, or the theorem numbering (Theorem 1 vs 2 vs 3).
- **Sampling (MCCFR).** I could not verify whether the ICML paper or Lanctot's thesis states the imperfect-recall bound for chance-sampled, outcome-sampling or external-sampling MCCFR, or which CFR variant was run in each domain. [UNVERIFIED-recall] Lanctot's 2013 thesis (abstract [V-abs]: "develops a theory for applying counterfactual regret minimization to a generic subset of imperfect recall games") has a chapter on this. MCCFR's high-probability bounds (Lanctot et al. 2009) would combine with the well-formed decomposition. Check the [Lanctot thesis PDF](https://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf) and the [ResearchGate record](https://www.researchgate.net/publication/259564336_Monte_Carlo_Sampling_and_Regret_Minimization_for_Equilibrium_Computation_and_Decision-Making_in_Large_Extensive_Form_Games).
- I found no source text saying how Lanctot et al. compute the average strategy in experiments (in Γ or in Γ′), or which metric they plot (full-game regret, Γ′ regret, or exploitability).

## Q2. Kroer & Sandholm (EC 2014, EC 2016, NeurIPS 2018): conditions, bounds, solution-quality guarantees

### Takeaway
- **EC 2014** gives algorithm-agnostic bounds for **perfect-recall** lossy abstractions. It explicitly leaves imperfect recall open.
- **EC 2016** introduces **chance-relaxed skew well-formed (CRSWF)** games. These relax Lanctot's skew well-formedness by also allowing chance error. EC 2016 proves that Nash equilibria and approximate **self-trembling equilibria** (strategies with bounded counterfactual regret at every infoset) computed in the abstraction are approximate NE in the original perfect-recall game, with linear dependence on tree height. It also shows that bound-minimizing abstraction reduces to clustering, with the bound increase as the distance.
- **NeurIPS 2018** gives a unified framework with an **exact ex-post** decomposition of abstraction error. It also gives the first bounds for ε-NE computed in abstractions.

### Cited Findings
- [V-abs] EC 2016: "first general, algorithm-agnostic, solution quality guarantees for Nash equilibria and approximate self-trembling equilibria computed in imperfect-recall abstractions, when implemented in the original (perfect-recall) game." For games with certain properties, "the problem of computing a bound-minimizing abstraction ... can be reduced to a clustering problem, where the increase in our bound is the distance function." — [ACM DL 10.1145/2940716.2940736](https://dl.acm.org/doi/10.1145/2940716.2940736); [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302); [EC'16 PDF](https://www.cs.cmu.edu/~sandholm/imperfect-recall-abstraction-with-bounds.ec16.pdf)
- [V-snip] "CRSWF games generalize SWFGs by also allowing chance error." That extra freedom lets nodes share an abstract information set even when their (chance) reach probabilities differ. "Typically, the perfect-recall refinement is the original game, and the CRSWF game is an abstraction that is easier to solve." Nash equilibria and strategies with bounded counterfactual regret in CRSWF abstractions are approximate NE in the original game, "with a linear dependence on game-tree height." — [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302); [ar5iv HTML](https://ar5iv.arxiv.org/html/1409.3302)
- [V-2nd] Kroer's thesis: a strategy profile with bounded counterfactual regret in an abstraction maps to an ε′-Nash equilibrium in the original game. — [Kroer PhD thesis](http://www.columbia.edu/~ck2945/files/ckroer_computer_science_2018.pdf)
- [V-2nd] Computing a minimum-error CRSWF abstraction is attributed to Kroer et al. (2016) as hard, by reduction from planar clustering. — ["The Value of Recall in Extensive-Form Games", arXiv 2412.19659](https://arxiv.org/pdf/2412.19659) (authors not verified)
- [V-snip] Search excerpt of the EC'16 experiments: CFR is run on the abstraction and the observed full-game regret is compared with the bound. The bound is "within the same order of magnitude as the observed regret" even after CFR converges. I am only moderately confident in this paraphrase. — [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302)
- [V-snip] The earlier workshop version (IJCAI-15 AGT WS) says that EC'14 left as "an open problem whether similar bounds can be achieved for imperfect-recall abstractions." — [Kroer & Sandholm WS PDF](https://www.cs.cmu.edu/~sandholm/imperfect-recall-abstraction-with-bounds.ijaci15agtWS.pdf); [EC'14 PDF](https://www.cs.cmu.edu/~sandholm/extensiveGameAbstraction.ec14.pdf)
- [V-abs] NeurIPS 2018 ("A Unified Framework for Extensive-Form Game Abstraction with Bounds", pp. 613–624):
  - It presents "a unified framework for analyzing abstractions that can express all types of abstractions and solution concepts used in prior papers with performance guarantees—while maintaining comparable bounds on abstraction quality."
  - "our framework gives an exact decomposition of abstraction error in a much broader class of games, albeit only in an ex-post sense, as our results depend on the specific strategy chosen."
  - It uses "slightly weaker assumptions than prior work", and a counterexample shows "such assumptions are necessary for some games."
  - It gives "the first bounds for how ε-Nash equilibria computed in abstractions perform in the original game."
  - It covers general-sum n-player games.

  — [NeurIPS 2018 abstract page](https://proceedings.neurips.cc/paper/2018/hash/aa942ab2bfa6ebda4840e7360ce6e7ef-Abstract.html); [PDF](https://proceedings.neurips.cc/paper/2018/file/aa942ab2bfa6ebda4840e7360ce6e7ef-Paper.pdf)
- [V-2nd] Sandholm's lecture summary of the 2018 theorem: given a perfect-recall game, an *acyclic* abstract game, a mapping that satisfies mild assumptions, and an ε-NE in the abstraction, any lifted strategy is an ε′-NE in the original game with ε′ = max_i ε_i. Here ε_i combines abstraction error terms and ε. There is "a similar result for bounded counterfactual regret in the abstract game." — [Sandholm 15-888 F23 Lecture 8](http://www.cs.cmu.edu/~sandholm/cs15-888F23/Lecture8.pdf)
- [V-snip] Lisý, Davis & Bowling (AAAI 2016) prove that normal-form games with sequential strategies can be modelled as a "generalized chance-relaxed skew well-formed imperfect-recall game", in which CFR is guaranteed to converge. Their Theorem 2 reportedly gives CRSWF "with zero errors". — [Lisý, Davis, Bowling AAAI-16 PDF](https://poker.cs.ualberta.ca/publications/lisy_davis_bowling_aaai16.pdf)

### Inferences
- The requester's abstraction-agnostic bound NashConv(avg) ≤ Σ_i (C_i + H_i)/T is closest in spirit to the **ex-post exact decomposition** of Kroer & Sandholm (2018). Both depend on the realized strategies, not only on the abstraction. The CRSWF bounds (2016) are **ex-ante**: they come from structural utility and chance errors, they apply to any solution with bounded abstract counterfactual regret, and they are not computed from realized regret vectors. I inferred this from the abstracts. The precise form of the 2018 decomposition has not been compared with C + H.
- EC 2016 needs *self-trembling* (bounded counterfactual regret at every abstract infoset, including ones the player does not reach) for approximate solutions. This strongly suggests the authors needed more than low exploitability in the abstraction, which bears on Q6. See Gaps for the definition.

### Gaps
- **[UNVERIFIED-recall] CRSWF definition (EC'16).** It uses the same bijection structure as Lanctot et al. On top of that:
  - a utility error |u_i(z) − ℓ u_i(φ(z))| ≤ δ^R;
  - a leaf chance error on the chance probability below I′ vs I″;
  - a *distribution* error on the conditional chance distribution over nodes inside I′ vs I″.

  The main theorem bounds ε by a sum over tree levels of 2·(reward error) plus W·(chance and distribution error) terms, with W the payoff range, hence the dependence on height. The exact formulas and theorem numbers must be checked in [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302).
- **[UNVERIFIED-recall] Self-trembling equilibrium.** A profile in which each player best-responds at *every* own information set under the counterfactual "play to reach" convention, including infosets the player itself never reaches. Approximate version: bounded counterfactual regret everywhere. Source to check: [EC'16 PDF](https://www.cs.cmu.edu/~sandholm/imperfect-recall-abstraction-with-bounds.ec16.pdf).
- **[UNVERIFIED-recall] EC 2014.** For perfect-recall abstractions, a lifted abstract NE is an ε-NE with ε ≈ 2ε^R + Σ_j ε^0_j W, which has no dependence on the number of infosets. Computing a min-error abstraction is NP-complete, and they give an IP formulation. Check the [EC'14 PDF](https://www.cs.cmu.edu/~sandholm/extensiveGameAbstraction.ec14.pdf).
- I could not determine whether the NeurIPS 2018 framework covers *imperfect-recall* abstract games (the lecture summary says "acyclic abstract game"), or what its exact ex-post decomposition terms are.
- None of the Kroer–Sandholm sources I could reach discusses the bucket-level stored average vs the Γ′ sequence-form average.

## Q3. Waugh et al., "A Practical Use of Imperfect Recall" (SARA 2009) and "Abstraction Pathologies in Extensive Games" (AAMAS 2009)

### Takeaway
SARA 2009 introduced imperfect-recall abstractions in poker, used since the 2008 ACPC. It acknowledged openly that CFR loses its convergence guarantee there, and justified the approach empirically: imperfect-recall strategies outperform perfect-recall ones. AAMAS 2009 showed that refining an abstraction can make the equilibrium more exploitable in the real game (pathologies). The one exception, monotonicity, holds when only one player is abstracted.

### Cited Findings
- [V-abs] SARA 2009 (Waugh, Zinkevich, Johanson, Kan, Schnizlein, Bowling; pp. 175–182): "all known efficient game solving algorithms rely on an assumption of perfect recall in their convergence proofs"; with imperfect recall "game solving algorithms such as CFR are no longer guaranteed to converge," but "the algorithm remains well defined and we can still attempt to run it"; imperfect-recall abstractions "can be used to generate strategies that perform better than perfect recall abstractions." — [SARA'09 PDF](https://webdocs.cs.ualberta.ca/~bowling/papers/09sara.pdf); [Johanson's publication page](http://johanson.ca/publications/poker/2009-sara-imperfect-recall/2009-sara-imperfect-recall.html)
- [V-snip] Lanctot et al. (2012) say that "previous counterexamples show that abstraction in general provides no guarantees in the full game." — [arXiv 1205.0622 HTML](https://arxiv.org/html/1205.0622)
- [V-abs] AAMAS 2009 (Waugh, Schnizlein, Bowling, Szafron; pp. 781–788): "refining an abstraction can actually lead to a weaker strategy", and pathologies arise "when abstracting both chance nodes as well as a player's actions." The counterexamples use a small six-card poker variant (Leduc-sized), which allows exact computation. — [AAMAS'09 PDF](https://poker.cs.ualberta.ca/publications/AAMAS09-abstraction.pdf); [AAMAS proceedings PDF](https://aamas.csc.liv.ac.uk/Proceedings/aamas2009/pdf/01_Full%20Papers/13_21_120_FP_0888.pdf)
- [V-2nd] Johanson et al. (AAAI 2012, CFR-BR) describe Waugh et al.'s Theorem 3: when one agent is abstracted and the other is not, "a strict refinement to the abstraction will result in a monotonic decrease in the abstracted player's exploitability." — [Finding Optimal Abstract Strategies (AAAI-12)](https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf)
- [V-2nd] Johanson et al. (AAMAS 2013) summarize: "strict refinements of abstractions can result in abstract strategy equilibria that are more exploitable in the real game." — [Evaluating State-Space Abstractions (AAMAS-13)](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf)

### Inferences
- Neither 2009 paper supplies a regret bound for imperfect recall. SARA 2009 is the empirical origin of the "run CFR anyway on IR buckets" practice whose theory Lanctot et al. (2012) later partly supplied.
- The pathology result implies that comparisons across abstraction granularities must be made in the real game. That matches the project rule to measure LBR or exploitability in the original game with matched evaluators.

### Gaps
- I could not verify what SARA 2009 says about computing the average strategy in the IR abstraction. I also could not verify whether it reports real-game exploitability or only match results.
- The exact theorem numbering in AAMAS 2009 is second-hand.

## Q4. Is in-bucket regret cancellation written anywhere as a Jensen / convex-function (Bregman) gap, or tied to clustering of regret or counterfactual-value vectors?

### Takeaway
I found **no** source that writes the bucket-aggregation loss as a Jensen or Bregman gap of ψ(x) = max(0, max_a x_a) over member regret vectors. The closest literature has three strands:
- Kroer & Sandholm (2016) reduce bound-minimizing abstraction to clustering, with the bound increase as the distance.
- Function-approximation CFR work bounds regret by the error in approximating regret vectors: RCFR (Waugh et al. 2015), f-RCFR (D'Orazio et al. 2020) and Deep CFR. Tabular abstraction is the special case of indicator features.
- Recent poker abstraction work clusters CFR-derived EV feature vectors, without regret bounds.

### Cited Findings
- [V-abs] Kroer & Sandholm EC 2016: the bound-minimizing abstraction problem reduces to clustering, "where the increase in our bound is the distance function." — [ACM DL](https://dl.acm.org/doi/10.1145/2940716.2940736)
- [V-abs] Waugh, Morrill, Bagnell, Bowling, "Solving Games with Functional Regret Estimation" (AAAI 2015, pp. 2138–2145). The method learns a function approximator online to estimate the regret of each action. They "prove the approach sound by providing a bound relating the quality of the function approximation and regret of the algorithm", with the corollary that it converges to NE in self-play "so long as the regrets are ultimately realizable by the function approximator". It is "a principled generalization of existing work on abstraction in large games." — [arXiv 1411.7974](https://arxiv.org/pdf/1411.7974); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/9445)
- [V-snip] D'Orazio, Morrill, Wright, Bowling (f-RCFR, AAMAS 2020) treat function approximation as "a natural generalization of abstraction." — [arXiv 1912.02967](https://arxiv.org/html/1912.02967)
- [V-snip] Recent preprint (2026): run a short CFR warm-up, extract per-hand expected-value features at each decision node, and cluster them with k-means++. The snippet showed no regret bound. — [arXiv 2605.10900](https://arxiv.org/html/2605.10900)
- [V-abs] Two history-aware IR abstractions: FROI (Fu et al., AAMAS 2026) and KrwEmd ("k-recall winrate" features). They argue outcome-based IR abstraction loses performance by discarding history. Neither analyses the average-strategy representation. — [arXiv 2510.15094](https://arxiv.org/pdf/2510.15094); [arXiv 2511.12089](https://arxiv.org/pdf/2511.12089)

### Inferences
- **ψ is positively homogeneous**, so the requester's H_B = Σ_{I∈B} W_I D_ψ(ρ_I, ρ̄_B) is a Jensen gap that **vanishes exactly when member regret vectors are positively collinear** (on a common ray). It is small when they are nearly collinear. Lanctot's well-formed condition (exact proportional utilities with matched future structure) is a *sufficient structural* condition for collinearity at every iteration (Q1 inferences). The H_B identity is therefore a strict generalization: an ex-post, strategy-dependent measure of the same thing Lanctot's ℓ-proportionality and Kroer–Sandholm's δ and chance errors bound ex ante. As far as these searches reach, the Bregman-divergence form appears **not to be in the literature**. Treat this as "not found" rather than "proven novel", because the full texts of Kroer–Sandholm 2016 and 2018 could not be read.
- Tabular bucket CFR is RCFR with one-hot bucket features. RCFR's bound is in terms of regression error, ‖R − R̃‖-type, and does not give the zero-cost-on-rays property. A Jensen gap in ψ is smaller than a squared-error proxy whenever members differ mainly in scale. This suggests that clustering on *direction* of regret or counterfactual-value vectors, plus a nonnegative scale, is the "right" metric for this loss (inference).

### Gaps
- **[UNVERIFIED-recall] RCFR and Deep CFR bound forms.** Deep CFR (Brown et al., ICML 2019) Theorem 1: with probability 1−ρ, average regret ≤ (1 + √2/√(ρK)) Δ|𝓘|√|A|/√T + 4|𝓘|√(|A|Δ ε_L), where ε_L is the regression MSE. The exact RCFR bound shape is also unverified. Check the [Deep CFR PDF](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf) and [arXiv 1411.7974](https://arxiv.org/pdf/1411.7974).
- I found nothing on Kroer & Sandholm's clustering distance written in counterfactual-value-vector form (it is in terms of utility and chance errors, as far as abstracts show).

## Q5. Perfect-recall vs imperfect-recall abstraction average strategy; A-loss recall; is the stored IR-CFR average the sequence-form average?

### Takeaway
Zinkevich's average-strategy formula needs player i's own reach π_i^σ(I) to be well-defined over the infoset. That holds under perfect recall, and in an abstraction it holds exactly when all members share their own abstract (bucket, action) history. In imperfect-recall abstractions where members' own-reach differs, the stored bucket average is a reach-weighted *mixture* of the members' true (Γ′) averages, not the sequence-form average of the lifted iterates. The true average is generally not even representable at bucket level. The literature recognizes the symptom: Čermák et al. note that the average strategy has no value guarantee in IR games. Kroer–Sandholm CRSWF lets members' chance reach differ, and the imperfect-recall literature discusses the "play to reach" assumption failing. But I found **no source stating the requester's proposition (3)** as an equality/only-if characterization. A-loss recall is a different, weaker notion and does not restore the identity.

### Cited Findings
- [V-snip] Čermák, Bošanský et al. give an imperfect-recall game "where CFR can reach a non-optimal strategy". The reason: "the average strategy, which can mix between actions evaluated when played deterministically only, has therefore, no guarantee for its expected value." Their A-loss-recall example has CFR converging to a profile "arbitrarily worse than the maxmin value". — [ResearchGate figure (Čermák et al. 2017)](https://www.researchgate.net/figure/An-imperfect-recall-game-where-CFR-can-reach-a-non-optimal-strategy_fig1_313861094); [Towards Solving Imperfect Recall Games (AAAI-17 WS)](https://cdn.aaai.org/ocs/ws/ws0384/15192-68362-1-PB.pdf)
- [V-snip] A-loss recall: "each loss of memory of a player can be traced back to a loss of memory of the player's own actions." In A-loss-recall games, best response can be found "using the same algorithm as in the perfect recall case." Computing maxmin strategies is NP-hard in IR games, and the negative results carry over to A-loss recall. Maxmin may need irrational probabilities even under A-loss recall. — [Towards Solving Imperfect Recall Games](https://cdn.aaai.org/ocs/ws/ws0384/15192-68362-1-PB.pdf); [Čermák et al., IJAR 2018 "Approximating maxmin strategies ... using A-loss recall property"](https://www.sciencedirect.com/science/article/pii/S0888613X17304097); [Computing Maxmin Strategies ... with Imperfect Recall, arXiv 1608.01510](https://arxiv.org/pdf/1608.01510)
- [V-abs] Čermák, Lisý, Bošanský (AIJ 282, 2020) introduce FPIRA and CFR+IRA. CFR+IRA iteratively solves the abstraction with CFR+ and refines "when the observed convergence is slower than the theoretical guarantee provided by CFR+ in the original game with perfect recall." They give a bound on the average external regret of CFR+IRA, so it converges to NE. The method "requires traversing the complete unabstracted game tree", and abstractions with as little as 0.9% of the original infosets suffice. — [arXiv 1803.05392](https://arxiv.org/abs/1803.05392); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370220300126); [IJCAI-17 precursor](https://www.ijcai.org/proceedings/2017/0130.pdf)
- [V-snip] Tewolde, Zhang, Anagnostides, Sandholm, Conitzer (2026): CFR "updates the action probabilities at each infoset based on expected utilities that counterfactually assume the player played in the past as if it only wanted to reach the infoset in question," and "under imperfect recall, and especially absentmindedness, notions such as 'in the past' become highly dubious." — [arXiv 2602.15252](https://arxiv.org/pdf/2602.15252); [author PDF](https://emanueltewolde.com/files/IRPractical.pdf)
- [V-snip] Lanctot's thesis abstract says it "develops a theory for applying counterfactual regrets minimization to a generic subset of imperfect recall games." — [Semantic Scholar record](https://www.semanticscholar.org/paper/Monte-carlo-sampling-and-regret-minimization-for-in-Bowling-Lanctot/9b0b91edea5a628679f1d0509dc8108a959db406)

### Inferences
- **An exact identity, derived here and not found in a source.** Suppose the CFR implementation accumulates π_i^{σ^t}(h)·σ^t(B) at each traversed history h of bucket B. This is vanilla CFR. MCCFR does the same in expectation after importance correction. Then the stored average is

  σ̄_B = Σ_{I′∈B} (W_{I′}/W_B)·σ̄′(I′), with W_{I′} = Σ_t π_i^{σ^t}(I′),

  where σ̄′(I′) = Σ_t π_i^{σ^t}(I′)σ^t(B) / W_{I′} is the true Γ′ (sequence-form) average at member I′. So the stored average equals every member's true average iff all σ̄′(I′), I′ ∈ B, coincide.
  - **Sufficient condition:** the members' own-reach sequences t ↦ π_i^{σ^t}(I′) are proportional. Members sharing the identical own (bucket, action) prefix, the requester's "own abstract recall", is the canonical sufficient condition.
  - **Necessity:** "only if" holds for arbitrary iterate sequences, apart from degenerate cases such as single-action buckets on the differing prefix, or prefixes whose probabilities coincide polynomially.
- "Own abstract recall" for every bucket is the same as the abstract game Γ having perfect recall for player i over its own (bucket, action) sequences, i.e., the poker "perfect-recall abstraction" (bucket sequences). Standard poker IR abstractions (Waugh 2009 style, which forget earlier card buckets) violate it.
- A-loss recall allows forgetting own *actions*. Members then differ in own action history and so in own reach, so A-loss recall does **not** give the identity. A-loss recall is relevant to best-response computability, not to average-strategy faithfulness.
- Lanctot's well-formedness (as recalled) constrains suffixes, not prefixes. It therefore does not imply the identity either. Its regret guarantee is on Γ′ regret, so the faithful object is the Γ′ average, not σ̄_B.
- CFR+IRA's requirement to traverse the full unabstracted tree is consistent with the authors computing original-game quantities (regret, and possibly the average) in Γ′. This is unconfirmed (see Gaps).
- **For the project's evaluation:** when the stored bucket average is evaluated in the original game, the LBR or NashConv measured is that of σ̄_B lifted, *not* of the Γ′ average, which is the object that Zinkevich/Lanctot guarantees bound. Reports should label which average is evaluated.

### Gaps
- I could not verify whether CFR+IRA (Čermák et al. 2020) stores the average strategy in the original game, nor its exact regret bound.
- I could not access Lanctot's thesis (2013), Gibson's thesis (2014), Johanson's thesis (2016) or Kroer's thesis (2018) for any explicit discussion of PR- vs IR-abstraction average-strategy computation. **No source found** for an explicit "stored IR average ≠ sequence-form average" statement.
- [UNVERIFIED-recall] Kaneko & Kline (1995) is the original A-loss-recall reference. None of the reachable snippets named it.

## Q6. Zinkevich et al. (2007) Theorem 3 and the converse: does low exploitability imply vanishing per-infoset counterfactual regret?

### Takeaway
Zinkevich's Theorem 3 is one-directional: the overall (external) regret is at most the sum of positive immediate counterfactual regrets, under perfect recall. It does **not** reverse. Immediate counterfactual regret weights by opponent and chance reach only, not by the player's own reach. Counterfactual regret at infosets the player itself (almost) never reaches can therefore stay large while the external regret, and hence exploitability, is ~0. I found no paper that states this converse failure as a theorem with a named counterexample. The closest is Kroer & Sandholm (2016) introducing *self-trembling* equilibria, i.e., a requirement of bounded counterfactual regret everywhere, which is precisely the gap between NE and per-infoset counterfactual optimality.

### Cited Findings
- [V-snip] Zinkevich, Johanson, Bowling, Piccione (NIPS 2007):
  - Theorem 3: a player's overall regret is bounded by the sum over that player's information sets of the positive part of the immediate counterfactual regret, R_i^T ≤ Σ_{I∈𝓘_i} R_{i,imm}^{T,+}(I).
  - Theorem 2: in a zero-sum game, if both players' average overall regret is below ε, σ̄^T is an approximate equilibrium. The search summary said "ε"; Deep CFR and other restatements give **2ε**, which I believe is the original statement.
  - Minimizing immediate counterfactual regret, which can be done independently per information set, therefore yields an approximate NE in self-play.

  — [Zinkevich et al. 2007 (NeurIPS)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf); [U. Alberta copy](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf); [Deep CFR restatement](https://arxiv.org/pdf/1811.00164)
- [V-snip] Counterfactual utility is defined with the *opponent's* (and chance's) reach probability only. — [Waugh et al. 2015, arXiv 1411.7974](https://arxiv.org/pdf/1411.7974)
- [V-abs] MCCFR: sampled updates equal CFR updates in expectation, and outcome or external sampling have "bounded overall regret with high probability." — [Lanctot et al. NIPS 2009](http://papers.neurips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games.pdf)
- [V-abs] Kroer & Sandholm EC 2016 frame guarantees for "Nash equilibria **and approximate self-trembling equilibria**". That is a solution concept beyond NE, needed for approximate solutions in IR abstractions. — [ACM DL](https://dl.acm.org/doi/10.1145/2940716.2940736)

### Inferences
- **Converse fails (elementary counterexample, derived here).** Take a perfect-recall game in which player i has an infoset I behind its own action b at the root, with opponent and chance reach to I positive. Let σ_i play b with probability 0 and play a dominated action at I. σ is an exact NE if the root choice is optimal. Exploitability and external regret do not depend on σ_i(I). Yet R_imm(I) = Θ(π_{−i}(I)·gap) > 0 persists. With averages, the iterates can have own-reach → 0 at I while CFR's per-infoset regret at I is never forced down, since regret matching still drives it to 0 at rate 1/√T unless the implementation prunes. So **low exploitability does not imply low per-infoset counterfactual regret**. Unweighted per-infoset counterfactual regret is a strictly stronger, "sequential-rationality-like" property.
- **For abstractions** this matters more. A bucket can merge an original infoset that is reached under the abstract solution with one that is not. Low abstract exploitability then says nothing about the unreached member, yet after lifting, the original-game best responder may exploit it. This is (inferred) the reason Kroer & Sandholm need self-trembling equilibria. It also supports the requester's choice of a counterfactual-regret-based (C + H) bound over an exploitability-based one.
- A tighter, reach-weighted version of Theorem 3 exists in the OCO literature: the laminar regret decomposition (Farina, Kroer, Sandholm), which weights local regrets by own reach of a comparator. It is not verified here (see Gaps). It explains *why* the converse fails: own-reach weights can be 0.

### Gaps
- **[UNVERIFIED-recall] Exact statements.**
  - Zinkevich Theorem 4: under regret matching, R_{i,imm}^T(I) ≤ Δ_{u,i}√|A_i|/√T, so R_i^T ≤ Δ_{u,i}|𝓘_i|√|A_i|/√T.
  - Average-strategy formula: σ̄_i^T(I)(a) = Σ_t π_i^{σ^t}(I)σ^t(I)(a) / Σ_t π_i^{σ^t}(I).
  - Check [the NIPS 2007 PDF](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).
- **[UNVERIFIED-recall] Laminar regret decomposition.** Farina, Kroer, Sandholm, "Online Convex Optimization for Sequential Decision Processes and Extensive-Form Games" (AAAI 2019): sequence-form regret ≤ max_{x̂} Σ_j x̂[p_j]·R̂_j^T. Not fetched.
- **No source found** that explicitly publishes a counterexample to the converse of Theorem 3, or that bounds per-infoset counterfactual regret by exploitability.

## Q7. Which of the requester's results (1)–(3) already appear in the literature?

### Takeaway
- **(1) NashConv(avg) ≤ Σ_i (C_i + H_i)/T.** The *structure* is precedented: CFR regret in Γ′ bounded by an abstract regret term plus abstraction-error terms.
  - Lanctot 2012: K-scaled CFR term plus skew δ term.
  - Kroer–Sandholm 2016: abstract counterfactual regret plus utility and chance errors.
  - Kroer–Sandholm 2018: exact ex-post decomposition.

  I found no *abstraction-agnostic, ex-post, per-bucket* term H defined from realized member regret vectors.
- **(2) The Jensen/Bregman identity with ψ = max(0, max_a ·).** I did not find it anywhere.
- **(3) "Own abstract recall" ⇔ stored average = true average.** I did not find it stated. Its ingredients are folklore: the average needs well-defined own reach (Zinkevich), and IR breaks CFR's guarantees (SARA 2009; Lanctot 2012; Čermák 2017).

### Cited Findings
- [V-snip] The bounds of the "abstract CFR regret + abstraction error ⇒ original-game regret or ε-NE" type are in Lanctot et al. 2012 (well-formed / skew, δ floor) and Kroer & Sandholm 2016 (CRSWF; bounded counterfactual regret ⇒ ε′-NE, linear in height). — [arXiv 1205.0622](https://arxiv.org/pdf/1205.0622); [arXiv 1409.3302](https://arxiv.org/pdf/1409.3302); [Kroer thesis](http://www.columbia.edu/~ck2945/files/ckroer_computer_science_2018.pdf)
- [V-abs] An exact but ex-post (strategy-dependent) abstraction-error decomposition is in Kroer & Sandholm 2018. — [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/aa942ab2bfa6ebda4840e7360ce6e7ef-Abstract.html)
- [V-abs] A regret bound in terms of how well a function class (which includes bucket indicators) approximates regret vectors is in Waugh et al. 2015. — [arXiv 1411.7974](https://arxiv.org/pdf/1411.7974)
- [V-snip] The average strategy in IR games lacks value guarantees. — [Čermák et al. figure](https://www.researchgate.net/figure/An-imperfect-recall-game-where-CFR-can-reach-a-non-optimal-strategy_fig1_313861094)

### Inferences
- **(1) Positioning.** Present it as an abstraction-agnostic, ex-post analogue of Lanctot/Kroer–Sandholm. It needs no well-formedness: H absorbs all structural mismatch, and H = 0 recovers the well-formed case through positive homogeneity of ψ. Two caveats:
  - The folk-theorem conclusion applies to the **Γ′ (lifted, sequence-form) average**. By (3), it transfers to the stored bucket average only under own abstract recall, or with an extra term for the mixture gap Σ_{I′} (W_{I′}/W_B)(σ̄′(I′) − σ̄_B).
  - Per the project AGENTS.md, keep NashConv/exploitability distinct from LBR measurements.
- **(2) Positioning.** This appears new as a closed form. It is worth relating to Kroer–Sandholm's "bound increase as clustering distance" and to RCFR-style regression error. The Bregman form explains why scale-only differences between members cost nothing, which squared-error clustering of regret or CFV vectors would penalize.
- **(3) Positioning.** The algebra is elementary: the stored average is the W-weighted mixture of member averages. What is new is stating it as a characterization and pointing out that in Waugh-style IR poker abstractions the faithful average is generally not bucket-measurable. Reference A-loss recall only to say that it is *not* the relevant condition.

### Gaps
- None of these positioning claims was checked against the full texts of Kroer–Sandholm 2016 and 2018, Lanctot's thesis, Gibson's thesis or Johanson's thesis. The proxy blocked all PDF hosts this session. A follow-up with full-text access should check:
  - (a) the exact Lanctot Theorem 1/2 constants and whether MCCFR is covered;
  - (b) the CRSWF error terms and the self-trembling definition;
  - (c) the form of the NeurIPS 2018 ex-post decomposition;
  - (d) whether CFR+IRA keeps the average in the original game;
  - (e) any thesis remark on IR average-strategy computation.
