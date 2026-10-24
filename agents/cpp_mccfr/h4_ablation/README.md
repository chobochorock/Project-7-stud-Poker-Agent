# H4 12-action ablation

초기 4장에서 버릴 카드 1장과 공개할 카드 1장을 고르는 12행동만 학습한다.
두 플레이어는 서로의 선택을 보기 전에 선택한다. 기존 5~7구 DT/KM **평균 베팅
전략은 10M 체크포인트 그대로 고정**한다. 3~4구 베팅 추가나 전체 게임 공동학습이 아니다.

## Run

Project root, PowerShell:

```powershell
python -m agents.cpp_mccfr.h4_ablation.run test --run agents/cpp_mccfr/data/h4_ablation/test_20261008
python -m agents.cpp_mccfr.h4_ablation.run train --run agents/cpp_mccfr/data/h4_ablation/seed2026100851_100k --hands 100000
python -m agents.cpp_mccfr.h4_ablation.run lbr --run agents/cpp_mccfr/data/h4_ablation/seed2026100851_100k --particles 64 --pairs 250 --eval-seeds 2026100841 2026100842
python -m agents.cpp_mccfr.h4_ablation.review --run agents/cpp_mccfr/data/h4_ablation/seed2026100851_100k
```

`--source` defaults to `../data/dynamic_tree_validated/centered_seed222_freeze1m_10m`.
One training hand means one fresh initial deal **per model**, two player updates and
24 terminal betting rollouts/model. Thus 100k is 2.4M rollouts/model, not 100k betting
nodes. DT/KM share deal/action seeds, but their evolving policies produce different
trajectories. No heuristic training labels or warm-start regrets are injected.

Repeat the same train command with a larger `--hands` to resume the saved checkpoint.
This is a total target, not additional hands. Models save at `--panel-every` (default
10k) and the requested end. Killing a process can lose the incomplete interval.
Training is deterministic per global hand; tests check uninterrupted/resumed identity.
If interrupted during a panel, `panel.csv` may have missing/duplicate rows on manual
restart; the binary model remains atomic. No automatic cleanup of historical logs.

## Algorithm and limits

- Separate signed regret and SIMPLE average tables for each model and player seat.
- Reuse existing rank/suit canonical H4 representation and its 12-action bijection.
  This reduces equivalent suit relabelings; it is not a new k-means hand bucket.
  The fixed downstream abstraction may not be exactly suit-invariant, so do not claim
  a full-game equilibrium guarantee for this approximate reduced game.
- On each player update: sample the opponent's current H4 action, enumerate all own
  actions, sample fixed downstream betting, add `Q(a) - dot(policy,Q)` to regrets.
  Freeze both root policies until both player updates from the deal are ready;
  apply the two updates simultaneously, avoiding adaptation to the same sampled deal.
  Preserve negative regret. Regret matching uses only positive entries. Uniform
  initialization/unseen H4 fallback, no heuristic fallback and no minimum-touch cutoff.
- Chance and opponent actions are sampled; learning is an application of sampled
  regret minimization, **not CFR+**, not CFR-D. It does not retrain the downstream
  betting policy on the changed hand distribution.
- A 100k pilot may have only a few visits per rare H4 information set. Neither training
  time nor the number of available actions is evidence of convergence.

## Metrics

`panel.csv`: every 10k, 64 fixed own-four-card/seat roots, 32 independent conditional
deals each. All unseen cards are reshuffled, including opponent cards and future deals.
Both H4 players use their current saved **average H4 policy**; all betting uses fixed
average betting. Average sampled action values **before** taking their maximum:

`gap(I) = max_a Q(I,a) - sum_a p(a|I) Q(I,a)`.

This is a **pre-selection H4 Local BR-gap**, not the old fifth-street metric. The
plug-in maximum is upward biased; cross-fitted selected-action advantage (can be
negative) and mean action SE are logged separately. Model/KM ratios use matched roots,
particles, seeds and training budgets. Average H4 is the evaluation target; current
H4 is used for training, not silently mixed into this curve.

LBR: compare DT/KM, each with heuristic H4 and learned average H4. The attacker still
uses heuristic H4 and the same existing 5~7구 myopic check-down LBR. It is **not a BR
over all 12 attacker H4 choices**, nor exact exploitability. Attacker profit is ante/hand,
lower is better for the target. Use linear axes because profit can be negative.

The learned target's H4 likelihood is included in the hidden-card posterior: for a
candidate initial four cards and the observed reveal, weight by the probability of
selecting that discard/reveal pair. Sampling and likelihood use the same seat-specific
average H4 policy. Zero likelihood is rejected; unseen H4 has uniform probability.
Opponent betting likelihoods, particle budgets and LBR EV formulas otherwise remain
unchanged. Metadata includes missing betting queries, unseen actual H4 and mean ESS.
Missing-query counters cover LBR's internal target-model queries, not every training
rollout or the target's direct action call.

Evaluation seeds are separate from training and panel seeds. Pair each shuffled deal
across attacker seats and all four arms; CI resampling preserves these pairs and
stratifies by evaluation seed. Multiple evaluation seeds do **not** replace multiple
independent training seeds. Heavy-tailed all-ins may dominate a small evaluation.

## Files and provenance

- `h4_checkpoint.bin`: both models' regret/average/touch tables, hand count and seed.
- `config.json`, `train.log`, `panel.csv`: configuration and learning diagnostics.
- `lbr_<seed>.csv/.json`, `summary.json`, `comparison.png`: evaluation and uncertainty.
- `input_integrity.json`: input checkpoint/atlas/config SHA256 unchanged.

`run.py` mechanically extracts the existing LBR/match/continuation bodies into a
generated build header. Only the H4 likelihood call and necessary template/type names
change; anchored replacements fail if the original source shape changes. This avoids
modifying sources whose hashes protect the user's ongoing separate training. Test
mode compares the heuristic adapter's decisions/payoffs to the original evaluator,
checks hidden-information privacy, H4 probability/action mapping and signed updates.

## Related work

[Lanctot et al., MCCFR (2009)](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)
provides the sampled-regret framework. Here only the initial simultaneous choice is
learned, with a frozen stochastic betting continuation. This is a controlled ablation,
not a new algorithm or proof of joint-game convergence.

[Lisy and Bowling, LBR (2017)](https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf)
motivates the approximate attacker. Our particle/rules/initial-action adapter is not
an exact replication of their no-limit hold'em implementation. No novelty claim.
