# Clustering Agent: Theoretical Basis and PAC Bounds

> 요약: 현재 에이전트는 cluster responsibility를 feature로 사용하는
> 선형 Q 근사기다. 고정된 encoder/cluster와 독립적인 bounded rollout을
> 가정하면 component Q에 PAC concentration bound를 줄 수 있고, 이를
> 유한 horizon의 정책 손실 bound로 연결할 수 있다. 그러나 현재의
> adaptive UCT label, 동일 shard 재사용, shared-table self-play, 무작위 H4까지
> 포함한 end-to-end PAC 보장은 아직 성립하지 않는다.

This note formalizes the agents currently implemented in
`agent/cluster_agent.py`, `clustering_train.py`, and
`cluster_q_learning.py`. It deliberately distinguishes a theorem that follows
under explicit assumptions from behavior that is only empirical in the current
code.

## 1. Scope

The current system has three stages.

1. UCT produces action-value estimates at observed information histories.
2. An MLP maps each history to a latent vector, and spherical k-means or a
   diagonal GMM maps that vector to soft cluster responsibilities.
3. Each cluster stores an action-value prototype. At inference, prototypes are
   mixed by responsibility and the legal action with the largest mixed value is
   selected.

The MLP representation is not purely unsupervised: its encoder is trained to
predict UCT Q targets. Clustering is unsupervised only after that value-aware
representation has been learned.

The present proof concerns betting decisions in heads-up EV mode. The base
`ClusterPokerAgent` still chooses the H4 discard/reveal pair randomly, so no
whole-agent optimality statement currently includes H4.

## 2. Decision Process

Let `h` be the player's observable information history: private and public
cards, public opponent cards, betting history, pot, investment, street, seat,
and legal actions. In a partially observed game, `h` is not automatically a
Markov state. The analysis therefore needs one of these assumptions.

- `h` determines an exact belief state `b(h)` and is a sufficient information
  state; or
- all values are interpreted only with respect to the fixed data-generating
  policy and opponent population.

The first interpretation permits an optimal-policy claim. The second permits
only prediction of that fixed teacher/data distribution.

Because the effective stack and raise count are bounded, an episode has a
finite maximum number `H` of the player's betting decisions. Let `A` be the
number of encoded betting actions; currently `A = 6`.

## 3. Current Function Class

### 3.1 Encoder

The MLP encoder gives

\[
z(h)=f_\theta(x(h))\in\mathbb R^d.
\]

The current default latent dimension is `d = 32`. Its policy and Q heads are
used during representation learning; the cluster agent later reuses `z`.

### 3.2 Spherical k-means gate

Let

\[
\bar z=\frac{z}{\max(\lVert z\rVert_2,10^{-12})}
\]

and let `c_k` be a normalized centroid. The score is

\[
s_k(h)=\frac{\bar z(h)^\top c_k}{\tau}.
\]

For the top-`L` score set `T_L(h)`, inference uses

\[
r_k(h)=
\begin{cases}
\dfrac{e^{s_k(h)}}{\sum_{j\in T_L(h)}e^{s_j(h)}} & k\in T_L(h),\\
0 & \text{otherwise}.
\end{cases}
\]

Training of the component table uses hard nearest-centroid membership, while
inference uses the soft top-`L` gate. This difference must be included in the
abstraction error below.

### 3.3 Diagonal GMM gate

The latent is first globally whitened:

\[
u(h)=(z(h)-m)W.
\]

For mixture weight `omega_k`, mean `mu_k`, and diagonal variance
`sigma_k^2`, the component log-joint is

\[
\ell_k(h)=\log\omega_k-rac12\left[
\sum_j\log(2\pi\sigma_{kj}^2)
+\sum_j\frac{(u_j(h)-\mu_{kj})^2}{\sigma_{kj}^2}
\right].
\]

The responsibility is the top-`L` softmax of these log-joints. Thus

\[
r(h)\in\Delta^{K-1},\qquad r_k(h)\ge 0,\qquad \sum_k r_k(h)=1.
\]

This is both a soft state aggregation and a gating network for a mixture of
local constant experts.

### 3.4 Component Q estimator

For training state `i` and action `a`, let

- `n_ia` be the UCT action visits,
- `X_iaj` be normalized rollout return `j`,
- `qhat_ia = (1/n_ia) sum_j X_iaj`, and
- `g_ik` be the training membership.

For GMM, `g_ik` is the soft responsibility. For spherical k-means, it is the
hard assignment. The implementation uses

\[
w_{ika}=g_{ik}\sqrt{n_{ia}}
\]

and stores

\[
\widehat q_{ka}
=
\frac{\sum_i w_{ika}\widehat q_{ia}}
     {\sum_i w_{ika}}.
\tag{1}
\]

At inference,

\[
\widehat Q(h,a)=\sum_{k=1}^{K}r_k(h)\widehat q_{ka}.
\tag{2}
\]

The Q agent selects

\[
\widehat\pi(h)\in
\arg\max_{a\in A(h)}\widehat Q(h,a).
\tag{3}
\]

The policy agent instead stores a component policy `p_k` and samples from

\[
\widehat\pi(a\mid h)=\sum_k r_k(h)p_k(a).
\tag{4}
\]

Equation (2) is a finite-basis linear Q approximator whose features are the
cluster responsibilities. It is also a soft nearest-prototype Q method.

## 4. Online TD Update

Let

\[
C_t=\max(\text{ante},\text{pot}_t,1)
\]

be the Q normalization and let `I_t` be the player's cumulative investment at
its decision time. The represented quantity is

\[
Q(h_t,a_t)
=
\mathbb E\left[\frac{G+I_t}{C_t}\mid h_t,a_t\right],
\tag{5}
\]

where `G` is final net chips. At the next decision,

\[
G+I_t=(G+I_{t+1})-(I_{t+1}-I_t),
\]

so the code's nonterminal target

\[
y_t=
\frac{C_{t+1}\max_a\widehat Q(h_{t+1},a)
      -(I_{t+1}-I_t)}{C_t}
\tag{6}
\]

is the correct Bellman rescaling if the next Q estimate is exact. At terminal,

\[
y_T=\frac{G+I_T}{C_T}.
\tag{7}
\]

For the selected action, `cluster_q_learning.py` performs

\[
\widehat q_{ka}leftarrow
\widehat q_{ka}
+\alpha r_k(h_t)\operatorname{clip}
\left(y_t-r(h_t)^\top\widehat q_a,-c,c\right).
\tag{8}
\]

Without clipping, (8) is the semi-gradient update for squared TD error. With
clipping, it is the semi-gradient of a Huber TD loss. The cluster map is fixed
during this phase.

## 5. What State Abstraction Must Preserve

Density similarity alone is insufficient. Define a teacher value `Q_T`, such
as the expected result of the fixed UCT search protocol. The relevant
abstraction error on a covered set `C` is

\[
\varepsilon_{\mathrm{abs}}
=
\sup_{h\in C,a}
\left|
Q_T(h,a)-r(h)^\top q_a^\dagger
\right|,
\tag{9}
\]

where `q^dagger` is the population version of the component table. This term
contains all of the following.

- states with different action values sharing a component;
- k-means hard-membership training versus soft-membership inference;
- GMM top-`L` truncation;
- information lost by the MLP encoder; and
- component-local Q variation that a constant prototype cannot represent.

For a full GMM posterior `p` with omitted top-`L` mass `m(h)`, truncation alone
changes an expectation of a bounded component value by at most

\[
m(h)\left(q_{\max,a}-q_{\min,a}\right).
\tag{10}
\]

Approximate Q-value state abstraction is theoretically appropriate here:
states may be merged when all action values are close. In the discounted case,
an abstraction with within-cluster optimal-Q diameter at most `epsilon` has a
known value-loss bound proportional to `epsilon/(1-gamma)^2`. The finite-
horizon bound used below is simpler and matches this game more directly.

## 6. A Conditional PAC Bound for the Offline Q Table

This section gives a finite-sample bound that matches estimator (1).

### Assumptions

For this theorem only, assume:

1. the encoder, clusters, memberships `g_ik`, and visit counts `n_ia` are fixed
   independently of the rollout returns used in (1);
2. each normalized return lies in an interval of width `B`;
3. returns are independent, or are conditionally centered bounded martingale
   differences with predictable weights; and
4. every component-action pair being certified has positive support.

Let

\[
\mu_{ia}=\mathbb E[X_{iaj}]
\]

and define the population target of estimator (1):

\[
q_{ka}^\dagger
=
\frac{\sum_i g_{ik}\sqrt{n_{ia}}\mu_{ia}}
     {\sum_i g_{ik}\sqrt{n_{ia}}}.
\tag{11}
\]

Define

\[
N_{\mathrm{eff}}(k,a)
=
\frac{
\left(\sum_{i:n_{ia}>0}g_{ik}\sqrt{n_{ia}}\right)^2
}{
\sum_{i:n_{ia}>0}g_{ik}^2
}.
\tag{12}
\]

When all memberships are hard and every included root has `m` rollouts for
the action, this is exactly `number_of_roots * m`.

### Theorem 1: component-table concentration

For `K` components and `A` actions, with probability at least `1-delta`,
simultaneously for every supported `(k,a)`,

\[
\left|\widehat q_{ka}-q_{ka}^\dagger\right|
\le
B\sqrt{
\frac{\log(2KA/\delta)}{2N_{\mathrm{eff}}(k,a)}
}.
\tag{13}
\]

### Proof

Write

\[
W_{ka}=\sum_i g_{ik}\sqrt{n_{ia}}.
\]

Substituting each root mean into (1) gives

\[
\widehat q_{ka}-q_{ka}^\dagger
=
\sum_i\sum_{j=1}^{n_{ia}}
\frac{g_{ik}}{W_{ka}\sqrt{n_{ia}}}
(X_{iaj}-\mu_{ia}).
\]

The sum of squared coefficients is

\[
\sum_i n_{ia}
\left(\frac{g_{ik}}{W_{ka}\sqrt{n_{ia}}}\right)^2
=
\frac{\sum_i g_{ik}^2}{W_{ka}^2}
=
\frac{1}{N_{\mathrm{eff}}(k,a)}.
\]

Weighted Hoeffding therefore bounds one component-action deviation by

\[
2\exp\left(-\frac{2N_{\mathrm{eff}}(k,a)t^2}{B^2}\right).
\]

Set this probability to `delta/(KA)` and apply a union bound over all
component-action pairs. This yields (13). QED.

For a state with responsibility `r(h)`, (13) implies

\[
\left|
r(h)^\top\widehat q_a-r(h)^\top q_a^\dagger
\right|
\le
\sum_k r_k(h)\beta_{ka}
\le \max_{k:r_k(h)>0}\beta_{ka},
\tag{14}
\]

where `beta_ka` is the right side of (13).

## 7. From Q Error to Policy Error

Let

\[
\varepsilon_{\mathrm{teacher}}
=
\sup_{h\in C,a}|Q^*(h,a)-Q_T(h,a)|
\tag{15}
\]

measure finite-budget UCT error, opponent-policy mismatch, and any difference
between the teacher target and the optimal value. Let

\[
\beta_{\max}=\max_{k,a}\beta_{ka}.
\]

Equations (9), (13), and (14) give, with probability at least `1-delta`,

\[
\sup_{h\in C,a}
|Q^*(h,a)-\widehat Q(h,a)|
\le
\varepsilon_{\mathrm{teacher}}
+\varepsilon_{\mathrm{abs}}
+\beta_{\max}.
\tag{16}
\]

### Theorem 2: finite-horizon greedy loss

Suppose all histories reachable by the learned policy remain in `C`, and the
right side of (16) is `epsilon_Q`. If all Q values use one common utility
scale, then

\[
V^*(h_0)-V^{\widehat\pi}(h_0)
\le 2H\varepsilon_Q.
\tag{17}
\]

The current implementation instead uses a state-dependent scale `C_t`. If
`epsilon_t` is measured in that normalized scale, the chip-value statement is

\[
V^*_{\mathrm{chip}}(h_0)-V^{\widehat\pi}_{\mathrm{chip}}(h_0)
\le 2\sum_{t=0}^{H-1} C_t\varepsilon_t
\le 2HC_{\max}\varepsilon_Q.
\tag{17a}
\]

### Proof

At any history, let `a*` maximize `Q*` and let `ahat` maximize `Qhat`. Then

\[
Q^*(h,a^*)
\le \widehat Q(h,a^*)+\varepsilon_Q
\le \widehat Q(h,\widehat a)+\varepsilon_Q
\le Q^*(h,\widehat a)+2\varepsilon_Q.
\]

Thus each decision loses at most `2 epsilon_Q` in common utility units, or
`2 C_t epsilon_t` in chip units under the current normalization. Summing the
performance-difference terms over at most `H` decisions gives (17) or (17a).
QED.

Combining (13) and (17), the current static Q architecture is conditionally
PAC in the following sense:

\[
\Pr\left[
V^*-V^{\widehat\pi}
\le
2HC_{\max}\left(
\varepsilon_{\mathrm{teacher}}
+\varepsilon_{\mathrm{abs}}
+B\sqrt{\frac{\log(2KA/\delta)}{2N_{\mathrm{eff,min}}}}
\right)
\right]
\ge 1-\delta.
\tag{18}
\]

For a desired loss `eta`, this requires

\[
N_{\mathrm{eff,min}}
\ge
\frac{B^2\log(2KA/\delta)}
{2\left[
\eta/(2HC_{\max})-\varepsilon_{\mathrm{teacher}}-
\varepsilon_{\mathrm{abs}}
\right]^2},
\tag{19}
\]

provided the denominator is positive. More data can shrink only the final
statistical term. It cannot repair a poor teacher or a lossy abstraction.

With the current `K = 256`, `A = 6`, and `delta = 0.05`, the statistical
radius in (13) is approximately

\[
\beta_{ka}\approx\frac{2.35B}{\sqrt{N_{\mathrm{eff}}(k,a)}}.
\]

This number is intentionally conservative. It also makes the design trade-off
plain: adding clusters reduces abstraction bias only if each new
component-action still receives enough effective samples.

This is a PAC policy-quality bound, not a full PAC-MDP learning-time theorem.
A PAC-MDP theorem also bounds the number of online steps on which behavior is
not near-optimal. The current learner does not have that property.

## 8. PAC Evaluation of the Final Agent

A simpler guarantee applies to an already frozen agent against a fixed
opponent. Let `Y_j` be the seat-paired profit in ante units and let the
effective stack be `S = 1000` antes. Then

\[
Y_j\in[-S,S].
\]

For `n` independent seat pairs, Hoeffding gives

\[
\Pr\left[
|\widehat J-J|
\le
S\sqrt{\frac{2\log(2/\delta)}{n}}
\right]
\ge 1-\delta.
\tag{20}
\]

This is rigorous but very loose because it pays for the full 1000-ante range.
For sample variance `s^2` and range width `D = 2S`, a two-sided empirical
Bernstein interval is

\[
|\widehat J-J|
\le
\sqrt{\frac{2s^2\log(4/\delta)}{n}}
+\frac{7D\log(4/\delta)}{3(n-1)}.
\tag{21}
\]

The evaluator currently prints `mean +/- 1.96 standard errors`. That is a
normal approximation, not the finite-sample PAC certificate in (20) or (21).

## 9. Why the Current End-to-End System Is Not Yet Proven PAC

The preceding bounds are conditional. The exact current pipeline violates or
does not establish several assumptions.

### 9.1 Data-dependent representation

The encoder, clusters, and component Q table are estimated from the same
training rows. The elementary union-bound proof treats the representation as
fixed before Q-label estimation. A clean certificate needs three independent
splits or shards:

```text
representation shard -> MLP and clusters
Q shard              -> component Q and confidence radii
test shard           -> frozen-agent evaluation
```

Without this split, one needs a uniform-complexity or algorithmic-stability
bound for the learned encoder and clustering procedure.

### 9.2 Adaptive UCT allocation

UCT chooses later simulations using earlier returns, so `n_ia` is generally
return-dependent. Theorem 1 assumes fixed counts or predictable weighting.
Stored return-square sums are sufficient raw material for an anytime empirical
Bernstein/Freedman-style confidence sequence, but `clustering_train.py` does
not yet load or propagate those confidence bounds.

### 9.3 Teacher error is unknown

UCT is consistent under its classical finite-horizon MDP assumptions, but a
finite simulation budget does not make
`epsilon_teacher` zero. Here root sampling, information sets, opponent policy,
and search abstraction add further assumptions. `opponent_policy=random` and
`opponent_policy=uct` also define different teachers.

### 9.4 Coverage is not certified

A GMM has nonzero density everywhere. Responsibility alone therefore does not
prove that an online state is supported by training data. A certified agent
needs a density or distance gate plus a lower bound on `N_eff(k,a)` for every
component used at that state. Outside the covered set `C`, (18) says nothing.

### 9.5 Self-play TD is nonstationary

Both players share and update the same Q table. Therefore each player's
environment changes during learning. In addition:

- the update bootstraps with a max target;
- soft aggregation is function approximation;
- behavior is epsilon-greedy while the target is greedy;
- the learning rate is constant; and
- TD errors are clipped.

Consequently, the current `ClusterQLearningAgent` has no general convergence or
PAC guarantee. Equation (8) is mathematically well-defined and useful as an
experiment, but model capacity growth or more hands alone does not prove
improvement.

For a provable baseline, use hard aggregation, a fixed opponent, every
aggregate-action visited, and Robbins-Monro learning rates. That reduces to a
finite tabular problem. For a genuine PAC-MDP baseline, use an optimistic
finite-state method such as R-MAX or Delayed Q-learning on those hard
aggregates. Soft responsibilities can remain in the practical agent, but they
should be compared with this certified baseline.

### 9.6 Policy-mixture mode

The policy mode is behavioral cloning of component-level UCT visit
distributions. Cross-entropy alone does not imply optimality. Under a fixed
teacher, Pinsker's inequality converts per-state KL error to total-variation
error, but distribution shift can compound across the horizon. Adaptive-root
UCT visit counts are search allocation rather than a policy target, which is
why Q-only training uses `policy_loss_weight = 0`.

## 10. Minimal Path to a Real Certificate

The smallest useful next theoretical experiment is:

1. freeze one encoder and one cluster atlas from shard A;
2. collect shard B with a fixed number of rollouts per root-action;
3. compute (12) and (13) for every component-action;
4. measure held-out abstraction residuals on shard C;
5. refuse cluster-Q reuse when responsibility touches an unsupported or
   high-radius component; and
6. report empirical Bernstein arena bounds in addition to the existing normal
   interval.

This would certify the static cluster-Q agent against a fixed teacher and
opponent distribution. Proving the full shared-table self-play loop should be
deferred until that simpler certificate succeeds.

## 11. Literature

- Singh, Jaakkola, and Jordan,
  [Reinforcement Learning with Soft State Aggregation](https://papers.nips.cc/paper_files/paper/1994/hash/287e03db1d99e0ec2edb90d079e142f3-Abstract.html),
  NeurIPS 1994. This is the closest classical foundation for responsibility-
  weighted cluster values. Its convergence result assumes a fixed soft
  aggregation and a stationary, persistently exploring process.
- Abel, Hershkowitz, and Littman,
  [Near Optimal Behavior via Approximate State Abstraction](https://proceedings.mlr.press/v48/abel16.html),
  ICML 2016. This gives value-loss bounds when aggregated states have similar
  optimal action values or approximately similar models.
- Kocsis and Szepesvari,
  [Bandit Based Monte-Carlo Planning](https://sites.ualberta.ca/~szepesva/papers/ecml06.pdf),
  ECML 2006. This establishes the UCT planning basis and consistency under its
  finite-horizon or discounted MDP assumptions.
- Strehl, Li, and Littman,
  [Reinforcement Learning in Finite MDPs: PAC Analysis](https://www.jmlr.org/papers/v10/strehl09a.html),
  JMLR 2009. This defines the stronger PAC-MDP learning-time standard and
  analyzes R-MAX and Delayed Q-learning.
- Maurer and Pontil,
  [Empirical Bernstein Bounds and Sample Variance Penalization](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf),
  COLT 2009. This supplies the finite-sample variance-sensitive bound used in
  (21).
- Baird,
  [Residual Algorithms: Reinforcement Learning with Function Approximation](https://leemon.com/papers/1995b.pdf),
  ICML 1995. This demonstrates why bootstrapping and general function
  approximation cannot inherit tabular convergence guarantees automatically.
