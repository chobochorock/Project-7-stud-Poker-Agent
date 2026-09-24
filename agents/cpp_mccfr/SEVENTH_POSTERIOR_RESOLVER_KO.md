# PBS 조건부 7th 실시간 resolver

## 1. 문서의 범위

이 문서는 7-stud heads-up EV 게임의 7th street에서 사용하는
`ExactSeventhResolverPolicy`의 설계, 수식, 구현, 실험 결과, 한계를 설명한다.
웹에서는 이 구성을 **PBS 7th (30M)** 이라는 이름으로 제공한다.

이 구현의 가장 중요한 성과는 다음 두 가지다.

1. 상대 hidden card를 남은 덱에서 균등하게 뽑던 기존 resolver를, 관측된
   H4 결과와 5th~7th betting history에 조건부인 posterior sampling으로
   바꾸었다.
2. 7th에서 posterior가 허용하는 사적 상태만 반복 탐색하므로, 깊은 탐색에서
   생성되는 local node와 실행 시간이 크게 감소했다.

다만 이름에 PBS가 들어가더라도 **원 논문의 완전한 ReBeL 구현은 아니다.**
현재 알고리즘은 의사결정자의 실제 private hand를 고정한 채 상대 range만
posterior로 구성하는 `PBS-conditioned local resolver`다. 두 플레이어의 전체
range를 공통 지식 상태로 유지하는 full PBS, value network, safe-resolving
gadget은 아직 없다.

---

## 2. 문제 정의

7th street에는 더 이상 새 카드가 배분되지 않는다. 따라서 현재까지의
공개 정보와 두 플레이어의 private hand가 정해지면 남은 게임은 베팅과
showdown만 있는 유한 게임이다.

의사결정자 `i`가 아는 정보는 다음과 같다.

- 자신의 hidden card 3장
- 자신의 public card 4장
- 자신의 H4 discarded card
- 상대 public card 4장
- 5th, 6th, 7th의 정확한 betting history
- 현재 pot, 투자액, legal action
- H4에서 관측한 상대 reveal card

모르는 것은 다음과 같다.

- 상대의 H4 hidden card 2장
- 상대가 H4에서 버린 card 1장
- 상대의 7th hidden card 1장

기존 resolver는 모르는 카드 네 장을 관측 이력과 무관하게 남은 덱에서
균등 추출했다. 이 방식은 카드 충돌만 방지할 뿐, 상대가 지금까지 보인
행동이 어떤 hand에서 나왔을 가능성이 큰지 반영하지 않는다.

예를 들어 상대가 5th와 6th에서 계속 강하게 bet했다면, 모든 합법 hand를
균등하게 뽑는 range는 지나치게 약하다. 반대로 상대가 계속 check/call했다면
강한 hand가 과대 대표될 수 있다. 잘못된 range를 아무리 깊게 풀어도 그
range에 대한 정답만 더 정확히 구하게 된다.

---

## 3. Public Belief State와 현재 구현의 위치

### 3.1 Full PBS

이론적인 public belief state는 public history `h_pub`가 주어졌을 때 각
플레이어의 private state에 대한 조건부 분포를 함께 가진다.

\[
\beta(h_{1},h_{2}\mid h_{\mathrm{pub}}).
\]

ReBeL 계열에서는 이 PBS와 continuation value를 상태로 사용해 search와
learning을 결합한다. 중요한 점은 PBS가 특정 한 플레이어만의 추측이 아니라
public history로부터 계산되는 공통 지식 표현이라는 것이다.

### 3.2 현재 구현

현재 resolver는 실제 의사결정자 `i`의 private hand `h_i`를 알고 있으므로,
full joint PBS 중 다음 조건부 slice만 만든다.

\[
\beta_i(h_{-i})
=P(h_{-i}\mid h_i,h_{\mathrm{pub}}).
\]

이 분포를 particle로 근사한 뒤 7th local MCCFR의 determinization 분포로
사용한다. 따라서 현재 구현은 다음과 같이 부르는 것이 정확하다.

> Blueprint-conditioned opponent posterior를 사용하는 exact-history 7th
> local MCCFR resolver

Full ReBeL과 공유하는 핵심은 **public history로 range를 갱신한 뒤 그 range
위에서 subgame을 다시 푼다**는 점이다. 다른 점은 value network 없이 7th를
terminal까지 직접 풀고, joint PBS 대신 한 플레이어 관점의 posterior만
만든다는 점이다.

---

## 4. Bayesian posterior

상대 private hand 후보를 `h`라고 하자. 관측된 상대 행동열을
`a_1, ..., a_T`, 해당 시점의 정보집합을 `I_t(h)`라고 하면 후보의
비정규화 가중치는 다음과 같다.

\[
w(h)
\propto
P(h\mid\text{known cards})
\mathbf 1[\text{H4-compatible}(h)]
\prod_{t:a_t\text{ is opponent action}}
\bar\sigma_{\mathrm{BP}}(a_t\mid I_t(h)).
\]

여기서 `BP`는 30M MCCFR blueprint다.

현재 candidate proposal은 합법인 남은 카드에서 균등하게 생성되므로
`P(h | known cards)` 항은 proposal 안에서 동일하다. 구현에서 실제로
비교하는 값은 다음 log-likelihood다.

\[
\ell(h)
=
\sum_{t:a_t\text{ is opponent action}}
\log\max\left(10^{-12},
\bar\sigma_{\mathrm{BP}}(a_t\mid I_t(h))\right).
\]

수치 underflow를 피하기 위해 최대 log-weight `m`을 빼고 정규화한다.

\[
\hat w_j = \exp(\ell(h_j)-m),
\qquad
p_j=\frac{\hat w_j}{\sum_k\hat w_k}.
\]

자신의 행동 확률은 likelihood에 곱하지 않는다. 자신의 행동은 자신의 정보와
정책에서 생성되며, 이미 자신의 private hand와 관측 history에 조건부이기
때문에 상대 hand를 추가로 구별하는 직접적인 관측으로 사용하지 않는다.

### 4.1 Posterior의 의미

이 posterior는 상대의 실제 정책을 아는 완전한 Bayesian posterior가 아니다.
상대가 blueprint를 따른다고 놓은 **모델 기반 posterior**다. 실제 상대가
blueprint와 크게 다르면 range도 편향된다. 그러나 기존 균등 range보다
betting history를 설명하는 hand에 질량을 주며, 적어도 이미 가진 30M 정책과
일관된 추론을 한다.

---

## 5. Particle 생성

### 5.1 Known-card mask

먼저 52장 전체에 대해 현재 viewer가 확실히 아는 카드를 제거한다.

- viewer hidden 3장
- viewer shown 4장
- viewer discarded 1장
- opponent shown 4장

남은 카드에서 상대의 미관측 상태를 제안한다. 중복 카드는 허용하지 않는다.

### 5.2 H4 조건

상대의 첫 public card는 H4에서 상대가 직접 공개한 카드다. 후보 생성 시
이 관측을 이용한다.

1. 관측된 reveal card와 남은 카드 3장을 합쳐 H4 초기 4장을 만든다.
2. 카드 순서를 무작위화한다.
3. 현재 deterministic heuristic `discard_reveal()`을 실행한다.
4. heuristic이 선택한 reveal이 실제 관측 reveal과 같은 후보만 수용한다.
5. 선택된 discard를 후보의 discarded card로 기록한다.
6. 나머지 hidden 2장에 별도로 뽑은 7th hidden card를 더한다.

즉, H4 정책이 deterministic이라는 현재 게임 가정 아래에서 관측 H4 행동과
모순되는 private hand는 posterior support에서 제거된다.

주의할 점은 이것이 학습된 H4 정책의 확률 likelihood가 아니라 hard
compatibility filter라는 것이다. H4가 stochastic policy로 바뀌면
indicator 대신 다음 항이 필요하다.

\[
P(\text{observed discard/reveal}\mid h,\pi_{H4}).
\]

### 5.3 Rejection limit

요청 particle 수를 `N`이라고 할 때 최대 proposal 수는 다음과 같다.

\[
\max(64N, N+64).
\]

현재 웹 프리셋은 `N=64`다. H4 조건을 만족하는 후보가 64개 모이거나 proposal
한도에 도달하면 중단한다. 후보가 하나도 없으면 posterior fallback을 기록하고
기존 uniform determinization을 사용한다.

---

## 6. Betting history likelihood

후보마다 실제 betting history를 5th 시작부터 재생한다.

### 6.1 5th 시작 상태 복원

복원 상태는 다음으로 초기화한다.

- `street = 5`
- `pot = 2 * ante`
- 양쪽 `invested = ante`
- 양쪽 public card는 5th 시점의 앞 3장
- viewer hidden은 H4 뒤의 hidden 2장
- opponent hidden은 후보의 H4 hidden 2장
- 양쪽 stack cap은 실제 상태에서 복사
- 첫 actor는 공개 카드 우선순위 규칙으로 계산

현재 규칙에서는 H4 뒤 5th부터 첫 betting round가 시작하므로 이 복원이
성립한다.

### 6.2 Street 전환

history event의 street가 증가하면 round bet, highest bet, raise count를
초기화한다.

- 6th 진입: 실제로 관측된 네 번째 public card를 양쪽에 추가
- 7th 진입: viewer의 실제 7th hidden과 후보의 7th hidden을 추가
- 새 street의 first bettor를 다시 계산

### 6.3 Action likelihood

각 event에서 actor와 street가 복원 상태와 일치하는지 검사한다. 상대의
action이면 30M blueprint의 평균 정책을 조회하고 해당 action 확률의 log를
누적한다. 그 뒤 실제 action을 적용해 pot, invested, current bet, legal action을
동일하게 전개한다.

불가능한 actor 순서, 조기 fold, 잘못된 street 전이는 후보를 무효화한다.

이 과정의 장점은 단순한 `raise count` 요약이 아니라 실제 관측된 action
sequence 전체가 posterior에 반영된다는 점이다. Blueprint 자체의 정보집합은
`power-memory16` abstraction이지만, likelihood 계산 시 어떤 action이 언제
발생했는지는 정확히 재생된다.

---

## 7. Exact 7th information set

Local resolver의 `ExactSeventhKey`는 다음을 포함한다.

- viewer seat
- legal action mask
- viewer hidden card 최대 3장
- viewer shown card 최대 4장
- opponent shown card 최대 4장
- viewer discarded card
- 5th~7th exact betting history 최대 32 event

각 history token에는 다음이 함께 들어간다.

- street
- viewer 기준 상대/자기 actor
- action 종류

`--seventh-hand-history`를 켜면 card의 관측 순서를 보존한다. 끄면 각 card
묶음을 정렬한다. 웹 프리셋과 성능 실험은 이 옵션을 켠다.

상대 hidden card는 상대의 정보집합에서는 자기 hidden으로 나타나지만,
viewer의 정보집합 key에는 들어가지 않는다. 따라서 local tree가 상대의
실제 hand를 viewer에게 누설하지 않는다.

현재 key에 pot과 invested를 직접 저장하지는 않는다. 고정 ante/stack 규칙과
정확한 betting history가 이 값을 결정하고 legal mask가 현재 행동 가능성을
확인한다. 향후 variable stack이나 다른 bet sizing을 섞을 때에는 stack/pot
field를 key에 명시적으로 추가해야 한다.

---

## 8. Local MCCFR

### 8.1 Blueprint prior

새 local information set `I`가 처음 나타나면 blueprint 평균 정책
`\bar\sigma_BP(I)`를 읽어 local regret과 average-strategy mass를 초기화한다.

\[
R_0(I,a)=\lambda\bar\sigma_{BP}(I,a),
\qquad
S_0(I,a)=\lambda\bar\sigma_{BP}(I,a).
\]

여기서 `\lambda = --seventh-resolve-prior`이며 웹 프리셋은 100이다. 이는
탐색 초기에 blueprint에서 완전히 벗어나는 것을 줄이는 warm start다. 반복이
쌓이면 local counterfactual regret가 prior를 덮어쓴다.

### 8.2 Traversal

각 iteration은 다음 순서로 실행된다.

1. posterior particle 하나를 가중 sampling한다.
2. 그 particle로 상대 hidden/discarded card를 determinize한다.
3. iteration마다 traverser를 두 플레이어 사이에서 교대한다.
4. traverser node에서는 legal action을 모두 전개한다.
5. 비-traverser node에서는 현재 regret-matching policy로 action 하나를
   sampling한다.
6. terminal fold 또는 showdown chip payoff를 root까지 반환한다.
7. traverser action regret를 CFR+ 방식으로 0 아래에서 clipping한다.

7th에는 미래 chance card가 없으므로 traversal은 betting tree를 terminal까지
직접 계산한다. Leaf value network나 V7 approximation은 사용하지 않는다.

### 8.3 출력 정책

실제 action은 local node의 평균 전략을 사용한다.

\[
\bar\sigma_{local}(a\mid I)
=\frac{S(I,a)}{\sum_b S(I,b)}.
\]

현재 웹 stdio 경로는 **각 7th 의사결정마다 현재 상태에서 subgame을 새로
구성한다.** 이전 핸드나 이전 root의 local node가 다음 요청으로 누출되지
않는다. 5th와 6th에서는 30M blueprint를 그대로 사용한다.

---

## 9. 전체 실행 흐름

```text
H4
  -> deterministic heuristic discard/reveal

5th, 6th
  -> power-memory16 30M blueprint policy

7th decision
  -> known-card mask 생성
  -> H4-compatible opponent hand 후보 64개 생성
  -> 5th부터 exact history replay
  -> blueprint action likelihood로 posterior weight 계산
  -> weighted posterior에서 determinization sampling
  -> exact-history local MCCFR 10,000회
  -> local average strategy에서 실제 action sampling

다음 7th decision
  -> 바뀐 public state와 history에서 위 과정을 다시 수행
```

의사코드로 쓰면 다음과 같다.

```text
resolve(root, viewer):
    posterior = []
    while len(posterior) < 64 and proposals < limit:
        h = propose_legal_private_hand(root)
        if not h4_compatible(h, observed_reveal):
            continue
        log_w = replay_history_and_score_blueprint(root.history, h)
        if finite(log_w):
            posterior.append((h, log_w))

    normalize_log_weights(posterior)
    local_table.clear()

    for t in 1..10000:
        h = weighted_sample(posterior)
        state = determinize(root, h)
        traverser = viewer if t is even else opponent
        external_sampling_cfr_plus(state, traverser, local_table)

    return sample(local_average_policy(root_information_set))
```

---

## 10. 계산량과 메모리

`N`을 posterior particle 수, `H`를 history event 수, `T`를 resolver iteration,
`V`를 한 traversal의 평균 방문 node 수라 하자.

Posterior 구축 비용은 대략 다음과 같다.

\[
O(NH \cdot C_{BP}),
\]

여기서 `C_BP`는 blueprint policy lookup과 power bucket 계산 비용이다.

Local solve 비용은 다음과 같다.

\[
O(TV).
\]

가중치 누적합을 한 번 만든 뒤 각 determinization은 `lower_bound`로 뽑으므로
particle sampling 비용은 다음과 같다.

\[
O(\log N).
\]

얕은 `T`에서는 posterior를 만드는 고정비가 손해일 수 있다. 깊은 `T`에서는
불가능하거나 매우 희박한 private state가 제거되어 local tree branching과
node 수가 크게 줄어든다. 실제 smoke test가 정확히 이 교차점을 보여준다.

Local table은 subgame마다 `clear()`되고 다음 root를 위해 보존되지 않는다.
따라서 장기 실행에서 모든 과거 local node가 누적되는 구조는 아니다. 최대
메모리는 주로 한 번의 7th resolve에서 생성되는 exact local node 수에 의해
결정된다.

---

## 11. CLI

핵심 옵션은 다음과 같다.

| 옵션 | 의미 | 웹 프리셋 |
|---|---|---:|
| `--seventh-resolve-iterations` | 7th local MCCFR 반복 수 | 10,000 |
| `--seventh-resolve-prior` | blueprint warm-start 질량 | 100 |
| `--seventh-posterior-particles` | 상대 posterior 후보 수, 0이면 uniform | 64 |
| `--seventh-hand-history` | exact key에서 card 관측 순서 보존 | 켬 |
| `--bucket power-memory16` | 5th/6th blueprint abstraction | 사용 |

재현 명령은 다음과 같다.

```powershell
.\cpp_mccfr\stud_mccfr_pbs_resolver.exe `
  --bucket power-memory16 `
  --load-atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --stack-ante 1000 `
  --load cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --seventh-resolve-iterations 10000 `
  --seventh-resolve-prior 100 `
  --seventh-hand-history `
  --seventh-posterior-particles 64 `
  --hands 10000 `
  --iterations 0 `
  --opponent policy-lbr `
  --belief-particles 240 `
  --seed 83501
```

`--seventh-posterior-particles 0`으로 바꾸면 기존 uniform resolver와 비교할 수
있다.

---

## 12. 로그 지표

### 12.1 Resolver 계산량

- `subgames`: 실제로 resolve한 7th root 수
- `traversals`: 수행한 local MCCFR traversal 수
- `node_visits`: traversal 중 방문한 node 총수
- `nodes_created`: 생성한 local exact information-set node 누적 수
- `maximum_local_nodes`: 한 subgame에서 동시에 존재한 local node 최대치
- `blueprint_fallbacks`: 요청한 7th exact key가 local table에 없어 blueprint를
  사용한 횟수

### 12.2 Posterior

- `posterior_builds`: posterior를 만든 7th subgame 수
- `posterior_proposals`: H4 rejection을 포함해 제안한 후보 수
- `posterior_policy_queries`: history likelihood를 위해 blueprint를 조회한 수
- `posterior_policy_misses`: 조회한 정보집합이 blueprint에 없던 수
- `posterior_fallbacks`: 후보 생성 실패로 uniform determinization을 사용한 수
- `average_effective_posterior_particles`: 평균 effective sample size

ESS는 다음과 같다.

\[
N_{eff}=\frac{(\sum_j\hat w_j)^2}{\sum_j\hat w_j^2}
=\frac{1}{\sum_j p_j^2}.
\]

64개 particle의 weight가 모두 같으면 ESS는 64다. 한 particle에 질량이 거의
몰리면 ESS는 1에 가깝다. ESS 약 22는 64개를 저장했지만 posterior 다양성은
균등 표본 약 22개 수준이라는 뜻이다. 이것은 곧바로 나쁜 결과를 의미하지
않는다. history가 실제로 range를 좁혔다면 ESS 감소는 정상이다. 다만 ESS가
지속적으로 1~2라면 particle 수 증가, proposal 개선, resampling이 필요하다.

---

## 13. 실험 결과

### 13.1 계산량 smoke test

동일 모델과 atlas에서 uniform determinization과 posterior-64를 비교했다.
성능 수치는 표본이 작아 이 표에서는 해석하지 않는다.

| resolver | hands | iterations | elapsed | node visits | nodes created | max local nodes |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1,000 | 100 | 9.10s | 580,228 | 155,040 | 1,286 |
| posterior-64 | 1,000 | 100 | 10.30s | 587,446 | 92,260 | 868 |
| uniform | 100 | 10,000 | 10.20s | 12,478,918 | 2,524,487 | 100,208 |
| posterior-64 | 100 | 10,000 | 5.34s | 9,137,840 | 205,982 | 13,521 |

해석은 다음과 같다.

- 100회 탐색에서는 posterior 구축 고정비 때문에 13% 느렸다.
- 10,000회 탐색에서는 약 1.9배 빨랐다.
- 10,000회에서 생성 node는 약 91.8% 감소했다.
- posterior-64의 평균 ESS는 약 22.5였다.

즉, 이 기법의 계산상 이득은 얕은 search가 아니라 **동일 posterior를 수천 번
재사용하는 깊은 local search**에서 나타난다.

### 13.2 10,000-hand LBR-240 평가

위 재현 명령의 결과는 다음과 같다.

| 지표 | 값 |
|---|---:|
| LBR 평균 이득 | 0.20333120 ante/hand |
| 표준오차 | 0.16083519 |
| 95% CI | [-0.11190576, 0.51856816] |
| hands | 10,000 |
| LBR particles | 240 |
| resolver subgames | 3,547 |
| resolver traversals | 35,470,000 |
| resolver node visits | 790,818,374 |
| resolver nodes created | 18,003,421 |
| maximum local nodes | 25,228 |
| posterior proposals | 884,698 |
| posterior policy queries | 522,712 |
| posterior policy misses | 0 |
| posterior fallbacks | 0 |
| 평균 posterior ESS | 21.7934 |
| elapsed | 698.49s |
| 처리량 | 14.32 hands/s |

과거 비교값은 다음과 같다. 서로 다른 run의 CI이므로 엄밀한 paired ablation은
아니며 방향을 보는 참고값이다.

| 설정 | LBR ante/hand | 95% CI |
|---|---:|---:|
| resolver 없음 | 약 1.299 | [1.107, 1.490] |
| uniform resolver, iter 100 / prior 100 | 약 1.223 | [1.021, 1.424] |
| uniform resolver, iter 10,000 | 약 0.934 | [0.274, 1.595] |
| posterior-64, iter 10,000 / prior 100 | 0.203 | [-0.112, 0.519] |

이 결과는 **관측 history에 맞는 range를 만들고 깊게 푸는 조합이 기존
uniform resolver보다 훨씬 유망하다**는 강한 실험 증거다. 그러나 CI가 0을
포함한다고 해서 Nash equilibrium이나 exploitability 0을 증명한 것은 아니다.
LBR은 최적 BR이 아니라 제한된 근사 공격자이며, 다음 절의 fallback 맹점도
있다.

---

## 14. 가장 중요한 평가 한계: blueprint fallback

10,000-hand 결과에서 `blueprint_fallbacks = 3,270,000`이었다. 이 값은 계산
성공을 부정하지 않지만, LBR 수치를 그대로 exploitability로 읽을 수 없게 하는
핵심 경고다.

### 14.1 왜 발생하는가

Online resolver는 실제 플레이 중인 target의 실제 private hand를 고정하고,
상대 hidden range만 바꿔 local tree를 만든다. 반면 policy-LBR은 target의
range를 추론하기 위해 여러 **가상의 target private hand**에서 target policy를
질의한다.

그 가상 target hand의 exact key는 실제 hand로 만든 local table에 존재하지
않을 수 있다. 이때 resolver는 30M blueprint로 fallback한다.

```text
실전 의사결정:
  실제 내 private hand의 local resolver policy 사용

LBR의 range query:
  여러 가상 target private hand 질의
  -> local node 없음
  -> blueprint fallback 빈번
```

따라서 현재 LBR은 resolver가 실제로 취하는 action은 일부 보지만, resolver의
전체 policy mapping을 공격자 range update에 완전히 노출하지 못한다. 이 때문에
0.203은 진짜 online policy의 exploitability lower bound를 과소평가할 가능성이
있다.

### 14.2 보완 방법

엄밀한 평가를 위해서는 다음 중 하나가 필요하다.

1. 같은 public root에서 target의 가능한 private hand마다 local policy를 함께
   풀어 policy oracle을 만든다.
2. full joint PBS 위에서 양쪽 range를 함께 가진 resolver를 사용한다.
3. resolver query를 deterministic하게 cache하고 LBR이 같은 public root의 모든
   target hand policy를 조회할 수 있게 한다.
4. 작은 7th subgame에서는 exact BR을 materialize하여 LBR과 교차 검증한다.

현재 결과는 “알고리즘이 계산량을 크게 줄였고 실제 trajectory 성능이 매우
좋아졌다”는 근거로는 충분하지만, “exploitability가 0.203 이하”라는 증명으로는
불충분하다.

---

## 15. 수렴과 안전성 보장

### 15.1 현재 보장할 수 있는 것

- 고정된 sampled posterior와 유한 7th game에서 local CFR+ 반복을 늘리면 그
  sampled subgame의 regret를 줄이는 방향으로 진행한다.
- exact key가 서로 다른 관측 history와 private information을 합치지 않으므로
  기존 `memory16` bucket보다 local recall이 훨씬 강하다.
- posterior support 밖의 hand를 반복 탐색하지 않아 deep search의 node 수를
  줄인다.
- subgame마다 local table을 비우므로 과거 핸드의 private state가 다음 핸드에
  남지 않는다.

### 15.2 보장할 수 없는 것

- 전체 5th~7th 원 게임의 exploitability 단조 감소
- full-game Nash equilibrium 수렴
- 잘못 지정된 blueprint opponent model 아래 posterior의 정확성
- particle approximation error 0
- unsafe subgame resolving이 trunk 전략을 악화시키지 않는다는 보장
- LBR 0.203이 exact exploitability와 같다는 주장

가장 큰 이론적 결손은 safe resolving이다. 현재 local solve는 trunk에서
상대에게 보장되던 counterfactual value를 constraint로 가져오지 않는다.
따라서 local subgame에서 좋아 보이는 전략이 전체 게임에서는 새로운 약점을
만들 수 있다.

안전한 다음 단계는 opponent counterfactual value를 blueprint에서 계산하고,
CFR-D/DeepStack 계열 gadget game으로 7th solve를 제한하는 것이다.

---

## 16. 기존 방법과의 비교

| 방법 | Range | 7th 정보집합 | Leaf | 주요 한계 |
|---|---|---|---|---|
| 30M blueprint | bucket에 암묵적 | power-memory16 | 없음 | abstraction/optimization 바닥 |
| 기존 7th resolver | uniform hidden hand | exact history | terminal | 잘못된 range에 깊은 탐색 |
| 현재 PBS resolver | history-conditioned particles | exact history | terminal | partial PBS, unsafe resolving |
| Full ReBeL | joint public belief | public state + ranges | value network/terminal | 구현 및 학습 비용 |
| Pluribus식 search | blueprint range | 제한 lookahead abstraction | continuation strategy | domain-specific engineering |

현재 구현의 실용적 위치는 blueprint와 full ReBeL 사이이다. 7th에는 미래
chance가 없어 value network가 필요 없다는 구조적 이점을 최대한 이용한다.

---

## 17. 웹 프리셋

웹의 agent 선택지 이름은 다음과 같다.

```text
PBS 7th (30M)
```

고정 설정은 다음과 같다.

```text
executable: stud_mccfr_pbs_resolver.exe
blueprint: made_call_r1000_k512_epsheur20_memory16_30m.bin
atlas: power512_epsheur20_memory16_v1.bin
bucket: power-memory16
start street: 5
7th iterations: 10,000
7th prior: 100
posterior particles: 64
preserve hand history: true
```

웹에서 시험할 때에는 다음과 같이 설정한다.

1. Mode를 `ev`로 선택한다.
2. Ante를 `1000`으로 둔다.
3. 한 자리는 `human`, 다른 자리는 `PBS 7th (30M)`을 선택한다.
4. 나머지 세 자리는 `empty`로 둔다.
5. Start를 누른다.

이 agent는 현재 heads-up 전용이다. 다인 웹 게임에서는 representative opponent
하나로 축약해야 하므로 PBS 의미가 깨진다. 그래서 웹 controller가 두 명이 아닌
설정을 거절한다.

5th와 6th 반응은 30M blueprint와 동일하다. 차이를 직접 관찰하려면 7th까지
도달한 뒤 같은 public/history 상황에서 bet, call, fold 빈도와 응답 시간을
비교해야 한다. 혼합전략이므로 같은 모양의 hand에서도 action이 달라질 수 있다.

---

## 18. 검증 체크리스트

### 빌드 및 기본 테스트

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_mccfr.cpp `
  -o cpp_mccfr\stud_mccfr_pbs_resolver.exe

.\cpp_mccfr\stud_mccfr_pbs_resolver.exe --self-test
```

기대 출력:

```json
{"self_test":"ok"}
```

### Posterior 확인

- `posterior_builds == subgames`
- `posterior_fallbacks == 0`이 이상적
- `posterior_policy_misses == 0`인지 확인
- ESS가 지나치게 1에 붙지 않는지 확인
- iteration을 늘릴 때 `maximum_local_nodes`와 시간이 감당 가능한지 확인

### 성능 확인

- 같은 deal seed로 uniform과 posterior를 paired 비교
- 64/120/240 posterior particle 민감도 비교
- prior 0/10/100/1000 비교
- 1k/10k/100k resolver iteration curve 측정
- LBR뿐 아니라 exact 7th BR 또는 safe-resolving value로 교차 검증

---

## 18.1 Joint-PBS LBR 검증 경로

웹의 `PBS 7th (30M)`은 실제 자기 hidden hand를 고정해 매 의사결정마다
subgame을 다시 푼다. 반면 `PolicyLBR`은 target의 여러 가상 hidden hand에
대한 정책을 동시에 질의한다. 웹 resolver의 local table을 그대로 넘기면 가상
hand 질의가 blueprint로 빠져, 배치된 정책과 LBR이 본 정책이 달라진다.

`stud_rebel_seventh_lbr.exe`는 실제 hidden hand를 제거한 동일한 7th public
root에서 양쪽 private range를 함께 표본화하고 local CFR를 한 번 수행한다.
실제 행동과 LBR의 모든 counterfactual hand 질의가 같은 local table을
조회한다. local node가 없는 경우에도 양쪽 모두 동일한 30M blueprint를
사용한다.

따라서 이 경로의 `blueprint_fallbacks`는 평가기 누락이 아니라 배치된 복합
정책에 명시된 fallback이다. `counterfactual_policy_consistent=true`가 이
조건을, `local_policy_coverage`가 local CFR가 직접 답한 질의 비율을 나타낸다.

LBR private range도 H4 공개 결과를 반영한다. 후보 hidden/discard 조합이 현재
결정론적 `discard_reveal`로 관측된 공개 카드를 만들 수 없으면 제거하고,
가능한 초기 카드 순열 수를 prior weight로 사용한다.

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_seventh.cpp `
  -o cpp_mccfr\stud_rebel_seventh_lbr.exe

.\cpp_mccfr\stud_rebel_seventh_lbr.exe `
  --bucket power-memory16 `
  --model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --hands 10000 `
  --lbr-particles 240 `
  --rebel-particles 240 `
  --rebel-iterations 100 `
  --prior 100 `
  --ante 1000 `
  --stack-ante 1000 `
  --power-cache-max 250000 `
  --progress-seconds 30 `
  --seed 71241
```

이 평가는 일관된 Local Best Response지만 exact BR은 아니다. 또한 웹의
hand-conditioned resolver와 동일한 정책이 아니라, LBR 검증이 가능한
joint-public-root resolver라는 별도 정책을 측정한다.

2026-08-23의 1,000-hand / LBR-240 screening 결과는 다음과 같다. 같은 deal
seed를 사용했지만 정책에 따라 이후 trajectory가 달라지므로 차이의 paired CI는
아니다.

| 정책 | LBR ante/hand | SE | 95% CI | local coverage |
|---|---:|---:|---:|---:|
| 30M blueprint | 1.273 | 0.239 | [0.804, 1.741] | - |
| joint-PBS 7th, iter 100 | 1.126 | 0.307 | [0.523, 1.728] | 80.3% |

방향은 개선이지만 1,000 hand에서는 유의하지 않다. 본 판정은 위 명령의
10,000-hand 결과로 한다.

## 19. 권장 후속 작업

우선순위는 다음과 같다.

1. **Resolver-aware LBR**: 가상 target private hand에도 local policy를 제공해
   `blueprint_fallbacks`를 제거한다.
2. **Paired ablation**: 같은 deal, 같은 LBR particle seed로 uniform과 posterior를
   비교한다.
3. **Safe resolving**: trunk opponent CFV를 보존하는 gadget을 추가한다.
4. **H4 likelihood 일반화**: deterministic filter를 learned/stochastic H4 policy
   likelihood로 바꾼다.
5. **Particle proposal 개선**: rejection sampling 대신 H4-compatible hand를 직접
   열거하거나 importance proposal을 사용한다.
6. **Full PBS 확장**: 양쪽 private range를 함께 표현하고 5th/6th에도 recursive
   resolving을 적용한다.
7. **Value network**: 5th/6th search의 leaf에서 V6/V7 counterfactual value를
   예측한다.

가장 먼저 해야 할 일은 1번이다. 현재 0.203이라는 숫자의 신뢰도를 결정하는
직접적인 병목이기 때문이다. 그 전까지는 이 결과를 **유망한 online-search
성능**으로 기록하되, 해결된 게임이나 정확한 exploitability로 표현하지 않는다.

---

## 20. 결론

기존 7th resolver의 병목은 탐색 횟수 자체만이 아니었다. 과거 행동과 무관한
uniform hidden-hand range를 풀고 있었기 때문에, 더 깊은 탐색이 필요 없는
사적 상태까지 대량으로 전개했다.

이번 구조는 H4 관측과 exact betting history를 이용해 상대 hand posterior를
만들고, 그 posterior를 10,000회의 local MCCFR에서 재사용한다. 그 결과 깊은
탐색에서 local node가 크게 줄었고, 10,000-hand LBR-240 실험에서 평균 공격자
이득이 0.203 ante/hand까지 내려갔다.

따라서 이번 결과는 단순한 iteration 증가보다 **올바른 range 위에서 탐색하는
것이 먼저**라는 점을 실증한다. 동시에 fallback과 safe-resolving 부재 때문에
아직 최종 결론은 아니다. 현재 구현은 버릴 실패작이 아니라, full PBS와 안전한
online resolving으로 넘어가기 위한 작동하는 중간 단계다.
