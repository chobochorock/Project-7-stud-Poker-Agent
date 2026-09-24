# 7-Stud reward-audited local bucket split 적용 명세

## 문서 상태

이 문서는 `Toy-Card-Game-Agent`에서 가장 좋은 결과를 보인 online bucket
split을 현재 C++ 7-Stud MCCFR에 옮기기 위한 구현 명세다. 아직 7-Stud에서
이 방식의 우월성이 확인된 것은 아니다.

용어는 다음처럼 고정한다.

- `base cluster`: 학습 전에 만든 고정 power centroid와 그 Voronoi 영역
- `route node`: 한 base cluster 내부를 다시 나누는 이진 트리의 내부 노드
- `leaf bucket`: 실제 `RegretNode`가 연결되는 route tree의 잎
- `audit`: 현재 정보 상태에서 각 합법 행동의 terminal return을 별도로 추정하는 탐색
- `split`: 한 leaf bucket을 두 자식으로 영구 분리하는 연산

## 결론

첫 구현은 아래 조합을 사용한다.

1. 5th street 이후의 2인용 external-sampling CFR+
2. 낮은 `K`로 미리 고정한 hard power atlas
3. 각 leaf 방문의 10%를 무작위 reward audit로 선택
4. audit마다 합법 행동별 terminal rollout 12회
5. feature 2-means와 reward-label SSE를 함께 통과할 때만 split
6. base cluster 밖의 배정에는 영향을 주지 않는 local binary tree
7. merge 없는 append-only 구조
8. 부모의 누적 regret과 average-strategy mass를 자식 비율로 보존
9. 초기화, CFR traversal, audit를 모두 합친 동일 node budget으로 비교
10. exact exploitability 대신 paired LBR lower bound로 평가

비용이 너무 크면 10% 방식을 폐기하지 않고, 새 leaf에는 10%를 유지하고
안정된 leaf만 geometric audit와 1% background audit로 전환한다.

## 확인된 실험 결과

toy 실험의 초기화는 uniform-random trajectory 2,000 hands였고, 각
`(player, round, legal action set)`에 처음에는 bucket 하나만 두었다. 아래
수치에서 낮을수록 좋다.

| Game, budget | Exact MCCFR+ | Static coarse | Adaptive local split | Leaves |
|---|---:|---:|---:|---:|
| Stud-Leduc, 300k, geometric 3 seeds | 0.345 | 2.326 | 1.983 | 16.7 |
| Limit high-card, 300k, geometric 3 seeds | 0.044 | 0.850 | 0.633 | 12.3 |
| Stud-Leduc, 100k, random 10%, seed 41 | - | 2.215 | 1.772 | 48 |
| Limit high-card, 100k, random 10%, seed 41 | - | 0.836 | 0.166 | 41 |

10% random audit가 현재 관측된 가장 좋은 adaptive 결과였지만, 이 비교는
seed 41 하나뿐이다. audit가 전체 learning-node budget의 Stud-Leduc 65.1%,
high-card 58.3%를 사용했다. 따라서 이는 강한 후보이지, 일반적인 우월성의
증명은 아니다.

현재 7-Stud의 `curve_adaptive_10m.json`은 LBR 4.681 ante, hard 방식의
`curve_hard_10m.json`은 3.031 ante였다. 신뢰구간은 각각
`[2.859, 6.504]`, `[1.862, 4.200]`이다. 이 adaptive 방식은 global centroid
append와 soft policy 초기화를 사용하므로 여기서 제안하는 local tree와
다른 알고리즘이다. 기존 결과를 새 방식의 반례나 근거로 재사용하면 안 된다.

## toy에서 사용한 정확한 split 규칙

재현 프로필은 다음 값을 그대로 사용한다.

```text
behavior initialization hands = 2,000
audit schedule               = Bernoulli(p=0.10) per leaf visit
rollouts per legal action    = 12
utility normalization        = known maximum utility
one-sided confidence z       = 1.645
strong-gap threshold         = 0.04
audit FIFO capacity          = 32
minimum strong violations    = 2
2-means iterations           = at most 20
minimum reward SSE gain      = 0.20
maximum split depth          = 8
maximum leaves               = 256
merge                        = disabled
```

audit 표본 `r`에서 행동 `a`의 정규화된 terminal return을 `U_r(a)`, 현재
bucket strategy를 `sigma(a)`라 한다.

```text
Q_hat(a) = mean_r U_r(a)
a_star   = argmax_a Q_hat(a)
d_r      = U_r(a_star) - sum_a sigma(a) U_r(a)
gap      = mean_r d_r
LCB90    = gap - 1.645 * std(d_r) / sqrt(12)
```

`LCB90 > 0.04`인 audit를 strong violation으로 센다. 최근 audit 32개 전체의
feature에 farthest-pair 초기화 2-means를 수행한다. 그 분할로 reward vector
`Q_hat`의 SSE를 계산한다.

```text
SSE(B) = sum_{i in B} ||Q_i - mean(Q_B)||^2
gain   = 1 - (SSE(child_0) + SSE(child_1)) / SSE(parent)
```

다음 조건을 모두 만족할 때만 split한다.

- strong violation이 2개 이상이다.
- 두 feature centroid가 서로 다르고 두 자식이 모두 비어 있지 않다.
- `gain >= 0.20`이다.
- 두 자식의 평균 reward에서 선택되는 최적 행동이 서로 다르다.
- depth와 leaf 수 제한을 넘지 않는다.

toy 구현은 `a_star` 선택과 LCB 계산에 같은 12개 표본을 사용한다. 따라서
winner's-curse 편향이 있을 수 있다. 7-Stud의 첫 parity 실험에서는 그대로
재현하되, 다음 실험에서는 6개 discovery와 6개 validation 표본으로 나누어
이 편향의 영향을 A/B 비교한다.

## 7-Stud 상태와 feature

첫 적용 범위는 현재 C++ 실험과 동일하게 heads-up, 5th-to-7th street다.
H4는 기존 heuristic으로 유지한다.

split feature는 기존 `PowerVector` 18차원을 그대로 사용한다.

- 최종 hand category 확률 9개에 `sqrt(p)` 적용
- expected primary rank
- 자신의 공개 패와 상대 공개 패의 pressure feature

`power_vector(state, viewer, sample_limit)`에는 viewer가 관측할 수 없는
상대 hole card를 절대 추가하지 않는다. pot odds, stack ratio, betting
history와 legal mask는 기존 `InfoKey`에 계속 남긴다. 즉 최종 정책 key는
다음 곱 구조를 유지한다.

```text
policy infoset = route-tree leaf
               x existing betting/stack/history abstraction
               x exact legal-action mask
```

audit 표본은 최소한 `(street, leaf, legal_mask)`별로 분리한다. 서로 다른
legal mask의 reward vector는 차원과 의미가 다르므로 같은 SSE에 넣지 않는다.
pot/history에 의한 label 혼합이 크게 관측되면 `InfoKey`에서
`power_cluster`만 제거한 `AuditContextKey`까지 층화한다. 첫 버전에서 이
세부 context를 cluster feature에 다시 넣지는 않는다.

## 초기 atlas

초기 centroid는 reward를 보지 않는 off-policy trajectory만으로 만든다.
첫 비교는 다음 두 초기화를 같은 random seed와 sample set으로 만든다.

- `static-low-k`: split하지 않는 대조군
- `local-split-low-k`: 같은 atlas에서 시작하고 online split 허용

7-Stud는 toy보다 18차원 feature와 훨씬 넓은 chance space를 가지므로
street당 bucket 하나로 시작하지 않는다. 첫 구현 기본값은 `8/8/8`, 비교점은
`16/16/16`으로 둔다. 이 값은 아직 실험 결과가 아니라 의도적으로 작은 초기
K에 대한 시작점이다. 현재 강한 frozen hard atlas도 별도 기준선으로 유지한다.

초기 trajectory의 행동은 uniform legal random을 기본으로 한다. 카드와
chance는 실제 규칙대로 표본화한다. 이후 coverage가 부족할 때만
epsilon-self-play나 heuristic/LBR 혼합을 별도 실험군으로 추가한다.

## local route tree

현재 `PowerAtlas::append`는 새 centroid를 같은 street의 모든 centroid와
다시 경쟁시킨다. 이 방식에서는 다른 cluster의 기존 상태까지 새 centroid로
이동할 수 있다. 새 모드는 이를 사용하지 않는다.

배정은 두 단계다.

1. 고정된 초기 centroid끼리만 비교하여 base cluster를 고른다.
2. 그 base cluster의 route tree 안에서만 두 child centroid를 비교하며 leaf까지 내려간다.

```text
street 5 fixed roots
  base 0
    split(feature) -> leaf 0 or leaf 513
      leaf 513 split(feature) -> leaf 513 or leaf 514
  base 1
  ...
```

한 leaf가 split될 때 기존 leaf ID는 child 0이 재사용하고 child 1만 새 policy
ID를 받는다. 내부 route node는 policy를 갖지 않는다. base root를 고를 때
쓰는 초기 centroid와 부모 영역은 영구 고정한다. 자식도 이후 같은 방식으로
다시 split될 수 있다.

권장 자료 구조는 다음 정도면 충분하다.

```cpp
struct RouteNode {
    uint32_t child[2];
    PowerVector child_center[2];
    uint16_t leaf_policy_id;
    uint16_t depth;
    bool is_leaf;
};

struct AuditSample {
    PowerVector feature;
    std::array<double, kActionCount> q;
    uint8_t legal_mask;
    uint8_t best_action;
    double gap;
    double lower_bound;
};
```

초기 root centroid와 route node를 분리해서 저장해야 root 영역이 움직이지
않는다. assignment cache는 split 직후 전부 비워도 의미는 local하게 유지된다.
성능 문제가 확인된 뒤에만 parent generation을 붙인 부분 무효화를 고려한다.

## 7-Stud audit

표준 CFR traversal 도중 leaf visit의 10%를 audit queue에 넣는다. recursion
중간에는 atlas나 regret table을 변경하지 않는다. 두 traverser의 update가
끝난 안전한 시점에 audit를 실행하고 split 후보를 처리한다.

현재 정보 상태에서 첫 행동만 각 합법 행동으로 강제한다. 이후의 플레이어
행동은 모두 현재 bucket strategy를 사용하고, 다음 상태도 계속 route tree를
통해 조회한다. audit rollout은 regret이나 `strategy_sum`을 업데이트하지 않는다.

terminal utility는 audit 시작 시점의 effective stack으로 나누어 대략
`[-1, 1]` 범위로 정규화한다.

```text
u = terminal_net_search(terminal, viewer)
    / max(ante, effective_stack_at_audit_root)
```

### 비공개 카드 처리

C++ `State` 안의 실제 상대 hole card를 feature로 사용해서는 안 된다. 또한
같은 실제 hole card를 고정한 12개 rollout을 서로 독립적인 정보상태 표본처럼
취급하면 LCB가 과도하게 좁아진다.

권장 estimator는 공개 카드와 betting history, 현재 self-play policy에 조건부인
상대 hand belief에서 outer particle을 다시 뽑고, 각 particle에서 미래 chance와
continuation을 표본화하는 방식이다. 기존 Policy-LBR particle machinery를 재사용할
수 있지만, likelihood는 고정 LBR opponent가 아니라 현재 MCCFR policy로 계산해야
한다.

조건부 sampler를 구현하기 전 임시 버전에서는 실제 hidden state 하나를 하나의
outer 표본으로만 센다. 서로 독립적으로 도달한 여러 audit event가 쌓이기 전에는
split하지 않고, 미래 chance만 바꾼 12개 rollout으로 hidden-card 불확실성까지
측정했다고 간주하지 않는다.

## regret 보존

split에 사용된 audit feature 중 child `j`로 간 표본의 비율을 `q_j`라 한다.

```text
q_0 + q_1 = 1
q_j = count(child_j) / count(parent)
```

parent power cluster를 사용하는 모든 기존 `InfoKey C`에 대해 다음을 적용한다.

```text
R(C, child_0, a)     = q_0 * R_old(C, parent, a)
R(C, child_1, a)     = q_1 * R_old(C, parent, a)
R_raw(C, child_j, a) = q_j * R_raw_old(C, parent, a)
S(C, child_j, a)     = q_j * S_old(C, parent, a)
```

여기서 `R`은 CFR+ regret, `R_raw`는 raw regret, `S`는 `strategy_sum`이다.
`touches`는 정수이므로 child 1에 `round(q_1 * old)`를 주고 child 0에 나머지를
준다. 이 방식에는 다음 성질이 있다.

- 각 배열의 두 자식 합은 split 전 부모 배열과 같다.
- 양의 상수를 곱하므로 두 자식의 즉시 regret-matching policy는 부모와 같다.
- average policy도 split 직후에는 부모와 같다.
- child를 부모 전체 값으로 복사하는 mass 이중 계산이 없다.
- 특정 행동 one-hot으로 child를 초기화해서 생기는 즉시 policy jump가 없다.

자식의 audit FIFO, visit count, confidence evidence는 0에서 시작한다. 부모의
과거 reward evidence는 경계 제안에만 사용했으므로 자식의 split 증거로 상속하지
않는다.

## 현재 C++ 코드의 변경 경계

첫 구현은 `cpp_mccfr/stud_mccfr.cpp` 안에서 다음 경계만 바꾼다.

| 현재 위치 | 변경 |
|---|---|
| `PowerAtlas::assign` | 초기 root 선택 뒤 local tree routing 추가 |
| `PowerAtlas::append` | 새 모드에서는 사용 금지, `split_leaf` 추가 |
| `PowerAtlas::save/load` | route tree와 leaf ID를 저장하는 새 format 추가 |
| `RegretNode` | 구조는 유지하고 split 시 mass partition 함수만 추가 |
| `MCCFR::soft_traverse` | hard tree 전용 traversal에서 audit queue 기록 |
| `MCCFR::adapt_soft_clusters` | 기존 실험용으로 유지, 새 알고리즘과 공유하지 않음 |
| root training loop | traversal 밖에서 audit와 최대 1개 split 처리 |

첫 버전은 soft mixture를 사용하지 않는다. `power-tree`는 하나의 hard leaf만
반환한다. 따라서 현재 soft temperature calibration, residual point와
`kChildMassFraction=0.1`은 새 모드에 적용하지 않는다.

구현 함수의 최소 인터페이스는 다음과 같다.

```cpp
uint16_t PowerAtlas::assign_tree(const State&, int viewer) const;
SplitResult PowerAtlas::split_leaf(
    int street,
    uint16_t parent_leaf,
    const PowerVector& child0,
    const PowerVector& child1);
void MCCFR::queue_split_audit(const State&, int viewer, uint16_t leaf);
void MCCFR::process_split_audits();
void MCCFR::partition_regret_nodes(
    int street, uint16_t parent_leaf, uint16_t child_leaf, double child_mass);
```

## checkpoint 형식

route tree와 regret table 중 하나만 저장되면 `power_cluster`의 의미가 깨진다.
따라서 둘은 같은 checkpoint generation으로 저장한다.

- 기존 `POWERAT1/2/3`은 읽기 전용 호환으로 유지한다.
- tree atlas는 새 magic, 예를 들어 `POWERAT4`, 를 사용한다.
- solver checkpoint에도 bucket mode와 atlas generation ID를 기록한다.
- load 시 generation ID, street별 base root 수, 최대 leaf ID를 검증한다.
- 구형 atlas를 tree mode로 읽으면 각 base cluster를 depth 0 leaf로 변환한다.

한 checkpoint에서 solver와 atlas를 모두 임시 파일에 쓴 뒤 마지막에 manifest를
교체한다. 중간 저장 실패 시 이전 generation을 계속 불러올 수 있어야 한다.

## 제안 CLI

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power-tree `
  --load-atlas cpp_mccfr\power8_v1.bin `
  --start-street 5 --algorithm mccfr-plus `
  --split-audit random --split-audit-probability 0.10 `
  --split-rollouts 12 --split-gap-threshold 0.04 `
  --split-confidence-z 1.645 --split-min-strong 2 `
  --split-reward-gain 0.20 --split-reservoir 32 `
  --split-max-depth 8 --split-max-leaves 512 `
  --root-node-budget 1000000 --seed 51001 `
  --save cpp_mccfr\local_split_seed51001_1m.bin `
  --save-atlas cpp_mccfr\local_split_seed51001_1m.atlas
```

이는 구현 후 사용할 목표 인터페이스이며, 현재 실행 가능한 명령은 아니다.

비용 절감형 후속 프로필은 다음과 같다.

```text
new or recently split leaf: p(audit) = 0.10
stable leaf: visits 10, 100, 1000, ... 에서 geometric audit
all stable leaves: p(audit) = 0.01 background audit
child after split: audit history reset, hot phase restart
```

## 공정한 평가

7-Stud에서는 exact exploitability를 계산할 수 없으므로 기존 paired Policy-LBR을
주 지표로 사용한다. 낮을수록 좋다. 각 방법은 같은 root/chance seed와 좌석쌍을
사용한다.

비교군은 최소 네 개다.

1. 같은 low-K atlas를 고정한 `static-low-k`
2. 현재 가장 강한 frozen hard atlas
3. 새 `local-split-random10`
4. 새 `local-split-hybrid`

현재 global-append adaptive 방식은 회귀 비교용 다섯 번째 실험군으로만 둔다.

예산에는 다음을 모두 포함한다.

```text
total learning nodes
  = initial trajectory nodes
  + CFR traversal nodes
  + every audit rollout node
```

같은 CFR iteration 수만 맞추면 random10 방식에 더 많은 계산을 허용하게 되므로
잘못된 비교가 된다. node budget 외에 wall time과 peak memory도 함께 기록한다.

각 checkpoint에서 다음을 저장한다.

- paired LBR lower bound와 95% CI, ante/hand
- heuristic 상대 paired EV
- total/train/init/audit node visits
- audit 비중과 audit당 평균 node 수
- street별 base roots, leaves, split 수, 최대 depth
- split 후보 수와 조건별 rejection 수
- leaf 방문 분포와 single-touch 비율
- split 직전과 직후 부모 표본의 policy KL
- parent 밖 assignment가 바뀐 상태 수

첫 판정 budget은 `1m`, `10m`, `30m total learning nodes`, seed는 최소 3개로
한다. toy의 random10 결과가 한 seed뿐이므로 7-Stud 단일 seed 결과로 기본
방법을 교체하지 않는다.

## 필수 테스트

구현 완료 조건은 다음과 같다.

- base cluster A를 split해도 base cluster B의 모든 test feature 배정이 같다.
- child를 다시 split해도 sibling subtree 배정이 같다.
- 동일 observation은 split 전후 checkpoint save/load에서 같은 leaf로 간다.
- 모든 기존 node에서 두 자식 regret, raw regret, strategy sum의 합이 부모 값이다.
- split 직후 두 자식의 current/average policy가 부모 policy와 같다.
- feature와 audit key에 상대 비공개 카드가 들어가지 않는다.
- 고정 seed에서 audit 선택, rollout과 split log가 재현된다.
- audit traversal이 regret와 strategy sum을 변경하지 않는다.
- initial, CFR, audit node가 total budget에 정확히 한 번씩 합산된다.
- max depth, max leaves와 기존 `uint16_t power_cluster` 범위를 넘지 않는다.
- 기존 non-tree MCCFR self-test와 checkpoint load가 그대로 통과한다.

## 구현 순서

1. `PowerAtlas`에 local route tree와 save/load를 넣고 수동 split routing test만 통과시킨다.
2. `RegretNode` mass partition과 policy-invariance test를 추가한다.
3. audit를 기록만 하고 split하지 않는 shadow mode를 실행한다.
4. 10% random split을 켜고 `static-low-k`와 1m node smoke test를 한다.
5. 3 seeds, 1m/10m/30m paired LBR 비교를 실행한다.
6. random10의 개선이 재현되면 hybrid schedule을 추가한다.

첫 3단계가 끝나기 전에는 기존 `PowerAtlas::append` 실험을 삭제하거나 현재 hard
baseline을 교체하지 않는다. 새 방식의 핵심 검증 대상은 leaf 수 자체가 아니라,
같은 총 계산량에서 LBR이 낮아지는지와 split 순간 기존 정책을 보존하는지다.

## 참고 파일

- toy 구현: `../../Toy-Card-Game-Agent/online_bucket_split_experiment.py`
- toy 결과: `../../Toy-Card-Game-Agent/ONLINE_BUCKET_SPLIT_RESULTS.md`
- 7-Stud 구현: `stud_mccfr.cpp`
- 기존 bucket 설명: `BUCKET_GROWTH.md`
- 기존 C++ 실행법: `README.md`
- 기존 7-Stud adaptive 결과: `curve_adaptive_10m.json`
- 기존 hard 결과: `curve_hard_10m.json`
