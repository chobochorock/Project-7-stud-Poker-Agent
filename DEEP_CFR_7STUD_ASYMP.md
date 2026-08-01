# 7-Stud Deep CFR와 7th-street AsymP 설계

## 1. 목표와 현재 경계

첫 대상은 현재 C++ 규칙과 같은 **2인 zero-sum chip-EV 게임**이다.

```text
H4: 기존 정책을 우선 동결
5th~6th: Deep CFR trunk
7th: Deep CFR 기본 정책 또는 안전한 AsymP subgame resolving
평가: heuristic 대전 + policy-LBR 하한
```

OpenSpiel의 PokerKit Seven Card Stud는 정보상태 tensor를 제공하지 않고,
현재 프로젝트의 H4 및 betting-rules-v3와도 다르다. 따라서 이를 그대로
학습시키지 않는다. C++ 게임을 규칙의 단일 기준으로 유지하고, 신경망이
사용할 정보상태 tensor와 표본 경계를 추가하는 것이 구현 경로다.

## 2. Deep CFR

플레이어별 advantage network와 공용 average-policy network를 둔다.

\[
\widehat r_t(I,a)=v_i^{\sigma^t}(I,a)-v_i^{\sigma^t}(I),
\]

\[
\sigma_i^t(a\mid I)=
\frac{[R_{\theta_i}(I,a)]_+}
{\sum_b[R_{\theta_i}(I,b)]_+}.
\]

양수 advantage가 모두 0이면 legal action에 균등분포를 사용한다.
External-sampling traversal은 traverser의 모든 legal action을 평가하고,
상대와 chance 행동은 현재 정책에서 표본화한다.

두 reservoir에 다음 표본을 보관한다.

```text
advantage: (information tensor, iteration, sampled counterfactual regret)
strategy:  (information tensor, iteration, sampled behavior strategy)
```

iteration weight를 포함한 회귀로 advantage를 다시 맞추고, 전체 반복이
끝나면 strategy network를 학습한다. 최종 행동은 마지막 advantage
network가 아니라 average-policy network에서 뽑는다.

### 2.1 7-Stud 정보상태 tensor v1

상대 hidden card처럼 플레이어가 모르는 정보는 절대로 넣지 않는다.

| 블록 | 표현 |
|---|---|
| seat/street | one-hot |
| 내 hidden cards | rank/suit one-hot + mask, 순서 불변 정렬 |
| 내 public cards | deal 순서의 rank/suit one-hot + mask |
| 상대 public cards | deal 순서의 rank/suit one-hot + mask |
| H4 기억 | discard/reveal 선택과 공개 결과 |
| betting history | street, 상대/자기, action을 순서대로 one-hot |
| 금액 | `log1p(pot/ante)`, `log1p(stack/ante)`, call/pot |
| legal actions | 7-action mask |
| 선택적 hand feature | 현재 족보와 future hand-power probability |

Betting history는 카운트로 뭉개지 않는다. 같은 카드라도 과거에 자신이
취한 행동이 다르면 다른 tensor가 되어 perfect recall을 유지해야 한다.
Hand-power feature는 raw card 블록을 대체하지 않고 보조 입력으로만 둔다.

현재 IPC prototype의 tensor는 총 1,832차원이다.

```text
seat                         2
street                       3
own hidden 3 slots       3 x 53
own public 4 slots       4 x 53
opponent public 4 slots  4 x 53
own discarded 1 slot     1 x 53
history 24 slots        24 x 49
chip/pressure scalars         7
legal-action mask             8
--------------------------------
total                       1832
```

각 card slot은 empty token과 52장 카드의 one-hot이다. History token은
empty 또는 `(street, relative actor, action)`의 one-hot이다. 상대 hidden
card를 바꾸어도 tensor가 같고, 관측 가능한 history가 바뀌면 tensor가
달라지는 self-test를 둔다.

### 2.2 7th-only trainer 상태

`deep_cfr_traverse.cpp`가 H4와 5th/6th를 frozen heuristic으로 진행한 뒤,
7th에 도달한 실제 state에서 external-sampling traversal을 수행한다. 현재
P0/P1 advantage network가 opponent action을 표본화하고, traverser의 모든
legal action은 C++ tree에서 정확히 평가한다. `train_deep_cfr_7th.py`는 두
advantage reservoir와 average-strategy reservoir를 갱신하고 매 iteration
checkpoint를 저장한다.

`3 iterations x 100 traversals/player`, `128x128` 파일럿은 21.76초에
완료되었다. advantage 표본은 P0 946개, P1 3,212개, strategy 표본은
6,675개였다. 10,000-hand heuristic 평가는 `-2.7596 ante/hand`, 95% CI
`[-3.0215, -2.4978]`였다. 이 결과는 학습량이 너무 작아 성능 판정에는
사용하지 않고, end-to-end 경로의 동작 확인값으로만 보존한다.

## 3. C++와 신경망의 결합

노드마다 Python 프로세스를 호출하면 느리므로 사용하지 않는다. 현재
prototype은 localhost TCP로 다음 두 단계를 구현한다.

1. C++ traverser가 같은 배치에 속한 정보상태 tensor와 legal mask를 모은다.
2. PyTorch/CUDA가 배치 단위로 advantage를 계산하고 결과를 돌려준다.

Python 서버는 임의 MLP 또는 TorchScript 모델을 로드할 수 있다. C++ probe는
실제 5th-street root들을 tensorize하여 배치로 보내고, 8개 advantage를
돌려받아 유한값인지 검사한다.

CPU `256x256` MLP, 총 4,096개 실제 상태에서 측정한 결과:

| Batch | states/second |
|---:|---:|
| 1 | 5,590 |
| 16 | 38,796 |
| 64 | **57,995** |
| 256 | 56,415 |
| 1,024 | 55,770 |

Batch 64부터 IPC 비용이 사실상 포화했다. 따라서 첫 online trainer는
`64~256` pending states를 모아 inference하는 구조로 시작한다.

이 probe만으로는 online Deep CFR가 아니다. 실제 학습에는 현재 advantage
network가 traversal 정책을 결정하고, 반환값으로 regret sample을 만들어야
한다. 다음 구현은 C++ traversal의 pending-node queue를 이 IPC client에
연결하는 것이다.
기존 MCCFR table을 단순 모방하는 것은 warm start이지 Deep CFR가 아니다.

체크포인트에는 다음을 함께 저장한다.

```text
tensor schema version
betting rules version
action ordering and legal-mask definition
policy network
two advantage networks
optimizer state
reservoir counters and RNG seed
```

## 4. 7th-street AsymP

7th에는 더 이상 미래 카드 chance가 없으므로 현재 public state 이후의
betting tree는 작다. 따라서 7th만 exact-history subgame으로 분리하여
bilinear saddle-point solver를 적용할 수 있다.

단, public state만 고정해 바로 다시 풀면 안전하지 않다. Deep CFR trunk가
해당 public state에 도달시킨 다음 자료가 필요하다.

- 양쪽 private-hand posterior range
- 각 hand의 trunk reach probability
- 상대가 trunk에서 보장받던 counterfactual value
- 정확한 7th betting history와 legal actions

두 플레이어의 sequence-form realization plan을 `x`, `y`라 하면 7th
subgame은 다음 꼴이다.

\[
\min_{x\in X}\max_{y\in Y} x^\top A y.
\]

`A`를 dense 행렬로 저장할 필요는 없다. 현재 7th tree를 순회하여 `Ay`와
`A^T x`를 계산하는 operator로 구현할 수 있다. Asymmetric perturbation은
이 exact gradient operator와 sequence-form 제약 위에서 수행한다.

### 4.0 먼저 만들 7th oracle benchmark

Deep CFR trunk에 바로 연결하기 전에, 하나의 고정된 7th public root와
private-hand posterior를 독립된 2인 zero-sum 게임으로 만든다. 이 게임은
미래 카드 chance가 없지만 root의 hidden-hand chance는 남아 있으므로,
호환되는 private hand와 그 확률을 명시적으로 열거해야 한다.

동일한 root game에 세 solver를 적용한다.

1. exact-history tabular CFR+를 충분히 돌려 기준 전략과 NashConv를 얻는다.
2. 같은 정보집합과 payoff operator에 AsymP를 적용해 NashConv/시간 곡선을
   CFR+와 비교한다.
3. 7th-only Deep CFR를 같은 게임에 적용해 exact 기준선과의 오차를 잰다.

이 단계는 safe resolving 실험이 아니다. 고정 root 내부에서 AsymP와 Deep
CFR 자체가 맞는지를 검증하는 정답지다. 여기서 AsymP가 CFR+보다 유리하지
않으면 trunk 연결과 safe-resolving 구현은 보류한다.

첫 oracle은 각 플레이어의 실제 카드 3종과 한 번의
`check/bet/call/fold`를 완전 열거한 `64 x 64` normal-form 게임이다. LP로
계산한 NashConv는 수치 오차 수준인 `4.44e-16`이었다. 동일한 strategy
update 예산 결과는 다음과 같다.

| updates | CFR+ NashConv | 최선 AsymP NashConv | AsymP `mu` |
|---:|---:|---:|---:|
| 100,000 | 0.000319 | 0.031085 | 0.5 |
| 1,000,000 | 0.0000953 | 0.002211 | 0.25 |

1M에서 격차는 줄었지만 AsymP가 약 23배 높았고, 벽시계 시간도 CFR+
`8.92초` 대비 AsymP `14.41초`였다. 따라서 이 root에서는 AsymP가 수렴하지
않는다고 결론내릴 수는 없지만, CFR+보다 빠르다는 근거도 없다. full v3
tree나 safe resolver로 확장하기 전에 더 복잡한 7th root에서도 이 비교를
반복해야 한다.

재현:

```powershell
python -B seventh_oracle_asymp.py `
  --updates 1000000 --mu 0.25 `
  --out cpp_mccfr\seventh_oracle_1m_mu0p25.json
```

### 4.1 안전 조건

다음 조건을 모두 만족해야 이론을 기대할 수 있다.

1. 2인 zero-sum이다.
2. 7th 정보집합이 perfect recall이다.
3. hand range와 gradient가 같은 trunk 정책에서 계산된다.
4. opponent counterfactual value를 보존하는 resolving gadget을 사용한다.

현재 `BucketAsymP`는 bucket behavior policy에 projected gradient를 적용한
실험 구현이다. 이것은 exact sequence-form safe resolver가 아니며, 그대로
Deep CFR 뒤에 붙였다고 전체 exploitability가 감소한다고 보장할 수 없다.

초기에는 AsymP를 online 행동에 사용하지 않고 다음 순서로 검증한다.

```text
frozen Deep CFR trunk
-> 반복 방문되는 7th public root 수집
-> exact-history AsymP resolve
-> resolved value가 trunk CFV 제약을 지키는지 검사
-> LBR와 duplicate match에서 개선될 때만 online 적용
```

## 5. Stud-Leduc 용량 실험

Stud-Leduc에는 72차원 perfect-recall tensor를 추가했다. 모든 실험은
20 iterations, player당 iteration당 200 traversals, seed 외 동일 조건이다.

| Policy/advantage MLP | Seed별 exploitability | 평균 | 평균 학습 시간 |
|---|---|---:|---:|
| `64x64 / 64x64` | 0.5285, 0.5736, 0.7486 | 0.6169 | 42.3초 |
| `256x256 / 256x256` | 0.4492, 0.5253, 0.6299 | **0.5348** | 50.8초 |

폭을 4배로 늘리자 평균 exploitability가 약 13.3% 감소했고 학습 시간은
약 20% 증가했다. 하지만 seed 분산이 커서 폭만으로 안정적인 개선이
보장되지는 않았다.

`256x256`, 50 iterations, 500 traversals, advantage 200 steps, policy
1,000 steps로 늘린 결과는 다음과 같다.

```text
exploitability: 0.277313
training:       273.09 seconds
strategy buffer: 250,000 (capacity 도달)
```

OpenSpiel의 strategy buffer는 FIFO가 아니라 reservoir sampling이다. 따라서
capacity 도달은 오래된 표본이 순서대로 밀려난다는 뜻이 아니라, 전체
stream에서 균등 표본을 유지하되 250,000개만 보존한다는 뜻이다. 용량 확대는
표본 분산을 줄일 수 있지만 구조 검증의 선행 조건은 아니다. `100 x 1000`
확장 실험은 7th oracle benchmark 뒤로 미룬다.

같은 seed의 짧은 `256x256` 결과 `0.525294`보다 약 47% 낮다. 현재 결과는
**MLP 폭보다 traversal 수와 회귀 학습량의 영향이 더 큼**을 시사한다.
그래도 flat CFR+ 2,000회의 `0.000082`와는 큰 차이가 있으므로 Stud-Leduc
크기에서는 Deep CFR가 효율적인 solver가 아니다. 이 실험의 목적은 큰
7-Stud로 옮기기 전 tensor, reservoir, 평가 경로를 검증하는 것이다.

재현 명령:

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -B D:\Experiment\Toy-Card-Game-Agent\deep_cfr_experiment.py `
  --game stud-leduc --iterations 50 --traversals 500 `
  --policy-hidden 256 --advantage-hidden 256 `
  --batch-size 512 --memory-capacity 250000 `
  --advantage-steps 200 --policy-steps 1000 `
  --cfrplus-iterations 0 --threads 1 --seed 17
```

## 6. 구현 순서와 성공 기준

1. C++ tensorizer와 collision/perfect-recall self-test를 만든다.
2. C++/PyTorch IPC batch inference를 연결한다.
3. 고정 7th root에 exact-history CFR+ oracle을 만든다.
4. 같은 root에서 AsymP와 7th-only Deep CFR를 oracle에 대조한다.
5. 이 비교를 통과한 solver만 5th~7th trunk 실험으로 확장한다.
6. LBR 하한, heuristic EV, reservoir 크기, 처리량을 기록한다.
7. 반복 방문되는 7th roots에 trunk CFV를 보존하는 safe resolver를 붙인다.
8. frozen deterministic H4와 discard-noise H4를 비교해 동결 손실을 잰다.
9. 필요할 때만 H4를 별도 advantage head 또는 별도 CFR로 학습한다.

Deep CFR 채택 기준은 같은 메모리 한도에서 MCCFR보다 낮은 LBR 하한을
얻거나, 같은 LBR 하한에 더 작은 checkpoint로 도달하는 경우다. AsymP는
trunk 대비 LBR를 악화시키지 않고 7th root value 제약을 지킬 때 채택한다.
