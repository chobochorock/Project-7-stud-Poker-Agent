# CFR/MCCFR 구현 및 검증 결과

## 결론

이번 실험에서 확인한 내용은 다음과 같다.

1. 표준 external-sampling MCCFR은 작은 Stud-Leduc에서 exploitability를
   지속적으로 낮췄다. 구현과 수렴 방향은 정상이다.
2. 전체 트리가 메모리에 들어가는 작은 게임에서는 exact CFR+가 MCCFR보다
   훨씬 빠르고 정확했다. 샘플링의 분산을 감수할 이유가 없기 때문이다.
3. 전체 7포커 트리 대신 매 iteration마다 하나의 카드 배치를 고정하고
   5th street 이후를 푸는 fixed-root MCCFR을 적용했다.
4. 7포커 모델은 휴리스틱 상대 성적이 1천 root에서 `-3.805 ante/hand`,
   6.1만 root에서 `-0.084 ante/hand`로 개선되어 통계적 무승부에 도달했다.
5. 이는 7포커의 Nash 수렴 증명이 아니다. H4 선택은 아직 휴리스틱이고,
   power bucket은 서로 다른 정보집합을 병합하며, exact best response도
   계산하지 않기 때문이다.

## 구현 구조

### 작은 게임: Stud-Leduc

파일:

- `D:\Experiment\Toy-Card-Game-Agent\stud_leduc_cfr.cpp`

두 알고리즘을 같은 게임 트리와 exact best-response 평가기에 연결했다.

#### Exact CFR+

한 player update가 모든 chance outcome, 상대 행동, 자기 행동을 순회한다.

```text
전체 게임 트리
→ counterfactual action value 계산
→ regret를 0 이상으로 절단
→ 평균전략 누적
→ exact best response로 exploitability 계산
```

#### External-sampling MCCFR

한 traverser update에서:

- chance node는 한 결과만 표본 추출한다.
- 상대 node는 현재 전략에서 한 행동만 표본 추출한다.
- traverser node는 모든 합법 행동을 평가한다.
- 두 player를 번갈아 traverser로 지정한다.
- 평균전략은 상대 node가 표본 경로에서 방문될 때 누적한다.

```text
sample chance
→ sample opponent action
→ enumerate traverser actions
→ terminal utility 역전파
→ sampled counterfactual regret update
```

2인 zero-sum, perfect recall, 모든 relevant action의 지속적인 sampling이라는
표준 가정 아래 external-sampling MCCFR의 평균전략은 Nash equilibrium으로
수렴한다. 유한 시간에서는 sampling variance 때문에 exact CFR보다 흔들린다.

### 7포커: fixed-root MCCFR

파일:

- `D:\Experiment\Project-7-stud-Poker-Agent\cpp_mccfr\stud_mccfr.cpp`

한 root iteration은 다음과 같다.

```text
52장 deck shuffle
→ 두 플레이어 H4를 기존 heuristic으로 선택
→ 5th street 공개 카드까지 배분
→ 동일한 root에서 P0 external-sampling traversal
→ 동일한 root에서 P1 external-sampling traversal
→ 다음 root용 deck shuffle
```

탐색 중에는 상대 행동 하나를 표본 추출하고, regret를 갱신하는 플레이어의
모든 행동을 평가한다. 이후 street의 카드는 root에 남은 deck에서 배분된다.

정보집합은 `power64_v1.bin` atlas를 사용한다. 따라서 exact card identity
대신 다음과 같은 전략적 power bucket을 공유한다.

```text
관측 가능한 카드와 betting state
→ hand/power feature
→ power atlas bucket
→ bucket별 regret와 average strategy
```

평가할 때는 학습을 완전히 끄고 `average strategy`만 사용한다.

## 중요한 수정

기존 C++ MCCFR은 평균전략을 regret를 갱신하는 traverser node에서 누적했다.
external-sampling의 alternating update에서는 다른 플레이어가 traverser일 때
표본 경로에서 방문된 opponent node의 현재 전략을 누적해야 한다.

이번 구현은 이를 수정했다. 따라서 수정 전 저장 모델의 `strategy_sum`과
수정 후 fixed-root 모델은 같은 의미로 비교하면 안 된다.

## 실험 결과

### Stud-Leduc, rank 3

트리 크기:

- nodes: 28,057
- information sets: 738

동일한 `10,000,000 training node visits` 비교:

| 알고리즘 | Exploitability | 학습 시간 |
|---|---:|---:|
| Exact CFR+ | 0.005182 | 0.210초 |
| External MCCFR | 0.046758 | 2.502초 |

External MCCFR exploitability:

| Node visits | Exploitability |
|---:|---:|
| 1M | 0.1597 |
| 2M | 0.0967 |
| 3M | 0.0727 |
| 5M | 0.0624 |
| 10M | 0.0468 |

### Mini-Stud, rank 4

트리 크기:

- nodes: 128,529
- information sets: 1,824

동일한 `20,000,000 training node visits` 비교:

| 알고리즘 | Exploitability | 학습 시간 |
|---|---:|---:|
| Exact CFR+ | 0.024831 | 0.430초 |
| External MCCFR | 0.044967 | 15.14초 |

External MCCFR exploitability:

| Node visits | Exploitability |
|---:|---:|
| 5M | 0.0991 |
| 10M | 0.0675 |
| 15M | 0.0527 |
| 20M | 0.0450 |

작은 두 게임에서 full CFR+가 더 빠른 이유는 전체 트리가 작고 contiguous
array로 저장되어 있기 때문이다. MCCFR의 장점은 full traversal 자체가
불가능해지는 큰 게임에서 나타난다.

### 7포커 v3, power64, 5th-street root

모든 checkpoint를 같은 seed의 paired 20,000-hand match로 평가했다.

| 누적 root iterations | 상대 | ante/hand | 95% CI |
|---:|---|---:|---:|
| 1,000 | heuristic | -3.8048 | [-4.3369, -3.2726] |
| 11,000 | heuristic | -0.9941 | [-1.3553, -0.6328] |
| 31,000 | heuristic | -0.1620 | [-0.3737, 0.0497] |
| 61,000 | heuristic | -0.0836 | [-0.3000, 0.1328] |

6.1만 root까지 누적된 실제 training node visits는 `74,129,668`회다.
최종 모델은 8,816개 bucket을 사용했고 평가 데이터의 bucket hit rate는
100%였다.

최종 모델을 `belief-br`과 5,000 hands 평가한 결과:

| 상대 | ante/hand | 95% CI |
|---|---:|---:|
| belief-br, 240 particles | -0.1640 | [-0.4456, 0.1176] |

두 최종 결과 모두 95% 신뢰구간이 0을 포함하므로 통계적으로는 무승부다.
`belief-br` 평가는 exact best response가 아니라 별도의 강한 상대에 대한
스트레스 테스트다.

## 재현 명령

### Stud-Leduc 빌드와 자체검사

```powershell
cd D:\Experiment\Toy-Card-Game-Agent
g++ -O3 -std=c++17 stud_leduc_cfr.cpp -o stud_leduc_cfr.exe
.\stud_leduc_cfr.exe --self-test
```

### Exact CFR+와 MCCFR 비교

```powershell
.\stud_leduc_cfr.exe --mode flat --algorithm cfr-plus `
  --node-budget 10000000 --report-nodes 1000000

.\stud_leduc_cfr.exe --mode flat --algorithm external-mccfr `
  --node-budget 10000000 --report-nodes 1000000 --seed 7
```

### Rank 4 mini-Stud

```powershell
g++ -O3 -std=c++17 -DSTUD_LEDUC_RANK_COUNT=4 `
  stud_leduc_cfr.cpp -o stud_leduc_cfr_rank4.exe

.\stud_leduc_cfr_rank4.exe --mode flat --algorithm external-mccfr `
  --node-budget 20000000 --report-nodes 5000000 --seed 7
```

### 7포커 fixed-root 학습

```powershell
cd D:\Experiment\Project-7-stud-Poker-Agent

.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --root-iterations 10000 `
  --root-report-every 10000 `
  --hands 2 `
  --iterations 0 `
  --opponent heuristic `
  --seed 6001 `
  --load cpp_mccfr\root_mccfr_61k.bin `
  --save cpp_mccfr\root_mccfr_71k.bin
```

`--load`와 `--save`를 함께 쓰면 기존 regret와 average strategy에 이어서
학습한다. `--hands 2`는 학습 뒤 형식상 수행하는 최소 평가이며 성능 판단에
사용하면 안 된다.

### 7포커 동결 평가

```powershell
.\cpp_mccfr\stud_mccfr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --load cpp_mccfr\root_mccfr_61k.bin `
  --root-iterations 0 `
  --hands 20000 `
  --iterations 0 `
  --opponent heuristic `
  --seed 5001
```

## 해석과 다음 기준

현재 결과는 “MCCFR이 안 돈다”가 아니라 다음을 의미한다.

- toy game에서는 exact CFR+를 쓰는 것이 맞다.
- 7포커에서는 full CFR가 현실적으로 너무 크므로 MCCFR sampling이 필요하다.
- fixed-root MCCFR은 실제로 휴리스틱과 통계적 무승부까지 개선됐다.
- 하지만 현재 성능이 Nash equilibrium에 가깝다는 증거는 없다.

다음 실험은 새 구조를 더 붙이는 것보다 같은 fixed-root 예산에서
`power64`와 더 세밀한 bucket 하나만 비교하는 것이 우선이다. 성능 차이가
없으면 bucket보다 H4 휴리스틱 또는 imperfect-recall aliasing이 병목이다.

## Policy-aware Local Best Response 평가 (2026-07-29)

기존 `belief-br`은 상대 정책을 조회하지 않고 수작업 action-strength
likelihood를 사용했다. 새 `policy-lbr`은 동결된 MCCFR average strategy를
직접 조회하여 상대 hidden-hand particle을 갱신한다.

관측된 target 행동을 `a_t`, 후보 hidden hand를 `h`라 하면:

```text
b_{t+1}(h) ∝ b_t(h) * sigma_target(a_t | I_t(h))
```

각 LBR 행동은 posterior showdown equity와 target의 실제 다음 fold 확률로
평가한다. 미래 전체를 푸는 exact best response는 아니며 한 단계의
Local Best Response다. 그럼에도 실제 게임에서 사용하는 합법적인 정책이므로
실측 수익은 true best-response value와 exploitability의 하한이다.

첫 구현의 범위:

- 2인 zero-sum, 5th street 이후
- 양쪽 H4 discard/reveal은 기존 heuristic으로 고정
- opponent range는 particle approximation
- target의 average strategy는 읽기 전용
- 가상 hand의 target policy lookup miss는 uniform fallback으로 기록

```powershell
.\cpp_mccfr\stud_mccfr_lbr.exe `
  --bucket power `
  --load-atlas cpp_mccfr\power64_v1.bin `
  --start-street 5 `
  --algorithm mccfr `
  --ante 1000 `
  --load cpp_mccfr\root_mccfr_ante1000_100m.bin `
  --root-iterations 0 `
  --hands 1000 `
  --iterations 0 `
  --opponent policy-lbr `
  --belief-particles 64 `
  --seed 23002
```

동일 seed의 첫 1,000-hand 비교:

| target | LBR ante/hand | 95% CI | policy miss |
|---|---:|---:|---:|
| 10M MCCFR | +1.7479 | [+1.0539, +2.4419] | 0 |
| 28.8M MCCFR checkpoint (`100m.bin` 학습 중) | +1.6086 | [+1.0007, +2.2166] | 0 |

28.8M checkpoint의 하한이 조금 낮지만 신뢰구간이 크게 겹치므로 10M보다
안전하다고 확정할 수는 없다. 반면 두 모델 모두 기존 heuristic 및 heuristic-strength
`belief-br` 대전만으로는 드러나지 않았던 명확한 취약성을 가진다.

이 수치는 exact exploitability도 상한도 아니다. 더 강한 하한이 필요하면
particle 수와 hand 수를 늘리거나 policy-aware IS-MCTS lookahead를 추가한다.
full-game 평가에는 LBR의 H4 선택과 target H4 likelihood도 포함해야 한다.
