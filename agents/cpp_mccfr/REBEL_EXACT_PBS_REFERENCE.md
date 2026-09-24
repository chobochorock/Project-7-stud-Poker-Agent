# 7-Stud exact-PBS ReBeL 기준 구현

## 결론부터

[`stud_rebel_exact_pbs.cpp`](stud_rebel_exact_pbs.cpp)는 기존의 particle/marginal
ReBeL 근사에서 가장 의심스러웠던 부분을 제거한 **정확성 기준 구현**이다.

- H4 이후 가능한 `hidden 2장 + discard 1장` 콤보를 명시적으로 보존한다.
- 상대와 카드가 겹치는 콤보는 joint compatibility로 제거한다.
- 관측된 행동은 현재 search 정책의 확률로 range를 베이즈 갱신한다.
- 베팅 정보집합은 카드와 전체 betting history를 기억한다.
- 한 public belief state(PBS)를 terminal까지 external-sampling CFR+로 푼다.
- 누적 regret, 정보집합 재방문률, 정책 entropy를 CSV와 그래프로 남긴다.
- local regret table과 난수 상태를 checkpoint로 저장하고 이어서 계산한다.

다만 이 파일은 아직 논문의 **전체 ReBeL 학습계가 아니다**. 논문의 value/policy
network 대신 terminal까지 전부 순회하는 full-depth solver를 사용한다. 이
차이는 아래 512-iteration 실험에서 바로 병목으로 나타났다.

## Exact factorized PBS

공개 상태를 `p`라고 하고, 플레이어 `i`의 private type을

```text
q_i = (initial hidden 2장, H4 discard 1장)
```

으로 둔다. 구현은 `q_i`마다 비정규화 reach factor `r_i(q_i)`를 저장하고
joint belief를 다음처럼 나타낸다.

\[
\beta(q_0,q_1\mid p)
\propto
\mathbf 1[\operatorname{compatible}(q_0,q_1,p)]
r_0(q_0)r_1(q_1).
\]

따라서 약 `10,000 x 10,000` joint matrix를 저장하지 않는다. 각 좌석의
약 1만 개 factor만 저장하고, world를 뽑을 때 두 factor에서 독립 표본화한 뒤
카드가 겹치면 거절한다. 이는 product distribution을 compatibility 조건으로
condition한 정확한 표본이다.

### H4 초기값

현재 환경의 H4는 deterministic heuristic이다. 후보 콤보마다 가능한 초기
4장 순서를 모두 열거하고, 실제 공개 카드가 reveal되는 순서의 개수를
`r_i(q_i)`의 초기 질량으로 둔다. discard는 공개되지 않지만 private type에는
포함된다.

H4를 학습 정책으로 바꾸면 이 항도 해당 H4 정책 확률로 바꿔야 한다.

### 행동 posterior

공개 상태 `p`에서 플레이어 `i`가 행동 `a`를 했으면 다음처럼 갱신한다.

\[
r_i'(q_i)
=
r_i(q_i)\,\sigma_i(a\mid I_i(p,q_i)).
\]

상대 factor는 그대로 둔다. 행동 정책은 자기 정보집합과 공개 정보에만
의존하므로, card compatibility를 별도로 유지하면 이 factor update는 정확하다.

이번 코드의 마지막 `belief_update`는 local average search policy로 이 계산을
실제로 수행한다. 기존 구현처럼 blueprint로 과거 range를 재구성한 뒤 local
search 결과를 버리지 않는다.

## 논문 ReBeL과 같은 점

| 항목 | 이 구현 |
|---|---|
| 상태 단위 | world state가 아니라 PBS에서 search 시작 |
| private belief | 각 플레이어 infostate의 reach distribution |
| card blocker | joint compatibility로 정확히 반영 |
| search | 두 플레이어 zero-sum CFR 계열 self-play |
| posterior | search policy의 행동 확률로 갱신 |
| 정보집합 | 상대 hidden을 제외하고 자기 private card와 전체 공개 history 사용 |
| 정책 | current regret-matching과 average strategy를 분리 |

ReBeL 논문은 PBS에서 search하고 그 결과로 value/policy 함수를 학습하는
구조이며, 2인 zero-sum imperfect-information game에서 이 결합을 다룬다.
[원 논문](https://arxiv.org/abs/2007.13544),
[공식 구현](https://github.com/facebookresearch/REBEL)을 기준으로 비교했다.

## 논문 ReBeL과 다른 점

| 논문 ReBeL | 현재 기준 구현 |
|---|---|
| depth-limited CFR + leaf PBS value | 5th에서 terminal까지 full-depth CFR |
| 많은 self-play PBS로 value network 학습 | 한 seed의 고정 5th PBS를 반복 solve |
| 학습된 policy network로 search warm start 가능 | uniform 또는 기존 MCCFR blueprint prior |
| leaf에서 private-state별 counterfactual value vector | terminal chip payoff를 직접 계산 |
| network가 서로 다른 PBS 사이에서 일반화 | local regret table은 해당 PBS 안에서만 유효 |
| training과 test에서 recursive PBS progression | 현재는 한 행동 posterior update까지 검증 |

따라서 실행 파일 이름의 `reference`가 중요하다. 이것은 exact PBS와 regret
회계의 정답지이며, 이 결과만으로 “7-Stud에 원본 ReBeL을 완성했다”고
주장할 수는 없다.

기존 [`stud_rebel_recursive.cpp`](stud_rebel_recursive.cpp)와의 차이는 반대
방향이다. 기존 구현은 V7 network와 6th depth limit는 있었지만 다음을
근사했다.

- joint particle 64/240개
- 208차원 card marginal belief summary
- 실제 다음 street에서 blueprint로 range 재구성
- power-memory16 private abstraction

새 기준 구현은 network를 포기하는 대신 이 네 항목을 exact combo/PBS와
perfect-recall key로 바꿨다. 두 구현 사이의 빈칸이 앞으로 구현해야 할
실제 ReBeL이다.

## Regret 지표

CSV의 핵심 열은 다음과 같다.

### `cumulative_positive_regret_mean`

관측된 정보집합과 합법 행동 전체에서 `max(R(I,a), 0)`의 평균이다. 누적값
자체이므로 반드시 감소하지 않는다.

### `average_positive_regret_mean`

위 평균을 완료 iteration `T`로 나눈 값이다.

\[
\frac{1}{T|\mathcal D|}
\sum_{(I,a)\in\mathcal D}\max(R(I,a),0).
\]

감소 여부는 볼 수 있지만, 새 정보집합이 계속 생길 때에는 `1/T` 분모 효과와
진짜 반복 개선을 구분해야 한다.

### `regret_bound_proxy`

\[
\frac{1}{2T}
\sum_i\sum_{I\in\mathcal D_i}\max_a R_i^+(I,a).
\]

형태는 Zinkevich regret decomposition과 같지만, 여기서는 sampled traversal이
아직 대부분의 원 정보집합을 한 번도 보지 못했다. 그러므로 현재 값은 exact
exploitability upper bound가 아니라 **발견된 정보집합에서의 sampled
decomposition diagnostic**이다.

### `touch_normalized_regret`

\[
\frac{\sum_{i,I}\max_a R_i^+(I,a)}
{\sum_{i,I}\operatorname{touch}(I)}.
\]

`T`가 아니라 실제 정보집합 방문 수로 나눈다. 새 node가 늘어나는 동안에도
해석하기 쉽다.

### `revisited_infoset_fraction`

두 번 이상 방문한 exact information set의 비율이다. 이 값이 매우 낮으면
`R/T` 감소를 수렴으로 해석하면 안 된다.

## 512-iteration 결과

실행 결과는
[`rebel_exact_pbs_512_v2.csv`](results/rebel_exact_pbs_512_v2.csv)와
[`rebel_exact_pbs_512_v2.png`](results/rebel_exact_pbs_512_v2.png)에 있다.

| 지표 | 32 iter | 512 iter |
|---|---:|---:|
| exact infosets | 26,483 | 475,679 |
| 5th infosets | 358 | 5,685 |
| 6th infosets | 2,035 | 33,837 |
| 7th infosets | 24,090 | 436,157 |
| cumulative positive regret mean | 10.63 | 11.01 |
| positive regret mean / T | 0.332 | 0.0215 |
| sampled decomposition proxy | 14,858 | 17,736 |
| average touches | 1.0000 | 1.0003 |
| revisited infoset fraction | 0% | 0.0296% |
| local action-policy coverage | - | 8.35% |

`positive regret mean / T`의 log-log slope는 `-0.992`지만 decomposition
proxy의 slope는 `+0.048`이다. 누적 regret 평균도 줄지 않았고 exact
정보집합 재방문률은 0.03%에 불과하다.

즉 **감소 곡선은 거의 전부 분모 효과**다. exact combo PBS의 메모리는
감당되지만, 5th에서 terminal까지 full-depth로 펴는 탐색은 새로운 7th
정보집합을 너무 빨리 만든다. 원본 ReBeL의 depth limit와 PBS value network가
필요한 이유가 실측으로 확인됐다.

![512-iteration regret diagnostics](results/rebel_exact_pbs_512_v2.png)

## 빌드와 self-test

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_exact_pbs.cpp `
  -o cpp_mccfr\stud_rebel_exact_pbs.exe

.\cpp_mccfr\stud_rebel_exact_pbs.exe --self-test
```

검사는 다음을 확인한다.

- true H4 private type이 exact range에 존재
- public/private 카드 중복 없는 compatible world sampling
- local CFR이 유한한 정보집합을 생성
- checkpoint가 iteration, RNG, regret table을 복원

## 실험과 재개

새 실험. full-depth 기준 구현은 정보집합 증가가 빠르므로 먼저 1,000회만
확인한다.

```powershell
.\cpp_mccfr\stud_rebel_exact_pbs.exe `
  --iterations 1000 `
  --report-every 50 `
  --seed 82001 `
  --ante 1000 --stack-ante 1000 `
  --metrics cpp_mccfr\results\rebel_exact_pbs_1k.csv `
  --checkpoint cpp_mccfr\results\rebel_exact_pbs_1k.bin
```

중단 후 같은 public root와 checkpoint를 재개한다. `--iterations`는 추가량이
아니라 목표 누적 iteration이다.

```powershell
.\cpp_mccfr\stud_rebel_exact_pbs.exe `
  --iterations 2000 `
  --report-every 50 `
  --seed 82001 `
  --ante 1000 --stack-ante 1000 `
  --metrics cpp_mccfr\results\rebel_exact_pbs_1k.csv `
  --checkpoint cpp_mccfr\results\rebel_exact_pbs_1k.bin `
  --resume cpp_mccfr\results\rebel_exact_pbs_1k.bin
```

기존 30M blueprint를 local prior로만 사용할 수도 있다.

```powershell
.\cpp_mccfr\stud_rebel_exact_pbs.exe `
  --iterations 1000 --report-every 50 `
  --blueprint-model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --blueprint-atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --blueprint-bucket power-memory16 `
  --blueprint-algorithm mccfr `
  --blueprint-prior 100 `
  --metrics cpp_mccfr\results\rebel_exact_pbs_prior.csv `
  --checkpoint cpp_mccfr\results\rebel_exact_pbs_prior.bin
```

이 prior는 첫 search policy를 빠르게 만들 뿐 leaf value를 대신하지 않는다.
iteration이 커지면 exact regret이 prior를 덮어쓴다.

그래프:

```powershell
python -B cpp_mccfr\plot_rebel_exact_pbs.py `
  --input cpp_mccfr\results\rebel_exact_pbs_1k.csv `
  --output cpp_mccfr\results\rebel_exact_pbs_1k.png
```

현재 512회 checkpoint가 약 `119 MiB`였다. 정보집합 수가 같은 속도로 늘면
10,000회는 수 GiB RAM/checkpoint가 될 수 있다. 이 reference에서 장시간
full-depth 학습을 돌리기보다, 다음 절의 depth limit를 먼저 구현하는 것이
맞다.

## 실제 ReBeL로 이어지는 최소 순서

1. **7th exact leaf oracle**: terminal CFR로 각 7th PBS의 private-type별
   counterfactual value target을 만든다.
2. **V7을 scalar가 아니라 type-wise value로 학습**: combo feature와 전체
   PBS context를 받아 각 private type의 counterfactual value를 예측한다.
3. **6th depth-limited search**: 한 public chance boundary까지만 풀고 V7을
   호출한다. local average policy로 다음 PBS를 갱신한다.
4. **self-play PBS replay**: 검색 중 방문한 PBS, search policy, value target을
   checkpoint 가능한 replay에 누적한다.
5. **V6 이후 5th 확장**: held-out type-wise value error와 LBR이 개선될 때만
   한 street씩 올린다.

이 순서에서는 이번 exact-PBS 구현이 range 생성, posterior update, compatible
sampling, perfect-recall key, regret 계측의 공통 기준으로 남는다. 다시 시작할
필요가 없도록 metrics CSV와 local checkpoint를 이미 분리해 두었다.
