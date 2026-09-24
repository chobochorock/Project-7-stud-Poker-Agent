# T=1 counterfactual-regret upper-bound estimator

## 목적

고정된 평균 정책 `sigma_bar`의 5th+ 게임 exploitability에 대해 다음
Zinkevich식 후회 분해를 계산한다.

```text
NashConv(sigma_bar)
  <= sum_i sum_{I in I_i}
       max(0, max_a v_tilde(I,a) - v_tilde(I))
```

여기서 `v_tilde`의 도달확률은 상대와 chance만 포함하는 비정규화
counterfactual reach다. 플레이어별 값을 합한 것이 NashConv이고,
OpenSpiel 관례의 2인 zero-sum exploitability는 그 절반이다.

## 구현

`stud_regret_upper.cpp`는 각 딜에 대해 플레이어 0과 1을 따로 순회한다.

- chance는 루트에서 완전한 덱 하나를 표본화한다.
- 상대 행동은 저장된 평균 정책에서 표본화한다.
- 대상 플레이어 행동은 모두 열거한다.
- 정책 조회만 기존 power bucket을 사용한다.
- 후회 합산 키는 카드와 정확한 betting history를 보존한 원 게임
  information set이다.
- H4는 현재 시스템과 동일한 heuristic으로 고정한다. 따라서 결과 범위는
  H4를 제외한 5th+ 부분게임이다.

상대와 chance를 그 도달분포에서 직접 표본화하므로 별도의 정규화된
`pi_-i`를 곱하지 않는다. 대상 플레이어 행동을 모두 열거하므로 자기
도달확률도 들어가지 않는다.

## 두 출력의 차이

### Original-infoset plug-in estimate

`original_infoset_plugin_estimate`는 표본을 원본 information set별로 모은
뒤 행동별 평균 이전의 합을 구하고 마지막에 `max`를 적용한다. 표본 수가
무한히 커지면 요청한 식으로 수렴한다.

유한 표본에서는 결정적인 상한이 아니다. `max` 때문에 보통 위쪽으로
편향되지만, 한 번의 임의 표본값이 참 상한보다 항상 크다는 PAC 보장은
없다.

### Trajectory-relaxed upper

`trajectory_relaxed_upper_mean`은 각 표본 안에서 먼저 `max`를 취한다.

```text
max_a E[X_a] <= E[max_a X_a]
```

따라서 그 기대값은 원래 T=1 후회 상한보다 더 느슨한 상한이다. 평균,
표준오차, 정규근사 95% 상측 신뢰한계를 함께 출력한다. 이 신뢰한계는
진단용이며 유한표본의 엄밀한 PAC 인증값은 아니다. 알려진 최악 범위를
사용한 Hoeffding 상한은 이 게임에서 지나치게 커 실용성이 없다.

## 실행

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_regret_upper.cpp `
  -o cpp_mccfr\stud_regret_upper.exe

.\cpp_mccfr\stud_regret_upper.exe `
  --bucket power-memory16 `
  --model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --samples 10000 `
  --report-every 1000 `
  --infoset-cap 2000000 `
  --ante 1000 `
  --stack-ante 1000 `
  --seed 72101
```

`--infoset-cap`은 원본 information set 맵이 RAM을 무제한 점유하지 못하게
한다. 상한에 닿으면 결과를 조용히 잘라내지 않고 오류로 종료한다.

## 30M 모델의 10,000표본 결과

```text
original infosets: 664,274
all infosets had one visit
plug-in exploitability estimate: 531.459 ante/hand
trajectory-relaxed exploitability 95% upper: 566.714 ante/hand
runtime: 5.17 seconds
```

이 값은 정책이 실제로 500 ante 이상 착취된다는 뜻이 아니다. 정확한 카드
information set가 거의 재방문되지 않아, 각 hidden-state 표본에서 가장
좋았던 행동을 사후 선택하는 완전정보 hindsight 상한에 가까워졌다는
뜻이다. 현재 LBR 하한이 약 1 ante라면 얻어진 샌드위치는 너무 넓어 실제
품질 판정에는 쓸 수 없다.

## 다음 tightening 순서

1. 카드 suit permutation을 정확한 게임 대칭으로 canonicalize한다.
2. 같은 원본 information set에서 상대 hidden card를 조건부 재표본화한다.
3. 7th-street public subgame처럼 chance 공간이 작은 범위에서 exact 합을
   계산한다.

첫 번째와 두 번째가 없으면 표본 수를 단순히 늘려도 새 information set가
거의 선형으로 늘어나므로 `max` 편향이 매우 느리게 줄어든다.
