# Hand-wise sampled BR-gap experiment

## 지표

각 외부표본 순회에서 대상 플레이어의 행동은 모두 열거하고 상대 행동은
현재 정책에서 표본화한다. 방문한 원 게임 정보집합에서 다음 값을 더한다.

```text
gap(I) = max_a Q(I,a) - sum_a pi(a|I) Q(I,a)
```

두 플레이어 합의 절반을 ante로 나눈 `sampled BR-gap proxy`를 한 핸드의
값으로 저장한다. 0이면 해당 표본 순회에서 발견한 수익성 있는 한 단계
이탈이 없다는 뜻이다.

이는 exact best response나 원 게임 exploitability의 인증 상한이 아니다.
상대/chance를 표본화한 뒤 max를 먼저 취하는 trajectory-relaxed 진단값이다.
정책 변화의 방향을 싸게 연속 측정하는 데 사용하고, 최종 LBR은 드물게
보정용으로만 사용한다.

새 학습 코드에서는 별도 평가 트리를 다시 만들지 말고 CFR 순회가 이미
계산한 `action_values`와 `node_value`에서 같은 gap을 누적한다. 이 실험은
고정 heuristic과 독립 평가 딜을 공정하게 비교하기 위해 별도 순회를
사용했으며, 그래서 일반 100k 학습보다 훨씬 느리다.

## 실험

- 고정 baseline heuristic
- 기존 k512 power atlas + memory16 abstraction
- k512 MCCFR은 regret과 평균전략이 없는 cold start
- 매 단계에서 독립 학습 딜 하나로 P0/P1을 갱신
- 별도의 공통 평가 딜에서 두 정책의 gap을 기록
- H4는 기존 heuristic으로 고정, 범위는 5th~7th

```powershell
g++ -O3 -std=c++17 br_gap_experiment\compare_heuristic_k512.cpp `
  -o br_gap_experiment\compare_heuristic_k512.exe

.\br_gap_experiment\compare_heuristic_k512.exe `
  --hands 100000 `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --csv br_gap_experiment\results\per_hand.csv `
  --save br_gap_experiment\results\k512_100k.bin `
  --report-every 1000 `
  --seed 81001

python -B br_gap_experiment\plot_br_gap.py `
  --input br_gap_experiment\results\per_hand.csv `
  --output br_gap_experiment\results\br_gap_curve.png
```

CSV에는 100,000개 핸드의 원자료가 모두 남는다. 그래프 위 패널은
geometric bin 평균과 95% CI를 log-log 축으로, 아래 패널은 누적 평균을
log-x 축으로 표시한다.

실행 결과와 해석은 [RESULTS_KO.md](RESULTS_KO.md)에 기록했다.
