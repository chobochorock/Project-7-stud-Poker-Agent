# 7-Stud Deep CFR 구현 감사와 교정 실험

## 결론

기존 실패의 주원인을 단순히 네트워크 크기로 볼 수 없다. 기존 H128 모델도
약 25만 파라미터였고, Deep CFR 원 논문의 실험망은 약 9.9만 파라미터였다.
교정판은 30만 파라미터의 branch/residual 모델을 사용했지만 같은 작은 계산
예산에서 성능이 개선되지 않았다.

가장 정확한 현재 결론은 다음과 같다.

1. 기존 구현에는 원래 Deep CFR와 다른 표본화와 loss가 있었다.
2. 이를 교정하고 논문형 네트워크를 넣어도 5 iteration x 200 traversal로는
   학습되지 않았다.
3. 지금 결과는 Deep CFR 자체의 실패가 아니라, 7-Stud에 비해 계산 예산과
   memory가 지나치게 작은 실험의 실패다.
4. 현재 단건 IPC 추론으로 논문 규모까지 늘리는 것은 비효율적이다.

## 확인한 기존 구조

- 5th street부터 두 플레이어를 번갈아 external-sampling MCCFR로 순회한다.
- traverser 정보집합에서는 모든 합법 행동을 열거한다.
- 상대 행동과 이후 chance는 한 경로를 표본화한다.
- 플레이어별 advantage network 두 개와 average-policy network 하나를 둔다.
- advantage와 strategy 표본을 reservoir에 저장하고 매 iteration 처음부터
  네트워크를 다시 적합한다.
- H4 discard/reveal은 heuristic으로 고정되어 있다.

이 골격은 Deep CFR와 맞다. 문제는 세부 학습 목적과 규모였다.

## 교정한 항목

### 1. Global uniform reservoir

기존 street-stratified 버전은 5th/6th/7th에 memory를 같은 크기로 강제
배분하고 batch에서도 street를 균등 선택했다. 중요도 보정이 없으므로 원래
external-sampling 방문분포와 다른 목적함수를 학습한다.

교정판은 모든 표본에 대해 하나의 Vitter reservoir를 사용한다. 따라서
보관될 확률이 street와 무관하게 같다.

### 2. 합법 행동만 loss에 포함

이전에는 8개 출력 전체에 MSE를 적용했다. 교정판은 tensor 마지막의 legal
mask를 사용해 현재 정보집합에서 합법인 행동의 오차만 평균한다.

### 3. LCFR weight 안정화

기존 `sqrt(t)`를 오차에 곱한 뒤 제곱한 방식은 결과적으로 weight `t`라서
수학적으로 틀리지는 않았다. 교정판은 논문 표기와 같은 `2t/T`를 직접
사용해 iteration이 증가해도 gradient scale이 불필요하게 커지지 않게 했다.

### 4. Advantage scale

chip payoff의 큰 범위 때문에 advantage target을 32로 나누어 적합한다.
모든 행동 advantage에 같은 양의 상수를 적용하므로 regret matching 정책은
바뀌지 않는다.

### 5. Network

plain 2-layer MLP를 다음으로 바꿨다.

- 카드/공개정보 branch
- betting-history branch
- branch 결합 layer
- residual layers
- 마지막 hidden feature LayerNorm
- action output

원 논문이 빠른 수렴에 중요했다고 보고한 branch, skip connection,
normalization을 최소 형태로 반영한 것이다.

### 6. 진단과 평가

- iteration별 궤적 BR-gap proxy와 street별 기여
- 모든 `policy_iN.pt` 체크포인트
- heuristic paired-seat 평가
- network policy를 대상으로 한 Policy-LBR
- 학습 이력 PNG와 `evaluations.json`

## 교정 실험

설정:

```text
iterations = 5
traversals/player/iteration = 200
memory = 50,000 per reservoir
hidden = 128
residual layers = 2
advantage SGD steps = 500
policy SGD steps = 1,000
```

학습 시간은 713.4초였다.

| iteration | heuristic screen | Policy-LBR screen | legacy per-record gap |
|---:|---:|---:|---:|
| 1 | -6.225 | 18.113 | 118.918 |
| 2 | -4.381 | 45.810 | 128.951 |
| 3 | -5.122 | 21.664 | 136.104 |
| 4 | -5.550 | 41.262 | 179.776 |
| 5 | -6.288 | 18.891 | 153.636 |

Policy-LBR는 checkpoint당 200 hands, 32 particles인 방향성 screen이라 CI가
매우 넓다. 절대 exploitability 추정값으로 사용하면 안 된다. 다만 큰 누수가
남았다는 판정에는 충분하다.

마지막 열은 초기 구현에서 advantage record별 gap을 평균한 값이다. 요청했던
궤적별 gap 합이 아니므로 BR-gap 곡선으로 사용하지 않는다. 이 실행은 원본
sample file을 삭제했기 때문에 올바른 값을 사후 복원할 수 없다.

10,000-hand heuristic 재평가:

| checkpoint | ante/hand | 95% CI |
|---|---:|---:|
| iteration 2 | -5.844 | [-6.715, -4.973] |
| iteration 5 | -5.917 | [-6.849, -4.985] |

따라서 이 예산에서 유의한 개선은 없었다.

## 왜 아직 실패했는가

마지막 iteration에서 strategy 표본은 누적 932,151개 생성됐지만 reservoir는
50,000개만 보관했다. 보관 표본은 다음처럼 7th에 집중됐다.

```text
5th:   407
6th: 2,321
7th: 47,272
```

이는 global reservoir의 버그가 아니라 실제 external-sampling 방문분포다.
현재는 5th/6th의 함수근사 자료가 너무 적고, 7th 표본조차 대부분 버린다.

원 논문 실험은 대략 다음 규모였다.

- iteration당 10,000 traversal
- advantage/strategy memory 각각 최대 40,000,000
- iteration당 4,000 SGD minibatch update
- batch size 10,000

현재 실험은 iteration당 traversal이 50배 작고 memory는 800배 작다. 따라서
모델 폭을 더 키우는 것보다 표본 생성과 memory 규모가 먼저다.

또한 C++ traverser가 각 network state를 Python IPC에 한 건씩 질의한다.
논문 규모로 확장하기 전에 batched inference 또는 C++ runtime 추론으로
왕복 병목을 제거해야 한다.

## 지표 해석

수정된 `trajectory_br_gap_mean_ante`는 각 training root에서 traverser가 이미
계산한 action value를 사용해 다음 값을 누적한다.

```math
g_t = E_{q \sim Q_t}\left[
  \sum_{I \in q}
  \max\left(0, \max_a Q(I,a)-\sum_a\pi(a|I)Q(I,a)\right)
\right]
```

P0와 P1의 평균을 그래프에 사용한다. CFR 순회가 이미 가진 `action_values`와
`node_value`만 사용하므로 추가 탐색이나 network query가 없다. 이 값이
감소하면 표본분포에서 수익성 있는 국소 일탈의 합이 줄고 있다는 신호다.
그러나 상대/chance를 표본화한 trajectory-relaxed proxy이므로 exact BR이나
exploitability 인증값은 아니다. 최종 Policy-LBR는 드물게 보정용으로만 쓴다.

## 재현 명령

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $py -B train_deep_cfr_7th.py `
  --run-dir models\deep_cfr_corrected_h128_v2 `
  --start-street 5 --iterations 5 --traversals 200 `
  --memory-capacity 50000 --hidden 128 --layers 2 `
  --batch-size 512 --advantage-steps 500 --policy-steps 1000 `
  --advantage-scale 32 --save-policy-every 1 `
  --threads 1 --seed 32001

& $py -B analyze_deep_cfr_history.py `
  --run-dir models\deep_cfr_corrected_h128_v2
```

위 명령은 학습 로그의 BR-gap만 읽으며 대전을 실행하지 않는다. 휴리스틱
평가는 `--hands 2000`을 명시했을 때만 수행한다. 보다 강한 최종 LBR은
마지막 checkpoint에만 별도로 수행한다.

```powershell
& $py -B evaluate_deep_cfr_7th.py `
  --model models\deep_cfr_corrected_h128_v1\policy_i5.pt `
  --start-street 5 --opponent policy-lbr `
  --hands 10000 --belief-particles 240 --threads 1 --seed 39001
```

## Random trajectory clustering과의 관계

50% 또는 70% random trajectory로 atlas를 만드는 아이디어는 tabular
hard-cluster CFR의 heuristic coverage 편향을 검사하는 유효한 별도 실험이다.
하지만 Deep CFR는 카드 power atlas를 사용하지 않으므로 이 변경이 Deep CFR
실패를 직접 고치지는 않는다.

다음 atlas 실험은 heuristic-only와 random-mixture atlas를 같은 CFR seed와
root budget으로 비교해야 한다. random 비율은 한꺼번에 50%와 70%를 모두
넣기보다 `0, 0.5, 0.7` 세 점을 고정해 coverage, heuristic EV, LBR 하한을
같이 측정하는 것이 가장 싼 판별법이다.

## 다음 게이트

Deep CFR를 계속하려면 다음 순서가 맞다.

1. IPC batch 또는 C++ inference로 traversal throughput 개선
2. memory를 최소 1M 이상으로 확대
3. `K=1,000` 이상에서 iteration scaling curve 확인
4. sampled gap이 감소할 때만 240-particle, 10k-hand LBR 수행

이 네 단계 전에는 hidden 256/512 확장은 우선순위가 아니다.
