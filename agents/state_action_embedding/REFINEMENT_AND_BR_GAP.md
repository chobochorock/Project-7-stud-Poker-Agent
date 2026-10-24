# 분할 기준과 Local BR-gap 평가

2026-10-03. **평가기는 구현됨. FPIRA/CFR+IRA 분할 학습은 아직 계획이며 실행하지 않음.**
기존 encoder, 군집, regret/평균전략 checkpoint를 수정하지 않는다.

후속 업데이트: [sampled local refinement 파일럿](REFINEMENT_EXPERIMENT.md)은100k까지
실행했다. 원형 IRA 재현은 아니며, 확인 gate를 통과한 분할이0개여서 채택된 분할의
효과는 아직 평가하지 못했다. 아래는 해당 파일럿 이전의 원형/확장 구분과 평가 설계다.

## 무엇을 문제로 볼 것인가

큰 후회나 큰 Local BR-gap 하나만으로는 abstraction 오류라고 판단할 수 없다.
같은 bucket의 모든 상황이 레이즈를 선호한다면, bucket을 나누기보다 공통 정책을
레이즈 쪽으로 학습하면 된다. 반대로 어떤 상황은 레이즈, 다른 상황은 폴드를
선호한다면 하나의 정책을 강제하는 것이 병목일 수 있다. 비교 대상은 마지막에
얻은 단일 reward가 아니라 행동별 counterfactual value/regret이다.

### FPIRA 원형

상대 평균전략에 대한 원게임의 정확 BR을 계산한다. 같은 bucket에 속한 원정보집합이
서로 다른 BR 행동을 요구하면 이를 표현할 수 있도록 분리한다. 또한 원정보집합에서
평균전략을 갱신한 결과와 bucket에서 갱신한 결과의 가치 차이를 검사해 필요한 구분을
복구한다. 단순한 centroid 거리 기준이나 한 번의 local rollout 분할과는 다르다.

### CFR+IRA 원형

**지속적인 후회 검사:** 원정보집합 일부를 감시하고, 일정 구간의 signed counterfactual
regret 합을 구간 길이로 나눈 최대값이 논문의 감소 기준을 넘으면 해당 정보집합을
부모에서 분리한다. 기준의 스케일은 payoff 범위, 행동 수, 감시 구간 길이에 따른다.
임의의 고정 'gap > 0.1' 검사가 아니다. 여기서 감시용 regret는 bucket의 clipped
누적 regret나 terminal reward label과 구분한다.

간략히 쓰면 `max_a sum(r_tau(I,a)) / n > L_I(n)`일 때 분리한다. `n`은 감시
구간 길이이며, 예를 들어 `L_I(n) = Delta_I * sqrt(|A(I)| / n)` 형태의 감소 기준을
쓸 수 있다. `Delta_I`는 payoff 범위다. 정확한 CF regret에 대한 원형 기준이며,
표본 regret의 오차를 빼고 이 부등식을 그대로 쓰면 근거 없는 분할이 될 수 있다.

**빠른 보조 검사:** 각 감시 정보집합에서 현재 counterfactual regret가 최대값에 가까운
행동들의 집합을 구한다. 같은 집합을 갖는 항목끼리 자식 bucket으로 나눈다.
감시하지 않은 항목은 가장 큰 그룹에 남긴다. 이 보조 검사만으로는 수렴 보장이 없다.

| 같은 bucket 안의 상황 | 폴드 regret | 레이즈 regret | 분할 그룹 |
|---|---:|---:|---|
| A (설명용 가상 값) | -2 | +2 | 레이즈 선호 |
| B (설명용 가상 값) | +3 | -3 | 폴드 선호 |

둘 다 레이즈 regret가 최대라면 이 보조 기준에서는 분할하지 않는다. 동점/근접한
행동은 하나를 무작위 선택하지 않고 집합으로 다룬다.

원형은 분할된 항목의 regret를 0으로 재초기화하고 해당 평균전략도 다시 누적한다.
사용자의 이전 '부모 regret 보존' 방식은 이 원형과 다르다. 보존형은 별도 warm-start
변형으로 검증해야 하며 부모의 전체 합계를 모든 자식에게 복제하면 안 된다.

출처: [원문 Sections 4.2, 4.3.1, 4.3.2 / Algorithms 9,10](https://arxiv.org/pdf/1803.05392),
[IJCAI 요약](https://www.ijcai.org/proceedings/2020/0701.pdf).
이번 계획은 확립된 방법의 재현과 확장이지 새 알고리즘의 주장으로 제시하지 않는다.

## 어떻게 나눌 것인가

1. Player/street/합법 행동집합 등의 호환 조건을 유지한다.
2. 정확한 원정보집합은 자신의 관측 가능한 사건 이력으로 식별한다. 상대의 실제 숨은 패로 나누지 않는다.
3. Bound 검사에서 걸린 정보집합은 `그 정보집합 / 나머지`로 분리한다.
4. 보조 검사에서는 `레이즈 선호 / 폴드 선호 / 동점 집합 ...`처럼 행동 선호 집합별로 나눈다.
5. 부모 내부의 소속만 바꾸며 다른 부모의 centroid나 영역은 움직이지 않는다.

처음에는 원정보집합 ID의 예외 매핑으로 분할을 표현하는 편이 검증하기 쉽다.
아직 보지 못한 정보집합을 feature 경계로 일반화하는 분할 트리는 추가적인 근사다.
분할 후 다시 학습하기 전의 정책 초기화 효과와, 정보 구분을 늘린 효과도 구별해야 한다.

7-stud에서 sampling으로 regret/BR을 추정하고 통계적으로 반복 확인하는 방식은
원형이 아닌 확장이다. 표본 잡음을 무시한 단일 argmax로 즉시 분리하지 않고,
독립 확인 표본과 불확실성을 고려하는 것이 제안이며 아직 threshold를 확정하지 않았다.
원형의 전 트리 순회 보장을 MCCFR 표본 학습에 그대로 적용하지 않는다.

## Local BR-gap은 bucket과 독립적으로 평가한다

고정한 원정보집합 I에서 현재 snapshot의 정책 sigma를 조회한다. 각 합법 행동 a만
강제하고 이후 양쪽 모두 그 snapshot의 정책을 따르는 수익을 여러 숨은 세계에서 평균한다.

```text
Q_hat(I,a) = mean over conditional hidden worlds and continuation rollouts
gap_hat(I) = max_a Q_hat(I,a) - sum_a sigma(a|I) Q_hat(I,a)
panel_gap = mean over the same fixed original information sets
```

**숨은 세계마다 max를 먼저 구하지 않는다.** 평가의 단위와 가중치는 원정보집합에
고정한다. bucket 수가 늘어도 root/표본/seed는 그대로이며 자식 수에 따라 가중치가
증가하지 않는다. `finish_from_fifth`는 agent의 행동 API만 호출하므로 미래 분할 정책도
같은 API에 연결할 수 있다. 단, 그 정책의 encoder/분할 구조/통계를 함께 snapshot해야 한다.

### 현재 구현 범위

- 원래 Python seven-poker v3, heads-up cash,1000chips/ante1, uniform discard/reveal.
- **5구 첫 베팅 결정만** 평가한다. 모든 street의 on-policy 상태 평가나 전체 BR은 아니다.
- 이 시점에는 과거 베팅이 없어 uniform H4에 조건화한 상대 초기 카드/버린 카드/초기 순서와
  미래 덱을 정확한 조건부 분포에서 표본화할 수 있다. 상대 관점 사건 이력도 함께 재구성한다.
- 각 root의 모든 합법 행동을 기존 환경 함수로 showdown/종료까지 rollout한다.
  원래 latent/history 추론과 실제 합법 행동을 사용한다. 초기 hidden state의 실제 정답을
  고정해 rollout하는 평가가 아니다.
- 저장된10k/100k/1M의 current/average를 따로 평가한다. Root/particle/action RNG를 raw와
  latent 사이에 맞춘다. 기존 C++ 고정 H4 결과와는 조건이 달라 직접 겹치지 않는다.
- 유한 표본 max의 낙관 편향 진단을 위해 전체/절반 particle 점수와 두 표본 절반을 교차 사용한
  held-out one-step deviation을 저장한다. 후자는 음수가 가능하며 log 축에 넣거나0으로 자르지 않는다.
- Root별 normal interval은 descriptive이다. 단일 학습 seed와 heavy-tailed 수익의 한계가 있다.
  Local BR-gap 감소만으로 LBR 수익이나 exploitability 개선을 단정하지 않는다.

## 실행 방법

프로젝트 root에서 실행한다. 새 output 디렉터리를 사용하며 기존 결과를 덮어쓰지 않는다.

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/reconstruction_cfr_20261002'
$out = 'agents/state_action_embedding/data/reconstruction_gap_20261003'
& $py -m unittest agents.state_action_embedding.test_reconstruction_br_gap -v
& $py -m agents.state_action_embedding.reconstruction_br_gap --run $run --out-dir $out --roots 128 --particles 64 --workers 8
# 중단 후 미완료 root만 평가: 위 명령 끝에 --resume 추가
python -m agents.state_action_embedding.reconstruction_cfr_report --gap-run $out

# 같은 첫 16개 root에서 표본 수만 늘린 정밀도 검사 (추가 학습 아님)
$audit = 'agents/state_action_embedding/data/reconstruction_gap_particle_audit_20261003'
& $py -m agents.state_action_embedding.reconstruction_br_gap --run $run --out-dir $audit --roots 16 --particles 512 --hands 10000 1000000 --workers 8
python -m agents.state_action_embedding.reconstruction_cfr_report --gap-run $audit --gap-reference $out
```

`config.json`: 조건, 코드/모델/checkpoint hash.
`per_root.jsonl`: 정보집합 fingerprint, 행동별 Q 평균/SE, 정책 확률, gap, 표본 수, 누락 수, 시간.
`summary.json`, `RESULTS.md`, `local_br_gap.png`: raw 비교/paired 차이/로그 그래프/비율.
대형 rollout replay나 새 학습 테이블은 저장하지 않는다.

저장된 세 checkpoint에서의 경향은 재학습 없이 볼 수 있다. 매 hand의 과거 정책은 남아
있지 않으므로 당시 매-hand 곡선 전체를 복원하는 것은 아니다. 후속 학습에서는 학습 전후와
각 분할 전후 snapshot을 남겨 같은 평가기로 추가 측정하는 계획이다.

## 2026-10-03 완료된 측정

학습 seed111, 평가 seed261003. 두 정책(current/average)을 각각 평가했다.
128개 원정보집합, 상태당64개 particle의 [전체 결과](data/reconstruction_gap_20261003/RESULTS.md)와
[로그 그래프](data/reconstruction_gap_20261003/local_br_gap.png)를 저장했다.
평균전략의 Local BR-gap은 다음과 같다 (ante 단위, 작을수록 좋음).

| 학습 hands | raw k-means | 복원 latent k-means |
|---:|---:|---:|
| 10k | 18.3891 | 15.3756 |
| 100k | 23.5700 | 23.3190 |
| 1M | 24.5737 | 25.3878 |

이 패널에서는 평균전략 gap의 감소가 보이지 않는다. 1M latent-minus-raw 차이는
0.8141, root 기준 normal95% 구간은[-4.3102,5.9384]로 우위가 확인되지 않는다.
이는 원게임 exploitability나 LBR 수익을 측정한 결과가 아니다.

**표본 잡음이 작지 않다.** [추가 검사](data/reconstruction_gap_particle_audit_20261003/RESULTS.md)는
위 패널 중 같은16개 상태에서 정책은 그대로 두고 particle만64개에서512개로 늘렸다.
1M 평균전략 gap은 raw30.3629->13.6816, latent29.2837->14.2625로 변했다.
128개 전체 평균과16개 부분 평균을 직접 비교한 숫자가 아니다. 유한 표본에서 max를
선택하는 평가의 낙관 편향/분산 때문에 낮은 표본 수의 gap 절댓값을 확정값으로 읽으면 안 된다.
512개 검사도16개 상태와 한 학습 seed뿐이므로 확정적인 수렴/열위 판정은 피한다.
교차 평가한 one-step deviation은 별도 보조 지표이며 정확 gap의 편향 보정값은 아니다.

두 실행에서 총983,040개 rollout,5,075,040개 정책 조회를 수행했으며 누락 partition은0개였다.
카드/관측/원환경 진행/강제 행동 테스트를 포함한16개 테스트가 통과했고, 실행 후에도
평가 코드와 모델/checkpoint hash가 일치했다. 분할 학습은 여전히 미실행이다.
