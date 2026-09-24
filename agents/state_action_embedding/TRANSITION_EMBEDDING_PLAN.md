# Observation / Action 전이 표현 구현 계획

작성: 2026-09-23, Asia/Seoul. **계획 문서이며 아래 모델과 실험은 아직 구현/실행하지 않았다.**
현재 관측은 MLP로 일반화하고, 작은 행동 어휘는 lookup으로 표현한다. 행동 발생 자체가 아니라
비공개 행동 대상만 가려진다는 사용자의 정정을 반영한다. Transformer/RL/CFR 학습은 범위 밖이다.

## 1. 목표와 기존 실험에서의 변경

- `E_o(o)`는 현재 관찰자의 관측만 받는다. History를 몰래 붙이거나 full state를 입력하지 않는다.
- `E_e(e)`는 관측 가능한 단일 이벤트를 표현한다. 플레이어의 베팅과 환경의 chance 모두 포함한다.
- 두 표현을 전이 학습으로 연결한다. Joint training이지 모든 `(state, action)`을 전수 저장한다는 뜻이 아니다.
- 별도 메모리 `m`은 초기 관측과 이후 이벤트를 순서대로 읽는다. `E_o(o)`와 `m`은 구분한다.
- MLP는 필수라는 수학적 주장은 하지 않는다. 이번에는 미관측 입력에 적용 가능한 저비용 기본값으로 채택한다.
- 기존 whole-state SGNS lookup -> MLP distillation은 비교 기준으로 보존한다. 그 teacher vector를
  새로운 실험의 정답으로 사용하지 않는다. 기존 학습 결과의 좌표를 임의로 새 모델과 직접 비교하지 않는다.

## 2. 이벤트의 의미와 관찰자

예를 들어 상대에게 카드가 배분되었다면 다음처럼 표현한다.

```text
kind=private_deal, source=chance, recipient=opponent, target=MASK
kind=bet, source=opponent, bet=CALL, paid=20, target=N/A
kind=private_deal, source=chance, recipient=self, target=visible_card
```

고정 관찰자 기준 self/opponent/chance를 사용한다. 행동자가 상대라고 상대 시점 관측으로 바꾸지 않는다.
기존 event의 `subject`는 deal에서는 수령인이고 bet에서는 행동자다. 무조건 actor로 해석하면 안 된다.
현재 `kind`와 `subject`에서 source/recipient를 파생할 수 있으므로 엔진 이벤트 형식을 전면 교체하지 않는다.

대상은 `VISIBLE`, `MASKED`, `NOT_APPLICABLE`을 구분한다. 기존 `card=0`의 의미는 kind로 복원한다.
`MASKED`는 비공개 정보 표시이고 legal-action mask는 불가능한 선택을 배제하는 별도 정보다.
카드 식별자, 베팅 종류, 상대적 주체는 유한 lookup을 사용하고 금액은 정규화한 scalar로 처리한다.
베팅 종류가 유한하다는 이유로 모든 금액/역사까지 하나의 ID vocabulary로 만들지 않는다.

## 3. 숨겨진 대상: 두 가지 주 비교

공개 종류/주체/구간/금액을 합친 표현을 `u_base`라 하자. 카드 lookup도 32D로 둔다.
고정 lookup은 행동 종류/주체/카드 ID에 대한 것이다. B의 최종 표현은 후보 분포에 의존하는
`E_e(e; b_t)`이지 문맥과 무관한 `E_a(a)`가 아니다. 모든 가능한 belief를 lookup ID로 열거하지 않는다.

$$
u_A=u_{base}+E_C(\mathrm{MASK}),\qquad
u_B=u_{base}+E_C(\mathrm{MASK})+\sum_{c\in\mathcal C_t} b_t(c)E_C(c).
$$

공개 카드이면 양쪽 모두 해당 카드 vector를, 대상 없는 이벤트이면 별도 N/A vector를 쓴다.
가중 평균에서도 MASK 표식을 남겨 실제로 확인한 카드와 혼동하지 않는다.

### 첫 가중치

- 참조 prior는 관찰자가 배제할 수 없는 카드 집합에 대한 균등분포다.
- 실제 엔진의 상대 패/남은 deck list를 읽어서 후보에서 제외하지 않는다. 상대에게 이미 갔지만
  관찰자가 모르는 카드는 관찰자 관점에서 다른 비공개 슬롯의 후보일 수 있다.
- `b_t`는 해당 이벤트 시점까지 허용된 정보만 사용한다. 미래 showdown으로 과거 token을 수정하지 않는다.
- 임의 정책 아래에서 균등분포가 정확한 posterior라고 주장하지 않는다. Discard/reveal 선택이나
  카드 기반 상대 베팅은 posterior를 바꾼다. 초기 데이터의 무작위 discard/reveal 및 카드 비의존 베팅과
  이후 heuristic 정책 데이터를 구분한다. 정확성이 필요한 posterior는 별도 정책/추론 모델을 요구한다.
- 후보 support/정규화/알려진 카드 제외를 검사하고, 불가능한 빈 support를 임의 uniform으로 덮지 않는다.

### 범위와 한계

첫 구현은 **대상 카드 한 장의 marginal**이다. 상대 손패 조합 전체나 모든 deck 순서를 열거하지 않는다.
복수 비공개 카드 사이의 비복원 추출 상관관계를 이 평균 하나가 보존하지는 않는다. 특히 32D 평균은
52개 카드의 모든 가능한 확률분포를 일대일로 표현하지 못한다.

또한 비선형 전이에서는 일반적으로 다음 두 계산이 다르다.

$$
T(z,\mathbb E[E_C(C)])\ne\mathbb E[T(z,E_C(C))].
$$

따라서 B는 정확한 belief filter가 아닌 표현 선택이다. 이후 필요하면 카드별 **출력 분포의 mixture**나
제한된 particle 방식과 비교한다. 처음부터 52번 전이를 계산하거나 전체 hand posterior를 만들지 않는다.
룰로 계산한 후보 prior는 명시적 inductive bias다. 이득이 발견되면 같은 52D prior를 별도 입력으로 주는
`MASK + prior features` 대조군을 추가해, 정보 제공 자체와 평균 embedding 방식의 효과를 분리한다.

## 4. 모델과 순서

```text
현재 관측 235D -> MLP(128, 128) -> z:32D
공개 이벤트 필드 -> 작은 lookup들의 합 + 금액 선형 투영 -> u:32D
[메모리 m:32D, 이벤트 u:32D] -> 공유 MLP(128, 128) -> 다음 메모리:32D
메모리 -> 관측 표현 readout P -> z_hat:32D
관측 표현 -> 관측 복원 head D -> 카드/공개 필드 예측
```

첫 버전은 `m_0=E_o(o_5)`로 초기화한다. 이후에는 `m_{j+1}=U(m_j,E_e(e_j))`를 순서대로 적용한다.
한 구간씩 현재 관측으로 다시 초기화하는 실험과, 5구부터 재초기화 없이 진행하는 실험을 구분한다.
후자는 누적 오차를 측정한다. 원본 관측으로 중간에 몰래 교정한 뒤 순수 event rollout이라 부르지 않는다.

행동을 더하는 고정 delta 모델은 간단한 비교군으로만 둔다. 공유 비선형 U의 합성은 순서에 따라 달라질 수
있으므로 초기 구현에 positional embedding이나 Transformer는 필요하지 않다. 실제로 순서가 의미 있는
합법 사례에서 달라지는지는 검사한다. 모든 행동 순열의 결과가 반드시 달라야 하는 것은 아니다.
`P(m)`만 관측 표현에 맞춘다. 메모리 전체를 관측과 같게 강제하여 과거 정보를 지우지 않는다.

## 5. 데이터와 시점 계약

재사용할 자료는 [기존 10k 데이터](data/separate_20260922/dataset.json)다.
환경은 표준 casino Stud가 아닌 custom seven-poker v3, 2인 cash, hand별 1000 chips/ante 1이다.
수집 seed 20260922, train 9000 / validation 1000 hands를 유지한다. 양쪽 관찰자는 같은 split에 둔다.
5/6/7구 fold 확률 2.625%/6.125%/8.75%, 나머지 합법 행동 균등, discard/reveal 무작위다.
기존 측정 showdown 비율은 51.33%다. 새 모델의 성능 측정 결과는 아니다.

현재 [EventGame](action2vec.py)은 전체 이벤트와 **베팅 직전 관측**을 기록한다. 저장된 corpus에는
각 원자 이벤트 직후의 전체 관측이 없다. 첫 학습 단위는 정확히 다음이다.

$$
(o_t,\ [e_{j_t+1},\ldots,e_{j_{t+1}}],\ o_{t+1}).
$$

`j_t`는 관측이 찍힌 TURN 위치다. 각 TURN과 state row를 대응시키고, count/actor/street/다음 BET가
`betting_action`과 맞는지 검사한다. 배열에 별도 인덱스를 추가하더라도 원본 NPZ는 덮어쓰지 않는다.
한 묶음에 베팅, street 변경, 여러 딜이 있을 수 있다. 이를 `(o, 단일 action, o')`라 부르지 않는다.
반대로 중간에 딜 없이 BET와 다음 TURN만 있는 구간은 별도 평가할 수 있다.

기존 hook 일부는 실제 엔진이 카드 배분 등을 한꺼번에 끝낸 뒤 개별 event를 emit한다. 그때마다
엔진 snapshot을 찍으면 뒤 이벤트의 정보가 앞 이벤트 target에 섞일 수 있다. 원자 전이가 필요해지면
기존 `reconstruct_prefix`를 확장하여 관찰자 event reducer를 만들거나 실제 원자 변경 위치에 hook을 둔다.
첫 버전은 이 작업을 건너뛰고 시점이 검증된 decision-boundary 묶음을 사용한다.

마지막 베팅 뒤 next snapshot이 없으면 전이 loss에서 제외한다. Terminal return은 별도 평가 label이며
입력에 붙이지 않는다. Terminal 공개/정산 event를 이전 관측 복원에 쓰지 않는다.
일단 기존 validation은 pilot 용도다. 최종 설정을 정한 뒤 seed 20260924의 독립 1000 hands를 test로
추가 생성할 계획이며 아직 생성하지 않았다. 같은 hand를 새 이름으로 test에 재분류하지 않는다.

## 6. 첫 학습 목적

우선 **이미 발생한 관측 이벤트를 받아 다음 관측을 갱신하는** 문제를 푼다. 미발생 chance의 실제
카드를 맞히는 문제와 다르다. 공개된 chance 결과는 입력하고, 상대 hidden card는 정답으로도 넣지 않는다.
별도 미래 예측을 추가할 경우에만 미관측 확률적 결과의 분포/NLL을 다룬다.

$$
\hat m'=U^{[e]}(E_o(o)),\quad\hat z'=P(\hat m'),
$$
$$
L=L_{obs}(D(\hat z'),o')
  +\lambda_{rec}L_{obs}(D(E_o(o)),o)
  +\lambda_{align}\|\hat z'-\operatorname{sg}(\bar E_o(o'))\|_2^2.
$$

- `L_obs`: 카드 multi-hot의 그룹별 균형 BCE와 공개 scalar의 정규화 회귀, 범주형 필드의 분류.
  카드 대부분이 0이라는 이유로 아무 카드도 없다고 예측하여 좋은 점수를 받지 않도록 별도 지표를 둔다.
- `bar E_o`: 천천히 갱신하는 target encoder. Stop-gradient/EMA만으로 collapse가 방지된다고 보지 않는다.
  현재 관측 복원과 표현 분산/effective-rank 검사를 함께 사용한다.
- 베팅으로 변한 공개 금액/합법 행동과 chance로 추가된 카드를 따로 집계한다. 고정 카드 복사만으로
  높은 전체 정확도를 내는 것을 전이 이해로 해석하지 않는다.
- 카드 중요도, 칩 중요도 등 loss 비율은 명시적 설계 선택이다. Train 통계로 정규화하고 validation으로만
  작은 범위를 선택한다. Return/족보를 훈련 label로 넣으면 그 실험은 별도 supervised 보조학습으로 표시한다.
- Lookup, MLP, 전이 모델 모두 일반 autodiff로 학습한다. 차원별 기여량으로 수동 gradient를 배분하지 않는다.

초기 pilot 제안: dimension 32, seeds 11/22/33, Adam lr 0.001, batch 256 decision 구간,
각 5000 optimizer steps, lambda_rec=lambda_align=1, target EMA 0.99. 동일 field 평균으로 loss 규모를
정리하고, 200-step 진단에서 NaN/collapse/복원 실패를 검사한다. 이 값은 미검증 시작값이지 최적값이 아니다.
MASK/weighted-MASK는 공통 가중치 초기값과 배치 순서를 맞춘다. Optimizer steps 외에 실제 event 수,
CPU/GPU 종류, wall time, peak RAM/VRAM, 파라미터 수와 prior 계산 시간도 보고한다.

## 7. 사용자가 제안한 네 방법론의 위치

| 제안 | 구현 순서와 수정 |
|---|---|
| 같은 좌우 관측 사이의 대체 중간 관측 | 기본 전이 모델 뒤, 합법적인 소규모 분기 표본에서 검사. 모든 중간 상태 탐색은 하지 않는다. 우연히 endpoint 한 번이 같다는 사실만으로 전역 positive label을 만들지 않는다. |
| 같은 시작/끝을 만드는 대체 행동 | 같은 현재 관측에 조건부인 행동 효과 유사도로 평가. 같은 bet 결과라도 행동자/정보 공개/보상이 다르면 동일 행동으로 합치지 않는다. |
| 길이가 다른 관측/행동열 | 공유 U를 순서대로 합성해 1/2/4개 decision 구간 및 5구 이후 전체 rollout 비교. 합 vector보다 순서 보존을 우선한다. |
| 끝 상태 유사도로 중간 표현 bootstrap | 마지막 단계. 고정된 관측 target과 의미 검사에 통과한 뒤 target encoder의 유사도로 제한된 보조 loss를 추가한다. 초기 random cosine은 의미 label이 아니다. |

별도 branching 실험에서는 simulator의 단일 hidden world 결과를 정보집합 전체의 동치라고 부르지 않는다.
가능한 hidden/chance 조건을 제한적으로 표본화하고 표본 수와 조건을 기록한다. 상대의 후속 행동에 대한
모델이 필요하면 규칙 전이와 행동 정책을 분리한다. 관측되지 않은 edge를 불가능한 negative로 판정하지 않는다.

## 8. 평가: 반드시 포함할 검사

### 정확성과 누출

1. 동일하게 허용된 관측 이력을 유지한 채 상대 hidden cards/deck order만 바꿔도 현재 z, u, prior가 동일한지 검사.
2. 한 prefix만 입력한 결과가 전체 sequence의 같은 시점 결과와 같은지 검사. 미래 reveal/보상을 읽지 않는다.
3. MASK/N/A/실제 카드 구별, 카드 중복/칩 보존, prefix 복원, TURN-state 대응, hand split 무결성 검사.
4. 복수-step 누적 오차, 변경 필드 성능, 카드 수/카드 그룹별 precision-recall, scalar MAE를 따로 보고.

### 요인 제거/삽입 및 의미

1. **전역 suit 치환:** 관련 카드/이벤트/prior를 함께 바꾼다. 실제 규칙상 대칭인 부분에서는 족보/보상
   probe의 일관성을 검사한다. 입력 특정 suit가 판정/순서의 tie-break에 쓰이면 먼저 그 예외를 분리한다.
   Vector 자체의 동일성은 불변성 학습을 넣었을 때만 요구하고, 그렇지 않으면 equivariance도 허용한다.
2. **의미 있는 한 장 편집:** 합법적인 5/7장 구성에서 suit 한 장을 바꿔 flush를 만들거나 깨고,
   frozen linear probe가 족보 변화에 반응하는지 평가한다. 모두 다른 flush를 같은 점으로 강제하지 않는다.
3. **요인 vector 산술:** train 편집쌍으로만 delta를 추정하고, held-out 구성의 `E_o(o)+delta`가
   실제 편집 후 `E_o(o_edit)`를 찾아내는지 top-k/정규화 오차를 측정한다. Test 정답으로 delta를 만들지 않는다.
4. 비교 기준은 raw one-hot/기존 raw features, random MLP, 편집 없는 vector와 동일 차원의 학습 표현이다.
   카드 rank 조합이 겹치지 않는 별도 편집 split을 둔다. 단일 관측의 synthetic 검사를 유효한 history라고
   꾸미지 않으며, sequence 검사는 엔진으로 합법 prefix까지 생성한다.

### Action 표현과 응용

- 같은 action을 서로 다른 o에서 적용할 때의 효과와, 같은 o에서 다른 합법 action의 효과를 비교한다.
  카드/칩의 영향을 제거한 raw action lookup cosine만으로 의미 품질을 판단하지 않는다.
- Action을 제거한 전이 모델, 다음 관측=현재 관측인 no-change baseline, 규칙으로 계산한 관측 갱신을 비교한다.
  규칙 계산은 신경망 학습의 효용을 판단할 기준이며 학습된 모델로 가장하지 않는다.
- Return 예측은 terminal 전의 prefix에 frozen probe로 별도 측정한다. 다중 정책 평가는 후속에 추가하고
  어떤 정책을 train/test에 썼는지 분리한다. 현재 무작위 데이터에서 전략 이해를 입증했다고 하지 않는다.
- Weighted-MASK가 숨은 카드에 관한 좋은 신호인지 주장하려면 미래의 관측 가능한 결과에 대한 분포
  예측도 별도로 필요하다. 단순 관측 복원만으로 posterior 품질을 검증하지 못한다.
- LBR/abstraction은 표현의 일차 검사 통과 뒤 같은 bucket 수/학습 예산에서 평가한다.
  Return probe, reconstruction, Local BR-gap, LBR profit, exploitability는 서로 다른 지표다.

## 9. 구현 파일과 실행 순서

기존 family 내부만 사용한다. 새 프레임워크/패키지나 게임 엔진은 만들지 않는다.

- 예정 `transition_embedding.py`: 기존 recorder/raw encoder 재사용, 데이터 정렬, 작은 모델과 train/evaluate.
- 예정 `test_transition_embedding.py`: 위 시점/정보누출/합법 편집/저장 복원 검사의 최소 집합.
- 현재 문서: 구현 시 실행 가능한 정확한 커맨드와 모델/데이터 schema를 추가한다. 지금은 없는 CLI를 제시하지 않는다.
- 예정 `data/transition_<run>/`: 원본 참조 manifest, 파생 NPZ, 설정, metrics/곡선, checkpoints.
- Checkpoint는 inference용 weights/schema와 별도로 optimizer/target encoder/RNG/step/split hash를
  저장해 실제 resume를 지원한다. 이전 실험의 inference export가 resume 가능했다고 소급하여 쓰지 않는다.

순서는 **데이터 시점 및 masking 검사 -> MASK 전이 모델 -> 가중 평균 비교 -> 요인 편집/누적 오차 ->
분기 기반 유사도와 bootstrap**이다. 첫 단계가 실패하면 데이터 계약부터 수정한다. 의미 검사에서 이득이
없으면 모델 크기나 CFR 실험을 확대하기 전에 어떤 정보를 복원/일반화하지 못했는지 보고한다.

## 10. 관련 근거와 범위

[CLOUD](https://proceedings.mlr.press/v155/wang21c.html)는 무작위 탐색 자료로 state/action 공간의
forward/inverse dynamics를 대조 학습하는 관련 연구다. 이번 계획과 동일한 poker masking 실험은 아니다.
[DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html)는 reward와 다음 latent-state 분포를 통한
표현 학습의 참고다. 현재 local observation이 Markov하다고 가정하거나 그 정리를 그대로 적용하지 않는다.
이 계획의 주장은 실험으로 검증할 가설이며, 좋은 embedding이나 NE 수렴의 보장은 아니다.
