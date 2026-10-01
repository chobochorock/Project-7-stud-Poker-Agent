# 행동 카드의 앞뒤 맞히기

## 다섯 살에게 설명한다면

게임에서 있었던 일을 그림 카드처럼 줄줄이 놓았다고 생각해 보자.
카드에는 "누가", "무엇을 했는지", "얼마를 냈는지"가 적혀 있어.

우리는 작은 로봇 여섯 개에게 **카드 한 장만** 보여 줄 거야.

| 로봇 | 맞혀 볼 것 |
| --- | --- |
| L1 | 바로 앞 카드 |
| L2 | 앞에 있는 두 카드 중 하나 |
| R1 | 바로 뒤 카드 |
| R2 | 뒤에 있는 두 카드 중 하나 |
| S1 | 바로 앞이나 바로 뒤 카드 |
| S2 | 앞뒤 두 칸 안에 있는 카드 중 하나 |

로봇은 카드의 내용을 숫자 32개로 된 작은 메모로 바꾸고, 그 메모로 답해.
두 칸을 보는 로봇에게 공부 시간을 더 주지는 않을 거야.

채점 도장은 세 가지야.

1. **기억 도장:** 실제 기록에 있던 행동을 맞혔니?
2. **규칙 도장:** 그때 할 수 있는 행동을 말했니?
3. **여러 답 도장:** 가능한 행동을 하나만 말하지 않고 여러 개 떠올렸니?

기록에는 CALL이라고 쓰여 있어도 HALF가 가능한 때라면 규칙 도장은 받을 수 있어.
돈의 숫자가 조금 다르다고 규칙 도장을 빼앗지는 않아.
하지만 항상 "포기!"만 말하면 규칙은 지킬 수 있으니, 그것만으로 잘했다고 하지는 않아.
답을 틀려도 어른이 먼저 고쳐 주지 않고, **로봇이 처음 말한 답**을 채점해.

마지막에는 모든 로봇에게 같은 추가 문제도 내서 공평하게 비교할 거야.
이건 게임에서 이기는 공부가 아니라, 작은 메모가 쓸모 있는지 알아보는 놀이야.

## 고정한 실험 계약

- 기존 custom seven-poker v3 low-fold 10k corpus를 읽기만 한다. 표준 카지노 Stud가 아니다.
  Train9000/validation1000 hand, 양쪽 관찰자는 같은 split. Fold5/6/7구는
  2.625/6.125/8.75%, 실제 showdown51.33%. 카드 강도에 따른 전략은 없다.
- ANTE/PRIVATE/PUBLIC/DISCARD/REVEAL/BET 유지, bookkeeping START/STREET/TURN 제외.
  Window 거리는 이 사건 단위다. Chance를 건너뛰어 BET만 세지 않는다.
- L1={-1}, L2={-2,-1}, R1={1}, R2={1,2}, S1={-1,1}, S2={-2,-1,1,2}.
  동일 관찰자/hand 내 +/-2가 존재하는 공통 center만 사용한다. 경계 제외 수를 기록한다.
  Center는 token-occurrence 균등, window의 offset도 균등이다. 거리 할인은 없다.
- 단일 현재 사건 -> typed raw85D -> MLP128/ReLU/128/ReLU/32 -> 필드별 linear heads.
  Actor/recipient를 분리한다. ANTE/DEAL의 actor는 environment, recipient는 기존 subject.
  DISCARD/REVEAL/BET의 actor는 기존 subject, recipient는 N/A다.
  Input enum: kind6, actor3, recipient3, card53(MASK 포함), bet9(N/A 포함), street7.
  숫자1(paid)과 card 존재/가시성/paid 존재3개 bit를 붙인다.
- Enum별 CE, 존재하는 paid의 train-only 표준화 MSE를 head별 평균하여 더한다.
  Card 정답은 공개된 identity만 사용한다. 숨은 카드와 없는 필드는 loss에서 제외한다.
  Numeric은 입력과 보조 loss로 유지하되 금액 오차는 합법성/성능 순위에 넣지 않는다.
  원형 lookup SGNS가 아니라 **typed-field skip-gram형 목적**이다.
- 각 조건2500 updates x512 pairs, Adam .001, seed11/22/33, CPU4 threads.
  같은 seed는 encoder/head 초기값과 center 추출열을 공유한다. Offset만 조건별로 다르다.
  매250 updates 고정 panel을 검사하고 마지막 step을 보고한다. Best-val 선택은 하지 않는다.
- 현재 사건 외의 state/legal mask/history/reward는 모델 입력에 없다. Delta를 받지 않는
  head는 window의 주변분포를 예측하며 순서/행동열의 joint distribution을 생성하지 않는다.
- 공통 비교: encoder를 고정하고 train-center 빈도로 표준화한32D에 offset별 선형 head4개를
  1000 updates x512 pairs로 학습한다. Raw85D와 무학습 random32D에도 같은 probe를 붙인다.
  Probe는 네 offset을 균등 추출하며 파라미터 수는 raw와32D에서 다름을 보고한다.
  Frequency 및 현재 kind/actor/street 조건부 빈도 baseline도 train pair로만 추정한다.

## 평가를 잘못하지 않기

- 원본 **target 사건 직전 상태**의 합법 행동 집합을 채점에만 쓴다. 예측에 mask를 곱하지
  않고 엔진의 불법행동 자동 대체도 거치지 않는다. Numeric 회귀값을 행동 금액으로 넣지 않는다.
- BET target에서 종류-level top1 accuracy/NLL, Valid@1, LegalMass, LegalRecall@3 및
  Precision@3을 계산한다. Kind/actor 정확도와 kind+actor+valid 결합 성공률도 따로 보고한다.
  LegalMass는 unmasked 확률의 합법 집합 합이며 sampling의 기대 합법률이다.
  Top1/Top3 동점은 해당 후보 사이의 균등 추출 기대로 계산한다.
- 실제 target이 BET인 조건부 지표다. Deal/Discard 등의 전체 사건은 kind/actor 등 필드별
  예측 평가에 포함하되, 카드 대상까지 실행 가능한 사건을 생성했다고 주장하지 않는다.
- 항상 FOLD는 현재 규칙에서 Valid@1=100%다. 이 baseline과 정확도/coverage를 함께 본다.
  NLL은 log(0)을 피하기 위해 확률 하한1e-12를 적용한다.
- 모든 최종 validation center/네 offset을 평가한다. Pair 평균과 hand 평균을 구분하고
  train에 등장한/미등장한 입력 tuple, 미관측 입력금액, 범위 밖 입력금액을 나눈다.
  범위는 실제 paid가 있는 train 사건에서만 추정한다. 조합 중복 제거는 인덱싱일 뿐,
  validation을 loss/normalizer/빈도 baseline에 사용하지 않는다.
- 동일 frozen probe 문제가 주 비교다. Direct head의 자기 window 점수는 서로 다른 문제의
  점수이므로 그것만으로 여섯 조건의 우열을 정하지 않는다.
- L 조건은 과거 복원이며, R2의 중간 상태는 원본 기록을 따른다. 새 행동열 전체를 실행하는
  rollout은 아니다. 기존 validation은 개발용으로 반복 사용되었다. 독립 test/다른 정책/
  게임 전체 coverage/승률/LBR/exploitability를 평가한 것이 아니다.
- RAM은 같은 프로세스에서10ms 간격으로 본 working set이다. Allocator 재사용이 있으므로
  조건별 독립 peak 비교로 해석하지 않는다. Pretraining/probe 비용과 시간은 구분한다.

## 실행 및 불러오기

기존 출력 폴더는 덮어쓰지 않는다. 출력과 checkpoint는 이 family의 `data/` 아래에 둔다.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_action_window -v
# 먼저 작은 시험. 학습량을 바꾼 smoke 결과는 본실험 결과가 아니다.
& $py -m agents.state_action_embedding.action_window --out-dir agents/state_action_embedding/data/action_window_smoke_20260927 --seeds 11 --steps 5 --probe-steps 5 --log-every 5
# 본실험: 6조건 x3seed. 원본 데이터 생성은 하지 않는다.
& $py -m agents.state_action_embedding.action_window --out-dir agents/state_action_embedding/data/action_window_20260927
# Matplotlib이 설치된 기본 Python으로 보고서/그래프 생성.
python -m agents.state_action_embedding.action_window_report --out-dir agents/state_action_embedding/data/action_window_20260927
```

`<condition>/seed<N>/model.pt`는 architecture/normalizer/model/optimizer/RNG/step을 저장한다.
`probe.pt`는 frozen representation 위에 학습한 head와 표준화 계수를 저장한다.
Optimizer/RNG를 남기지만 자동 resume CLI는 이번 범위에 넣지 않았다.

```python
from agents.state_action_embedding.action_window import load_model, field_arrays
from agents.state_action_embedding.experiment import torch

model, normalizer = load_model('agents/state_action_embedding/data/action_window_20260927/R1/seed11/model.pt')
# raw_events: shape[N,6], 기존 관찰자 기준 event tuple. 신뢰하는 checkpoint만 연다.
features, _, _ = field_arrays(raw_events, normalizer)
with torch.no_grad():
    z = model.encoder(torch.from_numpy(features))  # [N,32]
    logits = model(torch.from_numpy(features))
```

설계만으로 실행 결과를 주장하지 않는다. 완료 결과는 run 폴더의 `RESULTS.md`를 확인한다.
