# State / Action Skip-gram Distillation

현재 관측과 단일 관측 이벤트를 각각 32차원으로 만드는 실험이다. State-action
결합, history encoder, RL/CFR/abstraction 학습은 이번 실험에 넣지 않는다.

## 실행

프로젝트 root에서 기존 Python 3.12 CPU PyTorch 환경을 사용한다.
새 output 디렉터리를 사용해야 하며 기존 corpus/학습 디렉터리를 덮어쓰지 않는다.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/separate_20260922'
& $py -m unittest agents.state_action_embedding.test_separate_skipgram -v
& $py -m agents.state_action_embedding.separate_skipgram collect --out-dir $run --hands 10000 --pilot-hands 300 --seed 20260922
& $py -m agents.state_action_embedding.separate_skipgram train --out-dir $run --seed 11 --teacher-steps 2500 --student-steps 2000
# Default Python has Matplotlib; plotting imports no PyTorch.
python -m agents.state_action_embedding.separate_skipgram_report --out-dir $run
```

`--stream state` 또는 `--stream action`으로 개별 실행할 수 있다. `--dim 32`,
`--hidden 128`, `--batch 1024`, `--window 2`, `--negatives 5`, `--threads 4`가 기본이다.
다른 학습 seed는 같은 corpus에서 별도 `state_seedN`, `action_seedN` 폴더에 저장한다.
공식 본실험의 다중 seed/독립 test는 후속 단계이며 이번 9:1 pilot에 포함되지 않는다.

## 환경과 데이터

- 기존 Python `environments/seven_stud/poker_env.py`와 `EventGame`을 그대로 사용한다.
  환경은 custom seven-poker v3이며 표준 casino seven-card Stud와 동일하지 않다.
- 2인 cash, hand마다 1000 chips로 초기화, ante 1. 5/6/7구에 베팅한다.
- FOLD 확률은 결정마다 3%/7%/10%를 출발점으로 삼는다. FOLD가 아닌 경우 나머지
  합법 행동을 균등 선택한다. Check 가능 시에도 환경이 허용하는 fold가 선택될 수 있다.
- 별도의 최대 6 x 300-hand pilot에서 공통 배율을 조절해 showdown 50%에 접근한다.
  47.5~52.5%에 도달하면 종료한다. Pilot 데이터는 학습/검증에 넣지 않는다.
  배율 선택은 showdown만 사용하며 validation/model 결과를 사용하지 않는다.
- Main 10k에는 거절 샘플링이나 showdown 사후 균형화를 하지 않는다. 따라서 실제
  비율은 50%와 다를 수 있으며 `dataset.json`에 전체/분할별 비율을 보고한다.
- `resolve_showdown()`은 fold 종료에도 호출되므로 호출 횟수로 showdown을 세지 않는다.
  끝까지 fold하지 않은 두 플레이어가 남았는지로 구분한다.
- Discard/reveal은 무작위다. 카드 강도를 고려하는 전략적 behavior가 아니므로,
  깊은 hand와 보상 분포를 확보해도 카드 기반 상대 전략 다양성을 보장하지 않는다.

`separate_raw.npz`는 JSON table이 아닌 압축 배열이다. `state`, `action`, 각각의
`*_offsets`, sequence별 `hand/split/return_chips`, 각 state 이후 `betting_action`,
hand별 `showdown`을 저장한다. 각 hand에는 양쪽 고정 관찰자 sequence가 있으며
둘은 반드시 같은 partition이다. Train 9000 hands / validation 1000 hands.

State는 베팅 결정 직전의 현재 관측 235 features다. 기존 encoder의 action dummy
8개를 제거한다. Own private/public/discard와 상대 public cards, street, 상대적 actor,
공개된 acting-player legal mask와 chips/pot 등의 scalars만 사용한다. History/선택될
action/terminal reward/상대 private cards는 입력이 아니다.

Action은 기존 기록기의 단일 이벤트 6필드(kind, relative subject, visible card,
betting action, street, paid amount)를 83 features로 바꾼다. Start/ante, deal,
discard/reveal, street/turn, bet을 포함한다. 상대 private deal/discard의 card는 UNKNOWN이다.
전체 state를 붙이지 않는다. 현재 관측 event를 표현할 뿐 future event를 입력하지 않는다.
Terminal payout/전체 상대 패 공개는 embedding 입력이나 skip-gram pair에 넣지 않는다.

## 교사와 학생

1. Hand 분할을 먼저 수행한다. Vocabulary와 teacher는 train 데이터만 학습한다.
   전체 raw rows의 중복 제거는 저장/인덱싱 작업이며 validation row는 teacher 학습,
   negative 분포, normalizer, pseudo-target 추정에 사용하지 않는다.
2. State: 같은 관찰자/hand 안에서 다음 1, 2개의 현재 관측을 context로 사용한다.
3. Action: 각 관측 event에서 이후 첫 2개 BET event를 context로 사용한다.
   Chance는 center로 학습하지만 chance card 자체를 예측 표적으로 삼지 않는다.
4. 두 stream에서 각각 SGNS(negative sampling skip-gram)를 학습한다. 이는 대칭
   문장 window가 아닌 고정 forward window 변형이다. 할인/빈도 subsampling은 없다.
5. Train context 빈도의 0.75승에서 negative 5개를 뽑는다. Positive와 정확히 같은
   ID는 제외한다. Negative는 전역 분포이므로 street/actor 규칙만으로 쉬워질 수 있다.
6. Train vocabulary의 center/context vector를 고정된 pseudo-target으로 저장한다.
   이 결과는 객관적 ground truth나 전략적 최적 표현이 아니다.
7. Student는 `features -> 128 ReLU -> 128 ReLU -> 32`이다. SGNS의 두 역할을
   평가하기 위해 shared hidden trunk에 center/context 두 32D head를 둔다.
   사용자가 가져가는 representation은 center head이며 context head는 보조 평가용이다.
8. Train-only feature 표준화와 teacher 차원별 표준화 MSE로 역전파한다.
   Train token 빈도로 샘플링한다. Export는 표준화를 되돌린 teacher 좌표의 32D vector다.
9. Teacher 2500 / student 2000 optimizer steps, batch 1024의 고정 예산이다.
   Teacher는 SparseAdam(lr .02), student는 Adam(lr .001). Validation으로 best checkpoint를
   고르지 않고 마지막 step을 저장한다. 이 예산을 수렴 완료로 해석하지 않는다.

## Validation의 중요한 한계

미관측 상태는 train-only lookup teacher의 vector가 없다. 따라서 validation에
teacher를 미리 학습시키거나 새 teacher를 학습해 좌표별 MSE를 비교하지 않는다.

- Teacher MSE/cosine: train vocabulary에 존재하는 ID만 평가하고 coverage를 함께 보고한다.
  Coverage가 0이면 MSE는 0이 아닌 `null`이다. State에서 이런 경우가 많을 수 있다.
- Novel input: student의 center/context head로 실제 미래 context를 negative와 구분한다.
  고정된 train/val 평가 pair와 negative 후보를 모든 student checkpoint에 재사용한다.
- `sgns_nll_per_term`: positive logistic loss와 negative 5개 loss의 합을 6으로 나눈 값.
- `retrieval_nll`: 6개 후보 softmax에서 실제 context의 NLL.
- `retrieval_top1`: 가장 높은 점수의 후보가 정답인지; 동점은 균등 분배한다.
  Constant score의 값은 1/6이며, 빈도/단계 shortcut을 통제한 전략 지표는 아니다.
- Mean teacher-vector baseline, random initial MLP, train/val gap, embedding variance를 기록한다.
  이 그래프는 reward prediction, LBR profit, exploitability를 나타내지 않는다.
- 거의 모든 state가 한 hand에서만 등장하면 SGNS lookup이 hand별 임의 좌표를 외울 수 있다.
  Student MSE 감소나 train retrieval만으로 새 관측에 의미 있는 표현을 배웠다고 주장하지 않는다.

## 저장과 불러오기

`teacher.pt`와 `vocabulary.npz`는 학습 corpus에 대한 교사 vector와 raw ID 대응이다.
`encoder.pt`는 독립 inference용 architecture/normalizer/weights를 포함한다.
Optimizer/RNG 재개는 구현하지 않았으며 resume checkpoint라고 부르지 않는다.

```python
from agents.state_action_embedding.separate_skipgram import load_encoder

encoder = load_encoder('agents/state_action_embedding/data/separate_20260922/state_seed11/encoder.pt')
# raw_batch: float32 Tensor [batch, 235]; action model uses [batch, 83].
embedding = encoder(raw_batch)  # [batch, 32], no history required
```

## 채택한 변경과 아직 적용하지 않은 대안

채택: 별도 fold calibration, hand-level split, train-only pseudo-target, center/context
보조 head, forward window, chance 예측 제외, SGNS negative sampling, 평균 vector 기준선,
미관측 validation의 context 예측 지표. 10k/32D/2 hidden layers/9:1은 유지한다.

대안(이번에는 미실행): lookup teacher 없이 MLP를 SGNS로 직접 학습, 시간 거리 할인,
대칭 window, stage-matched negatives, masked-feature 복원, forward/inverse dynamics,
서로 다른 정책의 데이터 혼합, reward/족보 frozen probe, 독립 test와 다중 seed.
State lookup vocabulary가 거의 전부 고유하면 데이터/차원만 늘리기보다 direct MLP SGNS가
우선적인 비교 후보다. 일반화/전략 성능을 검증한 후 abstraction 연결을 별도로 진행한다.

원형 참고: [Mikolov et al., 2013](https://arxiv.org/abs/1310.4546).

완료한 첫 실행: [10k-hand 결과와 해석](data/separate_20260922/RESULTS.md).
