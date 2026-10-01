# 합법 분포와 GRU 실험

## 쉬운 설명

앞 실험에서는 사건 카드 한 장으로 주변 카드를 맞혔다.
이번에는 작은 메모장을 가진 GRU가 카드를 처음부터 차례대로 읽는다.
메모를 보고 다음에 누가 무엇을 할 수 있는지 말하게 한다.

- 가능한 답 세 개에 똑같이 확률을 주면 균등 점수는 좋다.
- 하지만 불가능한 답에 확률을 많이 줬다면 별도의 벌점을 받는다.
- 맞힐 답은 기록의 카드 한 장이 아니라 그때 가능한 카드들의 집합이다.
- 규칙을 잘 말하는 것과 포커를 잘하는 것은 다른 시험이다.

사용자의 연구 방향은 action embedding의 규칙/coverage 및 abstraction 유용성 평가다.
GRU는 assistant가 제안한 **기존 기법의 기준선**이다. GRU 자체를 사용자 발명으로 기록하지 않는다.
개념과 수식은 [한국어 설명글](GRU_EXPLAINED_KO.md)에 정리한다.

## 지표와 손실

모델의 mask 적용 전 확률을 p, 합법 집합을 L, 합법 확률합을 M이라 한다.

\[
M=\sum_{a\in L}p(a),\qquad q(a)=p(a)/M\ (a\in L),\qquad u(a)=1/|L|.
\]

사용자 요청 방향을 유지한 지표:

\[
D_{KL}(q\|u)=\sum_{a\in L}q(a)\log(q(a)|L|)
=\log|L|-H(q).
\]

불법 원소는 이 KL에 포함하지 않는다. 대신 아래를 따로 보고한다.

\[
\text{IllegalMass}=1-M,\quad P=-\log M,\quad J=D_{KL}(q\|u)+P.
\]

로그는 자연로그(nats), 가중치1이다. J는 이번 실험에서 정한 합성 손실이지
새로운 표준 KL 또는 논문 지표로 주장하지 않는다. 전체 p와 불법확률0인 u를
직접 KL(p||u)로 비교하면 불법확률이 조금만 있어도 무한대가 되므로 조건부 q를 쓴다.
학습에서는 logsumexp로 계산하여 합법 확률이 매우 작아도 0으로 반올림하지 않는다.
확률이 정확히 M=0인 외부 정책은 조건부 KL이 정의되지 않고 penalty는 무한대다.

예: 합법 CALL/FOLD에 각각0.25, 불법 RAISE에0.5면 KL=0이나
IllegalMass=0.5, P=log(2)다. FOLD에만1이면 IllegalMass=0이나 KL=log(2)다.
단일 합법 행동이면 KL은 항상0이므로 페널티/주체/종류 정확도를 함께 읽는다.
집합 크기에 따라 KL 상한 log|L|가 달라지므로 같은 집합/같은 평가 표본끼리 비교한다.

균등은 coverage 기준이지 좋은 전략이나 실제 chance posterior라는 주장이 아니다.
임베딩 자체는 확률분포가 아니다. 정확히는 **임베딩을 읽는 모델의 출력 분포**를 평가한다.

## 데이터와 시간 경계

- 기존 `data/separate_20260922/separate_raw.npz` 10k hands, train9000/val1000을 재사용.
- Custom seven-poker v3, 2인, 시작1000칩/ante1. 표준 카지노 7-stud가 아니다.
- Fold5/6/7구 2.625/6.125/8.75%, showdown51.33%. 나머지 합법행동 균등,
  discard/reveal 무작위. 이 데이터 정책 편향은 남는다. 새 데이터는 수집하지 않는다.
- 양쪽 관찰자는 같은 hand split. ANTE/PRIVATE/PUBLIC/DISCARD/REVEAL/BET만 유지.
  START/STREET/TURN 안내 토큰은 제외하며 street 값은 각 사건 필드에 남는다.
- 입력 e[0:t]를 읽은 출력이 다음 사건 e[t+1]을 예측한다. 첫 사건을 맞히는 문제는 제외한다.
  마지막 사건 다음에는 END를 예측한다. 전체 이력 사용, 잘라내기 없음, 미래 입력 없음.
- 데이터셋의 전체 핸드가 한 번에 batch에 있어도 단방향 GRU 출력 t는 t 이후를 읽지 않는다.
  실제 prefix를 매번 제공하는 teacher-forced one-step 평가이며 자유 rollout은 아니다.

## 정답 합법 집합과 출력

정적 출력 사전은 `(kind, subject, parameter)` 333종이다. Subject는 관찰자 기준 self/opponent,
deal/ante에서는 recipient이고 행동 주체는 environment다. END는 별도 terminal이다.
BET parameter는 행동 종류, 카드 사건 parameter는 card 또는 MASK다. 금액과 street는
입력에만 포함한다. 금액 회귀 오차는 출력 합법성에 반영하지 않는다.

이 사전은 정적 문법을 이미 준다. 예를 들어 상대 비공개 deal의 원소는 MASK뿐이다.
따라서 비공개 카드 식별을 학습했다고 주장하지 않는다. 현재 상태의 동적 mask는
출력에 곱하지 않으며, 모든333개 logits를 softmax한 원래 분포를 평가한다.

- 다음 kind/subject의 일정은 **실제 다음 기록**을 label로 사용한다. 모델 입력은 아니다.
- BET의 후보는 기존 recorder가 저장한 행동 직전 engine legal mask다.
- 관찰 가능한 새 deal은 앞서 보았던 카드(자신이 버린 카드 포함)를 제외한 모든 카드.
- 상대 비공개 deal/discard는 MASK. 자신의 discard/reveal은 현재 숨은 패 후보.
- 상대 reveal은 관찰 이력과 양립할 수 있는 카드 집합이며 실제 상대 패를 읽지 않는다.
- 실제 정답 카드 자체를 known set에 넣는 시점은 support 생성 **이후**다.
- 지표는 기록된 일정하의 관측 호환성이다. 별도 규칙 엔진으로 임의 생성 이력 전체를
  검사한 것이 아니다. 숨은 실제 덱의 순서/샘플된 상대 패를 기준으로 다른 가능성을 벌주지 않는다.
- 모든 실제 정답이 support에 포함되는지 검사한다. 합법 집합은 학습 label로도 사용하므로
  이번 GRU는 순수 skip-gram이 아니라 **규칙 감독학습**이다.

joint 사건 분포를 사용하므로 next kind/actor/recipient를 맞혀야 합법 확률에 포함된다.
추가로 actor 정확도, kind 정확도, subject 정확도를 주변분포로 계산한다.
BET, visible deal, MASK deal, discard/reveal, END를 나눠 보고한다.
MASK/END처럼 쉬운 singleton이 평균을 높이는 것을 visible deal과 섞어 숨기지 않는다.

## 고정 비교 조건

| 조건 | 입력 전처리 | 학습되는 부분 |
| --- | --- | --- |
| raw | 기존 typed85D | 85->32 projection, GRU, head |
| random | 고정 무작위85->128->128->32 | 32->32 projection, GRU, head |
| skipgram | 기존 S1 seed별 고정32D | random과 동일 |

S1은 양방향 window1이라는 개념적 기본형으로 사용한다. 새 GRU validation으로
6개 사전학습 모델 중 최선을 선택하지 않는다. 모든 encoder의 수치 표준화는 train 사건 빈도로만 계산.
Random/S1의 원래 encoder와 기존 데이터는 변경하지 않는다.

- GRU1층, hidden64, dropout0, 방향1개, trainable output333. 관측 state/legal mask는 입력 아님.
- Adam0.001, gradient norm clip1, 1000updates x32개의 관찰자별 전체 hand sequence.
- Seed11/22/33, CPU4 threads, 마지막 step 평가. Best-validation 선택 없음.
- 같은 seed는 학습 sequence 추출과 GRU/head 초기값 동일. Raw projection은 크기가 달라
  파라미터 수 차이가 있다. Random과 S1은 정확히 같은 trainable 구조다.
- 200step마다 고정128 validation hands의 양쪽 관찰자 평가. 마지막은 전체1000hands.
- 사건 평균과 hand 평균을 모두 저장. 표준편차는3seed 간 SD이며 신뢰구간이 아니다.
- 예산은 updates, sequence exposures, 실제 유효 event exposures 및 wall time으로 기록.
- 기존6window x3seed도 재학습 없이 새 BET KL/불법페널티로 다시 평가한다.
  그 모델은 단일 사건 입력/기존 CE/자기 window인 반면 GRU는 prefix/규칙 손실이므로
  양자의 차이를 GRU 구조만의 인과 효과라고 부르지 않는다.

## 실행과 산출물

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_gru_legality -v
& $py -m agents.state_action_embedding.gru_legality --out-dir agents/state_action_embedding/data/gru_legality_smoke_20260927 --seeds 11 --steps 5 --log-every 5
& $py -m agents.state_action_embedding.gru_legality --out-dir agents/state_action_embedding/data/gru_legality_20260927
& $py -m agents.state_action_embedding.gru_legality_report --run agents/state_action_embedding/data/gru_legality_20260927 --inspect
python -m agents.state_action_embedding.gru_legality_report --run agents/state_action_embedding/data/gru_legality_20260927
```

이미 존재하는 run은 덮어쓰지 않는다. Checkpoint는 projection/GRU/head, 고정 encoder,
normalizer, optimizer/RNG/step을 저장한다. 자동 resume CLI는 없다.
`validation_latents.npz`는 prefix hidden64, hand, 다음 target row, 개별 지표를 담는다.
향후 abstraction 연결을 위한 출력이며 이번에는 clustering/CFR를 실행하지 않는다.
Validation은 이전 연구에서 반복 사용한 개발용 집합이고, 새로운 독립 test가 아니다.
Exploitability/LBR/전략 개선이나 정보집합의 충분통계 보존을 입증하는 실험도 아니다.

`--inspect`는 저장된 모델을 읽기만 하여 조건부 BET/카드 분포 및 마지막 사건만 남긴
진단을 추가한다. 실제 prefix와 달리 memory를 매 사건0으로 초기화하는 입력 교란 검사이며,
새로 학습한 no-history 모델과의 공정한 architecture 비교가 아니다. 전체 정답 이력
중 이미 알려진 kind/subject를 조건으로 한 카드/BET 분포는 joint 오류를 감추므로
joint 지표와 반드시 따로 읽는다. 보고서의 카드52종 균등 기준선은 알려진 카드도
제외하지 않는 기준선이며, 동적 mask로 얻는 완벽한 oracle과 다르다.

### 저장 모델을 실제 사건열에 적용하기

프로젝트 root의 PyTorch Python에서 신뢰하는 checkpoint만 불러온다.
`raw_events`는 관찰자 기준으로 마스킹된 retained 사건들의 `[T,6]` 배열이다.
미래 사건을 붙이지 않고 현재까지의 prefix만 넣는다.

```python
from agents.state_action_embedding.gru_legality import HistoryGRU, ContextModel, torch
from agents.state_action_embedding.action_window import field_arrays

checkpoint = torch.load("agents/state_action_embedding/data/gru_legality_20260927/skipgram/seed11/model.pt",
                        map_location="cpu", weights_only=True)
features, _, _ = field_arrays(raw_events, checkpoint["normalizer"])
model = HistoryGRU(checkpoint["input_width"]).eval()
model.load_state_dict(checkpoint["model"])
encoding = checkpoint["encoding"]
with torch.no_grad():
    x = torch.from_numpy(features)
    if encoding["encoder"] is not None:
        encoder = ContextModel().encoder.eval()
        encoder.load_state_dict(encoding["encoder"])
        x = encoder(x)
    x = ((x.double() - encoding["mean"]) / encoding["std"]).float()
    logits, hidden = model(x[None])
    next_event_probability = logits[0, -1].softmax(-1)
    prefix_embedding = hidden[0, -1]  # 64D; not an exact information set.
```

사건 ID의 의미는 checkpoint의 `symbols` 배열에 있다. 금액 및 street 출력은 없으므로
이 확률만으로 엔진에 즉시 실행 가능한 완전한 사건 객체를 만들었다고 부르지 않는다.
