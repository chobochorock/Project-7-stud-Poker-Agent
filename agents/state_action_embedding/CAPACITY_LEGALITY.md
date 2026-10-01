# 모델 폭 증가와 소수 핸드 과적합

## 질문과 계획

같은 공부를 더 큰 메모장으로 해보는 것이 크기 비교다. 같은64개의 카드 기록을
계속 반복해서 외울 수 있는지 보는 것이 과적합 검사다. 외운 기록의 점수와 처음 보는
기록의 점수를 나눠서, 외우지도 못한 것인지 일반화하지 못한 것인지 구분한다.

사용자가 두 실험의 실행을 요청했다. 기존 GRU/Transformer 모델과 감독학습 손실을
그대로 확장한다. 새로운 알고리즘, CFR 또는 abstraction 실험이 아니다.

## 고정 조건

- Custom seven-poker v3,2players, hand마다1000chips/ante1의 기존 low-fold10k.
  Fold5/6/7구2.625/6.125/8.75%, showdown51.33%. 새 데이터 생성 없음.
- 기존9000hand train/1000hand validation을 유지한다. 같은 hand의 양쪽 관찰자 분리 금지.
  Raw85D와 고정 S1 skip-gram32D만 사용한다. Random은 이번 원인 진단에서 제외한다.
- 한 사건이 한 입력 token이다. Enum one-hot/표준화 paid/존재 여부를 사용한다.
  Float은 입력에만 있으며 회귀 head, 필드별 출력 head, RoPE는 추가하지 않는다.
- 이전 사건만 읽는 teacher-forced one-step, 미래 attention 차단, truncation 없음.
  Recorded next-kind/subject 및 관찰자에게 가능한 parameter를 label로 사용한다.
  실제 숨은 카드나 덱 순서는 label 생성에도 사용하지 않는다.
- 원래333개 출력 확률 p, legal mass M, legal 내부 q에 대한
  `KL(q||uniform_legal)-log(M)`을 사용한다. 금액을 제외하며 출력 legal mask는 없다.
  같은 Adam0.001, batch32, grad clip1, CPU4threads, dropout0.

## 두 실험

| 조건 | 학습 데이터 | 크기 | 예산 | seed |
| --- | --- | --- | --- | --- |
| full | 기존9000hands | small / wide | 1000updates | 11,22,33 |
| overfit | train에서 고른64hands(128관찰자열) | small / wide | 3000updates | 11 |

각각 GRU/Transformer x raw/S1이다. Full small12개는 이전 checkpoint를 읽고
동일 검증 점수 및 기존 조건부 카드 진단을 재현한 뒤 사용한다. 새 학습은 full wide12개와
overfit8개, 총20개다. Tiny는 한 seed의 진단이며 통계적 우위 주장에 사용하지 않는다.
Small 대 wide는 각 regime에서 같은 업데이트/배치/실제 사건 노출을 갖는다.
Full1000 대 overfit3000은 데이터 크기만의 통제 비교가 아니다. Overfit 가능성 검사다.

64hand는 `default_rng(seed+500000)`으로 기존 training hand ID에서 비복원 추출한다.
ID를 config에 기록한다. Full의 train 지표는 이64hand 패널이고, overfit에서는 실제
훈련 전체다. Validation curve는 고정128hand, 최종 평가는 기존1000hand 전체다.
평가로 early stopping/seed 교체/best checkpoint 선택은 하지 않는다.

## 크기와 초기화

- GRU: input projection32 ->128, hidden64 ->256,1층 유지.
- Transformer: width32 ->128, FF64 ->256,2블록/4heads 유지, hidden64 ->256.
  고정 sin/cos 절대 위치 표현, pre-LayerNorm을 그대로 사용한다. PE의 차원만 커진다.
- 같은 size/입력/seed에서 두 구조의 input projection과 출력 head 초기값을 맞춘다.
  Core는 별도 고정 seed로 초기화한다. Small 초기화는 기존 실행과 같다.
  Size 간 파라미터/계산량은 같지 않다. 정확한 수는 결과에 기록한다.
- 64hand 학습에서도 입력 정규화와 S1 사전학습은 원래9000hand 기반으로 고정한다.
  이는64hand만으로 처음부터 representation까지 배우는 실험이 아니라, 같은 전처리와
  고정 표현 위에서 sequence model을64hand label로 학습시키는 진단이다.

## 지표와 저장

- Train/validation 각각 visible-card 조건부 불법확률: 올바른 kind/recipient를
  조건으로 했을 때 이미 관측한 카드에 배정한 확률. Masked deal은 제외한다.
  해당 split 자체의52종 균등 기준선을 반드시 함께 보고한다.
- 합법 내부 KL, joint BET 불법확률, actor accuracy, 전체 합성 loss도 기록한다.
  Train card error가 크게 줄지만 validation이 유지되면 일반화 실패의 증거다.
  둘 다 개선되지 않으면 용량만으로 해결된다는 주장을 할 수 없다.
- 각 모델의 curve/metrics/checkpoint(optimizer,RNG,고정 encoder 포함)를 보관한다.
  자동 resume CLI는 없으며 큰 validation latent 배열은 이번에는 내보내지 않는다.
- RSS는 같은 프로세스/공통 데이터/allocator를 포함하며 순서 효과가 있다.
  모델 단독 최대 메모리 비교가 아니다. 새 모델 시간과 재사용 baseline의 과거 시간을 구분한다.
- 기존 validation을 재사용했다. 독립 test/다른 데이터 정책/자유 rollout은 미검증이다.
  Low-fold 정책 편향, 약52장 균등에 가까운 쉬운 해와 실제 카드 제외 규칙을 구분한다.
  CFR/LBR/exploitability/abstraction interval은 변경하거나 측정하지 않는다.

## 실행

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_capacity_legality agents.state_action_embedding.test_transformer_legality agents.state_action_embedding.test_gru_legality agents.state_action_embedding.test_action_window -v
& $py -m agents.state_action_embedding.capacity_legality --out-dir agents/state_action_embedding/data/capacity_legality_smoke_20260928 --seeds 11 --hands 8 --full-steps 5 --overfit-steps 5 --log-every 5
& $py -m agents.state_action_embedding.capacity_legality --out-dir agents/state_action_embedding/data/capacity_legality_20260928
python -m agents.state_action_embedding.capacity_legality_report --run agents/state_action_embedding/data/capacity_legality_20260928
```

출력 폴더가 이미 있으면 덮어쓰지 않고 중단한다. 모델 복원은 checkpoint의
`history_model`에 따라 `HistoryGRU`, `HistoryTransformer`, `WideGRU`, `WideTransformer`를
선택하고 `input_width`로 생성한 뒤 `model` state dict를 읽는다.
`encoding`에 전처리와 고정 encoder가 있고, wide의 출력 hidden 차원은256이다.
