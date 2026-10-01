# 같은 데이터로 GRU와 Transformer가 규칙을 배울 수 있는가

## 질문과 출처

사용자 질문은 표현 가능성의 증명이 아니라 **동일한 경험과 학습 예산에서 실제로
학습되는가**다. 기존 GRU 실험의 약점인 '이미 본 카드에 확률을 주는 문제'를 우선 본다.
Transformer/GRU 자체는 기존 기법이며 새로운 모델 발명으로 주장하지 않는다.

관측 사건을 읽는 방법만 바꾸고, [기존 GRU 계약](GRU_LEGALITY.md)의 데이터,
타깃 시점, 관찰자 MASK, 학습/검증 분리, 법칙 label, 손실을 재사용한다.
Raw85D / 고정 random-MLP32D / 고정 S1 skip-gram32D를 각각 비교한다.
Random은 입력별 독립 난수 lookup이 아니라 같은 고정 무작위 MLP를 통과한 표현이다.

## 어린이에게 설명한다면

GRU는 사건을 읽으며 작은 메모를 계속 고친다. 이번 Transformer는 지금까지 읽은
사건 메모들 중 필요한 것을 다시 살펴볼 수 있다. 둘에게 같은 카드 기록과 같은
공부 시간을 주고, 이미 나온 카드와 다음 차례를 잘 기억하는지 시험한다.
아직 나오지 않은 미래 카드에는 가림막을 씌운다. 미래 카드를 바꿔도 과거 답이
바뀌지 않는지 검사한 뒤 공부를 시작한다.

## 고정 조건

- 기존 custom seven-poker v3 low-fold10k: train9000/validation1000hands, 동일 양쪽 관찰자 split.
- Fold5/6/7구2.625/6.125/8.75%, showdown51.33%. 데이터 정책 편향은 그대로 유지된다.
- Prefix e[0:t] -> 다음 사건 e[t+1], 최종 END 포함. 초기 사건 하나의 예측은 제외한다.
- 실제 기록을 prefix로 주는 teacher-forced one-step, 원본 전체 이력 사용, truncation 없음.
- 다음 kind/subject는 recorded schedule label, BET는 engine mask, 카드 후보는 관측과
  양립 가능한 집합이다. 숨은 실제 패/덱 순서를 사용하지 않는다. 독립 rollout 검사는 아니다.
- `M=합법확률합`, `q=p/M on legal`, 손실은 `KL(q||uniform_legal)-log(M)`.
  Natural log/nats, penalty가중치1. 금액은 입력에만 있고 출력 평가에 없다.
- Seed11/22/33, 각1000updates x32 observer-hand sequences, Adam.001, grad clip1,
  CPU4threads. 실제 유효 event exposures도 짝마다 같은지 검사한다. 같은 샘플링 RNG다.
- Fixed128 validation hands의200step 곡선, 마지막은 전체1000hands. Best-val 선택 없음.

## 바뀌는 구조

| 항목 | GRU | Causal Transformer |
| --- | --- | --- |
| 공통 입력 | train-only 표준화 -> Linear to32 -> tanh | 동일 |
| 이력 처리 | GRU1층, hidden64 | 2블록, model32, head4, FF64, ReLU |
| 정규화 | 기본 GRU gate | pre-LayerNorm, 마지막 LayerNorm |
| 위치 | recurrent 순서 | 고정 sinusoidal32D, 최대128사건 |
| prefix 출력 | hidden64 | Linear32->64 + tanh |
| 공통 head | Linear64->333 | 동일 |
| dropout | 0 | 0 |
| trainable 파라미터(raw) | 43,213 | 43,661 |
| trainable 파라미터(random/S1) | 41,517 | 41,965 |

차이는448개(약1%)다. 완전히 같은 모델 용량이나 FLOPs를 보장하지는 않는다.
Transformer 블록은 각각 독립 초기화한다. Input projection과 최종 head의 초기값은
같은 조건/seed의 GRU와 정확히 일치시킨다. 서로 다른 core의 초기값 자체는 같을 수 없다.
Transformer의 positional encoding, normalization, attention/FFN을 합친 구조 비교이지
attention 연산 하나만의 인과 효과를 분리한 실험은 아니다.

오른쪽 padding은 모든 유효 query의 미래에 있으므로 causal attention으로 가린다.
유효 위치만 loss/평가에 포함한다. 미래 사건 변경/오른쪽 padding 및 prefix만 실행한
결과의 일관성을 train/eval 경로 모두 검사한다. 과거 출력의 미래 입력 gradient도0인지 검사한다.
추론 때 legal mask를 출력에 곱하지 않는다. Attention의 causal mask와 action legal mask는 다르다.

## 비교와 자원 기록

새 run에서 GRU9개도 재실행한다. 이전 GRU 모델의 학습 가중치와 bit 단위로 일치하는지
확인하여 공통 학습기 확장으로 기준선이 변하지 않았는지 검증한다. 과거 run은 수정하지 않는다.
이후 Transformer9개를 같은 budget으로 실행한다. Seed마다 순서를 번갈아 바꾼다.
Hyperparameter 탐색이나 실패 seed 교체는 하지 않는다.

매 모델 wall time 및10ms 간격 process RSS를 기록한다. 같은 프로세스의 allocator/cache,
공통 데이터 메모리가 포함되므로 RSS는 **모델 단독 메모리 또는 독립 peak 비교가 아니다**.
훈련 파라미터/초기 RSS/최고 RSS/증가량을 함께 남기며 이 측정으로 구조별 메모리 우위를
단정하지 않는다. 과거149.68초와 새 전체 시간은 작업 범위도 달라 직접 비교하지 않는다.

관측 가능 딜은 joint 불법확률과 '올바른 kind/recipient 조건부' 카드 불법확률을 분리한다.
후자에서52종 균등 기준선보다 낮아지는지가 카드 identity 기억의 주 검사다.
BET KL, joint 불법확률, 다음 actor 정확도도 보고한다. Mean +/-3seed SD 및 같은 seed
Transformer-GRU 차이를 보고한다. SD는 신뢰구간이 아니다.

`--inspect`의 마지막 사건만 남기는 검사는 추가 진단이다. Transformer에서는 이력을
없애면서 위치도0으로 재설정하므로 GRU의 memory 초기화와 완전히 같은 개입은 아니다.
별도로 학습한 no-history 비교로 해석하지 않는다.

## 실행

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_transformer_legality agents.state_action_embedding.test_gru_legality -v
& $py -m agents.state_action_embedding.transformer_legality --out-dir agents/state_action_embedding/data/transformer_legality_smoke_20260928 --seeds 11 --steps 5 --log-every 5
& $py -m agents.state_action_embedding.transformer_legality --out-dir agents/state_action_embedding/data/transformer_legality_20260928
& $py -m agents.state_action_embedding.transformer_legality --out-dir agents/state_action_embedding/data/transformer_legality_20260928 --inspect
python -m agents.state_action_embedding.transformer_legality_report --run agents/state_action_embedding/data/transformer_legality_20260928
```

Run 아래 `gru/`, `transformer/` 각각의 checkpoint/curve/metrics/64D validation latents를 저장한다.
기존 run은 덮어쓰지 않으며 checkpoint에는 고정 encoder와 학습 파라미터,
optimizer/RNG가 포함된다. 자동 resume CLI는 없다.

### 불러오기

기존 GRU 문서의 입력 전처리는 동일하다. Transformer checkpoint이면 모델 클래스만
`HistoryTransformer(checkpoint['input_width'])`로 바꿔 `model` state dict를 불러온다.
관찰된 사건열만 입력하고 마지막 위치의 hidden64와333종 확률을 얻는다.
이 클래스 이름은 checkpoint의 `history_model`에도 저장된다.

## 해석 제한

기존 validation을 재사용한 개발 실험이다. 다른 정책이나 독립 test로의 일반화는 미검증이다.
합법 규칙에 대한 supervised loss이므로 순수 skip-gram/자기지도만의 결과가 아니다.
차이가 없더라도 해당 규모와1000updates의 결과이지 Transformer의 학습 불가능성 증명이 아니다.
반대로 개선되더라도 abstraction/CFR/포커 전략/LBR/exploitability의 개선을 의미하지 않는다.
이번에는 abstraction interval과 다른 프로젝트를 변경하지 않는다.
