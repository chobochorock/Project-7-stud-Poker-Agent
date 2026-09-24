# Action token 및 동일 endpoint 경로 embedding

2026-09-24. 사용자의 두 제안을 실제로 비교하는 소규모 실험이다.
**현재 관측의 전이 효과를 구별하는가**를 묻고, 전략/정보집합/보상 동치를 증명하지 않는다.
결과는 [실행 보고서](data/action_paths_20260924/RESULTS.md)에 별도로 기록한다.

## 기존 실험과 같은 부분/다른 부분

| 항목 | 기존 `separate_skipgram.py` | 이번 `action_path_embedding.py` |
|---|---|---|
| 기본 단위 | 단일 이벤트의 6필드 tuple | 그대로 재사용, 각 tuple이 하나의 token ID |
| Chance | private/public deal 포함 | 포함 |
| 비공개 대상 | 상대 private deal/discard의 card=0 | 그대로 MASK 처리 |
| 주변 context | 이후 첫 두 BET만 | 좌우 거리 1, 2의 실제 action event |
| 시작 상태 | Action stream에 없음 | 없음 |
| 불필요한 bookkeeping | start/street/turn도 center에 포함 | start/street/turn 제외; ante/deal/discard/open/bet 유지 |
| 학습기 | Lookup SGNS 후 MLP에 distillation | SGNS lookup을 직접 사용; action MLP 불필요 |
| 같은 좌우 관측의 경로 학습 | 하지 않았음 | 정확한 관측 일치로 양성 쌍 구성 |

저장 필드는 `(kind, relative_subject, card, bet_kind, street, paid)`다.
문자열 조각으로 나누지 않는다. `DealPrivate(self,h4)` 전체가 한 단위처럼 동작하며,
실제 ID에는 기존 street/paid 등의 필드도 들어간다. DEAL의 subject는 수령인이고 BET에서는 행동자다.
Private/public를 구분하므로 opponent의 public card는 보이고 private card만 MASK다.
본 실험은 MASK 방식만 사용한다. 이전 계획의 weighted-MASK, state MLP, 전이 예측기는 구현하지 않는다.

초반에는 기존 엔진 순서 그대로 두 사람에게 한 장씩 번갈아 4장 딜, discard/open, 4구/5구 딜을 기록한다.
H4 discard/open은 엔진의 batch 처리 순서에 따라 직렬화한다. 동시 선택의 기록 순서를 게임의 인과관계로
과해석하지 않는다. 이후 6구 공개/7구 비공개 딜도 동일한 이벤트 단위를 사용한다.

## 실험 A: Action skip-gram

기존 10k corpus의 train9000/validation1000 hand 분할을 보존한다. 같은 hand의 양쪽 관찰자는 같은 split이다.
수집 정책은 5/6/7구 fold 확률 2.625/6.125/8.75%, 나머지 합법 행동 균등, random discard/reveal다.
실제 showdown 51.33%였으며 카드 강도를 반영하는 정책이 아니다.

주변 토큰이 함께 나타나는 빈도를 학습한다. 예컨대 `Discard(self,h4)` 주변에서 특정 Open/Deal 이벤트가
관측되었다면 그 center-context 조합의 내적을 키운다. 두 사건의 효과가 같다는 label은 아니다.
SGNS는 실제 이웃 하나와 무작위 비교 토큰 5개를 사용한다. 비교 분포는 train context 빈도의 0.75승,
정답과 완전히 동일한 token ID는 negative에서 제외한다. Window는 대칭 2, 출력은 center32D다.
이번에는 deal도 주변 context label이다. 이는 이전의 '미래 BET만 예측' 설정에서 명시적으로 바꾼 부분이다.
Chance의 무작위 카드 결과를 정책이 정확히 예측할 수 있다는 주장은 하지 않는다.

평가는 두 종류의 6후보 중 실제 이웃 찾기다. Global은 train 빈도 기반 negative, matched는 정답과
종류/상대적 주체/street가 같은 negative다. 후자는 종류나 구간만으로 맞히는 쉬운 문제를 줄이는 검사다.
Matched 후보가 없는 예는 제외하고 분모를 보고한다. 다른 실제 이웃과의 의미적 중복은 남을 수 있다.
비교 토큰은 복원 추출하므로 후보 ID가 중복될 수 있다. 1/6은 균등하게 후보 위치를 고르는 기준선이다.
추가 audit은 입력 center를 읽지 않고 train context 빈도만으로 **같은 후보**를 순위화한다.
모든 값은 '고정 후보 추출 방식 아래의 retrieval'이며, 기존 BET-only 점수와 직접 우열 비교하지 않는다.

## 실험 B: 정확히 같은 양끝의 다른 길이 행동열

`o`는 기존 현재 관측 encoder의 235개 필드 전체다. 카드 네 그룹, 구간, actor, legal mask, chip scalars가
포함된다. History는 원래 이 encoder의 입력이 아니다. Float32 배열이 bit-exact하게 같은 쌍만 허용한다.
유사도 threshold, 카드 제거, endpoint clustering, state embedding 유사도를 사용하지 않는다.

1. 실제 엔진을 실행하여 5구 첫 베팅 직전에 정지한다. 같은 root에서 완전 복제한 분기를 만든다.
2. 기존 `play_betting_round()`에 합법 script를 공급하여 5구의 non-fold 경로를 전부 실행한다.
   베팅 legality/pending/금액 계산을 실험 코드로 다시 구현하지 않는다.
3. 같은 deck에서 6구를 딜하고 기존 엔진이 다음 베팅 구간을 초기화한 첫 결정 직전까지 진행한다.
4. 같은 root/관찰자에서 오른쪽 235D 관측까지 동일한 경로를 묶는다. 입력 action열은 BET와 6구 딜을
   포함하고 bookkeeping인 TURN/STREET는 제외한다. 양성 쌍은 반드시 길이가 달라야 한다.
5. 비교용 negative는 같은 root에서 시작하지만 오른쪽 관측이 다른 합법 경로다.
   동일 root 후보들은 6구 딜 카드도 같아서 카드 ID만으로 양성을 찾을 수 없다.

예: 첫 행동자의 `BBING -> 상대 CALL`과 `CHECK -> 상대 BBING -> CALL`은 다음 구간에서 같은
stack/pot/cards를 만들 수 있다. 초기 pot 등도 같은 상태에서 실제 엔진으로 검증한다.
반면 public betting history는 다르므로 **perfect-recall information set은 같지 않다**.
관측 endpoint의 일치는 카드별 상대 range, 경로 확률, 효용, 이후 전략까지 같다는 뜻이 아니다.
전체 history를 o에 포함해야 한다면 이 실험의 같음 조건을 채택할 수 없다.

## 추가한 최소 개념의 설명

### GRU: 사용자가 쓴 f의 첫 구현

`f(a1,...,an)`에는 임의 길이의 순서를 읽는 함수가 필요하다. 여기서는 32D token lookup을
한 층 32D GRU에 순서대로 전달하고 마지막 표현을 쓴다. GRU는 '새 token으로 기존 기억을 어느 정도
바꾸고 남길지' 학습하는 작은 순환 신경망이다. Transformer/attention/positional encoding은 사용하지 않는다.
초기 기억은 0이며 좌우 observation이나 root ID는 모델 입력이 아니다. 모든 위치의 파라미터를 공유한다.
단순 평균한 SGNS vector도 비교하여 이 순차 함수와 학습이 실제로 필요한지 확인한다.

### 대조 학습: 모든 vector가 같아지는 해를 막기 위한 비교

정답 쌍을 가깝게 만드는 loss만 있으면 모든 vector를 같은 값으로 만드는 해가 가능하다.
그래서 query와 동일 endpoint의 한 경로, 다른 endpoint의 5경로를 비교한다.

\[
L=-\log\frac{\exp(\cos(f(a),f(a^+))/\tau)}
{\exp(\cos(f(a),f(a^+))/\tau)+\sum_{j=1}^5\exp(\cos(f(a),f(a_j^-))/\tau)},\qquad \tau=0.1.
\]

Temperature tau는 후보 사이 유사도 차이를 얼마나 강조할지 정하는 고정 숫자다.
표현은 cosine 비교를 위해 정규화하며, 정답·negative는 학습된 유사도가 아니라 실제 endpoint 일치로 정한다.
관측된 적 없다는 이유로 불가능한 경로를 만들거나 negative라고 판정하지 않는다.

## 비교군 및 데이터 분리

- SGNS-mean: 실험 A의 frozen center embedding을 경로 내 평균. 순서를 보존하지 않는다.
- Endpoint-GRU: 무작위 action lookup과 GRU를 위 endpoint loss로 함께 학습.
- SGNS+Endpoint-GRU: SGNS center lookup에서 시작하여 lookup과 GRU를 함께 학습.
- Constant: 모든 후보 동점이면 정답률 1/6. 동점을 첫 후보 정답으로 처리하지 않는다.
- Payment sum: 사건의 공개 paid를 플레이어별 합산하여 경로 간 차이를 비교하는 비학습 기준선.
  이 벤치는 카드가 후보 간 같고, 칩 합계로 endpoint를 구별할 수 있으므로 중요한 shortcut 검사다.

새 branch seed는 50260924부터 256개다. 기존 SGNS corpus의 hand seed 구간과 분리했다.
Root 단위 204/26/26 train/val/test이며 모든 분기와 양쪽 관찰자는 같은 split을 유지한다.
카드가 새로워도 동일 베팅 pattern을 암기할 수 있으므로 BET의 `(actor,kind,street,paid)` template hash로
약 20%를 path training에서 전역 제외한다. 실제 선택 비율은 데이터에 따라 다르고 manifest에 기록한다.
Test는 전체와 held-out-template query만 따로 보고한다. 후자의 정답 후보는 seen template여도 허용한다.
SGNS에는 그 template를 이루는 **개별 token**이 나타날 수 있다. 새 token 학습 능력을 평가하는 것은 아니다.

Root 분할과 template 분할은 다르며, 완전히 새로운 상대 정책/다른 구간/다른 starting pot 일반화는 미평가다.
평가 query는 조건별 고정 2048개를 복원 추출한다. 이를 독립 2048 hands로 해석하지 않는다.
Training path data의 root/endpoint 균등 가중을 주장하지 않는다. 적격 query는 균등, positive/negative는
그 query의 적격 경로 목록에서 균등하게 뽑으므로 경로가 많은 endpoint의 영향이 더 클 수 있다.

## 실행과 예산

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/action_paths_20260924'
& $py -m unittest agents.state_action_embedding.test_action_path_embedding -v
& $py -m agents.state_action_embedding.action_path_embedding collect --out-dir $run --roots 256
& $py -m agents.state_action_embedding.action_path_embedding train --out-dir $run --seed 11
& $py -m agents.state_action_embedding.action_path_embedding train --out-dir $run --seed 22
& $py -m agents.state_action_embedding.action_path_embedding train --out-dir $run --seed 33
& $py -m agents.state_action_embedding.action_path_embedding audit --out-dir $run
python -m agents.state_action_embedding.action_path_report --out-dir $run
```

기존 run에는 다시 collect/train하지 않는다. 재실행은 새 output 디렉터리를 사용한다.
PyTorch CPU 4 threads. SGNS 2500 updates x batch512, SparseAdam .02; 각 path 조건 1500 updates x
batch128, Adam .001, gradient norm cap1. Embedding/GRU 32D. Validation은 250 updates마다,
test는 조건별 최종 고정 step에서만 평가한다. Best checkpoint 선택이나 test 기반 조정은 하지 않는다.
두 path 조건은 같은 GRU 초기값/훈련 pair 추출을 공유하고 lookup 초기값만 다르다.
SGNS+Endpoint는 pretraining 비용이 추가되므로 총 연산량이 동일하다고 말하지 않는다.

`sgns.pt`, `endpoint.pt`, `sgns_endpoint.pt`에 weights/vocabulary와 optimizer/RNG 상태도 저장한다.
`load_path_encoder(path)`는 token ID 배열과 길이를 받는 inference 모델을 불러온다.
Optimizer 상태는 보존하지만 CLI의 resume 명령은 아직 제공하지 않는다.
토큰 ID는 checkpoint의 `vocab` 행번호+2, 0=padding, 1=unknown이다.
이 실험은 path token이 모두 train-only SGNS vocabulary에 있을 때만 학습하며 OOV가 있으면 중단한다.

## 반드시 남겨야 할 한계

이 벤치가 성공해도 확인되는 것은 **작은 합법 경로 집합에서 관측된 효과가 같은 행동열을 묶는 능력**이다.
모든 observation/action의 의미, 상대의 의도, hidden-state distribution, 게임의 승률을 배웠다고 할 수 없다.
특히 payment sum도 완벽하면 endpoint 과제가 쉽다는 뜻이며 모델의 전략적 발견 증거가 아니다.
GRU 경로 표현이 좋아져도 단일 action lookup 자체가 의미를 잘 분리했다고 단정하지 않는다.
SGNS-mean과의 차이에는 순서 처리와 학습 목적의 차이도 있으므로 SGNS 자체의 실패로 일반화하지 않는다.
다른 길이는 이번에는 4 vs 5 atomic actions뿐이다. 이후 다양한 구간/칩 규모/경로 길이에 확장하되,
같은 시작/끝이라는 사용자의 조건을 유사한 시작/끝으로 몰래 완화해서는 안 된다.

참고: [Skip-gram/negative sampling](https://arxiv.org/abs/1310.4546),
[가변 길이 sequence encoder](https://arxiv.org/abs/1406.1078).
