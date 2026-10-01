# 7-stud Raw 입력의 PPO / Neural Outcome-MCCFR 파일럿

## 목적과 범위

기존 85D raw 사건 표현을 그대로 사용해 전략학습을 시작한다. Skip-gram 모델,
constraint 추가 파라미터, 합법성 보조 loss는 사용하지 않는다.
이 환경은 프로젝트의 **custom seven-poker v3**이며 표준 casino Stud가 아니다.
2인 cash, 매 hand1000chips/ante1, 4장 비공개 딜 후 버림/공개 선택,
4th 딜 이후5th/6th/7th betting. 각 플레이어의 street별 공격적 베팅 상한은1/2/3이며,
CHECK 이후 재레이즈 제한과 stack/call 조건도 기존 엔진 그대로다.

## 입력 형식의 차이

| 실험 | 입력 | 출력/학습 대상 |
| --- | --- | --- |
| 이전 Kuhn/Leduc | 23종 사건 ID의 이력, learned lookup/position | 행동2/3종; full-tree regret 또는 PPO |
| 이전7-stud embedding/합법성 | 사건6필드에서85D raw 또는32D skip-gram으로 변환 | 주변 사건 또는 다음 사건333종 |
| 이번7-stud 전략학습 | 동일85D raw 사건 이력 + 학습 가능한 projection | betting8종 + discard/reveal 조합12종 |

저장 형식은 `[kind, subject, card, bet, street, paid]`다. Enum에 수치적 거리의
의미를 주지 않고 다음과 같이 확장한다.

| 85D 입력 필드 | 차원 | 의미 |
| --- | --- | --- |
| kind | 6 | ANTE / PRIVATE / PUBLIC / DISCARD / REVEAL / BET |
| actor | 3 | self / opponent / environment |
| recipient | 3 | self / opponent / 해당 없음 |
| card | 53 | 관측 가능한52종 + MASK/없음0 |
| betting action | 9 | 행동8종 + 해당 없음 |
| street | 7 | 기존 street enum |
| paid | 1 | 기존 train 통계로 표준화한 실제 지불 칩 |
| 존재 flags | 3 | card field 존재 / card 관측됨 / paid field 존재 |

총81D one-hot +4D 숫자/flags. 상대 PRIVATE/DISCARD 대상 card는0이다.
`START/STREET/TURN` bookkeeping은 기존 raw 실험처럼 제외한다. 결정 직전까지의
전체 retained event를 입력하며, 지금 선택할 행동/그 결과/terminal reward는 넣지 않는다.
마지막 실제 사건의 hidden state에서 출력을 만든다. 별도 query/constraint token 없음.

기존 `action_window.field_arrays`와 같은 변환을 Torch로 계산하고 수치 대조한다.
기존 raw checkpoint에서 **정규화 통계만** 읽으며 encoder/backbone/head 가중치는
읽어 학습 초기값으로 쓰지 않는다. Paid mean29.8470/std95.0571 및85D 후속 표준화는
과거 low-fold9k train의 통계로 고정한다. 새로운 사건도 필드로 직접 변환한다.

**표현 형식을 재사용하는 것이지, 과거 low-fold 데이터로 전략을 학습하는 것이 아니다.**
PPO/CFR 학습 hand는 현재 모델로 새로 생성한다. Fold 비율을 강제하지 않는다.
과거 low-fold validation은 고정 합법성 진단 panel에만 사용한다.

## 모델과 학습

공통:85D->Linear32->tanh, 고정 sinusoidal position, causal attention2층/4heads,
FFN64, dropout0, LayerNorm,64D hidden, 행동20 출력 +value1.
같은 seed의 두 조건은 같은 초기 weights, actor/value 마지막 층0으로 합법균등 시작.
RoPE/사전학습/모방학습/상태235D/별도 constraint 입력 없음. 전체 이력128까지이며
초과 시 잘라 쓰지 않고 실패한다. Engine의 합법 마스크는 출력에만 적용한다.

- **PPO:** 현재 정책끼리 self-play, 양쪽 플레이어 데이터를 함께 학습.
  기존 GAE gamma1/lambda.95, clip.2,4epochs,batch64,entropy.01,KL조기종료.03,
  grad norm.5. Terminal net chips/100을 보상으로 사용한다.
- **CFR 계열:** Neural outcome-sampling MCCFR 근사. Hand마다 갱신 대상 seat를
  번갈아 고르고, 그 seat만 `q=.6*uniform_legal+.4*sigma`로 탐색한다.
  상대는 현재 sigma, chance는 원래 환경 분포에서 샘플링한다. 확률을 보정한
  signed regret 표본을 uniform reservoir(capacity20k)에 넣고 합법 행동 MSE를 학습한다.
  매256hands마다32optimizer steps, Adam1e-4/eps1e-5,grad norm1.
  모델은 warm-start, 누적 regret clipping/target clipping/importance clipping 없음.
  순간 regret을 무조건 양수로 바꾸지 않고 **예측값의 양수 부분만** regret matching한다.
- 양쪽 모두 공유 Transformer1개로 두 좌석의 관측을 처리한다. CFR value head는
  사용하지 않는다. Paper Deep CFR의2advantage+1strategy-network 재현이 아니다.
  같은 hand 수이지만 PPO는 양쪽 행동, CFR는 지정된 traverser의 regret을 학습하므로
  정보량/optimizer step/목적함수/gradient cap까지 동일한 통제 실험은 아니다.

## Outcome Regret 보정

한 hand의 traverser 선택들만 시간순으로 남긴다. 그 이전 자신의 sampling 확률
곱을 `Q_prefix`라 하자. Terminal `v=net_chips/100`에서 거꾸로:

```text
child[a_taken] = v / q(a_taken); child[other] = 0
v = sum_a sigma(a) * child[a]
sampled_regret[a] = (child[a] - v) / Q_prefix
```

반환되는 v에 자신의 이후 경로 sampling 보정이 포함된다. Opponent/chance는
on-policy라 확률비가 상쇄된다. 앞쪽 자기 sampling 보정도 필요하므로 단순 `reward-V`
또는 PPO advantage와 같지 않다. 방문하지 않은 정보집합의 표본 증분은0이다.
Kuhn의 모든 terminal sampling 결과를 열거해 기대 증분이 정확한 CFR와 일치함을 검사한다.

Replay에서 같은 정보집합의 표본 평균을 정확히 맞출 수 있다면 누적 sampled regret에
양의 공통 배율을 곱한 방향을 얻는다. 실제로는 유한 buffer/함수근사/공유 모델의
오차가 있으므로 tabular MCCFR 수렴 보장을 주장하지 않는다. 특히 own-prefix 확률이
작으면 정답의 분산과 크기가 매우 커진다. `target_max`, loss, buffer 크기를 기록한다.

처음 검토한 replay-based external sampling은 한 sampled tree에서50k노드 제한을
넘어 중단했다. 그 partial 데이터는 학습에 사용하지 않았고 초기화부터 outcome 방식으로
다시 시작한다. 따라서 이전 full-tree Kuhn/Leduc과 sampler까지 동일한 실험이 아니다.

## 평가와 제약 아이디어

메인: 분리된 deal seeds에서 heuristic 및 uniform-random 상대에 대한 seat-paired
chips/hand. 각 deck을 두 좌석으로 바꾸어 실행하고 pair 단위95% normal interval을
기록한다. 작은 test256pairs는 불확실성이 크다. **Exploitability/LBR 측정이 아니다.**
Monitor64pairs와 test256pairs는 다른 seed영역이며, 최고 monitor 모델 선택 없이
고정16updates의 마지막 모델을 보고한다. 학습4096hands/seed/방법, seeds11/22/33.

현재전략과 평균전략을 구분한다. 모든 업데이트 직전 모델 snapshot을 보존하고,
평균전략 평가는 hand 시작에 snapshot 하나를 균등 선택하여 그 hand 내내 고정한다.
이것은 과거 behavioral strategy의 mixed-strategy 평균이며, 매 행동마다 모델을
무작위 교체하거나 logits/weights를 평균하는 방법이 아니다. 단일 신경망 평균 근사도 아니다.
저장/추론 메모리가 snapshot 수에 비례하므로 긴 학습에서는 별도의 압축 검증이 필요하다.

합법성 진단은 기존 validation의 actor-own-view512개 betting 직전 이력으로 고정한다.
실제 다음 행동은 prefix에서 제외한다. 과거 합법성 실험에 쓰인 validation이므로
새로운 독립 규칙-learning test라고 부르지 않는다. 입력으로 합법 mask를 주지 않는다.

- Betting8종 안에서 **마스크 전** 불법 확률합과 top1 불법률(동점 균등 처리)을 계산.
- PPO는 softmax, CFR는 positive-regret 정규화(전부0이면8종균등)라 분포의 생성법이 다르다.
  확률 calibration을 동일시하지 말고 각 방법의 초기값 대비 변화도 본다.
- 실제 실행은 합법 mask로 보호하므로 illegal execution=0이어도 규칙 학습 증거가 아니다.
- 현재 모형의 panel 진단만 제공한다. 평균 mixed policy의 panel posterior는 별도
  계산하지 않으며 그 칸은 null이다. Chance/다음 주체 예측을 하는 이전333종 과제와 다르다.

**사용자의 후속 아이디어:** 행동 이벤트에 남은 베팅 횟수, CHECK 이후 재레이즈 가능성,
call/stack 조건 등의 constraint metadata를 붙인다. 의미상으로는 여전히 관측 상태에
조건화된 행동 속성이며, 입력 위치를 행동 param으로 둘 수 있다. 선택 전부터 관측 가능한
값만 사용해야 하며, 선택된 행동이나 이후 결과를 거꾸로 넣으면 안 된다.
이번에는 추가하지 않았다. 이후 같은 데이터/seed/예산의 raw+constraint 비교가 필요하다.

## 실행과 산출물

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_raw_cfr_ppo -v
& $py -m agents.state_action_embedding.raw_cfr_ppo --out-dir agents/state_action_embedding/data/raw_cfr_ppo_20260928 --seeds 11 22 33 --updates 16 --hands 256 --fit-steps 32 --test-pairs 256
python -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir agents/state_action_embedding/data/raw_cfr_ppo_20260928
```

Smoke는 별도 폴더에 `--seeds 11 --updates 2 --hands 8 --fit-steps 2 --batch 8
--monitor-pairs 2 --test-pairs 2 --panel 16 --eval-every 1`.
학습/데이터 생성은 CPU1 thread. 시간에는 sampling/역전파를 포함하나 정규화통계 로딩,
panel 구성, checkpoint 복사/저장/평가는 제외한다. Decision visits를 기록하며 chance
노드를 포함한 이전 full-tree node visits와 직접 동일시하지 않는다.

`config.json`은 입력/코드 hash와 예산, `metrics.sqlite`는 곡선/평가,
`legality_panel.npz`는 진단 입력, 각 seed 폴더는 모델/좌석별 수익/replay를 저장한다.
`regret_replay.npz`는 bounded reservoir 표본이며 모든 history의 표가 아니다.
Checkpoints는 weights/정규화/구조를 저장한다. Optimizer/RNG-resume는 미구현이다.

```python
from agents.state_action_embedding.raw_cfr_ppo import load_model, inference
model = load_model('PATH/last.pt')
# tokens: 관측 가능한 retained raw 사건 [T,6]; legal: 행동20종 bool
probabilities, value, scores = inference(model, tokens, legal)
```

## 2026-09-28 완료 결과

[상세 결과](data/raw_cfr_ppo_20260928/RESULTS.md),
[학습곡선](data/raw_cfr_ppo_20260928/curves.png),
[seed별 요약](data/raw_cfr_ppo_20260928/summary.json).

각4096hands/3seeds에서 현재정책의 heuristic 상대 수익은 초기-17.229에서
PPO-24.759, neural outcome-MCCFR-31.352chips/hand로 악화됐다.
Random 상대는+4.830에서 각각+17.678/+32.480으로 개선됐다.
이 결과로 균형 수렴이나 두 학습법의 일반적 우열을 결론 내리지 않는다.
특히 MCCFR의 importance-corrected regret target은 최대1.449e8으로,
표현의 성능을 논하기 전에 샘플링 분산 문제를 분리해야 한다.

고정 panel의 마스크 전 불법확률은30.86%에서 PPO32.41%/MCCFR33.57%로 증가했다.
Constraint를 행동 param으로 추가하는 후속 실험은 아직 하지 않았다.
4개 단위검사, 같은seed 초기weights 일치,6개 최종 checkpoint의 복원 후panel 일치,
소스/기존raw checkpoint hash 불변을 확인했다. 전체 과거 실험을 재실행하지는 않았다.

## 예측 과정을 한 단계씩 따라가기

### 무엇을 입력하고 무엇을 예측하는가

이번에는 다음 사건이나 상대 행동을 맞히는 지도학습을 하지 않는다.
환경이 **내 결정 차례에 호출하면 지금 내가 할 행동의 분포**를 출력한다.
Player enum, 다음 card/chance, 지불 amount를 각각 예측하는 이전333종 과제와 다르다.
금액은 과거 사건의 입력 특징일 뿐이고, 현재 금액은 선택한 betting 종류와 엔진 규칙으로
정해진다. 별도 금액 회귀 head나 chance 예측 loss가 없다. Teacher forcing/모방 정답도 없다.

1. 결정 직전까지 자신의 관측 이력 `[T,6]`을 가져온다. 자기 카드와 공개 카드는 보지만
   상대 비공개 카드 대상은0이다. Deal의 발생/수신자는 관측에 남는다.
2. 각 행을 앞 표의85D로 변환/표준화해 `[T,85]`를 만든다.
   완성된 현재 상태 벡터나 족보/남은 raise 수/현재 합법성 벡터를 input에 붙이지 않는다.
3. `tanh(Linear85->32(x)) + sinusoidal_PE`로 `[T,32]`를 만든다.
   두 causal self-attention block(각4heads, FFN64)을 지나도 `[T,32]`다.
   구현 클래스는 `TransformerEncoderLayer`지만 미래를 가린 causal backbone이며,
   별도 cross-attention encoder-decoder나 RoPE는 없다.
4. `LayerNorm -> Linear32->64 -> tanh`로 `[T,64]`를 만든 뒤 **마지막 실제 사건**의64D를 읽는다.
   새로운 TURN/query token은 없다. 오른쪽 padding은 선택한 위치의 미래이므로 causal mask로 차단된다.
5. 행동 head는20개의 실수 score를, value head는1개의 예상 자기 수익을 출력한다.
   Parameters는23,381개이며 CFR에서는65개 value-head parameter가 학습에 사용되지 않는다.
6. 환경의 합법 mask로 지금 가능한 output만 남긴 뒤 행동을 확률적으로 뽑는다.
   최대 점수 행동을 항상 고르는 greedy 방식이 아니다. 합법 mask는 출력/손실에 쓰이지만
   Transformer 입력으로는 주지 않는다. 다음 결정에서는 실제로 발생한 사건까지 이어 읽는다.

출력0..7은 `CHECK, BBING, DDADANG, QUARTER, HALF, FULL, CALL, FOLD`다.
출력8..19는 초기4장의 서로 다른 `(버릴 카드 index, 공개할 카드 index)`12조합이다.
Discard/reveal 차례에는 그12개만, betting 차례에는 엔진이 허용한 betting만 합법이다.
초기 카드 순서는 자기 PRIVATE 사건의 순서로 입력되므로 index의 대상을 읽어야 한다.

### PPO의 score와 정답

Score `z_a`는 예상 칩이나 regret이 아니라 **logit**이다. 합법 집합을 `L`이라 할 때:

```text
pi(a | history) = exp(z_a) / sum_{b in L} exp(z_b),  a in L
pi(a | history) = 0,                              a not in L
```

현재 모델끼리256hands를 하고 양측의 자기 결정 이력을 모은다. 한 hand의 보상은
`G=(final_chips-1000)/100`. 각 플레이어의 마지막 결정에만 G를 배치하고 나머지는0이다.
아래 t는 **그 플레이어의 결정 순번**이며 chance/상대 행동은 별도 policy-loss 행이 아니다.

```text
delta_t = reward_t + V_old(h_{t+1}) - V_old(h_t)   # gamma=1, terminal next V=0
A_t     = delta_t + 0.95 * A_{t+1}
target_V_t = A_t + V_old(h_t)
rho_t   = pi_new(a_t | h_t) / pi_old(a_t | h_t)
L_actor = -mean(min(rho_t * normalized_A_t,
                    clip(rho_t, 0.8, 1.2) * normalized_A_t))
L = L_actor + 0.25 * MSE(V_new, target_V) - 0.01 * entropy(pi_new)
```

즉 "정답 행동=CALL"을 가르치거나 score를 G에 직접 회귀하지 않는다.
유리했다고 추정한 **실제로 선택한 행동**의 확률을 올리고, 불리했던 행동은 낮춘다.
행동/GAE/old logprob는 고정 데이터지만, 학습할 때 history를 다시 forward하여 현재
score/value에서 Transformer 전체까지 역전파한다. Rollout의 `inference_mode`는
이 재계산에 적용되지 않는다. 최대4epochs/미니배치64, KL>.03이면 그 업데이트를 조기 종료한다.
[PPO 원 논문](https://arxiv.org/abs/1707.06347)은 interaction 표본과 surrogate 최적화를 번갈아
진행하는 방법을 제시한다. 여기의 수치 설정과 value-loss 계수는 로컬 코드의 선택이다.

### CFR 쪽의 score와 정답

이번 score는 signed sampled regret들의 replay 평균을 근사한다. 정확한 표로 누적한
regret 자체를 읽는 것이 아니며, PPO의 logit과 단위/의미가 다르다.

```text
positive_a = max(score_a, 0) * legal_a
sigma_a = positive_a / sum(positive)   # 양수합0이면 합법균등
```

탐색 대상 플레이어는 `q=.6*합법균등+.4*sigma`에서 행동을 뽑고 상대는 sigma를 따른다.
**한 hand에서는 선택한 경로만 실제 진행**한다. 선택하지 않은 행동을 각각 끝까지
rollout한 값을 직접 얻는 것이 아니다. 앞 절의 own-prefix/tail 확률 보정으로
그 행동들에 대한 counterfactual regret의 noisy 추정량을 만든다.

Signed 표본을 누적 reservoir에 저장하고 미니배치64개씩32번 학습한다:

```text
L_CFR = mean_samples(mean_legal_actions((score - sampled_regret)**2))
```

역전파는 **clipping 이전 score**에서 시작한다. Regret matching의 `max(score,0)`를
미분해서 학습하는 것이 아니므로 음수 score도 회귀 gradient를 받는다.
환경/sampling/label 자체는 미분하지 않는다. 완전히 같은 정보집합의 표본 평균을
정확히 맞추면 저장된 누적 sampled regret과 행동 공통의 양수 배율만 다르지만,
유한 데이터와 신경망 공유/일반화에서는 이 관계를 정확히 보장하지 않는다.

### 같은 입력에서 실제 출력한 예

고정 panel0번의30사건 이력, seed11 마지막 checkpoint. 현재 합법 집합은 아래6종이다.
CHECK/BBING과 discard/reveal12종은 불법이라 실제 선택 확률0이다.

| 합법 행동 | PPO logit | PPO 확률 | CFR signed score | CFR 확률 |
| --- | ---: | ---: | ---: | ---: |
| DDADANG | 0.10264 | 18.03% | -0.23669 | 0% |
| QUARTER | -0.00718 | 16.16% | -0.09312 | 0% |
| HALF | -0.07409 | 15.11% | 0.05315 | 8.39% |
| FULL | -0.23187 | 12.91% | 0.26240 | 41.42% |
| CALL | 0.12064 | 18.36% | 0.31796 | 50.19% |
| FOLD | 0.17745 | 19.43% | -1.20788 | 0% |

이는 예측 메커니즘의 실제 예이지 이 history의 최적 행동 정답이 아니다.
PPO의 최고 확률은FOLD지만19.43%에 불과하므로 항상FOLD하지 않는다.
CFR 학습 중 traverser는 위 sigma가 아니라 탐색혼합q로 뽑는다. 평가 때는sigma를 쓴다.
전체 raw history/score20개/분포는 아래 diagnostics JSON에 남겼다.

## 2026-09-28 역전파와 학습 신호 진단

**이전 결과 해석 보충:** heuristic 수익 악화는 확인됐지만, 이를 "역전파가 안 된다"거나
"아무것도 학습하지 않는다"로 표현하면 부정확하다. 불법 score도 직접 낮추지 않는 목적함수이므로
마스크 전 불법확률을 이 전략학습의 최적화 성공 지표로 삼을 수 없다.

- **가중치 변화:** 모든6모델에서 projection과 두 attention block이 변했다.
  초기 대비 block별 parameter delta L2는 CFR0.4445~0.5770, PPO0.7481~1.4722.
  CFR value head만 그대로인데, 그것은 loss에 사용하지 않기 때문이다.
- **정책 변화:** 고정512prefix의 초기 합법균등 대비 평균 TV는 CFR0.476~0.556,
  PPO0.171~0.176. 동일한 균등정책을 계속 출력하고 있는 것이 아니다.
  이 변화가 전략적 개선을 의미하지는 않는다.
- **새 gradient probe:** 각seed8개 random minibatches(64개)에서 모두 유한한 gradient.
  CFR의 clipping 전 global norm은1.41e3~1.43e7, PPO는0.56~3.11이었다.
  원래 학습의 역사적 gradient 로그가 아니라 **저장 모델/데이터에 대한 새 진단**이다.
  Clip limit은 각각1/.5이므로 검사한24개씩 전부 clipping 대상이다.
  Gradient clipping이 곧 vanishing이나 학습 불가능을 뜻하지 않는다. Adam의 적응형 처리도 있다.
- **Zero-head 초기화:** 첫 역전파는 출력 head만 갱신하고 backbone gradient가0이다.
  두 번째 optimizer step 전에는 backbone에도 nonzero gradient가 생김을 직접 확인했다.
  따라서 초기 첫 스텝의0을 영구적인 gradient 차단으로 오해하면 안 된다.
- **Raw 포화 간이 검사:** 고정panel에서 input projection의 `abs(tanh)>0.99` 비율은
  여섯 모델 모두0. 다른 층의 포화/표현 품질까지 검증한 것은 아니다.

### CFR: noisy target이 loss를 지배한다

| seed | target RMS | prediction RMS | 상위1%행의 제곱 target 비중 |
| --- | ---: | ---: | ---: |
| 11 | 231,264 | 0.501 | 99.16% |
| 22 | 1,127,783 | 0.336 | 99.13% |
| 33 | 331,361 | 0.503 | 98.19% |

RMS와 비중은 학습 loss처럼 각행의 합법행동 평균으로 계산한다.
표본의 합법 target 원소 중55~60%는0이고 소수의 큰 importance-weight 표본에
MSE가 집중되어 있다. 전체 replay MSE / 항상0예측 MSE는0.99999988~0.99999998이다.
다만 stochastic label에는 irreducible variance가 있으므로, 이것만으로 **조건부 평균**
regret을 전혀 배우지 못했다고 증명할 수는 없다. 현재 loss가 드문 큰 표본에 매우
민감하다는 진단이며 더 많은 학습만 하면 해결된다는 보장도 없다.

MCCFR의 sampling variance와 baseline을 통한 완화는 별도로 연구된 문제다.
[Low-Variance and Zero-Variance Baselines](https://proceedings.mlr.press/v119/davis20a.html).
이번 구현에는 그 variance-reduction baseline을 넣지 않았다.

### PPO: value가 아직 약하다

16번째 업데이트 직전 snapshot과 원래 rollout seed로 마지막256hands를 재생성했다.
그때의 GAE target을 고정해 업데이트 전후 value MSE를 비교했다.

| seed | 업데이트 전 MSE | 업데이트 후 MSE | 후 explained variance |
| --- | ---: | ---: | ---: |
| 11 | 9.1066 | 8.8854 | 0.03955 |
| 22 | 8.9363 | 8.9326 | 0.00071 |
| 33 | 15.2457 | 15.0794 | 0.01350 |

`EV = 1 - Var(target-V)/Var(target)`. 약0.07~3.96%이며 실제 학습에 사용한 마지막
배치의 in-sample 진단이다. 독립 validation 성능/승률이 아니다. Chance/상대 비공개 정보로
인한 불확실성도 있으므로 EV가1이어야 한다고 가정하지 않는다. 세seed 모두 MSE는
작게나마 내려갔고 역전파 단절보다는 약한 value 예측/고분산 신호를 의심할 근거다.

기존 실행은 `update_ppo`가 반환한 value loss/entropy/KL/clip_fraction을 버리고
policy_loss만 DB에 기록했다. **진단용 로깅이 부족했던 부분**이다. Policy loss가0에
가깝다는 것은 advantage 표준화 때문에 생길 수도 있고, 높은 예측 정확도를 뜻하지 않는다.
새probe의 KL은 원래 모든 optimizer step의 KL 로그를 대체하지 않는다.

### 불법확률 진단의 한계

PPO는 불법 logit을 상수로 치환하고 CFR MSE는 불법항을 제외한다.
각 샘플에서 `d(loss)/d(score_illegal)=0`임을 이번 검사에서도 확인했다.
공유 weights 때문에 다른 상황에서 간접 변화는 가능하지만, 여기에는 불법확률 자체의
penalty가 없다. Constraint param을 붙이더라도 이 loss만으로 불법확률 감소를
자동 보장하지 않는다. 그 가설을 시험하려면 별도 합법성 loss/평가와 전략 수익을 구분해야 한다.

### 다음 실험의 우선순위 (아직 실행하지 않음)

1. PPO의 value/entropy/KL/clip fraction 및 양쪽의 gradient norm/clip rate를 지속 기록한다.
2. 작은 고정batch에서 회귀/정책 fitting이 되는지 확인한다. CFR는 noisy 단일표본과
   작은게임의 정확한 conditional-mean regret을 구분하고, 전역 양의 target scaling도
   별도 조건으로 비교한다. 전역 scaling은 정확한 regret matching을 바꾸지 않지만
   optimizer/함수근사 결과까지 같게 만드는 것은 아니다.
3. PPO는4096→16384→65536hands/seed처럼 동일 모델의 학습량 곡선을 먼저 비교한다.
   CFR는 sampling baseline 등 분산 완화와 budget 증가를 분리해서 비교한다.
4. 그 뒤 raw+constraint와 모델 크기를 한 가지씩 변경한다. 지금 결과만으로 작은 모델,
   데이터 부족, target variance, self-play 편향 중 하나를 단독 원인으로 확정하지 않는다.

### 진단 재실행

```powershell
& $py -m agents.state_action_embedding.raw_cfr_ppo_diagnose --run agents/state_action_embedding/data/raw_cfr_ppo_20260928 --out-dir agents/state_action_embedding/data/raw_cfr_ppo_diagnostics_20260928
```

`$py`는 앞 절의 PyTorch runtime이며 새 output 이름이 필요하다.
[진단 데이터](data/raw_cfr_ppo_diagnostics_20260928/diagnostics.json).
기존6개 모델을 수정하지 않고 해석/gradient probe만 수행했다. 최초seed의2-step
검사는 복사본에만 적용하고 저장하지 않았다. CFR replay는모두, PPO는마지막256hands만
분석했으며 예산 확대/성능 재평가는 하지 않았다. 사용 source hash의 불변성을 확인했다.

## MSE / Root / L1 비교

**정정 (2026-09-28)**: 아래의 RMSE 해석은 사용자 의도와 달랐다. 사용자는
`sqrt(mean(e))`를 요청했다. 아래 실험은 별도 RMSE/L1 비교 기록으로 보존하며,
요청한 식의 구현과 실행 결과는 마지막 절에 기록한다.

사용자 제안은 큰 regret 표본에 덜 흔들리도록 root 또는 L1 loss를 쓰는 것이다.
이번 root는 **batch MSE 전체에 root를 취하는 RMSE**로 해석한다.
`sqrt(error^2)`를 각각 적용한 뒤 평균하는 뜻이었다면 그것은 바로L1이다.
`sqrt(abs(error))`는 이와 다르며 이번에는 구현하지 않았다.

각 history의 합법행동에 먼저 동일 가중치를 주고 history끼리 평균한다. 아래 mean은
그 두 단계 평균이며 예전 MSE와 같은 가중 방식이다. `e=score-target`:

```text
MSE  = mean(e^2)
RMSE = sqrt(mean(e^2) + 1e-12)
L1   = mean(abs(e))
```

RMSE의epsilon은정확히맞힌batch에서root의미분이NaN이되는것을막는다.
MSE 기본값은이전과동일하며L1/RMSE도signed target과불법항제외를유지한다.
Enum/모델/샘플링/학습률1e-4/Adam/gradient clip1/reservoir20k는변경하지않았다.

### Root의 효과를 구분해야 하는 이유

같은batch에서는 `grad(RMSE)=grad(MSE)/(2*sqrt(MSE+epsilon))`다.
그러므로 gradient의전체크기는작아지지만큰오차와작은오차의**상대영향과방향은그대로**다.
오차1과100은MSE/RMSE에서gradient크기비율1:100, L1에서는1:1이다.
둘다global-norm clipping대상이면정규화된gradient는수학적으로같아질수도있다.
실제로는일부batch에서clipping여부와Adam의누적통계/epsilon이달라지고,
매batch root를취하면batch간weight도달라져학습경로가완전히같다는뜻은아니다.

L1은큰오차의직접gradient를제한하지만, stochastic label의conditional mean이아닌
median을목표로한다. 예를들어90%확률로0,10%로100인label의평균은10,
중앙값은0이다. CFR의기대regret이10이어야할때L1이0을배우면분산만줄인것이아니라
목표도달라진다. 따라서안정성뿐만아니라실제정책수익도확인하며수렴보장은주장하지않는다.

### 실행 조건과 저장

MSE는이전완료3seeds결과를기준선으로재사용한다. 새MSE구현의2update/16hand
smoke에서기존smoke의모든model parameter/buffer가정확히일치함을확인했다.
RMSE/L1은각각seeds11/22/33,16updates x256hands=4096hands,32fit steps/update,
총512optimizer steps/seed. 같은초기weights/정규화/환경/평가deck이며skip-gram없음.
각조건은자기정책으로fresh hand를생성하므로학습dataset자체는동일하지않다.
최종현재정책/과거snapshot평균혼합을각각test256pairs로측정한다.

새`fit_diagnostics` SQL table에512개optimizer step별loss와clipping전gradient norm을
저장한다. 서로다른loss의숫자가작아졌다는것을성능개선으로부르지않는다.
과거MSE의역사적gradient로그는없으므로새로그인것처럼소급기록하지않는다.

```powershell
foreach ($loss in 'rmse','l1') {
    & $py -m agents.state_action_embedding.raw_cfr_ppo --methods cfr --cfr-loss $loss --out-dir "agents/state_action_embedding/data/raw_cfr_${loss}_20260928" --seeds 11 22 33 --updates 16 --hands 256 --fit-steps 32 --test-pairs 256
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $loss" }
    python -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir "agents/state_action_embedding/data/raw_cfr_${loss}_20260928"
}
python -m agents.state_action_embedding.raw_cfr_loss_report --mse agents/state_action_embedding/data/raw_cfr_ppo_20260928 --rmse agents/state_action_embedding/data/raw_cfr_rmse_20260928 --l1 agents/state_action_embedding/data/raw_cfr_l1_20260928 --out-dir agents/state_action_embedding/data/raw_cfr_loss_compare_20260928
```

새훈련폴더가필요하다. 이전MSE/PPO의data폴더와원본source사본은수정하지않는다.
이번loss선택확장으로live source hash는이전실행과다르다. 이전diagnose의strict hash
검사는이차이를거부하므로예전진단을새코드로완전히동일한실행이라고주장하지않는다.
새조건은자기config에기록된source hash로진단할수있다.

### 완료 결과와 해석

[손실함수 비교 결과/신뢰구간](data/raw_cfr_loss_compare_20260928/RESULTS.md),
[그래프](data/raw_cfr_loss_compare_20260928/curves.png).

| loss | 현재 vs heuristic | 현재 vs random | 평균 vs heuristic | 평균 vs random |
| --- | ---: | ---: | ---: | ---: |
| MSE | -31.352 | +32.480 | -34.523 | +29.227 |
| RMSE | -35.451 | +33.951 | -34.310 | +30.408 |
| L1 | -29.074 | +12.389 | -26.482 | +6.743 |

각3seed평균chips/hand. L1의현재정책heuristic수익은seed SD17.899로크고,
MSE대비차이+2.277의조건부paired95%구간[-3.768,+8.310]은0을포함한다.
L1평균혼합의heuristic손실은줄었지만random상대수익은크게줄었다.
RMSE도일관된성능개선은확인하지못했다. Exploitability평가가아니다.

RMSE/L1에서실제1536optimizer step의gradient norm중앙값은.314/.286이며
threshold1clipping은둘다0%였다. 이는이전MSE의사후gradient probe와구분해야한다.
정책성능과는별개로역전파규모를줄이는효과는확인했다.

[RMSE진단](data/raw_cfr_rmse_diagnostics_20260928/diagnostics.json)에서predictionRMS는
.509~.541, [L1진단](data/raw_cfr_l1_diagnostics_20260928/diagnostics.json)에서는
.00126~.00523이었다. L1replay의target원소51.5~55.0%가0이었다.
다만전체target분포의중앙값을개별정보집합의중앙값이라고할수없고,
regret matching은공통양의배율에불변이므로작은score만으로정책붕괴를판정하지않는다.

새6개모델의초기weights/초기test수익배열은MSE와동일함을확인했고,
복원된최종모델의고정panel지표는SQL과일치했다. 5개단위검사통과.
모든학습/평가는완료했으며Huber나variance-reduction baseline은추가하지않았다.

## Literal sqrt(mean(e)) 실행

사용자 정정에 따라 `--cfr-loss root_mean`을 추가했다. 기존 표기의
`e = score - signed_regret_target`을 사용한다. 제곱, 절댓값, clamp, epsilon을
추가하지 않았다. 각 history의 합법행동 오차를 먼저 평균하고 history끼리 평균한다.
따라서 합법행동 개수가 다른 history도 동일한 가중치를 갖는다.

```text
mu = mean_history(mean_legal_actions(score - target))
loss = sqrt(mu)
```

위 raw 85차원 입력/causal Transformer/outcome-sampling CFR와 같은 초기화,
seeds 11/22/33, Adam 1e-4, batch 64, gradient norm clip 1을 사용했다.
계획 예산은 각 16 updates x 256 hands, 32 fit steps/update였다.
실제로는 각 seed가 첫 256 self-play hands 수집 후 첫 fitting에서 중단했다.
기존 MSE/RMSE/L1 결과와 비교 가능한 완성된 학습 모델은 얻지 못했다.
데이터는 각 실행의 self-play 정책에서 생성했고 별도 off-policy 자료나 abstraction은
사용하지 않았다. Abstraction 갱신 간격의 변경도 없다.

| seed | 성공한 optimizer steps | 중단 배치의 mean(e), float32 | 상태 |
| --- | ---: | ---: | --- |
| 11 | 1 | -0.002685282845 | 음수의 sqrt, NaN |
| 22 | 0 | -0.000325840199 | 음수의 sqrt, NaN |
| 33 | 1 | -0.001858317992 | 음수의 sqrt, NaN |

Loss/gradient가 유한하지 않으면 optimizer.step 전에 중단한다. `grad_norm=nan`은
loss가 이미 NaN이어서 backward를 실행하지 않았다는 뜻일 수도 있다.
`metrics.sqlite`의 status는 `interrupted_or_failed`이며 failure에 원인을 기록했다.
각 폴더의 `failed_fit_batch.npz`에 입력/합법 mask/예측/target/평균오차/성공 step 수를
보존했다. 마지막 update 완료 체크포인트(`last.pt`)와 최종 정책 평가는 없다.

- [seed 11 실패 배치](data/raw_cfr_root_mean_seed11_20260928/failed_fit_batch.npz)
- [seed 22 실패 배치](data/raw_cfr_root_mean_seed22_20260928/failed_fit_batch.npz)
- [seed 33 실패 배치](data/raw_cfr_root_mean_seed33_20260928/failed_fit_batch.npz)

초기 균등정책의 regret은 이론상 합법행동 평균이 0이다. 큰 signed target의 상쇄는
반올림에 민감하다. 같은 저장 배치를 float64로 재합산하면 seed 22 평균은
`+0.000327763827`로 부호가 바뀐다. Seed 11/33은 각각 `-0.002795593674`,
`-0.001879551021`로 여전히 음수다. 이는 float64 학습 실험이 아니라 저장값 재합산이다.

수학적으로 평균이 음수면 실수 loss가 정의되지 않고, 0이면 root 미분이 특이하다.
또한 큰 양/음 오차가 서로 상쇄될 수 있다. 따라서 이것은 학습량 부족이나
정책 성능 저하의 증거가 아니다. 음수/0 처리 규칙 또는 `e`의 다른 정의는
사용자 확인 없이 바꾸지 않았다.

실행한 명령 (프로젝트 root, `$py`는 위 PyTorch runtime):

```powershell
foreach ($seed in 11,22,33) {
    & $py -m agents.state_action_embedding.raw_cfr_ppo --methods cfr --cfr-loss root_mean --out-dir "agents/state_action_embedding/data/raw_cfr_root_mean_seed${seed}_20260928" --seeds $seed --updates 16 --hands 256 --fit-steps 32 --test-pairs 256
    Write-Output "seed=$seed exit_code=$LASTEXITCODE"
}
```

재실행에는 새 output 경로가 필요하다. 각 Python 실행의 exit code는 1이었다.
Loop 전체 PowerShell exit code 0을 학습 성공으로 해석하지 않는다.
Literal loss/미분/음수 정의역/잘못된 optimizer step 방지를 포함한 7개 단위검사 통과.

## 절댓값을 포함한 Root-MAE

**사용자 추가 정정:** 오차에 절댓값을 포함해야 한다. 따라서 요청한 새 조건을
`root_mae`로 구분한다. 앞의 signed `root_mean` 실패 기록을 덮어쓰지 않는다.

```text
MAE = mean_history(mean_legal_actions(abs(score - target)))
ROOT_MAE = sqrt(MAE)
```

RMSE나 `mean(sqrt(abs(error)))`, `sqrt(abs(mean(error)))`가 아니다.
Epsilon을 더하지 않는다. 모든 합법 오차가 정확히0이면 loss0/gradient0을 선택하는
수치 규약을 사용하며, 그 외에는 위 식을 그대로 미분한다. 평균이 양수일 때
`grad(ROOT_MAE) = grad(L1)/(2*sqrt(MAE))`다. 같은 batch 안에서 큰 개별 오차의
gradient를 키우지 않고, MAE가 큰 batch 전체의 gradient를 낮춘다. 작은 MAE에서는
오히려 커질 수 있으며 기존 global-norm clip1은 유지한다. 유한 고정 dataset 전체에
한 번 root를 씌운 목적은 L1과 최소점이 같지만, 미니배치별 root/Adam/self-play를
사용하는 이번 학습 경로가 같다는 뜻은 아니다. 기대 regret 보존/수렴을 주장하지 않는다.

위 기존 환경/입력/초기화/seeds11/22/33/학습 및 평가 예산을 유지한다.
각16updates x256fresh self-play hands,32fit steps/update,batch64,Adam1e-4,
test256개 deck의 좌석교환. MSE/RMSE/L1은 완료된 기존 결과를 재사용한다.
각 조건의 방문/label 분포는 자기 정책에 따라 달라진다. 새 사전학습/abstraction이나
갱신 간격 변경은 없다. PPO는 재학습하지 않는다.

```powershell
& $py -m agents.state_action_embedding.raw_cfr_ppo --methods cfr --cfr-loss root_mae --out-dir agents/state_action_embedding/data/raw_cfr_root_mae_20260928 --seeds 11 22 33 --updates 16 --hands 256 --fit-steps 32 --test-pairs 256
python -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir agents/state_action_embedding/data/raw_cfr_root_mae_20260928
python -m agents.state_action_embedding.raw_cfr_loss_report --mse agents/state_action_embedding/data/raw_cfr_ppo_20260928 --rmse agents/state_action_embedding/data/raw_cfr_rmse_20260928 --l1 agents/state_action_embedding/data/raw_cfr_l1_20260928 --root-mae agents/state_action_embedding/data/raw_cfr_root_mae_20260928 --out-dir agents/state_action_embedding/data/raw_cfr_root_mae_compare_20260928
```

프로젝트 root에서 실행하며 `$py`는 앞의 PyTorch runtime이다. 재실행 시 학습/비교
output은 새 이름이 필요하다. Root-MAE의 실제 결과는 아래 완료 기록에 덧붙인다.

### Root-MAE 완료 결과

세 seed 모두 완료: 총12,288 training hands,90,403 decision visits,
48 updates/1,536 optimizer steps. NaN 중단은 없었다. 아래는3seed 평균 chips/hand다.

| loss | 현재 vs heuristic | 현재 vs random | 평균 혼합 vs heuristic | 평균 혼합 vs random |
| --- | ---: | ---: | ---: | ---: |
| MSE | -31.352 | +32.480 | -34.523 | +29.227 |
| RMSE | -35.451 | +33.951 | -34.310 | +30.408 |
| L1 | -29.074 | +12.389 | -26.482 | +6.743 |
| Root-MAE | -29.840 | +8.531 | -21.438 | +8.945 |

Root-MAE의 heuristic 수익 seed SD는 현재7.269/평균혼합0.675다.
MSE 대비 현재 수익 차이+1.512의 paired-deal bootstrap95%구간은
[-3.953,+7.016]으로0을 포함한다. 평균혼합 차이는+13.085[+4.898,+21.630]이나
random 상대는 현재-23.949/평균혼합-20.282로 악화됐다. 구간은 고정된 학습모델에
조건부이며3seed 학습 불확실성을 완전히 포괄하지 않는다. 평균혼합은 매 hand 시작에
과거 snapshot 하나를 뽑는 정책이며 최종 단일 모델이 아니다.

초기 합법균등의 heuristic 수익 -17.229보다 Root-MAE 현재/평균혼합 모두 낮다.
따라서 기존 MSE 대비 일부 개선은 있어도 전반적인 정책 개선/균형 수렴으로 결론 내리지 않는다.
기존 test를 반복 사용한 탐색적 비교이며, exploitability/LBR 측정이 아니다.

Root-MAE 실제 학습 gradient norm 중앙값0.00258,clip1 비율0%였다.
기존 L1의0.286보다 작은 값이나 자기 정책으로 만든 다른 배치들이므로 직접적인
고정배치 배율 실험과 구분한다. 학습시간은seed당평균40.4초(평가/저장 제외).
큰 sampled target 자체는 남아 있으며 최대 절댓값은2.62e8이었다.

8개 단위검사와32hand smoke, 최종3개 모델 재로딩/panel의SQL지표 일치,
48snapshots/replay 유한성, MSE와 초기 weights/초기 평가 배열 일치,
실행 소스 및 원래 정규화 자료 hash 불변을 확인했다. 비교 그래프도 시각검사했다.

[전체 비교와 그래프](data/raw_cfr_root_mae_compare_20260928/RESULTS.md),
[Root-MAE 단독 결과](data/raw_cfr_root_mae_20260928/RESULTS.md).

## 네 손실의 100k Hand 비교

사용자 요청에 따라 MSE/RMSE/L1/Root-MAE를 모두 seeds11/22/33 각각100,000hands로
확대한다. 총1,200,000 training hands이며 PPO와 잘못 해석했던 signed root_mean은
포함하지 않는다. 이전 체크포인트는 optimizer/RNG 재개 정보가 없는 model-only이므로
기존 결과를 덮어쓰지 않고 같은 seed/초기weights부터 새로 학습한다.

정확히100k를 맞추기 위해 수집단위를256에서250hands로 바꾸고400updates 실행한다.
32fit steps/update를 유지하므로12,800optimizer steps/seed다. 기존4,096hand실험 대비
hand당 fitting 빈도가2.4% 증가한다. 네 loss는 동일 조건이지만 이전 짧은 실행의
정확한 연장이나 예산만 바뀐 bit-identical 재현으로 부르지 않는다.
모델/raw85/정규화/Adam1e-4/batch64/탐색0.6/reservoir20k/clip1은 그대로다.
Self-play 방문분포/label은 각 정책에 따라 달라진다. Abstraction/갱신 간격 변경 없음.

5,000hands(20updates)마다 같은 monitor64deck pairs로 현재 정책을 측정한다.
최종 고정100k에서 initial/current/average 혼합을 같은 test256deck pairs로 평가한다.
중간 성능으로 조기종료/최고 checkpoint 선택을 하지 않는다. 평균은400개 pre-update
snapshot에서 hand시작에 하나를 균등선택하는 기존 방식이다. 최종 test는 이전 실험과
같아 탐색적 비교이며 새 독립 test의 확증 결과로 부르지 않는다.

프로젝트 root 실행 명령 (`$py`는 앞 절 PyTorch runtime):

```powershell
foreach ($loss in 'mse','rmse','l1','root_mae') {
    & $py -u -m agents.state_action_embedding.raw_cfr_ppo --methods cfr --cfr-loss $loss --out-dir "agents/state_action_embedding/data/raw_cfr_100k_20260928/$loss" --seeds 11 22 33 --updates 400 --hands 250 --fit-steps 32 --eval-every 20 --test-pairs 256
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $loss" }
    python -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir "agents/state_action_embedding/data/raw_cfr_100k_20260928/$loss"
}
python -m agents.state_action_embedding.raw_cfr_loss_report --mse agents/state_action_embedding/data/raw_cfr_100k_20260928/mse --rmse agents/state_action_embedding/data/raw_cfr_100k_20260928/rmse --l1 agents/state_action_embedding/data/raw_cfr_100k_20260928/l1 --root-mae agents/state_action_embedding/data/raw_cfr_100k_20260928/root_mae --out-dir agents/state_action_embedding/data/raw_cfr_100k_20260928/comparison
```

실제 실행은 위 네 학습 명령을 각각 별도 hidden process로 병렬 실행하고, process당
Torch1thread를 유지한다. 같은 상위 data폴더에 loss별 stdout/stderr 로그를 보존한다.
병렬 경합이 있으므로 시간은 이전 단독 실행과 직접 속도비교하지 않는다.
실행 중 상태는 각 loss의 `metrics.sqlite` metadata/updates에서 확인한다.
완료 결과는 아래에 추가하며 진행 중 수치를 최종 성능으로 해석하지 않는다.

### 100k 완료 결과

네 프로세스 모두 exit0,12개 학습 모두100,000hands 완료했다.
총1,200,000hands/9,211,022decision visits/4,800updates/153,600optimizer steps,
324개 평가 기록과4,800개 pre-update snapshots를 보존했다. 모든 run status는completed다.

| loss | 현재 vs heuristic | 현재 vs random | 평균 혼합 vs heuristic | 평균 혼합 vs random |
| --- | ---: | ---: | ---: | ---: |
| MSE | -23.527 | +18.114 | -18.173 | +18.393 |
| RMSE | -23.874 | +23.924 | -22.778 | +23.264 |
| L1 | -12.773 | +5.057 | -17.927 | -1.630 |
| Root-MAE | -30.637 | +8.644 | -25.866 | +9.273 |

3seed 평균chips/hand이며 높을수록 좋다. 초기 합법균등은heuristic -17.229/random +4.830.
Root-MAE는이번표본평균에서MSE보다네칸모두낮았다. 현재heuristic 차이-7.111의
paired95%구간은[-13.110,-1.234],random차이-9.470은[-18.495,-0.309]다.
Root-MAE 현재heuristic의seed SD23.167로변동도크다. 고정학습모델에조건부인
평가deck bootstrap구간이므로학습seed전체에대한일반적열세를증명하는것은아니다.

L1현재정책은heuristic상대손실이가장작았으나random수익은낮았다.
초기균등대비heuristic차이+4.455의구간[-3.000,+13.356]은0을포함한다.
L1평균혼합의random수익은음수여서일관된정책개선으로해석하지않는다.
MSE/RMSE는random수익이높지만heuristic에는여전히손실이다. 이실험은
exploitability측정이나균형수렴성공의증거가아니다.

학습gradient norm중앙값(MSE/RMSE/L1/Root-MAE)은32,400/0.426/0.486/0.00164,
clip1비율은100.0%/1.5%/23.8%/0.0%였다. 작은gradient가좋은정책을보장하지않는다.
단위검사8개재통과,12모델복원및SQL의고정panel지표일치,동일seed초기weights와
초기평가배열일치,각replay20,000개/유한target,실행source/정규화hash불변을확인했다.
새그래프의축/음수수익/범례/3seed min-max음영을시각검사했고,중첩폴더에서도
보고서의사용법링크가올바르게열리도록상대경로계산을수정했다. 학습코드는변경하지않았다.

[100k 전체 결과/신뢰구간](data/raw_cfr_100k_20260928/comparison/RESULTS.md),
[100k 학습곡선](data/raw_cfr_100k_20260928/comparison/curves.png),
[기계 판독 요약](data/raw_cfr_100k_20260928/comparison/comparison.json).
모델은각loss의`cfr_seed11`, `cfr_seed22`, `cfr_seed33` 아래`last.pt`이며
`load_model`로불러온다. Optimizer/RNG resume지원으로확장한것은아니다.

## PPO 100k Hand 확장 (2026-09-30)

사용자 요청에 따라 PPO도 seeds 11/22/33 각각 100,000 fresh self-play hands로
실행한다. 총 300,000 hands, 400 updates x 250 hands이며 기존 4,096-hand 모델의
재개가 아니라 동일 seed의 초기 가중치부터 시작한다. 256에서 250으로 수집단위가
바뀌므로 이전 짧은 실행을 그대로 연장한 것은 아니다.

위 raw85 입력/32D causal Transformer/정규화/Adam 1e-4/GAE/4epochs/batch64/
clip .2/entropy .01/gradient cap .5/KL 조기종료 .03을 유지한다. 상대는 현재 자신과
같은 공유 모델이며 양측 궤적을 학습한다. 과거 snapshot은 평가용 평균혼합에만 쓴다.
사전학습, constraint 입력, 합법성 보조 loss, abstraction은 추가하지 않았다.
Low-fold 데이터는 기존 정규화 통계와 합법성 진단 panel에만 사용한다.

Monitor는 5,000 hands마다 동일 64 deck pairs, 마지막 test는 상대별 기존 256
deck pairs를 좌석 교환한다. 마지막 고정 100k 모델과 400 pre-update snapshot의
평균혼합을 평가하며 최고 monitor 모델을 선택하지 않는다. 이전 test를 재사용하므로
탐색적 비교다. CFR와 hand/평가 예산은 같지만 실제 행동 방문/optimizer 횟수와
목적함수는 다르다. 수익은 chips/hand이며 exploitability가 아니다.

프로젝트 root에서 실행 (`$py`는 앞 절 PyTorch runtime, 새 output 경로 필요):

```powershell
& $py -u -m agents.state_action_embedding.raw_cfr_ppo --methods ppo --out-dir agents/state_action_embedding/data/raw_ppo_100k_20260930 --seeds 11 22 33 --updates 400 --hands 250 --eval-every 20 --test-pairs 256
& $py -m agents.state_action_embedding.raw_cfr_ppo_report --out-dir agents/state_action_embedding/data/raw_ppo_100k_20260930
```

학습 코드는 변경하지 않았다. 실행 로그는 family `data/`의
`raw_ppo_100k_20260930.stdout.log`, `raw_ppo_100k_20260930.stderr.log`에 남긴다.
완료 전에는 `metrics.sqlite`의 진행 상태와 monitor 측정만 존재하며 최종 결과로
해석하지 않는다. Model-only 저장이며 optimizer/RNG resume는 지원하지 않는다.

### PPO 100k 완료 결과

실행 exit0, SQLite status completed. Seeds 11/22/33 모두 100,000 hands를 마쳤다.
총 300,000 hands/1,609,411 decision visits/1,200 updates/65,425 optimizer steps,
81개 평가 기록 및 1,200 pre-update snapshots를 보존했다. 실제 학습시간 합계는
2,819.7초(평가/저장 제외), 실행 전체는 약 50분이다. 아래는 3seed 평균 chips/hand다.

| 조건 | 현재 vs heuristic | 현재 vs random | 평균혼합 vs heuristic | 평균혼합 vs random |
| --- | ---: | ---: | ---: | ---: |
| 초기 합법균등 | -17.229 | +4.830 | -17.229 | +4.830 |
| PPO 4,096 hands | -24.759 | +17.678 | -23.997 | +11.148 |
| PPO 100k hands | -19.387 | +12.988 | -18.096 | +5.146 |
| CFR MSE 100k | -23.527 | +18.114 | -18.173 | +18.393 |
| CFR RMSE 100k | -23.874 | +23.924 | -22.778 | +23.264 |
| CFR L1 100k | -12.773 | +5.057 | -17.927 | -1.630 |
| CFR Root-MAE 100k | -30.637 | +8.644 | -25.866 | +9.273 |

PPO 100k current의 seed별 heuristic/random 수익은 각각
11: -5.250/-1.127, 22: -22.064/+20.084, 33: -30.848/+20.006이다.
Current의 seed SD(ddof=0)는 heuristic10.620/random9.981이며,
평균혼합은 heuristic6.096/random4.190이다. 한 seed만 개선된 결과를 일반화하지 않는다.

초기 대비 PPO100k current 차이는 heuristic -2.159 [-10.305,+7.040],
random +8.158 [-2.235,+18.524]다. 평균혼합 차이도 각각
-0.868 [-8.861,+8.161], +0.316 [-9.277,+9.229]로 0을 포함한다.
구간은 같은 평가 deck을 seed 간 함께 resample한 paired-deal bootstrap95%이며,
고정 학습모델에 조건부다. 전체 training-seed 불확실성을 포함하지 않는다.

이전4k PPO 대비 current heuristic 차이는 +5.372 [-2.329,+14.026],
random -4.690 [-15.541,+6.444]다. CFR MSE100k 대비 current 차이도
heuristic +4.139 [-1.958,+10.008], random -5.126 [-15.618,+4.757]로 0을 포함한다.
비교에는 기존 `bc_ppo_report.summarize`를 같은 seed/deck 순서의 저장 profits 차이에
적용했다. 다중 비교와 test 재사용을 포함하는 탐색적 분석이며 우열을 확증하지 않는다.

**해석:** 예산 확대 후에도 안정적인 전략 개선을 확인하지 못했다. 이것이 역전파가
전혀 되지 않았다는 뜻은 아니다. 마스크 전 불법확률은30.86% ->25.26%,
top1 불법률은30.86% ->18.82%로 감소했으나, 직접 학습 목표가 아닌 진단이며
실제 정책 수익/균형 수렴의 대체 지표가 아니다. 최종 모델은 heuristic 상대에 여전히
손실이고 seed 간 차이가 크다. 학습곡선에도 지속적인 개선 추세는 확인되지 않는다.

단위검사8개, 모든 seed의 초기weights/초기test배열과 CFR100k 일치,
최종 모델 유한성/복원 후 합법성 panel과 SQL 일치, source/정규화 hash 불변을
검사했다. Stderr는 비어 있고 그래프의 축/음수 수익/범례/seed min-max 음영도 확인했다.
PPO의 value loss/entropy/gradient 이력을 새로 추가 기록한 실험은 아니므로,
실패 원인을 특정할 만큼의 최적화 진단을 제공한다고 주장하지 않는다.

[PPO100k 상세 결과](data/raw_ppo_100k_20260930/RESULTS.md),
[학습곡선](data/raw_ppo_100k_20260930/curves.png),
[기계 판독 요약](data/raw_ppo_100k_20260930/summary.json).
최종 모델은 `raw_ppo_100k_20260930/ppo_seed{11,22,33}/last.pt`에 있다.
