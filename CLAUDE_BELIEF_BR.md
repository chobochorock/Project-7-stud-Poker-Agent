# Claude Belief-BR: Action-Conditioned Particle-Belief Best Response

> 요약: 상대 정책을 고정된 것으로 보고, 상대의 공개 베팅 행동으로 상대 히든
> 카드 belief를 Bayesian 으로 갱신한 뒤, 모든 합법 행동의 net-chip EV 를
> 비교해 최선응답을 고르는 헤즈업 에이전트다. `Toy-Card-Game-Agent` 의 exact
> POMDP best response 를 particle belief 로 7포커에 이식한 것이며, EV 모드에서
> 고정 휴리스틱을 실제로 착취한다(약 +1.0 ante/hand).

구현: [`agent/claude_belief_br.py`](agent/claude_belief_br.py) · 테스트:
[`test_claude_belief_br.py`](test_claude_belief_br.py)

## 1. 왜 이 에이전트인가

기존 `HA1` 은 상대 히든 카드에 대해 **균등(uniform) range** 를 가정하고 showdown
equity 를 계산한다. HA1 코드에는 다음 주석이 남아 있다.

```text
opponent discard strategy is uniform here; replace with the learned discard
likelihood when action-conditioned belief is added.
```

즉 저장소가 명시적으로 비워 둔 자리는 **action-conditioned belief** 다. 그리고
루트의 [`POMDP_FIXED_HEURISTIC_BEST_RESPONSE.md`](../POMDP_FIXED_HEURISTIC_BEST_RESPONSE.md)
는 그 핵심 식을 이미 적어 두었다.

```text
b'(h) proportional to b(h) * pi_H(observed_action | h, public_state)
```

이 에이전트는 그 식을 그대로 구현한다. 상대 정책 `pi_H` 가 고정되어 있으면
상대는 더 이상 동시에 학습하는 adversary 가 아니라 숨은 상태를 가진 환경의
일부이므로, 그 belief 위에서 정확한(표본화된) 최선응답을 계산할 수 있다.

opponent model `pi_H` 로는 저장소의 `HeuristicPokerAgent` 를 그대로 재사용한다.
따라서 실제 상대가 그 휴리스틱일 때 likelihood 가 정확해지고 belief 가 상대의
진짜 range 로 수렴한다. 이것이 toy 의 "고정 휴리스틱 -> POMDP best response"
아이디어를 7포커로 옮긴 것이다.

## 2. 알고리즘

내 매 베팅 결정마다 다음을 수행한다.

### 2.1 Particle belief

1. 관측으로 알려진 dead card(내 히든/공개/버린 카드, 모든 상대 공개 카드)를
   제외한 덱에서 상대 히든 카드 가설 `h` 를 여러 개 표본화한다. (7구에서 상대
   히든은 3장, 그 전에는 2장.)
2. 각 가설을 상대의 관측된 베팅 aggression 과의 일치도로 가중한다.

   ```text
   L(h) = exp( -(strength(h) - a_obs)^2 / (2 sigma^2) )
   b'(h) proportional to L(h)
   ```

   여기서 `strength(h)` 는 opponent model 이 그 가설 손에 매기는 강도이고
   `a_obs` 는 상대가 실제로 보인 가장 강한 행동의 강도 신호(예: `HALF`≈0.82,
   `CHECK`≈0.18)다. 상대가 크게 베팅했으면 강한 가설이, 체크만 했으면 약한
   가설이 up-weight 된다. 결정론적 휴리스틱의 극한(σ→0)에서는 toy 의 "모순
   가설 제거" 하드 필터와 같아진다.

3. belief-weighted showdown equity `eq` 를 같은 표본으로 계산한다. 진단용으로
   likelihood 를 끈 `uniform_equity` 도 함께 반환한다(둘의 차이가 belief 의
   실제 이동량이다).

### 2.2 EV 모델 (net chips)

현재 팟 `P`, 콜 비용 `c`, 내 누적 투자 `i`, raise 액 `r` 에 대해:

```text
EV(FOLD)  = -i
EV(CHECK) = eq * P            - i
EV(CALL)  = eq * (P + c)      - (i + c)
EV(raise) = sum over posterior particles:
              opp folds  -> (P - i)
              opp calls  -> win(h) * (P + c + 2r) - (i + c + r)
```

`EV(CALL) > EV(FOLD)` 는 정확히 `eq > c/(P+c)`, 즉 교과서적 pot-odds 콜과 같다.
raise 의 콜 분기는 **상대가 콜하는 가설에만** 조건부 equity `win(h)` 를 쓴다.
전체 range 의 평균 equity 를 쓰면 "좋은 손에만 콜당하는데 내 승률을 과대평가"
하는 낙관 편향이 생겨 raise 가 칩을 잃는데, 이 조건부화가 그것을 제거한다.
콜/폴드 판정도 같은 opponent model 로 한다.

### 2.3 Aggression margin (규율)

위 EV 는 **1-ply(myopic)** 다. 즉 내 행동 뒤 그대로 쇼다운까지 checkdown 된다고
가정한다. 다중 스트리트로 계속 베팅하는 상대에게는 이 가정이 **얇은 밸류벳을
과대평가**한다(6·7구 HALF 남발 -> 강한 콜 range 에 걸림). 그래서 raise/bet 은

```text
EV(best raise) >= EV(best passive) + aggression_margin * (P + c)
```

를 넘을 때만 선택한다. 기본값 `aggression_margin = 0.40` 은 EV 스윕으로 고른
값이다(아래 4절). 이 규율이 없으면 belief 가 정확해도 에이전트가 칩을 흘린다.

### 2.4 CARA 위험민감 효용 (자원 오목성)

각 행동을 기대 net chips 가 아니라 **certainty equivalent(CE)** 로 점수화한다.
per-particle·per-branch outcome 분포에 CARA 효용을 씌운다.

```text
CE(a) = -(1/lambda) * log( sum_o w_o * exp(-lambda * net_o / S) ) * S
      ~= E[net] - (lambda / 2) * Var[net] / S
```

`S` 는 유효 스택(`_risk_scale`), `lambda = risk_lambda`. **스택으로 정규화**하므로
lambda 는 "스택 한 개당 위험회피" 로 mode-agnostic 하고, 깊은 스택에서는 작은
팟에 거의 위험중립이며 **스택의 큰 비중을 거는 대형 팟·올인에서만** 강하게
작동한다. `lambda = 0` 이면 선형 EV 와 완전히 동일하다.

이는 `aggression_margin` 의 손튜닝을 원리적으로 보완한다. margin 은 EV 모드의
얇은 밸류벳 누수를, CARA 는 cash 모드의 대형 팟 분산 누수를 잡는다(서로 다른
누수). 기본값은 `risk_lambda = 12.0`, `aggression_margin = 0.40` 조합이다.
설계 근거는 [`CONCAVE_UTILITY_IDEAS.md`](CONCAVE_UTILITY_IDEAS.md) P1 참고.

`CE >= min(outcome)` 이 항상 성립하므로, CHECK 가능 시 CARA 가 공짜 쇼다운을
버리고 FOLD 하지 않는다(CHECK 의 최악 결과 = FOLD 확정값). CALL/RAISE 는 칩을 더
걸므로 CARA 가 올바르게 더 수동적으로 만들 수 있다.

## 3. 상태 공간과 범위

- belief 갱신은 **헤즈업(생존 상대 1명)** 을 대상으로 한다. 이론이 깨끗한
  설정이다. 생존 상대가 여럿이면 균등 range equity 최선응답으로 저하되며 fold
  equity 를 취하지 않는다(보수적). 이것이 이 에이전트의 기여 부분은 아니다.
- 학습하지 않는다. planning oracle 이며 상대 정책이 고정일 때만 best-response
  보장이 의미 있다. 상대가 동시에 학습하면 belief transition 과 target 이
  움직여 이 보장은 사라진다(POMDP 문서 10절과 동일한 한계).

## 4. 실험 결과

### 4.1 EV 모드 (선형 net chips)

EV 모드(헤즈업, 스택 없는 제로섬 net chips, `ante=1000`)에서 고정
`HeuristicPokerAgent` 상대로 측정했다. duplicate pairing(같은 덱을 좌석만 바꿔
두 번 플레이)으로 덱 분산을 줄이고 여러 시드로 반복했다. 단위는 ante/hand,
양수면 휴리스틱을 이긴다는 뜻이다.

| agent | vs heuristic (ante/hand) |
| --- | --- |
| **claude belief-BR (margin 0.40, lambda 0)** | **+1.024** (시드별 +0.287, +0.991, +1.793) |
| HA1 (uniform belief) | -10.474 +/- 3.973 |
| random | -6.334 +/- 2.252 |

- 세 시드 모두 양수이고 평균 **+1.024 ante/hand** 로, 고정 휴리스틱을 실제로
  **착취한다**. toy 의 exact POMDP best response(+0.715 ante/hand)와 같은 성격의
  결과다.
- 균등 belief 인 HA1 대비 **약 +11.5 ante/hand** 개선이며 이 격차는 통계적으로
  결정적이다. 즉 action-conditioned belief + 규율 있는 best response 가 승패를
  가른다.
- 절대 우위폭(약 +1)은 modest 하고 시드 분산이 있어 일부 시드의 95% 구간은 0 을
  살짝 포함한다. "압도적 지배" 가 아니라 "확실히 이기는 쪽" 으로 읽어야 한다.

aggression_margin 스윕(belief-BR vs heuristic, EV 모드):

```text
margin 0.00 : -5.7      (공격적, 얇은 밸류벳 남발 -> 큰 손실)
margin 0.20 : -2.8
margin 0.35 : +0.5
margin 0.40 : +0.5      (기본값, 분산 더 낮음)
margin 0.50+: -0.35     (거의 항상 passive, break-even 부근)
```

belief 자체의 정확성은 단위 테스트로 검증한다. 경계 손(페어 8)에서 같은
particle 로 likelihood on/off 를 비교하면, 공격적 상대는 내 equity 를 uniform
아래로(강한 range 로 판단), 체크만 한 상대는 uniform 위로(약한 range) 민다.

### 4.2 Cash 모드 (CARA 위험민감, 실제 스케일 1M 칩 / ante 1000)

실제 운영 스케일(스택 1,000,000 칩 = 1000 antes, ante 1000)의 cash 모드에서
duplicate pairing 으로 측정했다. cash 는 `FULL` 을 허용해 팟이 더 커지므로 선형
에이전트가 대형 팟에서 과투자하고, CARA 가 이 누수를 잡는다.

| 설정 | CASH 시드별 (ante/hand) | EV (ante/hand) |
| --- | --- | --- |
| 선형 + margin 0.40 (lambda 0) | +1.24 / -4.28 / -3.30 (평균 음수) | +1.02 |
| CARA lambda 16 (margin 0) | +0.81 / +3.67 | -0.72 (EV 회귀) |
| CARA lambda 32 (margin 0) | +1.69 / +4.04 (cash 최고) | 미측정 |
| **margin 0.40 + lambda 12 (기본값)** | **+1.55 / +4.43** | **+0.84** |

- lambda 를 키우면 cash 평균이 오르고(누수 교정) **분산도 준다**. lambda 0 -> 16 에서
  분산이 ±7.9 -> ±1.35 로 줄었다. Jensen 오목성이 그대로 나타난 것이다.
- **cash 는 높은 lambda, EV 는 낮은 lambda 를 원하는 tradeoff** 가 있다. 순수 CARA
  lambda 32 는 cash 최고지만 EV 를 해치고(lambda 16 에서 이미 EV 음수), margin 0.40
  + lambda 12 조합만 두 모드 모두 양수다. margin 은 EV 의 얇은벳 누수를, CARA 는
  cash 의 대형팟 분산 누수를 각각 잡기 때문이다.
- 따라서 기본값을 `margin 0.40 + lambda 12` 로 정했다. cash 우선이면서 EV 를 양수로
  유지하는 지점이다. cash 를 더 극대화하려면 `risk_lambda` 를 16~32 로 올리되 EV
  저하를 감수한다.

## 5. 발견

1. **belief 는 필요하지만 충분하지 않다.** 정확한 action-conditioned belief 만으로
   HA1 을 −10.5 에서 크게 끌어올리지만, myopic EV 는 그대로 두면 얇은 밸류벳으로
   칩을 흘린다. aggression margin 이라는 규율을 더해야 비로소 +가 된다.
2. **다중 스트리트 상대에게 1-ply 는 구조적으로 밸류벳을 과대평가한다.** 진 핸드에
   더 많이 투자하는 패턴(평균 투자 16.4 vs 12.75 ante)이 그 증거였다. margin 은
   이 증상을 막는 blunt 하지만 효과적인 처치다.
3. **자원(칩)의 오목 효용이 cash 의 대형 팟 누수를 잡는다.** cash 는 `FULL` 로 팟이
   커져 선형 에이전트가 큰 팟에서 과투자하는데(구 기본값 cash 평균 음수), CARA
   위험민감 효용을 넣으면 평균과 분산이 동시에 개선된다. 스택 정규화 덕에 깊은
   스택은 거의 위험중립이고 대형 팟에서만 작동한다. margin 과 CARA 는 서로 다른
   모드의 다른 누수를 잡으므로 조합이 두 모드 모두에서 양수다.

## 6. 한계와 다음 단계

- `aggression_margin` 과 `risk_lambda` 는 myopic 편향과 분산을 보정하는 처치다.
  근본 해법은 **다중 스트리트 rollout / IS-MCTS**: 각 belief particle 에서 남은
  스트리트를 opponent model 로 실제로 플레이아웃해 정확한 chip 을 정산하면, reverse
  implied odds 와 얇은 밸류벳 함정이 EV 에 자동 반영되어 margin 이 필요 없어진다.
  CARA 는 그 rollout return 에도 그대로 씌울 수 있다.
- 콜 분기 equity 만 콜 range 에 조건부화했고, CALL/CHECK 라인의 미래 베팅은 아직
  모델링하지 않는다. rollout 이 이것도 함께 해결한다.
- 헤즈업 전용 belief 를 다인전 particle 공유 belief 로 확장.
- 확률적 상대에 대한 soft-likelihood 보정(현재는 σ 고정).

## 7. 재현

```powershell
cd D:\Experiment\Project-7-stud-Poker-Agent

# 단위 테스트 (belief 이동, 밸류벳/폴드, 제로섬 통합)
python -B -m unittest -v test_claude_belief_br.py

# EV 모드 헤즈업 착취 측정
python -B evaluate_heads_up.py --agent-a claude --agent-b heuristic --mode ev --ante 1000 --hands 800 --simulations 120

# cash 모드(실제 스케일) 측정 -- CARA 위험민감 기본값
python -B evaluate_heads_up.py --agent-a claude --agent-b heuristic --mode cash --starting-chips 1000000 --ante 1000 --hands 800 --simulations 120

# 한 판 실행
python -B main.py --mode ev --rounds 10 --ante 1000 -p1 claude -p2 heuristic
```
