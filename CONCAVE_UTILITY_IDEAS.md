# 자원(칩)에 대한 오목 효용: 적용 가능성 보고서

> 목적: "칩의 효용을 선형이 아니라 오목(concave)하게 두면 좋다"(ICM류)는
> 아이디어를 현재 저장소 코드에 어디까지 적용할 수 있는지 정리한다. 이 문서는
> 설계 노트이며 **코드를 바꾸지 않는다.** 실제 구현은 각 제안의 검증 계획대로
> 별도로 진행한다.

관련 문서: [`CLAUDE_BELIEF_BR.md`](CLAUDE_BELIEF_BR.md),
[`CLUSTER_AGENT_THEORY.md`](CLUSTER_AGENT_THEORY.md),
루트 [`POMDP_FIXED_HEURISTIC_BEST_RESPONSE.md`](../POMDP_FIXED_HEURISTIC_BEST_RESPONSE.md)

## 1. 개념 요약

칩의 효용 `U(chips)`가 오목이면 Jensen 부등식으로

```text
E[U(X)] < U(E[X])            (U 오목)
```

이 되어, **같은 기대 칩이라도 분산이 큰 도박의 효용이 낮다** → 위험 회피가
자동으로 인코딩된다. 잃는 칩이 따는 칩보다 더 아프다는 성질이다.

세 가지 표준형:

- **CARA / 지수 효용** `U(x) = (1 - exp(-λx)) / λ`
  - `E[U] 최대화 ≈ E[X] − (λ/2)·Var[X]` (가우시안이면 정확). 곧 "기대값 −
    분산 페널티". λ 하나로 위험 성향 조절. `λ→0`이면 선형 EV로 환원.
- **로그 / Kelly** `U(x) = log(x)`. 뱅크롤 성장률 최적. 판돈 크기 결정용.
- **ICM (Independent Chip Model)** 토너먼트용. 칩 벡터 → 순위 확률
  (Malmuth-Harville) → 상금 기대값. `보상 = ΔICM`.

## 2. 코드 인벤토리: 자원(칩)이 들어오는 지점

| 위치 | 값 | 현재 형태 |
| --- | --- | --- |
| [`poker_env.py:713`](poker_env.py:713) | `reward = player.chips - hand_start_chips` | 선형 net chips |
| [`poker_env.py:403`](poker_env.py:403), `:419` | 상태 feature `my_chips`, `effective_stack` | 자원 feature 원본 |
| [`ev_rollout.py:178`](ev_rollout.py:178) | `pot, my_invested, opponent_invested, call_amount` | scalar 자원 features |
| [`agent/claude_belief_br.py`](agent/claude_belief_br.py) `_action_ev`/`_raise_ev` | `win·final_pot − cost` | 선형 EV |
| [`clustering_train.py:249`](clustering_train.py:249) | `q_scale = max(ante, pot)` | **스케일 정규화(≠ 효용)** |

두 가지 구분을 분명히 한다.

- **효용은 확률이 아니라 칩에 씌운다.** HA1의 `equity`는 이미 `[0,1]` 승률이므로
  대상이 아니다. 대상은 칩/스택/net chips다.
- **스케일 정규화 ≠ 오목 효용.** `clustering_train.py`의 `max(ante, pot)` 나눗셈은
  모든 행동을 같은 상수로 나눠 **행동 순위를 보존**한다. 오목 효용은 반대로
  **순위를 바꿔** 고분산 행동을 감점한다. 둘을 합칠 때는 `U`를 return에 먼저
  적용하고 정규화는 그 뒤에 별도로 다뤄야 한다(5절 open question).

## 3. 모드별 적용 가능성

| 모드 | 자원 성격 | 오목 효용 적용 | 근거 |
| --- | --- | --- | --- |
| **EV** | 선형 제로섬 net chips | **적용 금지** | 벤치마크 objective 자체가 선형. 넣으면 기준 왜곡 |
| **cash** | 매 라운드 1000칩 리셋 = 현금 | **불필요** | 파산 위험 없음. 칩=money 선형 |
| **tournament** | 스택 이월, 탈락 존재 | **이론상 올바른 자리** | 생존 가치·차등 상금이 오목성을 만듦 |

**토너먼트의 함정:** 현재 tournament은 "마지막 1인 = winner-take-all"
([README](README.md) 19행)이다. winner-take-all에서 ICM은

```text
P(i가 1등) = c_i / T        →  $EV_i = prize · c_i / T   (c_i에 선형)
```

이라 **오목 이득이 0이다.** ICM의 오목성은 **상금 사다리(상위 2~3등 차등 지급)**
에서만 나온다. 따라서 tournament에 ICM을 넣으려면 **먼저 분할 상금 구조를
추가**해야 실제 효과가 생긴다.

## 4. 지금 할 수 있는 것 (제안)

### 제안 P1 — belief-BR의 EV를 CARA 효용으로 ✅ 구현·검증 완료

> **상태:** [`agent/claude_belief_br.py`](agent/claude_belief_br.py)에 `risk_lambda`
> 로 구현. certainty equivalent 로 각 행동을 점수화하며, 스택 정규화(`_risk_scale`)
> 로 λ가 mode-agnostic 하다. 기본값 `margin 0.40 + λ=12`. 결과 요약: cash(실제
> 스케일 1M/1000)에서 구 기본값(λ0)은 시드 평균 음수(−4.28, −3.30)였으나 CARA 로
> **cash +1.55/+4.43, EV 여전히 +0.84** 로 두 모드 모두 양수. λ↑ 시 cash 평균 상승
> 과 분산 감소(±7.9→±1.35)가 동시에 나타나 오목성이 그대로 확인됐다. 자세한 표는
> [`CLAUDE_BELIEF_BR.md`](CLAUDE_BELIEF_BR.md) 4.2절. 아래는 최초 설계 메모다.

현재 `aggression_margin = 0.40`은 분산을 ±6 → ±0.5로 줄였는데, 이는 사실
**오목 효용의 손튜닝 대용품**이다. 원리적 대체:

```text
현재:  score(a) = Σ_particles b(h) · [선형 chip 결과];  raise는 margin으로 게이팅
변경:  score(a) = Σ_particles b(h) · U(net_chips 결과),  U = CARA(λ);  argmax 그대로
```

- **삽입 위치:** [`agent/claude_belief_br.py`](agent/claude_belief_br.py)의
  `_action_ev` / `_raise_ev`에서 per-particle·per-branch(fold/call) chip delta를
  합산하는 부분에 `U()`를 씌워 `E[U]`로 만든다. `choose_action`의 margin 게이팅은
  제거하거나 `λ`와 함께 남겨 비교한다.
- **효과:** 크게 딸/잃을 all-in·얇은 밸류벳이 스스로 `−`효용이 되어 잘려나간다.
  magic constant(0.40) 없이 분산 억제가 원리적으로 나온다. `λ→0`은 선형 EV.
- **검증:** EV 모드 duplicate-paired 평가에서 `λ`를 스윕(ante 단위 0.01~0.2)하고,
  현재 기준선(**margin 0.40 → +1.024 ante/hand, 시드별 +0.29/+0.99/+1.79**)의
  평균과 CI 폭을 함께 비교. 목표: 손튜닝 margin 없이 평균 유지 + 분산 축소.
- **주의/한계:** EV 모드에서 U를 "핸드 net chips"에 적용하면 **핸드 내부 위험
  회피**다(진짜 위험은 뱅크롤 단위). 벤치마크로서의 EV는 선형이 정답이므로, 이
  CARA 버전은 **별도 에이전트 옵션(`risk_lambda`)**으로 두고 선형 기준선과 항상
  같이 보고한다. 기존 선형 EV 결과를 덮어쓰지 않는다.
- **작업량/위험:** 작음 / 낮음. 파일 1개, 순수 planning 변경, 학습 없음.

### 제안 P2 — tournament에 분할 상금 + ICM 보상 (중간 규모)

- **선결 조건:** tournament에 **상금 사다리**(예: 상위 2등 60/40) 설정 추가. 이게
  없으면 3절 함정대로 ICM이 선형이라 무의미.
- **삽입 위치:** [`poker_env.py:713`](poker_env.py:713)의 터미널 보상을 tournament
  모드에서만 `ICM($equity_after) − ICM($equity_before)`로 교체. Malmuth-Harville
  1등 확률 `c_i/Σc`에서 재귀로 k등까지 전개.
- **효과:** 탈락 직전 스택 보존, 마진 콜/폴드 임계가 스택 분포에 따라 이론적으로
  올바르게 움직인다. 학습 에이전트(uct/cluster)의 보상 타깃이 위험 인지형이 된다.
- **검증:** ICM 함수 단위 테스트(합=총상금, 단조성, 대칭성). tournament 자기대전에서
  선형-칩 보상 대비 탈락 순번·최종 순위 분포 비교.
- **주의/한계:** 게임 규칙(상금 구조) 변경이므로 EV/cash에는 영향 없어야 한다.
  범위가 P1보다 큼.
- **작업량/위험:** 중간 / 중간(규칙 변경 파급).

### 제안 P3 — 학습 타깃의 위험 민감화 (탐색적)

- **아이디어:** uct/cluster의 터미널 return을 Q 타깃으로 쓰기 전에 `U(return)`로
  감싸 위험 민감 학습을 만든다.
- **주의:** [`clustering_train.py:249`](clustering_train.py:249)의 `max(ante, pot)`
  정규화와 **순서 문제**가 있다. `U`는 비선형이라 "정규화 후 U"와 "U 후 정규화"가
  다르다. return을 먼저 `U`로 변환하고, 스케일은 위험 척도와 분리해 재설계해야
  한다. 성급히 합치면 [`CLUSTER_AGENT_THEORY.md`](CLUSTER_AGENT_THEORY.md)의
  Bellman 재정규화 유도가 깨진다.
- **작업량/위험:** 중간 / 높음(학습 파이프라인·이론 재검토 필요). **후순위.**

## 5. 열린 질문

- ~~CARA `λ`를 어느 자원 단위로 정의할지~~ → **해결됨: 유효 스택 정규화**
  (`_risk_scale`). 깊은 스택은 per-hand 거의 위험중립, 대형 팟에서만 작동해
  mode-agnostic 하다.
- P3에서 오목 효용과 pot 스케일 정규화를 어떤 순서로 결합할지(비선형성 때문에
  순위·Bellman 성질에 영향).
- tournament 상금 사다리를 게임 규칙에 고정할지, 실험 설정으로 둘지.

## 6. 권장 순서

1. ✅ **P1 (CARA belief-BR)** — 구현·검증 완료. 기본값 `margin 0.40 + λ=12` 로 cash
   (실제 스케일)에서 구 기본값의 음수 평균을 양수로 돌리고 EV 도 양수 유지.
2. **P2 (tournament ICM)** — 단, **분할 상금부터**. 그 전에는 오목 이득이 없다.
3. **P3 (학습 타깃)** — P1 의 스택 정규화 방식을 재사용하되, pot 정규화와의 순서
   문제를 정리하고 착수.

핵심 한 줄 (갱신): **효용은 "칩"에 씌우되 스택으로 정규화한다. EV 벤치마크는
선형이 정답이라 낮은 λ, cash 는 대형 팟 분산 때문에 높은 λ 를 원하므로, margin
(EV 누수) + CARA(cash 누수) 조합이 두 모드 모두에서 양수다. tournament 는 분할
상금을 넣은 뒤 ICM(P2) 로 확장한다.**
