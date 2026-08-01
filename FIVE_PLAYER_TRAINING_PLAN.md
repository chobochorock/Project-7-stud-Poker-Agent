# 5인 Stud Poker 학습 및 평가 계획

작성일: 2026-07-31

## 1. 목표

5인 cash game 정책을 하나의 승률이나 한 상대에 대한 EV로 평가하지 않는다. 다음 다섯 축을
동시에 관리한다.

1. 자기대국 안정성: 다른 정책이 단독으로 이탈했을 때 얻는 이득이 작은가.
2. 비공모 방어력: 한 LBR이 네 개의 정책 복사본을 얼마나 착취하는가.
3. 필드 착취력: 약점을 가진 상대가 포함된 필드에서 칩을 얼마나 얻는가.
4. 공모 방어력: 정보를 직접 공유하지 않는 4인 팀이 목표 정책을 얼마나 공격할 수 있는가.
5. 스택 강건성: 활성 플레이어 수와 스택이 변해도 EV와 파산 위험이 안정적인가.

초기에는 다섯 지표를 가중합하지 않고 Pareto 지표로 보존한다. 특히 fish EV를 높이면서 LBR
방어력이 크게 나빠지는 모델은 채택하지 않는다.

5인 학습의 기본 산출물은 최신 단일 체크포인트가 아니라 **정책 population과 그 위의
meta-distribution**으로 둔다. 동일 정책 다섯 개의 self-play는 유용한 데이터 생성 방법이지만,
그 최신 정책이 5인 Nash equilibrium에 가까워진다는 보장은 없다. 단일 배포 정책이 필요할 때에는
population 혼합을 한 hand 단위로 표본화하거나, 별도의 검증을 통과한 대표 정책을 사용한다.

## 2. 평가 매트릭스

| ID | 구성 | 주 지표 | 좋은 방향 | 해석 |
|---|---|---:|---:|---|
| A | 5 target self-play | 근사 NashConv | 낮음 | 단독 정책 변경으로 얻을 수 있는 총 이득 |
| B | 4 target + 1 LBR | LBR EV | 낮음 | 일반적인 비공모 방어력의 하한 |
| C | 4 target + 1 fish | target 합산 EV, fish 손실 | 높음 | 서로 싸우면서 fish를 살려두는지 검사 |
| D | 1 target + 4 team-BR | target EV | 높음 | 비공개 정보 공유 없는 공모 공격에 대한 방어 |
| E | 1 target + 4 fish | target EV | 높음 | 소프트 필드에서의 개인 수익성 |
| F | target 수 1, 2, 3, 4 | 인원별 EV 곡선 | 완만함 | 필드가 강해질 때 edge가 무너지는 모양 |
| G | 활성 인원 2, 3, 4, 5 | 인원별 EV/LBR | 안정적 | 폴드 후 남은 subgame의 강건성 |
| H | 과거 체크포인트 population | 최악/평균 EV | 높음 | 순환적 약점과 망각 검사 |

### 2.1 5 self-play의 올바른 지표

대칭적인 5 self-play에서 플레이어들의 칩 수익 합은 항상 0이다. 따라서 평균 self-play EV는
개선 지표가 아니다. 대신 다음 근사 NashConv를 사용한다.

\[
\widehat{\mathrm{NashConv}}(\pi)
=
\sum_{i=1}^{5}
\left[
\widehat{BR}_i(\pi_{-i})-V_i(\pi)
\right].
\]

정확한 best response가 어려우므로 각 자리에서 동일 예산의 LBR 또는 PSRO response를 학습해
하한을 구한다. 함께 보고할 값은 다음과 같다.

- 자리별 EV와 최대 자리 편향
- LBR deviation gain의 합과 최댓값
- 과거 체크포인트에 대한 cross-play 행렬
- 행동 및 도달 정보집합의 다양성

이 값은 단일 정책 profile의 진단값이다. 5인 self-play의 최신 체크포인트를 자동 승격시키는
목표함수로 사용하지 않는다.

### 2.2 4 target + 1 LBR

LBR 한 명은 네 target의 공개 행동을 관측하고 자신의 hidden-card belief를 갱신한다. 네 target은
각자 정상적으로 플레이하며 공모하지 않는다. LBR을 모든 자리에 순환시켜 평균한다.

이 값은 exact exploitability가 아니라 제한된 response가 찾아낸 착취율 하한이다. 모든
체크포인트는 같은 hands, particles, compute budget으로 비교한다.

### 2.3 4 target + 1 fish

target 네 명의 개인 EV뿐 아니라 합산 EV를 본다. 합산 EV는 zero-sum이므로 fish 손실과 같다.
네 target이 서로 과도하게 싸워 fish를 살려두면 합산 EV가 낮아진다.

### 2.4 1 target + 4 team-BR

네 공격자는 팀 전체 칩의 합을 최대화한다. 팀원의 hidden card와 내부 상태는 공유하지 않고,
공개된 게임 행동만 서로 관측한다. 이는 공통 보상 Dec-POMDP이며 CTDE 또는 교대 best response로
근사한다.

이 평가는 일반적인 exploitability가 아니다. 공모 가능한 상대에 대한 더 강한 security stress
test이다. 팀은 행동을 신호로 사용할 수 있으므로 실제 비공모 필드보다 강하다.

## 3. 단단하지만 약점이 있는 fish 만들기

완전한 규칙 기반 heuristic을 여러 개 새로 작성하지 않는다. 기준 정책 \(\pi_0\)에 제한된
행동 편향을 주어 재현 가능한 fish population을 만든다.

\[
\pi_\theta(a\mid I)
\propto
\pi_0(a\mid I)
\exp\left(\theta^\top f(I,a)\right).
\]

여기서 \(f\)는 다음과 같은 leak feature다.

- 과도한 fold 또는 부족한 fold
- 과도한 call인 calling station
- 과도한 bluff 또는 부족한 bluff
- 특정 street의 과도한 aggression
- 작은 bet 또는 큰 bet 선호
- 약한 hand에서의 과도한 지속
- 강한 hand에서만 공격하는 투명한 range

\(D_{KL}(\pi_\theta\|\pi_0)\)는 perturbation 크기를 제한하는 정규화로만 사용한다. KL이 작아도
높은 reach 또는 큰 pot의 결정 하나가 크게 새면 쉽게 착취될 수 있고, 반대의 경우도 가능하므로
KL을 fish 난이도로 해석하지 않는다. 난이도는 고정 예산 response의 실제 이득 또는 target
population에 대한 `ante/hand`로 사후 보정한다. 예를 들어 LBR에 대해 `0.1`, `0.5`, `1`,
`2 ante/hand` 정도로 착취되는 정책을 각각 선별한다.

권장 opponent pool은 다음 조합이다.

```text
기준 heuristic
+ KL-제한 leak 정책들
+ 과거 self-play 체크포인트
+ 현재 정책의 제한된 LBR/PSRO response
+ 소량의 random explorer
```

이 방식은 특정 heuristic 하나에 과적합되는 문제를 줄이고, 약한 이유를 파라미터로 설명할 수
있다.

## 4. Population self-play와 JPSRO

### 4.1 핵심 구조

5인에서는 하나의 최신 정책을 계속 교체하기보다 다음 population 반복을 기본 학습 루프로 둔다.

```text
초기 policy pool
    -> 표본화한 5인 policy profile 평가
    -> sparse empirical meta-game
    -> CCE 계열 meta-distribution 계산
    -> meta-distribution에 대한 MCCFR response 학습
    -> 유효한 response만 pool에 추가
    -> 반복
```

초기 pool은 새로 만들지 않고 이미 존재하는 자산을 재사용한다.

- hard K64 100M과 주요 과거 체크포인트
- warm K128 이후 정책
- 기준 heuristic과 보정된 fish 정책
- 현재 정책을 상대로 학습한 제한된 LBR/response
- 필요할 때만 소량의 random explorer

이는 [PSRO](https://arxiv.org/abs/2403.02227)의 제한된 정책공간 확장과,
n-player extensive-form game에서 CE/CCE meta-solver를 사용하는
[JPSRO](https://arxiv.org/abs/2106.09435)를 현재 MCCFR 인프라에 맞게 축소한 형태다.

### 4.2 Empirical meta-game

정책이 `K`개라고 해서 `K^5` 조합을 전부 평가하지 않는다. 현재 meta-distribution에서 자주
표본화되는 profile, 새 정책이 포함된 profile, held-out stress profile만 duplicate seat rotation으로
평가하여 sparse payoff table을 유지한다.

첫 구현은 작은 support에서 no-regret update로 CCE를 근사한다. support가 작고 정확한 선택이
필요할 때만 CCE 선형계획 또는 JPSRO meta-solver를 추가한다.

정책 혼합은 우선 hand 시작 시 complete policy 하나를 표본화하고 hand가 끝날 때까지 유지한다.
정보집합마다 체크포인트의 action probability를 단순 평균하면 한 정책 안의 bluff/value line
상관구조가 깨질 수 있으므로 첫 구현에서는 사용하지 않는다.

### 4.3 MCCFR response oracle

새 response는 한 학습 자리의 기존 MCCFR 정책에서 warm start한다. 상대 네 자리는 매 hand마다
meta-distribution에서 policy profile을 표본화하고, 학습 자리는 다섯 좌석을 순환한다. 정확한 best
response가 아니어도 다음 조건을 만족하면 population에 추가한다.

- 현재 meta-distribution에 대한 unilateral deviation gain이 양수임
- 기존 population에 없는 cross-play 행을 만듦
- held-out profile에서 심각한 성능 붕괴가 없음

같은 상대 혼합에 대한 추가 학습으로 response gain이 더 이상 증가하지 않으면 oracle을 종료한다.
Population 자체의 순환성과 상성은 cross-play matrix를 주 지표로 보고, 필요하면
[Alpha-Rank](https://arxiv.org/abs/1903.01373)를 보조 진단으로 사용한다. Alpha-Rank는
exploitability나 학습 목표를 대신하지 않는다.

### 4.4 역할 구분

- 5-copy self-play: rollout과 초기 정책 생성
- JPSRO/CCE population: 주 학습 구조
- LBR: unilateral response 하한과 새 response 후보
- team-BR: 공모 방어 stress test만 담당
- fish ladder: exploitation 성능과 field 적응 평가

## 5. 활성 플레이어 수 2~5 처리

5인 게임에서 세 명이 fold하여 heads-up이 된 상태는 새로운 2인 게임과 같지 않다. 이미 생긴
dead money, 각자의 투자액, 좌석, action history와 fold를 유발한 range 정보가 남아 있기 때문이다.

따라서 fold bot을 넣은 2/3/4인 환경은 유용한 curriculum이지만 정확한 등가는 아니다.

공유 정책의 입력 또는 bucket에는 최소한 다음을 포함한다.

- 활성 플레이어 mask와 활성 인원 수
- target 기준 상대 좌석 순서
- 각 플레이어의 `log1p(stack / ante)`
- 각 플레이어의 투자액과 all-in/fold 상태
- pot의 dead money와 call amount
- street와 공개 action history
- 모든 공개 카드에서 계산한 hand-power/range feature

### 5.1 기존 2인 정책 연결

활성 인원이 정확히 두 명이 되면 현재 heads-up 정책을 다음 순서로 재사용한다.

1. heads-up 정책을 leaf value 또는 action prior로 사용한다.
2. 실제 5인 trajectory에서 두 명만 남은 상태를 별도로 수집한다.
3. dead money, position, stack과 range가 포함된 상태에서 fine-tune한다.
4. 새 조건부 heads-up 정책이 충분히 학습되면 prior의 영향력을 감쇠한다.

현재 2인 정책으로 즉시 교체하지 않는다. 기존 모델은 fresh heads-up root 분포에서 학습되어
5인 게임에서 유도된 heads-up belief를 정확히 나타내지 못하기 때문이다.

### 5.2 권장 curriculum

```text
5인 + fold bot 3명
-> 5인 + fold bot 2명
-> 5인 + fold bot 1명
-> 완전한 5인 게임
```

각 단계에서 이전 정책을 warm prior로 사용한다. 최종 학습 데이터에는 실제 self-play에서
자연스럽게 2/3/4명으로 줄어든 subgame도 반드시 포함한다.

## 6. 스택과 log 효용

### 6.1 기본 모델

주 solver는 선형 chip EV를 유지한다. 이 경우 게임은 constant-sum이고 기존 regret 기반 평가와
직접 비교할 수 있다.

각 플레이어의 스택은 독립적으로 `50~1000 ante`에서 log-uniform하게 표본화한다. 정책에는
절대 스택보다 `log1p(stack/ante)`, effective stack, pot 대비 잔여 스택을 입력한다.

### 6.2 log-risk 실험 모델

올인 방지 압력을 직접 실험할 때에는 별도 모델에서 다음 효용을 사용한다.

\[
U_i
=
\log
\frac{s_i+\Delta_i+\epsilon}{s_i+\epsilon}.
\]

스택을 모두 잃는 결과에는 매우 큰 음의 보상이 생긴다. 단, 이 변환은 chip zero-sum 구조를
깨뜨리므로 기존 exploitability와 동일하게 비교하지 않는다. 다음을 함께 보고한다.

- 선형 chip EV
- 평균 log growth
- stack floor 도달률
- all-in 시도율과 all-in 패배율
- 최대 drawdown과 생존 hand 수
- 근사 NashConv 또는 unilateral deviation gain

로그 효용은 스택이 hand 사이에 지속되는 bankroll episode에서 측정한다. 매 hand 스택을
초기화하면 장기 성장률이 아니라 단순한 hand-level risk aversion 실험이 된다.

## 7. 공통 평가 규칙

모든 5인 비교에는 다음 규칙을 적용한다.

1. 같은 deal을 target 자리만 바꿔 다섯 번 재생하는 duplicate seat rotation을 사용한다.
2. 평균뿐 아니라 paired standard error와 95% 신뢰구간을 출력한다.
3. LBR와 team-BR는 체크포인트마다 같은 계산 예산을 사용한다.
4. target 수가 다른 실험은 총 target EV와 target 1인당 EV를 함께 출력한다.
5. 학습에 사용한 opponent와 평가 전용 held-out opponent를 분리한다.
6. 승률보다 `ante/hand`, LBR gain, NashConv와 위험 지표를 우선한다.

## 8. 채택 기준

새 response 정책은 population에 추가하기 전에 다음 gate를 모두 만족해야 한다. 최신 정책 하나가
기존 population 전체를 교체하지 않는다.

1. 4 target + 1 LBR의 LBR EV가 이전 모델보다 악화되지 않는다.
2. 1 target + 4 fish의 target EV가 통계적으로 개선된다.
3. 4 target + 1 fish의 합산 EV도 개선되어 target끼리의 과도한 경쟁이 없다.
4. 활성 인원 2~5 곡선에서 특정 인원 수의 급격한 붕괴가 없다.
5. team-BR에 대한 target EV가 허용 한도 아래로 악화되지 않는다.
6. log-risk 모델은 chip EV 손실과 파산 위험 감소를 별도로 보고하고 자동 승격하지 않는다.
7. 현재 meta-distribution에 대한 response gain 또는 population coverage가 통계적으로 개선된다.

## 9. 구현 순서

1. 5인 duplicate evaluator와 활성-player mask를 먼저 고정한다.
2. 현재 heuristic과 2인 전이 정책으로 A~H 평가 표의 빈 baseline을 채운다.
3. 기존 체크포인트, heuristic, LBR response로 작은 초기 policy pool을 만든다.
4. 표본화 profile의 sparse cross-play table과 단순 CCE meta-distribution을 만든다.
5. 현재 MCCFR을 warm-start response oracle로 연결해 population 반복을 시작한다.
6. 기준 정책의 logit perturbation으로 fish를 만들고 실제 response EV로 난이도를 보정한다.
7. 단일 LBR를 held-out response 평가기로 유지한다.
8. 교대 team-BR를 구현해 공모 stress test를 추가한다.
9. 마지막으로 persistent-stack log-risk 모델을 별도 실험한다.

첫 성공 기준은 한 숫자의 최고점이 아니다. **LBR 방어력을 보존하면서 fish field EV와
활성 인원 강건성을 동시에 개선하고, population의 CCE deviation gain을 낮추는 것**이다.
