# 실험 계획서 (검토용)

> 상태: **계획만**. 구현·실행 없음. 검토 후 착수 여부를 결정.
> 원칙: 기존 코드 **수정·삭제 금지**. 새 파일 또는 브리지 드라이버로만 진행
> (`#define STUD_MCCFR_NO_MAIN` + `#include "stud_mccfr.cpp"`).

---

## 0. 확정된 사실 (이 계획의 전제)

| 항목 | 결과 | 근거 |
| --- | --- | --- |
| 빌드 | **`-static-libstdc++ -static-libgcc` 필수.** 환경의 libstdc++ 깨짐 | `--self-test` 통과 확인 |
| 측정 규약 | `prog \| tail; echo $?` 는 **tail 의 종료코드**. `prog > f 2>&1; echo $?` 로 읽을 것 | 오진 1건 발생 |
| #0 raise cap | 키에 정상 포함 (`street`,`own/opponent_bet_count`,`legal_mask`,`checked`) | `make_power_key` 1868–1910 |
| #1 centroid 정규화 | 효과 있음: distortion **−1.5% / −4.6% / −8.9%** (5th/6th/7th). 단 전제(단위노름)는 **거짓**(실측 ‖x‖≈2.0). `unit` ≈ `spherical` 이므로 한 줄이면 충분. **비용: 할당 25~37% 변경 → atlas·모델 전부 무효화** | 브리지 실험 |
| #6 CFR+ | 진단 정확. `strategy_sum += strategy`(균등), `regrets=max(0,·)`(clipping만). 선형 averaging 없음 | 3835/3997, 2085 |
| #3 1차 결과 | 100k roots 에서 **rising 32/64/128 이 최악(3.116)**, 16/32/64 최선(1.210). 총 버킷 수와 LBR 이 단조 | 아래 A 에서 재검증 |

---

## Plan A — rising 재검증: shape 인가 학습량 인가 ⭐ 최우선

**문제 제기(타당함).** 1차 실험은 모든 arm 에 **동일 roots(100k)** 를 줬다. 224버킷
arm 은 112버킷 arm 보다 버킷당 데이터가 절반이다. 따라서 관측된
"버킷 많을수록 나쁨"은 **abstraction pathology 가 아니라 단순 데이터 부족**일 수
있다. 단일 지점 측정으로는 두 가설을 구분할 수 없다.

### A1. 수렴 곡선 (핵심 실험)

각 arm 을 **roots = 100k / 300k / 1M / 3M / 10M** 체크포인트마다 LBR 측정.

```text
가설 B(데이터 부족)  → rising 곡선이 더 가파르게 하강, 어느 지점에서 교차
가설 A(추상화 병리)  → 곡선이 평행 또는 발산, 교차 없음
```

**교차점의 존재 여부가 판정 기준이다.** 교차하면 "7th 확대"는 옳고 단지 예산이
부족했던 것이고, 교차하지 않으면 shape 자체가 틀린 것이다.

### A2. 동일 compute 가 아니라 **동일 버킷당 방문수**로 정렬

곡선을 x축 = roots 가 아니라 **x축 = 평균 방문/인포셋** 으로 다시 그린다.
이 축에서 겹치면 차이는 전부 데이터 희석이고, 그래도 벌어지면 shape 효과다.

### A3. 데이터 기아 지표 (한 번의 테이블 스캔, 공짜)

arm 별로 기록:

- 총 인포셋 수, 평균/중앙값 방문수
- **방문수 < N 인 인포셋 비율** (N = 10, 100, 1000) ← 기아 정도
- 스트리트별 분해 (5th/6th/7th 각각)

### A4. 후회 상한 (원래 계획 #2, 여전히 유효)

```text
ε_abs ≤ (1/T) · Σ_I max_a R⁺(I,a)
```

arm 별로 계산. **최적화 오차와 추상화 오차를 분리**한다.

| 후회 상한 | LBR | 해석 |
| --- | --- | --- |
| 계속 감소 중 | 높음 | 아직 학습 중 → 가설 B. 더 돌릴 것 |
| 평평(≈0) | 높음 | 추상화 천장 → 가설 A. shape/세분이 문제 |

**A4 는 A1 보다 훨씬 싸다. A1 착수 전에 먼저 돌려서 예산을 아낄 것.**

### A5. 측정 품질 개선 (싸고 효과 큼)

- **paired LBR**: 현재 arm 별로 독립 CI 를 냈는데, 딜 시드는 공유하고 있다.
  arm 별 per-hand 값을 저장해 **같은 딜에서 짝지어 비교**하면 CI 가 크게 좁아진다.
  1차 실험의 상위 2개(1.210 vs 1.556)가 구분 안 된 이유가 이것이다.
- 확정 등급으로 승격: 스크리닝 5000핸드/64파티클 → 확정 10000핸드/240파티클.

---

## Plan B — 휴리스틱 시딩 (낮은 NashConv 출발점)

**아이디어.** 약한 손 check/fold, 강한 손 check/call 로 시작. 베팅을 하지 않으므로
블러프·얇은 밸류벳으로 잃지 않는다. 무작위 초기화보다 훨씬 균형에 가까운 지점에서
CFR 을 시작한다.

### B0. 정직한 사전 평가 — 이 시드의 LBR 을 먼저 측정

이 정책은 **저착취가 아닐 수 있다.** 절대 베팅하지 않고 약한 손을 다 접으므로,
상대가 매번 작게 베팅하면 팟을 계속 뺏긴다(과폴드 착취). 따라서:

> **B0 를 반드시 먼저 한다.** 시드 자체의 LBR 을 재고, 그 값이 CFR 초기값보다
> 실제로 낮은지 확인한다. 낮지 않으면 Plan B 는 여기서 중단한다.

동시에 **폴드 기준선**(원래 계획 #4-1, 아직 미측정)도 같이 잰다. 두 값은 모든
LBR 해석의 기준점이다. CLI 가 복구됐으므로 `--lbr-target fold` 로 즉시 가능하다.

### B1. 이미 있는 코드 자산 (신규 구현 최소)

| 필요 | 기존 자산 |
| --- | --- |
| 시드 정책 | `ConditionalParticipationPolicy("made-call", cat)` — 베팅 없으면 CHECK, 베팅 만나면 CALL, 약하면 FOLD. **사용자가 말한 정책과 사실상 동일** |
| 대안 시드 | `"fold"`, `"made-bet"`, `HalfPolicy` |
| 모방 주입 | `MCCFR::imitate_policy(state, seat, target, weight)` |
| 대량 모방 | `imitate_teacher_roots(...)` |
| 시드 감쇠 | `use_decaying_imitation_prior(true)`, `scale_imitation_prior(x)` |
| regret 초기화 | `--initial-fold-regret`, `--initial-other-regret`, `initial_regret_policy` |

### B2. 절차

1. 시드 정책을 teacher 로 `imitate_teacher_roots` → 초기 평균전략을 시드에 정렬
2. **감쇠 prior 사용 필수** (`use_decaying_imitation_prior`). 감쇠 없이 고정 prior 를
   두면 CFR 고정점이 편향된다
3. **warm start 시 regret 스케일 다운 필수** — 큰 regret 을 물려주면 새 iteration 이
   미미해져 학습이 얼어붙는다 (컨텍스트 문서에 명시된 함정)
4. CFR 진행, 체크포인트마다 LBR

### B3. 판정 기준

```text
성공: 시드 곡선이 무작위 초기화 곡선보다 (a) 낮게 시작하고
      (b) 같은 roots 에서 계속 낮으며 (c) 최종값도 같거나 낮다
경계: 초기엔 낮지만 후반에 역전당함 → 시드는 조기 가속용으로만 가치
실패: 시드가 특정 basin 에 가둬 최종값이 더 나쁨 → prior 감쇠를 더 빠르게
```

**(c) 가 핵심이다.** 조기 이득만 있고 최종값이 나빠지면 시딩은 함정이다.

---

## Plan C — LBR 을 상대군으로 학습 (분석 + 설계)

### C1. 순진한 방식은 발산한다 (경고)

"현재 정책의 BR 을 상대로 두고 학습"은 **best-response dynamics** 이고, 2인
제로섬에서도 **수렴하지 않고 순환**한다(가위바위보). 그대로 하면 안 된다.

### C2. 올바른 정식화: **CFR-BR**

> Johanson, Bard, Burch, Bowling (2012),
> *Finding Optimal Abstract Strategies in Extensive-Form Games*, AAAI.

한 플레이어는 CFR 로 갱신하고, **상대는 매 iteration 정확한 best response 를
플레이**한다. 이 조합은 **Nash 로 수렴하는 것이 증명돼 있고**, 어떤 경우엔 일반
CFR 보다 빠르다. 즉 "BR 을 상대로 학습"의 **유일하게 안전한 형태**다.

현재 컨텍스트 문서의 참고문헌에 이 논문이 없다. 추가를 권한다.

**단서:** CFR-BR 은 상대가 *정확한* BR 임을 가정한다. LBR 은 depth-limited +
belief 근사이므로 "국소 BR"이고, 보장은 그만큼 약해진다. 따라서 CFR-BR 의
근사판으로 취급해야 한다.

### C3. 더 견고한 대안: population / PSRO

매 iteration BR 계산은 LBR 비용상 비현실적이다(policy query 백만 단위).
대신:

1. 주기적으로 LBR 스냅샷을 뽑아 **상대 population 에 추가**
2. meta-game 을 풀어 population 위의 혼합(restricted Nash)에 대해 best respond
3. 반복 (Double Oracle / PSRO)

population 을 쓰면 단일 BR 의 순환과 **단일 착취 방향 과적합**을 동시에 완화한다.
LBR 이 현재 HALF 한 패턴에 46~56% 집중돼 있다는 관측이 정확히 그 위험을 보여준다.

### C4. 과적합 방지 프로토콜 (반드시)

컨텍스트 문서 §8: "벤치마크에 튜닝 = exploitability 의 존재 이유를 버리는 것".
LBR 로 학습하고 **같은 LBR 로 평가하면 숫자가 무의미**해진다.

```text
train probe : LBR-A  (particles=32, seed군 A)
eval  probe : LBR-B  (particles=240, seed군 B, 가능하면 다른 belief/탐색 설정)
+ 보조 평가 : 휴리스틱 필드, 폴드 기준선, heuristic-pool
```

train/eval 프로브를 분리하지 않으면 결과를 보고하지 않는다.

### C5. 구현 시 재사용할 자산

- `PolicyLBR<TargetPolicy>` — 템플릿이라 새 상대 타입도 그대로 꽂힌다
- `play_hand_policy_lbr(deck, lbr_seat, target, lbr, ante, stack_ante, h4)` —
  한 좌석이 LBR 인 핸드 진행 하네스가 **이미 있다**. 평가용으로 쓰이지만 데이터
  생성에도 그대로 쓸 수 있다
- `set_attribution_callback` — 어느 인포셋에서 털렸는지 기록 (Plan D 와 공유)

### C6. 판정 기준

```text
성공: eval-probe LBR 이 self-play 기준선보다 유의하게 낮음
      + 휴리스틱 필드 성능이 나빠지지 않음
실패: train-probe LBR 만 낮아지고 eval-probe 는 그대로/악화 → 과적합
위험: 순환 징후 (LBR 이 오르내리며 수렴 안 함) → C3 population 으로 전환
```

---

## Plan D — split bucket (이전 계획, 통합)

**목표:** 전체 K 를 올리는 대신 **실제로 털리는 버킷만** 쪼갠다.

1. **타겟 선정 — LBR 귀속 로깅.** `set_attribution_callback` 이 이미 있고
   `--dump-lbr-attribution` 경로도 있다(7412–7425). `(InfoKey, decisions, gain,
   max_gain)` 을 쌓아 **gain 상위 버킷**을 분할 후보로. `c(B)=touch×regret-range`
   보다 직접적인 증거다.
2. **분할.** 부모 클러스터 소속 샘플만 2-means 로 쪼개 atlas 해당 스트리트 블록에
   centroid **append**(기존 ID 보존 — adaptive 경로와 동일 규약).
3. **자식 초기화.** 부모의 현재 평균전략으로 warm start, **regret 스케일 다운 필수**.
4. **post-hoc merge.** README 가 "physical merge 미구현" 이라 명시 → 신규 구현.
   기준: 두 자식의 평균전략 TV distance < ε 이고 방문수 충분하면 되돌림.
   ID 재매핑이 필요하므로 **atlas 저장 시점에만** 수행.
5. **검증.** 분할 전/후를 동일 compute·동일 LBR 설정으로 비교.
   **총 버킷 수를 맞춘 uniform-K 대조군 필수** — 없으면 "분할이 좋았다"와
   "버킷이 많아졌다"를 구분할 수 없다.

**리스크:** abstraction pathology (Waugh 2009). 교집합(A1∩A2) 기준부터 시작하고
단독 기준 분할은 피한다.

---

## 통합 실행 순서 (권장)

```text
0. (즉시·공짜) B0 폴드 기준선 + 시드 정책 LBR        ← 모든 해석의 기준점
1. (싸다) A5 paired LBR 로 1차 결과 재검정           ← 기존 결론 확정/기각
2. (싸다) A3 데이터 기아 지표 + A4 후회 상한          ← 가설 A vs B 판정
3. (비쌈) A1 수렴 곡선 — 단, 2번이 B 를 지지할 때만   ← 교차점 존재 여부
4. Plan B 시딩 — B0 통과 시에만
5. Plan C — C2(CFR-BR) 소규모 검증 → 되면 C3(population)
6. Plan D — 2번이 "추상화 천장"을 지목할 때만
```

**게이트 논리:** 2번이 가장 싸면서 3·6번의 착수 여부를 결정한다. 순서를 지키면
가장 비싼 실험(A1, D)을 근거 없이 시작하는 일을 막는다.

---

## 검토가 필요한 열린 질문

1. **A1 예산.** 10M roots × 4 arm 은 큰 비용이다. arm 을 2개(uniform vs rising
   32/64/128)로 줄여 교차만 확인할까?
2. **Plan B 시드 강도.** `made_min_category` 임계값을 어디로? 이 값이 "약한 손"의
   정의이고 시드의 성격을 좌우한다.
3. **Plan C 우선순위.** CFR-BR 은 이론이 깨끗하지만 BR 비용이 크다. 먼저
   population 없이 소규모(7th 서브게임 한정)로 개념 검증할지?
4. **#1 centroid 정규화 적용 시점.** 효과는 확인됐지만 기존 atlas·모델을 전부
   무효화한다. A1 같은 큰 재학습을 할 때 **함께** 반영하는 것이 경제적이다.
5. **측정 등급.** 위 계획의 어느 단계부터 확정 등급(10000핸드/240파티클)을
   요구할지.
