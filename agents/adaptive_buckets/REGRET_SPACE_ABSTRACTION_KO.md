# 후회 공간 abstraction: 샘플링 CFR+IRA 수렴 보장을 위한 이론 정리

작성 2026-10-25. **실험은 아직 하지 않았다.** CFR+IRA 실행기(`mini_stud_ira`)는 이후에 들여온다.
이 문서는 이론 정리와 무작위 벡터 및 1인 소형 게임 수치 확인
([test_regret_space_theory.py](test_regret_space_theory.py))만 포함한다.

상태 표기:

- **[증명]** 이 문서 안에서 증명을 완결함. 모두 수치로도 확인함.
- **[스케치]** 가정을 명시하고 증명 개요만 적음.
- **[가설]** 검증하지 않음.
- **[미해결]** 아직 답이 없음.

## 0. 질문과 결론 요약

질문: imperfect-recall bucket 위에서 샘플링 CFR(+)을 돌린다. 학습 중에 bucket을 만들거나 쪼개면서 원 게임 수렴을 수식으로 보장할 수 있는가? 보장할 수 있다면 학습한 거리(비유사도)는 그 보장의 어디에 들어가는가?

1. 원 게임 `NashConv(σ̄) ≤ Σ_i (C_i + H_i)/T`는 abstraction의 형태와 무관하게 성립한다. **[증명]**
2. H는 bucket마다 **후회 공간에서의 질량가중 Bregman 분산**과 정확히 같다(`H_B = Σ_I W_I D_ψ(ρ_I, ρ̄_B)`). **[증명]**
   - k-means를 해야 할 공간은 카드 feature 공간이 아니다. 정보집합별 평균 후회 벡터 ρ의 공간이다.
   - 이 공간의 거리 D_ψ는 대칭이 아니고 삼각부등식도 없다.
   - 같은 Lloyd 반복을 이 공간과 D_ψ로 하면 목적함수가 정확히 H가 된다.
3. 분할만 하는 epoch 방식을 쓴다. 추적 표본으로 H를 추정하고, 추정값이 역치를 넘으면 분할한다. 그러면 거리 학습의 품질과 무관하게 거의 확실히 수렴한다. **[스케치]**
   - 거리 학습은 정확성을 정하지 않는다. 효율(최종 bucket 수, 마지막 분할 시점)을 정한다.
   - **선행 연구와 겹침:** 샘플링 없이 전체 순회하는 경우(p = 1)는 CFR+IRA(Čermák et al. 2020)의 refinement 수렴 보장과 본질적으로 같다. CFR+IRA는 감시 정보집합별 regret 역치로 판정하고, 여기서는 bucket 단위 H로 판정한다는 차이만 있다. 새로 주장할 수 있는 부분은 샘플링·추적 표본 판정, H의 정확한 기하, 학습 사상을 통한 일반화이며, 이것도 체계적인 문헌 조사 전에는 확정하지 않는다.
4. 학습 거리의 목표는 feature에서 ρ로 가는 예측기 f다. 예측 오차가 H를 늘리는 양은 최대 `4 Σ_I W_I ‖ρ̂_I − ρ_I‖∞`다. **[증명]** 이 식이 "regret을 보고 거리를 역전파로 조정"할 때의 손실을 정해 준다.
5. bucket 표에 저장된 평균전략은 일반적으로 이론의 σ̄와 다르다. 자기 abstract 경로를 기억하는 조건이 있으면 같아진다. **[증명]**
6. `H/T → 0`은 수렴의 충분조건이지 필요조건이 아니다. **[증명, 반례 수치 확인]**

## 1. 설정과 기호

- **게임:** G는 2인 영합, perfect recall, 유한 게임이다(mini_stud). `𝓘_i`는 player i의 원 정보집합 전체, `A(I)`는 정보집합 I의 행동집합이다.
- **분할:** 𝓟는 `𝓘_i`를 bucket B로 나눈 것이다. 같은 소유자, 같은 행동집합끼리만 묶는다. bucket 전략을 원 정보집합으로 들어 올리면(lifting) `σ_t(I) = σ_t(B(I))`이다.
- **도달확률:** `w_t(I) = π_{-i}^{σ_t}(I)`는 상대와 chance만의 도달확률(cf reach)이다. `π_i`는 자기 도달확률이다.
- **기대보상:** `q_t(I,a)`는 I에 도달했다는 조건에서 a를 둔 뒤의 기대보상이다.
- **순간 cf 후회:**

  ```text
  r_t(I,a) = w_t(I) · ( q_t(I,a) − Σ_b σ_t(B,b) q_t(I,b) )
  ```

- **구간 누적량:** 구간 𝒯(예: 한 epoch)에서 다음을 정의한다. W_I = 0인 정보집합은 제외한다.

  ```text
  R_I = Σ_{t∈𝒯} r_t(I),   W_I = Σ_{t∈𝒯} w_t(I),   ρ_I = R_I / W_I
  R_B = Σ_{I∈B} R_I,      W_B = Σ_{I∈B} W_I,      ρ̄_B = R_B / W_B = Σ_I λ_I ρ_I,   λ_I = W_I / W_B
  ```

  ρ_I는 "단위 도달량당 평균 후회"다. 이 문서에서 정보집합의 **위치**는 ρ, **질량**은 W다.
- **ψ:** `ψ(x) = max(0, max_a x_a)`. 성질은 다음과 같다.
  - convex이고 양의 동차(positively homogeneous)다.
  - ‖·‖∞에 대해 1-Lipschitz다.
  - subgradient는 `conv{0, e_a}` 안에 있어서 `‖g‖₁ ≤ 1`이다.
- **진단량:** 다음 정의는 `agents/evpa/endgame.py`의 L/C/H 진단과 같다. 거기서는 T로 나눈다.

  ```text
  L = Σ_I ψ(R_I),   C = Σ_B ψ(R_B),   H = L − C ≥ 0
  ```

알고리즘이 실제로 계산하는 것은 bucket별 `R_B`뿐이다. 샘플링이면 그 추정치 `R̃_B`에 regret matching(+)을 적용한다. 원 정보집합별 `R_I`는 분석과 분할 판단에만 쓴다.

## 2. 기본 부등식

**Lemma 2.1 [증명].** σ̄를 구간 내 반복들의 sequence-form 균등 평균이라 하고, 후회를 다음처럼 정의한다.

```text
Reg_i = max_{σ'} Σ_t u_i(σ', σ_{-i,t}) − Σ_t u_i(σ_t)
```

그러면 `NashConv(σ̄) = (Reg_1 + Reg_2)/|𝒯|`이다.

증명: 보상은 sequence form에서 쌍선형이므로 `Σ_t u_i(σ', σ_{-i,t}) = |𝒯| u_i(σ', σ̄_{-i})`이다. 영합이므로 `Σ_t u_1(σ_t) + Σ_t u_2(σ_t) = 0`이다. 두 식을 합치면 결론이 나온다.

**Lemma 2.2 (Zinkevich et al. 2007, Thm 3).** perfect recall 게임에서는 **임의의** 전략열에 대해 `Reg_i ≤ Σ_{I∈𝓘_i} ψ(R_I)`이다. bucket 전략을 들어 올린 것도 원 게임의 behavioral 전략이므로, abstraction 구조와 상관없이 이 부등식이 그대로 적용된다.

**Corollary 2.3 [증명].**

```text
NashConv(σ̄) ≤ Σ_i (C_i + H_i) / |𝒯|
```

- **C(bucket 후회)의 상계:**
  - RM은 `ψ(R_B) ≤ Δ_B √(|A| |𝒯|)`이다.
  - RM+은 누적 후회 Q가 `Q ≥ R`을 만족하므로 같은 상계가 성립한다.
  - 샘플링(MCCFR)이면 `R_B = R̃_B + M_B`이고, 오차 `M_B`는 martingale이다. 따라서 높은 확률로 `‖M_B‖∞ = O(Δ̃ √(|𝒯| log(1/δ)))`이다.
- **가중 평균:** linear averaging이나 DCFR처럼 시점별 가중을 주는 경우에도 위 식은 모두 r에 선형이다. 그래서 `r_t`를 가중된 `r_t`로 바꾸면 그대로 성립한다.
- **결론:** `H/|𝒯| → 0`이면 수렴한다.

**Remark 2.4 [증명, 수치 확인]. 역은 성립하지 않는다.** 이전 대화에서 두 조건이 동치라고 하셨다. 정확히는 다음 관계다.

- C → 0이라는 전제 아래에서 `L/T → 0 ⇔ H/T → 0`은 동치다.
- `L/T → 0 ⇒ NashConv → 0`도 성립한다.
- 그러나 `NashConv → 0 ⇒ L/T → 0`에는 반례가 있다.

반례 게임:

```text
chance 1/2 → I : a → +1, b → −1
chance 1/2 → X : out → 0, in → J : a → −2, b → −1
bucket B = {I, J}  (X는 단독)
```

- 공유 전략 σ(B) = (p, 1−p)에서 bucket 후회는 `½(1−p, −p)`이다. 그래서 p → 1이 된다.
- 이때 J의 cf 후회는 매 반복 `(0, ½)`씩 쌓인다.
- X는 out을 고르므로 J의 자기 도달확률은 0으로 간다.

T = 200,000에서의 측정값은 다음과 같다.

| NashConv(σ̄) | L/T | C/T | H/T |
|---:|---:|---:|---:|
| 0.00000 | 0.5000 | 0.0000 | 0.5000 |

원인은 cf 후회가 자기 도달확률을 빼고 계산된다는 점이다. 평균전략이 도달하지 않는 정보집합의 후회도 L에 남는다. 따라서 H로 분할을 판정하면 안전한 쪽(과분할)으로 틀린다. 필요충분한 양은 `Reg_i` 자체지만, 이 값은 정보집합별로 분해되지 않는다.

## 3. H의 구조

**Theorem 3.1 [증명].** 각 bucket B에 대해 다음이 성립한다. 여기서 `g_c ∈ ∂ψ(c)`는 어떤 subgradient를 골라도 된다.

```text
H_B = W_B [ Σ_I λ_I ψ(ρ_I) − ψ(ρ̄_B) ] = Σ_{I∈B} W_I · D_ψ(ρ_I, ρ̄_B)
D_ψ(x, c) = ψ(x) − ψ(c) − ⟨g_c, x − c⟩ ≥ 0
```

증명: 양의 동차성으로 `ψ(R_I) = W_I ψ(ρ_I)`이다. 또 `Σ_I W_I (ρ_I − ρ̄_B) = 0`이므로 선형항이 사라진다.

D_ψ를 풀어 쓰면 두 경우로 나뉜다.

- **c의 성분이 모두 0 이하일 때:** g = 0을 고르면 `D_ψ(x,c) = ψ(x)`이다.
- **c의 최댓값이 양수이고 그 위치가 a*일 때:** `D_ψ(x,c) = ψ(x) − x_{a*}`이다. 뜻은 "bucket이 선호하는 행동 a*를 따를 때 그 정보집합이 느끼는 후회"다.

D_ψ는 대칭이 아니고 삼각부등식도 없다. 즉 **이론이 요구하는 거리 자체가 metric이 아니다.** enum을 행렬 d(a,b)로 다루고 삼각부등식을 요구하지 않는다는 말씀과 일치한다. 이후의 정리도 삼각부등식을 쓰지 않는다.

**Corollary 3.2 (병합 가법성) [증명].** 두 정보집합 묶음 X, Y에 대해 다음이 성립한다.

```text
H(X ∪ Y) = H(X) + H(Y) + h(R_X, R_Y),   h(x, y) = ψ(x) + ψ(y) − ψ(x + y) ≥ 0
```

이로부터 세 가지가 나온다.

- **(a) 분할 이득의 정확한 식:** B를 B₁, B₂로 나누면, 같은 궤적에서 H는 정확히 `h(R_{B₁}, R_{B₂})`만큼 줄어든다. `dynamic_tree*`의 SSE 분할 기준 자리에 넣을 수 있는 정확한 기준이다.
- **(b) 응집형 군집:** h를 linkage로 쓰는 응집형 군집에서 병합 트리의 linkage 합은 정확히 H다.
- **(c) h = 0의 조건:** h = 0이면 ψ가 선분 [R_X, R_Y] 위에서 선형이다. 즉 다음 둘 중 하나다.
  - 두 벡터가 모두 0 이하다.
  - 같은 행동이 양의 최댓값을 가진다.

  카드 feature가 얼마나 닮았는지와는 무관한 조건이다.

**Corollary 3.3 (후회 공간 Bregman k-means) [증명].** {ρ_I, W_I}가 고정되어 있으면 `Σ_I W_I D_ψ(ρ_I, c)`는 질량가중 평균 `c = ρ̄`에서 최소가 된다. subgradient를 어떻게 골라도 마찬가지다. 이것은 Banerjee et al. 2005 결과의 미분 불가능 버전이다.

증명: 다음 식에서 마지막 부등식은 ψ의 볼록성에서 나온다.

```text
Σ W_I D_ψ(ρ_I, c) = Σ W_I ψ(ρ_I) − W_B [ ψ(c) + ⟨g_c, ρ̄ − c⟩ ] ≥ Σ W_I ψ(ρ_I) − W_B ψ(ρ̄)
```

따라서 Lloyd 반복은 H를 단조 감소시킨다. 반복은 두 단계다.
1. 배정: `argmin_k D_ψ(ρ_I, c_k)`
2. 중심 갱신: 질량가중 평균

기존 카드 feature k-means는 이 보장을 줄 수 없었다. 그 알고리즘이 줄인 SSE는 H와 연결되지 않기 때문이다.

**Remark 3.4 [증명]. 순간 advantage가 같아도 H > 0일 수 있다.**

```text
A₁ = (1, −1), A₂ = (−1, 1),   w(I) = (1, 0), w(J) = (0, 1)   ⇒   H = 2
```

두 정보집합의 도달확률 비율이 시간에 따라 변하면 이런 일이 생긴다. 그래서 거리 학습의 label은 순간 샘플 후회가 아니라 **구간 누적 ρ**여야 한다.

**Remark 3.5.** 느슨한 상계로 다음이 있다.

```text
H_B ≤ Σ W_I ‖ρ_I − ρ̄_B‖∞ ≤ W_B max_{I,J∈B} ‖ρ_I − ρ_J‖∞
```

이 상계는 0 이하 영역에서 생기는 무해한 차이까지 세므로 느슨하다. 분할 판정에는 Theorem 3.1의 정확한 식을 쓴다.

## 4. 적응 분할과 수렴

### 알고리즘 (epoch 분할 방식)

1. **epoch:** epoch k는 구간 `[t_k, t_k + L_k)`이다. 길이는 `L_k ≥ c·t_k`로 둔다(예: 매번 두 배).
2. **bucket 학습:** 각 bucket은 RM+로 학습한다. 분할할 때 자식의 Q는 부모 Q를 0 이상의 값으로 나눠 준다. 0으로 줘도 된다. 조건은 `‖Q_자식‖₂ ≤ ‖Q_부모‖₂`이다.
3. **추적 표본:** epoch마다 새 salt로 추적 표본 `S_k`를 뽑는다. 각 정보집합을 독립적으로 확률 p로 뽑는다(`hash(key, salt) < p`).
   - 추적은 수동 기록일 뿐이다. 전략에는 영향을 주지 않는다.
   - 추적 정보집합만 `R̃_I`, `W̃_I`를 따로 누적한다.
4. **분할 판정:** epoch가 끝나면 다음 추정치를 계산한다.

   ```text
   Ĥ_B = (1/p) Σ_{I ∈ B∩S_k} ψ(R̃_I) − ψ(R̃_B)
   ```

   `Ĥ_B > τ_k L_k`이면 B를 분할한다.
5. **분할 방식:** 학습 거리, h 최대화 등 무엇이든 쓸 수 있다. 단, 실제 멤버 기준으로 진분할이어야 한다(한쪽이 비면 안 됨).
   - 진분할이 안 되면 대체 규칙을 쓴다. `ψ(R̃_I)`가 가장 큰 추적 멤버를 단독 bucket으로 빼는 예외 매핑이다.
6. **제약:** 분할은 epoch 경계에서만 한다. 병합은 하지 않는다.

**Lemma 4.1 (RM+ 구간 성질) [증명].** 다음 두 성질이 있다.

- `Q_{t+1} = (Q_t + r̃_t)^+ ≥ Q_t + r̃_t`이므로 epoch 안에서 누적한 `r̃_B`는 `Q_end` 이하다.
- `σ_t ∝ Q_t`이고 각 멤버에서 `⟨σ_t, r̃_t⟩ = 0`이므로 `⟨Q_t, r̃_t⟩ = 0`이다. 따라서 다음이 성립한다.

  ```text
  ‖Q_end‖₂² ≤ ‖Q_start‖₂² + Σ_t ‖r̃_t‖₂²
  ```

결과적으로 epoch 구간의 bucket 후회는 `O(Δ̃ √(|A| (t_k + L_k)))`다. 또 epoch 구간 후회는 자식 bucket 자신의 합이므로, **분할 시 후회를 옮기면서 생기는 오차가 없다.** 이것이 RM+와 epoch 방식을 고른 이유다.

**Theorem 4.2 [스케치].**

가정:
- (A1) G는 유한, perfect recall, 2인 영합 게임이다.
- (A2) 샘플 후회는 불편추정량이고 `|r̃| ≤ Δ̃`다.
- (A3) `L_k ≥ c·t_k`이고 `τ_k → 0`이다.
- (A4) `p ∈ (0, 1]`은 고정이다.

결론(거의 확실하게):
- (i) 분할 횟수는 유한하다.
- (ii) epoch 평균 `σ̄^{(k)}`의 NashConv는 0으로 간다. NashConv가 볼록이므로 전체 평균의 NashConv도 0으로 간다.

증명 개요:
- **(i)** 매 분할은 실제 멤버 기준 진분할이다. 따라서 분할은 최대 `Σ_i |𝓘_i| − 1`회다.
- **(ii) 감지 확률:** 마지막 분할 이후 분할 𝓟*가 고정된다. bucket B와 η > 0을 정하고 사건 `E_k = {H_B^{(k)} ≥ η L_k}`를 본다.
  - 추정치의 핵심 항 `X = Σ_I ξ_I ψ(R_I)/p`는 궤적이 주어졌을 때 `E X = S`, `E X² ≤ S²/p`이다.
  - Paley–Zygmund 부등식으로 `P(X > θS) ≥ p(1−θ)²`이다.
  - Lemma 4.1과 martingale 항 때문에 `ψ(R_B) = O(√t_k)`이고, 샘플 노이즈는 `o(L_k)`다. 따라서 `θ → τ_k/η < 1`이다.
- **(ii) 모순:** salt가 epoch마다 독립이다. 그래서 `E_k`가 무한히 자주 일어나면, 조건부 Borel–Cantelli(Lévy)에 의해 분할도 무한히 자주 일어난다. 이는 (i)에 모순이다.
- **(ii) 결론:** η = 1/n에 대해 위를 적용하면 거의 확실히 `H^{(k)}/L_k → 0`이다. 여기에 Corollary 2.3과 Lemma 4.1을 합치면 다음을 얻는다.

  ```text
  NashConv(σ̄^{(k)}) ≤ Σ_i [ N_i Δ̃ √(|A|(t_k+L_k)) + ‖M‖ + H_i^{(k)} ] / L_k → 0
  ```

- **특수한 경우:** p = 1이고 샘플링이 없으면 결정론적 결과가 되고, Paley–Zygmund가 필요 없다. 이 경우는 CFR+IRA(Čermák et al. 2020)의 결과와 본질적으로 같다(§0 참고).

주의:
- **속도:** 마지막 분할 시점 t*에 대한 상계가 없다. 그래서 수렴 속도를 보장하지 못한다.
- **큰 게임:** 정보집합 수 N이 천문학적이면 이 정리는 정성적인 의미만 가진다.
- **노이즈:** 노이즈 때문에 생기는 과분할은 정확성은 해치지 않고 효율만 해친다. 과분할을 줄이려면 `τ_k L_k ≫ N Δ̃ √(L_k log k)/p`가 되도록 τ_k를 천천히 줄인다.

**Proposition 4.3. "bucket을 늘리면 좋아지는가" [증명].**

- 같은 궤적에서는 분할이 H를 정확히 h ≥ 0만큼 줄인다(Corollary 3.2).
- 그러나 분할은 이후의 궤적을 바꾼다. 그래서 서로 다른 실행 사이의 단조성은 없다. 이것이 abstraction pathology다(Waugh et al. 2009).
- 결국 보장은 단조성에서 나오지 않고 Theorem 4.2의 극한에서 나온다.

## 5. 배포: 평균전략을 bucket 표로 표현할 수 있는가

앞의 정리들이 말하는 σ̄는 원 정보집합별 sequence-form 평균이다. bucket 표에 저장되는 평균 σ̃는 다음과 같다. m_I는 시간에 따라 변하지 않는 가중이다(예: EVPA의 chance marginal).

```text
σ̃(B) = Σ_t Σ_{I∈B} m_I π_i^t(I) σ_t(B)  /  Σ_t Σ_{I∈B} m_I π_i^t(I)
```

**Proposition 5.1 [증명].** 각 bucket의 멤버들이 같은 **자기 abstract 경로**를 가지면 σ̃ = σ̄이다. 자기 abstract 경로는 자기의 이전 결정들의 (bucket, 행동) 열이다.

증명: bucket으로 표현되는 모든 σ에 대해 `π_i^σ(I) = Π σ(B', a')`이고, 곱하는 항들은 공유 경로를 따라간다. 그래서 멤버들의 자기 도달확률이 같다.

반례: 멤버들의 자기 도달확률이 시간에 따라 엇갈리는 경우다.
- t = 1에는 I만 도달하고 σ = a다.
- t = 2에는 J만 도달하고 σ = b다.
- 그러면 σ̄(I) = a, σ̄(J) = b인데 σ̃ = (½, ½)이다.

해결책:
- **평균 표만 재색인:** 후회 표는 imperfect-recall bucket을 그대로 쓴다. 평균 표만 (bucket, 자기 abstract 경로)로 색인한다. `π_i`는 abstract 경로만의 함수이므로 이 방식은 정확하다.
- **반복 표본 배포:** 반복 checkpoint의 표본을 혼합전략으로 배포한다. Kuhn 정리에 의해 σ̄와 realization-equivalent이다. 유한 표본이므로 근사다.

7-stud 기존 street별 독립 bucket은 이 조건을 만족하지 않는다. 그래서 거기서 보고한 평균전략 지표는 σ̄의 근사로 읽어야 한다.

## 6. 학습 거리(비유사도)의 위치

Theorem 4.2의 수렴은 거리의 품질과 무관하다. 거리가 정하는 것은 다음 세 가지다.
- 초기 분할
- 분할 방향
- 처음 보는 정보집합의 배정

이 세 가지가 최종 bucket 수와 마지막 분할 시점, 즉 실제 비용을 정한다.

### 6.1 권장 형태: 후회 공간으로 보내는 예측기

`f_θ : x(I) → ρ̂_I ∈ ℝ^{|A|}`. 출력 head는 행동집합 class마다 따로 둔다.

- **거리를 출력 공간에서 정의:** 거리는 입력 공간이 아니라 출력(후회) 공간에서 D_ψ로 정의한다. 그러면 enum과 실수의 정의역을 통일하는 문제가 아예 사라진다.
  - enum은 값마다 독립 차원으로 넣는다(one-hot 또는 값별 학습 벡터). "값마다 차원이 다르다"는 관점과 그대로 맞는다.
- **Proposition 6.1 [증명, 수치 확인].** 중심 {c_k}를 최적 K-분할의 질량가중 평균으로 둔다. 예측값으로 `π̂(I) = argmin_k D_ψ(ρ̂_I, c_k)`처럼 배정하면 다음이 성립한다.

  ```text
  H(π̂) ≤ H* + 4 Σ_I W_I ‖ρ̂_I − ρ_I‖∞
  ```

  증명:
  - ψ가 1-Lipschitz이고 `‖g‖₁ ≤ 1`이므로 `|D_ψ(x,c) − D_ψ(y,c)| ≤ 2‖x − y‖∞`이다.
  - 중심을 다시 평균으로 갱신하면 H는 더 줄어든다(Corollary 3.3).
- **역전파 손실:** 이 정리가 손실을 정해 준다.

  ```text
  ℒ(θ) = Σ_{I∈S_k} W_I ‖f_θ(x_I) − ρ_I‖∞
  ```

  W를 가중한 L2나 Huber를 매끄러운 대용으로 쓸 수 있다. 다만 L2는 ∞-norm을 상계하지 않으므로, 정리의 보장은 ∞-norm 버전에만 적용된다.
- **학습 데이터:** epoch마다 새로 뽑는 추적 표본 `S_k`를 쓴다. H 추정에 쓰는 표본과 같다. 전략과 독립으로 뽑으므로 편향되지 않은 표본이다.

### 6.2 사용자 제안 d(a,b) 행렬 형태와의 관계

- **쌍 label:** `δ(I,J) = h(R_I, R_J)/(W_I + W_J)`. 두 정보집합을 병합할 때 단위 질량당 생기는 상쇄량이다.
  - 대칭이고 삼각부등식이 없다. 그래서 행렬 형태로 학습할 수 있다.
  - 예: `D_f[a,b] = sigmoid(θ_ab)`로 두고 대각 성분은 0으로 고정한다. 0.5 초기화는 θ = 0에 해당한다. feature별 행렬을 `Σ_f w_f D_f`로 합친다.
- **한계:** H는 묶음 수준의 linkage `h(R_X, R_Y)`로 합성된다(Corollary 3.2). 그런데 ψ가 비선형이므로 개체 쌍의 δ로는 묶음 linkage가 정확히 나오지 않는다. 반면 6.1 형태에서는 묶음이 질량가중 평균이라서 정확히 합성된다.
  - 따라서 d(a,b) 행렬은 f_θ의 입력층(enum 값별 파라미터)으로 넣는 것이 이론과 맞는다.
- **AE/VAE/VQ:** 입력 인코더(trunk)로는 쓸 수 있다. 그러나 복원 손실은 H를 상계하지 않는다.
  - VQ의 codebook을 ρ 공간의 중심으로 쓰고 배정을 D_ψ로 바꾸면, Corollary 3.3의 Lloyd 반복과 같아진다.

### 6.3 비정상성

- ρ는 σ_t와 분할에 따라 달라진다. 분할이 σ를 공유시키므로 되먹임이 생긴다.
- 그래서 f는 epoch마다 다시 적합하고, 다음 epoch의 분할과 배정에만 쓴다.
- 고정된 ρ에서 Lloyd가 단조라는 사실이 epoch 사이의 단조성을 뜻하지는 않는다.

## 7. 미해결 항목과 가정 점검

1. **[미해결] 수렴 속도:** 마지막 분할 시점에 대한 상계가 없다. 적당한 구조 조건에서 t*를 제한할 수 있는지가 다음 이론 과제다. 구조 조건의 예는 Kroer–Sandholm 2016의 skew well-formed다.
2. **[미해결] 병합과 재군집:** 병합이나 재군집을 허용하면 분할 유한성 논증이 깨진다.
   - epoch 경계에서 다시 군집하는 것은 C에는 문제가 없다(Lemma 4.1).
   - 그러나 분할이 진동할 수 있다. 분리 제약(cannot-link)을 단조 증가시키는 장치가 필요하다.
3. **큰 게임:** N이 천문학적이면 Theorem 4.2는 정성적이다. 메모리는 대략 p × (방문한 정보집합 수)다. HT 추정의 분산도 문제다.
4. **배포 평균:** 5절의 문제다.
5. **[가설] 과분할 줄이기:** Remark 2.4 때문에 H로 판정하면 경로 밖 정보집합에서 과분할한다. 자기 도달확률을 가중한 변형으로 효율을 개선할 수 있을 것이다. 단, 그 변형이 필요조건이라는 보장은 없다.
6. **CFR+와 샘플링:** RM+를 고른 이유는 구간 성질(Lemma 4.1) 때문이다. 다른 할인 방식도 구간 보장이 있으면 대체할 수 있다.

## 8. mini_stud에서 확인할 것 (실행기를 들여온 뒤)

- **항등식:** `small` 프리셋의 정확한 트리에서 Theorem 3.1과 Corollary 3.2를 확인한다.
- **반례:** Remark 2.4 현상이 실제 게임에서도 나타나는지 본다. H/T와 NashConv를 함께 기록한다.
- **비교 조건:**
  - (a) perfect recall 샘플링
  - (b) 카드 k-means 고정 — AGENTS.md 기준선이므로 반드시 겹쳐 그린다
  - (c) exact ρ를 쓰는 오라클 후회 공간 Bregman k-means
  - (d) 학습한 f_θ
  - (e) Theorem 4.2 분할 알고리즘
- **보고 형식:**
  - current 정책과 average 정책을 분리한다.
  - log-log 그래프를 쓰고, 같은 평가기와 seed를 쓴다.
  - 원 게임 exact exploitability와 L/C/H를 함께 기록한다.
- **평균 표 차이:** 5절의 σ̃와 σ̄ 차이를 측정한다.

## 참고문헌

- Zinkevich, Johanson, Bowling, Piccione. Regret Minimization in Games with Incomplete Information. NeurIPS 2007.
- Lanctot, Waugh, Zinkevich, Bowling. Monte Carlo Sampling for Regret Minimization in Extensive Games. NeurIPS 2009.
- Lanctot, Gibson, Burch, Zinkevich, Bowling. No-Regret Learning in Extensive-Form Games with Imperfect Recall. ICML 2012.
- Kroer, Sandholm. Imperfect-Recall Abstractions with Bounds in Games. EC 2016.
- Waugh, Schnizlein, Bowling, Szafron. Abstraction Pathologies in Extensive Games. AAMAS 2009.
- Brown, Sandholm. Regret Transfer and Parameter Optimization. AAAI 2014. / Simultaneous Abstraction and Equilibrium Finding in Games. IJCAI 2015.
- Tammelin. Solving Large Imperfect Information Games Using CFR+. 2014.
- Banerjee, Merugu, Dhillon, Ghosh. Clustering with Bregman Divergences. JMLR 2005.
- Čermák, Lisý, Bošanský. Automated Construction of Bounded-Loss Imperfect-Recall Abstractions in Extensive-Form Games. IJCAI 2020 / arXiv 1803.05392 (FPIRA, CFR+IRA).
- Waugh, Morrill, Bagnell, Bowling. Solving Games with Functional Regret Estimation. AAAI 2015 (RCFR).
- Embedding CFR (arXiv 2511.12083, AAAI 2026): 사전학습한 정보집합 임베딩 공간에서 regret과 전략을 갱신한다. 원문 세부 내용은 아직 확인하지 않았다.
