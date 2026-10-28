# ASWISS 수렴 증명 정리 (초안 v1)

작성: 2026-10-27 (§9 추가: 2026-10-28)

**이 문서의 위치**
- [REGRET_SPACE_ABSTRACTION_KO.md](REGRET_SPACE_ABSTRACTION_KO.md) §4의 Theorem 4.2는 "H 역치 + 추적 표본" 분할 규칙에 대한 것이다.
- 실제로 구현하고 실험한 규칙은 이와 다른 **누적 regret 라벨 분할**이다(`aswiss/aswiss.cpp`, `aswiss/gated.cpp`, `aswiss/stud7/aswiss_stud.cpp`).
- 이 문서는 실험에 쓴 규칙 그대로를 대상으로 증명을 다시 세우고, 한계를 함께 적는다.
- 기호와 기본 보조정리는 이전 문서를 따른다.
- 수치 확인은 [test_aswiss_convergence.py](test_aswiss_convergence.py)에 있다. §2의 결정론적 보조정리 세 개와 §9의 정확 분해, 단일 프로필 인증, 두 반례, 여유 라벨 집합을 작은 게임에서 정확히 계산해 확인한다.

**상태 표기**
- **[증명]** 이 문서 안에서 증명을 완결함
- **[스케치]** 가정과 증명 개요만 적음
- **[미해결]** 아직 답이 없음
- **[한계]** 정리가 보장하지 않는 것

## 0. 요약

질문은 다음과 같다. 가장 거친 추상화(time step당 bucket 하나)에서 시작해, external sampling(ES)으로 얻은 **추정** regret만으로 bucket을 쪼갠다. 그래도 원게임 Nash 균형으로 수렴하는가?

| 결과 | 내용 | 상태 |
|---|---|---|
| 라벨 항등식 (Lemma 2) | bucket의 상쇄량 $H_B$는 "모든 멤버를 하나의 라벨에 맞출 때의 최소 비용"과 **정확히 같다** | 증명 |
| 분할 유한성 (Lemma 6) | 분할은 모든 표본 경로에서 많아야 $N-\lvert\mathcal P_0\rvert$번이다 | 증명 |
| 정리 1 (epoch 방식) | 분할이 없었던 epoch마다 $\mathrm{NashConv}(\bar\sigma^{(k)}) = O\big(\sqrt{\log(kN_A/\delta)/L_k}\big)$가 확률 $1-\delta$로 성립한다. 따라서 마지막 분할 이후 거의 확실히 0으로 수렴한다 | 증명 (동시 갱신, epoch 균등 평균 기준) |
| 정리 2 (수렴 조건부 방식) | 임의의 $\varepsilon_g>0$에 대해 같은 결론이다 | 증명 |
| 정리 2′ (ε 정지 인증) | 분할 없이 게이트를 통과하면 그 시점에 $\mathrm{NashConv}\le\varepsilon_g+(\text{검정 역치 항})+(\text{샘플 오차 항})$이다. 샘플링 버전의 정지 규칙이며, $t^\*$를 몰라도 쓸 수 있다 | 증명. 단 엄밀한 오차 항은 수치적으로 공허함 |
| 정리 3 (원자 일반화, 7-stud) | 원자 내부의 상쇄량 $H_{\mathcal A}$를 뺀 나머지 오차만 0으로 간다 | 증명 |
| Lemma 7 (NashConv 정확 분해, §9) | NashConv는 최선 응답의 자기 도달확률로 가중한 정보집합 regret의 합과 **같다**. Lemma 1은 그 상계다 | 증명, 수치 확인 |
| Lemma 9 (여유 라벨 집합, §9) | 추정 오차 η 안에서는 일치하는 bucket을 나누지 않는다. 잘못 배치되어도 손실은 4η 이내다 | 증명, 수치 확인 |
| 정리 A (저장한 순간 전략 재생, §9) | 정리 1과 같은 꼴이다. 학습 중 멤버별 추적 없이 미방문 정보집합까지 라벨을 붙인다 | 증명 |
| 정리 B1 (단일 프로필 인증, §9) | 모든 bucket이 일치하면 $\mathrm{NashConv}(\sigma)\le\rho_{\mathcal P}(\sigma)$. 출력할 전략 자체에 대한 인증이다 | 증명, 수치 확인 |
| 정리 B3 (분할 시점 평균 전략, §9) | 경로 밖까지 순간 regret이 ε 이하인 프로필을 주는 학습기가 있으면 유한 라운드 안에 $\mathrm{NashConv}\le\varepsilon$, ε=0이면 정확한 균형 | 증명. 단 CFR 평균은 이 조건을 만족하지 않는다(명제 B4, 반례) |
| 마지막 분할 시점 $t^\*$의 상계 | 없음 | 미해결 |
| 교대 갱신, 선형 평균 (실제 구현) | 정리의 가정 밖 | 미해결 |
| 학습 메모리 절감, 압축된 배포 정책 | 보장하지 않음 | 한계 |

한 문단 요약:
- 분할이 끝나면 모든 bucket의 멤버 라벨이 같다. 라벨이 같으면 그 bucket의 상쇄량 $H_B$는 "라벨 검정 역치 + 샘플링 오차"로 묶인다. 둘 다 구간 길이 $L$에 대해 $O(\sqrt{L\log})$다.
- bucket 자체의 regret $C$는 RM+가 $O(\sqrt{t})$로 묶는다.
- 분할 횟수는 유한하다. 따라서 마지막 분할 이후에는 원게임 NashConv가 $O(\sqrt{\log/L})$로 줄어든다.
- 대가는 두 가지다. 마지막 분할이 언제 일어나는지는 보장하지 못한다. 그리고 라벨 계산을 위해 모든 멤버의 구간 regret을 저장해야 한다.

## 1. 설정

### 1.1 게임과 기호

- **게임:** $G$는 2인 영합, perfect recall, 유한 게임이다.
  - $\mathcal I_i$는 player $i$의 원 정보집합 전체, $\mathcal I=\mathcal I_1\cup\mathcal I_2$, $N=\lvert\mathcal I\rvert$, $N_A=\sum_{I}\lvert A(I)\rvert$이다.
  - $\Delta_I$는 $I$ 아래 부분 트리에서 보상의 최댓값과 최솟값의 차이다.
- **분할:** $\mathcal P$는 $\mathcal I$를 bucket $B$로 나눈 것이다.
  - 같은 player, 같은 행동집합끼리만 묶는다. $A_B$는 그 행동집합이다.
  - **lifting:** bucket 전략을 멤버 정보집합에 그대로 쓰는 것이다. $\sigma_t(I)=\sigma_t(B(I))$.
- **cf regret:** $r_t(I,a)=\pi^{\sigma_t}_{-i}(I)\,\big(q_t(I,a)-\sum_b\sigma_t(B(I),b)\,q_t(I,b)\big)$이다.
  - $\pi_{-i}$는 상대와 chance만의 도달확률이다.
  - $q_t(I,a)$는 $I$에서 $a$를 둔 뒤의 조건부 기대보상이다.
- **구간:** $W=(s,t]$, $\lvert W\rvert=t-s$이다.
  - 정보집합 합 $R_I^W=\sum_{u\in W}r_u(I)$
  - bucket 합 $R_B^W=\sum_{I\in B}R_I^W$
  - 샘플 추정치에는 물결표를 붙인다($\tilde R$).
- **확장 표기:** 행동집합에 "없음" $\varnothing$을 더하고 $x(\varnothing):=0$으로 둔다. 그러면 다음이 성립한다.
  $$\psi(x)=\max\big(0,\max_a x_a\big)=\max_{\ell\in A\cup\{\varnothing\}}x(\ell).$$
- **진단량:**
  $$C^W=\sum_B\psi(R_B^W),\qquad H_B^W=\sum_{I\in B}\psi(R_I^W)-\psi(R_B^W)\ \ge 0,\qquad H^W=\sum_B H_B^W.$$
  - $C$는 bucket이 스스로 느끼는 regret이다.
  - $H$는 bucket 안에서 멤버끼리 상쇄되어 사라진 regret이다.
- **라벨:** 각 정보집합에 붙이는 표시 $\ell(I)\in A(I)\cup\{\varnothing\}$이다. "이 구간 동안 누적 regret이 유의하게 가장 큰 행동", 또는 그런 행동이 없음($\varnothing$)을 뜻한다.

### 1.2 샘플링 가정 (external sampling)

- **(S1) 동시 갱신:** 반복 $t$에서 두 player의 순회는 같은 프로필 $\sigma_t$를 쓴다. bucket 표는 그 반복의 순회가 모두 끝난 뒤에 갱신한다.
  - 그래서 한 순회 안에서는 같은 bucket의 모든 멤버가 같은 $\sigma_t(B)$를 쓴다.
  - 현재 구현은 이 가정과 다르다. 교대 갱신을 쓴다(§7.2).
- **(S2) 추정량:** $\tilde r_t(I,a)=\mathbf 1[I\text{ 방문}]\big(\tilde v_t(I,a)-\sum_b\sigma_t(B,b)\,\tilde v_t(I,b)\big)$이다. 방문하지 않으면 0이다.
  - $\mathbb E[\tilde r_t(I,a)\mid\mathcal F_{t-1}]=r_t(I,a)$이고 $\lvert\tilde r_t(I,a)\rvert\le\Delta_I$다.
  - $\mathcal F_{t-1}$은 반복 $t-1$까지의 모든 정보다. $\sigma_t$와 그때의 분할은 $\mathcal F_{t-1}$-가측이다.
  - **근거:** ES에서 traverser는 자기 행동을 모두 펼치고, 상대 행동과 chance만 표본화한다.
    - 그래서 $I$의 한 history $h$에 도달할 확률은 $\pi_{-i}(h)$다.
    - 부분 트리 표본값은 조건부 불편추정량이다.
    - perfect recall이므로 한 순회에서 한 정보집합은 많아야 한 번 방문된다.
    - 7-stud 구현처럼 두 traverser가 같은 deal을 공유해도 각 순회의 주변분포는 정확하므로 불편성은 유지된다.
- **(S3) bucket 샘플 regret의 크기:** $\tilde r_t(B)=\sum_{I\in B}\tilde r_t(I)$로 둔다. $D_B$는 한 순회에서 방문될 수 있는 $B$의 멤버들의 $\Delta_I$ 합의 최댓값이다($\le\sum_{I\in B}\Delta_I$). 그러면 $\lVert\tilde r_t(B)\rVert_\infty\le D_B$다.
  - imperfect recall bucket은 한 순회 안에서 여러 멤버가 방문될 수 있어서 이 합이 필요하다.

### 1.3 알고리즘 (이론 버전)

1. **bucket 학습 (RM+):**
   - bucket이 생길 때 $Q_B=0$에서 시작한다.
   - $\sigma_t(B)\propto Q_{B,t}^+$로 둔다. $Q^+=0$이면 균등 전략을 쓴다.
   - 갱신은 $Q_{B,t+1}=(Q_{B,t}+\tilde r_t(B))^+$다.
2. **추적:** 현재 판정 구간 동안 모든 멤버의 $\tilde R_I^W$를 누적한다. 정규 검정이면 $\sum\tilde r^2$도 누적한다.
3. **라벨:** $\hat a_I=\arg\max_a\tilde R_I^W(a)$로 둔다.
   - $\tilde R_I^W(\hat a_I)>\theta_I^W$이면 $\ell(I)=\hat a_I$, 아니면 $\ell(I)=\varnothing$이다.
   - 구간 동안 방문되지 않은 멤버는 $\varnothing$이다.
4. **역치 조건 (T):** $0\le\theta_I^W\le\Delta_I\,\kappa(W)$이고 $\kappa(W)=o(\lvert W\rvert)$여야 한다. 실험에 쓴 세 검정은 모두 이를 만족한다($n\le\lvert W\rvert$는 방문 수, $d$는 판정 번호, $\delta_d=\delta/(d+1)^2$).
   - 정규: $z\sqrt{\sum\tilde r^2}\le z\,\Delta_I\sqrt{\lvert W\rvert}$
   - Hoeffding (구현형): $\Delta_I\sqrt{2n\ln(1/\delta_d)}\le\Delta_I\sqrt{2\lvert W\rvert\ln(1/\delta_d)}$
   - Bernstein (구현형): $\sqrt{2nV\ln(2/\delta_d)}+\tfrac{14}{3}\Delta_I\tfrac{n}{n-1}\ln(2/\delta_d)=O\big(\Delta_I\sqrt{\lvert W\rvert\log d}\big)$
5. **분할 규칙:** 판정 시점에 멤버 라벨이 두 종류 이상인 bucket을 라벨별로 나눈다.
   - 자식은 $Q=0$, 평균 표 0에서 시작한다. 상속해도 되는 조건은 Remark 3.4에 있다.
   - 병합은 하지 않는다.
6. **판정 시점:**
   - **방식 E (epoch):** 구간 $W_k=(t_k,t_{k+1}]$, $L_k=t_{k+1}-t_k$, $t_{k+1}\le\gamma L_k$이다. epoch마다 두 배로 늘리면 $\gamma=2$다. 판정은 각 epoch 끝에 한다.
   - **방식 G (수렴 조건부):** 판정 구간은 마지막 분할 시점 $s$부터다. $g$반복마다 확인하고, $\lvert W\rvert\ge m$이고 $\tilde C^W/\lvert W\rvert\le\varepsilon_g$일 때만 라벨 판정을 한다.
7. **평균 전략:** $\bar\sigma^W$는 구간 $W$ 동안 lifting된 전략들의 sequence-form 균등 평균이다. 정보집합별 자기 도달확률로 가중한다. 구현과의 차이는 §7.2에 있다.

## 2. 보조정리

**Lemma 1 (NashConv 분해) [증명; 이전 문서 Lemma 2.1–2.2].** (S1) 아래에서 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^W)\le\frac1{\lvert W\rvert}\sum_{I\in\mathcal I}\psi(R_I^W)=\frac{C^W+H^W}{\lvert W\rvert}.$$
- 첫 부등식은 영합 항등식에 Zinkevich et al. 2007 Thm 3을 구간에 적용한 것이다. lifting된 전략도 $G$의 behavioral 전략이므로 그대로 적용된다.
- 등식은 $H$의 정의다.

**Lemma 2 (라벨 항등식) [증명, 수치 확인].** 임의의 bucket $B$와 구간 $W$에 대해 다음이 성립한다.
$$H_B^W=\min_{\ell\in A_B\cup\{\varnothing\}}\ \sum_{I\in B}e_I(\ell),\qquad e_I(\ell):=\psi(R_I^W)-R_I^W(\ell)\ \ge 0.$$
특히 $H_B=0$일 필요충분조건은 모든 멤버에 대해 동시에 $\psi$를 달성하는 공통 라벨 $\ell$이 있는 것이다. 이를 "공통 확장 argmax"라 부른다.

*증명.*
- $x\mapsto x(\ell)$은 선형이므로 $\psi(R_B)=\max_\ell R_B(\ell)=\max_\ell\sum_{I\in B}R_I(\ell)$다.
- 이를 $\sum_I\psi(R_I)$에서 빼면 바로 식이 나온다.
- $e_I(\ell)\ge0$은 $\psi(x)\ge x(\ell)$에서 나온다. □

*의미:*
- $e_I(\ell)$은 "멤버 $I$에게 라벨 $\ell$을 강요할 때 잃는 regret"이다.
- 이전 문서 Thm 3.1의 Bregman 형태 $\sum W_ID_\psi(\rho_I,\bar\rho_B)$는 $\ell$을 $R_B$의 확장 argmax로 고른 경우와 같다. Lemma 2는 그 선택이 모든 라벨 중 최적이라는 것까지 말해 준다.
- 라벨 분할 규칙은 이 공통 라벨이 존재하는지를 통계적으로 검사하는 절차다.

**Lemma 3 (샘플 regret에 대한 RM+ 구간 상계) [증명, 수치 확인].** bucket $B$가 시점 $\tau_B$에 생겼고, 구간 $W=(s,t]$에서 $s\ge\tau_B$라 하자. 그러면 다음이 성립한다.
$$\psi(\tilde R_B^W)\le\lVert Q_{B,t}\rVert_\infty\le D_B\sqrt{\lvert A_B\rvert\,(t-\tau_B)}\le D_B\sqrt{\lvert A_B\rvert\,t}.$$

*증명.* $x_u=\tilde r_u(B)$로 두자.
1. **직교성:** 방문된 각 멤버에서 $\tilde r_u(I)=\tilde v-\langle\sigma_u,\tilde v\rangle\mathbf 1$이다. 그래서 $\langle\sigma_u,\tilde r_u(I)\rangle=0$이고, 멤버들을 합해도 $\langle\sigma_u,x_u\rangle=0$이다.
2. **노름 상계:** $Q_u\ge0$이고 $\sigma_u\propto Q_u$이므로 $\langle Q_u,x_u\rangle=0$이다. 따라서 $\lVert Q_{u+1}\rVert_2^2\le\lVert Q_u+x_u\rVert_2^2=\lVert Q_u\rVert_2^2+\lVert x_u\rVert_2^2$이고, 귀납법으로 $\lVert Q_t\rVert_2^2\le\sum_{\tau_B\le u<t}\lVert x_u\rVert_2^2\le\lvert A_B\rvert D_B^2(t-\tau_B)$다.
3. **구간 합:** $Q_{u+1}\ge Q_u+x_u$이므로 $\sum_{u\in W}x_u\le Q_t-Q_s\le Q_t$다(성분별, $Q_s\ge0$). □

샘플링이 있어도 RM+의 구간 성질은 확률 없이 그대로 성립한다. 직교성이 "같은 $\sigma_u$를 쓰는 멤버들의 합"에서도 유지되기 때문이다.

**Lemma 4 (시간 균일 집중) [증명].** $X_u(I,a)=\tilde r_u(I,a)-r_u(I,a)$로 두고 다음을 정의한다.
$$\varepsilon_I(t):=4\Delta_I\sqrt{t\,\ln\!\Big(\frac{\pi^2N_A\,(1+\log_2t)^2}{3\delta}\Big)}.$$
그러면 사건 $\mathcal E_\delta$ = {모든 $I,a,t\ge1$에서 $\lvert\sum_{u\le t}X_u(I,a)\rvert\le\varepsilon_I(t)$}의 확률은 $1-\delta$ 이상이다. $\mathcal E_\delta$ 위에서는 모든 구간 $W=(s,t]$에 대해 다음이 성립한다.
$$\lVert R_I^W-\tilde R_I^W\rVert_\infty\le2\varepsilon_I(t).$$

*증명.*
1. (S2)에 의해 $X_u$는 $\lvert X_u\rvert\le2\Delta_I$인 마팅게일 차분열이다.
2. Azuma 부등식의 최대 버전을 블록 $[2^j,2^{j+1})$에 적용하면 다음을 얻는다.
   $$\Pr\Big(\max_{t<2^{j+1}}\big\lvert\textstyle\sum_{u\le t}X_u\big\rvert\ge\lambda_j\Big)\le2\exp\!\Big(-\frac{\lambda_j^2}{16\cdot2^j\Delta_I^2}\Big).$$
3. $\delta_j=6\delta/(\pi^2(j+1)^2)$로 두고 $\lambda_j=4\Delta_I\sqrt{2^j\ln(2N_A/\delta_j)}$를 고르면, 이 확률은 $\delta_j/N_A$다.
4. $t\in[2^j,2^{j+1})$이면 $2^j\le t$이고 $j+1\le1+\log_2t$이므로 $\lambda_j\le\varepsilon_I(t)$다.
5. $(I,a)$ $N_A$개와 $j$에 대해 합집합 상계를 쓰면 실패 확률은 $\sum_j\delta_j=\delta$ 이하다.
6. 구간 합은 두 누적합의 차이이므로 $2\varepsilon_I(t)$로 묶인다. □

판정 구간의 시작과 끝이 분할 시점, 즉 확률적인 정지시간이어도 상관없다. 시간 균일 상계이기 때문이다. 이 점이 방식 G에 필요하다.

**Lemma 5 (라벨이 같은 bucket의 상쇄량) [증명, 수치 확인].** $\mathcal E_\delta$ 위에서, 구간 $W=(s,t]$의 판정에서 bucket $B$의 모든 멤버가 같은 라벨 $\ell$을 받았다면 다음이 성립한다.
$$H_B^W\le\sum_{I\in B}e_I(\ell)\le\sum_{I\in B}\big(\theta_I^W+4\varepsilon_I(t)\big).$$

*증명.* 첫 부등식은 Lemma 2다. $\eta_I=2\varepsilon_I(t)$로 두고 경우를 나눈다.
- **$\ell=\varnothing$:** 검정에 실패했거나 방문이 없었으므로 $\psi(\tilde R_I)\le\theta_I$다. $\psi$가 $\lVert\cdot\rVert_\infty$에 대해 1-Lipschitz이므로 $e_I(\varnothing)=\psi(R_I)\le\psi(\tilde R_I)+\eta_I\le\theta_I+\eta_I$다.
- **$\ell=a\in A$:** $a$는 $\tilde R_I$의 argmax이고 $\tilde R_I(a)>\theta_I\ge0$이다.
  - $\max_bR_I(b)\ge0$이면 $e_I(a)=\max_bR_I(b)-R_I(a)\le(\tilde R_I(a)+\eta_I)-(\tilde R_I(a)-\eta_I)=2\eta_I$다.
  - $\max_bR_I(b)<0$이면 $e_I(a)=-R_I(a)\le-\tilde R_I(a)+\eta_I<\eta_I$다.
- 두 경우 모두 $e_I\le\theta_I+2\eta_I$다. □

검정의 통계적 타당성(가짜 라벨 확률)은 이 증명에 쓰이지 않는다. 쓰이는 것은 역치 크기 조건 (T)뿐이다. 가짜 라벨은 분할을 늘릴 뿐이고, 분할은 Lemma 6에 의해 유한하다.

**Lemma 6 (분할 유한성) [증명].** 분할마다 자식은 실제 멤버를 가진 둘 이상의 묶음이다. 따라서 bucket 수는 매 분할마다 엄격히 늘고, $N$을 넘지 못한다. 분할 횟수와 분할이 일어난 판정의 수는 모든 표본 경로에서 $N-\lvert\mathcal P_0\rvert$ 이하다.
- 방문되지 않은 멤버도 라벨 $\varnothing$을 가진 실제 멤버다.
- 7-stud의 rest bucket에서는 아직 보지 못한 클러스터가 이런 멤버다.
- 그래서 $\varnothing$ 자식도 비어 있지 않다. □

## 3. 정리 1: epoch 방식

**Theorem 1 [증명].** (S1)–(S3), (T), 방식 E를 가정한다. $\mathcal E_\delta$ 위에서(확률 $\ge1-\delta$), 끝에서 분할이 일어나지 않은 **모든** epoch $k$에 대해 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\sum_{B\in\mathcal P_k}D_B\sqrt{\lvert A_B\rvert\,t_{k+1}}+\sum_{I\in\mathcal I}\big(\theta_I^{(k)}+6\,\varepsilon_I(t_{k+1})\big)\Big].$$
$k^\*$를 마지막으로 분할이 일어난 epoch라 하자. Lemma 6에 의해 $k^\*<\infty$이고, $k>k^\*$인 모든 epoch에 위 식이 적용된다. 정규 검정($\theta\le z\Delta_I\sqrt{L_k}$)과 두 배 epoch($\gamma=2$)를 쓰면 다음이 된다.
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac{\sqrt2\sum_BD_B\sqrt{\lvert A_B\rvert}+\sum_I\Delta_I\Big(z+24\sqrt{2\ln\frac{\pi^2N_A(2+\log_2L_k)^2}{3\delta}}\Big)}{\sqrt{L_k}}=O\!\Big(\sqrt{\tfrac{\log(kN_A/\delta)}{L_k}}\Big).$$
또한 거의 확실하게 $\mathrm{NashConv}(\bar\sigma^{(k)})\to0$이다.

*증명.*
- **분해:** epoch $k$ 동안 분할 $\mathcal P_k$는 고정이다. 그 끝에서 분할이 없었으므로 모든 bucket의 라벨이 하나다. Lemma 1로 $C$와 $H$로 나눈다.
- **$C$ 항:** $\psi(R_B)\le\psi(\tilde R_B)+\sum_{I\in B}2\varepsilon_I(t_{k+1})$이고(Lemma 4), $\psi(\tilde R_B)\le D_B\sqrt{\lvert A_B\rvert\,t_{k+1}}$이다(Lemma 3).
- **$H$ 항:** Lemma 5를 쓴다. 두 항을 더하면 정리의 식이 된다.
- **거의 확실한 수렴:** 사건 $\{\limsup_k\mathrm{NashConv}(\bar\sigma^{(k)})>0\}$은 모든 $\delta$에 대해 $\mathcal E_\delta^c$에 포함된다. 따라서 그 확률은 0이다. □

**Corollary 3.1 (실행 전체의 균등 평균) [증명].** $T=t_{K+1}$($K>k^\*$)에서 $\bar\sigma^{(0,T]}$의 NashConv는 다음 이하다.
$$\frac1T\Big[\sum_I\Delta_I\,t_{k^\*+1}+\sum_{k^\*<k\le K}L_k\cdot\mathrm{(정리 1의 우변)}_k\Big]=O\!\Big(\frac{t_{k^\*+1}}T+\sqrt{\frac{\log}{T}}\Big).$$
- $\psi$의 준가법성 $\psi(\sum_kx_k)\le\sum_k\psi(x_k)$로 epoch별로 나눈다.
- 분할 시기의 epoch는 $\psi(R_I^{(k)})\le\Delta_IL_k$로 묶는다.
- 나머지는 등비합이다. □

**Proposition 3.2 (분할 중인 epoch의 상계) [증명].** 끝에서 일부 bucket이 분할된 epoch $k$에서도 다음이 성립한다. $S_k$는 분할된 bucket의 집합이다.
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\text{C항}+\sum_{B\notin S_k}\sum_{I\in B}(\theta_I+4\varepsilon_I)+\sum_{B\in S_k}\min_\ell\sum_{I\in B}e_I(\ell)\Big].$$
- 분할 시기의 exploitability는 **아직 라벨이 갈리는 bucket들의 "소수파 regret"**이 좌우한다.
- 실험의 41M 실행은 모두 이 시기에 있었다(§7.1-1). 이 식을 정확 진단으로 측정하는 것이 다음 실험 항목이다(§8).

**Remark 3.3 (속도 비교).**
- $\sum_BD_B\le\sum_I\Delta_I$다. 그래서 $k^\*$ 이후 정리 1의 차수는 완전 기억 CFR+의 상계 $\sum_I\Delta_I\sqrt{\lvert A\rvert/T}$와 로그 인자를 빼면 같다.
- 손해는 상수($z$, $\varepsilon$ 계수)와 $t^\*$까지의 지연뿐이다. 다만 $t^\*$ 자체는 묶지 못한다.

**Remark 3.4 (자식 표 상속).**
- 자식이 부모의 $Q$를 복사하거나 비율로 나눠 받아도 된다. 조건은 시작 노름이 $\lVert Q_{\text{자식}}\rVert_2\le\lVert Q_{\text{부모}}\rVert_2$인 것이다.
- 그러면 Lemma 3의 $\sqrt{t-\tau_B}$ 자리에 부모의 생성 시점이 들어갈 뿐이고, 같은 차수가 유지된다.

## 4. 정리 2: 수렴 조건부(gated) 방식

**Theorem 2 [증명].** (S1)–(S3), (T), 방식 G를 가정한다. 임의의 $\varepsilon_g>0$, $g,m\ge1$에 대해 다음이 성립한다.
1. **(i) 판정은 반드시 일어난다.** 어떤 분할이 고정된 시점 $s$ 이후 $\tilde C^{(s,t]}/(t-s)\le\sum_BD_B\sqrt{\lvert A_B\rvert t}/(t-s)\to0$이다(Lemma 3). 그래서 판정이 끝없이 미뤄지지 않는다.
2. **(ii) 분할은 유한하다.** 마지막 분할 시점 $s^\*$가 존재한다(Lemma 6).
3. **(iii) 수렴한다.** $\mathcal E_\delta$ 위에서, $s^\*$ 이후 분할 없이 판정을 통과한 모든 시점 $t$에 대해 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^{(s^\*,t]})\le\frac{\sum_BD_B\sqrt{\lvert A_B\rvert t}+\sum_I(\theta_I+6\varepsilon_I(t))}{t-s^\*}=O\!\Big(\frac{\sqrt{t\log t}}{t-s^\*}\Big)\to0.$$
   - 정규 검정이면 $\theta=O(\sqrt t)$라서 $\log t$ 자리가 $\log\log t$가 된다.
   - (i)에 의해 그런 $t$는 일정 시점 이후 $g$반복마다 있다.

증명은 정리 1과 같다. 판정 구간의 끝이 정지시간이어도 Lemma 4가 시간 균일이므로 그대로 성립한다.
- Hoeffding과 Bernstein의 판정 번호 $d$는 많아야 $t/g$다. 그래서 $\ln((d+1)^2/\delta)=O(\log t)$이고 조건 (T)가 유지된다. □

### 4.1 ε 정지 인증: 게이트의 ε가 정하는 보장 수준

정리 2는 "언젠가 0으로 간다"는 말이다. 하지만 게이트에는 더 직접적인 쓸모가 있다. **분할 없이 통과한 판정 시점마다 그 시점의 정밀도를 인증한다.** CFR+IRA는 목표 정밀도 ε를 입력으로 받고, 정확한 best response로 원게임 exploitability를 계산해 멈춘다. 샘플링 버전에서는 이 정지 판정을 "게이트 통과 + 라벨 일치"가 대신한다.

**Theorem 2′ (ε 정지 인증) [증명].** (S1)–(S3), 방식 G를 가정한다. $\mathcal E_\delta$ 위에서, 판정 구간 $W=(s,t]$의 게이트 통과에서 **어떤 bucket도 분할되지 않았다면** 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^W)\ \le\ \frac{\tilde C^W}{\lvert W\rvert}+\frac1{\lvert W\rvert}\sum_{I}\big(\theta_I^W+6\varepsilon_I(t)\big)\ \le\ \varepsilon_g+\underbrace{\frac{\sum_I\theta_I^W}{\lvert W\rvert}}_{\text{검정 역치 항}}+\underbrace{\frac{6\sum_I\varepsilon_I(t)}{\lvert W\rvert}}_{\text{샘플 오차 항}}.$$
일부 bucket $S$가 분할된 통과에서는 다음 항이 더해진다.
$$\frac1{\lvert W\rvert}\sum_{B\in S}\min_\ell\sum_{I\in B}e_I(\ell).$$

*증명.*
- $C^W\le\tilde C^W+2\sum_I\varepsilon_I(t)$이다(Lemma 4).
- 게이트 통과이므로 $\tilde C^W\le\varepsilon_g\lvert W\rvert$이다.
- $H^W$는 Lemma 5로 묶는다. Lemma 1로 합친다. □

**뜻: 판정 시점의 보장 수준(정밀도 바닥)은 두 손잡이가 정한다.**
- **ε:** bucket 게임 자체의 수렴 정도($C$ 항)를 정한다.
- **검정 역치 θ:** 라벨이 일치할 때 bucket 안 상쇄($H$ 항)가 얼마나 남을 수 있는지를 정한다.
  - 정규 검정은 $\theta\approx z\sqrt{\sum\tilde r^2}$로 작다. 그래서 라벨이 일치하면 $H$가 작다.
  - Hoeffding과 Bernstein은 θ가 보상 폭 $\Delta_I$와 로그 인자에 비례해 크다. 그래서 라벨이 모두 "없음"으로 일치해도 $H$가 크게 남을 수 있다.
- 구간이 길어지면 두 역치 항은 $O(1/\sqrt{\lvert W\rvert})$로 줄어든다. ε 자리에 실제로 측정한 $\tilde C^W/\lvert W\rvert$를 넣으면 바닥은 ε보다 아래로도 내려간다.
- **사용법:** "분할 없는 통과가 나오면 $\bar\sigma^W$를 인증과 함께 출력하고 멈춘다." 이것이 샘플링 버전의 정지 규칙이다. 마지막 분할 시점 $t^\*$를 몰라도 쓸 수 있다.

**41M 실험에서 확인한 것** (`decisions.csv`의 정확 진단, mini-stud small, seed 1)

| 실행 | 게이트 통과 | 분할 없는 통과 | 분할 없는 통과의 정확 $C/\lvert W\rvert$ | 같은 시점의 정확 $H/\lvert W\rvert$ |
|---|---:|---:|---:|---:|
| 정규 ε=0.05 / 0.1 / 0.2 / 0.4 | 4 / 6 / 11 / 40 | **0** | – | – |
| Bernstein ε=0.1 | 34,495 | 34,397 | 0.019–0.107 (평균 0.044) | 2.23–2.76 (평균 2.54) |

- **정규 검정 실행:** 게이트를 통과할 때마다 항상 분할이 일어났다. 즉 41M 동안 인증 조건이 한 번도 성립하지 않았다($t^\*$ 이전이라는 것과 같은 말이다).
  - 다만 ε=0.4의 마지막 판정들에서 정확 $H/\lvert W\rvert$는 0.0008까지 내려갔다. 남은 것은 거의 $C$ 항(약 0.35)이었다.
- **Bernstein 실행:** 인증 조건은 자주 성립했다. 하지만 라벨이 대부분 "없음"으로 일치한 것이고, 엄격한 역치 때문에 $H/\lvert W\rvert\approx2.5$가 남았다.
  - 마지막 통과의 정확 상계는 $(C+H)/(2\lvert W\rvert)=1.30$ ante였다(exploitability 단위, 구간 균등 평균 $\bar\sigma^W$ 기준).
  - 실측 exploitability는 0.467이었다. 다만 이 값은 실행 전체의 선형 평균 bucket 정책에서 잰 것이라 같은 대상은 아니다.
  - 위 식의 "검정 역치 항"이 바닥을 정한 사례다. Bernstein이 정체한 이유를 정량적으로 설명한다.

**한계: 엄밀한 인증은 수치적으로 공허하다.** 샘플 오차 항을 mini-stud small(정보집합 51,114개, $\sum\Delta_I=566{,}366$, $N_A=142{,}428$, $\delta=0.05$)에 넣어 계산했다. $t=4.1\times10^7$, $\lvert W\rvert=5.35\times10^5$일 때 $6\sum\varepsilon_I/\lvert W\rvert\approx7.7\times10^5$이다. 실제 NashConv(약 1) 대비 수십만 배 느슨하다.
- 원인 1: 모든 정보집합에 대한 합이다.
- 원인 2: 최악의 경우 Azuma 상계다.
- 실용적인 인증은 다음 둘 중 하나가 필요하다.
  - 방문 확률에 비례하는 분산을 쓰는 Freedman형 상계: $\sum_I\sqrt{n_I}\le\sqrt{N\sum_In_I}$ 덕분에 크게 줄지만, 여전히 큰 게임에서는 크다.
  - 판정 표본과 평가 표본을 나눈 교차 적합으로 $\sum_B\min_\ell\sum\hat e_I(\ell)$을 직접 추정하는 플러그인 인증(편향 있음).
- 이것은 [미해결]이다.

**Remark 4.1 (게이트가 실제로 보장하는 것).**
- $\mathcal E_\delta$ 위에서 판정 통과는 정확한 bucket regret에 대해 $C^W\le\varepsilon_g\lvert W\rvert+2\sum_I\varepsilon_I(t)$를 뜻한다.
- 반대로 $\mathbb E\,\psi(\tilde R_B)\ge\psi(R_B)$(Jensen)이므로 샘플 $C$는 위로 편향된다. 실험에서 bucket 2만 개 이상일 때 최대 1.18배였다.
- 그래서 게이트는 보수적으로(늦게) 통과한다.
- 게이트는 **수렴에 필요하지 않다.** 정리 1은 게이트 없이도 성립한다. 게이트의 목적은 "수렴한 추상 게임의 regret으로 라벨을 붙인다"는 해석과 분할 수 절약이다.

## 5. 정리 3: 원자(atom) 일반화와 7-stud

7-stud에서는 원 정보집합이 학습 중 거의 한 번씩만 나온다. 그래서 정보집합별 라벨은 유의해질 수 없다. 라벨 단위를 **원자**로 바꾼다.
- **원자의 정의:** $\mathcal A$는 $\mathcal I$의 분할이다. 같은 player, 같은 행동집합끼리 묶는다.
- **원자와 bucket의 관계:** bucket은 원자의 합집합이다. 원자가 분할의 가장 미세한 단위다.
- **7-stud 구현의 원자:** power 클러스터(1024) × 베팅 요약 × 합법 행동 집합이다.
- **원자 단위 라벨:** 라벨은 원자 합 $\tilde R_\alpha=\sum_{I\in\alpha}\tilde R_I$로 붙인다. $\Delta_I$ 자리에는 $D_\alpha$(한 순회에서 방문되는 원자 멤버들의 $\Delta$ 합의 최댓값), $N_A$ 자리에는 $\sum_\alpha\lvert A(\alpha)\rvert$가 들어간다.

**Theorem 3 [증명].** 위의 치환 아래 $\mathcal E_\delta$ 위에서, 분할이 없었던 epoch마다 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\sum_BD_B\sqrt{\lvert A_B\rvert t_{k+1}}+\sum_\alpha(\theta_\alpha+6\varepsilon_\alpha(t_{k+1}))+H_{\mathcal A}^{(k)}\Big],$$
$$H_{\mathcal A}^{(k)}:=\sum_\alpha\Big[\sum_{I\in\alpha}\psi(R_I^{(k)})-\psi(R_\alpha^{(k)})\Big]\ \ge0.$$

*증명.* 다음 항등식에서 출발한다.
$$\sum_I\psi(R_I)=\sum_\alpha\psi(R_\alpha)+H_{\mathcal A}=C+\sum_BH_{B\mid\mathcal A}+H_{\mathcal A}.$$
여기서 $H_{B\mid\mathcal A}=\sum_{\alpha\in B}\psi(R_\alpha)-\psi(R_B)$다. 앞의 두 항에 정리 1의 증명을 원자 단위로 그대로 적용한다. □

**해석과 한계:**
- ASWISS가 0으로 보내는 것은 "bucket과 원자 사이의 상쇄"까지다. 원자 안의 상쇄 $H_{\mathcal A}$는 남는다.
- 원자 분할을 처음부터 그대로 학습한 경우(7-stud의 k-means 1024 비교군)도 같은 꼴의 상계 $(C_{\mathcal A}+H_{\mathcal A})/L$을 가진다.
  - 따라서 정리 3의 의미는 "ASWISS는 점근적으로 원자 추상화와 같은 형태의 보장을 더 적은 bucket으로 얻는다"이다.
  - 다만 $H_{\mathcal A}$는 학습 궤적에 따라 달라지므로 "같은 값"이라는 뜻은 아니다.
- **7-stud 원자는 두 가지 면에서 imperfect recall이다.**
  - 카드를 클러스터로 묶는다.
  - 베팅 이력 대신 요약을 쓴다.
  - 그래서 $H_{\mathcal A}/L\to0$을 보장하는 것은 없다. 7-stud 실험으로 보일 수 있는 것은 "원자 추상화 대비 효율"까지다. 원게임 균형으로의 수렴은 아니다.
- **7-stud의 "원게임"은 4구 heuristic(H4)을 고정한 5–7구 게임이다.**
- **mini-stud는 원자 = 정보집합이므로 $H_{\mathcal A}=0$이고 정리 1이 그대로 적용된다.**

## 6. CFR+IRA와의 관계

비교 대상은 Čermák, Lisý, Bošanský의 CFR+IRA다(번역 문서 4.3.3절 기준).

| 항목 | CFR+IRA | ASWISS (이 문서) |
|---|---|---|
| regret 계산 | 매 반복 전체 트리 순회 (정확값) | external sampling (추정값) |
| 감시 대상 | 무작위 $k_b$개 정보집합, epoch마다 다시 뽑음 | 모든 멤버 ($p=1$) |
| 분할 기준 | 감시 정보집합의 자기 구간 regret이 RM+ 속도 역치를 넘으면 그 정보집합만 떼어 냄 | 구간 누적 샘플 regret의 라벨이 멤버끼리 다르면 라벨별로 나눔 |
| 아직 묶여 있는 정보집합의 regret | 원문에 유도가 없음 (번역 역자 주 8) | Lemma 2(항등식)와 Lemma 5로 직접 상계 |
| 보장 | 확률 $1-\delta$, $T\ge(\alpha(\delta)+1)^3$에서 평균 external regret $O(T^{-1/3})$. $\alpha$는 $\lvert\mathcal I\rvert$에 지수적 | 마지막 분할 이후 $O(\sqrt{\log k/L_k})$. 마지막 분할 시점의 상계는 없음 |
| 확률의 출처 | 감시 표본 추출 | 게임 샘플링(ES) |
| 학습 메모리 | 추상화 표 + 감시 $k_b$개 | 추상화 표 + 모든 멤버의 구간 합 ($\lvert\mathcal I\rvert$ 규모) |

**새로 주장할 수 있는 후보** (문헌 원문으로 아직 확인하지 않음):
1. 샘플 regret 라벨과 시간 균일 집중을 결합한 분할 판정
2. 라벨 항등식 $H_B=\min_\ell\sum e_I(\ell)$, 그리고 그것으로 묶인 멤버의 regret을 직접 상계하는 방법
3. 수렴 조건부 방식

**새롭지 않은 것** (CFR+IRA에 이미 있음): 두 배 epoch, 분할만 하는 정제, 구간 regret 감시.

## 7. 한계

### 7.1 정리가 보장하지 않는 것

1. **마지막 분할 시점 $t^\*$에 상계가 없다. [미해결]**
   - CFR+IRA도 이 시점을 직접 구하지 못했다. 원문의 $T_{lim}$은 확률적 최악 경우 논증이고 $\lvert\mathcal I\rvert$에 지수적이다.
   - 이 문서는 $t^\*$를 구하는 대신 정지 규칙(정리 2′)으로 우회한다.
   - 속도 보장은 $t^\*$ 이후에만 있다.
   - 실험과 연결하면 다음과 같다. mini-stud 41M 실행은 모두 $t^\*$ 이전이었다. 조건 없는 정규 방식은 마지막 epoch 끝(40.95M)에도 bucket 2,333개를 분할했고, ε=0.4도 마지막 판정에서 28개를 분할했다.
   - 따라서 **관찰된 수렴 곡선의 속도는 정리 1이 설명하지 않는다.** 분할 시기의 상계는 Proposition 3.2뿐이다.
   - 후보 해법은 margin 조건이다. 예를 들어 모든 멤버의 상위 두 행동 누적 regret 차이가 $\Omega(L)$이라는 조건 아래 $t^\*$를 묶어 볼 수 있다.
2. **최종 bucket 수에 보장이 없다.**
   - 최악의 경우 완전 기억(정보집합마다 bucket 하나)까지 간다.
   - 고정 수준의 정규 $z$ 검정은 가짜 라벨이 무한히 자주 나올 수 있다. 그래서 원리적으로 완전 기억까지 쪼개질 수 있다. 수렴은 유지되지만 압축 이득은 사라진다.
   - 누적 오경보율이 유한한 검정(아래 7.2의 엄밀한 버전)을 쓰면 가짜 분할은 확률 $1-\delta$로 멈춘다. 대신 탐지가 느려진다(실험의 엄격 검정 정체와 같은 현상).
   - §9.2 Lemma 9의 여유 라벨 집합은 좋은 사건 위에서 가짜 분할을 아예 없앤다.
3. **학습 메모리를 줄이지 않는다.**
   - 라벨 계산을 위해 모든 멤버의 구간 합 $\tilde R_I$(정규 검정이면 $\sum\tilde r^2$도)를 저장한다. 이는 $\lvert\mathcal I\rvert\cdot\lvert A\rvert$ 규모로, 완전 기억 CFR의 regret 표와 같은 차수다.
   - 부분 추적($p<1$)에서는 Lemma 5의 전제("모든 멤버의 라벨이 같다")를 확인할 수 없다. 그래서 증명이 성립하지 않는다. 이전의 집중 순회 실험도 성능이 낮았다.
   - 7-stud는 원자가 정보집합보다 훨씬 거칠어서 추적 메모리가 원자 수 규모다. 실용적이지만 정리 3의 한계가 따라온다.
   - §9.3의 방법 A(저장한 순간 전략 재생)는 이 추적을 bucket 표 $K_k$개로 바꾼다. 대가는 $H$ 항의 속도다.
4. **보장 대상 $\bar\sigma$는 정보집합별 평균이라 압축되지 않는다.**
   - $\bar\sigma$는 자기 도달확률로 가중하므로, 저장하려면 $\lvert\mathcal I\rvert$ 규모가 필요하다.
   - bucket 표의 평균 $\tilde\sigma$는 멤버들이 자기 abstract 경로를 공유하지 않으면 $\bar\sigma$와 다르다(이전 문서 Prop 5.1). 따라서 **압축된 배포 정책에는 보장이 없다.**
   - 대안은 반복 혼합 배포다. 저장한 checkpoint의 현재 정책 중 하나를 핸드 시작마다 무작위로 골라 둔다. Kuhn 정리에 의해 그 반복들의 평균과 같다. 수렴하려면 혼합에 쓰는 checkpoint 수가 늘어나야 한다.
   - mini-stud에서는 두 평균이 가까웠다(40.95M, ε=0.4: bucket 평균 0.0216, 정보집합별 평균 0.0198). 이는 경험적 사실일 뿐이다.
5. **상수가 크다.** 상계에 모든 정보집합(또는 원자)의 $\Delta$ 합이 들어간다. 7-stud 원자 수(수십만)에서는 현실적인 반복 수에서 의미 없는 크기다. 완전 기억 CFR의 상계도 같은 성질이 있다.
6. **현재 정책(마지막 반복)에는 보장이 없다.** 평균 전략만 보장한다.
7. **충분조건일 뿐이다.**
   - Lemma 1은 $\sum\psi(R_I)$로 상계한다. 이전 문서 Remark 2.4의 반례처럼, 자기 도달확률이 0으로 가는 정보집합의 regret은 NashConv에 영향을 주지 않는데도 $H$를 키운다.
   - 라벨 규칙은 이런 경로 밖 정보집합 때문에 불필요한 분할을 할 수 있다. 정확성 손해는 없고 효율 손해만 있다.
8. **실행 간 단조성이 없다.**
   - 분할은 이후 궤적을 바꾼다. 그래서 "bucket이 많을수록 같은 반복에서 더 좋다"는 단조성은 없다(abstraction pathology).
   - 41M 실험에서도 bucket 수와 exploitability 순서가 일치하지 않았다.
9. **외부 샘플링(ES)에만 해당한다.**
   - outcome sampling은 중요도 가중 때문에 $\lvert\tilde r\rvert\le\Delta/q$가 되어 크기가 커진다.
   - 탐험 하한 $q\ge q_{\min}$이 있으면 같은 논증에 상수만 바뀐다.

### 7.2 정리와 현재 구현의 차이

현재 실험(mini-stud, 7-stud)은 정리의 가정을 정확히 만족하지 않는다.

| 항목 | 정리 | 구현 | 영향과 맞추는 방법 |
|---|---|---|---|
| 갱신 순서 | 동시 갱신 (S1) | 교대 갱신: player 0 순회 후 적용, 그다음 player 1 순회 | Lemma 1의 항등식은 두 player가 같은 프로필 열을 쓸 것을 요구한다. 교대 갱신 CFR+는 Burch, Moravčík, Schmid 2019가 다뤘지만, bucket 공유와 샘플링이 겹친 경우는 확인하지 않았다. **맞추기:** 두 순회 뒤 한 번에 적용하는 변형으로 재실험한다(코드 한 줄). |
| 평균 | 판정 구간의 균등 평균, 또는 $k^\*$ 이후 균등 평균 | 실행 전체 선형 가중, 분할된 자식은 평균을 0부터 다시 쌓음 | 선형 가중에 분할이 겹치면 구간 안 부분합을 통제해야 하는데, 라벨은 구간 전체 합만 본다. 그래서 현재 증명 밖이다. **맞추기:** epoch 균등 평균 표를 추가한다. |
| 라벨 검정 | 역치 크기 조건 (T)만 요구 | 정규 $z=2$. Hoeffding/Bernstein 구현은 구간 길이 대신 방문 수 $n$을 씀 | 수렴에는 영향이 없다. 다만 "가짜 라벨을 확률 $\delta$로 억제한다"고 주장하려면 Freedman 부등식이나 마팅게일용 경험 Bernstein(시간 균일 신뢰열) 같은 예측 분산 기반 검정이 필요하다. 현재 구현은 근사다. |
| 순회 간 deal 공유 | 각 순회가 독립 표본 | 7-stud와 mini-stud 모두 한 반복의 두 순회가 같은 chance 표본을 공유 | 불편성은 유지된다. 상관은 집중 부등식에 영향이 없다(순회별 마팅게일). |
| 7-stud 원자 | 정리 1은 원자 = 정보집합 | 원자 = 클러스터 × 베팅 요약 | 정리 3만 적용된다. |

### 7.3 해석상의 한계

- **seed:** mini-stud 실험의 seed가 하나다.
- **7-stud 평가:** Policy-LBR은 exploitability의 근사 하한이다. 샘플 BR-gap proxy는 상계도 하한도 아니다.
- **신규성:** 문헌 조사 때 원문 PDF를 읽지 못했다. 그래서 §6의 신규성 후보는 원문으로 확인해야 한다.

## 8. 다음 단계

1. **이론에 맞춘 재실험 (mini-stud):** 동시 갱신 + epoch 균등 평균 + Azuma 역치로 돌린다. 정리 1의 우변을 실제로 계산해 정확 NashConv와 함께 그린다. 상계가 얼마나 느슨한지도 보고한다.
2. **분할 시기 진단:** 정확 regret으로 Proposition 3.2의 "소수파 regret" $\sum_{B\in S_k}\min_\ell\sum e_I(\ell)$을 기록한다. 실험 곡선의 속도를 설명하는지 본다.
3. **정지 인증 기록:** `gated.cpp`의 판정마다 세 값을 함께 기록한다.
   - 정리 2′의 우변: 플러그인 버전과 엄밀 버전
   - 정확 $(C+H)/\lvert W\rvert$
   - 구간 균등 평균의 실제 NashConv

   그다음 실용적인 인증(Freedman형 또는 교차 적합)이 얼마나 조여지는지 본다.
4. **$t^\*$ 상계:** margin 조건 아래의 증명을 시도한다. [미해결]
5. **부분 추적:** $p<1$에서 확률적 탐지 논증(이전 Theorem 4.2의 Paley–Zygmund 방향)을 라벨 규칙에 맞게 다시 세운다. [미해결]
6. **배포:** 반복 혼합 배포의 checkpoint 수와 exploitability 관계를 정리한다.
7. **방법 A·B 실험 (mini-stud):** 정확 재생으로 세 규칙을 비교한다. 기존 구간 라벨, 방법 A(저장한 순간 전략), 방법 B(분할 시점 평균 전략), 그리고 기존 규칙 + 방법 B 분할의 결합이다. 정확 NashConv 대 bucket 수, 분할 횟수, 재시작으로 버린 학습량, 정리 B1의 $\rho_{\mathcal P}(\bar\sigma)$를 기록한다.
8. **방법 B의 오라클:** 경로 밖까지 순간 regret을 줄이는 학습기(마지막 반복 수렴 학습기, 떨림 섭동 CFR)를 찾고, 명제 B4(b)의 반례에서 확인한다. [미해결]

## 9. 고정 전략 판정: 저장한 순간 전략(방법 A)과 분할 시점 평균 전략(방법 B)

작성 2026-10-28. 사용자가 제안한 두 판정 방식을 이 문서의 틀로 옮겨 증명한다.
- 공통점: 판정 시점에 저장해 둔 전략으로 각 정보집합의 순간 regret을 다시 계산해 라벨을 붙인다. 그래서 학습 중에 방문하지 않은 정보집합에도 라벨이 붙는다.
- 차이점: 다시 계산에 쓰는 전략이 다르다.

| | 방법 A | 방법 B |
|---|---|---|
| 판정에 쓰는 전략 | 판정 구간 $W$의 반복 전략 $\sigma_t$ 중 무작위 시점 $K$개를 저장한 것 | 판정 시점의 평균 전략 $\bar\sigma$ 하나 |
| 라벨이 겨냥하는 양 | 구간 누적 regret $R_I^W=\sum_{t\in W}r_t(I)$ | 단일 프로필의 순간 regret $r^{\bar\sigma}(I)$ |
| 결과 | 정리 A: 정리 1과 같은 꼴이고, 같은 학습기(CFR+, ES)로 성립 | 정리 B1(인증), 정리 B3(오라클 조건 아래 유한 종료와 정확 수렴). 표준 CFR 평균은 그 조건을 만족하지 않음(명제 B4) |

### 9.1 추가 기호와 가정

- **단일 프로필 순간 regret:** 프로필 σ를 고정하면 $r^\sigma(I,a)=\pi^\sigma_{-i}(I)\big(q^\sigma(I,a)-\sum_b\sigma(I,b)\,q^\sigma(I,b)\big)$이다. 반복 $t$의 $r_t(I,a)$는 $r^{\sigma_t}(I,a)$다.
- **bucket 합과 진단량:**
  $$r^\sigma(B)=\sum_{I\in B}r^\sigma(I),\quad\rho(\sigma)=\sum_I\psi(r^\sigma(I)),\quad\rho_{\mathcal P}(\sigma)=\sum_B\psi(r^\sigma(B)),\quad H_B^\sigma=\sum_{I\in B}\psi(r^\sigma(I))-\psi(r^\sigma(B)).$$
  - $\rho_{\mathcal P}$는 bucket들이 스스로 느끼는 순간 regret의 합이다. 구간 버전 $C^W$의 단일 프로필 판이다.
- **최대 라벨 집합과 일치:** $M^\sigma(I)=\arg\max_{\ell\in A(I)\cup\{\varnothing\}}r^\sigma(I,\ell)$다. 동률이면 원소가 여러 개다.
  - bucket $B$가 **σ-일치**한다는 것은 $\bigcap_{I\in B}M^\sigma(I)\ne\emptyset$이라는 뜻이다. 모든 멤버가 함께 최대로 하는 라벨이 있다는 말이다.
  - 구간 버전도 $R_I^W$로 같게 정의한다.
- **(A1) 경로 분리:** 같은 bucket의 두 멤버는 한 경로 위에 있지 않다. 한쪽이 다른 쪽의 선행 정보집합이 아니라는 뜻이다.
  - 정리 B3의 (iii)(존재)에만 쓴다.
  - time step과 context 분할은 한 street 안에서 같은 player의 연속 결정이 베팅 수나 체크 여부로 구별되므로 만족한다고 본다. 코드로 확인하지는 않았다.
- **최선 응답:** $\beta_i$는 원게임 $G$에서 상대 전략에 대한 순수 최선 응답이다. $\pi_i^{\beta}(I)\in\{0,1\}$은 β를 따를 때 $I$에 이르는 자기 도달확률이다.

### 9.2 공통 보조정리

**Lemma 7 (NashConv의 정확한 분해) [증명, 수치 확인].** $\bar\sigma^W$를 구간 $W$의 sequence-form 평균, $\beta_i$를 $\bar\sigma^W_{-i}$에 대한 원게임 순수 최선 응답이라 하자. 그러면 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^W)=\frac1{\lvert W\rvert}\sum_i\sum_{I\in\mathcal I_i}\pi_i^{\beta_i}(I)\,R_I^W\big(\beta_i(I)\big).$$
특히 단일 프로필($\lvert W\rvert=1$)이면 다음과 같다.
$$\mathrm{NashConv}(\sigma)=\sum_i\sum_{I\in\mathcal I_i}\pi_i^{\beta_i}(I)\,r^\sigma\big(I,\beta_i(I)\big)\le\rho(\sigma).$$

*증명.*
1. **성능 차이 항등식:** 고정된 σ와 player $i$의 순수 전략 β에 대해 $u_i(\beta,\sigma_{-i})-u_i(\sigma)=\sum_{I\in\mathcal I_i}\pi_i^\beta(I)\,r^\sigma(I,\beta(I))$이다.
   - player $i$의 정보집합을 자기 깊이 $d$(그 정보집합 전에 자기가 둔 수의 개수)로 나눈다. perfect recall이므로 잘 정의된다.
   - $\sigma^{(d)}$를 "깊이 $<d$에서는 β, $\ge d$에서는 $\sigma_i$"로 둔다. $\sigma^{(0)}=\sigma_i$이고, 깊이의 최댓값을 $D$라 하면 $\sigma^{(D+1)}=\beta$다.
   - $\sigma^{(d+1)}$과 $\sigma^{(d)}$는 깊이 $d$ 정보집합에서만 다르다. 그런 $I$의 history $h$에는 확률 $\pi_i^\beta(I)\,\pi_{-i}(h)$로 도달하고, 그 뒤로는 둘 다 σ를 따른다.
   - 그래서 효용 차이는 다음과 같다.
     $$\sum_{I:\,\text{깊이 }d}\pi_i^\beta(I)\sum_{h\in I}\pi_{-i}(h)\Big(q(h,\beta(I))-\sum_b\sigma(I,b)\,q(h,b)\Big)=\sum_{I:\,\text{깊이 }d}\pi_i^\beta(I)\,r^\sigma(I,\beta(I)).$$
   - $d$에 대해 더하면 된다.
2. 영합이므로 $\mathrm{NashConv}(\bar\sigma)=\sum_i\max_\beta u_i(\beta,\bar\sigma_{-i})$이다.
3. $\bar\sigma^W_{-i}$는 상대 실현계획의 평균이고, $u_i(\beta,\cdot)$는 그에 대해 선형이다. 그래서 $u_i(\beta,\bar\sigma_{-i})=\frac1{\lvert W\rvert}\sum_{t\in W}u_i(\beta,\sigma_{t,-i})$이다.
4. $\sum_iu_i(\sigma_t)=0$을 빼서 넣으면 $\mathrm{NashConv}(\bar\sigma^W)=\frac1{\lvert W\rvert}\sum_t\sum_i\big[u_i(\beta_i,\sigma_{t,-i})-u_i(\sigma_t)\big]$이다. 각 $t$에 1을 쓰고 더하면 식이 된다.
5. 부등식은 $\pi\le1$과 $R(\beta(I))\le\psi(R)$에서 나온다. □

- Lemma 1은 이 항등식의 상계 쪽이다.
- 등식에는 최선 응답의 자기 도달확률 $\pi^{\beta}$가 가중치로 붙는다. 그래서 최선 응답이 가지 않는 정보집합의 regret은 NashConv에 들어가지 않는다. §7.1-7의 "충분조건일 뿐"이 정확히 이 차이다.

**Lemma 8 (단일 프로필 라벨 항등식) [증명; Lemma 2의 $\lvert W\rvert=1$].** 임의의 σ와 분할 $\mathcal P$에 대해 다음이 성립한다.
$$\rho(\sigma)=\rho_{\mathcal P}(\sigma)+\sum_BH_B^\sigma,\qquad H_B^\sigma=\min_\ell\sum_{I\in B}\Big(\psi(r^\sigma(I))-r^\sigma(I,\ell)\Big).$$
그리고 $H_B^\sigma=0$일 필요충분조건은 $B$가 σ-일치하는 것이다.

*증명.*
- 앞의 두 식은 Lemma 2의 대수 그대로다. 구간 합 자리에 $r^\sigma$를 넣는다.
- 각 항이 0 이상이므로 $H_B^\sigma=0\iff\exists\ell\ \forall I:\ r^\sigma(I,\ell)=\psi(r^\sigma(I))\iff\bigcap_IM^\sigma(I)\ne\emptyset$이다. □

**Lemma 9 (여유 라벨 집합: 노이즈가 있어도 거짓 분할이 없다) [증명, 수치 확인].**

설정은 다음과 같다.
- 라벨이 겨냥하는 참 벡터를 $x_I$라 한다. 방법 A에서는 $R_I^W$, 방법 B에서는 $r^{\bar\sigma}(I)$다.
- 추정치 $\hat x_I$는 $\lVert\hat x_I-x_I\rVert_\infty\le\eta_I$를 만족한다. $\varnothing$ 성분은 둘 다 0이다.
- **여유 라벨 집합:** $\hat M(I)=\{\ell:\hat x_I(\ell)\ge\max_{\ell'}\hat x_I(\ell')-2\eta_I\}$로 둔다.
- **분할 규칙:** $\bigcap_{I\in B}\hat M(I)=\emptyset$이면 나눈다. 나눌 때는 각 멤버를 $\hat M(I)$의 대표 라벨(고정 순서의 첫 원소)별로 모은다.
- $e_I(\ell)=\psi(x_I)-x_I(\ell)$로 둔다.

그러면 다음이 성립한다.
1. **(i)** 참 최대 라벨은 늘 들어 있다: $M(I)\subseteq\hat M(I)$.
2. **(ii)** $\ell\in\hat M(I)$이면 $e_I(\ell)\le4\eta_I$이다.
3. **(iii) 거짓 분할이 없다:** 참으로 일치하는 bucket($\bigcap M\ne\emptyset$)은 나뉘지 않는다.
4. **(iv)** 나뉘지 않은 bucket은 $H_B\le4\sum_{I\in B}\eta_I$이다.
5. **(v) 잘못 배치되어도 손실은 4η 이내다:** 나뉠 때 각 멤버가 들어가는 자식의 라벨 $\hat c$에 대해 $e_I(\hat c)\le4\eta_I$이다.

*증명.* $\max_{\ell'}\hat x_I(\ell')\le\psi(x_I)+\eta_I$이다. 참 최대 라벨 $\ell^\*$에 대해서는 $\hat x_I(\ell^\*)\ge\psi(x_I)-\eta_I$이다.
- (i): $\ell\in M(I)$이면 $\hat x_I(\ell)\ge\psi(x_I)-\eta_I\ge\max\hat x_I-2\eta_I$다.
- (ii): $\ell\in\hat M(I)$이면 $x_I(\ell)\ge\hat x_I(\ell)-\eta_I\ge\max\hat x_I-3\eta_I\ge\hat x_I(\ell^\*)-3\eta_I\ge\psi(x_I)-4\eta_I$다.
- (iii)은 (i)에서, (iv)는 (ii)와 Lemma 2(또는 8)에서, (v)는 (ii)에서 나온다. □

- **현재 구현(argmax + 역치 θ)에는 (iii)이 없다.**
  - 동률이나 근소한 차이에서 argmax가 노이즈로 뒤집히면, 참으로 일치하는 bucket도 나뉜다.
  - §7.1-2의 "가짜 라벨이 무한히 자주 나올 수 있다"의 원인이다.
- 여유 라벨 집합을 쓰면 좋은 사건 위에서는 가짜 분할이 생기지 않는다. 이 개선은 방법 A·B뿐 아니라 기존 규칙에도 그대로 쓸 수 있다.

### 9.3 방법 A: 저장한 순간 전략 재생 [증명]

**알고리즘 (이론 버전)**
1. 학습은 §1.3 그대로다(ES + RM+, 방식 E). 다만 **멤버별 구간 합을 학습 중에 쌓지 않는다.**
2. epoch $k$를 시작할 때 시점 $\tau_{k,1},\dots,\tau_{k,K_k}$를 $W_k$에서 독립·균등하게 뽑아 둔다. 학습의 무작위성과는 독립이다. 그 시점이 오면 bucket 전략표(lifting된 $\sigma_\tau$)만 저장한다.
3. epoch 끝에서 저장한 전략마다 원게임을 한 번 정확히 순회해(작은 게임) 모든 정보집합의 $r_\tau(I)$를 계산한다. 그리고 $\hat R_I^{(k)}=\frac{L_k}{K_k}\sum_jr_{\tau_{k,j}}(I)$로 둔다. 학습 중 방문하지 않은 정보집합도 실제 값을 얻는다.
4. 라벨과 분할은 Lemma 9의 여유 라벨 집합으로 한다($\eta=\eta_I^{(k)}$, 아래).
- 방식 G처럼 구간 끝을 미리 모르면 저수지 표집(reservoir sampling)으로 $K$개를 비복원 균등 추출한다. Hoeffding 부등식은 비복원 추출에도 그대로 성립한다(Hoeffding 1963).

**Lemma 10 (재생 추정의 집중) [증명].** 다음과 같이 두자.
$$\eta_I^{(k)}:=L_k\Delta_I\sqrt{\frac2{K_k}\ln\frac{\pi^2N_A(k+1)^2}{3\delta}}.$$
사건 $\mathcal E'_\delta$ = {모든 $k,I,a$에서 $\lvert\hat R_I^{(k)}(a)-R_I^{(k)}(a)\rvert\le\eta_I^{(k)}$}의 확률은 $1-\delta$ 이상이다.

*증명.*
- 학습 궤적 전체를 조건으로 두면 $\tau$들은 독립·균등이다. 그래서 $X_j=L_k\,r_{\tau_j}(I,a)$는 평균이 $R_I^{(k)}(a)$인 독립 표본이고 $\lvert X_j\rvert\le L_k\Delta_I$다.
- Hoeffding 부등식으로 다음을 얻는다.
  $$\Pr\big(\lvert\hat R-R\rvert>\eta\big)\le2\exp\!\Big(-\frac{K_k\eta^2}{2L_k^2\Delta_I^2}\Big)=\frac{6\delta}{\pi^2N_A(k+1)^2}.$$
- $(I,a)$와 $k$에 대해 합집합 상계를 쓰면 $\sum_k6\delta/(\pi^2(k+1)^2)=\delta$다. □

**Theorem A [증명].** (S1)–(S3), 방식 E, 재생 라벨(Lemma 9, $\eta=\eta^{(k)}$)을 가정한다. $\mathcal E_\delta\cap\mathcal E'_\delta$ 위에서(확률 $\ge1-2\delta$) 다음이 성립한다.
1. 참으로 일치하는 bucket은 나뉘지 않는다.
2. 분할은 많아야 $N-\lvert\mathcal P_0\rvert$번이다(Lemma 6).
3. 끝에서 분할이 없었던 모든 epoch $k$에 대해 다음이 성립한다.
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\sum_BD_B\sqrt{\lvert A_B\rvert\,t_{k+1}}+2\sum_I\varepsilon_I(t_{k+1})\Big]+4\sum_I\Delta_I\sqrt{\frac2{K_k}\ln\frac{\pi^2N_A(k+1)^2}{3\delta}}.$$

*증명.*
- Lemma 1로 $C$와 $H$로 나눈다. $C$ 항은 정리 1과 같다(Lemma 3, 4).
- 분할이 없었으므로 모든 bucket이 여유 라벨로 일치했다. Lemma 9(iv)와 Lemma 10으로 $H^{(k)}\le4\sum_I\eta_I^{(k)}$다.
- $L_k$로 나누면 식이 된다. 1은 Lemma 9(iii)이다. □

**뜻**
- **수렴 조건:** $K_k/\log k\to\infty$이면 마지막 분할 이후 $\mathrm{NashConv}\to0$이다.
  - $K_k\propto L_k$이면 정리 1과 같은 차수 $O(\sqrt{\log(kN_A/\delta)/L_k})$다.
  - $K_k$가 이보다 작으면 $H$ 항의 속도가 $\sqrt{\log k/K_k}$로 느려진다.
- **정리 1보다 나은 점:**
  - 학습 중 멤버별 구간 합(정보집합 × 행동 규모)을 저장하지 않는다. 대신 bucket 표 $K_k$개를 저장한다. $K_k\lvert\mathcal P\rvert\ll N$이면 §7.1-3(학습 메모리)의 한계가 풀린다. 다만 그러면 위 속도 대가가 따른다.
  - $H$ 항에 게임 샘플링 노이즈가 없다(정확 재생). 시간 표집 노이즈만 남는다.
  - 방문하지 않은 정보집합도 실제 regret으로 라벨을 받는다. "미방문 = 없음" 규칙이 필요 없다.
  - 여유 라벨을 쓰므로 좋은 사건 위에서 가짜 분할이 없다.
- **같은 점:** 마지막 분할 시점 $t^\*$의 상계는 여전히 없다. 학습기가 CFR 계열이면 정리 1과 같다.
- **큰 게임(7-stud):** 정확 순회 대신 저장한 전략마다 ES 순회 $m$번으로 재생하면, 같은 Hoeffding 논증에서 $K_k$ 자리가 $K_km$이 된다. 원자 단위로 하면 정리 3과 같이 $H_{\mathcal A}$가 남는다.

### 9.4 방법 B: 분할 시점 평균 전략 [인증과 종료는 증명, 학습기 조건은 미해결]

**알고리즘 (이론 버전, 라운드 방식)**
1. 현재 분할 $\mathcal P_j$를 게이트를 통과할 때까지 학습해 평균 전략 $\bar\sigma_j$를 얻는다.
2. 모든 정보집합의 $r^{\bar\sigma_j}(I)$를 계산한다. 작은 게임은 정확히, 큰 게임은 별도 평가 순회로 추정한다.
3. 모든 bucket이 일치하면 $\bar\sigma_j$를 출력하고 멈춘다. 아니면 일치하지 않는 bucket을 대표 라벨별로 나누고 1로 돌아간다.

**Theorem B1 (단일 프로필 인증) [증명, 수치 확인].** 임의의 lifting된 프로필 σ(분할 시점의 평균 전략 포함)에 대해 다음이 성립한다.
$$\mathrm{NashConv}(\sigma)\le\rho_{\mathcal P}(\sigma)+\sum_BH_B^\sigma.$$
- 모든 bucket이 σ-일치하면 $\mathrm{NashConv}(\sigma)\le\rho_{\mathcal P}(\sigma)$이다.
- 추정 라벨(Lemma 9)로 판정해 나뉘지 않았으면 $\mathrm{NashConv}(\sigma)\le\rho_{\mathcal P}(\sigma)+4\sum_I\eta_I$이다.

*증명.* Lemma 7(단일 프로필), Lemma 8, Lemma 9(iv)를 차례로 쓴다. □

- 정리 2′와 달리 **실제로 내놓을 전략 σ 자체**에 대한 인증이다. 구간 평균이 아니고, 언제든 쓸 수 있다.
- $\rho_{\mathcal P}(\sigma)$는 bucket 단위로 모으는 양이다(bucket별 순간 regret의 양수 부분 합).

**Proposition B2 (bucket-행위자 균형의 존재) [증명].** (A1) 아래 모든 분할 $\mathcal P$에 대해 $\rho_{\mathcal P}(\sigma^\*)=0$인 lifting된 프로필 $\sigma^\*$가 존재한다.

*증명.*
- bucket 하나를 행위자 하나로 보고, 행위자 $B$의 보수를 $U_B(\sigma)=\sum_{I\in B}\pi_{-i}(I)\sum_b\sigma(B,b)\,q^\sigma(I,b)$로 둔다.
- (A1)이면 $\sigma(B)$는 멤버들의 하위 트리에도, $\pi_{-i}$에도 나타나지 않는다. 그래서 $U_B=\sum_b\sigma(B,b)\,c_B(b)$이고, $c_B(b)=\sum_{I\in B}\pi_{-i}(I)\,q(I,b)$는 $\sigma(B)$와 무관하다.
- 각 단말까지의 경로에서 한 행위자는 많아야 한 번 나타난다(두 player의 bucket 모두 (A1)). 그래서 $U_B$는 모든 행위자의 혼합전략에 대해 다중선형이고, 이것은 유한 게임이다. Nash 정리로 균형 $\sigma^\*$가 존재한다.
- 균형에서 $\sigma^\*(B)$는 $c_B$를 최대화한다. 그런데 $r^{\sigma}(B,a)=c_B(a)-\sum_b\sigma(B,b)\,c_B(b)$이므로 $\psi(r^{\sigma^\*}(B))=0$이다. □

- imperfect recall 추상 게임에는 behavioral 전략의 Nash 균형이 없을 수도 있다. 하지만 이 "bucket-행위자 균형"은 항상 있다.

**Theorem B3 (오라클 조건 아래 유한 종료와 정확 수렴) [증명].** 매 라운드 학습기가 $\rho_{\mathcal P_j}(\sigma_j)\le\varepsilon$인 lifting된 프로필 $\sigma_j$를 내놓는다고 하자(ε-오라클). 라벨은 정확히 계산한다.
1. **(i)** 분할이 일어나는 라운드는 많아야 $N-\lvert\mathcal P_0\rvert$번이다. 즉 알고리즘은 유한 라운드 안에 멈춘다.
2. **(ii)** 멈춘 라운드의 출력은 $\mathrm{NashConv}\le\varepsilon$을 만족한다. ε=0 오라클이면 **원게임의 정확한 Nash 균형**이다.
3. **(iii)** (A1) 아래 ε=0 오라클이 내놓을 프로필은 늘 존재한다(Prop B2). 그러므로 조건 자체는 공허하지 않다.
- 라벨을 추정하면(Lemma 9, 매 라운드 좋은 사건) (ii)는 $\varepsilon+4\sum_I\eta_I$가 된다.

*증명.*
- (i): 일치하지 않는 bucket을 대표 라벨로 나누면 자식이 둘 이상이다. 모든 대표 라벨이 같은 $c$라면 $c\in\bigcap M$이 되어 일치에 모순이기 때문이다. 그래서 Lemma 6이 적용된다.
- (ii): 멈춘다는 것은 모든 bucket이 일치한다는 뜻이므로 정리 B1이다. □

- 정리 1·A보다 강한 꼴이다. 마지막 분할까지의 **라운드 수**가 유한하게 묶이고, ε=0이면 정확한 균형이다.
- 다만 비용은 오라클 안에 숨어 있다. 한 라운드에 걸리는 학습 반복 수는 오라클이 ρ를 ε 아래로 만드는 데 걸리는 시간이다.

**Proposition B4 (오라클 조건이 왜 필요한가: 두 반례) [증명, 수치 확인].**

**(a) "추상 게임이 수렴했다"는 게이트로는 부족하다.**
- **게임:** player는 하나이고 상대는 수가 없다.
  - chance가 ½씩 L 또는 R을 고른다. L 뒤 정보집합이 $I_1$, R 뒤가 $I_2$이고, 둘은 bucket $B$를 공유한다(행동 a, b).
  - $I_1$: a → 1, b → $K_2$(두 행동 모두 −10).
  - $I_2$: a → 0, b → $K_4$(c → 2, d → −10).
  - $K_2$, $K_4$는 각자 bucket이다.
- **$\bar\sigma$:** $B$는 a, 경로 밖의 $K_4$는 d.
  - 추상 게임의 최적값은 $B$=a일 때 0.5다($B$=b면 −4). 그래서 $\bar\sigma$는 추상 게임의 **정확한 최적**이고, 추상 exploitability는 0이다.
  - 원게임의 최선은 $I_1$에서 a, $I_2$에서 b 다음 c로 1.5다. 그래서 $\mathrm{NashConv}(\bar\sigma)=1$이다.
  - $\bar\sigma$ 아래 순간 regret은 $I_1=(0,-5.5)$, $I_2=(0,-5)$이다. 둘 다 {a, 없음}을 최대로 하므로 $B$는 **일치**하고 나뉘지 않는다.
  - 정리 B1은 맞게 성립한다. $\rho_{\mathcal P}(\bar\sigma)=6$이기 때문이다. 경로 밖 $K_4$가 나빠서 c 쪽 regret이 6이다.
- **오라클 프로필**($K_4$=c)에서는 $I_2$의 regret이 $(0,1)$이 되어 라벨이 {b}로 바뀐다. 그러면 $B$가 일치하지 않아 나뉘고, 다음 라운드에 정확한 균형이 된다.
- **결론:** 방법 B의 게이트는 "추상 exploitability ≤ ε"이 아니라 "$\rho_{\mathcal P}(\bar\sigma)\le\varepsilon$"이어야 한다.

**(b) CFR/CFR+의 평균 전략은 오라클이 아니다.**
- **게임:** player 하나. $J$에서 L → 1, R → $K$. $K$에서 c → 0, d → 0.5. 완전 기억이다(추상화 없음).
- RM이든 RM+든 $t\ge2$부터 $J$는 L만 둔다. 그래서 $K$의 평균 전략은 자기 도달확률이 양수였던 첫 반복의 균등 전략에 고정된다.
- 그래서 모든 $T$에서 $\rho(\bar\sigma_T)\ge r^{\bar\sigma_T}(K,d)=0.25$다. 반면 $\mathrm{NashConv}(\bar\sigma_T)\to0$이다(균등 평균이면 $0.375/T$).
- **결론:** CFR 평균은 자기가 가지 않는 곳의 전략을 고치지 않는다. 그래서 $\rho_{\mathcal P}(\bar\sigma)\le\varepsilon$ 게이트는 ε < 0.25에서 영원히 통과하지 못한다. 전략은 이미 최적인데도 그렇다.

**뜻: 방법 B의 "더 좋은 조건"은 학습기 쪽 조건으로 옮겨진다.**
- 필요한 학습기는 경로 밖까지 포함해 모든 bucket의 순간 regret을 0으로 만드는 프로필을 내놓아야 한다. bucket-행위자 균형이 그것이다.
- 표준 CFR 평균은 그렇지 않다(B4(b)).
- CFR이 보장하는 것은 **반복들의 시간 평균 regret**이 모든 bucket에서 작다는 것이다. 그 양을 직접 쓰는 것이 방법 A(그리고 기존 구간 규칙)다.
- 오라클 후보는 다음과 같다. 모두 [미해결]이다.
  - 마지막 반복이 수렴하는 학습기의 현재 전략. RM+는 일반적으로 보장이 없다.
  - 경로 밖 정보집합에도 작은 확률을 강제하는 떨림(tremble) 섭동 CFR. 균형 정제 문헌의 방법이며 원문 확인이 필요하다.
  - 작은 게임에서는 선형계획으로 직접 계산.
- (A1)의 bucket-행위자 게임은 행위자마다 보수가 다른 일반합 게임이다. 그래서 그 균형 계산은 일반적으로 어려울 수 있다.

### 9.5 노이즈와 잘못된 배치 [증명]

사용자 논증을 방법 A·B 공통으로 정식화한다. 논증의 요지는 "노이즈로 다른 bucket에 들어가도, 그 bucket의 분할에 영향을 줬다면 그럴 만한 이유가 있으므로 문제없다"이다.

**Proposition N [증명].** 판정 $d$마다 좋은 사건 $\mathcal E^{(d)}$(Lemma 9의 $\lVert\hat x-x\rVert_\infty\le\eta$)가 확률 $1-\delta_d$ 이상이고, 여유 라벨 집합으로 판정한다고 하자.
1. **좋은 사건 위:**
   - 일치하는 bucket은 나뉘지 않는다.
   - 나뉠 때 잘못 배치되는 것은 참 최선과 $4\eta$ 이내로 차이 나는 "거의 동률" 라벨 사이뿐이다. 손실은 멤버마다 $4\eta_I$ 이하다(Lemma 9).
2. **나쁜 사건 위(확률 $\le\delta_d$):**
   - 임의로 잘못 나뉘거나 배치될 수 있다. 그래도 분할 총수는 $N-\lvert\mathcal P_0\rvert$ 이하다(Lemma 6).
   - 잘못 들어간 멤버 $I$는 자식 $B'$의 라벨 $\ell'$에 대해 $e_I(\ell')$이 클 수 있다. 그 크기가 곧 $H_{B'}$의 증가분이다.
   - 다음 판정 시점에도 $I$의 참 방향이 $\ell'$과 달라 그 시점의 $H_{B'}$가 $4\sum_{I\in B'}\eta_I$보다 크다고 하자. 그 판정이 좋은 사건 위에 있으면 $B'$는 일치하지 않으므로 다시 나뉜다(Lemma 9(iv)의 대우). 그래서 $I$는 떨어져 나간다.
3. **결국 맞아진다:**
   - $\sum_d\delta_d<\infty$이면 Borel–Cantelli 보조정리로 나쁜 판정은 거의 확실히 유한 번이다.
   - 분할도 유한하므로, 어느 판정 이후로는 모든 판정이 좋은 사건 위에 있다.
   - 그 뒤로는 분할 없는 판정마다 정리 A(또는 B1)의 인증이 성립한다.

**사용자 논증의 판정**
- **결론은 맞다.** 노이즈가 수렴을 깨지 않는다.
- **이유는 두 가지로 고쳐야 한다.**
  1. 좋은 사건 위에서 잘못 배치가 무해한 이유는 "그 bucket에 들어갈 만해서"가 아니다. 거의 동률(여유 4η 이내)인 라벨 사이에서만 일어나기 때문이다.
  2. 나쁜 사건에서 잘못 들어간 멤버는 그 bucket을 향상시키지 않는다. 상쇄 $H$를 늘려 오염시키고, 바로 그 오염 때문에 다음 판정에서 다시 분리된다.
- 수렴을 지키는 것은 "반복 판정 + 분할만 함"이다. 대가는 추가 분할과 자식 재시작 학습량이다.
- **필요조건 하나:** 라벨은 판정마다 한 번 정해 다음 판정까지 고정해야 한다. 방문할 때마다 다시 표집하면 같은 정보집합이 매번 다른 bucket 전략을 쓰게 된다. 그것은 분할이 아니므로 위 정리 밖이다.

### 9.6 요약: 더 좋은 조건으로 수렴하는가

| 항목 | 기존 (구간 누적 라벨) | 방법 A (저장한 순간 전략) | 방법 B (분할 시점 평균 전략) |
|---|---|---|---|
| 표준 학습기(CFR+, ES)로 증명 | 정리 1 | 정리 A | 인증(정리 B1)만. 수렴은 오라클 조건 필요 |
| 마지막 분할 이후 속도 | $O(\sqrt{\log k/L_k})$ | $K_k\propto L_k$면 같음. 아니면 $H$ 항이 $\sqrt{\log k/K_k}$ | 오라클 조건 아래 라운드 수 $\le N-\lvert\mathcal P_0\rvert$ |
| 마지막 분할 시점 | 미해결 | 미해결 | 라운드 수로는 유한. 반복 수는 오라클 비용에 달림 |
| 극한 | $\mathrm{NashConv}\to0$ | 같음 | ε-오라클이면 ε, 0-오라클이면 정확한 균형 |
| 미방문 정보집합 | "없음"으로 처리 | 재생으로 실제 라벨 | 재생으로 실제 라벨 |
| 학습 중 추적 메모리 | 정보집합 × 행동 | bucket 표 $K_k$개 | 평균 전략 하나 |
| 가짜 분할 | 무한히 생길 수 있음 | 좋은 사건 위에서 없음(여유 라벨) | 같음 |

- **방법 A:** 수렴 조건은 정리 1과 같다. 좋아지는 것은 메모리, 미방문 정보집합 처리, 가짜 분할이다.
- **방법 B:** 증명할 수 있는 가장 좋은 꼴(유한 라운드, 정확한 균형)은 정리 1보다 강하다. 하지만 그 조건을 CFR 평균이 만족하지 않는다는 것을 반례로 보였다. 조건을 만족하는 실용적인 학습기를 찾는 것이 남은 문제다.

## 참고문헌

- Zinkevich, Johanson, Bowling, Piccione. Regret Minimization in Games with Incomplete Information. NeurIPS 2007.
- Lanctot, Waugh, Zinkevich, Bowling. Monte Carlo Sampling for Regret Minimization in Extensive Games. NeurIPS 2009.
- Tammelin. Solving Large Imperfect Information Games Using CFR+. 2014.
- Burch, Moravčík, Schmid. Revisiting CFR+ and Alternating Updates. JAIR 2019.
- Čermák, Lisý, Bošanský. Automated Construction of Bounded-Loss Imperfect-Recall Abstractions in Extensive-Form Games. arXiv 1803.05392 (CFR+IRA).
- Freedman. On Tail Probabilities for Martingales. Annals of Probability 1975.
- Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. JASA 1963.
- Nash. Non-Cooperative Games. Annals of Mathematics 1951.
- Howard, Ramdas, McAuliffe, Sekhon. Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences. Annals of Statistics 2021.
- Waugh, Schnizlein, Bowling, Szafron. Abstraction Pathologies in Extensive Games. AAMAS 2009.
