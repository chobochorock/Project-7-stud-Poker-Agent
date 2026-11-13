# 단일 트리 ASWISS의 증명 초안 (2026-11-13)

**문서 정보**
- 대상: `PP_STUD7_DESIGN_KO.md` 16절의 단일 트리 구조(`--pp-structure tree`). 전략표가 트리의 잎에 있고, 잎의 표본 라벨이 갈리면 특징 검사 하나로 그 잎을 둘로 나눈다.
- 바탕: `ASWISS_CONVERGENCE_KO.md`(아래 "수렴 문서")의 Lemma 3, 4, 7, 8, 9와 정리 1, A, B1, B3. 여기서 새로 증명하는 것은 보조정리 T(표본 판정)와, 그것을 넣은 정리 T3–T5다.
- 표시:
  - [증명]: 이 문서나 수렴 문서에서 증명했다.
  - [가정]: 정리의 전제다.
  - [미해결]: 증명하지 못했다.
- 수치 확인은 하지 않았다. 수렴 문서의 보조정리들은 그 문서에서 수치로 확인한 것이다.

## 0. 요약

| 결과 | 내용 | 상태 |
|---|---|---|
| 정리 T1 (분할 유한성) | 분할은 모든 표본 경로에서 많아야 $N-\lvert\mathcal T_0\rvert$번이다 | [증명] |
| 정리 T2 (인증) | 배포 정책 σ에 대해 $\mathrm{NashConv}(\sigma)\le\rho_{\mathcal T}(\sigma)+\sum_LH_L^\sigma$ | [증명] (정리 B1) |
| 보조정리 T (표본 판정) | 잎의 표본 $n$개가 한 라벨에 일치하면, 높은 확률로 $H_L\le M_L\big(4\eta+2\Delta_L\varepsilon\big)$ | [증명] 새 결과 |
| 정리 T3 (표본 인증) | 분할이 없었던 판정에서 $\mathrm{NashConv}(\sigma)\le\rho_{\mathcal T}(\sigma)+V(\sigma)\,(4\eta+2\Delta\varepsilon)$ | [증명] |
| 정리 T4 (평균 전략 방향, 조건부 수렴) | 학습기가 $\rho_{\mathcal T}(\sigma_j)\le\varepsilon_j$를 내면 $\mathrm{NashConv}(\sigma_j)\le\varepsilon_j+o(1)$ (거의 확실하게) | [증명] 학습기 조건은 [가정] |
| 정리 T5 (누적 후회 방향, 무조건 수렴) | RM+ 학습기와 epoch 평균에서 $\mathrm{NashConv}(\bar\sigma^{(k)})\to0$ (거의 확실하게) | [증명] 이상화한 알고리즘에 대해 |
| 표준 학습기 + 평균 전략 방향의 무조건 수렴 | | [미해결] (수렴 문서 명제 B4) |

- pp 구조에서 따로 필요했던 것(진행 조건, 잘못된 할당의 수정, 계보)은 이 구조에 없다. T1이 그 자리를 대신한다.
- 지금 구현은 T2–T4의 틀(평균 전략 방향)에 해당한다. 구현이 정리의 전제를 어기는 곳은 6절에 있다.

## 1. 설정

**게임**
- $G$는 2인 영합, perfect recall, 유한 게임이다. $\mathcal I$는 정보집합 전체이고 $N=\lvert\mathcal I\rvert$다.
- 선언된 대칭(7-stud의 무늬 교환)으로 같은 정보집합은 하나로 센다.

**관측과 특징**
- (A0) [가정] 관측 사상 $I\mapsto(c(I),x(I))$는 단사다. $c$는 시작 칸, $x$는 관측 열이다.
  - 7-stud에서는 시험 T1이 표본 40만 개에서 확인했다(`PP_STUD7_DESIGN_KO.md` 15.4).
- (A0′) [가정] 특징 족 $\mathcal F$는 서로 다른 두 관측 열을 구분한다. 즉 $x\ne x'$이면 $f(x)\ne f(x')$인 $f\in\mathcal F$가 있다.
  - 완전 부호화 F1이 이를 만족한다.

**트리 분할**
- 시작 칸마다 이진 트리가 하나 있다. 내부 노드는 검사 $(f,\theta)$이고, 잎의 집합 $\mathcal T$가 $\mathcal I$의 분할이다.
- lifting: $\sigma(I)=\sigma(L(I))$. 잎의 전략을 그 멤버가 그대로 쓴다.

**후회**
- 방문당 후회: $d^\sigma_I(a)=q^\sigma(I,a)-\sum_b\sigma(I,b)\,q^\sigma(I,b)$, $\lvert d^\sigma_I(a)\rvert\le\Delta_I$.
- 반사실 후회: $r^\sigma(I,a)=\pi^\sigma_{-i}(I)\,d^\sigma_I(a)$.
- $\psi(x)=\max(0,\max_ax_a)$, 라벨 $\ell\in A\cup\{\varnothing\}$, $x(\varnothing)=0$.
- 방문당 손실: $e_I(\ell)=\psi(d_I)-d_I(\ell)\in[0,2\Delta_I]$.
- $\rho_{\mathcal T}(\sigma)=\sum_L\psi(r^\sigma(L))$, $r^\sigma(L)=\sum_{I\in L}r^\sigma(I)$.
- 잎 안 상쇄: $H_L^\sigma=\sum_{I\in L}\psi(r^\sigma(I))-\psi(r^\sigma(L))=\min_\ell\sum_{I\in L}\pi_{-i}(I)\,e_I(\ell)$ (수렴 문서 Lemma 8).
- 방문 질량: $M_L=\sum_{I\in L}\pi^\sigma_{-i}(I)$, $V(\sigma)=\sum_LM_L=\sum_I\pi^\sigma_{-i}(I)\le N$.
- 방문 분포: $\mu_L(I)=\pi^\sigma_{-i}(I)/M_L$ ($M_L>0$일 때).

## 2. 알고리즘 (이론 버전)

판정 $j=1,2,\dots$마다 다음을 한다.
1. 학습기가 트리 $\mathcal T_j$ 위의 lifting된 프로필 $\sigma_j$를 낸다.
2. $M_L>0$인 잎마다 $\mu_L$에서 표본 $I_1,\dots,I_{n_j}$를 독립으로 뽑는다. (S) [가정]
3. 표본마다 방향 추정 $\hat d_s$를 얻는다. 좋은 사건 $\mathcal G_j$(확률 $\ge1-\delta_j$) 위에서 모든 표본이 $\lVert\hat d_s-d_{I_s}\rVert_\infty\le\eta_j$다. (E) [가정]
4. 여유 라벨 집합 $\hat M(I_s)=\{\ell:\hat d_s(\ell)\ge\max_{\ell'}\hat d_s(\ell')-2\eta_j\}$를 만든다(수렴 문서 Lemma 9).
5. $\bigcap_s\hat M(I_s)=\emptyset$이고 표본에 서로 다른 정보집합이 둘 이상 있으면, 그 잎을 표본을 두 무리로 가르는 검사 $(f,\theta)$로 나눈다.
   - 어떤 검사를 고르는지는 정리에 영향이 없다(구현: 라벨의 지니 감소, 없으면 가장 고른 검사).
   - 새 잎의 표는 시작 노름이 부모 이하다(재시작 또는 약한 상속).
6. 나뉜 잎이 하나도 없으면 "분할 없는 판정"이다.

- 판정 대상 프로필에 따라 두 판이 있다.
  - **평균 전략 방향** (구현): $\sigma_j$는 판정 시점의 배포 정책(잎의 평균 표)이다. 정리 T2–T4.
  - **누적 후회 방향**: epoch $W_k$의 누적 후회 $R_I^{W}$를 겨냥한다. 정리 T5.

## 3. 분할 유한성과 인증

**정리 T1 (분할 유한성) [증명].** (A0), (A0′) 아래, 분할은 모든 표본 경로에서 많아야 $N-\lvert\mathcal T_0\rvert$번이다.

*증명.*
- 분할은 표본에 서로 다른 정보집합이 둘 이상 있을 때만 일어난다. (A0)과 (A0′)에 의해 그 둘을 가르는 검사가 있다.
- 고른 검사는 표본을 비어 있지 않은 두 무리로 가른다. 그래서 두 자식 잎은 각각 정보집합을 하나 이상 가진다.
- 분할마다 잎 수가 1 늘고, 잎 수는 $N$을 넘지 못한다. □

- 잡음으로 잘못 나눈 분할도 이 수에 포함된다. 잡음은 분할의 횟수를 늘릴 수 있을 뿐, 무한히 만들지는 못한다.
- pp 구조에는 이 성질이 없었다. 그쪽은 유효 분할이 그대로인 채 표만 다시 시작하는 사건이 가능했다.

**정리 T2 (인증) [증명; 수렴 문서 정리 B1].** 임의의 트리 $\mathcal T$와 lifting된 프로필 σ에 대해
$$\mathrm{NashConv}(\sigma)\le\rho_{\mathcal T}(\sigma)+\sum_{L\in\mathcal T}H_L^\sigma.$$

- 정리 B1은 분할이 어떻게 만들어졌는지를 쓰지 않는다. 그래서 그대로 적용된다.
- σ는 실제로 배포하는 정책(잎의 평균 표)이어도 된다.

## 4. 표본 판정

**보조정리 T (표본 판정) [증명].** σ와 잎 $L$($M_L>0$)을 고정한다. (S), (E) 아래 임의의 $\varepsilon\in(0,1)$에 대해
$$\Pr\Big[\textstyle\bigcap_s\hat M(I_s)\ne\emptyset\ \text{이고}\ H_L^\sigma>M_L\big(4\eta+2\Delta_L\varepsilon\big)\Big]\le\delta+(\lvert A_L\rvert+1)\,e^{-\varepsilon n}.$$
여기서 $\Delta_L=\max_{I\in L}\Delta_I$다.

*증명.*
1. 라벨 $\ell$마다 $G_\ell=\{I\in L:e_I(\ell)\le4\eta\}$로 둔다. 이것은 참값으로 정한 집합이라 표본과 무관하다.
2. 좋은 사건 위에서 $\ell\in\hat M(I_s)$이면 $e_{I_s}(\ell)\le4\eta$다(Lemma 9(ii)). 그러므로 $\ell\in\bigcap_s\hat M(I_s)$이면 모든 표본이 $G_\ell$에 있다.
3. $\mu_L(G_\ell^c)>\varepsilon$인 고정된 $\ell$에 대해, 독립 표본 $n$개가 모두 $G_\ell$에 있을 확률은 $(1-\varepsilon)^n\le e^{-\varepsilon n}$ 이하다. 라벨은 $\lvert A_L\rvert+1$개이므로 합집합 한계를 쓴다.
4. 그 밖에서는, 교집합의 라벨 $\hat\ell$이 $\mu_L(G_{\hat\ell}^c)\le\varepsilon$을 만족한다. 그러면
$$H_L^\sigma\le\sum_{I\in L}\pi_{-i}(I)\,e_I(\hat\ell)=M_L\,\mathbb E_{\mu_L}\big[e_I(\hat\ell)\big]\le M_L\big(4\eta+2\Delta_L\varepsilon\big).$$
   첫 부등식은 $H_L$이 라벨에 대한 최솟값이기 때문이다. 마지막은 $G_{\hat\ell}$ 안에서 $e\le4\eta$, 밖에서 $e\le2\Delta_L$이기 때문이다. □

- 뜻: 표본이 일치했다면, 일치하지 않는 멤버의 방문 질량은 $\ln(\cdot)/n$ 수준이다.
- 표본을 방문 분포로 뽑는 것이 핵심이다. 상쇄량 $H_L$에서 각 멤버의 무게가 바로 방문 질량 $\pi_{-i}(I)$이기 때문이다.

**정리 T3 (표본 인증) [증명].** 판정 $j$에서 "나뉜 잎이 없는데 아래 부등식이 깨질" 확률은 $\delta_j+\delta'_j$ 이하다.
$$\mathrm{NashConv}(\sigma_j)\le\rho_{\mathcal T_j}(\sigma_j)+V(\sigma_j)\Big(4\eta_j+2\Delta\,\frac{\ln\big(\lvert\mathcal T_j\rvert(\lvert A\rvert+1)/\delta'_j\big)}{n_j}\Big).$$
여기서 $\Delta=\max_I\Delta_I$, $\lvert A\rvert=\max_I\lvert A(I)\rvert$다.

*증명.*
- 좋은 사건 위에서는 같은 정보집합의 표본끼리 여유 라벨 집합이 겹친다(Lemma 9(i)). 그래서 나뉘지 않은 잎은 $\bigcap_s\hat M(I_s)\ne\emptyset$이다.
- 보조정리 T에서 $\varepsilon=\ln(\lvert\mathcal T_j\rvert(\lvert A\rvert+1)/\delta'_j)/n_j$로 두고 잎에 대해 합집합 한계를 쓴다. 추정의 좋은 사건은 판정 전체에 하나($\delta_j$)다.
- $M_L=0$인 잎은 $H_L=0$이다. 정리 T2에 넣는다. □

- $\rho_{\mathcal T}(\sigma)$는 잎 단위로 모으는 양이라 판정 순회에서 추정할 수 있다.
- 이 인증은 그 판정의 정책 하나에 대한 것이다. 과거의 분할 이력이나 학습 방법을 묻지 않는다.

## 5. 수렴

**정리 T4 (평균 전략 방향, 조건부 수렴) [증명].** 다음을 가정한다.
- (O) [가정] 학습기 조건: 판정마다 $\rho_{\mathcal T_j}(\sigma_j)\le\varepsilon_j$.
- (N) [가정] $\eta_j\to0$, $\ln(N/\delta'_j)/n_j\to0$, $\sum_j(\delta_j+\delta'_j)<\infty$.
- (F) [가정] 판정마다 $M_L>0$인 모든 잎을 판정한다.

그러면 거의 확실하게, 어떤 판정 이후로 모든 $j$에 대해
$$\mathrm{NashConv}(\sigma_j)\le\varepsilon_j+N\Big(4\eta_j+2\Delta\,\frac{\ln\big(N(\lvert A\rvert+1)/\delta'_j\big)}{n_j}\Big).$$
특히 $\varepsilon_j\to0$이면 $\mathrm{NashConv}(\sigma_j)\to0$이다.

*증명.*
- 정리 T1에 의해 분할은 유한 번이다. 그래서 마지막 분할이 일어난 판정 $J^\*<\infty$가 있고, 그 뒤의 판정은 모두 분할 없는 판정이다.
- $\sum_j(\delta_j+\delta'_j)<\infty$이므로 Borel–Cantelli 보조정리에 의해 정리 T3의 나쁜 사건은 거의 확실히 유한 번이다.
- 두 시점 이후의 모든 판정에 정리 T3을 쓰고, $V\le N$, $\lvert\mathcal T_j\rvert\le N$을 넣는다. □

- (O)가 남은 문제다. 표준 CFR의 평균 전략은 (O)를 만족하지 않는다(수렴 문서 명제 B4(b)). 경로 밖 잎의 평균 표가 고쳐지지 않기 때문이다.
- (O)를 만족하는 프로필 자체는 경로 분리 조건 아래 늘 존재한다(수렴 문서 명제 B2).

**정리 T5 (누적 후회 방향, 무조건 수렴) [증명].** 판정 대상을 epoch $W_k$(길이 $L_k$, 두 배씩)의 누적 후회로 바꾼 알고리즘을 생각한다.
- 학습기: 잎마다 RM+, 동시 갱신, external sampling(수렴 문서 (S1)–(S3)).
- 표본: 잎마다 그 epoch의 방문 질량 $m_I=\sum_{t\in W_k}\pi^{\sigma_t}_{-i}(I)$에 비례해 $n_k$개를 뽑는다. 학습 순회의 방문에서 저수지 추출을 하면 이 분포다.
- 방향: 표본의 정규화한 누적 후회 $D_I=R_I^{W_k}/m_I$를 오차 $\eta_k$ 이내로 추정한다(예: 저장한 순간 전략의 재생, 수렴 문서 Lemma 10).
- (N), (F)는 정리 T4와 같다.

그러면 분할이 없었던 epoch마다, 확률 $1-\delta-\delta_k-\delta'_k$ 이상으로
$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\sum_{L}D_L\sqrt{\lvert A_L\rvert\,t_{k+1}}+\sum_I2\varepsilon_I(t_{k+1})\Big]+\bar V_k\Big(4\eta_k+2\Delta\,\frac{\ln\big(N(\lvert A\rvert+1)/\delta'_k\big)}{n_k}\Big),$$
이고 거의 확실하게 $\mathrm{NashConv}(\bar\sigma^{(k)})\to0$이다. 여기서 $\bar\sigma^{(k)}$는 epoch의 sequence-form 평균, $\bar V_k=\frac1{L_k}\sum_Im_I\le N$, $D_L$과 $\varepsilon_I$는 수렴 문서 Lemma 3, 4의 것이다.

*증명.*
- 수렴 문서 Lemma 1로 $\mathrm{NashConv}(\bar\sigma^{(k)})\le(C^{W}+H^{W})/L_k$다. 이 분해는 lifting된 프로필 열이면 성립하고, 분할의 모양을 쓰지 않는다.
- $C$ 항은 정리 1의 증명 그대로다(Lemma 3, 4). 새 잎의 시작 노름이 부모 이하이므로 Lemma 3이 성립한다(Remark 3.4).
- $H$ 항: $R_I^{W}(a)=m_I\,D_I(a)$이고 $\lvert D_I(a)\rvert\le\Delta_I$다. 그래서 보조정리 T가 $\pi_{-i}(I)$를 $m_I$로, $d_I$를 $D_I$로 바꾼 채 그대로 성립하고, $H_L^{W}\le M_L^{W}(4\eta_k+2\Delta_L\varepsilon)$, $M_L^{W}=\sum_{I\in L}m_I$를 준다. 잎에 대해 더하면 $H^{W}/L_k\le\bar V_k(\cdots)$다.
- 분할의 유한성은 정리 T1이다. 거의 확실한 수렴은 정리 T4의 논증과 같다. □

- 이것은 정리 1을 "잎을 특징 검사로 나눈다"와 "멤버를 표본으로 본다"로 바꾼 것이다.
- 보장 대상은 epoch의 sequence-form 평균이다. 잎의 평균 표(배포 정책)와는 일반적으로 다르다(수렴 문서 §7.1-4).

## 6. 정리와 지금 구현의 차이

| 항목 | 정리의 전제 | 구현 (`--pp-structure tree`) | 영향 |
|---|---|---|---|
| 판정 대상 | T4: 평균 전략 + 학습기 조건 (O). T5: 누적 후회 | 평균 전략, 표준 MCCFR+ | (O)가 없으므로 T2, T3(인증)까지만 적용된다. 수렴은 [미해결] |
| 표본 수, 추정 오차 | $n_j\to\infty$, $\eta_j\to0$ | $n=64$ 고정(예산에 따라 8까지 줄어듦), 입자 64개 고정 | 인증의 표본 항이 0으로 가지 않는다 |
| 라벨 | 여유 라벨 집합 | 최대 행동 + $z=2$ 문턱 | 좋은 사건 위에서도 거짓 분할이 생길 수 있다. T1 덕분에 횟수는 유한하다 |
| 표본 추출 | 방문 분포에서 독립 추출 (S) | 평균 전략 순회의 방문에서 서로 다른 정보집합을 저수지 추출 | 가중치에 비례한 순차 비복원 추출이면 보조정리 T의 한계가 그대로 성립한다(불일치 멤버를 놓칠 확률이 더 작아진다). 구현의 저수지는 그것과 정확히 같지 않고, 한 순회 안의 방문도 독립이 아니다. 엄밀한 처리는 [미해결] |
| 공정성 (F) | 모든 잎을 판정 | 예산이 모자라면 방문이 많은 잎부터 판정하고 나머지는 건너뜀 | 영원히 판정되지 않는 잎이 있을 수 있다. 건너뛴 잎을 다음에 먼저 보면 해결된다 |
| 분할 상한 | 없음 | 한 판정에 잎 수 2배까지 | 분할이 늦어질 뿐이다 |
| 갱신과 평균 | T5: 동시 갱신, epoch 균등 평균 | 교대 갱신, 선형 평균 | 수렴 문서 §7.2와 같은 차이 |
| 상수 | $N$, $\Delta$가 그대로 들어감 | 7-stud의 $N$은 매우 크다 | 상계는 수치로는 공허하다. 수렴 문서의 정리들과 같은 성질 |

## 7. 남은 문제

1. **표준 학습기와 평균 전략 방향의 수렴** [미해결]. (O)를 만족하는 실용적인 학습기, 또는 (O) 없이 성립하는 논증이 필요하다. 후보는 수렴 문서 §9.4에 있다(마지막 반복이 수렴하는 학습기, 혼합 평균, 조건부 평균).
2. **마지막 분할 시점의 상계** [미해결]. 분할 횟수는 유한하지만 언제 끝나는지는 묶지 못한다.
3. **표본 추출의 의존성** [미해결]. 6절의 (S).
4. **수치 확인**: 보조정리 T를 mini-stud에서 정확한 $H_L$과 비교하는 시험을 아직 하지 않았다.
