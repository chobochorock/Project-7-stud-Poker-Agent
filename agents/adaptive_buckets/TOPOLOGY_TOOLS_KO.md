# 고정점·덮개 정리들과 ASWISS 이론의 결합 (검토, 2026-11-28)

사용자 요청: "Sperner's coloring, Brouwer's fixed point theorem, Poincare-Miranda theorem, Lebesgue's number등등과 조합해서 사용할 만한 정리가 있는지 확인해 보라."

**상태 표기**
- **[증명]**: 이 문서 안에서 증명을 끝냈다. 다만 다른 사람의 검토는 아직 없다.
- **[스케치]**: 가정과 논증의 뼈대만 적었다.
- **[문헌]**: 알려진 결과를 인용했다.

기호는 `ASWISS_CONVERGENCE_KO.md`와 논문 초안 5절을 따른다.
- 순간 반사실적 후회: $r^\sigma(I,a)$
- bucket 합: $r^\sigma(B,a)=\sum_{I\in B}r^\sigma(I,a)$
- 숨은 후회: $H_B^\sigma=\sum_{I\in B}\psi(r^\sigma(I))-\psi(r^\sigma(B))$
- 라벨 비용: $e_I(\ell)=\psi(r^\sigma(I))-r^\sigma(I,\ell)$

## 0. 요약

| 정리 | ASWISS에서 쓰는 자리 | 결론 | 상태 |
|---|---|---|---|
| **Brouwer** (Nash의 사상) | 어떤 분할이든 bucket 반사실 후회가 모두 0인 bucket 프로필이 있는가 | **있다.** 불완전 기억과 absent-mindedness가 있어도 된다(명제 1) | [증명] |
| Blackwell (접근 가능성, RM)로 대신하기 | 명제 1과 정리 3에 Brouwer가 꼭 필요한가 | 명제 1은 대신할 수 없다. 정수 보수 게임에서도 명제 1의 점이 무리수뿐일 수 있다. 정리 3은 비례 도달 가정 아래 평균 전략판이 있다(1.1절) | [증명] |
| **Lebesgue 수 보조정리** (또는 Lipschitz) | 특징 공간에서 칸의 지름을 얼마나 작게 하면 모든 칸이 라벨 일치(숨은 후회 ≤ ε)가 되는가 | 프로필 전체에 걸쳐 균일한 칸 크기 $\delta(\varepsilon)$가 있다(명제 2) | [증명] |
| Brouwer + Lebesgue + 라벨 항등식 | 작은 ε-추상화가 존재하는가, 몇 개의 bucket이면 되는가 | $\NC\le2L\delta(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert)$인 bucket 프로필이 덮개 수만큼의 bucket으로 존재한다(정리 3) | [증명] (가정이 강함) |
| Poincaré–Miranda, 정육면체 Sperner 보조정리 | 관측 분할의 문턱값 찾기, 라벨이 만나는 점의 존재 | 여러 특징의 문턱을 동시에 찾는 다차원 이분법, 분할 수의 하한 근거 | [스케치] |
| Sperner, KKM, Scarf | 라벨 = 색칠, 완전히 칠해진 칸 = 근사 고정점 | 추상 게임 균형 계산과 PPAD 난이도 맥락. 직접 쓸 자리는 작다 | [문헌] |
| **Helly** (분수 Helly) | 라벨 일치를 몇 개 멤버로 확인할 수 있는가 | $\lvert A\rvert$개씩의 부분집합이 모두 공통 최대 라벨을 가지면 전체가 가진다. 표본 판정의 근거 | [증명]/[스케치] |
| Kakutani, agent form | absent-mindedness가 있을 때의 해 개념 | 플레이어 단위 행동 전략 균형은 없을 수 있다(Wichardt 2008). agent 단위(bucket 단위)는 있다 | [문헌] |
| Tarski, ham sandwich, 목걸이 분할, Borsuk–Ulam, Shapley–Folkman | 분할 격자, 균형 분할 | 쓸모가 작다(6절) | – |

가장 쓸 만한 것은 명제 1, 명제 2, 그리고 둘을 묶은 정리 3이다. 정리 3은 다음 두 곳과 연결된다.
- 논문 초안 8절의 후속 연구 1(초기 추상화)과 2(분할 개선)
- "최종 bucket 수의 보장이 없다"는 한계에 대한 비교 기준

## 1. Brouwer: bucket 반사실 후회가 0인 프로필의 존재

**명제 1 [증명].**
- 완전 기억 유한 게임 $G$와, 같은 플레이어·같은 행동집합끼리 묶은 임의의 분할 $\mathcal P$를 생각한다.
- 불완전 기억이어도 되고, 한 경로가 같은 bucket을 여러 번 지나도(absent-mindedness) 된다.
- 그러면 bucket마다 상수인 행동 전략 $\sigma^\star$(lifting)가 존재해서 다음이 성립한다.

$$r^{\sigma^\star}(B,a)\le0\qquad\text{모든 bucket }B,\ \text{모든 행동 }a.$$

즉 $\rho_{\mathcal P}(\sigma^\star)=\sum_B\psi(r^{\sigma^\star}(B))=0$이다. bucket 단위 후회 매칭(CFR)이 겨냥하는 "정상 상태"가 언제나 존재한다는 말이다.

*증명.* Nash(1951)가 Brouwer 고정점 정리로 균형을 보인 방법을 bucket 반사실 후회에 그대로 쓴다.
1. **공간.** $X=\prod_B\Delta(A_B)$는 콤팩트 볼록 집합이다.
2. **연속성.** $r^\sigma(B,a)$는 $\sigma$의 다항식이다. 반사실 가치가 도달확률과 행동 확률의 곱의 합이기 때문이다.
3. **직교성.** 모든 $\sigma$에서 다음이 성립한다.
   $$\sum_a\sigma(B,a)\,r^\sigma(B,a)=\sum_{I\in B}\Big(\sum_a\sigma(I,a)v_i(\sigma,I,a)-v_i(\sigma,I)\Big)=0.$$
   - $v_i(\sigma,I)=\sum_a\sigma(I,a)v_i(\sigma,I,a)$는 첫 행동으로 나눈 항등식이다.
   - 뒤에 같은 bucket이 다시 나와도 $v_i(\sigma,I,a)$ 안에 $\sigma(B)$가 다시 곱해질 뿐이라 그대로 성립한다.
4. **사상.** Nash 사상 $T(\sigma)(B,a)=\dfrac{\sigma(B,a)+r^\sigma(B,a)^+}{1+\sum_br^\sigma(B,b)^+}$는 $X$에서 $X$로 가는 연속 사상이다. Brouwer 정리로 고정점 $\sigma^\star$가 있다.
5. **고정점의 성질.** 어떤 $B$에서 $s=\sum_br^{\sigma^\star}(B,b)^+>0$라고 하자.
   - 고정점 식에서 모든 $a$에 대해 $\sigma^\star(B,a)\,s=r^{\sigma^\star}(B,a)^+$다.
   - 그러면 $\sigma^\star(B,a)>0$인 모든 $a$에서 $r^{\sigma^\star}(B,a)>0$이고, $\sum_a\sigma^\star(B,a)r^{\sigma^\star}(B,a)>0$이 된다.
   - 이것은 3에 모순이다. 따라서 $s=0$이다. □

**의미.**
- **존재가 갈리는 지점.** 불완전 기억 추상 게임에서는 플레이어 단위의 행동 전략 내쉬 균형이 존재하지 않을 수 있다(Wichardt 2008).
  - 플레이어의 효용이 자기 행동 전략에 대해 준오목이 아니면 Kakutani 정리를 쓸 수 없다.
  - 명제 1의 대상은 다르다. 자기 도달확률로 가중하지 않은 반사실 합의 정상점이다. 이것은 언제나 존재한다.
  - 즉 bucket CFR이 수렴할 "자리"는 늘 있다. 다만 수렴한다는 보장은 아니다(Lanctot 외 2012).
- **원 게임에서의 평가.** 명제 1의 $\sigma^\star$에서는 단일 프로필 인증(정리 B1, 논문 보조정리 5.3의 단일 프로필판)으로 다음이 성립한다.
  $$\mathrm{NashConv}_G(\sigma^\star)\le\rho_{\mathcal P}(\sigma^\star)+\sum_BH_B^{\sigma^\star}=\sum_BH_B^{\sigma^\star}.$$
  원 게임에서의 오차가 정확히 숨은 후회만으로 묶인다.
- **이전 노트와의 관계.** `ASWISS_CONVERGENCE_KO.md`의 정리 B3는 "경로 분리 가정 아래 ε = 0인 전략이 항상 존재한다"고 적었다. 명제 1은 같은 종류의 존재를 경로 분리 없이 준다. B3 쪽 문장은 이것으로 바꿀 수 있다.
- **같은 진술이 이미 있는가.** 같은 진술이 문헌에 있는지는 확인하지 않았다.

### 1.1 Blackwell 정리로 대신할 수 있는가 (2026-11-28)

사용자 질문: "Brouwer는 사실 Blackwell 정리로도 되는 결과 아닌가?"

**답: 명제 1은 안 된다. 수렴 정리들은 이미 Blackwell 쪽이다. 정리 3은 가정을 하나 더 두면 Blackwell판이 있다.**

**Blackwell이 주는 것.**
- Blackwell 접근 가능성 정리의 증명은 분리 초평면(= minimax, LP 쌍대)과 평균 논증뿐이다. 후회 매칭(RM)은 이 정리를 후회 벡터에 쓴 것이다(Hart & Mas-Colell 2000).
- 결론은 **시간 평균**에 대한 것이다. 반복 $t=1..T$의 후회 평균 $\frac1T\sum_tr^{\sigma^t}(B,a)$가 0 이하로 간다.
- 그래서 Blackwell(또는 LP)로 존재를 보일 수 있는 대상은 평균이나 결합 분포 꼴이다.
  - 상관균형의 존재: LP 쌍대로 증명된다(Hart & Schmeidler 1989).
  - 2인 제로섬 완전 기억 게임의 균형: 수열형에서 보수가 쌍선형이라 평균 전략 쌍이 ε-균형이 된다. 그 극한으로 균형의 존재가 나온다(minimax).
- 우리 정리 1, 정리 A, 대기 조건판의 ε 인증은 모두 이 경로다. 출력은 반복들의 혼합(구간 균등 혼합, 평균 전략)이다. 명제 1에 기대지 않는다.

**명제 1이 요구하는 것.**
- 명제 1은 bucket 표 **하나** $\sigma^\star$에서 $r^{\sigma^\star}(B,a)\le0$을 요구한다.
- 평균으로 이것을 얻으려면 "평균 프로필의 후회 = 후회의 평균"이어야 한다. 즉 $r^\sigma(B,a)$가 $\sigma$에 대해 선형이어야 한다.
- 불완전 기억 분할에서는 선형이 아니다.
  - 같은 플레이어의 여러 bucket 확률이 한 경로에서 곱해진다.
  - absent-mindedness가 있으면 같은 bucket 확률이 한 경로에 여러 번 곱해진다.

**막히는 예: 정수 보수인데 명제 1의 점이 무리수뿐이다 [증명, 수치 확인].**
- 1인 게임(상대는 보수 0인 허수아비로 두면 2인 제로섬)에서 교차로 세 개 $h_1,h_2,h_3$을 한 bucket으로 묶는다. 원 게임은 완전 기억이고, 묶은 뒤 absent-minded가 된다.
  - 각 교차로에서 계속(C, 확률 $p$) 또는 나감(E)을 고른다.
  - 나갈 때의 보수: $h_1$에서 0, $h_2$에서 4, $h_3$에서 0. 셋 다 지나면 1.
- 상대·우연 도달확률은 모든 노드에서 1이다. 계산하면 다음과 같다.
  $$r(B,C)=(1-p)\,S(p),\qquad r(B,E)=-p\,S(p),\qquad S(p)=p^2-3p+1.$$
  - 직교성 $p\,r(B,C)+(1-p)\,r(B,E)=0$도 성립한다.
  - $p=0$이면 $r(B,C)=1>0$, $p=1$이면 $r(B,E)=1>0$이다. 그래서 명제 1의 점은 $S(p)=0$인 내부 점뿐이다.
  - 따라서 유일한 점은 $p^\star=(3-\sqrt5)/2\approx0.382$이다. 무리수다.
- 이 예가 말하는 것.
  - minimax, LP 쌍대, 분리 초평면 논증은 유리수 같은 임의의 순서체에서 그대로 성립한다. 그래서 그것만으로는 이 점의 존재를 낼 수 없다. 실수의 완비성을 쓰는 단계가 있어야 한다. 여기서는 중간값 정리(1차원 Brouwer)다.
  - 이 점은 기대 보상의 최대점도 아니다. $u(p)=4p-4p^2+p^3$의 최대점은 $p=2/3$($u=32/27$)이다. $p^\star$에서는 $u=1$이다. 최적화(최대점의 존재와 KKT)로도 명제 1이 나오지 않는다.
    - 이유: $\partial u/\partial\sigma(B,a)=\sum_{h\in B}\pi_i(h)\pi_{-i}(h)v(h,a)$는 자기 도달확률 $\pi_i(h)$로 가중한다. 반사실 후회는 그 가중이 없다.
    - 완전 기억이면 정보집합 안에서 $\pi_i(h)$가 같아서 둘이 비례한다. 불완전 기억 bucket에서는 다르다.
- 일반합으로 넓히면 명제 1에는 $n$인 내쉬 정리가 특수한 경우로 들어 있다.
  - 정규형 게임을 플레이어마다 정보집합 하나인 전개형으로 쓰고 항등 분할을 쓴다. 그러면 $r^\sigma(I,a)=u_i(a,\sigma_{-i})-u_i(\sigma)$이고, 명제 1이 곧 내쉬 균형이다.
  - 3인 정수 보수 게임에는 유일한 균형이 무리수 확률뿐인 예가 있다(Bubelis 1979; 보수 {0,1,2}의 예는 arXiv:2507.09422).
  - 결합되지 않은 동역학(각자 자기 보수만 보고 움직이는 것. 후회 기반 동역학 포함)은 일반적으로 내쉬 균형으로 가지 않는다(Hart & Mas-Colell 2003).
- 거꾸로 "내쉬 존재 ⇒ Brouwer(Kakutani)"는 McLennan과 Tourky의 모방 게임 논문으로 알려져 있다. 다만 이번에는 그 원문을 확인하지 못했다. 확인한 것은 "Imitation games and computation"(2인 게임을 모방 게임으로 바꾸면 균형이 일대일로 대응한다)뿐이다. [문헌, 미확인]

**정리 3의 Blackwell판 [증명; 비례 도달 가정].**
- 정리 3을 bucket 표 하나 대신 bucket CFR 반복들의 평균 전략 $\bar\sigma$에 대해 세울 수 있는지 본다.
- 경로는 다음과 같다.
  1. 원 게임 $G$는 완전 기억이다. folk theorem과 후회 분해로 $T\cdot\mathrm{NashConv}(\bar\sigma)\le\rho^T+\sum_BH_B^T$다. $\rho^T$는 bucket 누적 후회의 양수부 합이고, $H_B^T$는 누적 후회 벡터의 숨은 후회다.
  2. $\rho^T=O(\sqrt T)$는 bucket RM(Blackwell)으로 나온다.
  3. 남는 것은 $H_B^T$다.
- **문제: 숨은 후회는 시간에 대해 부분가법이 아니다.**
  - 반례: 멤버 $I_1,I_2$, 행동 하나와 "없음". $t=1$에 후회 $(2,0)$, $t=2$에 $(0,-2)$이다.
  - 각 시점의 숨은 후회는 0이다. 누적 $(2,-2)$의 숨은 후회는 2다.
  - 이 무늬는 두 멤버의 조건부 이점이 매 시점 같아도($\hat g_t(I_1)=\hat g_t(I_2)=\pm1$) 도달확률만 $(2,0)$, $(0,2)$로 엇갈리면 생긴다.
  - 되풀이하면 분해식의 상한이 $T$에 비례해 커진다. 이것은 상한의 반례이고, 실제 NashConv의 반례는 아니다.
  - 그래서 명제 2(a)를 프로필마다 쓰는 것만으로는 부족하다.
- **비례 도달 가정.** 같은 칸의 멤버마다 $\pi^{\sigma^t}_{-i}(I)=c_I\,w_t$라 하자(멤버 상수 × 시점 공통 인자).
  - 그러면 $R^T(I,\cdot)=c_I\sum_tw_t\hat g_t(I,\cdot)$이다. 정규화하면 $w$-가중 평균 $\hat G(I)=\sum_tw_t\hat g_t(I)/\sum_tw_t$이고, 이것도 $L$-Lipschitz다.
  - 명제 2(a)의 논증을 그대로 쓰면 $H_B^T\le2L\delta\sum_{I\in B}\sum_t\pi^t_{-i}(I)$다.
  - 정리 3의 4단계로 다음을 얻는다.
    $$\mathrm{NashConv}_G(\bar\sigma)\le\frac{\rho^T}{T}+2L\delta\big(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert\big)\ \xrightarrow{T\to\infty}\ 2L\delta\big(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert\big).$$
  - 이 판은 구성적이다(bucket CFR의 평균 전략). 출력은 반복들의 혼합이고, bucket 표 하나가 아니다.
- **가정이 맞는 경우.** 같은 공개 이력을 공유하는 멤버끼리, 사적 카드가 독립(카드 제거 효과 없음)이면 정확히 맞는다.
  - $\pi_{-i}(I)=\pi_c(\text{내 패})\times\sum_o\pi_c(o)\,\pi^t_{\text{상대}}(\text{공개 행동}\mid o)$로 갈라지기 때문이다.
  - mini-stud와 세븐 포커에서는 카드 제거 때문에 근사로만 맞는다.
  - 공개 이력이 다른 정보집합을 묶은 bucket에서는 상대 도달 인자가 이력마다 달라서 맞지 않는다.

**정리.**

| 결과 | 도구 | 무엇이 존재하거나 수렴하는가 | 구성적인가 |
|---|---|---|---|
| CFR, 정리 1·A, 대기 조건판 | Blackwell(RM) | 반복들의 혼합(평균 전략)의 ε 상한 | 예 |
| 정리 3의 Blackwell판 | Blackwell + 비례 도달 가정 | 평균 전략의 $2L\delta(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert)$ 상한 | 예 |
| 명제 1, 정리 3 | Brouwer | bucket 표 하나(압축된 단일 정책) | 아니오(존재만) |

## 2. Lebesgue 수 보조정리: 프로필 전체에 균일한 칸 크기

**설정.**
- 같은 constraint(같은 플레이어, 같은 행동집합) 안의 정보집합에 특징 $\phi(I)\in X$를 준다. $X$는 콤팩트 거리 공간이다.
- 정규화한 이점을 $\hat g_\sigma(I,a)=r^\sigma(I,a)/\pi^\sigma_{-i}(I)$로 둔다. 이것은 $I$에 도달했을 때의 조건부 이점이다. $\pi_{-i}=0$이면 $r=0$이므로 아무 값이나 둔다.
- "없음" 라벨은 $\hat g(I,\varnothing)=0$이다.

**명제 2 [증명].**

(a) **Lipschitz판.** 모든 프로필 $\sigma$, 같은 constraint의 $I,I'$, 행동 $a$에 대해 $\lvert\hat g_\sigma(I,a)-\hat g_\sigma(I',a)\rvert\le L\,d(\phi(I),\phi(I'))$라 하자.
- $\phi$-지름이 $\delta$ 이하인 칸 $B$에는 어떤 $\sigma$에서도 다음을 만족하는 공통 라벨 $\ell^\star$가 있다.
  $$e_I(\ell^\star)\le2L\delta\,\pi^\sigma_{-i}(I)\qquad\forall I\in B.$$
- 따라서 $H_B^\sigma\le2L\delta\sum_{I\in B}\pi^\sigma_{-i}(I)$다.

(b) **연속판(Lebesgue 수).** 정량적 $L$ 없이 "$(x,\sigma)\mapsto\hat g_\sigma(x)$가 콤팩트 공간 $X\times\Sigma$에서 연속"이라고만 하자.
- 그러면 임의의 $\varepsilon>0$에 대해 $\delta(\varepsilon)>0$이 있다.
- 지름이 $\delta$ 미만인 칸은 모든 $\sigma$에서 공통 라벨로 $e_I\le\varepsilon\,\pi_{-i}(I)$를 만족한다.

*증명.*
- **(a)**
  - 칸의 한 멤버 $I_0$에서 $\ell^\star=\arg\max_{\ell\in A\cup\{\varnothing\}}\hat g_\sigma(I_0,\ell)$로 둔다.
  - $\psi$는 sup 노름에 대해 1-Lipschitz이므로, 다른 멤버 $I$에서 다음이 성립한다.
    $$\psi(\hat g(I))-\hat g(I,\ell^\star)\le(\psi(\hat g(I_0))+L\delta)-(\hat g(I_0,\ell^\star)-L\delta)=2L\delta.$$
  - $\pi_{-i}(I)$를 곱하면 $e_I(\ell^\star)$다.
  - 라벨 항등식 $H_B=\min_\ell\sum_Ie_I(\ell)$로 $H_B$를 묶는다.
- **(b)**
  - $U_\ell=\{(x,\sigma):\hat g_\sigma(x,\ell)>\psi(\hat g_\sigma(x))-\varepsilon\}$는 연속성으로 열린 집합이다. 각 점에서 argmax 라벨이 속하므로 $X\times\Sigma$를 덮는다.
  - Lebesgue 수 보조정리로 $\delta$가 있다. 곱 거리에서 지름이 $\delta$ 미만인 집합은 어떤 $U_\ell$ 하나에 들어간다.
  - $\sigma$를 고정하고 $\{\phi(I):I\in B\}\times\{\sigma\}$에 쓰면 된다. □

**주의.**
- **유한 게임에서는 (b)가 자명하다.** 특징 집합이 유한하면 Lebesgue 수는 점들 사이의 최소 거리라서 칸이 한 점씩이 된다.
- **(b)가 의미를 갖는 경우.** 게임이 커지면서 연속 극한(예: 손패 세기를 연속으로 보는 포커 족)을 가질 때, 그 극한에서의 $\delta$가 게임 크기와 무관한 칸 크기를 준다.
- **(a)의 가정은 강하다.** "모든 프로필에서 같은 $L$"을 요구한다. 포커라면 같은 공개 이력에서 손패 특징이 가까운 정보집합끼리 이점이 가깝다는 뜻이다. 상대 전략이 바뀌어도 그래야 한다. 카드 제거 효과를 생각하면 이 가정을 따로 확인해야 한다.

## 3. 결합: 작은 ε-추상화의 존재

**정리 3 [증명; 가정은 명제 2(a)].**
- $\mathcal P$를 constraint 칸들을 세분한 분할로, 각 칸의 $\phi$-지름이 $\delta$ 이하라 하자.
- 그러면 bucket 상수 프로필 $\sigma^\star$가 존재해서 다음이 성립한다.
  $$\mathrm{NashConv}_G(\sigma^\star)\le2L\delta\,\big(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert\big),$$
  - $\lvert\Sigma_i\rvert$는 플레이어 $i$의 수열(sequence) 개수다(빈 수열 포함).
- bucket 수는 $\sum_c N_c(\delta)$로 잡을 수 있다. $N_c(\delta)$는 constraint 칸 $c$의 특징 집합을 지름 $\delta$ 이하의 조각으로 나누는 데 필요한 최소 개수(덮개 수)다.
- 예를 들어 $X=[0,1]^d$에서 sup 거리를 쓰면 $N_c(\delta)\le\lceil1/\delta\rceil^d$다.

*증명.*
1. 명제 1로 $\rho_{\mathcal P}(\sigma^\star)=0$인 $\sigma^\star$를 잡는다.
2. 단일 프로필 인증으로 $\mathrm{NashConv}(\sigma^\star)\le\sum_BH_B^{\sigma^\star}$다.
3. 명제 2(a)로 $\sum_BH_B^{\sigma^\star}\le2L\delta\sum_i\sum_{I\in\mathcal I_i}\pi^{\sigma^\star}_{-i}(I)$다.
4. 남은 것은 $\sum_{I\in\mathcal I_i}\pi_{-i}(I)\le\lvert\Sigma_i\rvert$다.
   - 정보집합을 그 직전의 자기 수열(부모 수열) $s$로 묶는다.
   - 부모 수열이 같은 두 정보집합은 한 경로 위에 있지 않다. 뒤의 것의 부모 수열에는 앞의 것의 행동이 들어가기 때문이다.
   - 그래서 $s$를 따르는 전략 아래에서 그 이력들은 서로 배반인 사건이고, 합이 1 이하다. $s$에 대해 더하면 된다. □

**읽는 법.**
- **원 게임 기준의 추상화 크기.** "원 게임 NashConv ≤ ε를 주는 bucket 프로필"이 있으려면 bucket이 $N(\varepsilon/(2L(\lvert\Sigma_1\rvert+\lvert\Sigma_2\rvert)))$개면 충분하다.
  - 고정 추상화의 오차 상한(Kroer & Sandholm 2014–2018: 묶인 정보집합의 보상·우연 확률 차이로 묶음)과 형태가 다르다.
  - 그쪽은 정해진 사상 아래 해의 품질을 묶고, 이쪽은 이점이 매끄러우면 작은 추상화에 좋은 bucket 프로필이 존재함을 Brouwer로 보인다.
- **존재만 준다.** 명제 1의 $\sigma^\star$를 bucket CFR이 찾는다는 보장은 없다. 불완전 기억 추상 게임에서 균형 계산은 일반적으로 어렵다(Koller & Megiddo 1992). 그래서 이 정리는 계산이 아니라 존재와 크기의 기준이다.
- **ASWISS와의 연결.**
  - 고정 문턱 판(논문 7.4절)과 관측 분할은 정리 3을 비교 기준으로 쓸 수 있다. "ε에 필요한 bucket 수가 덮개 수 정도인가"를 재 볼 수 있다.
  - 순수 ASWISS의 라벨 분할은 정리 3의 칸보다 거칠 수도, 고울 수도 있다. 라벨이 같은 정보집합은 특징이 멀어도 함께 남기 때문이다.
- **상수 $\lvert\Sigma_i\rvert$가 크다.** 깊은 게임에서는 이 상수가 크다. 단일 프로필 인증에서 최선 응답의 자기 도달확률 가중(논문 보조정리 5.3의 등식)을 쓰면 줄일 여지가 있다. 하지만 숨은 후회가 bucket 합으로만 묶여서 그 가중을 바로 쓸 수 없다. [미해결]

## 4. Poincaré–Miranda, 정육면체 Sperner, Sperner

**Poincaré–Miranda [문헌].**
- 진술: $f:[-1,1]^n\to\mathbb R^n$이 연속이고 모든 $i$에 대해 $f_i\le0$ on $\{x_i=-1\}$, $f_i\ge0$ on $\{x_i=1\}$이면, $f(x)=0$인 $x$가 있다.
- Brouwer 정리와 동치다(Miranda 1940).
- Kulpa(1997)는 정육면체 Sperner 보조정리로 증명했다.

ASWISS에서의 쓰임은 두 가지다.
1. **관측 분할의 문턱 찾기 [스케치].**
   - 칸 안에서 특징 $x\in[0,1]^n$을 따라 행동 $a_j$의 정규화 이점이 한쪽 면에서 음, 맞은편 면에서 양이라 하자(Poincaré–Miranda 경계 조건).
   - 그러면 모든 이점이 동시에 0인 점(무차별점)이 칸 안에 있다.
   - 지금의 후보 자르기는 무작위 (특징, 문턱값)이다. 대신 이 경계 조건을 표본으로 확인하고 다차원 이분법(정육면체를 나누며 경계 조건을 유지하는 칸을 따라감)으로 무차별점 근처를 찾을 수 있다.
   - 그 근처가 바로 라벨이 바뀌는 곳이라, 자르기 문턱의 좋은 후보가 된다.
   - 1차원이면 중간값 정리이고 이분 탐색이다.
2. **하한의 근거 [스케치].**
   - 무차별점 $x^\star$ 근처에서는 공통 라벨로 $e_I\le\varepsilon$을 지키는 칸의 크기가 이점의 기울기로 제한된다.
   - 그래서 라벨이 만나는 점의 개수가 필요한 bucket 수의 하한을 준다.
   - Sperner의 다면체 일반화(De Loera, Peterson, Su 2002)는 완전히 칠해진 칸이 최소 $n-d$개라는 셈을 준다. 이 셈을 하한으로 옮길 수 있는지는 확인하지 않았다.

**Sperner, KKM, Scarf [문헌].**
- ASWISS의 라벨(최대 후회 행동)은 색칠이다. "완전히 칠해진 칸" = 모든 행동이 무차별인 곳 = 근사 고정점이라는 Sperner–Scarf의 틀과 같은 모양이다.
- 쓸 자리는 셋이다.
  1. 추상 게임 균형의 근사 계산(Scarf 1967, Lemke–Howson 1964의 완전 라벨 경로)
  2. 일반합 게임 균형 계산의 PPAD-완전성 맥락(Daskalakis, Goldberg, Papadimitriou 2009; Chen, Deng, Teng 2009)
  3. 2인 제로섬 원 게임은 선형계획으로 다항 시간이라 존재나 계산에 이 도구가 필요 없다
- 그래서 직접 쓸 곳은 1절의 존재(Brouwer)와 4절의 문턱 탐색 정도다.

## 5. Helly: 라벨 일치를 적은 멤버로 확인

**명제 4 [증명].**
- bucket $B$의 행동 수를 $k=\lvert A_B\rvert+1$("없음" 포함)이라 하자.
- 멤버마다 최대 라벨 집합 $M(I)\subseteq A_B\cup\{\varnothing\}$가 있다.
- 멤버 $k$개로 이루어진 모든 부분집합이 공통 최대 라벨을 가지면, $B$ 전체가 공통 최대 라벨을 가진다. 즉 $H_B=0$이다.

*증명.*
- $M(I)$를 $(k-1)$차원 단체(꼭짓점 = 라벨)의 면 $\mathrm{conv}(M(I))$로 본다. 단체의 면은 볼록 집합이다.
- 두 면이 만나는 것은 공통 꼭짓점이 있는 것과 같다. 면들의 교집합도 그 꼭짓점들의 볼록 껍질이기 때문이다.
- $\mathbb R^{k-1}$에서 Helly 정리(모든 $k$개가 만나면 전체가 만남)를 쓴다. □

**분수 Helly판 [스케치].** 표본에서 $k$개 묶음의 큰 비율이 공통 라벨을 가지면, 멤버의 큰 비율이 한 라벨을 공유한다. 라벨 항등식으로 나머지 멤버의 $e_I$만 남는다.
- 이것은 순수 ASWISS의 저수지 표본 판정(멤버 16개)과 증명 개요의 L4(표본 추정)에 쓸 수 있다.
- 동률이 없으면(멤버마다 라벨 하나) 더 직접적인 계산이 된다. 무작위 $k$명이 같은 라벨일 확률은 $\sum_\ell p_\ell^k\le\max_\ell p_\ell^{k-1}$이다.
- 그래서 일치가 자주 보이면 최다 라벨의 비율이 크다.

## 6. absent-mindedness와 해 개념 [문헌]

- **플레이어 단위 균형.** 불완전 기억 게임에서 플레이어 단위 행동 전략 내쉬 균형은 존재하지 않을 수 있다(Wichardt 2008, GEB 63:366–369). 불완전 기억 게임의 균형 개념 정리는 Lambert, Marple, Shoham(GEB 2019)에 있다.
- **agent 단위 균형.** 정보집합(bucket)마다 행위자를 따로 두는 agent form에서는 Nash(Kakutani/Brouwer) 정리로 균형이 존재한다.
- **absent-minded driver.** absent-mindedness가 있을 때 계획 최적과 행동 최적(modified multiselves)의 관계는 Piccione & Rubinstein(1997), Aumann, Hart, Perry(1997)가 다뤘다.
- **논문 후속 연구 4와의 연결.** 명제 1은 이 두 개념과 또 다른 세 번째 정상점(반사실 합 기준)이다. absent-minded bucket에서 무엇을 학습 목표로 둘지 정할 때 비교 대상이 된다.

## 7. 쓸모가 작은 것

| 정리 | 이유 |
|---|---|
| Tarski (격자 고정점) | 분할은 유한 격자이고 분할만 하므로(세분 열), 멈춤은 유한성만으로 나온다(논문 보조정리 5.8). 단조성이 없어서(추상화 병리) 최소 고정점의 의미도 약하다 |
| ham sandwich, 목걸이 분할(Alon 1987), consensus halving | 여러 측도를 *똑같이* 나누는 정리다. 우리는 라벨별로 *가르고* 싶다. 방향이 반대다 |
| Borsuk–Ulam, Tucker | 대척 대칭이 필요하다. 제로섬의 부호 대칭을 쓸 자리를 찾지 못했다 |
| Shapley–Folkman | 많은 집합의 합이 거의 볼록이라는 결과다. 숨은 후회 $\sum\psi-\psi(\sum)$를 묶는 데는 직접 쓰이지 않는다 |
| Banach (축약 사상) | 후회 최소화 동역학은 축약이 아니다 |

**추측 수준의 연결 [스케치].**
- 추상화 세분을 거친 게임에서 고운 게임으로 가는 변형(homotopy)으로 보고, Brouwer 차수와 지표 이론의 균형 경로 추적(Govindan–Wilson, Herings–Peeters)을 쓰는 방향이다.
- 이렇게 하면 분할 직후의 일시적 악화(1단 게임 반례 B)를 경로 위의 연속 변화로 다룰 수 있을지 모른다. 아직 구체적인 진술은 없다.

## 8. 다음 단계 제안

1. 명제 1을 논문 5절(또는 7.4절)에 넣는다. 정리 B3의 "경로 분리 가정 아래 존재" 문장을 명제 1로 바꾼다.
2. 정리 3을 논문 8절 후속 연구 1(초기 추상화)과 2(분할 개선)에 "추상화 크기의 기준"으로 넣는다. mini-stud에서 손패 특징에 대한 이점의 Lipschitz 상수를 실측해 가정이 맞는지 본다.
3. Poincaré–Miranda 다차원 이분법을 관측 분할의 후보 생성으로 시험한다. 지금은 무작위 문턱을 쓴다.
4. 분수 Helly판을 표본 판정의 증명(증명 개요의 L4)에 붙일 수 있는지 정리한다.
5. 정리 3의 Blackwell판(1.1절)에서 비례 도달 가정이 mini-stud에서 얼마나 어긋나는지 잰다. 같은 bucket 멤버들의 $\pi^t_{-i}(I)/\sum_s\pi^s_{-i}(I)$ 시계열을 비교하면 된다.

## 참고 문헌

- Aumann, R. J., Hart, S., & Perry, M. (1997). The absent-minded driver. *Games and Economic Behavior*, 20(1), 102–116.
- Bubelis, V. (1979). On equilibria in finite games. *International Journal of Game Theory*, 8(2), 65–79.
- Chen, X., Deng, X., & Teng, S.-H. (2009). Settling the complexity of computing two-player Nash equilibria. *Journal of the ACM*, 56(3).
- Daskalakis, C., Goldberg, P. W., & Papadimitriou, C. H. (2009). The complexity of computing a Nash equilibrium. *SIAM Journal on Computing*, 39(1).
- De Loera, J. A., Peterson, E., & Su, F. E. (2002). A polytopal generalization of Sperner's lemma. *Journal of Combinatorial Theory, Series A*, 100(1), 1–26.
- Helly, E. (1923). Über Mengen konvexer Körper mit gemeinschaftlichen Punkten. *Jahresbericht der DMV*, 32.
- Koller, D., & Megiddo, N. (1992). The complexity of two-person zero-sum games in extensive form. *Games and Economic Behavior*, 4(4).
- Hart, S., & Mas-Colell, A. (2000). A simple adaptive procedure leading to correlated equilibrium. *Econometrica*, 68(5), 1127–1150.
- Hart, S., & Mas-Colell, A. (2003). Uncoupled dynamics do not lead to Nash equilibrium. *American Economic Review*, 93(5), 1830–1836.
- Hart, S., & Schmeidler, D. (1989). Existence of correlated equilibria. *Mathematics of Operations Research*, 14(1), 18–25.
- Kulpa, W. (1997). The Poincaré–Miranda theorem. *The American Mathematical Monthly*, 104(6), 545–550.
- Lambert, N. S., Marple, A., & Shoham, Y. (2019). On equilibria in games with imperfect recall. *Games and Economic Behavior*, 113, 164–185.
- Lemke, C. E., & Howson, J. T. (1964). Equilibrium points of bimatrix games. *SIAM Journal on Applied Mathematics*, 12(2).
- McLennan, A., & Tourky, R. Imitation games and computation. (ANU 연구 포털에서 제목과 초록만 확인)
- Miranda, C. (1940). Un'osservazione su un teorema di Brouwer. *Bollettino dell'Unione Matematica Italiana*.
- Nash, J. (1951). Non-cooperative games. *Annals of Mathematics*, 54(2), 286–295.
- Piccione, M., & Rubinstein, A. (1997). On the interpretation of decision problems with imperfect recall. *Games and Economic Behavior*, 20(1), 3–24.
- Scarf, H. (1967). The approximation of fixed points of a continuous mapping. *SIAM Journal on Applied Mathematics*, 15(5).
- (무리수 균형 예) Nash Equilibria with Irradical Probabilities, arXiv:2507.09422. 보수 {0,1,2}인 3인 게임의 유일 균형이 무리수 확률뿐인 예(명제 3.1)와 Bubelis(1979)의 인용.
- Wichardt, P. C. (2008). Existence of Nash equilibria in finite extensive form games with imperfect recall: A counterexample. *Games and Economic Behavior*, 63(1), 366–369.
- Lebesgue 수 보조정리: 표준 위상수학 교재(예: Munkres, *Topology*, 2판, 보조정리 27.5).
- (서지 정보 가운데 권호와 쪽은 원문으로 다시 확인할 것.)
