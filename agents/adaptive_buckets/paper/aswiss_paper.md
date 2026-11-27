# 불완전정보 전개형 게임에서 전략 기반 추상화 분할을 통한 내쉬 균형 수렴

**ASWISS: 표본 후회의 라벨로 추상화를 세분하는 MCCFR**

> 이 파일은 논문 초안(`main.tex`, `main.pdf`)의 Markdown 사본이자 편집본이다. 문장을 바로 고치거나, 고칠 곳 아래에 `>> 요청` 줄을 적는다. 고친 문장은 .tex에 그대로 옮긴다.

## 초록

유한 2인 제로섬 불완전정보 전개형 게임의 내쉬 균형을 근사하는 표준 방법은 반사실적 후회 최소화(CFR)와 그 표본판인 MCCFR이다.
큰 게임에서는 정보집합을 묶은 추상 게임을 푸는데, 고정 추상화는 추상화 오차 때문에 원 게임의 균형으로 수렴하지 않는다.
CFR+IRA는 학습 중에 추상화를 세분해 원 게임의 균형으로 수렴시키지만, 매 반복 원 게임 트리 전체를 순회해야 한다.

이 논문은 표본 기반으로 추상화를 세분하는 *순수 ASWISS*(Abstraction Split WIth Sampling by Strategy)를 제안한다.
ASWISS는 게임 구조만으로 만든 가장 거친 분할에서 시작해 외부 샘플링 MCCFR로 bucket 전략을 학습한다.
길이가 두 배씩 느는 epoch이 끝날 때마다, 각 정보집합에 그 epoch의 반복 전략들에 대한 누적 반사실적 후회가 가장 큰 행동을 라벨로 붙이고, 라벨이 갈린 bucket을 라벨별로 나눈다.
정보집합이 어느 bucket에 속하는지는 정보집합별 표 없이 분할 계보를 따라 다시 계산한다.

이론적으로, bucket 안에서 상쇄되어 보이지 않는 후회(숨은 후회)가 "모든 멤버에게 한 라벨을 강요할 때의 최소 비용"과 정확히 같다는 라벨 항등식을 보인다.
이 항등식에서 분할 횟수의 유한성과, 마지막 분할 이후 원 게임 NashConv가 높은 확률로 $O\big(\sqrt{\log(kN_A/\delta)/L_k}\big)$이고 거의 확실하게 0으로 간다는 결과가 나온다.
epoch의 반복 전략을 $K_k$개만 저장해 라벨을 재생해도 $K_k/\log k\to\infty$이면 같은 결론이 성립한다.
추상 게임의 후회가 작아질 때만 판정하는 변형에는 분할 없이 판정을 통과한 시점의 NashConv 인증을 준다.

정보집합 51,114개의 축소 세븐 포커(mini-stud)에서, 순수 ASWISS의 규칙은 1억 3,107만 반복에 bucket 17,455개로 정확한 exploitability 0.0140에 도달했다. 같은 반복의 카드 특징 k-means 고정 추상화는 0.297이었다(seed 1).
전체 순회판은 같은 반복의 CFR+IRA보다 26–28배 낮았다.
반면 정보집합별 표 없이 계보를 따라 다시 계산하는 비용은 분할 단계마다 20–40배씩 늘었다.
이 비용을 줄이는 파생 기법(평균 전략 방향, bucket 이력, 관측 분할과 전략 분할의 이중 구조, 숨은 후회 검정)은 뒤에서 따로 소개한다.

## 1. 서론

여러 플레이어가 차례로 행동을 고르는 순차적 상호작용은 게임 트리로 나타낼 수 있다. 이것이 전개형 게임(extensive-form game, EFG)이다.
전개형 게임은 우연에 의한 사건과, 플레이어가 서로 구별하지 못하는 상태(불완전정보)를 함께 표현한다.
마르코프 결정 과정(MDP)이나 부분 관측 MDP(POMDP)와 구조가 비슷하지만, 여러 에이전트가 서로의 보상에 영향을 준다는 점이 다르다.
그래서 한 에이전트의 보상을 최대화하는 것이 목표가 아니라, 어느 플레이어도 혼자 전략을 바꿔 이득을 볼 수 없는 내쉬 균형을 찾는 것이 목표다.
유한 게임에는 혼합 전략 내쉬 균형이 항상 존재한다 (Nash, 1951).
2인 제로섬 게임에서는 최소최대 정리에 의해 균형 값이 하나로 정해지고, 균형 전략끼리 바꿔 써도 균형이 유지된다. 그래서 균형 하나를 찾는 것이 의미 있는 목표가 된다.
또 완전 기억(perfect recall) 게임에서는 행동 전략만으로 모든 혼합 전략을 표현할 수 있다 (Kuhn, 1953).

2인 제로섬 불완전정보 게임의 균형을 근사하는 표준 방법은 반사실적 후회 최소화(counterfactual regret minimization, CFR)다 (Zinkevich 외, 2007).
CFR은 정보집합마다 "자신이 그 정보집합에 가려고 했다면"의 가치, 즉 상대와 우연의 도달확률만으로 가중한 반사실적 가치로 후회를 잰다.
이렇게 정의하면 전체 후회가 정보집합별 반사실적 후회의 합으로 위에서 묶인다. 그래서 정보집합마다 따로 후회를 줄이는 것만으로 전체 후회가 줄어든다.
정보집합마다 후회 매칭(regret matching)을 돌리면, 반복 전략들의 평균이 $O(1/\sqrt T)$의 exploitability로 균형에 다가간다.

그러나 CFR은 매 반복 게임 트리 전체를 순회한다. 트리는 깊이에 대해 지수적으로 커지므로, 큰 게임에서는 한 번의 순회도 어렵다.
MCCFR (Lanctot 외, 2009)은 트리의 일부만 표본으로 순회하면서 반사실적 가치의 불편 추정량으로 후회를 갱신한다.
반복당 계산량은 훨씬 작으면서, 높은 확률로 CFR과 같은 차수의 수렴률을 가진다.
하지만 정보집합마다 따로 표를 두는 한, 처음 방문한 정보집합에는 다른 정보집합에서 배운 것을 줄 수 없다. 또 드물게 방문되는 정보집합의 전략은 표본 분산 때문에 크게 흔들린다.

그래서 큰 게임에서는 비슷한 정보집합이나 행동을 하나로 묶은 추상 게임을 푼다. 추상화에는 두 방향이 있다.

- 연속적이거나 많은 행동을 대표 행동으로 줄이는 행동 추상화
- 비슷한 정보집합을 하나의 bucket으로 묶는 정보 추상화

둘 다 처음 보는 정보집합이나 행동에도 학습한 전략을 쓸 수 있게 한다. 이 논문은 정보 추상화를 다룬다.

고정된 정보 추상화에는 세 가지 문제가 있다.

1.  추상 게임의 균형은 원 게임의 균형이 아니다. 고정 추상화에서 CFR이 수렴하는 곳은 추상 게임의 균형이고, 원 게임에서는 추상화 오차만큼 공략당한다.
2.  추상화를 더 세밀하게 해도 원 게임에서 반드시 좋아지지는 않는다(추상화 병리 현상 (Waugh 외, 2009)).
    과거 정보를 잊는 불완전 기억 추상화에서는 CFR의 보장 자체가 일반적으로 성립하지 않는다 (Lanctot 외, 2012).
3.  좋은 추상화를 만들려면 손패 세기의 군집처럼 게임마다 사람이 고른 특징이 필요하다 (Johanson 외, 2013; Ganzfried & Sandholm, 2014). 이런 특징은 일반 전개형 게임으로 옮기기 어렵다.

CFR+IRA (Čermák 외, 2020)는 학습 중에 거친 초기 추상화를 세분해 원 게임 균형으로 수렴시킨다.
하지만 매 반복 원 게임 트리 전체를 순회해야 해서, 순회 한 번도 어려운 큰 게임에는 쓸 수 없다.
저자들도 이것을 주된 한계로 꼽고, 표본 기법과의 결합을 후속 과제로 남겼다 (Čermák 외, 2020, 6.1절).

**이 논문의 방법.**
이 논문은 표본만으로 추상화를 세분하는 *ASWISS*(Abstraction Split WIth Sampling by Strategy)를 제안한다. 본문은 가장 단순한 판인 *순수 ASWISS*를 다룬다.

- 게임 구조만으로 만든 가장 거친 분할(같은 플레이어, 같은 행동집합, 같은 시점)에서 시작한다.
- 현재 분할 위에서 외부 샘플링 MCCFR로 bucket 전략을 학습한다.
- 길이가 두 배씩 느는 epoch이 끝날 때마다, 각 정보집합에 그 epoch의 반복 전략들에 대한 *누적* 반사실적 후회가 가장 큰 행동을 라벨로 붙인다. 그런 행동이 없으면 "없음" 라벨을 붙인다.
- 라벨이 갈린 bucket을 라벨별로 나눈다. 병합은 하지 않는다.
- 정보집합의 bucket은 정보집합별 표 없이, 시작 분할에서 분할 사건을 따라 내려가며 매번 다시 정한다(계보 라우팅). 그래서 처음 보는 정보집합에도 bucket이 정해진다.

이름의 "by Strategy"는 bucket을 정하는 것이 정보집합의 특징이 아니라 전략이 그 정보집합에 주는 업데이트 방향이라는 뜻이다.

**기여.**

1.  **라벨 항등식과 수렴 정리.** bucket 안에서 상쇄되어 bucket이 느끼지 못하는 후회(숨은 후회)가 "모든 멤버에게 한 라벨을 강요할 때의 최소 비용"과 정확히 같음을 보인다(보조정리 5.4).
    이것으로 외부 샘플링 추정치만으로 분할을 판정해도, 분할이 유한 번이고 마지막 분할 이후 원 게임 NashConv가 0으로 감을 증명한다(정리 5.9).
    epoch의 반복 전략을 $K_k$개만 저장해 라벨을 재생하는 판에도 같은 꼴의 보장을 준다(정리 5.15).
2.  **추상 게임의 수렴을 기다리는 변형과 정지 인증.** 추상 게임의 후회가 $\varepsilon$ 이하일 때만 판정하는 변형을 다룬다. 분할 없이 판정을 통과한 시점의 NashConv 인증(정리 5.12)을 준다.
3.  **실험.** 정확한 exploitability를 잴 수 있는 작은 게임(Kuhn, Leduc, mini-stud)에서 CFR+IRA, 추상화 없는 CFR 계열, k-means 고정 추상화, 신경망 방법(MMD, ESCHER)과 비교한다.
    정보집합별 표 없이 계보를 따르는 비용도 실측한다.
4.  **파생 기법.** 계보 라우팅 비용과 방향 추정의 품질을 고치는 기법들을 같은 틀에서 정리한다(7절). 평균 전략 방향과 실제 업데이트 방향의 정확한 차이(정리 7.1)가 그중 하나다.

**구성.**
2절은 관련 연구, 3절은 배경(CFR, MCCFR, CFR+IRA의 정리)이다.
4절에서 순수 ASWISS를 정의하고, 5절에서 이론 결과를 정리한다. 증명은 부록 A에 있다.
6절은 실험, 7절은 파생 기법, 8절은 한계와 후속 연구다.

## 2. 관련 연구

### 2.1 가상 플레이(Fictitious Play)

가상 플레이 (Brown, 1951)는 매 반복 상대의 *경험적 평균 전략*에 대한 최선 응답을 두고, 자기 평균 전략을 갱신한다.
2인 제로섬 행렬 게임에서는 평균 전략이 균형으로 수렴한다 (Robinson, 1951).
전개형 게임에서는 정규형으로 바꾸면 전략 수가 지수적으로 커진다. 그래서 Heinrich 외 (Heinrich 외, 2015)는 행동 전략 위에서 같은 갱신을 하는 XFP와, 표본과 학습으로 근사하는 FSP를 제안했다.
NFSP (Heinrich & Silver, 2016)는 최선 응답을 강화학습으로, 평균 전략을 지도학습으로 근사한다.
추상화 세분과 직접 관련된 것은 FPIRA다 (Čermák 외, 2017; Čermák 외, 2020).
FPIRA는 불완전 기억 추상화 위에서 가상 플레이를 돌리면서, 필요한 정보를 추상화에 되돌려 넣는다. 그래서 원 게임에서 돌린 가상 플레이와 exploitability가 매 반복 정확히 같다 (Čermák 외, 2020, 정리 1).
가상 플레이 계열은 매 반복 원 게임의 최선 응답이 필요하다. 큰 게임에서는 이것이 병목이다.

### 2.2 후회 최소화와 CFR 계열

후회 매칭 (Hart & Mas-Colell, 2000)은 누적 후회의 양수 부분에 비례하는 전략을 두는 온라인 학습 규칙으로, Blackwell 접근성 (Blackwell, 1956)에서 나왔다.
CFR (Zinkevich 외, 2007)은 전체 후회를 정보집합별 반사실적 후회의 합으로 묶어, 정보집합마다 후회 매칭을 돌리는 것만으로 균형을 근사한다(3.3절).
CFR+ (Tammelin, 2014; Tammelin 외, 2015)는 음수 후회를 0에서 자르는 RM+, 반복 번호에 비례하는 평균 가중치, 교대 갱신을 쓴다. 헤즈업 리밋 홀덤을 사실상 풀었다 (Bowling 외, 2015).
교대 갱신의 이론은 Burch 외 (Burch 외, 2019)가 다시 정리했다.
DCFR (Brown & Sandholm, 2019)은 과거 후회와 평균을 할인하고, PCFR+ (Farina 외, 2021)는 다음 반복의 후회를 예측해 더 빨리 수렴한다.
이들은 모두 원 게임(또는 고정된 추상 게임)을 푸는 방법이다. 추상화를 바꾸지 않는다.

### 2.3 표본 기반 후회 최소화

MCCFR (Lanctot 외, 2009)은 종료 이력의 일부만 표본으로 뽑아 순회하고, 표본에 뽑힐 확률로 나눈 반사실적 가치로 후회를 갱신한다(3.4절).
외부 샘플링, 결과 샘플링, 우연 샘플링 같은 방식이 있다. 분산과 수렴의 관계는 Gibson 외 (Gibson 외, 2012)가 분석했다.
VR-MCCFR (Schmid 외, 2019)은 기준값(baseline)으로 분산을 줄이면서도 불편성을 유지한다.
Farina 외 (Farina 외, 2020)는 불편한 기울기 추정량을 쓰는 임의의 후회 최소화기에 대해 높은 확률의 상한을 주었다. 외부 샘플링 MCCFR에서는 실패 확률 $p$에 대한 의존을 $O(\sqrt{1/p})$에서 $O(\sqrt{\log(1/p)})$로 줄였다.

### 2.4 함수 근사

표 대신 신경망으로 후회나 가치를 근사하면 추상화를 따로 만들 필요가 없다.
Deep CFR (Brown 외, 2019)은 외부 샘플링으로 모은 표본 후회를 신경망으로 회귀하고, Single Deep CFR (Steinberger, 2019)은 평균 전략 신경망을 없앴다.
DREAM (Steinberger 외, 2020)은 결과 샘플링에 기준값을 더해 모델 없이 학습하고, ESCHER (McAleer 외, 2023)는 중요도 가중 없이 가치 신경망으로 후회를 추정한다.
그보다 앞서 RCFR (Waugh 외, 2015)은 후회를 특징 위의 회귀로 근사했다.
이 방법들은 근사 오차가 남기 때문에 원 게임 균형으로의 수렴이 보장되지 않는다.

### 2.5 1차 방법과 경사 하강–상승(GDA)

2인 제로섬 전개형 게임은 sequence form (Stengel, 1996) 위의 쌍선형 안장점 문제다.
그래서 경사 하강–상승(gradient descent–ascent, GDA) 같은 1차 방법을 그대로 쓸 수 있다.

- 단순한 동시 GDA는 쌍선형 문제에서 순환하거나 발산할 수 있다.
- 낙관적 GDA와 extragradient (Daskalakis 외, 2018; Mertikopoulos 외, 2019)는 마지막 반복의 수렴을 회복한다.
- 전개형 게임 전용으로는 다음 방법들이 있다.
  - 확장 정규화(dilated regularizer)를 쓴 평활화와 excessive gap technique (Hoda 외, 2010; Kroer 외, 2020)
  - 마지막 반복 수렴을 보인 낙관적 거울 하강 (Lee 외, 2021)
  - 강화학습과 같은 틀에서 마지막 반복 수렴을 주는 MMD (Sokota 외, 2023)

이들도 원 게임을 직접 다루며, 학습 중에 추상화를 세분하지는 않는다.

### 2.6 정보 추상화

- **무손실 추상화.** 무손실 추상화는 GameShrink (Gilpin & Sandholm, 2007)가 만든다.
- **손실 추상화.** 손실 추상화는 대개 손패 세기 분포를 군집해 만든다 (Johanson 외, 2013; Ganzfried & Sandholm, 2014).
- **불완전 기억.** 과거 정보를 잊는 불완전 기억 추상화는 같은 메모리로 더 좋은 전략을 준다 (Waugh 외, 2009). 하지만 CFR의 보장은 특정 조건에서만 유지된다 (Lanctot 외, 2012).
- **오차 상한.** 추상화 해의 원 게임 오차는 묶인 정보집합들의 보상과 우연 확률이 얼마나 다른지로 묶을 수 있다 (Kroer & Sandholm, 2014; Kroer & Sandholm, 2016; Kroer & Sandholm, 2018). 이 상한들은 추상화가 고정된 경우에 대한 것이다.
- **병리 현상.** 추상화를 세밀하게 해도 원 게임 성능이 나빠질 수 있다 (Waugh 외, 2009).
- **풀이 중 추상화 수정.** 풀이 중에 행동 추상화에 행동을 더하는 방법 (Brown & Sandholm, 2015)이 있다. 추상화가 바뀔 때 후회를 옮겨 쓰는 regret transfer (Brown & Sandholm, 2014)도 있다.

### 2.7 학습 중 정보 추상화 세분: FPIRA와 CFR+IRA

- **세분 방식.** FPIRA와 CFR+IRA (Čermák 외, 2017; Čermák 외, 2020)는 임의의 불완전 기억 추상화에서 시작해, 학습 중에 추상화를 세분하면서 원 게임 균형으로 수렴한다. 이 논문의 직접적인 선행 연구다(3.6절).
- **CFR+IRA의 보장.** 무작위로 감시하는 정보집합의 후회가 후회 매칭이 보장하는 속도보다 느리게 줄면 그 정보집합을 떼어 낸다. 이것으로 확률적 수렴 보장을 얻는다.
- **휴리스틱 갱신.** 실전 속도를 위해 한 반복의 순간 후회로 비슷한 정보집합끼리 묶는 휴리스틱 갱신을 함께 쓴다. 원문은 이 휴리스틱만으로는 수렴하지 않는 반례도 준다.
- **한계.** 두 방법 모두 매 반복 원 게임 트리 전체를 순회하거나 원 게임의 최선 응답을 계산한다.

### 2.8 실시간 탐색

DeepStack (Moravčík 외, 2017), Libratus (Brown & Sandholm, 2018), Pluribus (Brown & Sandholm, 2019), ReBeL (Brown 외, 2020)은 학습한 청사진이나 가치 함수 위에서 대국 중에 하위 게임을 다시 푼다.
상대의 반사실적 가치를 제약으로 두면 다시 푼 전략이 원래보다 나빠지지 않게 할 수 있다(CFR-D (Burch 외, 2014), safe subgame solving (Brown & Sandholm, 2017)).
이 논문은 청사진을 만드는 단계의 추상화를 다룬다. 탐색과는 함께 쓸 수 있다.

### 2.9 이 논문의 위치

- ASWISS는 CFR+IRA처럼 학습 중에 정보 추상화를 세분해 원 게임 균형으로의 수렴을 노린다. 차이는 셋이다.
  1.  분할 판정과 학습을 모두 외부 샘플링 추정치로 한다.
  2.  사람이 고른 특징 없이 게임 구조에서 시작한다.
  3.  분할 기준이 한 반복의 순간 후회가 아니라 epoch 동안의 *누적* 후회의 라벨이다.
- 셋째 차이 때문에 CFR+IRA의 휴리스틱 반례(3.6절)가 ASWISS의 규칙에는 해당하지 않는다. 누적 후회의 라벨이 같으면 숨은 후회가 0이기 때문이다(보조정리 5.4).
- 표본 기반으로 정보 추상화를 학습 중에 세분하면서 원 게임 수렴을 증명한 선행 연구는 찾지 못했다. **\[확인: 제출 전 문헌 조사를 다시 할 것\]**

## 3. 배경

### 3.1 전개형 게임과 기호

전개형 게임은 $G=(\mathcal{N},\mathcal{H},\mathcal{Z},A,P,\sigma_c,u,\mathcal{I})$로 쓴다.

- $\mathcal{N}=\{1,2,c\}$는 플레이어 집합이고, $c$는 우연이다. $i$는 한 플레이어, $-i$는 그 상대와 우연을 뜻한다.
- $\mathcal{H}$는 루트에서 시작하는 행동열(이력)의 유한 집합이다. $h\sqsubseteq h'$는 $h$가 $h'$의 앞부분이라는 뜻이다. $\mathcal{Z}\subseteq\mathcal{H}$는 종료 이력이다.
- $A(h)$는 비종료 이력 $h$에서 둘 수 있는 행동집합이고, $P(h)\in\mathcal{N}$은 $h$에서 행동하는 플레이어다.
- $\sigma_c(h,\cdot)$는 우연 노드 $h$의 고정 분포이고, $u_i:\mathcal{Z}\to\mathbb{R}$는 효용이다. 2인 제로섬이면 $u_1=-u_2$이고, $\Delta=\max_zu_1(z)-\min_zu_1(z)$를 효용 폭이라 한다.
- $\mathcal{I}_i$는 플레이어 $i$의 정보집합들로, $P(h)=i$인 이력들의 분할이다. 같은 정보집합의 이력은 행동집합이 같다($A(I)$).

**전략과 도달확률.**
행동 전략 $\sigma_i$는 각 $I\in\mathcal{I}_i$에 분포 $\sigma_i(I)\in\Delta(A(I))$를 준다. $\sigma=(\sigma_1,\sigma_2)$를 프로필이라 한다.

- $\pi^\sigma(h)$는 $\sigma$와 $\sigma_c$로 $h$에 이를 확률이다. 이것을 플레이어 $i$의 행동 확률의 곱 $\pi_i^\sigma(h)$와 나머지(상대와 우연)의 곱 $\pi_{-i}^\sigma(h)$로 나눈다. 그러면 $\pi^\sigma(h)=\pi_i^\sigma(h)\pi_{-i}^\sigma(h)$다.
- $h\sqsubseteq z$이면 $\pi^\sigma(h,z)$는 $h$에서 $z$까지 행동 확률의 곱이다.
- $\mathcal{Z}_I=\{(h,z):h\in I,\ z\in\mathcal{Z},\ h\sqsubseteq z\}$로 둔다.
- 기대 효용은 $u_i(\sigma)=\sum_z\pi^\sigma(z)u_i(z)$다.

**완전 기억.**
같은 정보집합 $I\in\mathcal{I}_i$의 모든 이력에서 플레이어 $i$가 지나온 (정보집합, 행동)의 열이 같으면, 그 게임을 완전 기억 게임이라 한다.
이때 $\pi_i^\sigma(h)$는 $h\in I$에 대해 같으므로 $\pi_i^\sigma(I)$로 쓴다. 정보집합 $I$ 앞에 $i$가 둔 수의 개수를 $I$의 *자기 깊이*라 한다.
완전 기억이면 행동 전략만으로 모든 혼합 전략을 표현할 수 있다 (Kuhn, 1953).
또 행동 전략 $\sigma_i$와 종료 이력마다의 자기 행동 확률 $x_i(z)=\pi_i^\sigma(z)$(실현 계획, sequence form (Stengel, 1996))가 일대일로 대응한다. $u_i$는 각 플레이어의 실현 계획에 대해 선형이다.

**균형과 exploitability.**
$e_i(\sigma)=\max_{\sigma'_i}u_i(\sigma'_i,\sigma_{-i})-u_i(\sigma)$를 플레이어 $i$의 최선 응답 이득이라 한다.
$\mathrm{NashConv}(\sigma)=\sum_ie_i(\sigma)$가 0이면 $\sigma$는 내쉬 균형이다. $\mathrm{NashConv}(\sigma)\le\epsilon$이면 $\epsilon$-균형이라 한다.
이 논문은 실험에서 exploitability를 $\mathrm{NashConv}/2$로 보고한다.

### 3.2 후회와 평균 전략

반복 $t=1,\dots,T$의 프로필 $\sigma^t$에 대해, 플레이어 $i$의 전체 후회는 다음과 같다.

$$R_i^T=\max_{\sigma'_i}\sum_{t=1}^T\Big(u_i(\sigma'_i,\sigma^t_{-i})-u_i(\sigma^t)\Big).$$

평균 전략 $\bar\sigma^T$는 실현 계획의 평균 $\bar x_i=\frac1T\sum_tx_i^t$에 대응하는 행동 전략이다. 정보집합마다 쓰면 자기 도달확률로 가중한 평균이다.

$$\tag{1}
\bar\sigma_i^T(I,a)=\frac{\sum_{t=1}^T\pi_i^{\sigma^t}(I)\,\sigma^t(I,a)}{\sum_{t=1}^T\pi_i^{\sigma^t}(I)}.$$

$\bar\sigma^T$는 반복 전략들을 균등하게 섞은 혼합 전략과 실현 동치다.

**정리 3.1 (평균 전략의 NashConv; folk theorem).**
2인 제로섬 게임에서 두 플레이어가 같은 프로필 열 $\sigma^1,\dots,\sigma^T$를 쓰면 $\mathrm{NashConv}(\bar\sigma^T)=(R_1^T+R_2^T)/T$다.

*증명.*
상대의 실현 계획에 대한 선형성으로 다음이 성립한다.

$$u_i(\sigma'_i,\bar\sigma^T_{-i})=\frac1T\sum_tu_i(\sigma'_i,\sigma^t_{-i}),\qquad
\max_{\sigma'_i}u_i(\sigma'_i,\bar\sigma_{-i}^T)=\frac1T\Big(R_i^T+\sum_tu_i(\sigma^t)\Big).$$

제로섬이므로 $\sum_iu_i(\bar\sigma^T)=0$이고 $\sum_iu_i(\sigma^t)=0$이다. 두 플레이어에 대해 더하면 된다.
∎

### 3.3 CFR

**정의 3.2 (반사실적 가치와 후회 (Zinkevich 외, 2007)).**
$I\in\mathcal{I}_i$의 반사실적 가치와 행동 $a$의 반사실적 가치를 다음과 같이 쓴다.

$$v_i(\sigma,I)=\sum_{(h,z)\in\mathcal{Z}_I}\pi^\sigma_{-i}(h)\,\pi^\sigma(h,z)\,u_i(z),\qquad
v_i(\sigma,I,a)=\sum_{(h,z)\in\mathcal{Z}_I,\ ha\sqsubseteq z}\pi^\sigma_{-i}(h)\,\pi^\sigma(ha,z)\,u_i(z).$$

정의상 $v_i(\sigma,I)=\sum_a\sigma(I,a)v_i(\sigma,I,a)$다.
프로필 $\sigma$에서의 순간 반사실적 후회는 $r^\sigma(I,a)=v_i(\sigma,I,a)-v_i(\sigma,I)$다.
반복 $t$의 순간 후회는 $r^t=r^{\sigma^t}$이고, 누적 후회는 $R^T(I,a)=\sum_{t\le T}r^t(I,a)$다.

$$\psi(x)=\max\big(0,\max_ax_a\big)$$

로 둔다. Zinkevich 외의 "직접 반사실적 후회의 양수 부분"은 $\psi(R^T(I))/T$다.

CFR은 정보집합마다 후회 매칭으로 다음 전략을 정한다.

$$\tag{2}
\sigma^{T+1}(I,a)=\frac{R^T(I,a)^+}{\sum_bR^T(I,b)^+}\qquad(\text{분모가 0이면 균등 전략}).$$

CFR의 정리는 세 부분으로 되어 있다.

1.  전체 후회가 정보집합별 후회로 묶인다(정리 3.4).
2.  정보집합마다 후회 매칭이 후회를 $O(\sqrt T)$로 묶는다(정리 3.6).
3.  이 둘과 정리 3.1로 평균 전략이 균형에 수렴한다(따름정리 3.7).

아래에서 (i)은 Zinkevich 외의 재귀적 증명 대신 성능 차이 항등식(보조정리 3.3)으로 증명한다. 같은 항등식을 5절에서 다시 쓴다.

**보조정리 3.3 (성능 차이 항등식).**
완전 기억 게임에서 임의의 프로필 $\sigma$와 플레이어 $i$의 임의의 순수 전략 $\beta$에 대해 다음이 성립한다.

$$u_i(\beta,\sigma_{-i})-u_i(\sigma)=\sum_{I\in\mathcal{I}_i}\pi_i^\beta(I)\,r^\sigma\big(I,\beta(I)\big).$$

여기서 $\pi_i^\beta(I)\in\{0,1\}$은 $\beta$를 따를 때 $I$에 이르는지를 나타낸다.

*증명.*
$q(h,a)=\sum_{z\sqsupseteq ha}\pi^\sigma(ha,z)u_i(z)$로 둔다. $D$를 자기 깊이의 최댓값이라 하자.
$d=0,\dots,D+1$에 대해 $\sigma^{(d)}$를 "자기 깊이 $<d$인 정보집합에서는 $\beta$, 깊이 $\ge d$에서는 $\sigma_i$"로 둔다. 그러면 $\sigma^{(0)}=\sigma_i$, $\sigma^{(D+1)}=\beta$다.

$\sigma^{(d+1)}$과 $\sigma^{(d)}$는 깊이 $d$인 정보집합에서만 다르다.

- 그런 $I$의 이력 $h$에는 두 전략 모두 확률 $\pi_i^\beta(I)\pi_{-i}^\sigma(h)$로 이른다. 그 앞의 자기 행동은 모두 $\beta$이기 때문이다.
- $h$ 뒤로는 $\sigma^{(d+1)}$이 $\beta(I)$를 둔 뒤 $\sigma$를 따르고, $\sigma^{(d)}$는 처음부터 $\sigma$를 따른다.
- 깊이 $d$의 정보집합을 지나지 않는 종료 이력의 확률은 둘이 같다.

따라서 다음이 성립한다.

$$\begin{aligned}
u_i(\sigma^{(d+1)},\sigma_{-i})-u_i(\sigma^{(d)},\sigma_{-i})
&=\sum_{I:\,\text{깊이 }d}\pi_i^\beta(I)\sum_{h\in I}\pi^\sigma_{-i}(h)\Big(q(h,\beta(I))-\sum_b\sigma(I,b)q(h,b)\Big)\\
&=\sum_{I:\,\text{깊이 }d}\pi_i^\beta(I)\,r^\sigma(I,\beta(I)).
\end{aligned}$$

$d=0,\dots,D$에 대해 더하면 왼쪽이 $u_i(\beta,\sigma_{-i})-u_i(\sigma)$로 소거된다.
∎

**정리 3.4 (후회 분해 (Zinkevich 외, 2007)).**
완전 기억 게임에서 임의의 프로필 열에 대해 $R_i^T\le\sum_{I\in\mathcal{I}_i}\psi\big(R^T(I)\big)$다.

*증명.*
$\sum_tu_i(\sigma'_i,\sigma^t_{-i})$는 $\sigma'_i$의 실현 계획에 대해 선형이다. 그래서 최댓값은 다면체의 꼭짓점, 즉 어떤 순수 전략 $\beta$에서 얻어진다.
보조정리 3.3를 $t$마다 쓰고 더하면 다음을 얻는다.

$$\sum_t\big(u_i(\beta,\sigma^t_{-i})-u_i(\sigma^t)\big)=\sum_I\pi_i^\beta(I)R^T(I,\beta(I))\le\sum_I\psi(R^T(I)).$$

마지막 부등식은 $\pi_i^\beta(I)\in\{0,1\}$과 $R^T(I,\beta(I))\le\psi(R^T(I))$에서 나온다.
∎

**보조정리 3.5 (순간 후회의 크기).**
완전 기억 게임에서 $\sum_{h\in I}\pi^\sigma_{-i}(h)\le1$이다. 따라서 $\left\lvert r^\sigma(I,a)\right\rvert\le\Delta_{u,i}=\max_zu_i(z)-\min_zu_i(z)$다.

*증명.*
$I$에 이르는 자기 행동열을 늘 두는 전략 $\beta$를 생각한다. $h\in I$에 대해 $\pi_i^\beta(h)=1$이고, 사건 "이력 $h$에 도달"들은 서로 배반이다.
그래서 $\sum_{h\in I}\pi_{-i}^\sigma(h)=\sum_{h\in I}\pi^{(\beta,\sigma_{-i})}(h)\le1$이다.
$q(h,a)-\sum_b\sigma(I,b)q(h,b)$는 효용 값들의 볼록결합의 차이이므로 그 절댓값은 $\Delta_{u,i}$ 이하다.
∎

**정리 3.6 (후회 매칭 (Hart & Mas-Colell, 2000; Zinkevich 외, 2007)).**
식 (2)을 쓰면 모든 $I\in\mathcal{I}_i$에서 $\psi(R^T(I))\le\Delta_{u,i}\sqrt{\left\lvert A(I)\right\rvert\,T}$다.

*증명.*
$x^+=\max(x,0)$을 성분별로 쓴다.

- 성분별로 $(R^{t-1}+r^t)^+\le\big((R^{t-1})^++r^t\big)^+$이므로 다음이 성립한다.

$$\left\lVert (R^t)^+\right\rVert_2^2\le\left\lVert (R^{t-1})^++r^t\right\rVert_2^2=\left\lVert (R^{t-1})^+\right\rVert_2^2+2\langle(R^{t-1})^+,r^t\rangle+\left\lVert r^t\right\rVert_2^2.$$

- 가운데 항은 0이다. $(R^{t-1})^+$가 0이 아니면 $\sigma^t(I)\propto(R^{t-1})^+$이고, $\langle\sigma^t(I),r^t(I)\rangle=v_i(\sigma^t,I)-v_i(\sigma^t,I)=0$이기 때문이다.
- 귀납법과 보조정리 3.5로 $\left\lVert (R^T)^+\right\rVert_2^2\le\sum_t\left\lVert r^t\right\rVert_2^2\le\left\lvert A(I)\right\rvert\,T\,\Delta_{u,i}^2$를 얻는다.
- $\psi(R^T)\le\left\lVert (R^T)^+\right\rVert_\infty\le\left\lVert (R^T)^+\right\rVert_2$다.

∎

**따름정리 3.7 (CFR의 수렴 (Zinkevich 외, 2007)).**
$A_i=\max_{I\in\mathcal{I}_i}\left\lvert A(I)\right\rvert$로 두면 $R_i^T/T\le\Delta_{u,i}\left\lvert \mathcal{I}_i\right\rvert\sqrt{A_i}/\sqrt T$이다.
두 플레이어를 동시에 갱신하면 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^T)\le\frac{\Delta_{u,1}\left\lvert \mathcal{I}_1\right\rvert\sqrt{A_1}+\Delta_{u,2}\left\lvert \mathcal{I}_2\right\rvert\sqrt{A_2}}{\sqrt T}.$$

**RM+와 CFR+.**
RM+는 $Q^t=(Q^{t-1}+r^t)^+$를 쌓고 $\sigma^{t+1}\propto Q^t$를 둔다 (Tammelin, 2014). 다음 보조정리는 RM+의 후회가 어느 구간에서 재도 묶인다는 것이다. ASWISS가 분할 뒤에도 남은 bucket의 표를 초기화하지 않아도 되는 근거다.

**보조정리 3.8 (RM+의 구간 후회).**
$Q^0=0$이고 $\left\lVert r^u\right\rVert_\infty\le\Delta$이면, 임의의 구간 $(s,t]$에 대해 성분별로 $\sum_{u\in(s,t]}r^u\le Q^t$이다. 또 $\left\lVert Q^t\right\rVert_2^2\le\sum_{u\le t}\left\lVert r^u\right\rVert_2^2$이다.
따라서 $\psi\big(\sum_{u\in(s,t]}r^u\big)\le\Delta\sqrt{\left\lvert A\right\rvert\,t}$다.

*증명.*

- $Q^{u}\ge Q^{u-1}+r^u$이므로 $\sum_{u\in(s,t]}r^u\le Q^t-Q^s\le Q^t$다.
- $\sigma^u\propto Q^{u-1}$이므로 $\langle Q^{u-1},r^u\rangle=0$이다. 그래서 $\left\lVert Q^u\right\rVert_2^2\le\left\lVert Q^{u-1}+r^u\right\rVert_2^2=\left\lVert Q^{u-1}\right\rVert_2^2+\left\lVert r^u\right\rVert_2^2$이다.

∎

CFR+는 RM+에 교대 갱신과 반복 번호에 비례하는 평균 가중치를 더한 것이다 (Tammelin 외, 2015). 교대 갱신의 보장은 Burch 외 (Burch 외, 2019)가 정리했다.

### 3.4 MCCFR

MCCFR (Lanctot 외, 2009)은 종료 이력의 덮개 $\mathcal{Q}=\{Q_1,\dots,Q_r\}$($\bigcup_jQ_j=\mathcal{Z}$)를 정하고, 매 반복 블록 하나 $Q_j$를 확률 $q_j>0$으로 뽑는다.
$q(z)=\sum_{j:\,z\in Q_j}q_j$는 $z$가 뽑힌 블록에 들어 있을 확률이다. 뽑힌 블록만으로 다음 표본 반사실적 가치를 계산한다.

$$\tilde v_i(\sigma,I\mid j)=\sum_{(h,z)\in\mathcal{Z}_I,\ z\in Q_j}\frac{1}{q(z)}\,\pi^\sigma_{-i}(h)\,\pi^\sigma(h,z)\,u_i(z).$$

$\tilde v_i(\sigma,I,a\mid j)$도 $ha\sqsubseteq z$인 항만 남겨 같은 식으로 정의한다. 표본 후회는 $\tilde r(I,a)=\tilde v_i(\sigma,I,a\mid j)-\tilde v_i(\sigma,I\mid j)$이고, CFR과 같은 후회 매칭에 넣는다.

**보조정리 3.9 (불편성 (Lanctot 외, 2009), Lemma 1).**
$\mathbb{E}_{j\sim q}\big[\tilde v_i(\sigma,I\mid j)\big]=v_i(\sigma,I)$이고 $\mathbb{E}_{j\sim q}\big[\tilde v_i(\sigma,I,a\mid j)\big]=v_i(\sigma,I,a)$다. 따라서 $\mathbb{E}[\tilde r(I,a)]=r^\sigma(I,a)$다.

*증명.*
합의 순서를 바꾸면 다음을 얻는다.

$$\begin{aligned}
\mathbb{E}_j\big[\tilde v_i(\sigma,I\mid j)\big]&=\sum_jq_j\sum_{(h,z)\in\mathcal{Z}_I,\ z\in Q_j}\frac{\pi^\sigma_{-i}(h)\pi^\sigma(h,z)u_i(z)}{q(z)}\\
&=\sum_{(h,z)\in\mathcal{Z}_I}\frac{\pi^\sigma_{-i}(h)\pi^\sigma(h,z)u_i(z)}{q(z)}\sum_{j:\,z\in Q_j}q_j=v_i(\sigma,I).
\end{aligned}$$

행동 $a$의 경우도 같다. 후회는 두 추정량의 차이이므로 기댓값의 선형성으로 불편하다.
∎

**외부 샘플링.**
갱신할 플레이어(traverser) $i$를 정한다.

- 우연 노드에서는 $\sigma_c$로, 상대 노드에서는 $\sigma_{-i}$로 행동을 하나씩 뽑는다. 상대와 우연의 순수 전략 $\tau$를 하나 뽑는 것과 같다.
- 자기 노드에서는 모든 행동을 펼친다.
- 블록은 $Q_\tau=\{z:\ z\text{의 상대·우연 행동이 }\tau\text{와 같다}\}$이고, $q(z)=\pi^\sigma_{-i}(z)$다.

$\pi^\sigma_{-i}(z)=\pi^\sigma_{-i}(h)\pi^\sigma_{-i}(h,z)$이고 $\pi^\sigma(h,z)=\pi_i^\sigma(h,z)\pi_{-i}^\sigma(h,z)$이므로 중요도 가중이 약분된다.

$$\tilde v_i(\sigma,I\mid\tau)=\sum_{(h,z)\in\mathcal{Z}_I\cap Q_\tau}\pi_i^\sigma(h,z)\,u_i(z).$$

즉 뽑힌 행동을 따라가며 자기 행동만 자기 확률로 펼치면 된다.
완전 기억이면 같은 $I$의 두 이력은 처음 갈라지는 곳이 상대나 우연의 노드다. 그래서 한 순회에서 $I$의 이력은 많아야 하나만 $\tau$와 맞는다. 따라서 $\left\lvert \tilde r(I,a)\right\rvert\le\Delta_{u,i}$다.

**정리 3.10 (외부 샘플링 MCCFR의 수렴 속도 (Farina 외, 2020), 명제 2와 식 (6)).**
두 플레이어가 외부 샘플링 추정량을 받는 CFR을 동시에 돌린다고 하자. $\tilde R_i^T$를 플레이어 $i$의 CFR이 추정된 손실에 대해 갖는 후회라 하자.
임의의 $p\in(0,1)$에 대해 확률 $1-2p$ 이상으로 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^T)\le\frac{\tilde R_1^T+\tilde R_2^T}{T}+4\Delta\sqrt{\frac2T\log\frac1p}.$$

외부 샘플링의 표본 손실은 상대와 우연을 $\tau$로 고정한 게임의 효용이다. 그래서 정리 3.4와 정리 3.6이 그대로 적용되어 $\tilde R_i^T\le\Delta\left\lvert \mathcal{I}_i\right\rvert\sqrt{A_iT}$다. 따라서 확률 $1-2p$ 이상으로 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^T)\le\frac{\Delta\big(\left\lvert \mathcal{I}_1\right\rvert\sqrt{A_1}+\left\lvert \mathcal{I}_2\right\rvert\sqrt{A_2}\big)+4\Delta\sqrt{2\log(1/p)}}{\sqrt T}.$$

*증명 개요.*

- 고정된 비교 전략 $\hat x$에 대해 $d_t=(\ell^t-\tilde\ell^t)^\top(x^t-\hat x)$는 마팅게일 차분이다($\ell^t$는 참 손실, $\tilde\ell^t$는 불편 추정). 크기는 $2\Delta$ 이하다.
- Azuma–Hoeffding 부등식으로 확률 $1-p$ 이상에서 $\sum_td_t\le2\Delta\sqrt{2T\log(1/p)}$다.
- 그래서 참 후회는 $\tilde R_i^T+2\Delta\sqrt{2T\log(1/p)}$ 이하다. 두 플레이어에 합집합 상계를 쓰고 정리 3.1를 적용한다.
- 비교 전략 전체에 대한 처리 등 세부는 원문을 따른다.

∎

**비고 3.11.**
Lanctot 외 (Lanctot 외, 2009)의 원래 분석도 외부 샘플링 MCCFR의 평균 후회가 높은 확률로 $O(1/\sqrt T)$임을 보였다. 다만 실패 확률에 대한 의존이 $O(\sqrt{1/p})$였고, 정리 3.10는 이것을 $O(\sqrt{\log(1/p)})$로 줄인 것이다 (Farina 외, 2020).
상수는 원문마다 표기가 다르므로 인용할 때 원문의 정의를 따른다.

### 3.5 정보 추상화와 불완전 기억

정보 추상화 $\mathcal{B}=(\mathcal{B}_1,\mathcal{B}_2)$는 각 플레이어의 정보집합 $\mathcal{I}_i$를 bucket들로 나눈 분할이다.

- 한 bucket $B$의 멤버들은 같은 플레이어의 것이고, 행동집합이 같다($A_B$). 행동의 대응은 미리 정한다.
- bucket 전략 $\sigma(B)$를 멤버에 그대로 쓰는 것을 lifting이라 하고 $\sigma(I)=\sigma(B(I))$로 쓴다. lifting된 전략은 원 게임 $G$의 행동 전략이므로 $G$에서 그대로 평가할 수 있다.
- bucket 위에서 CFR이나 MCCFR을 돌리면, bucket의 후회 갱신은 멤버들의 반사실적 후회의 합이다.

$$r^\sigma(B,a)=\sum_{I\in B}r^\sigma(I,a).$$

서로 다른 자기 행동열을 거친 정보집합이 같은 bucket에 들면 그 추상화는 불완전 기억이다 (Waugh 외, 2009; Lanctot 외, 2012).
그중 한 경로가 같은 bucket을 두 번 지나는 경우를 *absent-mindedness*라 한다 (Piccione & Rubinstein, 1997).

- 불완전 기억 추상 게임에서는 행동 전략이 모든 혼합 전략을 표현하지 못한다. 그래서 추상 게임에서 CFR의 보장이 일반적으로 성립하지 않는다 (Lanctot 외, 2012).
- 이 논문은 추상 게임의 균형 대신 원 게임 $G$의 NashConv를 직접 묶는다. 그래서 lifting된 전략이 $G$의 행동 전략이라는 사실만 쓴다(5절).

### 3.6 CFR+IRA

CFR+IRA (Čermák 외, 2020)는 2인 제로섬 완전 기억 게임 $G$에서 거친 불완전 기억 추상화로 시작한다. 학습 중에 추상화를 세분하며 $G$의 균형으로 수렴한다.
원문 4.3절의 구성은 다음과 같다.

**초기 추상화(원문 4.1.1절).**
플레이어 $i$의 정보집합 중 자기 행동열의 길이와 행동 수가 같은 것끼리 하나로 묶는다. 행동은 게임 기술의 순서대로 대응시킨다.

**학습.**
추상화되지 않은 트리 전체를 매 반복 순회하며 CFR+를 돈다(교대 갱신, 정해진 지연 뒤부터 평균). 후회와 평균 전략은 추상 정보집합 단위로 저장한다.
멈춤 조건은 원 게임 최선 응답으로 잰 평균 전략의 exploitability가 $\epsilon$ 이하가 되는 것이다.

**후회 상한 갱신(원문 4.3.1절, 수렴 보장용).**

- 반복 열 $T_1<T_2<\cdots$를 $T_{j+1}-T_j=2^j$로 둔다.
- 각 $T_j$에서, 아직 추상 정보집합에 묶여 있는 원 정보집합 가운데 $k_b$개를 균등하게 뽑는다($\hat\mathcal{I}_b$).
- 구간 $[T_j,T_{j+1})$ 동안 그 원 정보집합들의 후회를 따로 추적한다.
- 어떤 반복 $T'$에서 다음 조건이 성립하면, 그 $I$를 추상 정보집합에서 떼어 내고 후회와 평균을 새로 시작한다.

$$R^{T'}_{T_j,\mathrm{imm}}(I)=\frac{\max_{a}\sum_{t=T_j}^{T'}\big(v_i(b^t_{I\to a},I)-v_i(b^t,I)\big)}{T'-T_j}>L_I^{T_j}(T'),\qquad L_I^{T_j}(t)\le\frac{\Delta_I\sqrt{\left\lvert A(I)\right\rvert}}{\sqrt{t-T_j}}$$

즉 감시한 정보집합의 구간 평균 후회가, 그 정보집합이 따로 있었다면 후회 매칭이 보장했을 속도보다 느리게 줄면 떼어 낸다.

**휴리스틱 갱신(원문 4.3.2절, 실전 속도용).**

- 매 반복, 행동하는 플레이어의 정보집합 $k_h$개를 뽑아 *그 반복 하나의* 후회 $r^h$를 기록한다.
- 각 정보집합에 $\gamma_I=\{a:\ r^h(a)\ge\max_{a'}r^h(a')-1/(5\sqrt t)\}$를 붙이고, $\gamma_I$가 같은 것끼리 나눈다.
- 가장 큰 묶음이 추적하지 않은 멤버를 데려간다. 새 묶음들의 후회는 0에서 시작한다.

**정리 3.12 (CFR+IRA의 수렴 (Čermák 외, 2020), 정리 2).**
다음 식(원문 식 (26))의 $\alpha(\delta)$를 쓴다. **\[확인: 원문 PDF에서 근호와 지수의 배치를 다시 확인할 것\]**

$$\alpha(\delta)=\sqrt[3]{\,2^{\sqrt[\left\lvert \mathcal{I}\right\rvert]{\left\lvert \mathcal{I}\right\rvert!\,\log_{1-(k_b/\left\lvert \mathcal{I}\right\rvert)^{\left\lvert \mathcal{I}\right\rvert}}(\delta)}\;+\;\left\lvert \mathcal{I}\right\rvert+3}}\;A_{\max}$$

그러면 모든 $T\ge(\alpha(\delta)+1)^3$에서 확률 $1-\delta$로 다음이 성립한다.

$$\frac{R_i^T}{T}\le\Delta\left\lvert \mathcal{I}_i\right\rvert\frac{\sqrt{A_{\max}}}{\sqrt T}+\Delta\left\lvert \mathcal{I}_i\right\rvert\frac{\alpha(\delta)}{\sqrt[3]T}+\Delta\left\lvert \mathcal{I}_i\right\rvert\frac1T\ \in\ O\!\left(\frac1{\sqrt[3]T}\right).$$

**증명의 구조(원문 보조정리 2–6).**

- 처음 $T_{\lim}$ 반복 동안 추상화가 수렴을 막는다고 보고, 그 동안의 후회를 정보집합마다 최악 $\Delta T_{\lim}$로 센다.
- 그 뒤는 RM+의 $\Delta\sqrt{\left\lvert A(I)\right\rvert T}$를 쓴다. 그래서 $R_i^T/T\le B_i(T)=\frac1T\big(\Delta\left\lvert \mathcal{I}_i\right\rvert\sqrt{A_{\max}}\sqrt T+\Delta\left\lvert \mathcal{I}_i\right\rvert T_{\lim}\big)$이다(원문 보조정리 3).
- 이 상한이 성립할 확률의 하한은 최악의 경우로 계산한다. 모든 원 정보집합을 정해진 순서로 하나씩 떼어 내야 하는 경우다.

$$p_r(T,\bar R_i)=1-\Big(1-\big(k_b/\left\lvert \mathcal{I}\right\rvert\big)^{\left\lvert \mathcal{I}\right\rvert}\Big)^{\binom{\left\lvert \tau^T_{\bar R_i}\right\rvert}{\left\lvert \mathcal{I}\right\rvert}}$$

  여기서 $\tau^T_{\bar R_i}$는 감시 구간 중 충분히 긴 것들의 열이다.

- $T_{\lim}=\lceil\alpha(\delta)T^{2/3}\rceil$로 두면 이 확률이 $1-\delta$ 이상이 된다. 그 결과가 $O(T^{-1/3})$다.

**휴리스틱만으로는 수렴하지 않는다(원문의 반례).**

- 한 추상 정보집합에 두 원 정보집합 $I',I''$가 묶여 있고, 행동 가치가 두 표 사이를 오간다고 하자(원문 표 1).
- 그러면 매 반복 두 정보집합의 최선 행동 집합 $\gamma$가 같아서 휴리스틱은 한 번도 나누지 않는다.
- 그러나 추상 정보집합은 균등 전략(기댓값 2.75)으로 수렴하는 반면, 따로 풀면 기댓값 5인 서로 다른 순수 전략이 균형이다.
- 한 반복의 순간 후회로 묶는 규칙은 진동하는 가치에서 이런 실패를 겪는다.

ASWISS의 라벨은 epoch 동안의 *누적* 후회의 argmax다. 이 예에서 누적 후회는 두 정보집합에서 서로 다른 행동을 가리킨다(값의 크기가 1과 10으로 다르므로). 그래서 나뉜다.
일반적으로는 누적 라벨이 일치하면 숨은 후회가 0이다(보조정리 5.4).

**한계.**
매 반복 추상화되지 않은 트리 전체를 순회한다. 정리 3.12의 $\alpha(\delta)$는 정보집합 수에 지수적이라, 상한이 의미를 갖는 $T$가 현실적인 범위를 크게 넘는다.
원문 6.1절은 표본 기법과의 결합을 후속 과제로 남겼다.

## 4. 순수 ASWISS

이 절에서 정의하는 순수 ASWISS는 세 가지만 쓴다.

- 게임 구조만으로 만든 시작 분할
- 반복 전략들의 누적 반사실적 후회로 붙인 라벨
- 분할 사건의 계보

사람이 고른 특징이나 정보집합별 표는 쓰지 않는다.

### 4.1 시작 분할

lifting이 정의되려면 같은 bucket의 정보집합이 같은 플레이어의 것이고 행동집합이 같아야 한다. 수렴 정리(5절)에 필요한 조건은 이것뿐이다.
분할이 정보집합 하나까지 내려갈 수 있으므로, 이 조건을 만족하는 어떤 분할에서 시작해도 증명이 성립한다.

- 실험의 시작 분할 $\mathcal{P}_0$는 *시점 분할*이다.
  - 기준: (플레이어, 베팅 라운드, 그 라운드의 베팅 진행, 합법 행동집합)
  - 정보집합 51,114개의 mini-stud에서 40개다.
  - 한 경로가 같은 칸을 두 번 지나지 않는다. 즉 absent-mindedness가 없다.
- 더 깊은 구조까지 같은 것끼리 묶으면 필요한 분할이 줄 것으로 기대한다. 이것은 수렴의 조건이 아니라 효율의 문제다. 일반 게임에서 이런 시작 분할을 만드는 방법은 8절에서 다룬다.

### 4.2 라벨: 반복 전략들의 누적 반사실적 후회

판정 구간 $W=(s,t]$($\left\lvert W\right\rvert=t-s$)의 반복 전략 $\sigma^u$($u\in W$)에 대한 정보집합 $I$의 누적 반사실적 후회를 다음과 같이 쓴다.

$$R_I^W(a)=\sum_{u\in W}r^{\sigma^u}(I,a).$$

CFR의 후회 표가 매 반복 $r^{\sigma^u}$를 더하므로, $R_I^W$의 방향은 *그 구간 동안 학습이 실제로 움직인 방향*이다.

행동집합에 "없음" $\varnothing$을 더하고 $x(\varnothing)=0$으로 둔다. 그러면 $\psi(x)=\max_{\ell\in A\cup\{\varnothing\}}x(\ell)$다.

**라벨 규칙.**
추정치 $\tilde R_I^W$(4.5절)에서 $\hat a_I=\mathop{\mathrm{arg\,max}}_a\tilde R_I^W(a)$를 구한다.

$$\ell(I)=\begin{cases}\hat a_I,&\tilde R_I^W(\hat a_I)>\theta_I^W,\\ \varnothing,&\text{그 밖(구간 동안 방문하지 않은 경우 포함)}.\end{cases}$$

역치는 다음 크기 조건만 만족하면 된다.

$$\tag{T}
0\le\theta_I^W\le\Delta_I\,\kappa(W),\qquad\kappa(W)=o(\left\lvert W\right\rvert).$$

여기서 $\Delta_I$는 $I$ 아래의 효용 폭이다. 실험에 쓴 역치들은 다음과 같다.

- 정규 검정: $\theta=z\sqrt{\sum_u\tilde r_u(I,\hat a)^2}\le z\Delta_I\sqrt{\left\lvert W\right\rvert}$
- 정확한 계산: 반사실 도달확률 합당 $\mathrm{tol}$

정규 검정은 사용자 원안의 분할 기준 $R(I,a)/\sqrt{\sum r^2(I,a)}$와 같은 꼴이다.

### 4.3 분할 규칙

판정 시점에, 멤버들의 라벨이 두 종류 이상인 bucket을 라벨별로 나눈다.

- 자식은 라벨마다 하나다.
- 자식의 후회 표와 평균 표는 0에서 시작한다. 나뉘지 않은 bucket은 표를 그대로 이어 간다.
- 병합은 하지 않는다. 그래서 분할 열은 언제나 세분(refinement)이다.

### 4.4 계보 라우팅: 정보집합별 표 없이 bucket 정하기

ASWISS가 들고 있는 상태는 셋뿐이다.

1.  시작 분할 $\mathcal{P}_0$
2.  분할 사건 $E_1,E_2,\dots$. 사건 $E_k$는 나뉜 bucket마다 자식 라벨들과, 라벨 계산에 쓸 전략 집합 $\mathcal{S}_k$를 담는다.
3.  현재 bucket들의 후회 표와 평균 표

정보집합 $I$의 $k$번째 분할 뒤 bucket은 다음 재귀로 정한다.

> **function** `Route`$(I,k)$
>
> 1. **if** $k=0$ **then return** $\mathcal{P}_0(I)$
> 2. $B\gets$ `Route`$(I,k-1)$
> 3. **if** $E_k$에서 $B$가 나뉘지 않았다 **then return** $B$
> 4. $\ell\gets$ $\mathcal{S}_k$로 계산한 $I$의 라벨(자식 라벨 중에서 고름)
> 5. **return** $E_k$에서 라벨이 $\ell$인 $B$의 자식

- 정보집합의 특징이 아니라 분할 시점의 전략이 그 정보집합에 주는 업데이트 방향이 bucket을 정한다.
- 그래서 학습 중 방문한 적이 없는 정보집합도 bucket을 받는다.
- 대가는 비용이다. $I$의 라벨을 정확히 구하려면 다음 결정들의 분할 $k$ 시점 bucket이 필요하다.
  - $I$에 이르는 상대 결정(반사실 도달확률)
  - $I$ 아래의 모든 결정(행동 가치)

  이 bucket들도 `Route`로 다시 찾아야 한다.

- 정보집합별 표 없이는 비용이 분할 단계마다 곱으로 늘어난다(6.5절).

### 4.5 라벨 추정: 추적과 재생

이론은 $\tilde R_I^W$를 얻는 두 방법을 다룬다.

- **(가) 추적.** 학습 중 외부 샘플링 표본 후회 $\tilde r_u(I)$를 판정 구간 동안 모든 멤버에 대해 쌓는다. 메모리가 정보집합 수에 비례하므로 증명용 기준판이다(정리 5.9).
- **(나) 저장 전략 재생.** epoch $W_k$의 반복 중 $K_k$개($\tau_{k,1},\dots,\tau_{k,K_k}$)를 균등하게 뽑아 그 반복의 bucket 전략표만 저장한다.
  epoch 끝에 저장한 전략마다 원 게임을 정확히 순회해 $r^{\sigma_{\tau}}(I)$를 구하고 다음과 같이 둔다.

$$\hat R_I^{(k)}=\frac{L_k}{K_k}\sum_{j=1}^{K_k}r^{\sigma_{\tau_{k,j}}}(I).$$

  학습 중에는 정보집합에 대해 아무것도 저장하지 않으며, 방문하지 않은 정보집합도 실제 후회로 라벨을 받는다(정리 5.15).
  이것이 사용자 원안의 "순간 전략들의 합을 통한 $r(I,a)$"이다.

### 4.6 판정 시점

- **방식 E (epoch).** 구간 $W_k=(t_k,t_{k+1}]$, $L_k=t_{k+1}-t_k$, $t_{k+1}\le\gamma L_k$이다. epoch마다 두 배로 늘리면 $\gamma=2$다($t_k=1000\cdot2^k$). 판정은 epoch 끝마다 한다.
- **방식 G (수렴 조건부).** 마지막 분할 시점 $s$부터를 판정 구간으로 둔다. $g$반복마다 확인해, $\left\lvert W\right\rvert\ge m$이고 추상 게임의 표본 후회 $\tilde C^W/\left\lvert W\right\rvert\le\varepsilon_g$일 때만 판정한다.

$$\tilde C^W=\sum_B\psi(\tilde R_B^W)$$

  사용자 원안의 분할 시점 "전체 $\sum$bucket.regret $<\varepsilon$"에 해당한다.

> **알고리즘 1** 순수 ASWISS (방식 E, 재생 라벨)
>
> **입력:** 시작 분할 $\mathcal{P}_0$, 첫 epoch 길이 $t_0$, 저장 개수 $K_k$
>
> 1. 분할 사건 목록 $\mathcal{E}\gets\emptyset$, $k\gets0$, 모든 bucket의 표 $Q_B\gets0$, 평균 표 $S_B\gets0$
> 2. **for** $t=1,2,\dots$ **do**
>    - **for** traverser $i\in\{1,2\}$ **do**
>      - 외부 샘플링으로 순회한다. 정보집합 $I$에 닿으면 $B\gets$ `Route`$(I,k)$로 찾는다.
>      - $\sigma(I)\propto Q_B^+$로 행동하고, 표본 후회를 $Q_B$에, 평균을 $S_B$에 쌓는다.
>    - **if** $t$가 epoch $k$에서 뽑아 둔 $K_k$개의 반복 중 하나 **then** 현재 bucket 전략표를 $\mathcal{S}_k$에 저장
>    - **if** $t=t_{k+1}$ (epoch 끝) **then**
>      - **for** 각 bucket $B$ **do**
>        - 멤버의 라벨 $\ell(I)$를 $\mathcal{S}_k$의 재생으로 구한다(4.5절)
>        - **if** 라벨이 두 종류 이상 **then** $B$를 라벨별로 나누고 자식 표를 0에서 시작한다
>      - 사건 $E_{k+1}$(나뉜 bucket, 자식 라벨, $\mathcal{S}_k$)을 $\mathcal{E}$에 더한다. $k\gets k+1$

### 4.7 이론판과 구현의 차이

5절의 정리는 이론판을 다룬다. 실험의 구현(`pure_aswiss.cpp`)과 다른 점을 표 1에 적는다.

**표 1.** 이론판과 실험 구현의 차이

| 항목           | 이론판                                           | 구현                                                     |
|:---------------|:-------------------------------------------------|:---------------------------------------------------------|
| bucket 학습    | RM+, 자식은 0에서                                | RM(음수 후회를 자르지 않음), 자식은 0에서                |
| 갱신 순서      | 두 플레이어 동시 갱신                            | 교대 순회                                                |
| 출력 평균      | 판정 구간의 균등 혼합(정보집합별 실현 계획 평균) | bucket 단위 평균 표(실행 전체)                           |
| 분할 판정 대상 | bucket의 모든 멤버                               | bucket마다 그 epoch에 방문한 정보집합 16개의 저수지 표본 |
| 재생 개수      | $K_k/\log k\to\infty$                            | $K_k=\min(64,\,8(k+1))$                                  |

- 구현의 bucket 평균 표는 멤버들의 자기 도달확률 궤적이 비례하지 않으면 정보집합별 평균과 다르다. 그래서 정리의 대상이 아니다.
- 정보집합별 평균을 저장하지 않는 순수 구현에서는 정리의 대상인 평균을 잴 수 없다. 대신 저장한 반복 전략 하나를 판마다 무작위로 고르는 혼합 정책이 같은 평균과 실현 동치다. 이것을 내놓는 것은 8절의 과제로 둔다.

### 4.8 비용

- **학습 메모리.**
  - bucket 표와 분할 사건을 둔다.
  - 사건마다 bucket 전략표 $K_k$개를 둔다.
  - bucket마다 구성원 저수지를 둔다.

  정보집합 수에 비례하는 상태는 없다.

- **계산.**
  - 학습 순회는 외부 샘플링 MCCFR과 같다. 다만 정보집합마다 `Route`를 부른다.
  - 표 없이 `Route`를 부르면 그 비용이 분할 단계 $k$에 대해 곱으로 늘어난다. 같은 (정보집합, 단계)의 결과를 기억하는 메모를 무제한으로 두면 결국 정보집합별 표와 같아진다.
- **정책의 크기.** 배포하는 정책은 bucket 전략표와 분할 계보다.

## 5. 이론 결과

이 절은 결과만 적는다. 증명은 부록 A에 있다.

### 5.1 가정과 진단량

**가정 5.1.**

- **(S1) 동시 갱신.** 반복 $t$에서 두 traverser의 순회는 같은 프로필 $\sigma_t$를 쓰고, bucket 표는 그 반복의 순회가 모두 끝난 뒤 갱신한다.
- **(S2) 불편 추정.** 방문하지 않으면 $\tilde r_t(I,a)=0$이다. $\mathcal{F}_{t-1}$을 반복 $t-1$까지의 모든 정보라 하면 $\mathbb{E}[\tilde r_t(I,a)\mid\mathcal{F}_{t-1}]=r_t(I,a)$이고 $\left\lvert \tilde r_t(I,a)\right\rvert\le\Delta_I$다.
  외부 샘플링은 이를 만족한다(보조정리 3.9와 3.4절).
- **(S3) bucket 표본 후회의 크기.** $\tilde r_t(B)=\sum_{I\in B}\tilde r_t(I)$로 둔다. 한 순회에서 함께 방문될 수 있는 $B$의 멤버들에 대해 $\Delta_I$의 합의 최댓값을 $D_B$라 한다($D_B\le\sum_{I\in B}\Delta_I$). 그러면 $\left\lVert \tilde r_t(B)\right\rVert_\infty\le D_B$다.

학습기는 RM+이고(bucket이 생길 때 $Q_B=0$), 판정과 분할은 4절의 규칙을 따른다.
$\bar\sigma^W$는 구간 $W$ 동안 lifting된 전략들의 균등 평균(식 (1))이다.
$N=\left\lvert \mathcal{I}\right\rvert$, $N_A=\sum_I\left\lvert A(I)\right\rvert$로 둔다.

**정의 5.2 (bucket 후회와 숨은 후회).**

$$C^W=\sum_B\psi(R_B^W),\qquad H_B^W=\sum_{I\in B}\psi(R_I^W)-\psi(R_B^W)\ \ge0,\qquad H^W=\sum_BH_B^W.$$

- $R_B^W=\sum_{I\in B}R_I^W$다.
- $C$는 bucket이 스스로 느끼는 후회다. 추상 게임을 학습하면 줄어든다.
- $H$는 bucket 안에서 멤버끼리 상쇄되어 bucket이 느끼지 못하는 후회(숨은 후회)다. 추상 게임 학습으로는 줄지 않고, 분할해야만 줄어든다.

### 5.2 보조정리

**보조정리 5.3 (NashConv 분해).**
(S1) 아래에서 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^W)\le\frac1{\left\lvert W\right\rvert}\sum_{I\in\mathcal{I}}\psi(R_I^W)=\frac{C^W+H^W}{\left\lvert W\right\rvert}.$$

더 정확히는, $\beta_i$가 $\bar\sigma^W_{-i}$에 대한 원 게임의 순수 최선 응답일 때 등식이 된다.

$$\mathrm{NashConv}(\bar\sigma^W)=\frac1{\left\lvert W\right\rvert}\sum_i\sum_{I\in\mathcal{I}_i}\pi_i^{\beta_i}(I)\,R_I^W\big(\beta_i(I)\big).$$

**보조정리 5.4 (라벨 항등식).**
임의의 bucket $B$와 구간 $W$에 대해 다음이 성립한다.

$$H_B^W=\min_{\ell\in A_B\cup\{\varnothing\}}\sum_{I\in B}e_I(\ell),\qquad e_I(\ell)=\psi(R_I^W)-R_I^W(\ell)\ \ge0.$$

특히 $H_B^W=0$일 필요충분조건은 모든 멤버에서 동시에 $\psi$를 달성하는 공통 라벨이 있는 것이다.

$e_I(\ell)$은 멤버 $I$에게 라벨 $\ell$을 강요할 때 잃는 후회다. 그래서 라벨이 갈린 bucket을 라벨별로 나누는 것이 정확히 숨은 후회를 없애는 일이다.

**보조정리 5.5 (RM+의 구간 상한; 표본 후회).**
bucket $B$가 시점 $\tau_B$에 생겼고 구간 $W=(s,t]$에서 $s\ge\tau_B$라 하자. 그러면 다음이 성립한다.

$$\psi(\tilde R_B^W)\le\left\lVert Q_{B,t}\right\rVert_\infty\le D_B\sqrt{\left\lvert A_B\right\rvert(t-\tau_B)}.$$

**보조정리 5.6 (시간 균일 집중).**
다음과 같이 둔다.

$$\varepsilon_I(t)=4\Delta_I\sqrt{t\ln\frac{\pi^2N_A(1+\log_2t)^2}{3\delta}}.$$

사건 $\mathcal{E}_\delta$ = 모든 $I,a,t\ge1$에서 $\left\lvert \sum_{u\le t}(\tilde r_u(I,a)-r_u(I,a))\right\rvert\le\varepsilon_I(t)$의 확률은 $1-\delta$ 이상이다.
$\mathcal{E}_\delta$ 위에서는 임의의 구간 $W=(s,t]$에 대해 $\left\lVert R_I^W-\tilde R_I^W\right\rVert_\infty\le2\varepsilon_I(t)$다. 구간의 끝이 정지시간이어도 성립한다.

**보조정리 5.7 (라벨이 같은 bucket의 숨은 후회).**
$\mathcal{E}_\delta$ 위에서, 구간 $W=(s,t]$의 판정에서 $B$의 모든 멤버가 같은 라벨을 받았다면 다음이 성립한다.

$$H_B^W\le\sum_{I\in B}\big(\theta_I^W+4\varepsilon_I(t)\big).$$

**보조정리 5.8 (분할 유한성).**
분할은 모든 표본 경로에서 많아야 $N-\left\lvert \mathcal{P}_0\right\rvert$번이다.

### 5.3 정리

**정리 5.9 (방식 E의 수렴).**
(S1)–(S3), (T), 방식 E, 추적 라벨을 가정한다. $\mathcal{E}_\delta$ 위에서(확률 $1-\delta$ 이상), 끝에서 분할이 일어나지 않은 *모든* epoch $k$에 대해 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^{(k)})\le\frac1{L_k}\Big[\sum_{B\in\mathcal{P}_k}D_B\sqrt{\left\lvert A_B\right\rvert\,t_{k+1}}+\sum_{I\in\mathcal{I}}\big(\theta_I^{(k)}+6\varepsilon_I(t_{k+1})\big)\Big].$$

$k^*$를 마지막으로 분할이 일어난 epoch라 하면, 보조정리 5.8에 의해 $k^*<\infty$이고 $k>k^*$인 모든 epoch에 위 식이 적용된다.
정규 검정($\theta\le z\Delta_I\sqrt{L_k}$)과 두 배 epoch($\gamma=2$)를 쓰면 다음이 된다.

$$\mathrm{NashConv}(\bar\sigma^{(k)})=O\!\left(\sqrt{\frac{\log(kN_A/\delta)}{L_k}}\right).$$

또한 거의 확실하게 $\mathrm{NashConv}(\bar\sigma^{(k)})\to0$이다.

**따름정리 5.10 (실행 전체의 평균).**
$K>k^*$이고 $T=t_{K+1}$일 때 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^{(0,T]})=O\!\left(\frac{t_{k^*+1}}{T}+\sqrt{\frac{\log}{T}}\right).$$

**정리 5.11 (방식 G의 수렴).**
(S1)–(S3), (T), 방식 G를 가정한다. 임의의 $\varepsilon_g>0$, $g,m\ge1$에 대해 다음이 성립한다.

1.  판정은 끝없이 미뤄지지 않는다.
2.  마지막 분할 시점 $s^*$가 존재한다.
3.  $\mathcal{E}_\delta$ 위에서, $s^*$ 이후 분할 없이 판정을 통과한 모든 시점 $t$에 대해 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^{(s^*,t]})\le\frac{\sum_BD_B\sqrt{\left\lvert A_B\right\rvert t}+\sum_I(\theta_I+6\varepsilon_I(t))}{t-s^*}\to0.$$

**정리 5.12 ($\varepsilon$ 정지 인증).**
(S1)–(S3), 방식 G를 가정한다. $\mathcal{E}_\delta$ 위에서, 판정 구간 $W=(s,t]$의 판정 통과에서 어떤 bucket도 나뉘지 않았다면 다음이 성립한다.

$$\mathrm{NashConv}(\bar\sigma^W)\le\frac{\tilde C^W}{\left\lvert W\right\rvert}+\frac1{\left\lvert W\right\rvert}\sum_I\big(\theta_I^W+6\varepsilon_I(t)\big)\le\varepsilon_g+\frac{\sum_I\theta_I^W}{\left\lvert W\right\rvert}+\frac{6\sum_I\varepsilon_I(t)}{\left\lvert W\right\rvert}.$$

일부 bucket $S$가 나뉜 통과에서는 $\frac1{\left\lvert W\right\rvert}\sum_{B\in S}\min_\ell\sum_{I\in B}e_I(\ell)$이 더해진다.

이 인증은 마지막 분할 시점을 몰라도 쓸 수 있는 정지 규칙이다. 다만 모든 정보집합에 대한 합이 들어가므로, mini-stud 크기에서도 수치적으로는 매우 느슨하다(부록 A의 비고).

**재생 라벨.**
순수 ASWISS의 라벨 계산(4.5절 (나))에 대한 결과다. 추정 오차가 있어도 참으로 일치하는 bucket을 나누지 않으려면 *여유 라벨 집합*을 쓴다.

**보조정리 5.13 (여유 라벨 집합).**
$\left\lVert \hat x_I-x_I\right\rVert_\infty\le\eta_I$인 추정치에 대해 여유 라벨 집합을 다음과 같이 둔다.

$$\hat M(I)=\{\ell:\hat x_I(\ell)\ge\max_{\ell'}\hat x_I(\ell')-2\eta_I\}$$

$\bigcap_{I\in B}\hat M(I)=\emptyset$일 때만 나누고, 나눌 때는 각 멤버를 $\hat M(I)$의 대표 라벨(고정 순서의 첫 원소)별로 모은다. 그러면 다음이 성립한다.

1.  참 최대 라벨은 늘 $\hat M(I)$에 들어 있다.
2.  $\ell\in\hat M(I)$이면 $e_I(\ell)\le4\eta_I$다.
3.  참으로 일치하는 bucket은 나뉘지 않는다.
4.  나뉘지 않은 bucket은 $H_B\le4\sum_{I\in B}\eta_I$다.
5.  나뉠 때 각 멤버가 들어가는 자식의 라벨 $\hat c$에 대해 $e_I(\hat c)\le4\eta_I$다.

**보조정리 5.14 (재생 추정의 집중).**
다음과 같이 둔다.

$$\eta_I^{(k)}=L_k\Delta_I\sqrt{\frac{2}{K_k}\ln\frac{\pi^2N_A(k+1)^2}{3\delta}}.$$

사건 $\mathcal{E}'_\delta$ = 모든 $k,I,a$에서 $\left\lvert \hat R_I^{(k)}(a)-R_I^{(k)}(a)\right\rvert\le\eta_I^{(k)}$의 확률은 $1-\delta$ 이상이다.

**정리 5.15 (저장 전략 재생의 수렴).**
(S1)–(S3), 방식 E, 재생 라벨(보조정리 5.13, $\eta=\eta^{(k)}$)을 가정한다. $\mathcal{E}_\delta\cap\mathcal{E}'_\delta$ 위에서(확률 $1-2\delta$ 이상) 다음이 성립한다.

1.  참으로 일치하는 bucket은 나뉘지 않는다.
2.  분할은 많아야 $N-\left\lvert \mathcal{P}_0\right\rvert$번이다.
3.  끝에서 분할이 없었던 모든 epoch $k$에 대해 다음이 성립한다.

$$\begin{aligned}
      \mathrm{NashConv}(\bar\sigma^{(k)})\le{}&\frac1{L_k}\Big[\sum_BD_B\sqrt{\left\lvert A_B\right\rvert\,t_{k+1}}+2\sum_I\varepsilon_I(t_{k+1})\Big]\\
      &+4\sum_I\Delta_I\sqrt{\frac2{K_k}\ln\frac{\pi^2N_A(k+1)^2}{3\delta}}.
      
    \end{aligned}$$

따라서 $K_k/\log k\to\infty$이면 마지막 분할 이후 NashConv가 0으로 간다. $K_k\propto L_k$이면 정리 5.9와 같은 차수다.

### 5.4 보장하지 않는 것

- **마지막 분할 시점 $t^*$의 상한이 없다.** 속도 보장은 $t^*$ 이후에만 있다.
  CFR+IRA는 $T_{\lim}$을 묶지만, 모든 정보집합을 하나씩 떼어 내야 하는 최악의 경우로 묶기 때문에 그 크기가 정보집합 수에 지수적이다(정리 3.12).
- **최종 bucket 수에 보장이 없다.** 최악의 경우 정보집합마다 bucket 하나까지 간다. 그래도 수렴은 유지되지만 압축 이득은 사라진다.
- **보장 대상은 정보집합별 평균이다.** 구현의 bucket 평균 표는 대상이 아니다(4.7절).
- **현재 전략(마지막 반복)에는 보장이 없다.**
- **충분조건일 뿐이다.** 보조정리 5.3의 부등식은 최선 응답이 가지 않는 정보집합의 후회도 센다. 그래서 라벨 규칙이 그런 정보집합 때문에 불필요하게 나눌 수 있다. 정확성의 손해는 없고 효율의 손해만 있다.
- **실행 사이의 단조성이 없다.** 분할은 이후의 궤적을 바꾼다. 그래서 "bucket이 많을수록 같은 반복에서 더 좋다"가 성립하지 않는다(추상화 병리 현상 (Waugh 외, 2009)).
  1단 행렬 게임에서 라벨 분할 뒤 NashConv가 0.238에서 2.5로 오르는 예를 7.4절에 적었다.

## 6. 실험

정확한 exploitability를 잴 수 있는 작은 게임에서 실험했다. 질문은 셋이다.

1.  누적 후회 라벨로 나누는 규칙이 CFR+IRA, 고정 추상화와 비교해 어떤가?
2.  표본만 쓰는 순수 ASWISS의 규칙은 어디까지 내려가는가?
3.  정보집합별 표 없이 계보를 따르는 비용은 얼마인가?

모든 수치는 seed 1개다(CFR+IRA와 표본 실행 일부는 여러 seed이고 그렇게 적었다).
exploitability는 정확한 최선 응답으로 잰 $\mathrm{NashConv}/2$다.

### 6.1 설정

**게임.**

- Kuhn 포커(정보집합 12개), Leduc 포커(288개)
- mini-stud: 세븐 포커의 규칙을 유지하고 크기를 줄인 2인 게임이다(정보집합 51,114개, 노드 644,579개).
  - 무늬 없는 Q, K, A 각 4장, 처음 3장을 받고 버림과 공개는 고정 휴리스틱으로 한다.
  - 베팅 라운드는 둘이다. 5구(공개 카드)와 7구(비공개 카드)이고, 플레이어당 레이즈 상한은 라운드마다 1이다.
  - 행동은 다이, 체크/콜, 삥, 따당, 하프다. 보상은 ante 단위다.

**비교군.**

- 추상화 없는 CFR+, DCFR, PCFR+(전체 순회), 추상화 없는 외부 샘플링 MCCFR
- CFR+IRA: 원문 알고리즘 6–10을 따르고, 원문에 보고된 설정 B10H90과 B100H900을 썼다.
- 카드 특징 k-means 고정 추상화: 시점 칸마다 bucket 수를 비교 대상에 맞췄다.
- 시작 분할 $\mathcal{P}_0$를 고정한 것
- MMD (Sokota 외, 2023), ESCHER (McAleer 외, 2023): 원문 설정, 같은 C++ 신경망 구현

**ASWISS의 판.**

- **추적판**: 정리 5.9의 규칙이다. 정보집합마다 구간 누적 후회를 쌓아 라벨을 붙인다. 전체 순회판과 외부 샘플링판이 있다.
- **순수판**: 정리 5.15의 규칙이다. 정보집합에 대해 아무것도 저장하지 않는다.
  - 저장한 반복 전략 $K_k$개를 재생해 라벨을 구한다.
  - 분할 판정은 bucket마다 저수지 표본 16개로 한다.
  - bucket은 계보를 따라 정한다(4.7절).

### 6.2 전체 순회: CFR+IRA와의 비교

모든 비교군이 같은 전체 순회 학습기(교대 갱신, 선형 평균)를 쓴다.
ASWISS 추적판은 epoch(10, 30, 70, … 반복) 끝마다 정확한 구간 누적 후회로 라벨을 붙인다. 반사실 도달확률 합당 후회가 $10^{-3}$ 이하면 "없음"이다.
표 2와 표 3은 평균 정책(배포 가능한 bucket 평균)의 exploitability다.

**표 2.** Leduc(정보집합 288개). CFR+IRA는 seed 10개의 중앙값이다.

| 비교군                              | 최종 bucket |   1,270 |  20,470 |  81,910 | 655,350 |
|:------------------------------------|------------:|--------:|--------:|--------:|--------:|
| CFR+ (추상화 없음)                  |         288 | 1.69e-4 | 2.37e-6 | 9.75e-7 | 1.99e-7 |
| PCFR+ (추상화 없음)                 |         288 |  5.0e-4 | 1.89e-6 | 2.95e-8 | 5.8e-11 |
| ASWISS 추적판 ($\mathcal{P}_0$에서) |         222 | 2.34e-2 | 9.19e-5 | 5.87e-6 | 2.35e-7 |
| CFR+IRA B10H90                      |         287 |  6.8e-4 |  8.7e-6 |  2.6e-6 | 1.87e-7 |
| k-means 고정 (222개)                |         222 |   0.935 |   0.939 |   0.955 |   0.947 |
| $\mathcal{P}_0$ 고정                |          18 |    2.35 |    2.35 |    2.35 |    2.35 |

**표 3.** mini-stud(정보집합 51,114개), 81,910반복. CFR+IRA는 seed 5개의 중앙값 \[최소, 최대\]이다.

| 비교군                                  | bucket |                    평균 정책 | 현재 정책 |
|:----------------------------------------|-------:|-----------------------------:|----------:|
| CFR+ (추상화 없음)                      | 51,114 |                      7.04e-7 |   1.03e-3 |
| PCFR+ (추상화 없음)                     | 51,114 |                      3.23e-8 |   4.66e-5 |
| ASWISS 추적판 ($\mathcal{P}_0$에서)     | 14,145 |                  **2.05e-3** |   5.37e-3 |
| ASWISS 추적판 (CFR+IRA 초기 추상화에서) | 21,215 |                      6.85e-3 |   6.97e-3 |
| CFR+IRA B10H90                          | 21,066 | 5.64e-2 \[5.32e-2, 5.79e-2\] |   4.40e-2 |
| CFR+IRA B100H900                        | 22,218 | 5.33e-2 \[4.83e-2, 6.20e-2\] |   3.84e-2 |
| k-means 고정 (11,229개)                 | 11,229 |                        0.297 |     0.298 |
| $\mathcal{P}_0$ 고정                    |     40 |                        0.764 |     0.764 |

- **Leduc.** 두 방법 모두 원 게임 정확도에 도달한다.
  - CFR+IRA는 처음 30반복 안에 휴리스틱 갱신이 288개 중 284–288개로 나눠, 이후로는 사실상 추상화 없는 CFR+다.
  - ASWISS 추적판은 bucket 222개(77%)로 같은 수준(2.35e-7)에 도달했다.
- **mini-stud.** 같은 반복에서 ASWISS 추적판이 CFR+IRA보다 26–28배 낮았고, bucket은 3분의 1 적었다.
  - 같은 출발점(CFR+IRA의 초기 추상화)끼리 비교해도 약 8배 낮았다.
  - 출발점 $\mathcal{P}_0$(시점 분할)이 CFR+IRA의 초기 추상화(자기 행동열 길이와 행동 수, 14개)보다 bucket도 적고 3.3배 낮았다.
- **고정 추상화.** k-means 고정 추상화는 추상화 오차의 바닥(Leduc 0.95, mini-stud 0.30)에서 멈췄다.
- **추상화 없는 풀이기.** 두 게임 모두 원 게임을 바로 풀 수 있는 크기라서 추상화 없는 PCFR+가 가장 낮았다. 추상화의 쓰임은 정보집합 표나 전체 순회를 감당할 수 없는 게임에 있다.

### 6.3 같은 학습 시간: 표본판

CFR+IRA는 전체 순회가 필요하다. ASWISS는 외부 샘플링으로도 돈다.
**같은 시간의 CFR+IRA 비교(mini-stud).**

- 약 1,000초에서 ASWISS 표본 추적판(seed 4개 중앙값, $z=2$)은 7.67e-3이었고, 전체 순회 CFR+IRA(seed 1)는 9.4e-2였다. 12배 차이다.
- 약 500초에서는 1.20e-2 대 0.155–0.165로 13배 차이였다.
- 같은 ASWISS라도 이 크기에서는 전체 순회판(915초에 3.60e-3, bucket 13,710)이 표본판(1,139초에 7.67e-3, bucket 28,130)보다 낮았다.
- 표본판이 bucket을 더 많이 만든 것은 표본 잡음 때문이다. 정규 검정이 가짜 라벨을 내서 더 많이 나눴다.

**같은 시간의 표본 방법 비교.**

- ASWISS(외부 샘플링, 후회 매칭, 균등 평균)와 같은 순수 MCCFR, 같은 학습기의 k-means, MMD, ESCHER에 같은 학습 시간을 주었다.
- 조건: 같은 기계, 코어 하나, 같은 C++ 구현이다. 학습 시간은 Kuhn 300초, Leduc과 mini-stud 1,800초다.

**표 4.** 같은 학습 시간 끝의 exploitability(평균 정책; MMD는 마지막 반복)

| 게임                |           ASWISS |  MCCFR |         k-means |   MMD | ESCHER |
|:--------------------|-----------------:|-------:|----------------:|------:|-------:|
| Kuhn (300초)        |           1.4e-5 | 2.2e-5 |          3.0e-5 | 0.047 |  0.154 |
| Leduc (1,800초)     |   2.6e-4 (206개) | 1.9e-4 |    0.77 (206개) |  0.42 |   2.09 |
| mini-stud (1,800초) | 0.047 (21,722개) | 1.1e-3 | 0.31 (11,693개) |  0.62 |   2.24 |

- **고정 추상화.** k-means는 추상화 오차의 바닥에 막히고, ASWISS는 그 아래로 계속 내려간다.
  - 같은 반복의 비율(ASWISS/k-means)은 Leduc에서 순회 $10^9$회에 0.00079, mini-stud에서 0.079였다.
  - mini-stud의 아주 짧은 예산(순회 $2\times10^6$회)에서는 ASWISS가 1.7배 높았다.
- **추상화 없는 MCCFR.** 원 게임 표를 메모리에 그대로 둘 수 있는 크기에서는 추상화 없는 MCCFR이 더 낮았다.
  - mini-stud에서 같은 반복의 비율은 15–20배였다.
  - ASWISS가 아끼는 것은 메모리다(mini-stud에서 bucket 수가 정보집합의 42%).
- **신경망 방법.** 같은 CPU 시간에서 MMD와 ESCHER는 두 자릿수 이상 높았다.
  - 신경망 방법은 GPU를 상정한 알고리즘이다. 우리 CPU 구현은 BLAS를 쓰지 않아, 측정상 같은 코어의 numpy보다 2.1–3.6배 느리다.
  - 그래서 이 비교는 같은 CPU 조건에서만의 결과다.

### 6.4 순수 ASWISS

순수판은 정보집합에 대해 아무것도 저장하지 않는다.

- 라벨은 저장한 반복 전략 $K_k=\min(64,8(k+1))$개의 재생으로 구한다.
- 분할 판정은 bucket마다 그 epoch에 방문한 정보집합 16개의 저수지 표본으로 한다.
- 학습기는 외부 샘플링 MCCFR(후회 매칭, 균등 평균, 교대 순회)이다.
- bucket을 정하는 방법은 둘이다.
  - 엄격판: 매번 계보를 따라 다시 계산한다.
  - 표판: 분할 뒤 한 번 계산해 표에 둔다.

  두 방법은 같은 규칙이다. Leduc 1.5만 반복에서 세 구현(엄격, 반복 안 메모, 표)의 정책 파일이 바이트 단위로 같았다. mini-stud 1,000–7,000반복에서도 같은 값이었다. 그래서 규칙의 품질은 표판으로 잰다.

**표 5.** mini-stud, 같은 반복의 평균 정책 exploitability(외부 샘플링, seed 1)

| 방법                          |   3,280만 |    6,550만 |                     1.31억 |
|:------------------------------|----------:|-----------:|---------------------------:|
| 순수 ASWISS (재생 라벨, 표판) | **0.024** | **0.0177** | **0.0140** (bucket 17,455) |
| ASWISS 추적판 (창 라벨)       |     0.131 |      0.086 |                      0.056 |
| 추상화 없는 MCCFR             |    0.0078 |     0.0049 |                     0.0032 |
| k-means 고정 (11,693개)       |     0.294 |          – |                      0.297 |

- 같은 반복에서 순수판은 지금까지의 ASWISS 변형 중 가장 낮았다.
- 1.31억 반복(학습 482초, 분할 17단계, bucket 17,455)에서 0.0140이었다. 같은 반복의 k-means 고정 추상화보다 21배 낮았다.
- 추상화 없는 MCCFR보다는 여전히 4.4배 높았다.

### 6.5 계보 라우팅의 비용

정보집합별 표 없이 `Route`를 부르면, 한 정보집합의 라벨 계산이 상대의 앞선 결정과 이후 결정들의 `Route`를 연쇄적으로 부른다.

**표 6.** mini-stud, 계보 라우팅이 반복 하나에 계산한 트리 노드 수(분할 단계별)

| 메모                     | 1단계 | 2단계 | 3단계 | 4단계 |
|:-------------------------|------:|------:|------:|------:|
| 없음                     | 6.5e3 | 1.4e5 | 1.0e7 | 4.1e8 |
| 500                      | 5.4e3 | 5.1e4 | 6.6e5 | 2.4e7 |
| 무제한 (= 정보집합별 표) |   512 |   494 |   377 |   225 |

- 메모가 없으면 반복당 비용이 분할 단계마다 20–40배씩 늘었다.
- 노트북에서 순수판 엄격 구현의 반복당 시간은 다음과 같았다. 같은 구간의 표판은 반복당 약 0.07밀리초였다.
  - 3단계: 약 0.34초
  - 4단계: 약 21초(트리 노드 약 40억 개)
- 이것이 파생 기법(7절)의 동기다. 규칙의 품질은 좋지만 계보를 따르는 비용이 단계에 대해 곱으로 늘어난다.

### 6.6 정리와 한계

- 누적 후회 라벨로 나누는 규칙은 전체 순회에서 CFR+IRA보다 같은 반복에 26–28배 낮았다. 표본판도 같은 시간에 12–13배 낮았다(mini-stud).
- 순수판의 규칙(재생 라벨 + 구성원 표본 판정)은 같은 반복에서 추적판보다 낮았다.
- 추상화 없는 풀이기와 비교하면, 이 크기의 게임에서는 추상화 없는 쪽이 낮았다. 추상화의 이득은 원 게임 표를 들 수 없는 크기에서만 의미가 있다. 그런 게임(헤즈업 세븐 포커)의 결과는 파생 기법과 함께 7.5절에 둔다.
- 모든 비교가 seed 1개(일부 4–10개)이고 게임도 작다. 곡선이 만나는 지점을 기울기로 외삽하지 않았다.

## 7. 파생 기법

순수 ASWISS의 약점은 둘이다.

1.  정보집합별 표 없이 계보를 따르는 비용이 분할 단계에 대해 곱으로 늘어난다(6.5절).
2.  라벨에 필요한 반복 전략을 모두 저장할 수 없어 표본으로 근사한다.

이 절은 이 약점을 고치는 기법들을 정리한다. 각 기법이 정리의 가정을 얼마나 지키는지 함께 적는다.

### 7.1 평균 전략 방향

epoch의 반복 전략 전부 대신 *평균 전략 하나*로 라벨을 구하면 저장과 계산이 epoch 길이만큼 줄어든다.
두 방향은 일반적으로 같지 않다. 그 차이를 정확히 나타낸 것이 다음 정리다.

**기호.**
플레이어 $p$의 정보집합 $I$, 행동 $a$, 프로필 열 $\sigma_1,\dots,\sigma_T$를 고정한다. $\bar\sigma$는 식 (1)의 평균이다.

- 상대와 우연의 실현 계획: $y_t(z)=\pi^{\sigma_t}_{-p}(z)$, $\bar y=\frac1T\sum_ty_t$
- $Ia$에서 $z$까지 자기 행동 확률: $w_t^{Ia}(z)=\pi_p^{\sigma_t}(Ia\to z)$, $\bar w^{Ia}=\frac1T\sum_tw_t^{Ia}$
- $\hat\sigma_{Ia}$: $Ia$ 아래 $p$의 결정만 $Ia$ 조건부 평균으로 바꾼 $\bar\sigma$
- 도달확률 가중 항: $\Omega_T(I,a)=v^{\hat\sigma_{Ia}}(I,a)-v^{\bar\sigma}(I,a)$
- 공분산 항: $K_T(I,a)=\frac1T\sum_t\sum_{z\sqsupseteq Ia}(y_t-\bar y)(z)\,(w_t^{Ia}-\bar w^{Ia})(z)\,u_p(z)$
- 행동과 무관한 상수: $c_T(I)=T\langle\bar\sigma(I),v^{\bar\sigma}(I)\rangle-\sum_t\langle\sigma_t(I),v^{\sigma_t}(I)\rangle$

**정리 7.1 (평균 전략 방향의 분해).**
모든 행동 $a$에 대해 다음이 성립한다.

$$\sum_{t=1}^Tr^{\sigma_t}(I,a)=T\,r^{\bar\sigma}(I,a)+T\,\Omega_T(I,a)+T\,K_T(I,a)+c_T(I).$$

- 상대의 전략이 어떻게 바뀌어도 차이를 만들지 않는다. 차이는 자기 후속 전략의 도달확률 가중($\Omega$)과, 상대와 자기 후속 전략의 시간 공분산($K$)뿐이다.
- $I$ 아래에 $p$의 결정이 없으면(전선 정보집합) $\Omega=K=0$이고, 행동 순위가 정확히 같다.
- CFR의 반복 전략은 진동할 수 있으므로 $K\to0$의 보장은 없다.
- 그래서 정리 5.9, 5.15의 보장은 평균 전략 방향으로 옮겨지지 않는다.

**실험(mini-stud, 외부 샘플링, 학습 1,800초, seed 1).**
분할 시점 전략 하나로 방향을 구해 계보를 따르는 판(원안)에서 그 전략만 바꿔 보았다(2.62억 반복).

- 순간 전략 하나를 평균 전략으로 바꾸자 0.147에서 0.035로, 같은 반복의 exploitability가 4배 낮아졌다.
- 분할 판정까지 평균 전략의 정확한 방향으로 하면 0.012였다. 같은 반복의 창 라벨(0.041)보다 3.5배 낮고 bucket은 28% 적었다.

### 7.2 bucket 이력 시작 칸

- **문제.** 합법 행동집합만으로 묶은 시작 분할은 한 경로가 같은 칸을 두 번 지날 수 있다(absent-mindedness). 예를 들어 같은 플레이어의 5구와 7구 결정, 또는 레이즈 싸움의 반복된 결정이 같은 칸에 든다.
- **정의.** 그래서 시작 칸을 "그 플레이어가 이번 판에서 지나온 자기 칸(constraint)의 순서" $(i,c_1,\dots,c_k)$로 정한다. 그러면 한 경로가 같은 칸을 두 번 지나지 않는다.
- **칸 수.** mini-stud에서는 20칸이다. 헤즈업 세븐 포커에서는 constraint를 (스트리트, 첫/둘째 행동자, 합법 행동집합)으로 두면 규칙으로 세어 284칸이다.
- **정리와의 관계.** 5절의 보장은 absent-mindedness를 금지하지 않아도 성립한다. 원 게임에서 lifting된 전략은 여전히 행동 전략이기 때문이다. 이 기법의 목적은 추상 게임 학습의 질이다.

### 7.3 관측 분할과 전략 분할의 이중 구조

계보 라우팅의 비용을 없애려고, 정보집합이 들어갈 칸을 정하는 일과 전략표를 나누는 일을 분리한다.
설계 문서에서는 이것을 prior–posterior 구조라고 불렀다. 이 논문에서는 하는 일을 따라 *관측 분할*(observation partition, 옛 이름 prior)과 *전략 분할*(strategy partition, 옛 이름 posterior)이라 부른다.

- **관측 분할.**
  - 정보집합의 관측(카드, 베팅 등)의 특징과 문턱값으로 만든 결정 트리다.
  - 잎(모둠)마다 전략표 하나를 가리키는 포인터가 있다. 여러 잎이 한 표를 같이 쓸 수 있다.
  - 정보집합의 칸은 트리를 한 번 내려가면 정해지므로, 계보를 다시 계산할 필요가 없다.
- **관측 분할의 세분.**
  - 잎마다 후보 자르기(특징, 문턱값)를 몇 개 들고, 각 쪽의 후회 합을 쌓는다.
  - 양쪽 방문이 합쳐 1,000이 되면 한 번 시험한다.
  - 통과하면 잎을 둘로 나누고, 두 자식은 같은 표를 계속 가리킨다(표는 그대로).
- **전략 분할의 세분.**
  - 정해진 간격마다 각 표를 쓰는 잎들을 기록의 argmax 행동으로 묶는다.
  - 무리마다 "그 무리 대 나머지"를 시험한다.
  - 통과한 무리는 새 표로 옮긴다.

두 시험은 모두 *두 묶음의 숨은 후회*를 쓴다.

$$\Delta=\psi(R_1)+\psi(R_2)-\psi(R_1+R_2)\ \ge0.$$

이것은 보조정리 5.4의 숨은 후회를 두 묶음으로 나눈 것이다. 잡음 항은 둘 중 하나를 쓴다.

- 정규 근사: $\Delta\ge\varepsilon t+z_\delta(\hat\sigma_1\sqrt{n_1}+\hat\sigma_2\sqrt{n_2})$
- Freedman 부등식 (Freedman, 1975)의 해: $x_j=c+\sqrt{c^2+2V_j\ln(1/\delta)}$, $c=L\ln(1/\delta)/3$로 두고 $\Delta\ge\varepsilon T+x_1+x_2$

$\varepsilon$은 "나눠서 숨은 후회가 반복당 이만큼은 줄어야 나눈다"는 문턱이다.

**장기 실행(mini-stud, seed 1).**

- 설정: bucket 이력 시작 칸, 관측 분할 실시간 시험, 전략 분할 실시간 Freedman 시험, 새 표는 부모 표를 복사(완전 상속)
- 이 설정으로 $\varepsilon=0,0.01,0.03,0.1$을 167.77억 반복까지 돌렸다(표 7).

**표 7.** mini-stud 장기 실행, 평균 정책 exploitability(괄호는 쓰이는 표 수)

| 반복     |    $\varepsilon=0$ | $\varepsilon=0.01$ | $\varepsilon=0.03$ | $\varepsilon=0.1$ |
|:---------|-------------------:|-------------------:|-------------------:|------------------:|
| 1.31억   |      0.528 (5,668) |        0.552 (613) |        0.639 (250) |        0.571 (68) |
| 20.97억  |     0.338 (14,790) |        0.506 (871) |        0.623 (311) |        0.626 (77) |
| 167.77억 | **0.173** (20,353) |        0.496 (962) |        0.661 (340) |        0.592 (82) |

- **$\varepsilon=0$.** 끝까지 내려간 것은 $\varepsilon=0$뿐이다. 숨은 후회도 반복당 2에서 0.066으로 줄었다.
- **$\varepsilon>0$.** 10억 반복 근처부터 평평했고, 숨은 후회가 3–4에서 줄지 않았다.
- **k-means 기준선.** 같은 학습기의 k-means 고정 추상화는 900초(약 2.24억 반복)까지만 있고, 1.31억 반복에서 0.339였다. 그 구간에서는 이 판이 더 높았다. 그 뒤의 비교는 기준선이 없어서 하지 않았다.
- **같은 반복의 다른 판.** 같은 반복에서는 이 판이 순수판(표 5, 1.31억 반복 0.0140)보다 훨씬 높다.
  - 표를 완전 상속한 것과 관측 분할의 칸이 다른 것이 섞여 있어서 원인을 나누지 않았다.
  - 이 구조의 이득은 계보 비용이 없다는 데 있다.

### 7.4 고정 문턱의 $\varepsilon$-수렴과 1단 게임의 반례

분할을 "추정한 숨은 후회 $\hat C_W(B)$가 $\theta\left\lvert W\right\rvert$를 넘을 때"로 정하는 판을 생각한다.

- **주장.** 고정 문턱 $\theta>\eta>0$으로 판정하면 다음이 성립한다는 개요가 있다.
  - 분할은 유한 번이다.
  - 그 뒤 모든 bucket의 숨은 후회 비율이 $\theta+\eta$ 아래에 있다.
  - $\mathrm{NashConv}\le O(1/\sqrt{\left\lvert W\right\rvert})+\left\lvert \mathcal{B}^*\right\rvert(\theta+\eta)$이다(`ASWISS_PROOF_OUTLINE_KO.md`).
- **논리의 상태.** 논리는 정리 5.9와 같은 뼈대다. 다만 버킷 크기와 무관한 표본 추정(그 개요의 L4)은 아직 증명하지 않았다. 라벨 항등식으로 선형화하는 방법을 제안해 두었다.
- **우리 실행과의 대응.** 7.3절의 $\varepsilon>0$ 실행이 이 문턱판에 해당한다. 평평해진 결과와 결이 맞는다.

같은 개요가 쓴 1단 행렬 게임 실험(공개 우연이 copy를 고르고 두 플레이어가 copy를 봄)을 다시 돌렸다. 기록된 수치가 모두 재현되었다.

- **반례 B.** 라벨대로 한 플레이어만 나누면 NashConv가 0.238에서 2.5로 오른다. 다음 분할에서 0이 된다.
  - 분할 전 묶인 추상 균형은 $x=y=(10/21,11/21)$이다.
  - 분할 뒤 P1의 추상 maximin은 하나로 정해진 순수에 가까운 전략이다.
  - 그래서 묶인 상대가 원 게임에서 크게 공략한다.
- **$\varepsilon$-계열.** 분할 이득이 0.00023밖에 안 되는 분할도 2.5로 오를 수 있다. 문턱이 있으면 이런 분할을 하지 않는다.
- **분할 규칙 비교.** 같은 게임 2,892개에서, 한 번이라도 상한이 오른 게임의 비율은 다음과 같았다.
  - 한 단계에 한 bucket을 나누는 규칙: 21.2%
  - 순수 ASWISS처럼 라벨이 갈린 bucket을 모두 동시에 나누는 규칙: 10.1%

이 게임에는 사적 정보가 없고, 균형이 하나인 경우만 썼다. 그래서 포커로 그대로 옮기지 않는다.

### 7.5 헤즈업 세븐 포커

실제 크기의 헤즈업 세븐 포커(5–7구 베팅, 4구 버림과 공개는 고정 휴리스틱)에서는 순수판을 돌릴 수 없다. 원 정보집합이 학습 중 거의 다시 오지 않고, 계보 비용도 감당할 수 없다.
이 게임의 ASWISS(`hist_bbf`)는 다음을 함께 썼다.

- 평균 전략 방향(7.1절)
- 정보집합별 표 대신 bucket 이력 표
- 역방향 분할
- 카드 군집 codebook을 관측으로 쓰는 사전 정보
- 학습기 MCCFR+

평가는 Policy-LBR(국소 최선 응답 공격자가 따는 양, ante/판)이다. exploitability의 근사 하한이지 exploitability가 아니다.

**표 8.** 세븐 포커, 같은 학습 CPU 24시간(스레드 하나, seed 1, 평균 정책, 64파티클, 같은 20,000쌍 deal, 95%)

| 비교군                        |                         학습량 |                   LBR | k-means 256과의 짝지은 차이 |
|:------------------------------|-------------------------------:|----------------------:|----------------------------:|
| ASWISS (파생 기법 + codebook) | 1.28억 roots, bucket 1,199,549 | **0.995 $\pm$ 0.116** |            $-0.548\pm0.170$ |
| k-means 256                   |    2.56억 roots, bucket 37,277 |     1.543 $\pm$ 0.121 |                           0 |
| MMD                           |                     239.5만 판 |      27.10 $\pm$ 1.77 |             $+25.55\pm1.77$ |
| ESCHER                        |                      53.1만 판 |      32.28 $\pm$ 1.80 |             $+30.73\pm1.81$ |
| ReBeL                         |                      23.9만 판 |     2.439 $\pm$ 0.246 |            $+0.896\pm0.270$ |

- **같은 roots에서의 비율.** ASWISS/k-means 256 비율은 0.15M roots에서 1.50, 10.23M에서 0.82, 81.91M에서 0.70이었다. 초반에는 ASWISS가 나쁘고 1천만 roots 무렵부터 앞섰다.
- **이득의 출처.** ASWISS와 k-means 256은 같은 자료로 만든 codebook을 공유하고, ASWISS의 bucket 수가 32배다. 그래서 이득이 방법에서 왔는지 크기에서 왔는지는 이 실험으로 나눌 수 없다.
- **숨은 후회 판.** 관측·전략 이중 구조와 숨은 후회 시험의 세븐 포커 판(`pphist_freedman_rt`)은 같은 조건의 비교군으로 넣어 두었고, 결과는 아직 없다.

## 8. 한계와 후속 연구

### 8.1 한계

- **속도.** 마지막 분할 시점 $t^*$의 상한이 없다. 정리의 속도는 $t^*$ 이후에만 적용된다. 실험의 곡선 대부분은 $t^*$ 이전이다.
- **가정과 구현의 차이.** 정리는 동시 갱신, RM+, 판정 구간의 균등 혼합을 가정한다. 구현은 교대 순회, 후회 매칭, bucket 평균 표를 쓴다(표 1).
- **계보 비용.** 정보집합별 표 없는 순수판은 계보 비용이 분할 단계에 대해 곱으로 늘어난다. 큰 게임에는 7절의 파생 기법이 필요하다. 그 기법들은 정리의 가정을 일부 벗어난다.
- **실험의 범위.** 실험은 seed 1개(일부 4–10개)이고, 정확한 exploitability는 작은 게임에서만 쟀다. 세븐 포커의 지표(LBR)는 exploitability가 아니다.

### 8.2 후속 연구

**1. 더 나은 초기 추상화.**

- **구조 서명.** 수렴에 필요한 것은 같은 플레이어, 같은 행동집합뿐이다. 그러나 앞으로의 구조까지 같은 것끼리 묶으면 필요한 분할이 줄 것으로 기대한다.
  - 정보집합 $I$에서 행동 $a$ 뒤에 같은 플레이어가 다음으로 마주칠 수 있는 정보집합의 집합을 $\mathrm{Next}(I,a)$라 한다.
  - 깊이 $d$의 구조 서명을 다음과 같이 재귀적으로 정의한다.

$$\sigma_0(I)=(P(I),A(I)),\qquad\sigma_d(I)=\big(\sigma_0(I),\ (a\mapsto\{\sigma_{d-1}(J):J\in\mathrm{Next}(I,a)\})_{a}\big)$$

  - 서명이 같은 것끼리 묶으면 게임 지식 없이 시뮬레이터만으로 시작 분할을 만들 수 있다.
  - 거리를 $(1-\gamma)\sum_d\gamma^d\mathbb 1[\sigma_d(I)\ne\sigma_d(I')]$로 두면 초거리(ultrametric)가 된다. 그래서 "거리 $\le\varepsilon$"이 동치관계가 되어 그대로 bucket으로 쓸 수 있다.
- **보상과 우연 확률.** 보상과 우연 확률의 차이까지 넣는 확장은 고정 추상화의 오차 상한 (Kroer & Sandholm, 2016; Kroer & Sandholm, 2018)과 연결할 수 있다.
- **이전 실행의 재사용.** 이전 실행의 최종 분할을 다음 실행의 시작점으로 쓰는 것도 후보다.

**2. 추상화 분할의 개선.**

- **거짓 분할 없애기.** 여유 라벨 집합(보조정리 5.13)을 표본 판정에 넣으면 좋은 사건 위에서는 거짓 분할이 없다. 지금의 argmax와 역치 판정에는 이 성질이 없다.
- **숨은 후회를 직접 추정하기.** 라벨 항등식으로 $C_W(B)=\min_\ell\sum_{I\in B}e_I(\ell)$을 행동마다 선형인 합으로 바꿀 수 있다. 도달 질량에 비례해 뽑은 구성원으로 중요도 가중 추정을 하면, 표본 수가 bucket 크기와 무관해질 것으로 본다(7.4절). 증명은 남아 있다.
- **문턱의 크기.** 문턱을 bucket의 도달 질량에 비례하게 잡으면, 고정 문턱판의 오차가 최종 bucket 수에 비례하지 않게 할 수 있다.
- **일시적 악화 줄이기.** 1단 게임에서는 라벨이 갈린 bucket을 두 플레이어 모두 한꺼번에 나눌 때 일시적인 악화가 절반으로 줄었다. 사적 정보가 있는 게임에서 같은지 확인해야 한다.
- **자식 표를 어디서 시작할지.** 0에서 다시 시작할지, 부모 표를 물려줄지(regret transfer (Brown & Sandholm, 2014)) 정해야 한다. RM+에서는 자식의 시작 노름이 부모 이하이면 보장이 유지된다.
- **평균 전략 방향의 치우침.** $\Omega$는 조건부 평균 표로 없앨 수 있다. $K$를 줄이는 방법은 반복 전략 몇 개를 섞는 것이다.
- **$t^*$의 상한.** 상위 두 행동의 누적 후회 차이에 대한 margin 조건 아래 $t^*$를 묶어 볼 수 있다.

**3. 3인 이상의 게임.**

- **성립하는 것.** 보조정리 5.3의 첫 부등식(후회 분해)과 라벨 항등식은 플레이어마다 성립한다. 그래서 ASWISS를 그대로 쓰면 각 플레이어의 평균 후회를 0으로 보내는 논증이 그대로 남을 것으로 본다.
- **성립하지 않는 것.** 2인 제로섬이 아니면 평균 후회가 0이라는 것이 내쉬 균형을 뜻하지 않는다. 반복 프로필들의 균등 분포가 근사 coarse correlated equilibrium이라는 데서 멈춘다.
- **과제.** 정리해야 할 것은 둘이다.
  - 어떤 해 개념을 목표로 할지
  - 6인 포커처럼 실전에서 쓰이는 설정 (Brown & Sandholm, 2019)에서 분할 규칙이 무엇을 보장하는지

**4. Absent-mindedness가 있을 때의 전략 개선.**

- **지금 보장되는 것.** 시작 분할이나 분할 결과가 absent-minded이면(한 경로가 같은 bucket을 두 번 지나면), 원 게임 NashConv의 상한(5절)은 그대로 성립한다.
- **문제.** 그러나 추상 게임 안의 학습 목표가 흐려진다.
  - 계획 단계의 최적과 행동 단계의 최적이 다를 수 있다 (Piccione & Rubinstein, 1997).
  - bucket의 후회 합이 한 경로의 여러 방문을 따로 세기 때문이다.
- **지금의 방법.** 지금은 bucket 이력 시작 칸(7.2절)으로 absent-mindedness 자체를 피한다.
- **과제.** 피할 수 없는 큰 게임을 위해 두 가지가 필요하다.
  - absent-minded bucket에서 일관된 행동 전략을 학습하는 갱신 규칙
  - 그 규칙이 원 게임 상한과 어떻게 맞물리는지에 대한 분석

**5. 관측 분할(옛 prior 추상화)의 구성.**

- **이름.** 설계 문서의 "prior 추상화"는 전략을 보기 전에 관측만으로 정하는 분할이라 붙은 이름이다. 베이즈 사전분포와 혼동되므로 이 논문은 *관측 분할*이라 부른다(7.3절).
- **지금 방식의 약점.** 지금의 관측 분할은 무작위 후보 자르기(특징, 문턱값)를 시험해 나눈다. 후보가 무작위라서 필요한 자르기를 찾기까지 오래 걸린다.
- **개선 후보.**
  - 분할 전 전략으로 얻은 표본의 후회 벡터나 가치 벡터 사이의 거리로 라벨을 붙인다. 그 라벨을 맞히는 결정 트리를 학습해 자르기를 고른다.
  - 학습한 관측 임베딩이나 카드 codebook을 특징으로 쓴다.
  - 관측 분할이 세분된 뒤 포인터를 다시 배정하는 탐색 기간을 둔다.
  - 잎 수에 상한을 둔다.
- **필요한 분석.** 관측 분할은 표를 바꾸지 않으므로 수렴 보장과 독립적으로 고칠 수 있다. 다만 관측 분할이 충분히 세밀하지 않으면 전략 분할이 나눌 수 없는 숨은 후회가 남는다. 그 크기에 대한 분석이 필요하다.

**6. 그 밖의 과제.**

- 교대 갱신과 선형 평균으로 증명을 넓힌다.
- 저장한 반복 전략의 혼합으로 정리의 대상인 평균 정책을 내놓는다.
- seed 반복과 신뢰구간을 낸다.
- 같은 크기와 같은 정보를 가진 고정 추상화와 비교한다.
- 신경망 기준선을 GPU에서 비교한다.

## 9. 결론

이 논문은 표본만으로 정보 추상화를 세분하는 순수 ASWISS를 제안했다.

- **방법.**
  - 게임 구조만으로 만든 시작 분할에서 외부 샘플링 MCCFR로 bucket 전략을 학습한다.
  - 반복 전략들의 누적 반사실적 후회로 라벨을 붙이고, 라벨이 갈린 bucket을 나눈다.
  - 정보집합의 bucket은 분할 계보로 정한다.
- **이론.**
  - 숨은 후회가 공통 라벨을 강요하는 최소 비용과 같다는 라벨 항등식을 보였다.
  - 이 항등식으로 분할의 유한성과 마지막 분할 이후 원 게임 NashConv의 수렴을 증명했다.
  - 저장한 반복 전략 일부로 라벨을 재생해도 같은 꼴의 보장이 있음을 보였다.
- **실험.** 작은 게임에서 이 규칙은 같은 반복의 CFR+IRA와 고정 추상화보다 낮은 exploitability에 도달했다.
- **남은 문제.** 정보집합별 표 없이 계보를 따르는 비용과 마지막 분할 시점의 상한이다. 앞의 것은 관측 분할과 전략 분할을 나누는 파생 기법으로 줄였지만, 그 기법의 보장은 아직 순수판만큼 정리되지 않았다.

## 부록 A. 증명

이 부록은 5절과 7.1절의 결과를 증명한다. 기호는 5절을 따른다. 같은 증명과 작은 게임에서의 수치 확인은 다음 파일에 있다.

- 증명: `ASWISS_CONVERGENCE_KO.md`
- 수치 확인: `test_aswiss_convergence.py`, `test_avg_direction.py`

### A.1 보조정리 5.3 (NashConv 분해)

*증명.*

- 제로섬이므로 $\mathrm{NashConv}(\bar\sigma^W)=\sum_i\max_\beta u_i(\beta,\bar\sigma^W_{-i})$다.
- $\beta_i$를 $\bar\sigma^W_{-i}$에 대한 순수 최선 응답이라 하자. 상대의 실현 계획에 대한 선형성으로 $u_i(\beta_i,\bar\sigma^W_{-i})=\frac1{\left\lvert W\right\rvert}\sum_{t\in W}u_i(\beta_i,\sigma_{t,-i})$다.
- (S1)로 두 플레이어가 같은 $\sigma_t$를 쓰므로 $\sum_iu_i(\sigma_t)=0$을 빼 넣을 수 있다.

$$\mathrm{NashConv}(\bar\sigma^W)=\frac1{\left\lvert W\right\rvert}\sum_{t\in W}\sum_i\big(u_i(\beta_i,\sigma_{t,-i})-u_i(\sigma_t)\big)$$

- 보조정리 3.3를 각 $t$와 $i$에 쓰고 $t$에 대해 더하면 등식을 얻는다.
- 부등식은 $\pi_i^{\beta_i}(I)\in\{0,1\}$과 $R_I^W(\beta_i(I))\le\psi(R_I^W)$에서 나온다.
- 마지막 등식은 $H$의 정의다.

∎

### A.2 보조정리 5.4 (라벨 항등식)

*증명.*

- $x\mapsto x(\ell)$은 선형이고 $\psi(x)=\max_\ell x(\ell)$이다. 그래서 $\psi(R_B^W)=\max_\ell\sum_{I\in B}R_I^W(\ell)$다.
- 이것을 $\sum_I\psi(R_I^W)$에서 빼면 $H_B^W=\min_\ell\sum_I\big(\psi(R_I^W)-R_I^W(\ell)\big)$이다.
- $\psi(x)\ge x(\ell)$이므로 각 항은 0 이상이다.
- 따라서 $H_B^W=0$은 어떤 $\ell$에서 모든 항이 0이라는 것, 즉 공통 최대 라벨이 있다는 것과 같다.

∎

### A.3 보조정리 5.5 (RM+의 구간 상한)

*증명.*
$x_u=\tilde r_u(B)$로 둔다.

- 방문된 멤버마다 $\tilde r_u(I)=\tilde v-\langle\sigma_u(B),\tilde v\rangle\mathbf 1$이므로 $\langle\sigma_u(B),\tilde r_u(I)\rangle=0$이다. 멤버들에 대해 더하면 $\langle\sigma_u(B),x_u\rangle=0$이다.
- $Q_u\ge0$이고 $\sigma_u\propto Q_u$이므로 $\langle Q_u,x_u\rangle=0$이다. 따라서 다음이 성립한다.

$$\left\lVert Q_{u+1}\right\rVert_2^2\le\left\lVert Q_u+x_u\right\rVert_2^2=\left\lVert Q_u\right\rVert_2^2+\left\lVert x_u\right\rVert_2^2$$

- 귀납법과 (S3)로 $\left\lVert Q_t\right\rVert_2^2\le\left\lvert A_B\right\rvert D_B^2(t-\tau_B)$다.
- $Q_{u+1}\ge Q_u+x_u$이므로 성분별로 $\sum_{u\in W}x_u\le Q_t-Q_s\le Q_t$다.

확률을 쓰지 않는 결론이다. 직교성이 같은 $\sigma_u(B)$를 쓰는 멤버들의 합에서도 유지되기 때문이다.
∎

### A.4 보조정리 5.6 (시간 균일 집중)

*증명.*

- (S2)에 의해 $X_u=\tilde r_u(I,a)-r_u(I,a)$는 $\left\lvert X_u\right\rvert\le2\Delta_I$인 마팅게일 차분이다.
- 블록 $[2^j,2^{j+1})$에 Azuma 부등식의 최대 버전을 쓰면 다음을 얻는다.

$$\Pr\Big(\max_{t<2^{j+1}}\big\lvert\textstyle\sum_{u\le t}X_u\big\rvert\ge\lambda_j\Big)\le2\exp\!\Big(-\frac{\lambda_j^2}{16\cdot2^j\Delta_I^2}\Big)$$

- $\delta_j=6\delta/(\pi^2(j+1)^2)$, $\lambda_j=4\Delta_I\sqrt{2^j\ln(2N_A/\delta_j)}$로 고르면 이 확률은 $\delta_j/N_A$다.
- $t\in[2^j,2^{j+1})$이면 $\lambda_j\le\varepsilon_I(t)$다.
- $(I,a)$ $N_A$개와 $j$에 대해 합집합 상계를 쓰면 실패 확률은 $\sum_j\delta_j=\delta$ 이하다.
- 구간 합은 두 누적합의 차이이므로 $2\varepsilon_I(t)$로 묶인다. 시간 균일 상계이므로 구간의 끝이 정지시간이어도 된다.

∎

### A.5 보조정리 5.7 (라벨이 같은 bucket)

*증명.*
첫 부등식 $H_B^W\le\sum_{I\in B}e_I(\ell)$은 보조정리 5.4이다. $\eta_I=2\varepsilon_I(t)$로 두면 보조정리 5.6로 $\left\lVert R_I^W-\tilde R_I^W\right\rVert_\infty\le\eta_I$다.

- **$\ell=\varnothing$.** 검정에 실패했거나 방문이 없었으므로 $\psi(\tilde R_I)\le\theta_I$다. $\psi$는 $\left\lVert \cdot\right\rVert_\infty$에 대해 1-립시츠이므로 $e_I(\varnothing)=\psi(R_I)\le\theta_I+\eta_I$다.
- **$\ell=a\in A$.** $a$는 $\tilde R_I$의 argmax이고 $\tilde R_I(a)>\theta_I\ge0$이다.
  - $\max_bR_I(b)\ge0$이면 $e_I(a)\le(\tilde R_I(a)+\eta_I)-(\tilde R_I(a)-\eta_I)=2\eta_I$다.
  - 그렇지 않으면 $e_I(a)=-R_I(a)<\eta_I$다.

어느 경우든 $e_I\le\theta_I+2\eta_I=\theta_I+4\varepsilon_I(t)$다.
∎

검정의 통계적 타당성(가짜 라벨이 나올 확률)은 이 증명에 쓰이지 않는다. 쓰이는 것은 역치의 크기 조건 (T)뿐이다. 가짜 라벨은 분할을 늘릴 뿐이고, 분할은 유한하다.

### A.6 보조정리 5.8 (분할 유한성)

*증명.*
분할마다 자식은 실제 멤버를 가진 둘 이상의 묶음이다. 방문되지 않은 멤버도 라벨 $\varnothing$을 가진 실제 멤버이므로 $\varnothing$ 자식도 비어 있지 않다.
그래서 bucket 수는 분할마다 엄격히 늘고, $N$을 넘지 못한다.
∎

### A.7 정리 5.9과 따름정리 5.10

*증명.*

- epoch $k$ 동안 분할 $\mathcal{P}_k$는 고정이다. 그 끝에서 분할이 없었으므로 모든 bucket의 라벨이 하나다.
- 보조정리 5.3로 $C$와 $H$로 나눈다.
- **$C$ 항.** 보조정리 5.6과 5.5로 다음을 얻는다.

$$\psi(R_B)\le\psi(\tilde R_B)+\sum_{I\in B}2\varepsilon_I(t_{k+1}),\qquad\psi(\tilde R_B)\le D_B\sqrt{\left\lvert A_B\right\rvert t_{k+1}}$$

- **$H$ 항.** 보조정리 5.7로 묶는다. 두 항을 더하면 정리의 식이다.
- **차수.** 정규 검정이면 $\theta_I\le z\Delta_I\sqrt{L_k}$이고, $\gamma=2$이면 $t_{k+1}\le2L_k$다. 그래서 전체가 $O(\sqrt{\log(kN_A/\delta)/L_k})$다.
- **거의 확실한 수렴.** 사건 $\{\limsup_k\mathrm{NashConv}>0\}$은 모든 $\delta$에 대해 $\mathcal{E}_\delta^c$에 포함되므로 확률이 0이다.
- **따름정리.** $\psi$의 준가법성 $\psi(\sum_kx_k)\le\sum_k\psi(x_k)$로 epoch별로 나눈다. 분할 시기의 epoch는 $\psi(R_I^{(k)})\le\Delta_IL_k$로 묶고, 나머지는 등비합이다.

∎

### A.8 정리 5.11과 정리 5.12

*증명.*

- **(i)** 분할이 고정된 시점 $s$ 이후 다음이 성립하므로 판정이 끝없이 미뤄지지 않는다.

$$\tilde C^{(s,t]}/(t-s)\le\sum_BD_B\sqrt{\left\lvert A_B\right\rvert t}/(t-s)\to0\quad(\text{보조정리 5.5})$$

- **(ii)**는 보조정리 5.8이다.
- **(iii)**은 정리 5.9의 증명과 같다. 보조정리 5.6이 시간 균일이라 판정 구간의 끝이 정지시간이어도 된다.
- **인증.**
  - 보조정리 5.6로 $C^W\le\tilde C^W+2\sum_I\varepsilon_I(t)$다.
  - 판정 통과이므로 $\tilde C^W\le\varepsilon_g\left\lvert W\right\rvert$다.
  - $H^W$는 보조정리 5.7로 묶고, 보조정리 5.3로 합친다.
  - 나뉜 bucket은 보조정리 5.4의 최솟값을 그대로 더한다.

∎

**비고 A.1 (인증의 크기).**
mini-stud(정보집합 51,114개, $\sum_I\Delta_I=566{,}366$, $N_A=142{,}428$, $\delta=0.05$)에 넣어 보았다. $t=4.1\times10^7$, $\left\lvert W\right\rvert=5.35\times10^5$에서 표본 오차 항 $6\sum_I\varepsilon_I/\left\lvert W\right\rvert$는 약 $7.7\times10^5$다. 실제 NashConv(약 1)에 비해 매우 느슨하다.
원인은 모든 정보집합에 대한 합과 최악의 경우의 Azuma 상한이다. 실용적인 인증에는 분산을 쓰는 상한(Freedman형)이나 표본을 나눈 직접 추정이 필요하다.

### A.9 보조정리 5.13 (여유 라벨 집합)

*증명.*
$\max_{\ell'}\hat x_I(\ell')\le\psi(x_I)+\eta_I$이다. 참 최대 라벨 $\ell^*$에 대해서는 $\hat x_I(\ell^*)\ge\psi(x_I)-\eta_I$다.

- **(i)** $\ell$이 참 최대 라벨이면 $\hat x_I(\ell)\ge\psi(x_I)-\eta_I\ge\max\hat x_I-2\eta_I$다.
- **(ii)** $\ell\in\hat M(I)$이면 $x_I(\ell)\ge\hat x_I(\ell)-\eta_I\ge\max\hat x_I-3\eta_I\ge\psi(x_I)-4\eta_I$다.
- **(iii)**은 (i)에서, (iv)는 (ii)와 보조정리 5.4에서, (v)는 (ii)에서 나온다.

∎

### A.10 보조정리 5.14과 정리 5.15 (재생 라벨)

*증명.*

- **보조정리 5.14.**
  - 학습 궤적 전체를 조건으로 두면 $\tau_{k,j}$들은 독립이고 균등하다. 그래서 $X_j=L_kr^{\sigma_{\tau_{k,j}}}(I,a)$는 평균이 $R_I^{(k)}(a)$이고 $\left\lvert X_j\right\rvert\le L_k\Delta_I$인 독립 표본이다.
  - Hoeffding 부등식으로 다음을 얻는다.

$$\Pr(\left\lvert \hat R-R\right\rvert>\eta)\le2\exp\!\big(-K_k\eta^2/(2L_k^2\Delta_I^2)\big)=6\delta/(\pi^2N_A(k+1)^2)$$

  - $(I,a)$와 $k$에 대해 합집합 상계를 쓴다.
  - 구간 끝을 미리 모르면 저수지 표집으로 비복원 균등 추출을 한다. Hoeffding 부등식은 비복원 추출에도 성립한다 (Hoeffding, 1963).
- **정리 5.15.**
  - 보조정리 5.3로 나누고, $C$ 항은 정리 5.9과 같이 묶는다.
  - 분할이 없었으므로 모든 bucket이 여유 라벨로 일치했다. 보조정리 5.13(iv)와 보조정리 5.14로 $H^{(k)}\le4\sum_I\eta_I^{(k)}$다.
  - (i)은 보조정리 5.13(iii)이다.

∎

### A.11 정리 7.1 (평균 전략 방향의 분해)

*증명.*
완전 기억이면 $v^\sigma(I,a)=\sum_{z\sqsupseteq Ia}\pi^\sigma_{-p}(z)\,\pi_p^\sigma(Ia\to z)\,u_p(z)$다.

1.  곱의 평균은 평균의 곱에 공분산을 더한 것이다.

$$\frac1T\sum_tv^{\sigma_t}(I,a)=\frac1T\sum_t\sum_zy_tw_tu=\sum_z\bar y\,\bar w\,u+K_T$$

2.  상대의 평균은 실현 계획의 평균이고 (Stengel, 1996) 우연은 고정이므로 $\pi^{\bar\sigma}_{-p}(z)=\bar y(z)$다.
3.  같은 논증을 $Ia$ 아래 부분 트리에 쓰면, $Ia$ 조건부 평균에서 $\pi_p(Ia\to z)=\bar w^{Ia}(z)$다.
4.  따라서 1의 첫 항은 $v^{\hat\sigma_{Ia}}(I,a)=v^{\bar\sigma}(I,a)+\Omega_T(I,a)$다.
5.  $\sum_tr^{\sigma_t}(I,a)=\sum_tv^{\sigma_t}(I,a)-\sum_t\langle\sigma_t(I),v^{\sigma_t}(I)\rangle$에 4를 넣고 $T\,r^{\bar\sigma}(I,a)$를 빼면 식이 된다.

∎

- $I$ 아래에 $p$의 결정이 없으면 $w\equiv1$이므로 $\Omega=K=0$이다. 이때 두 방향의 행동 순위는 같다.
- 자기 후속 전략이 $t$에 무관하면 역시 $\Omega=K=0$이다.
- $\left\lvert K_T\right\rvert$는 상대 실현 계획의 변동과 자기 후속 전략의 변동의 곱으로 묶인다. 다만 CFR 반복이 진동할 수 있으므로 0으로 간다는 보장은 없다.

## 부록 B. 재현

모든 경로는 저장소 `agents/adaptive_buckets/` 아래다.

| 내용                             | 위치                                                          |
|:---------------------------------|:--------------------------------------------------------------|
| 5절의 증명과 수치 확인           | `ASWISS_CONVERGENCE_KO.md`, `test_aswiss_convergence.py`      |
| 정리 7.1의 수치 확인             | `test_avg_direction.py`                                       |
| 순수 ASWISS 학습기               | `aswiss/pure_aswiss.cpp` (`–route-impl lazy|table`, `–memo`)  |
| 추적판과 원안의 전이 실험        | `aswiss/aswiss.cpp`, 결과 `aswiss/compare/README_KO.md`       |
| CFR+IRA 구현과 전체 순회 비교    | `aswiss/compare/refine.cpp`, `run_compare.sh`                 |
| 같은 시간 비교(MMD, ESCHER 포함) | `aswiss/compare/run_walltime.sh`                              |
| 정확한 평가                      | `aswiss/eval.py`                                              |
| mini-stud 트리와 시작 분할       | `aswiss/export.py`, `environments/mini_stud/game.py`          |
| 관측·전략 이중 구조의 장기 실행  | `PP_STUD7_DESIGN_KO.md` 22.12–22.13절, `aswiss/pp_aswiss.cpp` |
| 1단 행렬 게임 실험               | `ASWISS_PROOF_OUTLINE_KO.md`, `aswiss/proof_experiments/`     |
| 세븐 포커 비교                   | `aswiss/stud7/README_KO.md`의 `runs_fair`                     |

## 참고 문헌

- David Blackwell (1956). An Analog of the Minimax Theorem for Vector Payoffs. *Pacific Journal of Mathematics*, 6(1), 1–8.
- Michael Bowling, Neil Burch, Michael Johanson, Oskari Tammelin (2015). Heads-up Limit Hold'em Poker Is Solved. *Science*, 347(6218), 145–149.
- George W. Brown (1951). Iterative Solution of Games by Fictitious Play. *Activity Analysis of Production and Allocation*.
- Noam Brown, Tuomas Sandholm (2014). Regret Transfer and Parameter Optimization. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Noam Brown, Tuomas Sandholm (2015). Simultaneous Abstraction and Equilibrium Finding in Games. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*.
- Noam Brown, Tuomas Sandholm (2017). Safe and Nested Subgame Solving for Imperfect-Information Games. *Advances in Neural Information Processing Systems 30 (NIPS)*.
- Noam Brown, Tuomas Sandholm (2018). Superhuman AI for Heads-up No-Limit Poker: Libratus Beats Top Professionals. *Science*, 359(6374), 418–424.
- Noam Brown, Tuomas Sandholm (2019). Solving Imperfect-Information Games via Discounted Regret Minimization. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Noam Brown, Tuomas Sandholm (2019). Superhuman AI for Multiplayer Poker. *Science*, 365(6456), 885–890.
- Noam Brown, Adam Lerer, Sam Gross, Tuomas Sandholm (2019). Deep Counterfactual Regret Minimization. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
- Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong (2020). Combining Deep Reinforcement Learning and Search for Imperfect-Information Games. *Advances in Neural Information Processing Systems 33 (NeurIPS)*.
- Neil Burch, Michael Johanson, Michael Bowling (2014). Solving Imperfect Information Games Using Decomposition. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Neil Burch, Matej Moravcik, Martin Schmid (2019). Revisiting CFR+ and Alternating Updates. *Journal of Artificial Intelligence Research*, 64.
- Jiří Čermák, Branislav Bošanský, Viliam Lisý (2017). An Algorithm for Constructing and Solving Imperfect Recall Abstractions of Large Extensive-Form Games. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*.
- Jiří Čermák, Viliam Lisý, Branislav Bošanský (2020). Automated Construction of Bounded-Loss Imperfect-Recall Abstractions in Extensive-Form Games. *Artificial Intelligence*, 282, arXiv:1803.05392.
- Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, Haoyang Zeng (2018). Training GANs with Optimism. *International Conference on Learning Representations (ICLR)*.
- Gabriele Farina, Christian Kroer, Tuomas Sandholm (2020). Stochastic Regret Minimization in Extensive-Form Games. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, arXiv:2002.08493.
- Gabriele Farina, Christian Kroer, Tuomas Sandholm (2021). Faster Game Solving via Predictive Blackwell Approachability: Connecting Regret Matching and Mirror Descent. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- David A. Freedman (1975). On Tail Probabilities for Martingales. *The Annals of Probability*, 3(1), 100–118.
- Sam Ganzfried, Tuomas Sandholm (2014). Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance in Imperfect-Information Games. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Richard Gibson, Marc Lanctot, Neil Burch, Duane Szafron, Michael Bowling (2012). Generalized Sampling and Variance in Counterfactual Regret Minimization. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Andrew Gilpin, Tuomas Sandholm (2007). Lossless Abstraction of Imperfect Information Games. *Journal of the ACM*, 54(5).
- Sergiu Hart, Andreu Mas-Colell (2000). A Simple Adaptive Procedure Leading to Correlated Equilibrium. *Econometrica*, 68(5), 1127–1150.
- Johannes Heinrich, David Silver (2016). Deep Reinforcement Learning from Self-Play in Imperfect-Information Games. *arXiv:1603.01121*.
- Johannes Heinrich, Marc Lanctot, David Silver (2015). Fictitious Self-Play in Extensive-Form Games. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
- Samid Hoda, Andrew Gilpin, Javier Peña, Tuomas Sandholm (2010). Smoothing Techniques for Computing Nash Equilibria of Sequential Games. *Mathematics of Operations Research*, 35(2), 494–512.
- Wassily Hoeffding (1963). Probability Inequalities for Sums of Bounded Random Variables. *Journal of the American Statistical Association*, 58(301), 13–30.
- Michael Johanson, Neil Burch, Richard Valenzano, Michael Bowling (2013). Evaluating State-Space Abstractions in Extensive-Form Games. *Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*.
- Christian Kroer, Tuomas Sandholm (2014). Extensive-Form Game Abstraction with Bounds. *Proceedings of the ACM Conference on Economics and Computation (EC)*.
- Christian Kroer, Tuomas Sandholm (2016). Imperfect-Recall Abstractions with Bounds in Games. *Proceedings of the ACM Conference on Economics and Computation (EC)*.
- Christian Kroer, Tuomas Sandholm (2018). A Unified Framework for Extensive-Form Game Abstraction with Bounds. *Advances in Neural Information Processing Systems 31 (NeurIPS)*.
- Christian Kroer, Kevin Waugh, Fatma Kılınç-Karzan, Tuomas Sandholm (2020). Faster Algorithms for Extensive-Form Game Solving via Improved Smoothing Functions. *Mathematical Programming*, 179.
- Harold W. Kuhn (1953). Extensive Games and the Problem of Information. *Contributions to the Theory of Games II*, 28.
- Marc Lanctot, Kevin Waugh, Martin Zinkevich, Michael Bowling (2009). Monte Carlo Sampling for Regret Minimization in Extensive Games. *Advances in Neural Information Processing Systems 22 (NIPS)*.
- Marc Lanctot, Richard Gibson, Neil Burch, Martin Zinkevich, Michael Bowling (2012). No-Regret Learning in Extensive-Form Games with Imperfect Recall. *Proceedings of the 29th International Conference on Machine Learning (ICML)*.
- Chung-Wei Lee, Christian Kroer, Haipeng Luo (2021). Last-iterate Convergence in Extensive-Form Games. *Advances in Neural Information Processing Systems 34 (NeurIPS)*.
- Stephen McAleer, Gabriele Farina, Marc Lanctot, Tuomas Sandholm (2023). ESCHER: Eschewing Importance Sampling in Games by Computing a History Value Function to Estimate Regret. *International Conference on Learning Representations (ICLR)*.
- Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar, Georgios Piliouras (2019). Optimistic Mirror Descent in Saddle-Point Problems: Going the Extra (Gradient) Mile. *International Conference on Learning Representations (ICLR)*.
- Matej Moravčík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, Michael Bowling (2017). DeepStack: Expert-Level Artificial Intelligence in Heads-up No-Limit Poker. *Science*, 356(6337), 508–513.
- John Nash (1951). Non-Cooperative Games. *Annals of Mathematics*, 54(2), 286–295.
- Michele Piccione, Ariel Rubinstein (1997). On the Interpretation of Decision Problems with Imperfect Recall. *Games and Economic Behavior*, 20(1), 3–24.
- Julia Robinson (1951). An Iterative Method of Solving a Game. *Annals of Mathematics*, 54(2), 296–301.
- Martin Schmid, Neil Burch, Marc Lanctot, Matej Moravcik, Rudolf Kadlec, Michael Bowling (2019). Variance Reduction in Monte Carlo Counterfactual Regret Minimization (VR-MCCFR) for Extensive Form Games using Baselines. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, Christian Kroer (2023). A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games. *International Conference on Learning Representations (ICLR)*.
- Eric Steinberger (2019). Single Deep Counterfactual Regret Minimization. *arXiv:1901.07621*.
- Eric Steinberger, Adam Lerer, Noam Brown (2020). DREAM: Deep Regret Minimization with Advantage Baselines and Model-free Learning. *arXiv:2006.10410*.
- Oskari Tammelin (2014). Solving Large Imperfect Information Games Using CFR+. *arXiv:1407.5042*.
- Oskari Tammelin, Neil Burch, Michael Johanson, Michael Bowling (2015). Solving Heads-up Limit Texas Hold'em. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*.
- Bernhard von Stengel (1996). Efficient Computation of Behavior Strategies. *Games and Economic Behavior*, 14(2), 220–246.
- Kevin Waugh, David Schnizlein, Michael Bowling, Duane Szafron (2009). Abstraction Pathologies in Extensive Games. *Proceedings of the International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*.
- Kevin Waugh, Martin Zinkevich, Michael Johanson, Morgan Kan, David Schnizlein, Michael Bowling (2009). A Practical Use of Imperfect Recall. *Proceedings of the Symposium on Abstraction, Reformulation and Approximation (SARA)*.
- Kevin Waugh, Dustin Morrill, J. Andrew Bagnell, Michael Bowling (2015). Solving Games with Functional Regret Estimation. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Martin Zinkevich, Michael Johanson, Michael Bowling, Carmelo Piccione (2007). Regret Minimization in Games with Incomplete Information. *Advances in Neural Information Processing Systems 20 (NIPS)*.
