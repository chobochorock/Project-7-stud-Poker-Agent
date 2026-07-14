# 설계 및 구현 계보

이 문서는 게임, 에이전트, 학습 방법에 관한 아이디어와 실제 구현 결과를 날짜순으로 기록한다.

## 기록 규칙

- 새로운 아이디어는 `검토`, `결정`, `구현`, `보류` 중 하나로 표시한다.
- 구현 항목에는 변경 파일, 검증 방법, 추천 커밋 메시지를 함께 기록한다.
- 아직 구현하지 않은 내용을 구현된 기능처럼 적지 않는다.
- 폐기한 설계도 이유와 대안을 남긴다.

## 2026-07-15: 스택 없는 헤즈업 EV 학습 모드

상태: `검토`

### 목표

- 스택과 올인의 영향을 제거한 헤즈업 환경에서 카드와 베팅라인의 EV를 먼저 학습한다.
- 가능한 모든 상태를 테이블에 저장하지 않고, 표본 trajectory와 함수 근사기로 일반화한다.
- 상대의 신원은 입력에서 제외하고, 관측 가능한 정보와 암묵적 uniform range만 사용한다.

### 현재 결정 방향

- EV 모드는 헤즈업부터 시작한다.
- `ALL_IN`, `MAX`, `FULL`은 EV 모드에서 제외한다.
- 보상은 `최종 획득 칩 - 총투입 칩`인 선형 chip EV로 둔다.
- 할인은 적용하지 않고 `gamma = 1`로 둔다.
- 최초 베팅 직전의 무작위 4핸드 상태에서 rollout을 시작한다.
- 자신의 버린 카드는 알려진 dead card로 상태에 포함한다.
- 한 라운드에서 양쪽 플레이어 관점의 trajectory를 모두 저장한다.

### 베팅 상한 제안

- 한 베팅 라운드에서 `QUARTER`와 `HALF`를 합쳐 총 6회까지 허용하는 안을 우선 검토한다.
- `BBING`은 최초 오픈 액션으로 별도 계산한다.
- 상한에 도달하면 추가 레이즈 없이 `CALL` 또는 `FOLD`만 허용한다.
- 6회 상한이 실제 정책을 자주 제한하는지는 `raise_cap_reached` 통계로 확인한다.
- 상한 도달률이 의미 있게 높을 때만 8회로 확장한다.

매 베팅 라운드가 `BBING + HALF 6회 + CALL`로 끝나는 최악의 헤즈업 경로에서는 최종 팟이 약 67,641,472 ante까지 증가한다. 따라서 EV 모드는 유한 스택을 가장하지 않고 완전히 stackless로 처리해야 한다.

### 보상 처리

- reward clipping은 큰 팟과 작은 팟의 차이를 없애므로 사용하지 않는다.
- `sign(x) * log(1 + abs(x))`도 위험 선호를 바꾸므로 EV 모드 보상으로 사용하지 않는다.
- 학습 안정화가 필요하면 모든 보상을 같은 양의 상수로 나눈다. 이는 행동의 우열을 보존한다.
- 추가 안정화는 reward 변형보다 Huber loss와 gradient clipping을 우선한다.
- 로그 효용은 유한 bankroll의 성장률을 최적화하는 별도 스택 게임에서만 다시 검토한다.

### Fold equity와 fold regret

상대의 fold equity 성분은 다음처럼 액션 가치에 포함된다.

`Q(s, raise) = P(fold) * V(fold 결과) + P(continue) * V(계속 진행)`

상대의 `FOLD`가 rollout에 포함되어 있으면 이 값은 Q 추정치에 자동으로 반영되므로 별도의 fold-equity 변수를 반드시 만들 필요는 없다. 다만 uniform action 상대에게서 측정된 `P(fold)`는 실제 전략적 압박이 아니라 uniform 정책의 성질일 뿐이다.

`max_a Q(s, a) - Q(s, FOLD)`는 자신이 폴드하여 포기한 가치이며 fold equity가 아니라 fold regret 또는 기회비용이다. 해석용 UI가 필요하면 두 값을 따로 표시한다.

### Sweep와 value iteration의 구분

- 알려진 전이확률로 모든 상태에 Bellman backup을 적용하면 value iteration의 한 sweep이다.
- 한 sweep은 value iteration의 한 반복일 뿐이며 일반적으로 수렴을 뜻하지 않는다.
- 유한 비순환 게임 트리를 terminal부터 역순으로 정확히 계산할 때만 한 번의 backward pass로 해를 구할 수 있다.
- 무작위 rollout 결과를 terminal return으로 한 번 업데이트하는 것은 Monte Carlo batch update이다.
- replay 데이터에 `r + max Q(s', a')` target을 반복 적용하면 fitted Q iteration에 가깝다.
- 불완전정보 게임에서 uniform 상대를 환경으로 고정하면 그 상대에 대한 best response를 학습한다. 이것만으로 균형 정책이 보장되지는 않는다.

### 첫 학습기 제안

1. 실제 딜링 규칙으로 무작위 첫 베팅 상태를 생성한다.
2. 관측에는 자신의 히든/공개/버린 카드, 상대 공개 카드, 팟, 콜 비용, street, 베팅 기록과 남은 레이즈 횟수를 넣는다.
3. 상대 히든 카드와 버린 카드는 남은 덱에서 중복 없이 uniform sampling한다.
4. 합법 액션으로 terminal까지 rollout한다.
5. 양쪽 플레이어의 각 의사결정에 해당 관점의 terminal net chip을 저장한다.
6. 처음에는 full-return Monte Carlo로 학습한다. 이는 episodic `TD(lambda=1)`에 해당한다.
7. 분산이나 학습 속도가 실제 문제가 될 때 n-step TD를 추가한다. eligibility trace는 그 이후에 검토한다.

첫 구현은 변형 없는 uniform action rollout으로 둔다. `FOLD` 때문에 짧은 라인이 많이 생길 수 있으므로 `(street, raise_depth, action)` 방문 횟수만 기록한다. 깊은 라인의 coverage 부족이 실제로 확인될 때만 목표 레이즈 깊이를 먼저 뽑는 보정 sampler를 추가한다.

### 한 라운드에서 두 관점 저장

- 헤즈업의 같은 라운드에서 두 플레이어의 trajectory를 모두 저장할 수 있다.
- 각 상태는 항상 행동하는 플레이어를 `self`로 보는 정규화된 관점으로 기록한다.
- 상대 히든 카드는 해당 플레이어의 입력에 유출하지 않는다.
- terminal reward는 칩 보존이 성립할 때 `G_A = -G_B`이다.
- 두 trajectory는 독립 표본은 아니지만, 공유 모델을 학습하는 데 모두 사용할 수 있다.

### 하드 버킷 없이 시작하는 방법

모든 상태를 저장할 수 없다는 사실이 곧 하드 버킷이 필수라는 뜻은 아니다. 신경망 자체가 비슷한 상태를 연속적으로 일반화할 수 있다. 첫 구현에서는 다음 순서를 사용한다.

1. 문양 이름을 최초 등장 순서로 치환해 동등한 문양 순열을 손실 없이 canonicalize한다.
2. 카드 역할을 구분한 원본 표현을 유지한다: 내 히든, 내 공개, 내 버린 카드, 상대 공개 카드.
3. `agent/HA1.py`의 uniform joint-range Monte Carlo equity를 연속 특징으로 재사용한다.
4. 동일한 canonical 정보상태에는 안정적인 seed를 사용해 equity 특징의 재현성을 높인다.
5. 팟 크기, pot odds, street, 액션 순서와 레이즈 깊이는 별도 특징으로 넣는다.

uniform equity 하나만으로 상태를 대체하지는 않는다. 같은 equity라도 draw 구조와 베팅라인에 따라 행동 가치가 다를 수 있으므로 원본 카드 표현과 함께 사용한다. 하드 버킷은 메모리나 학습 안정성 문제가 측정된 뒤, street별 equity quantile 방식으로만 검토한다.

### 미결정 사항

- 6회 상한을 `BBING`을 제외한 레이즈 횟수로 확정할지 여부
- coverage rollout에서 목표 레이즈 깊이를 어떤 분포로 뽑을지
- 고정 reward scale의 값
- uniform 상대 이후 self-play 또는 CFR 계열로 넘어가는 시점
- trajectory 저장 형식과 최대 replay 크기

### 참고 자료

- [Reinforcement Learning: An Introduction](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf): Monte Carlo, exploring starts, TD 방법
- [Q-Learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html): 반복적인 상태-행동 방문을 전제로 한 tabular Q-learning 수렴 조건
- [Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html): 포커와 같은 불완전정보 게임의 CFR
- [A New Interpretation of Information Rate](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x): 유한 bankroll 성장률과 로그 효용

### 구현 시 추천 커밋 메시지

`feat: add stackless heads-up EV rollout mode`
