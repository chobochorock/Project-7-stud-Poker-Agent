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

1. 네 문양의 24가지 전역 치환 중 카드 배열이 사전순으로 가장 작은 표현을 골라, 동등한 문양 순열을 손실 없이 canonicalize한다.
2. 카드 역할을 구분한 원본 표현을 유지한다: 내 히든, 내 공개, 내 버린 카드, 상대 공개 카드.
3. `agent/HA1.py`의 uniform joint-range Monte Carlo equity를 연속 특징으로 재사용한다.
4. 동일한 canonical 정보상태에는 안정적인 seed를 사용해 equity 특징의 재현성을 높인다.
5. 팟 크기, pot odds, street, 액션 순서와 레이즈 깊이는 별도 특징으로 넣는다.

uniform equity 하나만으로 상태를 대체하지는 않는다. 같은 equity라도 draw 구조와 베팅라인에 따라 행동 가치가 다를 수 있으므로 원본 카드 표현과 함께 사용한다. 하드 버킷은 메모리나 학습 안정성 문제가 측정된 뒤, street별 equity quantile 방식으로만 검토한다.

### 미결정 사항

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

## 2026-07-15: 균등 rollout 실험 1차

상태: `구현`

### 구현 결과

- `poker_env.py`에 헤즈업 전용 `ev` 모드를 추가했다.
- EV 장부는 0에서 시작해 베팅액만큼 음수가 되고 정산액을 더한다. 따라서 스택이나 올인 없이 최종 chip net을 직접 얻는다.
- `BBING`을 제외한 `QUARTER/HALF` 합계 6회를 street별 상한으로 적용했다.
- `ev_rollout.py`에서 첫 4핸드, discard/reveal, 이후 모든 합법 액션을 균등 표본화한다.
- 양쪽 플레이어의 관점을 `self` 기준으로 canonicalize해 모두 저장한다.
- 진단용 SQLite의 `q_values`에는 방문 수와 terminal return 합계를, `coverage`에는 street/레이즈 깊이/액션 방문 수를 저장한다.

### 30초 실측

- 실행 시간: 30.17초
- 처리량: 17,750핸드, 평균 588.27핸드/초
- 상태-액션 표본: 76,414개, 평균 4.305개/핸드
- 고유 상태-액션: 76,414개
- 재방문된 상태-액션: 0개
- SQLite 크기: 9,707,520바이트, 약 9.26MiB
- 고유 상태-액션당 평균: 약 127.04바이트

30초 결과를 단순히 60배 한 30분 선형 예상치는 약 1,065,000핸드, 4,584,840개 상태-액션, 555MiB이다. 다만 SQLite 인덱스가 커지며 처리량이 감소했으므로 실제 30분 결과는 이보다 작을 가능성이 높다.

street별 표본은 4th 46,668개, 5th 19,099개, 6th 7,653개, 7th hidden 2,994개였다. 균등 액션에 `FOLD`가 포함되므로 뒤 street coverage가 빠르게 줄어든다는 예상도 확인됐다.

### 결론

정확 상태 키는 짧은 실험에서도 표본마다 새로 나타났다. 이 테이블을 최종 모델로 키우지 않고, 다음 단계에서는 원본 canonical 카드 표현과 연속 equity 특징을 받는 함수 근사기를 사용한다. 하드 버킷은 아직 도입하지 않는다.

### 검증

- `python -B -m py_compile poker_env.py web_controller.py ev_rollout.py test_poker_env.py`
- `python -B -m unittest -v test_poker_env.py test_web_controller.py`
- 총 33개 테스트 통과

### 추천 커밋 메시지

`feat: add stackless EV rollout benchmark`

## 2026-07-15: rollout 저장 포맷과 용량 제한

상태: `구현`

### 저장 포맷 version 1

- 각 `state_json` 숫자 배열의 첫 값으로 `schema_version = 1`을 저장한다.
- `decode_state()`는 숫자 배열을 street, 카드 역할, 팟, 투자액, 콜 비용, 레이즈 횟수와 베팅 기록이 있는 딕셔너리로 복원한다.
- 기존 30초 실험의 version 0 배열도 계속 읽는다.
- canonical suit는 `u0`부터 `u3`으로 표시한다. 원래 `s/h/d/c` 이름은 의도적으로 복원하지 않지만 같은 문양 관계와 플러시 구조는 보존한다.
- `--inspect {DB} --limit N`으로 SQLite 행을 사람이 읽을 수 있는 JSON으로 출력한다.

### 1GiB 제한

- SQLite 파일은 1GiB를 미리 할당하지 않고 필요한 만큼 자동으로 증가한다.
- `--max-gib 1`은 SQLite 본체, WAL, SHM 파일의 실행 중 합계를 확인한다.
- 용량 검사는 flush 단위로 수행되므로 최종 파일은 한 batch 이내에서 제한을 조금 넘을 수 있다.
- 장시간 실행은 시간 제한과 용량 제한 중 먼저 도달한 조건에서 종료한다.

### 야간 실행 명령

```powershell
python -B ev_rollout.py --seconds 28800 --max-gib 1 --output replays/ev_rollout_overnight_1g.sqlite3
```

### 추천 커밋 메시지

`feat: version and inspect EV rollout tables`

## 2026-07-15: 8시간 균등 rollout 실측

상태: `구현`

### 결과

- 실행 시간: 28,803.29초, 약 8시간
- 종료 조건: 시간 제한
- 핸드: 1,469,250
- 상태-액션 표본: 6,254,259개
- 고유 상태-액션: 6,254,235개
- 재방문: 24회, 재방문율 약 0.000384%
- 한 상태-액션의 최대 방문 수: 2회
- SQLite 크기: 797,753,344바이트, 약 760.8MiB
- 평균 처리량: 51.01핸드/초
- 상태-액션당 평균: 약 127.55바이트

1GiB까지는 약 263.2MiB가 남았다. 현재 행 밀도가 유지되면 약 216만 행, 50.8만 핸드가 추가로 필요하지만, 커진 인덱스 때문에 실제 소요 시간은 선형 예상보다 길 수 있다.

### Coverage

- 4th: 3,848,788개, 약 61.5%
- 5th: 1,541,863개, 약 24.7%
- 6th: 617,703개, 약 9.9%
- 7th hidden: 245,905개, 약 3.9%
- 레이즈 깊이 6: 44,638개

### 해석

30초 빈 테이블에서는 평균 588.27핸드/초였지만 8시간 전체 평균은 51.01핸드/초로 낮아졌다. `state_json + action`이라는 긴 primary key를 가진 B-tree에 거의 전부 새로운 행을 삽입하면서 인덱스 탐색, 페이지 분할과 디스크 쓰기 비용이 커진 것이 주된 원인이다.

이 실험은 1GiB를 채우지 않았지만 목적에는 충분하다. 정확 상태 테이블은 반복 방문으로 평균값을 개선하지 못하고 거의 append-only 데이터셋처럼 증가한다. 동일 방식의 추가 수집보다 현재 625만 표본으로 첫 함수 근사기를 학습하고, 검증 결과에 따라 데이터를 더 모으는 편이 낫다.

### 상태 수에 관한 정정

이전에 계산한 155,937,600은 전체 게임 핸드 수가 아니다.

`C(52, 4) * 4 * 3 * 48 = 155,937,600`

이는 한 플레이어의 최초 4장, 버릴 카드, 공개할 카드, 다음 공개 카드까지 구분한 원시 초기 카드 이력 수다. 상대 카드, 이후 공개/히든 카드와 베팅라인을 포함한 전체 헤즈업 trajectory 공간은 이보다 훨씬 크다.

### 다음 단계

- 현재 DB를 더 키우지 않는다.
- exact-state Q 테이블 대신 canonical 카드 입력과 연속 equity 특징을 받는 함수 근사기를 학습한다.
- uniform fold로 인한 street 편향은 첫 모델 평가 후 필요할 때만 depth-balanced sampler로 보정한다.

## 문양 대칭성 압축과 버킷 구분 (2026-07-15)

문양에 우열이 없는 현재 규칙에서는 `s/h/d/c`를 `a/b/c/d`로 일관되게 치환해도 게임 상태와 보상은 변하지 않는다. 이 치환은 서로 다른 상태를 근사적으로 합치는 버킷이 아니라, 동일한 상태의 이름만 바꾸는 손실 없는 canonicalization이다.

현재 `ev_rollout.py`는 내 히든 카드, 내 공개 카드, 버린 카드와 상대 공개 카드 전체에 동일한 문양 치환을 적용한다. 가능한 24가지 치환 중 사전순 최소 표현을 선택하므로, 특정 문양을 높은 랭크 순으로 `a/b/c/d`에 배정하는 별도 규칙을 추가할 필요가 없다. 사용된 문양 수와 카드 구조에 따라 이론적 원시 상태 공간은 최대 24배 줄지만, 표본이 매우 희소하므로 이미 수집한 DB 파일의 행 수가 즉시 같은 비율로 줄어드는 것은 아니다.

따라서 다음 단계에서도 하드 버킷은 보류한다. 현재 canonical 카드 표현을 그대로 함수 근사기에 넣어 일반화를 먼저 측정한다. 정확 Q 테이블을 계속 사용해 상태 재방문을 확보하려는 경우에만 equity, draw 구조와 베팅 문맥을 포함한 근사 버킷을 검토한다.

## 다음 단계 결정: 압축보다 첫 함수 근사기 (2026-07-15)

현재 SQLite는 최종 학습 저장 포맷이 아니라 exact-state 테이블의 한계를 확인하기 위한 진단용 구현이다. 따라서 가능한 무손실 압축을 모두 적용한 상태는 아니다.

### 남아 있는 무손실 정리

- 최초 히든 두 장의 배분 순서는 게임 의미가 없으므로 정렬해 최대 2배의 중복을 더 없앨 수 있다.
- EV 실험의 고정 ante는 파일 metadata에 한 번만 기록하면 된다.
- 헤즈업에서 `pot = my_invested + opponent_invested`, `call_amount = max(0, opponent_round_bet - my_round_bet)`이며 `raise_count`도 현재 street의 행동 기록에서 계산할 수 있다.
- 더 나아가 고정 ante와 현재 EV 규칙에서는 `street + actor + action` 순서만으로 매 행동의 콜, 레이즈, 투자액과 팟을 `ceil` 규칙까지 동일하게 재생할 수 있으므로 상태 키의 모든 베팅 금액은 중복이다.
- 카드, 행동과 베팅 기록을 JSON/TEXT 대신 작은 정수 BLOB으로 저장하면 파일과 인덱스를 줄일 수 있다.
- 재방문율이 거의 0이므로 최종 학습 데이터는 unique primary key와 UPSERT보다 append-only 압축 batch가 적합하다.

앞의 항목들은 상태 의미를 보존하지만 대부분 저장 공간과 생성 속도만 개선한다. 랭크, 드로 구조나 equity가 비슷한 상태를 합치기 시작하면 그때부터는 손실이 있는 하드 버킷이다.

### 학습 타깃 보정

현재 DB는 각 의사결정에 핸드 전체의 최종 순손익을 붙였다. 의사결정 이전 투자액은 이미 매몰비용이므로 기존 데이터에서 다음과 같이 시점 이후 수익을 계산한다.

`future_return = terminal_net + my_invested`

학습 시에는 이를 ante로 나눠 reward scale을 고정한다. 이 보정은 기존 DB의 `my_invested`로 가능하므로 rollout을 다시 실행할 필요가 없다.

### 결정

1. 기존 8시간 DB는 수정하거나 추가 수집하지 않는다.
2. 하드 버킷 없이 canonical 카드, 베팅 기록과 정규화된 연속 수치를 입력받는 작은 함수 근사기를 먼저 학습한다.
3. 동일 DB의 무작위 행 분할 대신 별도 seed로 만든 작은 DB를 검증셋으로 사용한다.
4. 현재 표본의 타깃은 이후 행동도 균등 정책인 `Q^uniform`이므로, 첫 모델은 최종해가 아니라 한 번의 정책 개선 출발점으로 취급한다.
5. 검증에서 일반화가 부족할 때만 equity 특징 또는 하드 버킷을 추가한다.

## sparse 상태에 대한 정보집합 MCTS 검토 (2026-07-15)

상태: `설계 검토`

전역 exact-state 테이블은 거의 재방문되지 않지만, 탐색은 현재 정보집합의 각 액션에 표본을 집중해 상대 히든과 이후 카드를 주변화할 수 있다. 헤즈업, 정확한 게임 시뮬레이터와 6회 레이즈 상한이 있다는 점도 온라인 탐색에 유리하다.

일반 MCTS에서 매 simulation마다 상대 히든을 하나 확정한 뒤 실제 상태 노드를 공유하면, 같은 정보집합에서 확정된 히든에 따라 다른 결정을 내리는 strategy fusion과 비공개 정보 누출이 발생한다. 따라서 현재 게임에는 vanilla MCTS가 아니라 다음 최소 구조가 적합하다.

1. 관측된 카드와 dead card에 모순되지 않는 상대 히든, 상대 버린 카드와 미래 덱을 uniform belief에서 표본화한다.
2. 트리는 실제 전체 상태가 아니라 canonical 정보집합과 공개 행동 history를 키로 사용한다.
3. root-sampling ISMCTS로 현재 액션별 방문 수와 시점 이후 EV를 계산한다.
4. 첫 rollout policy는 이미 있는 `HA1`을 재사용한다.
5. root 방문 분포와 EV를 append-only 학습 표본으로 저장해 작은 policy/value 모델에 증류한다.

이 방식은 MCTS를 최종 저장 모델로 쓰는 것이 아니라 국소적인 정책 개선 교사로 사용한다. search 없이 빠르게 행동해야 할 때는 증류 모델을 사용하고, 계산 여유가 있을 때만 그 모델을 prior 또는 leaf value로 다시 탐색에 넣는다.

현재 agent 인터페이스에는 관측 상태만 전달되므로, 실제 상대 히든이나 덱을 노출하지 않는 정보집합용 generative simulator가 먼저 필요하다. 초기 구현은 uniform belief 기반 ISMCTS와 HA1 rollout까지만 만든다. 상대 행동 likelihood를 이용한 belief 갱신, ReBeL, MCCFR과 equilibrium 보장은 실측 후로 미룬다. ISMCTS는 유용한 휴리스틱이지만 자체적으로 Nash 전략을 보장하지 않으며, 순수 MCTS 에이전트는 상대 모델에 대한 best response로 치우칠 수 있다.

### 갱신된 우선순위

1. 기존 exact-state DB 추가 수집과 하드 버킷 설계를 중단한다.
2. uniform belief 정보집합 sampler와 ISMCTS baseline을 구현해 HA1과 헤즈업 대전한다.
3. 탐색 시간, simulation 수, 액션별 분산과 승률을 측정한다.
4. 탐색 결과를 저장한 뒤 policy/value 모델 증류 여부를 결정한다.

### 근거

- Cowling, Powley, Whitehouse (2012), Information Set Monte Carlo Tree Search: https://doi.org/10.1109/TCIAIG.2012.2200894
- Silver, Veness (2010), Monte-Carlo Planning in Large POMDPs: https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html
- Brown et al. (2020), Combining Deep Reinforcement Learning and Search for Imperfect-Information Games: https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html

## uniform 상대 핸드 테이블 1차 구현 (2026-07-15)

상태: `구현`

### 범위와 가정

- 헤즈업에서 상대의 초반 히든 2장 가능한 조합을 모두 열거한다.
- 내 활성 카드, 내 discarded card와 상대 공개 카드는 known/dead card로 제외한다.
- 각 히든 2장 조합의 prior는 동일하다.
- 상대의 알려지지 않은 discarded card, 아직 배분되지 않은 공개 카드와 마지막 히든은 각 조합 안에서 중복 없이 Monte Carlo 표본화한다.
- 상대 신원과 베팅·discard/reveal 행동 likelihood는 사용하지 않는 `uniform_card_only` belief다.

### 출력

`agent/hand_range.py`의 `estimate_uniform_hand_range()`는 다음을 반환한다.

- 가능한 히든 2장 조합 수와 조합별 균등 확률
- 조합별 승·무·패 확률과 split을 반영한 equity
- 전체 승·무·패, equity와 근사 95% Monte Carlo 구간
- 상대 최종 족보별 확률
- 실제 completion 표본 수

`HA1PokerAgent.estimate_hand_range()`에서 같은 계산기를 필요할 때 호출할 수 있다. 기존 `HA1.choose_action()`은 매 행동 응답 속도를 위해 기존 128회 joint Monte Carlo equity를 계속 사용한다.

### 4구 상태 실측

- 내 패: 히든 `h6, s5`, 공개 `dQ, d10`, 버린 카드 `h9`
- 상대 공개: `sJ, h5`
- 가능한 상대 히든 2장: 990조합
- 조합당 completion: 16회, 총 15,840표본
- 승 43.0114%, 무 0.0063%, 패 56.9823%
- split equity: 43.0145%
- 근사 95% Monte Carlo 구간: 42.2633~43.7657%
- 실행 시간: 3.02초

상대 최종 족보 분포는 하이카드 20.53%, 원페어 46.09%, 투페어 21.84%, 트리플 4.02%, 스트레이트 3.88%, 플러시 1.69%, 풀하우스 1.85%, 포카드 0.09%, 스트레이트 플러시 0.01%였다.

조합당 completion을 64회로 늘린 교차 검증에서는 총 63,360표본, equity 42.5213%, 근사 95% 구간 42.1461~42.8965%, 실행 시간 12.22초였다. 기존 HA1 joint Monte Carlo 16,384회의 equity 42.2394%가 이 구간 안에 들어 두 계산 방식의 일관성을 확인했다. 기본 16회는 약 3초에 이 상태 기준 오차폭 약 ±0.75%p인 빠른 분석 설정으로 유지한다.

### 해석과 다음 단계

전역 상태를 버킷화하지 않고도 현재 정보집합에서 사용할 명시적인 belief particle 집합을 확보했다. 계산 비용상 매 행동의 기본 휴리스틱으로 쓰기보다는 분석 UI와 ISMCTS root sampling에 사용한다. 다음 단계에서는 이 균등 table에서 상대 히든과 나머지 deal을 표본화하는 정보집합용 generative simulator를 만든다.

### 검증

- known/dead card가 모든 상대 히든 조합에서 제외됨을 확인했다.
- 7구 확정 포카드 상태에서 780개 조합, 1,560 completion의 승률과 equity가 모두 100%였다.
- 전체 34개 테스트가 통과했다.

### 추천 커밋 메시지

`feat: add uniform opponent hand range table`

## 상대 핸드 레인지 격자 UI (2026-07-15)

상태: `구현`

홀덤식 13×13 rank 표는 간결하지만 현재 게임에서는 문양별 blocker와 known/dead card를 숨긴다. 분석 화면은 정보 손실을 피하기 위해 52장의 실제 카드를 양 축으로 둔 52×52 상삼각 격자를 사용한다. 같은 두 카드의 순서는 의미가 없으므로 하삼각은 만들지 않는다.

- 헤즈업의 4구, 5구, 6구와 마지막 히든 거리에서 관측자를 선택할 수 있다.
- `Calculate`를 눌렀을 때만 `estimate_uniform_hand_range()`를 호출한다. 기본 설정은 조합당 16표본이며 실측 약 3초가 걸리므로 게임 상태 갱신 때 자동 재계산하지 않는다.
- 각 유효 칸은 상대의 초반 히든 2장과 그 조건에서의 내 equity를 나타낸다. 붉은색에서 녹색으로 갈수록 equity가 높다.
- known/dead card가 들어간 조합은 회색으로 비활성화한다.
- 칸을 선택하면 조합별 승·무·패, equity와 균등 prior를 표시한다.
- 전체 승·무·패, split equity, 근사 95% 구간, 조합/표본 수와 상대 최종 족보 분포를 함께 표시한다.
- `web_controller.py`가 관측자 전용 상태를 계산기에 전달하고 `POST /api/hand_range`가 결과를 반환한다. 게임 진행과 HA1 행동 정책은 바꾸지 않았다.

데스크톱과 390×844 모바일 화면을 검수했다. 모바일에서는 52×52 격자만 내부 스크롤되며 페이지 폭을 넓히지 않는다. 브라우저 콘솔 오류는 없었다.

### 검증

- 전체 35개 테스트 통과
- `node --check web/static/app.js` 통과
- 웹 컨트롤러 테스트에서 4구 상태의 990개 조합과 확률 합을 확인

### 추천 커밋 메시지

`feat: add hand range heatmap to web UI`

## MCTS의 장기 전략 한계와 CFR 대안 검토 (2026-07-15)

상태: `설계 검토`

### 장기 이득의 구분

MCTS가 한 에피소드에만 greedy하다는 표현은 다음 세 목표를 나누어야 정확하다.

1. 서로 독립인 캐시 라운드에서 순칩 보상이 단순 합산된다면, 매 라운드의 기대 순칩을 최대화하는 정책은 장기 기대 순칩도 최대화한다. 이 경우 라운드 사이에 추가로 계획할 상태가 없으므로 단기 EV 최적화 자체는 문제가 아니다.
2. 탐색 트리와 결과를 매번 버리는 순수 MCTS는 여러 라운드에서 얻은 지식을 지속 정책으로 축적하지 못한다. 이는 탐색 결과를 policy/value 모델이나 평균 전략에 저장하면 해결할 수 있다.
3. 고정된 상대 모델에 대한 MCTS best response는 그 상대에게는 강해도 다른 상대에게 exploit될 수 있다. 장기적으로 쉽게 exploit되지 않는 혼합전략을 원하는 문제는 탐색 깊이가 아니라 equilibrium 학습 문제다.

스택, 파산 확률, 상대의 적응과 과거 행동이 다음 라운드 보상에 영향을 준다면 그 정보까지 상태에 넣어야 한다. 현재 스택 없는 EV 게임에서는 그런 라운드 간 상태를 의도적으로 제거했으므로 보상은 라운드 순칩의 합으로 충분하다.

### CFR만 가능한가

아니다. 선택지는 다음과 같다.

- `탐색 + 증류`: 정보집합 MCTS의 root action 분포와 EV를 저장해 지속 policy/value 모델을 학습한다. 구현이 가장 작지만 equilibrium 보장은 없다.
- `NFSP/FSP`: 현재 best response와 과거 평균 전략을 함께 학습한다. 탐색 결과를 장기 혼합전략으로 만드는 비교적 직관적인 대안이다.
- `PSRO`: 여러 정책의 집합과 상대 조합별 payoff를 유지하고 새 best response를 추가한다. 상대 풀이 다양하거나 다인전으로 갈 때 CFR보다 자연스럽지만 운영 비용이 크다.
- `MCCFR/DCFR/Deep CFR`: counterfactual regret를 최소화하여 헤즈업 제로섬 게임의 평균 전략을 근사 equilibrium으로 보낸다. 현재처럼 상태 공간이 큰 게임에서는 전수 tabular CFR보다 external/outcome-sampling MCCFR 또는 함수 근사형 Deep CFR가 후보가 된다.
- `ReBeL 계열`: public belief state에서 search와 self-play value 학습을 결합한다. 현재 hand range와 가장 잘 이어지지만 첫 구현으로는 가장 복잡하다.

### 현재 프로젝트의 최소 경로

1. uniform hand range를 재사용하는 정보집합 generative simulator를 먼저 만든다.
2. ISMCTS를 국소 정책 개선 교사로 사용하고 root action 분포, 시점 이후 순칩 EV와 정보집합 표현만 저장한다.
3. 이 표본으로 작은 policy/value 모델을 학습해 탐색 결과가 라운드 사이에도 남게 한다.
4. HA1, random, 서로 다른 탐색 예산의 에이전트로 상대 population을 만들어 교차 평가한다.
5. 학습 정책이 특정 상대에 과적합되거나 순환적으로 패하면, 헤즈업 EV 게임에 external-sampling MCCFR 기준선을 추가한다.
6. tabular regret의 재방문율과 메모리 사용량을 먼저 측정하고, 부족할 때만 Deep CFR의 advantage network와 average-policy network를 도입한다.

따라서 지금 즉시 Deep CFR 전체 구조를 만드는 것은 이르다. 먼저 `ISMCTS teacher -> persistent policy/value`가 장기 지식 축적 문제를 해결하는지 확인하고, equilibrium 안정성이 실제 병목일 때 sampled CFR로 넘어간다. CFR은 강한 기준선이지만 장기 학습을 가능하게 하는 유일한 방법은 아니다.

### 근거

- Lanctot et al. (2009), Monte Carlo Sampling for Regret Minimization in Extensive Games: https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html
- Heinrich, Lanctot, Silver (2015), Fictitious Self-Play in Extensive-Form Games: https://proceedings.mlr.press/v37/heinrich15.html
- Brown et al. (2019), Deep Counterfactual Regret Minimization: https://proceedings.mlr.press/v97/brown19b.html
- Brown et al. (2020), Combining Deep Reinforcement Learning and Search for Imperfect-Information Games: https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html

### 추천 커밋 메시지

`docs: compare persistent search and CFR training paths`

## UCT가 선별한 정보집합만 저장하는 방안 (2026-07-15)

상태: `설계 후보`

모든 실제 관측 상태를 저장하지 않고, 정보집합 MCTS에서 충분한 방문 통계가 쌓인 노드만 학습 표본으로 저장한다. 이 방식은 무작위 rollout 경로 전체를 보존하는 현재 exact-state DB보다 훨씬 적은 수의 고품질 target을 만든다.

다만 일반 UCT는 노드를 처음 방문할 때 바로 확장할 수 있으므로 단순히 `노드가 생성됨`을 저장 기준으로 삼으면 압축 효과가 없다. 저장 자격을 별도로 둔다.

- 노드 키는 determinization에서 표본화한 실제 상대 히든이 아니라 canonical 관측 상태와 public belief 요약으로 만든다.
- 탐색 트리는 메모리에서만 사용하고 영구 저장하지 않는다.
- 총 방문 수가 최소 기준을 넘은 root 또는 내부 정보집합 노드만 학습 표본으로 승격한다.
- raw rollout과 child 객체 대신 합법 행동 mask, 행동별 방문 수, 평균 시점 이후 순칩 EV, 보상 제곱합 또는 분산, 전체 방문 수와 탐색 정책 버전만 저장한다.
- 정책 target은 `N(s,a) / N(s)`, value target은 해당 정보집합의 평균 시점 이후 순칩 EV로 만든다.
- 동일 canonical 정보집합은 새 행을 계속 추가하기보다 모델 버전별 집계 또는 제한된 최근 표본으로 관리한다.

현재 행동 수는 최대 7개라 action branching은 크지 않다. 큰 분기는 상대 히든, discard와 미래 deal 같은 chance 변수에서 생긴다. 따라서 sampled hidden hand마다 별도 영구 노드를 만들지 않고 root-sampling한 결과를 같은 정보집합 노드의 통계로 합쳐야 한다. 필요하면 chance outcome에만 progressive widening을 적용한다.

### 저장 자격의 최소안

첫 실험에서는 다음 정도만 둔다.

```text
eligible = total_visits >= 256 and legal_actions_sampled >= 2
```

256은 확정값이 아니라 측정용 시작점이다. 저장 후에는 action별 방문 수가 적은 표본에 낮은 학습 가중치를 준다. 모든 행동을 같은 횟수만큼 방문하도록 강제하면 UCT의 이점을 잃으므로 `각 행동 8회 이상` 같은 강한 조건은 처음부터 두지 않는다.

### 선택 편향과 보완

UCT가 자주 도달하는 상태만 저장하면 현재 정책이 좋아하는 line에 표본이 몰리고 드물지만 중요한 상태가 사라질 수 있다. 이를 막기 위해 전체 상태 DB를 다시 만들기보다 다음 두 장치만 유지한다.

1. exploring start로 거리와 초기 카드·베팅 문맥을 무작위화한다.
2. 저장 자격을 충족한 표본은 고정 크기 reservoir에 넣어 오래 실행해도 파일 크기가 무한히 늘지 않게 한다.

이 데이터는 UCT가 만든 search-policy/value target이지 counterfactual regret가 아니다. 따라서 정책 증류에는 적합하지만 equilibrium 보장은 없으며, exploitability가 실제 문제로 확인되면 같은 정보집합 표현을 MCCFR의 regret target 저장에도 재사용한다.

### 권장 저장 단위

한 행은 한 rollout이나 한 action이 아니라 한 정보집합 노드다. 행동별 배열을 한 행에 묶으면 최대 7개 action의 통계를 함께 읽을 수 있고 중복 상태 키도 줄어든다. 첫 구현은 고정 크기 replay buffer와 간단한 decoder만 만들고 전체 MCTS tree 직렬화, 영구 transposition table과 우선순위 replay는 보류한다.

### 추천 커밋 메시지

`docs: define UCT-gated search replay storage`

## UCT 집중 탐색과 observ2vec clustering (2026-07-15)

상태: `설계 후보`

### coverage의 재정의

모든 exact observation을 저장하는 coverage는 현재 게임에서 실용적이지 않다. 이후 coverage는 다음 두 값으로 측정한다.

1. `cluster coverage`: 전략적으로 비슷한 observation 군집 중 몇 개를 방문했는가.
2. `cluster-action coverage`: 각 군집에서 합법 행동이 최소한 표본화되었는가.

UCT는 유망한 베팅 line에 방문을 집중하되 각 합법 행동을 처음 몇 회는 강제 표본화한다. 거리와 초기 카드·베팅 문맥은 exploring start로 넓게 생성하고, 이후 탐색 예산은 UCT가 배분한다. 이 조합이면 모든 exact state를 저장하지 않으면서도 현재 표현이 구분하는 군집과 행동의 누락을 계측할 수 있다.

### EV 하한과 clipping

보상 자체를 임의 하한으로 clipping하면 Q의 크기와 행동 간 순서를 왜곡하고 원래 chip EV 목적함수를 바꾼다. 특히 표본이 적을 때 우연히 낮게 나온 행동을 영구 제거할 위험이 있다.

현재 `future_return = terminal_net + my_invested` 정의에서는 fold의 시점 이후 EV를 0으로 둘 수 있다. 따라서 낮은 EV line은 값 clipping보다 신뢰구간으로 제거한다.

```text
최소 방문 수를 먼저 보장한다.
UCB(action) < LCB(fold)일 때만 해당 action의 추가 탐색 우선순위를 낮춘다.
raw mean, visit count와 return variance는 보존한다.
```

신경망 학습의 큰 손익 때문에 gradient가 불안정하면 raw target을 지우는 clipping 대신 ante 또는 pot 기준 정규화와 Huber loss를 먼저 사용한다. action별 `sum_return`과 `sum_squared_return`은 작은 고정 길이 숫자 배열이므로 이를 버려도 저장량 이득이 거의 없고, 오히려 탐색 신뢰도와 재분석 능력을 잃는다.

### observ2vec

`observ2vec`는 이 프로젝트에서 사용할 observation encoder의 작업명으로 둔다. 일반적으로 확립된 단일 알고리즘 이름이라기보다 다음 지도학습 표현을 뜻한다.

- 입력: canonical 카드, 거리, 공개/비공개 구분, legal-action mask, pot·call·베팅 이력, 행동 순서와 uniform belief 요약
- 출력: 우선 16차원 또는 32차원 embedding
- 정책 target: MCTS의 `N(s,a) / N(s)`
- 가치 target: MCTS의 평균 시점 이후 chip EV
- 보조 target: 상대 최종 족보 분포 또는 다음 거리의 belief 요약

카드 원문을 복원하는 autoencoder만 사용하면 전략적으로 무관한 표면 유사성을 학습할 수 있다. 정책 분포와 value가 비슷한 observation을 가깝게 만드는 것이 핵심이다. reward와 다음 latent-state 분포를 함께 예측해 행동 결과가 비슷한 상태를 묶는 DeepMDP/bisimulation 계열 목적은 두 번째 실험 후보로 둔다.

### clustering 절차

1. 거리, 생존 인원과 legal-action mask가 다른 표본은 먼저 분리한다.
2. 학습된 observ2vec embedding에 MiniBatch K-means를 적용한다.
3. 군집 내부 MCTS 정책의 Jensen-Shannon divergence와 value 분산을 측정한다.
4. 두 값이 큰 군집만 다시 나누고, 거의 같은 군집은 합친다.
5. 각 `cluster × action`의 방문 수를 coverage 표로 관리한다.

encoder가 갱신되면 cluster id가 이동하므로 cluster id만 영구 저장해서는 안 된다. 재임베딩할 수 있는 compact canonical observation과 encoder version을 함께 보존한다. 문양 정규화처럼 규칙상 완전히 같은 상태는 학습 전에 합치고, observ2vec은 그 이후의 근사적 전략 유사성만 담당한다.

### 최소 구현 순서

1. 정보집합 MCTS와 UCT 집계 저장부터 구현한다.
2. 저장 표본으로 policy/value 예측 MLP를 학습한다.
3. 마지막 hidden layer를 observ2vec으로 사용해 clustering 품질만 오프라인 평가한다.
4. cluster를 실제 탐색 공유 키나 학습 target 집계에 사용하는 것은 정책 divergence와 value 분산이 충분히 낮을 때만 진행한다.

처음부터 clustering을 탐색의 정답으로 사용하지 않는다. 잘못 합친 정보집합은 되돌리기 어렵지만, 오프라인 분석용 cluster는 원본 표본을 보존한 채 언제든 다시 만들 수 있다.

### 근거

- Gelada et al. (2019), DeepMDP: Learning Continuous Latent Space Models for Representation Learning: https://proceedings.mlr.press/v97/gelada19a.html
- Brown et al. (2019), Deep Counterfactual Regret Minimization: https://proceedings.mlr.press/v97/brown19b.html

### 추천 커밋 메시지

`docs: design UCT coverage and observ2vec clustering`

## UCT 전체 라운드 rollout과 고정 크기 상태 표현 (2026-07-15)

상태: `구현 전 설계`

### 탐색 단위

EV 모드는 스택과 탈락이 없으므로 한 명만 생존할 때 끝나는 에피소드가 존재하지 않는다. 따라서 UCT가 탐색할 유한 단위는 현재 의사결정부터 해당 라운드의 fold 또는 showdown까지다. 여러 라운드의 장기 지식은 트리를 무한히 이어서 얻는 것이 아니라 저장된 search target으로 policy/value 모델을 갱신해 얻는다.

### 핸드 테이블과 직접 rollout의 역할

`agent/hand_range.py`는 4구에서 상대 히든 990조합을 모두 열거하고 조합마다 여러 completion을 계산하므로 기본 설정도 15,840회 showdown이 필요하다. 이를 MCTS simulation 안에서 반복하면 중첩 Monte Carlo가 된다.

UCT에서는 다음 직접 rollout을 사용한다.

1. root observation과 모순되지 않는 상대 히든과 알려지지 않은 discard를 한 번 표본화한다.
2. 이후 공개 카드와 마지막 히든은 simulation이 해당 chance node에 도달할 때 남은 덱에서 뽑는다.
3. 선택은 UCT, 상대 행동은 첫 기준선에서 가벼운 random 또는 고정 heuristic policy로 진행한다.
4. fold 또는 showdown의 시점 이후 순칩을 root까지 backup한다.

한 simulation은 한 determinization만 사용한다. 핸드 테이블은 UI 분석, belief sampler 검증과 observ2vec 보조 target 생성에 남기고 매 탐색의 선행 조건으로 사용하지 않는다. HA1은 행동마다 자체 Monte Carlo를 수행하므로 첫 rollout policy로 사용하면 중첩 표본화가 발생한다. 처음에는 `RandomPokerAgent` 또는 Monte Carlo를 호출하지 않는 `HeuristicPokerAgent`만 사용한다.

### 불완전정보 누출 방지

sampled 상대 히든을 UCT node key나 policy 입력에 넣지 않는다. node key는 해당 행동자가 실제로 관측할 수 있는 정보집합으로 만든다. 첫 버전은 root 플레이어 행동만 UCT로 선택하고 상대는 고정 관측 정책으로 행동하게 하여, 상대 node가 root의 실제 히든에 맞추어 학습하는 strategy fusion을 피한다. 상대도 탐색하게 만드는 adversarial ISMCTS와 re-determinization은 별도 단계다.

### EV 헤즈업 observation의 고정 크기

현재 규칙에서는 모든 구성요소에 상한이 있다.

- 플레이어: self와 opponent 한 명으로 고정
- self 카드: 히든 최대 3장, 공개 최대 4장, discarded 1장
- opponent 관측 카드: 공개 최대 4장
- betting action vocabulary: `FOLD`, `CHECK`, `CALL`, `BBING`, `QUARTER`, `HALF` 6개
- 거리: `4th`, `5th`, `6th`, `7th_hidden` 4개
- raise: 거리당 최대 6회
- history: 거리당 최대 `선행 CHECK + BBING + 6 raise + 마지막 CALL/FOLD`인 9토큰, 전체 최대 36토큰

따라서 betting observation은 다음 고정 구조로 padding할 수 있다.

```text
cards:
  self_hidden[3]
  self_public[4]
  self_discarded[1]
  opponent_public[4]
scalars:
  street, actor/order, ante, pot, invested, round_bet,
  call_amount, raise_count, legal_action_mask
history[36]:
  street, actor, action
```

고정 ante와 현재 베팅 공식에서는 street·actor·action 순서로 과거 금액을 재생할 수 있다. 현재 pot, invested, round bet과 call amount도 observation에 있으므로 history token에 `paid`, `pot_after` 같은 금액을 중복 저장하지 않는다. 큰 베팅 수치는 `log1p(value / ante)`로 정규화한다.

초기 discard/reveal은 betting observation과 카드 수가 다르므로 첫 구현에서는 기존 agent 메서드를 그대로 사용한다. 추후 선택까지 학습할 때만 히든 4장 전용 작은 encoder head를 추가한다.

### belief state의 고정 크기

상대 히든 전체 조합표를 모델 입력으로 넣지 않는다. UCT에서는 belief를 고정 개수 `K`의 particle로 표현하고 simulation마다 하나를 뽑는다. 첫 측정값은 `K=64`로 두되 탐색 simulation 수가 더 크면 particle을 재사용하거나 즉석에서 추가 표본화한다.

policy/value 모델이 belief 특징을 필요로 할 때는 각 particle의 상대 히든 최대 3장을 같은 작은 MLP로 encode한 뒤 mean pooling하여 고정 길이 벡터로 만든다. 이렇게 하면 `K`를 바꾸어도 최종 입력 차원은 변하지 않는다.

`opponents' my hidden cards estimation`을 별도의 재귀 belief로 저장하지 않는다. 상대 행동을 구할 때 동일한 observation encoder를 상대 시점으로 호출하면 상대가 볼 수 있는 self hidden과 내 public card가 자동으로 뒤집힌다. 더 강한 상호 추론이 필요해지면 두 플레이어의 private state를 함께 가진 public-belief particle로 확장한다.

### 첫 observ2vec 구조

과도한 sequence 모델 없이 다음 구조로 시작한다.

1. 각 카드의 rank, canonical suit와 역할을 작은 embedding으로 변환하고 그룹별 mean pooling한다.
2. 36개 history token에 위치를 포함한 embedding을 붙여 고정 길이로 펼친다.
3. 정규화한 scalar, legal mask와 belief particle 요약을 이어 붙인다.
4. 2~3층 MLP로 32차원 `observ2vec`을 만든다.
5. 같은 vector에서 policy head와 value head를 분기한다.

학습 목적은 다음 최소 조합이다.

```text
L = KL(pi_MCTS || pi_model)
    + lambda * Huber(v_MCTS, v_model)
    + mu * symmetry_loss
```

`symmetry_loss`는 문양 치환 전후처럼 규칙상 같은 observation의 embedding을 가깝게 만든다. 카드 복원 loss는 필수가 아니다. 정책만 같고 손익 규모가 다른 상태가 잘못 합쳐지지 않도록 value target을 함께 사용한다. reward와 다음 latent state 예측을 통한 DeepMDP/bisimulation loss는 이 기준선의 clustering 품질이 부족할 때만 추가한다.

### 구현 순서

1. 현재 observation에서 determinization을 표본화하는 헤즈업 EV sampler를 만든다.
2. fold/showdown까지 진행할 수 있는 조용한 simulation step API를 기존 `PokerGame` 규칙 위에 만든다.
3. `agent/uct_agent.py`에 `PokerAgent`를 상속한 root-UCT baseline을 만든다.
4. simulation 수별 처리량과 HA1·heuristic 상대 순칩을 측정한다.
5. 충분히 방문한 root의 action 방문 수, return 합과 분산만 저장한다.
6. 그 데이터가 쌓인 뒤 observ2vec MLP와 clustering을 구현한다.

### 근거

- Cowling, Powley, Whitehouse (2012), Information Set Monte Carlo Tree Search: https://doi.org/10.1109/TCIAIG.2012.2200894
- Brown et al. (2020), Combining Deep Reinforcement Learning and Search for Imperfect-Information Games: https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html
- Gelada et al. (2019), DeepMDP: Learning Continuous Latent Space Models for Representation Learning: https://proceedings.mlr.press/v97/gelada19a.html

### 추천 커밋 메시지

`docs: design fixed-size UCT rollout state`

## UCT rollout 예산과 기존 SQLite 저장소 결합 (2026-07-15)

상태: `구현 전 설계`

### 1000 rollout의 해석

MCTS는 실제 라운드 시작점에서 한 번만 탐색하는 것이 아니라 새 공개 카드와 베팅 행동으로 observation이 바뀔 때마다 현재 의사결정 root를 다시 탐색해야 한다. 기존 uniform 수집의 실측 의사결정 수는 핸드당 평균 약 4.26개였다.

첫 기준선은 `결정 root당 256 simulation`으로 둔다. 그러면 평균 핸드당 약 1,090 simulation이 되어 목표한 1000회와 거의 같다. 한 root의 합법 행동은 최대 6개이고 UCT가 유망한 행동에 방문을 집중하므로 초기 search-policy target을 만들기에는 합리적인 시작값이다.

고정 256회가 충분하다는 보장은 없다. EV 모드의 큰 베팅 line은 return 분산이 크므로 다음 통계를 저장하고 seed를 바꾼 반복 탐색으로 판단한다.

- action별 visit count
- return sum과 squared-return sum
- 최다 방문 action의 일치율
- 256회와 512회 search-policy의 Jensen-Shannon divergence
- 상위 두 action의 평균 EV 차이와 신뢰구간

첫 구현은 고정 256회를 사용하고, 이후에는 최소 256회에서 시작해 상위 action의 구간이 겹치면 512회, 최대 1024회까지 늘리는 adaptive budget을 검토한다. 고정 1000회를 모든 root에 주는 것은 측정 전에 도입하지 않는다.

### 기존 테이블과 값의 의미

기존 `q_values`는 선택한 action 뒤에도 모든 플레이어가 균등 행동을 하는 `Q^uniform` 표본이다. UCT 결과는 search policy, 탐색 예산과 상대 rollout policy에 따라 달라진다. 같은 `(state, action)` 행에 두 return을 더하면 서로 다른 target이 섞인다.

따라서 같은 SQLite 파일을 사용하더라도 별도 `uct_nodes` 테이블에 저장한다. 기존 8시간 데이터는 수정하지 않는다.

```text
uct_nodes
  canonical_state
  search_version
  opponent_policy
  simulations
  legal_mask[6]
  action_visits[6]
  return_sums[6]
  return_squared_sums[6]
  chosen_action
```

한 행은 action 하나가 아니라 root 정보집합 하나다. 고정 길이 6개 배열로 묶으면 긴 state key를 여섯 번 반복하지 않는다. 초기 데이터 수는 기존 exact-state 수집보다 훨씬 작으므로 먼저 단순한 JSON 또는 SQLite 숫자 열로 구현하고, 실제 파일 크기가 문제가 될 때만 BLOB packing을 적용한다.

`search_version`에는 UCT 상수, simulation budget, rollout 종료 규칙과 backup 관점을 식별할 버전을 기록한다. `opponent_policy`에는 `random`, `heuristic`처럼 상대 행동 생성기를 기록한다. 두 값이 다른 표본은 자동 합산하지 않는다.

### 저장 범위

실제 플레이 경로에서 만난 각 의사결정 root만 저장한다. simulation 중 방문한 모든 child node를 영구 저장하지 않는다. child 통계는 root action을 선택하고 target을 만든 뒤 버린다. 이후 observ2vec이 준비되면 충분히 방문한 내부 정보집합을 추가 저장할지 별도 측정한다.

라운드당 양쪽 플레이어 시점의 실제 decision root를 저장할 수 있으므로 한 라운드에서 여러 개의 고품질 node target을 얻는다. 다만 같은 simulation 결과를 두 플레이어의 서로 다른 정보집합 표본으로 복제하지 않는다. 각 플레이어가 실제로 행동한 root만 그 플레이어 관측으로 기록한다.

### 첫 품질 기준

1. 동일 root를 서로 다른 seed로 10회 탐색한다.
2. 최다 방문 action이 8회 이상 일치하는지 본다.
3. 256회와 512회의 평균 정책 divergence를 비교한다.
4. 안정적이면 256회를 유지하고, 불안정한 거리나 raise depth에만 budget을 늘린다.

이 기준은 equilibrium을 증명하지 않는다. 목적은 UCT가 집중한 betting line의 재현 가능한 policy/value target을 적은 핸드로 수집하는 것이다.

### 추천 커밋 메시지

`docs: define UCT rollout budget and storage schema`

## 헤즈업 EV UCT 수집기 1차 구현 (2026-07-15)

상태: `구현`

### 구현 범위

- `agent/uct_agent.py`에 `PokerAgent`를 상속한 `UCTPokerAgent`를 추가했다.
- 각 simulation은 현재 observation과 모순되지 않는 상대 히든, 상대 discard와 남은 덱을 새로 표본화한다.
- sampled 비공개 카드는 node key에 넣지 않고 canonical observation만 사용한다.
- root 플레이어의 현재와 미래 의사결정은 UCT로 선택하고 상대 simulation 행동은 균등 무작위로 선택한다.
- fold 또는 showdown에서 시점 이후 순칩을 방문한 root-player node에 backup한다.
- 실제 라운드는 양쪽 UCT 에이전트가 진행하며 각 실제 의사결정 root의 집계만 저장한다.
- 정확한 베팅 우선순위 tie-break를 복원할 수 있도록 agent observation에 `seat_index`를 추가했다.

이 구현은 상대 node까지 adversarial하게 탐색하는 완전한 equilibrium solver가 아니다. 상대 비공개 정보에 맞추어 opponent tree가 적응하는 누출을 피하기 위해 첫 버전에서는 상대를 고정 random policy로 두었다. 현재 값은 `random opponent에 대한 UCT best response`로 해석한다.

### 저장

`uct_rollout.py`는 기존 SQLite 파일을 열어도 `q_values`와 `coverage`를 건드리지 않고 `uct_nodes`만 추가한다. 한 행은 한 canonical root이며 여섯 행동별 visit count, return sum과 squared-return sum을 숫자 열로 저장한다. `seat_index`, `search_version`, `opponent_policy`와 `simulation_budget`이 primary key에 포함된다.

### 20핸드 smoke 실측

동일 seed에서 로컬 Python 3.10으로 측정했다.

| root 예산 | 핸드 | root | simulation/hand | 처리량 |
|---:|---:|---:|---:|---:|
| 256 | 20 | 45 | 576.0 | 7.59 hands/s |
| 512 | 20 | 43 | 1,100.8 | 4.00 hands/s |

UCT 실제 행동으로 빠른 fold가 늘어 기존 uniform rollout의 4.26 decision/hand보다 적은 약 2.2 root/hand가 관측됐다. 따라서 목표한 핸드당 약 1000 simulation에는 root당 512회가 더 가깝다. 512회 결과는 43개 root, 22,016 simulation과 20KiB SQLite 파일을 만들었고 저장된 전체 simulation 수와 여섯 action visit 합이 일치했다.

20핸드는 처리량 smoke test일 뿐 policy 안정성을 증명하지 않는다. 장시간 실행 전후에는 다른 seed의 256/512 결과에서 최다 방문 action 일치율과 policy divergence를 별도로 비교해야 한다.

### 검증

- UCT root action visit 합이 설정한 simulation budget과 일치함을 확인했다.
- 기존 `q_values`가 있는 DB에 `uct_nodes`를 추가해도 기존 marker 행이 유지됨을 확인했다.
- 전체 37개 테스트가 통과했다.

### 실행 명령

```powershell
python -B uct_rollout.py --output replays/ev_rollout_overnight_1g.sqlite3 --hands 0 --seconds 28800 --simulations 512 --max-gib 1 --flush-hands 25 --progress-seconds 30
```

### 추천 커밋 메시지

`feat: add heads-up EV UCT rollout collector`

## UCT 수집기의 다중 프로세스 실행 검토 (2026-07-15)

상태: `운영 지침`

UCT simulation은 CPU 계산 비중이 높고 SQLite flush는 여러 핸드마다 짧게 발생하므로 서로 다른 프로세스를 실행하면 멀티코어에서 처리량이 늘 수 있다. Python GIL은 프로세스 사이에 공유되지 않는다.

같은 SQLite 파일을 사용해도 WAL과 transaction 때문에 파일이 손상되지는 않으며 `uct_nodes` UPSERT는 writer lock 아래 직렬화된다. 다만 SQLite는 한 번에 writer 하나만 허용하므로 프로세스 수가 많아지면 flush 대기와 `database is locked` 오류 가능성이 커진다.

현재 주의점은 다음과 같다.

- 프로세스마다 반드시 다른 `--seed`를 사용한다. 같은 seed는 카드와 탐색 표본을 거의 그대로 중복한다.
- `uct_last_run_*` metadata는 공용 key라 마지막으로 종료한 프로세스의 값이 남는다. 실제 node 통계에는 영향이 없다.
- 여러 프로세스가 같은 `--max-gib`를 확인하므로 flush batch만큼 파일 상한을 조금 넘을 수 있다.
- 동일 node의 동시 UPSERT는 합산되지만 writer가 직렬화되므로 프로세스 수에 비례한 완전한 선형 가속은 기대하지 않는다.

첫 실행은 같은 DB에 두 프로세스만 사용하고 `--flush-hands 50`으로 writer 횟수를 줄인다. 두 프로세스의 합산 hands/s가 단일 프로세스의 1.5배 미만이면 추가 worker를 늘리지 않는다. lock 오류가 발생하면 process별 shard DB로 분리하고 이후 학습기가 여러 파일을 순회하게 한다.

### 두 터미널 실행 예시

첫 번째 터미널:

```powershell
python -B uct_rollout.py --output replays/ev_rollout_overnight_1g.sqlite3 --hands 0 --seconds 28800 --simulations 512 --seed 101 --max-gib 1 --flush-hands 50 --progress-seconds 30
```

두 번째 터미널:

```powershell
python -B uct_rollout.py --output replays/ev_rollout_overnight_1g.sqlite3 --hands 0 --seconds 28800 --simulations 512 --seed 202 --max-gib 1 --flush-hands 50 --progress-seconds 30
```

완전한 충돌 회피가 우선이면 `--output`을 각각 `uct_w1.sqlite3`, `uct_w2.sqlite3`로 바꾼다. 현재는 shard 병합 명령을 구현하지 않았으므로 먼저 두 프로세스의 동일 DB 처리량을 짧게 측정한다.

### 추천 커밋 메시지

`docs: add parallel UCT collection guidance`

## observ2vec의 연산 일관성과 최소 encoder (2026-07-15)

상태: `구현 전 설계`

### 결론

고정 길이 입력을 그대로 펼친 MLP도 기준선으로는 동작할 수 있다. 그러나 카드 집합의 순서까지 MLP가 다시 학습하게 만들 필요는 없다. 첫 구현은 Transformer 대신 `공유 card MLP + 집합 pooling + 고정 history slot + fusion MLP`로 둔다. 이 구조는 현재 최대 36개인 베팅 이력에 충분히 작고, 규칙상 정확한 대칭성을 직접 주입할 수 있다.

Transformer는 5인 게임처럼 플레이어와 이력이 길어지거나, 이 기준선이 held-out UCT policy와 action EV를 충분히 예측하지 못할 때만 비교한다. 순서 없는 집합끼리의 상호작용이 실제 병목이면 그때 Set Transformer를 검토한다.

### word2vec식 산술과 다른 목표

포커에서는 `현재 latent + 카드 vector = 다음 latent` 같은 전역 선형 관계를 목표로 하지 않는다. 같은 카드도 현재 패와 공개 정보에 따라 플러시 완성, 스트레이트 완성 또는 무관한 카드가 되며, 같은 액션도 pot과 call 비용에 따라 결과가 달라진다.

대신 다음 두 종류의 일관성을 사용한다.

```text
# 규칙상 완전히 같은 관측
E(suit_permutation(o)) = E(o)
E(hidden_card_permutation(o)) = E(o)
E(belief_particle_permutation(o)) = E(o)

# 관측에 작용하는 문맥 의존 연산
T(E(o), action, chance_card) ~= E(o_next)
R(E(o), action) ~= reward
```

첫 세 식은 encoder 구조와 전처리로 정확히 만족시킨다. 뒤의 두 식은 transition 자료가 생긴 뒤 학습할 비선형 latent dynamics이며, word2vec의 벡터 덧셈에 해당하는 성질을 억지로 요구하지 않는다.

EV 모드의 칩 단위는 모든 금액과 결과를 ante로 나누어 없앤다. 입력 금액에는 `log1p(amount / ante)`, action EV target에는 `return / ante`를 사용하면 ante의 배율만 다른 상태가 같은 전략 표현을 공유할 수 있다.

### 카드 순서에서 보존할 정보

현재 `canonical_state`는 카드 그룹의 나열 순서를 보존한다. 여기서 비공개 패의 내부 순서는 규칙상 의미가 없지만, 공개 패의 순서는 카드가 어느 street에 나타났는지와 과거 베팅을 연결하므로 의미가 있다.

학습 전처리는 각 문양 치환 후보에서 비공개 카드만 정렬하고, 자신의 공개 카드와 상대 공개 카드의 도착 순서는 유지한 채 사전식 최소 표현을 다시 고른다. 실행 중인 SQLite를 다시 쓰거나 현재 수집 schema를 즉시 바꾸지는 않는다. 이후 수집기에 같은 정규화를 적용할 때는 schema version을 올린다.

### 최소 encoder

```text
self hidden set
  -> shared CardMLP -> mean pooling

self/opponent public cards
  -> card embedding + owner/arrival-slot embedding -> fixed slots

belief particles (선택 입력)
  -> hidden-hand pooling -> particle mean pooling

history[36]
  -> street/actor/action embedding + mask -> flatten

normalized scalars + legal-action mask
  -> concatenate all -> 2~3 layer MLP -> observ2vec[32]
                                         -> policy head[6]
                                         -> action-Q head[6]
```

공개 카드와 이력에는 도착 순서가 있으므로 pooling하지 않는다. 첫 기준선에서는 history가 짧고 상한이 고정되어 있어 위치별 embedding을 펼치는 것으로 충분하다. 순서 일반화가 실제로 부족하면 1층 GRU를 먼저 비교하고, Transformer는 그 뒤에 둔다.

### decoding의 의미

원본 observation을 정확히 복원하는 autoencoder는 사용하지 않는다. 전략적으로 같은 여러 observation을 하나로 합치려는 목적과 exact reconstruction은 서로 충돌한다. observ2vec의 decoder 역할은 다음 task head가 맡는다.

- UCT action 방문 비율인 search policy
- action별 평균 chip EV
- 추후 추가할 reward와 next-latent prediction

사람이 cluster를 해석할 때는 latent를 카드 문자열로 생성하지 않고, cluster 중심에 가장 가까운 실제 observation인 medoid와 대표 표본을 보여준다. 이것이 손실 없는 decoder는 아니지만 UI와 검증에는 더 정직하고 읽기 쉽다.

### 첫 학습 목적

현재 `uct_nodes`에는 action별 방문 수와 return 합이 있으므로 다음 목적을 바로 만들 수 있다.

```text
pi_uct(a|o) = N(o,a) / sum_a N(o,a)
q_uct(o,a)  = return_sum(o,a) / N(o,a)

L = KL(pi_uct || pi_model)
    + lambda * sum_a visit_weight(a) * Huber(q_uct(a) / ante, q_model(a))
    + mu * ||E(o) - E(g(o))||^2
```

`g`는 비공개 카드 순서 변경이나 문양 치환처럼 동일한 상태를 만드는 변환이다. legal하지 않은 action은 두 head와 loss에서 mask한다. 서로 다른 `search_version`, `opponent_policy`, simulation budget은 같은 target으로 섞지 않는다. 현재 target은 `uct-v1`의 random opponent에 대한 값이므로 이 embedding 역시 우선 그 조건에서의 전략 유사성을 뜻한다.

### clustering과 판정 기준

street와 legal-action mask가 다른 표본은 먼저 분리하고, 그 안에서 observ2vec에 MiniBatch K-means를 적용한다. 좋은 cluster인지는 좌표 모양이 아니라 다음 값으로 판정한다.

- cluster 내부 UCT policy의 Jensen-Shannon divergence
- cluster 내부 action-Q 분산과 최선 action 일치율
- 다른 seed와 held-out state에서도 유지되는 cluster 품질
- cluster별 표본 수와 action 방문 coverage

현재 저장소에는 2026-07-15 확인 시점에 1만 4천 개 이상 UCT root와 700만 회 이상 simulation이 있어 첫 policy/Q encoder 실험은 시작할 수 있다. 다만 root 사이의 실제 transition 연결은 저장하지 않으므로 `T`와 `R` 일관성은 아직 학습할 수 없다. 첫 encoder의 clustering이 부족하다고 확인된 뒤에만 실제 플레이 경로의 `(state, action, next_state, reward)`를 작은 별도 table로 저장한다. simulation 내부 node 전체는 저장하지 않는다.

### 구현 순서

1. 기존 `uct_nodes`를 읽는 학습 전처리에서 비공개 카드와 문양의 정확한 대칭성을 정규화한다.
2. structured MLP의 policy/action-Q 예측 성능을 held-out split으로 측정한다.
3. 32차원 latent를 clustering하고 policy divergence와 Q 분산을 측정한다.
4. 품질이 부족할 때 history GRU, transition loss, Set Transformer 순서로 하나씩 비교한다.

### 근거

- Zaheer et al. (2017), Deep Sets: https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Lee et al. (2019), Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Gelada et al. (2019), DeepMDP: Learning Continuous Latent Space Models for Representation Learning: https://proceedings.mlr.press/v97/gelada19a.html

### 추천 커밋 메시지

`docs: define observ2vec invariances and training targets`

## clustering label과 RL fine-tuning의 경계 (2026-07-15)

상태: `개념 정리`

### 결론

K-means가 붙이는 cluster ID 자체는 RL label이 아니다. `cluster 17` 같은 번호는 centroid를 가리키는 임의 주소이며 번호의 순서나 의미도 없다. 이것을 맞히도록 학습하는 단계는 pseudo-label을 사용한 분류 또는 양자화이지, 보상을 최적화하는 RL fine-tuning은 아니다.

현재 파이프라인의 target은 다음처럼 구분한다.

| target 또는 update | 성격 |
|---|---|
| K-means cluster ID | 비지도 clustering의 pseudo-label |
| UCT action 방문 비율 | search-policy distillation, 지도학습 |
| UCT action별 평균 return | 경험적 return을 label로 삼는 value learning, 즉 RL update |
| 실제 trajectory return이나 TD target으로 반복 갱신 | RL의 policy evaluation/control |
| policy gradient, GRPO로 정책 자체 갱신 | RL fine-tuning |

RL은 label이 없는 학습을 뜻하지 않는다. 환경에서 얻은 reward와 return이 스스로 만든 label 역할을 한다. 따라서 observ2vec encoder를 `EV = expected return` 오차로 갱신하면 optimizer의 모양은 회귀이더라도 학습 내용은 value learning이며, embedding과 이후 bucket은 reward-aware state abstraction이 된다. 질문에서 말한 `EV가 label이 되어 RL화된다`는 해석이 맞다.

다만 `fine-tuning`은 이미 학습된 모델을 새로운 목적과 데이터로 추가 갱신한다는 뜻이다. 처음부터 UCT EV target으로 학습하면 reward-based value learning 또는 search distillation이고, 먼저 다른 목적으로 만든 observ2vec을 EV 또는 TD loss로 다시 조정하면 RL fine-tuning이다. 그 목적이 policy/value 개선과 반복적인 새 trajectory 수집까지 이어질 때 전체 과정은 approximate policy iteration에 가까워진다.

### bucket을 label로 고정하지 않는 이유

cluster ID를 영구 정답으로 삼으면 encoder가 갱신될 때 centroid와 번호가 바뀌는 문제, 경계 근처 상태가 임의로 갈리는 문제, 잘못 합친 초기 bucket을 그대로 모방하는 문제가 생긴다. 따라서 다음 단방향 관계만 사용한다.

```text
observation -> encoder -> continuous latent -> nearest centroid -> bucket ID
                         -> policy/action-Q heads
```

모델이 bucket ID를 다시 예측하도록 학습하지 않는다. policy와 action-Q가 학습의 의미 있는 label이고, bucket은 저장과 집계에 쓰는 파생 주소다. encoder와 centroid에는 각각 version을 붙이고 원본 compact observation은 재임베딩할 수 있게 남긴다.

### 현재 프로젝트의 해석

현재 `uct-v1` 자료로 observ2vec을 학습하는 것은 `random opponent에 대한 UCT search EV를 value function으로 근사하는 RL 학습`이면서, 동시에 search 결과를 증류하는 지도 회귀다. 두 표현은 학습을 보는 층위가 다를 뿐 서로 모순되지 않는다. 이 모델로 새 UCT trajectory를 만들고 그 결과로 같은 모델을 반복 갱신하면 search와 학습이 닫힌 RL control loop가 된다.

첫 실험에서는 cluster label loss나 VQ codebook을 추가하지 않는다. policy/action-Q loss로 continuous latent를 먼저 만들고, clustering이 실제로 DB 크기와 일반화를 개선하는지만 측정한다.

### 추천 커밋 메시지

`docs: clarify clustering labels and RL fine-tuning`

## cluster 수를 고정하지 않는 bucket 후보 (2026-07-15)

상태: `기법 선택 전`

### 공통 전제

cluster 수 `K`를 입력하지 않는 것은 가능하지만 아무 기준도 없이 cluster 수를 정할 수는 없다. `K` 대신 거리 반경, 허용 의사결정 regret, 최소 밀도, Bayesian concentration 같은 복잡도 기준이 필요하다. 이 프로젝트에서는 임의의 latent 거리보다 action EV 손실을 기준으로 삼는 것이 해석하기 쉽다.

32차원 observ2vec은 먼저 UCT policy와 action-Q를 예측하도록 학습한다. clustering은 `street + legal-action mask`가 같은 표본 안에서만 수행한다. threshold 후보를 여러 개 평가하고 다음 제약을 만족하는 가장 큰 압축률을 선택한다.

```text
bucket_action = argmax_a mean_Q(bucket, a)
regret(o) = max_a Q(o, a) - Q(o, bucket_action)

평가값:
  mean / p95 regret (ante 단위)
  best-action disagreement
  policy JS divergence
  state 수 / bucket 수 압축률
```

### 추천 순위

#### 1. BIRCH + task-aware observ2vec

현재 MVP 추천이다. `Birch(n_clusters=None)`은 새 점이 기존 subcluster의 반경 threshold를 넘으면 subcluster를 추가하고 `partial_fit`으로 이어 학습할 수 있다. 현재 설치된 scikit-learn 1.1.3에서 바로 사용할 수 있고 별도 패키지가 필요 없다.

- 장점: cluster 수 불필요, centroid 제공, streaming 처리, 메모리 효율적, 새 상태 `predict` 가능
- 단점: threshold와 입력 순서에 민감하고, 32차원 Euclidean 거리가 전략 유사성을 잘 반영해야 함
- 사용법: latent를 표준화하거나 L2 정규화하고, shuffled calibration batch로 먼저 tree를 만든 뒤 수집 데이터를 `partial_fit`한다.
- 선택 기준: threshold sweep 후 p95 decision regret 제한을 만족하는 가장 적은 bucket 수

#### 2. online DP-means + EV-regret gate

새 latent가 가장 가까운 centroid에서 거리 `lambda` 이상이면 centroid를 하나 추가하는 방식이다. DP-means는 k-means distortion에 cluster 수 penalty를 더해 cluster 수를 데이터에서 늘린다. 이 프로젝트에서는 거리만 보지 않고 해당 centroid의 대표 action을 썼을 때 EV regret이 허용치를 넘는 경우에도 새 centroid를 만들 수 있다.

- 장점: 사용자가 말한 `필요할 때 centroid 증가`와 가장 직접적으로 일치하고, EV 허용 오차를 명시할 수 있음
- 단점: scikit-learn 구현이 없어 작은 custom 구현과 저장 형식이 필요하며, 데이터 순서에 민감함
- 승격 조건: BIRCH의 latent 반경과 실제 action regret의 상관이 낮을 때

#### 3. multi-output regression tree의 leaf bucket

`Q(o, action)` 여섯 값을 예측하는 회귀나무의 leaf를 bucket으로 사용한다. cluster centroid는 아니지만 leaf 수를 미리 정하지 않고 `min_samples_leaf`, `min_impurity_decrease`, pruning penalty로 늘릴 수 있다.

- 장점: bucket이 EV 오차를 직접 줄이고, 비교 기준선 구현이 매우 짧음
- 단점: 32차원 공간을 축별로만 분할하며 online centroid 갱신 구조가 아님
- 용도: unsupervised 거리 bucket이 실제 value abstraction보다 나은지 확인하는 대조군

#### 4. HDBSCAN

밀도가 다른 cluster와 noise를 찾고 cluster 수를 자동 결정하는 오프라인 분석 후보이다.

- 장점: 구형이 아닌 군집과 outlier를 찾을 수 있음
- 단점: persistent centroid와 streaming 배정이 핵심인 현재 DB bucket에는 맞지 않고, 현재 환경에 패키지가 설치되어 있지 않음
- 용도: observ2vec 공간이 실제로 군집 구조를 갖는지 표본에서 진단할 때만 사용

#### 5. Bayesian Gaussian mixture 또는 deep clustering

Dirichlet-process 계열 mixture는 cluster 수에 확률적 prior를 둘 수 있지만 실용 구현은 최대 component 수를 두며 32차원 covariance 추정 비용이 크다. DEC, VQ-VAE, Growing Neural Gas처럼 encoder와 cluster를 함께 바꾸는 기법은 초기 cluster 오류가 representation에 다시 주입될 수 있다.

첫 실험에서는 사용하지 않는다. BIRCH와 DP-means가 decision regret 기준을 만족하지 못한 뒤에만 검토한다. Transformer는 sequence encoder 후보이지 cluster 수를 자동으로 정하는 기법이 아니므로 이 선택과는 무관하다.

### 첫 실험안

1. action-Q/policy 지도학습으로 32차원 observ2vec을 만든다.
2. encoder를 고정하고 street/legal-mask별 BIRCH를 학습한다.
3. threshold sweep으로 bucket 수와 held-out regret 곡선을 출력한다.
4. BIRCH가 실패할 때만 같은 latent에 online DP-means를 비교한다.

### 근거

- Zhang, Ramakrishnan, Livny (1996), BIRCH: An Efficient Data Clustering Method for Very Large Databases: https://doi.org/10.1145/233269.233324
- Kulis, Jordan (2012), Revisiting k-means: New Algorithms via Bayesian Nonparametrics: https://icml.cc/2012/papers/291.pdf
- McInnes, Healy, Astels (2017), hdbscan: Hierarchical density based clustering: https://joss.theoj.org/papers/10.21105/joss.00205
- Xie, Girshick, Farhadi (2016), Unsupervised Deep Embedding for Clustering Analysis: https://proceedings.mlr.press/v48/xieb16.html

### 추천 커밋 메시지

`docs: compare adaptive bucket clustering methods`

## Mahalanobis metric, BIRCH, DP-means와 Gaussian EM 검토 (2026-07-15)

상태: `기법 선택 보충`

### Mahalanobis distance의 가정

32차원 latent `z`와 centroid `mu_k` 사이의 Mahalanobis 제곱거리는 다음과 같다.

```text
d_M^2(z, mu_k) = (z - mu_k)^T Sigma^-1 (z - mu_k)
```

이 거리를 사용하는 데 정규분포 가정이 반드시 필요한 것은 아니다. 공분산은 각 축의 scale과 선형 상관을 보정하는 metric으로만 사용할 수 있다. 다만 이 거리가 cluster의 확률이나 chi-square 임계값을 뜻한다고 해석하려면 Gaussian 또는 타원형 분포 가정이 추가로 필요하다.

Mahalanobis metric은 observation에서 latent로 가는 encoder가 선형이라고 가정하지 않는다. latent 공간 안에서 한 cluster의 유사도가 공분산으로 설명되는 타원형이라는 국소 가정을 한다. 곡선형 manifold나 하나의 centroid 주위에 여러 mode가 있는 경우에는 충분하지 않다.

공유 공분산 `Sigma = L L^T`를 사용하면 다음 whitening과 완전히 같다.

```text
y = L^-1 (z - mean_z)
d_M^2(z_i, z_j) = ||y_i - y_j||_2^2
```

이는 neural latent에 특히 유용하다. policy/Q head 앞의 latent는 회전과 축별 scale을 바꾸고 다음 layer가 역변환해도 같은 예측을 낼 수 있으므로 raw Euclidean 거리는 표현 좌표계에 민감하다. 공분산도 함께 변환한 global Mahalanobis 거리는 이런 가역 선형 좌표 변환에 불변이다.

### 공분산을 cluster마다 둘 때의 문제

cluster별 `Sigma_k`를 사용하면 단순 Mahalanobis 거리만 비교해서는 안 된다. Gaussian negative log-likelihood에서 component 선택 점수는 상수항을 제외하면 다음과 같다.

```text
score_k(z) = d_M,k^2(z, mu_k) + log|Sigma_k| - 2 log pi_k
```

`log|Sigma_k|`를 빼면 공분산이 큰 cluster는 모든 점의 거리를 작게 만들어 과도하게 많은 점을 흡수한다. `pi_k`까지 포함하면 이미 Gaussian mixture의 hard assignment에 가까워진다.

32차원 full covariance에는 cluster마다 `32 * 33 / 2 = 528`개 covariance parameter가 필요하다. 작은 신규 centroid는 역행렬을 계산할 표본조차 부족하고, 그보다 많은 표본이 있어도 condition number가 불안정할 수 있다. 첫 버전에서는 다음 순서로 제한한다.

1. 전체 또는 street별 shared covariance 하나를 사용한다.
2. sample covariance 대신 shrinkage covariance로 작은 eigenvalue를 안정화한다.
3. cluster별 차이가 필요해도 먼저 diagonal covariance만 비교한다.
4. full per-cluster covariance는 bucket마다 충분한 표본이 쌓이고 residual anisotropy가 확인된 뒤에만 사용한다.

현재 설치된 scikit-learn에는 `LedoitWolf` shrinkage estimator가 있으므로 새 의존성 없이 shared precision을 만들 수 있다. whitening이 작은 고유값 방향의 noise를 증폭하지 않도록 eigenvalue floor 또는 shrinkage를 반드시 둔다.

### 1순위: whitened BIRCH

BIRCH는 각 subcluster를 Clustering Feature인 `(N, LS, SS)`로 요약한다.

- `N`: 표본 수
- `LS`: latent vector 합
- `SS`: 제곱합

새 표본은 root에서 가까운 centroid를 따라 leaf까지 내려간다. leaf subcluster에 합쳤을 때 반경이 `threshold` 이하이면 CF만 갱신하고, 넘으면 새 subcluster를 만든다. 한 node의 subcluster 수가 `branching_factor`를 넘으면 node를 분할한다. `n_clusters=None`이면 마지막에 고정 K로 다시 합치지 않고 leaf subcluster를 그대로 bucket으로 쓴다.

기본 BIRCH의 반경은 Euclidean이므로 full Mahalanobis metric을 직접 넣을 수 없다. 대신 shared covariance로 latent를 먼저 whitening하면 기존 BIRCH를 수정하지 않고 Mahalanobis BIRCH와 같은 효과를 얻는다.

```text
observation -> encoder -> z[32]
                        -> shared shrinkage whitening -> y[32]
                        -> Birch(threshold=t, n_clusters=None)
```

장점은 한 번의 순회와 작은 CF tree로 대량 DB를 처리하고, `partial_fit`으로 뒤에 수집된 상태를 추가할 수 있다는 점이다. 단점은 다음과 같다.

- 입력 순서에 따라 초기 leaf 구성이 달라질 수 있다.
- 하나의 threshold가 밀도가 다른 모든 전략 영역에 맞지 않을 수 있다.
- CF 요약은 개별 표본의 복잡한 분포를 보존하지 않는다.
- 32차원에서는 거리 집중 현상이 생길 수 있으므로 latent 학습과 whitening 품질이 중요하다.

첫 fit은 무작위로 섞은 calibration batch를 사용하고, 서로 다른 세 순서에서 bucket 수와 held-out EV regret의 변동을 확인한다. threshold는 Gaussian chi-square 분위수로 정하지 않고 실제 action regret 곡선으로 선택한다.

### 2순위: Mahalanobis DP-means

DP-means의 목적함수는 다음과 같은 k-means distortion과 cluster 수 penalty의 합이다.

```text
J = sum_i d^2(z_i, mu_c(i)) + lambda * K
```

각 표본을 가장 가까운 centroid에 할당하되 최소 제곱거리가 `lambda`보다 크면 그 표본 위치에 새 centroid를 만든다. 전체 할당 뒤 centroid 평균을 다시 계산하고 반복한다. 따라서 `K`는 입력하지 않지만 `lambda`가 새 cluster 하나를 만드는 비용을 결정한다.

shared Mahalanobis를 쓰는 경우 `d^2 = d_M^2`로 바꾸거나, BIRCH와 동일하게 whitened `y`에서 Euclidean DP-means를 실행하면 된다. 후자가 구현과 수치 검증이 더 단순하다.

원 논문의 DP-means는 전체 자료에 대해 할당과 centroid 갱신을 반복한다. streaming 한 번만 통과하면서 centroid를 이동시키는 변형은 실용적이지만 데이터 순서에 더 민감하고 같은 수렴 성질을 그대로 주장할 수 없다. 현재 DB에서는 다음 절충이 적절하다.

1. calibration 표본에 batch DP-means를 실행한다.
2. 이후 표본은 가장 가까운 centroid에 누적한다.
3. 일정량이 쌓이면 centroid 재계산과 merge pass를 수행한다.

`lambda` 역시 latent 거리만으로 고르지 않는다. 여러 값을 sweep하여 held-out decision regret 제한을 만족하는 가장 작은 `K`를 선택한다. EV-regret gate를 직접 넣는 변형도 가능하다.

```text
새 centroid 생성 조건:
  nearest Mahalanobis distance > lambda
  또는
  centroid 대표 action 사용 시 labeled Q-regret > epsilon
```

두 번째 조건은 학습 자료에 UCT action-Q label이 있을 때만 정확히 계산할 수 있다. label이 없는 새 상태에서는 Q head의 추정치를 사용해야 하므로, 첫 구현에서는 cluster 생성 조건에 직접 넣기보다 held-out 평가와 threshold 선택에 사용한다.

### BIRCH와 DP-means의 직접 비교

| 항목 | BIRCH | DP-means |
|---|---|---|
| cluster 수 입력 | 불필요 | 불필요 |
| 대신 필요한 값 | 반경 threshold | cluster penalty `lambda` |
| 대량 streaming | 강함 | 별도 online 절충 필요 |
| centroid 재배치 | CF 누적 평균 | 반복 평균 갱신 |
| 전역 목적함수 | 명시적이지 않음 | distortion + `lambda * K` |
| 입력 순서 영향 | 있음 | batch는 비교적 작고 one-pass는 큼 |
| 현재 코드 비용 | scikit-learn으로 작음 | custom 구현 필요 |

둘 다 동일한 whitened 32차원 latent와 동일한 held-out regret 평가를 사용해야 알고리즘 자체의 차이를 비교할 수 있다. 따라서 BIRCH를 처리량 기준선으로 먼저 구현하고, DP-means를 품질 비교 대상으로 두는 기존 추천을 유지한다.

### Gaussian mixture와 EM

Gaussian mixture model은 latent가 다음 밀도에서 생성된다고 가정한다.

```text
p(z) = sum_k pi_k * Normal(z | mu_k, Sigma_k)
```

어느 component에서 표본이 왔는지가 숨은 변수이므로 EM을 흔히 사용한다.

1. E-step: 현재 `pi`, `mu`, `Sigma`로 component별 responsibility를 계산한다.
2. M-step: responsibility 가중치로 mixture weight, mean과 covariance를 다시 계산한다.
3. likelihood 변화가 작아질 때까지 반복한다.

따라서 사용자가 떠올린 `공분산 + 정규분포 + EM`의 연결은 맞다. 다만 보통의 GMM-EM은 component 수 `K`를 고정해야 한다. K를 자동으로 늘리려면 여러 K의 BIC 비교, split-and-merge EM, Dirichlet-process Gaussian mixture 같은 추가 구조가 필요하다. 실용적인 Bayesian mixture 구현도 보통 최대 component 수를 둔 finite truncation과 variational inference를 사용한다.

또한 GMM은 관측 밀도를 잘 설명하도록 cluster를 나눈다. 이 프로젝트의 목표는 같은 action decision과 EV를 갖는 상태를 묶는 것이므로, 빈도가 높은 저중요 영역을 여러 Gaussian으로 쪼개고 희귀하지만 전략적으로 중요한 상태를 합칠 수 있다. soft assignment와 불확실성이 꼭 필요해질 때는 유용하지만 첫 bucket 기준으로는 목적이 어긋난다.

DP-means는 Dirichlet-process Gaussian mixture의 small-variance limit에서 유도된 hard-clustering 방법이다. 따라서 `Gaussian mixture의 자동 component 생성 아이디어를 단순한 centroid와 penalty로 축약한 것`으로 이해할 수 있다.

### 최종 권장 실험

1. policy/action-Q loss로 32차원 encoder를 학습하고 고정한다.
2. train latent에 shared Ledoit-Wolf covariance를 맞추고 whitening한다.
3. street와 legal-action mask별로 `Birch(n_clusters=None)` threshold sweep을 수행한다.
4. bucket 수, 평균/p95 regret, best-action disagreement와 순서 민감도를 기록한다.
5. 같은 whitened latent에 batch DP-means를 실행해 품질과 처리량을 비교한다.
6. 두 방법 모두 타원형 구조를 제대로 표현하지 못한다는 증거가 있을 때만 diagonal GMM을 검토한다.

이 설계에서는 Mahalanobis가 두 clustering 알고리즘 앞의 공통 metric layer이고, Gaussian 분포와 EM은 3단계 이후의 별도 확장이다.

### 근거

- Zhang, Ramakrishnan, Livny (1996), BIRCH: An Efficient Data Clustering Method for Very Large Databases: https://doi.org/10.1145/233269.233324
- Kulis, Jordan (2012), Revisiting k-means: New Algorithms via Bayesian Nonparametrics: https://icml.cc/2012/papers/291.pdf
- Ledoit, Wolf (2004), A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices: https://doi.org/10.1016/S0047-259X(03)00096-4
- Dempster, Laird, Rubin (1977), Maximum Likelihood from Incomplete Data via the EM Algorithm: https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

### 추천 커밋 메시지

`docs: detail Mahalanobis adaptive clustering design`

## Soft centroid mixture와 cluster attention (2026-07-15)

상태: `기법 선택 보충`

### 구조의 해석

cluster 수가 늘어날 때 32차원 latent 자체가 늘어나는 것은 아니다. prototype key와 해당 prototype이 보관하는 strategy value의 행이 늘어난다.

```text
query:   y(o)              shape [32]
keys:    centroid_k        shape [K, 32]
values:  action_Q_k        shape [K, 6]
scores:  score(o, k)       shape [K]
```

현재 observation이 query, centroid가 key, cluster별 action-Q 또는 policy가 value가 된다. score를 softmax하여 value를 섞으면 single-query attention이면서 mixture-of-experts 또는 normalized RBF network로 해석할 수 있다. 별도의 Transformer는 필요하지 않다.

### dot product와 거리 score

직접 dot product를 쓰면 다음과 같다.

```text
score_k = y^T c_k / temperature
w_k = softmax(score)_k
```

하지만 vector norm을 제한하지 않으면 큰 norm을 가진 centroid가 방향 유사성과 무관하게 높은 score를 받을 수 있다. query와 centroid를 L2 정규화하면 cosine attention이 되고, 단위 vector에서는 다음 관계가 성립한다.

```text
||y - c_k||^2 = 2 - 2 y^T c_k
```

따라서 단위 vector에서 dot-product softmax와 negative squared-distance softmax는 temperature 차이만 있는 같은 순위를 만든다. 앞에서 shared covariance whitening을 사용하기로 했으므로 첫 구현은 metric과 직접 연결되는 다음 score가 더 자연스럽다.

```text
score_k = -||y - c_k||^2 / (2 * tau^2)
w_k = softmax(score)_k
```

whitening 전 좌표로 쓰면 이는 shared Mahalanobis RBF이다. Gaussian mixture의 responsibility에 있는 `log pi_k`를 더하면 빈도가 높은 cluster prior까지 반영할 수 있지만, 전략 bucket에서는 방문 빈도가 전략 중요도를 덮을 수 있으므로 첫 기준선은 equal prior로 둔다.

### 전략을 섞는 방법

각 cluster에 action별 평균 EV인 `q_k[a]`를 보관하고 다음처럼 섞는다.

```text
q_mix[a] = sum_k w_k * q_k[a]
action = argmax_a q_mix[a]
```

UCT policy를 직접 보관한다면 categorical mixture는 다음이다.

```text
pi_mix[a] = sum_k w_k * pi_k[a]
```

각 policy의 logit을 평균한 뒤 softmax하는 것은 확률 mixture와 다른 연산이므로 첫 구현에서는 사용하지 않는다. action-Q mixture가 같은 EV 기준을 유지하고 regret도 직접 측정할 수 있으므로 우선한다.

Soft mixture는 cluster 경계에서 행동이 갑자기 바뀌는 문제를 줄이고, 표본이 적은 상태가 이웃 centroid의 값을 빌릴 수 있게 한다. 반대로 서로 다른 이유로 같은 위치에 놓인 cluster를 섞으면 raise와 fold 같은 상반된 전략을 부정확하게 평균낼 수 있다. 따라서 street와 legal-action mask는 여전히 hard partition하고 held-out action regret으로 soft mixing의 이득을 판정한다.

### hard bucket과의 연속성

temperature `tau`가 작아질수록 가장 가까운 centroid 하나의 weight가 1에 가까워진다.

```text
tau -> 0: hard nearest-centroid bucket
tau 증가: 여러 centroid의 local strategy mixture
```

따라서 hard BIRCH와 soft mixture는 별도 모델이 아니다. 동일한 encoder, whitening, centroid와 Q 통계를 두고 `tau`만 바꾸어 비교할 수 있다. held-out mean/p95 regret가 가장 작은 `tau`를 고르고, soft 방식이 hard 방식보다 개선되지 않으면 추가 구조 없이 hard bucket을 유지한다.

### 계산량과 sparse mixture

모든 K개 centroid를 scoring하면 상태 하나당 `O(K * 32)` 연산이 필요하다. K가 수백이면 충분히 작지만 수만 개로 늘면 UCT simulation 안에서 부담이 된다. 처음에는 dense matrix multiplication으로 측정하고, 병목이 확인된 뒤에만 가장 가까운 `M=4`개 centroid를 선택해 softmax한다.

```text
candidates = top_4 nearest centroids
w = softmax(scores[candidates])
q_mix = sum(w * q[candidates])
```

BIRCH의 tree 또는 별도 nearest-neighbor index는 top-k 탐색이 실제 병목일 때만 연결한다. 각 상태의 K차원 weight를 DB에 저장하지 않고 centroid version과 필요시 선택된 top-k ID만 기록한다.

### centroid 증가와 학습 안정성

BIRCH 또는 DP-means가 새 centroid를 만들면 다음 통계 한 행만 추가한다.

```text
centroid[32], action_Q[6], visits[6], return_variance[6], sample_count
```

첫 구현에서 centroid는 neural parameter로 만들지 않고 표본 평균으로 갱신한다. 동적으로 `nn.Parameter` 행을 추가하면 optimizer state 변경, checkpoint 호환성과 cluster collapse 문제가 함께 생긴다. encoder와 policy/Q head는 gradient로 학습하고, centroid와 cluster Q는 통계적으로 갱신하는 분리를 먼저 유지한다.

Soft mixture가 확실히 개선된 뒤에만 다음 learned gate를 비교한다.

```text
query = W_q y
key_k = W_k c_k
score_k = query^T key_k / sqrt(d)
```

이 경우 cluster의 거리 의미가 학습 중 이동하므로 encoder, query/key projection, centroid를 같은 version으로 묶어야 한다.

### 첫 비교 실험

1. whitened BIRCH로 centroid와 cluster별 action-Q를 만든다.
2. hard nearest-centroid의 held-out regret를 측정한다.
3. 동일 centroid에서 `tau` sweep으로 dense RBF soft mixture를 측정한다.
4. soft 방식이 개선되면 top-4 mixture의 품질과 속도를 비교한다.
5. learned dot-product gate는 위 결과가 유효한 뒤에만 구현한다.

이 순서라면 clustering, Mahalanobis와 attention을 한 번에 새로 학습하지 않고 각 요소가 실제로 주는 개선을 분리해 확인할 수 있다.

### 근거

- Jacobs, Jordan, Nowlan, Hinton (1991), Adaptive Mixtures of Local Experts: https://doi.org/10.1162/neco.1991.3.1.79
- Bugmann (1998), Normalized Gaussian Radial Basis Function Networks: https://doi.org/10.1016/S0925-2312(98)00027-7
- Vaswani et al. (2017), Attention Is All You Need: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf

### 추천 커밋 메시지

`docs: design soft centroid strategy mixture`

## 가변 K Gaussian split-EM과 분열 검정 (2026-07-16)

상태: `기법 선택 보충`

### 결론

EM 바깥에 component split을 제안하고 model-selection 기준으로 채택하는 outer loop를 두면 K가 증가하는 Gaussian mixture를 만들 수 있다. 다만 평균과 공분산만으로는 split 필요성을 판정할 수 없다. 단일 Gaussian과 두 mode의 mixture가 같은 평균과 공분산을 가질 수 있으므로, split 검정에는 해당 component가 담당하는 실제 latent 표본이나 그보다 높은 차수의 통계가 필요하다.

이 프로젝트의 첫 Gaussian 버전은 다음 조합이 적절하다.

```text
shared shrinkage whitening
  -> diagonal-covariance GMM
  -> projection Gaussianity test로 split 후보 생성
  -> local 1-vs-2 component EM
  -> held-out likelihood/BIC와 action-Q regret로 split 승인
  -> responsibility로 cluster Q를 soft mixture
```

### 알려진 분열 계열과 차이

- G-means는 cluster의 주요 분할축으로 표본을 투영하고 Anderson-Darling Gaussianity test를 사용해 split을 결정한다. K를 자동 증가시키는 명시적인 통계 검정이 있다는 장점이 있다.
- X-means는 한 cluster와 두 child model의 BIC/AIC를 비교하여 split을 채택한다. Gaussianity p-value보다 모델 복잡도 penalty를 직접 반영한다.
- Dip-means는 거리 분포에 dip test를 적용해 unimodality가 깨지는 cluster를 분할한다. Gaussian 가정보다 약하지만 pairwise distance 비용이 커질 수 있다.
- Split-and-Merge EM은 component를 분할하거나 병합한 뒤 EM으로 다시 최적화한다. 고전적인 SMEM은 동일한 K 안에서 local optimum을 탈출하려고 split과 merge를 동시에 수행하는 경우가 많다. K 자체를 늘리려면 split 전후의 penalized score를 비교하는 별도 model-selection outer loop가 필요하다.

따라서 `EM이 스스로 새 component를 만든다`기보다 `outer loop가 split을 제안하고, 고정 K EM이 각 후보의 parameter를 다시 맞춘다`고 이해하는 편이 정확하다.

### 분열 후보 생성

component `k`의 responsibility가 큰 표본을 모으고 다음 조건을 먼저 확인한다.

```text
effective_count_k = (sum_i r_ik)^2 / sum_i r_ik^2
```

effective count가 작으면 split과 covariance 추정을 건너뛴다. 충분한 표본이 있으면 weighted covariance의 가장 큰 eigenvector 방향으로 두 child mean을 초기화한다.

```text
mu_left  = mu_k - delta * principal_axis
mu_right = mu_k + delta * principal_axis
weight_left = weight_right = weight_k / 2
```

그 뒤 해당 표본만 대상으로 1-component model과 2-component local EM을 맞춘다. 단순히 가장 큰 eigenvalue가 크다는 이유만으로 split하지 않는다. 길쭉하지만 단봉인 Gaussian은 하나의 covariance로 표현할 수 있기 때문이다.

### 통계적인 분열 신호

분열 필요성을 보는 threshold는 하나가 아니라 다음 역할별로 나눈다.

#### 1. Gaussianity 또는 unimodality test

주요 분할축에 투영한 1차원 표본에 Anderson-Darling test를 적용할 수 있다. 귀무가설인 단일 Gaussian을 유의수준 `alpha`에서 기각하면 split 후보가 된다. Gaussian을 가정하기 싫으면 dip test로 unimodality를 검사할 수 있다.

p-value만 최종 승인 기준으로 쓰지는 않는다. 표본이 매우 많으면 전략적으로 무의미한 작은 비정규성도 기각되고, 여러 cluster에서 반복 검정하면 false positive가 증가한다. 이 검정은 비싼 local EM을 실행할 후보를 줄이는 filter로만 사용한다.

#### 2. Penalized likelihood 또는 held-out likelihood

일반적인 BIC 표기는 다음과 같다.

```text
BIC = -2 * log_likelihood + parameter_count * log(sample_count)
```

두 component model의 BIC가 더 작으면 density 설명력 증가가 parameter 증가 비용을 이겼다는 뜻이다. 그러나 Gaussian mixture의 1-vs-2 likelihood-ratio는 component weight 0과 covariance singularity 때문에 단순한 chi-square threshold로 해석하기 어렵다. 첫 구현에서는 BIC와 별도 held-out negative log-likelihood를 사용한다.

#### 3. 최소 child support

outlier 몇 개가 별도 Gaussian을 만들지 않도록 두 child 모두 최소 effective count와 최소 mixture weight를 만족해야 한다. 이 조건은 유의수준보다 실제 과분할 방지에 중요하다.

#### 4. 전략적 split 기준

이 프로젝트의 최종 목적은 density가 아니라 action decision 보존이다.

```text
regret_before = regret(parent Q policy)
regret_after  = weighted regret(two child Q policies)
regret_gain   = regret_before - regret_after
```

Gaussian split이 likelihood를 높여도 held-out regret와 policy JS divergence가 거의 개선되지 않으면 bucket을 나누지 않는다. 반대로 density가 단봉이어도 parent 내부의 action-Q 차이 때문에 regret 제한을 넘으면 Q-aware split 후보를 별도로 만든다.

### 권장 승인 규칙

첫 구현은 복잡한 종합 점수보다 다음 gate를 순서대로 사용한다.

```text
1. parent effective count 충분
2. projection Gaussianity 기각 또는 parent p95 regret 초과
3. 두 child의 minimum support 충족
4. held-out BIC 또는 held-out NLL 개선
5. held-out regret가 설정한 최소량 이상 감소
```

모든 threshold는 train에서 정하고 최종 평가는 분리한 state split에서 수행한다. 같은 exact state의 반복 search가 train과 validation에 동시에 들어가지 않도록 canonical state 기준으로 나눈다.

### 공분산 저장 비용

32차원 component 하나의 대략적인 float32 저장량은 다음과 같다.

| covariance 형태 | 저장할 covariance 값 | component당 크기 |
|---|---:|---:|
| shared/tied full | 전체 모델에 528개 한 번 | 전체 약 2.1 KiB |
| spherical | 1개 | 4 B |
| diagonal | 32개 | 128 B |
| full symmetric | 528개 | 약 2.1 KiB |
| rank-4 + diagonal | 160개 | 640 B |

full covariance만 보면 1만 component에서 약 21 MiB이므로 저장 자체는 치명적이지 않다. 더 큰 문제는 component마다 528개 parameter를 안정적으로 추정할 표본이 필요하고, E-step에서 full Mahalanobis 계산이 `O(K * d^2)`라는 점이다. diagonal은 `O(K * d)`이며 global whitening 뒤에는 대부분의 공통 상관이 제거되므로 첫 버전에 충분히 합리적이다.

DB에는 mean, diagonal variance, mixture weight, action-Q 통계와 sample count만 저장한다. covariance와 precision을 둘 다 영구 저장하지 않고 covariance에서 precision 또는 Cholesky cache를 재생성한다.

### 표본을 별도로 저장해야 하는가

`N`, vector sum과 outer-product sum만 있으면 mean과 covariance는 복원할 수 있지만 bimodality는 검출할 수 없다. 즉 online sufficient statistics만으로 split test를 완결할 수 없다.

현재는 원본 compact observation이 SQLite에 있으므로 cluster별 대형 reservoir를 중복 저장하지 않는다. split epoch가 실행될 때 source DB에서 해당 cluster 표본을 다시 읽어 encoder로 재임베딩한다. 완전한 streaming split이 필요해질 때만 큰 component에 한정해 작은 reservoir를 둔다.

### Soft strategy mixture와의 결합

Diagonal GMM의 E-step responsibility는 이전에 제안한 soft centroid score의 확률적 버전이다.

```text
score_k = log pi_k
          - 0.5 * sum_j ((y_j - mu_kj)^2 / variance_kj)
          - 0.5 * sum_j log variance_kj

r_k = softmax(score)_k
Q_mix[a] = sum_k r_k * Q_k[a]
```

따라서 Gaussian split-EM을 선택하면 `cluster score -> softmax -> strategy mix` 구조가 자연스럽게 함께 구현된다. responsibility가 넓게 퍼진 상태는 cluster 경계의 불확실성을 나타내며, top-k responsibility만 사용하면 계산량도 제한할 수 있다.

### 최종 추천

full-covariance split-and-merge EM부터 시작하지 않는다. 첫 비교 모델은 `whitened diagonal split-GMM`으로 둔다.

1. BIRCH centroid로 초기 component를 만든다.
2. diagonal GMM EM으로 soft responsibility를 맞춘다.
3. G-means식 projection test로 split 후보를 찾는다.
4. local 1-vs-2 EM 뒤 held-out BIC/NLL과 EV regret로 승인한다.
5. 승인되는 split이 없을 때 종료한다.
6. 과분할이 실제 관측될 때만 merge pass를 추가한다.

이 모델은 Gaussian의 통계적 split 판단, 자동 K 증가와 soft strategy mixture를 모두 연결하면서도 공분산 저장과 계산을 통제한다.

### 근거

- Hamerly, Elkan (2003), Learning the k in k-means: https://papers.nips.cc/paper_files/paper/2003/hash/234833147b97bb6aed53a8f4f1c7a7d8-Abstract.html
- Pelleg, Moore (2000), X-means: Extending K-means with Efficient Estimation of the Number of Clusters: https://www.cs.cmu.edu/~dpelleg/kmeans.html
- Ueda, Nakano, Ghahramani, Hinton (2000), SMEM Algorithm for Mixture Models: https://doi.org/10.1162/089976600300015088
- Kalogeratos, Likas (2012), Dip-means: An Incremental Clustering Method for Estimating the Number of Clusters: https://proceedings.neurips.cc/paper/2012/hash/a8240cb8235e9c493a0c30607586166c-Abstract.html

### 추천 커밋 메시지

`docs: design adaptive split EM buckets`

## observ2vec bucket의 현재 선택 (2026-07-16)

상태: `1차 구현 기법 결정`

첫 구현은 `shared-whitened diagonal Gaussian split-EM`으로 결정한다.

```text
32d observ2vec
  -> shared Ledoit-Wolf whitening
  -> street/legal-mask별 1개 diagonal Gaussian에서 시작
  -> principal projection Gaussianity 또는 높은 parent regret로 split 후보 생성
  -> local 1-vs-2 diagonal EM
  -> minimum child support 확인
  -> held-out density가 악화되지 않고 EV regret가 감소할 때 split 승인
  -> responsibility top-k로 action-Q soft mixture
```

Gaussianity test는 `어디를 나눠 볼지` 결정하고, EV regret는 `그 분열을 전략 bucket으로 보존할지` 결정한다. cluster 수는 미리 정하지 않으며 승인되는 split이 없을 때 종료한다.

BIRCH는 첫 정확성 모델에서 제외한다. 표본 수가 커져 recursive split이 느려질 때 초기 centroid 생성과 데이터 축약에만 추가한다. DP-means는 Gaussian split 기준의 대조군으로 남긴다. Full covariance, learned dot-product gate, merge pass는 diagonal 모델의 residual correlation, soft-mixture 이득 또는 과분할이 실제 측정된 뒤에만 구현한다.

### 추천 커밋 메시지

`docs: select diagonal Gaussian split EM buckets`

## UCT 수집기의 1GiB 한도 해석 (2026-07-16)

상태: `동작 확인`

`uct_rollout.py --max-gib`는 `uct_nodes` table의 크기가 아니라 메인 SQLite 파일, WAL과 SHM의 전체 크기 합을 제한한다. 따라서 같은 DB에 이미 저장된 `q_values`도 한도에 포함된다.

크기 검사는 매 `flush_hands` batch를 저장한 직후 수행한다. DB가 이미 한도를 넘었어도 시작 즉시 종료하는 것이 아니라 첫 flush까지 실행하며, 마지막 batch와 동시 worker의 WAL만큼 한도를 조금 초과할 수 있다.

2026-07-16 확인 시점의 `ev_rollout_overnight_1g.sqlite3` 상태는 다음과 같다.

- SQLite + WAL + SHM: 약 `790.04 MiB`
- 1GiB까지 남은 크기: 약 `233.96 MiB`
- UCT unique root: `125,354`
- 저장된 UCT simulation: `64,181,248`

UCT에 별도 1GiB를 배정하려면 기존 mixed DB의 한도 의미를 바꾸기보다 새 출력 파일을 사용하는 것이 가장 단순하다.

```powershell
python -B uct_rollout.py --output replays/uct_rollout_1g.sqlite3 --hands 0 --seconds 28800 --simulations 512 --seed 303 --max-gib 1 --flush-hands 50 --progress-seconds 30
```

기존 mixed DB의 `uct_nodes`는 유효하므로 삭제하거나 이동하지 않는다. 추후 observ2vec 학습기는 여러 SQLite shard의 `uct_nodes`를 순회하게 만들면 병합 복사 없이 함께 사용할 수 있다.

### 추천 커밋 메시지

`docs: clarify UCT database size limit`

## observ2vec와 split-EM의 입출력 계약 (2026-07-16)

상태: `코드 리뷰 기반 1차 계약`

### 코드 리뷰 결과

현재 `uct_nodes` 한 행에는 첫 policy/Q encoder를 학습하는 데 필요한 자료가 있다.

- `state_json`: schema version, street, 네 카드 그룹, ante와 여섯 betting scalar, raise count, betting history
- `seat_index`: 현재 행동자의 실제 seat
- `legal_mask`: 여섯 EV action의 합법 여부
- action별 `visits`, `return_sum`, `return_sq_sum`
- `search_version`, `opponent_policy`, `simulation_budget`: target 생성 조건

다음 정보는 현재 행에 없다.

- 명시적인 opponent belief particle
- root 사이의 `(state, action, next_state)` transition
- sampled opponent hidden hand

첫 모델에서는 이들을 요구하지 않는다. 현재 action-Q는 observation과 모순되지 않는 상대 히든을 UCT가 uniform하게 표본화한 결과를 평균했으므로, uniform belief가 target에 암묵적으로 포함되어 있다. transition loss와 explicit belief encoder는 별도 자료가 생긴 뒤에만 추가한다.

현재 `canonical_state`는 `seat_index`와 `legal_mask`를 JSON 밖에 저장하므로 loader가 반드시 세 값을 함께 읽어야 한다. 또한 비공개 카드 나열 순서는 전략적으로 무의미하지만 현재 serializer가 보존한다. tensorizer는 각 suit permutation에서 비공개 카드만 정렬하고 공개 카드 도착 순서를 보존하는 training canonicalization을 적용한다.

### DB row에서 학습 표본으로

학습기는 같은 조건의 UCT 자료만 한 dataset으로 묶는다.

```text
filter:
  search_version = uct-v1
  opponent_policy = random
  simulation_budget = 512

key:
  state_json, seat_index, search_version,
  opponent_policy, simulation_budget
```

여러 SQLite shard에서 같은 key가 나오면 action별 visit과 return 통계를 합산한 뒤 tensor로 만든다. train/validation/test 분리는 합산된 canonical key의 stable hash로 수행하여 같은 상태가 둘 이상의 split에 들어가지 않게 한다.

### Encoder 입력

DB의 JSON을 그대로 neural model에 넘기지 않고 다음 typed tensor로 변환한다. `B`는 batch 크기다.

```text
self_hidden_rank/suit       int64 [B, 3]
self_hidden_mask            bool  [B, 3]

self_public_rank/suit       int64 [B, 4]
self_public_mask            bool  [B, 4]
opponent_public_rank/suit   int64 [B, 4]
opponent_public_mask        bool  [B, 4]
discarded_rank/suit         int64 [B, 1]
discarded_mask              bool  [B, 1]

history_street/actor/action int64 [B, 36]
history_mask                bool  [B, 36]

street                      int64 [B]
seat_index                  int64 [B]
legal_mask                  bool  [B, 6]
betting_scalars             float [B, 7]
```

카드 code는 `rank = code // 4`, `canonical_suit = code % 4`로 분리한다. self hidden만 shared card encoder 뒤 mean pooling하여 순서 불변으로 만들고, 공개 카드는 공개 시점과 betting history를 연결해야 하므로 slot 순서를 유지한다.

일곱 betting scalar는 다음 순서로 고정한다.

```text
log1p(pot / ante)
log1p(my_invested / ante)
log1p(opponent_invested / ante)
log1p(my_round_bet / ante)
log1p(opponent_round_bet / ante)
log1p(call_amount / ante)
raise_count / raise_cap
```

ante의 절대값은 EV 모드의 scale-invariant 입력에서 제외한다. `search_version`, `opponent_policy`, `simulation_budget`도 feature가 아니라 dataset 조건으로 사용한다.

### Encoder target

action 순서는 기존 `ev_rollout.ACTIONS`와 동일하게 고정한다.

```text
[CHECK, BBING, QUARTER, HALF, CALL, FOLD]
```

각 action의 target은 다음이다.

```text
policy_target[a] = visits[a] / sum_legal visits
q_target[a] = return_sum[a] / visits[a] / ante
q_variance[a] = return_sq_sum[a] / visits[a]
                - (return_sum[a] / visits[a])^2
target_mask[a] = legal_mask[a] and visits[a] > 0
```

첫 loss는 legal action만 사용한다.

```text
L = KL(policy_target || policy_pred)
    + lambda * weighted_Huber(q_target, q_pred)
    + mu * symmetry_loss
```

Q loss weight는 첫 기준선에서 `sqrt(visits)`를 batch 안에서 정규화해 사용한다. `return_sq_sum`은 target uncertainty와 calibration 평가에 사용하고, inverse-variance weight는 작은 분산 action이 loss를 독점하는 문제가 실제 관측될 때만 추가한다.

### Encoder 출력

```text
embedding       z              float [B, 32]
policy_logits                  float [B, 6]
action_q                       float [B, 6]
```

32차원 `z`는 별도 reconstruction decoder 없이 policy/Q head가 전략적 의미를 부여한다. 학습 완료 뒤 encoder를 고정하고 clustering 자료를 생성한다.

### Split-EM 입력

```text
z               float [N, 32]
street          int   [N]
legal_mask      bool  [N, 6]
q_target        float [N, 6]
policy_target   float [N, 6]
sample_weight   float [N]
canonical_key          [N]
```

street와 legal mask는 hard stratum key다. Gaussian component는 서로 다른 stratum을 가로질러 생성되지 않는다. shared Ledoit-Wolf whitening은 train `z`에만 fit하고 validation/test에는 같은 transform을 적용한다.

### Split-EM 출력 artifact

Ragged component를 pickle object로 저장하지 않고 numeric array와 작은 schema JSON으로 분리한다.

```text
whitening_mean          float32 [32]
whitening_transform     float32 [32, 32]

stratum_keys            int     [S, 2]     # street, legal bits
stratum_offsets         int     [S + 1]

component_mean          float32 [K, 32]
component_diag_var      float32 [K, 32]
component_log_prior     float32 [K]
component_q             float32 [K, 6]
component_policy        float32 [K, 6]
component_support       float32 [K]
```

`stratum_offsets`로 각 stratum의 component 구간을 찾는다. model artifact에는 encoder version, state schema, action order, source search version, opponent policy와 split threshold를 함께 기록한다.

### 추론 입력과 출력

Agent가 한 번 행동할 때는 기존 `get_ai_state()`와 `valid_actions`를 tensorizer에 전달한다.

```text
observation
  -> tensors
  -> encoder z[32], direct policy/Q[6]
  -> whitening
  -> current street/legal stratum components
  -> diagonal Gaussian scores
  -> top-M responsibility
  -> mixed action-Q[6]
```

첫 `M`은 4로 두되 component 수가 4보다 작으면 전부 사용한다. clustering layer의 반환 계약은 다음으로 둔다.

```text
mixed_q          float [6]
mixed_policy     float [6]
component_ids    int   [M]
responsibilities float [M]
```

실제 agent는 illegal action을 mask한 뒤 `mixed_q`의 argmax를 선택한다. stochastic policy가 필요해질 때만 `mixed_policy` sampling을 추가한다. UI와 진단에서는 component ID, responsibility와 32차원 embedding을 표시할 수 있지만 trajectory DB에는 dense K차원 weight를 저장하지 않는다.

### 구현 과정

1. 여러 UCT SQLite를 읽고 중복 key를 합산하는 streaming loader를 만든다.
2. state JSON을 위 fixed-shape tensor로 바꾸는 tensorizer와 round-trip shape test를 만든다.
3. 작은 policy/Q encoder를 학습하고 held-out loss와 best-action agreement를 저장한다.
4. encoder를 고정해 32차원 latent를 생성하고 shared whitening을 fit한다.
5. stratum별 diagonal split-EM을 실행하고 split 전후 NLL과 EV regret를 기록한다.
6. hard nearest component, dense responsibility와 top-4 mixture의 regret를 비교한다.
7. 선택된 artifact를 읽는 agent는 마지막에 별도 파일로 만들고 `PokerAgent`를 상속한다.

### 코드 리뷰에서 확인된 구현 전 수정점

- 기존 `canonical_state` DB를 다시 쓰지 않고 tensorizer에서 hidden-order canonicalization을 보정한다.
- `seat_index`와 `legal_mask`를 state JSON에서 찾지 말고 `uct_nodes` column에서 읽는다.
- 공개 카드 slot은 정렬하지 않는다.
- `last_action`은 visit argmax와 중복되므로 학습 target으로 사용하지 않는다.
- explicit belief와 transition decoder를 첫 구현 범위에 넣지 않는다.

### 추천 커밋 메시지

`docs: define observ2vec clustering IO contract`

## 수집 중 clustering과 Gaussian local geometry (2026-07-16)

상태: `운영 및 기하 해석`

### 수집 중 같은 SQLite를 읽을 수 있는가

현재 DB는 WAL mode이므로 원칙적으로 reader와 writer가 동시에 동작할 수 있고 reader는 읽기 시작 시점의 snapshot을 본다. 그러나 긴 clustering scan은 다음 문제를 만든다.

- long read transaction이 WAL checkpoint의 완료와 reset을 막아 WAL이 커질 수 있다.
- `uct_rollout.py --max-gib`는 WAL까지 합산하므로 clustering reader 때문에 수집기가 예상보다 빨리 size limit에 도달할 수 있다.
- SQLite 손상과 별개로 같은 SSD의 순차 scan, UCT UPSERT와 CPU 학습이 서로 처리량을 빼앗는다.
- clustering 결과를 같은 DB에 쓰면 두 번째 writer가 되어 lock 경쟁이 생긴다.
- 현재 `uct_nodes`는 `WITHOUT ROWID`이고 update timestamp가 없으므로 새 행과 기존 행의 UPSERT 증가분을 정확히 증분 추적하기 어렵다.

따라서 full training과 split-EM은 live DB를 직접 오래 읽지 않는다. 가장 안전하고 단순한 운영 방식은 shard rotation이다.

```text
collector -> active_002.sqlite3
trainer   -> finished_001.sqlite3 (immutable)

다음 주기:
collector -> active_003.sqlite3
trainer   -> finished_002.sqlite3
```

현재처럼 mixed DB 하나를 수집 중이라면 수집 종료 뒤 읽는 것이 우선이다. 수집 중 실험이 꼭 필요하면 SQLite Online Backup API로 별도 snapshot을 만든 뒤 그 파일에서 학습한다. WAL mode의 메인 `.sqlite3` 파일만 OS copy하면 WAL의 commit을 놓칠 수 있으므로 사용하지 않는다. cluster artifact도 source SQLite가 아니라 별도 `.pt`, `.npz`, `.json`에 저장한다.

### 현재 SQLite 버전 주의

2026-07-16 확인 결과 현재 collector가 사용하는 Python `sqlite3`는 SQLite `3.35.5`, 설치된 CLI는 `3.44.3`이다. SQLite 공식 문서에 따르면 WAL mode의 여러 connection이 동시에 write/checkpoint할 때 발생할 수 있는 드문 WAL-reset bug는 `3.44.6`, `3.50.7`, `3.51.3` 이상 계열에서 수정되었다.

Read-only clustering 하나가 이 race를 직접 만드는 것은 아니지만, 현재 버전에서는 같은 DB에 여러 collector writer나 별도 checkpoint process를 추가하지 않는다. 과거의 `같은 DB에 두 writer를 먼저 측정` 지침은 이 확인으로 폐기하고 worker마다 별도 shard를 사용한다.

### 각 Gaussian은 local standard-normal chart인가

그렇게 해석할 수 있다. component `k`의 covariance를 `Sigma_k = L_k L_k^T`로 두면 다음 local 좌표에서 해당 component는 표준정규가 된다.

```text
y_k(z) = L_k^-1 (z - mu_k)

z ~ Normal(mu_k, Sigma_k)
=> y_k ~ Normal(0, I)
```

Mahalanobis 거리는 local chart의 원점까지 Euclidean 거리다.

```text
d_k^2(z) = (z - mu_k)^T Sigma_k^-1 (z - mu_k)
         = ||y_k(z)||^2
```

따라서 각 Gaussian은 `자기 영역을 평평하게 펴는 서로 다른 좌표계`를 가진다. GMM responsibility는 여러 local chart 중 현재 점을 가장 잘 설명하는 chart와 그 혼합 비율을 선택한다.

### 모든 component를 하나의 평면으로 동시에 펼칠 수 있는가

하나의 invertible linear transform `A`가 모든 component를 동시에 표준정규로 만들려면 다음을 모두 만족해야 한다.

```text
A Sigma_k A^T = I  for every k
```

이는 사실상 모든 `Sigma_k`가 같은 경우에만 가능하다. covariance가 다르면 component별 `A_k = L_k^-1`가 필요하다.

공간 위치에 따라 local precision이 달라진다고 보고 다음처럼 부드러운 metric field를 만들 수 있다.

```text
G(z) = sum_k responsibility_k(z) * Sigma_k^-1

path_length = integral sqrt(dz^T G(z) dz)
```

각 `Sigma_k^-1`가 positive definite이면 이 가중합도 positive definite이므로 Riemannian metric으로 볼 수 있다. metric이 위치마다 달라지면 최단 경로인 geodesic은 휘어질 수 있다. 한 점 근처에서는 normal coordinate로 평평하게 만들 수 있지만 curvature가 0이 아니면 하나의 동일 차원 Euclidean 좌표로 전역 왜곡 없이 펼칠 수는 없다.

이 `G(z)`는 현재 모델을 해석하기 위한 실용적인 local metric이며 Gaussian-mixture Fisher information metric의 유일한 정의라고 주장하지 않는다.

### 실용적인 평면 사영은 responsibility simplex

전역 geodesic을 계산하는 대신 다음 map을 사용한다.

```text
phi(z) = [r_1(z), r_2(z), ..., r_K(z)]
sum_k r_k(z) = 1
```

이는 32차원 latent를 K차원 probability simplex로 사영한다. component 중심에 가까운 점은 해당 component의 one-hot vertex에 가깝고 경계의 점은 여러 vertex 사이에 놓인다.

```text
cluster k         <-> e_k
soft state        <-> mixture of e_k
strategy Q(z)     = phi(z)^T component_Q
```

이것이 앞서 제안한 `cluster마다 score 차원 하나가 늘고 softmax로 전략을 섞는다`는 구조의 정확한 기하학적 표현이다. K차원 dense vector를 저장하지 않고 추론 시 top-k responsibility만 계산한다.

### Normalizing flow는 필요한가

invertible nonlinear transform인 normalizing flow를 학습하면 복잡한 전체 density와 표준정규 사이의 map을 만들 수 있다. 그러나 이 목적은 density를 평평하게 만드는 것이며 action-Q가 같은 상태를 보존한다는 보장은 없다. component identity와 해석 가능성도 flow 안으로 숨고 학습 비용이 크게 늘어난다.

현재는 local whitening과 responsibility simplex로 충분하다. 실제 geodesic distance가 Euclidean 또는 responsibility distance보다 held-out EV regret를 유의하게 줄인다는 증거가 생길 때만 manifold metric이나 flow를 검토한다.

### 최종 운영 및 모델 선택

1. collector와 trainer는 서로 다른 SQLite shard를 사용한다.
2. finished shard에서 32차원 encoder와 split-GMM을 학습한다.
3. 각 Gaussian은 diagonal local whitening chart로 해석한다.
4. 공통 표현은 top-k responsibility simplex와 mixed action-Q다.
5. live DB 장시간 scan, 같은 DB writer, global geodesic과 normalizing flow는 첫 구현에서 제외한다.

### 근거

- SQLite, Write-Ahead Logging: https://sqlite.org/wal.html
- SQLite, Online Backup API: https://sqlite.org/backup.html
- Fetaya, Ullman (2015), Learning Local Invariant Mahalanobis Distances: https://proceedings.mlr.press/v37/fetaya15.html
- Rezende, Mohamed (2015), Variational Inference with Normalizing Flows: https://proceedings.mlr.press/v37/rezende15.html
- Shao, Kumar, Fletcher (2018), The Riemannian Geometry of Deep Generative Models: https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Shao_The_Riemannian_Geometry_CVPR_2018_paper.pdf

### 추천 커밋 메시지

`docs: define snapshot clustering and local Gaussian geometry`

## CFR 대화에서 선별한 에이전트 설계 원칙 (2026-07-16)

상태: `아이디어 선별 및 현재 구현과의 연결`

### 한 문장 결론

현재 프로젝트의 가장 일관된 방향은 `정확한 게임 규칙 -> 후반부터 탐색 -> 탐색 결과를 전역 모델로 압축 -> 가치 오차가 큰 영역만 동적으로 세분화`다.

최종 목표는 5인 불완전정보 게임이지만, 지금 수집 중인 heads-up UCT 데이터는 모델과 자료구조를 검증하기 위한 첫 단계다. 특히 현재 `uct-v1`은 랜덤 상대에 대한 root search 결과이므로 `Q*`, Nash equilibrium 또는 5인 최적 전략으로 해석하지 않는다.

### 반드시 유지할 구분

- 에이전트의 현재 관측은 완전한 게임 state가 아니다. 정책은 `pi(a | information history)` 또는 충분한 belief가 있을 때 `pi(a | belief)`로 정의한다.
- 모든 행동을 방문하는 coverage는 잘못된 값을 수정할 기회를 줄 뿐이다. policy improvement나 search가 없으면 rollout 평균은 사용한 continuation policy의 `Q^pi`에 수렴한다.
- MCTS/UCT는 현재 상태 주변의 임시 정밀 table이고, encoder와 cluster model은 여러 탐색 결과를 전역적으로 손실 압축하는 장치다.
- 카드 equity, 족보와 팟 계산처럼 환경에서 정확히 계산되는 값과, 그 값에 임계치를 적용해 행동을 결정하는 휴리스틱 정책을 구분한다.
- 정책 분포 자체가 `pi*`와 얼마나 닮았는지보다 실제 chip-EV 손실, action regret, held-out search 일치도를 평가한다.

### 채택할 아이디어

1. **학습하지 않아도 되는 것은 정확하게 계산한다.** 족보, 백스트레이트와 마운틴 순서, split pot, side pot, 합법 행동과 chip net은 simulator의 결정적 규칙으로 둔다.
2. **첫 목적함수는 한 라운드의 net chip이다.** 승률은 진단과 belief 보조 target으로 사용할 수 있지만 캐시/EV 정책의 최종 value는 행동별 기대 net chip이다.
3. **후반에서 앞으로 학습한다.** terminal evaluator, late betting, full betting, H4 순으로 horizon을 늘린다. H4는 continuation policy의 오차와 전체 return 분산을 모두 받으므로 먼저 풀 대상이 아니다.
4. **휴리스틱은 soft prior로만 사용한다.** 기존 heuristic agent는 rollout policy, warm-start, 테스트 상대가 될 수 있지만 합법 행동을 제거하거나 최종 target을 고정하지 않는다.
5. **지역 탐색과 전역 압축을 반복한다.** UCT가 현재 상태의 행동가치를 보정하고, encoder/cluster model이 그 결과를 비슷한 상태에 재사용한다. 이후 탐색 결과를 다시 모델에 학습시키는 expert-iteration 구조를 지향한다.
6. **정확한 대칭성만 먼저 압축한다.** suit 이름의 순열처럼 보상과 전이를 보존하는 canonicalization은 사용한다. 전략적으로 비슷해 보인다는 이유만으로 상태를 합치는 것은 별도의 근사 오차로 취급한다.
7. **고정 크기 bucket보다 가치 기반 dynamic abstraction을 사용한다.** density는 split 후보를 찾는 데 쓰고, 실제 분열 승인은 held-out EV regret, action 순위 변화와 충분한 표본 수로 결정한다.

### Dynamic Gaussian atlas의 역할

첫 모델은 현재 결정대로 `32차원 observ2vec + shared whitening + diagonal Gaussian split-EM`을 사용한다.

```text
state/history
    -> 32d encoder
    -> Gaussian responsibilities
    -> top-k component Q/policy mixture
    -> UCT prior 또는 행동 점수
```

각 component는 최종 정책을 고정하는 bucket이 아니라 다음을 공유하는 local expert다.

- component 중심의 행동별 Q와 policy
- diagonal covariance에 따른 local whitening 좌표
- 행동별 유효 표본 수와 return variance
- top-k responsibility를 통한 soft mixture

Gaussian coverage는 `이 상태가 offline 분포와 비슷한가`를 나타내는 기하학적 gate일 뿐, 행동가치의 정확성을 보증하지 않는다. 다음 조건도 함께 만족할 때만 기존 local expert를 강하게 신뢰한다.

- held-out action-Q 오차가 작음
- top action 사이의 gap이 충분함
- local expert들의 disagreement가 작음
- online search와의 regret가 작음

atlas 밖이거나 위 조건을 위반한 상태는 추가 rollout 대상으로 보낸다. 한 개의 outlier마다 component를 만들지 않고 novelty 표본이 충분히 모이고 전략적 오차가 반복될 때만 split 또는 신규 component를 승인한다.

### Belief와 5인 확장

상대 hidden hand의 정확한 joint table은 5인에서 너무 커진다. 최종 구조에서는 관측 history로부터 joint hidden-card particle을 만들고, 불완전정보 search가 particle을 root sampling하는 방향이 자연스럽다.

다만 belief model, ISMCTS/POMCP, 5인 self-play를 현재 clustering 실험과 동시에 넣지 않는다. 순서는 다음과 같다.

```text
heads-up random-opponent UCT로 데이터 계약 검증
-> heads-up opponent population
-> belief-aware heads-up search
-> 3인 상호작용 검증
-> 5인 population self-play
```

heads-up에서 얻은 전략을 5인 전략으로 간주하지 않는다. heads-up은 value/search/representation 오류를 측정할 수 있는 축소 실험이다.

### 구할 수 있는 오차의 형태

정책 확률 비율 `pi_current / pi*`는 support가 0인 행동, 여러 최적 정책과 중요하지 않은 상태 때문에 적절한 주 지표가 아니다. 다음과 같이 실제 성능 오차를 나누는 편이 낫다.

```text
total loss
<= belief error
 + representation/cluster error
 + finite-search error
 + value approximation error
 + optimization/distillation error
```

한 의사결정에서 모든 행동에 대해 Q 오차가 최대 `epsilon`이면 근사 Q의 greedy 행동이 만드는 regret는 최대 대략 `2 * epsilon`이다. 유한 horizon에서는 단계별 오차를 합산하는 형태로 전체 손실을 추적할 수 있다. 현재 프로젝트에서는 전역 정리를 먼저 주장하지 않고 held-out UCT root에서 다음을 측정한다.

- action별 Q 오차와 confidence interval
- best-action agreement
- mixture policy의 search regret
- hard bucket 대비 soft top-k mixture의 regret 감소
- component 수 대비 저장량과 추론 시간

### 검증 후에만 도입할 아이디어

| 아이디어 | 현재 판단 | 도입 조건 |
|---|---|---|
| 정확한 equity | 보조 feature/target | raw 입력만 쓴 모델과의 ablation에서 표본 효율 개선 |
| online OOD deep search | 유망 | offline atlas가 held-out regret를 실제로 줄인 뒤 |
| H4 action-value table | 후순위 | full Bet continuation이 안정되고 모든 H4 경계 상태를 평가할 수 있을 때 |
| stack residual | 후순위 | 고정 기준 stack의 chip-EV 모델이 먼저 작동할 때 |
| tournament residual/meta-policy | 별도 단계 | cash/EV 하위 모델과 payout simulator가 검증된 뒤 |
| Beta-box 또는 radial Beta cluster | 연구 가설 | Gaussian이 bounded/asymmetric latent를 반복적으로 잘못 모델링할 때 |
| MoE expert 추가 | 보류 | 단일 encoder와 Gaussian soft mixture가 측정상 underfit일 때 |
| Transformer history encoder | 보류 | 현재 고정 길이 MLP encoder가 history ablation에서 부족할 때 |
| progressive network growth | 보류 | 작은 모델의 training error로 capacity 부족이 확인될 때 |
| potential reward shaping | 불필요 | terminal net-chip 학습이 실제 병목이고 안전한 potential이 있을 때만 |

Beta-box는 명시적 bounded support와 유연한 내부 밀도라는 장점이 있지만, 현재 Gaussian split-EM보다 fitting과 검증이 복잡하다. 따라서 첫 atlas를 대체하지 않고 동일 데이터에서 held-out likelihood와 EV regret를 비교할 수 있을 때만 실험한다.

### 현재 코드에 맞춘 다음 순서

1. 완결된 UCT shard만 읽고 `search_version`, `opponent_policy`, `simulation_budget`이 같은 row를 합친다.
2. state JSON을 고정 크기 tensor로 바꾸고 card/order/mask 규칙을 test로 고정한다.
3. `policy=visits/sum(visits)`, `Q=return_sum/visits/ante`를 예측하는 32차원 encoder를 학습한다.
4. frozen latent에서 shared whitening과 diagonal split-EM을 학습한다.
5. raw MLP, hard nearest component, dense mixture, top-4 mixture의 held-out regret를 비교한다.
6. top-4 mixture가 유효한 경우에만 online search budget gate와 novelty buffer를 추가한다.
7. 그 이후에 belief-aware search와 opponent population을 별도 버전으로 시작한다.

이 순서의 핵심은 clustering을 먼저 믿는 것이 아니라, 현재 UCT 데이터에서 clustering이 실제 action-value 손실을 줄이는지 가장 작은 실험으로 확인하는 것이다.

### 추천 커밋 메시지

`docs: distill agent design principles from CFR discussion`

## Clustering 비교 실험 순서 (2026-07-16)

상태: `실험 순서 확정`

### 결론

다음 순서가 가장 작고 해석하기 쉽다.

```text
raw encoder head
-> spherical k-means + dot-product gate
-> diagonal GMM/EM
-> value-aware dynamic split-EM
-> 측정된 실패가 있을 때만 별도 custom density
```

### Dot product와 centroid의 관계

Dot product는 그 자체로 cluster를 만드는 알고리즘이 아니라 현재 latent와 prototype의 유사도를 계산하는 방법이다. 단위 벡터 `u`, `v`에서는 다음이 성립한다.

```text
||u - v||^2 = 2 - 2 * dot(u, v)
```

따라서 latent와 centroid를 L2 normalize하면:

- 최대 dot product
- 최소 cosine distance
- 최소 squared Euclidean distance

의 순위가 같다. 별도의 `dot-product 단계`와 `centroid 단계`를 중복 구현하지 않고 **spherical k-means로 centroid를 만든 뒤 dot product로 hard/soft assignment**한다.

Word2vec의 skip-gram objective는 어떤 상태들이 문맥상 함께 등장해야 하는지 정의해야 한다. 현재는 policy visit와 action-Q라는 더 직접적인 target이 있으므로 co-occurrence objective를 추가하지 않는다.

### 0단계: cluster 없는 기준선

32차원 encoder의 policy/Q head를 그대로 평가한다. 이후 clustering 모델은 반드시 이 기준선과 비교한다.

### 1단계: spherical k-means

고정된 `K`개의 normalized centroid를 만들고 다음 score를 사용한다.

```text
score_k(z) = dot(normalize(z), centroid_k) / temperature
responsibility = softmax(score)
```

같은 centroid로 두 방식을 모두 비교한다.

- hard: 가장 가까운 centroid 하나의 component Q 사용
- soft: top-4 responsibility로 component Q 혼합

이 단계가 가장 싼 `dot-product prototype mixture` 기준선이다. 첫 실험은 `K=256` 하나로 시작하고, 가능성이 확인된 뒤에만 K를 바꾼다.

### 2단계: diagonal GMM과 EM

같은 `K=256`과 같은 frozen latent를 사용하고 spherical k-means 결과로 초기화한다. component마다 평균, diagonal variance와 prior를 학습한다.

```text
score_k(z)
= log prior_k
  - 0.5 * mahalanobis_k^2
  - 0.5 * log determinant_k
```

이 단계는 cluster별 크기와 축별 분산을 허용했을 때 held-out EV regret가 실제로 감소하는지 확인한다. k-means와 GMM은 같은 component 수와 top-4 mixture 조건에서 비교한다.

### 3단계: value-aware dynamic split-EM

고정 K GMM보다 나은 가능성이 확인되면 component 수를 자동으로 늘린다. density만으로 split하지 않고 다음을 모두 확인한다.

- 충분한 child support
- held-out NLL 비악화
- held-out EV regret 감소
- action 순위 또는 policy 혼합의 개선

이것이 현재 프로젝트에서 말하는 첫 `custom clustering`이다. 새로운 분포를 발명하는 것이 아니라 검증된 GMM 위에 **전략적 분열 승인 규칙**만 추가한다.

### 추가 custom 모델의 도입 조건

- cluster 내부 Q 변화가 완만하지만 상수 prototype이 부족함: local linear Q expert 추가
- Gaussian이 bounded/asymmetric latent를 반복적으로 잘못 모델링함: Beta-box 비교
- Gaussian responsibility가 policy mixture에 부적합함: learned dot-product gate 비교
- component 수가 과도하게 증가함: merge 또는 encoder 수정

위 문제가 측정되기 전에는 Beta, full covariance, Transformer gate와 별도 MoE를 구현하지 않는다.

### 공통 평가 지표

각 단계는 동일한 train/validation shard와 frozen encoder에서 다음을 비교한다.

- held-out action-Q error
- search visit policy cross-entropy
- best-action agreement
- search-Q 기준 mixture regret
- component 수와 artifact 용량
- 한 상태의 top-4 추론 시간

NLL은 GMM 내부 선택 지표일 뿐 최종 우승 지표가 아니다. 최종 선택은 held-out EV regret와 계산 비용으로 한다.

### 추천 커밋 메시지

`docs: define progressive clustering baselines`

## Raw MLP, spherical k-means와 diagonal GMM 구현 (2026-07-16)

상태: `구현 및 smoke test 완료`

### 구현 파일

- `clustering_train.py`: 세 기준 모델의 공통 학습 CLI
- `test_clustering_train.py`: 작은 UCT SQLite를 이용한 end-to-end 검사

### 데이터 처리

`uct_nodes`에서 `search_version`, `opponent_policy`, `simulation_budget`이 같은 row만 읽는다. 현재 기본값은 `uct-v1`, `random`, `512`다. state JSON은 다음 compact 배열로 변환하며 MLP batch에서만 one-hot으로 확장한다.

```text
cards          uint8 [N, 12]
history        uint8 [N, 36, 3]
scalars        float32 [N, 7]
street/seat    uint8
legal mask     uint8
policy target  visits / sum(visits)
Q target       return_sum / visits / ante
```

hidden card 순서는 tensorizer에서 정렬하고 public card와 betting history 순서는 보존한다. train/validation 분할은 `state_json + seat`의 stable hash로 정하여 같은 정보상태가 실행마다 다른 split으로 이동하지 않게 한다.

### 학습 구조

Raw MLP는 667차원 one-hot/scalar feature를 `256 -> 128 -> 32` latent로 압축하고 policy와 action-Q head를 함께 학습한다. policy는 UCT visit distribution의 cross-entropy, Q는 방문 횟수의 제곱근으로 가중한 Huber loss를 사용한다.

Spherical k-means는 normalized 32차원 latent에 `MiniBatchKMeans`를 적용하고 normalized centroid와 dot product로 assignment한다. 동일 centroid에 대해 hard assignment와 temperature softmax top-4 mixture를 모두 평가한다.

Diagonal GMM은 train latent의 Ledoit-Wolf covariance로 shared whitening한 뒤 scikit-learn EM을 실행한다. 전체 responsibility를 영구 저장하지 않고 배치마다 top-4만 남겨 component Q/policy와 support를 집계한다.

### Artifact와 평가

```text
models/<run>/raw_mlp.pt
models/<run>/spherical_kmeans.npz
models/<run>/diagonal_gmm.npz
models/<run>/metrics.json
```

공통 validation 지표는 action-Q MAE, visit policy cross-entropy, Q-best agreement, visit agreement와 UCT target-Q regret다. NLL은 GMM 내부 진단값으로만 저장한다.

83만 row 전체 latent를 메모리에 둘 수 있도록 DB 원문을 거대한 float one-hot 배열로 펼치지 않는다. GMM fit은 기본 10만 표본으로 제한하지만 component 통계와 평가는 전체 train/validation row를 사용한다.

### 검증 결과

- 합성 64-row UCT DB에서 세 모델 학습과 네 artifact 저장 완료
- 실제 DB 5,000 row, CUDA MLP, 16 component, EM 2회 smoke test 완료
- `models/`를 `.gitignore`에 추가

smoke test의 수치는 학습 품질을 평가하기 위한 것이 아니다. 1 epoch와 2 EM iteration에서 실제 schema, CUDA 경로와 artifact 생성이 끝까지 동작하는지만 확인했다.

### 추천 커밋 메시지

`feat: add UCT clustering baselines`

## 첫 전체 clustering baseline 실측 (2026-07-16)

상태: `833,839 UCT root 학습 완료`

### 실행 조건

```text
rows                 833,839
train                750,175
validation            83,664
UCT simulations           512
latent dimensions          32
clusters                   256
GMM fit rows           100,000
top-k                        4
MLP epochs                   8
GMM maximum iterations      30
GMM actual iterations        8
total elapsed             69.37s
```

전체 UCT simulation을 다시 실행한 것이 아니다. DB에 이미 집계된 root별 visit와 return 통계를 읽어 지도학습했다. DB decode가 약 31초였고, RTX 3060에서 작은 MLP의 각 epoch는 약 1.2초였다. GMM은 전체 83만 row가 아니라 10만 개의 32차원 latent로 fit했으며 8번째 EM iteration에서 수렴했다. component 통계와 validation 평가는 전체 row에 배치 적용했다.

### 결과

| 모델 | Q MAE | Q-best 일치 | visit 일치 | search-Q regret |
|---|---:|---:|---:|---:|
| Raw MLP | 22.6964 | 38.82% | 37.37% | 16.9287 |
| Spherical k-means hard | 23.0229 | 50.33% | 47.62% | 10.2032 |
| Spherical k-means soft top-4 | 22.8146 | 50.55% | 47.82% | 10.1224 |
| Diagonal GMM top-4 | 22.4156 | 50.52% | 47.78% | 10.1077 |

### 해석

- Spherical k-means soft mixture는 raw MLP 대비 search-Q regret를 약 40.2% 줄이고 visit agreement를 약 10.46%p 높였다.
- Hard에서 soft top-4로 바꾸면서 regret가 약 0.79% 줄었다. 작은 차이지만 soft mixture 방향과 일치한다.
- Diagonal GMM은 spherical k-means soft보다 Q MAE를 약 1.75% 줄였지만 regret 개선은 약 0.15%뿐이며 action agreement는 사실상 같았다.
- 현재 결과는 `clustering/평균화가 noisy raw Q head보다 행동 순위를 안정화한다`는 증거다.
- 현재 결과만으로 component별 covariance가 전략적으로 필요하다고 말하기는 어렵다. GMM의 `lower_bound`는 GMM 내부 likelihood 지표이며 다른 모델과 직접 비교하지 않는다.
- 절대 regret `10.1 ante`의 품질은 target return과 action-gap 분포를 함께 보아야 판단할 수 있다. 또한 target 자체가 random opponent UCT 결과이므로 최적 정책 오차로 해석하지 않는다.

### 다음 판정

동일 설정의 seed를 두세 개 더 실행하여 k-means와 GMM의 작은 차이가 반복되는지 확인한다. GMM의 전략 지표가 계속 같은 수준이면 첫 online prototype은 더 단순하고 빠른 spherical k-means top-4를 선택한다. custom split이나 Beta density는 아직 도입하지 않는다.

### Cluster 구조 분석

`clustering_analyze.py`를 추가하여 artifact만으로 support quantile, entropy/Simpson effective component 수, 대표 Q/policy 행동과 최근접 component를 계산하고 component별 CSV를 저장한다.

첫 10만-row GMM fit artifact의 결과는 다음과 같다.

- k-means는 256개 모두 사용되며 entropy effective count는 `173.6`, Simpson count는 `136.6`이다.
- k-means support 중앙값은 `1,794.5`, 최소 `232`, 최대 `11,919`다.
- k-means centroid의 최근접 cosine 중앙값은 `0.9989`다. latent가 큰 공통 방향에 몰렸거나 작은 각도 차이만으로 cluster가 갈렸을 가능성이 있다.
- GMM은 256개 모두 support가 0보다 크지만 entropy effective count는 `51.5`, Simpson count는 `30.1`뿐이다.
- GMM component 132개는 평균 support의 10% 미만이고 62개는 1% 미만이다. 일부 component는 사실상 single-sample component다.
- GMM variance는 다수 축에서 regularization floor `1e-4`에 닿고 component anisotropy 중앙값은 약 `30`, p90은 약 `474`다.

따라서 현재 GMM은 256개의 균형 잡힌 전략 영역을 발견한 것이 아니라 소수 대형 component와 다수 극소 component로 붕괴한 형태다. 10만 표본 제한이 원인인지 확인하려면 validation 83,664개는 계속 분리하고 train 750,175개 전체에 fresh EM을 실행한다. 기존 collapsed component를 warm-start하기보다 full-train fresh fit이 더 깨끗한 비교다.

### CUDA diagonal EM

전체 train fit의 CPU와 responsibility 메모리 부담을 줄이기 위해 scikit-learn `GaussianMixture`를 batched Torch EM으로 교체했다. Diagonal Gaussian의 E-step은 whitening latent와 component precision 사이의 행렬곱으로 계산하고, M-step은 component mass, `r^T x`, `r^T x^2`만 누적한다. 전체 `N x K` responsibility는 저장하지 않으므로 CUDA에서는 기본적으로 최소 32,768-row batch만 VRAM에 올린다. GMM 초기 중심만 `MiniBatchKMeans`로 CPU에서 계산하며 artifact 형식은 유지한다.

실제 DB 5,000 row, 16 component, EM 3회 CUDA smoke test에서 `backend = torch-cuda`를 확인했으며 전체 MLP, k-means, GMM 파이프라인은 2.66초에 완료됐다.

### 실제 EV 게임 평가

저장된 cluster artifact를 실제 stackless heads-up EV 환경에서 실행하는 `ClusterPokerAgent`와 좌석 교대 평가기를 추가한다. 비교 전략은 spherical k-means와 diagonal GMM 각각에 대해 responsibility-weighted component policy를 그대로 표본화하는 `policy` 방식과 responsibility-weighted component Q의 greedy action을 사용하는 `q` 방식이다. H4 discard/reveal은 UCT 수집 조건과 같이 무작위로 유지한다. 결과는 random 상대에 대한 ante 단위 chip net과 paired 95% 신뢰구간이며 Nash exploitability가 아니다.

Full-train artifact를 분석하면 GMM entropy effective count는 `153.0`, Simpson count는 `97.9`로 10만-row artifact의 `51.5`, `30.1`보다 크게 증가했고 극소 support component도 사라졌다. 그러나 축별 variance 중앙값은 여전히 regularization floor `1e-4`이며 최대 anisotropy는 약 `1.26e6`이다. 데이터 확대는 component 붕괴를 줄였지만 전략적 분리까지 보장하지 않았다. Held-out regret도 k-means soft `10.1496`, GMM `10.2239`로 GMM이 더 나빴다.

2,000-hand 첫 평가에서 k-means-Q와 GMM-Q는 각각 1,784회와 1,780회 fold하여 약 `-0.78 ante/hand`를 기록했다. Cluster별 action Q를 평균한 뒤 greedy하게 선택하면 서로 다른 상태의 공격 행동 가치가 상쇄되고 incremental value가 0에 가까운 fold가 과도하게 선택된다. 현재 Q-mixture는 폐기한다.

Policy-mixture를 같은 seed의 10,000 hand로 평가한 결과는 다음과 같다.

| fit | strategy | ante/hand | paired 95% CI | avg top responsibility |
|---|---|---:|---:|---:|
| 100k | k-means policy | +1.98 | [-2.15, 6.11] | 0.252 |
| 100k | GMM policy | -0.22 | [-3.26, 2.82] | 0.714 |
| full train | k-means policy | +3.58 | [-1.85, 9.02] | 0.252 |
| full train | GMM policy | +2.46 | [-1.72, 6.64] | 0.866 |

모든 신뢰구간이 0을 포함하고 서로 크게 겹치므로 실제 수익 우열은 판정할 수 없다. Stackless EV의 드문 대형 팟이 평균 분산을 크게 만든다. 다만 k-means top-4 responsibility는 거의 균등한 `0.25`이고 평균 entropy도 `log(4)`에 가까워 cosine gate가 실질적인 local 선택을 하지 못한다. GMM gate는 훨씬 선택적이지만 현재 표본에서 수익 우위를 입증하지 못했다.

평가기에는 `random`, `heuristic`, `uct`, k-means/GMM policy와 Q를 임의로 지정하는 단일 대전 모드와 여러 참가자의 모든 조합을 실행하는 round-robin 리그 모드를 추가한다. 각 대진은 같은 deal seed를 좌석만 바꿔 실행하고 A 관점 ante/hand, paired standard error와 95% 신뢰구간을 출력한다. 리그 순위는 상대별 point estimate의 단순 평균이며 각 대진 신뢰구간을 대체하지 않는다.

Heuristic 상대 10,000-hand 실측은 다음과 같다.

| agent A | ante/hand | paired 95% CI | 처리량 |
|---|---:|---:|---:|
| k-means policy | -1.547 | [-1.680, -1.413] | 628 hands/s |
| GMM policy | -1.635 | [-1.829, -1.441] | 549 hands/s |

두 cluster policy 모두 현재 heuristic agent보다 명확히 약하다. 다만 cluster agent의 H4 discard/reveal은 UCT 수집 조건과 같이 무작위이고 heuristic은 수작업 규칙을 사용하므로, 이 결과는 betting abstraction만의 비교가 아니라 현재 전체 에이전트의 비교다. 네 참가자 `k-means policy`, `GMM policy`, `heuristic`, `random`의 100-hand-per-match 리그 smoke는 6대진 600 hand를 약 1초에 완료했다. UCT 없는 10,000-hand-per-match 리그는 선형적으로 약 1분 30초에서 2분 수준으로 예상한다.

같은 조건의 heuristic 대 random 10,000-hand 기준은 `+1.873 ante/hand`, paired 95% CI `[1.684, 2.062]`였다. 따라서 cluster policy의 heuristic 상대 패배는 작은 평가 표본의 우연이 아니다. 현재 원인 우선순위는 동일한 random-continuation UCT 데이터의 절대 개수 부족보다 `random 상대 search target`, UCT-vs-UCT root 분포, 무작위 H4, density 기반 cluster와 component 평균 policy를 결합한 체계적 mismatch에 있다. 추가 학습은 같은 objective의 epoch/EM을 늘리는 방식보다 현재 policy가 방문한 상태에서 opponent population을 대상으로 search target을 다시 만드는 iterative distillation이어야 한다.

## H4 action-EV signature와 adaptive conditional sampling (2026-07-16)

현재 UCT cluster 데이터는 discard/reveal 이후 4th, 5th, 6th와 7th betting root만 포함한다. Pre-H4의 hidden 4장 상태 자체는 encoder 입력이나 UCT root가 아니지만, post-H4 4th state에는 남은 hidden 2장, reveal card, discarded card와 새 4th public card가 모두 남아 있어 실제로 선택된 H4 action의 결과는 역으로 재구성할 수 있다. 따라서 현재 atlas를 H4 classifier로 직접 조회할 수는 없지만 후보 H4 action 이후의 continuation evaluator 또는 기존 데이터에서 H4 label을 만드는 source로 재사용할 수 있다.

H4는 4개 discard 후보와 나머지 3개 reveal 후보의 12-action contextual bandit으로 본다. 자기 initial four-card information state를 고정하고, 각 action에 대해 상대 initial hand, 상대 H4 action과 이후 chance card를 조건부로 다시 표본화한다. 같은 opponent/chance sample을 12개 action 모두에 재사용하는 common-random-number 방식으로 action 간 paired EV 차이의 분산을 줄인다. 후보별 terminal chip-net 또는 search continuation value의 `count`, `sum`, `sum_sq`를 저장한다.

현재 UCT도 하나의 betting root 안에서는 자기 관측 상태를 고정하며 512 simulation마다 상대 hidden과 미래 deck을 새로 표본화한다. 새로 필요한 것은 fixed 512 budget이 아니라 adaptive stopping이다. 연속 batch 평균의 단순 diff는 우연히 작아질 수 있으므로 다음 중 하나를 사용한다.

```text
모든 action의 confidence radius <= epsilon
또는
LCB(best action) > max UCB(other actions)
```

Stackless EV return은 bounded이지만 heavy-tailed하므로 정규근사만 믿기보다 empirical Bernstein bound 또는 paired bootstrap을 우선 검토한다. 최소 표본 수와 최대 표본 수도 둔다. 안정된 initial hand는 종료하고 confidence width가 큰 다음 hand 또는 reach probability와 pot 영향도가 큰 hand를 priority queue에서 선택한다.

각 initial hand의 최종 representation은 raw card geometry가 아니라 12-action value 또는 advantage signature로 둔다.

```text
Q_H4(h) = [Q(h,a_1), ..., Q(h,a_12)]
A_H4(h,a) = Q_H4(h,a) - baseline(h)
```

이 signature가 가까운 hand를 같은 cluster로 묶으면 같은 H4 전략을 가져야 한다는 목적을 직접 반영한다. Betting state도 같은 방식으로 legal-action advantage signature를 사용한다. 이는 현재의 latent-density GMM보다 value-aware clustering에 직접 대응하며, fold의 0 baseline만 보존되고 다른 action의 조건부 양수 EV가 평균으로 사라지는 문제를 줄일 수 있다.

첫 구현은 `h4_rollout.py`다. 한 initial four-card context에서 12개 행동 모두에 같은 opponent/future seed를 사용하고, 95% confidence radius가 `epsilon_ante` 이하가 될 때까지 batch를 추가한다. 기존 cluster policy를 continuation으로, heuristic을 상대 정책으로 사용한다. 결과는 기존 UCT SQLite의 별도 `h4_nodes` 테이블에 action별 `samples`, `return_sum_ante`, `return_sq_sum_ante`로 누적한다. 동일 terminal의 반복은 제거하지 않는다. 그것은 조건부 환경에서 해당 terminal이 발생하는 확률의 Monte Carlo 표본이기 때문이다.

초기 smoke에서 5개 context를 action당 최대 32회 평가했을 때 최대 95% CI 반경은 각각 `0.963`, `7.190`, `1.925`, `0.672`, `1.974 ante`였고 `epsilon=0.5`에 수렴한 context는 없었다. 100-hand heuristic 대전에서 기존 GMM policy는 `-1.216 ante/hand`, paired 95% CI `[-1.910, -0.522]`였고, action당 8~32회와 `epsilon=1.0`을 사용한 online H4 GMM은 `-1.361`, CI `[-1.725, -0.996]`였다. H4 100개 중 24개만 threshold에 수렴했다. 불안정한 12개 추정치 중 최대를 고르는 optimizer's curse 때문에 작은 online budget은 개선 신호가 아니며, H4는 먼저 offline에서 context별 누적 표본을 충분히 확보한 뒤 정책으로 distill해야 한다.

## Symmetric UCT, distance softmax와 raw MLP agent (2026-07-17)

기존 `uct_rollout.py`의 외부 실제 게임은 이미 UCT agent 두 명의 self-play이므로 저장 root의 방문분포는 UCT 대 UCT다. 그러나 각 `choose_action` 내부 simulation에서는 root player의 정보집합만 UCB로 선택하고 상대 행동은 random이었다. terminal chip-net은 simulation 중 방문한 root-player node 전체로 backpropagation되지만 SQLite에는 실제 의사결정 root 하나만 저장되고 내부 descendant node는 버려졌다.

`--opponent-policy uct`는 simulation의 양쪽 actor가 각자 관측하는 information set에서 UCB를 최대화하고 terminal에서 각 actor의 chip-net을 각 경로에 역전파한다. `--record-tree-nodes --record-min-visits N`을 사용하면 충분히 방문한 내부 node도 저장한다. 같은 고정 4th betting root를 20회 평가한 512-simulation benchmark는 internal random이 root당 `0.129초`, symmetric UCT가 `0.211초`로 약 `1.64배`였다. 10-hand collector smoke에서 random/root-only는 21 records, `0.128초/root`였고 symmetric UCT/min-8 descendant 저장은 77 records, `0.183초/root`였다. 새 `opponent_policy=uct` 데이터는 기존 clustering trainer로 바로 학습됐다.

거리 `d_k >= 0`가 작을수록 가까우면 responsibility는 `softmax(-d_k / T)`가 기본이다. squared Mahalanobis에는 `softmax(-d_k^2 / (2T))`, cosine/dot-product similarity처럼 클수록 가까우면 `softmax(s_k / T)`를 쓴다. 현재 k-means는 dot-product softmax, GMM은 log mixture likelihood softmax이므로 이미 이 방향이다. Softmax는 모든 component가 절대적으로 멀어도 합을 1로 만들기 때문에 `min distance` 또는 `max log likelihood`의 별도 OOD threshold가 필요하다.

`raw_mlp.pt`의 policy/Q head도 `raw-policy`, `raw-q` agent로 직접 연결했다. 1,000-hand heuristic 상대에서 raw policy는 `-1.306 ante/hand`, 95% CI `[-1.557, -1.055]`, raw Q는 `-2.441`, CI `[-2.977, -1.905]`였다. Raw MLP를 이전에 agent화하지 않은 것은 표현상의 이유가 아니라 cluster 비교 실행기에서 누락한 것이었다.

반복 개선은 `UCT self-play roots -> UCT labels -> MLP/clustering -> current policy가 만든 roots 재수집 -> UCT relabel`의 Expert Iteration으로 구성할 수 있다. 다만 현재 구현의 첫 단계는 cluster policy를 UCT prior/leaf로 넣지 않으므로, 새 symmetric UCT dataset을 독립적으로 만들고 성능을 비교한 뒤 prior 결합을 추가한다.

Descendant 저장은 무조건 켜면 안 된다. 동일한 10-hand, 512-simulation symmetric UCT smoke에서 `record_min_visits=1`은 18개의 실제 root로부터 16,716 records, 16,703 unique nodes와 약 3.38MB DB를 만들었다. `record_min_visits=8`은 77 records, 72 unique nodes와 24KB였다. 99%가 넘는 descendant가 방문 수 8 미만이므로 기본 운영값은 8로 둔다. 이는 용량뿐 아니라 단일 determinization에 가까운 noisy target을 제거하는 품질 필터다.

Betting root에도 `--min-simulations`, `--simulation-batch`, `--epsilon-ante` adaptive 수집을 추가했다. 한 root에서 자기 관측 카드와 공개 상태는 고정되고 상대 hidden hand와 미래 deck은 simulation마다 다시 표본화된다. 최소 예산까지 root action을 균등 표본화한 뒤 95% CI 반경이 가장 넓은 action에 표본을 추가하며, 모든 action 반경이 threshold 이하이면 다음 실제 root로 넘어간다. Descendant와 상대 actor의 선택은 계속 UCT다.

이 모드의 `uct-v1-ci95` root visit count는 정책이 아니라 action-Q 추정용 sample allocation이다. 따라서 visit-policy distillation에 섞지 않고 Q target으로 사용한다. 4th street 고정 root 하나의 symmetric UCT 실측에서 최대 4,096회, 최소 128회, batch 128 조건은 `epsilon=10`과 `5 ante`에서 256회와 약 0.13초에 종료했으며 최종 최대 반경은 `4.121 ante`였다. `epsilon=2`는 4,096회와 약 1.97초 후에도 최대 반경 `4.219 ante`로 수렴하지 않았다. Stackless EV의 큰 분산과 학습 중인 descendant UCT의 비정상성 때문에 현재 95% CI는 실용적 normal approximation이지 time-uniform confidence sequence가 아니다.

### Adaptive UCT 100-hand pilot

최대 4,096회, 최소 128회, batch 128, `epsilon=5 ante`, symmetric UCT와 descendant 최소 방문 8 조건으로 100 hand를 수집했다. 1,219개의 실제 decision root에서 4,240,000 simulation을 수행했고 root당 평균 3,478회, 2.366초가 들었다. threshold에 도달한 root는 312개로 수렴률은 25.6%였으며 최종 CI95 반경 평균은 `211.686 ante`였다. 평균은 소수의 극단적으로 불안정한 root에 크게 끌려 올라갔지만, 대부분의 계산이 최대 예산 근처까지 간다는 사실도 확인됐다.

DB에는 32,113 record write와 29,760 unique node, 총 6,624,725 stored simulation이 들어갔다. 실제 root는 1,219개뿐이므로 descendant가 cluster geometry를 지배한다. 최초 구현은 adaptive root와 일반 UCT descendant를 모두 `uct-v1-ci95`로 기록해 사후에 정확히 분리할 marker가 없었다. 이 pilot DB는 삭제하지 않고 frozen diagnostic/Q-only shard로 보존하며 기존 policy dataset에는 합치지 않는다. 이후 수집은 root를 `uct-v1-ci95`, descendant를 `uct-v1-ci95-tree`로 기록해 primary key와 loader filter에서 분리한다.

Adaptive root의 action visit은 CI 폭이 큰 action에 계산을 더 배정한 결과라 policy target이 아니다. Trainer에 `policy_loss_weight`를 추가했고 이 데이터는 `--policy-loss-weight 0`으로 Q-only 학습한다. Q-only artifact는 cluster agent의 `policy` decision으로 열 수 없도록 막는다. 기존 pilot 파일에 새 collector를 이어 쓰면 과거에 섞인 descendant와 새 root가 다시 합쳐질 수 있으므로 새 출력 shard를 사용한다.

첫 Q-only 학습에서 target의 절댓값은 중앙값 `54 ante`, p99 `27,953 ante`, 최대 `1,078,141 ante`였다. 이는 loader 단위 오류가 아니라 stackless EV에서 여러 street의 반복 raise로 팟이 폭증한 결과다. Current pot으로 나누면 중앙값 `0.574`, p99 `10.420`으로 줄고 각 상태의 action 순위는 보존된다. 따라서 trainer에 `q_normalization=pot`을 추가해 이 shard의 Q/value-aware geometry에 사용한다. 극단값은 여전히 남으므로 Smooth-L1을 유지한다.

Pot-normalized Q-only 학습은 29,760 row 전체와 128 component에서 약 4.7초가 걸렸다. Validation search-Q regret은 raw MLP `1.531`, spherical k-means hard `1.236`, soft `1.276`, diagonal GMM `1.316`이었다. K-means와 GMM의 128개 component는 모두 활성이고 entropy effective component는 각각 `111.5`, `101.9`라 빈 component 붕괴는 없었다. 그러나 k-means centroid의 최근접 cosine 중앙값은 `0.99847`이고 online top-4 responsibility는 거의 `0.25`씩, entropy는 `ln(4)`에 가까워 soft dot-product gate가 실질적으로 네 expert를 균등 혼합했다. GMM은 online top responsibility가 약 `0.92`였지만 실제 대전의 effective cluster usage가 약 5~9개로 줄어 train/online coverage mismatch를 보였다.

2,000-hand-per-match Q-only 리그에서 heuristic 상대 결과는 k-means Q `-3.064 ante/hand`, 95% CI `[-4.396, -1.732]`, raw Q `-5.078`, CI `[-6.478, -3.677]`, GMM Q `-5.228`, CI `[-7.367, -3.088]`였다. 같은 리그에서 heuristic은 random에 `+2.041`, CI `[1.607, 2.475]`였다. 이 결과는 H4가 여전히 random이라는 confound를 포함하지만 세 Q agent의 현 betting policy가 production candidate가 아님을 충분히 보여준다. 특히 component Q 대표 행동이 `BBING/HALF`에 치우치고 EV의 heavy tail 때문에 Q-agent끼리의 평균 손익과 신뢰구간은 극단적으로 커졌다.

따라서 이 pilot의 용도는 (1) adaptive root와 descendant 분리 필요성, (2) absolute-ante Q와 CI의 scale 문제, (3) soft cosine gate collapse, (4) online cluster coverage drift를 확인한 diagnostic benchmark다. 다음 수집은 새 shard에서 root/tree를 분리하고, root는 pot-normalized Q target으로, descendant는 별도 UCT policy/Q target으로 평가한다. 같은 형식의 데이터를 단순 증량하거나 현재 mixed artifact를 장기 self-play policy로 채택하지 않는다.

## Cluster time-tree와 시간층별 abstract transition model (2026-07-17)

온라인에서 사용할 cluster assignment는 현재 시점까지 관측 가능한 정보만 입력으로 사용하고 미래 정보는 masking한다. 시간층 `t`마다 cluster membership `r_t(i|s)`를 계산한 뒤 transition 표본 `(s_t, a_t, r_t, s_{t+1})`로 action-conditioned soft edge를 만든다.

```text
N_t(i,a,j) = sum_n r_t(i|s_n) * 1[a_n=a] * r_(t+1)(j|s'_n)
P_t(j|i,a) = N_t(i,a,j) / sum_k N_t(i,a,k)
```

가장 큰 `P_t(j|i,a)`만 남기는 hard connection은 deterministic approximation이지만 일반적으로 tree는 아니다. 여러 parent가 같은 child로 합쳐지고 action별 branch도 생기므로 시간방향 DAG다. Soft connection은 chance와 opponent uncertainty를 보존하는 action-labeled weighted DAG이며 포커에는 이쪽이 더 자연스럽다.

이 graph의 첫 용도는 graph 분석이나 GNN이 아니라 abstract transition model이다. Terminal value에서 역위상 순서로 다음 backup을 수행한다.

```text
Q_t(i,a) = R_t(i,a) + sum_j P_t(j|i,a) * V_(t+1)(j)
```

단일-agent에서는 `max_a`, 고정 상대에서는 상대 policy expectation, 불완전정보 self-play에서는 별도 equilibrium/regret 처리가 필요하다. GNN message passing은 이 backup을 학습하여 새로운 graph나 미관측 node로 일반화해야 할 때만 후보로 둔다. 현재처럼 수백 component의 유한 DAG라면 sparse transition matrix와 backward induction이 더 단순하고 검증 가능하다.

포커의 시간축은 `(street, betting_depth)`의 두 층으로 표현한다. Street은 macro time, 같은 street 안의 행동 수 또는 raise depth는 micro time이다. Action edge, opponent edge, chance-card edge와 street-advance edge를 구분하고 lexicographic order로 DAG를 유지한다.

첫 실험은 GNN 없이 다음만 수행한다.

1. `(street, betting_depth)`별 cluster를 만든다.
2. Soft action-conditioned transition count와 reward mean을 계산한다.
3. Terminal에서 backward value propagation을 수행한다.
4. Held-out UCT action-Q와 graph-derived Q를 비교한다.
5. 같은 cluster 내부의 reward/next-cluster 분포가 이질적이면 split한다.

현재 `uct_nodes`에는 node 통계만 있고 parent-action-child transition은 없다. 이 실험을 구현할 때에만 별도 transition 표본 또는 집계 table을 추가한다. Graph-derived Q가 기존 component Q를 개선하지 못하면 GNN은 도입하지 않는다.

### 추천 커밋 메시지

`docs: define temporal cluster transition model`

## Cluster-Q의 이론적 정리와 현재 한계 (2026-07-17)

UCT target을 학습한 encoder 위에서 spherical k-means 또는 diagonal GMM
responsibility를 계산하고, component별 action-Q를 혼합하는 구조를 수식으로
정리했다.

```text
observable history h
→ value-aware encoder z(h)
→ responsibility r_k(h)
→ component action value q_(k,a)
→ Q_hat(h,a) = sum_k r_k(h) q_(k,a)
```

이는 soft state aggregation이자 finite-basis linear Q approximator이며,
nearest-neighbor Q-learning과 mixture-of-local-constant-experts의 중간 형태다.
고정 encoder, 고정 cluster, 독립적이고 bounded한 rollout, 충분한 information
state라는 가정을 두면 component action value에 concentration bound를 붙이고
cluster 내부 abstraction error와 유한 horizon의 정책 손실을 합성할 수 있다.

Effective sample size는 responsibility 또는 importance weight가 서로 다른 표본의
실질 표본 수다.

```text
N_eff = (sum_i w_i)^2 / sum_i w_i^2
```

모든 weight가 같으면 실제 표본 수와 같고, 일부 표본에 weight가 몰리면 작아진다.
PAC 형태의 오차는 대략 `1 / sqrt(N_eff)`로 줄지만, 현재 시스템 전체에는 다음
가정이 성립하지 않는다.

- UCT label은 adaptive하고 서로 독립이 아니다.
- self-play 중 상대 정책과 방문분포가 바뀐다.
- encoder와 cluster를 같은 shard에서 선택하고 평가한다.
- H4는 random 또는 별도 휴리스틱이다.
- 같은 cluster 내부에서 action-Q가 가깝다는 보장이 없다.

따라서 `CLUSTER_AGENT_THEORY.md`의 bound는 조건부 분석이지 현재 agent의
end-to-end PAC 보장이 아니다. 이 정리를 통해 “Gaussian 안에 들어가면 같은
전략일 것”이라는 가정과 “그 가정을 검증할 수 있다”는 주장을 분리했다.

### Gaussian/Beta cover 아이디어의 위치

Gaussian ellipsoid, Mahalanobis box, bounded radial Beta 또는 product-Beta
component는 명시적 support와 OOD gate를 만들 수 있다. Pearson type II는
단위공 또는 타원체 안에서 밀도가 `f(x) ∝ (1 - ||x||^2)^beta` 꼴인 bounded
elliptical distribution이며, Gaussian의 무한 support 대신 유한 support를 준다.

이 계열로 얻는 것은 다음과 같다.

- 기존 atlas가 online state를 cover하는지 판정
- support 밖 상태에 추가 rollout 배정
- novelty buffer와 dynamic component 생성
- responsibility를 이용한 soft policy mixture

그러나 density cover는 value cover가 아니다. 현재 결론은 geometry를 먼저
복잡하게 만들기보다 action-Q, advantage, reward/transition residual로 split을
검증해야 한다는 것이다. Beta/Pearson atlas는 후보로 보존하되 production
경로에는 넣지 않았다.

## 1000-ante 상한과 데이터 재해석 (2026-07-17~18)

Stackless EV 환경에서 반복 raise가 허용되자 Q target의 최대값이 백만 ante를
넘고 adaptive CI가 사실상 수렴하지 않았다. 이는 단순한 학습률 문제가 아니라
게임 자체의 reward range가 지나치게 큰 문제였다.

실제 피망식 고정 스택에 가까운 `1000 ante` effective stack 상한을 기준으로
게임을 다시 정의했다. 이 변경은 다음 효과를 가진다.

- terminal utility와 empirical bound의 range를 유한하게 고정
- 올인 이후 추가 raise 제거
- 무한히 커지는 팟과 6-bet tail 차단
- 서로 다른 agent의 ante/hand 비교를 일관되게 유지

규칙이나 reward range가 바뀐 뒤에는 과거 UCT/MCCFR data를 같은 문제의
표본으로 볼 수 없다. 기존 stackless shard는 diagnostic으로만 보존하고 새
규칙의 학습에는 섞지 않는 것으로 결정했다.

## Kuhn CFR과 staged 7포커 MCCFR (2026-07-19~20)

가위바위보, Kuhn poker와 같은 작은 불완전정보 게임에서 Q-learning, UCT와
CFR의 차이를 확인했다. 고정 상대에 대한 Q/UCT는 best response를 배울 수
있지만, 양쪽 전략이 동시에 변하는 zero-sum game의 평균전략 수렴에는
counterfactual regret가 더 직접적인 기준이다.

`kuhn_cfr.py`에 작은 exact CFR 기준선을 만들고 정보집합별 regret matching과
평균전략을 검증했다. 이후 7포커에는 모든 street을 한꺼번에 넣지 않고 terminal에
가까운 street부터 넓히는 staged MCCFR을 적용했다.

```text
7th table
→ freeze 또는 warm-start
→ 6th+7th table
→ 5th+6th+7th table
```

초기 Python 7th 모델은 100,000 hands, 약 3.61M traversals에서 173,373개
bucket을 만들었다. 추가 20,000 hands 뒤에는 약 182,745개가 되었다. 마지막
출력의 `last_strategy`는 전체 평균 행동률이 아니라 마지막으로 조회된 한
information bucket의 평균전략이었다.

6th+7th 모델은 7th 모델을 동결된 continuation으로 연결해 학습했지만,
20,000-hand 대전에서 7th-only 모델에 다음과 같이 패했다.

```text
6th+7th vs 7th-only
ante/hand = -0.1123008
95% CI    = [-0.1565662, -0.0680354]
```

이 실패는 “앞 street을 배우면 성능이 반드시 보존된다”는 가정이 틀렸음을
보였다. 새 6th bucket의 sparse regret, 다른 도달분포, frozen continuation과의
coordination mismatch가 함께 작용했다. 더 많은 hands를 바로 투입하기보다
bucket reuse를 먼저 높여야 한다는 결론으로 이어졌다.

## Bottom-up bound와 고정 상대 POMDP best response (2026-07-21~23)

UCT가 root에서 아래로 내려가며 계산을 배분하는 것과 반대로, terminal에 가까운
action return부터 confidence interval을 계산하고 열등한 행동을 제거하는
bottom-up bounding을 toy high-card game에 구현했다.

`bottom_up_bound.py`는 root card를 고정하고 모든 첫 행동을 같은 opponent
card/chance sample에 평가한다. empirical Bernstein radius는 다음 형태다.

```text
r_n =
  sqrt(2 * sample_variance * log(3 / delta) / n)
  + 3 * reward_range * log(3 / delta) / n

LCB(a) = mean(a) - r_n(a)
UCB(a) = mean(a) + r_n(a)
```

`UCB(a) < max_b LCB(b)`인 행동을 제거할 수 있다. 다만 이 구현은 terminal
바로 위 root action의 인증 실험이며 전체 belief tree를 재귀적으로 인증한
알고리즘은 아니다.

고정 휴리스틱은 동시에 학습하는 adversary가 아니므로, 상대 hidden card를
숨은 환경 상태로 보고 exact POMDP best response를 계산했다.

```text
상대 카드 39개 branch
→ 관측된 상대 행동별 posterior partition
→ 내 차례에는 모든 합법 행동 max
→ 상대 차례에는 고정 heuristic expectation
→ terminal chip net
```

`pomdp_best_response.py`는 시작 스택 6에서 모든 rank와 양쪽 좌석을 약 10초에
풀었고 다음 결과를 냈다.

```text
fixed heuristic average EV = approximately 0
exact best-response EV      = +0.7153846 ante/hand
```

이 solver는 약한 카드로 체크한 뒤 상대의 작은 raise가 특정 range를 드러내면
큰 재레이즈로 fold를 유도하는 check-raise bluff도 스스로 발견했다. 중요한
교훈은 bluff를 reward shaping으로 넣지 않아도 belief update와 상대 반응을
정확히 모델링하면 best response 안에서 자연스럽게 나온다는 것이다.

이 경로는 고정 상대 착취에는 강하지만 self-play equilibrium을 보장하지 않는다.
따라서 이후 연구를 두 갈래로 분리했다.

```text
fixed opponent → POMDP/belief best response
adaptive opponent → CFR/MCCFR equilibrium learning
```

## Saddle-point와 다인 게임 검토 (2026-07-23~24)

bilinear zero-sum saddle-point에서 asymmetric perturbation을 사용하는 GDA를
`asymp_gda.py`로 재현했다. L2 perturbation을 KL 또는 entropy geometry로
바꾸면 simplex 전략에 더 자연스러운 mirror/prox update가 되지만, 논문의
수렴률을 그대로 유지하려면 해당 geometry의 strong convexity와 smoothness에
맞춘 별도 증명이 필요하다.

3인 Kuhn을 단순 trilinear tensor payoff로 확장하는
`three_player_kuhn_gda.py`도 실험했다. 다인 extensive-form game은 각
플레이어의 realization plan을 고정하면 다른 플레이어에 대해 multilinear지만,
2인 zero-sum의 convex-concave saddle point 구조는 일반적으로 사라진다.
따라서 2인 알고리즘을 tensor로 바꾼 것만으로 Nash 수렴은 따라오지 않는다.

PED, exploitability descent와 strong Nash도 검토했다.

- PED류는 현재 정책을 착취하는 방향으로 population을 확장할 수 있지만 모든
  일반합 다인 게임에서 exploitability가 0으로 수렴한다는 보장은 없다.
- 두 플레이어의 공동 deviation에도 견디는 strong Nash는 존재하지 않는 게임이
  많아 기본 목표로 두기 어렵다.
- 현재 7포커의 첫 기준은 2인 zero-sum chip EV와 Nash exploitability 또는
  고정 상대 head-to-head EV로 제한한다.

## Stud-Leduc 2계층 HRL 실험 (2026-07-24~27)

OpenSpiel의 Leduc exploitability를 기준으로 삼고, public community card 대신
각 플레이어가 서로 다른 public up-card를 받는 작은 Stud-Leduc을 만들었다.
전체 트리를 정확히 만들 수 있어 HRL 구조가 균형을 훼손하는지를 exact best
response로 측정할 수 있다.

독립 C++ solver `D:\Experiment\Toy-Card-Game-Agent\stud_leduc_cfr.cpp`에 다음
모드를 구현했다.

```text
flat
  기존 exact CFR+

latent
  라운드 전 비공개 option 3개 선택
  상위/하위 모두 terminal chip EV regret

semantic
  목표 투입률 tau 선택
  상위 regret = terminal chip EV
  하위 regret = 1 - |actual_commitment / max_commitment - tau|

adaptive
  0, 0.5, 1에서 시작
  많이 사용되는 구간의 midpoint goal 활성화
```

두 플레이어 모두 각 betting round 직전에 자기 option을 비공개로 선택한다.
자기 option과 과거 자기 option은 정보집합에 들어가고 상대 option은 제외되어
perfect recall과 비공개성을 유지한다. 모든 모드의 평가는 원래 chip payoff의
exact best response와 exploitability를 사용한다.

목표 집합을 `{0, 0.25, 0.5, 0.75, 1}`로 조밀하게 하거나 adaptive split하는
기능도 구현했다. 그러나 goal 수가 늘어난다는 사실만으로 exploitability가
단조 감소하지는 않는다.

```text
더 조밀한 goal
→ 표현 가능한 하위 정책 집합은 커질 수 있음
→ 정보집합과 학습해야 할 regret도 증가
→ 같은 node budget의 표본/순회 효율은 나빠질 수 있음
```

Semantic 하위 reward는 원래 zero-sum payoff와 다른 objective이므로 원 게임의
Nash 수렴 보장을 잃는다. 이 실험의 의미는 HRL이 항상 좋다는 주장이 아니라,
동일 training node visits에서 flat CFR+보다 exploitability AUC를 실제로
낮추는지 반증 가능하게 만든 것이다. 현재 HRL은 연구 branch로 보존하고
7포커 production 학습에는 아직 연결하지 않았다.

## 피망식 5구 규칙 v3 확정 (2026-07-27)

실제 플레이 관찰을 바탕으로 betting 규칙을 다시 정의했다.

```text
H4
→ 5th: player당 aggressive action 최대 1회
→ 6th: player당 최대 2회
→ 7th: player당 최대 3회
```

추가 규칙:

- 4th street에는 betting round가 없다.
- `BBING`은 아직 베팅이 없을 때의 첫 베팅이다.
- 그 뒤에는 직전 최고 bet의 2배인 `DDADANG`을 사용할 수 있다.
- 한 번 check한 플레이어는 같은 street에서 다시 raise할 수 없다.
- `BBING`, `DDADANG`, `QUARTER`, `HALF`, `FULL`은 aggressive action이다.

이 변경으로 betting history의 최대 길이가 유한해지고 4th round 전체가
사라졌다. 동시에 v1/v2의 UCT dataset과 MCCFR table은 v3와 호환되지 않게
되었다. 과거 데이터를 억지로 재사용하지 않고 규칙 버전을 model/data 계약의
일부로 둔다.

## C++ 7포커 환경과 MCCFR 분리 (2026-07-27)

Python의 deepcopy, JSON key 직렬화와 객체 dictionary overhead를 제거하기 위해
v3 heads-up EV 환경과 MCCFR agent를 독립 C++ 실험으로 옮겼다.

```text
D:\Experiment\Project-7-stud-Poker-Agent\cpp_mccfr\
  stud_mccfr.cpp
  power64_v1.bin
  *.bin checkpoints
  README.md
  IMPLEMENTATION_GUIDE.md
  BUCKET_GROWTH.md
```

환경과 solver는 현재 한 translation unit에 있지만 state transition 함수와
MCCFR class는 코드상 분리되어 있다. 같은 프로세스와 연속 자료구조를 사용하므로
Python-C++ IPC에서 생기는 serialization, process call, information loss가 없다.
학습 병목이 확인되기 전에는 별도 library/interface로 나누지 않는다.

기존 legacy bucket은 다음 항목의 곱으로 커졌다.

```text
hand category
x rank/public-card feature
x pot/stack band
x raise count
x exact relative history
x legal action mask
```

새 information set을 만나면 즉시 permanent entry를 만들고 merge나 eviction을
하지 않으므로 sparse history가 table을 폭발시켰다. C++은 lookup을 빠르게 할
뿐 coverage를 해결하지 않는다.

## Frozen power bucket과 sparse table 완화 (2026-07-27)

각 street에서 미래 완성 족보의 확률분포를 Monte Carlo로 계산하고
`sqrt(probability)`로 변환했다. 두 power vector의 dot product는
Bhattacharyya similarity에 해당한다. 이 vector와 expected tie-break rank,
opponent public pressure를 묶어 centroid를 미리 fitting한 뒤 MCCFR 중에는
고정한다.

```text
street
x frozen power centroid
x pot odds / stack-to-pot band
x own/opponent raise count
x checked flag
x recent public action/intent
x legal action mask
```

Centroid가 학습 중 움직이면 기존 regret entry의 의미가 바뀌므로 atlas는
반드시 먼저 만들고 freeze한다. 첫 identical-seed 2,000-hand 7th-street
비교는 다음과 같았다.

```text
legacy: 268,027 buckets, 61.3% single-touch, about -4.12 ante/hand
power:    2,604 buckets,  4.9% single-touch, about -0.50 ante/hand
```

이는 clustering 자체의 승리가 아니라, 전략적으로 관련된 hand-power
확률을 이용해 sparse information sets를 재사용한 결과다. garbage line을
임의로 삭제하는 대신 low-touch ratio, hit rate, bucket growth로 실제 희소성을
측정한다.

초기 pruning은 속도를 높였지만 EV를 낮췄다. 따라서 regret-based cold branch
pruning은 기본값을 off로 유지한다.

## pmang_v2 cluster policy의 최종 실패 기록 (2026-07-27)

새 규칙 데이터로 GMM policy agent를 다시 평가했지만 고정 휴리스틱에 명확히
패했다.

```text
gmm-policy vs heuristic, 10,000 hands
ante/hand = -4.9529389
95% CI    = [-5.8048074, -4.1010704]
average top responsibility = 0.8781
effective clusters used    = 82.44
```

Responsibility가 날카롭고 많은 component를 사용했다는 사실은 좋은 전략을
보장하지 않았다. 현재 cluster policy 실패의 주원인은 component 개수보다
off-policy UCT target, action-value averaging, H4 mismatch와 원 게임의
equilibrium objective 부재다.

따라서 다음 항목은 production 경로에서 중단했다.

- GMM/k-means component policy를 그대로 최종 agent로 사용
- cluster 평균 Q의 greedy action을 정답으로 간주
- 동일한 off-policy shard에 EM epoch만 추가
- geometric coverage를 decision confidence로 해석

Clustering은 폐기하지 않았지만 power abstraction, confidence/OOD gate,
search prior처럼 효과를 별도로 측정할 수 있는 보조 역할로 축소했다.

## 7포커 particle belief best response (2026-07-27)

Toy POMDP의 고정 상대 best-response 경로를 7포커에 옮긴
`agent/claude_belief_br.py`를 구현했다.

상대 hidden-card particle `h`를 공개 betting action과의 일치도로 갱신한다.

```text
b'(h) ∝ b(h) exp(
  -(strength(h) - observed_aggression)^2 / (2 sigma^2)
)
```

각 합법 행동은 posterior particle에 대한 net-chip EV로 평가한다. Raise는
상대가 fold하는 particle과 call하는 particle을 나누고, call range에
조건부인 equity를 사용한다. 1-ply rollout이 얇은 value bet을 과대평가하는
문제는 passive action보다 일정 margin 이상 좋을 때만 공격하는
`aggression_margin`으로 제한했다.

EV mode, `ante=1000`, fixed heuristic 상대의 실측:

```text
claude belief-BR average = +1.024 ante/hand
seed results             = +0.287, +0.991, +1.793
HA1 uniform belief       = -10.474 ± 3.973
random                   = -6.334 ± 2.252
```

이 결과는 action-conditioned belief가 고정 상대 착취에 실제로 유용함을
보인다. 그러나 `aggression_margin=0.40`은 myopic rollout의 보정 휴리스틱이고,
상대가 동시에 학습하면 opponent likelihood와 best-response target이 움직인다.
따라서 이것은 equilibrium agent가 아니라 고정 정책에 대한 planning oracle다.

## Textbook external-sampling MCCFR 검증 (2026-07-27)

7포커 결과를 해석하기 전에 작은 exact game에서 구현 자체를 검증했다.
Stud-Leduc C++ solver에 다음 표준 external-sampling MCCFR을 추가했다.

```text
chance node     → 한 outcome sample
opponent node   → current strategy에서 한 action sample
traverser node  → 모든 legal action 평가
terminal        → traverser utility 역전파
```

중요한 수정도 발견했다. 기존 C++ MCCFR은 average strategy를 regret를
갱신하는 traverser node에서 누적했다. Alternating external sampling에서는
다른 player가 traverser일 때 표본 경로에서 방문한 opponent node의 현재
strategy를 누적해야 한다. 수정 전 model의 `strategy_sum`은 수정 후 checkpoint와
같은 의미가 아니므로 이어 학습하거나 직접 비교하지 않는다.

### Rank 3 Stud-Leduc

```text
nodes            = 28,057
information sets = 738
node budget      = 10,000,000

Exact CFR+:
  exploitability = 0.005182
  elapsed        = 0.210 sec

External MCCFR:
  exploitability = 0.046758
  elapsed        = 2.502 sec
```

External MCCFR은 1M visits의 `0.1597`에서 10M의 `0.0468`로 지속적으로
exploitability를 낮췄다.

### Rank 4 mini-Stud

```text
nodes            = 128,529
information sets = 1,824
node budget      = 20,000,000

Exact CFR+:
  exploitability = 0.024831
  elapsed        = 0.430 sec

External MCCFR:
  exploitability = 0.044967
  elapsed        = 15.14 sec
```

전체 tree가 작고 contiguous array에 들어가는 게임에서는 full CFR+가 더
빠르다. MCCFR은 더 좋은 알고리즘이라서 쓰는 것이 아니라 full traversal을
할 수 없는 게임에서 unbiased sampled regret update를 얻기 위해 쓴다.

## 7포커 fixed-root MCCFR self-play (2026-07-27)

기존 `choose_action`마다 짧게 MCCFR을 돌리는 online 방식 외에, 5th street
경계 상태를 반복 표본화해 offline으로 regret와 average strategy를 누적하는
fixed-root 경로를 추가했다.

한 root iteration:

```text
52-card deck shuffle
→ 두 player H4를 기존 heuristic으로 선택
→ 5th public card까지 deal
→ 같은 root에서 P0 external-sampling traversal
→ 같은 root에서 P1 external-sampling traversal
→ 다음 deck/root
```

이는 H4 이후 5th~7th street의 power-abstracted game에 대한 self-play다.
상대 node는 현재 regret-matching strategy에서 sample되며, 두 player 모두
번갈아 regret를 갱신한다. H4까지 포함한 전체 7포커 self-play는 아니다.

61,000 root까지의 누적 training node visits는 `74,129,668`회였다. 모든
checkpoint를 같은 seed의 paired 20,000-hand match로 동결 평가했다.

| 누적 root | heuristic 상대 ante/hand | paired 95% CI |
|---:|---:|---:|
| 1,000 | -3.8048 | [-4.3369, -3.2726] |
| 11,000 | -0.9941 | [-1.3553, -0.6328] |
| 31,000 | -0.1620 | [-0.3737, 0.0497] |
| 61,000 | -0.0836 | [-0.3000, 0.1328] |

최종 model은 8,816개 bucket을 사용했고 평가 중 hit rate는 100%였다.
`belief-br`, 240 particles, 5,000-hand 스트레스 테스트도 다음과 같이
통계적 무승부였다.

```text
root_mccfr_61k vs belief-br
ante/hand = -0.1640
95% CI    = [-0.4456, 0.1176]
```

현재 결과는 “7포커 Nash equilibrium을 구했다”가 아니라 다음을 입증한다.

- 수정된 external-sampling regret와 average strategy가 toy에서 수렴한다.
- fixed-root self-play를 늘리면 실제 7포커 table의 상대 성능이 개선된다.
- 31k root부터 고정 휴리스틱과 통계적 무승부에 도달한다.
- 61k 이후의 작은 EV 차이는 현재 평가 noise보다 작다.

`root_mccfr_61k.bin`을 100k root까지 늘리는 것은 유효하다. 다만
`-0.084 → -0.076` 같은 선형 예측은 할 수 없다. 20,000-hand 평가의
standard error가 약 `0.110 ante/hand`라 그 차이는 측정할 수 없는 크기다.

## 현재 설계 판단과 보존할 기준선 (2026-07-27)

### 계속 사용하는 것

- exact terminal evaluator와 zero-sum chip-net 정산
- paired-seat evaluation과 95% confidence interval
- 규칙 version을 model/data 계약에 포함
- 작은 game의 exact exploitability oracle
- 고정된 power atlas와 bucket growth diagnostics
- 표준 external-sampling MCCFR
- 5th-street fixed-root self-play
- 고정 상대용 particle-belief best response

### 연구 branch로 보존하는 것

- Gaussian/Beta/Pearson bounded atlas
- value-aware split/merge와 dynamic bucket
- temporal cluster DAG와 abstract Bellman backup
- semantic/adaptive HRL goal
- bottom-up confidence-bound belief planning
- asymmetric perturbation과 다인 equilibrium 실험

### 현재 production 경로에서 사용하지 않는 것

- random H4를 가진 cluster policy
- descendant와 root가 섞인 adaptive UCT shard
- component 평균 Q의 단순 greedy policy
- latest-only clone self-play와 반복 distillation
- 규칙이 다른 과거 v1/v2 table 또는 rollout
- 수정 전 average-strategy MCCFR checkpoint

### 다음 한 번의 실험

새 구조를 더 추가하기 전에 다음 비교만 한다.

```text
같은 fixed-root training node visits
power64 vs 더 세밀한 frozen power atlas
→ 같은 frozen heuristic/belief-BR 평가
```

더 세밀한 atlas가 개선되지 않으면 현재 병목은 cluster 수가 아니라 H4
휴리스틱 또는 imperfect-recall aliasing이다. 그때 다음 구조적 확장은 H4
action을 CFR game tree에 넣거나 별도의 exact/strong continuation target으로
학습하는 것이다.

상세 재현 명령과 최신 수치는 `CFR_MCCFR_VALIDATION.md`에 둔다.

## Power bucket 진단 필드 수정 (2026-07-27)

Power model 출력에서 `buckets_by_hand_category`와
`buckets_by_history_length`가 모두 index 0에만 나타났다. 실제 전략이
street만 사용한 것이 아니라, 통계 코드가 legacy key의 `category`와
`history_length`를 읽은 반면 `make_power_key()`는 두 필드를 의도적으로
채우지 않아 기본값 0이 남은 진단 오류였다.

Power key의 실제 분화축은 다음과 같다.

```text
street
x power_cluster
x pot_odds / stack_pot
x own/opponent bet count
x checked
x last_action_class
x betting_goal
x legal_action_mask
```

Hand category는 별도 key가 아니라 power vector의 final hand-category
distribution 9차원과 expected primary rank에 포함된다. Exact history는 power
abstraction에서 의도적으로 제거하고 bet count, checked, recent action과 goal로
근사한다. 따라서 hand 정보는 보존되지만 exact perfect recall은 보장하지 않는다.

진단 출력은 power mode에서 의미 없는 legacy 배열을 `null`로 표시하고 실제
분화축을 출력하도록 수정했다. 기존 model key와 binary checkpoint format은
변경하지 않았다. `root_mccfr_100k.bin`의 새 출력은 다음을 보였다.

```text
power clusters active: 64 / 64 / 64 by street
own bet count buckets: [1647, 2523, 3210, 1547]
opponent bet count:    [382, 2525, 2777, 3243]
checked:               [8299, 628]
last action class:     [190, 192, 0, 3500, 5045, 0]
betting goal:          [1647, 3421, 3859]
legal action count:    [0, 0, 3454, 0, 0, 5473, 0, 0, 0]
street:                [762, 1838, 6327]
```

즉 기존의 all-zero category/history 출력은 학습 실패의 증거가 아니었다.
다만 exact history를 버린 power abstraction의 equilibrium error 가능성은
별도의 문제로 남는다.

### 추천 커밋 메시지

`docs: record clustering, belief BR, HRL, and MCCFR lineage`

## 100M snapshot의 top-p 및 적응형 cluster 실험 (2026-07-29)

`root_mccfr_current_snapshot.bin`을 동결하고 다음 순서로 실험했다.

```text
hard MCCFR table
→ Gaussian local bandwidth + top-p 0.99
→ local temperature calibration
→ 평균 surrogate regret가 threshold를 넘는 parent에서 centroid 1개 append
→ 기존 top-p 혼합전략으로 새 information-set node 초기화
→ local temperature 재조정
```

20,000 roots를 두 번 순회하는 데 `1,021.6초`가 걸렸다. 7th street의
cluster 6에서 새 cluster 64가 생성되었고 초기 전략은
`QUARTER 0.0206355, FOLD 0.979364`였다.

내부 surrogate loss는 감소했다.

```text
첫 calibration:       20.7274 → 20.1358
cluster 추가 이후:   18.6093 → 18.1469
```

하지만 동일 seed, 1,000-hand policy-LBR 평가는 다음과 같았다. 값은
LBR이 얻은 ante/hand이므로 낮을수록 방어 정책이 좋다.

| 정책 | LBR | 95% CI |
|---|---:|---:|
| Hard snapshot | 1.3970 | [0.7564, 2.0376] |
| Top-p 0.99, 기본 local temperature | 1.3675 | [0.4602, 2.2748] |
| 새 cluster + 보정 temperature | 1.7925 | [1.0162, 2.5688] |
| 새 cluster, temperature 파일 제외 | 1.3675 | [0.4602, 2.2748] |

마지막 두 비교에서 temperature 파일을 제외하자 결과가 top-p-only와
정확히 같았다. 이번 표본에서는 새 cluster 하나의 실질적 영향은 없었고,
악화는 local temperature calibration에서 발생했다. 현재 surrogate는
실제 LBR 목적과 정렬되지 않으므로 temperature를 surrogate만으로 채택하면
안 된다. 이후에는 후보 temperature를 별도 validation LBR 또는 여러 seed의
accept/reject gate로 검증해야 한다.

이 실험 직후 구현상의 원인도 확인했다. Power cluster 하나는 전역 정책
하나가 아니라 `power cluster × betting context`별 정책 묶음인데, 최초
구현은 발견된 context 하나만 초기화했다. 이후 구현은 새 centroid를 만들
때 해당 street의 기존 betting context 전체를 순회하고, 각 context마다
이웃 cluster의 전략을 responsibility로 혼합해 child node를 초기화하도록
수정했다. 100-root smoke test에서는 7th-street child cluster 하나에
103개 context node가 생성되었고 self-test를 통과했다. 위 20,000-root
adaptive 산출물은 수정 전 구현의 진단 결과이므로 최종 후보로 사용하지
않는다.

수정된 구현으로 5,000-root calibration을 성장 전후 한 번씩 수행했다.
소요 시간은 `252초`였다. 7th-street parent cluster 45에서 child 64를
추가하고 103개 betting-context node를 초기화했다. 발견 상태의 상속
전략은 다음과 같았다.

```text
DDADANG  0.150065
QUARTER  0.713256
HALF     0.041378
CALL     0.037878
FOLD     0.057423
```

동일 seed의 1,000-hand policy-LBR 결과:

| 정책 | LBR | 95% CI |
|---|---:|---:|
| Hard snapshot | 1.39700 | [0.7564, 2.0376] |
| Top-p 0.99 | 1.36750 | [0.4602, 2.2748] |
| Child cluster, temperature 미적용 | 1.36750 | [0.4602, 2.2748] |
| Child cluster + local temperature | 1.36125 | [0.4457, 2.2768] |

혼합전략 상속은 함수 보존 초기화이므로 temperature 조정 전 결과가
동일한 것이 정상이다. Local temperature 적용 후 명목상 `0.00625a/hand`
개선됐지만 표준오차 `0.46712`보다 매우 작아 효과를 주장할 수 없다.
현재 결론은 "파괴 없이 자유도 하나를 추가했다"까지이며, 개선 판정에는
더 큰 paired validation 또는 여러 seed가 필요하다.

Adaptive 평가에서 처음 관측된 `4 / 297,696` policy miss는 새 centroid가
top-p를 독점했지만 아직 child node가 없는 신규 문맥이었다. 이 경우 전체
기존 이웃으로 fallback하도록 수정했으며, 같은 평가에서 LBR 값은 유지되고
policy miss는 0이 되었다.

## 처음부터 학습하는 adaptive soft-cluster 비교 (2026-07-29)

기존 10M/100M hard table을 soft policy로 사후 변환하지 않고, 빈 regret
table에서 `mix`와 `simple` 두 정책을 같은 self-play 조건으로 학습했다.

공통 설정은 street별 초기 centroid 8개, `top-p=0.99`, local Gaussian
bandwidth, 1000-ante stack, plain MCCFR, 동일 deal seed이다. 한 상태의
cluster responsibility를 `r_k`라고 하면 local expert regret은 다음처럼
갱신한다.

```text
Delta R_k(a) = r_k * (Q(a) - V_k)
V_k = sum_a pi_k(a) Q(a)
pi_mix(a) = sum_k r_k pi_k(a)
```

10,000 roots마다 dominant cluster별 sampled one-step action gap의 평균을
계산한다. 최대 평균 gap이 `1 ante`를 넘으면 최악 사례의 power vector를
새 centroid로 추가한다.

- `mix`: 양의 `Q(a) - V_mix` 전체를 정규화한 혼합정책으로 초기화
- `simple`: 가장 큰 `Q(a) - V_mix` 행동 하나의 one-hot 정책으로 초기화
- threshold 이하: centroid 대신 local temperature의 이산 후보
  `{0.25, 0.5, 1, 2, 4}` 중 하나를 선택

온도 후보 계산은 성장 regret 표본 16개당 하나만 사용한다. 여기서 gap은
formal counterfactual regret/exploitability가 아니라 sampled opponent
trajectory에서 얻은 one-step surrogate이다.

### 결과

10k roots의 첫 적응에서 최대 평균 gap은 `79.5861 ante`였고, 두 방식
모두 첫 신규 7th-street expert가 `CALL 100%`여서 정책과 평가값이 같았다.

| 10k 평가 | mix | simple |
|---|---:|---:|
| heuristic, 20k hands | -1.8771 | -1.8771 |
| policy-LBR lower bound, 1k hands | 6.8230 | 6.8230 |

10k checkpoint에서 90k roots를 더 학습했다. 모든 구간의 최대 평균
gap이 `42~198 ante`여서 temperature 분기는 한 번도 실행되지 않았다.
따라서 아래 결과는 새 expert의 mix/simple 초기화만 비교한다.

| 누적 100k 평가 | mix | simple |
|---|---:|---:|
| heuristic, 100k hands | -0.70757 | -0.56738 |
| heuristic 95% CI | [-0.78736, -0.62778] | [-0.64458, -0.49018] |
| policy-LBR lower bound, 5k hands | 4.33505 | 3.97153 |
| policy-LBR 95% CI | [2.28530, 6.38480] | [2.31655, 5.62650] |

`simple`은 heuristic 상대로 약 `0.1402 ante/hand` 좋았다. LBR도 명목상
낮았지만 신뢰구간이 크게 겹쳐 방어력 우위를 주장할 수 없다. 두 모델
모두 heuristic에 지고 LBR에 크게 착취되므로 기존 hard MCCFR 결과를
대체하지 못한다.

신규 centroid 10개가 전부 7th street에 생긴 것도 중요한 실패 신호다.
global maximum gap 하나만 고르는 규칙은 street별 payoff 분산과 표본 수
차이에 민감해 5th/6th street 성장을 굶길 수 있다. 다음 실험은 새 모델을
더 얹기 전에 street별 gap 정규화 또는 street별 성장 quota를 비교한다.

## 동일 node-visit 예산의 hard/fixed/adaptive 비교 (2026-07-29)

앞선 비교는 같은 root 수를 사용했지만 soft traversal은 여러 expert를
동시에 갱신하므로 실제 연산량이 달랐다. 이를 바로잡기 위해
`--root-node-budget`을 추가하고, 다음 세 방식을 동일한 누적 게임 트리
node 방문 수에서 비교했다.

- `hard`: 가장 가까운 centroid 하나만 사용
- `fixed`: Gaussian responsibility로 정책을 혼합하되 atlas와 온도는 고정
- `adaptive`: `simple` 방식으로 centroid를 추가하고 local temperature를 조정

공통 조건은 1000-ante stack, street별 초기 centroid 8개, 빈 regret table,
plain MCCFR, 같은 seed이다. Soft 방식은 `top-p=0.99`, 기본 온도 1,
local bandwidth를 사용했다. Adaptive의 성장 threshold는 `1 ante`,
적응 주기는 10,000 roots이다.

### 10M node visits

| 방식 | 학습 시간 | heuristic EV (20k) | policy-LBR (1k) |
|---|---:|---:|---:|
| hard | 44.5s | +0.02752 | 1.14463 |
| fixed | 193.2s | -0.89458 | 5.65625 |
| adaptive | 216.1s | -0.91880 | 5.07350 |

이 시점의 adaptive는 마지막에 7th-street cluster 하나를 추가했을 뿐,
신규 expert를 후속 self-play로 충분히 학습하지 못했다. LBR 표본도
1,000 hands라 신뢰구간이 넓어 10M 수치는 방향 확인용이다.

### 누적 100M node visits

| 방식 | 추가 90M 시간 | heuristic EV (100k, 95% CI) | policy-LBR (5k, 95% CI) |
|---|---:|---:|---:|
| hard | 301.7s | +0.04476 [0.00128, 0.08824] | 2.50558 [1.74079, 3.27036] |
| fixed | 1745.5s | -0.76399 [-0.83864, -0.68935] | 4.88243 [3.58723, 6.17762] |
| adaptive | 2123.7s | -0.20693 [-0.27847, -0.13539] | 2.55863 [1.62129, 3.49596] |

Adaptive는 fixed-soft보다 heuristic EV를 `0.55706 ante/hand` 개선했고,
policy-LBR lower bound를 `2.32380 ante/hand` 낮췄다. 따라서 centroid
성장이 고정 soft atlas의 추상화 오차를 줄인다는 증거는 얻었다.

그러나 adaptive와 hard의 LBR 신뢰구간은 크게 겹친다. Adaptive가 hard보다
낮은 asymptotic exploitability floor를 갖는다는 증거는 아직 없다.
Heuristic EV에서는 hard가 유의하게 더 좋고, 학습 시간도 adaptive가 약
7배 길었다.

Adaptive atlas는 최종적으로 street별 `8/8/25` clusters가 됐다. 모든 신규
cluster가 7th street에 생겼고, 최대 sampled surrogate gap은 계속
`83~205 ante` 수준이어서 temperature 분기는 한 번도 실행되지 않았다.
이는 현재 global maximum 기준이 7th-street payoff scale에 편향되며,
surrogate가 안정적으로 감소하지 않는다는 뜻이다.

현재 결론은 다음과 같다.

1. Adaptive growth는 fixed-soft보다 확실히 유용하다.
2. 100M에서 adaptive의 방어력은 hard와 구분되지 않을 정도까지 따라왔다.
3. 더 낮은 수렴 바닥은 확인되지 않았고, 1B 학습을 정당화할 곡선도 없다.
4. 다음 실험은 street별 gap 정규화 또는 street별 성장 quota를 먼저
   적용하고, 같은 LBR seed와 예산으로 30M/100M/300M checkpoint를 비교한다.

Policy-LBR은 알려진 평가 정책에 대한 근사 best response가 얻은
착취 가능 이득의 lower bound이다. 이것은 원 게임 exploitability의 정확한
값이나 upper bound가 아니며, sampled one-step gap도 그 대용물이 아니다.

## Street-balanced temperature/growth 실험 (2026-07-29)

이전 adaptive는 모든 street의 sampled gap 중 전역 최댓값 하나를
선택했다. 7th street의 payoff 변동이 가장 커 신규 cluster가 모두 7th에
생겼고, 고정 threshold `1 ante`를 항상 초과해 temperature 분기는 한 번도
실행되지 않았다.

다음 최소 변경을 적용했다.

```text
적응 street: 5th -> 6th -> 7th -> 반복
threshold(j) = max(1, 200 * 0.9^j)

street의 최대 평균 gap > threshold
-> 최악 사례를 centroid로 새 simple-policy cluster 생성

그 외
-> 해당 street의 local temperature만 이산 후보로 조정
```

CFR regret update 자체는 계속 모든 soft expert의 과도한 행동 확률을
수정한다. Temperature는 기존 local policy들의 혼합 비율만 바꾸며,
현재 atlas로 설명하기 어려운 큰 action gap만 신규 cluster가 담당한다.

### 기존 adaptive의 학습 기울기 재측정

기존 hard/adaptive 10M과 100M을 동일한 5,000-hand policy-LBR,
32 particles, 같은 평가 seed로 다시 측정했다.

| 방식 | 10M LBR | 100M LBR | 명목 감소량 |
|---|---:|---:|---:|
| hard | 3.03108 | 2.40121 | 0.62986 |
| 기존 global adaptive | 4.68133 | 2.83803 | 1.84330 |

기존 adaptive의 명목 감소량은 hard의 약 2.9배였지만 각 checkpoint의
신뢰구간이 겹치므로 더 빠른 asymptotic rate의 증명은 아니다. 또한 학습
seed가 동일하지 않아 이 표는 장기 추세 신호로만 사용한다.

### 같은 seed의 10M/30M 비교

새 방식과 hard를 같은 학습 seed `15101`, 같은 초기 8-cluster atlas,
같은 node-visit 예산으로 각각 처음부터 학습했다. 평가는 100,000-hand
heuristic match와 5,000-hand policy-LBR에 같은 평가 seed를 사용했다.

| 방식 | 예산 | heuristic EV | policy-LBR |
|---|---:|---:|---:|
| hard | 10M | +0.13104 | 2.56137 |
| hard | 30M | +0.19546 | 1.94507 |
| balanced decay | 10M | -0.39203 | 4.43216 |
| balanced decay | 30M | +0.04661 | 4.21800 |

10M까지는 threshold가 높아 cluster를 만들지 않고 temperature만 조정했다.
30M에서는 threshold가 내려가며 7th cluster 세 개가 생성됐다. 5th/6th의
평균 gap은 `2~9 ante`였으므로 강제 분할하지 않고 temperature만 조정했다.

결과는 두 목표를 분리해서 봐야 한다.

- heuristic EV 개선량: hard `+0.06442`, balanced `+0.43864`
- LBR 감소량: hard `0.61631`, balanced `0.21416`

Balanced 방식은 특정 heuristic에 대한 활용 정책을 훨씬 빨리 찾았지만,
근사 best response에 대한 방어력은 hard보다 느리게 개선됐다. Soft
mixture의 유연성이 곧 equilibrium 수렴 가속을 뜻하지는 않는다.

따라서 이 설정의 100M 연장은 중단했다. 다음 실험에서는 하나의 surrogate로
temperature와 split을 모두 결정하지 않는다. Cluster 생성은
counterfactual/action gap을 사용하되, temperature 변경은 별도 validation
LBR에서 개선된 후보만 채택하는 acceptance gate가 필요하다.

## Temperature-free weighted cluster growth (2026-07-29)

Local temperature를 제거하고 cluster의 고정 state mass만 responsibility에
포함했다.

```text
r_k(z) proportional to
    m_k * exp(-distance_k(z) / (2 * local_variance_k))
```

초기 `POWERAT1` atlas는 cluster mass를 균등하게 초기화한다. 새 cluster는
선택된 parent mass의 10%를 받고 parent는 나머지 90%를 유지한다. 이 mass는
정책의 우수성이 아니라 state support prior이며 CFR 구간 중에는 바뀌지
않는다. 저장 형식은 mass를 포함하는 `POWERAT2`로 확장했고 기존 atlas는
계속 읽을 수 있다.

같은 seed와 초기 8-cluster atlas에서 centroid 생성법 두 개만 비교했다.

- `point`: interval의 최대 one-step regret 상태 한 개를 centroid로 사용
- `residual`: 선택된 parent에서 regret이 큰 상위 64개 상태를 보관하고
  `max(regret - threshold, 0)`으로 가중 평균한 centroid 사용

둘 다 신규 expert는 최대-regret 행동의 one-hot으로 초기화하고, 이후 정책은
CFR regret으로만 갱신했다. 적응 주기는 10,000 roots, threshold는
`1 ante`이며 local temperature 조정은 없다.

### 10k roots

두 방식 모두 같은 7th-street parent를 분할하고 같은 행동으로 초기화했다.
새 child가 후속 CFR update를 받기 전이라 평가 정책도 완전히 같았다.

| 방식 | heuristic EV (20k) | policy-LBR (1k) |
|---|---:|---:|
| point | -0.59921 | 4.06063 |
| residual | -0.59921 | 4.06063 |

### 누적 100k roots

10k checkpoint에서 각 방식으로 90k roots를 추가 학습했다. 평가는
100,000-hand heuristic match와 5,000-hand policy-LBR를 사용했다.

| 방식 | heuristic EV | policy-LBR | LBR 95% CI |
|---|---:|---:|---:|
| point | -0.37998 | 2.31000 | [1.15790, 3.46210] |
| residual | -0.40453 | 2.71855 | [1.91420, 3.52290] |

Point의 LBR이 명목상 `0.40855 ante/hand` 낮지만 신뢰구간이 겹치므로
통계적 우위를 주장할 수 없다. Residual 평균 centroid가 point보다 낫다는
증거는 없었다. Point는 구현과 계산이 더 단순하므로 두 방식 중에는 point를
기준선으로 남긴다.

두 run 모두 신규 cluster 10개가 전부 7th street에 생겼다. 이는 centroid
계산법이 아니라 global maximum gap 선택의 결과다. 또한 100k에서도 두
정책 모두 heuristic에 지므로 순정 hard MCCFR을 대체하지 못한다.

### Ante1000 hard 100k 기준선

Point/residual과 학습 deal 순서를 맞추기 위해 hard도 첫 10k roots는
seed `16101`, 이어지는 90k roots는 seed `16102`로 학습했다. 모든 모델은
`ante=1000`, effective stack `1,000,000 chips = 1000 ante`를 사용했다.
평가도 같은 seed `16301`로 수행했다.

| 방식 | 누적 node visits | 학습 시간 | heuristic EV | policy-LBR |
|---|---:|---:|---:|---:|
| hard | 76.28M | 319.5s | -0.02596 | 1.54909 |
| point mass | 55.89M | 1688.8s | -0.37998 | 2.31000 |
| residual mass | 54.86M | 1681.3s | -0.40453 | 2.71855 |

Heuristic 95% CI는 hard `[-0.07505, 0.02313]`, point
`[-0.45007, -0.30989]`, residual `[-0.47246, -0.33659]`였다. Hard만
heuristic과 통계적으로 구분되지 않는 동률이고 두 soft 방식은 명확히 졌다.

LBR 95% CI는 hard `[0.75571, 2.34246]`, point `[1.15790, 3.46210]`,
residual `[1.91420, 3.52290]`로 겹친다. 따라서 hard의 명목 LBR 우위는
관측됐지만 통계적 확정은 아니다. Hard 평가에는 20회의 policy miss가
있어 uniform fallback이 사용됐지만 전체 query 대비 극소수였다.

같은 root 수에서 hard는 정책 경로상 더 많은 tree node를 방문했음에도
벽시계는 soft보다 약 5.3배 빨랐고 bucket도 `1,143`개로 soft의 `2,198`개보다
적었다. 이 실험에서도 hard abstraction + CFR regret matching이 주력
기준선이다.
