# 7-stud sampled local refinement pilot

2026-10-03. 이번 실험은 **IRA-inspired MCCFR 휴리스틱 확장**이다.
정확 FPIRA, 원형 CFR+IRA, regret-bound 검사 또는 수렴 보장 구현이 아니다.

## 질문과 선행연구

- [Cermak et al.](https://arxiv.org/pdf/1803.05392), 특히 Algorithm10의 행동 선호 충돌 기반 보조 분할과 관련된다. 원형은 전체 원게임 트리를 순회하고 원정보집합별 순간 CF regret를 얻는다.
- [Lanctot et al.](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)의 outcome-sampling MCCFR를 기존 코드에서 재사용한다. 원형 IRA를 그대로 재현하는 새 방법이라고 주장하지 않는다.
- 검증 질문: 같은 frozen latent k-means와 signed MCCFR에서, 학습 궤적의 행동 선호 충돌로 부모 내부만 분리하면 고정형 대비 Local BR-gap이 줄어드는가?
- 비교군: raw k-means 고정형, latent k-means 고정형, latent 분할형. 서로 같은 초기화/학습 예산을 사용한다. 추가 분할 검토 비용은 별도로 포함해 보고한다.

## 이번에 고정한 조건

- Python seven-poker v3, heads-up cash1000chips/ante1, uniform H4 discard/reveal, 5/6/7구 베팅.
- `reconstruction_20261002/transformer/seed11/best.pt`의 전체 관측 복원 모델과 저장된 raw/latent512 군집. 새 모델이나 군집을 만들지 않는다.
- 세 방법 모두 regret/평균전략0에서 시작. 학습 seed111,100k hand, batch128, traverser epsilon0.6. 1hand는 교대 traverser의 단일 표본 궤적.
- 10k마다 분할 검토. 두 고정형도 같은10k 경계에서 batch를 끝낸다. 따라서 과거 실행과의 완전한 재현 비교는 첫10k까지만 가능하고, 이번의 새 고정형을 주 비교군으로 쓴다.
- signed regret matching을 유지한다. CFR+ clipping/linear averaging을 추가하지 않는다.
- 베팅 이력/카드 정보를 함께 담은 기존 latent를 계속 쓴다. **핸드만 abstraction하고 베팅 이력을 정확히 보존하는 실험은 다음 실험이며 이번에는 구현하지 않는다.**

## 검출과 분할

1. 기존 key `(seat, own-decision depth, centroid)`마다 최대512개 `(latent, sampled regret vector)`를 reservoir에 유지한다. Encoder/centroid는 고정한다. 관측하지 못한 카드 정답은 routing feature에 넣지 않는다.
2. 원정보집합 ID 하나의 예외 분리는 재방문이 희박해 일반화하지 못한다. 대신 부모 내부 latent 좌표 하나의 부등식으로 binary split을 제안한다. 원형 IRA와 달리 latent에서 이미 사라진 정보를 복원하지 못한다.
3. 최소256개 표본이 쌓인 부모만 검토한다. Hand ID의 별도 hash로 제안/확인 데이터를 나눠 같은 hand가 두 쪽에 들어가지 않게 한다.
4. 제안 쪽에서 각 latent 차원의25/50/75% 경계를 검색한다. 양쪽에서 최선인 행동이 다르고, 양쪽에 각각 최소24개 표본이 있는 경계 중 행동별 평균 regret의 분할 이득이 가장 큰 하나를 고른다. regret을 정규화하거나 clipping하지 않는다.
5. 별도 확인 쪽에서도 각 자식24개 이상이어야 한다. 두 행동의 상대적 선호 방향이 재현되고, `gain_mean - 2.58*SE > 0`이면 채택한다. 이 gate는 adaptive/heavy-tailed 자료의 유한표본 보장이 아닌 탐색적 잡음 필터다. 여러 후보/시점 검사의 전체 오류율을 보장하지 않는다.
6. 검토한 부모의 확인 데이터는 채택/거절과 무관하게 소비한다. 표본 부족 부모만 다음 구간까지 모은다. 한 검토당 최대4개, 전체32개 분할. 거절됐다고 실험 도중 역치를 낮추지 않는다.
7. 부모 안에 자식2개를 만들고, 자식 regret/평균전략/방문수는0으로 시작한다. 부모 통계는 버리고 다른 부모/seat/depth는 그대로 둔다. 자식도 다시 분할할 수 있다. 미관측 상태도 부등식으로 반드시 자식 하나에 배정된다.

분할 로그의 `validation_gain`은 순수익/100 기반 importance-weighted regret 표본의 차이다.
별도로 원정보집합 posterior rollout에서 계산하는 ante 단위 Local BR-gap과 같지 않다.

현재 latent 좌표 경계는 원정보집합의 정확 선호 그룹과 다르다. 이 실험은 imperfect recall을
해결했다고 가정하지 않으며, 분할과 재초기화의 합친 효과를 비교한다. 별도 reset-only
비교군 없이 어느 효과인지 완전히 분리할 수는 없다.

## 분할 기준의 수식과 구현상 주의

2026-10-03 사용자 요청에 따른 상세 보충이다. 기존100k 실험의 기준을 사후 변경한 것이 아니다.

**데이터와 label.** 부모마다 `x=latent observation`, `y=행동별 sampled counterfactual
regret vector`를 최대512개 저장한다. `y`는 reward 숫자 하나나 현재 누적 regret table이 아니라,
이번 outcome-sampling trajectory에서 importance correction을 거쳐 얻은 **증분**이다.
같은 합법 행동집합만 비교하며 부모의 `(seat, own-decision depth, legal partition)`을 유지한다.
현재 실제 구현은 사용자가 과거 제안한10% 방문의 추가 true-rollout probe 방식이 아니다.
기존 학습 표본을 이용한 충돌 검토이고, 별도 Local BR-gap은 평가 전용이다.

**후보 선택.** 부모의 reservoir에256개 이상이 모이면10k 경계에서 검토한다. 같은 hand의
모든 표본은 hash로 정한 같은 절반 A 또는 B에 넣는다. A는 경계를 고르는 자료, B는 확인
자료다. 각 latent 좌표의25/50/75백분위를 문턱으로 시험한다. 각 자식에24표본 이상이
필요하며, 행동 후보는 모두 합법이어야 한다. 다음을 계산한다.

```text
a_parent = argmax_a mean_{i in A} y_i(a)
a_left   = argmax_a mean_{i in A, x_i[d] <= t} y_i(a)
a_right  = argmax_a mean_{i in A, x_i[d] >  t} y_i(a)

gain_A(d,t) = mean_{i in A} [y_i(a_child(x_i)) - y_i(a_parent)]
```

`a_left != a_right`인 후보 중 `gain_A`가 최대인 하나만 B에서 확인한다. 즉 거리나 분산이
커서 분리하는 것이 아니라 **같은 정책을 강요하면 행동 선호가 충돌하는지**를 찾는 방식이다.

**독립 확인 gate.** B의 두 자식도 각각24개 이상이어야 한다. A에서 정한 두 행동을 바꾸지
않고, B에서 왼쪽은 왼쪽 행동을, 오른쪽은 오른쪽 행동을 선호하는지 확인한다.

```text
mean_{B_left}  [y(a_left)  - y(a_right)] > 0
mean_{B_right} [y(a_right) - y(a_left)]  > 0

delta_i = y_i(a_child(x_i)) - y_i(a_parent), i in B
SE      = sample_std(delta) / sqrt(len(B))
accept  = both_preference_checks AND mean(delta) - 2.58 * SE > 0
```

여기서 표본들이 완전히 독립·정규라고 증명한 것은 아니다. 같은 hand 안의 상관, 학습 중
정책 변화, 중요도 가중의 heavy tail, 반복 검토가 존재한다. 따라서2.58을 썼다고 통계적
99% 보장이나 원형 CFR+IRA의 regret bound 검사라고 해석하면 안 된다.

**실제 분할.** 채택하면 기존 부모에 `latent[d] <= threshold`인 왼쪽 자식과 나머지 오른쪽
자식을 붙인다. 기존 centroid는 이동하지 않고 다른 부모의 영역도 바꾸지 않는다. 같은
정보집합은 고정 encoder를 통과한 뒤 부모에서 자식으로 부등식에 따라 항상 routing된다.
자식도 나중에 부모가 될 수 있다. 전체32회/한 검토4회의 한도가 있다.

**중요한 원안과의 차이.** 현재 구현은 부모 regret/평균전략/방문 통계를 두 자식에게
보존·배분하지 않는다. 부모와 자식들을0으로 초기화한다. 사용자가 제안했던 "분할 시 기존
regret 보존"과 다르며, 그 원안을 검증했다고 말할 수 없다. 실제 분할이0회여서 이번 완료
실험에서는 이 초기화 분기 자체가 실행되지 않았다. 다음 hand-only 실험에도 이 reset이나
split을 섞지 않는다. 후속 split 실험을 하려면 보존 방식과 reset-only 대조군을 먼저 정해야 한다.

**완료 실험의 해석.** 후보311개 중11개는 확인 자식 표본 부족,300개는 gate 미통과였다.
68개는 선호 충돌 방향이 재현됐지만 `mean/SE` 최대1.927로2.58에 못 미쳤다. 따라서
fixed와 split 결과가 같은 이유는 **분할했는데 효과가 없어서가 아니라 채택된 분할이0개여서**다.

## 평가와 저장

- 학습10k/50k/100k의 current/average를 각각 저장하고 평가한다.
- 새 평가 seed861003, 고정32개 원정보집합,256 conditional particles. 학습/분할 확인 데이터와 평가 패널을 공유하지 않는다.
- 분할이 실제 발생한 경계의 직전/직후도 같은 첫8개 원정보집합,256particles로 측정한다. 이 작은 패널은 즉시 정책 변화 진단이며 전체32개 평균과 혼합하지 않는다.
- 기존 `reconstruction_br_gap`의 uniform-H4 posterior/원환경 rollout/mean-before-max/crossfit을 재사용한다. 5구 첫 결정만 측정하며 전체 BR/LBR/exploitability가 아니다.
- log-log gap와 raw baseline 대비 비율, signed crossfit, 실제 분할 수와 table/monitor 크기를 기록한다. 기울기로 미래 교차점을 예언하지 않는다.
- raw의 현재 관측 입력과 latent의 사건 이력 입력은 서로 다르다. Encoder/군집 학습은 과거 low-fold 궤적에 기반하며, 새 학습 궤적과 분포가 다르다는 제한이 남는다.
- `last.npz`만 reservoir/RNG를 포함한 중단 재개용 파일이다. 각 milestone 및 분할 전후 NPZ에는 모델과 경계/통계만 저장한다. 구간별 대형 replay 사본을 만들지 않는다.
- 이 NPZ는 정책 통계/경계 snapshot이다. 공유하는 frozen encoder와 centroid atlas는 별도 reference 디렉터리에 있으므로 함께 유지해야 한다. NPZ 크기만으로 전체 모델 저장 크기라고 부르지 않는다.
- 메모리 지표의 `allocated_table_bytes`는 세 통계 array의 할당 크기, `monitor_array_bytes`는 표본 배열 payload 크기다. Python 객체 overhead/encoder/worker RSS 전체와 같지 않다. 공통 용량을 예약하므로 논리적 leaf 증가와 할당량 증가가 다를 수 있다.

## 실행

프로젝트 root에서 다음을 실행한다. 새 경로를 사용하며 기존 모델/결과를 덮어쓰지 않는다.

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/refinement_20261003'
& $py -m unittest agents.state_action_embedding.test_refinement_cfr -v
& $py -m agents.state_action_embedding.refinement_cfr train --out-dir $run
& $py -m agents.state_action_embedding.refinement_cfr evaluate --out-dir $run
# 같은 조건의 중단 재개는 해당 명령 끝에 --resume 추가
python -m agents.state_action_embedding.refinement_report --run $run
```

분할 검사 없이 원본과 같은 sampler 결과가 나오는지, 독립 확인/부모 경계/재귀/초기화/
checkpoint 재개를 테스트한다. 테스트 통과는 연구 가설의 입증이 아니다.

## 완료 결과 (2026-10-03)

[전체 결과](data/refinement_20261003/RESULTS.md),
[Local BR-gap 로그 그래프](data/refinement_20261003/local_br_gap.png),
[분할/메모리 그래프](data/refinement_20261003/refinement_cost.png).

세 방법 모두100k hand를 완료했다.311개 후보 중11개는 확인 표본 부족,
300개는 확인 gate에서 거절되어 **실제 분할은0개**이다. 행동 선호 방향이 재현된
68개를 포함해, 확인 이득의 `mean/SE` 최고값은1.927로 사전 기준2.58에 못 미쳤다.
조건을 사후 변경하지 않았다. Fixed/split의10k/50k/100k regret/평균전략/방문수와
모든 paired 평가값이 정확히 같다. 이것은 채택된 분할의 효과를 검증한 결과가 아니다.

| Hands | Raw 평균전략 gap | Fixed latent 평균전략 gap | Split 평균전략 gap |
|---:|---:|---:|---:|
| 10k | 11.6178 | 9.2290 | 9.2290 |
| 50k | 16.3779 | 11.9462 | 11.9462 |
| 100k | 21.1744 | 12.9446 | 12.9446 |

32root/256particle, ante 단위이다. 평균전략 gap 감소는 나타나지 않았으며, raw 대비
차이를 분할의 성과로 해석할 수 없다. 표본 잡음/지나치게 보수적인 gate/latent 정보 손실/
단일 좌표 경계의 한계 중 무엇이 원인인지는 분리되지 않았다. Exact exploitability/LBR는 미측정이다.

모두 합해884,736개 평가 rollout,4,516,614개 정책 조회, 누락 partition0개였다.
평가 약967.5초. 학습+검토는 raw32.4초, fixed166.7초, split233.2초였다.
Split의 마지막 표본 배열 payload174.713MiB, 재개 파일150.935MiB,
추론용 정책-table/routing NPZ0.198MiB이다. 공통 encoder/atlas 크기와 전체 RSS는 제외한다.
모델/코드/체크포인트 hash를 확인했고, 최종21개 테스트가 통과했다.

## 더 촘촘한 평가 (2026-10-03 후속 요청)

사용자가 제시한 VQ10M 그림의 원본은10k마다128root/32particle로 current/average를
평가했다. 이를 따라 같은100k 학습의 저장 지점을10k 간격으로 늘린 별도 실행과,
매 hand 정보상태1개를 평가하고100hand 이동평균을 그리는 보조 실행을 추가했다.
**첫 실행은100k까지, 매-hand 실행은 비용 점검을 위한 첫1k만** 대상으로 한다.

- [10k 간격 고정 패널의 조건·명령](data/refinement_dense_20261003/README.md)
- [매-hand 평가/100hand 이동평균의 조건·명령](data/refinement_per_hand_20261003/README.md)
- [계측 wrapper](refinement_per_hand.py): 기존 trainer/evaluator를 재사용하며 batch128을 바꾸지 않는다.

후속 평가를 이유로 기존 분할 기준/학습법을 바꾸지 않았다. 원형 CFR+IRA로 명칭을
바꾸거나, 실제 분할0회를 분할 성능 비교로 해석하지 않는다. 기존32root/256particle
수치와 새128root/32particle 수치는 동일 조건의 시계열로 연결하지 않는다.

## 다음 실험 제안 (이번에는 미구현)

사용자 제안: 카드/핸드에만 abstraction을 적용하고, 베팅라인 및 이력은 정확히 저장한다.
검증할 가설은 정보 이력을 함께 압축하는 것이 현재 학습 병목인지이다. 이번 실험에는 섞지 않는다.

다음 key의 후보는 `(player, street, 정확한 행동/지불액 이력, 과거 card-bucket 이력, 현재 card bucket)`이다.
베팅 이력만 남기고 현재 hand bucket 하나로 과거 카드 관측을 대체하면 여전히 imperfect recall이
남을 수 있으므로 과거에 인식했던 bucket과 자신의 행동도 보존할 필요가 있다. 카드 부분에서
자기 비공개 패/공개 패/버린 카드/상대 공개 패를 어디까지 묶을지는 후속 설계에서 명시한다.
정확 상대 숨은 패를 넣는다는 뜻이 아니다. 이 방향은 전통적인 포커 정보 abstraction과
가까우므로 신규 기법으로 주장하지 않고, 현재 전체 latent 방식과 동일 예산에서 비교할 계획이다.

### 후속 준비 (2026-10-03)

위 미구현 제안과 별도로 [hand-only k-means 실험 실행기](../cpp_mccfr/HAND_RECALL.md)를
추가했다. 사용자가 직접 실행할1M 명령을 제공하며, 이번 요청에서는 학습하지 않았다.
같은 hand k-means/정확한 betting history에서 과거 card bucket 기억만 달리하는 대조군을
새로 학습한다. 기존 Python latent 실험과 C++의 H4/샘플링 방식이 달라 수치를 직접 이어
붙이지 않는다. 기존 split trainer의 분할 기준이나 초기화 방식은 바꾸지 않았다.
