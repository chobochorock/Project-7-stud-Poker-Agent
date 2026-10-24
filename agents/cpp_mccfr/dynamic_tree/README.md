# 단순 동적 Decision Tree MCCFR

7-stud C++ 환경의 실험용 온라인 abstraction이다. 매번 전체 트리를 다시
학습하지 않고 기존 leaf 내부만 `feature <= threshold`로 둘로 나눈다.
합법 행동집합이 다른 상태는 같은 leaf에 들어가지 않는다.
**CFR+IRA 원문 구현이나 수렴 보장이 있는 방법은 아니다.**

## 사용법

프로젝트 root에서 실행한다. 기존 실행과 다른 data 폴더를 사용한다.
기본값은 두 모델 각각100k hand이며 여기서 장기 실행을 자동 시작하지 않는다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
$run = "agents/cpp_mccfr/data/dynamic_tree/seed111"
python -m agents.cpp_mccfr.dynamic_tree.run build --check
python -m agents.cpp_mccfr.dynamic_tree.run train --run $run --hands 100000
python -m agents.cpp_mccfr.dynamic_tree.run plot --run $run --window 100

# 중단 후 총 목표까지 이어서 학습한다. 추가 핸드 수가 아니다.
python -m agents.cpp_mccfr.dynamic_tree.run train --run $run --hands 100000 --resume
```

`plot`은 학습 중에도 가능하다. 완성된 CSV행만 읽고 checkpoint를 건드리지
않는다. stdout/CSV는100hand마다 flush되므로 최신100hand 미만은 안 보일 수 있다.
반복 자동 실행은 하지 않는다. 모델/로그는 `agents/cpp_mccfr/data/dynamic_tree/`
아래, 코드/설명은 이 폴더, 실행파일은 기존 계열 `bin/`에 둔다.

`local_br_gap_log_zoom.png`는 로그축 확대본이다. 화면 하한 아래의 min/분위수는
잘릴 수 있음을 그림에 표시하며, 0을 포함한 평균/비율/원자료는 변경하지 않는다.
전체 범위는 `local_br_gap_log.png`, 0을 포함한 선형축은 `local_br_gap_linear.png`다.

분할만 끈 통제군은 새 run에서 `--max-splits 0`으로 실행한다. 이때 트리는
그룹별root만 유지하면서 regret을 학습하며 k-means 비교군도 같이 학습한다.
분할 hyperparameter를 바꿀 때는 새 run을 만든다. `--resume`은 코드/atlas/
seed/분할/평가 조건이 다르면 거절한다. 목표hand,저장 주기,용량 한도는
변경할 수 있다.

## 학습 및 분할

1. `(street, player seat, legal action mask)`마다 root 하나로 시작한다.
   따라서 전체 게임을 행동집합까지 무시한 단 하나의root에 넣지는 않는다.
2. 기존34D 관측 feature를 쓴다.18D 족보확률/공개패 요약에 pot,call 관련
   칩 상태,양쪽 잔여stack/베팅횟수/check 여부,공격성/이력길이/마지막 행동
   요약 등을 더한다. 상대 hidden cards/미래deck을 입력하지 않는다.
3. 각 leaf에 signed 누적regret와 평균전략 누적량을 저장한다. 두 플레이어의
   external-sampling traversal을 번갈아 수행한다. 상대 노드는 정책으로 하나
   선택하고 traverser의 합법 행동은 모두 평가한다. CFR+ clipping/linear
   weighting/CFR-D/LightGBM은 사용하지 않는다.
4. 같은 leaf가 한 traversal에서 반복되더라도 그 traversal 안에서는 최초
   조회 정책을 동결해 사용한다. 두 모델 모두 동일하게 적용한다.
5. 분할 label은 순간 sampled regret vector `r(a)=Q(a)-sum_b sigma(b)Q(b)`다.
   단위는ante. 커지는 누적합과 한 번의 순간 regret를 직접 비교하지 않는다.
6. **leaf당 한 hand에서 방문한 traverser 노드 중 하나를 균등 reservoir
   sampling**하여 최근256hand 표본만 남긴다. 별도monitor RNG를 쓴다.
   실제 regret 업데이트는 모든 방문을 사용하며 monitor만 제한한다.
   대형 traversal 하나가256개 표본을 모두 채우는 것을 막는다.
7. 새 표본256개마다 hand 종료 후 분할 후보를 검사한다. 모든feature의
   서로 다른 관측값 사이 경계를 검사해 행동별 regret SSE 감소가 가장 큰
   경계를 선택한다. 부모RMSE>=1ante,각 자식의 서로 다른hand 표본>=32,
   상대 SSE감소>=20%일 때 채택한다. 구체적 식은 아래와 같다.
8. 부모 regret/평균전략 누적량을 표본 좌우 비율로 나눠 자식에 준다.
   합계와 분할 직후의 각 자식 정책은 보존된다. 부모 통계를 다시 더하지
   않으며 자식 monitor는 비운다. 자식도 추후 다시 분할할 수 있다.

```text
mu_B[a] = mean(r_i[a] for i in B)
SSE(B) = sum_i sum_legal_a (r_i[a] - mu_B[a])^2
RMSE(B) = sqrt(SSE(B) / (sample_count * legal_action_count))
gain = (SSE(parent) - SSE(left) - SSE(right)) / SSE(parent)

alpha = n_left / n_parent
R_left = alpha * R_parent; R_right = (1-alpha) * R_parent
S_left = alpha * S_parent; S_right = (1-alpha) * S_parent
```

`S`는 평균전략의 누적량이다. 위 승계는 실제 과거 표본을 자식별로 정확하게
재배분한 것이 아니라 근사다. Monitor는 hand별 균등 표본이고 누적 regret는
방문별 합계여서 두 가중치가 같다고 가정하지 않는다. 분할에는 평가용gap을
사용하지 않으며 추가LBR 탐색도 없다.

조절할 기본값:

| 인자 | 기본값 | 의미 |
|---|---:|---|
| `--window` |256| leaf monitor 표본 수 및 새 표본 검사 간격 |
| `--min-child` |32| 자식의 최소 표본 수 |
| `--gain` |0.20| 분할 전후 상대SSE감소 역치 |
| `--min-rmse` |1.0| 부모 regret RMSE 최소값,ante |
| `--max-depth` |12| 그룹root 깊이0부터 최대깊이 |
| `--max-splits` |1024| 전 그룹 합계 성공 분할 수;0은분할금지 |
| `--max-kmeans-states` |500000| 비교군의 상태 안전한도;hand 경계 저장 후 중단 |
| `--reserve-gib` |10| 디스크 여유 하한 |

분할상한에 도달하면 tree의 추가 분할/monitor 수집을 멈추고 기존leaf 학습은
계속한다. 새 합법 그룹root는 별도로 만들 수 있으므로 leaf 수는 root 수+분할수다.
Root는 이 환경의1536개 미만 가능한group으로 제한된다. 비교군과feature
cache도 메모리를 사용하므로 전체RSS의 엄밀한 상한이나1M 완주를 보장하지 않는다.
`max-kmeans-states`에 걸리면 원인을 출력하고 checkpoint를 보존한다.

## 비교군과 평가

- 같은 seed111의 새 deal,동일 규칙/hand예산으로 signed MCCFR 두 모델을0부터
  학습한다. 한 hand당 각 모델 두traversal이다. 서로 전략이 달라 방문수와
  sampling 경로는 달라질 수 있다. `tree_nodes/kmeans_nodes`를 별도 기록한다.
- 비교군은 기존 frozen hard256 atlas와 기존 betting-summary key에 seat를
  명시적으로 추가한 새학습이다. 현재 돌아가는 exact-betting-history
  hand-recall 모델이나30M/100M checkpoint를 불러오는 것이 아니다.
- Tree는 연속34D 요약,비교군은hand cluster와고정 베팅요약bins를 쓴다.
  따라서 평가 조건은 맞지만 표현/초기 partition은 동일하지 않다.
  분할효과만 분리하려면 위의root고정 `--max-splits 0`도 비교해야 한다.
- Atlas는 기존모델 epsilon0.2 self-play100khand/seed37301에서 생성된
  데이터 편향을 유지한다. Tree의split label은 자신의 변화하는 정책에 의존한다.
- 기존 C++7-stud v3 HU/heuristic H4/5~7구/ante1000/stack1000ante 그대로다.
  앞선 low-fold 데이터수집 정책으로 조건을 바꾼 실험은 아니다.
- 매hand 업데이트/분할 후 독립5구 첫결정root1개,기본32conditional particles,
  eval seed307.10k마다 고정128root. 두군의current/average를 별도평가하며
  root/particles/rollout seed를 맞춘다. 숨은world별max가 아닌mean-before-max다.
- `gap=max_a mean(Q_a)-sum_a sigma(a)mean(Q_a)`,단위ante.
  Signed crossfit과평균 action SE도CSV에 남긴다.0/음수를 버리지 않는다.
  Finite-particle max 편향은MA100으로 없어지지 않는다. Crossfit은 정확gap의
  불편추정량이 아니고 action SE는gap 신뢰구간이 아니다.
- 그래프는 빨강tree/파랑k-means,MA100,min-max 옅은띠/5~95% 진한띠/평균선,
  current/average 분리,log-log와선형 및동일budget 비율이다. 띠는CI가 아니다.
  지표는Local BR-gap이며LBR 수익/exploitability와 구분한다.

## 출력과 복원

- `checkpoint.bin`: 두 모델regret/평균전략/RNG,트리 경계/monitor RNG와buffer,
  완료hand 및로그commit 위치. 임시파일 작성 후 원자적으로 교체한다.
- `per_hand.csv`, `panel.csv`: gap/crossfit/action SE/미등록key 조회/leaf 수/
  비교군 상태수/방문노드수/최고RSS. `splits.csv`: 모든검사와채택여부/분할feature/
  threshold/부모RMSE/상대gain.0split과0성능개선을 혼동하지 않는다.
- `config.json`, `source/`, `atlas_provenance.json`, `invocations.jsonl`,
  `console.log`: 조건과소스/실행파일hash/명령/진행보존.
- Ctrl+C는 현재hand 뒤 저장을 요청한다. 강제종료는 마지막checkpoint에서
  재개하며 그 이후의로그suffix만 되돌린다. 한run에writer하나만허용한다.
- 원래 학습파일/DB는 수정하지 않는다. Checkpoint는 같은코드/컴파일환경의
  자체생성파일을 대상으로 하며 임의외부binary를 받는 형식은 아니다.

## 선행연구와 한계

[RCFR, Waugh et al.2015](https://webdocs.cs.ualberta.ca/~mbowling/papers/15aaai-rcfr.pdf)의
regret 회귀/트리분할과 직접 관련된다. 원형처럼 모든학습자료로 회귀tree를
재학습하는 대신,여기서는 제한된 최근hand표본으로 기존leaf만분할한다.
[CFR+IRA](https://arxiv.org/pdf/1803.05392)의 전체순회/regret-bound 기반
분할 알고리즘을 재현하거나 그 보장을 계승하지 않는다.

이는 기존 아이디어의 단순 변형을 시험할 구현이지 신규성을 입증한 방법이 아니다.
훈련SSE가 줄었다고 전략적으로 좋아지는 것은 아니며 sampling잡음/정책변화에
과적합할 수 있다. Hold-out split 검증,통계적오류제어,merge,가지치기는 넣지 않았다.
기존34D 요약만 사용하므로 **perfect recall이 아니다**. Feature에 없는 차이는
아무리 분할해도 복원하지 못한다. 분할상한도 있으므로원게임NE수렴보장은없다.

## 검증

```powershell
python -m unittest agents.cpp_mccfr.dynamic_tree.test_dynamic_tree -v
```

합성분할/부호있는regret와평균전략승계/행동mask분리/비공개정보불변/
기존gap계산일치/24hand 연속 대12+12재개/미commit로그복구를 검사한다.
작은검증이지100k학습이나전략우위실험이아니다. Include한기존C++의
anonymous-namespace linkage 경고는 남아있다.
