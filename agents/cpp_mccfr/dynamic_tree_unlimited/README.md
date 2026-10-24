# 무제한 분할 DT-MCCFR

기존 `dynamic_tree/`와 실행 중인 모델을 그대로 보존하기 위한 독립 실험 버전이다.
모델 코드와 runner를 이 폴더에 두고 출력은 `agents/cpp_mccfr/data/dynamic_tree_unlimited/`
아래에 둔다. 기존 공유 게임/feature/evaluator는 재사용하며 시각화도 기존 helper를 쓴다.
이 버전의 작성 때문에 기존 학습 소스/체크포인트/config를 수정하지 않았다.

## 1M 새 학습

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
$run = "agents/cpp_mccfr/data/dynamic_tree_unlimited/seed111_1m"
python -m agents.cpp_mccfr.dynamic_tree_unlimited.run train --run $run --hands 1000000 --max-splits -1 --max-depth -1
```

위 명령은 두 모델을 **0부터 총1Mhand씩** 학습한다. 기존100k를 이어붙이지 않는다.
기존 학습과 동시 실행할 수 있도록 코드/바이너리명/run 경로를 분리했지만 CPU/RAM은 공유한다.
준비 과정에서는1M 학습을 시작하지 않았다.

중단 후에는 동일 명령에 `--resume`을 붙인다. Ctrl+C 후 저장완료를 기다린다.
새 버전은 옛 `data/dynamic_tree/` run을 받지 않으며 구버전 checkpoint를 복사해 재개하지 않는다.
같은 run에 writer 둘을 실행하지 않는다.

중간 또는 완료 그래프:

```powershell
python -m agents.cpp_mccfr.dynamic_tree_unlimited.run plot --run $run --window 100
```

빨강DT/파랑k-means, current/average 분리, MA100/min-max/중앙90%띠/비율과
선형/log-log/로그확대본을 출력한다. 띠는신뢰구간이아니며0도평균에포함한다.

## 변경점과 남는 조건

- `--max-splits -1`: 분할 횟수 상한 없음. `0`은 분할 없는 통제군이고 양수는 명시적 상한이다.
- `--max-depth -1`: 깊이 상한 없음. `0`은 root만 유지하며 양수는 명시적 상한이다.
- 두 값 모두 기본값이 `-1`이다. 큰 양수로 상한을 숨긴 것이 아니라 제한 검사를 건너뛴다.
- 시작시 `CONFIG max_splits=unlimited max_depth=unlimited ...`를 출력한다.
- Split 조건은 그대로다: leaf별최근256개 서로다른hand,자식각32개이상,
  부모regret RMSE>=1ante/상대SSE감소>=20%. 조건을 못 맞추면 분할하지 않는다.
- 새로운 자식도 같은 조건으로 재분할하며 부모통계의 비율 승계 방식은 유지한다.
- 분할상한과 별개로 비교군 `--max-kmeans-states 500000`,디스크여유 `--reserve-gib 10`은
  저장후전체학습을중단하는운영안전장치로남겼다. 이때DT분할만몰래정지시키지않는다.
- DT 전체 RAM 상한은 없다. Leaf 수에 따라 monitor와모델이계속증가할수있고 OOM/완주를
  보장하지않는다. 1,000hand마다checkpoint를저장한다. 정수index의기계적표현한도는 남는다.

## 유지하는 비교와 해석

기존과 같은 C++7-studv3 HU/heuristicH4/5~7구,seed111/eval307,
signed alternating external-sampling MCCFR/SIMPLE평균전략을사용한다.
두모델은zero regrets/평균전략부터시작한다. 새 hard256 betting-summary+seat k-means를
동일hand예산으로학습하며 방문node수는별도기록한다.
Tree34D관측요약과baseline partition은서로다르며정확history/perfect recall을보장하지않는다.

매hand 업데이트 후 독립5구root1개/32particles로current/average Local BR-gap을측정하고
10k마다동일128root고정panel을평가한다. Crossfit/actionSE/미등록key조회도그대로남긴다.
Local BR-gap은LBR수익/exploitability가아니고32particles의max표본편향도남는다.

Atlas의이전epsilon0.2 self-play100k/seed37301 데이터편향과DT자신의변화하는정책에서
수집한표본편향은바꾸지않는다. 분할monitor 간격도그대로다.
기존실행의1,024분할/깊이12와무제한성장을비교하는조건변경이지새알고리즘이아니다.
관련선행연구는[RCFR](https://webdocs.cs.ualberta.ca/~mbowling/papers/15aaai-rcfr.pdf)이며,
[CFR+IRA](https://arxiv.org/pdf/1803.05392)의full-tree/regret-bound 기반분할재현이나
수렴보장으로제시하지않는다. 자세한기존분할식은[기존설계](../dynamic_tree/README.md)를참고한다.

## 검증

```powershell
python -m unittest agents.cpp_mccfr.dynamic_tree_unlimited.test_dynamic_tree -v
```

합성표본으로1,025회분할/깊이1,025에실제로도달하고그트리의저장/복원을검사한다.
24hand연속대12+12재개(완료로그/split log/통계와RNG),숨은정보불변,기존gap연산일치,
기본무제한값/음수오입력거절/실제CLI실행을검사한다. 짧은검증이지1M성능실험은아니다.
