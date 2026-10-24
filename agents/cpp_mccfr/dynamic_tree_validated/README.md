# DT 분할 개선 실험

기존 `dynamic_tree_unlimited`를 보존한 비교용 변형이다. C++ 7-stud HU/H4,
34D 관측 요약, signed alternating external-sampling MCCFR, traversal 안 정책
동결, SIMPLE 평균전략을 그대로 사용한다. 완전정보 CFR+IRA 재현이 아니다.

## 실행

새 모델을 로그 감쇠 분할 역치 `1/(1+log10(t))`로 0부터1M hand까지 실행하려면
[LOG_GAIN.md](LOG_GAIN.md)를 따른다. 기본10k 간격 평가/저장과 완료 후 비교 그래프를 포함한다.
기존 checkpoint를 재개하는 split schedule과는 별도 진입점이다.

프로젝트 root의 PowerShell에서 실행한다. `--hands`는 추가량이 아닌 총 목표다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_dynamic_tree -v
$run = "agents/cpp_mccfr/data/dynamic_tree_validated/validated_seed111_100k"
python -m agents.cpp_mccfr.dynamic_tree_validated.run train --run $run --hands 100000 --split-rule validated --gain 0.05
python -m agents.cpp_mccfr.dynamic_tree_validated.run plot --run $run --window 100
python -m agents.cpp_mccfr.dynamic_tree_validated.run lbr --run $run --output "$run/lbr" --pairs 250 --particles 64 --seeds 2026100701 2026100702
```

Ctrl+C는 현재 hand 종료 후 저장한다. 같은 조건에 `--resume`을 붙이면 이어간다.
원래 DT run은 복사해서 재개하지 않는다. 모델/RNG/monitor/원로그는 이 계열의
`data/dynamic_tree_validated/`에 저장한다. LBR는 고정 checkpoint를 읽기만 한다.
새 output만 받으며 같은 run의 writer lock으로 학습과 평가의 중복 실행을 막는다.

## 세 가지 비교 조건

- `--split-rule raw`: 기존 순간 regret SSE 분할. 원 구현과 짧은 실제 학습
  policy/metric의 일치를 검사하는 대조 조건이다.
- `--split-rule centered`: **분할 label만** `y[a] = r[a] - mean_legal(r)`로
  바꾼다. `r[a] = Q[a] - sigma dot Q`이므로 `y[a] = Q[a] - mean_legal(Q)`다.
  행동 간 차이를 보존하면서 모든 행동에 공통인 정책 baseline 이동을 제거한다.
  실제 누적 regret에는 원래 signed r를 넣는다. Regret-matching에 centered r를
  넣는 실험이 아니며 centered label만으로 RCFR의 보장을 주장하지 않는다.
  Regret-matching 자체는 상수 이동에 불변하지 않으므로, 이 변경이 언제나
  유익하다는 뜻도 아니다. 분할 표현의 우선순위를 바꾸는 ablation이다.
- `--split-rule validated` (기본): centered label의 최근 256개 서로 다른 hand를
  시간순으로 정렬한다. 앞 절반만으로 CART 후보 하나와 부모/좌우 평균 label을
  정하고, 뒤 절반에서 부모 대비 자식 예측 SSE 개선을 검사한다. 학습·검증 각각
  양쪽 자식에 최소 32개가 있어야 한다. 양쪽 상대 SSE 개선이 20% 이상이며
  검증 hand별 개선량 평균이 `--validation-z` (기본 2) x SE보다 커야 분할한다.
  검증 label을 보고 다른 후보로 재선택하지 않는다.

시간에 따라 정책이 변하고 반복 검사를 하므로 2SE 역치는 정확한 95% 유의성이나
sequential error-control 보장이 아니다. 기존 SSE 기준도 전략 개선을 보장하지 않는다.
검증 표본도 MCCFR 업데이트에는 쓰이며, 여기서의 holdout은 **분할 후보 선택에만**
쓰지 않았다는 뜻이다. 독립적인 원게임 테스트셋이라는 뜻은 아니다.

부모 regret와 평균전략은 기존대로 관측 좌우 비율로 분배한다. 합계를 보존하며
자식의 분할 직후 정책은 부모와 같다. 정확한 과거 history 재배분은 아니다.
한 leaf/hand에서 traverser 방문 하나를 reservoir sampling한다. 누적 regret는
모든 traverser 방문을 사용하므로 monitor와 업데이트의 가중치는 다르다.

분할 횟수/깊이는 기본 무제한이며 숨긴 cap이 없다. 선택 인자
`--freeze-after N`은 N hand 이후 분할만 멈추는 명시적 실험 조건이며 기본 0은
계속 분할이다. 메모리는 leaf당 최대 256개 monitor에 비례하고 OOM을 보장하지 않는다.
기존 `--max-kmeans-states`, `--reserve-gib`는 저장 후 전체 실행을 중단하는 안전장치다.

## 비교와 지표

모든 실행은 새 hard256 k-means + betting-summary + seat 비교군을 함께 학습한다.
같은 seed/hand/evaluation roots/particles/RNG를 사용하되 node visits와 계산 시간은
정책별로 다를 수 있다. Atlas는 과거 epsilon0.2 self-play100k/seed37301에서 생성된
것으로 그 편향을 유지한다. DT는 자신의 변화하는 정책에서 label을 모은다.
두 모델의 partition이 같다는 의미는 아니며 34D 요약은 perfect recall이 아니다.

매 hand 독립 5구 정보상태 1개에서 Local BR-gap을 측정하고 10k마다 고정 128개
panel을 기록한다. 기본 32particles, hidden 평균 후 max, current/average 분리,
crossfit/action SE/조회 미등록률을 기록한다. 빨강 DT/파랑 k-means, MA100 평균선,
min-max 및 중앙 90% 띠, 선형/log-log/동일 예산 비율을 유지한다. 띠는 CI가 아니다.
Finite-particle max 편향은 이동평균으로 사라지지 않는다.

LBR는 기존 `PolicyLBR`로 실제 terminal까지 대전하여 공격자 ante/hand를 기록한다.
기본 2 seeds x250 deal pairs x2 seats = 정책당 1,000hand, 64particles다.
Current/average와 matched k-means 및 uniform을 포함한다. 좌석 교환 pair를 단위로
seed별 bootstrap 95%CI를 계산한다. LBR 수익/Local BR-gap/exploitability는 다르다.

## 선행연구와 실험 질문

- [RCFR, Waugh et al. 2015](https://webdocs.cs.ualberta.ca/~mbowling/papers/15aaai-rcfr.pdf):
  regret 회귀와 직접 관련된다. 이 구현은 전체 자료 재회귀 대신 local split을 사용한다.
- [Athey & Imbens 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941430/): 분할 선택과
  검증/추정을 다른 표본으로 수행하는 발상과 관련된다. 여기서는 비정상적인 self-play
  표본을 사용하므로 causal tree의 독립표본 추론 보장을 가져오지 않는다.
- [CFR+IRA](https://arxiv.org/abs/1803.05392): 원문은 원게임 전체 순회를 요구한다.
  이번 sampled DT는 그 알고리즘이나 수렴 보장의 재현이 아니다.

질문은 공통 label 이동과 in-sample 분할 과적합을 줄이면 같은 예산에서 원 DT보다
나아지는가다. 새로운 방법의 최초 제안이라고 주장하지 않는다. 실제 결과는 별도
run/report에 남기며, 개선 실패도 보존한다. 1M 이상 실행은 사용자가 직접 시작한다.

## Frozen recall 진단

`audit_recall.cpp`는 학습하지 않는다. 저장된 tree에서 uniform/current/average
각 정책의 terminal까지 경로를 생성하고 같은 leaf의 경로 내 재방문을 센다.
Public history 길이를 root group에 추가했을 때 재방문이 사라지는지도 검사하지만,
그 partition으로 **학습하는 코드는 아니다**. 길이가 같은 서로 다른 과거와 카드
관측을 여전히 합칠 수 있으므로 perfect recall을 회복했다는 뜻은 아니다.
[Lanctot et al. 2012](https://mlanctot.info/files/papers/12icml-ir.pdf)의 imperfect-recall
결과도 같은 정보집합의 경로 내 재방문을 배제하며 추가 조건들을 요구한다.
이 진단만으로 해당 정리의 적용이나 성능 악화의 원인을 입증하지 않는다.

```powershell
g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc agents/cpp_mccfr/dynamic_tree_validated/audit_recall.cpp -o agents/cpp_mccfr/bin/dt_audit_recall.exe -lpsapi
$run = 'agents/cpp_mccfr/data/dynamic_tree_validated/confirm_raw_seed222_50k'
$identity = (Get-Content -Raw "$run/config.json" | ConvertFrom-Json).identity
& agents/cpp_mccfr/bin/dt_audit_recall.exe $run $identity 10000 2026100791
```

학습 중인 checkpoint에 실행하지 않는다. 결과의 uniform 경로 빈도는 MCCFR가
모든 traverser 행동을 펼치는 탐색에서의 빈도와 다르다. Source/atlas hash 검증은
먼저 `report` 또는 `lbr` 명령으로 수행한다. JSON은 새 파일에 보관한다.

## 비교 보고서

```powershell
$base = 'agents/cpp_mccfr/data/dynamic_tree_validated'
python -m agents.cpp_mccfr.dynamic_tree_validated.report --output "$base/my_review" --runs "raw=$base/confirm_raw_seed222_50k" "centered=$base/confirm_centered_seed222_50k" "validated=$base/confirm_validated_seed222_50k"
```

새 output만 허용한다. 각 run에 `lbr/summary.json`이 있으면 실제 LBR 및 동일
deal의 raw 대비 차이 CI도 포함한다. 학습 seed/evaluator/atlas와 k-means 통계가
일치해야 같은 그림에 넣는다. 서로 다른 budget/seed의 결과를 한 순위로 합치지 않는다.

## 2026-10-07 결과

Seed111/20k에서 6조건을 탐색하고, seed222/50k에서 원형과 두 후보를 확인했다.
모두 zero 초기화이며 기존 1M 모델의 연장이 아니다. [20k 대조 보고서](../data/dynamic_tree_validated/review_pilot_final_20261007/README.md),
[50k 확인 보고서](../data/dynamic_tree_validated/review_confirm_20261007/README.md)에
매 hand Local BR-gap, matched k-means, LBR CI와 자원 측정이 있다.

50k의 실제 LBR는 새 evaluation seed 두 개, 정책당1,000hand/64particles다.

| 조건 | LBR current | LBR average | 최고 RSS MiB |
|---|---:|---:|---:|
| 원형, gain20% | 2.3935 | 2.99525 | 68.95 |
| Centered, gain5% | 1.995812 | 2.130125 | 137.98 |
| Validated, gain5% | 2.649312 | 3.721124 | 81.60 |
| 동일 예산 k-means | 2.32375 | 3.433999 | 위 공동 실행에 포함 |

공격자 ante/hand이며 낮을수록 좋다. Centered 평균전략의 원형 대비 차이는
-0.865125, paired bootstrap95%CI[-2.11435,0.40410]다. **유망한 점추정이지만
확정 개선은 아니다.** Validated는 성장/메모리를 줄였으나 전략적 개선을 재현하지
못했다. 50k에서는 raw5%가 없으므로 centering 단독 효과를 분리하지 못하며,
20k의 같은5% 대조가 이 질문을 일부 보완한다. Training seed 불확실성과
LBR 근사 오차는 위 CI에 포함되지 않는다. 정확 exploitability는 측정하지 않았다.

Centering 후보의 마지막10k Local BR-gap은 current0.66720/average0.67655로,
원형0.29414/0.45073보다 오히려 높았다. LBR 점추정과 순위가 다르므로
Local BR-gap만으로 후보를 확정하지 않는다. 더 오래 학습하면 반드시 앞선다는
추세 외삽도 하지 않는다. 현재 코드가 최선이라고 판단할 근거도 없다.

## 사용자 실행용 1M 연장

아래 명령은 기존 50k action-contrast 실험을 총 1M까지 이어가는 **탐색용**이다.
50k에서의 성능만으로 1M 우위나 수렴을 보장하지 않는다. 작성 시점에 에이전트는
이 명령을 실행하지 않았다. 평가까지 마친 뒤 사용자가 직접 시작한다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
$base = 'agents/cpp_mccfr/data/dynamic_tree_validated'
$run = "$base/confirm_centered_seed222_50k"
python -m agents.cpp_mccfr.dynamic_tree_validated.run train --run $run --hands 1000000 --seed 222 --split-rule centered --gain 0.05 --resume
python -m agents.cpp_mccfr.dynamic_tree_validated.run lbr --run $run --output "$base/centered_seed222_1m_lbr" --pairs 250 --particles 64 --seeds 2026100721 2026100722
python -m agents.cpp_mccfr.dynamic_tree_validated.report --output "$base/centered_seed222_1m_review" --runs "centered=$run" --lbrs "centered=$base/centered_seed222_1m_lbr"
```

폴더명의 `50k`는 최초 목표를 나타낼 뿐이다. 연장 후 실제 hand 수는 checkpoint와
로그에 기록된다. 최초 50k의 `lbr/`는 과거 frozen 평가로 남으므로 연장 뒤의 보고서에서
이를 현 checkpoint의 평가로 사용하면 안 된다. 완료 예산과 일치하는 LBR만 읽도록
보고서에서 검사한다. 새 평가 output을 사용하고 과거 결과를 덮어쓰지 않는다.
대형 로그 그림에는 모든 CSV 문자열을 한꺼번에 읽는 `plot`보다 위의 streaming
`report`를 권장한다. 1M에서의 RAM 사용량은 아직 측정하지 않았다.

## 2026-10-08 후속: 같은 1M에서 분할 정지/지속

사용자가 위 1M 연장을 완료했다. [완료 검토](../data/dynamic_tree_validated/centered_seed222_1m_review/README.md)에
Local BR-gap/LBR/자원 결과가 있다. 최고 공동 RSS는2.62GiB였다. 위의 미측정 문장은10-07 작성 당시 기록이다.

총10M 비교는 [분할 스케줄 사용법](SCHEDULE.md)을 따른다. 새 `schedule` 진입점으로
동일1M 모델에서 정지/지속 분기를 만들며 기존 학습 코드와 원본 모델은 변경하지 않는다.
매10k 고정 panel 평가, 사용자 실행 명령, 느린 분할 파라미터, LBR/그래프 명령이 문서에 있다.

## 2026-10-08 후속: 분할 및 regret 상속 감사

고정5%와 로그 역치1M의 구현 차이, SSE 표본 수, 자식의 regret 상속을
[구현 감사 보고서](../data/dynamic_tree_validated/split_audit_20261008_verified/README.md)에 정리했다.
같은 역치/실제 window256에서 두 실행기를 각각2,048hand 돌려 전체 학습 상태가 일치함을
확인했다. 합성 검사에서는 최근 표본192:64에 따라 부모 regret을3:1로 나누고,
자식의 초기 정책은 보존하되 monitor만 초기화함을 확인했다.

재검사는 `python -m agents.cpp_mccfr.dynamic_tree_validated.audit_split --output agents/cpp_mccfr/data/dynamic_tree_validated/my_split_audit`.
기존 모델/학습 소스를 수정하는 명령이 아니며 새 출력 폴더만 허용한다.

## 로그 역치 10M 연장 및 사용자 인계

기존 로그 역치1M을 보존하고 별도 폴더에서1M씩 이어 학습하는 명령, 안전 중단 및 재개는
[LOG_GAIN_CHUNKS.md](LOG_GAIN_CHUNKS.md)를 따른다. 학습 수식은 그대로 유지하며,
가용 RAM이3GiB 미만이면 분할을 제한하지 않고 저장 후 정지한다.
