# 로그 감쇠 분할: 새 모델 0 → 1M hand

## 한 번에 실행

프로젝트 root의 PowerShell에서 실행한다. 자동 C++ 빌드, 새 모델 초기화,
1M hand 학습, 최종 비교 그래프 생성을 한 명령으로 수행한다.
10k마다 평가/저장하지만 그 지점에서 학습을 멈추지는 않는다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain train --hands 1000000
```

기본 출력: `agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_1m/`.
기존 fixed-gain 1M이나 freeze/continue checkpoint를 이어받지 않는다.
같은 경로가 이미 비어 있지 않으면 덮어쓰지 않고 실패한다.

중단 후에는 **같은 명령에 `--resume`**을 추가한다. `--hands`는 추가량이 아닌 총량이다.
기본값 이외의 학습 설정을 사용했다면 재개 시 같은 설정을 유지해야 한다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain train --hands 1000000 --resume
```

Ctrl+C는 현재 hand, 평가, checkpoint 저장 후 종료한다. 저장 완료까지 기다릴 것.
강제 종료하면 마지막 checkpoint 이후 최대10k 미만 hand를 잃을 수 있다.
다시 실행하면 저장 시점 이후의 미커밋 CSV suffix만 복구한다.

## 정확한 변경

- 역치 `gain(t) = 1 / (1 + log10(t))`, t는 **전체 누적 학습 hand 수**, t>=1.
  0hand 초기 평가에는1을 표시한다. 재개/새 leaf 생성 때 t를 초기화하지 않는다.
- 1k:25%,10k:20%,100k:16.67%,1M:14.29%. 실제 검사 hand의 역치를 사용한다.
- 현재 leaf의 최근256개 monitor 표본으로 후보를 평가한다. leaf에서256개 monitored hand가
  추가될 때마다 검사하며, 자식 최소32개, 부모 centered-label RMSE 최소1ante를 유지한다.
- 이전과 같은 `centered` SSE 분할: 기준을 넘는 가장 좋은 feature/threshold를 채택한다.
  자식별 사전 CFR 학습은 없다. 부모 regret/평균전략 누적량은 표본 수 비율대로 상속한다.
- signed alternating external-sampling MCCFR, traversal 내 frozen policy,
  SIMPLE 평균전략 업데이트를 그대로 사용한다. CFR+, CFR-D, Hoeffding Tree가 아니다.
- 분할 수/깊이 상한, 분할 정지 시점은 **없다**. 전역 t가 커져도 최근 평가 표본 수는256개로
  고정이므로 통계적 신뢰도 또는 수렴 보장을 의미하지 않는다. Imperfect recall도 그대로다.
- 기존 `solver.cpp`, `run.py`, `schedule.cpp/.py`는 수정하지 않는다. 새 진입점만 추가하여
  과거 checkpoint의 source hash를 보존한다. 확장 소스 hash와 snapshot을 별도로 검증한다.

## 실험 조건과 비교군

- C++7-stud HU/H4/5~7구, ante1000, stack1000ante. 학습seed222, 평가seed307.
- 양쪽 모두 zero regret/zero 평균전략 누적량으로 시작한다.
- 1hand는 새로운5구 root에서 각 플레이어1회씩 총2회 traversal, 모델별로 적용한다.
  동일 hand-index deal로 DT와 기존 hard256 k-means+betting summary+seat를 함께 학습한다.
- DT는 자기 정책에서 얻은 label로 분할한다. K-means atlas는 기존epsilon0.2 self-play
  100k/seed37301 데이터로 만들어진 고정본이다. 이 초기 데이터 정책 차이는 제거하지 않았다.
- fixed0.05/0.1 및 종료 역치와 비슷한 fixed0.15가 스케줄 효과 확인용 비교 대상이다.
  현재 one-shot은 로그 스케줄과 k-means만 학습한다. 선택적 고정역치 통제군 명령:

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain train --run agents/cpp_mccfr/data/dynamic_tree_validated/fixed015_seed222_1m --hands 1000000 --fixed-gain 0.15
```

이는 추가 실행이며 자동으로 시작하지 않는다. 한 seed만으로 우위를 확정할 수 없다.

## 평가와 출력

- 매10k 및 시작/최종/정상중단 때 동일128개5구 root, 각32particles로 평가한다.
- **매 hand Local BR-gap 평가가 아니다.** 학습마다256개 monitor를 수집하는 것과도 별개다.
- hidden particle별 행동 가치를 먼저 평균하고 max를 취한 one-step gap:
  `max_a mean_hidden(Q[a]) - sum_a sigma[a] * mean_hidden(Q[a])`, 단위ante.
  Current/average DT 및 matched k-means 네 정책을 모두 평가한다.
  유한 particle max 편향이 남으며, crossfit/action SE/미등록 query 원자료도 저장한다.
  Local BR-gap은 exploitability나 실제 LBR 수익이 아니다.
- `panel.csv`: fixed-root 지표, 자원 통계, `gain_threshold`.
- `splits.csv`: 검사 결과, 실제 SSE 감소율과 그때의 `gain_threshold`.
- `per_hand.csv`: 기존 checkpoint 형식 호환용 **header만 존재**, per-hand 지표 없음.
- `console.log`: `LOCAL_BR_GAP hands=... tree_current=... kmeans_current=...` 및
  `PROGRESS hands=... target=... gain_threshold=... tree_leaves=... splits=... checks=...`.
  `tree_nodes/kmeans_nodes`는 방문 노드 누적량, `peak_rss_mib`는 해당 프로세스 최고working set,
  `wall_seconds`는 이번 실행 경과초다. 재개하면 프로세스 시간/peak memory는 다시 시작한다.
- `checkpoint.bin`: 두 모델의 regret/평균전략/RNG 및 DT monitor. 최신본으로 원자적 교체한다.
  매10k 별도 모델 파일을 쌓지 않는다. source/config/atlas/invocations로 재현 조건을 기록한다.
- 정상 완료 시 `review_1000000/`에 log-log/선형 비교, DT/KM 비율, 분할/자원 그래프,
  `summary.json`을 생성한다. 빨강DT/파랑k-means, 옅은min-max/진한중앙90% 박스 띠와 평균선.
  이 띠는128root의 분포이며 신뢰구간 또는 매hand MA100가 아니다. log 그림은0hand를 제외하고
  표시만0.001로 하한 처리한다. 평균/비율 원자료는0을 유지한다.

중단·저장된 모델을 별도 경로에 시각화하거나 LBR 평가:

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain plot --output agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_review
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain lbr --output agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_lbr --pairs 250 --particles 64 --seeds 2026100821 2026100822
```

학습이 실행 중이면 writer lock 때문에 평가/plot은 거부한다. LBR는 고정 정책 상대 실제 대전의
공격자 ante/hand이며 낮을수록 좋다. 기본2seed ×250 deal pair ×2seat로 정책당1000hand다.
자동 LBR 평가는 하지 않는다.

## 자원과 안전

분할 기준이 엄격해져도 RAM 절감량은 아직 측정하지 않았다. RAM 상한이나 자동 RAM 중단은 없다.
기존과 동일하게 디스크 여유10GiB 미만 또는 k-means 상태500k 도달 시 저장 후 중단한다.
`--reserve-gib`는 디스크 조건이지 RAM 제한이 아니다. 큰 checkpoint의 원자적 교체에는
이전 파일+임시 파일 공간이 동시에 필요하므로10GiB가 항상 충분하다는 보장은 없다.
`--checkpoint-every 10000`으로 최신본을 저장하며, 변경 시 강제 종료 손실 구간도 달라진다.

## 선행 연구와 검증

[VFDT / Hoeffding Tree (Domingos & Hulten, 2000)](https://alchemy.cs.washington.edu/papers/pdfs/domingos-hulten00.pdf)
는 해당 노드의 표본 수와 bound로 후보를 비교한다. 이 제안은 전역 시간에 따른 상대SSE 개선율
역치이므로 그 방법을 재현하지 않는다. 관련 문헌과 구별한 휴리스틱 통제 실험이며 신규성 주장은 없다.
검증 질문은 초반 coarse bucket을 유지하다 완화하는 것이 같은 예산에서 더 유리한가이다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain build --check
python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_log_gain -v
```

작은 테스트는 역치, one-shot/재개 전체 상태 일치, 미커밋 로그 복구, 고정역치 기존 학습기와의
동일성, k-means 일치, source 검증, 보고서 및 LBR 호환성을 검사한다. 테스트 통과는1M 결과가 아니다.
