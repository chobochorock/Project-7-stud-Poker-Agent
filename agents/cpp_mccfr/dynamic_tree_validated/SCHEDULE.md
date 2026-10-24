# DT 분할 정지와 지속 비교

## 질문과 범위

분할로 표현력을 늘리는 이득보다 leaf별 학습 표본 분산과 통계 상속의 비용이 큰가?
사용자가 완료한 **같은 centered DT 1M checkpoint**에서 두 분기를 만들고 총10M까지 비교한다.
10M 역전이나 성능 개선을 가정하지 않는다. 정확 exploitability 실험도 아니다.

- `freeze`: 1M의 partition을 유지한다. **Regret와 평균전략 학습은 계속**한다.
- `continue`: 기존 centered/gain5% 분할을 계속한다. 분할 수/깊이 상한 없음.
- 두 분기는 DT뿐 아니라 동시 학습한 k-means, 평균전략 누적값, RNG를 동일하게 물려받는다.
  부모 체크포인트/기존 소스를 변경하지 않는다. 원본의 1M 평가 자료도 그대로 남긴다.
- 초기 key `(street, seat, legal mask)`와 34D feature도 유지한다. Absent-mindedness 방지까지
  동시에 바꾸지 않는다. Perfect recall 또는 CFR+IRA 보장 실험이 아니다.
- 같은 seed222와 hand index별 deal을 사용하지만 정책이 달라지면 탐색 경로/node 수도 달라진다.
  초기 self-play atlas와 학습정책의 데이터 편향도 이전 실험과 동일하게 남는다.

## 선행연구

- [Domingos & Hulten, VFDT, 2000](https://alchemy.cs.washington.edu/papers/pdfs/domingos-hulten00.pdf):
  충분한 streaming 표본을 확보한 뒤 분할을 판단하는 접근이 관련된다. 이번에는 그 Hoeffding
  검정을 구현하지 않았으며 변화하는 MCCFR regret label에 그 통계 보장을 적용하지 않는다.
- [Čermák 외, CFR+IRA, 2020](https://arxiv.org/html/1803.05392v2): 학습 중 추상화 refinement가
  직접 관련된다. 이번 것은 sampled DT의 고정/증식 ablation이며 논문 알고리즘 재현이 아니다.
- 고정 추상화 대 동적 refinement 비교는 새로운 방법 자체의 제안이 아니다. 비교군은 반드시
  동일1M 출발의 두 DT와 기존 hard256 k-means다. 이번 검토는 관련 연구이지 완전한 신규성 조사도 아니다.

## 준비된 두 실행

프로젝트 root: `D:\Experiment\Project-7-stud-Poker-Agent`.

```powershell
$base = 'agents/cpp_mccfr/data/dynamic_tree_validated'
$source = "$base/confirm_centered_seed222_50k"
$freeze = "$base/centered_seed222_freeze1m_10m"
$continue = "$base/centered_seed222_continue1m_10m"
```

원본 폴더명 `50k`와 달리 실제 원본 checkpoint는1M이다. 아래 `fork`는 학습하지 않고
checkpoint를 로드하여 독립 분기를 저장하고 시작점의 고정 panel을 평가한다.
**이미 준비된 폴더에는 다시 fork하지 않는다.** 다른 실험은 새 경로를 지정한다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule fork --source $source --run $freeze --mode freeze --eval-every 10000
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule fork --source $source --run $continue --mode continue --eval-every 10000
```

## 사용자 실행 명령

에이전트는 아래 10M 학습을 실행하지 않는다. `--hands`는 추가량이 아닌 **총 예산**이므로
각각1M에서9M을 더 진행한다. 매번 저장점부터 자동으로 재개하므로 `--resume`은 필요 없다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule train --run agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_freeze1m_10m --hands 10000000
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule train --run agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_continue1m_10m --hands 10000000
```

**차례로 실행하는 것을 권장한다.** 지속 버전의 RAM/체크포인트는 분할과 함께 증가하므로
1M의2.62GiB에서10M 메모리를 단순 보장할 수 없다. 숨긴 RAM/split 상한은 없다.
정지 버전은 원본 로딩 직후 분할 monitor만 버려 가벼워지지만 초기 로딩 때에는 원본 크기의 RAM이 든다.
분할 monitor는 지속 학습에는 필요하므로 지속 버전과 원본에서는 보존한다.

- 기본 checkpoint 간격은100k다. 대형 checkpoint 반복 쓰기를 줄이기 위한 선택이다.
  강제 종료/전원 장애에는 마지막 checkpoint 이후 최대100k 미만의 학습을 잃을 수 있다.
  자주 저장하려면 `--checkpoint-every 10000`을 지정한다.
- Ctrl+C는 현재 hand를 마치고 panel 평가 후 checkpoint를 저장한다. 저장 완료까지 기다린다.
- `--reserve-gib 10`, `--max-kmeans-states 500000`은 명시적 운영 안전장치다. 도달 시
  저장 후 중단하며 분할만 몰래 멈추는 옵션이 아니다. 저장 중 새 checkpoint를 위한 여유 공간도 필요하다.
- 원래 `run train`이 아니라 **`schedule train`으로 재개**한다. 소스/config hash 불일치 시 거부한다.

## 평가 및 출력

**매 hand 평가는 끈다.** 분기 시점과 이후10k마다, 그리고 종료/정상 중단 시
같은128개 5구 정보상태에서32particles로 current/average Local BR-gap을 측정한다.
기존1M의 고정 panel과 eval seed307/roots/particles가 같다. 평가 RNG는 학습 RNG와 분리된다.

```text
LOCAL_BR_GAP hands=1010000 tree_current=... kmeans_current=... tree_average=... kmeans_average=... roots=128 particles_per_root=32 unit=ante
PROGRESS hands=... target=... tree_leaves=... splits=... checks=... kmeans_states=... tree_nodes=... kmeans_nodes=... peak_rss_mib=... wall_seconds=...
```

- `panel.csv`: 각 평가 시점의 root별 원자료, signed crossfit/행동값 SE/미등록 조회수도 저장.
- `splits.csv`: **분기 이후** 검사/분할 로그. 부모의 누적 횟수는 config의 `schedule`에 기록.
- `per_hand.csv`: 기존 checkpoint 형식 호환용 **header-only** 파일. 매 hand 측정이 아니다.
- `checkpoint.bin`: 전체 정책/RNG/재개 상태. Freeze는 분할용 표본만 제거하고 전략은 보존한다.
- `config.json`: 부모 경로/identity/checkpoint SHA256/분기 hand, 분할 및 평가 조건.

그림은 기존의 시간축 MA100이 아니라 **각 시점128root 평균선과 root 분포의 min-max/중앙90% 띠**다.
표본 단위가 다르므로 과거 per-hand MA100와 같은 시계열로 합치지 않는다. 비교에는 과거 고정 panel만
붙인다. 띠는 신뢰구간이 아니다. Finite-particle max 편향은 남고 Local BR-gap은 exploitability가 아니다.

학습 완료 또는 Ctrl+C 저장 후, 새 output 경로로 그림을 만든다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule_report --runs freeze=agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_freeze1m_10m continue=agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_continue1m_10m --output agents/cpp_mccfr/data/dynamic_tree_validated/review_split_schedule_10m
```

각 분기의 선형/log-log 그림은 빨강DT/파랑k-means이고, DT/KM 비율도 함께 나온다.
종합 평균선 그림은 정지=빨강/지속=초록/k-means=파랑이다. 동일 예산의 k-means 원자료가
두 분기에서 일치하는지도 확인한다. 학습 중인 실행의 미저장 로그는 보고서로 확정하지 않는다.

## LBR 확인

Local BR-gap의 우위를 곧바로 실제 방어 성능이나 균형으로 해석하지 않는다. 완료 후 같은 평가
seed로 두 frozen 정책을 평가한다. 기본은 정책당1000hand/64particles다. 새 output만 허용한다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule lbr --run agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_freeze1m_10m --output agents/cpp_mccfr/data/dynamic_tree_validated/lbr_freeze_10m --seeds 2026100811 2026100812
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule lbr --run agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_continue1m_10m --output agents/cpp_mccfr/data/dynamic_tree_validated/lbr_continue_10m --seeds 2026100811 2026100812
```

## 천천히 분할하는 선택지

첫 비교에서는 정지 여부만 바꾼다. 아래는 **선택적인 별도 조건**이며 자동 실행하지 않는다.
Monitor window256은 그대로 유지하고 최소 방문 수와 검사 간격만 늘릴 수 있다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.schedule fork --source agents/cpp_mccfr/data/dynamic_tree_validated/confirm_centered_seed222_50k --run agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_slow1m_10m --mode continue --min-split-visits 1024 --split-check-visits 1024
```

Leaf 생성 이후 서로 다른 monitored hand 수를 n이라 할 때 추가 gate는
`n >= min_split_visits and n % split_check_visits == 0`이다. 기본값은 둘 다256으로 기존과 같다.
여기에 기존 parentRMSE/centered SSE gain5% 조건이 그대로 적용된다. 두 수는 window의 양의 배수다.
1024는 독립표본1024개로 검정한다는 뜻이 아니다. 분할 점수는 여전히 최근256개이고,
기존 leaf의 생성 이후 횟수를 이어받는다. 부하/표본 축적/일시적 분할 기회를 모두 바꿀 수 있다.

게임별 숫자는 위 인자로 조절한다. 다른 함수가 필요하면 `schedule.cpp`의 작은 `split_due`
함수를 새 실험용으로 변경할 수 있다. 임의 plugin 체계는 추가하지 않았다. 이미 시작한 분기의
소스를 바꾸면 재개 hash 검사가 실패하므로 반드시 별도 버전/실행으로 다뤄야 한다.

## 검증

```powershell
python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_schedule -v
```

작은 실제 학습으로 fork 전후 정책 일치, 부모 불변, 분할 정지와 regret 학습 지속,
기존 학습기와 지속 분기의 정책/node 일치, 저장/재개와 미저장 로그 복구,
동일 k-means 대조군, LBR 로딩 및 그림 생성 경로를 검사한다. 테스트는 수렴 보장이 아니다.

### 2026-10-08 준비 완료

위 두 분기 폴더는 실제로 생성했다. 둘 다 현재1M이며 **10M 학습은 실행하지 않았다**.
정지 분기의 checkpoint는29.21MiB, 지속 분기는2.056GiB다. Leaf80,438/split80,420 및
128root의 current/average 평가값이 부모와 같다. 원본 SHA256도 변하지 않았다.
[시작점 검증](../data/dynamic_tree_validated/review_schedule_initial_20261008/summary.json).

통합 테스트는 통과했다. 기존 학습기와 continue의 정책뿐 아니라 key별 regret/평균전략,
RNG/monitor/tree를 비교했다. 해시맵 저장 순서는 재개 후 달라질 수 있어 직렬화 바이트 순서가
아닌 정규화한 상태를 비교한다. 실제10M의 성능/메모리는 아직 측정하지 않았다.
