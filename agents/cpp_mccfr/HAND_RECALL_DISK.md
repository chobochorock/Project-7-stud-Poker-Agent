# Hand-recall: 디스크 저장형 1M 재개

기존 [실험 정의](HAND_RECALL.md)의 학습법, 정보집합 key, atlas, seed와
Local BR-gap을 유지하고 **저장 방식만 변경**한다. Signed alternating
external-sampling MCCFR + SIMPLE 평균전략이며 CFR+가 아니다.
새로운 학습법 연구가 아닌 저장 계층 변경으로, RAM 버전과 정확한 업데이트
일치가 필수 비교 기준이다. 장기1M 학습은 사용자가 시작한다.

## 실행

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
$old = "agents/cpp_mccfr/data/hand_recall_kmeans256_1m_seed111"
$run = "agents/cpp_mccfr/data/hand_recall_kmeans256_disk_1m_seed111"

# 최초 한 번 변환만 수행. 준비 완료이면 덮어쓰지 않는다.
python -m agents.cpp_mccfr.hand_recall_disk prepare --source-run $old --run $run

# 두 군 모두 총 1M까지. 중단 뒤에도 같은 명령으로 자동 재개한다.
python -m agents.cpp_mccfr.hand_recall_disk train --run $run --hands 1000000

# 상태는 학습 중에도 읽을 수 있다.
python -m agents.cpp_mccfr.hand_recall_disk status --run $run

# 종료/일시 중단 뒤 CSV 내보내기 및 그래프 생성
python -m agents.cpp_mccfr.hand_recall_disk plot --run $run --window 100
```

`--hands`는 추가량이 아닌 **총 목표**다.2,439hand를 가져온 경우997,561hand를
추가한다. 원본 checkpoint/소스/지표를 덮어쓰거나 삭제하지 않는다.
Plot은 기존처럼 빨강/파랑 비교, current/average 분리, MA100의 min~max,
5~95백분위 띠, 평균선, 로그/선형 축과 비교군 대비 비율을 출력한다.
학습 중 plot은 쓰기 잠금으로 거절한다. `status`는 읽기만 한다.

학습을 계속하면서 중간 그래프를 보려면 별도 읽기 전용 도구를 사용한다.

```powershell
python -m agents.cpp_mccfr.hand_recall_live_plot --run $run --window 100
```

시작 시 완료 hand 수를 고정하고, 변경되지 않는 `meta/metrics` prefix만
1,000hand씩 짧은 read-only 연결로 나누어 읽는다. 각 연결을 닫은 뒤
`$run/reviews/시각_핸드수/`에 CSV, 로그/선형 그래프, sampling 진단과 요약을
저장한다. 원본 CSV/모델/학습 코드는 덮어쓰지 않으며 이 출력은 재개용
checkpoint가 아니다. 읽는 동안 WAL 정리가 잠시 지연될 수 있으므로
각 조회에는 15초 제한이 있다. 전체 조회 동안 WAL snapshot을 붙잡지 않으며,
학습 중 append된 새 hand는 이번 그림에 포함하지 않는다. 자동 반복 조회는 하지 않는다.
Crossfit은 음수를 보존하며 정확 gap의 불편 추정량이 아니다.

## RAM, 저장과 복원

- 기존5M 상태 제한은 없다. Full abstract information key를 SQLite 기본키로
  사용하고 필요한 node만 읽어 수정한다. Cache miss는 미학습 상태라는 뜻이
  아니며 DB에도 없을 때만 기존 fallback을 쓴다. Raw hand로 별도 sharding하지 않는다.
- 기본 `--db-cache-mib 256`, `--node-cache-entries 32768`, atlas cache100,000개.
  모델 전체를 RAM에 올리지 않지만 이는 **전체 process RSS의 엄밀한 상한은 아니다**.
- 노드16개 double 중 비트가0인 항목만 생략한다. 음수 regret, 평균전략
  누적량, 방문수, -0 비트까지 보존한다. 양자화/가지치기/재군집화는 없다.
- `model.sqlite`에 두 군 table, RNG/진행 상태, 평가 로그를 함께 저장한다.
  매 저장마다 전체 모델을 복사하지 않는다. WAL/SHM은 정상 작업 파일이다.
  **실행 중 삭제하거나 DB 본체만 복사하지 않는다.**
- 기본 `--commit-every 25`: 두 군 node, RNG, 완료 hand와 평가행을 같은
  transaction으로 저장한다. Ctrl+C는 진행 중 hand를 마치고 저장을 요청한다.
  강제 종료 후에는 마지막 commit에서 이어가며 미저장 구간을 다시 계산한다.
  장치 고장/파일 손상까지 복구하는 백업은 아니다.
- 기본 `--reserve-gib 20`: 여유 공간이 부족하면 hand 경계에서 저장하고
  종료코드2로 일시 중단한다. 갑작스러운 디스크 쓰기 실패는 마지막 성공한
  commit으로 복원한다. 공간 확보 후 같은 명령으로 재개한다.
- 상태 증가 자체를 해결하지 않으므로 **1M의 용량/시간을 보장하지 않는다**.
  SSD random I/O로 RAM 버전보다 느릴 수 있다. Cache는 가득 차면 flush하는
  단순 구조이며 LRU/분산 저장은 구현하지 않았다.
- CSV는 종료 또는 plot 때 생성하는 파생 파일이다. 뒤처져도 DB가 기준이다.
  변환 중 중단되면 같은 `prepare`로 원본부터 idempotent upsert한다.
  변환 완료 전에는 학습을 거절한다. 변환 도중 원본 run을 변경하지 않는다.
- 원본 source hash를 확인한다. 기존 `hand_recall.cpp`는 독립실행 main의
  compile guard만 추가했으며 importer는 이 차이만 허용한다. 이후에는 기존
  RAM 명령의 hash-resume 대신 위 명령을 사용한다. 새 run도 준비 이후 학습/
  평가 코드가 바뀌면 재개를 거절한다. 캐시/저장 주기/목표량은 변경 가능하다.

## 지표

매 hand 업데이트 후 별도5구 첫 결정 정보상태1개, seed307,32conditional
particles에서 평가한다. 두 군/current/average에 같은 root/particles/rollout
seeds를 사용하고10k마다 고정128root도 평가한다. 평가를 학습에 사용하지 않는다.

기존 gap은 `max_a mean(Q_a) - sum_a sigma(a) mean(Q_a)`, 단위는 ante다.
원본과 같은 연산 순서를 유지하고, 아래 진단만 **변환 후부터** 추가한다.

- `*_crossfit`: 한쪽 절반에서 고른 행동을 다른 절반에서 평가한 signed
  deviation을 양방향 평균한다. 음수도 남긴다. 정확 gap의 불편 추정량이나
  정확한 편향 보정값은 아니다.
- `*_mean_action_se`: 합법 행동별 rollout 수익 표준오차의 평균.
  행동 간 공분산/학습 seed 분산을 포함하는 gap 신뢰구간은 아니다.
- `*_uniform_fallback`: root/continuation 전체 조회 중 uniform fallback 횟수.
  평균전략 mass가0이면 현재 regret 정책을 먼저 쓰고, 양의 regret도 없을 때
  uniform이다. 균등하게 학습된 정책 전체를 세는 값도 아니다.
- 고정 패널 `action_values` 테이블: Q평균8, SE8, 정책8개 double.
  policy0~3은 recall-current, baseline-current, recall-average, baseline-average.
  불법 행동의0값은 분석에서 제외한다. 합법 mask는 대응하는 metrics의
  `root_seed`를 기존 `epoch7::root_for`에 넣어 복원한다.
  매 hand에는 큰 rollout replay를 저장하지 않는다.
- 이전2,439행의 신규 진단은 빈칸이다. 과거 값을 계산한 척0으로 채우지 않는다.
  기존21개 열은 유지한다. `payload_bytes`는 기존 RAM RegretNode 기준 논리
  payload를 연속 기록하는 값이지 SQLite 용량이나 현재 RAM 사용량이 아니다.

이는 gap 상승 원인을 해결하거나 수렴을 보장하는 변경이 아니다. Local BR-gap,
LBR 수익, exploitability를 구분한다. 고정 정책/root의 particle 수 검사는 별도다.

## 출력과 검증

`PROGRESS`의 값에는 이름이 붙는다: 완료 `hands`, 안전 저장 `committed_hands`,
총 `target`, 군별 `*_states`, 최고 process working set `peak_rss_mib`,
DB 본체 `db_gib`, 남은 공간 `free_gib`, 이번 process의 실제 node
`db_reads/writes`. `train_seconds/eval_seconds`는 기존 누적 계측에 이어지지만
마지막 commit/CSV export 등 모든 비용을 포함한 wall time은 아니다.

```powershell
python -m agents.cpp_mccfr.hand_recall_disk build --check
python -m unittest agents.cpp_mccfr.test_hand_recall_disk agents.cpp_mccfr.test_hand_recall -v
```

작은7-stud 수 hand와 합성 데이터만 사용한다. RAM/DB regret, 평균전략,
RNG, 방문수, traversal return, 기존 gap의 정확한 일치,3항목 cache eviction,
lossless codec, rollback/재개, 구 checkpoint 변환, 연속/분할 실행과 CSV/plot
호환성을 검사한다. 장기1M 성능 시험을 대신하지 않는다.

설치된 MinGW `opt/include/sqlite3.h`, `opt/lib/libsqlite3.a`를 재사용한다.
추가 설치는 없다. 저장 설계 근거: [WITHOUT ROWID](https://sqlite.org/withoutrowid.html),
[cache_size](https://sqlite.org/pragma.html#pragma_cache_size), [WAL](https://sqlite.org/wal.html).
실행 시 확인한 SQLite3.35.5에는 복수 connection이 동시에 쓰기/checkpoint할 때 발생할 수 있는
WAL-reset 이슈의 수정이 없다. 이 경로는 C++ writer/checkpointer 하나만 허용하고
상태 조회는 read-only로 연다. 실행 중 외부 SQLite 도구로 쓰기/checkpoint하지 않는다.
기존 anonymous-namespace 빌드 경고는 남아 있으며 정확성 검증과 구분한다.

실제 기존 run의 변환 및 무결성 검사 결과:
[MIGRATION.md](data/hand_recall_kmeans256_disk_1m_seed111/MIGRATION.md).

## 재방문율 확인

```powershell
python -m agents.cpp_mccfr.hand_recall_growth --run $run
```

학습 방문 횟수와 고유 상태 수를 읽어 `revisit_audit_HANDS.json`으로 저장한다.
모델을 변경하지 않고 일관된 commit snapshot을 streaming으로 읽는다.
큰 DB에서는 전체 순회가 오래 걸리며 읽는 동안 WAL checkpoint가 지연될 수 있다.
종료 노드/평가 조회는 제외하고, 같은 hand 안의 재방문은 포함한다.
[3,689hand 측정 및 해석](data/hand_recall_kmeans256_disk_1m_seed111/REVISITS_3689.md).
