# 로그 역치 DT 1M -> 10M 연장 및 인계

## 직접 이어 돌리기

먼저 기존 실행이 `PAUSED` 또는 `COMPLETE`로 끝났는지 확인한다. 같은 폴더에 두 학습기를
동시에 시작하지 않는다. 아래는 **현재 저장 지점부터 총10M까지**이며 추가10M가 아니다.
`train` 명령 자체가 재개 승인이고, 기존 중단 요청 파일을 지운 뒤 이어 간다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain_chunks train --hands 10000000 --chunk-hands 1000000
```

총 hand가 다음 정수1M 경계에 도달하면 C++ 프로세스가 정상 종료하여 RAM을 반환한다.
체크포인트/로그 일관성 및 그래프 생성이 끝나면 다음1M 구간을 자동 시작한다.
구간별 graph/summary는 보존하지만 큰 checkpoint는 최신1개만 유지한다.
원본1M 폴더는 별도로 보존한다. 분할은 멈추거나 재초기화하지 않는다.

## 안전 중단

실행 터미널에서 Ctrl+C를 누르거나 다른 터미널에서 다음을 실행한다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.log_gain_chunks stop
```

이 명령은 요청만 기록하고 즉시 반환한다. 실행기는 C++의 중단 핸들러가 준비된 뒤
CTRL_BREAK를 보내며 현재 hand, 마지막 평가와 체크포인트 쓰기를 마친 후 종료한다.
**`PAUSED` 및 실행 프로세스 종료를 확인한 후** 직접 재개한다. 로딩/저장/그래프 도중에는
해당 작업 종료를 기다릴 수 있으므로 즉시 강제 종료하는 기능은 아니다.
중간 hand에서 멈춰도 모델/regret/평균전략/RNG/monitor를 저장한다.

## 경로와 상태

- 보존된 원본: `agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_1m/`
- 연장/인계 폴더: `agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_10m/`
- 최신 상태: 위 연장 폴더의 `chunk_status.json`, 이력: `chunks.jsonl`.
- 학습 출력: `console.log`, 정확 실행 인자/바이너리 hash: `invocations.jsonl`.
- 지표: `panel.csv`, 분할 검사: `splits.csv`, 재개용 전체 상태: `checkpoint.bin`.
- 매1M 결과: `review_2000000/`, `review_3000000/`, ...의 `summary.json`,
  `local_br_gap_log.png`, `local_br_gap_linear.png`, `split_resources.png`.
- `controller/writer.lock`는 실행 관리자 중복을, `writer.lock`는 모델 동시 쓰기를 방지한다.

이번 연장은 `log_gain_chunks fork`로 원본1M의 체크포인트/로그/atlas/소스 스냅샷을 복사했다.
체크포인트 SHA256은 복사 전후 같으며 원본1M을 덮어쓰지 않는다.
같은 configuration identity를 유지하고 실제 분기 출처는 `chunks.jsonl`에 별도로 기록한다.
이미 만든 폴더에 `fork`를 다시 실행하지 않는다.

## 조건과 운영 한계

기존 학습 코드와 수식은 바꾸지 않았다. C++ 7-stud HU/H4/5~7구, seed222/eval307,
signed alternating ES-MCCFR/SIMPLE average, 매 hand 모델별2 traversals,
gain=`1/(1+log10(전역hand))`, window256/min_child32/min_rmse1ante가 그대로다.
Regret/평균전략은 표본 개수 비례 상속이며 monitor는 분할 시 초기화한다.
시간 t와 난수 상태는1M 경계에서 리셋하지 않는다. Split/depth cap은 없다.

비교군은 함께 이어 학습하는 hard256 k-means+betting-summary+seat다. Atlas의 기존
epsilon0.2 self-play100k/seed37301 편향과 DT 자체 정책 label 편향이 유지된다.
매10k마다 같은128root/32particles에서 current/average Local BR-gap을 기록한다.
붉은 DT/파란 k-means의 log-log, 평균비율 및 root분포 띠를 함께 그린다.
이는 매hand 평가, 신뢰구간, LBR 또는 정확 exploitability가 아니다.
이번 연장에 추가 LBR 평가는 자동 포함하지 않는다.

- 기본 체크포인트 간격100khand. 평가는 계속10k마다 수행한다. 원래10k 저장으로 시작했으나
  대형 체크포인트의 I/O 비용 때문에1,045,656hand에서 정상 저장 후 정기 저장만100k로 변경했다.
  중단/최종/1M 경계에서는 이 간격과 무관하게 저장한다. 학습 수식이나 평가 빈도 변경이 아니다.
- 시스템 가용 RAM이3GiB 미만이면 저장 후 정지한다(`--min-free-gib`). 모델 분할 수를
  줄이는 제약이 아니라 운영 중단 조건이다. 실제 남은 RAM은 다른 앱에도 좌우된다.
- 디스크 여유10GiB, k-means500k 상태의 기존 정지 조건도 유지한다.
- 자동 중단 후 자동 재시작하지 않는다. 자원 문제를 해결한 뒤 `train`을 다시 실행한다.
- 강제 종료/전원 장애로 checkpoint 뒤에 미커밋 로그가 남으면 관리자는 불일치를 거부한다.
  기존 `log_gain train --resume`의 suffix 복구 기능을 사용하여 저장 상태를 복구한 뒤 재개한다.
  정상 인계는 이런 강제 종료를 사용하지 않는다.
- 모든1M 체크포인트를 별도 보관하지 않으므로, 중간 예산의 LBR에는 해당 시점에서 일시
  중단하여 평가하거나 별도 snapshot 보관이 필요하다. 그래프와 지표는 계속 보존한다.

이번 질문은 기존 휴리스틱 스케줄의 장기 학습 여부이며 새 방법의 신규성/수렴 보장을 주장하지
않는다. 앞선 VFDT/Hoeffding-tree 검토와 달리 이 스케줄은 전역 시간에 따른 경험적 gain gate다.
고정0.15 장기 통제군은 아직 없고 고정5% DT의 보유 결과는1M이므로, 그 이후 동예산 직접
비교는 함께 학습하는 k-means에 한정된다.

## 실행기 검증

```powershell
python -m unittest agents.cpp_mccfr.dynamic_tree_validated.test_log_gain_chunks -v
```

작은32hand 체크포인트를 복사하여 연속학습 대16hand씩 나눈 재개의 전체 solver 상태를
비교한다. 별도 프로세스에서 stop 파일을 요청해 `PAUSED`/로그와 checkpoint 일치를 검사하고,
중단 없이 동일 hand까지 학습한 상태와 비교한다. Explicit 재개, 원본 hash 보존 및
구간별 report 생성도 확인한다. 테스트 통과는 큰 예산의 수렴이나 완주 보장이 아니다.
