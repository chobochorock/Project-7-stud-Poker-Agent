# Compact Reservoir Deep CFR

기존 7-Stud Deep CFR의 게임 순회와 신경망 입력은 유지하면서 reservoir만
무손실 token 형식으로 저장하는 격리 실험이다. 기존 학습 코드와 실행 중인
모델의 기본 동작은 변경하지 않으며 compact 경로는 별도 실행 파일을 쓴다.

2026-09-22 저장 정리: 사용자 요청으로 `reference_t100_k3000_v1`에 남은
6회차 임시 `samples_i6_p0.bin`(68,742,765 bytes)을 삭제했다.
5회차 `checkpoint.pt`, `reservoirs.npz`, 모든 신경망과 로그는 보존했다.
재개 코드는 checkpoint/reservoir를 읽고 다음 회차 sample을 다시 생성하므로
이 임시 출력은 재개 입력이 아니다. 삭제 후 세 신경망의 로딩/유한 출력 검사를 통과했다.
전체 학습 재개를 실행한 것은 아니다. [삭제 기록](../data/storage_cleanup_20260922.json).

## 저장 형식

- viewer/street: `uint8`
- 카드 12칸: 53-way one-hot 대신 `uint8[12]`
- betting history 24칸: 49-way one-hot 대신 `uint8[24]`
- 연속 feature 7개: 기존 `float32[7]` 유지
- legal actions: 8-way one-hot 대신 `uint8` bit mask
- regret/policy target: 기존 `float32[8]` 유지

Minibatch를 만들 때 기존 1,832차원 `float32` tensor를 정확히 복원한다.
수치 양자화는 하지 않는다.

## 검사

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py -B deep_cfr_compact\test_compact_reservoir.py
```

## Smoke test

기존 학습과 포트가 충돌하지 않도록 `29731`을 사용한다.

```powershell
& $py -B deep_cfr_compact\train.py `
  --run-dir deep_cfr_compact\runs\smoke_v1 `
  --start-street 5 --iterations 1 --traversals 2 `
  --memory-capacity 100 --hidden 32 --layers 1 `
  --batch-size 8 --advantage-steps 1 --policy-steps 1 `
  --save-policy-every 1 --threads 1 --port 29731 --seed 42001
```

`deep_cfr_compact_traverse.exe`는 `--compact-output 1`을 사용하므로 장기
학습의 sample 파일도 compact schema다. 동일 traversal을 dense/compact로
각각 기록한 회귀 검사에서 1,043 records의 tensor, target, metadata,
trajectory BR-gap이 정확히 일치했고 파일은 `7,684,852 -> 109,543 bytes`로
줄었다.

## 2026-08-23 smoke 결과

- dense tensor 무손실 왕복: 통과 (`np.array_equal`)
- reservoir 교체 표본화 및 checkpoint 복원: 통과
- 슬롯 크기: `7,365 -> 103 bytes` (`71.50x`)
- 100-slot reservoir 3개 checkpoint: `2,213,540 -> 38,104 bytes`
- 1 iteration 학습 및 2 iteration resume: 통과
- trajectory BR-gap, street별 fit metric, policy checkpoint 출력: 통과
- 생성한 `policy.pt`의 100-hand evaluator 로드/대전 경로: 통과
- dense/compact C++ writer 동일성 및 BR-gap 동일성: 통과 (`70.15x`)
- root 진행 heartbeat와 중간 BR-gap 출력: 통과

두 traversal만 사용한 smoke이므로 regret과 대전 성능 수치는 학습 성능으로
해석하지 않는다.

## BR-gap

각 external-sampling root에서 traverser 정보집합마다 이미 계산된 action
value로 다음을 합산한다.

```math
g(q)=\sum_{I\in q}\left[\max_a Q(I,a)-\sum_a\pi(a\mid I)Q(I,a)\right].
```

별도 BR/LBR, rollout, network query는 발생하지 않는다. 다음 필드가 P0/P1
generator 결과와 `history.jsonl`에 저장된다.

- `trajectory_br_gap_mean_ante`: root별 합의 평균
- `trajectory_br_gap_standard_error_ante`: 평균의 표준오차
- `trajectory_br_gap_by_street_mean_ante`: 5th/6th/7th 기여

이는 sampled trajectory-relaxed 진단값이며 exact exploitability는 아니다.

## 100-iteration 기준 실험

[`train_100_reference.ps1`](train_100_reference.ps1)은 다음 설정을 사용한다.

- `T=100`
- `K=3,000 traversals/player/iteration`
- advantage reservoir `2M/player`
- strategy reservoir `5M`
- `hidden=256`, residual layers `2`
- advantage SGD `2,000`, policy SGD `4,000`
- checkpoint `5 iterations`, root heartbeat `100 traversals`
- LBR 실행 없음

Compact reservoir의 예약 메모리는 약
`(2M + 2M + 5M) * 103 = 927MB`이고 모델·minibatch·일시 배열을 더한 실제
process RAM은 이보다 크다. 현재 single-state IPC에서는 약 1~3일이 걸릴 수
있으므로 기존 장기 학습과 동시에 시작하지 않는 편이 안전하다.

```powershell
.\deep_cfr_compact\train_100_reference.ps1
```

중간 출력:

```powershell
Get-Content deep_cfr_compact\runs\reference_t100_k3000_v1\train.stdout.log -Wait
```

스크립트를 중단 후 다시 실행하면 같은 폴더의 `checkpoint.pt`를 감지해
자동으로 `--resume`한다.

학습 종료 후 BR-gap 그래프:

```powershell
& $py -B analyze_deep_cfr_history.py `
  --run-dir deep_cfr_compact\runs\reference_t100_k3000_v1 `
  --force
```

분석 명령의 기본값은 heuristic/LBR 대전을 실행하지 않고 저장된 BR-gap만
그린다.
