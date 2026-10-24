# Small-K abstraction: convergence speed and plateau

질문: power cluster 수 K가 작으면 MCCFR이 얼마나 빨리 평평해지고, 그 정체 수준은 어디인가.
K512 power-memory16 기준(`data/made_call_r1000_k512_epsheur20_memory16_*`)과 K만 다르게 맞춘 통제 비교다.
계획·선행연구 위치·허용되는 주장은 `data/small_k_plateau_20261010/PLAN.md`에 있다.

## Usage (PowerShell, project root)

```powershell
# 1. 학습: "K:seed offset" arm마다 단일 thread 프로세스 하나 (1,000만 root에 arm당 약 70~100분)
.\agents\cpp_mccfr\small_k_plateau\launch_train.ps1 -Arms "1:0","2:0","4:0","8:0","16:0","64:0","512:0","1:1000","1:2000"
# 2. 평가: checkpoint가 생기는 대로 평가 (학습과 동시에 실행 가능)
.\agents\cpp_mccfr\small_k_plateau\evaluate_checkpoints.ps1 -Parallel 6 -Wait
# 3. 집계와 그림
python -m agents.cpp_mccfr.small_k_plateau.analyze
```

## Input

- `bin/stud_mccfr_memory16.exe` (2026-08-03 빌드. 현재 `stud_mccfr.cpp`보다 오래된 실행 파일이며, 기준 K512를 만든 것과 같은 파일).
- `agents/autoencoder_abstraction/bin/neural_mccfr.exe --evaluate-checkpoint` (동적 링크라 PowerShell에서만 실행됨).

## Output (`data/small_k_plateau_20261010/`)

- `k<K>[_s<offset>]/`: atlas, checkpoint `k<K>_{1k,3k,10k,30k,100k,300k,1m,…,10m}.bin`, 프로세스별 log.
- `eval/<arm>/<budget>/seed_<s>/`: `lbr_pairs.csv`, `local_gap.csv`, `lbr_queries.csv`.
- `summary.json`, `tables.md`, `fig_*.png`.

## Metric

- Policy-LBR 공격자 수익(ante/hand, 64 particles, seat를 바꾼 deal pair). exploitability의 **근사 하한**이며 exploitability가 아니다.
- root Local BR-gap(ante, 128 root × 32 particles, plug-in 추정, 5구 첫 결정만).
- 예산 단위: root 반복 1회 = 5구 deal 1개 = traversal 2회. node visit은 따로 기록한다.

## Limit

- K=1을 빼면 학습 seed가 1개다. 구간은 평가 deal 잡음만 반영한다.
- 1k, 3k, 30k, 300k는 주 사슬에 속하지 않는 곁가지 checkpoint다.
- regret prior(made-call)는 기준 run의 이력대로 100만 root까지만 적용된다.
- '정체 수준'은 이 학습기와 이 평가기 기준의 값이며, 추상화가 표현할 수 있는 한계가 아니다.
