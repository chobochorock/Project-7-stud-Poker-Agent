# ASWISS on 7-stud (heads-up, 5–7구)

작성 2026-10-27. 학습 seed 1 결과와, seed 2 및 상속 변형 결과(아래 절)를 담았다. 모두 완료된 실험이다.

## 무엇을 비교했나

**게임**
- 엔진은 `../../../cpp_mccfr/stud_mccfr.cpp`를 수정 없이 include해서 쓴다.
- 규칙은 v3다. heads-up이고, ante 1000, stack 1000 ante다.
- 학습과 평가 모두 5구 루트에서 시작한다. 4구의 버림/공개 선택(12액션)은 두 player 모두 기존 heuristic(`discard_reveal`)으로 고정했다.
- 따라서 실제로 푸는 게임은 "4구 heuristic을 고정한 5–7구 베팅 게임"이다.

**원자(atom)**
- 원자 = K=1024 power 클러스터 × 베팅 요약 × 합법 행동 집합이다. 엔진의 `--bucket power` key와 같다.
- 원 정보집합은 학습 중 거의 한 번씩만 나와서 정보집합별 라벨이 유의해질 수 없다. 그래서 라벨 단위를 원자로 바꿨다.
- 이 바꿈의 이론적 의미는 [수렴 증명 문서](../../ASWISS_CONVERGENCE_KO.md) 정리 3에 있다. ASWISS가 0으로 보내는 것은 bucket과 원자 사이의 상쇄까지다. 원자 안의 상쇄는 남는다.

**비교군**
- 모두 같은 학습기를 쓴다: external-sampling MCCFR+(RM+), 순회 뒤 bucket 일괄 갱신, 선형 평균.
- 학습량은 10.23M roots, seed 1이다.

| 비교군 | 분할 | atlas |
|---|---|---|
| context | context당 bucket 하나. 카드 정보 없음(mini-stud의 time step에 해당) | 없음 |
| k-means 256 | 원자 = bucket. 프로젝트 표준 K | K=256 |
| k-means 341 (bucket 수 맞춤) | 원자 = bucket. ASWISS의 최종 bucket 수에 맞춰 K를 **사후에** 정함 | K=341 |
| ASWISS | context에서 시작. 두 배 epoch마다 원자 라벨(누적 샘플 regret, z=2)로 분할. 자식 표는 0에서 다시 시작 | K=1024 |
| k-means 1024 | 원자 = bucket. mini-stud의 "완전 기억"에 해당 | K=1024 |

- **atlas 생성:** 세 atlas 모두 여기서 같은 자료로 새로 만들었다(heuristic ε=0.2 self-play 10만 핸드, seed 37301).
- **이전 자료:** 저장소의 이전 atlas와 LBR 결과 파일은 이 환경에 없어서 쓰지 못했다(`data/`가 로컬에만 있음).

**평가**
- **Policy-LBR:** 엔진의 `PolicyLBR`를 쓴다. exploitability의 근사 **하한**이며 단위는 ante/hand다.
  - 모든 스냅숏에 같은 deal, 같은 LBR seed, particle 64를 쓴다.
  - 학습 곡선은 2,000 pair로 평가했다. 마지막 스냅숏은 20,000 pair로 다시 평가했다.
  - LBR 한 pair의 표준편차는 약 17.5 ante이고 최대 482 ante까지 꼬리가 길다. 그래서 2,000 pair로는 구간이 넓다.
- **샘플 BR-gap 근사치:** 고정 deal 4,000개에서 external sampling 순회로 잰다. traverser가 방문한 모든 상태의 $\max_aQ-\sum\pi Q$를 더한 값이다.
  - 상계도 하한도 아니다.
  - 단위는 ante다. 기존 `agents/evaluation/br_gap`는 ante로 한 번 더 나눈다.
- 평균 정책과 현재 정책을 따로 잰다.

## 결과 (seed 1, 마지막 스냅숏, 20,000 pair)

| 비교군 | bucket 수 | 평균 정책 LBR (95%) | 현재 정책 LBR (95%) | BR-gap 근사치 (평균/현재) |
|---|---:|---:|---:|---:|
| context | 146 | 63.66 ± 2.56 | 65.97 ± 2.59 | 70,848 / 70,808 |
| k-means 256 | 37,060 | 1.520 ± 0.136 | 1.506 ± 0.157 | 653 / 623 |
| **k-means 341 (bucket 수 맞춤)** | 49,182 | **1.206 ± 0.111** | **1.373 ± 0.127** | 721 / 699 |
| ASWISS | 49,384 | 1.335 ± 0.133 | 1.461 ± 0.138 | 1,037 / 805 |
| k-means 1024 (원자) | 142,761 | 1.200 ± 0.143 | 1.500 ± 0.151 | 813 / 701 |

ASWISS에서 각 비교군을 뺀 차이다(같은 deal로 짝지은 95% 구간).

| ASWISS − | 평균 정책 | 현재 정책 |
|---|---:|---:|
| k-means 256 | −0.186 [−0.364, −0.007] | −0.045 [−0.227, +0.138] |
| **k-means 341 (bucket 수 맞춤)** | **+0.128 [−0.033, +0.290]** | +0.088 [−0.092, +0.269] |
| k-means 1024 | +0.135 [−0.050, +0.320] | −0.039 [−0.222, +0.144] |

![학습 곡선, k-means 256 대비 차이와 비율, bucket 수](results_stud_seed1_curves.png)

![마지막 평균 정책의 LBR과 BR-gap 근사치 대 bucket 수](results_stud_seed1_frontier.png)

## 해석 (seed 1 기준)

1. **bucket 수를 맞추면 ASWISS가 k-means보다 낫지 않다.**
   - 점추정은 0.13 ante 나쁘다. 구간은 0을 포함한다.
   - k-means 256 대비 우위(−0.19)는 bucket이 33% 더 많은 데서 온 것으로 보는 것이 맞다.
2. **이 게임에서는 카드 분할 해상도가 이미 포화로 보인다.**
   - k-means 341(4.9만 bucket)과 k-means 1024(14.3만 bucket)의 평균 정책 LBR이 같다(1.206 대 1.200).
   - 원자(1024) 수준을 넘어서는 이득이 원리적으로 없다(정리 3). 그래서 ASWISS가 이길 여지가 거의 없는 설정이었다.
   - 남은 오차는 카드 클러스터가 아니라 다른 곳에서 온다고 볼 수 있다. 후보는 베팅 요약 key(이력을 잃는 imperfect recall), 고정 H4, 학습량이다.
3. **분할된 자식은 표를 0에서 다시 시작한다.**
   - 마지막 epoch에도 6,655개 bucket이 분할되었다. 그래서 최종 분할의 많은 bucket은 5.12M roots만 학습했다. 반면 k-means 표는 10.23M roots를 모두 학습했다.
   - mini-stud에서도 같은 손해를 가설로 적었다. 확인하기 위해 상속 변형(`--inherit 1`, 자식이 부모 표를 복사)을 실행 중이다.
4. **context(카드 정보 없음)는 LBR 64 ante다.** 카드 정보가 결정적이라는 점은 분명하다.
5. **BR-gap 근사치는 LBR과 순위가 다르다.**
   - LBR이 가장 나쁜 k-means 256이 근사치는 가장 작다. ASWISS는 근사치가 가장 크다.
   - 이 근사치는 7구의 큰 팟 갈래 합이 지배하는 것으로 보인다. 그래서 이 문서의 결론에는 쓰지 않는다.
6. **이전 프로젝트 최고치와의 비교:** 사용자 기록상 이전 최고치는 LBR 1.7–1.8이다. 이 실험의 모든 카드 기반 비교군(1.20–1.52)이 그보다 낮다.
   - 평가 deal, pair 수, atlas가 달라 직접 비교는 아니다.
   - 낮아진 몫은 ASWISS가 아니라 학습기(MCCFR+, 선형 평균, 일괄 갱신) 쪽일 가능성이 크다.

## seed 2와 상속 변형 (완료, 2026-10-27)

**조건**
- 같은 atlas와 같은 학습량(10.23M roots)이다.
- 마지막 스냅숏을 seed 1과 **같은 deal**로 20,000 pair 평가했다.
- 상속 변형(`--inherit 1`)에서는 분할된 자식이 부모의 regret 표와 평균 표를 복사해 시작한다.

| 실행 | bucket 수 | 평균 정책 LBR | 현재 정책 LBR |
|---|---:|---:|---:|
| ASWISS seed 1 / seed 2 | 49,384 / 49,872 | 1.335 / 1.378 | 1.461 / 1.490 |
| k-means 341 seed 1 / seed 2 | 49,182 / 49,229 | 1.206 / 1.390 | 1.373 / 1.550 |
| k-means 1024 seed 1 / seed 2 | 142,761 / 142,869 | 1.200 / 1.320 | 1.500 / 1.586 |
| ASWISS 상속 seed 1 | 39,663 | 1.575 | 1.636 |

같은 deal로 짝지은 차이(95%)다. "두 seed"는 seed별 차이의 평균이고, 구간은 평가 deal 잡음만 반영한다.

| 비교 | 평균 정책 | 현재 정책 |
|---|---:|---:|
| ASWISS − k-means 341, seed 2 | −0.012 [−0.204, +0.179] | −0.060 [−0.233, +0.113] |
| **ASWISS − k-means 341, 두 seed** | **+0.058 [−0.067, +0.183]** | **+0.014 [−0.111, +0.139]** |
| ASWISS − k-means 1024, 두 seed | +0.097 [−0.039, +0.232] | −0.068 [−0.202, +0.067] |
| 상속 − 재시작 (ASWISS seed 1) | **+0.240 [+0.067, +0.413]** | +0.175 [+0.002, +0.347] |

**결론** (2개 seed 기준)
1. **bucket 수가 같으면 ASWISS와 k-means는 구별되지 않는다.** k-means 341과 k-means 1024도 구별되지 않는다.
   - 같은 비교군도 seed에 따라 0.04–0.18 ante 움직인다(k-means 341: 1.206과 1.390).
   - 이 학습 seed 변동이 평가 deal 잡음보다 크다. 위 구간은 그 변동을 반영하지 않으므로 실제보다 좁다.
2. **상속은 오히려 나빴다.**
   - LBR이 0.24 나빠졌고 bucket도 약 20% 적었다.
   - 부모의 regret을 물려받으면 자식들의 regret 방향이 같은 쪽에서 출발한다. 그래서 라벨 차이가 늦게 드러나 분할이 줄어든 것으로 보인다.
   - "자식 재시작의 학습 손해" 가설은 이 실험에서 지지되지 않는다.
3. **이 설정에서는 카드 해상도가 포화되어 ASWISS의 이득을 보일 여지가 없다.**
   - 원자(1024)의 학습 결과가 bucket 4.9만 개 k-means와 같다.
   - 다음 실험은 해상도가 실제로 필요한 설정이어야 한다. 후보는 다음과 같다.
     - 더 미세한 원자(K=4096)
     - 베팅 이력을 보존하는 원자
     - 4구 12액션 추가

## 메모리

**최종 정책 파일:** 스냅숏 = 현재 표 + 평균 표, 행동 8개 × double. ASWISS는 분할 지도도 포함한다.

| context | k-means 256 | k-means 341 | ASWISS | k-means 1024 |
|---:|---:|---:|---:|---:|
| 24 KB | 5.0 MB | 6.6 MB | 6.9 MB | 19.1 MB |

**프로세스 메모리** (실제 상주 메모리)
- 평가 프로세스(k-means 1024)는 138 MB다. 대부분은 카드 → 클러스터 배정 캐시(최대 100만 항목)다.
- context 학습은 14 MB다.
- 학습 전체(10.23M roots)의 최대 상주 메모리다(seed 2, 메모리 기록판).

| 실행 | 최대 메모리 | bucket 기록 | bucket 표 | 원자 구간 통계 | 분할 지도 |
|---|---:|---:|---:|---:|---:|
| ASWISS | 156 MB | 95,842 (살아 있는 것 58,097) | 26.2 MB | 24.1 MB | 2.2 MB |
| ASWISS 상속 (seed 1) | 158 MB | 78,911 | 13.1 MB | 23.4 MB | 2.0 MB |
| k-means 341 | 113 MB | 49,229 | 13.1 MB | – | 1.1 MB |
| k-means 1024 | 157 MB | 142,869 | 52.4 MB | – | 3.2 MB |

- bucket 표 MB는 vector capacity 기준이다. 나머지는 대부분 배정 캐시(최대 100만 항목)다.

**이론 문서의 한계와 일치하는 점**
- 학습 메모리는 k-means 1024와 같은 차수다. 라벨을 붙이려면 원자 통계가 필요하기 때문이다.
- 최종 정책은 k-means 1024의 약 1/2.8이다. 다만 bucket 수를 맞춘 k-means와는 크기도 성능도 비슷하다.

## AGENTS.md 점검 항목

- **k-means 기준선:** 같은 학습기로 새로 학습한 k-means 256을 모든 곡선에 겹쳐 그렸다. bucket 수를 맞춘 k-means 341도 함께 그렸다.
  - 이전에 기록된 hard256 결과(`curve_hard_10m.json`)는 이 환경에 없다. 조건도 달라서(이전 학습기, 다른 평가 seed) 겹쳐 그리지 않았다.
- **평가 조건:** 모든 비교군이 같은 평가 deal, LBR seed, particle 수, gap deal을 쓴다. 학습 seed도 같다(1).
- **남은 차이:**
  - 학습 deal 경로는 전략이 달라지는 순간부터 갈라진다.
  - k-means K=341은 ASWISS 결과를 보고 사후에 정했다.
  - ASWISS의 분할은 자기 실행의 샘플 regret으로 정해진다.
  - k-means는 카드 feature만 쓴다. 배팅 요약 key는 모든 비교군에 공통이다.
- **축과 지표:** LBR은 음수가 될 수 있어 선형 축에 그렸다. BR-gap 근사치만 log 축이다. 어느 것도 exploitability와 같은 것으로 다루지 않았다.
- **외삽:** 교차점이나 기울기로 외삽하지 않았다.

## 4구 12액션 학습 (`--h4 learn`, 실행 중)

4구에서 4장 중 버릴 카드 1장과 공개할 카드 1장을 고르는 결정(12액션)을 5–7구 베팅과 함께 학습한다.

- **정보 상태:** (좌석, 자기 4장). 문양 재배치에 대해 정규화한 엔진 key(`make_h4_view`)를 쓴다.
  - 카드 추상화가 없는 원 정보집합이다. 좌석당 약 1.6만 개다.
  - 12액션은 엔진의 정규 순서 (`h4_positions`, `h4_choice`)를 따른다.
- **학습 방법:** external sampling을 쓴다.
  - 상대의 버림/공개는 현재 전략에서 표본화하고, 상대의 평균 전략은 선형 가중으로 쌓는다.
  - traverser는 12가지를 모두 펼친다. 각 갈래 뒤에 같은 카드로 베팅 순회를 한다.
  - 4구 regret은 베팅 regret과 함께 순회가 끝난 뒤 RM+로 갱신한다.
  - 그래서 root 하나의 비용이 약 12–20배다. 이 환경에서 초당 약 50 roots였다.
- **4구 분할:** `--partition` 규칙을 따른다.

  | partition | 4구 표 |
  |---|---|
  | context | 좌석당 bucket 하나 |
  | atom | 4장 key마다 bucket 하나. 정확한 표이며 k-means 비교군에 해당 |
  | aswiss | 좌석당 하나에서 시작해 같은 라벨 규칙으로 분할 |

  4구에서는 원자가 곧 원 정보집합이다. 그래서 mini-stud와 같은 조건(정리 1)이 그대로 적용되는 결정이다.
- **평가:** `H4PolicyLBR`을 쓴다.
  - 이것은 `make_h4_lbr.py`가 엔진의 `PolicyLBR`에서 기계적으로 만든다. `cpp_mccfr/h4_ablation`과 같은 치환이다.
  - 상대 카드 추정(particle)의 사전확률로 학습된 4구 정책의 likelihood(24 × 확률)를 쓴다.
  - target은 자기 4구 선택을 학습된 정책에서 표본화한다.
  - LBR 공격자는 기존 heuristic 4구를 그대로 쓴다. 공격자의 12액션에 대한 BR이 아니다.
  - BR-gap 근사치는 양쪽 4구 선택을 평가 정책에서 표본화한 뒤 5–7구 베팅의 gap만 잰다.
- **검증:**
  - 학습된 4구 루트에 heuristic 선택을 넣으면 엔진의 5구 루트와 같다.
  - 4구 표가 스냅숏 저장/복원에서 보존된다.
  - 학습 전(균등) likelihood는 2(= 24/12)다.
  - 4구 모드도 SIGINT와 `kill -9` 뒤 재개한 스냅숏이 비트 단위로 같다.
  - heuristic 모드는 이전 실행과 같다(스냅숏 버전 바이트만 다름).
- **실행 중인 비교:** ASWISS(4구도 분할), k-means 256 / 341 / 1024(4구는 정확한 표)를 2.55M roots까지 돌린다. 스냅숏은 25만 roots마다 남긴다.
  - 이 환경에서 약 10–14시간 걸린다.
  - 더 길게 돌리려면 `run_100m.*`에 `-H4 learn`(`H4=learn`)을 준다.

## 100M 실행 (중단 후 재개 가능)

`run_100m.ps1`(Windows)와 `run_100m.sh`(Linux/WSL)는 비교군을 하나씩 별도 프로세스(코어 1개)로 돌린다.
- 기본 비교군은 ASWISS와 k-means 256 / 341 / 512 / 1024다.
- k-means 341은 10M에서 맞춘 K다. 100M에서 ASWISS의 bucket 수가 어디까지 늘지 모르므로 512로 범위를 넓혀 두었다.
- atlas 네 개는 `atlas/`에 들어 있다. 모두 heuristic ε=0.2 self-play 10만 핸드, seed 37301로 만들었다.

```powershell
# 저장소 루트에서 실행
$s = "agents\adaptive_buckets\aswiss\stud7\run_100m.ps1"
powershell -ExecutionPolicy Bypass -File $s                  # 시작 (멈춘 뒤 같은 명령을 다시 실행하면 이어서 진행)
powershell -ExecutionPolicy Bypass -File $s -Action status   # 진행 상황
powershell -ExecutionPolicy Bypass -File $s -Action stop     # checkpoint 저장 후 종료
powershell -ExecutionPolicy Bypass -File $s -Action eval     # 새 스냅숏 평가 (LBR 2,000 pair, 이미 한 것은 건너뜀)
powershell -ExecutionPolicy Bypass -File $s -Action final    # 가장 최근 스냅숏, LBR 20,000 pair
powershell -ExecutionPolicy Bypass -File $s -Action plot     # 그림
# 옵션: -Roots 100000000 -Seed 1 -Arms aswiss,k256,k341,k512,k1024 -Workers 4 -Out <폴더> -Python python
#       -H4 learn (4구 12액션 학습) -SnapshotEvery 5000000 -CheckpointEvery 1000000 -CurveEvery 100000
```

```bash
bash agents/adaptive_buckets/aswiss/stud7/run_100m.sh run      # status | stop | eval | final | plot
# 환경 변수: ROOTS SEED ARMS OUT WORKERS PYTHON H4 SNAPSHOT_EVERY CHECKPOINT_EVERY CURVE_EVERY
```

**중단과 재개**
- **checkpoint 저장 시점:** 다음 네 경우에 학습 상태 전체를 `checkpoint.bin`에 원자적으로 저장한다.
  - 1M roots마다
  - epoch 끝마다
  - Ctrl+C나 종료 신호를 받았을 때
  - 출력 폴더에 `STOP` 파일이 생겼을 때 (`-Action stop`이 만든다)
- **저장 내용:** bucket 표, 분할 지도, 원자 구간 통계, 두 RNG 상태, 출력 파일 길이다.
- **재개:** 같은 명령을 다시 실행하면 checkpoint 이후 출력 줄을 잘라 내고 그 시점부터 이어 간다.
- **재현성 확인:** SIGINT, `kill -9`, STOP으로 끊고 이어 간 실행의 스냅숏이 끊지 않은 실행과 비트 단위로 같았다. 끊지 않은 실행은 10M 원래 실행의 스냅숏과도 같았다.
- **강제 종료:** 작업 관리자 강제 종료나 정전이면 마지막 checkpoint 이후(최대 1M roots)만 다시 계산한다.
- **이미 끝난 비교군:** 다시 실행해도 바로 끝난다.
- **실행 중인 비교군:** `.pid`로 확인해 건너뛴다.
- **학습량 늘리기:** `--roots`를 늘려 재개할 수 있다. 다만 마지막 epoch가 잘린 지점에서 이미 분할했다면, 처음부터 더 길게 돌린 실행과 epoch 경계가 다르다.

**저장되는 그래프용 자료** (비교군 폴더마다)

| 파일 | 내용 |
|---|---|
| `log.csv` | epoch마다: bucket 수(street별), 분할 수, 원자 수, 샘플 L/C/H, 시간, 메모리(최대 RSS, 표 크기) |
| `curve.csv` | 10만 roots마다: 학습 deal에서 잰 현재 정책의 샘플 BR-gap 근사치(평균, 표준오차), bucket 수, 최대 RSS. 학습 순회의 부산물이라 추가 계산이 없다 |
| `snapshots.csv`, `snapshot_*.bin` | epoch 끝과 500만 roots마다의 정책 스냅숏(현재 표 + 평균 표 + 분할) |
| `eval.jsonl`, `lbr_pairs_*.csv` | `eval`/`final` 결과. 스냅숏 × 평균/현재 정책의 LBR(평균, 표준오차)과 BR-gap 근사치, 그리고 pair별 원자료(짝지은 차이 계산용) |
| `results_s1_*.png` (12액션은 `results_h4_s1_*.png`) | `plot`이 그리는 그림: LBR과 근사치 학습 곡선, k-means 256 대비 차이와 비율, bucket 수, `curve.csv` 곡선, bucket 수 대비 LBR |

**규모 추정** (이 환경 기준: 4개 병렬, 10M에 2–2.5시간)
- **시간:** 100M에 비교군당 약 22–25시간이다. 평가는 2,000 pair에 약 2분, 20,000 pair에 약 20분이다.
- **디스크:** 스냅숏이 비교군당 약 27개라 전체 약 1.5 GB다(k-means 1024는 개당 약 20 MB). checkpoint는 비교군당 수십 MB이고 덮어쓴다.
- **메모리:** 비교군당 약 110–160 MB(10M 측정값)다. 100M에서는 bucket 기록이 늘어나는 만큼 조금 커진다.

## 실행

```bash
g++ -O3 -std=c++17 aswiss_stud.cpp -o aswiss_stud
./aswiss_stud self-test --atlas ATLAS_K1024
./aswiss_stud train --partition aswiss --atlas ATLAS_K1024 --roots 10230000 --z 2 --seed 1 --out runs/aswiss_k1024_z2_s1
./aswiss_stud train --partition atom --atlas ATLAS_K --roots 10230000 --seed 1 --out runs/atom_kK_s1
./aswiss_stud train --partition context --roots 10230000 --seed 1 --out runs/context_s1
python evaluate.py ./aswiss_stud runs --runs aswiss_k1024_z2_s1:aswiss:ATLAS_K1024 atom_k256_s1:atom:ATLAS_K256 ...
python evaluate.py ./aswiss_stud runs --runs ... --epochs 9 --pairs 20000 --tag final
python plot_stud.py runs results_stud_seed1
```

atlas는 엔진으로 만든다. 예:

```bash
stud_mccfr --bucket power --fit-hands 100000 --clusters K --power-samples 128 --fit-sample-cap 50000 \
  --fit-policy-epsilon 0.2 --save-atlas ATLAS --start-street 5 --algorithm mccfr --ante 1000 \
  --hands 2 --iterations 0 --seed 37301
```

**실행 시간:** 4개 병렬 기준으로 원자 기반 학습은 비교군당 약 2–2.5시간, context는 약 5.5시간이었다. 평가는 2,000 pair에 약 2분, 20,000 pair에 약 20분이었다.
