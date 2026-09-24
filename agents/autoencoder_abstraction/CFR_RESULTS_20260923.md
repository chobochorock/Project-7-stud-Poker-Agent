# AE/VQ Frozen Abstraction: 100k MCCFR Pilot

## 완료 범위

기존에 생성한 AE+k-means와 VQ-VAE의 bucket을 C++ 학습기에 연결하고,
각각 **100,000 training hands = 200,000 external-sampling traversals**를 완료했다.
Signed MCCFR이며 CFR+나 CFR-D가 아니다. Encoder/codebook은 고정이고,
그 bucket에 연결된 regret와 평균전략 테이블을 학습했다. 기존 k-means 재학습은 하지 않았다.

환경은 기존 heads-up C++ 7-stud v3, fixed H4, 5구 시작, raise cap 1/2/3,
ante 1000칩, stack 1000 antes다. 18D power feature/completion limit 128,
각 구 256 card codes, seat/합법 행동집합/기존 cumulative betting context를 분리한다.
완전한 information set representation 또는 perfect recall 실험은 아니다.

Abstraction 원본은 low-fold 10k 데이터의 representation seed 11이다.
Solver seed 7, evaluation seed 307. Low-fold 확률은 원본 데이터 수집 조건이며
이번 self-play에 강제로 적용하지 않았다. Regret 0, 합법 행동 균등에서 시작했다.
Representation 선택은 CFR 평가 이전에 고정했고 test score로 seed를 고르지 않았다.

[사용법과 전체 조건](CFR.md). 실제 실행 명령은 다음과 같으며 기존 출력은 덮어쓰지 않는다.
재실행하려면 새 출력 폴더명을 지정해야 한다.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
python -m agents.autoencoder_abstraction.run_cfr --method ae_kmeans --out-dir agents/autoencoder_abstraction/data/cfr_ae_100k_seed7_20260923
python -m agents.autoencoder_abstraction.run_cfr --method vqvae --out-dir agents/autoencoder_abstraction/data/cfr_vq_100k_seed7_20260923
```

기본 `--hands 100000`, Local gap은 시작 및 매 10k/최종 시점, 128 roots x 32 particles.
최종 LBR은 정책별 500 seat-pairs x 64 particles, 즉 **정책별 1000 평가 핸드**다.
현재/평균전략 각각을 평가하므로 모델 하나당 LBR 평가 핸드는 합계 2000이다.
100k 학습 핸드 수와 평가 핸드 수를 혼동하지 않는다.

## 최종 지표

괄호는 정규근사 95% CI. Gap은 root 단위, LBR은 좌석 교환 pair 평균 단위로 산출했다.
양의 LBR profit은 **평가자 수익 = 대상 정책의 손실**이다. 어느 값도 exact exploitability가 아니다.

| Abstraction | 전략 | Root Local BR-gap, ante | LBR profit, ante/hand |
|---|---|---:|---:|
| AE+k-means | 현재 | 3.836 [2.664, 5.008] | 7.173 [4.081, 10.265] |
| AE+k-means | 평균 | 6.178 [4.747, 7.608] | 8.485 [4.278, 12.692] |
| VQ-VAE | 현재 | 3.430 [2.461, 4.399] | 5.533 [2.281, 8.786] |
| VQ-VAE | 평균 | 4.783 [3.809, 5.756] | 6.083 [1.638, 10.528] |

동일 500 deal-pair에서 `VQ LBR profit - AE LBR profit`의 평균 및 CI:

- 현재전략: **-1.640 [-4.480, +1.201]** ante/hand.
- 평균전략: **-2.402 [-7.813, +3.009]** ante/hand.

VQ가 표본 평균상 낮지만 paired difference CI가 모두 0을 포함한다. 우위 확정은 불가하다.
두 모델 모두 LBR에 양의 수익을 허용하므로 균형 수렴을 확인한 결과도 아니다.
CI는 이 두 고정 정책에 대한 deal sampling 변동만 반영하며 학습 seed 변동은 포함하지 않는다.

LBR query 중 미방문 전체 전략 key 비율: AE 현재 0.0315%, 평균 0.0176%;
VQ 현재 0.0381%, 평균 0.0228%. Card code를 찾지 못한 것이 아니라 해당 베팅 문맥까지
포함한 key가 학습 테이블에 없다는 뜻이며 uniform legal fallback을 사용했다.

## 그래프

![AE metrics](data/cfr_ae_100k_seed7_20260923/metrics.png)

![VQ metrics](data/cfr_vq_100k_seed7_20260923/metrics.png)

두 모델 모두 uniform 초기 정책의 gap 약 1.770에서 초반 상승 후 대체로 감소했다.
이는 매 시점 continuation policy도 함께 변하는 root-only 지표다. 초기 uniform의 작은
gap만 보고 초기 정책이 강하다고 할 수 없다. 그 원인에 베팅 크기/rollout 분산이 얼마나
기여하는지는 이번 실험에서 분리하지 않았으며, 유한 particle max bias도 남아 있다.
그래프의 곡선은 평가 시점별 평균이지 지금까지의 gap을 누적 평균한 값이 아니다.

## 비용과 저장

| 항목 | AE+k-means | VQ-VAE |
|---|---:|---:|
| Training node visits | 235,125,711 | 224,536,923 |
| 학습된 전체 전략 key | 413,753 | 369,154 |
| 순수 학습 시간 | 127.53 s | 122.11 s |
| 검증/학습/평가/그래프 총 wall time, 빌드 제외 | 153.34 s | 148.41 s |
| 학습 중 process peak RSS | 141.61 MiB | 130.59 MiB |
| 최종 regret/평균전략 테이블 | 91.54 MiB | 81.68 MiB |
| 3개 구의 추론용 encoder+centers 합계 | 44.40 KiB | 44.40 KiB |

평가 비용은 training nodes에 포함하지 않는다. 같은 핸드 예산이지만 정책별 탐색 분기로
node budget은 다르다. 시간은 단일 머신 관측값이며 엄격한 독립 성능 benchmark가 아니다.
메모리는 표준 hash table과 cache를 포함하며 최종 LBR/로드 단계의 peak까지 측정한 값은 아니다.
작은 encoder를 써도 베팅 문맥별 regret table의 비용이 사라지는 것은 아니다.

각 run에 final policy, frozen atlas, CSV, 그래프, source/model hash와 실행 로그를 보관했다.
10k별 대형 checkpoint나 원시 궤적 복제본은 저장하지 않았다. 최종 테이블은 재추론용이며,
RNG와 카운터가 없으므로 exact resume checkpoint라고 부르지 않는다.

- [AE 데이터/모델](data/cfr_ae_100k_seed7_20260923/), [요약 JSON](data/cfr_ae_100k_seed7_20260923/summary.json)
- [VQ 데이터/모델](data/cfr_vq_100k_seed7_20260923/), [요약 JSON](data/cfr_vq_100k_seed7_20260923/summary.json)

## 검증 및 제한

- 모델별 source observation 81,513개 모두 Python/C++ bucket 배정 일치, mismatch 0.
- 두 종류 모두 10핸드 학습/평가 smoke, 비공개 정보 불변성, seat partition,
  확률합/합법 행동, 잘린/비유한/추가 byte weight 파일 거절 검사 통과.
- 100k 테이블을 저장/재로딩하고 50개 root의 현재/평균전략 완전 일치 확인 후 LBR 수행.
- 기존 C++ engine self-test 및 Python 7개 테스트 통과. 과거 대형 실험 전체를 재실행한 것은 아니다.
- 기존 hard-256의 actor-shared key와 달리 이번에는 seat까지 hard partition한다.
  기존 k-means와 feature 범위는 같지만 과거 점수를 동일 조건 대조군으로 인용할 수 없다.
- 18D/베팅 요약의 imperfect recall은 그대로다. 1개 representation seed + 1개 solver seed
  파일럿으로 AE/VQ의 보편적 순위나 원래 게임에서의 CFR 수렴을 주장하지 않는다.

## 이전 K-means와 종합

사용자 요청에 따라 기존 저장 결과를 합쳐 검토했다. **이번 보강에서는 추가 학습이나
재평가를 실행하지 않았다.** 아래는 서로 다른 평가 조건의 기록을 병기한 것이며 동일
조건 통계 검정이나 알고리즘 순위표가 아니다. K-means는 기존 frozen hard-256 power atlas다.

### LBR: 실제 대결에서 허용한 손실

단위 ante/hand, 낮을수록 대상 정책에 유리하다. 괄호는 deal-pair 정규근사 95% CI.

| Abstraction | 학습 hands | 현재전략 상대 LBR 수익 | 평균전략 상대 LBR 수익 |
|---|---:|---:|---:|
| 과거 k-means hard-256 | 100k | 3.478 [2.982, 3.975] | 5.516 [4.634, 6.399] |
| 이번 AE+k-means | 100k | 7.173 [4.081, 10.265] | 8.485 [4.278, 12.692] |
| 이번 VQ-VAE | 100k | 5.533 [2.281, 8.786] | 6.083 [1.638, 10.528] |
| 과거 k-means hard-256, 참고 | 1M | 1.954 [1.791, 2.118] | 1.711 [1.455, 1.967] |

- 과거 k-means 두 결과: 정책별 5,000 pairs = 10,000 평가 hands, 240 particles,
  seed 307. 이번 AE/VQ: 500 pairs = 1,000 평가 hands, 64 particles, 별도 seed stream.
  공격자의 근사 강도와 표본 수/난수 경로가 다르다. 같은 원래 seed 숫자도 같은 평가 deal을
  뜻하지 않는다. 이들 사이의 CI 겹침 여부로 동일 조건 유의성을 판단하지 않는다.
- 100k끼리 점추정만 보면 k-means, VQ, AE 순으로 손실이 작다. 하지만 그 순서가
  representation 자체의 우열이라는 증거는 아니다. AE/VQ끼리도 paired 차이는 미확정이다.
- K-means의 1M 결과는 100k보다 더 낮은 손실을 기록했다. 같은 계열의 학습량 증가 참고이며,
  100k만 학습한 AE/VQ와 동일 예산 비교로 사용하지 않는다.
- 과거 joint-leaf ensemble의 구현 의도 불일치는 이 독립 hard-256 비교군의 오류가 아니다.
  Leaf-refined 평가와 앞선 joint 평가의 hard-256 100k checkpoint/수익은 동일했으므로
  이를 두 독립 학습 seed 실험으로 중복 계산하지 않는다.

### Root Gap과 모델 크기

아래 Local gap도 평가 조건이 달라 직접 비율 비교나 순위 검정에는 쓰지 않는다.

| 100k 모델 | Local gap 현재 / 평균 (ante) | 후회/전략 테이블 | Training node visits |
|---|---:|---:|---:|
| 과거 k-means | 1.542 / 2.540 | 43.51 MiB | 187,666,720 |
| 이번 AE | 3.836 / 6.178 | 91.54 MiB | 235,125,711 |
| 이번 VQ | 3.430 / 4.783 | 81.68 MiB | 224,536,923 |

K-means gap은 seed 520, 512 roots x 128 particles의 독립 audit이고, 이번 AE/VQ는
128 roots x 32 particles다. 특히 유한 particle max bias가 달라 gap의 차이를 성능 차이와
동일시하지 않는다. Table 파일 크기는 실제 파일을 확인했으며 atlas/encoder는 제외했다.
과거 atlas는 추가 116,760 bytes, 이번 neural atlas는 각 45,464 bytes다.

과거는 actor-shared 전략 key, 이번은 actor별 hard partition이므로 전략 테이블 수와
메모리 차이를 encoder 고유 비용으로 해석할 수 없다. 또한 같은 100k hands에서도 node
visits가 다르다. 과거 메모리/시간은 ensemble과 함께 측정한 값이 있어 새 단독 학습의
RSS/시간과 직접 비교하지 않는다.

### 데이터 편향과 현재 결론

기존 k-means atlas는 과거 epsilon-0.2 self-play에서 생성했고, AE/VQ는 지정 low-fold
10k trajectory에서 학습했다. Atlas 이름의 `100m`은 원래 representation source를 가리키며
여기 적은 100k solver에 100M regret table을 로드했다는 뜻이 아니다. 원본 데이터/fit 예산이
다르므로 역사적 k-means 우위를 주장하기 위한 통제 비교는 아직 아니다.

반면 **동일 low-fold 데이터의 압축 진단**은 이미 있다. 3 seeds 평균 test MSE:

| 방법 | 5구 | 6구 | 7구 |
|---|---:|---:|---:|
| 새 raw k-means | 0.002827 | 0.002550 | 0.002195 |
| AE+k-means | 0.003114 | 0.002820 | 0.002471 |
| VQ-VAE | 0.003129 | 0.003052 | 0.002888 |

그 새 raw k-means와 과거 solver에 쓰인 atlas는 다른 모델이다. 새 raw k-means의 MCCFR은
아직 돌리지 않았다. MSE는 k-means의 목적과 가까운 복원 지표이지 CFR 전략 품질 지표가 아니다.

**종합: 현재 기록에서는 k-means가 유력한 기준선이고, AE/VQ가 이를 개선했다고 할 근거는
없다. 다만 같은 조건에서 AE/VQ가 열등하다고 확정한 실험도 아니다.** 다음 통제 비교의
최소 조건은 이미 저장된 동일 low-fold raw-kmeans seed 11에 이번과 같은 seat 분리/100k
MCCFR/동일 deal·particle LBR을 적용하는 것이다. 이는 제안일 뿐 이번에 실행하지 않았다.

역사적 원본:
[100k LBR](../lightgbm_regret_ensemble/data/leaf_refined_100k_seed7_20260920/lbr_seed307_p240/summary.json),
[100k gap와 학습 조건](../lightgbm_regret_ensemble/data/leaf_refined_100k_seed7_20260920/README.md),
[1M LBR](../lightgbm_regret_ensemble/data/seven_stud_1m_seed7_20260919_204332/lbr_seed307_p240/summary.json),
[동일 데이터의 압축 비교](RESULTS_20260923.md).
