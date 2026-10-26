# ASWISS 실험: 누적 regret 라벨로 bucket 분할 (mini-stud)

2026-10-26 첫 실행. 이론은 [REGRET_SPACE_ABSTRACTION_KO.md](../REGRET_SPACE_ABSTRACTION_KO.md)의 식 (6) 기반 분할((B) 방식)이다.

## 무엇을 하는가

가장 거친 추상화에서 출발한다. time step마다 bucket 하나를 둔다. time step은 (좌석, street, 현재 street의 베팅 진행, 행동집합)이다. 카드와 이전 street 베팅은 구분하지 않는다. `small`에서는 bucket 40개가 된다.

학습은 다음 두 방식 중 하나다.

- external sampling MCCFR+: 매 반복 양쪽 traverser가 한 번씩 traversal하고, bucket별로 RM+를 적용하며, 평균은 선형 가중이다.
- 전체 순회 CFR+

epoch 길이는 두 배씩 늘린다. 각 epoch가 끝나면 다음 순서로 분할한다.

1. **라벨:** 원 정보집합마다 그 epoch의 누적 샘플 regret R̃_I를 보고 라벨을 붙인다.
   - external sampling: `R̃_I(a*) − z·sqrt(Σ r̃²) > 0`이면 argmax 행동 a*를 라벨로 쓰고, 아니면 "없음"이다.
   - 전체 순회: 도달 확률당 regret이 `tol`보다 크면 argmax 행동, 아니면 "없음"이다.
2. **분할:** 라벨이 둘 이상 섞인 bucket을 라벨별로 나눈다. 자식 bucket은 regret과 평균을 새로 시작한다. 병합은 하지 않는다.
3. **추적 범위:** 이번 실행은 모든 정보집합을 추적한다(p = 1). **그래서 학습 중 메모리는 아직 줄지 않는다.** 결과로 얻은 bucket 표가 작다는 것만 보인다.

## 실행

```bash
python agents/adaptive_buckets/aswiss/export.py small DATA/small
g++ -O3 -std=c++17 agents/adaptive_buckets/aswiss/aswiss.cpp -o aswiss
./aswiss --tree DATA/small --buckets DATA/small/bucket_timestep.bin --mode es --iters 2560000 \
  --epoch0 10000 --split cum --z 2 --seed 1 --out RUN
python agents/adaptive_buckets/aswiss/eval.py DATA/small RUN
python agents/adaptive_buckets/aswiss/kmeans.py DATA/small RUN/final_bucket_map.bin DATA/small/bucket_kmeans.bin 0
python agents/adaptive_buckets/aswiss/plot.py RUNS_DIR out.png
```

- **추상화 없음(perfect recall):** `--buckets`에 항등 사상을 준다.
- **고정 추상화:** `--split none`을 준다.

`eval.py`는 checkpoint마다 정책 세 가지의 exact exploitability(NashConv/2, ante)를 `tree_best_response.py`로 계산한다.

- bucket 평균: 실제로 배포할 수 있는 정책
- 진짜 sequence-form 평균: 원 정보집합별 누적이며, 이론이 다루는 정책
- 현재 전략

**구현 검증:** C++ 전체 순회 perfect-recall CFR+를 `tiny`에서 1,000반복 돌린 exploitability가 Python 구현과 같은 0.00205로 나왔다.

## 결과 (`small`, seed 1, external sampling 2,550,000반복)

![결과](results_small_seed1.png)

| 방식 | 마지막 epoch 동안 bucket 수 | 평균 정책 | 현재 정책 | 실행 시간 |
|---|---:|---:|---:|---:|
| perfect recall | 51,114 | 0.067 | 0.134 | 738초 |
| 분할 (time step 40개에서 시작, z=2) | 5,988 (종료 후 9,856) | 0.233 (진짜 평균 0.199) | 0.308 | 63초 |
| 카드 k-means 고정 | 7,051 | 0.356 | 0.418 | 96초 |
| time step만 고정 | 40 | 0.916 | 0.917 | 15초 |

- **정체와 감소:**
  - time step만 쓰는 고정 추상화는 0.92 근처에서 정체한다. 추상화 오차만 남은 상태다.
  - 분할 실행은 계속 감소한다.
- **k-means와의 비교:**
  - 마지막 checkpoint에서 분할/k-means 비율은 평균 정책 0.65, 현재 정책 0.74다.
  - 이때 분할 쪽 bucket(5,988개)이 k-means(7,051개)보다 적다.
  - 중간 구간(7만–60만 반복)에서는 분할이 k-means보다 나빴다. 분할된 자식이 regret과 평균을 새로 시작하기 때문으로 보인다. 원인은 검증하지 않았다.
- **아직 끝나지 않음:** 분할은 멈추지 않았다. 마지막 epoch에서도 2,313개를 분할했다. 더 길게 돌렸을 때 perfect recall보다 작은 크기에서 멈추는지는 아직 모른다.
- **노이즈 없는 기준(전체 순회 + 분할, tol 1e-3):** 4,000반복에서 bucket 5,273개, 평균 정책 0.18. 정확한 라벨로도 분할이 많이 일어나므로, 분할 대부분이 노이즈 때문은 아닌 것으로 보인다. 전체 순회 perfect-recall CFR+는 2,000반복에서 0.00083이었다.

## H 추적 (2026-10-26)

H는 같은 bucket에 묶인 정보집합들의 regret이 서로 상쇄되는 양이다. 정의는 `H = L − C`이다.

- `L = Σ_I max(0, max_a R_I(a))`: 원 정보집합마다 계산한 값의 합
- `C`: 같은 식을 bucket 합에 대해 계산한 값

추적 방식은 다음과 같다.

- `--exact-every 100`이면 100반복마다 진단용 전체 순회를 한 번 한다. 이때 원 정보집합의 정확한 regret을 계산해 100배로 누적하므로, epoch 합을 시간 표본으로 근사한 값이다.
- 이 진단값은 분할 판정에 쓰지 않는다. 진단을 켠 분할 실행의 최종 bucket 사상은 끄고 돌린 실행과 바이트 단위로 같다.

반복당 값이다. 각 행은 그 epoch 동안 쓰던 분할 기준이다.

| 누적 반복 | 분할: 정확 H | 분할: 샘플 H | time step 고정: 정확 H | k-means 고정: 정확 H |
|---:|---:|---:|---:|---:|
| 10,000 | 6.26 | 7.72 | 6.26 | 2.37 |
| 150,000 | 3.76 | 4.15 | 6.80 | 3.26 |
| 630,000 | 2.91 | 3.03 | 6.91 | 3.34 |
| 1,270,000 | 2.24 | 2.30 | 6.92 | 3.42 |
| 2,550,000 | 1.37 | 1.40 | 6.94 | 3.39 |

- **분할:** 분할이 진행될수록 H가 줄어든다(6.26 → 1.37).
- **고정 추상화:** time step 고정과 k-means 고정은 H가 줄지 않는다. 이 둘이 정체하는 원인이 추상화 오차라는 점과 맞는다.
- **샘플 H의 편향:** 알고리즘이 보는 샘플 H는 정확 H보다 크다(위쪽 편향). epoch가 길어질수록 차이가 줄어든다. 예를 들어 k-means 첫 epoch는 5.02 대 2.37이고, 마지막은 3.44 대 3.39다.
- **누적 상계:** `exploitability(선형 평균) ≤ Σ_I max(0, max_a Σ_t t·r_t(I,a)) / (2 Σ_t t)`로 계산한다. 2,550,000반복에서 분할 0.82, k-means 1.84, time step 고정 3.48이다. 실제로 잰 값(0.20 / 0.36 / 0.92)보다 느슨하지만 순서는 같다.

## 라벨 판정 역치 비교 (2026-10-26)

라벨 판정에 쓰는 역치를 세 가지로 바꿔 비교했다. 모든 역치는 정보집합마다, 그 epoch의 샘플 regret 합 R̃(a*)에 대해 적용한다. a*는 R̃가 가장 큰 행동이다.

- `normal`(z=2): `R̃ > 2·√(Σ r̃²)`. 정규분포 근사라서 엄밀한 보장은 없다.
- `hoeffding`: `R̃ > Δ_I·√(2 n ln(1/δ_k))`. Azuma–Hoeffding 부등식에서 나온 역치다.
- `bernstein`: `R̃ > √(2 n V ln(2/δ_k)) + 7·(2Δ_I)·n·ln(2/δ_k)/(3(n−1))`. 경험적 Bernstein 부등식에서 나온 역치다.

기호는 다음과 같다.

- n: 그 epoch의 방문 횟수
- V: 샘플 regret의 표본분산
- Δ_I: 정보집합 아래 보수의 최대−최소. 1–32 ante 범위다.
- δ_k = 0.05/(k+1)²: epoch k의 허용 오판정 확률. epoch에 걸쳐 합이 유한하므로, 노이즈로 인한 잘못된 라벨은 거의 확실히 유한 번만 생긴다.

Bernstein 역치는 원래 독립 표본용이다. 마팅게일 표본에 쓰려면 엄밀하게는 마팅게일용 버전이 필요하다.

2,550,000반복 기준이다. 정책 평가는 마지막 epoch 동안 쓰던 분할 위에서 한 것이고, H는 그 epoch의 정확 H(반복당)다.

| 역치 | 마지막 epoch bucket 수 (종료 후) | 정확 H | 평균 정책 | 현재 정책 |
|---|---:|---:|---:|---:|
| normal z=2 | 5,988 (9,856) | 1.37 | 0.233 | 0.308 |
| bernstein δ=0.05 | 164 (220) | 3.44 | 0.499 | 0.502 |
| hoeffding δ=0.05 | 114 (154) | 4.29 | 0.701 | 0.730 |
| 카드 k-means 고정 | 7,051 | 3.39 | 0.356 | 0.418 |
| time step 고정 | 40 | 6.94 | 0.916 | 0.917 |

- **보수적인 역치일수록 분할이 늦다.** Hoeffding과 Bernstein은 최악 범위 Δ_I를 쓰므로 분할이 훨씬 늦다. 그래도 둘 다 마지막 epoch까지 분할이 계속됐다.
- **bucket당 상쇄 감소 효율.** Bernstein은 bucket 164개로 k-means 7,051개와 비슷한 정확 H(3.44 대 3.39)에 도달했다. 다만 exploitability는 k-means가 더 낮았다.
- **seed는 하나뿐이다.** δ와 epoch 길이에 대한 민감도는 아직 보지 않았다.

## 비교 조건과 남은 차이

- **평가:** 모든 방식은 같은 정확한 BR 평가기, 같은 seed, 같은 반복 수로 평가했다. LBR이나 Local BR-gap이 아니다.
- **seed:** 하나뿐이다. 신뢰구간은 없다.
- **k-means 기준선의 조건:**
  - bucket 수를 분할 실행의 최종 time step별 개수(10,129개)에 **사후적으로** 맞췄다.
  - 카드 특징(자기 rank 개수, 자기 공개 rank, 상대 공개 rank, 이전 street 베팅)의 서로 다른 행 수가 상한이라 7,051개에서 멈췄다.
  - 학습 데이터는 쓰지 않았다.
- **마지막 epoch 제외:** 마지막 1만 반복짜리 짧은 epoch는 모든 곡선에서 뺐다. 분할 직후 자식의 평균이 거의 비어 있어서 bucket 평균 값이 왜곡된다.
- **메모리 절감 아님:** 추적을 p = 1로 했으므로 학습 메모리 절감을 보인 결과가 아니다. 일부만 추적(p < 1)하는 버전은 아직 만들지 않았다.
