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

## 비교 조건과 남은 차이

- **평가:** 모든 방식은 같은 정확한 BR 평가기, 같은 seed, 같은 반복 수로 평가했다. LBR이나 Local BR-gap이 아니다.
- **seed:** 하나뿐이다. 신뢰구간은 없다.
- **k-means 기준선의 조건:**
  - bucket 수를 분할 실행의 최종 time step별 개수(10,129개)에 **사후적으로** 맞췄다.
  - 카드 특징(자기 rank 개수, 자기 공개 rank, 상대 공개 rank, 이전 street 베팅)의 서로 다른 행 수가 상한이라 7,051개에서 멈췄다.
  - 학습 데이터는 쓰지 않았다.
- **마지막 epoch 제외:** 마지막 1만 반복짜리 짧은 epoch는 모든 곡선에서 뺐다. 분할 직후 자식의 평균이 거의 비어 있어서 bucket 평균 값이 왜곡된다.
- **메모리 절감 아님:** 추적을 p = 1로 했으므로 학습 메모리 절감을 보인 결과가 아니다. 일부만 추적(p < 1)하는 버전은 아직 만들지 않았다.
