# prior–posterior 분해: 관측 색인과 후회 bucket을 번갈아 나누기 (설계 초안, 2026-11-10)

상태: **설계 초안이다. 구현과 실험은 하지 않았다.** 사용자가 제시한 구조를 정식화하고, 이전 실험 기록과 기존 정리에 연결한다.

## 1. 동기 (사용자 제안 요약)

- 원안 ASWISS는 전략의 후회만 보고 bucket을 나누고, 어느 bucket에 넣을지도 전략으로 정한다.
  - 그래서 분할이 늘수록 판단 시간이 늘고(전이 비용이 분할 단계마다 곱으로 증가, `aswiss/compare/README_KO.md`), 메모리도 는다.
- 이를 풀려면 prior 정보가 필요하다. 지금은 atlas(k-means codebook)로 가속하지만, atlas 자체는 나누지 않는다.
- 제안: **prior 쪽에서는 atlas를 나누고, posterior 쪽에서는 bucket을 나눈다.** 둘을 번갈아 나눈다.
  - 둘 다 가장 거칠게 시작한다.
  - posterior를 고정하고 prior를 나눠 성능을 최대한 올린다.
  - 더 나아지지 않으면 prior를 고정하고 posterior를 나눠 성능을 올린다.
  - 이를 반복한다.
- 두 구조의 출발점은 다르다.
  - prior는 모든 time step의 정보를 포함한다.
  - posterior는 같은 constraint끼리만 묶는다.
- **전략은 posterior에만 저장한다. prior는 어느 bucket인지를 빨리 찾는 포인터일 뿐이다.**

## 2. 이전의 prior 쪽 분할 실험 (기록에 있는 것)

| 실험 | 무엇을 나눴나 | 결과 (기록) | 문서 |
|---|---|---|---|
| 가중 cluster 성장 (2026-07-29) | power atlas에 새 중심을 더함(최대 one-step regret 상태) | 100k에서 휴리스틱에 지고, 새 cluster가 모두 7구에만 생김 | `DESIGN_LINEAGE.md` |
| DT / DT validated | 34D 관측 요약 위 회귀 트리를 온라인으로 분할(순간 regret SSE, centered label, 검증 gain) | 분할 기준이 SSE 계열이다. 결과는 이번에 다시 확인하지 않았다 | `agents/cpp_mccfr/dynamic_tree_validated/` |
| reward-audited local split | k-means 칸 안을 이진 트리로 나눔(행동별 rollout audit) | 7-stud 우월성 미확인(명세) | `agents/cpp_mccfr/ONLINE_LOCAL_BUCKET_SPLIT.md` |
| **TARA** (2026-10-08/09, 5 seed) | 공용 노드(베팅 요약)마다 카드 특징 22개의 회귀 트리를 구간마다 새로 적합(regret 표본의 제곱오차), 트리 숲 | 1M에서 co-trained k-means를 못 이김. current LBR 판단 불가(+0.43 ± 0.75), **평균 정책 붕괴**(+2.74), 같은 특징의 고정 k-means 64가 127k에서 더 나음 | `agents/cpp_mccfr/tara/README.md` |

TARA 기록에 남은 실패 원인 후보(대부분 검증되지 않은 가설):
1. **분할 기준이 후회의 제곱오차 회귀였다.** 수렴에 남는 양(라벨 항등식의 H)과 다른 목표다. random 분할보다 current LBR에서 나은 점이 없었다.
2. **구간마다 새 트리를 만들고 표를 0에서 다시 쌓았다.** 여러 트리의 평균을 합칠 때 카드를 구분하지 못하는 첫 트리의 몫이 커져 평균 정책이 무너졌다(가설).
3. 트리당 leaf 2,048 고정, 노드당 저장 표본 4,096행, 잡음이 큰 표본 regret 라벨.
4. 베팅은 공유 요약(약 290개 노드)이었다. 지금의 bucket 이력과 다르다.

**지금 다른 점:** 분할 기준이 라벨 항등식(H)으로 정해져 있고, 교차 추정으로 잡음 치우침을 줄이는 방법도 있다(ρ̂, bucket-regret). 그리고 아래 성질 P1로 표 재시작 문제를 피할 수 있다.

## 3. 정식화

### 3.1 대상

- **context k(I):** 같은 constraint를 가진 결정들. 예를 들어 (street, 합법 행동집합), 또는 bucket 이력 키다. posterior bucket은 한 context 안에서만 묶는다.
- **prior 분할(관측 색인) 𝒞:** 관측 특징 φ(I)(카드 등)의 분할이다. 칸 c(I)는 **모든 time step에서 공유된다.** 지금의 atlas(K개 Voronoi 칸)가 출발점이다.
- **원자 α = (k, c):** context와 prior 칸의 쌍이다(7-stud 원자 = 카드 군집 × 베팅 요약 × 합법 행동과 같은 구조).
- **posterior 분할 𝒫:** 각 context에서 원자들을 묶은 bucket이다. 포인터 π(k, c) = bucket이다. 전략(후회표, 평균표)은 bucket에만 있다.
- **조회:** I → (k(I), c(I)) → π(k, c). prior는 특징 검사의 트리를 내려가고 posterior는 표를 한 번 찾는다. 분할 계보를 되짚는 전이가 없으므로 분할이 늘어도 조회 비용이 곱으로 늘지 않는다.

### 3.2 분해 항등식 (정리 3의 항등식을 두 층으로)

구간 W의 누적 후회 R과 ψ(x) = max(0, max_a x_a)에 대해 다음이 성립한다.

$$\sum_I\psi(R_I)=\underbrace{\sum_B\psi(R_B)}_{C:\ \text{bucket이 느끼는 후회}}+\underbrace{\sum_B\Big[\sum_{\alpha\in B}\psi(R_\alpha)-\psi(R_B)\Big]}_{H_{\text{post}}}+\underbrace{\sum_\alpha\Big[\sum_{I\in\alpha}\psi(R_I)-\psi(R_\alpha)\Big]}_{H_{\text{prior}}}.$$

- 라벨 항등식(보조정리 2)이 두 층에 모두 적용된다.
  - $H_{\text{post}}=\sum_B\min_\ell\sum_{\alpha\in B}e_\alpha(\ell)$: bucket 안 원자들의 방향 불일치
  - $H_{\text{prior}}=\sum_\alpha\min_\ell\sum_{I\in\alpha}e_I(\ell)$: 원자 안 정보집합들의 방향 불일치
- NashConv(평균) ≤ (C + H_post + H_prior)/|W| (보조정리 1).
- 지금의 7-stud ASWISS는 H_post만 줄이고 H_prior는 남긴다(정리 3의 잔여항). k-means 고정 추상화는 둘 다 남긴다.

### 3.3 두 가지 수

**posterior 분할 (prior 고정): 지금의 ASWISS 그대로.**
- 한 bucket 안 원자들의 라벨이 갈리면 라벨별로 나눈다. 자식은 새 전략이다.
- H_post를 줄인다.

**prior 분할 (posterior 고정): 칸 c를 특징 검사로 c′, c″로 나눈다.**
- **이득 = H_prior의 감소량:** $\mathrm{gain}(c\to c',c'')=\sum_k\big[\psi(R_{(k,c')})+\psi(R_{(k,c'')})-\psi(R_{(k,c)})\big]\ge0$이다. 칸이 모든 context에 걸쳐 있으므로 context에 대해 더한다.
  - 참값은 언제나 0 이상이다(ψ의 준가법성). 그래서 "이득이 있나"가 아니라 "잡음보다 큰가"를 봐야 한다.
  - 표본 후회로는 위로 치우치므로 **교차 추정**으로 판정한다. 한 절반으로 분할 검사를 고르고 다른 절반으로 이득을 잰다(ρ̂와 같은 방법).
  - TARA의 제곱오차와 달리 이 기준은 수렴에 남는 양 자체다.
- **P1 (분할 순간 정책 불변):** 새 칸 c′, c″는 모든 context에서 부모의 포인터를 물려받는다(π(k, c′) = π(k, c″) = π(k, c)). 그러면 분할 순간의 정책도 표도 그대로다. 표를 다시 쌓지 않으므로 TARA의 평균 붕괴 원인(2번)이 생기지 않는다.
- **P2 (prior 분할만으로는 총합이 줄지 않는다):** 포인터를 그대로 두면 C는 그대로이고, H_prior에서 줄어든 만큼 H_post가 늘어난다. 상쇄가 "원자 안"에서 "bucket 안의 원자 사이"로 옮겨 가 posterior가 볼 수 있게 될 뿐이다.
- **그래서 prior 단계에서 성능을 올리는 것은 포인터 재지정이다.**
  - 새 원자 (k, c′)를 같은 context의 기존 bucket 중 더 맞는 것으로 옮긴다. 새 전략은 만들지 않는다.
  - 판정은 한 단계 개선량으로 한다. bucket b에서 b′로 옮길 때의 이득은 $\langle\sigma_{b'},R_{(k,c')}\rangle$다(원자의 누적 후회와 후보 전략의 내적; $r(I,a)=v(I,a)-v(I,\sigma_b)$이므로 $v(I,\sigma_{b'})-v(I,\sigma_b)=\langle\sigma_{b'},r(I)\rangle$). 교차 추정으로 유의할 때만 옮긴다.
  - 이것은 원안의 전이("방향이 맞는 자식으로 보낸다")를 원자 단위로, 그리고 이미 있는 bucket들 사이에서 하는 것이다.

### 3.4 교대 절차

1. **시작:** prior는 가장 거친 관측 분할(칸 하나, 또는 작은 K의 codebook), posterior는 context마다 bucket 하나.
2. **prior 단계 (posterior 고정):** epoch마다 교차 추정 이득이 유의한 칸을 나누고, 새 원자의 포인터를 재지정한다. 이득이 유의한 분할이 더 없으면 끝낸다.
3. **posterior 단계 (prior 고정):** epoch마다 원자 라벨로 bucket을 나눈다(지금의 ASWISS). 분할이 더 없으면 끝낸다.
4. 2–3을 반복한다.

- **전환 기준 후보:** "개선이 멈춤"을 성능 지표 대신 항등식의 항으로 잰다.
  - prior 단계 종료: 교차 추정한 H_prior의 감소가 멈춤(유의한 분할 없음)
  - posterior 단계 종료: H_post 쪽 라벨 불일치가 없음
  - 작은 게임에서는 세 항을 정확히 잴 수 있다. 큰 게임에서는 교차 추정한다.
- 두 수는 서로 다른 항을 줄이고 둘 다 세분만 하므로 동시에 해도 된다. 교대는 어느 쪽이 무엇을 줄였는지 나눠 보기 쉽고, 한쪽의 잡음이 다른 쪽 판정을 흔들지 않게 한다.

### 3.5 수렴에 필요한 조건 [가설, 증명 안 함]

- prior가 결국 정보집합을 구분할 수 있을 만큼 세분될 수 있어야 한다. 즉 칸 + context가 서로 다른 방향의 정보집합을 언젠가 가를 수 있어야 한다.
  - k-means Voronoi 칸은 그대로는 정확한 관측까지 내려가지 못한다. 그래서 prior는 **원시 관측 특징(실제 카드)에 대한 검사의 트리**로 두고, 첫 층만 codebook으로 시작하는 형태를 권한다.
- 그 조건과 두 판정이 계속 돌면, H_prior와 H_post가 모두 0으로 갈 수 있어 정리 1의 형태(원 게임 수렴)를 기대할 수 있다.
- prior를 일정 해상도에서 멈추면 보장은 정리 3(그 해상도까지)이다.
- 포인터 재지정(3.3)은 기존 정리 밖이다. 재지정은 분할이 아니라 이동이라 Lemma 6(분할 유한성)이 그대로 적용되지 않는다. 진동하지 않게 하려면 재지정 횟수에 상한을 두거나 유의성 문턱을 둔다.

## 4. 실험 계획 (제안)

**1단계: mini-stud (정확한 exploitability와 세 항을 정확히 잼)**
- prior 특징: 카드 관측(자기 카드, 자기 공개, 상대 공개의 숫자별 개수; `export.py`의 features)
- prior 출발: 칸 하나, 그리고 k-means K=8
- posterior: 시점 context
- 비교군:
  1. posterior만 (prior 고정 = 지금의 원자 ASWISS)
  2. prior–posterior 교대
  3. 동시
  4. 정보집합 단위 ASWISS(표; 위쪽 기준)
  5. 고정 k-means(같은 bucket 수)
  6. 순수 ASWISS(전이; `aswiss/pure_aswiss.cpp`, 24시간 실행 중)
- 잴 것:
  - exploitability 대 반복과 학습 시간
  - C, H_post, H_prior의 궤적: 어느 단계가 어느 항을 줄였는지
  - 조회 비용과 메모리: prior 트리 깊이, 원자 수

**2단계: 7-stud**
- prior: power 특징 + 원시 카드 특징의 트리, codebook K(예: 256)로 시작
- posterior: bucket 이력
- 기준선: 지금의 `hist_bbf`(codebook 바닥)와 k-means 1024
- 평가: LBR, 리그, bucket-regret

## 5. 이름 제안

atlas를 바꿀 이름 후보:
- **관측 색인(observation index)**: 포인터라는 역할이 드러난다.
- **prior 분할 / posterior 분할**: 두 층의 대비가 드러난다.
- **카드 트리(card tree)**: 7-stud에 한정된 이름이다.

## 6. 정할 것

- 3.3의 해석: "posterior를 고정하고 prior를 나눠 성능을 올린다"를 **prior 분할 + 기존 bucket으로의 포인터 재지정**으로 읽었다. P2 때문에 포인터를 바꾸지 않는 prior 분할은 성능을 바꾸지 못한다.
- 포인터 재지정의 기준: 평균 전략의 내적(위)으로 할지, 원안처럼 분할 사건의 라벨로 할지.
