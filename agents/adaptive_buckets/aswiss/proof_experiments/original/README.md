# ASWISS 검토 세션 실험 기록 (2026-11-20 ~ 11-23)

모든 실험은 **공개 chance로 K개의 행렬게임(copy) 중 하나를 고르는 1단 게임**에서 돌렸다.
P1은 행 최대화, P2는 열 최소화이며 두 플레이어 모두 copy를 관측한다.
추상화는 copy들의 분할이다. 수렴 상태는 **LP로 구한 정확한 추상 균형**(scipy HiGHS)으로 대신했고,
균형이 유일하지 않은 경우는 제외했다(`chain_search.side_lp`의 좌표 범위 검사).
이 구조에서는 상한(Σ 양의 advantage)이 원게임 NashConv와 정확히 같다.
연구 저장소의 게임(Leduc, 7-stud 등)이 아니므로 일반화에는 주의가 필요하다.

환경: Python 3, numpy 2.5.3, scipy 1.18.1. 의존 관계: 대부분의 스크립트가 `chain_search.py` 또는 `search_pathology.py`를 import한다.

| 주장 | 스크립트 | 명령 | 결과 |
|---|---|---|---|
| 반례 A: NE가 버킷-상수인데 hidden regret이 선형 | `sim_tied.py`, `sim2.py`, `sim3.py` | `python3 sim3.py` | CFR+: P2 exploitability 0.0777, hidden/T 0.0772 (t=10^6) |
| 반례 B: 라벨 분할 후 NashConv 0.238→2.5 | `search_pathology.py` | `python3 search_pathology.py 0` (~5분) | A1=[[3,-3],[-3,2]], A2=[[2,-3],[-3,2]] |
| 반례 B의 상한 경로 (CFR+) | `bound_sequence.py` | `python3 bound_sequence.py` | 0.239 → 2.500 → 0.0006 (T=10^5) |
| ε-계열 (이득≈0인데 상승 2.5) | `eps_family.py` | `python3 eps_family.py` | ε=0.001: 0.00023 → 2.5 |
| 평균전략 라벨 vs BR 라벨 | `label_direction_test.py` | `python3 label_direction_test.py` (~4분) | 증가 27.3% vs 0% |
| 한쪽 추상화에서도 NashConv 증가 | `one_sided_search.py` | `python3 one_sided_search.py 0` (~3분) | 1860건 중 2건, 1.0→1.333 |
| 분할 경로 탐색(최대 이득 규칙) | `chain_search.py` | `python3 chain_search.py 0 4 1500` | 길이 3 경로만 발견 |
| ∇ 부호, Γ<0 비율 | `nabla_sign.py` | `python3 nabla_sign.py 5 3 2500` | ∇<0: 357/2576 (13.9%), Γ<0: 253 (9.8%) |
| 공동 분할(각자 라벨) | `paired_split.py` | `python3 paired_split.py 5 3 2500` | 증가 9.3% → 3.6% |
| 공개 정렬 공동 분할 | `public_aligned.py` | `python3 public_aligned.py 5 3 1500` | 증가 76/1997 (3.8%) |
| 균형 분산도와 P(Γ<0) | `consistency_test.py` | `python3 consistency_test.py 7 4 40 40` | Spearman −0.463 (p=0.0027), E[Γ]>0 40/40 |
| 연속 증가·회복·큰 단계 길이 | `double_split2.py`, `analyze_double.py` | `python3 double_split2.py 101 ds_a.jsonl 1500` 와 `... 202 ds_b.jsonl 1500` 병렬 후 `python3 analyze_double.py ds_a.jsonl ds_b.jsonl` | 아래 표 |

### 연속 증가 실험 최종 집계 (`ds_a.jsonl`, `ds_b.jsonl`, 경로 12,282개)

- 상한 증가 3,900건
- 다음 분할로 완전 회복(B_{i+1} ≤ B_{i-1}) 3,818건 (97.9%), 일부만 회복 82건 (2.1%)
- 두 번 연속 증가 16건 (증가의 0.41%)
- 적응형 큰 단계 길이: 대부분 1–2, 길이 3이 80번, 길이 4가 2번, 최대 4

대화 중에 보고한 "10,690개, 98.0%, 13건"은 작업이 끝나기 전의 중간 집계였다. 위 수치가 최종값이다.

생성기: `rand`(2×2, 성분 −3..3, K∈{4,5}), `near`(같은 기저 행렬 + 드문 ±1/±2 섭동, K∈{3,4,5}),
`rand3`(3×3, K=3). 출발점: 모두 묶음(`tied`)과 무작위 분할(`random`)을 번갈아 사용.

`whole_run.py`는 실행이 중단되어 결과가 없다. `double_split.py`는 소규모 예비 실행(경로 42개)이다.
