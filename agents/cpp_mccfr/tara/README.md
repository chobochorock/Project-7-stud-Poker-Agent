# TARA: 트리 누적 regret abstraction (7-stud 이식)

사용자 설계 "트리 누적 regret abstraction (가칭 TARA)"를 C++ 7-stud v3에 옮긴 계열이다.
사용자는 Leduc 예제(`tara_leduc.zip`)로 정확 exploitability 실험을 이미 했다. 그래서 여기서는 실제 7-stud에서
**함께 학습하는 hard256 k-means arm**과 Local BR-gap / Policy-LBR / 메모리로 비교한다.
Local BR-gap과 LBR은 exploitability가 아니다.

공유 소스(`stud_mccfr.cpp`, `hand_recall.cpp`, `dynamic_tree_validated/*`, `abstraction_lbr/evaluate.hpp` 등)는
수정하지 않고 include만 한다.

> **2026-10-13 용량 정리 (사용자 지시).** `data/tara/` 아래의 큰 파일 722개(20.4 GB)를 빼냈다.
> - 빼낸 종류: `checkpoint.bin`, `models/model_*.bin`, `buffer_*.bin`, `per_hand.csv`, `probes.csv`.
>   목록은 `data/tara/MOVED_OUT_20261013_manifest.csv`에 있다.
> - 남긴 것: 모델 하나(`main_1m_20261008/tara_seed222/models/model_1000000.bin`)와 그 forest·atlas,
>   `review_*` 보고서, `lbr/` 결과, `panel.csv`, `forests.csv`, `fits.jsonl`, 설정·로그·소스 사본.
> - 그래서 아래의 재개(`--resume`), `report.py`, `paired.py`, 중간 snapshot의 LBR 명령은 이 데이터로 다시 돌릴 수 없다.
>   B2 적합에 쓴 `main_127k_20261008/tara_seed*/buffer_1.bin`도 없다. 필요하면 같은 명령으로 다시 학습해야 한다.
> - 아래 결과 표와 수치는 정리 전에 만든 것이며 바뀌지 않았다.

## 알고리즘 (구현된 것)

- **학습기:** external-sampling MCCFR. 1 hand = 새 5구 deal 1개 + 모델별 순회 2회(좌석마다 1회).
  k-means arm과 같은 deal을 같은 순서로 쓴다(`root_for(seed ^ hand·0xa0761d6478bd642f)`).
- **regret 키:** 구간마다 트리 하나(= 공용 노드별 작은 회귀 트리들의 forest)를 만들고, 모든 트리를 계속 유지한다.
  정보집합 I의 트리 j 버킷은 (공용 노드(I), leaf_j(φ(I)))다. 베팅은 공용 노드 키로만 나뉘고 카드만 트리로 병합된다.
- **구간:** doubling N_t = N0·2^(t−1)(기본, N0 = 1000) 또는 고정 N_t = N0.
  경계는 1k, 3k, 7k, …, 255k, 511k이므로 1M에서는 트리 10개를 쓴다. 트리 1은 공용 노드당 leaf 1개(카드 무시)다.
- **갱신 (RM+):** 구간 t에서 traverser 노드의 표본 regret r로 트리마다, 행동마다 매번 clip한다.
  - 이전 트리: R_j ← max(0, R_j + 2^(t−1)·r)
  - 현재 트리: R_t ← max(0, R_t + 2^t·r)
  - 정책: σ ∝ Σ_j R_j[b_j(I)], 합이 0이면 균일
  - 한 순회 안에서 같은 버킷 조합이 다시 나오면 첫 정책을 고정해 쓴다(DT 계열과 같은 규칙).
- **가중치 옵션 (`weighting`):**
  - `pow2`: 기본
  - `halving`: 경계마다 R(과 `same_as_regret`이면 S)을 반으로 줄이고 0.5/1을 쓴다.
    self-test에서 pow2와 정책이 **비트 단위로 같다**(max |dp| = 0).
  - `uniform`: B4
  - `index`: 트리 j에 j·r
  - `index_frozen`: 현재 트리만 t·r
  - `prev_scope=last`: 직전 트리만 계속 쌓고 그 이전 트리는 동결한다.
  - `latest_only`: B3. 경계에서 이전 트리를 버린다.
- **평균전략:** 상대(표본) 노드에서 가중치가 양수인 모든 트리에 S_j += w_avg·σ를 더한다.
  w_avg는 `linear`(hand 번호, 기본) 또는 `same_as_regret`이다. 출력은 세 가지를 모두 기록한다.
  - current: RM(Σ R_j)
  - average(sum): Σ_j S_j (사양)
  - average(newest): 최신 트리의 S만. 최신 트리가 자기 구간의 절반을 채우기 전에는 직전 트리를 쓴다
    (Leduc 예제 규칙). 경계 snapshot과 LBR 모델에서는 항상 최신 트리다.
  - 평균 질량이 0인 버킷은 기본(`avg_fallback=current`)으로 현재 RM(Σ R_j)을 낸다. k-means arm의
    `average_strategy`와 같은 규칙이다(리뷰 후 수정. 이전에는 균일).
- **다음 트리 데이터:** 구간 t의 traverser 방문마다 (φ, 가중치 없는 표본 regret r)을 공용 노드별 균일
  reservoir(기본 4096행)에 넣는다. 공용 노드별로 방문 수와 regret 질량 Σ‖r‖₂도 센다. 경계에서
  `buffer_t.bin`을 쓰고 비운다. 적합은 공용 노드 키 순으로 하므로 재개 여부와 무관하게 결정적이다.
- **적합 (`fit.py`):**
  - 공용 노드마다 sklearn `DecisionTreeRegressor(criterion=squared_error, max_leaf_nodes=k, min_samples_leaf=50)`를
    쓴다. multi-output으로, 합법 행동의 r을 예측한다.
  - leaf 예산 L(기본 2048)을 regret 질량 비례 D'Hondt로 나눈다(공용 노드당 최소 1, 최대 행수/50).
  - 트리가 배정량을 실현하지 못하면(서로 다른 행이 적은 경우) 실현값으로 상한을 낮추고 남은 leaf를 다시 나눈다
    (최대 6회). 리뷰 전에는 약 7.7%가 버려졌다. 지금은 예산 2048을 다 쓰고, 분할되는 공용 노드도 58개에서 96개로 늘었다(seed222 buffer_1 기준).
  - 분기 임계값은 float32 내림(`floor32`)으로 저장해, C++의 float32 `x <= t` 경로가 sklearn과 같다(단위 시험).
  - 새 트리의 R = S = 0이다. 분할에 쓴 데이터(구간 t)와 regret을 쌓는 데이터(구간 t+1 이후)는 겹치지 않는다.
- **카드 특성 φ (22개):** power vector 0–17(k-means atlas와 같은 입력), 균일 상대에 대한 MC equity(64 표본,
  관측 키 해시로 시드), 죽은 flush/straight/pair 카드 수(상대 공개 카드 + 자신의 버린 카드).

## 공용 노드 (`key_mode`)

| 모드 | 정의 | 성질 |
|---|---|---|
| `summary` (기본) | k-means arm과 같은 베팅 요약 + 좌석. 카드 cluster만 뺀다 | 약 290개다. 두 arm의 베팅 분할이 같고 카드 추상화만 다르다 |
| `exact` | 좌석, 스트리트, 합법 행동, 정확한 베팅 이력 | 사양 그대로다. 시연만 한다(아래) |

## 대조군과 모드

| 이름 (`configs/`) | mode | 무엇을 가르나 |
|---|---|---|
| k-means hard256 (co-trained) | — | 프로젝트 기준선. **signed regret + 단순 평균**. 같은 seed의 DT 실행과 비트 단위로 같다(시험) |
| `tara` | tara | 주 실험 |
| `b0_kmeans_sched` (B0-sched) | kmeans_sched | TARA와 같은 일정·가중치·트리별 RM+·합 평균. 다만 **모든 트리가 k-means arm의 파티션**이다. TARA 대 B0-sched = 학습된 파티션의 효과 |
| `b0_kmeans_partition` (B0-single) | kmeans_partition | k-means 파티션 하나에 plain RM+ + 선형 평균. k-means arm 대 B0-single = 갱신 규칙의 효과 |
| `b3_latest` | tara | B3: 누적 대 최신 트리만 |
| `b4_uniform` | tara | B4: 2^t 가중치 대 균일 |
| `b5_fixed` | tara | B5: doubling 대 고정 구간(18,143 hands, 127k까지 사용 트리 7개로 맞춤) |
| `b7_prev_last` | tara | 이전 트리 전부 대 직전 트리만 |
| `index` | tara | Leduc 예제에서 가장 좋았던 j·r 가중치 |
| `random` | tara | 같은 행·leaf 배분·일정에서 분할만 무작위. 학습된 분할이 의미가 있는가(f-RCFR식 대조) |
| `b2_k8`, `b2_k64` | fixed | B2: 같은 22특성의 스트리트별 표준화 k-means, 처음부터 고정. 같은 seed TARA의 `buffer_1.bin`으로 적합 |
| B1 완전 recall | — | **미실행.** 7-stud에서는 표가 너무 크다(`HAND_RECALL.md`: 상태 한도 5M에 2,439 hands에서 도달) |
| B6 Λ 온라인 분할 | — | **미실행.** 기존 `shadow_split` 코드가 해당한다(다른 코드, 비대응 참고) |
| `avg_same_as_regret` | tara | 설정만 있고 **미실행** |

B2–B5, prev_last, index, random, B0-single은 **127k에서만** 5 seed로 돌렸다. 1M은 TARA와 B0-sched뿐이다.

- **B2 크기:**
  - k는 대체된 구 코드 런(트리당 ~1,900 leaf, 127k 전체 ~11,600)을 기준으로 정했다(buffer_1 기준 셀 1,741 / 9,955).
  - 현재 코드에서 127k의 실현 버킷은 다음과 같다.

    | arm | 버킷 |
    |---|---|
    | TARA | 트리당 2,052–2,064 leaf, 합계 12,640 |
    | B2 k=8 | 2,315 |
    | B2 k=64 | 17,517 (TARA의 1.39배) |

  - B2는 `main_127k_20261008/tara_seed*/buffer_1.bin`으로 적합했다.
    - 이 파일은 `main_1m_20261008`의 buffer_1과 sha256이 같다(검증 workflow).
    - 내용은 카드 무시 트리 1 아래 첫 1,000 hands의 방문 행이다(공용 노드당 4,096행 상한, 방문 가중 없음).
- **B0 kind-2 forest:** 버킷 = (summary 공용 키, atlas `power_cluster`)이고, k-means arm의 키와 1:1이다
  (self-test: 1,158개 결정, 950 버킷).
- **남는 차이:** B0-sched와 k-means arm 사이에는 갱신 규칙(RM+ 대 signed), 평균 가중치(선형 대 단순),
  트리 수와 경계 할인이 남는다. 파티션은 같다.
- **snapshot:** 모든 모드가 doubling 지점(1k, 3k, …)과 끝에서 `models/model_<hand>.bin`을 저장한다.
  그래서 B0/B2도 LBR 곡선이 있다.

## 실행

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m unittest agents.cpp_mccfr.tara.test_tara -v
# 한 런 (설정 파일 + 명령줄 덮어쓰기). 경계마다 C++가 멈추고 Python이 다음 트리를 적합한 뒤 이어서 학습한다.
python -m agents.cpp_mccfr.tara.run train --run agents/cpp_mccfr/data/tara/<new> --config agents/cpp_mccfr/tara/configs/tara.json --seed 222 --hands 1000000
python -m agents.cpp_mccfr.tara.run train --run agents/cpp_mccfr/data/tara/<same> --resume [--hands 2000000]
# 여러 arm × seed 병렬 실행 (메모리/커밋 여유를 보고 시작), 이어서 snapshot마다 Policy-LBR
python -m agents.cpp_mccfr.tara.campaign --name main_1m_20261008 --arms tara b0_kmeans_sched --seeds 222 111 333 444 555 --hands 1000000 --jobs 10
python -m agents.cpp_mccfr.tara.campaign --name main_1m_20261008 --arms tara b0_kmeans_sched --seeds 222 111 333 444 555 --hands 1000000 --lbr sum latest --all-models
python -m agents.cpp_mccfr.tara.report --campaign agents/cpp_mccfr/data/tara/main_1m_20261008 --output agents/cpp_mccfr/data/tara/<new review>
```

- **옵션:** 모든 옵션은 `configs/*.json`이나 명령줄로 바꾼다(`run.py train --help`).
- **재개 (`--resume`):**
  - 저장된 옵션을 쓰고, `--hands` 연장만 허용한다.
  - 최종 hand가 경계이면 다음 트리까지 준비해 두므로, 연장해도 일정이 이어진다.
  - 중간 정지·재개·연장 결과는 새로 돌린 것과 CSV/모델이 비트 단위로 같다(시험).
  - 경계 hand에서 정지 요청이 와도 그 구간을 마친 것으로 기록된다(리뷰 후 수정).
- **메모리:**
  - numpy의 BLAS는 프로세스마다 하드웨어 스레드 수만큼 버퍼를 잡아 약 0.5 GiB를 커밋한다(실사용 33 MB).
  - `campaign.py`는 자식에게 `OPENBLAS_NUM_THREADS=1`을 넘겨 이를 19 MB로 줄이고, 물리 메모리와 커밋 여유를
    모두 보고 작업을 시작한다. 직접 여러 개를 돌릴 때도 같은 환경 변수를 쓴다.
  - C++ 쪽 peak RSS는 1M에서 약 200 MiB, 127k에서 약 180 MiB였다(reservoir가 차면 거의 포화).
- **출력 (런 폴더):**
  - `per_hand.csv`: 판마다 첫 5구 결정의 Local BR-gap. 32 particle, eval seed 307. plug-in, crossfit, action SE를 담는다.
  - `panel.csv`: 10k마다 128 root. tag는 `regular`, `before`/`at`(경계 직전과 경계), `after`(+100)다.
  - `forests.csv`: 트리별 leaf, 엔트리, 구조 단위, Σ R⁺, Σ S
  - `probes.csv`: 아래 안정성 측정
  - 기타: `fits.jsonl`, `models/model_<hand>.bin`, `config.json`(소스 SHA256), `options.json`, `invocations.jsonl`
- **Local BR-gap 추정치:**
  - plug-in max_a Q̂ − σ·Q̂는 위로 치우친다.
  - crossfit은 아래로 치우친다(16 particle로 argmax를 고르고 나머지 16으로 평가).
  - plug-in − crossfit 차이는 arm마다 0.5–0.9 ante로 다르다. arm 간 차이(0.03–0.10)보다 약 10배 크다.
  - 그래서 Local BR-gap 결론은 **두 추정치가 같은 방향일 때만** 확정적으로 쓴다.
  - 처음 세운 "RM+ 정책이 희소해서 plug-in 편향이 작다"는 설명은 **철회했다**(검증 workflow).
    - TARA의 action SE가 오히려 크다.
    - 같은 RM+인 B0-single의 차이가 k-means보다 크다.
    - 1M에서는 plug-in이 TARA를 더 불리하게 본다.
  - `report.py`는 두 추정치와, 같은 root·particle·rollout seed에서의 paired 차이를 함께 보고한다.
- **안정성:**
  - 경계 B−100에서 합 평균 self-play로 상태 200 episode를 뽑아 (deal seed, 행동 prefix)로 저장한다.
  - 같은 상태의 정책을 네 번 기록한다: B−100(`before`), 트리 추가 전 B(`at`), 추가 직후(`added`), B+100(`after`).
  - 새 트리는 0에서 시작하므로 at = added가 세 출력 모두 비트 단위로 같아야 한다(시험). B3는 정의상 예외다.
  - 대조 창(before→at, 트리 추가 없음)과 경계 창(at→after)을 같은 경계 집합에서 비교한다.
  - 지표는 합법 행동 위의 TV, KL(ε=1e-4), support 변화율이다. RM+의 정확한 0 때문에 ε=1e-9 KL은 TV와 반대 결론을 냈다(리뷰).
- **메모리 보고:**
  - 사양 식은 Σ_j(leaf_j × 합법 행동) × 2 + 구조 × 5다. 구조 단위는 트리 노드이고, centroid/atlas forest에서는
    centroid와 셀 항목이다. atlas 자체는 k-means arm과 공유하므로 세지 않는다.
  - 버킷 수 Σ_j leaf_j와 k-means 상태 수, 다음 트리 버퍼(`buffer_rows` × 120 bytes), peak RSS를 따로 기록한다.
- **LBR:**
  - deal seed 2026100721/22, 250 pair × 2, 64 particle이다. DT 1M LBR과 같아서, seed222 k-means baseline 값이
    DT 1M 결과와 정확히 같은지로 평가 경로를 확인한다.
  - 출력 폴더는 `lbr/<run>_<avg>_s<seeds>_h<hands>`다.

## 사양과 다른 점과 그 이유

1. **정확 exploitability, B1, 정확 평균 표:** 7-stud에서는 계산할 수 없다. 사용자 지시대로 LBR / Local BR-gap으로 대신한다.
2. **공용 노드 = 베팅 요약:** 정확한 베팅 이력의 공용 노드는 7구에서 17,371,168개다.
   - exact 데모(`data/tara/exact_key_demo_20261008`, seed222, reservoir 64)에서는 1,000 hands 만에 297,649개가 나왔다.
   - 3,000 hands에서는 551,147개였고, 노드당 표본이 100행 미만이라 **분할된 노드가 0개**였다.
   - 3k hands의 Local BR-gap(2,001–3,000 hands 평균, current)은 다음과 같다.

     | | exact 키 | summary 키 TARA (`main_1m` seed222) | k-means |
     |---|---|---|---|
     | plug-in | 18.3 | 1.35 | 2.69 |
     | crossfit | 3.32 | 0.19 | 0.20 |

   - leaf는 295만 개, 모델은 584 MB였다. 그래서 기본값은 summary다.
   - 이 데모는 리뷰 전 코드와 reservoir 64로 돌렸다.
3. **정보집합별 집계 대신 방문 행 reservoir:** 7-stud의 정확한 정보집합은 거의 반복되지 않는다(방문 1회).
   그래서 "정보집합별 평균 regret, 가중치 = 방문 수"는 "방문별 행, 가중치 1"과 같다. 공용 노드당 4096행으로
   제한한다(균일 표본). leaf 배분의 질량은 Σ‖r‖₂다.
4. **평가 출력:** current, 합 평균, 최신 트리 평균을 각각 평가한다. 예제에서 합 평균이 나빴기 때문이다.

## 실행 이력

- **`data/tara/smoke_seed222_20261008`, `smoke4000b_seed222_20261008`:** 초기 구현 점검용이다(resume·연장 비트 일치 확인).
- **`data/tara/main_127k_20261008/` (구 코드, 대체됨):**
  - TARA ×5와 B3 ×3을 127k까지 학습하고, LBR(deal seed 01/02)까지 했다.
  - 리뷰 후 D'Hondt 재배분과 버퍼 정렬로 **트리 2부터 학습 경로가 바뀌었다.** 평균 fallback, 최신 트리 평균 규칙, B0도 추가했다.
  - 그래서 전부 새 코드로 다시 돌렸다.
  - 구 코드에서 보였던 "TARA current LBR 1.44 대 k-means 2.41"은 **재현되지 않았다.** 새 코드에서는 2.45 대 2.58이고,
    paired −0.12 ± 0.52, p = 0.62다.
- **`data/tara/exact_key_demo_20261008/`:** exact 키 시연이다. 메모리 때문에 3k hands에서 중단했다.
- **`data/tara/main_1m_20261008/`:** TARA, B0-sched × seed 222/111/333/444/555, 1M hands
- **`data/tara/ablation_127k_20261008/`:** B3, B4, B5, prev_last, index, random, B0-single, B2 k8/k64 × 5 seeds, 127k
- **실행 경과 (KST):**
  - 10-08 22:31: 시작
  - 22:33: numpy BLAS 커밋 문제로 campaign을 재시작했다. hand 3,000 체크포인트에서 재개했다.
  - 약 23:12: 사용자 DT 10M 작업(약 15 GB)과 커밋 한도가 겹쳤다. ablation과 B2를 assistant가 직접 멈췄다(85–97k).
  - 23:45: 감시 스크립트가 커밋 여유 0.56 GB에서 B0-sched ×5를 강제 종료했다(151–155k).
    `main_1m_20261008.campaign.txt`의 `FAILED code=1`은 이 의도적 종료이며, 이후 완료됐다.
  - 10-09 00:21: 사용자 작업이 끝난 뒤 전부 체크포인트에서 재개했다(B0는 `main_1m_20261008.b0.campaign.txt`).
    TARA는 06:07, B0-sched는 07:05–07:08에 완료됐다.
- **재개 무결성:**
  - 중복 hand는 0개다.
  - co-trained k-means arm은 모든 arm, 모든 seed에서 비트 단위로 같다. 중단된 런과 중단되지 않은 런이 같다는 뜻이다.
  - 단위 시험은 경계 정지 후 재개만 다루고, 구간 중간의 강제 종료는 다루지 않는다.
- **같은 코드 버전:** 1M과 ablation 런은 같은 소스 해시와 실행 파일(`tara_solver_7bacc2b8…`)을 쓴다.

### 실제로 쓴 명령

```powershell
python -m agents.cpp_mccfr.tara.campaign --name main_1m_20261008 --arms tara b0_kmeans_sched --seeds 222 111 333 444 555 --hands 1000000 --jobs 10 --min-free-gib 1.5
python -m agents.cpp_mccfr.tara.campaign --name ablation_127k_20261008 --arms b3_latest b4_uniform b5_fixed b7_prev_last index random b0_kmeans_partition --seeds 222 111 333 444 555 --hands 127000 --jobs 5 --min-free-gib 1.5
python -m agents.cpp_mccfr.tara.campaign --name ablation_127k_20261008 --arms b2_k8 b2_k64 --seeds 222 111 333 444 555 --hands 127000 --jobs 2 --extra "--b2-buffer D:/Experiment/Project-7-stud-Poker-Agent/agents/cpp_mccfr/data/tara/main_127k_20261008/tara_seed{seed}/buffer_1.bin"
# LBR: 다중 forest arm은 sum latest, 단일 forest arm(B2, B3, B0-single)은 sum만 (newest = sum)
python -m agents.cpp_mccfr.tara.campaign --name main_1m_20261008 --arms tara b0_kmeans_sched --seeds 222 111 333 444 555 --hands 1000000 --lbr sum latest --all-models
python -m agents.cpp_mccfr.tara.campaign --name ablation_127k_20261008 --arms b4_uniform b5_fixed b7_prev_last index random --seeds 222 111 333 444 555 --hands 127000 --lbr sum latest --all-models
python -m agents.cpp_mccfr.tara.campaign --name ablation_127k_20261008 --arms b2_k8 b2_k64 b3_latest b0_kmeans_partition --seeds 222 111 333 444 555 --hands 127000 --lbr sum --all-models
$D = "agents/cpp_mccfr/data/tara"
python -m agents.cpp_mccfr.tara.report --campaign $D/main_1m_20261008 --output $D/review_1m_20261009_v2 --window 20000 --dt-run centered=agents/cpp_mccfr/data/dynamic_tree_validated/confirm_centered_seed222_50k --dt-lbr centered=agents/cpp_mccfr/data/dynamic_tree_validated/centered_seed222_1m_lbr/summary.json
python -m agents.cpp_mccfr.tara.report --campaign $D/main_1m_20261008 $D/ablation_127k_20261008 --arms tara b0_kmeans_sched b0_kmeans_partition b2_k8 b2_k64 random --max-hands 127000 --window 5000 --lbr-hand 127000 --output $D/review_127k_partitions_20261009_v2
python -m agents.cpp_mccfr.tara.report --campaign $D/main_1m_20261008 $D/ablation_127k_20261008 --arms tara b3_latest b4_uniform b5_fixed b7_prev_last index --max-hands 127000 --window 5000 --lbr-hand 127000 --output $D/review_127k_variants_20261009_v2
python -m agents.cpp_mccfr.tara.paired --campaign $D/main_1m_20261008 --hand 1000000 --window 20000 --pairs tara:kmeans tara:b0_kmeans_sched b0_kmeans_sched:kmeans --output $D/review_1m_20261009_v2/paired_1000000.md
python -m agents.cpp_mccfr.tara.paired --campaign $D/main_1m_20261008 $D/ablation_127k_20261008 --hand 127000 --pairs tara:kmeans tara:b2_k64 tara:b2_k8 tara:random tara:b0_kmeans_sched tara:b7_prev_last tara:b3_latest tara:b4_uniform tara:b5_fixed tara:index b2_k64:kmeans b2_k64:b0_kmeans_partition b0_kmeans_partition:kmeans b0_kmeans_sched:kmeans b0_kmeans_sched:b0_kmeans_partition --output $D/review_127k_partitions_20261009_v2/paired_127000.md
```

- `review_1m_20261009`, `review_127k_20261009`는 같은 데이터의 첫 판이다(선형 LBR 축, 다중비교 보정 없음).
- `_v2`가 최종판이다. 바뀐 점은 다음과 같다.
  - LBR log-log
  - 얇은 선과 k-means와 겹치지 않는 색
  - 127k를 두 묶음으로 분리
  - BH/Holm과 95% CI
  - KL과 ±100 패널 변화

## 결과 (측정 완료, 5 seed, mean ± sample sd)

게임은 HU 7-stud v3다. 5–7구를 학습하고, 3–4구는 H4 휴리스틱이다. 공용 키는 k-means arm과 같은 베팅 요약이다.

**지표:**
- **Policy-LBR:** 5–7구 공격자, 낮을수록 좋다. exploitability의 휴리스틱 하한이다.
- **Local BR-gap:** 첫 5구 결정의 한 단계 지표다. crossfit은 편향 보정판이다.
- 둘 다 exploitability가 아니다.

**통계:** paired는 같은 seed끼리의 차이(= 같은 deal과 같은 k-means arm)다. BH/Holm은 파일 안의 모든 검정에 대해 보정했다.
5 seed라서 "유의하지 않음"은 "같음"이 아니라 "판단 불가"다.

### 1M (`data/tara/review_1m_20261009_v2/`, paired는 `paired_1000000.md`)

| | LBR current | LBR 평균 Σ S_j | LBR 평균 최신 트리 | crossfit current (arm − k-means) | 버킷 |
|---|---|---|---|---|---|
| TARA | 2.14 ± 0.59 | 4.41 ± 0.67 | 3.48 ± 0.88 | +0.073 ± 0.048 | Σ leaf 18,802 (트리 10개) |
| B0-sched | 2.01 ± 0.39 | 1.58 ± 0.52 | 1.88 ± 0.49 | −0.009 ± 0.008 | 711,948 (k-means 파티션 10벌) |
| k-means hard256 | 1.71 ± 0.54 | 1.68 ± 0.19 | — | 0 | 상태 70,067 |
| 참고: DT centered (seed222 1개, 비대응 구조) | 1.83 | 1.57 | — | — | — |

- **평가 경로 검증:** seed222 k-means의 LBR(1.96625 / 1.89275)이 DT 1M LBR과 **pair 단위로 같다.**
- **TARA − k-means (paired):**

  | 지표 | 차이 | 판정 |
  |---|---|---|
  | current LBR | +0.43 ± 0.75 | 95% CI [−0.50, +1.35], p = 0.27, 판단 불가 |
  | Σ S 평균 LBR | +2.74 ± 0.79 | 5/5 seed, p = 0.002, BH 0.023, Holm 0.023 |
  | 최신 트리 평균 LBR | +1.81 ± 0.82 | 5/5 seed, p = 0.008, BH 0.039 |
  | crossfit current | +0.073 | 5/5 seed, p = 0.027, BH 0.067. plug-in도 같은 방향(+0.103) |
  | crossfit average | +0.017 | n.s. |

- **TARA − B0-sched:**
  - Σ S 평균 LBR: +2.84 ± 1.12 (5/5, p = 0.005, BH 0.036)
  - crossfit current: +0.082 (5/5, p = 0.010, BH 0.039). 다만 plug-in은 −0.051(2/5)이라 **추정치 의존적이다.**
  - current LBR: +0.13 (판단 불가)
- **B0-sched − k-means:** 모든 지표에서 판단 불가다. 같은 파티션이면 RM+, 선형 평균, 2^t 일정이 1M에서 큰 차이를 만들지 않았다.
- **학습 곡선 (seed 평균 current LBR, TARA 대 k-means):**

  | hands | 3k | 15k | 63k | 127k | 255k | 511k | 1M |
  |---|---|---|---|---|---|---|---|
  | TARA | 8.07 | 3.33 | 2.36 | 2.45 | 2.68 | 1.97 | 2.14 |
  | k-means | 11.63 | 4.55 | 2.71 | 2.58 | 2.35 | 1.99 | 1.71 |

  - 기술 통계일 뿐이고 snapshot별 검정은 하지 않았다.
  - crossfit current로 보면 TARA는 10–40k에서 더 낮고(5/5), 40–80k에서 구별되지 않으며, 약 80k부터 더 높다(5/5).
    MA 20k 곡선은 약 75k에서 교차하고, 실제 교차는 60–65k다.
  - crossfit average는 127–255k 구간까지 TARA가 낮고(−0.034, p = 0.04), 약 300k에서 교차한다. 1M에서는 +0.017(n.s.)다.
- **안정성 (9개 경계 × 5 seed):**
  - 트리 추가 순간(at → added)의 정책 변화는 세 출력 모두 **정확히 0**이다.
  - current의 TV는 대조 창 0.047, 경계 창(B → B+100) 0.073이고, 44/45에서 경계 창이 더 크다.
  - B0-sched는 0.018 대 0.028이다.
  - 128-root 패널의 ±100 Local BR-gap 변화는 패널 잡음(sd ~0.2–0.9)에 묻혀 구별되지 않는다.
- **트리별 regret 비중 (`forests.png`):** 트리 1(카드 무시)은 약 3%다. 트리 2–8은 각각 10–12%, 최신 트리는 약 17%다(511k).

### 127k (`review_127k_partitions_20261009_v2/`, `review_127k_variants_20261009_v2/`, paired는 partitions 폴더의 `paired_127000.md`)

TARA와 B0-sched는 1M 런의 127k 지점에서 잘랐다. LBR은 127k snapshot이다.

| arm | LBR current | LBR Σ S | LBR 최신 트리 | crossfit current (− k-means) | 버킷 |
|---|---|---|---|---|---|
| **B2 k=64** (22특성 고정 k-means) | **1.38 ± 0.21** | **1.05 ± 0.36** | = Σ | −0.069 | 17,517 |
| B2 k=8 | 1.97 ± 0.51 | 1.84 ± 0.78 | = Σ | −0.071 | 2,315 |
| B4 균일 가중치 | 1.99 ± 0.32 | 5.72 ± 1.61 | 3.74 ± 0.41 | +0.040 | 12,640 |
| j·r 가중치 | 2.26 ± 0.30 | 6.11 ± 1.62 | 3.35 ± 1.22 | +0.041 | 12,641 |
| prev_scope=last | 2.39 ± 0.83 | 3.04 ± 0.81 | 3.08 ± 0.45 | −0.021 | 12,630 |
| **TARA** | 2.45 ± 0.51 | 6.41 ± 1.60 | 3.70 ± 0.80 | +0.032 | 12,640 |
| k-means hard256 | 2.58 ± 0.21 | 2.84 ± 0.84 | — | 0 | 65,407 |
| B5 고정 구간 | 2.75 ± 0.74 | 8.85 ± 0.57 | 3.86 ± 0.53 | +0.057 | 12,627 |
| B0-single | 2.87 ± 0.44 | 2.30 ± 1.03 | = Σ | −0.047 | 66,329 |
| random 분할 | 2.88 ± 0.49 | 11.00 ± 1.12 | 6.25 ± 2.03 | −0.025 | 12,643 |
| B0-sched | 3.10 ± 0.60 | 2.32 ± 0.77 | 2.32 ± 0.85 | −0.025 | 461,074 |
| B3 최신 트리만 | 3.56 ± 1.66 | 2.95 ± 1.45 | = Σ | +0.038 | 2,054 |

**BH q ≤ 0.05를 통과한 paired 차이 (75개 검정, 서로 상관됨):**
- TARA의 Σ S 평균이 더 나쁨:

  | 비교 대상 | 차이 | q (BH) |
  |---|---|---|
  | k-means | +3.57 | 0.039 |
  | B2 k=64 | +5.36 | 0.019 |
  | B2 k=8 | +4.57 | 0.019 |
  | B0-sched | +4.09 | 0.019 |
  | prev_last | +3.37 | 0.039 |
  | random (TARA가 더 나음) | −4.59 | 0.019 |

  - TARA − B2 k=64의 최신 트리 평균도 +2.65 (q = 0.024)다.
- TARA의 crossfit current가 더 나쁨:
  - B2 k=64 +0.101, B2 k=8 +0.103, prev_last +0.054 (각 q = 0.019)
- B2 k=64 − k-means:
  - LBR current −1.20 ± 0.37 (q = 0.019)
  - crossfit current −0.069 (q = 0.019)
- B2 k=64 − B0-single: LBR current −1.49 (q = 0.033)
- 기타:
  - TARA − B5 crossfit average +0.028 (q = 0.019, Holm 0.032)
  - TARA − B0-sched crossfit average −0.056 (q = 0.019; TARA가 나음)
- **경계선:** TARA − B2 k=64의 LBR current는 +1.07 ± 0.63 (5/5, p = 0.019, q = 0.071)이다.

**판단 불가 (5 seed):**
- B4, j·r, B5, prev_last, B3가 current LBR에 미치는 효과
- B0-single − k-means: LBR current +0.30, p = 0.063. budget에 따라 부호가 바뀐다.

### 해석 (assistant 판단과 남는 교란)

1. **1M에서 TARA는 co-trained k-means를 이기지 못했다.**
   - current LBR은 판단 불가다(CI가 −0.5에서 +1.35까지 걸친다).
   - 평균 출력은 명확히 나쁘다.
   - crossfit current는 약 60–80k 이후 나쁘다.
   - 저장 leaf 수는 k-means 상태의 27%(18,802 대 70,067)다. 다만 current 정책은 10개 파티션의 교차로 정해지므로 해상도와 같지 않다.
   - 특성 계산, 조회, 버퍼 비용은 비교하지 않았다.
2. **127k에서 같은 22개 특성의 고정 k-means(B2 k=64)가 TARA와 k-means보다 낫다.**
   - 이것은 "이 두 특정 파티션, 이 budget"에 대한 결과다.
   - B2와 k-means/B0-single은 특성(22 표준화 대 power 18), k, 버킷 수, 거리, 적합 데이터가 한꺼번에 다르다.
   - B2의 우위는 budget이 커질수록 줄었다: 1k −7.5 → 31k −1.37 → 127k −1.20.
   - TARA와 B2는 특성이 같으므로, TARA − B2는 특성 문제가 아니다. 파티션 방법(regret 회귀 트리 + 다중 트리 누적) 대 고정 클러스터의 차이다.
   - B2 k=64는 TARA보다 버킷이 많으므로 "거칠어서 빠르다"로도 설명되지 않는다.
   - 1M 미검증이다.
3. **학습된 분할의 효과:**
   - random보다 나은 것은 Σ S 평균 LBR뿐이다(−4.6).
   - current LBR에는 이점이 없고, crossfit Local BR-gap은 random이 오히려 약간 낫다(current +0.058, q = 0.137; average +0.053).
   - random arm도 TARA의 D'Hondt 배분, 행, 일정은 그대로 쓴다.
4. **TARA current 정체의 원인은 미해결이다.**
   - 31k–1M 동안 current LBR은 약 2.0–2.7에 머물렀고, 그동안 k-means는 3.7에서 1.7로 내려갔다.
   - 가중치 ablation(B4 +0.47, j·r +0.20, prev_last +0.06)은 current LBR을 유의하게 바꾸지 않았다.
   - 후보 원인(미검정):
     - 트리당 leaf 2,048 고정. 반면 데이터는 늘어난다(트리 10 구간의 방문은 3,330만).
     - 공용 노드당 reservoir 4,096행
     - 잡음이 큰 표본 regret 라벨
     - 공유 베팅 요약 290개 노드
   - 미분할 공용 노드는 개수로는 절반이지만 질량은 4–5%뿐이다. 방문된 정보집합은 평균 7.5/9개 트리에서 분할된 노드에 있다.

## 이상하게 나온 부분과 추정 원인

1. **Σ_j S_j 평균의 붕괴.**
   - TARA, B4, B5, j·r, random에서 나타난다. TARA는 1M LBR 4.41 대 k-means 1.68이다.
   - B0-sched(모든 트리가 같은 파티션)와 prev_last(옛 트리가 S를 더 쌓지 않음)는 붕괴하지 않거나 훨씬 덜하다.
   - **추정 메커니즘 (가설):** average()는 정보집합마다 각 트리 leaf의 S 질량으로 가중된다.
     - 공용 노드 전체가 leaf 하나인 트리 1의 몫은 1M에서 약 28–30%다.
     - 카드를 구분하지 못하는 성분을 모두 합치면 약 45–50%다(current 정책에서는 약 9%).
     - 이 몫은 127k seed222의 arm 순서와 상관된다: prev_last 0.09 < TARA 0.58 < random 0.67.
   - 인과 검증은 하지 않았다.
   - 처음 쓴 "트리별 S 총량이 같아 1/9씩 섞인다"는 설명은 틀렸다. 같은 총량은 선형 평균의 기계적 결과이고 B0-sched도 똑같다.
   - 재학습 없는 검증: 기존 1M 모델을 Σ_{j≥2} S_j나 leaf 질량 정규화 평균으로 다시 LBR 평가한다(**미실행**).
2. **최신 트리 평균도 k-means보다 나쁘다**(3.48 대 1.68). 거친 단일 파티션의 그룹 평균 자체도 약하다.
3. **Local BR-gap 추정치 불일치:** plug-in과 crossfit의 arm별 차이가 결과 차이보다 크다(앞 절). TARA − B0-sched는 추정치에 따라 부호가 바뀐다.
4. **1k hands에서 TARA LBR이 균일 정책(35)과 같다:** 트리 1이 카드를 무시하기 때문이다. 예상된 결과다.
5. **B5는 18k hands에서 LBR이 53이다:** 첫 적합 트리가 18,143 hands에야 생긴다. 그래프에는 B5의 고정 경계가 표시되지 않는다.
6. **B3는 경계마다 정책이 크게 바뀐다:** TV 0.52, KL 1.84, 패널 crossfit +0.29. LBR 분산도 크다(3.56 ± 1.66).
7. **B0-sched의 트리별 버킷 수가 k-means 상태 수보다 1–2% 많다:** 예를 들어 511k seed222에서 70,633 대 69,058이다. 원인은 확인하지 않았다.
8. **구 코드의 유리한 127k 결과가 재현되지 않았다:** 앞 절 실행 이력을 보라.

## 사양 대비 미완료/변경

- **B1과 정확 평균 표:** B1은 7-stud에서 불가능하다. 정확 평균 표도 쓰지 않는다.
- **미실행:** B6(별도 코드 `shadow_split`)과 `avg_same_as_regret`
- **127k에서만 비교:** B2–B5와 기타 ablation
- **단위 시험:**
  - "identity 특성 → 완전 recall" 시험은 없다. 7-stud 정보집합으로는 불가능하다.
  - 대신 C++ self-test가 다음을 확인한다: pow2 = halving 비트 일치, 트리 1 카드 무시, B0 파티션 = k-means 키 1:1.
  - 버킷 합법성은 `solver.cpp`의 `require`(버킷 하나에 합법 행동 집합이 둘 이상이면 중단)로 강제한다.
- **메모리 그래프:**
  - x축은 사양의 메모리 값이 아니라 버킷 수(Σ leaf, k-means 상태)이고, LBR만 그렸다(Local BR-gap 대 메모리 그림은 없다).
  - 트리 추가 표시는 hands 축에만 있다.
  - 사양식 메모리 값은 summary 표에 있다.
- **안정성 가중:**
  - probe 상태는 B−100에서 합 평균 self-play로 뽑아 같은 가중을 준다. 정책별 reach 가중이 아니다.
  - ±100 평가 변화는 128-root 패널에서만 쟀다(LBR은 경계마다 재지 않음).
