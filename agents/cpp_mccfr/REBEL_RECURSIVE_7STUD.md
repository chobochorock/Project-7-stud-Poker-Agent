# 7-Stud recursive ReBeL: V7 -> 6th search

## 현재 실행 경로

```text
5th: 30M MCCFR blueprint
6th: particle PBS + depth-limited local CFR
     leaf value = learned V7(infostate, public belief summary)
7th: particle PBS + terminal local CFR
```

사용하는 blueprint는 다음 기존 checkpoint 쌍이다.

```text
model: made_call_r1000_k512_epsheur20_memory16_30m.bin
atlas: power512_epsheur20_memory16_v1.bin
```

Blueprint는 5th 실제 정책, H4/5th/6th action likelihood, 6th/7th local
node prior, local coverage 밖 fallback에 사용된다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `stud_rebel_seventh.cpp` | 7th posterior PBS와 terminal local CFR |
| `stud_rebel_v7_generate.cpp` | V7 학습 표본 생성 |
| `train_rebel_v7.py` | scalar infostate-value network 학습 |
| `rebel_value_server.py` | TorchScript V7 IPC 추론 |
| `stud_rebel_recursive.cpp` | V7 leaf를 사용하는 6th local CFR |

## V7 target

7th public root에서 기존 particle PBS solver를 먼저 푼다. Posterior joint
private deal을 표본화하고 local average policy의 fixed-deal EV를 계산한다.
입력에는 viewer의 정보집합과 그 public root의 particle posterior 요약이
함께 들어간다.

```text
input:  deep_cfr_tensor(I), 1832 floats
        + public belief summary, 208 floats
        = 2040 floats
target: local CFR continuation value, ante units
```

Belief summary는 viewer 기준 자기/상대 좌석마다 `52 hidden-card inclusion
probabilities + 52 discarded-card probabilities`를 담는다. 같은 공개
히스토리라도 posterior range가 달라지면 V7 입력도 달라진다.

데이터는 public root ID를 함께 저장한다. Train/validation 분할도 root
단위여서 같은 subgame의 P0/P1 particle이 양쪽 split에 새지 않는다.

## 6th search

6th public state에서 H4 결과와 과거 blueprint action likelihood로 joint
private posterior를 만든다. 각 local CFR iteration은 posterior particle을
뽑고 두 플레이어를 번갈아 traverser로 둔다.

6th betting round가 끝나면 particle에 보존된 7th hidden card를 배분한다.
현재 local CFR 정책이 관측된 6th 행동을 낼 likelihood로 모든 particle을
다시 가중해 7th PBS를 만든 뒤, terminal까지 펼치지 않고 V7을 조회한다.

```text
v_leaf(I7) = V7(deep_cfr_tensor(I7), summary(PBS7))
```

Local CFR iteration마다 range가 변할 수 있으므로 exact-card key만으로 V7
결과를 캐시하지 않는다. 현재 정책에서 확률 0인 반사실적 행동에는 likelihood
`1e-12`를 사용해 빈 belief를 피한다. 실제 플레이가 7th에 도달하면 value
network로 행동하지 않고 `stud_rebel_seventh.cpp`의 terminal PBS resolver를
다시 실행한다.

## 빌드

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_seventh.cpp `
  -o cpp_mccfr\stud_rebel_seventh.exe

g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_v7_generate.cpp `
  -o cpp_mccfr\stud_rebel_v7_generate.exe

g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_recursive.cpp `
  -lws2_32 -o cpp_mccfr\stud_rebel_recursive.exe
```

## V7 데이터 생성

작은 첫 실험:

```powershell
.\cpp_mccfr\stud_rebel_v7_generate.exe `
  --bucket power-memory16 `
  --model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --roots 1000 `
  --particles 64 `
  --iterations 32 `
  --samples-per-root 4 `
  --behavior-epsilon 0.1 `
  --output cpp_mccfr\rebel_v7_1k.bin `
  --report-every 50 `
  --seed 74101
```

`samples`는 `2 * roots * samples-per-root`다. 레코드당 약 8.0 KiB이므로
위 설정은 약 62 MiB다. `behavior-epsilon`은 6th local CFR이 만들 수 있는
blueprint 밖 betting leaf도 V7 학습에 일부 포함한다. 학습용 posterior의
action likelihood에도 같은 epsilon mixture를 적용한다.

## V7 학습

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $py -B train_rebel_v7.py `
  --input cpp_mccfr\rebel_v7_1k.bin `
  --output models\rebel_v7_1k.pt `
  --hidden 128 --layers 2 `
  --epochs 20 --batch-size 256 `
  --threads 1 --seed 74102
```

Trainer는 target을 표준화해 Huber loss로 학습하고 validation MAE가 가장
낮았던 epoch를 export한다. `constant_baseline_mae_ante`보다
`best_validation_mae_ante`가 충분히 작아야 다음 평가가 의미 있다.

## 6th 평가

터미널 1에서 value server를 실행한다.

```powershell
& $py -B rebel_value_server.py `
  --model models\rebel_v7_1k.pt `
  --port 28741 --threads 1
```

터미널 2에서 recursive resolver를 평가한다.

```powershell
.\cpp_mccfr\stud_rebel_recursive.exe `
  --bucket power-memory16 `
  --model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --value-port 28741 `
  --hands 1000 --lbr-particles 64 `
  --sixth-particles 64 --sixth-iterations 32 --sixth-prior 100 `
  --seventh-particles 64 --seventh-iterations 100 --seventh-prior 100 `
  --progress-seconds 30 --seed 74103
```

## 로그

| 필드 | 의미 |
|---|---|
| `sixth_subgames` | 새로 푼 6th public root 수 |
| `sixth_traversals` | 양쪽 traverser를 포함한 local CFR 순회 수 |
| `sixth_node_visits` | 6th search node 방문 수 |
| `sixth_leaf_queries` | V7 leaf 평가 요청 수, cache hit 포함 |
| `sixth_average_effective_particles` | 6th posterior ESS |
| `sixth_blueprint_fallbacks` | local node가 없어 blueprint를 쓴 횟수 |
| `seventh_subgames` | 실제 행동용 7th terminal solve 수 |
| `v7_network_queries` | 실제 Python/TorchScript 호출 수 |
| `v7_cache_hits` | belief-aware 모드에서는 stale range 방지를 위해 항상 `0` |

## 검증 결과

- V7 generator self-test: 통과
- root-grouped V7 trainer/export: 통과
- 6th CFR -> local-policy PBS update -> V7 IPC -> 7th terminal resolver
  end-to-end self-test: 통과
- 이전 V2 100 roots / 800 labels probe: constant MAE `5.46`, best V7 MAE
  `5.07`

800 labels는 구조 검사에는 충분하지만 value model 품질에는 부족하다.
Root 단위 validation을 적용하기 전의 더 좋은 수치는 leakage였으므로 사용하지
않는다.

### 2026-08-18 explicit-PBS V3 본 실험

V7은 1,000 public roots, 8,000 labels로 학습했다. 입력은 2,040차원이고
root-grouped validation 결과는 다음과 같다.

| 지표 | 값 |
|---|---:|
| constant baseline MAE | 4.974 ante |
| best V7 MAE | 4.709 ante |
| best epoch | 3 |

이 모델을 6th 32 iterations, 7th 100 iterations로 사용하고 LBR 240
particles, 10,000 hands로 평가했다.

| 지표 | 값 |
|---|---:|
| LBR ante/hand | 1.2432 |
| SE | 0.1089 |
| 95% CI | [1.0297, 1.4566] |
| 6th posterior ESS | 10.90 / 64 |
| policy misses | 4,372,300 / 22,639,920 |
| V7 calls | 5,145,824 |
| elapsed | 2,924 s |

기존 7th-only resolver의 LBR `1.2227` (SE `0.1029`)과 통계적으로
구별되지 않는다. 서로 다른 deal seed의 독립 비교로 보면 차이는 `0.0205`,
결합 SE는 약 `0.1499`다.
따라서 현재 1k-root V7과 6th search는 **추가 개선을 입증하지 못했다**.
V7 held-out MAE가 여전히 4.7 ante이고 6th posterior ESS와 local policy
coverage가 낮은 것이 주요 병목 후보다. 원본 결과는
`results/rebel_v7_pbs_1k_lbr240_10k.json`에 저장했다.

## 아직 정식 ReBeL이 아닌 부분

1. 208차원 marginal summary는 카드 사이의 joint correlation과 정확한
   private-hand 조합 분포를 모두 보존하지 않는다.
2. 실제 7th resolver는 실제로 관측된 6th 행동까지는 blueprint likelihood로
   public belief를 재구성한다. 가상 search 정책은 실제 역사로 채택되기 전에는
   다음 의사결정으로 지속되지 않는다.
3. 6th local policy가 만든 leaf는 V7 데이터 분포 밖일 수 있다.
   `behavior-epsilon`은 이 문제를 완화하지만 제거하지 않는다.
4. 5th depth-limited resolver와 V6는 아직 구현하지 않았다.

다음 단계는 더 많은 V7 데이터를 수집해 explicit-PBS 모델의 held-out MAE와
LBR을 검증하는 것이다. 효과가 확인되면 V6 target을 수집하고 같은 구조를
5th로 올린다.
