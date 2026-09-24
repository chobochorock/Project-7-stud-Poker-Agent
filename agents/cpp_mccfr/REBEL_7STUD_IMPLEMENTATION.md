# 7-Stud ReBeL 7th-street PBS 구현

## 현재 범위

[`stud_rebel_seventh.cpp`](stud_rebel_seventh.cpp)는 ReBeL 전체가 아니라
첫 실행 단계인 **7th-street public-belief subgame solver**다. 5th/6th는 기존
MCCFR blueprint를 사용하고, 7th에 도달하면 공개 정보만으로 private range를
다시 만든 뒤 local CFR-AVG를 수행한다.

신경망 leaf value는 아직 없다. 7th 이후에는 chance card가 없으므로 terminal
chip payoff를 정확히 계산할 수 있다. 이 구간에서 posterior와 local solving을
먼저 검증한 뒤 `V7 -> 6th resolve -> V6` 순서로 확장한다.

## Public-belief 구성

solver가 사용하는 공개 정보는 다음뿐이다.

- 두 플레이어의 공개 카드
- 5th/6th betting history
- pot, ante, stack cap
- 현재 blueprint의 공개 행동 확률

환경 객체에 들어 있는 실제 상대 hidden card는 posterior 생성에 사용하지
않는다. 남은 덱에서 두 플레이어의 초기 hidden 2장, discard 1장, 7th hidden
1장을 함께 표본화한다.

각 joint particle의 가중치는 다음과 같다.

\[
w(h_0,h_1) \propto
P(\text{observed H4 reveal/discard}\mid h_0,h_1)
\prod_{t:\,street<7}\pi_{blueprint}(a_t\mid I_t(h_0,h_1)).
\]

H4 항은 가능한 초기 카드 순열 중 현재 공개 카드와 discard를 선택하는 순열
수로 계산한다. Betting 항은 후보 hidden card로 5th부터 상태를 재생하면서
blueprint 정책 확률을 곱한다. blocker는 joint particle을 중복 없이 뽑는
과정에서 반영된다.

## Local CFR-AVG

정규화된 posterior에서 chance particle을 뽑고 두 플레이어를 번갈아
traverser로 삼는다. Traverser action은 모두 순회하고 상대 action은 현재
regret-matching 정책에서 표본화한다.

로컬 정보집합은 다음을 사용한다.

```text
power-tree private bucket + exact 7th betting history
```

따라서 exact private-card ReBeL은 아니며 fixed private abstraction을 쓰는
particle ReBeL 근사다. Local 정보집합이 particle에 없으면 기존 blueprint로
fallback한다. `local_policy_coverage`가 이 fallback을 제외한 비율이다.

새 public root가 나타날 때마다 이전 particle과 local regret table을 비운다.
장기간 실행에서 증가 가능한 전역 자료는 power assignment cache뿐이며
`--power-cache-max`로 제한된다.

## 빌드와 검사

```powershell
g++ -O3 -std=c++17 cpp_mccfr\stud_rebel_seventh.cpp `
  -o cpp_mccfr\stud_rebel_seventh.exe

.\cpp_mccfr\stud_rebel_seventh.exe --self-test --progress-seconds 0
```

## 권장 평가

```powershell
.\cpp_mccfr\stud_rebel_seventh.exe `
  --bucket power-memory16 `
  --model cpp_mccfr\made_call_r1000_k512_epsheur20_memory16_30m.bin `
  --atlas cpp_mccfr\power512_epsheur20_memory16_v1.bin `
  --hands 10000 `
  --lbr-particles 240 `
  --rebel-particles 240 `
  --rebel-iterations 100 `
  --prior 100 `
  --ante 1000 `
  --stack-ante 1000 `
  --power-cache-max 250000 `
  --progress-seconds 30 `
  --seed 71201
```

`power-tree` 체크포인트도 지원한다.

```powershell
.\cpp_mccfr\stud_rebel_seventh.exe `
  --bucket power-tree `
  --model cpp_mccfr\power_tree_cooldown_static_1p1m.bin `
  --atlas cpp_mccfr\power_tree_cooldown_1m.atlas `
  --hands 1000 --lbr-particles 64 `
  --rebel-particles 240 --rebel-iterations 100 `
  --power-cache-max 250000 --seed 71202
```

## 출력 지표

| 지표 | 의미 |
|---|---|
| `subgames` | 새로 푼 7th public subgame 수 |
| `proposals` | posterior particle 후보 수 |
| `h4_rejected` | 공개 H4 결과와 맞지 않은 후보 수 |
| `replay_rejected` | 공개 betting history를 재생하지 못한 후보 수 |
| `average_effective_particles` | posterior의 ESS |
| `average_posterior_entropy` | 정규화 posterior entropy |
| `local_policy_coverage` | local CFR 정책이 존재한 7th query 비율 |
| `blueprint_fallbacks` | local particle cover 밖이라 blueprint를 사용한 횟수 |
| `blueprint_history_misses` | 과거 행동 likelihood 계산 시 blueprint miss 수 |
| `maximum_local_nodes` | 한 subgame에서 사용한 local regret node 최대치 |
| `maximum_particles` | 한 subgame의 particle 최대치 |
| `power_cache_entries/limit/resets` | bounded power assignment cache 상태 |

## 2026-08-11 smoke 결과

동일한 1,000 paired-seat hands, LBR 64 particles, local 100 iterations 기준:

| resolver | LBR ante/hand | SE | local coverage | peak RAM |
|---|---:|---:|---:|---:|
| blueprint | 0.906 | 0.207 | - | - |
| 기존 determinized exact-key | 0.925 | 0.252 | 거의 100% | - |
| PBS, 64 particles | 0.847 | 0.262 | 64.6% | 151 MiB |
| PBS, 240 particles | 0.756 | 0.232 | 72.7% | 151 MiB |

표준오차가 커서 성능 우열은 아직 확정할 수 없다. 다만 PBS particle 증가가
ESS와 coverage를 개선하고, 100핸드에서 1,000핸드로 늘려도 peak RAM이
약 132 MiB에서 151 MiB에 머문 것은 확인했다. 10,000 hands / LBR 240
particles 결과가 첫 비교 가능한 본 실험이다.

## 다음 단계

1. 동일 deal seed로 blueprint/exact/PBS의 per-hand 차이를 저장해 paired CI 계산
2. local coverage를 90% 이상으로 올릴 private-state stratified sampling 추가
3. exact 7th solve로 private value-vector target 생성
4. `V7` 학습 후 6th depth-limited PBS resolve 구현

