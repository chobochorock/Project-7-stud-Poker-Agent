# Shadow-Regret Splitting (7-stud 이식)

사용자 설계 "261007 — Sampling IRA (가칭: Shadow-Regret Splitting)"를 C++ 7-stud v3에 옮긴 계열이다.
Toy 구현과 결과는 `Toy-Card-Game-Agent/agents/adaptive_buckets/SHADOW_SPLIT.md`에 있다.
학습기, 평가기, 기준선은 `dynamic_tree_validated`를 그대로 쓴다. 공유 소스는 수정하지 않고 include만 한다.
완전정보 CFR+IRA 재현이 아니며, 수렴 보장도 없다.

## 무엇이 같고 무엇이 다른가

- **같은 것 (`dynamic_tree_validated`와 동일):**
  - 규칙: C++ 7-stud v3 HU, 휴리스틱 H4, ante 1000, stack 1000 ante
  - 학습기: signed alternating external-sampling MCCFR, 순회 안 정책 동결, SIMPLE 평균, 선형 가중치 없음
  - 예산 단위: 1 hand = 새 5구 deal 1개 + 모델별 순회 2회
  - 평가: 판마다 첫 5구 결정의 Local BR-gap(32 particle, eval seed 307), 10k판마다 128-root panel,
    고정 checkpoint의 Policy-LBR
  - 기준선: 같은 판에서 함께 학습하는 hard256 k-means arm. 같은 seed면 DT 실행의 k-means 열과
    **비트 단위로 같다**(시험으로 확인).
- **tree arm (새로 만든 것):**
  - 공용 노드마다 카드 트리가 하나 있다.
  - leaf는 (특성, bin, 행동)별로 clip하지 않은 표본 regret shadow를 둔다. 순회 단위 합의 제곱도 함께 쌓는다.
  - Λ = (g(L) + g(R) − g(L+R))/W, g = max⁺ 상쇄 통계로 분할한다.
  - 상속은 B다: shadow 합에, 나머지 상속분을 방문 비율로 나눠 더한다.

## 공용 노드(`--key-mode`)

| 모드 | 공용 노드 정의 | 성질 |
|---|---|---|
| `summary` (기본) | k-means 기준선과 같은 베팅 요약 + 좌석. 카드 cluster만 뺀다 | 두 arm이 같은 베팅 분할을 쓰고 카드 추상화만 다르다. 한 leaf가 한 순회에 여러 번 나올 수 있어 정책을 고정한다 |
| `own` | 좌석, 스트리트, 합법 행동, 자기의 이전 스트리트 행동, 현재 스트리트 전체 이력 | traverser leaf는 순회당 한 번만 갱신된다(A1). 공용 노드가 수십만 개다 |
| `exact` | 정확한 베팅 이력 전체 (설계 그대로) | 공용 노드가 1,700만 개까지 늘어난다. 노드당 표본이 부족하다 |

정확한 이력의 공용 결정 노드는 34 / 8,000 / 17,371,168개다(5/6/7구, 읽기 전용 열거).
그래서 설계 그대로의 "베팅 이력은 추상화하지 않음"은 7-stud에서 시연만 한다.

## 특성과 판정

- **특성:** power vector 0–17(카드)에 pot, 상대 투입, 공격성, 이력 길이를 더한다. `summary` 모드는 0–17로만 자른다.
- **bin:** 스트리트별 8개 분위다. no-fold 균일 무작위 플레이 상태 20k개(seed 2026100801)에서 만든다.
  학습 정책이 아닌 균일 플레이 기준이라는 data-policy bias가 있다.
- **판정(`--rule z`, 기본):** Λ − z·(max√QL + max√QR + max√QP)/W > ε(t)이면 분할한다.
  - Q는 순회 단위 합의 제곱이다.
  - z = 0.66은 Toy에서 사후에 고른 ρ×0.1에 해당하는 값이다(√(2L)·0.1, L≈22). 7-stud에서는 결과를 보기 전에 정했다.
  - 동시 보장이 없는 검사별 규칙이다.
- **`--rule time_uniform`:** 설계 그대로의 시간 균일 반경이다(span 2000 ante). 한 순회에 leaf가 한 번만
  나오는 `own`/`exact`에서만 유효하다.
- **기타:**
  - ε(t) = max(eps_min, κ·2 ante/√t). 2 ante는 처음 pot이다.
  - grace는 shadow 방문 100회다. shadow는 leaf가 100번 갱신된 뒤 할당한다.
  - shadow 메모리는 `--shadow-gib`(기본 4)로 제한한다.

## 실행

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m unittest agents.cpp_mccfr.shadow_split.test_shadow_split -v
$run = "agents/cpp_mccfr/data/shadow_split/summary_z066_seed222_50k"
python -m agents.cpp_mccfr.shadow_split.run train --run $run --hands 50000 --seed 222
python -m agents.cpp_mccfr.shadow_split.run lbr --run $run --output "$run/lbr_2026100711_12" --seeds 2026100711 2026100712
python -m agents.cpp_mccfr.shadow_split.report --output agents/cpp_mccfr/data/shadow_split/<new review> `
  --runs summary_z066=$run --references dt_centered=agents/cpp_mccfr/data/dynamic_tree_validated/confirm_centered_seed222_50k `
  --max-hands 50000 --lbrs summary_z066="$run/lbr_2026100711_12"
```

`--hands`는 총 목표다. Ctrl+C를 누르면 현재 판을 마치고 저장한다. 같은 조건에 `--resume`을 붙이면 이어간다.
데이터는 `data/shadow_split/<run>`에 있고, 실행 파일은 소스 해시 이름(`bin/shadow_split_*`)이다.

## 비교 규칙 (Project-7 AGENTS.md)

- k-means 기준선을 항상 겹쳐 그린다. 평가 root, particle, seed가 다른 참조는 unmatched로 표기한다.
- 같은 판 수라도 노드 방문, 메모리, 표 크기가 다르다. leaf, state, shadow MiB, 노드 수를 함께 보고한다.
- Local BR-gap, LBR, exploitability는 서로 다른 양이다. 어느 것도 exploitability가 아니다.
