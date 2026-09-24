# Sampled BR-gap 100k 결과

## 설정

- 7-Stud betting rules v3, ante 1000, stack 1000 ante
- H4 heuristic 고정, 5th~7th 평가
- 고정 baseline heuristic
- `power512_epsheur20_memory16_v1.bin` atlas를 사용한 k512 MCCFR cold start
- k512는 매 hand마다 독립 training deal로 P0/P1을 한 번씩 갱신
- 별도의 공통 evaluation deal에서 두 정책을 동일한 random seed로 평가
- 총 100,000 training hands와 100,000 evaluation hands

## 결과

| 구간 | heuristic gap | k512 gap |
|---:|---:|---:|
| 0~10k | 4.450 | 26.119 |
| 10k~20k | 4.496 | 23.854 |
| 20k~30k | 4.662 | 20.671 |
| 30k~40k | 4.749 | 19.088 |
| 40k~50k | 4.656 | 17.602 |
| 50k~60k | 4.677 | 16.421 |
| 60k~70k | 4.637 | 15.279 |
| 70k~80k | 4.225 | 14.395 |
| 80k~90k | 4.508 | 13.828 |
| 90k~100k | 4.392 | 13.449 |

마지막 10k의 95% CI는 heuristic `[4.023, 4.761]`, k512
`[13.037, 13.860]`이다. k512의 구간 평균은 첫 10k 대비 마지막 10k에서
48.5% 감소했다. 10k window 평균의 log-log 기울기는 약 `-0.309`다.

전체 100k 누적 평균은 heuristic `4.545`, k512 `18.071`이다. 이 값은
초기의 나쁜 k512 정책을 모두 포함하므로 최종 정책 품질은 마지막 구간
평균을 보는 편이 낫다.

## 스트리트

마지막 10k에서 평균 gap 분해는 다음과 같다.

| 정책 | 5th | 6th | 7th |
|---|---:|---:|---:|
| heuristic | 0.003 | 0.051 | 4.338 |
| k512 | 0.020 | 0.258 | 13.170 |

두 정책 모두 지표가 7th에 거의 전부 집중된다. 7th는 미래 chance가 없지만
가능한 반사실적 betting branch를 가장 많이 합산한다. 따라서 이 수치를
LBR ante/hand와 직접 비교하면 안 된다.

## 해석

- 지표는 고정 heuristic을 거의 평평한 기준선으로, 학습 k512를 감소
  곡선으로 구별했다.
- k512는 100k에서도 heuristic보다 큰 국소 이탈 여지를 남긴다.
- 0으로 향하는 방향은 보이지만 100k 범위에서 0 수렴을 주장할 수 없다.
- absolute value는 exact exploitability가 아니라 sampled,
  trajectory-relaxed counterfactual deviation proxy다.
- 다음 구현에서는 학습 traversal의 action values를 재사용해 같은 지표를
  거의 공짜로 기록하고, 독립 LBR은 드문 calibration에만 사용한다.

## 산출물

- `results/per_hand.csv`: 100,000개 hand 원자료
- `results/br_gap_curve.png`: 로그 구간 평균 및 누적 평균 그래프
- `results/k512_100k.bin`: 실험 종료 시점 cold-start k512 모델

## 고정 100k 모델의 10k regret 재평가

`k512_100k.bin`을 고정하고 독립 딜 10,000개에서 같은 행동가치 regret을
두 방식으로 합산했다.

| 회계 | exploitability형 평균 |
|---|---:|
| 원본 상태에서 먼저 max한 trajectory-relaxed 값 | 13.468 ante |
| 같은 k512 bucket의 행동별 regret을 먼저 합친 값 | 8.592 ante |

원본 정보집합 3,583,632개는 모두 한 번만 방문했다. 따라서 원본 정보집합
기준으로는 표본을 합친 효과가 없었다. 반면 실제 abstraction key는
312,385개가 재사용되어, 행동별 regret을 먼저 합치자 값이 36.2% 낮아졌다.
이는 표본별 사후 max 효과와 같은 bucket 내부의 상충이 크다는 뜻이다.

Bucket 방문 4회 이하 구간의 기여는 0.746 ante로 전체의 8.7%뿐이었다.
반대로 17회 이상 방문한 bucket이 6.372 ante, 전체의 74.2%를 만들었다.
따라서 8.592가 희소 bucket의 유한표본 `max` 편향만으로 생겼다고 보기는
어렵다. 또한 7th의 기여가 8.443 ante로 98.3%를 차지했다. 이 100k
정책은 abstraction 바닥에 도달했다기보다, 우선 7th의 잘 방문된 abstract
bucket 안에서도 큰 국소 이탈 여지를 남긴 상태로 해석하는 편이 맞다.

두 값 모두 exact exploitability가 아니다. 특히 bucket 값은 imperfect-recall
abstract game의 국소 이탈 진단값이다. Abstraction 병목은 여러 checkpoint에서
bucket 값은 하강·정체하는데 LBR이 높은 값에서 정체하는지를 함께 봐야 판정할
수 있다.

전체 JSON은 `results/k512_100k_regret_10k.json`에 저장했다.

## Memory16 30M의 동일 10k 평가

동일 atlas와 동일 10,000딜 seed로
`made_call_r1000_k512_epsheur20_memory16_30m.bin`을 평가했다.

| 모델 | trajectory/original | bucket-first |
|---|---:|---:|
| k512 cold-start 100k | 13.468 | 8.592 |
| memory16 30M | 0.496 | 0.263 |

30M은 각각 96.3%, 96.9% 감소했다. 30M bucket 값의 94.9%는 여전히
7th에서 발생했다. 방문 4회 이하 bucket이 0.117 ante로 44.6%를 차지해
유한표본 max 편향도 남아 있지만, 100k와 비교하면 abstract policy가 국소
stationarity에 훨씬 가까워진 것은 명확하다.

반면 같은 계열 모델의 LBR 하한은 과거 평가에서 약 1 ante 이상이었다.
따라서 이 값을 exploitability 상한으로 해석하면 안 된다. 현재 관측은
`abstract/local regret은 작지만 원 게임의 다단계 착취는 남음`이라는
abstraction·imperfect-recall·평가기 사각지대 후보를 보여준다. 동일 계열의
10M/30M/추가 checkpoint 곡선으로 bucket 값의 바닥을 확인해야 한다.

전체 JSON은 `results/memory16_30m_regret_10k.json`에 저장했다.

## Memory16 30M + 7th 실시간 resolver

`memory16 30M` blueprint와 `ExactSeventhResolverPolicy`를 동일한 10,000딜에서
비교했다. Resolver는 7th 진입마다 양쪽 좌석의 관측으로 별도 local game을
만들고, 정확한 베팅 히스토리와 핸드 히스토리를 사용했다. 설정은
`iterations=100`, `prior=100`이다.

| 정책 | sampled BR-gap | 95% CI |
|---|---:|---:|
| memory16 blueprint | 0.501 | [0.469, 0.533] |
| 7th resolver | 0.591 | [0.561, 0.622] |

같은 딜의 핸드별 차이를 직접 계산한 결과는 `resolver - blueprint = +0.0904`
ante이고 95% CI는 `[+0.0779, +0.1029]`다. 낮을수록 좋은 지표이므로 이
resolver 설정은 약 18.0% 악화됐다.

| 정책 | 5th | 6th | 7th |
|---|---:|---:|---:|
| memory16 blueprint | 0.0033 | 0.0249 | 0.4728 |
| 7th resolver | 0.0034 | 0.0263 | 0.5616 |

악화분은 사실상 7th에서 발생했다. 5th/6th 값도 terminal continuation인
7th 정책이 달라지므로 미세하게 변한다.

이번 평가기는 bucket regret을 합산하지 않는다. 원래 관측 정보, 자기 카드,
상대 공개 카드, discard 기억, 정확한 베팅 히스토리로 information-set key를
만들고, 각 key에 행동별 `Q(a)-V`를 먼저 누적한 뒤 `max_a`를 취한다. 다만
7-Stud의 exact information-set 공간은 매우 커서 10,000딜에서도 재방문이
거의 없다. 그 결과 exact-infoset plug-in 값과 trajectory-relaxed 값이 같은
수준으로 남는다. 이는 exact exploitability나 인증된 상한이 아니라 유한표본
local-deviation 진단값이다.

Resolver 평가에는 817.9초가 걸렸고, 534,182개 local subgame에서 53,418,200
traversal과 949,788,558 node visit을 수행했다. 배포 시 한 번만 풀 subgame을
counterfactual 평가가 여러 branch에서 반복해 풀기 때문에 일반 대전 평가보다
비싸다.

전체 JSON은 `results/memory16_30m_resolver100_gap_10k.json`에 저장했다.
