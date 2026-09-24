# 7-Stud ReBeL 체크포인트 결과

## 실험 범위

이 결과는 원 논문의 완성형 ReBeL이 아니라 현재 구현된 partial recursive
경로를 측정한다.

```text
5th = 30M MCCFR blueprint
6th = 64-particle local CFR, 32 iterations, leaf = V7
7th = 64-particle terminal CFR, 100 iterations
평가 = policy-LBR 240 particles, 10,000 paired-seat hands
```

V7 데이터는 self-play 시도 hand 수를 정확히 10,000 단위로 끊었다. 각
checkpoint의 LBR seed는 서로 다르므로 차이의 표준오차는 독립 추정치처럼
합성했다.

## 결과

| V7 self-play hands | labels | held-out V7 MAE | LBR ante/hand | 95% CI |
|---:|---:|---:|---:|---:|
| 10,000 | 31,896 | 5.2657 | 1.3764 | [1.1643, 1.5884] |
| 20,000 | 64,168 | 5.1478 | 1.3664 | [1.1051, 1.6277] |

10k에서 20k의 LBR 변화는 `-0.00994 ante/hand`이고 결합 표준오차는 약
`0.1717`이다. 유의한 개선이 아니다.

기존 1,000-root V7 결과는 `1.2432 +/- 0.1089 ante/hand`였다. 20k-hand
결과와 차이는 `+0.1233`, 결합 표준오차는 약 `0.1722`로 역시 통계적으로
구별되지 않는다.

## 진단

- 10k V7은 constant MAE `5.1881`보다 나쁜 `5.2657`이었다.
- 20k V7은 constant MAE `5.1798`을 `5.1478`로 아주 조금 이겼다.
- label을 두 배로 늘려도 LBR은 움직이지 않았다.
- 20k 평가에서 6th posterior ESS는 `10.61 / 64`로 낮았다.
- 20k 평가에서 `5,281,549`개 V7 query와 `3,173,263`번 blueprint fallback이
  발생했다.

따라서 현재 병목은 단순 label 수보다 다음 구조에 가깝다.

1. 208차원 card marginal summary가 joint private-range correlation을 잃는다.
2. 실제 6th local policy posterior가 다음 실제 7th resolver까지 지속되지 않는다.
3. scalar V7의 held-out 오차가 continuation value 규모에 비해 크다.
4. 5th는 여전히 blueprint여서 ReBeL 학습의 영향을 받지 않는다.

이 결과만으로 ReBeL 자체가 실패했다고 볼 수는 없다. 다만 같은 partial
구조에 V7 데이터를 30k, 40k로 더 붓는 것은 우선순위가 낮다. 다음 유효한
실험은 persistent PBS와 type-wise V7을 먼저 구현한 뒤 같은 10k/20k 프로토콜을
반복하는 것이다.

## 산출물

- `run/lbr_checkpoints.json`: 그래프 원자료
- `run/lbr_00010000.json`, `run/lbr_00020000.json`: 전체 LBR 출력
- `run/lbr_curve.png`: LBR CI와 held-out V7 MAE 곡선
- `run/v7_00010000_hands.pt`, `run/v7_00020000_hands.pt`: TorchScript V7
