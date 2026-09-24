# 실험 결과

## 설정

```text
outer iterations       12
root CFR+ / outer       5
leaf CFR+ / PBS         20
PBS / outer             45
hidden width            64
online train steps      40 + PBS별 즉시 1회
target Polyak tau       0.2
search exploration      0.02
CPU threads             1
seed                    88101
```

총 실행 시간은 약 294초다. 원시 로그는
[`results/online_v2_12x5x20.jsonl`](results/online_v2_12x5x20.jsonl), 모델은
`results/online_v2_12x5x20.pt`에 저장했다.

## 결과

| 지표 | 첫 보고(5 root iter) | 마지막(60 root iter) |
|---|---:|---:|
| online exact exploitability | 0.8443 | 0.2378 |
| flat CFR+ exploitability | 1.3139 | 0.0306 |
| 새 PBS의 update 전 CFV MAE | 1.7947 | 0.2446 |
| update 후 CFV MAE | 1.1097 | 0.2126 |
| root regret proxy | 0.9128 | 0.1305 |
| replay targets | 90 | 585 |

online exploitability의 최저값은 55 root iteration의 `0.2345`다. root regret
proxy의 log-log 기울기는 약 `-0.774`였고, CFV MAE는 최초 update 전 오차의
약 11.8%까지 감소했다.

![Online PBS experiment](results/online_v2_12x5x20.png)

## Leaf budget 분리 실험

60회까지 학습한 동일 root 정책과 동일 V2 checkpoint를 고정하고 마지막 라운드
resolve 반복만 바꿨다.

| leaf CFR+ iterations | exploitability | leaf regret proxy |
|---:|---:|---:|
| 20 | 0.2378 | 0.9174 |
| 100 | 0.1963 | 0.1882 |
| 500 | 0.1946 | 0.0384 |

`20 -> 100`에서는 분명히 개선됐지만 `100 -> 500`은 거의 포화했다. 따라서
최종 `~0.195` 중 일부는 leaf search 오차였고, 남은 대부분은 root 정책 및
V2 depth-limit 근사의 바닥이다.

## 판정

확인된 것:

- self-play search가 만든 PBS를 즉시 value target으로 쓰는 파이프라인은
  동작한다.
- private-type CFV network가 on-policy PBS 분포에서 실제로 학습된다.
- 그 target network를 호출하는 depth-limited root search의 regret과 exact
  exploitability도 장기적으로 감소했다.
- leaf search와 root/value 오차를 별도로 측정할 수 있다.

확인되지 않았거나 부정적인 것:

- 이 작은 게임에서는 flat CFR+가 훨씬 빠르고 정확하다.
- exploitability는 단조 감소하지 않았다.
- 현재 구현에는 safe resolving과 ReBeL 논문의 전체 수렴 조건이 없다.
- iteration 비교도 online 한 회가 훨씬 비싸므로 online에 유리한 비교가 아니다.

따라서 이 결과는 `ReBeL이 Stud-Leduc에서 flat CFR+를 이겼다`가 아니다. 정확한
결론은 **7-Stud에 필요한 online PBS -> search -> CFV 학습 연결을 작은 exact
게임에서 end-to-end로 검증했고, 현재 근사의 바닥도 측정했다**는 것이다.
