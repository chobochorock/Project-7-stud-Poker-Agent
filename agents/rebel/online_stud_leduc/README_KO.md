# Online PBS search + value learning 실험

이 폴더는 기존 `cpp_mccfr` 구현과 완전히 분리된 Stud-Leduc 기준 실험이다.
목적은 다음 순환이 실제로 작동하는지 exact exploitability로 확인하는 것이다.

실측 결과와 판정은 [`RESULTS.md`](RESULTS.md)에 분리해 두었다.

```text
현재 V2 target network
  -> 1라운드 depth-limited CFR+ search
  -> search 평균 정책으로 2라운드 PBS 생성
  -> 각 PBS를 terminal까지 exact CFR+ resolve
  -> private-rank counterfactual value target 생성
  -> replay에 즉시 저장하고 V2 SGD
  -> target network 갱신
  -> 반복
```

Stud-Leduc의 2라운드는 7-Stud의 `7th`, 1라운드는 `6th`에 대응한다. 따라서
이 실험은 `V7 -> 6th search -> V6` 재귀에서 가장 작은 한 단계를 검증한다.

## 네트워크 계약

입력은 현재 2라운드 PBS다.

- private rank joint posterior `3 x 3`
- 양쪽 공개 rank one-hot
- 누적 투입량
- 이전 라운드의 exact betting history

출력은 scalar 하나가 아니라 양쪽 플레이어의 private rank별 CFV 6개다.

\[
V_2(\beta)=(v_0(J),v_0(Q),v_0(K),v_1(J),v_1(Q),v_1(K)).
\]

학습 표본은 고정 데이터셋에서 오지 않는다. 현재 search 평균 정책이 만든 PBS를
그 자리에서 exact resolve하고 바로 replay와 SGD에 넣는다. 느리게 움직이는 target
network만 다음 root search의 leaf evaluator로 사용한다.

## 중요한 한계

이 코드는 ReBeL 논문의 전체 수렴 보장을 재현한 구현이 아니다.

- root CFR의 leaf에서 private-type CFV를 직접 사용한다.
- 안전 재해결 gadget은 없다.
- PBS coverage를 위해 search policy에 작은 exploration floor를 둔다.
- full-game exploitability는 exact이지만, online 알고리즘 자체의 단조 수렴은
  보장되지 않는다.

따라서 실험의 질문은 `이 구조가 보장되는가`가 아니라 다음 두 가지다.

1. online search가 만드는 PBS에서 V2 오차가 실제로 감소하는가?
2. 그 V2를 쓴 depth-limited policy의 exact exploitability가 flat CFR+와 함께
   감소하는가?

## 실행

```powershell
$py = 'C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

& $py -B .\online_rebel_stud_leduc\online_rebel.py --self-test

& $py -B .\online_rebel_stud_leduc\online_rebel.py `
  --outer-iterations 30 `
  --root-iterations 10 `
  --leaf-iterations 100 `
  --warmup-steps 500 `
  --train-steps 100 `
  --metrics .\online_rebel_stud_leduc\results\online_v2.jsonl `
  --checkpoint .\online_rebel_stud_leduc\results\online_v2.pt

python -B .\online_rebel_stud_leduc\plot_metrics.py `
  --input .\online_rebel_stud_leduc\results\online_v2.jsonl `
  --output .\online_rebel_stud_leduc\results\online_v2.png
```

현재 환경에서는 OpenSpiel/PyTorch가 `$py` 쪽에, `matplotlib`은 시스템
`python` 쪽에 설치되어 있어 그래프 명령만 런타임이 다르다.

마지막 root 정책을 고정하고 leaf solve 예산만 바꾸는 병목 검사는 다음과 같다.

```powershell
& $py -B .\online_rebel_stud_leduc\online_rebel.py `
  --load-checkpoint .\online_rebel_stud_leduc\results\online_v2.pt `
  --eval-only --leaf-iterations 200
```

## 로그

| 필드 | 의미 |
|---|---|
| `exploitability` | 결합 정책의 exact exploitability. 낮을수록 좋다. |
| `baseline_exploitability` | 같은 시점까지 진행한 flat CFR+ 기준선 |
| `value_mae_before_update` | 새 PBS target을 학습하기 전 V2 오차 |
| `value_mae_after_update` | 즉시 학습 후 같은 target에서의 V2 오차 |
| `root_regret_proxy` | 관측 root infoset의 `sum max R+ / T` 진단값 |
| `leaf_reach_weighted_regret_proxy` | PBS reach로 가중한 local solve regret 진단값 |
| `public_belief_states` | 현재 평균 정책이 만든 2라운드 PBS 수 |
| `replay_seen` | 지금까지 생성된 online value target 수 |

그래프의 위쪽은 정책 품질과 CFV 오차, 아래쪽은 regret 진단과 데이터 coverage다.

## 7-Stud로 옮길 때

동일 계약을 street별로 반복한다.

```text
7th PBS -> terminal search -> V7 target
6th PBS -> V7 leaf search -> V6 target
5th PBS -> V6 leaf search -> V5 target
H4 simultaneous choice -> V5/5th resolver target
```

각 `Vt` 입력은 이전 street의 raw belief가 아니라 공개 카드와 행동으로 Bayes
갱신된 **현재 street PBS**다. 출력은 해당 PBS 안의 private type별 CFV다. 7-Stud
에서는 joint range를 통째로 dense matrix로 저장하지 않고, 각 좌석 range factor와
card-compatibility mask를 유지해야 한다.
