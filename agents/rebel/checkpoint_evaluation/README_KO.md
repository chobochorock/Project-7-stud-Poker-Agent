# 7-Stud ReBeL 체크포인트 실험

## 이 실험이 실제로 재는 것

현재 7-Stud 구현은 완성형 ReBeL이 아니다. 실행 경로는 다음과 같다.

```text
5th: 기존 30M MCCFR blueprint
6th: particle PBS local CFR, leaf = learned V7
7th: particle PBS terminal CFR
```

이 폴더는 위 **partial recursive ReBeL**에서 V7 학습 데이터를 self-play 시도
10,000 hand씩 늘렸을 때 LBR 하한이 변하는지를 자동 측정한다. 각 체크포인트는
10,000 evaluation hands와 240 LBR particles를 기본값으로 사용한다.

다음 두 가지는 아직 빠져 있다.

- 실제 6th local-search 정책으로 갱신한 PBS를 실제 7th 의사결정까지 지속
- V7 target으로 V6를 학습하고 5th를 depth-limited search로 전환

따라서 결과가 좋아져도 원 논문의 완성형 ReBeL 성능이라고 부르면 안 된다.
반대로 결과가 나빠도 ReBeL 자체의 반증은 아니다.

## 실행

첫 10k 체크포인트만:

```powershell
$py = "C:\Users\choi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $py -B rebel_7stud_checkpoints\run_checkpoints.py `
  --checkpoints 10000 `
  --lbr-hands 10000 `
  --lbr-particles 240 `
  --device cuda
```

10k, 20k, 30k 누적 곡선:

```powershell
& $py -B rebel_7stud_checkpoints\run_checkpoints.py `
  --checkpoints 10000 20000 30000 `
  --lbr-hands 10000 `
  --lbr-particles 240 `
  --device cuda
```

중간 checkpoint는 약 8분짜리 2,000-hand screening을 하고 마지막 100k만
10,000 hands로 확정하려면 다음처럼 실행한다.

```powershell
& $py -B rebel_7stud_checkpoints\run_checkpoints.py `
  --checkpoints 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 `
  --quick-lbr-hands 2000 `
  --full-lbr-checkpoints 100000 `
  --lbr-hands 10000 `
  --lbr-particles 240 `
  --device cuda `
  --skip-build
```

이미 2,000 hands로 평가한 checkpoint를 나중에 10,000 hands로 다시 지정하면
더 큰 평가가 기존 행을 교체한다.

완료된 chunk와 평가 JSON은 재사용하므로 중단 후 같은 명령을 다시 실행할 수
있다. 결과는 `run/lbr_checkpoints.json`, 그래프는 `run/lbr_curve.png`에 저장된다.

현재 10k/20k 실행 결과와 해석은 [`RESULTS_KO.md`](RESULTS_KO.md)에 있다.

## 로그 해석

- `v7_hands`: 해당 데이터 chunk에서 처리한 self-play hand 수
- `v7_roots`: fold되지 않고 7th PBS까지 도달해 target을 만든 수
- `samples`: V7 scalar counterfactual-value label 수
- `validation_mae_ante`: held-out public root에서 V7 value 오차
- `lbr_ante_per_hand`: LBR이 평가 정책에서 뽑은 평균 수익. 낮을수록 좋다.
- `ci95`: 위 LBR 하한 추정치의 95% 신뢰구간
