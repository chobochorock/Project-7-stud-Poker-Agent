# 사건 이력에서 관측 복원

## 범위

Custom seven-poker v3, 2인 cash, 초기 1000chips/ante1. 최초 딜링부터
관측 가능한 ANTE/PRIVATE/PUBLIC/DISCARD/REVEAL/BET만 85D로 인코딩한다.
초기 관측 벡터, TURN, 현재 선택할 BET 및 그 지불액, 미래 카드/보상은
입력하지 않는다. 고정 초기 stack은 암묵적인 조건이지 미지의 가변 stack이 아니다.
상대 비공개 카드 대상은 MASK이며 실제 hidden state를 복원하지 않는다.

원본 `data/separate_20260922/separate_raw.npz`: 10k hands, 9k/1k hand split,
fold 5/6/7구 2.625/6.125/8.75%, 나머지 합법 행동 균등, showdown 51.33%.
두 observer와 한 hand의 모든 prefix를 같은 split에 둔다. 원본은 수정하지 않는다.
새 test는 915200000부터 1000hands, uniform/heuristic shift는 각각
916200000/917200000부터 250hands다. Test는 모델 선택 이후 수집하며 재학습하지 않는다.

## 학습

- 전체 235D label을 허용된 6종 prefix와 엔진의 공개 규칙만으로 먼저 audit한다.
  TURN은 정렬 위치에만 사용한다. 상대의 hidden 카드나 recorder label을
  reconstruction parser의 입력에 제공하지 않는다. Audit 허용오차 1e-6.
- 기존 causal HistoryTransformer: 85 -> 32, sinusoidal PE, 2층/4heads/FF64,
  dropout0, 최대128, 마지막 유효 위치의 64D hidden -> 235D reconstruction head.
  새로운 무작위 초기화다. 기존 PPO/skip-gram weight를 가져오지 않는다.
- 카드/Binary와 street/actor는 raw logits로 BCEWithLogits/CE를 계산한다.
  카드 pos_weight 및 입력/회귀 scale은 train에서만 계산한다. Chip 9/count 3은
  train 필드별 표준편차로 표준화하여 MSE를 구하고 원래 단위로 평가한다.
- 그룹 가중치 cards/street/actor/legal/chips/counts=1, all-in=0.1.
  All-in은 보조 지표이며 통과 기준에 넣지 않는다. Threshold는 0.5로 사전 고정한다.
- Adam 3e-4, clip1, batch64. Hand -> observer -> 결정 시점 순서로 균등 표집한다.
  Tiny: seed11, 고정32 train hands, 최대2000steps. 본 실험: seeds11/22/33,
  최대5000steps, 250steps마다 validation, 8번 미개선 시 중단.
  Checkpoint 선택은 weighted group loss의 hand macro 평균이다.
- Last-event MLP는 동일한 출력/loss/예산, 이력 없이 마지막 사건만 읽는다.
  상수 baseline은 train 관측의 필드 평균이다. 파라미터/FLOPs가 같은 비교는 아니다.
- 원본 validation은 재사용된 데이터다. 독립 test라고 부르지 않는다.
  95% bootstrap은 hand를 묶어 재표집하며 해당 학습 모델에 조건부인 불확실성이다.

## 실행

프로젝트 루트에서:

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_observation_reconstruction -v
& $py -m agents.state_action_embedding.observation_reconstruction --out-dir agents/state_action_embedding/data/reconstruction_20261002 --width 128 --tiny-steps 4000
# 중단된 동일 실행 재개: 같은 옵션에 --resume 추가. 코드/데이터/설정 변경은 거부한다.
python -m unittest agents.state_action_embedding.test_observation_reconstruction_report -v
python -m agents.state_action_embedding.observation_reconstruction_report --run agents/state_action_embedding/data/reconstruction_20261002
```

`python`은 sklearn/matplotlib가 이미 있는 Python3.10이며, 학습용 `$py`와 다르다.
처음 검사에는 `--tiny-only`를 사용했다. 같은32 train hands/seed11/2000steps에서
width32의 hidden-card exact는0%, width128은 약91.5%였다. 본 실험은 width128로
변경하고 tiny 검사만4000steps까지 확장했다. 본 학습은 기존5000steps 상한이다.
이는 사전에 정한 tiny 진단을 통한 용량 조정이며 fresh test 결과로 고른 설정이 아니다.
Width128의 head 직전 latent는256D이다. Raw 관측235D보다 작게 압축한 실험은 아니다.

`best.pt`는 모델/정규화/target scale, `last.pt`는 optimizer/step/CPU Torch RNG/
NumPy sampler RNG까지 포함한다. 모델 정의는 이 폴더의 소스를 사용한다.
`metrics.sqlite`는 업데이트별 loss/gradient와 최종 지표를 저장한다.
`*_latents.npz`에는 prefix별 latent, 예측/실제 관측, hand/row/길이가 있다.
Data/manifest와 결과 파일은 원본 데이터와 별도 디렉터리에 둔다.

기본 width32에서는64D, 본 실행 width128에서는256D latent가 저장된다. 각 모델은
직접 불러와 `(prediction_logits, latent) = model(x, lengths)`로 사용할 수 있다.
`decode(logits, statistics)`는 target scale을 원래 관측 단위로 되돌린다.
Packed 이력의 `rows`는 원본 관측 행과 대응한다. `source/`는 본 실행 소스 사본이다.

## 정적 k-means 진단

Encoder와 centroid를 고정한 raw / latent / predicted-raw의 MiniBatchKMeans 비교다.
Train-only 필드 표준화, n_init3/max_iter100, seeds11/22/33, 총128/512buckets.
내 차례 관측만 사용하고 street+실제 합법 집합으로 동일 partition을 구성한다.
한 hand의 관측 수가 많다고 가중치가 늘지 않도록 fit과 점수에 hand 가중치를 쓴다.
각 bucket의 train 관측 평균으로 fresh 관측을 예측하여 카드 Brier, chip/count MAE를
측정한다. 이 prototype은 clustering 평가용이지 행동전략/Q/regret table이 아니다.
실제 사용 bucket 수/미지원 partition coverage도 공개한다. 원본 raw 입력의
관측 보존에 유리한 과제이므로 이 지표에서의 우열을 전략 성능으로 확대하지 않는다.

## 결과를 해석할 때

1. 복원 성공은 전략 성능/균형 수렴의 증거가 아니다. 현재 관측235D는 전체
   perfect-recall 정보집합이 아니며, 상대의 행동 이력이 갖는 믿음을 보존한다고 보장하지 않는다.
2. 실패하면 입력 부족, 구현 오류, 최적화/용량, 분포 이동을 나눈다. Tiny 학습 표본의
   오차와 held-out 오차를 구별한다. 본 학습이 pilot 예산에서 멈춰도 수렴 완료라고 부르지 않는다.
3. Low-fold 정책에 조건부인 데이터다. 합법 상태 전체의 coverage나 가변 stack
   일반화를 입증하지 않는다. 카드/칩을 정답 규칙으로 보정한 점수는 사용하지 않는다.
4. 주요 성능은 내 차례 observer=actor로 보고한다. 전체/상대 차례/street/길이별
   결과도 저장한다. 내 차례 subset의 actor 정답은 상수다.
5. K-means에서 정보가 잘 보존되어도 좋은 regret bucket이라는 보장은 없다.
   과거 7구 전용 k-means 정책과 전체 street latent는 solver/데이터 계약이 달라
   직접 전략 비교할 수 없다. 이번 정적 군집 진단을 LBR/Local BR-gap/exploitability로 부르지 않는다.

## 2026-10-02 완료 결과

[전체 보고서](data/reconstruction_20261002/RESULTS.md),
[실제/예측 예시](data/reconstruction_20261002/EXAMPLES.md),
[복원 그래프](data/reconstruction_20261002/reconstruction_curves.png),
[군집 비교](data/reconstruction_20261002/clustering_comparison.png).

Fresh low-fold1000hands의 내 차례에서3seed 평균:
Transformer legal exact99.867%, chip MAE10.852, hidden exact0%, discard exact26.859%.
Last-event MLP는 legal86.727%, chip MAE7.829, hidden/discard exact0%였다.
따라서 규칙과 일부 과거 사건을 학습했지만 정확한 전체 관측 복원에는 미달한다.
특히 train32hands의 거의 완벽한 카드 암기를 held-out 일반화로 해석하면 안 된다.

512bucket의 latent는 raw 대비 chip prototype MAE11.774 -> 10.387,
count MAE0.1804 -> 0.1228로 개선됐지만 카드 Brier0.043414 -> 0.044519로 악화됐다.
이것은 서로 다른 정보의 보존 차이이며, 전략적 abstraction의 승패가 아니다.
이번 latent는256D이며, 모델 및 centroid를 고정한 정적 실험이다.

## 카드 시점별 후속 진단

`card_memory_diagnostic.py`는 기존 checkpoint를 바꾸지 않고 최초 딜링,
버린 직후, 5/6/7구 첫/마지막 자기 차례 및 discard 이후 사건 거리별로 읽어낸다.
카드 exact-set, 정답 순위, 과잉 예측 수, pos_weight의 분석적 보정 결과를 구별한다.
Setup은 기존 모델에서 supervision하지 않은 시점이므로, 실패를 망각의 증거로
취급하지 않는다. 같은 view가 7구까지 생존한 paired 결과와 available 결과를 저장한다.

별도 무작위 모델로 고정32hand의 초기4장만(weighted/plain BCE), 버린 직후
1장만(weighted BCE/CE) 각각2000steps 과적합시킨다. Seeds11/22/33, 동일 초기화와
batch, 별도128hand validation이며 마지막 고정 step을 보고한다. 기존 multi-task와
과제/학습 분포가 다르므로 원인 분리는 제한적이다. 카드 수를 맞춘 top-k는
별도 진단일 뿐 원래0.5 threshold 점수를 대체하지 않는다.

```powershell
& $py -m unittest agents.state_action_embedding.test_card_memory_diagnostic -v
& $py -m agents.state_action_embedding.card_memory_diagnostic run --out-dir agents/state_action_embedding/data/card_memory_20261002
& $py -m agents.state_action_embedding.card_memory_diagnostic inspect --out-dir agents/state_action_embedding/data/card_memory_20261002
python -m agents.state_action_embedding.card_memory_diagnostic report --out-dir agents/state_action_embedding/data/card_memory_20261002
```

새 경로만 생성하며 과거 run을 덮어쓰지 않는다. 학습용은 위 Python3.12,
보고서용은 matplotlib가 있는 Python3.10이다. Latent 정보량 자체, 전략 성능,
abstraction 우위는 이 진단으로 측정하지 않는다.

완료 결과: [시점별 카드 진단 보고서](data/card_memory_20261002/RESULTS.md).
기존 모델의 discard top-1은 같은 전체 내 차례 분모에서95.42%였으므로,
집합 일치율26.86%를 곧바로 심한 망각으로 해석하지 않는다. Hidden은
장수를 맞춘 ranking도 낮았다. 초기4장만 tiny 학습한6개 모델은 train100%지만
validation top-4는약1%였다. Discard tiny의 BCE/CE는 모두 train100%,
validation69.14%이며 학습에서 본 target만100%/미관측 target0%였다.
관측 복원 불가능성보다는 출력 선택, 일반화, target coverage를 구분해야 한다.
