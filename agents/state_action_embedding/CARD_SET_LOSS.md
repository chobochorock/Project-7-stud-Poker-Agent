# 최초 4장 복원: 손실함수와 데이터 크기

## 질문

처음 받은 카드조차 새로운 조합에서 복원하지 못한 문제를 먼저 분리한다.
한 아이가 받은 네 장의 카드 이름을 읽어 준 뒤, 어떤 네 장이었는지 묻는 과제다.
이번에는 게임을 잘하는지를 묻지 않고, 들은 카드들을 새 조합에서도 기억하는지 본다.

같은 모델이 세 방식으로 연습한다.

- BCE: 카드마다 내 카드인지 아닌지 맞춘다.
- Ranking: 내 카드 점수를 아닌 카드보다 높게 만든다.
- Softmax CE: 정답 네 장에 각각1/4씩 점수를 배분한다.

답을 낼 때에는 모두 가장 높은 네 장만 고른다. Loss 이외의 모델 구조는 같다.

## 데이터와 예산

- 기존 custom seven-poker v3, 2인 cash,1000chips/ante1. 새50,000hands를 생성한다.
- 기존 low-fold 확률2.625/6.125/8.75%를 고정한다. 재보정하지 않는다.
- 45,000 train/5,000 validation, 별도의 새로운5,000 test hands.
- 작은 대조군은 같은 train에서9,000hands를 뽑는다. 검증/test는 공유한다.
- 카드/사건/보상 전체를5,000hand NPZ shards로 보존한다. 현재 모델에는 최초
  ante2개+private deal8개만85D로 입력한다. 미래 행동/공개/보상은 읽지 않는다.
- Label은 자기 최초4장의52D multi-hot. 상대 private 대상은MASK이다.
- Backbone은 기존 causal Transformer width128/2layers/4heads/FF256, dropout0,
  sinusoidal PE, latent256D. 기존235D head의 첫52D에만 손실을 적용한다.
- Seeds11/22/33, Adam3e-4, batch64, clip1, 각5,000updates.18개 조건 총90,000updates.
- Mini-batch는 해당 train 관측에서 복원 추출한다. 모델당320,000회 표본 제시이며,
  9k/45k 조건의 명목상 반복 횟수는 각각17.78/3.56회다.
- 각 seed의 초기 가중치와 같은 데이터 크기 내 표본 순서를 공유한다.
- 입력 mean/std는 해당 학습 크기의 초기 prefix만으로 계산한다. 검증/test 제외.
- 마지막 고정 step을 평가한다. Test는 모든 조건의 학습 이후에만 점수화한다.

## 손실과 평가

정답 집합T의 크기4, 카드 score z에 대해:

1. BCE:52개 binary cross entropy의 평균. 이번에는pos_weight를 넣지 않는다.
2. Ranking: `mean(softplus(z_negative - z_positive))`,4x48쌍 전체를 사용한다.
3. Softmax CE: `-sum((y/4) * log_softmax(z))`. 정답이4개이므로 최적 CE값은
   0이 아니라log(4)에 접근한다. 각 카드의 보유확률과는 다른 출력 의미다.

주 지표는Top-4 exact-set 및recall@4. 각 hand의 두 관측자를 먼저 평균한다.
유한한 카드 조합이 train/test에 우연히 중복될 수 있어, 보지 못한 unordered
4-card-set에 대해서도 별도 집계한다. Threshold0.5 지표는 BCE의 보조 진단이다.

## 실행

프로젝트 루트에서 기존 런타임을 사용한다.

```powershell
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_card_set_loss_comparison -v
& $py -m agents.state_action_embedding.card_set_loss_comparison collect --out-dir agents/state_action_embedding/data/card_set_50k_20261002
& $py -m agents.state_action_embedding.card_set_loss_comparison train --out-dir agents/state_action_embedding/data/card_set_50k_20261002
python -m agents.state_action_embedding.card_set_loss_comparison report --out-dir agents/state_action_embedding/data/card_set_50k_20261002
```

`python`은 matplotlib가 있는Python3.10이다. 중단된 학습은 같은train 명령에
`--resume`을 붙인다. 코드/데이터/옵션이 바뀌면 거부한다. `last.pt`에는 모델,
optimizer, RNG, 입력 정규화, step과조건이 있다. 수집은 새 output만 허용한다.
Smoke는별도경로에서 `collect --hands 20 --test-hands 4` 후
`train --small-train-hands 4 --steps 2 --seeds 11`로 검사한다.

## 2026-10-02 완료 결과

[최종 test 표](data/card_set_50k_20261002/RESULTS.md)와
[학습 곡선](data/card_set_50k_20261002/learning_curves.png)을 저장했다.
두 데이터 크기, 세 손실, 세 seed의 **18개 모델 모두 독립5,000 test hands의
초기 네 장을100% 복원**했다. 각 모델의10,000 observer view에서 오답0개였다.
학습에서 보지 못한 카드 조합만 평가해도100%였다. 그 조합의 test 비중은
9k/45k 조건에서 각각93.56%/71.53%다. 전체 가능한 조합에 대한 보장은 아니다.

최종 점수는 같았지만, 첫 측정인500updates에서는 차이가 있었다.
아래는 공통5,000 validation hands의 top-4 exact-set, 세 seed 평균 +/- SD다.

| 손실 | 9k train | 45k train |
|---|---:|---:|
| BCE | 99.08 +/- 0.21% | 99.15 +/- 0.15% |
| Pairwise ranking | 99.75 +/- 0.09% | 99.80 +/- 0.04% |
| Soft-target softmax CE | 99.98 +/- 0.02% | 99.97 +/- 0.01% |

9k BCE의 같은500updates 출력에0.5 threshold를 적용하면 exact는32.41%였다.
관측당 false negative0.99277장, false positive0.00003장으로, 이 시점에는
정답 score가0.5에 못 미쳐 누락되는 경우가 주된 문제였다. Top-4는 장수를
알려주어 이 문제를 줄였다. 충분히 학습한 최종 BCE는 threshold 방식도
두 데이터 크기/세 seed 모두 test100%였다. 이전 weighted BCE의 후기 오류와
이번 plain BCE 초기 오류를 같은 현상으로 단정하지 않는다.

이 과제에서는 **새 loss가 없어도 BCE로 새 조합을 복원할 수 있었다.**
Ranking/CE는 이 설정의 초기500updates에서 더 높았지만, 최적 학습률이나
gradient scale을 맞춘 보편적 속도 우위는 검증하지 않았다.9k부터 최종 점수가
포화되어45k의 추가 이점을 최종 정확도로 구별하지 못했다.

수집/audit92.6초,18개 모델의 학습/중간 검증 합계1,241.9초였다.
전처리/최종 test/보고서 시간은 이 학습 시간에서 제외된다. 새 단위시험2개,
smoke/resume,18개 checkpoint 재로딩, 공통 test/유한 logits/원본 shard와
학습 소스 hash 검사를 통과했다. 실제 학습 코드 snapshot은 run의 `source/`다.

## 주의

손실함수의 숫자 크기와 최적 학습률이 같지는 않다. 이번은 동일optimizer 설정의
통제 비교이지 각 방법을 최적으로 튜닝한 비교가 아니다. 큰 데이터 조건은 같은
update에서 반복 횟수가 적다.18k/90k 관측을 한 번씩 도는epoch와update를 혼동하지 않는다.

초기 카드 분포는 이후fold 정책에 영향을 받지 않는다. 저장된 후반 궤적에는
low-fold 편향이 있다. 이번 결과로 후반 memory, 전체 관측 복원, 전략, LBR,
exploitability 또는abstraction 개선을 주장하지 않는다. 추가 게임 학습은 하지 않는다.
