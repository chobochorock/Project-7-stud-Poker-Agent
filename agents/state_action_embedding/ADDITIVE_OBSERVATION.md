# Action Sum -> Observation Embedding

## 이번에 적용한 것

사용자 제안 `o' = o + a`를 기존 atomic-action skip-gram 데이터에 적용한다.
정확히는 raw 관측과 벡터를 구분하여 다음과 같이 쓴다.

\[
z_0=0,\quad z_t=\sum_{j<t}E_a(e_j),\quad z_{t+1}=z_t+E_a(e_t).
\]

`E_a`는 [이전 실험](ACTION_PATH_EMBEDDING.md)의 고정된 32D SGNS center lookup이다.
ANTE, PRIVATE, PUBLIC, DISCARD, REVEAL, BET 전체 atomic event를 사용한다.
Deal(player,card), Discard(player,card)처럼 사건 종류/상대적 주체/공개된 대상 등을
하나의 토큰으로 구분한다. 상대 비공개 대상은 기존 MASK를 유지한다.
START/STREET/TURN은 합에 포함하지 않는다. 각 hand의 처음부터 누적하므로 별도 초기
관측 벡터는 사용하지 않는다. action 벡터의 재학습/정규화/중심화는 하지 않는다.

이 합은 우선 **관측 가능한 이력의 요약**이지, 정확한 현재 관측의 정답은 아니다.
현재 관측만 받는 `235 -> 128 ReLU -> 128 ReLU -> 32` MLP `E_o`를 추가하여
`E_o(o_t)`가 이 합을 재현하도록 학습한다. 50,848 parameters, Adam lr=0.001,
batch=512, 2,000 optimizer updates이며 최종 step을 사용한다.
학습셋 평균/표준편차만으로 입력과 좌표별 target loss를 표준화한다.
보상, hidden opponent cards, history는 MLP 입력에 넣지 않는다.
합은 **pseudo-target**이며 환경의 ground truth라고 부르지 않는다.

## 비교 조건과 필수적인 정정

| 조건 | 누적 벡터 | 목적 |
| --- | --- | --- |
| `sum` | `sum(E_a(e_j))` | 요청한 기본 합산 |
| `sum_pe` | `sum(E_a(e_j) + p_j)` | 단순 positional embedding 추가 검사 |
| `rotary_sum` | `sum(R_j E_a(e_j))` | 위치와 action을 결합하는 추가 대조군 |

단순 PE 추가 후 합산은 순서를 구별하지 못한다.
`sum(a_j + p_j) = sum(a_j) + sum(p_j)`이므로 같은 길이에서 action 순서만 바꾸면
결과가 같다. Transformer에서는 그 뒤의 attention 등이 token과 position을 결합하지만,
여기서는 그런 비선형 연산이 없다. PE는 길이 정보를 추가할 수 있을 뿐이다.

회전 대조군은 각 좌표쌍을 위치 `j`에 따라 회전시킨다.
`theta(j,k) = j * 10000^(-2k/32)`일 때
`(x,y) -> (x*cos(theta)-y*sin(theta), x*sin(theta)+y*cos(theta))`이다.
각 action의 norm은 보존하지만 위치가 다르면 방향이 달라져 순서에 민감해질 수 있다.
[RoPE](https://arxiv.org/abs/2104.09864)의 위치별 회전만 참고한 검사이며,
RoFormer/attention을 구현한 것이 아니다. 모든 순열을 구별한다는 보장도 없다.

## 데이터와 누출 방지

- 기존 `data/separate_20260922/separate_raw.npz`: 10,000 heads-up hands,
  custom seven-poker v3, stack=1,000, ante=1. 표준 카지노 Stud와 다르다.
- 데이터 seed=20260922, hand 단위 train 9,000 / validation 1,000.
  두 관찰자 시점은 항상 같은 split에 속한다. 새로운 독립 test는 만들지 않았다.
- 5/6/7구 fold=2.625%/6.125%/8.75%, 나머지는 legal action 균등 선택,
  discard/reveal 무작위. 기존 showdown 비율 51.33%. 패의 강도에 따른 행동은 아니다.
- SGNS seed=11/22/33 각각 사용. 기존 teacher는 2,500 updates, window=2,
  negative=5, batch=512로 학습되었다. 이번에는 teacher 비용을 재지불하지 않는다.
- 관측은 의사결정 직전 TURN에 대응한다. 그 뒤 선택한 BET는 현재 target에서 제외한다.
  이후 카드/보상/상대 비공개 대상도 사용하지 않는다.
- 168,564개 관측 중 168,384개 유효 prefix를 저장한다. 관측 MLP train=151,578,
  val=16,806. validation prefix coverage=98.9403%.
- 학습 vocabulary 밖 action을 하나라도 포함한 prefix는 제외한다. UNK를 0벡터로
  대체한 후 정상 데이터처럼 평가하지 않는다. vocabulary는 exhaustive하지 않다.
- 고정 1,024개 prefix를 기존 symbolic replay로 재생하여 관측 카드와 stack 일치를
  검사한다. 모든 TURN의 stage/actor/선택 action 정렬도 검사한다.

## 무엇을 측정하는가

1. **MLP target fit**: 고정 4,096 validation 관측의 좌표 표준화 MSE.
   평균 target만 출력하는 baseline도 기록한다. 이것은 의미 정확도가 아니다.
   조건마다 target와 분산이 달라 MSE만으로 세 표현의 품질을 순위 매기면 안 된다.
2. **증분 일치**: 같은 hand의 인접 관측 4,096쌍에서
   `sum(||E_o(o')-E_o(o)-(z'-z)||^2) / sum(||z'-z||^2)`.
   변화 없는 encoder는 1이다. 여러 event가 두 관측 사이에 들어갈 수 있다.
   다음에 나올 action/chance의 예측 성능을 측정한 것은 아니다.
3. **선형 probe**: 누적 벡터 또는 학습된 관측 MLP를 고정한 뒤,
   train 관측 20,000개로 ridge regression을 학습한다. 카드 4그룹, pot/양측 stack을
   검증한다. card recall@k의 k는 각 그룹의 실제 관측 카드 수이다. 카드 수까지
   예측하거나 전체 패를 정확히 복구한 비율이 아니다. chip MAE는 pot/양측 stack의
   절대오차 평균이다. 이 probe loss는 관측 MLP 학습에 쓰이지 않는다.
4. **대조군**: 같은 32D 무작위 action lookup의 합, street+prefix length만의 probe.
   현재 raw observation은 검사 대상 카드를 직접 포함하므로 정보의 상한이다.
   관측 MLP도 이 raw 입력을 받으므로 prefix vector보다 높은 probe 성능이 나와도
   action sum이 원래 그 정보를 완전히 보존했다는 뜻은 아니다.
5. **경로 독립성**: 이전 `action_paths_20260924/paths.npz`의 235D 관측 양 끝이
   완전히 같은 서로 다른 길이의 합법 경로 8,192쌍을 검사한다. 시작 prefix 길이는 18.
   모든 시드에 동일한 경로를 재사용한 대수적 진단이며 신규 test 성능이 아니다.
   `||delta_A-delta_B|| / ((||delta_A||+||delta_B||)/2)`를 보고한다.
   같은 관측을 하나의 벡터로 표현하며 모든 증분을 정확히 만족하려면 이 값이 0이어야 한다.
   두 증분을 하나의 공통 벡터로 맞추는 최적 pair-average MSE는
   `mean((delta_A-delta_B)^2)/4`이므로 양수이면 동시에 완벽하게 맞출 수 없다.
   이 값은 전체 데이터셋 MSE의 직접적인 하한은 아니다.
6. **순서 검사**: 별도로 512개 경로를 역순으로 재배열하여 합의 변화를 검사한다.
   역순이 합법적인 게임 경로라는 주장은 하지 않는다.

다른 betting history가 같은 현재 관측으로 끝나도 정보집합은 다를 수 있다.
경로 독립성이 깨졌다는 결과는 현재 관측만의 정확한 embedding 가정을 반박하지만,
전략적으로 유용한 이력 표현의 가능성까지 부정하지 않는다.
현재 데이터는 특정 low-fold 정책의 분포이며, seed SD는 새 게임/정책에 대한 CI가 아니다.
승률/LBR/exploitability/CFR abstraction 성능을 측정하지 않았다.

## 실행 및 저장 파일

프로젝트 루트에서 실행한다. 기존 run을 덮어쓰지 않도록 새 출력 폴더를 지정한다.

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
$py = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$run = 'agents/state_action_embedding/data/additive_observation_new'
foreach ($seed in 11,22,33) {
    & $py -m agents.state_action_embedding.additive_observation --out-dir $run --seed $seed
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed' }
}
& $py -m unittest agents.state_action_embedding.test_additive_observation -v
# This machine's default Python has Matplotlib; no Torch needed for plotting.
python -m agents.state_action_embedding.additive_observation_report --out-dir $run
```

`seedN/sum_targets.npz`: `state_row`로 원본 raw 관측 행을 참조하고 `z`에
32D float32 합을 저장한다. `prefix_length`, `split`도 함께 저장한다.
원본 raw 관측을 복제하지 않는다. PE 조건은 동일한 action 벡터/위치에서 재생성된다.
`seedN/{sum,sum_pe,rotary_sum}.pt`: 모델, 정규화 통계, optimizer/RNG 상태.
현재 CLI는 재개 기능을 제공하지 않지만 inference 로드는 지원한다.

```python
from agents.state_action_embedding.additive_observation import load_encoder
from agents.state_action_embedding.experiment import torch

encoder = load_encoder("agents/state_action_embedding/data/additive_observation_20260924/seed11/sum.pt")
# observation: torch.float32, shape [batch, 235], original unstandardized raw format
with torch.no_grad():
    z = encoder(observation)
```

원본 corpus/SGNS/code hash, data coverage, 학습 곡선, probe, algebra audit를 JSON으로
저장한다. `resources.json`은 전체 seed 프로세스 작업 구간의 10ms sampled CPU RSS이며
정확한 tensor peak나 GPU 메모리 측정은 아니다.

실제 완료 결과: [RESULTS.md](data/additive_observation_20260924/RESULTS.md).
완전 일치가 아닌 내적/cosine과 길이 일치 negative를 확인한
[후속 진단 및 실행법](data/additive_observation_20260924/similarity/RESULTS.md)도 제공한다.
[방향 집중·32D 차원 사용 진단](data/additive_observation_20260924/geometry/RESULTS.md)은
action/누적합/관측 MLP를 구분하고 train-mean 제거 효과 및 연산자 대안을 정리한다.
다음 단계로 action을 증분 loss와 공동 학습하는 방법은 가능하지만 이번에는 적용하지 않았다.
