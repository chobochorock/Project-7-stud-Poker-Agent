# Heuristic 모방과 PPO 개선 실험 (2026-09-19)

## 결론

**불완전하게 모방한 정책의 손실은 PPO로 크게 줄었다. 그러나 heuristic보다 우월하다는 근거는 아직 없다.**

세 seed 평균 칩 손익은 BC만 사용했을 때 -3.084칩/핸드, PPO 후에는 -0.003칩/핸드였다.
개선 폭은 +3.081칩/핸드이며 paired-deal bootstrap 95% 구간은 [+2.719, +3.451]이다.
반면 PPO 자체 손익의 구간은 [-0.053, +0.049]로 0을 포함한다. 따라서 이번 결과는
"모방 정책보다 개선되어 고정 heuristic과 거의 비김"으로 해석한다.

- [손익·승패 비교 그래프](data/bc_ppo_20260919/comparison.png)
- [모방 정확도·PPO 검증 학습 곡선](data/bc_ppo_20260919/learning.png)
- [전체 수치](data/bc_ppo_20260919/summary.json)
- [실행법과 알고리즘 조건](BC_PPO.md)
- [구현](bc_ppo.py), [검사](test_bc_ppo.py)

## 실제 조건

| 항목 | 조건 |
|---|---|
| 게임 | 기존 seven-poker v3, 2인 cash, 초기 1000칩, ante 1; 표준 카지노 Stud와는 다름 |
| 행동 | 기존 베팅 8종 + 초기 discard/reveal 순서쌍 12종. 두 단계 모두 모델이 실행 |
| 입력 | 행동자 관점의 chance/행동 사건열. 상대 비공개 패·미래·교사 strength는 입력하지 않음 |
| 모델 | 기존 causal Transformer, 2층·4-head·64차원, policy/value head, 86,229 parameters |
| BC 데이터 | heuristic 대 heuristic 12000핸드, 101318결정 |
| BC 분할 | 9600/1200/1200핸드, 결정 수 81045/10127/10146; 양쪽 관점 함께 분할 |
| BC 학습 | 4000 updates, batch 64, Adam 0.0005; 검증 NLL로 선택 |
| PPO 상대 | 고정된 기존 `HeuristicPokerAgent`; self-play 아님 |
| PPO 예산 | seed당 새 궤적 16384핸드, 128 rollout batches, batch당 128핸드 |
| PPO 핵심 설정 | ratio clip 0.2, gamma 1, GAE lambda 0.95, Adam 0.0001, 최대 4 epochs/batch |
| PPO 보상 | 최종 실제 순수익 / 100. 모방 label·heuristic 점수·보상 shaping 없음 |
| 모델 선택 | 학습과 분리한 256 deal pairs의 검증 손익. BC 초기점도 후보에 포함 |
| 테스트 | 학습·검증과 분리한 4000 deal pairs, 자리 교대 8000핸드/정책/seed |
| seed | 11, 22, 33. BC/PPO에 같은 테스트 덱과 행동 RNG seed 사용 |
| 행동 선택 | BC와 PPO 모두 확률적 categorical sampling. BC만 greedy로 평가하지 않음 |

BC/PPO는 이전 action2vec의 상대 행동 head를 자기 정책으로 그대로 쓰지 않았다.
새 actor-view BC를 학습한 후 그 가중치에서 PPO를 시작했다. 기존 게임과 heuristic은
변경하지 않았으며, 학습 정책은 discard/reveal을 heuristic에 위임하지 않는다.
Qwen/Gemma의 가중치나 외부 LLM은 사용하지 않았다.

PPO의 critic과 policy는 backbone을 공유한다. 모방으로 학습된 backbone을 고정하지 않고
함께 개선했으며, value head의 초기 출력은 0이다. PPO에 별도의 BC 유지 loss는 없다.
기존 행동의 old log probability와 GAE를 고정한 상태에서 clipped surrogate를 계산한다.
Chance/상대 사건에는 policy loss를 주지 않는다. 자세한 loss 계수와 KL 중단 조건은
BC_PPO.md 및 각 seed의 config.json에 기록했다.

## 모방 정확도

학습에 쓰지 않은 heuristic 궤적에서 top-1 행동 일치율을 측정했다.
대전에서는 이 top-1을 고정 선택하는 것이 아니라 학습된 확률로 sampling한다.

| 단계 | 학습 빈도 + legal mask 기준선 | BC, 세 seed 평균 |
|---|---:|---:|
| 베팅 | 65.89% | 77.62% |
| discard/reveal 조합 전체 일치 | 10.88% | 66.86% |
| 전체 | 52.88% | 75.08% |

따라서 유의미한 모방은 했지만 완전 복제는 아니다. **이 실험은 완벽한 heuristic 복제본의
PPO 개선이 아니라, 제한된 예산으로 얻은 불완전한 BC 정책의 개선 실험**이다.

## 독립 대전 결과

모든 상대는 고정 heuristic. 표의 신뢰구간은 seed별 return을 평균한 후 4000개의
자리 교대 deal pair를 3000회 재표본추출한 조건부 구간이다. 학습 seed 불확실성 전체를
포함하는 구간이 아니다. Seed별 값도 아래에 별도로 제시한다.

| 정책 | 평균 칩/핸드 | 조건부 95% 구간 | 승률 | 패배율 |
|---|---:|---:|---:|---:|
| Uniform random | -15.796 | [-17.258, -14.408] | 29.42% | 70.57% |
| Heuristic 자체 | 0.000 | [0.000, 0.000] | 50.00% | 50.00% |
| BC | -3.084 | [-3.458, -2.707] | 47.31% | 52.68% |
| BC → PPO, 검증 선택 | -0.003 | [-0.053, +0.049] | 38.58% | 61.41% |
| BC → PPO, 마지막 budget | -0.017 | [-0.072, +0.041] | 38.85% | 61.14% |

나머지는 극소수 무승부다. Heuristic 자체의 정확한 0은 같은 결정적 정책을 같은 덱에서
양쪽 자리로 교대해 합산한 검사 결과이지, 샘플링 오차를 무시한 추정이 아니다.

| Seed | BC | PPO 검증 선택 | PPO 마지막 | 검증 선택 PPO - BC |
|---|---:|---:|---:|---:|
| 11 | -4.2305 | -0.0131 | +0.0549 | +4.2174 |
| 22 | -2.8536 | -0.0016 | -0.0730 | +2.8520 |
| 33 | -2.1686 | +0.0056 | -0.0321 | +2.1743 |

검증 선택 update는 각각 64, 64, 96이다. 테스트 결과로 이 checkpoint를 고르지 않았다.
마지막 update 128도 모두 평가했으며 BC 대비 +3.068칩/핸드 개선,
조건부 95% 구간 [+2.699, +3.440]이었다. 좋은 중간 checkpoint만 제시한 결과는 아니다.

## 무엇이 달라졌나

승률은 오히려 낮아졌으므로 "더 자주 이기게 됐다"고 해석하면 틀리다.

| 결과 분해 | BC | PPO 검증 선택 |
|---|---:|---:|
| 승리한 핸드의 평균 이익 | +3.549칩 | +3.540칩 |
| 패배한 핸드의 평균 손실 | -9.042칩 | -2.229칩 |
| 100칩 이상 잃은 핸드 비율 | 0.7125% | 이번 표본에서 0% |

이번 손익 개선의 핵심은 **큰 손실 감소**로 나타났다. PPO가 더 자주 포기하거나 큰 팟을
피하는 정책을 배웠을 가능성과 일치하지만, 행동별 원인 분해는 아직 하지 않았다.
드문 큰 손실이 앞으로도 절대 없다는 의미는 아니다.

이것만으로 장기 문맥 추론, 카드 카운팅, 상대 hand-range 추정 능력이 생겼다고 주장할
수 없다. 고정 heuristic에 특화된 보수적 정책일 수 있다. 새로운 상대 정책과 다른
스택/규칙에 대한 일반화, 더 높은 모방 정확도에서의 PPO, PPO-from-scratch와의 비교는
이번에 실행하지 않았다. 우선은 "BC 정책 대비 실제 칩 손익 개선"만 확인했다.

## 실행 비용과 저장물

- CPU 실행. Seed당 BC 평균 85.96초, PPO 평균 162.49초. 각 학습 phase의 검증 비용 포함.
- PPO 학습은 각 seed 16384핸드, 실제 learner decision 수는 59303 / 58055 / 58692.
- PPO 선택용 대전은 seed당 9회 × 512 = 4608핸드로, 학습 핸드 수와 별도다.
- 최종 대전은 seed당 5정책 × 8000 = 40000핸드, 세 seed 합계 120000핸드. 평균 약 132.56초/seed.
- 별도 BC 데이터 수집 12000핸드, 약 21.07초. 테스트·수집 비용은 PPO 학습 예산에 섞지 않았다.
- Sampled peak process RSS의 seed 평균: BC 약 451.85 MiB, PPO 약 345.23 MiB. GPU VRAM 측정 아님.
- 기존·신규 단위 검사 11개 통과. 합법 행동, 비공개 정보, 저장/로드, on-policy 확률 재계산, GAE terminal 처리, clipping 및 실제 가중치 갱신을 검사했다.
- 데이터 hand split, actor query에 target 미포함, opponent private-card 차단, chip conservation, heuristic 자리 교대 zero-sum 검사를 통과했다.

각 `data/bc_ppo_20260919/seed{11,22,33}/`에 `bc.pt`, `ppo_best.pt`, `ppo_last.pt`,
`config.json`, `training.json`, `evaluation.json`, `evaluation.npz`가 있다.
원시 테스트 수익은 재집계할 수 있다. Checkpoint는 추론용이며 optimizer/RNG를 포함한
중단 지점 재개용은 아니다. 구현 점검용 smoke는 별도 폴더이며 위 집계에 포함하지 않았다.
