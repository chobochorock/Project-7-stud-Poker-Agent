# 7-stud teacher distillation: event-history Transformer / MLP students under Policy-LBR

질문: 고정된 C++ MCCFR teacher(abstraction 표)를 신경망이 따라 하도록 학습하면, 같은 Policy-LBR 공격자에게
student가 teacher와 비슷한 ante/hand만 잃는가? 관측 사건 token을 읽는 Transformer와, 같은 정보를 고정
slot으로 펼친 MLP를 비교한다.

## 선행연구와 위치

CFR 전략을 신경망으로 지도학습해 근사하는 것은 확립된 방법이다. [Deep CFR](https://arxiv.org/abs/1811.00164)는
advantage 네트워크가 tabular regret에 "대략 비례"하도록 학습하고(Lemma 1: 도달확률로 정규화된 advantage),
평균전략 네트워크를 따로 학습한다. [SD-CFR](https://arxiv.org/abs/1901.07621)은 평균전략 네트워크의 근사 오차를
없애는 변형이고, [ReBeL](https://arxiv.org/abs/2007.13544)은 subgame 평균전략으로 정책망을 학습한다.
같은 프로젝트의 LightGBM regret ensemble(`agents/lightgbm_regret_ensemble`)도 teacher 표의 학습된 근사를
같은 PolicyLBR로 평가했다. 이 폴더는 새 방법이 아니라, 고정 teacher와 student를 같은 deal·seed·particle의
Policy-LBR로 짝지어 비교하는 통제 실험이다.

## 구성

| 파일 | 역할 |
|---|---|
| `tokens.hpp` | `State`의 관측 가능한 부분만 token/수치/MLP용 고정 벡터로 만든다(상대 비공개 카드·discard·남은 덱은 읽지 않음). |
| `teacher.hpp` | teacher spec: `dt=RUN_DIR`(dynamic-tree checkpoint의 matched hard256 k-means) 또는 `mccfr=ATLAS,MODEL,BUCKET,HANDS`(stud_mccfr 표, 예: K512 power-memory16 10M). 576바이트 학습 record. |
| `export.cpp` | 5구 root(`epoch7::root_for`, heuristic H4)에서 두 좌석이 teacher current/average 정책(손·좌석별 선택)에 ε∈{0,0.1,0.3} 균등 탐색을 섞어 행동하고, 모든 결정을 행동자 시점으로 기록한다. |
| `train.py` | Torch 학습. regret head = 정보집합 최대 \|regret\|로 나눈 teacher regret에 MSE(regret matching은 정보집합별 양의 배율에 불변), average head = teacher 평균전략에 CE. hand%20==0은 검증. |
| `net.hpp` | train.py와 같은 계산의 C++ CPU forward(가중치 `NDNET1`). |
| `evaluate.cpp` | `abstraction_lbr/evaluate.hpp`에 붙인다: model_current/average = student, baseline_current/average = teacher, uniform. `--parity`, `--self-test` 포함. |
| `run.py` | build → export → train → check → lbr → analyze. |

## 실행 (프로젝트 root)

```powershell
python -m agents.cpp_mccfr.neural_distill.run build   --run agents/cpp_mccfr/data/neural_distill_NEW [--teacher SPEC]
python -m agents.cpp_mccfr.neural_distill.run export  --run agents/cpp_mccfr/data/neural_distill_NEW --hands 400000 --seed 20261009
python -m agents.cpp_mccfr.neural_distill.run train   --run agents/cpp_mccfr/data/neural_distill_NEW --models transformer:11 mlp:11 --steps 20000 --threads 4 --parallel 4
python -m agents.cpp_mccfr.neural_distill.run check   --run agents/cpp_mccfr/data/neural_distill_NEW
python -m agents.cpp_mccfr.neural_distill.run lbr     --run agents/cpp_mccfr/data/neural_distill_NEW --seeds 2026100401 2026100402 --pairs 250 --particles 64 --parallel 6
python -m agents.cpp_mccfr.neural_distill.run analyze --run agents/cpp_mccfr/data/neural_distill_NEW --seeds 2026100401 2026100402
```

Torch는 `TORCH_PY`(기본 codex runtime)와 `Toy-Card-Game-Agent/.deep_cfr_deps`를 쓴다. g++는 abstraction_lbr와
같은 플래그(`-O3 -std=c++17 -DNOMINMAX -static-libstdc++ -static-libgcc`)로 빌드해 teacher 행의 재현성을 지킨다.

## 검증

- C++/Torch forward parity(검증 상태 2,000개, 출력·정책 최대 차이 1e-3 미만이어야 통과).
- abstraction_lbr privacy self-test: 상대 비공개 정보를 지운 상태와 같은 정책, LBR 행동 동일, 재현성, 단위.
- export는 매 record에서 teacher 표에서 계산한 current/average가 teacher 자신의 정책 함수와 같음을 검사한다.
- teacher 행은 같은 deal seed에서 이전 평가값을 그대로 재현해야 한다(dt teacher: 2026100401/02×250 pairs,
  64 particles에서 current 2.003375, average 2.4895).

## 지표와 한계

- 주 지표: Policy-LBR 공격자 수익(ante/hand, 높을수록 target에 불리). 근사 BR이며 exploitability가 아니다.
  student-minus-teacher는 같은 deal pair로 짝지은 차이다.
- 보조 지표: 검증 상태와 LBR이 조회한 상태(가상 particle 상태 포함)에서 teacher와의 TV·최선행동 일치.
- 학습 분포는 teacher self-play+ε 탐색이다. LBR은 주로 target의 숨은 카드를 다시 뽑은 가상 상태를 조회한다.
- teacher 하나를 모방하는 실험이며 CFR 갱신을 하지 않는다. student가 teacher보다 덜 잃어도 원게임
  균형에 더 가깝다는 증명이 아니다.
