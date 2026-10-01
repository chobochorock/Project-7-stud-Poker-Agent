# 여섯 Action Embedding의 동일 조건 비교

## 쉽게 설명하면

행동을 숫자로 바꾸는 번역기 여섯 개를 만들었는데, 앞 실험에서는 하나만 써 봤다.
이번에는 나머지 다섯 번역기도 같은 문제를 풀게 한다. 원래 숫자를 그대로 읽는
raw 기준선도 같이 표시한다. 작은 모델과 큰 모델, 많은 문제와64hand만 반복해서
외우는 경우를 구분한다. 어느 한 번역기가 좋을 것이라고 미리 가정하지 않는다.

## 고정 조건

- 환경: custom seven-poker v3 heads-up, 표준 casino Stud가 아니다.
- 데이터: `data/separate_20260922/separate_raw.npz`, seed20260922의 low-fold10k.
  Train9000/validation1000hand, 두 관찰자는 같은 partition.
  5/6/7구 fold확률2.625/6.125/8.75%, showdown51.33%; 나머지는 합법행동 균등.
  수집 정책 편향을 유지하며 새 데이터/독립 test를 생성하지 않는다.
- Raw는 여섯 encoder 사전학습에 쓴 동일한85D `prepare(...)["x"]`다.
  사건종류/행동주체/수신자/카드/베팅/구간 one-hot81D와 표준화금액/존재 flags4D.
  저장 파일의 원본6열을 그대로 거리 계산에 쓰는 것이 아니다.
- Frozen encoder: `data/action_window_20260927/{window}/seed{seed}/model.pt`.
  각2500updates x512 context pairs로 학습한32D 표현; 이번에 재학습하지 않는다.
  L1={-1}, L2={-2,-1}, R1={1}, R2={1,2}, S1={-1,1}, S2={-2,-1,1,2}.
  사전학습은 공통의 양끝2칸을 중심 표본에서 제외했다. 후속 학습은 전체 이력을
  사용하므로 사전학습 중심 표본의 coverage와 완전히 같지는 않다. 이번에는 변경하지 않는다.
- 모든 입력은 원래9000hand 빈도로 수치 표준화한다. Raw는85D, encoder는32D에서
  trainable Linear+tanh를 거쳐 GRU/Transformer에 들어간다. Frozen 표현은 역전파하지 않는다.
- GRU1layer, Transformer2blocks/4heads/fixed sinusoidal PE/causal mask.
  Small width32/hidden64, wide128/hidden256. 기존 초기화/Adam0.001/gradclip1 유지.
- 목표:333종 다음 사건 분포의 합법 집합 내 `KL(q||uniform) - log(M)`.
  실제 기록 일치 CE/금액 회귀/RoPE/PPO/CFR/abstraction은 추가하지 않는다.
- Full: 각1000updates x32sequence, seeds11/22/33,9000traininghands.
- Overfit: seed11,64traininghands,3000updates x32sequence. Hand선택은
  `default_rng(seed+500000)`로 고정. 전처리/encoder 사전학습은9000hand 기반이다.
- Train곡선은64hand, validation곡선은고정128hand, 최종평가는전체1000validationhands.
  마지막 checkpoint를 평가하며 best-val/early stopping은 없다.
- 앞 실험의 raw/S1 32개 결과를 재사용하고, 나머지5표현의80개를 새로 학습한다.
  동일 regime/seed의 실제 event exposures도 검사한다. FLOPs/시간 동일 비교는 아니다.

## 지표와 해석

- BET joint 불법확률, 다음 행동주체 정확도.
- 올바른 사건종류/수신자 조건부 카드 불법확률, 합법 카드 KL과 합성손실.
  불법확률 감소만으로 균등분포까지 잘 학습했다고 판단하지 않는다.
- 52카드 균등 기준선과 raw/S1을 모든 관련 그림에 표시한다.
- Full은 hand macro 후3seed 평균/SD, 같은seed의 raw 대비 차이도 저장한다.
  SD는 신뢰구간이 아니다. Overfit은 한seed의 진단이다.
- 반복 사용한 validation으로 여섯 후보를 비교하므로 최상의 결과는 탐색적이다.
  독립 test/새 seed 재검증 없이 선정 모델의 일반화 개선을 확정하지 않는다.
- True-prefix teacher-forced 규칙 예측이지 자유 rollout/전략 성능/exploitability가 아니다.
  주어진 schedule의 합법 후보이며 미래 attention과 비공개 카드 입력은 금지한다.

## 실행

프로젝트 root에서 실행한다. 기존 output은 덮어쓰지 않는다.

```powershell
$py='C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py -m unittest agents.state_action_embedding.test_capacity_legality agents.state_action_embedding.test_transformer_legality agents.state_action_embedding.test_gru_legality agents.state_action_embedding.test_action_window -v
& $py -m agents.state_action_embedding.capacity_legality --out-dir agents/state_action_embedding/data/all_windows_smoke_20260928 --conditions L1 L2 R1 R2 S2 --seeds 11 --hands 8 --full-steps 5 --overfit-steps 5 --log-every 5
& $py -m agents.state_action_embedding.capacity_legality --out-dir agents/state_action_embedding/data/all_windows_legality_20260928 --conditions L1 L2 R1 R2 S2
python -m agents.state_action_embedding.all_windows_report --run agents/state_action_embedding/data/all_windows_legality_20260928 --reference agents/state_action_embedding/data/capacity_legality_20260928
```

결과/곡선/checkpoint는run아래 `regime/window/architecture/size/seedN/`에 저장한다.
매 모델 완료 시 `progress.json`, 전체 완료 시 `summary.json`이 갱신된다.
Checkpoint는 optimizer/RNG/고정encoder를 포함하나 자동 resume CLI는 없다.
기존 raw/S1 원본은 수정하지 않는다. 이전 학습/데이터 생성 비용은 새 실행 시간에 포함하지 않는다.
