# Hand-only k-means와 abstract perfect recall

2026-10-03. **구현과 비학습 검사만 완료. 사용자 요청에 따라 학습/성능 평가는 실행하지 않았다.**

후속 기록: 사용자가 실행한 결과는2,439hand에서5M 상태 한도로 중단되었다.
원래 설계/명령은 아래에 보존한다. 이후 재개에는 제한된 RAM으로 실행하는
[디스크 저장형 명령](HAND_RECALL_DISK.md)을 사용한다.1M 완료 기록은 아직 없다.

## 질문과 선행연구

현재 hand bucket만 저장할 때 잊는 과거 카드 관측이 학습 병목인가?
같은 frozen hand k-means에 과거 bucket 이력을 추가하고 Local BR-gap, 상태 수,
시간을 비교한다. 새로운 clustering/IRA 기법 제안이 아닌 **기억 보존의 통제 비교**다.

- Waugh et al., [A Practical Use of Imperfect Recall](https://www.cs.cmu.edu/~waugh/publications/sara09.pdf), 2009:
  과거 관측 기억을 버리면 공간이 줄어들지만 perfect-recall 알고리즘의 보장이 그대로
  적용되지 않는다는 배경. 기억을 보존한 모델이 같은 예산에서 항상 더 강하다는 뜻은 아니다.
- Lanctot et al., [Monte Carlo Sampling for Regret Minimization in Extensive Games](https://www.cs.cmu.edu/~waugh/publications/nips09b.pdf), 2009:
  external-sampling MCCFR를 사용한다. 표본 기반 local search/CFR-D/CFR+IRA가 아니다.
- 이전 분할 실험은 [별도 문서](../state_action_embedding/REFINEMENT_EXPERIMENT.md#분할-기준의-수식과-구현상-주의)에
  상세히 기록했다. 이번 실험에서는 split/merge, encoder, 새 clustering을 하지 않는다.

## 무엇을 보존하는가

빨강 `recall`의 정보 key:

```text
(seat, street, legal_action_mask, ante, exact_initial_stack_caps,
 [(first_actor_5, bucket_5), ..., (first_actor_s, bucket_s)],
 full [(street, actor, action), ...],
 exact current pot/highest_bet/raise_count/invested/round_bet/fold/allin)
```

파랑 `baseline`은 **같은 key에서 과거 bucket만 생략**한다. 과거 street의 선행 플레이어,
전체 베팅 이력, 칩 수, 합법 행동집합은 두 모델 모두 정확히 보존한다.
이 비교군은 기존 hand k-means를 이용한 **새 학습 대조군**이다. 기존 `power`의 pot-ratio
bin/행동 요약 key나 기존 30M/100M checkpoint를 그대로 불러온 모델은 아니다.
따라서 기존 실험 수치와 이어 붙이지 않는다. 이 차이를 고정해 과거 bucket 기억만 비교한다.

- 5구부터 게임이 시작한다. 4구 discard/reveal은 양쪽 모두 기존 고정 heuristic이다.
- 5구 bucket은 그때 자기 hidden2/shown3, 상대 shown3만으로 계산한다.
  6구에서는 hidden2/shown4, 7구에서는 hidden3/shown4다. 자기 discard는 기존 확률
  계산에서 unseen deck에서 제외한다. 상대 숨은 패/버린 카드/미래 deck은 key에 없다.
- 7구에서 5/6구 bucket을 재구성할 때 **자기 7번째 카드와 양쪽 미래 공개 카드를 제거**한다.
  전체 공개카드를 그대로 넣으면 미래 정보 누출이 되므로 prefix 불변성 검사로 막았다.
- 기존 규칙의 지불액은 ante/초기 stack과 `(street, actor, action)` 이력에서 정수 단위로
  정확히 재생 가능하다. 이력을 16개로 자르거나 금액을 구간화하지 않는다.
- 과거 선행 플레이어도 보존하므로 각 과거 결정 당시 공개 chip 상태와 actor, legal mask,
  자신의 과거 bucket을 복원할 수 있다. 자기가 과거에 구분하던 추상 정보집합을 잊지 않는다.
- 카드의 원본 rank/suit 자체는 bucket으로 합쳐진다. **원게임 정보를 손실 없이 보존한
  perfect recall이 아니라, 추상화된 카드 관측에 대한 perfect recall**이다.
  원게임의 Nash 수렴이나 낮은 exploitability를 보장하지 않는다.

두 팔 모두 정확한 betting history를 보존하므로 memory-free 옛 모델보다 클 수 있다.
`256 buckets`는 table 전체가256개라는 뜻이 아니다. `(bucket history, betting history)`
조합이 table의 상태 수를 결정하며 상한은 대략256의 거듭제곱 방향으로 커질 수 있다.

## 기존 hand feature와 frozen atlas

입력은 기존 `power_vector`의18차원이다. 순수한9차원 족보 확률만으로 바꾸지 않았다.

1. 최종7장 best-hand category9종의 메이드 확률에 제곱근을 적용한9개 좌표.
2. 완성 패의 primary-rank 평균을 정규화한1개 좌표.
3. 상대 공개 패의 category/최고 rank/최대 같은 suit 수/연속 rank 수4개.
4. 자기 공개 패의 같은4개 요약.

5/6구에는 가능한 완성 카드를 관측 정보에서 샘플링/열거한다. 상한128개이며 관측만으로
결정되는 seed를 사용한다. 7구에는 최종 패가 알려져 있다. 메이드 확률을 실제 상대 hidden
card 정답이나 미래 deck으로 계산하지 않는다. 추가 public 요약도 카드 abstraction 범위다.

`data/power256_selfplay100m_eps20_v1.bin`의 street별 hard k-means256을 두 모델이 공유한다.
SHA256: `2944bed6ef966cf49a412c1ca3aca981f07436b6fb370cb7fb77a954737170a0`.
Atlas 생성 데이터는 과거100M 모델의 epsilon0.2 self-play100k hands, street별 최대50k
표본, seed37301이다. **이번 데이터로 refit한 것이 아니며 과거 정책의 분포 편향이 남는다.**
Run 안에 atlas와 provenance를 복사한다. 모델은 uniform/0 regret부터 새로 학습한다.

## 학습과 Local BR-gap

- 게임: C++ seven-stud v3, HU, 5~7구, ante1000 chips, stack1000 ante,
  player별 raise cap1/2/3, 기존의 유한한 bet action 집합. 전체 자유 금액 no-limit이 아니다.
- signed external-sampling MCCFR. 한 hand = 새5구 deal 하나에서 player0, player1의
  traversal 각각1회, **각 비교군마다** 수행. 1M hands는 각 군2M traversals이며 nodes와 다르다.
- traverser 행동은 모두 확장하고 상대 행동은 현재 regret-matching 정책으로 샘플링한다.
  음수 누적 regret을 유지하고 matching할 때만 positive part를 쓴다. CFR+ clipping 없음.
- 평균전략은 기존 external-sampling SIMPLE 방식: 상대 노드에서 도달한 현재전략을 누적한다.
  추가 iteration linear weighting이나 own-reach를 이중 곱하지 않는다. 현재/평균을 별도 평가한다.
- seed111, evaluation seed307. 두 군은 같은 deal을 사용하나 정책이 달라 분기 수/난수 소비가
  달라질 수 있다. node visits, 전체 학습/평가 시간도 함께 출력한다.
- **매 hand 업데이트 후**, 별도 seed로 생성한5구 첫 결정 정보상태1개를 평가한다.
  두 군/현재·평균 전략은 같은 root, hidden particles, rollout seed를 공유한다.
- 원정보집합에 조건부인32개 hidden worlds를 기존 heuristic-H4 posterior에서 샘플링한다.
  각 합법 행동을 첫 수로 강제한 뒤 양쪽이 해당 snapshot 정책을 따라 terminal까지 진행한다.

```text
Qhat(I,a) = mean_particle terminal_net_ante(force a, then follow sigma)
gap(I)    = max_a Qhat(I,a) - sum_a sigma(a|I) Qhat(I,a)
```

Hidden-world 평균을 구한 **뒤** max를 취한다. 정답 hidden world만 사용하지 않는다.
이 값은 한 번의 행동 변경 이득 추정치이며, 뒤의 모든 행동을 최적화하는 BR/LBR 수익/
exploitability가 아니다. Max에 따른 양의 표본 편향은 이동평균으로 제거되지 않는다.
5구 첫 결정만 평가하므로 6/7구의 모든 정보상태 품질을 직접 대표하지도 않는다.

0 hand와 이후10k마다 고정128 root/32 particle 패널도 측정해 `panel.csv`에 별도 저장한다.
이 패널 평균은 고정 정책의 반복 평가이고, 매-hand 이동평균은 서로 다른 root와 시간의
정책을 섞는다. 둘을 동일한 통계로 해석하지 않는다. 패널 요약은 `plot_summary.json`에 있다.
기존 Python uniform-H4/outcome-sampling 실험은 조건이 달라 여기의 matched baseline이 아니다.

Regret table에 없는 key는 합법 행동에 균등분포를 반환한다. 임의 최근접 table로 fallback하지
않는다. 각 군·정책별 `queries`, `missing`을 기록하므로 gap만 보고 미학습 상태를 오해하지 않는다.
Evaluation은 table/regret/RNG를 변경하지 않는다. 관측 기반 atlas cache만 채운다.

## 실행 명령

PowerShell에서 프로젝트 root로 이동한다. 기본 명령 하나가 **두 비교군 모두** 학습한다.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
$run = 'agents/cpp_mccfr/data/hand_recall_kmeans256_1m_seed111'

# 실행하지 않은 새1M 실험. 자동 C++ 빌드 후 학습한다.
python -m agents.cpp_mccfr.hand_recall train --run $run --hands 1000000 `
  --seed 111 --eval-seed 307 --particles 32 `
  --panel-every 10000 --panel-roots 128 --checkpoint-every 10000 `
  --max-states 5000000

# 기록된 부분까지만 그래프 생성; 학습하지 않는다. 최소100hand 필요.
python -m agents.cpp_mccfr.hand_recall plot --run $run --window 100

# 중단 재개: 동일한 조건. 필요할 때 안전 한도만 올릴 수 있다.
python -m agents.cpp_mccfr.hand_recall train --run $run --hands 1000000 --resume `
  --seed 111 --eval-seed 307 --particles 32 `
  --panel-every 10000 --panel-roots 128 --checkpoint-every 10000 `
  --max-states 5000000
```

Python은 기본 Python3.10의 numpy/matplotlib로 충분하고 Torch는 필요 없다.
빌드에는 기존 g++을 사용하며 이 Windows 환경에 필요한 static-libstdc++/static-libgcc를 넣는다.
Compile/self-test만 하려면 다음을 사용한다. 학습·gap rollout은 하지 않는다.

```powershell
python -m agents.cpp_mccfr.hand_recall build --check
python -m unittest agents.cpp_mccfr.test_hand_recall -v
```

### 비용·중단·복원

- 매 hand1 root, 두 군/두 정책,32 particles이므로, 합법 행동이5개인 예에서는
  약640 rollout/hand,1M에 **약6.4억 rollout**이다. 고정 패널/학습 비용은 별도다.
  완주 시간이나 RAM을 측정한 적은 없다. 이 실행은 평가가 학습보다 비쌀 수 있다.
- `--max-states 5000000`은 두 군 table의 **합계 상태 수** 한도다. 정확한 RAM 한도는 아니다.
  한도에 도달하면 해당 hand를 마친 뒤 atomic checkpoint를 저장하고 종료코드2로 중단한다.
  이것을1M 완주로 표기하지 않는다. RAM 여유를 보고 값을 키워 `--resume`할 수 있다.
- `peak_rss_mib`는 C++ 전체 process의 최고 working set이다. 군별 payload는 key문자열+
  RegretNode 크기만으로, hash table/allocator/cache overhead가 제외된 하한에 가깝다.
- 체크포인트는 두 table, signed regret, 평균전략 누적량, touches, sampling RNG,
  완료 hand 수, 평가 로그의 commit 위치를 포함한다. Atlas/코드 identity가 다르면 재개를 거절한다.
- 기본10k마다 **최신 checkpoint 한 개를 교체**한다. 매10k 대형 파일 사본을 모두 남기지 않는다.
  저장 중에는 이전 checkpoint와 임시 파일이 함께 있어 약2배 저장 공간이 필요하다.
- Ctrl+C/프로세스 오류 뒤 재개하면 마지막 저장 뒤의 CSV 꼬리만 버리고 그 구간을 다시 계산한다.
  복원 파일보다 CSV가 짧으면 오류로 중단한다. 같은 run을 여러 프로세스에서 동시에 실행하지 않는다.
- source/atlas/metric 조건과 실행 파일 SHA를 기록한다. 중단 시 `console.log`, 마지막
  `checkpoint.bin`을 남긴다. 일반 코드 검사는 통과했지만 사용자 지시로 end-to-end 학습 및
  실제 학습 중단/재개 smoke는 **아직 수행하지 않았다**.

## 그래프와 파일

- `per_hand.csv`:1hand당1행, 빨강/파랑의 current/average4 gap, root seed, 조회/누락 수,
  table 수/node 수/통계 payload/process RSS. 모든 raw gap을 보존한다.
- `panel.csv`:0/10k/.../1M의 fixed-root 패널, 같은 열 구조.
- `local_br_gap_linear.png`, `local_br_gap_log.png`: 두 정책 별도 패널, 비교군 중첩,
  log-log와 선형 축, 아래에는 같은 시점 이동평균의 recall/baseline 비율.
- 색: recall **빨강**, k-means current-only 비교군 **파랑**. 산점도를 그리지 않는다.
- 각 full trailing100hand 창의 **min~max 옅은 띠 / 5~95백분위 진한 띠 / 평균 실선**.
  `step=post` 사각 구간 형태다. 첫99hand는 창이 완성되지 않아 이동 통계가 없다.
- 90%는 표본 분포 구간이며 신뢰구간이 아니다. Log에서는0을 표시할 수 없어 그림에만
  명시된 작은 양수로 clip한다. 선형 그림, raw CSV, 평균/분위수 통계에는0을 그대로 남긴다.
- `plot_summary.json`: 마지막 이동 구간,0값 수, 고정 패널의 실제 예산별 평균.
- 이 결과만으로 미래 crossing이나 우열을 확정하지 않는다. 후속 multi-seed/LBR는 별도 요청 사항이다.
