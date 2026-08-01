# 7-Stud Poker Agent

최근 CFR/MCCFR 구현 구조, 수렴 검증, 7포커 fixed-root 실험 결과는
[CFR_MCCFR_VALIDATION.md](CFR_MCCFR_VALIDATION.md)에 정리되어 있습니다.
현재 C++ power-bucket MCCFR 모델의 정확한 동작은
[CPP_MCCFR_MECHANISM_KO.md](CPP_MCCFR_MECHANISM_KO.md)에 정리되어 있습니다.

## Current betting rules (v3)

- Fourth street is dealt without a betting round.
- Betting starts on fifth street.
- Each player may make at most 1 aggressive action on fifth street, 2 on
  sixth street, and 3 on seventh street.
- `BBING`, `DDADANG`, `QUARTER`, `HALF`, and `FULL` count as aggressive
  actions.
- Checking still removes the player's right to bet or raise for the rest of
  that street.
- Rules v1/v2 MCCFR tables and UCT datasets are incompatible with v3.

The live seventh-street external-sampling variants can be compared with:

```powershell
python -B evaluate_cluster_agent.py --model-dir models\mccfr_plus_v3_live_eval --agent-a mccfr-plus --agent-b heuristic --hands 200 --mccfr-iterations 16 --ante 1000
```

Train and evaluate a reusable v3 MCCFR+ table:

```powershell
python -B mccfr_train.py --output models\mccfr_7th_plus.json --hands 100000 --iterations 16 --cfr-plus
python -B evaluate_cluster_agent.py --model-dir models --agent-a mccfr-plus-table --agent-b heuristic --hands 10000 --ante 1000
```

이 프로젝트는 최대 5명의 사람 또는 AI가 참여할 수 있는 7포커 캐시게임 환경과 에이전트 인터페이스를 제공합니다. 기본 칩은 플레이어당 1000칩이고, 각 라운드의 기본금은 1칩입니다. 캐시 모드가 기본이며 기존 탈락식 진행은 선택적인 토너먼트 모드로 남아 있습니다. 별도의 헤즈업 EV 모드는 스택과 올인을 제거한 rollout 실험에 사용합니다.

## 주요 규칙

- 처음에는 각 플레이어가 비공개 카드 4장을 받습니다.
- 각 플레이어는 카드 1장을 버리고, 카드 1장을 공개합니다.
- 이후 공개 카드 1장을 더 받은 뒤 베팅을 시작합니다.
- 4구, 5구, 6구는 공개 카드로 받고, 마지막 7구는 비공개 카드로 받습니다.
- 가능한 베팅 행동은 `CHECK`, `BBING`, `DDADANG`, `QUARTER`, `HALF`, `FULL`, `CALL`, `FOLD`입니다.
- `BBING`은 아직 아무도 베팅하지 않은 라운드에서 첫 베팅으로만 사용할 수 있습니다. 이후 `DDADANG`은 현재 최고 round bet을 2배로 만듭니다. 한 번 `CHECK`한 플레이어는 같은 베팅 라운드에서 다시 레이즈할 수 없고 `CALL` 또는 `FOLD`만 할 수 있습니다.
- `QUARTER`, `HALF`, `FULL`은 팟의 1/4, 1/2, 전체를 추가로 올리는 행동입니다. 이미 따라가야 하는 콜 비용이 있으면 `콜 비용 + (현재 팟 + 콜 비용) 기준 추가 베팅액`을 냅니다. 예를 들어 팟이 80일 때 `QUARTER`는 20을 내서 팟을 100으로 만들고, 다음 플레이어가 `HALF`로 레이즈하면 콜 비용 20에 `현재 팟 100 + 콜 비용 20`의 절반인 60을 더해 총 80을 냅니다. 피망 7포커의 정확한 계산식은 공개 확인이 어려우므로, 이 프로젝트는 콜 이후 기준 팟을 사용해 쿼터/하프/풀의 상대적 팟 압박률이 일정하게 유지되도록 합니다.
- 매 베팅 라운드는 공개된 패로 가장 높은 족보를 형성한 플레이어부터 시작합니다. 공개 카드가 4장뿐이면 그 4장 안에서 확정된 트리플, 투페어, 원페어, 하이카드 순으로 우선권을 계산합니다. 동점이면 문양 우위 없이 좌석 순서를 따릅니다.
- 백스트레이트(`A, 2, 3, 4, 5`)는 일반 스트레이트보다 높고, 마운틴(`A, K, Q, J, 10`)보다 낮습니다.
- 문양 우위는 없습니다. 족보가 같으면 팟을 나눕니다.
- 올인 플레이어는 자신이 낸 금액 한도 안에서만 팟을 받을 수 있고, 남은 금액은 사이드팟으로 다시 정산합니다.
- 캐시 모드에서는 매 라운드를 독립적인 1000칩 스택으로 시작하고, 라운드별 손익을 세션 누적 손익에 합산합니다. 한 라운드에서 칩을 모두 잃어도 다음 라운드에 다시 1000칩으로 참가합니다.
- 토너먼트 모드에서는 스택을 다음 라운드로 넘기며, 모든 칩을 잃은 플레이어가 탈락하고 한 명만 남으면 종료합니다.
- EV 모드는 정확히 2명만 참가하며 `FULL`을 사용하지 않습니다. 한 베팅 라운드의 `DDADANG`, `QUARTER`, `HALF`는 합계 6회까지이고 `BBING`은 이 횟수에 포함하지 않습니다. 보상은 라운드의 최종 chip net이며 두 플레이어의 합은 0입니다.
- 카드 문양은 특수기호를 쓰지 않고 소문자 `s`, `d`, `h`, `c`로 표기합니다. 예: `sA`, `d10`, `hK`, `c2`

## 파일 구조

- `poker_env.py`: 7포커 게임 환경, 카드/덱/플레이어, 베팅, 쇼다운 규칙
- `main.py`: 터미널에서 플레이어 구성을 선택하고 게임을 실행하는 진입점
- `web_app.py`: 로컬 웹 GUI 서버
- `web_controller.py`: 웹에서 한 단계씩 진행하기 위한 게임 컨트롤러
- `web/static/`: 웹 화면, 동작, 스타일 파일
- `agent/`: 모든 에이전트 구현
- `agent/base.py`: 기본 인터페이스와 무작위 에이전트
- `agent/heuristic_agent.py`: 기존 규칙 기반 에이전트
- `agent/HA1.py`: 균등 belief와 Monte Carlo equity를 사용하는 비학습 에이전트
- `agent/hand_range.py`: 상대의 가능한 초반 히든 2장 조합과 조건부 equity를 계산하는 균등 range 계산기
- `agent/uct_agent.py`: 상대 히든과 미래 deal을 직접 표본화하는 헤즈업 EV UCT 에이전트
- `agent/cluster_agent.py`: 저장된 k-means/GMM responsibility로 component policy 또는 Q를 혼합하는 EV 에이전트
- `agent/claude_belief_br.py`: 상대의 공개 베팅으로 상대 히든 카드 belief를 Bayesian 갱신한 뒤 net-chip EV를 비교해 최선응답하는 particle-belief best response. HA1의 균등 belief를 action-conditioned posterior로 교체한 것이다. 자세한 내용은 `CLAUDE_BELIEF_BR.md` 참고.
- `agent/learning_agent.py`: 공유 데이터베이스를 사용하는 학습 에이전트 예시
- `evaluate_heads_up.py`: 좌석을 교대하는 캐시게임 헤즈업 평가기
- `ev_rollout.py`: stackless EV 균등 rollout과 진단용 Monte Carlo SQLite 테이블 생성기
- `uct_rollout.py`: 실제 UCT 플레이 경로의 root 방문 수와 EV 통계를 SQLite에 수집하는 실행기
- `clustering_train.py`: UCT 데이터로 raw MLP, spherical k-means와 diagonal GMM/EM을 비교 학습하는 실행기
- `clustering_analyze.py`: 저장된 cluster의 support, 유효 개수, geometry와 행동 구조를 JSON/CSV로 분석하는 실행기
- `evaluate_cluster_agent.py`: soft cluster 전략을 실제 stackless EV hand에서 좌석 교대 평가하는 실행기
- `DESIGN_LINEAGE.md`: 아이디어 검토와 실제 구현 결과를 날짜순으로 기록하는 계보
- `exp.py`: 사이드팟 정산 확인용 간단한 데모
- `test_poker_env.py`: 핵심 규칙 테스트
- `LearningAgent_Shared_db.json`: 학습 에이전트가 공유하는 기본 데이터베이스 파일
- `state_log.txt`: 최근 게임 실행 로그

## 실행 방법

프로젝트 폴더로 이동합니다.

```powershell
cd D:\Experiment\Project-7-stud-Poker-Agent
```

무작위 에이전트 2명으로 실행합니다.

```powershell
python -B main.py -p1 random -p2 random
```

HA1과 기존 휴리스틱을 캐시게임 10라운드로 실행합니다.

```powershell
python -B main.py --mode cash --rounds 10 -p1 ha1 -p2 heuristic
```

기존 탈락식 게임은 토너먼트 모드로 실행합니다.

```powershell
python -B main.py --mode tournament --rounds 10000 -p1 heuristic -p2 heuristic -p3 random
```

스택 없는 헤즈업 EV 모드를 일반 게임 루프로 실행합니다. 정수 정밀도를 위해 내부 `1 ante`를 1000단위로 두는 예시입니다.

```powershell
python -B main.py --mode ev --rounds 10 --ante 1000 -p1 random -p2 random
```

30초 동안 양쪽 관점의 균등 액션 rollout을 수집하고 진단용 SQLite 테이블을 만듭니다.

```powershell
python -B ev_rollout.py --seconds 30
```

30분 실측 명령은 다음과 같습니다. SQLite 본체와 WAL의 합이 1GiB에 도달하면 시간보다 먼저 자동 종료합니다. 진행 중 `Ctrl+C`를 눌러도 수집된 내용을 마감하고 요약을 출력합니다.

```powershell
python -B ev_rollout.py --seconds 1800 --max-gib 1
```

잠든 동안 최대 8시간 실행하되 1GiB에서 멈추려면 다음 명령을 사용합니다.

```powershell
python -B ev_rollout.py --seconds 28800 --max-gib 1 --output replays/ev_rollout_overnight_1g.sqlite3
```

결과는 `replays/ev_rollout_{시간}.sqlite3`에 저장됩니다. `q_values`에는 canonical 정보상태, 액션, 방문 횟수, terminal return 합계가 저장되고 `coverage`에는 street, 레이즈 깊이, 액션별 방문 횟수가 저장됩니다. 상태 배열에는 `schema_version`이 포함됩니다. 이 파일은 최종 학습 DB 형식을 결정하기 위한 것이 아니라 정확 상태 테이블의 증가율을 측정하는 실험 결과입니다.

저장된 숫자 배열을 카드와 이름 있는 필드로 해석합니다. 기존 schema version 0 실험 파일도 읽을 수 있습니다.

```powershell
python -B ev_rollout.py --inspect replays/ev_rollout_overnight_1g.sqlite3 --limit 5
```

UCT 수집기를 짧게 확인합니다. 매 실제 의사결정 root에서 512회 rollout하며, 현재 상대 simulation 정책은 균등 무작위입니다.

```powershell
python -B uct_rollout.py --output replays/uct_rollout.sqlite3 --hands 100 --simulations 512
```

기존 8시간 균등 rollout DB를 유지하면서 같은 파일의 별도 `uct_nodes` 테이블에 최대 8시간 추가 수집합니다. `--hands 0`은 핸드 수 제한 없이 시간 제한만 사용한다는 뜻입니다.

```powershell
python -B uct_rollout.py --output replays/ev_rollout_overnight_1g.sqlite3 --hands 0 --seconds 28800 --simulations 512 --max-gib 1 --flush-hands 25 --progress-seconds 30
```

내부 rollout의 상대도 UCT로 선택하고, 충분히 방문한 descendant information set까지 같은 DB에 저장합니다. 결과에는 `seconds_per_search_root`가 포함됩니다.

```powershell
python -B uct_rollout.py --output replays\ev_rollout_uct_vs_uct.sqlite3 --hands 100 --simulations 512 --opponent-policy uct --record-tree-nodes --record-min-visits 8 --progress-seconds 10
```

한 betting information root의 내 카드와 공개 상태를 고정하고 상대 hidden hand와 미래 카드를 다시 뽑으면서, 모든 root action의 95% CI 반경이 기준 이하가 될 때 다음 root로 넘어갑니다. `--simulations`는 최대 예산입니다.

```powershell
python -B uct_rollout.py --output replays\ev_rollout_uct_adaptive.sqlite3 --hands 1000 --simulations 4096 --min-simulations 128 --simulation-batch 128 --epsilon-ante 5 --opponent-policy uct --record-tree-nodes --record-min-visits 8 --progress-seconds 10
```

Adaptive root는 최소 예산까지 action을 균등 표본화한 뒤 CI가 가장 넓은 action에 표본을 추가합니다. 따라서 `uct-v1-ci95`의 root visit count는 정책 target이 아니라 Q 추정용 allocation입니다. 이 DB로 학습한 agent는 visit-policy보다 action-Q 기준으로 평가해야 합니다. 초기 4장 discard/reveal context의 별도 adaptive 수집은 아래 `h4_rollout.py`가 담당합니다.

`--record-tree-nodes`를 함께 쓰면 실제 decision root는 `uct-v1-ci95`, simulation descendant는 `uct-v1-ci95-tree`로 저장됩니다. 두 종류는 visit count의 의미가 다르므로 따로 학습해야 합니다. 이 구분이 도입되기 전에 만든 `replays\uct_adaptive.sqlite3`에는 두 종류가 섞여 있으므로 파일을 동결하고 새 수집은 다른 출력 파일을 사용합니다.

Adaptive allocation 데이터에서 Q 표현만 학습할 때는 policy loss를 끕니다.

```powershell
python -B clustering_train.py --input replays\uct_adaptive.sqlite3 --output models\uct_adaptive_mixed_q --search-version uct-v1-ci95 --opponent-policy uct --simulation-budget 4096 --max-rows 0 --cluster-max-rows 0 --epochs 8 --clusters 128 --top-k 4 --em-iterations 30 --policy-loss-weight 0 --q-normalization pot --device cuda
```

이 모델은 `raw-q`, `kmeans-q`, `gmm-q`로만 실행할 수 있습니다. `*-policy`를 선택하면 trainer metadata를 확인해 오류를 냅니다. `pot` 정규화는 한 상태의 모든 action-Q를 같은 `max(ante, current pot)`으로 나누므로 action 순위를 바꾸지 않으면서 반복 raise의 큰 수치 범위를 줄입니다.

`uct_nodes`는 canonical observation, 좌석, 탐색 버전과 예산별로 여섯 행동의 방문 수, return 합과 제곱합을 저장합니다. 기존 `q_values`와 `coverage`는 수정하지 않습니다. 전체 SQLite와 WAL 크기가 1GiB에 도달하거나 `Ctrl+C`를 누르면 현재 batch를 저장하고 요약을 출력합니다.

### UCT clustering 학습

수집이 끝난 UCT DB에서 다음 세 기준 모델을 같은 train/validation split으로 학습합니다.

1. raw feature를 입력받아 policy와 action-Q를 예측하는 32차원 MLP encoder
2. normalized latent에 대한 spherical k-means와 dot-product top-4 mixture
3. shared whitening latent에 대한 diagonal Gaussian mixture와 EM

먼저 5천 row로 전체 파이프라인을 빠르게 확인합니다.

```powershell
python -B clustering_train.py --input replays\ev_rollout_overnight_1g.sqlite3 --output models\clustering_smoke --max-rows 5000 --cluster-max-rows 4000 --epochs 1 --clusters 16 --em-iterations 2
```

현재 `uct-v1`, random opponent, 512 simulation 데이터 전체를 학습하는 기본 명령은 다음과 같습니다. `--max-rows 0`은 일치하는 row를 전부 사용한다는 뜻입니다.

```powershell
python -B clustering_train.py --input replays\ev_rollout_overnight_1g.sqlite3 --output models\clustering_uct_v1 --max-rows 0 --cluster-max-rows 100000 --epochs 8 --batch-size 4096 --clusters 256 --top-k 4 --em-iterations 30 --seed 7
```

MLP와 diagonal GMM/EM은 CUDA가 있으면 자동으로 GPU를 사용합니다. GMM은 전체 responsibility 행렬을 저장하지 않고 배치별 충분통계만 누적합니다. k-means와 GMM 초기 중심 계산은 scikit-learn으로 CPU에서 실행합니다. `--cluster-max-rows`는 분포 파라미터를 fit할 표본 수만 제한합니다. component별 Q/policy 집계와 validation 평가는 전체 row에 배치 적용합니다.

출력 폴더에는 다음 파일이 생성됩니다.

- `raw_mlp.pt`: MLP encoder와 policy/Q head
- `spherical_kmeans.npz`: centroid, component Q/policy와 support
- `diagonal_gmm.npz`: whitening, Gaussian 파라미터와 component Q/policy
- `metrics.json`: Q MAE, policy cross-entropy, best-action agreement와 search-Q regret 비교

현재 UCT target은 랜덤 상대에 대한 search 결과이며 `Q*`나 균형 전략이 아닙니다. 학습 중 같은 SQLite에 collector를 실행하지 말고, 수집이 끝난 shard 또는 Online Backup으로 만든 snapshot을 입력으로 사용합니다. 생성되는 `models/` 폴더는 Git에서 제외됩니다.

학습된 256개 cluster의 실제 support와 구조를 분석합니다.

```powershell
python -B clustering_analyze.py --model-dir models\clustering_uct_v1
```

요약은 `cluster_analysis.json`, component별 support, 대표 행동과 최근접 cluster는 `spherical_kmeans_clusters.csv`, `diagonal_gmm_clusters.csv`에 저장됩니다. GMM의 `active_components`는 support가 0보다 큰 수이고, `effective_components_entropy`와 `effective_components_simpson`은 support 집중도를 반영한 실질적인 component 수입니다.

저장된 soft cluster mixture를 실제 stackless EV 게임에서 random 상대와 평가합니다. `policy`는 component 정책을 responsibility로 혼합해 표본화하고, `q`는 혼합 Q가 가장 큰 합법 행동을 선택합니다. 같은 deal을 좌석만 바꿔 두 번 실행하며 평균 chip net과 paired 95% 신뢰구간을 저장합니다.

```powershell
python -B evaluate_cluster_agent.py --model-dir models\clustering_uct_v1_full_em_cuda --hands 10000 --opponent random --device cpu
```

결과는 모델 폴더의 `game_evaluation.json`에 저장됩니다. 실제 게임은 한 상태씩 추론하므로 작은 MLP에서는 CUDA보다 CPU가 빠를 수 있습니다. 이 평가는 학습과 같은 random-opponent stackless EV 분포를 측정하며, 일반 cash 게임 성능이나 Nash exploitability를 뜻하지 않습니다.

두 에이전트의 단일 대전을 실행합니다. `--hands`는 짝수여야 하며 총 hand 수의 절반씩 좌석을 바꿉니다.

```powershell
python -B evaluate_cluster_agent.py --model-dir models\clustering_uct_v1_full_em_cuda --agent-a gmm-policy --agent-b heuristic --hands 10000 --device cpu
```

Clustering 없이 raw MLP의 policy/Q head를 직접 평가할 수도 있습니다.

```powershell
python -B evaluate_cluster_agent.py --model-dir models\clustering_uct_v1_full_em_cuda --league raw-policy raw-q kmeans-policy gmm-policy heuristic --hands 10000 --device cpu
```

여러 에이전트의 모든 조합을 대전시키는 round-robin 리그를 실행합니다. 네 참가자면 6대진이므로 아래 명령은 총 60,000 hand입니다.

```powershell
python -B evaluate_cluster_agent.py --model-dir models\clustering_uct_v1_full_em_cuda --league kmeans-policy gmm-policy heuristic random --hands 10000 --device cpu
```

H4에서 자기 4장을 고정하고 상대 패와 미래 카드를 다시 뽑아 discard/reveal 12개를 비교한 뒤 heuristic과 EV 대전을 실행합니다.

```powershell
python -B evaluate_cluster_agent.py --model-dir models\clustering_uct_v1_full_em_cuda --agent-a gmm-policy-h4 --agent-b heuristic --hands 200 --h4-min-rollouts 8 --h4-max-rollouts 64 --h4-batch-size 8 --h4-epsilon-ante 0.25 --device cpu
```

기존 UCT SQLite 파일을 그대로 사용하되 별도 `h4_nodes` 테이블에 H4 action-EV를 수집합니다. 같은 파일에 collector를 동시에 실행하지 않습니다.

```powershell
python -B h4_rollout.py --output replays\ev_rollout_overnight_1g.sqlite3 --model-dir models\clustering_uct_v1_full_em_cuda --contexts 1000 --clusterer gmm --decision policy --min-rollouts 16 --max-rollouts 128 --batch-size 8 --epsilon-ante 0.25 --device cpu
```

단일 대전은 `match_evaluation.json`, 리그는 `league_evaluation.json`에 저장됩니다. 각 대진에는 A 관점의 `average_profit_ante_for_a`, paired standard error와 `ci95_ante_for_a`가 포함됩니다. `uct`도 참가시킬 수 있지만 `--uct-simulations`번 rollout을 매 행동마다 수행하므로 다른 참가자보다 훨씬 느립니다.

사람 1명과 무작위 에이전트 1명으로 실행합니다.

```powershell
python -B main.py -p1 human -p2 random
```

학습 에이전트 2명으로 실행합니다.

```powershell
python -B main.py -p1 learning -p2 learning --db LearningAgent_Shared_db.json
```

압축 데이터베이스를 쓰고 싶으면 `.json.gz` 파일명을 사용합니다.

```powershell
python -B main.py -p1 learning -p2 learning --db LearningAgent_Shared_db.json.gz
```

최대 5명까지 지정할 수 있으며, 빈 자리는 `empty`로 둡니다.

```powershell
python -B main.py -p1 random -p2 learning -p3 human -p4 random -p5 empty
```

터미널에서 질문을 보며 플레이어 구성을 고르려면 `--interactive`를 붙입니다.

```powershell
python -B main.py --interactive
```

규칙 테스트를 실행합니다.

```powershell
python -B -m unittest -v test_poker_env.py test_web_controller.py
```

사이드팟 데모를 실행합니다.

```powershell
python -B exp.py
```

좌석을 번갈아 배정하여 HA1과 기존 휴리스틱의 헤즈업 성능을 측정합니다. 캐시게임에서는 핸드 승률과 평균 chip net을 함께 확인해야 합니다.

```powershell
python -B evaluate_heads_up.py --agent-a ha1 --agent-b heuristic --hands 400 --simulations 128
```

`--mode ev`로 스택 없는 제로섬 net-chip 기준으로 평가합니다. `claude` belief best response가 고정 휴리스틱을 얼마나 착취하는지 이 모드에서 측정합니다. 정수 정밀도를 위해 `--ante 1000`을 사용합니다.

```powershell
python -B evaluate_heads_up.py --agent-a claude --agent-b heuristic --mode ev --ante 1000 --hands 800 --simulations 120
```

코드에서 현재 관측 상태의 균등 상대 핸드 테이블과 대략적인 showdown 승률을 계산할 수 있습니다. 상대 초반 히든 2장 조합은 전수 열거하고, 상대 discard와 아직 배분되지 않은 카드는 조합마다 Monte Carlo 표본화합니다.

```python
result = ha1_agent.estimate_hand_range(state, samples_per_hand=16, seed=7)
print(result["win_probability"], result["tie_probability"], result["equity"])
print(result["opponent_hand_categories"])
print(result["hands"])
```

현재 구현은 헤즈업 전용이며 모든 가능한 히든 2장에 같은 prior를 둡니다. 상대의 이름, 베팅 행동과 discard/reveal 성향은 반영하지 않습니다. 계산량이 크므로 기존 HA1의 매 행동 판단에는 자동 사용하지 않고 분석과 이후 ISMCTS belief 표본화에 사용합니다.

## 로컬 웹 GUI

MVP 웹 GUI는 표준 라이브러리 기반 로컬 서버로 실행합니다.

```powershell
python -B web_app.py --port 8765
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8765
```

웹 화면에서는 `cash` 또는 `tournament` 모드와 `human`, `random`, `heuristic`, `ha1`, `learning`, `empty` 플레이어 타입을 고른 뒤 `Start`를 누릅니다. 모든 플레이어가 AI이면 한 라운드가 자동으로 끝까지 진행됩니다. `human`이 포함되면 카드 버리기/공개하기와 베팅 행동을 화면 버튼으로 선택합니다.

헤즈업에서 첫 카드 선택이 끝난 뒤에는 `Opponent Hand Range`에서 관측자와 조합당 표본 수를 고르고 `Calculate`를 누를 수 있습니다. 52장의 실제 카드로 만든 삼각 격자에서 상대 히든 2장 조합별 내 equity를 색으로 표시하며, known/dead card가 포함된 조합은 회색으로 비활성화됩니다. 칸을 누르면 해당 조합의 승·무·패와 균등 prior를 확인할 수 있습니다. `4`, `16`, `64` 표본 설정은 각각 빠른 확인, 기본 분석, 정밀 분석용입니다.

카드 딜링부터 쇼다운까지의 한 단위를 라운드라고 부릅니다. `Next Round` 또는 `Auto Next`로 다음 라운드를 진행합니다. 캐시 모드는 매번 고정 스택으로 재시작하고 누적 손익을 별도로 표시합니다. 토너먼트 모드는 기존 스택을 유지하며 최후의 한 명이 남으면 종료합니다. 복기 JSON은 캐시 세션 또는 토너먼트 에피소드마다 하나씩 `replays/` 폴더에 저장됩니다.

나중에 UI를 다듬을 때는 주로 `web/static/styles.css`와 `web/static/app.js`를 수정하면 됩니다. 서버/API 흐름은 `web_app.py`, 게임 진행 흐름은 `web_controller.py`에 분리되어 있습니다.

## 새 에이전트 작성 규칙

새로운 에이전트는 반드시 `agent/` 폴더의 새로운 `.py` 파일에 작성합니다. 기존 에이전트 파일에 새 클래스를 끼워 넣지 않습니다.

새 에이전트 클래스는 반드시 `agent/base.py`의 에이전트를 상속받아야 합니다. 일반적으로 `PokerAgent`를 상속합니다.

```python
from agent import PokerAgent


class MyAgent(PokerAgent):
    def choose_action(self, state, valid_actions):
        return valid_actions[0] if valid_actions else None

    def choose_discard_and_reveal(self, hidden_cards):
        return 0, 1

    def learn_from_database(self, database=None):
        return {"agent": type(self).__name__, "trained": False}
```

새 파일을 만든 뒤에는 `main.py`의 `create_agent()`에 새 에이전트 타입을 등록해야 CLI에서 사용할 수 있습니다.

에이전트가 게임 중 받는 `state`에는 자신의 비공개/공개 카드, 상대의 공개 카드, 칩 수, 팟, 콜 금액, 가능한 행동, 베팅 기록이 포함됩니다. 상대 플레이어 이름이나 에이전트 종류는 학습 데이터에 저장하지 않는 것을 원칙으로 합니다.

## 학습 데이터베이스

`LearningAgent`는 같은 파일명을 쓰는 모든 인스턴스가 하나의 데이터베이스를 공유합니다. 한 게임 안에 여러 학습 에이전트가 있어도 같은 DB를 함께 사용합니다.

데이터베이스에는 익명화된 상태, 선택 행동, 가능한 행동, 최종 보상, 궤적이 저장됩니다. 이 구조는 바로 모델은 아니지만, 모델을 만들 수 있는 학습 데이터셋이자 간단한 테이블 기반 가치 추정 저장소입니다.

가독성이 중요하면 `.json`을 사용하고, 파일 크기가 커지면 `.json.gz`를 사용합니다.

## 커밋 메시지 추천

커밋을 만들 때는 변경 목적이 드러나도록 짧게 작성하는 것을 권장합니다. 상황별 예시는 다음과 같습니다.

- `feat: add main entrypoint for poker game`
- `feat: support terminal player type selection`
- `feat: add public-card betting order rule`
- `refactor: keep poker environment separate from CLI`
- `docs: update run instructions`
- `test: cover five random player game`
- `fix: correct side pot settlement`

`__pycache__`, `.pyc`, 임시 테스트 DB 같은 실행 부산물은 커밋 대상에서 제외하는 것을 권장합니다.
