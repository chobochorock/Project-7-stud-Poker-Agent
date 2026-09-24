# Project-7-stud-Poker-Agent 구조와 실행

정리일: 2026-09-17. [중심 원리](D:/Experiment/EXPERIMENT_PRINCIPLES.md), [코딩 스타일](D:/Experiment/CODE_STYLE.md), [검토 결과](D:/Experiment/EXPERIMENT_REVIEW_2026-09-17.md).

## 정식 위치

| 폴더 | 내용과 읽기 시작점 |
|---|---|
| `environments/seven_stud/` | Python `poker_env.py`, 공유 C++ `stud_rules.hpp` |
| `agents/common/` | `base.py`, hand-range 계산 |
| `agents/baselines/` | human, heuristic, HA1 |
| `agents/tabular_rl/` | 기존 `learning_agent.py` 및 JSON checkpoint |
| `agents/search/` | UCT, belief BR, rollout 및 관련 테스트 |
| `agents/clustering/` | encoder/k-means 학습, cluster Q-learning, 분석/평가 |
| `agents/state_action_embedding/` | 현재 관측·행동 encoder, forward Skip-gram/CPC 비교, 공통 probe 및 메모리 실측 |
| `agents/python_mccfr/` | Python MCCFR, k-means, Kuhn CFR 및 테스트 |
| `agents/cpp_mccfr/` | 호환되는 C++ MCCFR/5인/Deep CFR traversal/ReBeL probe 파일군 |
| `agents/lightgbm_regret_ensemble/` | 구간별 LightGBM regret ensemble 코드와 README. 학습 결과·모델·LBR 평가는 이 폴더의 `data/` |
| `agents/deep_cfr/` | Python trainer, IPC server, 평가, `compact/` 하위 구현 |
| `agents/rebel/` | value trainer/server, `online_stud_leduc/`, `checkpoint_evaluation/` |
| `agents/evaluation/` | heads-up/5인/stack 평가 및 `br_gap/` |
| `data/` | 공유 replay, runtime 로그 |

데이터·checkpoint·결과는 각 계열의 `data/`; 바이너리는 `bin/`이다. 기존 자체 실행 단위인 `compact`, `online_stud_leduc`, `br_gap`은 각 하위 폴더의 `data/`를 사용한다. 기존 `models/<실험>`은 대응 계열 `data/models/<실험>`으로 옮겼다.

C++ traversal들이 `stud_mccfr.cpp`의 모델/상태/IPC 정의를 include하므로 파일별로 다른 agent 폴더로 찢지 않았다. Python Deep CFR·ReBeL trainer는 별도 계열이며 C++ 공용 엔진을 사용한다. **현재 C++ 규칙은 heads-up용 상태/정산도 포함하므로 Python 다인 환경과 완전히 같은 인터페이스라고 가정하면 안 된다.**

기존 `agent/`, `cpp_mccfr/`, `models/` 내 항목과 root 실험 파일들은 원본으로 연결되는 compatibility 경로다. `main.py`, `web_app.py`, `web_controller.py`와 환경 통합 테스트는 root에 남겼다. 정식 소스는 위 표에서 읽는다. 매핑 전체와 이동 전 크기/소스 hash는 `PROJECT_LAYOUT.json`, 이동 완료 기록은 `LAYOUT_MIGRATION.jsonl`이다.

기존 `agent.*` import는 직렬화/외부 코드 호환 API로 유지했다. 동일 모듈을 구·신 경로 양쪽으로 동시에 import하지 않는다. 원래 있던 사용자 수정과 과거 결과 파일은 되돌리거나 삭제하지 않았다.

## 실행

```powershell
Set-Location D:/Experiment/Project-7-stud-Poker-Agent
python -m unittest discover -p 'test_*.py' -q

$py312 = 'C:/Users/choi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $py312 -m agents.deep_cfr.train_deep_cfr_7th --help
& $py312 agents/deep_cfr/compact/train.py --help
& $py312 agents/rebel/online_stud_leduc/online_rebel.py --help
```

OpenSpiel/PyTorch를 사용하는 일부 스크립트는 Toy 프로젝트에 설치된 CPython 3.12 dependency를 공유한다. 기본 Python 3.10으로 실행하면 NumPy/OpenSpiel binary ABI 오류가 날 수 있다. 두 프로젝트를 서로 독립된 pip package로 만든 것은 아니다.

Deep CFR는 `--run-dir agents/deep_cfr/data/<새 실험명>`을 명시한다. 기존 학습을 재개할 때만 같은 경로와 `--resume`을 사용한다. compact 계열은 `agents/deep_cfr/compact/data/<새 실험명>`을 사용한다. `data`로 이동했다고 checkpoint 형식이 호환되거나 과거 stratified reservoir가 현재 global reservoir로 변환된 것은 아니다.

C++ 현재 소스 빌드 및 self-test:

```powershell
& C:/ProgramData/mingw64/mingw64/bin/g++.exe -std=c++17 -O2 -pthread -static-libgcc -static-libstdc++ agents/cpp_mccfr/stud_mccfr.cpp -o agents/cpp_mccfr/bin/stud_mccfr_review_20260917.exe
& ./agents/cpp_mccfr/bin/stud_mccfr_review_20260917.exe --self-test
```

기존 바이너리는 보존했다. 새 검증 바이너리는 별도 이름이다. C++ 환경 header는 기존 상수/행동 ID 선언 뒤 namespace 안에 include하며, 기존 include 진입점은 링크로 유지한다. 대형 solver 전체를 새 클래스 체계로 재작성하지 않았다.

## 저장과 관리

새 `data/`, `bin/`은 `.gitignore`에 추가했다. 파일을 Git에 대량 추가하기 전에 기존 dirty worktree 및 이동 내역을 직접 검토한다. 링크를 지원하지 않는 압축/복사/checkout에서는 호환 경로가 깨질 수 있으므로 symlink 지원을 확인한다. 새 환경/agent 작업은 정식 위치에 작성하고, 생성 결과는 소스와 섞지 않는다.
