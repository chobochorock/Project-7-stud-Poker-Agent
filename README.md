# 7-Stud Poker Agent

이 프로젝트는 최대 5명의 사람 또는 AI가 참여할 수 있는 7포커 캐시게임 환경과 에이전트 인터페이스를 제공합니다. 기본 칩은 플레이어당 1000칩이고, 각 라운드의 기본금은 1칩입니다. 캐시 모드가 기본이며 기존 탈락식 진행은 선택적인 토너먼트 모드로 남아 있습니다.

## 주요 규칙

- 처음에는 각 플레이어가 비공개 카드 4장을 받습니다.
- 각 플레이어는 카드 1장을 버리고, 카드 1장을 공개합니다.
- 이후 공개 카드 1장을 더 받은 뒤 베팅을 시작합니다.
- 4구, 5구, 6구는 공개 카드로 받고, 마지막 7구는 비공개 카드로 받습니다.
- 가능한 베팅 행동은 `CHECK`, `BBING`, `QUARTER`, `HALF`, `FULL`, `CALL`, `FOLD`입니다.
- `BBING`은 아직 아무도 베팅하지 않은 라운드에서 첫 베팅으로만 사용할 수 있습니다. `BBING` 이후에는 `BBING`으로 다시 레이즈할 수 없습니다.
- `QUARTER`, `HALF`, `FULL`은 팟의 1/4, 1/2, 전체를 추가로 올리는 행동입니다. 이미 따라가야 하는 콜 비용이 있으면 `콜 비용 + (현재 팟 + 콜 비용) 기준 추가 베팅액`을 냅니다. 예를 들어 팟이 80일 때 `QUARTER`는 20을 내서 팟을 100으로 만들고, 다음 플레이어가 `HALF`로 레이즈하면 콜 비용 20에 `현재 팟 100 + 콜 비용 20`의 절반인 60을 더해 총 80을 냅니다. 피망 7포커의 정확한 계산식은 공개 확인이 어려우므로, 이 프로젝트는 콜 이후 기준 팟을 사용해 쿼터/하프/풀의 상대적 팟 압박률이 일정하게 유지되도록 합니다.
- 매 베팅 라운드는 공개된 패로 가장 높은 족보를 형성한 플레이어부터 시작합니다. 공개 카드가 4장뿐이면 그 4장 안에서 확정된 트리플, 투페어, 원페어, 하이카드 순으로 우선권을 계산합니다. 동점이면 문양 우위 없이 좌석 순서를 따릅니다.
- 백스트레이트(`A, 2, 3, 4, 5`)는 일반 스트레이트보다 높고, 마운틴(`A, K, Q, J, 10`)보다 낮습니다.
- 문양 우위는 없습니다. 족보가 같으면 팟을 나눕니다.
- 올인 플레이어는 자신이 낸 금액 한도 안에서만 팟을 받을 수 있고, 남은 금액은 사이드팟으로 다시 정산합니다.
- 캐시 모드에서는 매 라운드를 독립적인 1000칩 스택으로 시작하고, 라운드별 손익을 세션 누적 손익에 합산합니다. 한 라운드에서 칩을 모두 잃어도 다음 라운드에 다시 1000칩으로 참가합니다.
- 토너먼트 모드에서는 스택을 다음 라운드로 넘기며, 모든 칩을 잃은 플레이어가 탈락하고 한 명만 남으면 종료합니다.
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
- `agent/learning_agent.py`: 공유 데이터베이스를 사용하는 학습 에이전트 예시
- `evaluate_heads_up.py`: 좌석을 교대하는 캐시게임 헤즈업 평가기
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
python -B -m unittest -v test_poker_env.py
```

사이드팟 데모를 실행합니다.

```powershell
python -B exp.py
```

좌석을 번갈아 배정하여 HA1과 기존 휴리스틱의 헤즈업 성능을 측정합니다. 캐시게임에서는 핸드 승률과 평균 chip net을 함께 확인해야 합니다.

```powershell
python -B evaluate_heads_up.py --agent-a ha1 --agent-b heuristic --hands 400 --simulations 128
```

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
