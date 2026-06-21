# 7-Stud Poker Agent

이 프로젝트는 최대 5명의 사람 또는 AI가 참여할 수 있는 7포커 게임 환경과 에이전트 인터페이스를 제공합니다. 기본 칩은 플레이어당 1000칩이고, 각 판의 기본금은 1칩입니다.

## 주요 규칙

- 처음에는 각 플레이어가 비공개 카드 4장을 받습니다.
- 각 플레이어는 카드 1장을 버리고, 카드 1장을 공개합니다.
- 이후 공개 카드 1장을 더 받은 뒤 베팅을 시작합니다.
- 4구, 5구, 6구는 공개 카드로 받고, 마지막 7구는 비공개 카드로 받습니다.
- 가능한 베팅 행동은 `CHECK`, `BBING`, `QUARTER`, `HALF`, `FULL`, `CALL`, `FOLD`입니다.
- 매 베팅 라운드는 공개된 패로 가장 높은 족보를 형성한 플레이어부터 시작합니다. 공개 카드가 4장뿐이면 그 4장 안에서 확정된 트리플, 투페어, 원페어, 하이카드 순으로 우선권을 계산합니다. 동점이면 문양 우위 없이 좌석 순서를 따릅니다.
- 백스트레이트(`A, 2, 3, 4, 5`)는 일반 스트레이트보다 높고, 마운틴(`A, K, Q, J, 10`)보다 낮습니다.
- 문양 우위는 없습니다. 족보가 같으면 팟을 나눕니다.
- 올인 플레이어는 자신이 낸 금액 한도 안에서만 팟을 받을 수 있고, 남은 금액은 사이드팟으로 다시 정산합니다.
- 모든 칩을 잃은 플레이어는 탈락합니다.
- 카드 문양은 특수기호를 쓰지 않고 소문자 `s`, `d`, `h`, `c`로 표기합니다. 예: `sA`, `d10`, `hK`, `c2`

## 파일 구조

- `poker_env.py`: 7포커 게임 환경, 카드/덱/플레이어, 베팅, 쇼다운 규칙
- `main.py`: 터미널에서 플레이어 구성을 선택하고 게임을 실행하는 진입점
- `web_app.py`: 로컬 웹 GUI 서버
- `web_controller.py`: 웹에서 한 단계씩 진행하기 위한 게임 컨트롤러
- `web/static/`: 웹 화면, 동작, 스타일 파일
- `agent.py`: 모든 에이전트가 따라야 하는 기본 인터페이스와 무작위 에이전트
- `LearningAgent.py`: 공유 데이터베이스를 사용하는 학습 에이전트 예시
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

## 로컬 웹 GUI

MVP 웹 GUI는 표준 라이브러리 기반 로컬 서버로 실행합니다.

```powershell
python -B web_app.py --port 8765
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8765
```

웹 화면에서는 1번부터 5번까지의 플레이어 타입을 `human`, `random`, `learning`, `empty` 중에서 고른 뒤 `Start`를 누릅니다. 모든 플레이어를 `random`으로 두면 한 판이 자동으로 끝까지 진행됩니다. `human`이 포함되면 카드 버리기/공개하기와 베팅 행동을 화면 버튼으로 선택합니다.

나중에 UI를 다듬을 때는 주로 `web/static/styles.css`와 `web/static/app.js`를 수정하면 됩니다. 서버/API 흐름은 `web_app.py`, 게임 진행 흐름은 `web_controller.py`에 분리되어 있습니다.

## 새 에이전트 작성 규칙

새로운 에이전트는 반드시 새로운 `.py` 파일에 작성합니다. 기존 `agent.py`를 직접 수정해서 새 에이전트를 끼워 넣지 않습니다.

새 에이전트 클래스는 반드시 `agent.py`의 에이전트를 상속받아야 합니다. 일반적으로 `PokerAgent`를 상속합니다.

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
