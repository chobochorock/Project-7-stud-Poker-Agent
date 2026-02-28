import json
import os
import random
from agent import PokerAgent

class LearningAgent(PokerAgent):
    def __init__(self, name):
        # 부모 클래스(PokerAgent)의 초기화 메서드 실행
        super().__init__(name)
        
        # 클래스명과 에이전트명을 조합하여 DB 파일명 생성 (예: LearningAgent_AI_1_db.json)
        self.db_filename = f"{self.__class__.__name__}_{self.name}_db.json"
        
        # 에이전트가 생성될 때 스스로 DB를 불러옵니다.
        self.memory = self._load_db()

    def _load_db(self):
        """기존 DB 파일이 있으면 불러오고, 없으면 빈 딕셔너리를 반환합니다."""
        if os.path.exists(self.db_filename):
            with open(self.db_filename, 'r', encoding='utf-8') as f:
                print(f"[{self.name}] 기존 학습 데이터베이스를 성공적으로 불러왔습니다.")
                return json.load(f)
        else:
            print(f"[{self.name}] 새로운 학습 데이터베이스를 생성합니다.")
            return {}

    def _save_db(self):
        """현재 메모리를 DB 파일에 저장합니다."""
        with open(self.db_filename, 'w', encoding='utf-8') as f:
            # indent=4 를 주어 사람이 파일 텍스트를 열어봤을 때 읽기 편하게 포맷팅합니다.
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def _state_to_key(self, state):
        """
        딕셔너리 형태의 상태(state)를 DB의 키(Key)로 사용하기 위해 
        정렬된 문자열 형태로 변환합니다.
        """
        return json.dumps(state, sort_keys=True)

    def choose_action(self, state, valid_actions):
        """경험(DB)을 바탕으로 행동을 샘플링합니다."""
        if not valid_actions:
            return None
            
        state_key = self._state_to_key(state)
        
        # DB에 한 번도 경험하지 않은 상태라면, 가능한 액션들의 초기 점수를 0으로 셋팅
        if state_key not in self.memory:
            self.memory[state_key] = {action: 0 for action in valid_actions}
            self._save_db()

        # 탐험(Exploration) vs 활용(Exploitation) 설정
        # 30% 확률로 새로운 시도(랜덤)를 하고, 70% 확률로 과거에 점수가 가장 높았던 액션을 선택합니다.
        exploration_rate = 0.3 
        
        if random.random() < exploration_rate:
            chosen_action = random.choice(valid_actions)
            print(f"[{self.name}] [탐험] 새로운 시도로 '{chosen_action}' 액션을 선택했습니다!")
        else:
            # 활용: 과거 경험 기반 선택
            action_scores = self.memory[state_key]
            
            # 현재 턴에서 유효한(valid) 액션들의 점수만 가져옵니다.
            valid_scores = {a: action_scores.get(a, 0) for a in valid_actions}
            
            # 가장 높은 점수를 찾고, 그 점수를 가진 액션들 중 하나를 고릅니다 (동점 대비)
            max_score = max(valid_scores.values())
            best_actions = [a for a, score in valid_scores.items() if score == max_score]
            chosen_action = random.choice(best_actions)
            
            print(f"[{self.name}] [활용] 과거 경험(최고점: {max_score})을 바탕으로 '{chosen_action}' 액션을 선택했습니다!")

        return chosen_action

    def update_memory(self, state, action, reward):
        """
        행동에 대한 결과(보상)를 에이전트가 스스로 DB에 업데이트합니다.
        """
        state_key = self._state_to_key(state)
        
        # 만약 메모리에 없는 상태/액션이라면 구조를 먼저 만들어줍니다.
        if state_key not in self.memory:
            self.memory[state_key] = {}
        if action not in self.memory[state_key]:
            self.memory[state_key][action] = 0
            
        # 획득한 보상(Reward)을 기존 점수에 누적합니다. (간단한 Q-러닝의 기초 형태)
        self.memory[state_key][action] += reward
        self._save_db()
        print(f"[{self.name}] 경험치 업데이트 완료: {action} 액션으로 {reward}의 보상을 반영했습니다.")
