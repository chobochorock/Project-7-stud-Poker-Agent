import json
import os
import random
from agent import PokerAgent  # 구조에 따라 알맞게 임포트 유지

class LearningAgent(PokerAgent):
    # --- [핵심] 클래스 변수로 선언하여 모든 LearningAgent가 하나를 공유합니다 ---
    shared_memory = None
    db_filename = "LearningAgent_Shared_db.json"

    def __init__(self, name):
        super().__init__(name)
        
        # 최초의 LearningAgent가 생성될 때 딱 한 번만 DB를 읽어옵니다.
        if LearningAgent.shared_memory is None:
            LearningAgent.shared_memory = self._load_db()
        
        # 내 개인 메모리 변수가 중앙 공유 메모리를 바라보게 연결합니다.
        self.memory = LearningAgent.shared_memory

    def _load_db(self):
        """단일 공유 DB 파일을 불러옵니다."""
        if os.path.exists(self.db_filename):
            with open(self.db_filename, 'r', encoding='utf-8') as f:
                print(f"[시스템] 중앙 공유 학습 데이터베이스를 성공적으로 불러왔습니다.")
                return json.load(f)
        else:
            print(f"[시스템] 새로운 중앙 공유 학습 데이터베이스를 생성합니다.")
            return {}

    def _save_db(self):
        """공유 메모리 상태를 단일 파일에 저장합니다."""
        with open(self.db_filename, 'w', encoding='utf-8') as f:
            json.dump(LearningAgent.shared_memory, f, ensure_ascii=False, indent=4)

    def _state_to_key(self, state):
        return json.dumps(state, sort_keys=True)

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
            
        state_key = self._state_to_key(state)
        
        # 공유 메모리에 처음 보는 상태라면 0점으로 초기화
        if state_key not in self.memory:
            self.memory[state_key] = {action: 0 for action in valid_actions}
            self._save_db()

        exploration_rate = 0.3 
        
        if random.random() < exploration_rate:
            chosen_action = random.choice(valid_actions)
            print(f"[{self.name}] [탐험] 새로운 시도: '{chosen_action}'")
        else:
            action_scores = self.memory[state_key]
            valid_scores = {a: action_scores.get(a, 0) for a in valid_actions}
            
            max_score = max(valid_scores.values())
            best_actions = [a for a, score in valid_scores.items() if score == max_score]
            chosen_action = random.choice(best_actions)
            
            print(f"[{self.name}] [활용] 과거 경험(최고점: {max_score}) 기반: '{chosen_action}'")

        return chosen_action

    def update_memory(self, state, action, reward):
        """
        행동에 대한 결과(보상)를 중앙 공유 DB에 업데이트합니다.
        """
        state_key = self._state_to_key(state)
        
        if state_key not in self.memory:
            self.memory[state_key] = {}
        if action not in self.memory[state_key]:
            self.memory[state_key][action] = 0
            
        self.memory[state_key][action] += reward
        self._save_db()
        print(f"[{self.name}] 경험치 공유 완료: {action} 액션으로 {reward} 보상 획득")
