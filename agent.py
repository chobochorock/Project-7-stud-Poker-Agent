import random

class PokerAgent:
    def __init__(self, name):
        self.name = name

    def choose_action(self, state, valid_actions):
        """
        주어진 상태와 가능한 액션을 바탕으로 다음 행동을 결정합니다.
        
        :param state: 현재 게임의 상태 (자신의 패, 상대방 공개 패 등)
        :param valid_actions: 현재 턴에 취할 수 있는 액션 리스트 (예: ['CALL', 'FOLD', 'HALF'])
        :return: 선택된 하나의 액션 (문자열)
        """
        if not valid_actions:
            return None
            
        # [TODO] 여기에 딥러닝 모델의 예측 로직이나 확률 기반 룰을 추가하게 됩니다.
        # 현재는 주어진 가능한 액션 중 무작위로 하나를 선택하도록 기초 뼈대를 잡았습니다.
        chosen_action = random.choice(valid_actions)
        
        print(f"[{self.name}] 에이전트가 고민 끝에 '{chosen_action}' 액션을 선택했습니다!")
        return chosen_action

    def choose_discard_and_reveal(self, hidden_cards):
        """
        초반 4장의 카드 중 버릴 카드 1장과 공개할 카드 1장의 인덱스를 선택합니다.
        (임시로 무조건 0번을 버리고, 1번을 공개하도록 설정)
        """
        # 실제 AI는 카드의 가치를 판단하여 선택하겠지만, 지금은 고정값으로 둡니다.
        return 0, 1
