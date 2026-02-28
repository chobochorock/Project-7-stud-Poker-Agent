import sys
import random
import pprint
import argparse
import itertools
from collections import Counter

# 에이전트 파일 임포트 (파일 구조에 맞게 유지)
from agent import PokerAgent
from LearningAgent import LearningAgent 

# --- HumanAgent 클래스 (터미널에서 직접 플레이) ---
class HumanAgent:
    def __init__(self, name):
        self.name = name

    def choose_action(self, state, valid_actions):
        print(f"\n[{self.name}님의 턴]")
        print(f"내 칩: {state['my_chips']} | 콜 필요 금액: {state['call_amount']}")
        print(f"내 패: {state['my_hidden_cards']} | 공개 패: {state['my_public_cards']}")
        print(f"가능한 액션: {valid_actions}")
        
        while True:
            action = input("액션을 입력하세요: ").strip().upper()
            if action in valid_actions:
                return action
            print("잘못된 입력입니다. 가능한 액션 중에서 정확히 입력해 주세요.")

    def choose_discard_and_reveal(self, hidden_cards):
        """사람 플레이어는 임시로 0번을 버리고 1번을 공개합니다."""
        return 0, 1

# --- 카드 및 덱 ---
class Card:
    def __init__(self, suit, rank):
        self.suit = suit # 'S', 'H', 'D', 'C'
        self.rank = rank # '2'~'9', 'T', 'J', 'Q', 'K', 'A'
        
        # 족보 비교를 위한 숫자 값 (2 ~ 14)
        rank_values = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        self.value = rank_values.get(rank) or int(rank)

    def __repr__(self):
        return f"{self.suit}{self.rank}"

class Deck:
    def __init__(self):
        suits = ['S', 'H', 'D', 'C']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.cards = [Card(s, r) for s in suits for r in ranks]
        random.shuffle(self.cards)
    def draw(self):
        return self.cards.pop() if self.cards else None

# --- 플레이어 구조 ---
class Player:
    def __init__(self, name):
        self.name = name
        self.chips = 1000
        self.hidden_cards = []
        self.public_cards = []
        self.is_folded = False
        self.is_all_in = False
        self.invested = 0 # 이번 팟에 넣은 총 칩 (사이드 팟 계산용)
        self.current_bet = 0 # 이번 베팅 라운드에 넣은 칩
        self.hand_score: tuple[int] = (-1,) # 최종 족보 점수

    def receive_card(self, card, is_public=False):
        if is_public:
            self.public_cards.append(card)
        else:
            self.hidden_cards.append(card)

    def discard_and_reveal(self, discard_idx, reveal_idx):
        if len(self.hidden_cards) != 4:
            return False
        indices = sorted([discard_idx, reveal_idx], reverse=True)
        revealed_card = self.hidden_cards[reveal_idx]
        for idx in indices:
            self.hidden_cards.pop(idx)
        self.public_cards.append(revealed_card)
        return True

    def get_all_cards(self):
        return self.hidden_cards + self.public_cards

# --- 족보 판별 모듈 (7장 중 5장 최고 조합 찾기) ---
def get_best_hand(cards):
    """주어진 카드 중 5장을 뽑아 가장 높은 족보의 점수 튜플을 반환합니다."""
    if len(cards) < 5:
        return (0,)

    best_score = (-1,)
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
    return best_score

def evaluate_5_cards(cards):
    values = sorted([c.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    is_flush = len(set(suits)) == 1
    
    is_straight = False
    if len(set(values)) == 5 and values[0] - values[-1] == 4:
        is_straight = True
    elif values == [14, 5, 4, 3, 2]: # 백스트레이트 예외 처리
        is_straight = True
        # values = [5, 4, 3, 2, 1] # in 7stud poker, back is stronger than other straight except for mountain.

    counts = Counter(values)
    freq = sorted([(count, val) for val, count in counts.items()], reverse=True)
    
    if is_straight and is_flush: return (8, values[0])
    if freq[0][0] == 4: return (7, freq[0][1], freq[1][1])
    if freq[0][0] == 3 and freq[1][0] == 2: return (6, freq[0][1], freq[1][1])
    if is_flush: return (5, *values)
    if is_straight: return (4, values[0])
    if freq[0][0] == 3: return (3, freq[0][1], freq[1][1], freq[2][1])
    if freq[0][0] == 2 and freq[1][0] == 2: return (2, freq[0][1], freq[1][1], freq[2][1])
    if freq[0][0] == 2: return (1, freq[0][1], freq[1][1], freq[2][1], freq[3][1])
    
    return (0, *values)

# --- 포커 게임 핵심 로직 ---
class PokerGame:
    players : list[Player]

    def __init__(self, player_names, log_file="state_log.txt"):
        self.players = [Player(name) for name in player_names][:5]
        self.deck = Deck()
        self.ante = 1
        self.current_highest_bet = 0
        self.pot = 0 # 화면 표시용 총 팟 크기 추적

        self.log_file = log_file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 포커 게임 로그 시작 ({len(self.players)}인 플레이) ===\n")

    def log_global_state(self, event_message=""):
        global_state = {
            "pot": self.pot,
            "current_highest_bet": self.current_highest_bet,
            "players": {}
        }
        for p in self.players:
            global_state["players"][p.name] = {
                "chips": p.chips,
                "invested": p.invested,
                "hidden_cards": [str(c) for c in p.hidden_cards],
                "public_cards": [str(c) for c in p.public_cards],
                "is_folded": p.is_folded
            }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- [상태 업데이트] {event_message} ---\n")
            pprint.pprint(global_state, stream=f)

    def start_game(self):
        print(f"=== {len(self.players)}인 게임을 시작합니다 ===")
        # 1. 앤티 징수 및 투자금(invested) 기록
        for player in self.players:
            player.chips -= self.ante
            player.invested += self.ante
            self.pot += self.ante
        
        # 2. 4장씩 딜링
        for _ in range(4):
            for player in self.players:
                player.receive_card(self.deck.draw())
                
        self.log_global_state("초기 4장 딜링 완료")

    def get_valid_actions(self, player):
        if player.is_folded or player.is_all_in: return []
        actions = ["FOLD", "CALL"]
        call_amount = self.current_highest_bet - player.current_bet
        if player.chips >= call_amount + (self.pot * 0.5): actions.append("HALF")
        if player.chips >= call_amount + (self.pot * 0.25): actions.append("QUARTER")
        if player.chips >= call_amount + self.ante: actions.append("BBING")
        return actions

    def get_ai_state(self, player):
        state = {
            "pot": self.pot,
            "my_chips": player.chips,
            "my_hidden_cards": [str(c) for c in player.hidden_cards],
            "my_public_cards": [str(c) for c in player.public_cards],
            "call_amount": self.current_highest_bet - player.current_bet,
            "opponents": {}
        }
        for p in self.players:
            if p != player:
                state["opponents"][p.name] = {
                    "public_cards": [str(c) for c in p.public_cards],
                    "is_folded": p.is_folded,
                    "chips": p.chips
                }
        return state
    
    def apply_action(self, player, action):
        """
        플레이어의 액션을 해석하여 칩을 차감하고 팟에 더합니다.
        판돈이 올라갔는지(Raise) 여부를 True/False로 반환합니다.
        """
        if action == "FOLD":
            player.is_folded = True
            print(f"  -> {player.name}님이 FOLD 했습니다.")
            return False

        # 콜을 하기 위해 내야 하는 기본 금액
        call_amount = self.current_highest_bet - player.current_bet
        raise_amount = 0

        # 베팅 종류에 따른 추가 금액 계산 (한국식 룰: 콜 금액을 더한 가상 팟을 기준으로 계산)
        if action == "HALF":
            raise_amount = int((self.pot + call_amount) * 0.5)
        elif action == "QUARTER":
            raise_amount = int((self.pot + call_amount) * 0.25)
        elif action == "BBING":
            raise_amount = self.ante

        total_bet = call_amount + raise_amount

        # 보유 칩이 부족하면 올인(All-in) 처리
        if player.chips <= total_bet:
            total_bet = player.chips
            player.is_all_in = True
            print(f"  -> {player.name}님이 올인(ALL-IN)! ({total_bet} 칩)")
        else:
            print(f"  -> {player.name}님이 {action}! ({total_bet} 칩 베팅)")

        # 칩 이동: 내 칩 감소 -> 투자금 및 현재 베팅금 증가 -> 중앙 팟 증가
        player.chips -= total_bet
        player.invested += total_bet
        player.current_bet += total_bet
        self.pot += total_bet

        # 누군가 최고액을 갱신했다면(레이즈가 발생했다면) True 반환
        if player.current_bet > self.current_highest_bet:
            self.current_highest_bet = player.current_bet
            return True 
        
        return False

    def play_betting_round(self, active_agents):
        """
        모든 플레이어가 콜을 맞추거나 폴드할 때까지 턴을 반복하는 루프입니다.
        """
        print(f"\n=== 베팅 라운드 시작 (현재 팟: {self.pot}) ===")
        
        # 라운드 시작 시 이번 라운드 누적 베팅액 초기화
        for p in self.players:
            p.current_bet = 0
        self.current_highest_bet = 0

        # 폴드하거나 올인하지 않은, 행동 가능한 플레이어들만 추림
        acting_players = [p for p in self.players if not p.is_folded and not p.is_all_in]
        
        if len(acting_players) <= 1:
            print("  -> 행동 가능한 플레이어가 1명 이하이므로 베팅을 생략합니다.")
            return

        # 행동해야 할 사람 수 (누군가 레이즈하면 다시 인원수만큼 늘어남)
        players_to_act = len(acting_players)
        current_idx = 0

        while players_to_act > 0:
            player = acting_players[current_idx]
            
            # 1명 빼고 전부 폴드했는지 체크 (즉시 라운드 종료)
            survivors = sum(1 for p in self.players if not p.is_folded)
            if survivors <= 1:
                break

            # 이미 폴드했거나 올인한 상태면 턴을 넘김
            if player.is_folded or player.is_all_in:
                current_idx = (current_idx + 1) % len(acting_players)
                continue

            # 가능한 액션 가져오기
            valid_actions = self.get_valid_actions(player)
            if not valid_actions:
                players_to_act -= 1
                current_idx = (current_idx + 1) % len(acting_players)
                continue

            # 에이전트에게 상태를 주고 액션을 받아옴
            state = self.get_ai_state(player)
            agent = active_agents[player.name]
            action = agent.choose_action(state, valid_actions)
            
            self.log_global_state(f"{player.name}의 선택: {action}")
            
            # 액션 적용 및 레이즈 여부 확인
            is_raise = self.apply_action(player, action)
            
            if is_raise:
                # 판돈이 올랐으므로, 방금 베팅한 본인을 제외한 나머지 모두가 다시 턴을 가져야 함
                players_to_act = sum(1 for p in acting_players if not p.is_folded and not p.is_all_in) - 1
            else:
                # 콜이나 폴드라면 한 명이 숙제를 마친 것으로 카운트다운
                players_to_act -= 1

            # 다음 사람으로 순서 넘김
            current_idx = (current_idx + 1) % len(acting_players)
            
        print(f"=== 베팅 라운드 종료 (현재 팟: {self.pot}) ===")

    def resolve_showdown(self):
        """사이드 팟을 고려하여 승자들에게 칩을 분배합니다."""
        print("\n=== 쇼다운 및 팟 분배 ===")
        for p in self.players:
            if not p.is_folded:
                p.hand_score = get_best_hand(p.get_all_cards())
        
        active_investors = [p for p in self.players if p.invested > 0]
        pot_number = 1
        
        while any(p.invested > 0 for p in active_investors):
            current_min_invest = min(p.invested for p in active_investors if p.invested > 0)
            current_pot = 0
            eligible_players = []
            
            for p in active_investors:
                if p.invested > 0:
                    deduction = min(p.invested, current_min_invest)
                    p.invested -= deduction
                    current_pot += deduction
                    
                    if not p.is_folded and deduction == current_min_invest:
                        eligible_players.append(p)
            
            if current_pot == 0: break
            if not eligible_players: continue
                
            best_score = max(p.hand_score for p in eligible_players)
            winners = [p for p in eligible_players if p.hand_score == best_score]
            
            split_amount = current_pot // len(winners)
            winner_names = ", ".join([w.name for w in winners])
            print(f"[팟 {pot_number}] 크기: {current_pot} | 승자: {winner_names} (각 {split_amount} 칩 획득)")
            
            for w in winners:
                w.chips += split_amount
                
            pot_number += 1

    def deal_cards_to_active(self, is_public=True):
        """폴드하지 않고 살아있는 플레이어들에게만 카드를 1장씩 분배합니다."""
        for p in self.players:
            if not p.is_folded:
                card = self.deck.draw()
                if card:
                    p.receive_card(card, is_public=is_public)

    def play_hand(self, active_agents):
        """한 판의 전체 7포커 게임 흐름을 제어합니다."""
        # 1. 앤티 징수 및 4장 딜링
        self.start_game()
        
        # 2. 1장 버리고 1장 공개 (3구 완성)
        for p in self.players:
            agent = active_agents[p.name]
            discard_idx, reveal_idx = agent.choose_discard_and_reveal(p.hidden_cards)
            p.discard_and_reveal(discard_idx, reveal_idx)
            
        self.log_global_state("1장 버리고 1장 공개 완료 (3구 세팅)")

        # 3. 각 스트리트(Street)별 분배 및 베팅 페이즈 정의
        # 형태: (페이즈 이름, 공개 여부)
        streets = [
            ("4구", True),
            ("5구", True),
            ("6구", True),
            ("7구(히든)", False)
        ]

        for street_name, is_public in streets:
            # 이번 턴에 살아있는 사람(폴드하지 않은 사람) 수 확인
            survivors = [p for p in self.players if not p.is_folded]
            
            # 나 빼고 다 죽었으면 묻고 더블로 갈 필요 없이 바로 조기 승리!
            if len(survivors) == 1:
                print(f"\n[{street_name} 페이즈] {survivors[0].name}님을 제외한 모두가 기권했습니다. 조기 승리!")
                break # 카드 딜링을 멈추고 바로 쇼다운(결산)으로 이동
                
            # 카드 딜링
            print(f"\n--- {street_name} 분배 ---")
            self.deal_cards_to_active(is_public=is_public)
            self.log_global_state(f"{street_name} 딜링 완료")

            # 베팅할 수 있는 사람(폴드X, 올인X)이 2명 이상인지 확인
            bettors = [p for p in survivors if not p.is_all_in]
            
            if len(bettors) >= 2:
                self.play_betting_round(active_agents)
            else:
                print(f"  -> 베팅 가능한 플레이어가 부족하여 {street_name} 베팅을 생략하고 턴을 넘깁니다. (올인 발생)")

        # 4. 최종 쇼다운 및 결산
        self.resolve_showdown()
        
        # 결과 출력
        print("\n=== 최종 결과 ===")
        for p in self.players:
            status = "FOLD" if p.is_folded else "ALL-IN" if p.is_all_in else "SURVIVED"
            print(f"{p.name}: {p.chips} 칩 (이번 판 투자금: {p.invested}) | 상태: {status}")



# --- 실행 메인 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7 Poker AI Simulation Environment")
    parser.add_argument('-p1', type=str, default='Human', help='Player 1 Type')
    parser.add_argument('-p2', type=str, default='Human', help='Player 2 Type')
    parser.add_argument('-p3', type=str, default='Empty', help='Player 3 Type')
    parser.add_argument('-p4', type=str, default='Empty', help='Player 4 Type')
    parser.add_argument('-p5', type=str, default='Empty', help='Player 5 Type')
    
    args = parser.parse_args()
    
    def create_agent(agent_type, name):
        agent_type = agent_type.lower()
        if agent_type == 'learning': return LearningAgent(name)
        elif agent_type == 'random': return PokerAgent(name)
        elif agent_type == 'human': return HumanAgent(name)
        return None

    agent_names = ["Player_1", "Player_2", "Player_3", "Player_4", "Player_5"]
    agent_types = [args.p1, args.p2, args.p3, args.p4, args.p5]
    
    active_agents = {}
    for name, a_type in zip(agent_names, agent_types):
        agent = create_agent(a_type, name)
        if agent is not None:
            active_agents[name] = agent
            print(f"[{name}] 참전! (타입: {a_type})")

    if len(active_agents) < 2:
        print("\n[오류] 게임을 시작하려면 최소 2명의 플레이어가 필요합니다.")
        sys.exit()
    

    game = PokerGame(list(active_agents.keys()), log_file="state_log.txt")
    game.play_hand(active_agents)

    print("\ngame complete successfully. You can check the whole processing in state_log.txt.")
