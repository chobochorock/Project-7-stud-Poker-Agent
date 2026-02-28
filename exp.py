import random
import itertools
from collections import Counter
import pprint

# --- 카드 및 덱 (이모지 제거, 랭크 값 추가) ---
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

# --- 플레이어 구조 (투자금 관리 추가) ---
class Player:
    def __init__(self, name):
        self.name = name
        self.chips = 1000
        self.hidden_cards = []
        self.public_cards = []
        self.is_folded = False
        self.is_all_in = False
        self.invested = 0 # 이번 게임에 팟에 넣은 총 칩 (사이드 팟 계산용)
        self.hand_score = None # 최종 족보 점수

    def get_all_cards(self):
        return self.hidden_cards + self.public_cards

# --- 족보 판별 모듈 (7장 중 5장 최고 조합 찾기) ---
def get_best_hand(cards):
    """주어진 카드(최대 7장) 중 5장을 뽑아 가장 높은 족보의 점수를 반환합니다."""
    if len(cards) < 5:
        return (0,)

    best_score = (-1,)
    # 7장 중 5장을 선택하는 모든 조합(21개)을 확인
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5_cards(combo)
        if score > best_score:
            best_score = score
    return best_score

def evaluate_5_cards(cards):
    """
    5장 카드의 족보를 평가하여 비교 가능한 튜플을 반환합니다.
    (족보 랭크, 타이브레이커1, 타이브레이커2, ...)
    족보 랭크: 8=스트레이트플러시, 7=포카드, 6=풀하우스, 5=플러시, 4=스트레이트, 3=트리플, 2=투페어, 1=원페어, 0=하이카드
    """
    values = sorted([c.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    is_flush = len(set(suits)) == 1
    
    # 스트레이트 판별 (A-5-4-3-2 예외 처리 포함)
    is_straight = False
    if len(set(values)) == 5 and values[0] - values[-1] == 4:
        is_straight = True
    elif values == [14, 5, 4, 3, 2]: # 백스트레이트 (A=1)
        is_straight = True
        values = [5, 4, 3, 2, 1] # 비교를 위해 A값을 1로 취급하여 뒤로 보냄

    counts = Counter(values)
    freq = sorted([(count, val) for val, count in counts.items()], reverse=True)
    
    # freq 형태: [(개수, 숫자), (개수, 숫자), ...] 
    # 예: 8포카드면 [(4, 8), (1, 13)]
    
    if is_straight and is_flush:
        return (8, values[0])
    if freq[0][0] == 4:
        return (7, freq[0][1], freq[1][1])
    if freq[0][0] == 3 and freq[1][0] == 2:
        return (6, freq[0][1], freq[1][1])
    if is_flush:
        return (5, *values)
    if is_straight:
        return (4, values[0])
    if freq[0][0] == 3:
        return (3, freq[0][1], freq[1][1], freq[2][1])
    if freq[0][0] == 2 and freq[1][0] == 2:
        return (2, freq[0][1], freq[1][1], freq[2][1])
    if freq[0][0] == 2:
        return (1, freq[0][1], freq[1][1], freq[2][1], freq[3][1])
    
    return (0, *values)

# --- 포커 게임 핵심 로직 (사이드 팟 처리) ---
class PokerGame:
    def __init__(self, players):
        self.players = players
        self.deck = Deck()
        # self.pot 대신 각 플레이어의 invested를 합산하여 계산합니다.

    def resolve_showdown(self):
        """
        사이드 팟을 고려하여 승자들에게 칩을 분배합니다. (스플릿 포함)
        """
        print("\n=== 쇼다운 및 팟 분배 ===")
        # 각 플레이어의 족보 점수 계산 (폴드하지 않은 사람만)
        for p in self.players:
            if not p.is_folded:
                p.hand_score = get_best_hand(p.get_all_cards())
        
        active_investors = [p for p in self.players if p.invested > 0]
        
        pot_number = 1
        # 투자된 칩이 남아있는 동안 사이드 팟을 계속 생성 및 분배
        while any(p.invested > 0 for p in active_investors):
            # 1. 이번 팟의 기준액 (남아있는 투자액 중 0이 아닌 최소값)
            current_min_invest = min(p.invested for p in active_investors if p.invested > 0)
            
            # 2. 이번 팟 생성 및 각 플레이어 투자액 차감
            current_pot = 0
            eligible_players = []
            
            for p in active_investors:
                if p.invested > 0:
                    deduction = min(p.invested, current_min_invest)
                    p.invested -= deduction
                    current_pot += deduction
                    
                    # 폴드하지 않은 사람만 이 팟을 먹을 자격이 있음
                    if not p.is_folded and deduction == current_min_invest:
                        eligible_players.append(p)
            
            if current_pot == 0:
                break
                
            # 3. 이번 팟의 승자 판별 (점수가 가장 높은 사람, 동률이면 여러 명)
            if not eligible_players:
                continue # 모두 폴드한 돈만 남은 경우 (규칙상 발생하기 어려움)
                
            best_score = max(p.hand_score for p in eligible_players)
            winners = [p for p in eligible_players if p.hand_score == best_score]
            
            # 4. 스플릿 분배 (나머지 버림은 임의로 처리하거나 첫 승자에게 줄 수 있으나 일단 정수 나누기)
            split_amount = current_pot // len(winners)
            
            winner_names = ", ".join([w.name for w in winners])
            print(f"[팟 {pot_number}] 크기: {current_pot} | 승자: {winner_names} (각 {split_amount} 칩 획득)")
            
            for w in winners:
                w.chips += split_amount
                
            pot_number += 1

# --- 단위 테스트 실행 ---
if __name__ == "__main__":
    p1 = Player("Player_A")
    p2 = Player("Player_B")
    p3 = Player("Player_C")
    
    # 임의로 투자액과 폴드/올인 상태 설정 (올인 테스트)
    # A는 올인(100), B와 C는 남아서 더 베팅(300)
    p1.chips    = 0
    p1.invested = 100
    p1.is_all_in = True 
    
    p2.invested = 300
    p3.invested = 300
    
    game = PokerGame([p1, p2, p3])
    
    # 승부 조작: A는 최고 족보(A포카드), B는 투페어, C는 트리플
    p1.hidden_cards = [Card('S', 'A'), Card('D', 'A'), Card('H', 'A'), Card('C', 'A'), Card('S', 'K')]
    p2.hidden_cards = [Card('S', 'K'), Card('D', 'K'), Card('H', 'Q'), Card('C', 'Q'), Card('S', '2')]
    p3.hidden_cards = [Card('S', 'J'), Card('D', 'J'), Card('H', 'J'), Card('C', 'T'), Card('S', '3')]
    
    game.resolve_showdown()
    
    print(f"\n[최종 칩 현황]")
    for p in game.players:
         print(f"{p.name}: {p.chips} 칩")
