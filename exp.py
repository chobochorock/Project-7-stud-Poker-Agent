from poker_env import Card, PokerGame, get_best_hand


def make_cards(labels: list[str]) -> list[Card]:
    return [Card(label[0], label[1:]) for label in labels]


def side_pot_demo() -> None:
    game = PokerGame(["AllIn_A", "Deep_B", "Deep_C"], log_file=None)
    all_in_a, deep_b, deep_c = game.players

    all_in_a.chips = 0
    deep_b.chips = 0
    deep_c.chips = 0
    all_in_a.invested = 100
    deep_b.invested = 300
    deep_c.invested = 300
    game.pot = 700

    all_in_a.hidden_cards = make_cards(["sA", "hA", "dA", "cA", "sK"])
    deep_b.hidden_cards = make_cards(["sK", "hK", "dK", "c2", "d3"])
    deep_c.hidden_cards = make_cards(["sQ", "hQ", "dQ", "c4", "d5"])

    print("A best hand:", get_best_hand(all_in_a.get_all_cards()))
    print("B best hand:", get_best_hand(deep_b.get_all_cards()))
    print("C best hand:", get_best_hand(deep_c.get_all_cards()))

    result = game.resolve_showdown()
    print(result)


if __name__ == "__main__":
    side_pot_demo()
