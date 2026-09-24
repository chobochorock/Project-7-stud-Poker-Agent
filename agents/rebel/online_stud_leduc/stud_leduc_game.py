from __future__ import annotations

import enum

import numpy as np
import pyspiel


class Action(enum.IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2


NUM_PLAYERS = 2
DECK = tuple(range(6))
RANKS = "JQK"
MAX_RAISES = 2
HISTORY_SLOTS = 8
INFORMATION_STATE_SIZE = 2 + 2 + 4 + 4 + 4 + HISTORY_SLOTS * 7

GAME_TYPE = pyspiel.GameType(
    short_name="python_online_rebel_stud_leduc",
    long_name="Online ReBeL Stud Leduc",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
)

GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(DECK),
    max_chance_outcomes=len(DECK),
    num_players=NUM_PLAYERS,
    min_utility=-13.0,
    max_utility=13.0,
    utility_sum=0.0,
    max_game_length=12,
)


def rank(card: int) -> int:
    return card // 2


def information_key(
    player: int,
    round_index: int,
    private_rank: int,
    public_ranks: tuple[int, int] | None,
    history: tuple[tuple[int, int], ...],
) -> str:
    own_up = "-" if public_ranks is None else RANKS[public_ranks[player]]
    other_up = "-" if public_ranks is None else RANKS[public_ranks[1 - player]]
    actions = "".join(f"{street}:{action}" for street, action in history)
    return (
        f"p{player}|r{round_index}|h{RANKS[private_rank]}|"
        f"u{own_up}{other_up}|{actions}"
    )


class StudLeducGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(GAME_TYPE, GAME_INFO, params or {})

    def new_initial_state(self):
        return StudLeducState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return StudLeducObserver(params)

    def information_state_tensor_size(self):
        return INFORMATION_STATE_SIZE


class StudLeducState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.private: list[int] = []
        self.public: list[int] = []
        self.contribution = [1, 1]
        self.round_bet = [0, 0]
        self.round_index = 0
        self.raises = 0
        self.actor = 0
        self.history: list[tuple[int, int]] = []
        self.folded: int | None = None
        self.terminal = False

    def current_player(self):
        if self.terminal:
            return pyspiel.PlayerId.TERMINAL
        if len(self.private) < 2 or (len(self.public) < 2 and self.round_index == 1):
            return pyspiel.PlayerId.CHANCE
        return self.actor

    def chance_outcomes(self):
        used = set(self.private + self.public)
        outcomes = [card for card in DECK if card not in used]
        probability = 1.0 / len(outcomes)
        return [(card, probability) for card in outcomes]

    def _legal_actions(self, player):
        call = max(self.round_bet) - self.round_bet[player]
        actions = [Action.FOLD, Action.CHECK_CALL] if call else [Action.CHECK_CALL]
        if self.raises < MAX_RAISES:
            actions.append(Action.BET_RAISE)
        return actions

    def _apply_action(self, action):
        if self.is_chance_node():
            if len(self.private) < 2:
                self.private.append(action)
            else:
                self.public.append(action)
                if len(self.public) == 2:
                    self.actor = 0
            return

        player = self.actor
        call = max(self.round_bet) - self.round_bet[player]
        previous = (
            self.history[-1][1]
            if self.history and self.history[-1][0] == self.round_index
            else None
        )
        self.history.append((self.round_index, int(action)))
        if action == Action.FOLD:
            self.folded = player
            self.terminal = True
        elif action == Action.BET_RAISE:
            paid = call + (2 if self.round_index == 0 else 4)
            self.round_bet[player] += paid
            self.contribution[player] += paid
            self.raises += 1
            self.actor = 1 - player
        elif call:
            self.contribution[player] += call
            self.round_bet[player] += call
            self._finish_round()
        elif previous == Action.CHECK_CALL:
            self._finish_round()
        else:
            self.actor = 1 - player

    def _finish_round(self):
        if self.round_index == 1:
            self.terminal = True
            return
        self.round_index = 1
        self.round_bet = [0, 0]
        self.raises = 0

    def is_terminal(self):
        return self.terminal

    def returns(self):
        if not self.terminal:
            return [0.0, 0.0]
        pot = sum(self.contribution)
        if self.folded is not None:
            winners = [1 - self.folded]
        else:
            values = [self._hand_value(player) for player in range(2)]
            winners = [
                player for player, value in enumerate(values) if value == max(values)
            ]
        awards = [0.0, 0.0]
        for winner in winners:
            awards[winner] = pot / len(winners)
        return [awards[player] - self.contribution[player] for player in range(2)]

    def _hand_value(self, player):
        private = rank(self.private[player])
        public = rank(self.public[player])
        return (int(private == public), max(private, public), min(private, public))

    def information_state(self, player):
        private = rank(self.private[player]) if len(self.private) > player else 0
        public = (
            (rank(self.public[0]), rank(self.public[1]))
            if len(self.public) == 2
            else None
        )
        return information_key(
            player, self.round_index, private, public, tuple(self.history)
        )

    def information_state_tensor(self, player=None):
        if player is None:
            player = self.current_player()
        tensor = np.zeros(INFORMATION_STATE_SIZE, np.float32)
        offset = 0
        tensor[offset + player] = 1
        offset += 2
        tensor[offset + self.round_index] = 1
        offset += 2
        private = 0 if len(self.private) <= player else rank(self.private[player]) + 1
        tensor[offset + private] = 1
        offset += 4
        own_up = 0 if len(self.public) <= player else rank(self.public[player]) + 1
        tensor[offset + own_up] = 1
        offset += 4
        other = 1 - player
        other_up = 0 if len(self.public) <= other else rank(self.public[other]) + 1
        tensor[offset + other_up] = 1
        offset += 4
        for slot, (street, action) in enumerate(self.history):
            tensor[offset + slot * 7 + 1 + street * 3 + action] = 1
        for slot in range(len(self.history), HISTORY_SLOTS):
            tensor[offset + slot * 7] = 1
        return tensor.tolist()

    def observation_string(self, player):
        return self.information_state(player)

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{RANKS[rank(action)]}{action % 2}"
        return ("Fold", "Check/Call", "Bet/Raise")[action]


class StudLeducObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Unsupported observer parameters: {params}")
        self.tensor = np.zeros(INFORMATION_STATE_SIZE, np.float32)
        self.dict = {}

    def set_from(self, state, player):
        self.tensor[:] = state.information_state_tensor(player)

    def string_from(self, state, player):
        return state.information_state(player)


pyspiel.register_game(GAME_TYPE, StudLeducGame)


def load_game():
    return pyspiel.load_game("python_online_rebel_stud_leduc")
