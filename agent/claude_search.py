"""Multi-street particle-belief best response (depth-limited search).

This is the principled fix for the 1-ply myopia shared by both the Python
``ClaudeBeliefBRAgent`` and the C++ ``PolicyLBR``: instead of assuming an
immediate showdown after my action, it plays each candidate action out to the
end of the hand over belief-weighted determinizations of the opponent, using a
base continuation policy for my future decisions and the fixed opponent model
for the opponent. Choosing the root action that maximises this rollout value is
one step of rollout policy improvement over the base policy, so against a fixed
opponent it is >= the base policy in expectation -- and it captures the
multi-street exploit lines (bet this street AND the next) that a one-ply model
cannot see.

It reuses ``poker_env`` rules unmodified: the hand is reconstructed from the
observable state plus a determinized opponent, then driven to showdown with the
real ``apply_action`` / ``play_betting_round`` / ``resolve_showdown`` methods.
Heads-up only; multiway falls back to the base belief agent.

See CLAUDE_BELIEF_BR.md and CONCAVE_UTILITY_IDEAS.md for the belief lineage, and
cpp_mccfr/stud_mccfr.cpp PolicyLBR::action_ev (line ~4289) for the 1-ply probe
this generalises.
"""
from __future__ import annotations

import contextlib
import io
import math
import random
from typing import Any, Sequence

from agent.claude_belief_br import ClaudeBeliefBRAgent, ACTION_STRENGTH_SIGNAL
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import PokerGame, Card, ALL_CARDS

_STREETS = [("4th", True), ("5th", True), ("6th", True), ("7th_hidden", False)]
_STREET_INDEX = {name: i for i, (name, _) in enumerate(_STREETS)}
_RAISES = {"BBING", "DDADANG", "QUARTER", "HALF", "FULL"}


class ClaudeSearchAgent(ClaudeBeliefBRAgent):
    def __init__(
        self,
        name: str,
        rollouts: int = 32,
        cont_particles: int = 24,
        belief_particles: int = 120,
        seed: int | None = None,
        opponent_model: HeuristicPokerAgent | None = None,
    ):
        # risk-neutral: as a probe/exploiter we want the true best response.
        super().__init__(name, belief_particles=belief_particles, aggression_margin=0.0,
                         risk_lambda=0.0, opponent_model=opponent_model, seed=seed)
        self.rollouts = rollouts
        # The base continuation policy for MY future streets: the tuned 1-ply
        # belief BR (margin 0.40 -- the +1.0-EV configuration, not the over-betting
        # margin-0 one). Rollout policy improvement is only as good as its base,
        # so the base must itself play sensibly. Opponent future play uses the
        # fixed opponent model (the real target being exploited).
        self._cont = ClaudeBeliefBRAgent(f"{name}__cont", belief_particles=cont_particles,
                                         aggression_margin=0.40, risk_lambda=0.0,
                                         opponent_model=self.opp_model, seed=(seed or 0) + 991)

    # ------------------------------------------------------------------ actions
    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None
        active = [o for o in state.get("opponents", []) if not o.get("is_folded") and not o.get("is_eliminated")]
        if len(active) != 1:
            return super().choose_action(state, valid_actions)  # multiway: base agent

        dets = self._sample_determinizations(state, self.rollouts)
        if not dets:
            return super().choose_action(state, valid_actions)

        best_action, best_value = None, -math.inf
        for action in valid_actions:
            value = self._rollout_value(state, action, dets)
            if value > best_value:
                best_action, best_value = action, value
        return best_action

    # -------------------------------------------------------- determinizations
    def _sample_determinizations(self, state: dict[str, Any], count: int):
        """Sample (weight, opp_hidden, opp_discard, my_future, opp_future) tuples.

        Weights are the same action-conditioned likelihood the base belief uses,
        so this is the posterior over opponent holdings, carried as real cards.
        """
        own = self._cards(state.get("my_hidden_cards", [])) + self._cards(state.get("my_public_cards", []))
        opp = state["opponents"][0]
        opp_public = self._cards(opp.get("public_cards", []))
        street = state.get("street")
        opp_hidden_count = 3 if street == "7th_hidden" else 2

        known = set(own) | set(opp_public)
        discard = state.get("my_discarded_card")
        if discard:
            known.add(self._coerce_card(discard))
        deck = [c for c in ALL_CARDS if c not in known]

        street_idx = _STREET_INDEX.get(street, 1)
        remaining = _STREETS[street_idx + 1:]          # streets still to be dealt
        my_future_n = len(remaining)
        opp_future_n = len(remaining)
        need = 1 + opp_hidden_count + my_future_n + opp_future_n  # +1 opp discard
        if need > len(deck):
            return []

        a_obs = self._observed_opponent_strength(state)
        dets = []
        for _ in range(count):
            draw = self.rng.sample(deck, need)
            c = 0
            opp_discard = draw[c]; c += 1
            opp_hidden = draw[c:c + opp_hidden_count]; c += opp_hidden_count
            my_future = draw[c:c + my_future_n]; c += my_future_n
            opp_future = draw[c:c + opp_future_n]; c += opp_future_n
            if a_obs is not None:
                strength = self._opp_strength(opp_hidden, opp_public, state)
                weight = math.exp(-((strength - a_obs) ** 2) / (2 * self.likelihood_sigma ** 2))
            else:
                weight = 1.0
            dets.append((weight, opp_hidden, opp_discard, my_future, opp_future, remaining))
        return dets

    # ---------------------------------------------------------------- rollouts
    def _rollout_value(self, state: dict[str, Any], action: str, dets) -> float:
        total_w = value = 0.0
        for weight, opp_hidden, opp_discard, my_future, opp_future, remaining in dets:
            net = self._one_rollout(state, action, opp_hidden, opp_discard, my_future, opp_future, remaining)
            value += weight * net
            total_w += weight
        return value / total_w if total_w > 0 else -math.inf

    def _one_rollout(self, state, action, opp_hidden, opp_discard, my_future, opp_future, remaining) -> float:
        game, me, opp = self._reconstruct(state, opp_hidden, opp_discard, my_future, opp_future, remaining)
        agents = {me.name: self._cont, opp.name: self.opp_model}
        with contextlib.redirect_stdout(io.StringIO()):
            was_raise = game.apply_action(me, action)
            if not me.is_folded:
                self._finish_current_round(game, agents, me, opp, was_raise)
            self._play_remaining_streets(game, agents, remaining)
            game.resolve_showdown()
        ante = float(state.get("ante", 1) or 1)
        return (me.chips - me.hand_start_chips) / ante

    # --------------------------------------------------------- reconstruction
    def _reconstruct(self, state, opp_hidden, opp_discard, my_future, opp_future, remaining):
        mode = state.get("game_mode", "ev")
        ante = int(state.get("ante", 1) or 1)
        ev_ante = int(state.get("effective_stack_ante") or 1000)
        my_seat = int(state.get("seat_index", 0))
        opp_state = state["opponents"][0]
        opp_seat = int(opp_state.get("seat_index", 1 - my_seat))

        names = [None, None]
        names[my_seat], names[opp_seat] = "S_me", "S_opp"
        game = PokerGame(names, log_file=None, ante=ante, game_mode=mode, ev_stack_ante=ev_ante)
        game.street = state.get("street")
        game.pot = float(state.get("pot", 0) or 0)
        game.current_highest_bet = float(state.get("current_highest_bet", 0) or 0)
        game.raise_count = int(state.get("raise_count", 0) or 0)
        game.betting_history = self._rebuild_history(state, my_seat, opp_seat)

        me = game.players[my_seat]
        opp = game.players[opp_seat]
        self._set_player(me, mode, ev_ante, ante,
                         hidden=self._cards(state.get("my_hidden_cards", [])),
                         public=self._cards(state.get("my_public_cards", [])),
                         discarded=state.get("my_discarded_card"),
                         invested=float(state.get("my_invested", 0) or 0),
                         round_bet=float(state.get("my_round_bet", 0) or 0),
                         all_in=bool(state.get("my_is_all_in")),
                         chips_available=float(state.get("my_chips", 0) or 0))
        self._set_player(opp, mode, ev_ante, ante,
                         hidden=list(opp_hidden),
                         public=self._cards(opp_state.get("public_cards", [])),
                         discarded=opp_discard,
                         invested=float(opp_state.get("invested", 0) or 0),
                         round_bet=float(opp_state.get("round_bet", 0) or 0),
                         all_in=bool(opp_state.get("is_all_in")),
                         chips_available=float(opp_state.get("chips", 0) or 0))

        # Deck: remaining streets deal seat0 then seat1; draw() pops from the end.
        deal_order: list[Card] = []
        for i, (_street, _pub) in enumerate(remaining):
            card0 = my_future[i] if my_seat == 0 else opp_future[i]
            card1 = my_future[i] if my_seat == 1 else opp_future[i]
            deal_order.extend([card0, card1])
        game.deck.cards = list(reversed(deal_order))
        return game, me, opp

    def _set_player(self, player, mode, ev_ante, ante, hidden, public, discarded,
                    invested, round_bet, all_in, chips_available):
        player.stackless = mode == "ev"
        player.hidden_cards = [self._coerce_card(c) for c in hidden]
        player.public_cards = [self._coerce_card(c) for c in public]
        player.discarded_card = self._coerce_card(discarded) if discarded else None
        player.invested = invested
        player.current_bet = round_bet
        player.is_folded = False
        player.is_all_in = all_in
        player.is_eliminated = False
        if mode == "ev":
            # real ev bookkeeping: chips start at 0 and decrease by the investment
            player.chips = -invested
            player.hand_start_chips = 0
        else:
            player.chips = chips_available
            player.hand_start_chips = chips_available + invested

    def _rebuild_history(self, state, my_seat, opp_seat):
        history = []
        for event in state.get("betting_history", []):
            actor = event.get("actor")
            index = my_seat if actor == "self" else opp_seat
            new_event = {k: v for k, v in event.items() if k != "actor"}
            new_event["actor_index"] = index
            history.append(new_event)
        return history

    # -------------------------------------------------------------- driving
    def _alive(self, game) -> int:
        return sum(1 for p in game.players if not p.is_folded and not p.is_eliminated)

    def _acted_this_street(self, game, player) -> bool:
        idx = game.players.index(player)
        return any(e.get("actor_index") == idx and e.get("street") == game.street
                   for e in game.betting_history)

    def _finish_current_round(self, game, agents, me, opp, my_was_raise) -> None:
        if self._alive(game) <= 1:
            return
        if my_was_raise:
            pending = {opp}
        else:
            opp_matched = game.current_highest_bet - opp.current_bet == 0
            if game.current_highest_bet - me.current_bet == 0 and opp_matched and self._acted_this_street(game, opp):
                pending = set()
            elif not opp_matched:
                pending = set()   # I called the opponent's bet -> round closed
            else:
                pending = {opp}   # I checked first; opponent still to act
        order = [opp, me]
        i = 0
        guard = 0
        while pending and guard < 16:
            guard += 1
            player = order[i % 2]; i += 1
            if player not in pending:
                continue
            if not player.can_act():
                pending.discard(player)
                continue
            valids = game.get_valid_actions(player)
            if not valids:
                pending.discard(player)
                continue
            act = agents[player.name].choose_action(game.get_ai_state(player, valids), valids)
            if act not in valids:
                act = "CHECK" if "CHECK" in valids else "CALL" if "CALL" in valids else "FOLD"
            was_raise = game.apply_action(player, act)
            if player.is_folded or self._alive(game) <= 1:
                return
            pending = {me if player is opp else opp} if was_raise else (pending - {player})

    def _play_remaining_streets(self, game, agents, remaining) -> None:
        for street, is_public in remaining:
            if self._alive(game) <= 1:
                break
            game.street = street
            game.deal_cards_to_active(is_public=is_public)
            if sum(1 for p in game.players if p.can_act()) >= 2:
                game.play_betting_round(agents)
