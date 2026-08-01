from __future__ import annotations

import math
import random
from typing import Any, Sequence

from agent.base import PokerAgent
from agent.heuristic_agent import HeuristicPokerAgent
from poker_env import ALL_CARDS, Card, get_best_hand


RAISE_ACTIONS = ("BBING", "DDADANG", "QUARTER", "HALF", "FULL")

# Implied opponent strength for each observed action, aligned with the action
# thresholds inside HeuristicPokerAgent (e.g. it needs strength >= 0.76 to bet
# HALF for free, >= 0.90 to raise HALF facing a bet). These are the anchors the
# action-conditioned belief update centers on.
ACTION_STRENGTH_SIGNAL = {
    "FULL": 0.93,
    "HALF": 0.82,
    "DDADANG": 0.80,
    "QUARTER": 0.63,
    "BBING": 0.52,
    "CALL": 0.33,
    "CHECK": 0.18,
}


class ClaudeBeliefBRAgent(PokerAgent):
    """Action-conditioned particle-belief best response (sampled one-sided POMDP).

    한국어 요약: 상대 정책을 고정된 것으로 보고, 상대의 공개 베팅 행동으로
    상대 히든 카드에 대한 belief를 Bayesian 으로 갱신한 뒤, 모든 합법 행동의
    net-chip EV 를 비교해 최선응답을 고른다. `Toy-Card-Game-Agent` 의 exact
    POMDP best response 를 유한한 particle belief 로 7포커에 이식한 것이다.

    How it differs from HA1
    -----------------------
    HA1 estimates showdown equity under a *uniform* range over the opponent's
    hidden cards. This agent instead weights each hidden-card hypothesis ``h`` by

        b'(h) proportional to b(h) * pi_opp(observed betting | h),

    the exact Bayesian filter written in ``POMDP_FIXED_HEURISTIC_BEST_RESPONSE``.
    The opponent model ``pi_opp`` reuses ``HeuristicPokerAgent``'s strength
    estimate, so when the real opponent is that heuristic the likelihood is
    correct and the belief collapses toward the opponent's true range -- exactly
    the "fixed heuristic -> POMDP best response" idea, ported to seven-stud.

    Decision model
    --------------
    Equity ``eq`` (belief-weighted showdown share) is combined with a one-ply
    fold-equity model. Writing the current pot ``P``, my call amount ``c``, my
    committed chips ``i``, a raise size ``r`` and the opponent fold probability
    ``pf`` implied by the posterior:

        EV(FOLD)  = -i
        EV(CHECK) = eq * P            - i
        EV(CALL)  = eq * (P + c)      - (i + c)
        EV(raise) = pf * (P - i)
                  + (1 - pf) * (eq * (P + c + 2r) - (i + c + r))

    EV(CALL) beats EV(FOLD) exactly when ``eq`` clears the pot odds
    ``c / (P + c)``, so the myopic model already reproduces textbook pot-odds
    calling and adds belief-driven fold equity on top. Multi-street value is not
    searched yet; that is the natural next extension (see module notes).

    Scope: the belief update targets heads-up (one live opponent), the mode
    where the theory is clean. Against several live opponents it degrades to a
    uniform-range equity best response with no fold equity, which stays sound but
    is not the contribution.
    """

    def __init__(
        self,
        name: str,
        belief_particles: int = 240,
        likelihood_sigma: float = 0.18,
        aggression_margin: float = 0.40,
        risk_lambda: float = 12.0,
        opponent_model: HeuristicPokerAgent | None = None,
        seed: int | None = None,
    ):
        super().__init__(name)
        if belief_particles <= 0:
            raise ValueError("belief_particles must be positive.")
        if likelihood_sigma <= 0:
            raise ValueError("likelihood_sigma must be positive.")
        if risk_lambda < 0:
            raise ValueError("risk_lambda must be non-negative.")
        self.belief_particles = belief_particles
        self.likelihood_sigma = likelihood_sigma
        # A raise/bet is only taken when its EV beats the best passive action by
        # at least aggression_margin * (pot it builds). The myopic one-ply model
        # over-values thin value bets against a multi-street bettor; this margin
        # is the discipline that keeps the agent from bleeding chips on them.
        self.aggression_margin = aggression_margin
        # CARA risk aversion, per full effective stack (see _risk_scale). Each
        # action is scored by the certainty equivalent of its net-chip outcome
        # distribution, not the plain mean:
        #   CE = -(1/lambda) log E[exp(-lambda * net/stack)] * stack
        # which is approximately E[net] - (lambda / 2) Var[net] / stack. lambda = 0
        # is exactly the risk-neutral EV. The default lambda = 12 stays near
        # risk-neutral on small pots (deep stacks) but discounts high-variance
        # stack-offs, which fixes the linear agent's big-pot leak in cash mode
        # while keeping EV mode positive. See CONCAVE_UTILITY_IDEAS.md (P1).
        self.risk_lambda = risk_lambda
        self.rng = random.Random(seed)
        # The opponent model does the pi_opp(action | hand) reasoning. Reusing the
        # repo heuristic keeps the belief consistent with the fixed opponent.
        self.opp_model = opponent_model or HeuristicPokerAgent(f"{name}__oppmodel")
        self.last_equity = 0.0
        self.last_uniform_equity = 0.0

    # ------------------------------------------------------------------ actions
    def choose_action(self, state: dict[str, Any], valid_actions: Sequence[str]) -> str | None:
        if not valid_actions:
            return None
        valid = list(valid_actions)

        belief = self.estimate_belief(state)
        self.last_equity = belief["equity"]
        self.last_uniform_equity = belief["uniform_equity"]

        evs = {action: self._action_ev(action, state, belief) for action in valid}
        passive = [a for a in valid if a not in RAISE_ACTIONS]
        aggressive = [a for a in valid if a in RAISE_ACTIONS]

        # Default to the best passive line; only escalate to a bet/raise when it
        # clears the aggression margin, so the myopic model cannot spam thin bets.
        best_action = max(passive, key=lambda a: evs[a]) if passive else max(valid, key=lambda a: evs[a])
        if aggressive:
            best_raise = max(aggressive, key=lambda a: evs[a])
            pot = float(state.get("pot", 0) or 0)
            call = float(state.get("call_amount", 0) or 0)
            margin = self.aggression_margin * (pot + call + 1.0)
            if evs[best_raise] >= evs[best_action] + margin:
                best_action = best_raise

        shift = belief["equity"] - belief["uniform_equity"]
        print(
            f"[{self.name}] belief-br: {best_action} "
            f"(eq={belief['equity']:.3f}, unif={belief['uniform_equity']:.3f}, "
            f"shift={shift:+.3f}, ev={evs[best_action]:+.2f})"
        )
        return best_action

    def choose_discard_and_reveal(self, hidden_cards: Sequence[Any]) -> tuple[int, int]:
        if len(hidden_cards) != 4:
            raise ValueError("Belief-BR needs exactly four initial cards.")
        cards = [self._coerce_card(card) for card in hidden_cards]

        # Keep the three cards whose retained triple has the best rollout equity,
        # i.e. discard the index whose removal hurts least.
        discard_idx = max(
            range(4),
            key=lambda index: self._retained_equity(
                [card for card_index, card in enumerate(cards) if card_index != index],
                cards,
            ),
        )
        retained = [index for index in range(4) if index != discard_idx]
        reveal_idx = max(retained, key=lambda index: self._reveal_value(cards[index], cards))
        return discard_idx, reveal_idx

    def learn_from_database(self, database: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "agent": type(self).__name__,
            "trained": False,
            "method": "action-conditioned particle-belief best response (planning, no training)",
        }

    # ------------------------------------------------------------------- belief
    def estimate_belief(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return posterior belief particles and belief-weighted equity.

        The returned dict carries ``equity`` (posterior-weighted showdown share),
        ``uniform_equity`` (the same estimate with the likelihood switched off,
        for diagnostics), and ``particles`` as ``(opp_strength, weight)`` pairs
        used later for the fold-equity model.
        """
        own_cards = self._cards(state.get("my_hidden_cards", [])) + self._cards(state.get("my_public_cards", []))
        my_public_labels = list(state.get("my_public_cards", []))
        opponents = list(state.get("opponents", []))
        active = [o for o in opponents if not o.get("is_folded") and not o.get("is_eliminated")]

        fair_share = 1.0 / max(1, len(active) + 1)
        blank = {
            "equity": fair_share,
            "uniform_equity": fair_share,
            "particles": [],
            "headsup": False,
        }
        if not own_cards or not active:
            return {**blank, "equity": 1.0 if own_cards and not active else fair_share,
                    "uniform_equity": 1.0 if own_cards and not active else fair_share}

        known = set(own_cards)
        for opponent in opponents:
            known.update(self._cards(opponent.get("public_cards", [])))
        discard = state.get("my_discarded_card")
        if discard:
            known.add(self._coerce_card(discard))
        deck = [card for card in ALL_CARDS if card not in known]

        headsup = len(active) == 1
        a_obs = self._observed_opponent_strength(state) if headsup else None

        # Cards each active opponent still hides, plus each opponent's unknown
        # discarded card, plus the cards I have yet to receive.
        opp_publics = [self._cards(o.get("public_cards", [])) for o in active]
        opp_hidden_counts = [self._opponent_hidden_count(state, public) for public in opp_publics]
        opp_future_counts = [max(0, 7 - len(public) - hidden) for public, hidden in zip(opp_publics, opp_hidden_counts)]
        my_needed = max(0, 7 - len(own_cards))
        dead_count = len(active)  # each live opponent discarded exactly one card at H4
        sample_size = dead_count + my_needed + sum(opp_hidden_counts) + sum(opp_future_counts)
        if sample_size > len(deck):
            return blank

        weighted_win = weight_sum = 0.0
        uniform_win = 0.0
        particles: list[tuple[float, float]] = []
        for _ in range(self.belief_particles):
            draw = self.rng.sample(deck, sample_size)
            cursor = dead_count  # first cards stand in for unknown discards (dead)

            own_final = own_cards + draw[cursor:cursor + my_needed]
            cursor += my_needed
            own_score = get_best_hand(own_final)

            opp_scores = []
            primary_strength = fair_share
            for index, (public, hidden_count, future_count) in enumerate(
                zip(opp_publics, opp_hidden_counts, opp_future_counts)
            ):
                hidden = draw[cursor:cursor + hidden_count]
                cursor += hidden_count
                future = draw[cursor:cursor + future_count]
                cursor += future_count
                opp_scores.append(get_best_hand(public + hidden + future))
                if index == 0:
                    primary_strength = self._opp_strength(hidden, public, state)

            best_score = max([own_score, *opp_scores])
            if own_score == best_score:
                tied = sum(score == best_score for score in opp_scores)
                win = 1.0 / (tied + 1)
            else:
                win = 0.0

            likelihood = 1.0
            if headsup and a_obs is not None:
                likelihood = math.exp(-((primary_strength - a_obs) ** 2) / (2 * self.likelihood_sigma ** 2))

            weighted_win += likelihood * win
            weight_sum += likelihood
            uniform_win += win
            particles.append((primary_strength, likelihood, win))

        equity = weighted_win / weight_sum if weight_sum > 0 else uniform_win / self.belief_particles
        uniform_equity = uniform_win / self.belief_particles
        return {
            "equity": equity,
            "uniform_equity": uniform_equity,
            "particles": particles,
            "headsup": headsup,
        }

    def _observed_opponent_strength(self, state: dict[str, Any]) -> float | None:
        """Strongest strength signal the opponent has leaked through betting."""
        signals = [
            ACTION_STRENGTH_SIGNAL[event["action"]]
            for event in state.get("betting_history", [])
            if event.get("actor") != "self" and event.get("action") in ACTION_STRENGTH_SIGNAL
        ]
        return max(signals) if signals else None

    def _opp_strength(self, hidden: Sequence[Card], public: Sequence[Card], state: dict[str, Any]) -> float:
        """Opponent hand strength under the opponent model, as it would see it."""
        opp_state = {
            "my_hidden_cards": [str(card) for card in hidden],
            "my_public_cards": [str(card) for card in public],
            "opponents": [
                {
                    "seat": "opponent_1",
                    "public_cards": list(state.get("my_public_cards", [])),
                    "is_folded": False,
                    "is_eliminated": False,
                }
            ],
            "betting_history": [],
            "call_amount": 0,
            "ante": state.get("ante", 1),
            "pot": state.get("pot", 0),
            "my_chips": state.get("effective_stack") or 1000,
        }
        return self.opp_model._estimate_strength(opp_state)

    # ----------------------------------------------------------------- EV model
    def _action_ev(self, action: str, state: dict[str, Any], belief: dict[str, Any]) -> float:
        pot = float(state.get("pot", 0) or 0)
        call = float(state.get("call_amount", 0) or 0)
        invested = float(state.get("my_invested", 0) or 0)
        scale = self._risk_scale(state)
        particles = belief["particles"]

        if action == "FOLD":
            return -invested
        if action == "CHECK":
            if not particles:
                return belief["equity"] * pot - invested
            return self._ce([(w, win * pot - invested) for _, w, win in particles], scale)
        if action == "CALL":
            if not particles:
                return belief["equity"] * (pot + call) - (invested + call)
            outcomes = [(w, win * (pot + call) - (invested + call)) for _, w, win in particles]
            return self._ce(outcomes, scale)
        return self._raise_ev(state, belief, self._raise_amount(action, state))

    def _risk_scale(self, state: dict[str, Any]) -> float:
        """Chip scale for CARA risk aversion: the effective stack at risk this hand.

        Normalising by the stack (not the ante) makes ``risk_lambda`` mode-agnostic
        and interpretable as risk aversion per full stack. Deep stacks then stay
        nearly risk-neutral per hand, and CARA only bites when a decision commits a
        large fraction of the stack -- big pots and all-ins -- which is the correct
        place for risk aversion in a cash game.
        """
        effective = state.get("effective_stack")
        if effective:
            return float(effective)
        total = float(state.get("my_chips", 0) or 0) + float(state.get("my_invested", 0) or 0)
        if total > 0:
            return total
        return max(float(state.get("ante", 1) or 1), 1.0)

    def _ce(self, outcomes: Sequence[tuple[float, float]], scale: float) -> float:
        """Certainty equivalent of a weighted net-chip outcome distribution.

        With ``risk_lambda == 0`` this is the plain belief-weighted mean, so the
        agent is exactly risk-neutral. With ``risk_lambda > 0`` it is the CARA
        certainty equivalent in units of ``scale`` (the effective stack), which
        sits below the mean by roughly ``(lambda / 2) * variance / scale``.
        """
        total = sum(weight for weight, _ in outcomes)
        if total <= 0:
            return 0.0
        if self.risk_lambda <= 0.0:
            return sum(weight * net for weight, net in outcomes) / total
        lam = self.risk_lambda
        scale = max(scale, 1.0)
        units = [(weight, net / scale) for weight, net in outcomes]
        # log-sum-exp anchored at the worst outcome keeps every exponent <= 0.
        ref = min(net for _, net in units)
        soft = sum(weight * math.exp(-lam * (net - ref)) for weight, net in units) / total
        return (ref - math.log(soft) / lam) * scale

    def _raise_ev(self, state: dict[str, Any], belief: dict[str, Any], raise_amount: float) -> float:
        """Belief-weighted EV of a raise, conditioning equity on the calling range.

        The earlier version valued a called raise at the *unconditional* posterior
        equity, which is optimistic: an opponent that calls has a strong hand, so
        my true equity when called is lower. Here each posterior particle either
        folds (I win the current pot) or calls (I go to showdown against *that*
        specific hand), which removes the optimism that made raises lose chips.
        """
        pot = float(state.get("pot", 0) or 0)
        call = float(state.get("call_amount", 0) or 0)
        invested = float(state.get("my_invested", 0) or 0)
        scale = self._risk_scale(state)
        final_pot = pot + call + 2 * raise_amount
        ev_fold_branch = pot - invested
        ev_call_cost = invested + call + raise_amount

        if not belief["particles"]:
            return ev_fold_branch
        facing_state = {
            "call_amount": raise_amount,
            "pot": pot + call + raise_amount,
            "ante": state.get("ante", 1),
            "my_chips": state.get("effective_stack") or 1000,
        }
        # Heads-up uses the opponent model to split fold vs call; multiway takes no
        # fold equity (everyone "calls") so raises are valued conservatively. Each
        # particle contributes one net-chip outcome; the certainty equivalent then
        # penalises the high-variance called-raise branch when risk_lambda > 0.
        headsup = belief["headsup"]
        outcomes = []
        for strength, weight, win in belief["particles"]:
            calls = (not headsup) or strength >= 0.90 or self.opp_model._should_call(strength, facing_state)
            net = (win * final_pot - ev_call_cost) if calls else ev_fold_branch
            outcomes.append((weight, net))
        return self._ce(outcomes, scale)

    def _raise_amount(self, action: str, state: dict[str, Any]) -> float:
        """Mirror ``PokerGame._raise_amount`` so EV matches the real chip cost."""
        pot = float(state.get("pot", 0) or 0)
        call = float(state.get("call_amount", 0) or 0)
        highest = float(state.get("current_highest_bet", 0) or 0)
        ante = float(state.get("ante", 1) or 1)
        pot_after_call = pot + call
        if action == "BBING":
            return ante
        if action == "DDADANG":
            return max(1.0, highest)
        if action == "QUARTER":
            return max(1.0, math.ceil(pot_after_call / 4))
        if action == "HALF":
            return max(1.0, math.ceil(pot_after_call / 2))
        if action == "FULL":
            return max(1.0, pot_after_call)
        return 0.0

    # -------------------------------------------------------------- discard MC
    def _retained_equity(self, retained: Sequence[Card], initial: Sequence[Card]) -> float:
        deck = [card for card in ALL_CARDS if card not in set(initial)]
        simulations = max(32, self.belief_particles // 4)
        total = 0.0
        for _ in range(simulations):
            sampled = self.rng.sample(deck, 12)
            own_score = get_best_hand(list(retained) + sampled[1:5])
            opp_score = get_best_hand(sampled[5:12])
            if own_score > opp_score:
                total += 1.0
            elif own_score == opp_score:
                total += 0.5
        return total / simulations

    def _reveal_value(self, card: Card, cards: Sequence[Card]) -> float:
        same_rank = sum(other.value == card.value for other in cards) - 1
        return card.value + same_rank * 5

    # ---------------------------------------------------------------- utilities
    def _opponent_hidden_count(self, state: dict[str, Any], public: Sequence[Card]) -> int:
        # 2 hidden through 6th street, 3 after the final hidden card on 7th.
        return 3 if state.get("street") == "7th_hidden" else 2

    def _cards(self, labels: Sequence[Any]) -> list[Card]:
        return [self._coerce_card(label) for label in labels]

    def _coerce_card(self, value: Any) -> Card:
        if isinstance(value, Card):
            return value
        label = str(value)
        return Card(label[0], label[1:])


def _self_check() -> None:
    """Play one heads-up EV hand against the heuristic without crashing."""
    import contextlib
    import io

    from poker_env import PokerGame

    random.seed(7)
    agents = {
        "Player_1": ClaudeBeliefBRAgent("Player_1", belief_particles=80, seed=1),
        "Player_2": HeuristicPokerAgent("Player_2"),
    }
    game = PokerGame(["Player_1", "Player_2"], log_file=None, ante=1000, game_mode="ev")
    with contextlib.redirect_stdout(io.StringIO()):
        result = game.play_hand(agents)
    net = result["final_chips"]["Player_1"]
    assert result["final_chips"]["Player_1"] + result["final_chips"]["Player_2"] == 0
    print(f"ok: one EV hand played, belief-br net = {net:+d}")


if __name__ == "__main__":
    _self_check()
