const setupForm = document.querySelector("#setupForm");
const autoRoundsEl = document.querySelector("#autoRounds");
const autoIntervalEl = document.querySelector("#autoInterval");
const playersEl = document.querySelector("#players");
const actionPanel = document.querySelector("#actionPanel");
const bettingLog = document.querySelector("#bettingLog");
const eventLog = document.querySelector("#eventLog");
const roundBadge = document.querySelector("#roundBadge");
const phaseBadge = document.querySelector("#phaseBadge");
const modeBadge = document.querySelector("#modeBadge");
const streetBadge = document.querySelector("#streetBadge");
const potBadge = document.querySelector("#potBadge");
const rangeObserver = document.querySelector("#rangeObserver");
const rangeSamples = document.querySelector("#rangeSamples");
const calculateRangeButton = document.querySelector("#calculateRange");
const rangeStatus = document.querySelector("#rangeStatus");
const rangeSummary = document.querySelector("#rangeSummary");
const rangeWorkspace = document.querySelector("#rangeWorkspace");
const rangeGrid = document.querySelector("#rangeGrid");
const rangeDetail = document.querySelector("#rangeDetail");

const RANGE_RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"];
const RANGE_SUITS = ["s", "h", "d", "c"];
const RANGE_CARDS = RANGE_RANKS.flatMap((rank) => RANGE_SUITS.map((suit) => `${suit}${rank}`));
const RANGE_CARD_INDEX = new Map(RANGE_CARDS.map((card, index) => [card, index]));
const CATEGORY_LABELS = {
  high_card: "High card",
  one_pair: "One pair",
  two_pair: "Two pair",
  three_of_a_kind: "Trips",
  straight: "Straight",
  flush: "Flush",
  full_house: "Full house",
  four_of_a_kind: "Quads",
  straight_flush: "Straight flush",
};

let currentState = null;
let autoRoundTimer = null;
let autoRoundSettings = { enabled: false, intervalMs: 2000 };
let currentRange = null;
let currentRangeKey = null;
let selectedRangeCell = null;
let rangeLoading = false;

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  syncAutoRoundSettings();
  const form = new FormData(setupForm);
  await postJson("/api/start", {
    player_types: ["p1", "p2", "p3", "p4", "p5"].map((key) => form.get(key)),
    game_mode: form.get("game_mode"),
  });
});

autoRoundsEl.addEventListener("change", () => {
  syncAutoRoundSettings();
  scheduleAutoNextRound(currentState);
});

autoIntervalEl.addEventListener("input", () => {
  syncAutoRoundSettings();
  scheduleAutoNextRound(currentState);
});

rangeObserver.addEventListener("change", () => {
  clearRangeResult();
  renderRangeControls(currentState);
});

calculateRangeButton.addEventListener("click", loadHandRange);

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    renderError(data.error || "Request failed");
    if (data.state) render(data.state);
    return null;
  }
  render(data);
  return data;
}

async function loadState() {
  const response = await fetch("/api/state");
  render(await response.json());
}

function render(state) {
  currentState = state;
  roundBadge.textContent = `Round ${state.round_number || state.hand_number || 0}`;
  modeBadge.textContent = state.game_mode || "cash";
  phaseBadge.textContent = state.phase || "idle";
  streetBadge.textContent = state.street || "-";
  potBadge.textContent = `Pot ${state.pot || 0}`;
  renderPlayers(state.players || [], state);
  renderActionPanel(state);
  renderLogs(state);
  renderRangeControls(state);
  scheduleAutoNextRound(state);
}

function renderRangeControls(state) {
  const players = state?.players || [];
  const previousObserver = rangeObserver.value;
  rangeObserver.replaceChildren();
  for (const player of players.filter((player) => !player.is_eliminated)) {
    const option = document.createElement("option");
    option.value = player.name;
    option.textContent = player.name;
    rangeObserver.appendChild(option);
  }

  const preferred = players.some((player) => player.name === previousObserver)
    ? previousObserver
    : state?.waiting?.player || players[0]?.name || "";
  rangeObserver.value = preferred;

  const activePlayers = players.filter((player) => !player.is_folded && !player.is_eliminated);
  const validStreet = ["4th", "5th", "6th", "7th_hidden"].includes(state?.street);
  const available = players.length === 2 && activePlayers.length === 2 && validStreet;
  const key = rangeContextKey(state);
  if (currentRangeKey && currentRangeKey !== key) clearRangeResult();

  calculateRangeButton.disabled = !available || rangeLoading;
  rangeObserver.disabled = players.length === 0 || rangeLoading;
  rangeSamples.disabled = !available || rangeLoading;
  if (!currentRange && !rangeLoading) {
    rangeStatus.textContent = available ? `${state.street} · uniform card-only` : "Unavailable";
  }
}

async function loadHandRange() {
  if (!currentState || calculateRangeButton.disabled) return;
  const requestKey = rangeContextKey(currentState);
  let loadError = null;
  rangeLoading = true;
  calculateRangeButton.disabled = true;
  rangeObserver.disabled = true;
  rangeSamples.disabled = true;
  rangeStatus.textContent = "Calculating";

  try {
    const response = await fetch("/api/hand_range", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        player: rangeObserver.value,
        samples_per_hand: Number(rangeSamples.value),
        seed: 7,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Range calculation failed");
    if (requestKey !== rangeContextKey(currentState)) return;
    currentRange = result;
    currentRangeKey = requestKey;
    renderHandRange(result);
  } catch (error) {
    loadError = error.message;
    clearRangeResult();
  } finally {
    rangeLoading = false;
    renderRangeControls(currentState);
    if (loadError) {
      rangeStatus.textContent = loadError;
      rangeStatus.classList.add("error");
    }
  }
}

function renderHandRange(result) {
  rangeStatus.classList.remove("error");
  rangeStatus.textContent = `${result.observer} → ${result.opponent} · ${result.street} · ${result.elapsed_seconds.toFixed(2)}s`;
  rangeSummary.classList.remove("is-hidden");
  rangeSummary.innerHTML = [
    rangeMetric("Hero equity", formatPercent(result.equity)),
    rangeMetric("Win", formatPercent(result.win_probability)),
    rangeMetric("Tie", formatPercent(result.tie_probability)),
    rangeMetric("Loss", formatPercent(result.loss_probability)),
    rangeMetric("Combos", result.possible_hands.toLocaleString()),
    rangeMetric("Samples", result.total_samples.toLocaleString()),
    rangeMetric("95% CI", `${formatPercent(result.equity_95ci[0])}–${formatPercent(result.equity_95ci[1])}`),
  ].join("");

  const handMap = new Map(result.hands.map((hand) => [rangeHandKey(hand.cards), hand]));
  const fragment = document.createDocumentFragment();
  fragment.appendChild(rangeAxisLabel("", "corner"));
  for (const card of RANGE_CARDS) fragment.appendChild(rangeAxisLabel(card, "column"));

  for (let row = 0; row < RANGE_CARDS.length; row += 1) {
    const rowCard = RANGE_CARDS[row];
    fragment.appendChild(rangeAxisLabel(rowCard, "row"));
    for (let column = 0; column < RANGE_CARDS.length; column += 1) {
      if (column <= row) {
        const blank = document.createElement("span");
        blank.className = "range-cell mirror";
        blank.setAttribute("aria-hidden", "true");
        fragment.appendChild(blank);
        continue;
      }

      const columnCard = RANGE_CARDS[column];
      const hand = handMap.get(rangeHandKey([rowCard, columnCard]));
      if (!hand) {
        const unavailable = document.createElement("span");
        unavailable.className = "range-cell unavailable";
        unavailable.title = `${rowCard} ${columnCard} · unavailable`;
        fragment.appendChild(unavailable);
        continue;
      }

      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "range-cell available";
      cell.style.backgroundColor = equityColor(hand.equity);
      cell.title = `${hand.cards.join(" + ")} · Hero equity ${formatPercent(hand.equity)}`;
      cell.setAttribute("aria-label", cell.title);
      cell.addEventListener("click", () => {
        selectedRangeCell?.classList.remove("selected");
        selectedRangeCell = cell;
        cell.classList.add("selected");
        renderRangeDetail(result, hand);
      });
      fragment.appendChild(cell);
    }
  }

  rangeGrid.replaceChildren(fragment);
  rangeWorkspace.classList.remove("is-hidden");
  renderRangeDetail(result, null);
}

function renderRangeDetail(result, hand) {
  const selected = hand
    ? `<div class="selected-hand"><strong>${hand.cards.map(escapeHtml).join(" + ")}</strong><span>Hero equity ${formatPercent(hand.equity)}</span><span>W ${formatPercent(hand.win_probability)} · T ${formatPercent(hand.tie_probability)} · L ${formatPercent(hand.loss_probability)}</span><span>Prior ${formatPercent(hand.probability, 3)}</span></div>`
    : `<div class="selected-hand"><strong>All combinations</strong><span>Hero equity ${formatPercent(result.equity)}</span></div>`;

  const categories = Object.entries(result.opponent_hand_categories)
    .sort((left, right) => right[1] - left[1])
    .map(([category, probability]) => `
      <div class="category-row">
        <span>${escapeHtml(CATEGORY_LABELS[category] || category)}</span>
        <div><i style="width:${Math.max(1, probability * 100)}%"></i></div>
        <strong>${formatPercent(probability)}</strong>
      </div>
    `)
    .join("");
  rangeDetail.innerHTML = `${selected}<h3>Opponent final hand</h3><div class="category-list">${categories}</div>`;
}

function clearRangeResult() {
  currentRange = null;
  currentRangeKey = null;
  selectedRangeCell = null;
  rangeSummary.classList.add("is-hidden");
  rangeWorkspace.classList.add("is-hidden");
  rangeSummary.replaceChildren();
  rangeGrid.replaceChildren();
  rangeDetail.replaceChildren();
  rangeStatus.classList.remove("error");
}

function rangeContextKey(state) {
  const statuses = (state?.players || []).map((player) => `${player.name}:${player.status}`).join("|");
  return `${state?.round_number || 0}:${state?.street || "idle"}:${rangeObserver.value}:${statuses}`;
}

function rangeHandKey(cards) {
  return [...cards].sort((left, right) => RANGE_CARD_INDEX.get(left) - RANGE_CARD_INDEX.get(right)).join("|");
}

function rangeAxisLabel(card, position) {
  const label = document.createElement("span");
  label.className = `range-axis ${position} ${card[0] === "h" || card[0] === "d" ? "red" : "black"}`;
  label.textContent = card;
  return label;
}

function rangeMetric(label, value) {
  return `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function formatPercent(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function equityColor(equity) {
  const value = Math.max(0, Math.min(1, Number(equity)));
  return `hsl(${Math.round(value * 120)} 55% 43%)`;
}

function renderPlayers(players, state) {
  playersEl.innerHTML = "";
  for (const player of players) {
    const card = document.createElement("article");
    const classes = ["player-card", player.status.toLowerCase()];
    if (player.is_acting) classes.push("acting");
    if (player.has_priority) classes.push("priority");
    card.className = classes.join(" ");

    const badges = [
      player.has_priority ? `<span class="player-badge dealer">Dealer</span>` : "",
      player.is_acting ? `<span class="player-badge turn">Turn</span>` : "",
    ].join("");

    card.innerHTML = `
      <div class="player-head">
        <div>
          <h2>${escapeHtml(player.name)}</h2>
          <p>${escapeHtml(player.type)} · ${escapeHtml(player.status)}</p>
        </div>
        <strong>${player.chips}</strong>
      </div>
      <div class="badge-row">${badges}</div>
      <div class="metric-row">
        <span>Invested ${player.invested}</span>
        <span>Bet ${player.current_bet}</span>
        <span>Net ${formatSigned(player.net || 0)}</span>
      </div>
      <div class="hand-name">${escapeHtml(player.hand_name || "-")}</div>
      <div class="card-group">
        <span>Public</span>
        <div class="cards">${renderCards(player.public_cards, false)}</div>
      </div>
      <div class="card-group">
        <span>Hidden</span>
        <div class="cards">${renderCards(player.hidden_cards, player.hidden_count > 0 && player.hidden_cards.length === 0, player.hidden_count)}</div>
      </div>
    `;
    playersEl.appendChild(card);
  }
}

function renderActionPanel(state) {
  actionPanel.innerHTML = "";
  const title = document.createElement("h2");
  title.textContent = panelTitle(state);
  actionPanel.appendChild(title);
  renderTurnOrder(state);

  if (state.phase === "idle") {
    actionPanel.appendChild(textBlock("Ready"));
    return;
  }
  if (state.phase === "complete" || state.phase === "game_over") {
    renderResultPanel(state);
    return;
  }
  if (!state.waiting) {
    actionPanel.appendChild(textBlock("Running"));
    return;
  }
  if (state.waiting.type === "discard") {
    renderDiscardControl(state.waiting);
    return;
  }
  if (state.waiting.type === "bet") {
    renderBetControl(state.waiting);
  }
}

function renderDiscardControl(waiting) {
  const form = document.createElement("form");
  form.className = "action-form";
  const options = waiting.cards.map((card, index) => `<option value="${index}">${index}: ${escapeHtml(card)}</option>`).join("");
  form.innerHTML = `
    <label>Discard<select name="discard">${options}</select></label>
    <label>Reveal<select name="reveal">${options}</select></label>
    <button type="submit">Apply</button>
  `;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    await postJson("/api/discard", {
      player: waiting.player,
      discard_index: Number(formData.get("discard")),
      reveal_index: Number(formData.get("reveal")),
    });
  });
  actionPanel.appendChild(form);
}

function renderBetControl(waiting) {
  const meta = document.createElement("p");
  meta.className = "callout";
  meta.textContent = `Call ${waiting.call_amount}`;
  actionPanel.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "action-buttons";
  for (const action of waiting.valid_actions) {
    const cost = waiting.action_costs?.[action];
    const button = document.createElement("button");
    button.type = "button";
    button.appendChild(document.createTextNode(action));
    const costText = formatActionCost(action, cost);
    if (costText) {
      const small = document.createElement("small");
      small.textContent = costText;
      button.appendChild(small);
    }
    if (cost) {
      button.title = actionTooltip(action, cost);
    }
    button.addEventListener("click", () => postJson("/api/action", { player: waiting.player, action }));
    actions.appendChild(button);
  }
  actionPanel.appendChild(actions);
}

function renderResultPanel(state) {
  if (state.phase === "game_over") {
    const winner = state.episode?.winner || state.session?.winner || "No winner";
    actionPanel.appendChild(textBlock(`Winner ${winner}`));
    actionPanel.appendChild(textBlock(`Total rounds ${state.episode?.total_rounds || state.session?.total_hands || 0}`));
  } else {
    actionPanel.appendChild(textBlock(`Round ${state.round_number || state.hand_number} complete`));
  }

  const summaries = state.result?.round_summaries || state.result?.hand_summaries || [];
  if (summaries.length) {
    const list = document.createElement("div");
    list.className = "result-grid wide";
    for (const player of summaries) {
      const profit = state.episode?.cumulative_profit?.[player.name] ?? state.session?.cumulative_profit?.[player.name] ?? 0;
      list.innerHTML += `
        <span>${escapeHtml(player.name)}</span>
        <span>${escapeHtml(player.hand_name)}</span>
        <strong>${player.chips} (${formatSigned(profit)})</strong>
      `;
    }
    actionPanel.appendChild(list);
  }

  if (state.replay_file || state.result?.replay_file) {
    const replay = state.replay_file || state.result.replay_file;
    const node = textBlock(`Replay saved: ${replay}`);
    node.className = "replay-path";
    actionPanel.appendChild(node);
  }

  if (state.next_round_available || state.next_hand_available) {
    if (autoRoundSettings.enabled) {
      actionPanel.appendChild(textBlock(`Auto next in ${formatSeconds(autoRoundSettings.intervalMs)}`));
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "primary-action";
    button.textContent = "Next Round";
    button.addEventListener("click", () => {
      clearAutoRoundTimer();
      postJson("/api/next_round", {});
    });
    actionPanel.appendChild(button);
  }
}

function renderTurnOrder(state) {
  if (!state.turn_order || state.turn_order.length === 0) return;

  const wrap = document.createElement("div");
  wrap.className = "turn-order";
  for (const name of state.turn_order) {
    const chip = document.createElement("span");
    chip.className = "order-chip";
    if (name === state.priority_player) chip.classList.add("dealer");
    if (name === state.acting_player) chip.classList.add("active");
    chip.textContent = name === state.priority_player ? `${name} · Dealer` : name;
    wrap.appendChild(chip);
  }
  actionPanel.appendChild(wrap);
}

function renderLogs(state) {
  bettingLog.innerHTML = "";
  for (const item of [...(state.betting_history || [])].reverse()) {
    const raiseText = item.raise_amount ? ` · raise ${item.raise_amount}` : "";
    bettingLog.appendChild(textBlock(`${item.street} · P${Number(item.actor_index) + 1} · ${item.action} · +${item.paid}${raiseText}`));
  }
  eventLog.innerHTML = "";
  for (const item of [...(state.events || [])].reverse()) {
    eventLog.appendChild(textBlock(item));
  }
}

function renderCards(cards, hidden, hiddenCount = 0) {
  if (hidden) {
    return Array.from({ length: hiddenCount }, () => `<span class="playing-card back">?</span>`).join("");
  }
  if (!cards || cards.length === 0) {
    return `<span class="empty-slot">-</span>`;
  }
  return cards.map((card) => `<span class="playing-card ${card[0] === "h" || card[0] === "d" ? "red" : "black"}">${escapeHtml(card)}</span>`).join("");
}

function textBlock(text) {
  const node = document.createElement("p");
  node.textContent = text;
  return node;
}

function panelTitle(state) {
  if (state.waiting) return `${state.waiting.player} Turn`;
  if (state.phase === "game_over") return "Episode Over";
  if (state.phase === "complete") return "Round Complete";
  return "Table";
}

function formatActionCost(action, cost) {
  if (!cost) return "";
  if (action === "CHECK" || action === "FOLD") return "";
  const prefix = action === "CALL" ? "" : "+";
  const suffix = cost.all_in ? " all-in" : "";
  return `(${prefix}${cost.paid}${suffix})`;
}

function actionTooltip(action, cost) {
  if (action === "CHECK" || action === "FOLD") return action;
  if (action === "CALL") return `Pay ${cost.paid}`;
  return `Pay ${cost.paid}: call ${cost.call_amount}, raise ${cost.raise_amount}`;
}

function formatSigned(value) {
  const number = Number(value || 0);
  return number > 0 ? `+${number}` : String(number);
}

function syncAutoRoundSettings() {
  const seconds = Math.max(0.5, Number(autoIntervalEl.value) || 2);
  autoRoundSettings = {
    enabled: autoRoundsEl.checked,
    intervalMs: Math.round(seconds * 1000),
  };
}

function scheduleAutoNextRound(state) {
  clearAutoRoundTimer();
  if (!autoRoundSettings.enabled || !state?.next_round_available || state.phase !== "complete") {
    return;
  }

  autoRoundTimer = setTimeout(async () => {
    if (currentState?.next_round_available && currentState.phase === "complete") {
      await postJson("/api/next_round", {});
    }
  }, autoRoundSettings.intervalMs);
}

function clearAutoRoundTimer() {
  if (autoRoundTimer) {
    clearTimeout(autoRoundTimer);
    autoRoundTimer = null;
  }
}

function formatSeconds(milliseconds) {
  const seconds = milliseconds / 1000;
  return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)}s`;
}

function renderError(message) {
  actionPanel.innerHTML = `<h2>Error</h2><p class="error">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadState();
