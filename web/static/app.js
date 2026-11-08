const setupForm = document.querySelector("#setupForm");
const autoRoundsEl = document.querySelector("#autoRounds");
const autoIntervalEl = document.querySelector("#autoInterval");
const playersEl = document.querySelector("#players");
const tablePot = document.querySelector("#tablePot");
const actionPanel = document.querySelector("#actionPanel");
const bettingLog = document.querySelector("#bettingLog");
const eventLog = document.querySelector("#eventLog");
const roundBadge = document.querySelector("#roundBadge");
const phaseBadge = document.querySelector("#phaseBadge");
const modeBadge = document.querySelector("#modeBadge");
const streetBadge = document.querySelector("#streetBadge");
const potBadge = document.querySelector("#potBadge");
const anteBadge = document.querySelector("#anteBadge");
const stackBadge = document.querySelector("#stackBadge");
const gameModeEl = document.querySelector("#gameMode");
const anteInputEl = document.querySelector("#anteInput");
const displayUnitEls = [...document.querySelectorAll('input[name="display_unit"]')];
const rangeObserver = document.querySelector("#rangeObserver");
const rangeSamples = document.querySelector("#rangeSamples");
const calculateRangeButton = document.querySelector("#calculateRange");
const rangeStatus = document.querySelector("#rangeStatus");
const rangeSummary = document.querySelector("#rangeSummary");
const rangeWorkspace = document.querySelector("#rangeWorkspace");
const rangeGrid = document.querySelector("#rangeGrid");
const rangeDetail = document.querySelector("#rangeDetail");
const analyzeButton = document.querySelector("#analyzeDecision");
const analysisStatus = document.querySelector("#analysisStatus");
const analysisCurrent = document.querySelector("#analysisCurrent");
const analysisList = document.querySelector("#analysisList");

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
let displayUnit = "chips";
let discardSelection = { key: null, discard: null, reveal: null };
let discardSubmitting = false;
let lastBetEffectKey = null;

playersEl.addEventListener("click", (event) => {
  const card = event.target.closest("[data-discard-index]");
  if (card) selectDiscardCard(Number(card.dataset.discardIndex));
});

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  syncAutoRoundSettings();
  const form = new FormData(setupForm);
  await postJson("/api/start", {
    player_types: ["p1", "p2", "p3", "p4", "p5"].map((key) => form.get(key)),
    game_mode: form.get("game_mode"),
    ante: Number(form.get("ante")),
  });
});

gameModeEl.addEventListener("change", () => {
  anteInputEl.value = gameModeEl.value === "ev" ? "1000" : "1";
});

for (const input of displayUnitEls) {
  input.addEventListener("change", () => {
    displayUnit = input.value;
    if (currentState) render(currentState);
  });
}

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
analyzeButton.addEventListener("click", analyzeCurrentDecision);

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
  syncDiscardSelection(state);
  gameModeEl.value = state.game_mode || "cash";
  anteInputEl.value = state.ante || 1;
  roundBadge.textContent = `Round ${state.round_number || state.hand_number || 0}`;
  modeBadge.textContent = state.game_mode || "cash";
  phaseBadge.textContent = state.phase || "idle";
  streetBadge.textContent = state.street || "-";
  potBadge.textContent = `Pot ${formatAmount(state.pot || 0)}`;
  anteBadge.textContent = `Ante ${formatAmount(state.ante ?? anteInputEl.value)}`;
  stackBadge.textContent = `Stack ${formatAmount(state.effective_stack || 0)}`;
  renderPlayers(state.players || [], state);
  renderTablePot(state);
  renderActionPanel(state);
  renderLogs(state);
  renderRangeControls(state);
  renderAnalysis(state);
  scheduleAutoNextRound(state);
}

// ---- ASWISS strategy analysis -----------------------------------------------------------------------------------
let analysisDecisionKey = null;

function humanDecisionKey(state) {
  const waiting = state?.waiting;
  if (!waiting || waiting.type !== "bet") return null;
  return `${state.round_number}:${state.street}:${(state.betting_history || []).length}:${waiting.player}`;
}

function renderAnalysis(state) {
  const available = Boolean(state.analysis_available);
  const key = humanDecisionKey(state);
  analyzeButton.disabled = !(available && key);
  if (key !== analysisDecisionKey) {
    analysisDecisionKey = key;
    analysisCurrent.classList.add("is-hidden");
    analysisCurrent.innerHTML = "";
  }
  const records = state.analysis || [];
  if (!available) {
    analysisStatus.textContent = "ASWISS 분석은 헤즈업 EV 게임(ante 1000)에서 쓸 수 있습니다.";
  } else if (key) {
    analysisStatus.textContent = `${state.waiting.player}의 차례입니다. "ASWISS 추천 보기"로 이 결정에서의 ASWISS 정책을 볼 수 있습니다.`;
  } else {
    analysisStatus.textContent = records.length ? `이번 판의 결정 ${records.length}개` : "이번 판에 분석할 결정이 아직 없습니다.";
  }
  analysisList.innerHTML = "";
  for (const record of records.slice().reverse()) analysisList.appendChild(analysisCard(record, false));
}

async function analyzeCurrentDecision() {
  const waiting = currentState?.waiting;
  if (!waiting || waiting.type !== "bet") return;
  analyzeButton.disabled = true;
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player: waiting.player }),
    });
    const data = await response.json();
    analysisCurrent.innerHTML = "";
    analysisCurrent.classList.remove("is-hidden");
    if (!response.ok) {
      analysisCurrent.appendChild(textBlock(data.error || "분석 실패"));
    } else {
      analysisCurrent.appendChild(analysisCard({ ...data, player: waiting.player, by: "now" }, true));
    }
  } finally {
    analyzeButton.disabled = !(currentState?.analysis_available && humanDecisionKey(currentState));
  }
}

function analysisCard(record, isCurrent) {
  const card = document.createElement("article");
  card.className = "analysis-card";
  const who = record.by === "aswiss" ? "ASWISS" : record.by === "human" ? "사람" : "지금 결정";
  const head = document.createElement("header");
  const taken = record.action_taken
    ? ` · 선택 <strong>${escapeHtml(record.action_taken)}</strong>` +
      (record.probability_of_action_taken != null
        ? ` (ASWISS 확률 ${formatPercent(record.probability_of_action_taken)})`
        : "")
    : "";
  head.innerHTML = `<span class="analysis-tag tag-${escapeHtml(record.by)}">${who}</span> ${escapeHtml(record.player || "")} · ${escapeHtml(record.street || "")}${taken}`;
  card.appendChild(head);
  if (record.error) {
    card.appendChild(textBlock(`분석 불가: ${record.error}`));
    if (record.fallback) card.appendChild(textBlock(`이 결정은 ${record.fallback} 에이전트가 대신 두었습니다.`));
    return card;
  }
  if (record.fallback) card.appendChild(textBlock(`이 결정은 ${record.fallback} 에이전트가 대신 두었습니다.`));
  card.appendChild(probabilityBars(record.probabilities || {}, record.action_taken, isCurrent));
  const meta = document.createElement("dl");
  meta.className = "analysis-meta";
  const key = record.key || {};
  const rows = [
    ["조회", `${record.resolved || "-"}${record.hops ? ` (조상 ${record.hops}단계)` : ""}`],
    ["bucket", record.bucket ?? "-"],
    ["계보", (record.genealogy || []).join(" ← ") || "-"],
    ["직전 내 결정", key.prev_bucket == null ? "없음(이 판 첫 결정)" : `bucket ${key.prev_bucket} · ${key.prev_action || "-"}`],
    ["그 뒤 상대 행동", (key.opponent_actions || []).join(", ") || "없음"],
    ["카드 군집", key.cluster == null ? "새 카드 없음" : key.cluster],
  ];
  if (record.current) rows.push(["현재 정책", Object.entries(record.current).map(([a, p]) => `${a} ${formatPercent(p, 0)}`).join(" · ")]);
  if (record.stack_matches_training === false) rows.push(["주의", "스택이 학습(ante의 1000배)과 다릅니다."]);
  for (const [label, value] of rows) {
    const dt = document.createElement("dt"); dt.textContent = label;
    const dd = document.createElement("dd"); dd.textContent = String(value);
    meta.append(dt, dd);
  }
  card.appendChild(meta);
  return card;
}

function probabilityBars(probabilities, taken, isCurrent) {
  const wrap = document.createElement("div");
  wrap.className = "prob-bars";
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  for (const [action, p] of entries) {
    const row = document.createElement("div");
    row.className = "prob-row" + (action === taken ? " is-taken" : "");
    row.innerHTML = `<span class="prob-label">${escapeHtml(action)}</span>
      <span class="prob-track"><span class="prob-fill" style="width:${Math.max(0, Math.min(100, p * 100)).toFixed(1)}%"></span></span>
      <span class="prob-value">${formatPercent(p)}</span>`;
    wrap.appendChild(row);
  }
  if (isCurrent && entries.length) {
    const best = entries[0];
    const note = document.createElement("p");
    note.className = "prob-note";
    note.textContent = `가장 자주 고르는 행동: ${best[0]} (${formatPercent(best[1])}). 혼합 전략이므로 다른 행동도 위 확률로 섞어 씁니다.`;
    wrap.appendChild(note);
  }
  return wrap;
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
  label.className = `range-axis ${position} ${cardSuitClass(card)}`;
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
  const latestActions = new Map();
  for (const event of state.betting_history || []) {
    if (event.street === state.street) {
      latestActions.set(players[event.actor_index]?.name, event);
    }
  }
  const latestEvent = (state.betting_history || []).at(-1);
  const latestEventKey = latestEvent
    ? `${state.round_number}:${state.betting_history.length}:${latestEvent.actor_index}:${latestEvent.action}:${latestEvent.pot_after}`
    : null;
  const animateLatest = latestEventKey !== lastBetEffectKey;
  const humanIndex = players.findIndex((player) => player.type === "human");
  const seatedPlayers = humanIndex < 0
    ? players
    : players.map((_, index) => players[(humanIndex + index) % players.length]);
  const seats = seatPositions(seatedPlayers.length);
  seatedPlayers.forEach((player, index) => {
    const card = document.createElement("article");
    const classes = ["player-card", player.status.toLowerCase(), `seat-${index}`];
    if (player.type === "human") classes.push("hero");
    if (state.waiting?.type === "discard" && state.waiting.player === player.name) classes.push("selecting");
    if (player.is_acting) classes.push("acting");
    if (player.has_priority) classes.push("priority");
    card.className = classes.join(" ");
    card.style.setProperty("--seat-x", `${seats[index][0]}%`);
    card.style.setProperty("--seat-y", `${seats[index][1]}%`);

    const badges = [
      player.has_priority ? `<span class="player-badge dealer">Dealer</span>` : "",
      player.is_acting ? `<span class="player-badge turn">Turn</span>` : "",
    ].join("");
    const playerAction = latestActions.get(player.name);
    const actionText = playerAction ? formatPlayerAction(playerAction) : "";
    const actionEffect = animateLatest && playerAction === latestEvent ? " action-effect" : "";
    card.innerHTML = `
      <div class="player-head">
        <div>
          <h2>${escapeHtml(player.name)}</h2>
          <p>${escapeHtml(player.type)} · ${escapeHtml(player.status)}</p>
        </div>
        <strong>${formatAmount(player.chips)}</strong>
      </div>
      <div class="badge-row">${badges}</div>
      <div class="metric-row">
        <span>Invested ${formatAmount(player.invested)}</span>
        <span>Bet ${formatAmount(player.current_bet)}</span>
        <span>Net ${formatSignedAmount(player.net || 0)}</span>
      </div>
      <div class="hand-name">Current hand · ${escapeHtml(player.hand_name || "-")}</div>
      <div class="card-group">
        <span>Public</span>
        <div class="cards">${renderCards(player.public_cards, false)}</div>
      </div>
      <div class="card-group">
        <span>Hidden</span>
        <div class="cards">${renderCards(
          player.hidden_cards,
          player.hidden_count > 0 && player.hidden_cards.length === 0,
          player.hidden_count,
          state.waiting?.type === "discard" && state.waiting.player === player.name,
        )}</div>
      </div>
    `;
    card.querySelector(".hand-name").textContent =
      `${player.type === "human" ? "확정 족보 · " : ""}${player.hand_name || "-"}`;
    const cardGroups = card.querySelectorAll(".card-group");
    cardGroups[0].classList.add("public-cards");
    cardGroups[1].classList.add("private-cards");
    const actionNode = document.createElement("div");
    actionNode.className = `player-action${playerAction ? ` action-${playerAction.action.toLowerCase()}` : ""}${actionEffect}`;
    actionNode.textContent = actionText;
    card.appendChild(actionNode);
    playersEl.appendChild(card);
  });
  lastBetEffectKey = latestEventKey;
}

function seatPositions(count) {
  const layouts = {
    1: [[50, 68]],
    2: [[50, 68], [50, 3]],
    3: [[50, 68], [22, 3], [78, 3]],
    4: [[50, 68], [22, 36], [50, 3], [78, 36]],
    5: [[50, 68], [22, 37], [22, 3], [78, 3], [78, 37]],
  };
  return layouts[count] || layouts[5];
}

function renderTablePot(state) {
  const history = state.betting_history || [];
  const latest = history.at(-1);
  tablePot.className = "table-pot";
  tablePot.querySelector("strong").textContent = formatAmount(state.pot || 0);

  if (!latest) {
    tablePot.querySelector("small").textContent = "Waiting for action";
    return;
  }

  const actor = state.players?.[latest.actor_index]?.name || `P${Number(latest.actor_index) + 1}`;
  const potBefore = Math.max(0, Number(latest.pot_after) - Number(latest.paid));
  const growth = latest.paid && potBefore > 0 ? ` · Pot +${((latest.paid / potBefore) * 100).toFixed(1)}%` : "";
  const paid = latest.paid ? ` · +${formatAmount(latest.paid)}` : "";
  tablePot.querySelector("small").textContent = `${actor} · ${latest.action}${paid}${growth}`;
  tablePot.classList.add(`action-${latest.action.toLowerCase()}`);

  const effectKey = `${state.round_number}:${history.length}:${latest.actor_index}:${latest.action}:${latest.pot_after}`;
  if (effectKey !== lastBetEffectKey) {
    lastBetEffectKey = effectKey;
    tablePot.classList.add("pot-effect");
  }
}

function formatPlayerAction(event) {
  const potBefore = Math.max(0, Number(event.pot_after) - Number(event.paid));
  const growth = event.paid && potBefore > 0 ? ` · Pot +${((event.paid / potBefore) * 100).toFixed(1)}%` : "";
  const paid = event.paid ? ` · +${formatAmount(event.paid)}` : "";
  return `${event.action}${paid}${growth}`;
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
  const controls = document.createElement("div");
  controls.className = "discard-control";
  const discardCard = discardSelection.discard === null ? "-" : waiting.cards[discardSelection.discard];
  const revealCard = discardSelection.reveal === null ? "-" : waiting.cards[discardSelection.reveal];
  controls.innerHTML = `
    <div class="discard-summary">
      <span>Discard <strong>${escapeHtml(discardCard)}</strong></span>
      <span>Reveal <strong>${escapeHtml(revealCard)}</strong></span>
    </div>
  `;
  actionPanel.appendChild(controls);
}

function renderBetControl(waiting) {
  const meta = document.createElement("p");
  meta.className = "callout";
  meta.textContent = `Call ${formatAmount(waiting.call_amount)}`;
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
        <strong>${formatAmount(player.chips)} (${formatSignedAmount(profit)})</strong>
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
    const raiseText = item.raise_amount ? ` · raise ${formatAmount(item.raise_amount)}` : "";
    bettingLog.appendChild(textBlock(`${item.street} · P${Number(item.actor_index) + 1} · ${item.action} · +${formatAmount(item.paid)}${raiseText}`));
  }
  eventLog.innerHTML = "";
  for (const item of [...(state.events || [])].reverse()) {
    eventLog.appendChild(textBlock(item));
  }
}

function renderCards(cards, hidden, hiddenCount = 0, interactive = false) {
  if (hidden) {
    return Array.from({ length: hiddenCount }, () => `<span class="playing-card back">?</span>`).join("");
  }
  if (!cards || cards.length === 0) {
    return `<span class="empty-slot">-</span>`;
  }
  return cards.map((card, index) => {
    const choice = discardSelection.discard === index
      ? ["selected-discard", "Discard"]
      : discardSelection.reveal === index
        ? ["selected-reveal", "Reveal"]
        : ["", ""];
    const tag = interactive ? "button" : "span";
    const attributes = interactive
      ? ` type="button" data-discard-index="${index}" aria-pressed="${choice[0] ? "true" : "false"}"`
      : "";
    const marker = choice[1] ? `<small>${choice[1]}</small>` : "";
    return `<${tag}${attributes} class="playing-card ${cardSuitClass(card)} ${choice[0]}">${escapeHtml(card)}${marker}</${tag}>`;
  }).join("");
}

function syncDiscardSelection(state) {
  const waiting = state.waiting;
  const key = waiting?.type === "discard"
    ? `${state.round_number}:${waiting.player}:${waiting.cards.join(",")}`
    : null;
  if (key !== discardSelection.key) {
    discardSelection = { key, discard: null, reveal: null };
  }
}

async function selectDiscardCard(index) {
  if (discardSubmitting) return;
  if (discardSelection.discard === index) {
    discardSelection.discard = null;
  } else if (discardSelection.reveal === index) {
    discardSelection.reveal = null;
  } else if (discardSelection.discard === null) {
    discardSelection.discard = index;
  } else {
    discardSelection.reveal = index;
  }
  render(currentState);

  if (discardSelection.discard !== null && discardSelection.reveal !== null) {
    discardSubmitting = true;
    try {
      await postJson("/api/discard", {
        player: currentState.waiting.player,
        discard_index: discardSelection.discard,
        reveal_index: discardSelection.reveal,
      });
    } finally {
      discardSubmitting = false;
    }
  }
}

function cardSuitClass(card) {
  return `suit-${String(card || "")[0] || "unknown"}`;
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
  return `(${prefix}${formatAmount(cost.paid)}${suffix})`;
}

function actionTooltip(action, cost) {
  if (action === "CHECK" || action === "FOLD") return action;
  if (action === "CALL") return `Pay ${formatAmount(cost.paid)}`;
  return `Pay ${formatAmount(cost.paid)}: call ${formatAmount(cost.call_amount)}, raise ${formatAmount(cost.raise_amount)}`;
}

function formatAmount(value) {
  const number = Number(value || 0);
  if (displayUnit === "ante") {
    const ante = Math.max(1, Number(currentState?.ante ?? anteInputEl.value ?? 1));
    return `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(number / ante)}a`;
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(number);
}

function formatSignedAmount(value) {
  const number = Number(value || 0);
  return number > 0 ? `+${formatAmount(number)}` : formatAmount(number);
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
