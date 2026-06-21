const setupForm = document.querySelector("#setupForm");
const playersEl = document.querySelector("#players");
const actionPanel = document.querySelector("#actionPanel");
const bettingLog = document.querySelector("#bettingLog");
const eventLog = document.querySelector("#eventLog");
const phaseBadge = document.querySelector("#phaseBadge");
const streetBadge = document.querySelector("#streetBadge");
const potBadge = document.querySelector("#potBadge");

let currentState = null;

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(setupForm);
  await postJson("/api/start", {
    player_types: ["p1", "p2", "p3", "p4", "p5"].map((key) => form.get(key)),
  });
});

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
  phaseBadge.textContent = state.phase || "idle";
  streetBadge.textContent = state.street || "-";
  potBadge.textContent = `Pot ${state.pot || 0}`;
  renderPlayers(state.players || []);
  renderActionPanel(state);
  renderLogs(state);
}

function renderPlayers(players) {
  playersEl.innerHTML = "";
  for (const player of players) {
    const card = document.createElement("article");
    card.className = `player-card ${player.status.toLowerCase()}`;
    card.innerHTML = `
      <div class="player-head">
        <div>
          <h2>${escapeHtml(player.name)}</h2>
          <p>${escapeHtml(player.type)} · ${escapeHtml(player.status)}</p>
        </div>
        <strong>${player.chips}</strong>
      </div>
      <div class="metric-row">
        <span>Invested ${player.invested}</span>
        <span>Round ${player.current_bet}</span>
      </div>
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
  title.textContent = state.waiting ? `${state.waiting.player} Turn` : "Table";
  actionPanel.appendChild(title);

  if (state.phase === "idle") {
    actionPanel.appendChild(textBlock("Ready"));
    return;
  }
  if (state.phase === "complete") {
    actionPanel.appendChild(textBlock("Complete"));
    if (state.result?.final_chips) {
      const list = document.createElement("div");
      list.className = "result-grid";
      for (const [name, chips] of Object.entries(state.result.final_chips)) {
        list.innerHTML += `<span>${escapeHtml(name)}</span><strong>${chips}</strong>`;
      }
      actionPanel.appendChild(list);
    }
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
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = action;
    button.addEventListener("click", () => postJson("/api/action", { player: waiting.player, action }));
    actions.appendChild(button);
  }
  actionPanel.appendChild(actions);
}

function renderLogs(state) {
  bettingLog.innerHTML = "";
  for (const item of [...(state.betting_history || [])].reverse()) {
    bettingLog.appendChild(textBlock(`${item.street} · P${Number(item.actor_index) + 1} · ${item.action} · +${item.paid}`));
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
