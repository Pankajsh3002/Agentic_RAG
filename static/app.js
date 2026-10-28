(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const TABS = ["workspaces", "ask", "summary", "timeline", "connections"];
  const TAB_LABELS = {
    workspaces: "Workspace", ask: "Query Document", summary: "Summary",
    timeline: "Timelines", connections: "Connections",
  };
  const STEP_ORDER = ["chunks", "summary", "events", "graph"];
  const STEP_LABELS = { chunks: "Text index", summary: "Summary", events: "Key dates", graph: "Connections" };
  const STATUS_LABELS = { done: "Ready for query", pending: "Waiting", processing: "Processing", failed: "Failed" };

  // Icon shapes (24x24 outline icons). These are fixed constants, never user data.
  const ICONS = {
    layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    chat: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    summary: '<line x1="17" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="17" y1="18" x2="3" y2="18"/>',
    clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    share: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    chevron: '<polyline points="6 9 12 15 18 9"/>',
    folder: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>',
    check: '<polyline points="20 6 9 17 4 12"/>',
    alert: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
    send: '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
  };

  const state = {
    workspaces: [], wsId: null, expanded: true,
    docs: [], lastDone: null, pollTimer: null,
    tab: "workspaces",
    chats: {}, askBuiltFor: null,
    uploadTarget: null, uploading: [], notices: [], errors: [], md: null,
    summaries: [], summaryDoc: null,
    events: [], timelineDoc: "all",
    graphQ: "",
  };

  // ---------- helpers ----------
  function icon(name, size = 18) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = ICONS[name] || "";
    return svg;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (e) {}
    if (!res.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : "Request failed (" + res.status + ")";
      throw new Error(detail);
    }
    return data;
  }

  // Build elements with textContent, never innerHTML: filenames, quotes and errors come from users.
  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  let bannerTimer = null;
  function showError(e) {
    const b = $("banner");
    b.textContent = e && e.message ? e.message : String(e);
    b.classList.remove("hidden");
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => b.classList.add("hidden"), 7000);
  }
  const guard = (fn) => async (...args) => {
    try { await fn(...args); } catch (e) { showError(e); }
  };

  function storedWs() { try { return localStorage.getItem("rag_ws"); } catch (e) { return null; } }
  function saveWs(id) {
    try { id ? localStorage.setItem("rag_ws", id) : localStorage.removeItem("rag_ws"); } catch (e) {}
  }

  function buildTable(headers, rows) {
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    headers.forEach((h) => hr.appendChild(el("th", h)));
    thead.appendChild(hr);
    const tbody = document.createElement("tbody");
    rows.forEach((cells) => {
      const tr = document.createElement("tr");
      cells.forEach((c) => tr.appendChild(el("td", c == null ? "" : String(c))));
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    return table;
  }

  const currentWs = () => state.workspaces.find((w) => w.workspace_id === state.wsId) || null;
  const needWorkspace = (box) =>
    box.appendChild(el("p", "Create a workspace in the Workspace section first.", "muted"));

  // ---------- sections (tabs) ----------
  function updateCrumbs() {
    const c = $("crumbs");
    c.replaceChildren();
    const ws = currentWs();
    const parts = ["Agentic RAG"];
    if (ws) parts.push(ws.name);
    parts.push(TAB_LABELS[state.tab]);
    parts.forEach((p, i) => {
      if (i) c.appendChild(el("span", ">", "sep"));
      c.appendChild(el("span", p, i === parts.length - 1 ? "current" : ""));
    });
  }

  function applyTab(name) {
    if (!TABS.includes(name)) name = "workspaces";
    state.tab = name;
    document.querySelectorAll(".nav-item").forEach((b) => {
      const on = b.dataset.tab === name;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".tabpanel").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + name));
    updateCrumbs();
  }

  async function showTab(name) {
    applyTab(name);
    try { history.replaceState(null, "", "#" + state.tab); } catch (e) {}
    await loadActiveTab();
  }

  async function loadActiveTab() {
    switch (state.tab) {
      case "summary": return loadSummaries();
      case "timeline": return loadTimeline();
      case "connections": return loadConnections();
      case "ask": return renderAsk();
      default: return;
    }
  }

  // ---------- workspaces ----------
  async function fetchWorkspaces() { state.workspaces = await api("/workspaces"); }

  function resetData() {
    clearTimeout(state.pollTimer);
    state.docs = []; state.lastDone = null;
    state.uploading = []; state.notices = []; state.errors = []; state.md = null;
    state.summaries = []; state.summaryDoc = null;
    state.events = []; state.timelineDoc = "all";
    state.graphQ = ""; $("graphQ").value = "";
  }

  async function loadWorkspaces(preferId) {
    await fetchWorkspaces();
    const ids = state.workspaces.map((w) => w.workspace_id);
    let pick = preferId || state.wsId || storedWs();
    if (!ids.includes(pick)) pick = ids[0] || null;
    if (pick !== state.wsId) await selectWorkspace(pick);
    else renderChrome();
  }

  async function selectWorkspace(id) {
    resetData();
    state.wsId = id;
    state.expanded = !!id;
    saveWs(id);
    renderChrome();
    if (id) await loadDocs();
    await loadActiveTab();
  }

  async function refreshWorkspaceList() {
    await fetchWorkspaces();
    if (state.wsId && !currentWs()) { state.wsId = null; await loadWorkspaces(); return; }
    renderChrome();
  }

  async function createWorkspace() {
    $("wsErr").textContent = "";
    try {
      const ws = await api("/workspaces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("wsName").value.trim() || "My workspace" }),
      });
      $("wsName").value = "";
      await loadWorkspaces(ws.workspace_id);
    } catch (e) { $("wsErr").textContent = e.message; }
  }

  async function deleteWorkspace(w) {
    if (!confirm('Delete "' + w.name + '" and all its documents?')) return;
    await api("/workspaces/" + w.workspace_id, { method: "DELETE" });
    if (w.workspace_id === state.wsId) state.wsId = null;
    await loadWorkspaces();
    if (!state.wsId) { resetData(); renderChrome(); await loadActiveTab(); }
  }

  async function toggleWorkspace(id) {
    if (id === state.wsId) { state.expanded = !state.expanded; renderChrome(); return; }
    await selectWorkspace(id);
  }

  async function openQuery(id) {
    if (id !== state.wsId) await selectWorkspace(id);
    await showTab("ask");
  }

  // ---------- rendering: sidebar and workspace cards ----------
  function renderChrome() {
    renderSidebar();
    renderWorkspaceCards();
    updateCrumbs();
    if (state.tab === "ask") renderAsk();
  }

  function renderSidebar() {
    const sel = $("wsSelect");
    sel.replaceChildren();
    if (!state.workspaces.length) {
      sel.appendChild(new Option("No workspaces yet", ""));
      sel.disabled = true;
    } else {
      sel.disabled = false;
      state.workspaces.forEach((w) => sel.appendChild(new Option(w.name, w.workspace_id)));
      sel.value = state.wsId || "";
    }
    const ws = currentWs();
    $("wsMeta").textContent = ws ? ws.ready_documents + " of " + ws.documents + " file(s) ready for query" : "";
    $("askDot").classList.toggle("hidden", !(ws && ws.ready_documents > 0));
  }

  function renderWorkspaceCards() {
    const box = $("wsCards");
    box.replaceChildren();
    if (!state.workspaces.length) {
      box.appendChild(el("p", "No workspaces yet. Create one to get started.", "muted"));
      return;
    }

    state.workspaces.forEach((w) => {
      const selected = w.workspace_id === state.wsId;
      const open = selected && state.expanded;
      const card = el("div", "", "ws-card" + (selected ? " selected" : "") + (open ? " open" : ""));

      const head = el("div", "", "ws-head");
      head.tabIndex = 0;
      head.setAttribute("role", "button");
      head.setAttribute("aria-expanded", open ? "true" : "false");
      head.onclick = guard(() => toggleWorkspace(w.workspace_id));
      head.addEventListener("keydown", (ev) => {
        if (ev.target === head && (ev.key === "Enter" || ev.key === " ")) {
          ev.preventDefault();
          guard(() => toggleWorkspace(w.workspace_id))();
        }
      });

      const chev = el("span", "", "chev");
      chev.appendChild(icon("chevron", 18));
      const folder = el("span", "", "ws-folder");
      folder.appendChild(icon("folder", 18));
      const info = el("div", "", "grow");
      info.append(
        el("div", w.name, "ws-name"),
        el("div", w.documents + " file(s), " + w.ready_documents + " ready for query", "muted small")
      );

      const actions = el("div", "", "ws-actions");
      if (w.ready_documents > 0) {
        const query = el("button", "", "icon-btn query");
        query.title = "Ask questions about this workspace's documents";
        query.append(icon("chat", 16), document.createTextNode("Query"));
        query.onclick = guard(async (ev) => { ev.stopPropagation(); await openQuery(w.workspace_id); });
        actions.appendChild(query);
      }
      const up = el("button", "", "icon-btn upload");
      up.title = "Upload documents";
      up.append(icon("upload", 16), document.createTextNode("Upload"));
      up.onclick = (ev) => {
        ev.stopPropagation();
        state.uploadTarget = w.workspace_id;
        $("file").click();            // must run directly inside the click
      };
      actions.appendChild(up);

      const del = el("button", "", "icon-btn danger-btn");
      del.title = "Delete workspace";
      del.setAttribute("aria-label", "Delete workspace " + w.name);
      del.appendChild(icon("trash", 16));
      del.onclick = guard(async (ev) => { ev.stopPropagation(); await deleteWorkspace(w); });
      actions.appendChild(del);

      head.append(chev, folder, info, actions);
      card.appendChild(head);
      if (open) card.appendChild(workspaceBody());

      card.addEventListener("dragover", (ev) => { ev.preventDefault(); card.classList.add("drag"); });
      card.addEventListener("dragleave", (ev) => {
        if (!card.contains(ev.relatedTarget)) card.classList.remove("drag");
      });
      card.addEventListener("drop", (ev) => {
        ev.preventDefault();
        card.classList.remove("drag");
        const files = Array.from((ev.dataTransfer && ev.dataTransfer.files) || []);
        if (files.length) guard(handleFiles)(w.workspace_id, files);
      });

      box.appendChild(card);
    });
  }

  function workspaceBody() {
    const body = el("div", "", "ws-body");

    state.errors.forEach((m) => body.appendChild(el("div", m, "err-line")));

    state.notices.forEach((n) => {
      const row = el("div", "", "notice");
      row.appendChild(el("span", "\u201C" + n.file.name + "\u201D is already in this workspace (" +
                                 n.status + "). It was not processed again."));
      const replace = el("button", "Replace it", "btn-sm");
      replace.onclick = guard(() => replaceNotice(n));
      const keep = el("button", "Keep existing", "btn-sm");
      keep.onclick = () => { state.notices = state.notices.filter((x) => x !== n); renderChrome(); };
      row.append(replace, keep);
      body.appendChild(row);
    });

    state.uploading.forEach((name) => {
      const row = el("div", "", "doc-row");
      const top = el("div", "", "doc-top");
      top.append(icon("file", 16), el("span", name, "doc-name"), el("span", "uploading...", "small st-processing"));
      row.appendChild(top);
      body.appendChild(row);
    });

    if (!state.docs.length && !state.uploading.length) {
      body.appendChild(el("p", "No documents yet. Click Upload, or drop files onto this workspace.", "muted small"));
    }
    state.docs.forEach((d) => body.appendChild(docRow(d)));
    return body;
  }

  function docRow(d) {
    const row = el("div", "", "doc-row");
    const top = el("div", "", "doc-top");
    top.append(icon("file", 16), el("span", d.filename, "doc-name"),
               el("span", STATUS_LABELS[d.status] || d.status, "pill pill-" + d.status));

    const actions = el("span", "", "doc-actions");
    const showingMd = state.md && state.md.docId === d.doc_id;
    if (d.status === "done") {
      const m = el("button", showingMd ? "Hide markdown" : "View markdown", "btn-sm");
      m.onclick = guard(() => toggleMarkdown(d.doc_id));
      actions.appendChild(m);
    }
    const hasFailedStep = Object.values(d.branches || {}).some((b) => b.status === "failed");
    if (hasFailedStep && (d.status === "done" || d.status === "failed")) {
      const r = el("button", "Retry failed steps", "btn-sm");
      r.onclick = guard(() => retryDoc(d.doc_id));
      actions.appendChild(r);
    }
    top.appendChild(actions);
    row.appendChild(top);

    const entries = Object.entries(d.branches || {})
      .sort((a, b) => STEP_ORDER.indexOf(a[0]) - STEP_ORDER.indexOf(b[0]));
    if (entries.length) {
      const steps = el("div", "", "doc-steps");
      entries.forEach(([name, b]) => {
        const chip = el("span", (STEP_LABELS[name] || name) + ": " + b.status, "step st-" + b.status);
        if (b.error) chip.title = b.error;
        steps.appendChild(chip);
      });
      row.appendChild(steps);
    }
    if (d.status === "failed" && d.error) row.appendChild(el("div", d.error.slice(0, 160), "doc-err muted small"));
    if (showingMd) row.appendChild(el("pre", state.md.text, "md"));
    return row;
  }

  // ---------- documents ----------
  async function loadDocs() {
    const id = state.wsId;
    if (!id) return;
    const docs = await api("/workspaces/" + id + "/documents");
    if (id !== state.wsId) return;   // the user switched workspace meanwhile
    state.docs = docs;
    renderChrome();

    const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
    const done = docs.filter((d) => d.status === "done").length;
    if (state.lastDone !== null && done !== state.lastDone) loadActiveTab().catch(showError);
    state.lastDone = done;

    clearTimeout(state.pollTimer);
    if (busy) state.pollTimer = setTimeout(guard(pollTick), 2000);
  }

  async function pollTick() { await Promise.all([loadDocs(), refreshWorkspaceList()]); }

  async function handleFiles(wsId, files) {
    if (!files.length) return;
    if (state.wsId !== wsId) await selectWorkspace(wsId);
    state.expanded = true;
    state.errors = [];
    state.notices = [];
    renderChrome();
    for (const file of files) await uploadOne(wsId, file, false);
  }

  async function uploadOne(wsId, file, force) {
    state.uploading.push(file.name);
    renderChrome();
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("force", force ? "true" : "false");
      const data = await api("/workspaces/" + wsId + "/documents", { method: "POST", body: form });
      if (state.wsId !== wsId) return;
      if (data.duplicate) state.notices.push({ file, status: data.status });
    } catch (e) {
      if (state.wsId === wsId) state.errors.push(file.name + ": " + e.message);
    } finally {
      const i = state.uploading.indexOf(file.name);
      if (i >= 0) state.uploading.splice(i, 1);
    }
    if (state.wsId !== wsId) return;
    await Promise.all([loadDocs(), refreshWorkspaceList()]);
  }

  async function replaceNotice(n) {
    state.notices = state.notices.filter((x) => x !== n);
    renderChrome();
    await uploadOne(state.wsId, n.file, true);
  }

  async function toggleMarkdown(docId) {
    if (state.md && state.md.docId === docId) { state.md = null; renderChrome(); return; }
    const data = await api("/workspaces/" + state.wsId + "/documents/" + docId + "/markdown");
    state.md = { docId, text: data.markdown };
    renderChrome();
  }

  async function retryDoc(docId) {
    state.errors = [];
    await api("/workspaces/" + state.wsId + "/documents/" + docId + "/retry", { method: "POST" });
    await loadDocs();
  }

  // ---------- query document ----------
  const STARTERS = [
    "What is each document about?",
    "What are the key dates and deadlines?",
    "Which people and organizations are involved?",
  ];

  function chatFor(id) {
    if (!state.chats[id]) state.chats[id] = { messages: [], busy: false };
    return state.chats[id];
  }

  function renderAsk() {
    const box = $("askBody");
    const ws = currentWs();
    if (!ws) { state.askBuiltFor = null; box.replaceChildren(); needWorkspace(box); return; }
    // Build the screen once per workspace, so typing is never wiped by background refreshes.
    if (state.askBuiltFor !== ws.workspace_id || !$("chatLog")) buildAskShell(box, ws.workspace_id);
    else updateAskControls();
  }

  function autoGrow(input) {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  function buildAskShell(box, wsId) {
    box.replaceChildren();
    const col = el("div", "", "chat-col");

    const top = el("div", "", "chat-top");
    const status = el("span", "", "small");
    status.id = "askStatus";
    const clear = el("button", "Clear chat", "btn-sm");
    clear.id = "chatClear";
    clear.onclick = () => clearChat();
    top.append(status, clear);

    const log = el("div", "", "chat-log");
    log.id = "chatLog";

    const composer = el("div", "", "composer");
    const input = document.createElement("textarea");
    input.id = "chatInput";
    input.rows = 1;
    input.maxLength = 500;
    input.addEventListener("input", () => autoGrow(input));
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); guard(submitChat)(); }
    });
    const send = el("button", "", "primary send");
    send.id = "chatSend";
    send.setAttribute("aria-label", "Send question");
    send.append(icon("send", 16), document.createTextNode("Send"));
    send.onclick = guard(submitChat);
    composer.append(input, send);

    const hint = el("p", "Answers come only from this workspace's documents. Enter sends, Shift+Enter adds a new line.",
                    "muted small chat-hint");

    col.append(top, log, composer, hint);
    box.appendChild(col);
    state.askBuiltFor = wsId;
    renderChat();
  }

  function updateAskControls() {
    const ws = currentWs(), input = $("chatInput");
    if (!ws || !input) return;
    const chat = chatFor(ws.workspace_id);
    const ready = ws.ready_documents > 0;
    const status = $("askStatus");
    status.textContent = ready
      ? ws.ready_documents + " of " + ws.documents + " file(s) ready for query"
      : "Upload a document and wait for processing to finish. Then you can ask questions here.";
    status.className = "small " + (ready ? "ok" : "muted");
    input.disabled = !ready || chat.busy;
    $("chatSend").disabled = !ready || chat.busy;
    $("chatClear").classList.toggle("hidden", !chat.messages.length || chat.busy);
    input.placeholder = ready ? "Ask a question about this workspace's documents"
                              : "Waiting for a document to finish processing";
  }

  function renderChat() {
    const ws = currentWs(), log = $("chatLog");
    if (!ws || !log) return;
    const chat = chatFor(ws.workspace_id);
    log.replaceChildren();

    if (!chat.messages.length) {
      const empty = el("div", "", "chat-empty");
      empty.append(
        el("h3", "Ask about your documents"),
        el("p", "Answers are checked against the documents and come with their sources.", "muted small")
      );
      const row = el("div", "", "chips center");
      STARTERS.forEach((q) => {
        const b = el("button", q, "chip");
        b.onclick = guard(() => sendQuestion(q));
        row.appendChild(b);
      });
      empty.appendChild(row);
      log.appendChild(empty);
    }

    chat.messages.forEach((m) => log.appendChild(messageEl(m)));
    if (chat.busy) log.appendChild(el("div", "Searching your documents...", "msg assistant thinking"));
    updateAskControls();

    // After an answer arrives, show its beginning. While waiting, stay at the bottom.
    const last = log.lastElementChild;
    if (!chat.busy && last && last.classList.contains("assistant")) log.scrollTop = Math.max(0, last.offsetTop - 8);
    else log.scrollTop = log.scrollHeight;
  }

  function clearChat() {
    const ws = currentWs();
    if (!ws) return;
    const chat = chatFor(ws.workspace_id);
    if (chat.busy) return;
    chat.messages = [];
    renderChat();
  }

  function messageEl(m) {
    if (m.role === "user") return el("div", m.text, "msg user");
    const box = el("div", "", "msg assistant" + (m.error ? " error-msg" : ""));
    box.appendChild(el("div", m.text, "msg-text"));
    if (m.verification) box.appendChild(verificationEl(m.verification));
    if (m.steps && m.steps.length) {
      const tools = el("div", "", "msg-tools");
      tools.appendChild(el("div", "Tools used", "muted small"));
      m.steps.forEach((s) => tools.appendChild(stepEl(s)));
      box.appendChild(tools);
    }
    return box;
  }

  function badgeText(v) {
    const claims = v.claims || [];
    const total = claims.length;
    const ok = claims.filter((c) => c.verdict === "supported").length;
    const bad = total - ok;
    switch (v.status) {
      case "grounded": return "Verified: " + ok + " of " + total + " claims supported";
      case "partly": return bad
        ? bad + " of " + total + " claims not verified"
        : "Some citations or numbers could not be verified";
      case "ungrounded": return "Only " + ok + " of " + total + " claims supported. Treat this answer with caution";
      case "no_claims": return "No factual claims to verify";
      case "basic": return "Citations and numbers checked (claim check unavailable)";
      default: return "";
    }
  }

  function verificationEl(v) {
    const wrap = el("div", "", "verify");
    const text = badgeText(v);
    if (text) {
      const cls = { grounded: "v-ok", partly: "v-warn", ungrounded: "v-bad" }[v.status] || "v-muted";
      const iconName = v.status === "grounded" ? "shield"
        : (v.status === "partly" || v.status === "ungrounded") ? "alert" : "check";
      const badge = el("div", "", "v-badge " + cls);
      badge.append(icon(iconName, 14), document.createTextNode(text));
      wrap.appendChild(badge);
    }
    if (v.repaired) {
      wrap.appendChild(el("p", "This answer was rewritten once after a check found unsupported statements.", "small muted"));
    }
    (v.issues || []).forEach((i) => wrap.appendChild(el("p", i, "v-issue")));

    if ((v.sources || []).length) {
      const src = el("div", "", "v-sources");
      src.appendChild(el("div", "Sources", "muted small"));
      v.sources.forEach((s) => {
        const row = el("div", "", "v-src" + (s.verified ? "" : " flagged"));
        row.appendChild(icon(s.verified ? "check" : "alert", 14));
        const body = el("span");
        body.appendChild(el("span", s.file || "file not stated", "v-file"));
        if (s.quote) body.appendChild(document.createTextNode(" \u201C" + s.quote + "\u201D"));
        if (s.note) body.appendChild(el("div", s.note, "v-note"));
        row.appendChild(body);
        src.appendChild(row);
      });
      wrap.appendChild(src);
    }

    if ((v.claims || []).length) {
      const ok = v.claims.filter((c) => c.verdict === "supported").length;
      const det = document.createElement("details");
      det.className = "tool-step";
      det.appendChild(el("summary", "Claim check: " + ok + " of " + v.claims.length + " supported" +
        (v.faithfulness != null ? " (faithfulness " + Math.round(v.faithfulness * 100) + "%)" : "")));
      v.claims.forEach((c) => {
        const row = el("div", "", "v-claim");
        const label = c.verdict === "supported" ? "Supported: "
          : c.verdict === "contradicted" ? "Contradicted: " : "Not supported: ";
        row.appendChild(el("div", label + c.claim, c.verdict === "supported" ? "ok" : "bad"));
        if (c.evidence_quote) row.appendChild(el("div", "\u201C" + c.evidence_quote + "\u201D", "muted small"));
        if (c.note) row.appendChild(el("div", c.note, "muted small"));
        det.appendChild(row);
      });
      wrap.appendChild(det);
    }
    return wrap;
  }

  function stepEl(s) {
    const d = document.createElement("details");
    d.className = "tool-step";
    const args = Object.entries(s.args || {})
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => k + "=" + String(v).slice(0, 60))
      .join(", ");
    d.appendChild(el("summary", s.tool + "(" + args + ")"));
    d.appendChild(el("pre", s.result || "(no result)", "tool-result"));
    return d;
  }

  async function submitChat() {
    const input = $("chatInput");
    if (!input || input.disabled) return;
    const text = input.value;
    input.value = "";
    autoGrow(input);
    await sendQuestion(text);
  }

  async function sendQuestion(text) {
    const ws = currentWs();
    const q = (text || "").trim();
    if (!ws || !q || ws.ready_documents < 1) return;
    const wsId = ws.workspace_id;
    const chat = chatFor(wsId);
    if (chat.busy) return;

    const history = chat.messages.filter((m) => !m.error)
      .slice(-8).map((m) => ({ role: m.role, content: m.text }));
    chat.messages.push({ role: "user", text: q });
    chat.busy = true;
    if (state.wsId === wsId && state.tab === "ask") renderChat();

    try {
      const data = await api("/workspaces/" + wsId + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history }),
      });
      chat.messages.push({ role: "assistant", text: data.answer, steps: data.steps || [],
                           verification: data.verification || null });
    } catch (e) {
      chat.messages.push({ role: "assistant", text: e.message, error: true });
    } finally {
      chat.busy = false;
    }

    if (state.wsId === wsId && state.tab === "ask") {
      renderChat();
      const input = $("chatInput");
      if (input && !input.disabled) input.focus();
    }
  }

  // ---------- summary ----------
  async function loadSummaries() {
    if (!state.wsId) { renderSummary(); return; }
    const id = state.wsId;
    const list = await api("/workspaces/" + id + "/summaries");
    if (id !== state.wsId) return;
    state.summaries = list;
    if (!list.some((s) => s.doc_id === state.summaryDoc)) {
      const first = list.find((s) => s.summary) || list[0];
      state.summaryDoc = first ? first.doc_id : null;
    }
    renderSummary();
  }

  function summaryState(s) {
    if (s.summary) return { label: "ready", cls: "ok" };
    if (s.status === "pending" || s.status === "processing") return { label: "creating...", cls: "warn" };
    if (s.summary_status === "failed") return { label: "failed", cls: "bad" };
    return { label: "not available", cls: "muted" };
  }

  function renderSummary() {
    const list = $("sumList"), detail = $("sumDetail");
    list.replaceChildren();
    detail.replaceChildren();
    if (!state.wsId) { needWorkspace(detail); return; }
    if (!state.summaries.length) {
      detail.appendChild(el("p", "No documents yet. Upload one in the Workspace section.", "muted"));
      return;
    }

    state.summaries.forEach((s) => {
      const st = summaryState(s);
      const item = el("div", "", "list-item" + (s.doc_id === state.summaryDoc ? " selected" : ""));
      item.append(el("div", s.filename, "li-title"), el("div", st.label, "small " + st.cls));
      item.onclick = () => { state.summaryDoc = s.doc_id; renderSummary(); };
      list.appendChild(item);
    });

    const cur = state.summaries.find((s) => s.doc_id === state.summaryDoc);
    if (!cur) return;
    detail.appendChild(el("h3", cur.filename));
    if (cur.summary) {
      detail.appendChild(el("div", cur.summary, "prose"));
      if (cur.model) detail.appendChild(el("p", "Written by " + cur.model, "muted small"));
      return;
    }
    const st = summaryState(cur);
    let msg;
    if (st.label === "creating...") msg = "The summary is still being created. This page updates when processing finishes.";
    else if (cur.summary_status === "failed")
      msg = "The summary step failed" + (cur.summary_error ? ": " + cur.summary_error : ".") +
            " Use Retry failed steps in the Workspace section.";
    else msg = "No summary for this document. It was uploaded before summaries existed. Upload it again and choose Replace it.";
    detail.appendChild(el("p", msg, "muted"));
  }

  // ---------- timelines ----------
  async function loadTimeline() {
    if (!state.wsId) { renderTimeline(); return; }
    const id = state.wsId;
    const events = await api("/workspaces/" + id + "/events");
    if (id !== state.wsId) return;
    state.events = events;
    renderTimeline();
  }

  function fmtDate(iso, precision) {
    const [y, m, d] = iso.split("-").map(Number);
    if (precision === "year") return String(y);
    if (precision === "month") return MONTHS[m - 1] + " " + y;
    return d + " " + MONTHS[m - 1] + " " + y;
  }

  function eventLabel(e) {
    if (!e.date_start) return e.date_text || "No date";
    const startPrecision = e.date_precision === "range" ? "day" : e.date_precision;
    let s = fmtDate(e.date_start, startPrecision);
    if (e.date_end) s += " - " + fmtDate(e.date_end, "day");
    else if (e.is_ongoing) s += " - present";
    return s;
  }

  function docBadge(docId, filename) {
    const i = state.docs.findIndex((d) => d.doc_id === docId);
    return el("span", filename, "badge d" + ((i < 0 ? 0 : i) % 6));
  }

  function timelineItem(e, dated) {
    const item = el("div", "", "tl-item");
    if (dated) item.appendChild(el("span", "", "tl-dot"));
    const head = el("div", "", "tl-head");
    head.append(el("span", eventLabel(e), "tl-date"), el("span", e.title, "tl-title"),
                docBadge(e.doc_id, e.filename));
    item.appendChild(head);
    if (e.description) item.appendChild(el("p", e.description, "tl-desc"));
    item.appendChild(el("p", "\u201C" + e.quote + "\u201D", "tl-quote"));
    if (dated && e.date_precision === "range" && e.date_text)
      item.appendChild(el("p", "Written as: " + e.date_text, "tl-quote"));
    return item;
  }

  function renderTimeline() {
    const filters = $("tlFilters"), body = $("tlBody");
    filters.replaceChildren();
    body.replaceChildren();
    if (!state.wsId) { needWorkspace(body); return; }
    if (!state.docs.length) {
      body.appendChild(el("p", "No documents yet. Upload one in the Workspace section.", "muted"));
      return;
    }
    if (state.timelineDoc !== "all" && !state.docs.some((d) => d.doc_id === state.timelineDoc))
      state.timelineDoc = "all";

    const chip = (label, value) => {
      const b = el("button", label, "chip" + (state.timelineDoc === value ? " active" : ""));
      b.onclick = () => { state.timelineDoc = value; renderTimeline(); };
      return b;
    };
    filters.appendChild(chip("All Documents", "all"));
    state.docs.forEach((d) => filters.appendChild(chip(d.filename, d.doc_id)));

    const events = state.timelineDoc === "all"
      ? state.events
      : state.events.filter((e) => e.doc_id === state.timelineDoc);
    if (!events.length) {
      body.appendChild(el("p", state.timelineDoc === "all"
        ? "No events found yet. Events appear once a document has finished processing, and only if it mentions dates."
        : "No events found in this document.", "muted"));
      return;
    }

    body.appendChild(el("p", events.length + " event(s), oldest first", "muted small"));
    const dated = events.filter((e) => e.date_start);
    const undated = events.filter((e) => !e.date_start);

    let year = null, line = null;
    dated.forEach((e) => {
      const y = e.date_start.slice(0, 4);
      if (y !== year) {
        year = y;
        body.appendChild(el("h4", y, "year"));
        line = el("div", "", "timeline");
        body.appendChild(line);
      }
      line.appendChild(timelineItem(e, true));
    });

    if (undated.length) {
      body.appendChild(el("h4", "No exact date", "year"));
      const box = el("div", "", "undated");
      undated.forEach((e) => box.appendChild(timelineItem(e, false)));
      body.appendChild(box);
    }
  }

  // ---------- connections (knowledge graph) ----------
  async function loadConnections() {
    if (!state.wsId) { renderConnections(null); return; }
    const id = state.wsId, q = state.graphQ;
    const g = await api("/workspaces/" + id + "/graph" + (q ? "?q=" + encodeURIComponent(q) : ""));
    if (id !== state.wsId) return;
    renderConnections(g);
  }

  function renderConnections(g) {
    const body = $("graphBody");
    body.replaceChildren();
    if (!state.wsId) { needWorkspace(body); return; }
    if (!g || !g.relations.length) {
      body.appendChild(el("p", state.graphQ
        ? "No connections match that name."
        : "No connections found yet. They appear once a document has finished processing.", "muted"));
      return;
    }
    body.appendChild(el("p", g.entities.length + " entities, " + g.relations.length + " connections" +
                             (state.graphQ ? " matching \u201C" + state.graphQ + "\u201D" : ""), "muted small"));
    const wrap = el("div", "", "table-wrap");
    wrap.appendChild(buildTable(
      ["From", "Relation", "To", "File", "Source quote"],
      g.relations.map((r) => [r.source + " (" + r.source_type + ")", r.relation,
                              r.target + " (" + r.target_type + ")", r.filename, r.quote])
    ));
    body.appendChild(wrap);
  }

  // ---------- wiring ----------
  document.querySelectorAll("[data-icon]").forEach((s) => s.appendChild(icon(s.dataset.icon, 18)));

  $("wsSelect").onchange = guard(() => selectWorkspace($("wsSelect").value || null));
  $("wsCreate").onclick = guard(createWorkspace);
  $("wsName").addEventListener("keydown", (e) => { if (e.key === "Enter") guard(createWorkspace)(); });

  // Choosing files uploads them straight away.
  $("file").addEventListener("change", () => {
    const files = Array.from($("file").files);
    const target = state.uploadTarget;
    $("file").value = "";            // lets you choose the same file again later
    if (target && files.length) guard(handleFiles)(target, files);
  });

  const runGraphSearch = guard(() => { state.graphQ = $("graphQ").value.trim(); return loadConnections(); });
  $("graphSearch").onclick = runGraphSearch;
  $("graphQ").addEventListener("keydown", (e) => { if (e.key === "Enter") runGraphSearch(); });
  document.querySelectorAll(".nav-item").forEach((b) => { b.onclick = guard(() => showTab(b.dataset.tab)); });

  (async () => {
    applyTab(location.hash.slice(1));
    await loadWorkspaces();
    if (!state.wsId) { renderChrome(); await loadActiveTab(); }
  })().catch(showError);
})();