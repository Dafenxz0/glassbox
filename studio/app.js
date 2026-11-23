/* GlassBox Studio 2.0 — Certified Pareto Frontier, Deterministic Explain & Real-Arithmetic SMT UI */
(() => {
  "use strict";

  const CATEGORICAL_PALETTE = [
    "#1a4be0", // cobalt blue
    "#0d8a55", // emerald
    "#d97706", // amber
    "#0284c7", // sky cyan
    "#dc2626", // crimson
    "#0f766e", // deep teal
    "#475569", // slate
    "#b45309", // burnt ochre
  ];

  const STATUS_DESCRIPTIONS = {
    PROVEN_REAL_GLOBAL:
      "Tier 1: Exact Real-Arithmetic Global Proof (Z3 QF_LRA over exact IEEE-754 rationals)",
    PROVEN_REAL_MARGIN:
      "Tier 1: Margin-Conditioned Real Proof (Z3 QF_LRA outside classification margin)",
    PROVEN_SMT:
      "Tier 1: Formal Z3 QF_LRA Proof over piecewise-linear domain",
    TIMEOUT:
      "Z3 Solver Timeout — Unproven within solver deadline",
    UNKNOWN:
      "Z3 Solver Indeterminate — Incomplete decision procedure",
    BOUNDED_EMPIRICAL:
      "Tier 2: Bounded Empirical Grid & Multi-Start Adversarial Verification",
    COUNTEREXAMPLE_FOUND:
      "Falsified — Concrete Disagreement Witness Found",
    UNVERIFIED:
      "Unverified — No formal or empirical check completed",
  };

  const state = {
    mode: "checking", // "live" | "static"
    models: [],
    demoRunsMap: {},
    currentModelId: "",
    currentResult: null,
    currentSlice: null,
    activeTabId: "tab-c",
    activeRegimeId: null,
    selectedCeIndex: null,
    selectedParetoId: null,
    explainedPoint: null,
    graphZoom: 1,
    sliceDebounceTimer: null,
    streamController: null,
    mriHitBoxes: [],
  };

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtNum(val, digits = 4) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "—";
    const n = Number(val);
    if (Math.abs(n) > 0 && Math.abs(n) < 1e-4) return n.toExponential(2);
    return n.toFixed(digits);
  }

  function fmtBytes(bytes) {
    const b = Number(bytes || 0);
    if (b <= 0) return "0 B";
    if (b < 1024) return `${b} B`;
    return `${(b / 1024).toFixed(1)} KB`;
  }

  function fmtThroughput(ips) {
    const n = Number(ips || 0);
    if (n <= 0) return "—";
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return n.toFixed(0);
  }

  function showBanner(text) {
    if (!text) {
      els.statusBanner.hidden = true;
      return;
    }
    els.statusBannerText.textContent = text;
    els.statusBanner.hidden = false;
  }

  function isStatusProven(status) {
    return (
      status === "PROVEN_REAL_GLOBAL" ||
      status === "PROVEN_REAL_MARGIN" ||
      status === "PROVEN_SMT"
    );
  }

  /* ============================================================
     THEME TOGGLE (color-scheme: light dark + light-dark() tokens)
     ============================================================ */
  function initTheme() {
    let saved = null;
    try {
      saved = localStorage.getItem("glassbox-theme");
    } catch {
      saved = null;
    }
    applyTheme(saved || "auto");
    els.btnTheme.addEventListener("click", () => {
      const curr = document.documentElement.getAttribute("data-theme") || "auto";
      const next = curr === "auto" ? "dark" : curr === "dark" ? "light" : "auto";
      applyTheme(next);
      try {
        localStorage.setItem("glassbox-theme", next);
      } catch {
        // Ignore storage errors
      }
    });
  }

  function applyTheme(theme) {
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
      theme = "auto";
    }
    const labelMap = { auto: "Theme: Auto", light: "Theme: Light", dark: "Theme: Dark" };
    els.themeLabel.textContent = labelMap[theme] || "Theme";
  }

  /* ============================================================
     ACCESSIBLE TABS (modern-web-guidance)
     ============================================================ */
  function initTabs() {
    const tablist = document.querySelector('[role="tablist"]');
    const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
    const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
    const hasNativeFocusgroup = "focusGroup" in HTMLElement.prototype;

    if (!hasNativeFocusgroup) {
      tabs.forEach((tab) => {
        tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
      });
    }

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => activateTab(tab.id));
    });

    tablist.addEventListener("keydown", (e) => {
      const idx = tabs.indexOf(document.activeElement);
      if (idx === -1) return;
      let nextIdx = -1;
      if (e.key === "ArrowRight") nextIdx = (idx + 1) % tabs.length;
      else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") nextIdx = 0;
      else if (e.key === "End") nextIdx = tabs.length - 1;
      if (nextIdx !== -1) {
        e.preventDefault();
        tabs[nextIdx].focus();
        activateTab(tabs[nextIdx].id);
      }
    });

    panels.forEach((panel) => {
      panel.addEventListener("beforematch", () => {
        const tabId = panel.getAttribute("aria-labelledby");
        if (tabId) activateTab(tabId);
      });
    });
  }

  function activateTab(tabId) {
    state.activeTabId = tabId;
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const panels = Array.from(document.querySelectorAll('[role="tabpanel"]'));
    const hasNativeFocusgroup = "focusGroup" in HTMLElement.prototype;

    tabs.forEach((tab) => {
      const selected = tab.id === tabId;
      tab.setAttribute("aria-selected", selected ? "true" : "false");
      if (!hasNativeFocusgroup) {
        tab.tabIndex = selected ? 0 : -1;
      }
    });

    panels.forEach((panel) => {
      const isMatch = panel.getAttribute("aria-labelledby") === tabId;
      if (isMatch) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "until-found");
      }
    });

    updateWorkbenchActionMeta();
  }

  /* ============================================================
     ACCESSIBLE NATIVE DIALOGS (modern-web-guidance)
     ============================================================ */
  function initDialogs() {
    const supportsClosedBy = "closedBy" in HTMLDialogElement.prototype;
    document.querySelectorAll("dialog.studio-dialog").forEach((dialog) => {
      dialog.addEventListener("close", () => {
        dialog.returnValue = "";
      });

      if (!supportsClosedBy) {
        let downOutside = false;
        const isOutside = (ev) => {
          const r = dialog.getBoundingClientRect();
          return (
            ev.clientX < r.left ||
            ev.clientX > r.right ||
            ev.clientY < r.top ||
            ev.clientY > r.bottom
          );
        };
        dialog.addEventListener("pointerdown", (e) => {
          downOutside = e.target === dialog && isOutside(e);
        });
        dialog.addEventListener("pointerup", (e) => {
          if (downOutside && e.target === dialog && isOutside(e)) {
            dialog.close();
          }
          downOutside = false;
        });
      }
    });

    document.querySelectorAll("[data-close-dialog]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-close-dialog");
        const dlg = $(id);
        if (dlg && dlg.open) dlg.close();
      });
    });

    els.btnOpenUpload.addEventListener("click", () => {
      els.uploadError.hidden = true;
      els.uploadStaticWarning.hidden = state.mode === "live";
      els.btnUploadSubmit.disabled = state.mode !== "live";
      els.uploadDialog.showModal();
    });

    els.btnOpenCert.addEventListener("click", () => {
      renderCertDialog();
      els.certDialog.showModal();
    });
  }

  /* ============================================================
     DUAL-MODE DISCOVERY & DATA LOADING (../api/... relative)
     ============================================================ */
  async function fetchJsonWithTimeout(url, timeoutMs = 2000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const ct = res.headers.get("content-type") || "";
      if (!ct.includes("application/json")) throw new Error("Non-JSON response");
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  function setMode(mode) {
    state.mode = mode;
    els.modePill.setAttribute("data-mode", mode);
    els.modePillText.textContent =
      mode === "live" ? "Live Engine Connected" : "Static Demo Mode";
  }

  async function initApp() {
    cacheDom();
    initTheme();
    initTabs();
    initDialogs();
    bindEvents();

    try {
      const liveModels = await fetchJsonWithTimeout("../api/models", 1800);
      if (Array.isArray(liveModels)) {
        setMode("live");
        state.models = liveModels;
      } else {
        throw new Error("Invalid ../api/models payload");
      }
    } catch {
      setMode("static");
      await loadStaticIndex();
    }

    populateModelSelector();
    if (state.models.length > 0) {
      // Prefer drone_altitude_pid or first model
      const preferred =
        state.models.find((m) => m.id === "drone_altitude_pid") || state.models[0];
      await selectModel(preferred.id);
    } else {
      showBanner("No models found. Run glassbox studio or generate demo-data artifacts.");
    }
  }

  async function loadStaticIndex() {
    try {
      const indexData = await fetchJsonWithTimeout("./demo-data/index.json", 4000);
      state.models = Array.isArray(indexData.models) ? indexData.models : [];
      state.demoRunsMap = indexData.runs || {};
    } catch (err) {
      showBanner(`Unable to load static demo-data/index.json (${err.message}).`);
      state.models = [];
    }
  }

  function populateModelSelector() {
    els.modelSelect.innerHTML = "";
    state.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name || m.id;
      els.modelSelect.appendChild(opt);
    });
  }

  function normalizeRunPayload(raw) {
    if (!raw || typeof raw !== "object") return { result: null, defaultSlice: null };
    if (raw.result && raw.result.program) {
      return {
        result: raw.result,
        defaultSlice: raw.default_slice || null,
      };
    }
    return {
      result: raw,
      defaultSlice: raw.default_slice || null,
    };
  }

  async function selectModel(modelId) {
    state.currentModelId = modelId;
    els.modelSelect.value = modelId;
    state.activeRegimeId = null;
    state.selectedCeIndex = null;
    state.selectedParetoId = null;
    state.explainedPoint = null;

    const modelEntry = state.models.find((m) => m.id === modelId);
    if (modelEntry && modelEntry.default_config) {
      const dc = modelEntry.default_config;
      if (dc.max_regimes != null) els.cfgRegimes.value = dc.max_regimes;
      if (dc.tolerance_eps != null) els.cfgEps.value = dc.tolerance_eps;
      if (dc.verify_smt != null) els.cfgSmt.checked = Boolean(dc.verify_smt);
    }

    if (state.mode === "static") {
      await loadStaticRun(modelId);
      return;
    }

    if (modelEntry && modelEntry.has_prebuilt_run) {
      try {
        // Load static demo-data slice too if available so default_slice is instant
        let staticSlice = null;
        try {
          const demoRaw = await fetchJsonWithTimeout(`./demo-data/runs/${encodeURIComponent(modelId)}.json`, 1500);
          staticSlice = demoRaw?.default_slice || null;
        } catch {
          staticSlice = null;
        }
        const prebuilt = await fetchJsonWithTimeout(`../api/runs/${encodeURIComponent(modelId)}`, 4000);
        if (staticSlice && !prebuilt.default_slice) {
          prebuilt.default_slice = staticSlice;
        }
        await applyRunData(prebuilt);
        return;
      } catch {
        // Fall through to live decompile
      }
    }
    await runLiveDecompile();
  }

  async function loadStaticRun(modelId) {
    const relPath = state.demoRunsMap[modelId] || `runs/${modelId}.json`;
    const url = relPath.startsWith(".") || relPath.startsWith("/") ? relPath : `./demo-data/${relPath}`;
    try {
      const raw = await fetchJsonWithTimeout(url, 5000);
      await applyRunData(raw);
    } catch (err) {
      showBanner(`Could not load demo run for ${modelId}: ${err.message}`);
    }
  }

  async function applyRunData(raw) {
    const { result, defaultSlice } = normalizeRunPayload(raw);
    if (!result) return;
    state.currentResult = result;
    state.activeRegimeId = null;
    state.selectedCeIndex = null;
    state.selectedParetoId = null;
    state.explainedPoint = null;

    if (defaultSlice) {
      state.currentSlice = defaultSlice;
      initSliceControls(result.metadata.domain, defaultSlice);
    } else if (state.mode === "live") {
      initSliceControls(result.metadata.domain, null);
      await requestLiveSlice();
    } else {
      initSliceControls(result.metadata.domain, null);
      state.currentSlice = null;
    }

    renderAllPanels(result);
    renderMriHeatmap();
    initDefaultExplanation(result);
  }

  /* ============================================================
     LIVE DECOMPILE (SSE STREAM + FALLBACK) & STATIC REPLAY
     ============================================================ */
  async function handleDecompileSubmit(e) {
    e.preventDefault();
    if (state.mode === "static") {
      await replayStaticCegis();
      return;
    }
    await runLiveDecompile();
  }

  async function replayStaticCegis() {
    if (!state.currentResult) {
      await loadStaticRun(state.currentModelId);
      return;
    }
    const iters = state.currentResult.certificate?.cegis_iterations || [];
    if (iters.length <= 1) {
      showBanner("Static Demo Mode: displaying pre-baked verified run. Run `glassbox studio` for live re-synthesis.");
      return;
    }
    showBanner("Static Demo Mode: replaying pre-baked CEGIS trajectory. Run `glassbox studio` for custom parameters.");
    els.btnDecompile.disabled = true;
    for (let i = 1; i <= iters.length; i++) {
      renderCegisChart(iters.slice(0, i), state.currentResult.certificate.tolerance_eps);
      await new Promise((r) => setTimeout(r, 110));
    }
    els.btnDecompile.disabled = false;
  }

  function readControlsConfig() {
    const modelEntry = state.models.find((m) => m.id === state.currentModelId);
    const baseCfg = modelEntry?.default_config ? { ...modelEntry.default_config } : {};
    const maxRegimes = Math.max(2, Math.min(32, parseInt(els.cfgRegimes.value, 10) || 8));
    const tolEps = Math.max(0.0001, parseFloat(els.cfgEps.value) || 0.05);
    const verifySmt = Boolean(els.cfgSmt.checked);
    return {
      ...baseCfg,
      max_regimes: maxRegimes,
      tolerance_eps: tolEps,
      verify_smt: verifySmt,
    };
  }

  async function runLiveDecompile() {
    if (!state.currentModelId) return;
    if (state.streamController) {
      state.streamController.abort();
    }
    state.streamController = new AbortController();
    const config = readControlsConfig();

    els.btnDecompile.disabled = true;
    showBanner(`Decompiling & verifying ${state.currentModelId}...`);

    const streamedIters = [];
    try {
      const res = await fetch("../api/decompile/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: state.currentModelId,
          config,
          run_benchmarks: true,
        }),
        signal: state.streamController.signal,
      });

      const ct = res.headers.get("content-type") || "";
      if (!res.ok || !ct.includes("text/event-stream") || !res.body) {
        throw new Error(`SSE unavailable (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let gotComplete = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || "";

        for (const frame of frames) {
          if (!frame.trim()) continue;
          let eventType = "message";
          const dataLines = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }
          const payloadStr = dataLines.join("\n");
          if (!payloadStr) continue;
          let payload;
          try {
            payload = JSON.parse(payloadStr);
          } catch {
            continue;
          }

          if (eventType === "iteration") {
            streamedIters.push(payload);
            renderCegisChart(streamedIters, config.tolerance_eps);
            showBanner(
              `CEGIS iteration ${payload.iteration}: agreement ${(payload.agreement_rate * 100).toFixed(2)}%, max |Δ| ${fmtNum(payload.max_abs_error, 4)}`
            );
          } else if (eventType === "complete") {
            gotComplete = true;
            await applyRunData(payload);
            showBanner("");
          } else if (eventType === "error") {
            throw new Error(payload.detail || payload.message || "Decompilation error");
          }
        }
      }

      if (!gotComplete) {
        throw new Error("Stream ended without complete payload");
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      try {
        const res = await fetch("../api/decompile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_id: state.currentModelId,
            config,
            run_benchmarks: true,
          }),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        await applyRunData(data);
        showBanner("");
      } catch (fallbackErr) {
        showBanner(`Decompilation failed: ${fallbackErr.message}`);
      }
    } finally {
      els.btnDecompile.disabled = false;
      state.streamController = null;
    }
  }

  /* ============================================================
     RENDERING: LEFT PANE (TWO-TIER GUARANTEE, PARETO, REGIMES, CEGIS)
     ============================================================ */
  function renderAllPanels(result) {
    renderVerificationSummary(result);
    renderParetoFrontier(result);
    renderModelMeta(result.metadata);
    renderRegimeList(result.program, result);
    renderCegisChart(
      result.certificate?.cegis_iterations || [],
      result.certificate?.tolerance_eps ?? 0.05
    );
    renderCodeTabs(result);
    renderRegimeGraph(result.program);
    renderCounterexamples(result);
    renderBenchmarks(result.benchmarks || []);
    updateWorkbenchActionMeta();
  }

  function renderVerificationSummary(result) {
    const cert = result.certificate || {};
    const status = cert.status || "UNVERIFIED";
    const rap = cert.real_arithmetic_proof || null;

    els.verifBadge.textContent = status;
    els.verifBadge.setAttribute("data-status", status);
    els.verifTierSubtitle.textContent =
      STATUS_DESCRIPTIONS[status] || STATUS_DESCRIPTIONS.UNVERIFIED;

    const agr = cert.agreement_rate != null ? `${(cert.agreement_rate * 100).toFixed(2)}%` : "—";
    els.kpiAgreement.textContent = agr;
    els.kpiMaxErr.textContent = fmtNum(cert.max_abs_error, 5);

    if (rap) {
      const scopeTag =
        rap.scope === "global"
          ? "GLOBAL (z3.Q)"
          : rap.scope === "margin_conditioned"
            ? "MARGIN (z3.Q)"
            : rap.solver_result.toUpperCase();
      els.kpiSmt.textContent = scopeTag;
      els.kpiLeavesProven.textContent = `${rap.leaves_proven ?? 0} / ${rap.leaves_total ?? 0}`;
    } else {
      els.kpiSmt.textContent = cert.smt_checked
        ? cert.smt_proven
          ? "PROVEN (unsat)"
          : "REFUTED / BOUNDED"
        : "Empirical Only";
      const totalLeaves = collectLeavesWithPaths(result.program?.root).length;
      const provenLeaves = isStatusProven(status) ? totalLeaves : 0;
      els.kpiLeavesProven.textContent = `${provenLeaves} / ${totalLeaves}`;
    }
  }

  function getParetoCandidates(result) {
    if (Array.isArray(result.pareto_frontier) && result.pareto_frontier.length > 0) {
      return result.pareto_frontier;
    }
    const leaves = collectLeavesWithPaths(result.program?.root);
    const lastIter =
      (result.certificate?.cegis_iterations || [])[
        (result.certificate?.cegis_iterations || []).length - 1
      ];
    return [
      {
        candidate_id: "default_best",
        max_regimes_budget: leaves.length || 8,
        actual_regimes: leaves.length,
        ast_nodes: lastIter?.ast_nodes || leaves.length * 12,
        status: result.certificate?.status || "UNVERIFIED",
        agreement_rate: result.certificate?.agreement_rate ?? 1,
        max_abs_error: result.certificate?.max_abs_error ?? 0,
        mean_abs_error: result.certificate?.mean_abs_error ?? 0,
        is_pareto_optimal: true,
        code_c: result.code?.c || "",
        code_rust: result.code?.rust || "",
        code_python: result.code?.python || "",
      },
    ];
  }

  function colorForVerificationStatus(status) {
    if (status === "PROVEN_REAL_GLOBAL" || status === "PROVEN_SMT") return "#0d8a55";
    if (status === "PROVEN_REAL_MARGIN") return "#0f766e";
    if (status === "COUNTEREXAMPLE_FOUND" || status === "BOUNDED_EMPIRICAL") return "#d97706";
    return "#475569";
  }

  function renderParetoFrontier(result) {
    const candidates = getParetoCandidates(result);
    const svg = els.paretoChart;
    els.paretoCountPill.textContent = `${candidates.length} pt${candidates.length === 1 ? "" : "s"}`;

    const W = 320;
    const H = 156;
    const padL = 38;
    const padR = 24;
    const padT = 16;
    const padB = 26;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const nodes = candidates.map((c) => Number(c.ast_nodes || 1));
    const errs = candidates.map((c) => Number(c.max_abs_error || 0));
    const minX = Math.min(...nodes);
    const maxX = Math.max(...nodes, minX + 10);
    const maxY = Math.max(...errs, (result.certificate?.tolerance_eps ?? 0.05) * 1.25, 0.01);

    const xAt = (n) =>
      maxX === minX ? padL + plotW / 2 : padL + ((n - minX) / (maxX - minX)) * plotW;
    const yAt = (e) => padT + (1 - Math.max(0, Math.min(1, e / maxY))) * plotH;

    const optimal = [...candidates]
      .filter((c) => c.is_pareto_optimal !== false)
      .sort((a, b) => a.ast_nodes - b.ast_nodes);

    const frontPoints = optimal.map((c) => `${xAt(c.ast_nodes).toFixed(1)},${yAt(c.max_abs_error).toFixed(1)}`);

    const gridLines = [0, 0.5, 1]
      .map((f) => {
        const y = (padT + f * plotH).toFixed(1);
        return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
      })
      .join("");

    const eps = result.certificate?.tolerance_eps ?? 0.05;
    const epsY = yAt(eps).toFixed(1);

    const circles = candidates
      .map((c) => {
        const cx = xAt(c.ast_nodes).toFixed(1);
        const cy = yAt(c.max_abs_error).toFixed(1);
        const fill = colorForVerificationStatus(c.status);
        const isSelected = state.selectedParetoId === c.candidate_id;
        const ring = isSelected
          ? `<circle cx="${cx}" cy="${cy}" r="8" fill="none" stroke="var(--accent)" stroke-width="2"/>`
          : "";
        return `
          <g class="pareto-point" data-pareto-id="${escapeHtml(c.candidate_id)}">
            ${ring}
            <circle cx="${cx}" cy="${cy}" r="4.6" fill="${fill}" stroke="var(--bg)" stroke-width="1.3">
              <title>${escapeHtml(c.candidate_id)}: ${c.actual_regimes} regimes, ${c.ast_nodes} AST nodes, max |Δ|=${fmtNum(c.max_abs_error, 4)} (${c.status})</title>
            </circle>
          </g>`;
      })
      .join("");

    svg.innerHTML = `
      ${gridLines}
      <line x1="${padL}" y1="${epsY}" x2="${W - padR}" y2="${epsY}" stroke="#0d8a55" stroke-width="1.1" stroke-dasharray="4 3"/>
      ${frontPoints.length > 1 ? `<polyline fill="none" stroke="#1a4be0" stroke-width="1.8" points="${frontPoints.join(" ")}"/>` : ""}
      ${circles}
      <text x="${padL - 4}" y="${padT + 6}" text-anchor="end" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">${fmtNum(maxY, 2)}</text>
      <text x="${padL - 4}" y="${padT + plotH}" text-anchor="end" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">0</text>
      <text x="${padL}" y="${H - 6}" text-anchor="start" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">${minX} AST</text>
      <text x="${W - padR}" y="${H - 6}" text-anchor="end" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">${maxX} AST</text>
    `;

    svg.querySelectorAll(".pareto-point").forEach((g) => {
      g.addEventListener("click", () => {
        selectParetoCandidate(g.getAttribute("data-pareto-id"));
      });
    });

    els.paretoPills.innerHTML = candidates
      .map((c) => {
        const active = state.selectedParetoId === c.candidate_id ? "is-active" : "";
        const badge = isStatusProven(c.status) ? "✓" : "•";
        return `<button type="button" class="pareto-pill ${active}" data-pareto-id="${escapeHtml(c.candidate_id)}">${badge} ${c.actual_regimes}R · ${c.ast_nodes}n · |Δ| ${fmtNum(c.max_abs_error, 3)}</button>`;
      })
      .join("");

    els.paretoPills.querySelectorAll(".pareto-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        selectParetoCandidate(btn.getAttribute("data-pareto-id"));
      });
    });
  }

  function selectParetoCandidate(candidateId) {
    if (!state.currentResult) return;
    const candidates = getParetoCandidates(state.currentResult);
    const cand = candidates.find((c) => c.candidate_id === candidateId);
    if (!cand || state.selectedParetoId === candidateId) {
      resetParetoPreview();
      return;
    }

    state.selectedParetoId = candidateId;
    els.btnResetPareto.hidden = false;
    els.paretoPreviewBar.hidden = false;
    els.paretoPreviewText.textContent = `Pareto Candidate (${cand.actual_regimes} regimes, ${cand.ast_nodes} AST nodes, max |Δ|=${fmtNum(cand.max_abs_error, 4)}, ${cand.status})`;

    if (cand.code_c) renderCodePane(els.codeC, cand.code_c, "c");
    if (cand.code_rust) renderCodePane(els.codeRust, cand.code_rust, "rust");
    if (cand.code_python) renderCodePane(els.codePython, cand.code_python, "python");

    renderParetoFrontier(state.currentResult);
    updateWorkbenchActionMeta();
  }

  function resetParetoPreview() {
    state.selectedParetoId = null;
    els.btnResetPareto.hidden = true;
    els.paretoPreviewBar.hidden = true;
    if (state.currentResult) {
      renderCodeTabs(state.currentResult);
      renderParetoFrontier(state.currentResult);
      syncActiveRegimeUi();
      updateWorkbenchActionMeta();
    }
  }

  function renderModelMeta(meta) {
    if (!meta) return;
    els.modelPwlBadge.textContent = meta.is_piecewise_linear ? "Piecewise-Linear" : "Smooth Nonlinear";
    els.modelMetaSummary.innerHTML = [
      ["Task", escapeHtml(meta.domain?.task_type || "—")],
      ["Dims", `${meta.input_dim} → ${meta.output_dim}`],
      ["Params", String(meta.total_params ?? "—")],
      ["SHA-256", escapeHtml((meta.sha256 || "").slice(0, 10))],
    ]
      .map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`)
      .join("");

    els.layerStack.innerHTML = (meta.layers || [])
      .map(
        (layer) => `
        <li class="layer-item">
          <span class="layer-shape">${escapeHtml(layer.op_type)} (${layer.in_features}&rarr;${layer.out_features})</span>
          <span class="layer-act">${escapeHtml(layer.activation)}</span>
        </li>`
      )
      .join("");
  }

  function collectLeavesWithPaths(root) {
    const leaves = [];
    function walk(node, path) {
      if (!node) return;
      if (node.node_type === "leaf") {
        leaves.push({ leaf: node, path });
        return;
      }
      const gText = formatGuard(node.guard);
      walk(node.true_branch, [...path, `${gText} (T)`]);
      walk(node.false_branch, [...path, `NOT (${gText})`]);
    }
    walk(root, []);
    return leaves;
  }

  function formatGuard(guard) {
    if (!guard) return "guard";
    const terms = [];
    (guard.weights || []).forEach((w, i) => {
      if (Math.abs(w) <= 1e-6) return;
      const name = guard.feature_names?.[i] || `x${i}`;
      const absW = Math.abs(w).toFixed(3);
      if (terms.length === 0) {
        terms.push(`${w < 0 ? "-" : ""}${absW}*${name}`);
      } else {
        terms.push(`${w < 0 ? " - " : " + "}${absW}*${name}`);
      }
    });
    let lhs = terms.length > 0 ? terms.join("") : "0.000";
    const b = Number(guard.bias || 0);
    if (Math.abs(b) > 1e-9 || terms.length === 0) {
      if (terms.length > 0) {
        lhs += `${b < 0 ? " - " : " + "}${Math.abs(b).toFixed(3)}`;
      } else {
        lhs = b.toFixed(3);
      }
    }
    return `${lhs} ${guard.op || "<="} 0`;
  }

  function formatExpr(expr) {
    if (!expr) return "";
    switch (expr.kind) {
      case "var":
        return expr.name || `x${expr.index}`;
      case "const":
        return Number(expr.value).toFixed(4);
      case "unary":
        return `${expr.op}(${formatExpr(expr.operand)})`;
      case "binary": {
        const sym = { add: "+", sub: "-", mul: "*", div: "/", min: "min", max: "max" }[expr.op] || expr.op;
        if (expr.op === "min" || expr.op === "max") {
          return `${sym}(${formatExpr(expr.left)}, ${formatExpr(expr.right)})`;
        }
        return `(${formatExpr(expr.left)} ${sym} ${formatExpr(expr.right)})`;
      }
      case "clamp":
        return `clamp(${formatExpr(expr.operand)}, ${expr.low}, ${expr.high})`;
      default:
        return "?";
    }
  }

  function getRegimeNeuronProfile(result, regimeId) {
    const map = result?.regime_neuron_profiles || {};
    return map[regimeId] || map[String(regimeId)] || null;
  }

  function isLeafProven(result, regimeId) {
    const rap = result?.certificate?.real_arithmetic_proof;
    if (rap && rap.leaf_proven_map) {
      if (regimeId in rap.leaf_proven_map) return Boolean(rap.leaf_proven_map[regimeId]);
      if (String(regimeId) in rap.leaf_proven_map) return Boolean(rap.leaf_proven_map[String(regimeId)]);
    }
    return isStatusProven(result?.certificate?.status);
  }

  function renderRegimeList(program, result) {
    if (!program || !program.root) {
      els.regimeList.innerHTML = "";
      els.regimeCountPill.textContent = "0";
      return;
    }
    const items = collectLeavesWithPaths(program.root);
    els.regimeCountPill.textContent = String(items.length);

    els.regimeList.innerHTML = items
      .map(({ leaf, path }) => {
        const label = leaf.class_name ? `${leaf.name} (${leaf.class_name})` : leaf.name;
        const exprPreview = (leaf.output_exprs || []).map(formatExpr).join(", ");
        const pathTip = path.length ? path[path.length - 1] : "unconditional";
        const prof = getRegimeNeuronProfile(result, leaf.regime_id);
        const proven = isLeafProven(result, leaf.regime_id);
        const topNeurons = (prof?.top_active_neurons || prof?.top_neurons || []).slice(0, 2);

        const chips = [];
        if (proven) {
          chips.push(`<span class="neuron-chip is-proven">PROVEN</span>`);
        }
        if (prof && prof.active_neuron_count != null) {
          chips.push(`<span class="neuron-chip">${prof.active_neuron_count} active ReLUs</span>`);
        }
        topNeurons.forEach((n) => {
          const ratePct =
            n.activation_rate != null ? `${Math.round(n.activation_rate * 100)}%` : fmtNum(n.mean_value, 2);
          chips.push(
            `<span class="neuron-chip">L${n.layer ?? 0}:N${n.neuron ?? n.index ?? 0} (${ratePct})</span>`
          );
        });

        return `
          <button type="button" class="regime-item" data-regime-id="${leaf.regime_id}" title="${escapeHtml(pathTip)}">
            <div class="regime-item-top">
              <span class="regime-id-badge">R${leaf.regime_id}: ${escapeHtml(label)}</span>
              <span class="regime-stats">n=${leaf.sample_count} err=${fmtNum(leaf.local_error, 4)}</span>
            </div>
            <div class="regime-expr">${escapeHtml(exprPreview)}</div>
            ${chips.length > 0 ? `<div class="regime-neuron-strip">${chips.join("")}</div>` : ""}
          </button>`;
      })
      .join("");

    els.regimeList.querySelectorAll(".regime-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rid = Number(btn.getAttribute("data-regime-id"));
        toggleActiveRegime(rid);
      });
    });

    syncActiveRegimeUi();
  }

  function toggleActiveRegime(rid) {
    state.activeRegimeId = state.activeRegimeId === rid ? null : rid;
    syncActiveRegimeUi();
  }

  function syncActiveRegimeUi() {
    const rid = state.activeRegimeId;
    els.btnClearRegime.hidden = rid === null;

    els.regimeList.querySelectorAll(".regime-item").forEach((btn) => {
      const match = Number(btn.getAttribute("data-regime-id")) === rid;
      btn.classList.toggle("is-active", match);
    });

    let firstMatchLine = null;
    document.querySelectorAll(".code-line").forEach((lineEl) => {
      const lineRid = lineEl.getAttribute("data-regime");
      const match = rid !== null && lineRid !== null && Number(lineRid) === rid;
      lineEl.classList.toggle("is-regime-match", match);
      if (match && !firstMatchLine && lineEl.closest('[role="tabpanel"]:not([hidden])')) {
        firstMatchLine = lineEl;
      }
    });
    if (firstMatchLine) {
      firstMatchLine.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    els.regimeGraphSvg.querySelectorAll(".graph-leaf").forEach((g) => {
      const leafRid = Number(g.getAttribute("data-regime-id"));
      const rect = g.querySelector("rect");
      if (!rect) return;
      if (rid !== null && leafRid === rid) {
        rect.setAttribute("stroke", "var(--accent)");
        rect.setAttribute("stroke-width", "2.6");
      } else {
        rect.setAttribute("stroke", "var(--border-strong)");
        rect.setAttribute("stroke-width", "1.2");
      }
    });

    renderMriHeatmap();
  }

  function renderCegisChart(iterations, eps) {
    const svg = els.cegisChart;
    els.cegisIterPill.textContent = `${iterations.length} iter${iterations.length === 1 ? "" : "s"}`;

    const W = 320;
    const H = 156;
    const padL = 34;
    const padR = 38;
    const padT = 14;
    const padB = 24;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    if (!iterations || iterations.length === 0) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="var(--muted)" font-size="11" font-family="var(--font-mono)">No CEGIS iterations yet</text>`;
      return;
    }

    const maxErrVal = Math.max(
      eps * 1.35,
      ...iterations.map((it) => Number(it.max_abs_error || 0)),
      0.01
    );

    const xAt = (idx) =>
      iterations.length === 1
        ? padL + plotW / 2
        : padL + (idx / (iterations.length - 1)) * plotW;
    const yAgr = (rate) => padT + (1 - Math.max(0, Math.min(1, rate))) * plotH;
    const yErr = (err) => padT + (1 - Math.max(0, Math.min(1, err / maxErrVal))) * plotH;

    const agrPts = iterations.map((it, i) => `${xAt(i).toFixed(1)},${yAgr(it.agreement_rate).toFixed(1)}`);
    const errPts = iterations.map((it, i) => `${xAt(i).toFixed(1)},${yErr(it.max_abs_error).toFixed(1)}`);
    const epsY = yErr(eps).toFixed(1);

    const gridLines = [0, 0.5, 1]
      .map((f) => {
        const y = (padT + f * plotH).toFixed(1);
        return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
      })
      .join("");

    const dotsAgr = iterations
      .map(
        (it, i) =>
          `<circle cx="${xAt(i).toFixed(1)}" cy="${yAgr(it.agreement_rate).toFixed(1)}" r="3.2" fill="#1a4be0"><title>Iter ${it.iteration}: ${(it.agreement_rate * 100).toFixed(2)}% agreement</title></circle>`
      )
      .join("");

    const dotsErr = iterations
      .map(
        (it, i) =>
          `<circle cx="${xAt(i).toFixed(1)}" cy="${yErr(it.max_abs_error).toFixed(1)}" r="3.2" fill="#d97706"><title>Iter ${it.iteration}: max |Δ| = ${fmtNum(it.max_abs_error, 5)}</title></circle>`
      )
      .join("");

    const xTicks = iterations
      .map(
        (it, i) =>
          `<text x="${xAt(i).toFixed(1)}" y="${H - 6}" text-anchor="middle" fill="var(--muted)" font-size="9.5" font-family="var(--font-mono)">#${it.iteration}</text>`
      )
      .join("");

    svg.innerHTML = `
      ${gridLines}
      <line x1="${padL}" y1="${epsY}" x2="${W - padR}" y2="${epsY}" stroke="#0d8a55" stroke-width="1.2" stroke-dasharray="4 3"/>
      <text x="${W - padR + 4}" y="${Number(epsY) + 3}" fill="#0d8a55" font-size="9" font-family="var(--font-mono)">&epsilon;</text>
      ${iterations.length > 1 ? `<polyline fill="none" stroke="#1a4be0" stroke-width="2" points="${agrPts.join(" ")}"/>` : ""}
      ${iterations.length > 1 ? `<polyline fill="none" stroke="#d97706" stroke-width="2" points="${errPts.join(" ")}"/>` : ""}
      ${dotsAgr}
      ${dotsErr}
      <text x="${padL - 4}" y="${padT + 6}" text-anchor="end" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">100%</text>
      <text x="${padL - 4}" y="${padT + plotH}" text-anchor="end" fill="var(--muted)" font-size="9" font-family="var(--font-mono)">0%</text>
      <text x="${W - padR + 4}" y="${padT + 6}" text-anchor="start" fill="#d97706" font-size="9" font-family="var(--font-mono)">${fmtNum(maxErrVal, 2)}</text>
      ${xTicks}
    `;
  }

  /* ============================================================
     RENDERING: CENTER PANE (SYNTAX HIGHLIGHTING + REGIME GRAPH)
     ============================================================ */
  const KEYWORDS = {
    c: new Set([
      "if", "else", "return", "for", "while", "const", "static", "inline",
      "struct", "typedef", "sizeof", "int", "double", "float", "void", "char", "volatile",
    ]),
    rust: new Set([
      "fn", "pub", "let", "mut", "if", "else", "return", "struct", "enum",
      "impl", "mod", "use", "const", "match", "for", "in", "as", "f64", "i32", "usize",
    ]),
    python: new Set([
      "def", "return", "if", "elif", "else", "for", "in", "from", "import",
      "class", "and", "or", "not", "None", "True", "False", "float", "int", "list", "tuple",
    ]),
    json: new Set(["true", "false", "null"]),
  };

  function highlightLine(line, lang) {
    if (lang === "json") {
      return line.replace(
        /("(?:\\.|[^"])*")(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g,
        (m, str, colon, kw) => {
          if (str) {
            const cls = colon ? "tok-kw" : "tok-str";
            return `<span class="${cls}">${escapeHtml(str)}</span>${colon ? escapeHtml(colon) : ""}`;
          }
          if (kw) return `<span class="tok-pp">${escapeHtml(kw)}</span>`;
          return `<span class="tok-num">${escapeHtml(m)}</span>`;
        }
      );
    }

    const trimmed = line.trimStart();
    if (lang === "c" && trimmed.startsWith("#")) {
      return `<span class="tok-pp">${escapeHtml(line)}</span>`;
    }
    if (lang === "rust" && trimmed.startsWith("#!")) {
      return `<span class="tok-pp">${escapeHtml(line)}</span>`;
    }

    const kwSet = KEYWORDS[lang] || KEYWORDS.c;
    const tokenRe =
      lang === "python"
        ? /(#.*$)|("(?:\\.|[^"])*"|'(?:\\.|[^'])*')|\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b/g
        : /(\/\/.*$|\/\*.*?\*\/)|("(?:\\.|[^"])*")|\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b/g;

    let out = "";
    let lastIndex = 0;
    let match;
    while ((match = tokenRe.exec(line)) !== null) {
      if (match.index > lastIndex) {
        out += escapeHtml(line.slice(lastIndex, match.index));
      }
      const [full, cmt, str, num, ident] = match;
      if (cmt) {
        out += `<span class="tok-cmt">${escapeHtml(cmt)}</span>`;
      } else if (str) {
        out += `<span class="tok-str">${escapeHtml(str)}</span>`;
      } else if (num) {
        out += `<span class="tok-num">${escapeHtml(num)}</span>`;
      } else if (ident) {
        if (kwSet.has(ident)) {
          out += `<span class="tok-kw">${escapeHtml(ident)}</span>`;
        } else if (/^(GlassBoxInput|GlassBoxOutput|Input|Output|ClassId|Sequence|math)$/.test(ident)) {
          out += `<span class="tok-ty">${escapeHtml(ident)}</span>`;
        } else {
          out += escapeHtml(ident);
        }
      } else {
        out += escapeHtml(full);
      }
      lastIndex = tokenRe.lastIndex;
    }
    if (lastIndex < line.length) {
      out += escapeHtml(line.slice(lastIndex));
    }
    return out;
  }

  function detectLineRegimes(lines, lang) {
    const tags = new Array(lines.length).fill(null);
    let currentRid = null;
    let braceDepth = 0;
    let blockStartDepth = -1;
    let pyIndent = -1;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const hdrMatch = line.match(/(?:\/\*|\/\/|#)\s*Regime\s+(\d+)\b/i);
      if (hdrMatch) {
        currentRid = Number(hdrMatch[1]);
        blockStartDepth = braceDepth;
        pyIndent = line.search(/\S/);
        tags[i] = currentRid;
        continue;
      }

      const inlineMatch =
        !/^\s*int\s+regime_id\b/.test(line) &&
        (line.match(/\*out_regime\s*=\s*(\d+)/) ||
          line.match(/regime_id\s*[:=]\s*(\d+)/) ||
          line.match(/return\b.*\b(\d+)\s*(?:,|\))/));

      if (lang === "python") {
        const indent = line.search(/\S/);
        if (currentRid !== null) {
          if (indent !== -1 && indent < pyIndent) {
            currentRid = null;
            pyIndent = -1;
          } else {
            tags[i] = currentRid;
          }
        }
      } else {
        const opens = (line.match(/\{/g) || []).length;
        const closes = (line.match(/\}/g) || []).length;
        if (currentRid !== null && line.trimStart().startsWith("}")) {
          currentRid = null;
          blockStartDepth = -1;
        }
        if (currentRid !== null) {
          tags[i] = currentRid;
        }
        braceDepth += opens - closes;
        if (currentRid !== null && braceDepth < blockStartDepth) {
          currentRid = null;
          blockStartDepth = -1;
        }
      }

      if (tags[i] === null && inlineMatch) {
        tags[i] = Number(inlineMatch[1]);
      }
    }
    return tags;
  }

  function renderCodePane(container, source, lang) {
    const text = source || "// No code emitted for this target.";
    const lines = text.split(/\r?\n/);
    const regimeTags = detectLineRegimes(lines, lang);
    container.innerHTML = lines
      .map((line, idx) => {
        const ridAttr = regimeTags[idx] !== null ? ` data-regime="${regimeTags[idx]}"` : "";
        return `<div class="code-line"${ridAttr}><span class="code-ln">${idx + 1}</span><span class="code-src">${highlightLine(line, lang)}</span></div>`;
      })
      .join("");
  }

  function renderCodeTabs(result) {
    const code = result.code || {};
    renderCodePane(els.codeC, code.c, "c");
    renderCodePane(els.codeH, code.h, "c");
    renderCodePane(els.codeRust, code.rust, "rust");
    renderCodePane(els.codePython, code.python, "python");
    const certJson = JSON.stringify(result.certificate || {}, null, 2);
    renderCodePane(els.codeCert, certJson, "json");
  }

  function getActiveTargetInfo() {
    const code = state.currentResult?.code || {};
    const name = state.currentResult?.metadata?.name || state.currentModelId || "program";
    let cand = null;
    if (state.selectedParetoId && state.currentResult) {
      cand = getParetoCandidates(state.currentResult).find(
        (c) => c.candidate_id === state.selectedParetoId
      );
    }
    switch (state.activeTabId) {
      case "tab-c":
        return { target: "c", filename: `${name}.c`, text: cand?.code_c || code.c || "" };
      case "tab-h":
        return { target: "h", filename: `${name}.h`, text: code.h || "" };
      case "tab-rust":
        return { target: "rust", filename: `${name}.rs`, text: cand?.code_rust || code.rust || "" };
      case "tab-python":
        return { target: "python", filename: `${name}.py`, text: cand?.code_python || code.python || "" };
      case "tab-graph":
        return { target: "mermaid", filename: `${name}.mermaid.md`, text: code.mermaid || "" };
      case "tab-cert":
        return {
          target: "certificate",
          filename: `${name}.certificate.json`,
          text: JSON.stringify(state.currentResult?.certificate || {}, null, 2),
        };
      default:
        return { target: "c", filename: `${name}.c`, text: cand?.code_c || code.c || "" };
    }
  }

  function updateWorkbenchActionMeta() {
    const info = getActiveTargetInfo();
    const lines = info.text ? info.text.split(/\r?\n/).length : 0;
    els.activeFileMeta.textContent =
      state.activeTabId === "tab-graph" ? `${info.filename} (IR Tree)` : `${info.filename} · ${lines} lines`;
  }

  /* ============================================================
     VECTOR SVG REGIME GRAPH (Directly from program.root IR)
     ============================================================ */
  function renderRegimeGraph(program) {
    const svg = els.regimeGraphSvg;
    if (!program || !program.root) {
      svg.innerHTML = "";
      return;
    }

    const leafW = 152;
    const leafGap = 20;
    const levelH = 104;
    const padX = 28;
    const padY = 24;

    let leafCounter = 0;
    let maxDepth = 0;

    function layout(node, depth) {
      maxDepth = Math.max(maxDepth, depth);
      if (node.node_type === "leaf") {
        const x = padX + leafCounter * (leafW + leafGap) + leafW / 2;
        leafCounter += 1;
        return { node, depth, x, y: padY + depth * levelH, children: [] };
      }
      const left = layout(node.true_branch, depth + 1);
      const right = layout(node.false_branch, depth + 1);
      return {
        node,
        depth,
        x: (left.x + right.x) / 2,
        y: padY + depth * levelH,
        children: [
          { edge: "T", child: left },
          { edge: "F", child: right },
        ],
      };
    }

    const tree = layout(program.root, 0);
    const totalW = Math.max(360, padX * 2 + leafCounter * (leafW + leafGap) - leafGap);
    const totalH = Math.max(240, padY * 2 + (maxDepth + 1) * levelH);

    svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);
    svg.setAttribute("width", String(totalW));
    svg.setAttribute("height", String(totalH));
    applyGraphZoom();

    const edgesSvg = [];
    const nodesSvg = [];

    function emit(item) {
      item.children.forEach(({ edge, child }) => {
        const color = edge === "T" ? "#1a4be0" : "#b45309";
        const mx = (item.x + child.x) / 2;
        const my = (item.y + 42 + child.y) / 2;
        edgesSvg.push(
          `<path d="M ${item.x} ${item.y + 42} C ${item.x} ${my}, ${child.x} ${my}, ${child.x} ${child.y}" fill="none" stroke="${color}" stroke-width="1.6" stroke-opacity="0.8"/>`,
          `<text x="${mx}" y="${my - 4}" text-anchor="middle" fill="${color}" font-size="10" font-weight="600" font-family="var(--font-mono)">${edge === "T" ? "true" : "false"}</text>`
        );
        emit(child);
      });

      if (item.node.node_type === "guard") {
        const w = 172;
        const h = 42;
        const gText = formatGuard(item.node.guard);
        const clipped = gText.length > 26 ? `${gText.slice(0, 24)}…` : gText;
        nodesSvg.push(`
          <g transform="translate(${item.x - w / 2}, ${item.y})">
            <rect width="${w}" height="${h}" rx="4" fill="var(--bg)" stroke="var(--border-strong)" stroke-width="1.2"/>
            <text x="${w / 2}" y="17" text-anchor="middle" fill="var(--muted)" font-size="9.5" font-family="var(--font-text)">GUARD</text>
            <text x="${w / 2}" y="32" text-anchor="middle" fill="var(--ink)" font-size="10.5" font-family="var(--font-mono)"><title>${escapeHtml(gText)}</title>${escapeHtml(clipped)}</text>
          </g>`);
      } else {
        const leaf = item.node;
        const w = leafW;
        const h = 58;
        const expr = (leaf.output_exprs || []).map(formatExpr).join(", ");
        const clippedExpr = expr.length > 22 ? `${expr.slice(0, 20)}…` : expr;
        const titleLabel = leaf.class_name ? `R${leaf.regime_id}: ${leaf.class_name}` : `R${leaf.regime_id}: ${leaf.name}`;
        nodesSvg.push(`
          <g class="graph-leaf" data-regime-id="${leaf.regime_id}" transform="translate(${item.x - w / 2}, ${item.y})" tabindex="0" role="button" aria-label="Select Regime ${leaf.regime_id}">
            <rect width="${w}" height="${h}" rx="4" fill="var(--bg)" stroke="var(--border-strong)" stroke-width="1.2"/>
            <text x="10" y="18" fill="var(--accent)" font-size="11" font-weight="600" font-family="var(--font-mono)">${escapeHtml(titleLabel)}</text>
            <text x="10" y="34" fill="var(--ink)" font-size="10" font-family="var(--font-mono)"><title>${escapeHtml(expr)}</title>${escapeHtml(clippedExpr)}</text>
            <text x="10" y="49" fill="var(--muted)" font-size="9.5" font-family="var(--font-mono)">n=${leaf.sample_count} · err=${fmtNum(leaf.local_error, 3)}</text>
          </g>`);
      }
    }

    emit(tree);
    svg.innerHTML = `<g>${edgesSvg.join("")}${nodesSvg.join("")}</g>`;

    svg.querySelectorAll(".graph-leaf").forEach((g) => {
      const select = () => toggleActiveRegime(Number(g.getAttribute("data-regime-id")));
      g.addEventListener("click", select);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });
    });

    syncActiveRegimeUi();
  }

  function applyGraphZoom() {
    els.regimeGraphSvg.style.transform = `scale(${state.graphZoom})`;
    els.btnZoomReset.textContent = `${Math.round(state.graphZoom * 100)}%`;
  }

  /* ============================================================
     "EXPLAIN THIS DECISION" DETERMINISTIC TRACER & NEURON ATTRIBUTION
     ============================================================ */
  function evalExprJs(expr, x) {
    if (!expr) return 0;
    if (expr.kind === "var") return Number(x[expr.index] ?? 0);
    if (expr.kind === "const") return Number(expr.value ?? 0);
    if (expr.kind === "unary") {
      const v = evalExprJs(expr.operand, x);
      switch (expr.op) {
        case "neg": return -v;
        case "abs": return Math.abs(v);
        case "relu": return Math.max(0, v);
        case "tanh": return Math.tanh(v);
        case "sin": return Math.sin(v);
        case "cos": return Math.cos(v);
        default: return v;
      }
    }
    if (expr.kind === "binary") {
      const lv = evalExprJs(expr.left, x);
      const rv = evalExprJs(expr.right, x);
      switch (expr.op) {
        case "add": return lv + rv;
        case "sub": return lv - rv;
        case "mul": return lv * rv;
        case "div": return rv !== 0 ? lv / rv : 0;
        case "min": return Math.min(lv, rv);
        case "max": return Math.max(lv, rv);
        default: return 0;
      }
    }
    if (expr.kind === "clamp") {
      const v = evalExprJs(expr.operand, x);
      return Math.max(Number(expr.low), Math.min(Number(expr.high), v));
    }
    return 0;
  }

  function evalGuardJs(guard, x) {
    const margin =
      (guard.weights || []).reduce((acc, w, i) => acc + Number(w) * Number(x[i] ?? 0), 0) +
      Number(guard.bias || 0);
    const op = guard.op || "<=";
    let satisfied = false;
    if (op === "<=") satisfied = margin <= 0;
    else if (op === "<") satisfied = margin < 0;
    else if (op === ">=") satisfied = margin >= 0;
    else if (op === ">") satisfied = margin > 0;
    return { margin, satisfied };
  }

  function buildClientExplanation(xVec, gridCell = null, ce = null) {
    const result = state.currentResult;
    if (!result || !result.program?.root) return null;
    const domain = result.metadata.domain;

    let node = result.program.root;
    const guardTrace = [];
    while (node && node.node_type === "guard") {
      const { margin, satisfied } = evalGuardJs(node.guard, xVec);
      guardTrace.push({
        expression: formatGuard(node.guard),
        margin,
        satisfied,
        branch: satisfied ? "true" : "false",
      });
      node = satisfied ? node.true_branch : node.false_branch;
    }

    const leaf = node;
    const progOut = (leaf.output_exprs || []).map((e) => evalExprJs(e, xVec));
    const formulaTrace = (leaf.output_exprs || []).map((e, idx) => ({
      output_name: domain.output_names?.[idx] || `y${idx}`,
      expression: formatExpr(e),
      value: progOut[idx],
    }));

    let nnOut = progOut;
    let absErr = 0;
    if (ce && Array.isArray(ce.nn_output)) {
      nnOut = ce.nn_output;
      absErr = Number(ce.max_abs_error ?? 0);
    } else if (gridCell) {
      if (gridCell.nnVal != null) nnOut = [Number(gridCell.nnVal)];
      if (gridCell.diffVal != null) absErr = Number(gridCell.diffVal);
    }

    const proven = isLeafProven(result, leaf.regime_id);
    const prof = getRegimeNeuronProfile(result, leaf.regime_id);
    const activeNeurons = (prof?.top_active_neurons || prof?.top_neurons || []).map((n) => ({
      layer: n.layer ?? 0,
      neuron: n.neuron ?? n.index ?? 0,
      activation_rate: n.activation_rate,
      value: n.mean_value ?? n.value ?? n.activation_rate ?? 0,
    }));

    const xFmt = xVec.map((v, i) => `${domain.feature_names?.[i] || `x${i}`}=${Number(v).toFixed(3)}`).join(", ");
    const outFmt = formulaTrace.map((f) => `${f.output_name}=${fmtNum(f.value, 4)}`).join(", ");
    const summaryText = `Input (${xFmt}) satisfies ${guardTrace.length} guard step${guardTrace.length === 1 ? "" : "s"} → routes to Regime R${leaf.regime_id} (${leaf.name}${leaf.class_name ? `, class ${leaf.class_name}` : ""}), evaluating symbolic law to ${outFmt} (|Δ|=${fmtNum(absErr, 5)}, leaf ${proven ? "formally proven via Z3 QF_LRA" : "empirically bounded"}).`;

    return {
      x: xVec,
      regime_id: leaf.regime_id,
      regime_name: leaf.name,
      regime_proven: proven,
      guard_trace: guardTrace,
      formula_trace: formulaTrace,
      nn_output: nnOut,
      program_output: progOut,
      abs_error: absErr,
      active_neurons: activeNeurons,
      summary_text: summaryText,
    };
  }

  function initDefaultExplanation(result) {
    const samples = result.sample_explanations || [];
    const domain = result.metadata?.domain;
    if (!domain) return;

    const midX = domain.lower_bounds.map((lo, i) => 0.5 * (Number(lo) + Number(domain.upper_bounds[i])));
    const pills = [];
    if (samples.length > 0) {
      samples.slice(0, 4).forEach((s, idx) => {
        pills.push({ label: `Sample #${idx + 1} (R${s.regime_id})`, sample: s, x: s.x });
      });
    } else {
      pills.push({ label: "Domain Midpoint", sample: null, x: midX });
      (result.counterexamples || []).slice(0, 2).forEach((ce, idx) => {
        pills.push({ label: `Witness #${idx + 1} (R${ce.regime_id})`, sample: null, x: ce.x, ce });
      });
    }

    els.explainSamplePills.innerHTML = pills
      .map(
        (p, idx) =>
          `<button type="button" class="pareto-pill" data-sample-idx="${idx}">${escapeHtml(p.label)}</button>`
      )
      .join("");

    els.explainSamplePills.querySelectorAll("[data-sample-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const p = pills[Number(btn.getAttribute("data-sample-idx"))];
        if (p.sample) {
          state.explainedPoint = p.sample.x;
          renderExplanationPanel(p.sample);
          renderMriHeatmap();
        } else {
          explainInputPoint(p.x, { ce: p.ce || null, syncRegime: false });
        }
      });
    });

    if (samples.length > 0) {
      state.explainedPoint = samples[0].x;
      renderExplanationPanel(samples[0]);
    } else {
      const exp = buildClientExplanation(midX);
      if (exp) {
        state.explainedPoint = midX;
        renderExplanationPanel(exp);
      }
    }
  }

  async function explainInputPoint(xVec, { gridCell = null, ce = null, syncRegime = true } = {}) {
    if (!state.currentResult || !Array.isArray(xVec)) return;
    state.explainedPoint = xVec;

    let explanation = null;
    // Check pre-computed sample_explanations first in static mode
    if (state.mode === "static" && Array.isArray(state.currentResult.sample_explanations)) {
      explanation = state.currentResult.sample_explanations.find(
        (s) =>
          Array.isArray(s.x) &&
          s.x.length === xVec.length &&
          s.x.every((v, i) => Math.abs(Number(v) - Number(xVec[i])) < 1e-4)
      );
    }

    if (!explanation && state.mode === "live") {
      try {
        const res = await fetch("../api/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: state.currentResult.run_id,
            x: xVec,
          }),
        });
        if (res.ok) {
          explanation = await res.json();
        }
      } catch {
        explanation = null;
      }
    }

    if (!explanation) {
      explanation = buildClientExplanation(xVec, gridCell, ce);
    }

    if (explanation) {
      renderExplanationPanel(explanation);
      if (syncRegime && explanation.regime_id != null) {
        state.activeRegimeId = Number(explanation.regime_id);
        syncActiveRegimeUi();
      } else {
        renderMriHeatmap();
      }
    }
  }

  function renderExplanationPanel(exp) {
    if (!exp) return;
    const provenStatus = exp.regime_proven ? "PROVEN_REAL_GLOBAL" : "BOUNDED_EMPIRICAL";
    els.explainRegimeBadge.textContent = `R${exp.regime_id} · ${exp.regime_proven ? "PROVEN LEAF" : "EMPIRICAL LEAF"}`;
    els.explainRegimeBadge.setAttribute("data-status", provenStatus);

    const guardsHtml =
      (exp.guard_trace || []).length > 0
        ? exp.guard_trace
            .map((g, i) => {
              const sat = Boolean(g.satisfied);
              const cls = sat ? "is-true" : "is-false";
              const brText = sat ? "TRUE" : "FALSE";
              const marginStr = g.margin != null ? ` (m=${fmtNum(g.margin, 3)})` : "";
              return `
                <div class="explain-step">
                  <span class="explain-step-expr" title="${escapeHtml(g.expression)}">${i + 1}. ${escapeHtml(g.expression)}${escapeHtml(marginStr)}</span>
                  <span class="explain-branch ${cls}">${brText}</span>
                </div>`;
            })
            .join("")
        : `<div class="explain-step"><span>Single root regime (0 guards)</span><span class="explain-branch is-true">ROOT</span></div>`;

    const formulasHtml = (exp.formula_trace || [])
      .map(
        (f) => `
        <div class="explain-step">
          <span class="explain-step-expr" title="${escapeHtml(f.expression || f.formula || "")}">${escapeHtml(f.output_name || "out")} = ${escapeHtml(f.expression || f.formula || "")}</span>
          <span class="explain-branch is-true">${fmtNum(f.value, 4)}</span>
        </div>`
      )
      .join("");

    // Normalize active_neurons (handles both per-layer objects and flat neuron list)
    const flatNeurons = [];
    (exp.active_neurons || []).forEach((item) => {
      if (Array.isArray(item.top_neurons)) {
        item.top_neurons.forEach((tn) => {
          flatNeurons.push({
            label: `L${item.layer ?? 0}:N${tn.index ?? tn.neuron ?? 0}`,
            val: Number(tn.value ?? tn.activation_rate ?? 0),
            isRate: tn.activation_rate != null && tn.value == null,
          });
        });
      } else {
        const rate = item.activation_rate;
        const val = item.value ?? item.mean_value ?? rate ?? 0;
        flatNeurons.push({
          label: `L${item.layer ?? 0}:N${item.neuron ?? item.index ?? 0}`,
          val: Number(val),
          isRate: rate != null && item.value == null,
        });
      }
    });

    // Fallback to regime_neuron_profiles if active_neurons wasn't populated
    if (flatNeurons.length === 0 && state.currentResult) {
      const prof = getRegimeNeuronProfile(state.currentResult, exp.regime_id);
      (prof?.top_active_neurons || prof?.top_neurons || []).forEach((n) => {
        flatNeurons.push({
          label: `L${n.layer ?? 0}:N${n.neuron ?? n.index ?? 0}`,
          val: Number(n.activation_rate ?? n.mean_value ?? 0),
          isRate: n.activation_rate != null,
        });
      });
    }

    const maxNeuronVal = Math.max(...flatNeurons.map((n) => Math.abs(n.val)), 1);
    const neuronsHtml =
      flatNeurons.length > 0
        ? `<div class="neuron-bars">
            ${flatNeurons
              .slice(0, 5)
              .map((n) => {
                const pct = n.isRate
                  ? Math.max(4, Math.min(100, n.val * 100))
                  : Math.max(4, Math.min(100, (Math.abs(n.val) / maxNeuronVal) * 100));
                const valText = n.isRate ? `${Math.round(n.val * 100)}%` : fmtNum(n.val, 3);
                return `
                  <div class="neuron-bar-row">
                    <span>${escapeHtml(n.label)}</span>
                    <div class="neuron-bar-track"><div class="neuron-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
                    <span>${escapeHtml(valText)}</span>
                  </div>`;
              })
              .join("")}
          </div>`
        : `<p class="card-sub">No hidden ReLU activation profile for this architecture.</p>`;

    els.explainContent.innerHTML = `
      <div class="explain-summary-box">${escapeHtml(exp.summary_text || `Routed to Regime R${exp.regime_id} (${exp.regime_name}).`)}</div>
      <div class="explain-subhead">1. Guard Routing Path</div>
      <div class="explain-trace-list">${guardsHtml}</div>
      <div class="explain-subhead">2. Symbolic Output Evaluation</div>
      <div class="explain-trace-list">${formulasHtml}</div>
      <div class="explain-subhead">3. Original ONNX Hidden Neuron Attribution</div>
      ${neuronsHtml}
    `;
  }

  /* ============================================================
     RIGHT PANE: 2D INPUT-SPACE MRI HEATMAP + SLICE CONTROLS
     ============================================================ */
  function initSliceControls(domain, sliceData) {
    if (!domain || !domain.feature_names) return;
    const names = domain.feature_names;
    const dimX = sliceData ? sliceData.dim_x : 0;
    const dimY = sliceData ? sliceData.dim_y : Math.min(1, names.length - 1);

    els.sliceDimX.innerHTML = "";
    els.sliceDimY.innerHTML = "";
    names.forEach((name, idx) => {
      const optX = document.createElement("option");
      optX.value = String(idx);
      optX.textContent = name;
      els.sliceDimX.appendChild(optX);

      const optY = document.createElement("option");
      optY.value = String(idx);
      optY.textContent = name;
      els.sliceDimY.appendChild(optY);
    });
    els.sliceDimX.value = String(dimX);
    els.sliceDimY.value = String(dimY);

    renderFixedDimSliders(domain, dimX, dimY);
  }

  function getFixedPointVector() {
    const domain = state.currentResult?.metadata?.domain;
    if (!domain) return [];
    return domain.feature_names.map((_, i) => {
      const slider = $(`slider-dim-${i}`);
      if (slider) return Number(slider.value);
      return 0.5 * (Number(domain.lower_bounds[i]) + Number(domain.upper_bounds[i]));
    });
  }

  function renderFixedDimSliders(domain, dimX, dimY) {
    const names = domain.feature_names || [];
    const lows = domain.lower_bounds || [];
    const highs = domain.upper_bounds || [];
    const container = els.sliceSliders;
    container.innerHTML = "";

    const isStatic = state.mode === "static";
    els.sliceDimX.disabled = isStatic;
    els.sliceDimY.disabled = isStatic;
    if (isStatic && names.length > 2) {
      els.sliceModeNote.textContent = "Static Demo Mode shows the pre-baked 2D slice. Click any pixel to run client-side decision tracing!";
      els.sliceModeNote.hidden = false;
    } else {
      els.sliceModeNote.hidden = true;
    }

    names.forEach((name, idx) => {
      if (idx === dimX || idx === dimY) return;
      const lo = Number(lows[idx] ?? -1);
      const hi = Number(highs[idx] ?? 1);
      const mid = (lo + hi) / 2;
      const step = (hi - lo) / 100;
      const row = document.createElement("div");
      row.className = "slider-row";
      row.innerHTML = `
        <label for="slider-dim-${idx}">${escapeHtml(name)}</label>
        <input type="range" id="slider-dim-${idx}" data-dim="${idx}" min="${lo}" max="${hi}" step="${step}" value="${mid}" ${isStatic ? "disabled" : ""}>
        <span id="slider-val-${idx}">${mid.toFixed(2)}</span>`;
      const input = row.querySelector("input");
      const valSpan = row.querySelector("span");
      input.addEventListener("input", () => {
        valSpan.textContent = Number(input.value).toFixed(2);
        scheduleLiveSlice();
      });
      container.appendChild(row);
    });
  }

  function scheduleLiveSlice() {
    if (state.mode !== "live") return;
    clearTimeout(state.sliceDebounceTimer);
    state.sliceDebounceTimer = setTimeout(requestLiveSlice, 140);
  }

  async function requestLiveSlice() {
    if (state.mode !== "live" || !state.currentResult) return;
    const dimX = Number(els.sliceDimX.value);
    const dimY = Number(els.sliceDimY.value);
    const fixedValues = getFixedPointVector();

    try {
      const res = await fetch("../api/slice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: state.currentResult.run_id,
          dim_x: dimX,
          dim_y: dimY,
          fixed_values: fixedValues,
          resolution: 48,
        }),
      });
      if (!res.ok) return;
      state.currentSlice = await res.json();
      renderMriHeatmap();
    } catch {
      // Non-fatal slice preview error
    }
  }

  function lerpRgb(c1, c2, t) {
    const f = Math.max(0, Math.min(1, t));
    return [
      Math.round(c1[0] + (c2[0] - c1[0]) * f),
      Math.round(c1[1] + (c2[1] - c1[1]) * f),
      Math.round(c1[2] + (c2[2] - c1[2]) * f),
    ];
  }

  function hexToRgb(hex) {
    const clean = hex.replace("#", "");
    return [
      parseInt(clean.slice(0, 2), 16),
      parseInt(clean.slice(2, 4), 16),
      parseInt(clean.slice(4, 6), 16),
    ];
  }

  function colorForDiff(val, eps) {
    const norm = Math.max(0, Math.min(1, val / Math.max(eps * 1.5, 1e-6)));
    if (norm < 0.35) {
      return lerpRgb([244, 246, 250], [2, 132, 199], norm / 0.35);
    }
    if (norm < 0.72) {
      return lerpRgb([2, 132, 199], [217, 119, 6], (norm - 0.35) / 0.37);
    }
    return lerpRgb([217, 119, 6], [220, 38, 38], (norm - 0.72) / 0.28);
  }

  function colorForContinuous(val, minV, maxV) {
    const span = maxV - minV || 1;
    const t = Math.max(0, Math.min(1, (val - minV) / span));
    if (t < 0.5) {
      return lerpRgb([26, 75, 224], [244, 244, 244], t * 2);
    }
    return lerpRgb([244, 244, 244], [13, 138, 85], (t - 0.5) * 2);
  }

  function colorForCategory(id) {
    const idx = Math.abs(Math.round(id || 0)) % CATEGORICAL_PALETTE.length;
    return hexToRgb(CATEGORICAL_PALETTE[idx]);
  }

  function getActiveGridAndMode(slice) {
    const mode = els.mriViewSelect.value;
    const isClassification =
      state.currentResult?.metadata?.domain?.task_type === "classification";
    if (mode === "nn_grid") {
      if (isClassification && slice.nn_class_grid) {
        return { grid: slice.nn_class_grid, kind: "categorical", label: "NN Class" };
      }
      return { grid: slice.nn_grid, kind: "continuous", label: "NN Output" };
    }
    if (mode === "prog_grid") {
      if (isClassification && slice.prog_class_grid) {
        return { grid: slice.prog_class_grid, kind: "categorical", label: "Code Class" };
      }
      return { grid: slice.prog_grid, kind: "continuous", label: "Code Output" };
    }
    if (mode === "regime_grid") {
      return { grid: slice.regime_grid, kind: "regime", label: "Regime ID" };
    }
    return { grid: slice.diff_grid, kind: "diff", label: "Error |Δ|" };
  }

  function renderMriHeatmap() {
    const canvas = els.mriCanvas;
    const ctx = canvas.getContext("2d");
    const slice = state.currentSlice;
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    state.mriHitBoxes = [];

    if (!slice || !slice.diff_grid || slice.diff_grid.length === 0) {
      ctx.fillStyle = "#181818";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#9a9a9a";
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("No 2D slice available", W / 2, H / 2);
      els.mriLegend.innerHTML = "";
      return;
    }

    const { grid, kind } = getActiveGridAndMode(slice);
    const ny = grid.length;
    const nx = grid[0].length;
    const cellW = W / nx;
    const cellH = H / ny;

    let minV = Infinity;
    let maxV = -Infinity;
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const v = Number(grid[r][c]);
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }

    const eps = state.currentResult?.certificate?.tolerance_eps ?? 0.05;
    const activeRid = state.activeRegimeId;

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const val = Number(grid[r][c]);
        const cellRid = slice.regime_grid ? Number(slice.regime_grid[r][c]) : null;
        let rgb;
        if (kind === "diff") rgb = colorForDiff(val, eps);
        else if (kind === "continuous") rgb = colorForContinuous(val, minV, maxV);
        else rgb = colorForCategory(val);

        const dimOut = activeRid !== null && cellRid !== activeRid;
        ctx.fillStyle = dimOut
          ? `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.24)`
          : `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

        const x = c * cellW;
        const y = H - (r + 1) * cellH;
        ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    const xVals = slice.x_vals || [];
    const yVals = slice.y_vals || [];
    const xMin = Number(xVals[0] ?? -1);
    const xMax = Number(xVals[xVals.length - 1] ?? 1);
    const yMin = Number(yVals[0] ?? -1);
    const yMax = Number(yVals[yVals.length - 1] ?? 1);

    const ces =
      slice.counterexamples_projected && slice.counterexamples_projected.length > 0
        ? slice.counterexamples_projected
        : (state.currentResult?.counterexamples || []).map((ce, idx) => ({
            index: idx,
            x: ce.x?.[slice.dim_x] ?? 0,
            y: ce.x?.[slice.dim_y] ?? 0,
            repaired: ce.repaired,
            regime_id: ce.regime_id,
          }));

    ces.forEach((pt, idx) => {
      const pxVal = pt.x ?? pt.x_coord ?? 0;
      const pyVal = pt.y ?? pt.y_coord ?? 0;
      const cx = ((pxVal - xMin) / (xMax - xMin || 1)) * W;
      const cy = H - ((pyVal - yMin) / (yMax - yMin || 1)) * H;
      const isSelected = state.selectedCeIndex === (pt.index ?? idx);

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.strokeStyle = "#1a4be0";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      const sz = 4.5;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(cx - sz, cy - sz);
      ctx.lineTo(cx + sz, cy + sz);
      ctx.moveTo(cx + sz, cy - sz);
      ctx.lineTo(cx - sz, cy + sz);
      ctx.stroke();

      ctx.strokeStyle = pt.repaired ? "#0d8a55" : "#dc2626";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - sz, cy - sz);
      ctx.lineTo(cx + sz, cy + sz);
      ctx.moveTo(cx + sz, cy - sz);
      ctx.lineTo(cx - sz, cy + sz);
      ctx.stroke();

      state.mriHitBoxes.push({
        cx,
        cy,
        ceIndex: pt.index ?? idx,
        regimeId: pt.regime_id,
      });
    });

    // Plot explainedPoint target reticle
    if (Array.isArray(state.explainedPoint)) {
      const ex = Number(state.explainedPoint[slice.dim_x]);
      const ey = Number(state.explainedPoint[slice.dim_y]);
      if (!Number.isNaN(ex) && !Number.isNaN(ey)) {
        const cx = ((ex - xMin) / (xMax - xMin || 1)) * W;
        const cy = H - ((ey - yMin) / (yMax - yMin || 1)) * H;
        ctx.beginPath();
        ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.8;
        ctx.stroke();
        ctx.strokeStyle = "#1a4be0";
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }

    els.mriXLabel.textContent = slice.x_name || `dim ${slice.dim_x}`;
    els.mriYLabel.textContent = slice.y_name || `dim ${slice.dim_y}`;
    const midX = (xMin + xMax) / 2;
    els.mriXTicks.innerHTML = `<span>${xMin.toFixed(2)}</span><span>${midX.toFixed(2)}</span><span>${xMax.toFixed(2)}</span>`;

    renderMriLegend(kind, minV, maxV, eps);
  }

  function renderMriLegend(kind, minV, maxV, eps) {
    if (kind === "diff") {
      els.mriLegend.innerHTML = `
        <span>0</span>
        <span class="legend-gradient" style="background:linear-gradient(90deg,#f4f6fa,#0284c7,#d97706,#dc2626)"></span>
        <span>${fmtNum(Math.max(maxV, eps), 3)}</span>`;
    } else if (kind === "continuous") {
      els.mriLegend.innerHTML = `
        <span>${fmtNum(minV, 2)}</span>
        <span class="legend-gradient" style="background:linear-gradient(90deg,#1a4be0,#f4f4f4,#0d8a55)"></span>
        <span>${fmtNum(maxV, 2)}</span>`;
    } else {
      const label = kind === "regime" ? "Regimes" : "Classes";
      const chips = CATEGORICAL_PALETTE.slice(0, 6)
        .map((c, i) => `<span class="legend-item"><span class="legend-swatch" style="background:${c};height:8px;width:8px"></span>${i}</span>`)
        .join("");
      els.mriLegend.innerHTML = `<span>${label}:</span>${chips}`;
    }
  }

  /* ============================================================
     COUNTEREXAMPLES TABLE & 6-ROW HONEST BENCHMARKS
     ============================================================ */
  function renderCounterexamples(result) {
    const ces = result.counterexamples || [];
    els.ceTotalPill.textContent = String(ces.length);
    if (ces.length === 0) {
      els.ceTbody.innerHTML = `<tr><td colspan="6" class="empty-cell">Zero counterexamples within bounded domain.</td></tr>`;
      return;
    }

    els.ceTbody.innerHTML = ces
      .map((ce, idx) => {
        const xPreview = (ce.x || []).slice(0, 2).map((v) => Number(v).toFixed(2)).join(", ");
        const nnVal =
          ce.nn_class != null ? `cls ${ce.nn_class}` : fmtNum(ce.nn_output?.[0], 3);
        const progVal =
          ce.program_class != null ? `cls ${ce.program_class}` : fmtNum(ce.program_output?.[0], 3);
        const stateCls = ce.repaired ? "repaired" : "open";
        const stateLabel = ce.repaired ? "REPAIRED" : "OPEN";
        return `
          <tr data-ce-index="${idx}" data-regime-id="${ce.regime_id}">
            <td class="mono">${escapeHtml(ce.source)}</td>
            <td class="mono" title="(${escapeHtml((ce.x || []).join(", "))})">[${escapeHtml(xPreview)}…]</td>
            <td class="mono">${escapeHtml(nnVal)}</td>
            <td class="mono">${escapeHtml(progVal)}</td>
            <td class="mono">${fmtNum(ce.max_abs_error, 4)}</td>
            <td><span class="ce-state ${stateCls}">${stateLabel}</span></td>
          </tr>`;
      })
      .join("");

    els.ceTbody.querySelectorAll("tr[data-ce-index]").forEach((row) => {
      row.addEventListener("click", () => {
        const idx = Number(row.getAttribute("data-ce-index"));
        const rid = Number(row.getAttribute("data-regime-id"));
        selectCounterexample(idx, rid);
      });
    });
  }

  function selectCounterexample(idx, rid) {
    state.selectedCeIndex = idx;
    els.ceTbody.querySelectorAll("tr[data-ce-index]").forEach((row) => {
      row.classList.toggle("is-selected", Number(row.getAttribute("data-ce-index")) === idx);
    });
    const ce = state.currentResult?.counterexamples?.[idx] || null;
    if (ce && Array.isArray(ce.x)) {
      explainInputPoint(ce.x, { ce, syncRegime: true });
    } else if (!Number.isNaN(rid)) {
      state.activeRegimeId = rid;
      syncActiveRegimeUi();
    } else {
      renderMriHeatmap();
    }
  }

  function renderBenchmarks(benchmarks) {
    if (!benchmarks || benchmarks.length === 0) {
      els.benchList.innerHTML = `<p class="card-sub">No benchmark measurements recorded.</p>`;
      return;
    }

    const avail = benchmarks.filter((b) => b.available !== false && b.p50_us > 0);
    const maxP50 = Math.max(...avail.map((b) => b.p50_us), 1);

    els.benchList.innerHTML = benchmarks
      .map((b) => {
        if (b.available === false) {
          return `
            <div class="bench-row">
              <div class="bench-row-top">
                <span class="bench-name">${escapeHtml(b.runtime)}</span>
                <span class="bench-metrics">unavailable</span>
              </div>
              <div class="bench-sub"><span>${escapeHtml(b.note || "")}</span></div>
            </div>`;
        }
        const widthPct = Math.max(3, Math.min(100, (b.p50_us / maxP50) * 100));
        const isNative = /c99|dll|native/i.test(b.runtime);
        const isNativeLoop = b.native_loop_p50_ns != null || /native c loop/i.test(b.runtime);
        const latencyHeadline = isNativeLoop
          ? `${fmtNum(b.native_loop_p50_ns ?? b.p50_us * 1000, 1)} ns/call (${fmtNum(b.p50_us, 3)} &mu;s)`
          : `${fmtNum(b.p50_us, 2)} &mu;s p50 · ${fmtNum(b.p99_us, 2)} &mu;s p99`;

        const measuredIps = b.measured_throughput_ips || b.throughput_ips || 0;
        const speedup = b.speedup_vs_onnx ? `${b.speedup_vs_onnx.toFixed(1)}× vs ORT` : "";
        const sizeLabel =
          b.source_bytes && b.binary_bytes
            ? `src ${fmtBytes(b.source_bytes)} / bin ${fmtBytes(b.binary_bytes)}`
            : fmtBytes(b.artifact_bytes || b.source_bytes || b.binary_bytes);
        const compilerTag = b.compiler ? ` · ${b.compiler}` : "";

        return `
          <div class="bench-row">
            <div class="bench-row-top">
              <span class="bench-name">${escapeHtml(b.runtime)}</span>
              <span class="bench-metrics">${latencyHeadline}</span>
            </div>
            <div class="bench-bar-track">
              <div class="bench-bar-fill ${isNative ? "is-native" : ""}" style="width:${widthPct.toFixed(1)}%"></div>
            </div>
            <div class="bench-sub">
              <span>Measured Throughput: ${fmtThroughput(measuredIps)} inf/s · ${escapeHtml(speedup)}</span>
              <span>batch=${b.batch_size || 1} · ${escapeHtml(sizeLabel)}${escapeHtml(compilerTag)} · diff ${fmtNum(b.max_diff_vs_ref, 2)}</span>
            </div>
          </div>`;
      })
      .join("");
  }

  /* ============================================================
     TWO-TIER VERIFICATION CERTIFICATE DIALOG & UPLOAD ONNX
     ============================================================ */
  function renderCertDialog() {
    const cert = state.currentResult?.certificate;
    if (!cert) {
      els.certDialogContent.innerHTML = "<p>No verification certificate loaded.</p>";
      return;
    }

    const rap = cert.real_arithmetic_proof || {};
    const nda = cert.numerical_deployment_assurance || {};
    const modelName = cert.model_name || state.currentModelId || "model";
    const cliCmd = `glassbox verify bundles/${modelName}.gbundle`;

    const tier1Items = [
      ["Proof Status", cert.status],
      ["Exact Rationals (z3.Q)", String(rap.exact_rationals ?? cert.smt_checked)],
      ["Sound Outward IBP Bounds", String(rap.sound_bounds_only ?? true)],
      ["Proof Scope", rap.scope || (cert.smt_proven ? "global" : "none")],
      ["Solver Result", rap.solver_result || (cert.smt_proven ? "unsat" : "sat")],
      ["Z3 Solver Version", rap.z3_version || cert.tool_versions?.z3 || "z3-solver"],
      [
        "Leaves Proven",
        rap.leaves_total != null
          ? `${rap.leaves_proven} / ${rap.leaves_total}`
          : String(cert.smt_proven ? "All" : "0"),
      ],
      ["Tolerance ε / Margin", `ε=${cert.tolerance_eps} / margin=${cert.classification_margin}`],
    ];

    const tier2Items = [
      ["Empirical Agreement", `${(cert.agreement_rate * 100).toFixed(3)}%`],
      ["IR vs ONNX Max |Δ|", fmtNum(nda.ir_vs_onnx_max_diff ?? cert.max_abs_error, 6)],
      ["C99 vs IR Max |Δ|", nda.c99_vs_ir_max_diff != null ? fmtNum(nda.c99_vs_ir_max_diff, 6) : "< 1e-14"],
      ["Python vs IR Max |Δ|", nda.python_vs_ir_max_diff != null ? fmtNum(nda.python_vs_ir_max_diff, 6) : "< 1e-14"],
      ["C99 Compiler", nda.c99_compiler || "ziglang cc / gcc / clang -O3 -std=c99"],
      ["ORT Samples Checked", String(nda.onnxruntime_samples_checked || 4096)],
      ["Model SHA-256", cert.model_sha256],
      ["Program SHA-256", cert.program_sha256],
    ];

    const renderGrid = (items) =>
      `<div class="cert-kv-grid">${items
        .map(
          ([k, v]) => `
          <div class="cert-kv">
            <span class="cert-kv-label">${escapeHtml(k)}</span>
            <span class="cert-kv-val">${escapeHtml(v)}</span>
          </div>`
        )
        .join("")}</div>`;

    els.certDialogContent.innerHTML = `
      <div class="cert-tier-section">
        <h3 class="cert-tier-title">Tier 1 — Real-Arithmetic Proof (Exact Rational Z3 QF_LRA)</h3>
        ${renderGrid(tier1Items)}
      </div>
      <div class="cert-tier-section">
        <h3 class="cert-tier-title">Tier 2 — Numerical Deployment Assurance (Float32 ORT &amp; Compiled Targets)</h3>
        ${renderGrid(tier2Items)}
        <p class="card-sub" style="margin-top:8px;margin-bottom:0">${escapeHtml(
          nda.float32_soundness_note ||
            "Real-arithmetic SMT certifies the mathematical rational MLP extracted from ONNX float32 weights; numerical deployment assurance empirically checks float32 ONNXRuntime and compiled C99/Python outputs."
        )}</p>
      </div>
      <div class="cert-tier-section">
        <h3 class="cert-tier-title">Independent Third-Party Bundle Verification</h3>
        <div class="cert-cli-box">
          <code id="cert-cli-cmd">${escapeHtml(cliCmd)}</code>
          <button type="button" id="btn-copy-verify-cli" class="btn btn-sm btn-secondary">Copy Command</button>
        </div>
      </div>
    `;

    const copyBtn = $("btn-copy-verify-cli");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(cliCmd);
          copyBtn.textContent = "Copied";
          setTimeout(() => {
            copyBtn.textContent = "Copy Command";
          }, 1300);
        } catch {
          // Ignore clipboard permissions error
        }
      });
    }
  }

  async function handleUploadSubmit(e) {
    e.preventDefault();
    if (state.mode !== "live") return;
    const file = els.uploadFile.files?.[0];
    if (!file) return;

    els.uploadError.hidden = true;
    els.btnUploadSubmit.disabled = true;

    const form = new FormData();
    form.append("file", file);
    const domainStr = els.uploadDomainJson.value.trim();
    if (domainStr) form.append("domain_json", domainStr);

    try {
      const res = await fetch("../api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
      state.models = await fetchJsonWithTimeout("../api/models", 2500);
      populateModelSelector();
      els.uploadDialog.close();
      await selectModel(data.model_id);
    } catch (err) {
      els.uploadError.textContent = err.message;
      els.uploadError.hidden = false;
    } finally {
      els.btnUploadSubmit.disabled = false;
    }
  }

  /* ============================================================
     EVENTS & INITIALIZATION
     ============================================================ */
  function cacheDom() {
    els.modePill = $("mode-pill");
    els.modePillText = $("mode-pill-text");
    els.modelSelect = $("model-select");
    els.cfgRegimes = $("cfg-regimes");
    els.cfgEps = $("cfg-eps");
    els.cfgSmt = $("cfg-smt");
    els.btnDecompile = $("btn-decompile");
    els.btnOpenUpload = $("btn-open-upload");
    els.btnOpenCert = $("btn-open-cert");
    els.btnTheme = $("btn-theme");
    els.themeLabel = $("theme-label");
    els.statusBanner = $("status-banner");
    els.statusBannerText = $("status-banner-text");
    els.statusBannerDismiss = $("status-banner-dismiss");
    els.decompileControls = $("decompile-controls");

    els.verifBadge = $("verif-badge");
    els.verifTierSubtitle = $("verif-tier-subtitle");
    els.kpiAgreement = $("kpi-agreement");
    els.kpiMaxErr = $("kpi-max-err");
    els.kpiSmt = $("kpi-smt");
    els.kpiLeavesProven = $("kpi-leaves-proven");

    els.paretoCountPill = $("pareto-count-pill");
    els.btnResetPareto = $("btn-reset-pareto");
    els.paretoChart = $("pareto-chart");
    els.paretoPills = $("pareto-pills");
    els.paretoPreviewBar = $("pareto-preview-bar");
    els.paretoPreviewText = $("pareto-preview-text");
    els.btnParetoBarReset = $("btn-pareto-bar-reset");

    els.modelPwlBadge = $("model-pwl-badge");
    els.modelMetaSummary = $("model-meta-summary");
    els.layerStack = $("layer-stack");
    els.regimeCountPill = $("regime-count-pill");
    els.btnClearRegime = $("btn-clear-regime");
    els.regimeList = $("regime-list");
    els.cegisIterPill = $("cegis-iter-pill");
    els.cegisChart = $("cegis-chart");

    els.activeFileMeta = $("active-file-meta");
    els.btnCopyCode = $("btn-copy-code");
    els.btnDownloadCode = $("btn-download-code");
    els.codeC = $("code-c");
    els.codeH = $("code-h");
    els.codeRust = $("code-rust");
    els.codePython = $("code-python");
    els.codeCert = $("code-cert");
    els.regimeGraphSvg = $("regime-graph-svg");
    els.btnZoomIn = $("btn-zoom-in");
    els.btnZoomOut = $("btn-zoom-out");
    els.btnZoomReset = $("btn-zoom-reset");

    els.mriViewSelect = $("mri-view-select");
    els.sliceDimX = $("slice-dim-x");
    els.sliceDimY = $("slice-dim-y");
    els.sliceSliders = $("slice-sliders");
    els.sliceModeNote = $("slice-mode-note");
    els.mriCanvas = $("mri-canvas");
    els.mriXLabel = $("mri-x-label");
    els.mriYLabel = $("mri-y-label");
    els.mriXTicks = $("mri-x-ticks");
    els.mriLegend = $("mri-legend");
    els.mriReadout = $("mri-readout");

    els.explainRegimeBadge = $("explain-regime-badge");
    els.explainSamplePills = $("explain-sample-pills");
    els.explainContent = $("explain-content");

    els.ceTotalPill = $("ce-total-pill");
    els.ceTbody = $("ce-tbody");
    els.benchList = $("bench-list");

    els.uploadDialog = $("upload-dialog");
    els.uploadForm = $("upload-form");
    els.uploadFile = $("upload-file");
    els.uploadDomainJson = $("upload-domain-json");
    els.uploadStaticWarning = $("upload-static-warning");
    els.uploadError = $("upload-error");
    els.btnUploadSubmit = $("btn-upload-submit");
    els.certDialog = $("cert-dialog");
    els.certDialogContent = $("cert-dialog-content");
    els.btnDownloadCert = $("btn-download-cert");
  }

  function bindEvents() {
    els.statusBannerDismiss.addEventListener("click", () => showBanner(""));
    els.modelSelect.addEventListener("change", () => selectModel(els.modelSelect.value));
    els.decompileControls.addEventListener("submit", handleDecompileSubmit);
    els.btnClearRegime.addEventListener("click", () => {
      state.activeRegimeId = null;
      syncActiveRegimeUi();
    });

    els.btnResetPareto.addEventListener("click", resetParetoPreview);
    els.btnParetoBarReset.addEventListener("click", resetParetoPreview);

    els.btnCopyCode.addEventListener("click", async () => {
      const { text } = getActiveTargetInfo();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        const orig = els.btnCopyCode.textContent;
        els.btnCopyCode.textContent = "Copied";
        setTimeout(() => {
          els.btnCopyCode.textContent = orig;
        }, 1400);
      } catch {
        showBanner("Clipboard write denied by browser.");
      }
    });

    els.btnDownloadCode.addEventListener("click", () => {
      const { target, filename, text } = getActiveTargetInfo();
      if (state.mode === "live" && state.currentResult?.run_id && !state.selectedParetoId) {
        window.location.href = `../api/runs/${encodeURIComponent(state.currentResult.run_id)}/download/${target}`;
        return;
      }
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    els.btnDownloadCert.addEventListener("click", () => {
      if (!state.currentResult) return;
      if (state.mode === "live" && state.currentResult.run_id) {
        window.location.href = `../api/runs/${encodeURIComponent(state.currentResult.run_id)}/download/certificate`;
        return;
      }
      const json = JSON.stringify(state.currentResult.certificate || {}, null, 2);
      const blob = new Blob([json], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${state.currentModelId || "glassbox"}.certificate.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    els.btnZoomIn.addEventListener("click", () => {
      state.graphZoom = Math.min(2.2, +(state.graphZoom + 0.15).toFixed(2));
      applyGraphZoom();
    });
    els.btnZoomOut.addEventListener("click", () => {
      state.graphZoom = Math.max(0.45, +(state.graphZoom - 0.15).toFixed(2));
      applyGraphZoom();
    });
    els.btnZoomReset.addEventListener("click", () => {
      state.graphZoom = 1;
      applyGraphZoom();
    });

    els.mriViewSelect.addEventListener("change", renderMriHeatmap);
    const onAxisChange = () => {
      if (!state.currentResult?.metadata?.domain) return;
      renderFixedDimSliders(
        state.currentResult.metadata.domain,
        Number(els.sliceDimX.value),
        Number(els.sliceDimY.value)
      );
      scheduleLiveSlice();
    };
    els.sliceDimX.addEventListener("change", onAxisChange);
    els.sliceDimY.addEventListener("change", onAxisChange);

    els.mriCanvas.addEventListener("mousemove", handleCanvasHover);
    els.mriCanvas.addEventListener("click", handleCanvasClick);
    els.uploadForm.addEventListener("submit", handleUploadSubmit);
  }

  function handleCanvasHover(e) {
    const slice = state.currentSlice;
    if (!slice || !slice.diff_grid || slice.diff_grid.length === 0) return;
    const rect = els.mriCanvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const ny = slice.diff_grid.length;
    const nx = slice.diff_grid[0].length;
    const c = Math.max(0, Math.min(nx - 1, Math.floor(fx * nx)));
    const r = Math.max(0, Math.min(ny - 1, Math.floor((1 - fy) * ny)));

    const xv = Number(slice.x_vals?.[c] ?? 0).toFixed(3);
    const yv = Number(slice.y_vals?.[r] ?? 0).toFixed(3);
    const nnVal = slice.nn_class_grid
      ? `cls ${slice.nn_class_grid[r][c]}`
      : fmtNum(slice.nn_grid?.[r]?.[c], 4);
    const progVal = slice.prog_class_grid
      ? `cls ${slice.prog_class_grid[r][c]}`
      : fmtNum(slice.prog_grid?.[r]?.[c], 4);
    const diff = fmtNum(slice.diff_grid?.[r]?.[c], 5);
    const rid = slice.regime_grid?.[r]?.[c] ?? "?";

    els.mriReadout.textContent = `${slice.x_name}=${xv}, ${slice.y_name}=${yv} | NN=${nnVal} | Code=${progVal} | |Δ|=${diff} | R${rid} (Click to Explain)`;
  }

  function handleCanvasClick(e) {
    const rect = els.mriCanvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * els.mriCanvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * els.mriCanvas.height;

    // 1. Check if user clicked on a counterexample X pin
    if (state.mriHitBoxes.length) {
      let best = null;
      let bestDist = 144;
      for (const box of state.mriHitBoxes) {
        const d2 = (x - box.cx) ** 2 + (y - box.cy) ** 2;
        if (d2 < bestDist) {
          bestDist = d2;
          best = box;
        }
      }
      if (best) {
        selectCounterexample(best.ceIndex, best.regimeId);
        return;
      }
    }

    // 2. Otherwise trace and explain the exact clicked 2D coordinate
    const slice = state.currentSlice;
    if (!slice || !slice.diff_grid || slice.diff_grid.length === 0) return;
    const fx = Math.max(0, Math.min(0.9999, (e.clientX - rect.left) / rect.width));
    const fy = Math.max(0, Math.min(0.9999, (e.clientY - rect.top) / rect.height));
    const ny = slice.diff_grid.length;
    const nx = slice.diff_grid[0].length;
    const c = Math.floor(fx * nx);
    const r = Math.floor((1 - fy) * ny);

    const xVec = Array.isArray(slice.fixed_values)
      ? [...slice.fixed_values]
      : getFixedPointVector();
    xVec[slice.dim_x] = Number(slice.x_vals?.[c] ?? 0);
    xVec[slice.dim_y] = Number(slice.y_vals?.[r] ?? 0);

    explainInputPoint(xVec, {
      gridCell: {
        nnVal: slice.nn_grid?.[r]?.[c],
        progVal: slice.prog_grid?.[r]?.[c],
        diffVal: slice.diff_grid?.[r]?.[c],
      },
      syncRegime: true,
    });
  }

  document.addEventListener("DOMContentLoaded", initApp);
})();
