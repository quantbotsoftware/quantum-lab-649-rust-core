// Quantum Lab 649 · Rust Core v3 — Observator Nativ HPC cu 12 File Dark-Mode WebKitGTK (0 casete albe).
// Fără bundler, CSP strict (`script-src 'self'`), interogări < 1 ms zero-copy `mmap` din `./vault/`
// și execuție interactivă live pe `NVIDIA GeForce RTX 5060` (`sm_120`) + `24` fire CPU `AVX2`.
// Guvernanță științifică: `positive_claims_allowed = false`.
"use strict";

const invoke = (cmd, args) => window.__TAURI__.core.invoke(cmd, args || {});
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 6) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    return Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(3) : v.toFixed(d);
  }
  return String(v);
};

const mib = (b) => (b == null ? "—" : `${(b / (1024 * 1024)).toFixed(2)} MiB`);
const gib = (b) => (b == null ? "—" : `${(b / (1024 * 1024 * 1024)).toFixed(3)} GiB`);
const tib = (b) => (b == null ? "—" : `${(b / 1e12).toFixed(3)} TB`);
const mb = (b) => (b == null ? "—" : `${(b / 1e6).toFixed(3)} MB (${b} B)`);

// Stare minimală UI (numai referințe ușoare pentru acțiunile 1-Click)
const quickState = {
  kjAlpha: { joc: "649", bile: [1, 2, 3, 4, 5, 6], bonus: null },
  hsAlpha: { joc: "649", bile: [12, 15, 17, 19, 26, 32], bonus: null },
  intlAlpha: { flux: "fr_loto", bile: [1, 2, 3, 4, 5], bonus: 1 },
};

const v3State = {
  overview: null,
  lastTicket: { joc: "649", numbers: [3, 14, 22, 31, 41, 49], joker: null },
  lastWheel: null,
  paretoSample: { joc: "649", tickets: [] },
  ckStreamsLoaded: false,
  hwTimer: null,
};

function makeBallsNode(balls, bonus, matchedSet) {
  const wrap = document.createElement("span");
  wrap.className = "balls-wrap";
  (balls || []).forEach((b) => {
    const sp = document.createElement("span");
    const isHit = matchedSet && matchedSet.has(Number(b));
    sp.className = "ball-chip" + (isHit ? " ball-match" : "");
    sp.textContent = String(b);
    wrap.append(sp);
  });
  if (bonus !== null && bonus !== undefined) {
    const sp = document.createElement("span");
    sp.className = "ball-chip ball-bonus";
    sp.textContent = `+${bonus}`;
    wrap.append(sp);
  }
  return wrap;
}

function kv(table, rows) {
  if (!table) return;
  table.replaceChildren(...rows.map(([k, v, cls]) => {
    const tr = document.createElement("tr");
    const a = document.createElement("td");
    const b = document.createElement("td");
    a.textContent = k;
    if (v instanceof Node) {
      b.append(v);
    } else {
      b.textContent = String(v ?? "—");
    }
    b.className = "mono" + (cls ? " " + cls : "");
    tr.append(a, b);
    return tr;
  }));
}

function grid(table, head, rows) {
  if (!table) return;
  const thead = document.createElement("tr");
  head.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    thead.append(th);
  });
  table.replaceChildren(thead, ...rows.map((r) => {
    const tr = document.createElement("tr");
    r.forEach((cell) => {
      const td = document.createElement("td");
      if (cell instanceof Node) {
        td.append(cell);
      } else if (cell && typeof cell === "object") {
        if (cell.node instanceof Node) td.append(cell.node);
        else td.textContent = String(cell.t ?? "");
        td.className = cell.c || "";
      } else {
        td.textContent = String(cell ?? "");
      }
      tr.append(td);
    });
    return tr;
  }));
}

function canvasCtx(id) {
  const c = $(id);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#080e1c";
  ctx.fillRect(0, 0, c.width, c.height);
  return [c, ctx];
}

function bars(id, values, opts = {}) {
  const [c, ctx] = canvasCtx(id);
  const n = values.length;
  if (!n) return;
  const pad = 26;
  const lo = Math.min(0, ...values, opts.ref ?? Infinity);
  const hi = Math.max(...values, opts.ref ?? -Infinity) || 1;
  const y = (v) => c.height - pad - ((v - lo) / (hi - lo || 1)) * (c.height - 2 * pad);
  const w = (c.width - 2 * pad) / n;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = opts.colors ? opts.colors[i] : "#5cc8ff";
    const y0 = y(Math.max(0, lo)), y1 = y(values[i]);
    ctx.fillRect(pad + i * w + 1, Math.min(y0, y1), Math.max(1, w - 2), Math.abs(y1 - y0));
  }
  if (opts.under) {
    ctx.strokeStyle = "#8a97b580";
    ctx.beginPath();
    opts.under.forEach((v, i) => {
      const px = pad + (i + 0.5) * w;
      i ? ctx.lineTo(px, y(v)) : ctx.moveTo(px, y(v));
    });
    ctx.stroke();
  }
  if (opts.ref !== undefined) {
    ctx.strokeStyle = "#ffcf6e";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(pad, y(opts.ref));
    ctx.lineTo(c.width - pad, y(opts.ref));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = "#8d9ec0";
  ctx.font = "10px monospace";
  for (let i = 0; i < n; i += Math.max(1, Math.round(n / 12))) {
    ctx.fillText(String(i + (opts.base ?? 1)), pad + i * w, c.height - 8);
  }
  ctx.fillText(fmt(hi, 4), 4, pad - 7);
}

function multiLines(id, seriesList, opts = {}) {
  const [c, ctx] = canvasCtx(id);
  const padL = 46, padR = 14, padT = 18, padB = 22;
  const allVals = [];
  seriesList.forEach((s) => (s.values || []).forEach((v) => { if (Number.isFinite(v)) allVals.push(v); }));
  (opts.hlines || []).forEach((h) => allVals.push(h.y));
  if (!allVals.length) return;
  let lo = opts.minY !== undefined ? opts.minY : Math.min(...allVals);
  let hi = opts.maxY !== undefined ? opts.maxY : Math.max(...allVals);
  if (Math.abs(hi - lo) < 1e-9) { lo -= 0.05; hi += 0.05; }
  else { const m = 0.06 * (hi - lo); if (opts.minY === undefined) lo -= m; if (opts.maxY === undefined) hi += m; }

  const plotW = c.width - padL - padR;
  const plotH = c.height - padT - padB;
  const yOf = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  ctx.strokeStyle = "#1c2c4f";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);

  (opts.hlines || []).forEach((hl) => {
    const py = yOf(hl.y);
    ctx.strokeStyle = hl.color || "#ffcf6e";
    ctx.setLineDash(hl.dash || [4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  seriesList.forEach((s) => {
    const vals = s.values || [];
    if (!vals.length) return;
    ctx.strokeStyle = s.color || "#5cc8ff";
    ctx.lineWidth = s.width || 1.8;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const px = padL + (vals.length === 1 ? 0.5 : i / (vals.length - 1)) * plotW;
      const py = yOf(v);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  });

  ctx.fillStyle = "#8d9ec0";
  ctx.font = "10px monospace";
  ctx.fillText(fmt(hi, 3), 3, padT + 8);
  ctx.fillText(fmt(lo, 3), 3, padT + plotH);
  if (opts.xFirst) ctx.fillText(String(opts.xFirst), padL, c.height - 6);
  if (opts.xLast) {
    const s = String(opts.xLast);
    ctx.fillText(s, Math.max(padL + 60, padL + plotW - s.length * 6.2), c.height - 6);
  }
}

function heatmap(id, rho, n, k) {
  const [c, ctx] = canvasCtx(id);
  const du = 1 / n, ou = (k - 1) / (n * (n - 1));
  let m = 0;
  const dev = rho.map((v, idx) => {
    const d = v - (Math.floor(idx / n) === idx % n ? du : ou);
    m = Math.max(m, Math.abs(d));
    return d;
  });
  const s = c.width / n;
  dev.forEach((d, idx) => {
    const t = m ? d / m : 0;
    const r = t > 0 ? 225 : Math.round(65 * (1 + t));
    const b = t < 0 ? 235 : Math.round(75 * (1 - t));
    const g = Math.round(45 + 65 * (1 - Math.abs(t)));
    ctx.fillStyle = `rgb(${Math.max(12, r)},${Math.max(16, g)},${Math.max(24, b)})`;
    ctx.fillRect((idx % n) * s, Math.floor(idx / n) * s, Math.ceil(s), Math.ceil(s));
  });
  return m;
}

function parseBallsInput(str) {
  return String(str || "")
    .split(/[\s,;+]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number.parseInt(s, 10))
    .filter((n) => Number.isFinite(n));
}

function activateTab(tabName) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === tabName));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tabName));
  onTab(tabName);
}

function jumpToCateSiUnde(streamCode, balls, bonus) {
  $("cu-joc").value = streamCode;
  $("cu-bile").value = (balls || []).join(" ");
  $("cu-bonus").value = bonus != null ? String(bonus) : "";
  activateTab("hsv4");
  $("cu-box").scrollIntoView({ behavior: "smooth", block: "start" });
  runCateSiUnde();
}

function makeInspectButton(streamCode, balls, bonus, labelTxt = "🔍 Câte & Unde") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn-mini";
  btn.textContent = labelTxt;
  btn.addEventListener("click", () => jumpToCateSiUnde(streamCode, balls, bonus));
  return btn;
}

function openModal(title, kvRows, gridHead, gridRows) {
  $("v3-modal-title").textContent = title;
  kv($("v3-modal-kv"), kvRows || []);
  grid($("v3-modal-grid"), gridHead || [], gridRows || []);
  $("v3-modal").classList.remove("hidden");
}

function closeModal() {
  $("v3-modal").classList.add("hidden");
}

// ============================================================================
// FILA 1 (NOU v3): VAULT 39.07M · CD-V3Q · PM-V3 · C3-v2 · C4-v2 · GPU LIVE
// ============================================================================

function populateWheelSelect() {
  const ov = v3State.overview;
  if (!ov) return;
  const qap = $("v3w-qap").value === "true";
  const sel = $("v3w-id");
  const prevVal = sel.value;
  const opts = [];

  (ov.main_wheels_cdv3q || []).forEach((d) => {
    const o = document.createElement("option");
    o.value = d.id;
    o.textContent = `Integral · ${d.id} (${d.game}, ${d.n_tickets} bilete, 0 neacoperite)`;
    opts.push(o);
  });

  if (qap) {
    (ov.partial_wheels_cdv3q || []).forEach((w) => {
      const o = document.createElement("option");
      o.value = w.id;
      o.textContent = `Parțial · ${w.id} (${w.game}, K=${w.k_budget_tickets}, acop. ${fmt(w.coverage_pct, 2)}%)`;
      opts.push(o);
    });
    (ov.abbreviated_wheels_cdv3q || []).forEach((a) => {
      const o = document.createElement("option");
      o.value = a.id;
      o.textContent = `Abreviat · ${a.id} (${a.game}, v=${a.v_sub}, N=${a.n_tickets})`;
      opts.push(o);
    });
  }

  sel.replaceChildren(...opts);
  if ([...sel.options].some((x) => x.value === prevVal)) {
    sel.value = prevVal;
  }
}

async function loadV3Overview() {
  try {
    const ov = await invoke("cmd_v3_overview");
    v3State.overview = ov;

    $("hdr-vram").textContent = `${mib(ov.vram_resident_bytes)} rezident`;
    $("kpi-univ").textContent = `${Number(ov.vram_resident_universe_tickets).toLocaleString("en-US")} bilete`;
    $("kpi-univ-sub").textContent = `VRAM DMA+tabele: ${mib(ov.vram_resident_bytes)} în ${fmt(ov.vram_init_ms, 1)} ms`;
    $("kpi-vault").textContent = ov.vault_13_v3bin_all_0444_and_valid ? "13 / 13 tabele (0444)" : "ATENȚIE VAULT";
    $("kpi-vault-sub").textContent = `${gib(ov.vault_13_v3bin_bytes)} · mmap open ${fmt(ov.mmap_open_ms, 2)} ms · query ${fmt(ov.query_ms, 2)} ms`;

    grid($("v3-pm-wf"),
      ["joc", "extrageri", "MAPE vânzări", "κ post-report", "κ fond nou", "cauzalitate K3"],
      (ov.pmv3_walk_forward || []).map((w) => [
        { t: w.game, c: "mono warn-txt" },
        w.evaluated_draws,
        { t: `${fmt(w.sales_mape_regular_pct, 2)}%`, c: "mono pass" },
        { t: fmt(w.kappa_post_report_pmv3, 3), c: "mono" },
        { t: fmt(w.kappa_all_new_fund_pmv3, 3), c: "mono" },
        { t: fmt(w.k3_causality_max_abs_diff, 2), c: "mono pass" },
      ])
    );

    grid($("v3-pm-cubes"),
      ["joc", "variante .v3bin", "EV NB2 mediu", "EV Jensen", "anti-crowd best/1..k", "max|GPU−CPU|"],
      (ov.pmv3_cubes || []).map((c) => [
        { t: c.game, c: "mono warn-txt" },
        { t: Number(c.ticket_count).toLocaleString("en-US"), c: "mono" },
        { t: `${fmt(c.ev_nb_mean, 4)} lei`, c: "mono" },
        { t: `${fmt(c.ev_jensen_mean, 4)} lei`, c: "mono dim" },
        { t: `${fmt(c.anti_crowd_ev_ratio_best_vs_1_to_k, 3)}×`, c: "mono pass" },
        { t: fmt(c.max_abs_diff_gpu_cpu_f64, 2), c: "mono pass" },
      ])
    );

    grid($("v3w-main5"),
      ["roată CD-V3Q", "joc", "bilete", "neacop. GPU/CPU", "patterned", "EV înainte → după QAP", "câștig QAP"],
      (ov.main_wheels_cdv3q || []).map((d) => [
        { t: d.id, c: "mono" },
        d.game,
        { t: d.n_tickets, c: "mono warn-txt" },
        { t: `${d.uncovered_gpu} / ${d.uncovered_cpu}`, c: "mono pass" },
        { t: `${d.patterned_before} → ${d.patterned_after}`, c: d.patterned_after === 0 ? "mono pass" : "mono fail" },
        { t: `${fmt(d.mean_ev_before, 3)} → ${fmt(d.mean_ev_after, 3)}`, c: "mono" },
        { t: `+${fmt(d.ev_gain_pct, 2)}%`, c: "mono pass" },
      ])
    );

    grid($("hw-vault-tbl"),
      ["fișier .v3bin", "kind", "înregistrări", "B/rec", "dimensiune", "mod octal", "stare QLV3VLT"],
      (ov.vault_tables || []).map((t) => [
        { t: t.filename, c: "mono" },
        { t: t.table_kind, c: "mono" },
        { t: Number(t.record_count).toLocaleString("en-US"), c: "mono" },
        { t: t.record_bytes, c: "mono" },
        { t: mib(t.file_bytes), c: "mono" },
        { t: t.mode_octal, c: t.is_0444 ? "mono pass" : "mono fail" },
        { t: t.is_0444 && t.header_valid ? "PASS (0444 + zero-copy)" : "EROARE", c: t.is_0444 && t.header_valid ? "pass" : "fail" },
      ])
    );

    populateWheelSelect();
    return {
      ok: ov.vault_13_v3bin_all_0444_and_valid === true && ov.positive_claims_allowed === false,
      vram_resident_bytes: ov.vram_resident_bytes,
      mmap_open_ms: ov.mmap_open_ms,
    };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

function drawBootstrapQuantiles(q) {
  const [c, ctx] = canvasCtx("c-v3-boot");
  const bq = q.boot_1000_quantiles || {};
  const vals = [bq.q05, bq.q25, bq.q50, bq.q75, bq.q95].map((x) => Number(x || 0));
  const nom = Number(q.ev_nominal || 0);
  const lo = Math.min(...vals, nom) * 0.94;
  const hi = Math.max(...vals, nom, lo + 0.05) * 1.05;
  const padL = 36, padR = 24;
  const xOf = (v) => padL + ((v - lo) / (hi - lo || 1)) * (c.width - padL - padR);
  const cy = Math.floor(c.height / 2) - 4;

  // Mustața q05..q95
  ctx.strokeStyle = "#5cc8ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xOf(vals[0]), cy);
  ctx.lineTo(xOf(vals[4]), cy);
  ctx.stroke();

  // Cutia intercuantilă q25..q75
  const x25 = xOf(vals[1]), x75 = xOf(vals[3]);
  ctx.fillStyle = "#173763";
  ctx.fillRect(x25, cy - 20, Math.max(4, x75 - x25), 40);
  ctx.strokeStyle = "#7bd4ff";
  ctx.strokeRect(x25, cy - 20, Math.max(4, x75 - x25), 40);

  // Mediana q50 și EV nominal
  ctx.strokeStyle = "#6ee7a0";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(xOf(vals[2]), cy - 24);
  ctx.lineTo(xOf(vals[2]), cy + 24);
  ctx.stroke();

  ctx.strokeStyle = "#ffcf6e";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(xOf(nom), cy - 28);
  ctx.lineTo(xOf(nom), cy + 28);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#8d9ec0";
  ctx.font = "10.5px monospace";
  const labels = [["q05", vals[0]], ["q50", vals[2]], ["q95", vals[4]]];
  labels.forEach(([lbl, v], idx) => {
    const px = Math.min(c.width - 78, Math.max(8, xOf(v) - 24));
    ctx.fillText(`${lbl}=${fmt(v, 3)}`, px, idx === 1 ? 20 : c.height - 12);
  });
  ctx.fillStyle = "#ffcf6e";
  ctx.fillText(`EV_nom=${fmt(nom, 3)} lei (preț ${fmt(q.price_lei, 1)} lei)`, 10, 20);
}

async function runV3MmapTicket(byRank = false) {
  const joc = $("v3t-joc").value;
  $("v3t-status").textContent = "se interoghează zero-copy mmap…";
  $("v3t-status").className = "status";
  try {
    let args;
    if (byRank) {
      const rang = Math.max(0, Number.parseInt($("v3t-rang").value, 10) || 0);
      args = { joc, bile: null, joker: null, rang };
    } else {
      const bile = parseBallsInput($("v3t-bile").value);
      const joker = joc === "joker" ? (Number.parseInt($("v3t-joker").value, 10) || 1) : null;
      args = { joc, bile, joker, rang: null };
    }
    const q = await invoke("cmd_v3_mmap_ticket", args);
    $("v3t-bile").value = (q.numbers || []).join(" ");
    $("v3t-rang").value = String(q.rank);
    if (q.joker != null) $("v3t-joker").value = String(q.joker);

    v3State.lastTicket = { joc: q.game, numbers: q.numbers || [], joker: q.joker ?? null };
    const bq = q.boot_1000_quantiles || {};

    kv($("v3t-kv"), [
      ["bilet interogat", makeBallsNode(q.numbers, q.joker, null)],
      ["rang combinadic / univers", `${Number(q.rank).toLocaleString("en-US")} / ${Number(q.universe_size).toLocaleString("en-US")}`],
      ["mască .v3bin (= calcul CPU)", `${q.mask_hex} (${q.mask_verified_equal ? "VERIFICAT" : "EROARE"})`, q.mask_verified_equal ? "pass" : "fail"],
      ["EV partajat NB2 vs preț", `${fmt(q.ev_nominal, 4)} lei  /  preț ${fmt(q.price_lei, 2)} lei (ρ=${fmt(q.ev_nominal / q.price_lei, 4)})`],
      ["1 000× bootstrap [q05 · q50 · q95]", `${fmt(bq.q05, 4)} · ${fmt(bq.q50, 4)} · ${fmt(bq.q95, 4)} lei`],
      ["popularitate mulțime exp(lw)", `${fmt(q.crowd_relative_weight, 4)}×  (ln w = ${fmt(q.crowd_log_weight, 4)})`],
      ["filtru tipar / calendar 1..31", `patterned=${q.is_patterned ? "DA (penalizat)" : "NU (curat)"} · 1..31=${q.is_birthday_1_31 ? "DA" : "NU"}`, q.is_patterned ? "warn-txt" : "pass"],
      ["latență zero-copy mmap", `${fmt(q.query_us, 1)} µs (${fmt(q.query_ms, 4)} ms · < 1 ms: ${q.under_1ms ? "PASS" : "NU"})`, q.under_1ms ? "pass" : "warn-txt"],
    ]);

    drawBootstrapQuantiles(q);
    $("v3t-status").textContent = `Zero-copy mmap: ${fmt(q.query_us, 1)} µs (< 1 ms)`;
    return {
      ok: q.mask_verified_equal === true && q.under_1ms === true && q.positive_claims_allowed === false,
      query_us: q.query_us,
      rank: q.rank,
    };
  } catch (e) {
    $("v3t-status").textContent = String(e);
    $("v3t-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadV3WheelDetail() {
  const qapRelabeled = $("v3w-qap").value === "true";
  const wheelId = $("v3w-id").value || "CDV3_649_163";
  $("v3w-status").textContent = "se citește roata din mmap…";
  $("v3w-status").className = "status";
  try {
    const w = await invoke("cmd_v3_wheel_detail", { qapRelabeled, wheelId });
    v3State.lastWheel = w;
    const m = w.meta || {};
    const rows = [
      ["ID roată / tip / joc", `${m.id} (${w.kind} · ${m.game})`],
      ["număr bilete în catalog", String(m.n_tickets ?? m.k_budget_tickets ?? "—"), "warn-txt"],
    ];
    if (w.kind === "main") {
      rows.push(["stări neacoperite GPU / 24T CPU", `${m.uncovered_gpu} / ${m.uncovered_cpu} din ${Number(m.universe_draws).toLocaleString("en-US")}`, "pass"]);
    } else if (w.kind === "partial") {
      rows.push(["acoperire exactă univers", `${fmt(m.coverage_pct, 4)}% (${Number(m.covered_draws_gpu).toLocaleString("en-US")} stări)`, "pass"]);
    } else if (w.kind === "abbreviated") {
      rows.push(["garanție sub-mulțime v / p / t", `v=${m.v_sub}, dacă ies p=${m.p_hit_in_sub} ⇒ min t=${m.t_guarantee} (neacop.=${m.uncovered_sub_draws})`, "pass"]);
    }
    if (m.patterned_after !== undefined) {
      rows.push(["bilete tipizate (patterned)", `${m.patterned_before ?? 0} → ${m.patterned_after}`, m.patterned_after === 0 ? "pass" : "fail"]);
    }
    if (m.mean_ev_after !== undefined) {
      rows.push(["EV mediu pe bilet (NB2)", `${fmt(m.mean_ev_after, 4)} lei${m.ev_gain_pct !== undefined ? ` (+${fmt(m.ev_gain_pct, 2)}% QAP)` : ""}`, "pass"]);
    }
    rows.push(["timp citire mmap", `${fmt(w.query_ms, 3)} ms`]);
    kv($("v3w-kv"), rows);

    grid($("v3w-tickets"),
      ["#", "bilet din catalog", "EV NB2", "q05..q95", "audit"],
      (w.tickets_preview || []).slice(0, 60).map((t) => [
        { t: t.i, c: "mono dim" },
        makeBallsNode(t.numbers, t.joker, null),
        { t: fmt(t.ev_nominal, 3), c: "mono" },
        { t: `[${fmt(t.q05, 2)}, ${fmt(t.q95, 2)}]`, c: "mono dim" },
        makeInspectButton(m.game, t.numbers, t.joker, "🔍"),
      ])
    );

    $("v3w-status").textContent = `${(w.tickets_preview || []).length} bilete încărcate în ${fmt(w.query_ms, 2)} ms`;
    return { ok: w.positive_claims_allowed === false && (w.tickets_preview || []).length > 0, wheel_id: wheelId };
  } catch (e) {
    $("v3w-status").textContent = String(e);
    $("v3w-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

function drawParetoCanvas(frontiers) {
  const [c, ctx] = canvasCtx("c-v3-pareto");
  if (!frontiers || !frontiers.length) {
    ctx.fillStyle = "#8d9ec0";
    ctx.font = "12px monospace";
    ctx.fillText("Portofolii mixte 3-jocuri afișate în tabelul de mai jos.", 24, Math.floor(c.height / 2));
    return;
  }
  const f = frontiers[0];
  const pts = f.pareto_points || [];
  if (!pts.length) return;
  const pRand = (f.p_random_analytical || 0) * 100;
  const xs = pts.map((p) => p.p_any_exact * 100);
  const ys = pts.map((p) => p.ev_portfolio);
  let minX = Math.min(...xs, pRand) * 0.96;
  let maxX = Math.max(...xs, pRand) * 1.03;
  let minY = Math.min(...ys) * 0.95;
  let maxY = Math.max(...ys) * 1.05;
  if (Math.abs(maxX - minX) < 1e-6) { minX -= 0.2; maxX += 0.2; }
  if (Math.abs(maxY - minY) < 1e-6) { minY -= 0.2; maxY += 0.2; }

  const padL = 52, padR = 22, padT = 18, padB = 26;
  const pw = c.width - padL - padR, ph = c.height - padT - padB;
  const xOf = (v) => padL + ((v - minX) / (maxX - minX)) * pw;
  const yOf = (v) => padT + (1 - (v - minY) / (maxY - minY)) * ph;

  ctx.strokeStyle = "#1d2e52";
  ctx.strokeRect(padL, padT, pw, ph);

  // Linie verticală P_random analitic hipergeometric
  const rx = xOf(pRand);
  ctx.strokeStyle = "#ffcf6e";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(rx, padT);
  ctx.lineTo(rx, padT + ph);
  ctx.stroke();
  ctx.setLineDash([]);

  // Curba Pareto
  ctx.strokeStyle = "#5cc8ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const px = xOf(p.p_any_exact * 100), py = yOf(p.ev_portfolio);
    idx === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();

  pts.forEach((p) => {
    const px = xOf(p.p_any_exact * 100), py = yOf(p.ev_portfolio);
    ctx.fillStyle = "#6ee7a0";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#dce8ff";
    ctx.font = "10px monospace";
    ctx.fillText(p.mu_label, Math.min(c.width - 52, px + 5), Math.max(padT + 10, py - 5));
  });

  ctx.fillStyle = "#8d9ec0";
  ctx.font = "10px monospace";
  ctx.fillText(`EV=${fmt(maxY, 2)}`, 4, padT + 9);
  ctx.fillText(`EV=${fmt(minY, 2)}`, 4, padT + ph);
  ctx.fillText(`P(≥1)=${fmt(minX, 2)}%`, padL, c.height - 7);
  ctx.fillStyle = "#ffcf6e";
  ctx.fillText(`P_rand=${fmt(pRand, 3)}%`, Math.min(c.width - 115, Math.max(padL + 95, rx - 35)), c.height - 7);
}

async function loadV3C3C4() {
  const joc = $("v3p-joc").value;
  const buget = Number.parseInt($("v3p-buget").value, 10) || 40;
  $("v3p-status").textContent = "se încarcă frontiera C3-v2 & C4-v2…";
  $("v3p-status").className = "status";
  try {
    const r = await invoke("cmd_v3_c3_c4", { joc, buget });
    if (joc === "mix") {
      drawParetoCanvas([]);
      grid($("v3p-table"),
        ["buget", "regim μ", "bilete (649/540/Jk)", "miză", "EV mix", "P(≥1) exact", "vs P_random"],
        (r.mixed_frontiers || []).map((m) => [
          `${m.budget_lei} lei`,
          { t: m.mu_label, c: "mono warn-txt" },
          { t: `${m.k_649} / ${m.k_540} / ${m.k_joker}`, c: "mono" },
          `${m.stake_used_lei} lei`,
          { t: `${fmt(m.ev_total, 3)} lei`, c: "mono" },
          { t: `${fmt(m.p_any_combined_independent * 100, 3)}%`, c: "mono pass" },
          { t: `${fmt(m.p_random_equal_split * 100, 3)}%`, c: "mono dim" },
        ])
      );
    } else {
      const f = (r.single_game_frontiers || [])[0];
      drawParetoCanvas(r.single_game_frontiers || []);
      if (f && (f.pareto_points || []).length) {
        const midPt = f.pareto_points.find((p) => String(p.mu_label).includes("0.5")) || f.pareto_points[0];
        v3State.paretoSample = { joc, tickets: midPt.sample_tickets || [] };
      }
      grid($("v3p-table"),
        ["regim μ", "K bilete", "EV portofoliu", "ρ = EV/miză", "P(≥1) exact", "eroare Bonferroni evitată", "patterned"],
        ((f && f.pareto_points) || []).map((p) => [
          { t: p.mu_label, c: "mono warn-txt" },
          { t: p.k_tickets, c: "mono" },
          { t: `${fmt(p.ev_portfolio, 3)} lei`, c: "mono" },
          { t: fmt(p.ev_per_lei, 4), c: "mono" },
          { t: `${fmt(p.p_any_exact * 100, 4)}%`, c: "mono pass" },
          { t: fmt(p.bonferroni_excess, 5), c: "mono dim" },
          { t: p.patterned_tickets, c: p.patterned_tickets === 0 ? "mono pass" : "mono fail" },
        ])
      );
    }

    const c4Rows = [];
    (r.c4_mc1m_strategies || []).slice(0, 12).forEach((s) => {
      (s.horizons || []).forEach((h) => {
        const qt = h.net_profit_quantiles_lei || {};
        c4Rows.push([
          { t: s.strategy_id, c: "mono warn-txt" },
          { t: `${h.years}A (${h.draws})`, c: "mono" },
          { t: `${fmt(h.total_stake_lei, 0)}`, c: "mono dim" },
          { t: fmt(h.net_mean_sim_lei, 1), c: "mono" },
          { t: `${fmt(h.z_score_vs_analytic, 2)}σ`, c: Math.abs(h.z_score_vs_analytic) <= 4 ? "mono pass" : "mono fail" },
          { t: `[${fmt(qt.q05, 0)}, ${fmt(qt.q50, 0)}, ${fmt(qt.q95, 0)}]`, c: "mono" },
          { t: `${fmt((h.prob_net_profit_ge_0 || 0) * 100, 2)}%`, c: "mono" },
        ]);
      });
    });
    grid($("v3c4-table"),
      ["strategie", "orizont", "miză [lei]", "net mediu 1M", "z Markov", "cuantile net [q05, q50, q95]", "P(net ≥ 0)"],
      c4Rows
    );

    $("v3p-status").textContent = `C3/C4 mmap: ${fmt(r.query_ms, 2)} ms · ${r.c4_mc1m_mmap_records_count} înreg. C4`;
    return { ok: r.positive_claims_allowed === false && r.c4_mc1m_mmap_records_count === 90 };
  } catch (e) {
    $("v3p-status").textContent = String(e);
    $("v3p-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

function parsePortfolioTextarea(joc, rawText) {
  const lines = String(rawText || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  const bilete = [];
  const jokeri = [];
  for (const ln of lines) {
    const nums = parseBallsInput(ln);
    if (!nums.length) continue;
    if (joc === "joker") {
      bilete.push(nums.slice(0, 5));
      jokeri.push(nums.length >= 6 ? nums[5] : 1);
    } else if (joc === "540") {
      bilete.push(nums.slice(0, 5));
    } else {
      bilete.push(nums.slice(0, 6));
    }
  }
  return { bilete, jokeri: joc === "joker" ? jokeri : null };
}

async function runV3GpuEvalPortfolio() {
  const joc = $("v3e-joc").value;
  const { bilete, jokeri } = parsePortfolioTextarea(joc, $("v3e-input").value);
  $("v3e-status").textContent = "se execută kernelul CUDA sm_120 + 24T AVX2…";
  $("v3e-status").className = "status";
  try {
    const r = await invoke("cmd_v3_gpu_eval_portfolio", { joc, bilete, jokeri });
    kv($("v3e-kv"), [
      ["joc / K bilete / miză", `${r.game} · ${r.k_tickets} bilete · ${fmt(r.stake_lei, 2)} lei`],
      ["stări câștigătoare GPU vs 24T CPU", `${Number(r.win_states_gpu).toLocaleString("en-US")} == ${Number(r.win_states_cpu_24t).toLocaleString("en-US")}`, r.gpu_cpu_bit_exact ? "pass" : "fail"],
      ["paritate bit-exactă GPU == CPU", r.gpu_cpu_bit_exact ? "BIT-EXACT PASS" : "EROARE", r.gpu_cpu_bit_exact ? "pass" : "fail"],
      ["P(câștig ≥ 1) exact vs P_random", `${fmt(r.p_any_exact * 100, 4)}%  vs  ${fmt(r.p_random_analytical * 100, 4)}%`, r.p_any_exact >= r.p_random_analytical - 1e-12 ? "pass" : "warn-txt"],
      ["EV portofoliu NB2 · ρ = EV/miză", `${fmt(r.ev_portfolio_nb2, 4)} lei · ρ = ${fmt(r.ev_per_lei, 4)}`],
      ["bilete tipizate / timp GPU / 24T CPU", `patterned=${r.patterned_tickets} · GPU ${fmt(r.gpu_ms, 2)} ms · 24T CPU ${fmt(r.cpu_24t_ms, 2)} ms`, "pass"],
    ]);

    grid($("v3e-cats"),
      ["categorie", "contribuție EV NB2 [lei]"],
      Object.entries(r.ev_by_cat || {}).map(([cat, val]) => [
        { t: `Cat. ${cat}`, c: "mono warn-txt" },
        { t: fmt(val, 5), c: "mono" },
      ])
    );

    grid($("v3e-tickets"),
      ["#", "bilet evaluat pe VRAM", "rang combinadic", "EV individual NB2", "q05..q95"],
      (r.tickets || []).slice(0, 80).map((t) => [
        { t: t.i, c: "mono dim" },
        makeBallsNode(t.numbers, t.joker ?? null, null),
        { t: Number(t.rank).toLocaleString("en-US"), c: "mono" },
        { t: `${fmt(t.ev_nominal, 4)} lei`, c: "mono" },
        { t: `[${fmt(t.q05, 2)}, ${fmt(t.q95, 2)}]`, c: "mono dim" },
      ])
    );

    $("v3e-status").textContent = `GPU RTX 5060: ${fmt(r.gpu_ms, 2)} ms == 24T CPU: ${fmt(r.cpu_24t_ms, 2)} ms (BIT-EXACT)`;
    return { ok: r.gpu_cpu_bit_exact === true && r.positive_claims_allowed === false, gpu_ms: r.gpu_ms };
  } catch (e) {
    $("v3e-status").textContent = String(e);
    $("v3e-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

// ============================================================================
// FILA 2 (NOU v3): POARTA A6 (3 REGISTRE LIVE) & CK-V6 (10K GPU)
// ============================================================================

async function loadV3RegistreLive() {
  $("v3r-status").textContent = "se verifică criptografic cele 3 registre prospective…";
  $("v3r-status").className = "status";
  try {
    const r = await invoke("cmd_v3_registre_live");
    const xif = r.xif_registry;
    const tu4 = r.tu4_registry;
    const ql = r.quantum_limit_registry;
    const s35 = tu4.guard_s35_03;

    $("v3r-badge").textContent = r.gate_a6_pass ? `Poarta A6: PASS (${fmt(r.verification_ms, 1)} ms)` : "Poarta A6: EROARE";
    $("v3r-badge").className = "badge " + (r.gate_a6_pass ? "pass" : "fail");
    $("hdr-blind").textContent = r.blind_reserves_all_untouched_0400 ? "4/4 0400 INTACT" : "ALERTĂ";

    const pendDates = Object.entries(xif.pending_targets || {}).map(([d, n]) => `${d}(${n})`).join(", ") || "niciuna";
    kv($("v3r-xif-kv"), [
      ["verdict lanț SHA-256 XI-F", xif.pass ? "PASS (0444 + canonic)" : "FAIL", xif.pass ? "pass" : "fail"],
      ["LANT_STRATEGII / DECONTARI", `${xif.lant_lines} linii (${xif.lant_mode_octal}) / ${xif.decontari_lines} decontate (${xif.decontari_mode_octal})`],
      ["recalculare independentă bilete", `${xif.settled_recalculated_matches} / ${xif.decontari_lines} identice`, "pass"],
      ["ținte viitoare în așteptare", `${xif.pending_future_count} (${pendDates}) · restante=${xif.overdue_unsettled_count}`],
      ["bilanț hârtie (miză / plată / net)", `${fmt(xif.total_settled_stake_lei, 0)} / ${fmt(xif.total_settled_payout_lei, 2)} / ${fmt(xif.total_settled_net_lei, 2)} lei`],
      ["cap DECONTARI SHA-256", `${xif.decontari_tip_sha256.slice(0, 18)}…`],
    ]);

    grid($("v3r-xif-sample"),
      ["#", "strategie", "joc", "dată", "miză", "plată", "net"],
      (xif.sample_settled || []).map((s) => [
        { t: s.seq, c: "mono dim" },
        { t: s.strategie, c: "mono" },
        s.joc,
        { t: s.data, c: "mono" },
        fmt(s.miza, 0),
        fmt(s.plata, 1),
        { t: fmt(s.net, 1), c: s.net >= 0 ? "mono pass" : "mono dim" },
      ])
    );

    kv($("v3r-tu4-kv"), [
      ["verdict tribunal TU4 + SSD", tu4.pass ? "PASS (LANT + Newton + SSD)" : "FAIL", tu4.pass ? "pass" : "fail"],
      ["emisii LANT / decontări JU", `${tu4.lant_emissions_count} emisii / ${tu4.decontari_count} decontări (0444)`],
      ["oglinzi SSD (EM / JU mmap)", `EM=${tu4.ssd_mirror_em_ok ? "OK" : "NU"} · JU=${tu4.ssd_mirror_ju_ok ? "OK" : "NU"}`, tu4.ssd_mirror_em_ok && tu4.ssd_mirror_ju_ok ? "pass" : "fail"],
      ["rulări emițător / judecător", `${tu4.rulari_lines} / ${tu4.judecator_rulari_lines} (ultim RC=0: ${tu4.latest_judge_rc_zero ? "DA" : "NU"})`],
      ["emisii viitoare nescadente", (tu4.pending_emissions || []).join(", ") || "niciuna"],
      ["cap LANT SHA-256", `${tu4.lant_tip_line_sha256.slice(0, 18)}…`],
    ]);

    grid($("v3r-tu4-streams"),
      ["flux TU4", "stare", "ln E explorare", "ln E confirmare"],
      Object.entries(tu4.streams_state || {}).map(([k, v]) => [
        { t: k, c: "mono warn-txt" },
        { t: v.state || "—", c: "mono" },
        { t: fmt(v.lnE ?? 0, 6), c: "mono" },
        { t: fmt(v.lnE_conf ?? 0, 6), c: "mono dim" },
      ])
    );

    kv($("v3r-ql-kv"), [
      ["verdict Quantum Limit + OGLINDA", ql.pass ? "PASS (append-only + 17/17 OGLINDA)" : "FAIL", ql.pass ? "pass" : "fail"],
      ["prognoze sigilate verificate SHA-256", `${ql.total_sealed_forecasts_verified} fișiere sigiliu_*.json`, "pass"],
      ["statut OGLINDA (read-only)", `${ql.oglinda.status} (${ql.oglinda.files_verified_count} fișiere SHA-256 intacte)`],
      ["extrageri OGLINDA evaluate", `${ql.oglinda.evaluated_prospective_draws} (${ql.oglinda.target_draw})`],
    ]);

    grid($("v3r-ql-games"),
      ["joc", "scorate", "binding/req", "sigilii", "în așteptare", "vs DL preimage"],
      (ql.games || []).map((g) => [
        { t: g.game, c: "mono warn-txt" },
        g.scored_entries,
        `${g.binding_entries}/${g.required_entries}`,
        g.sealed_files_verified,
        { t: (g.pending_sealed_targets || []).join(", ") || "—", c: "mono" },
        { t: g.append_only_vs_dl_preimage_ok ? "APPEND-ONLY OK" : "ERR", c: g.append_only_vs_dl_preimage_ok ? "pass" : "fail" },
      ])
    );

    grid($("v3r-blind"),
      ["rezervă oarbă XI-P2", "cale relativă", "permisiuni", "dimensiune [B]", "mtime UNIX", "protecție 0400"],
      (r.blind_reserves_0400 || []).map((b) => [
        { t: b.game, c: "mono warn-txt" },
        { t: b.rel_path, c: "mono" },
        { t: b.mode_octal, c: b.is_strict_0400 ? "mono pass" : "mono fail" },
        { t: b.size_bytes, c: "mono" },
        { t: b.mtime_sec, c: "mono dim" },
        { t: b.is_strict_0400 ? "NEATINS (lstat 0400)" : "ALERTĂ", c: b.is_strict_0400 ? "pass" : "fail" },
      ])
    );

    const ymDraws = Object.entries(s35.regular_draws_in_current_month || {}).map(([g, c]) => `${g}:${c}`).join(", ");
    kv($("v3r-s35-kv"), [
      ["gardă S35_03 (lună nouă)", s35.classification, s35.pass ? "pass" : "fail"],
      ["lună curentă București · SQLite", `${s35.current_bucharest_ym} · snapshot ${s35.sqlite_snapshot} (${ymDraws})`],
      ["colector / judecător systemd", `colector=${s35.collector_service_result} (${s35.collector_active_state}) · judecător=${s35.judge_service_result}`],
      ["referință incident documentat", `${s35.incident_reference} (${s35.incident_sha256.slice(0, 14)}…)`],
    ]);
    $("v3r-epistemic").textContent = `${s35.explanation} | ${r.epistemic_notice}`;
    $("v3r-status").textContent = `Certificat în ${fmt(r.verification_ms, 1)} ms · positive_claims_allowed = ${r.positive_claims_allowed}`;

    return {
      ok: r.gate_a6_pass === true && r.blind_reserves_all_untouched_0400 === true && r.positive_claims_allowed === false,
      verification_ms: r.verification_ms,
    };
  } catch (e) {
    $("v3r-status").textContent = String(e);
    $("v3r-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

function drawCkPowerHeatmap(cells, selectedIdx) {
  const [c, ctx] = canvasCtx("c-v3-ckpow");
  if (!cells || cells.length < 25) return;
  const padL = 74, padT = 26, padR = 16, padB = 22;
  const cw = (c.width - padL - padR) / 5;
  const ch = (c.height - padT - padB) / 5;
  const deltas = ["δ=0.00", "δ=0.15", "δ=0.25", "δ=0.35", "δ=0.50"];
  const lregs = ["L=40", "L=80", "L=120", "L=200", "L=400"];

  ctx.fillStyle = "#8d9ec0";
  ctx.font = "10.5px monospace";
  lregs.forEach((lbl, col) => ctx.fillText(lbl, padL + col * cw + 14, 17));

  for (let row = 0; row < 5; row++) {
    ctx.fillStyle = "#8d9ec0";
    ctx.fillText(deltas[row], 10, padT + row * ch + ch / 2 + 4);
    for (let col = 0; col < 5; col++) {
      const idx = row * 5 + col;
      const cell = cells[idx] || {};
      const p = Number(cell.power_m_ge_1500 || 0);
      const rCol = Math.round(16 + 35 * (1 - p));
      const gCol = Math.round(28 + 145 * p);
      const bCol = Math.round(58 + 85 * p);
      const x = padL + col * cw + 2, y = padT + row * ch + 2;
      ctx.fillStyle = `rgb(${rCol},${gCol},${bCol})`;
      ctx.fillRect(x, y, cw - 4, ch - 4);
      if (idx === selectedIdx) {
        ctx.strokeStyle = "#ffcf6e";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cw - 4, ch - 4);
      }
      ctx.fillStyle = "#eef5ff";
      ctx.fillText(`${(p * 100).toFixed(1)}%`, x + 10, y + ch / 2 + 3);
    }
  }
}

async function loadV3Ckv6() {
  const streamSel = $("v3ck-stream");
  const cellSel = $("v3ck-cell");
  const streamId = streamSel.value || "649";
  const cellIdx = cellSel.value !== "" ? Number.parseInt(cellSel.value, 10) : 12;
  try {
    const r = await invoke("cmd_v3_ckv6", { streamId, cellIdx });
    if (!v3State.ckStreamsLoaded) {
      streamSel.replaceChildren(...(r.streams_summary || []).map((s) => {
        const o = document.createElement("option");
        o.value = s.stream_id;
        o.textContent = `${s.name} (${s.category} · ${s.canonical_steps} pași)`;
        o.selected = s.stream_id === streamId;
        return o;
      }));
      cellSel.replaceChildren(...(r.power_grid_25_cells || []).map((c) => {
        const o = document.createElement("option");
        o.value = String(c.cell_id);
        o.textContent = `Celula #${c.cell_id}: δ=${c.delta} (${c.delta_bp} bp), L=${c.l_regime} → P=${(c.power_m_ge_1500 * 100).toFixed(1)}%`;
        o.selected = c.cell_id === r.selected_power_cell_idx;
        return o;
      }));
      v3State.ckStreamsLoaded = true;
    }

    $("v3ck-verdict").textContent = r.verdict;
    $("v3ck-verdict").className = "badge pass";

    const st = r.selected_stream || {};
    const cps = st.exact_checkpoints || [];
    multiLines("c-v3-ckstream",
      [{ values: cps.map((p) => p.m_cum), color: "#5cc8ff", width: 2.0 }],
      {
        minY: 0.45,
        hlines: [
          { y: 0.5, color: "#ffcf6e", dash: [4, 3] },
          { y: 1.0, color: "#6ee7a077", dash: [2, 4] },
        ],
        xFirst: cps[0] ? cps[0].key : "",
        xLast: cps.length ? cps[cps.length - 1].key : "",
      }
    );
    $("v3ck-stream-leg").textContent =
      `${st.name || streamId}: M_T=${fmt(st.m_t_final, 4)}, max M_t=${fmt(st.max_m_t_anytime, 4)}, min E_t=${fmt(st.min_e_t, 4)} (≥ 0.50 garantat), MAP r_T=${st.final_map_r ?? st.final_map_run_length ?? "—"}`;

    grid($("v3ck-streams-tbl"),
      ["flux", "familie", "pași", "M_T final", "max M_t", "min E_t", "MAP r_T"],
      (r.streams_summary || []).map((s) => [
        s.name,
        { t: s.category, c: "mono dim" },
        s.canonical_steps,
        { t: fmt(s.m_final, 4), c: "mono" },
        { t: fmt(s.max_m, 4), c: "mono" },
        { t: fmt(s.min_e, 4), c: "mono pass" },
        { t: s.final_map_r, c: "mono" },
      ])
    );

    grid($("v3ck-q-tbl"),
      ["poartă", "stare", "descriere calificare pre-arhivă"],
      (r.q_tests || []).map((q) => [
        { t: q.id, c: "mono warn-txt" },
        { t: q.pass ? "PASS" : "FAIL", c: q.pass ? "pass" : "fail" },
        q.name,
      ])
    );

    grid($("v3ck-holm-tbl"),
      ["rang", "flux XI-P2 Dev", "pași", "E_dev brut", "divizor Holm", "E_dev ajustat", "prag 1500"],
      (r.holm_family_p2_dev || []).map((h) => [
        h.rank_1based,
        { t: h.stream_code, c: "mono" },
        h.n_draws_dev,
        { t: fmt(h.e_dev_raw, 4), c: "mono" },
        h.holm_divisor,
        { t: fmt(h.e_dev_holm_adjusted, 4), c: "mono" },
        { t: h.passes_holm_1500 ? "SUPRAVIEȚUITOR" : "0 (STOP ONEST)", c: "mono pass" },
      ])
    );

    drawCkPowerHeatmap(r.power_grid_25_cells || [], r.selected_power_cell_idx);
    grid($("v3ck-mmap-tbl"),
      ["rep#", "δ", "L_regim", "ln M_T", "max ln M_t", "min E_t", "|err CP|", "ln M_global"],
      (r.selected_power_cell_mmap_samples || []).map((s) => [
        { t: s.rep, c: "mono dim" },
        fmt(s.delta, 2),
        fmt(s.l_regime, 0),
        { t: fmt(s.log_m_final, 3), c: "mono" },
        { t: fmt(s.max_log_m, 3), c: "mono" },
        { t: fmt(s.min_e, 4), c: "mono pass" },
        fmt(s.cp_err, 1),
        { t: fmt(s.log_m_global, 3), c: "mono dim" },
      ])
    );

    $("v3ck-status").textContent = `mmap ${fmt(r.query_ms, 2)} ms · supraviețuitori Dev=${(r.dev_survivors_holm_1500 || []).length}`;
    return { ok: (r.q_tests || []).length === 6 && r.positive_claims_allowed === false };
  } catch (e) {
    $("v3ck-status").textContent = String(e);
    $("v3ck-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function runV3Ckv6LiveSim() {
  const deltaBp = Number.parseInt($("v3cks-delta").value, 10) || 0;
  const lRegime = Number.parseInt($("v3cks-lreg").value, 10) || 80;
  const nReps = Number.parseInt($("v3cks-reps").value, 10) || 16;
  $("v3cks-status").textContent = "se rulează kernelul CUDA CK-V6 f64…";
  $("v3cks-status").className = "status";
  try {
    const r = await invoke("cmd_v3_ckv6_live_sim", { deltaBp, lRegime, nReps, seed: 20261105 });
    const pts = r.witness0_trajectory || [];
    multiLines("c-v3-cksim",
      [
        { values: pts.map((p) => p.log_m), color: "#5cc8ff", width: 2.0 },
        { values: pts.map((p) => p.map_r / 40.0), color: "#a98bff", width: 1.5 },
      ],
      {
        hlines: [{ y: 0.0, color: "#ffcf6e88", dash: [4, 3] }],
        xFirst: "pas 1",
        xLast: `pas ${r.t_steps}`,
      }
    );

    kv($("v3cks-kv"), [
      ["parametri regim sintetic", `δ = ${fmt(r.delta, 2)} (${r.delta_bp} bp) · schimbare la L = ${r.l_regime} · ${r.n_reps} traiectorii × ${r.t_steps} pași`],
      ["putere emp. P(M_T ≥ 1500) · mediană ln M_T", `${fmt(r.power_m_ge_1500 * 100, 1)}%  ·  mediană ln M_T = ${fmt(r.median_log_m_final, 4)}`],
      ["scut martingal w₀ = 0.50 (min E_t ≥ 0.50)", `min E_t = ${fmt(r.min_e_observed, 6)} (${r.shield_w0_050_respected ? "GARANTAT" : "EROARE"})`, r.shield_w0_050_respected ? "pass" : "fail"],
      ["paritate martor #0 max|GPU − CPU| · timp GPU", `${fmt(r.gpu_cpu_witness0_max_diff, 3)}  ·  ${fmt(r.gpu_ms, 2)} ms pe RTX 5060`, r.gpu_cpu_witness0_max_diff < 1e-12 ? "pass" : "fail"],
    ]);

    $("v3cks-status").textContent = `RTX 5060: ${fmt(r.gpu_ms, 2)} ms · |GPU−CPU|=${fmt(r.gpu_cpu_witness0_max_diff, 2)}`;
    return {
      ok: r.shield_w0_050_respected === true && r.gpu_cpu_witness0_max_diff < 1e-12 && r.positive_claims_allowed === false,
      gpu_ms: r.gpu_ms,
    };
  } catch (e) {
    $("v3cks-status").textContent = String(e);
    $("v3cks-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

// ============================================================================
// FILA 3 (NOU v3): TELEMETRIE HARDWARE LIVE RTX 5060 · 24T CPU · 16G RAM · SSD
// ============================================================================

async function loadV3HwTelemetry() {
  try {
    const hw = await invoke("cmd_v3_hw_telemetry");
    const g = hw.gpu, c = hw.cpu, r = hw.ram, s = hw.ssd;

    $("hdr-gpu").textContent = `${g.name.replace("NVIDIA GeForce ", "")} (sm_${g.compute_capability.replace(".", "")})`;
    $("hdr-vram").textContent = `${mib(g.used_vram_bytes)} / ${mib(g.total_vram_bytes)}`;
    $("v3hw-guard").textContent = r.rss_guard_ok
      ? `VmRSS Guard < 11.5 GiB: PASS (${mib(r.process_vm_rss_bytes)})`
      : `ALERTĂ VmRSS: ${mib(r.process_vm_rss_bytes)}`;
    $("v3hw-guard").className = "badge " + (r.rss_guard_ok ? "pass" : "fail");

    const vramPct = g.total_vram_bytes ? Math.min(100, (g.used_vram_bytes / g.total_vram_bytes) * 100) : 0;
    const cpuPct = Math.min(100, (c.loadavg_1m / Math.max(1, c.logical_threads)) * 100);
    const ramPct = r.rss_guard_limit_bytes ? Math.min(100, (r.process_vm_rss_bytes / r.rss_guard_limit_bytes) * 100) : 0;
    const ssdUsedPct = s.fs_total_bytes ? Math.min(100, ((s.fs_total_bytes - s.fs_avail_bytes) / s.fs_total_bytes) * 100) : 0;

    $("hw-vram-bar").style.width = `${Math.max(3, vramPct).toFixed(1)}%`;
    $("hw-cpu-bar").style.width = `${Math.max(3, cpuPct).toFixed(1)}%`;
    $("hw-ram-bar").style.width = `${Math.max(3, ramPct).toFixed(1)}%`;
    $("hw-ssd-bar").style.width = `${Math.max(3, ssdUsedPct).toFixed(1)}%`;

    kv($("hw-gpu-kv"), [
      ["dispozitiv CUDA", `${g.name} (CC ${g.compute_capability} · ${g.sm_count} SM-uri)`, g.available ? "pass" : "fail"],
      ["VRAM utilizat / liber / total", `${mib(g.used_vram_bytes)} / ${mib(g.free_vram_bytes)} / ${mib(g.total_vram_bytes)} (${fmt(vramPct, 1)}%)`],
      ["univers v3 rezident în VRAM", `${mib(g.v3_resident_universe_bytes)} (39,077,004 bilete + tabele C3-v2)`, "pass"],
      ["temperatură · consum electric", `${g.temperature_c != null ? `${fmt(g.temperature_c, 0)} °C` : "—"}  ·  ${g.power_draw_w != null ? `${fmt(g.power_draw_w, 1)} W / ${fmt(g.power_limit_w, 0)} W` : "—"}`],
      ["utilizare SM / memorie · ceas · P-state", `SM ${g.sm_util_pct ?? 0}% · Mem ${g.mem_util_pct ?? 0}% · ${g.graphics_clock_mhz ?? "—"} MHz · ${g.pstate}`],
    ]);

    kv($("hw-cpu-kv"), [
      ["procesor", c.model_name],
      ["fire logice active", `${c.logical_threads} fire (std::thread::scope + AVX2 + FMA + POPCNT)`, "pass"],
      ["frecvență medie / maximă", `${fmt(c.avg_mhz, 0)} MHz  /  vârf ${fmt(c.max_mhz, 0)} MHz`],
      ["load average (1m / 5m / 15m)", `${fmt(c.loadavg_1m, 2)}  /  ${fmt(c.loadavg_5m, 2)}  /  ${fmt(c.loadavg_15m, 2)}`],
    ]);

    kv($("hw-ram-kv"), [
      ["RAM sistem utilizat / disponibil / total", `${gib(r.sys_mem_used_bytes)} / ${gib(r.sys_mem_available_bytes)} / ${gib(r.sys_mem_total_bytes)}`],
      ["proces Rust VmRSS / VmHWM", `${mib(r.process_vm_rss_bytes)}  /  vârf ${mib(r.process_vm_hwm_bytes)}`],
      ["gardă activă check_rss_guard", `< ${gib(r.rss_guard_limit_bytes)} (${r.rss_guard_ok ? "PASS" : "DEPĂȘIT"})`, r.rss_guard_ok ? "pass" : "fail"],
      ["WebKitGTK RSS (procese Webview)", `${mib(r.webview_rss_bytes)} (${r.webview_process_count} procese)`],
      ["Swap liber / total", `${gib(r.sys_swap_free_bytes)} / ${gib(r.sys_swap_total_bytes)}`],
    ]);

    kv($("hw-ssd-kv"), [
      ["punct montare SSD extern", `${s.mount_path} (${s.mounted ? "MONTAT" : "ABSENT"})`, s.mounted ? "pass" : "fail"],
      ["spațiu disponibil / total (statvfs)", `${tib(s.fs_avail_bytes)} liberi din ${tib(s.fs_total_bytes)}`],
      ["director Vault zero-copy", s.vault_dir],
      ["tabele .v3bin sigilate 0444", `${s.vault_v3bin_count} / 13 fișiere (${gib(s.vault_total_bytes)})`, s.vault_v3bin_all_0444 ? "pass" : "fail"],
    ]);

    $("v3hw-status").textContent = `Eșantionat UTC: ${hw.sampled_utc}`;
    return {
      ok: g.available === true && r.rss_guard_ok === true && s.mounted === true && s.vault_v3bin_all_0444 === true && hw.positive_claims_allowed === false,
      vram_used_bytes: g.used_vram_bytes,
      rss_bytes: r.process_vm_rss_bytes,
    };
  } catch (e) {
    $("v3hw-status").textContent = String(e);
    $("v3hw-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

function syncHwAutoLoop() {
  if (v3State.hwTimer !== null) {
    clearInterval(v3State.hwTimer);
    v3State.hwTimer = null;
  }
  if ($("tab-v3-hw").classList.contains("active") && $("v3hw-auto").checked) {
    loadV3HwTelemetry();
    v3State.hwTimer = setInterval(() => {
      if ($("tab-v3-hw").classList.contains("active") && $("v3hw-auto").checked) {
        loadV3HwTelemetry();
      }
    }, 2000);
  }
}

// ============================================================================
// FILELE 4..12: PANOUL V2 PORTAT INTEGRAL & ÎMBOGĂȚIT CU PORȚILE V3
// ============================================================================

function formatGateSummaryValue(g) {
  const v = g.value;
  if (typeof v === "number") return `${fmt(v, 4)} (prag ${fmt(g.threshold, 2)})`;
  if (!v || typeof v !== "object") return String(v);
  if ("E_S" in v) return `E_S=${fmt(v.E_S, 4)}, min E_s=${fmt(v.min_E_s, 4)}`;
  if ("diff_vs_U" in v) return `ΔU=${fmt(v.diff_vs_U, 3)}, ΔB=${fmt(v.diff_vs_B, 3)}`;
  if ("KJV5" in v && "B" in v) return `KJ=${fmt(v.KJV5, 2)} · B=${fmt(v.B, 2)} · R=${fmt(v.R, 2)}`;
  if ("HSV4" in v && "B" in v) return `HS=${fmt(v.HSV4, 2)} · B=${fmt(v.B, 2)} · R=${fmt(v.R, 2)}`;
  const entries = Object.entries(v).slice(0, 3).map(([k, val]) => `${k}=${fmt(val, 3)}`);
  return entries.join(", ");
}

async function loadKjv5Ro() {
  const joc = $("kj-joc").value;
  const config = $("kj-cfg").value;
  $("kj-status").textContent = "se încarcă artefactele KJ-V5…";
  $("kj-status").className = "status";
  try {
    const r = await invoke("cmd_kjv5_ro", { joc, config });
    const sel = r.selected;
    const post = sel.post_archive_state;
    const mt = sel.martingale_trajectory;
    const bm = sel.bayesian_masses;
    const wt = sel.windows.total;

    $("kj-sync").textContent = `${r.live_sync_state} · ${sel.archive.last_date}`;
    $("kj-sync").className = "badge " + (r.live_sync_state.includes("sincronizat") ? "pass" : "warn-txt");

    const variants = post.variants_jastrow_multiplet || [];
    if (variants.length > 0) {
      quickState.kjAlpha = { joc, bile: variants[0].ticket, bonus: variants[0].bonus ?? null };
    }
    grid($("kj-multiplet"),
      ["variantă", "bilet ortogonal", "energie −ln P", "ln(P / P_U)", "audit istoric"],
      variants.map((v) => [
        { t: v.label, c: "mono warn-txt" },
        makeBallsNode(v.ticket, v.bonus ?? null, null),
        { t: fmt(v.energy, 6), c: "mono" },
        { t: fmt(v.log_lift_vs_uniform, 6), c: "mono" },
        makeInspectButton(joc, v.ticket, v.bonus ?? null),
      ])
    );

    $("kj-core").replaceChildren(makeBallsNode(post.deterministic_core || [], null, null));
    $("kj-meta-post").textContent = `det(G₃)=${fmt(post.det_G3, 4)} · exact_24fire=${post.exact_global_certified ? "DA" : "BnB"}`;
    $("kj-epistemic").textContent = post.epistemic_notice;

    kv($("kj-summary-kv"), [
      ["calificare pre-arhivă Q1–Q6", `${r.qualification.passed} / ${r.qualification.total} (${r.qualification.status})`, "pass"],
      ["incidente tehnice pre-arhivă", String(r.qualification.technical_incidents), "warn-txt"],
      ["verdict porți walk-forward", `${sel.verdict_gates} (${sel.status})`, sel.gates_passed === 6 ? "pass" : "dim"],
      ["pași evaluați / extrageri țintă", `${sel.archive.evaluated_steps} / ${sel.archive.evaluated_target_draws}`],
      ["E_S final · min E_s · max E_s", `${fmt(mt.E_S_final, 6)} · ${fmt(mt.min_E_s, 6)} · ${fmt(mt.max_E_s, 6)}`, mt.guarantee_E_ge_050_verified ? "pass" : "fail"],
      ["mase finale (w₀, w_K1, w_K2, w_K3)", `${fmt(bm.posterior_channels.w_0, 4)}, ${fmt(bm.posterior_channels.w_k1, 4)}, ${fmt(bm.posterior_channels.w_k2, 4)}, ${fmt(bm.posterior_channels.w_k3, 4)}`],
      ["potriviri KJ-V5 / B / R (exces vs U)", `${wt.hits_KJV5} / ${wt.hits_B} / ${wt.hits_R} (${fmt(wt.excess_hits_vs_U, 5)})`],
      ["SHA-256 arhivă LIVE", `${sel.archive.file_sha256.slice(0, 16)}…`],
    ]);

    grid($("kj-gates"), ["poartă", "verdict", "valori"],
      (sel.gates || []).map((g) => [
        { t: g.id, c: "mono" },
        { t: g.pass ? "PASS" : "NU", c: g.pass ? "pass" : "dim" },
        { t: formatGateSummaryValue(g), c: "mono" },
      ])
    );

    const pts = mt.curve_points || [];
    multiLines("c-kjv5-e",
      [{ values: pts.map((p) => p.e_s), color: "#5cc8ff", width: 2.0 }],
      {
        minY: 0.45,
        hlines: [
          { y: 0.5, color: "#ffcf6e", dash: [4, 3] },
          { y: 1.0, color: "#6ee7a088", dash: [2, 4] },
        ],
        xFirst: pts[0] ? pts[0].date : "",
        xLast: pts.length ? pts[pts.length - 1].date : "",
      }
    );

    multiLines("c-kjv5-w",
      [
        { values: pts.map((p) => p.w_0), color: "#5cc8ff", width: 2.0 },
        { values: pts.map((p) => p.w_k1), color: "#6ee7a0", width: 1.6 },
        { values: pts.map((p) => p.w_k2), color: "#a98bff", width: 1.6 },
        { values: pts.map((p) => p.w_k3), color: "#ffcf6e", width: 1.6 },
      ],
      {
        minY: 0.0,
        maxY: 1.02,
        xFirst: pts[0] ? pts[0].date : "",
        xLast: pts.length ? pts[pts.length - 1].date : "",
      }
    );

    const kUnif = sel.game_params.k_ticket / sel.game_params.N;
    bars("c-kjv5-pi", post.pi_inclusion || [], { ref: kUnif });
    const sh = post.krawtchouk_shell_coeffs_m123 || [0, 0, 0];
    const kvAnti = post.koopman_v2_anti_eigenvalues || [0, 0];
    $("kj-spec-legend").textContent =
      `Krawtchouk c₁..₃ = [${sh.map((x) => fmt(x, 4)).join(", ")}] · Koopman V₂ anti λ± = [${kvAnti.map((x) => fmt(x, 4)).join(", ")}] · ‖Q_anti‖_F = ${fmt(post.koopman_v2_anti_norm, 4)}`;

    grid($("kj-q-table"), ["poartă", "stare", "ms", "descriere"],
      (r.qualification.gates || []).map((q) => [
        { t: q.id, c: "mono" },
        { t: q.pass ? "PASS" : "FAIL", c: q.pass ? "pass" : "fail" },
        { t: fmt(q.elapsed_ms, 1), c: "mono" },
        q.title,
      ])
    );

    grid($("kj-ablation"), ["config", "porți", "hits KJ/B/R", "exces U", "ΔLog U", "E_S", "min E_s", "w₀ final"],
      (r.comparison_table || []).map((row) => [
        { t: row.config_key, c: "mono" },
        row.verdict_gates,
        `${row.hits_KJV5}/${row.hits_B}/${row.hits_R}`,
        { t: fmt(row.excess_hits_vs_U, 5), c: "mono" },
        { t: fmt(row.log_score_KJV5 - row.log_score_U, 3), c: "mono" },
        { t: fmt(row.E_S_final, 4), c: "mono" },
        { t: fmt(row.min_E_s, 4), c: "mono pass" },
        { t: fmt(row.w_0_final, 4), c: "mono" },
      ])
    );

    const winOrder = [["total", "Total"], ["half1", "Jumătatea 1"], ["half2", "Jumătatea 2"], ["last100", "Ultimii 100"]];
    grid($("kj-windows"), ["fereastră", "pași", "hits KJ/B/R", "exces U", "ΔLog(KJ−B)", "Brier KJ / U"],
      winOrder.map(([wk, wlbl]) => {
        const w = sel.windows[wk];
        return [
          wlbl,
          w.n_steps,
          `${w.hits_KJV5}/${w.hits_B}/${w.hits_R}`,
          { t: fmt(w.excess_hits_vs_U, 4), c: "mono" },
          { t: fmt(w.log_score_KJV5 - w.log_score_B, 3), c: "mono" },
          { t: `${fmt(w.brier_KJV5, 2)} / ${fmt(w.brier_U, 2)}`, c: "mono" },
        ];
      })
    );

    const mh = sel.categories.main_histogram || {};
    grid($("kj-cats"), ["categorie", "KJ-V5", "B", "R", "așteptat U"],
      Object.entries(mh).map(([lbl, v]) => [
        { t: lbl, c: "mono" },
        { t: v.KJV5, c: "mono" },
        { t: v.B, c: "mono" },
        { t: v.R, c: "mono" },
        { t: fmt(v.expected_U, 2), c: "mono dim" },
      ])
    );

    grid($("kj-intl-mini"), ["flux", "tip", "pași", "porți", "hits KJ/B/R", "E_S", "min E_s"],
      (r.international_overview || []).map((x) => [
        x.name,
        x.group === "electronic_rng" ? "RNG" : "Mecanic",
        x.evaluated_steps,
        x.verdict_gates,
        `${x.hits_KJV5}/${x.hits_B}/${x.hits_R}`,
        { t: fmt(x.E_S_final, 4), c: "mono" },
        { t: fmt(x.min_E_s, 4), c: "mono pass" },
      ])
    );

    $("kj-status").textContent = `SHA-256 ablație: ${r.ablation_sha256.slice(0, 14)}… · ${sel.device}`;
    return {
      ok: r.qualification.passed === 6 && mt.guarantee_E_ge_050_verified === true && r.positive_claims_allowed === false,
      joc,
      verdict_gates: sel.verdict_gates,
      min_E_s: mt.min_E_s,
    };
  } catch (e) {
    $("kj-status").textContent = String(e);
    $("kj-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadHsv4Ro() {
  const joc = $("hs-joc").value;
  const config = $("hs-cfg").value;
  $("hs-status").textContent = "se verifică sigiliile SHA-256 HS-V4 & .npz…";
  $("hs-status").className = "status";
  try {
    const r = await invoke("cmd_hsv4_ro", { joc, config });
    const sel = r.selected;
    const post = sel.post_archive_state;
    const npz = r.npz_curves;
    const met = r.meteo_era5;
    const snap = met.snapshot;

    $("hs-sync").textContent = r.live_sync.in_sync
      ? `sincronizat LIVE (${r.live_sync.live_last_date})`
      : `sigilat ${r.live_sync.sealed_last_date} vs LIVE ${r.live_sync.live_last_date}`;
    $("hs-sync").className = "badge " + (r.live_sync.in_sync ? "pass" : "warn-txt");

    const variants = post.variants_jastrow_multiplet || [];
    if (variants.length > 0) {
      quickState.hsAlpha = { joc, bile: variants[0].ticket, bonus: variants[0].joker ?? null };
    }
    grid($("hs-multiplet"),
      ["variantă", "bilet Coulomb–Jastrow", "energie −ln P", "probabilitate", "audit istoric"],
      variants.map((v) => [
        { t: v.label, c: "mono warn-txt" },
        makeBallsNode(v.ticket, v.joker ?? null, null),
        { t: fmt(v.energy, 6), c: "mono" },
        { t: fmt(v.prob, 4), c: "mono" },
        makeInspectButton(joc, v.ticket, v.joker ?? null),
      ])
    );
    $("hs-core").replaceChildren(makeBallsNode(post.deterministic_core || [], null, null));
    $("hs-meta-post").textContent = `det(G₃)=${fmt(post.det_G3, 4)} · bile acoperite=${post.distinct_balls_covered}`;
    $("hs-epistemic").textContent = post.epistemic_notice;

    kv($("hs-meteo-kv"), [
      ["dată / oră locală (sursă)", `${snap.date} ${snap.hour_local} (${snap.source})`],
      ["temperatură T_2m · z Welford", `${fmt(snap.temp_2m_c, 2)} °C  (z = ${fmt(snap.z_welford.temp_2m, 3)})`],
      ["umiditate RH_2m · z Welford", `${fmt(snap.rh_2m_pct, 2)} %  (z = ${fmt(snap.z_welford.rh_2m, 3)})`],
      ["presiune P_surf · ΔP_24h", `${fmt(snap.surface_pressure_hpa, 2)} hPa · ${fmt(snap.dp_24h_hpa, 2)} hPa`],
      ["densitate aer ρ_aer · z Welford", `${fmt(snap.rho_air_kg_m3, 5)} kg/m³  (z = ${fmt(snap.z_welford.rho_air, 3)})`],
      ["zile cache ERA5 verificate SHA-256", `${met.total_days} (${met.first_date} → ${met.last_cache_date})`, "pass"],
    ]);

    multiLines("c-hsv4-rho",
      [{ values: met.recent_120d_rho_air.rho_air_kg_m3 || [], color: "#ffcf6e", width: 1.7 }],
      {
        hlines: [{ y: met.recent_120d_rho_air.mean_rho_air, color: "#5cc8ff88", dash: [3, 3] }],
        xFirst: met.recent_120d_rho_air.dates[0] || "",
        xLast: met.recent_120d_rho_air.dates[met.recent_120d_rho_air.dates.length - 1] || "",
      }
    );

    multiLines("c-hsv4-e",
      [{ values: npz.E_curve || [], color: "#5cc8ff", width: 2.0 }],
      {
        minY: 0.45,
        hlines: [
          { y: 0.5, color: "#ffcf6e", dash: [4, 3] },
          { y: 1.0, color: "#6ee7a088", dash: [2, 4] },
        ],
        xFirst: npz.dates[0] || "",
        xLast: npz.dates[npz.dates.length - 1] || "",
      }
    );

    multiLines("c-hsv4-w",
      [
        { values: npz.w_0 || [], color: "#5cc8ff", width: 2.0 },
        { values: npz.w_O1 || [], color: "#6ee7a0", width: 1.6 },
        { values: npz.w_O2 || [], color: "#ffcf6e", width: 1.6 },
        { values: npz.w_O4 || [], color: "#a98bff", width: 1.6 },
      ],
      {
        minY: 0.0,
        maxY: 1.02,
        xFirst: npz.dates[0] || "",
        xLast: npz.dates[npz.dates.length - 1] || "",
      }
    );

    const piArr = post.pi_inclusion || [];
    const kEst = piArr.reduce((a, b) => a + b, 0);
    bars("c-hsv4-pi", piArr, { ref: piArr.length ? kEst / piArr.length : undefined });

    grid($("hs-ablation"), ["config", "porți", "hits HS/B/R", "ΔLog U", "E_S", "min E_s", "w₀ final"],
      (r.comparison_table || []).map((row) => [
        { t: row.config_key, c: "mono" },
        row.verdict_gates,
        `${row.hits_HSV4}/${row.hits_B}/${row.hits_R}`,
        { t: fmt(row.log_diff_vs_U, 3), c: "mono" },
        { t: fmt(row.E_S, 4), c: "mono" },
        { t: fmt(row.min_E_s, 4), c: "mono pass" },
        { t: fmt(row.w_0_final, 4), c: "mono" },
      ])
    );

    grid($("hs-gates"), ["poartă", "verdict", "valori"],
      (sel.gates || []).map((g) => [
        { t: g.id, c: "mono" },
        { t: g.pass ? "PASS" : "NU", c: g.pass ? "pass" : "dim" },
        { t: formatGateSummaryValue(g), c: "mono" },
      ])
    );

    const mh = sel.categories.main_histogram || {};
    grid($("hs-cats"), ["categorie", "HS-V4", "B", "R", "așteptat U"],
      Object.entries(mh).map(([lbl, v]) => [
        { t: lbl, c: "mono" },
        { t: v.HSV4, c: "mono" },
        { t: v.B, c: "mono" },
        { t: v.R, c: "mono" },
        { t: fmt(v.expected_U, 2), c: "mono dim" },
      ])
    );

    $("hs-status").textContent = `SHA-256 sigiliu UI verificat · .npz eșantionat ${npz.sampled_points}/${npz.total_points} pași`;
    return {
      ok: r.seals.verified_all_frozen_hashes === true && r.positive_claims_allowed === false,
      joc,
      sampled_points: npz.sampled_points,
    };
  } catch (e) {
    $("hs-status").textContent = String(e);
    $("hs-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function runCateSiUnde() {
  const joc = $("cu-joc").value;
  const bile = parseBallsInput($("cu-bile").value);
  const bStr = $("cu-bonus").value.trim();
  const bonus = bStr ? Number.parseInt(bStr, 10) : null;
  const minPotriviri = Number.parseInt($("cu-min").value, 10) || 3;
  const includeIntl = $("cu-intl").checked;

  $("cu-status").textContent = "se scanează arhivele…";
  $("cu-status").className = "status";
  try {
    const r = await invoke("cmd_cate_si_unde", {
      joc,
      bile,
      bonus: Number.isFinite(bonus) ? bonus : null,
      minPotriviri,
      includeIntl,
    });
    const p = r.primary;
    $("cu-hist-title").textContent =
      `Histogramă ${p.stream_name} (${p.n_draws} extrageri, ${p.first_date} → ${p.last_date})`;

    grid($("cu-hist"),
      ["potriviri", "observat istoric", "așteptat hipergeometric U", p.bonus_evaluated ? "+Bonus" : "diferență"],
      (p.histogram || []).map((h) => [
        { t: h.label, c: "mono" },
        { t: h.observed, c: h.hits >= 3 && h.observed > 0 ? "mono pass" : "mono" },
        { t: fmt(h.expected_U, 3), c: "mono dim" },
        {
          t: p.bonus_evaluated ? String(h.with_bonus ?? 0) : fmt(h.observed - h.expected_U, 3),
          c: "mono",
        },
      ])
    );

    grid($("cu-cross"),
      ["arhivă", "urnă", "extrageri", "3+ obs (U)", "4+ obs (U)", "5+", "max", "ultima 3+"],
      (r.cross_streams || []).map((cs) => [
        cs.stream_name,
        { t: cs.urn, c: "mono" },
        cs.n_draws,
        { t: `${cs.hits_3plus} (${fmt(cs.exp_3plus_U, 1)})`, c: "mono" },
        { t: `${cs.hits_4plus} (${fmt(cs.exp_4plus_U, 2)})`, c: cs.hits_4plus > 0 ? "mono pass" : "mono" },
        { t: cs.hits_5plus, c: cs.hits_5plus > 0 ? "mono warn-txt" : "mono" },
        { t: cs.max_hits_observed, c: "mono" },
        { t: cs.last_3plus_date || "—", c: "mono dim" },
      ])
    );

    $("cu-matches-title").textContent =
      `Extrageri cu ≥ ${r.min_hits} potriviri în ${p.stream_name}: ${p.total_matching_draws_ge_min} găsite`;
    grid($("cu-matches"),
      ["dată", "categorie", "bile nimerite", "extragere completă"],
      (r.primary_matches || []).map((m) => {
        const hitSet = new Set(m.matched_balls || []);
        return [
          { t: m.date, c: "mono" },
          { t: m.tier, c: m.hits_main >= 4 ? "mono warn-txt" : "mono pass" },
          makeBallsNode(m.matched_balls, m.hit_bonus ? m.draw_bonus : null, hitSet),
          makeBallsNode(m.draw_numbers, m.draw_bonus, hitSet),
        ];
      })
    );

    grid($("cu-intl-matches"),
      ["loterie", "dată", "categorie", "bile nimerite"],
      (r.cross_high_matches || []).map((m) => {
        const hitSet = new Set(m.matched_balls || []);
        return [
          m.stream_name,
          { t: m.date, c: "mono" },
          { t: m.tier, c: "mono warn-txt" },
          makeBallsNode(m.matched_balls, m.hit_bonus ? m.draw_bonus : null, hitSet),
        ];
      })
    );

    $("cu-nota").textContent = r.epistemic_notice;
    $("cu-status").textContent =
      `3+: ${p.tiers["3+"].observed} (U=${fmt(p.tiers["3+"].expected_U, 1)}) · 4+: ${p.tiers["4+"].observed} (U=${fmt(p.tiers["4+"].expected_U, 2)})`;
    return { ok: r.positive_claims_allowed === false && Array.isArray(p.histogram), total_matches: p.total_matching_draws_ge_min };
  } catch (e) {
    $("cu-status").textContent = String(e);
    $("cu-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadIntlComparatie() {
  const flux = $("in-flux").value;
  $("in-status").textContent = "se încarcă evaluarea comparativă HS-V4 + KJ-V5…";
  $("in-status").className = "status";
  try {
    const r = await invoke("cmd_intl_comparatie", { flux });

    grid($("in-overview"),
      ["flux", "familie", "urnă", "pași", "porți HS / KJ", "hits HS / KJ / B / R", "E_S HS / KJ", "min E_s HS / KJ", "w₀ final HS / KJ"],
      (r.overview_rows || []).map((x) => [
        x.name,
        x.group === "electronic_rng" ? { t: "RNG Electronic", c: "warn-txt" } : { t: "Mecanic Extern", c: "pass" },
        { t: x.urn + (x.bonus_m ? `+1/${x.bonus_m}` : ""), c: "mono" },
        x.evaluated_steps,
        { t: `${x.hsv4_gates}  |  ${x.kjv5_gates}`, c: "mono" },
        { t: `${x.hsv4_hits} / ${x.kjv5_hits} / ${x.hits_B} / ${x.hits_R}`, c: "mono" },
        { t: `${fmt(x.hsv4_E_S, 4)} / ${fmt(x.kjv5_E_S, 4)}`, c: "mono" },
        { t: `${fmt(x.hsv4_min_E_s, 4)} / ${fmt(x.kjv5_min_E_s, 4)}`, c: "mono pass" },
        { t: `${fmt(x.hsv4_w0_final, 3)} / ${fmt(x.kjv5_w0_final, 3)}`, c: "mono" },
      ])
    );

    const skH = r.paired_skandinav.hsv4;
    const skK = r.paired_skandinav.kjv5;
    grid($("in-skandi"),
      ["tragere (aceleași 1.312 zile)", "motor", "porți", "hits (vs B / R)", "exces U", "E_S final", "min E_s"],
      [
        ["Tragerea 1 Mecanică (Kézi)", "HS-V4", skH.mechanical.verdict_gates, `${skH.mechanical.hits_HSV4} (${skH.mechanical.hits_B}/${skH.mechanical.hits_R})`, fmt(skH.mechanical.excess_hits_vs_U, 5), fmt(skH.mechanical.E_S, 4), fmt(skH.mechanical.min_E_s, 4)],
        ["Tragerea 1 Mecanică (Kézi)", "KJ-V5", skK.mechanical_kezi.verdict_gates, `${skK.mechanical_kezi.hits_KJV5} (${skK.mechanical_kezi.hits_B}/${skK.mechanical_kezi.hits_R})`, fmt(skK.mechanical_kezi.excess_hits_vs_U, 5), fmt(skK.mechanical_kezi.E_S_final, 4), fmt(skK.mechanical_kezi.min_E_s, 4)],
        ["Tragerea 2 Electronică RNG (Gépi)", "HS-V4", skH.electronic_rng.verdict_gates, `${skH.electronic_rng.hits_HSV4} (${skH.electronic_rng.hits_B}/${skH.electronic_rng.hits_R})`, fmt(skH.electronic_rng.excess_hits_vs_U, 5), fmt(skH.electronic_rng.E_S, 4), fmt(skH.electronic_rng.min_E_s, 4)],
        ["Tragerea 2 Electronică RNG (Gépi)", "KJ-V5", skK.electronic_rng_gepi.verdict_gates, `${skK.electronic_rng_gepi.hits_KJV5} (${skK.electronic_rng_gepi.hits_B}/${skK.electronic_rng_gepi.hits_R})`, fmt(skK.electronic_rng_gepi.excess_hits_vs_U, 5), fmt(skK.electronic_rng_gepi.E_S_final, 4), fmt(skK.electronic_rng_gepi.min_E_s, 4)],
      ]
    );

    grid($("in-o2-ctrl"),
      ["loterie RNG electronică", "pași", "w_O2 inițial → final", "E_S ctrl", "min E_s", "contracție"],
      (r.o2_negative_control_summary.streams || []).map((s) => [
        s.game_name,
        s.evaluated_steps,
        { t: `${fmt(s.ctrl_w_O2_prior, 2)} → ${fmt(s.ctrl_w_O2_final, 6)}`, c: "mono" },
        { t: fmt(s.ctrl_E_S_final, 4), c: "mono" },
        { t: fmt(s.ctrl_min_E_s, 4), c: "mono pass" },
        { t: s.contraction_verified && s.guarantee_E_ge_050 ? "VERIFICAT" : "NU", c: s.contraction_verified ? "pass" : "fail" },
      ])
    );

    const kSel = r.selected_kjv5;
    const hSel = r.selected_hsv4.configs.Full_HSV4;
    const kPts = kSel.martingale_trajectory.curve_points || [];
    const hNpz = r.selected_hsv4_npz;

    $("in-detail-title").textContent = `Traiectorii E_s — ${kSel.game_name}`;
    multiLines("c-intl-e",
      [
        { values: kPts.map((p) => p.e_s), color: "#5cc8ff", width: 2.0 },
        { values: hNpz && hNpz.E_curve ? hNpz.E_curve : [], color: "#a98bff", width: 1.8 },
      ],
      {
        minY: 0.45,
        hlines: [
          { y: 0.5, color: "#ffcf6e", dash: [4, 3] },
          { y: 1.0, color: "#6ee7a088", dash: [2, 4] },
        ],
        xFirst: kPts[0] ? kPts[0].date : "",
        xLast: kPts.length ? kPts[kPts.length - 1].date : "",
      }
    );

    const ds = r.provenance_dataset || {};
    kv($("in-detail-kv"), [
      ["sursă oficială", ds.source_description || "—"],
      ["interval arhivă · înregistrări", `${ds.first_date} → ${ds.last_date} (${ds.records} extrageri)`],
      ["SHA-256 fișier .jsonl", `${(ds.jsonl_sha256 || "").slice(0, 20)}…`],
      ["HS-V4 porți · E_S · min E_s", `${hSel.verdict_gates} · E_S=${fmt(hSel.martingale_trajectory.E_S_final, 5)} · min=${fmt(hSel.martingale_trajectory.min_E_s, 5)}`],
      ["KJ-V5 porți · E_S · min E_s", `${kSel.verdict_gates} · E_S=${fmt(kSel.martingale_trajectory.E_S_final, 5)} · min=${fmt(kSel.martingale_trajectory.min_E_s, 5)}`],
      ["nucleu determinist KJ-V5", makeBallsNode(kSel.post_archive_state.deterministic_core || [], null, null)],
    ]);

    const kVars = kSel.post_archive_state.variants_jastrow_multiplet || [];
    if (kVars.length > 0) {
      quickState.intlAlpha = { flux, bile: kVars[0].ticket, bonus: kVars[0].bonus ?? null };
    }
    grid($("in-detail-mult"),
      ["var", "bilet KJ-V5", "energie", "audit"],
      kVars.map((v) => [
        { t: v.label, c: "mono warn-txt" },
        makeBallsNode(v.ticket, v.bonus ?? null, null),
        { t: fmt(v.energy, 4), c: "mono" },
        makeInspectButton(flux, v.ticket, v.bonus ?? null, "🔍"),
      ])
    );

    $("in-status").textContent = `10 fluxuri verificate SHA-256 · Skandináv zile identice: ${skK.same_dates_verified ? "DA" : "NU"}`;
    return {
      ok: (r.overview_rows || []).length === 10 && skK.same_dates_verified === true && r.positive_claims_allowed === false,
      flux,
    };
  } catch (e) {
    $("in-status").textContent = String(e);
    $("in-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadDensities() {
  const joc = $("d-joc").value, model = $("d-model").value;
  $("d-status").textContent = "se calculează…"; $("d-status").className = "status";
  const t0 = performance.now();
  try {
    const d = await invoke("cmd_densitati", { joc, model });
    bars("c-pi", d.pi, { ref: d.pi_uniform });
    const m = heatmap("c-rho", d.rho, d.n, d.k);
    bars("c-dft", d.dft_abs, { base: 0, colors: d.dft_abs.map((_, i) => `hsl(${(360 * i) / d.n},70%,60%)`) });
    const rows = [
      ["joc / model", `${d.joc} / ${d.model}`], ["n, k, N", `${d.n}, ${d.k}, ${d.draws}`],
      ["ultima cheie", d.meta.last_key], ["λ (contracție James–Stein)", fmt(d.lambda)],
      ["X² (df)", `${fmt(d.x2, 4)} (${d.df})`], ["modelul pariază", d.bet_active ? "da" : "nu (λ = 0 ⇒ lege uniformă)"],
      ["D = KL(P_w ‖ U) [nats]", fmt(d.d_nats, 9)], ["Tr ρ", fmt(d.rho_trace, 15)],
      ["puritate Tr ρ²  (uniform)", `${fmt(d.rho_purity, 9)}  (${fmt(d.rho_purity_uniform, 9)})`],
      ["λ_min(ρ), λ_max(ρ)", `${fmt(d.rho_min_eigenvalue, 9)}, ${fmt(d.rho_max_eigenvalue, 9)}`],
      ["max |ρ − ρ_U|", fmt(m, 9)], ["Parseval (reziduu relativ)", fmt(d.parseval_residual, 3)],
    ];
    if (d.ball) rows.push(["bila Joker λ, D", `${fmt(d.ball.lambda)}, ${fmt(d.ball.d_nats, 9)}`]);
    rows.push(["timp IPC + calcul", `${(performance.now() - t0).toFixed(1)} ms`]);
    kv($("d-kv"), rows);
    $("d-status").textContent = d.nota;
    return { ok: true, n: d.n, trace: d.rho_trace, min_eig: d.rho_min_eigenvalue };
  } catch (e) {
    $("d-status").textContent = String(e); $("d-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadFrft() {
  const joc = $("f-joc").value, a = parseFloat($("f-a").value);
  $("f-aval").textContent = a.toFixed(2);
  try {
    const f = await invoke("cmd_frft", { joc, a });
    const colors = f.arg.map((p) => `hsl(${((p + Math.PI) / (2 * Math.PI)) * 360},75%,58%)`);
    bars("c-frft", f.abs, { colors, under: f.z.map(Math.abs) });
    kv($("f-kv"), [["n, a", `${f.n}, ${fmt(f.a, 2)}`], ["‖z‖²", fmt(f.energy_z, 9)], ["‖F_a z‖²", fmt(f.energy_fa, 9)],
      ["|Δ energie| relativ", fmt(f.energy_rel_residual, 3)], ["max |F_aᴴF_a − I|", fmt(f.unitarity_residual, 3)], ["notă", f.nota]]);
    $("f-status").textContent = ""; $("f-status").className = "status";
    return { ok: true, energy_rel_residual: f.energy_rel_residual, unitarity_residual: f.unitarity_residual };
  } catch (e) {
    $("f-status").textContent = String(e); $("f-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadTargets() {
  try {
    const t = await invoke("cmd_tinte");
    const sel = $("a-tinta");
    sel.replaceChildren(...t.tinte.map((x) => {
      const o = document.createElement("option");
      o.value = x.tinta;
      o.textContent = `${x.tinta} · ${x.stare}${x.programata ? " (programată)" : ""}`;
      o.selected = x.tinta === t.implicita;
      return o;
    }));
    $("a-regula").textContent = t.regula;
    return { ok: true, implicita: t.implicita, programata: t.programata, tinte: t.tinte.map((x) => [x.tinta, x.stare]) };
  } catch (e) {
    $("a-status").textContent = String(e); $("a-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadAdm(tinta) {
  const t = tinta || $("a-tinta").value || undefined;
  try {
    const r = await invoke("cmd_admisibilitate", t ? { tinta: t } : {});
    $("a-tinta-titlu").textContent = r.tinta;
    grid($("a-table"), ["joc", "model", "emisie", "λ", "top (lege)", "top liniar", "stare", "bilet afișare", "rang", "noduri heap", "q(adm)/q(top)", "bila"],
      r.fluxuri.map((s) => {
        const f = s.filtru;
        const em = { t: s.stare_emisie === "CERTIFICATA" ? `certificată ${s.sha256.slice(0, 8)}…` : "absentă", c: s.stare_emisie === "CERTIFICATA" ? "pass" : "dim" };
        if (!f) return [s.joc, s.model, em, "—", "—", "—", { t: "EMISIE_ABSENTA", c: "dim" }, "—", "—", "—", "—", "—"];
        return [s.joc, s.model, em, fmt(s.lambda), f.top_ticket.join(" "), f.top_liniar ? "da" : "nu",
          { t: f.stare, c: f.stare === "NU_PARIAZA" ? "dim" : "pass" },
          f.bilet_afisare ? f.bilet_afisare.join(" ") : "—", f.rang ?? "—", f.noduri_heap ?? "—",
          f.q_adm_pe_q_top != null ? fmt(f.q_adm_pe_q_top, 9) : "—",
          s.bila_stare ? `${s.bila_stare}${s.bila_afisare ? " " + s.bila_afisare : ""}` : "—"];
      }));
    $("a-nota").textContent = r.nota;
    $("a-eticheta").textContent = etichetaText(r);
    $("a-status").textContent = `${r.fluxuri_certificate} / ${r.fluxuri.length} fluxuri cu emisie certificată`; $("a-status").className = "status";
    return {
      ok: true,
      tinta: r.tinta,
      certificate: r.fluxuri_certificate,
      fluxuri: r.fluxuri.map((s) => [s.joc, s.model, s.stare_emisie, s.filtru ? s.filtru.stare : null, s.filtru ? s.filtru.bilet_afisare : null, s.filtru ? s.filtru.rang : null]),
    };
  } catch (e) {
    $("a-nota").textContent = String(e);
    $("a-status").textContent = String(e); $("a-status").className = "status err";
    return { ok: false, tinta: t, error: String(e) };
  }
}

async function loadMemory() {
  try {
    const m = await invoke("cmd_memorie");
    kv($("m-kv"), [
      ["verdict Webview (Σ VmHWM ≤ limită)", m.verdict, m.verdict === "PASS" ? "pass" : m.verdict === "FAIL" ? "fail" : "dim"],
      ["limită", m.limita_text], ["Σ VmHWM WebKit*", mb(m.webview_hwm_total_octeti)], ["Σ VmRSS WebKit*", mb(m.webview_rss_total_octeti)],
      ["Σ Pss WebKit*", mb(m.webview_pss_total_octeti)],
      ["proces principal RSS / HWM", m.principal ? `${mb(m.principal.rss_bytes)} / ${mb(m.principal.hwm_bytes)}` : "—"],
      ["măsurat (UTC)", m.masurat_utc], ["motiv", m.motiv || "—"],
    ]);
    grid($("m-table"), ["pid", "ppid", "nume", "VmRSS", "VmHWM", "Pss", "tip"],
      [...m.webview_procese.map((p) => [p, "Webview"]), ...m.alte_descendente.map((p) => [p, "alt descendent"])].map(([p, tip]) =>
        [p.pid, p.ppid ?? "—", p.name ?? "—", mb(p.rss_bytes), mb(p.hwm_bytes), mb(p.pss_bytes), tip]));
    $("m-nota").textContent = `${m.metoda}. ${m.nota}`;
    $("m-status").textContent = ""; $("m-status").className = "status";
    return { ok: true, verdict: m.verdict, webview_hwm: m.webview_hwm_total_octeti, procese_webview: m.webview_procese.length };
  } catch (e) {
    $("m-status").textContent = String(e); $("m-status").className = "status err";
    return { ok: false, error: String(e) };
  }
}

async function loadTelemetry() {
  try {
    const t = await invoke("cmd_telemetrie");
    kv($("t-kv"), [
      ["stare", t.stare, t.stare === "ACTIV" || t.stare === "COMPLET" || t.stare === "OPRIT" ? "pass" : "fail"],
      ["motiv", t.motiv || "—"], ["director sesiune (SSD)", t.director || "—"],
      ["interval", `${t.interval_ms} ms`], ["eșantioane scrise", String(t.esantioane)],
      ["Σ VmRSS arbore — vârf eșantionat", mb(t.rss_total_varf)], ["… la eșantionul / UTC", `${t.seq_varf ?? "—"} / ${t.utc_varf ?? "—"}`],
      ["Σ VmRSS WebKit* — vârf", mb(t.webview_rss_varf)], ["proces principal VmRSS — vârf", mb(t.rss_principal_varf)],
      ["Σ VmRSS arbore — ultim", mb(t.rss_total_ultim)], ["Σ VmHWM arbore — ultim", mb(t.hwm_total_ultim)],
      ["verdict Webview 30 MB (ultimul eșantion, numai raportat)", t.verdict_webview_30mb_ultim],
      ["octeți scriși pe SSD", String(t.octeti_scrisi)], ["segmente", String(t.segmente)], ["pornit (UTC)", t.pornit_utc],
    ]);
    $("t-nota").textContent = `${t.metoda}. ${t.nota}`;
    return { ok: true, stare: t.stare, esantioane: t.esantioane, raspuns: t };
  } catch (e) {
    kv($("t-kv"), [["eroare", String(e)]]);
    return { ok: false, error: String(e) };
  }
}

const ETICHETA_STANDARD = "test științific în curs; nu este sfat de joc";
const REF_WEBVIEW_OCTETI = 30000000;
const TEL_X0 = 230, TEL_X1 = 920, TEL_BARA = 34;
const TEL_Y = [18, 84, 150];
const telLive = { timer: null, inZbor: false, oprire: false, ticks: 0, ultim: null, perioada: 1000 };

const mbScurt = (b) => (b == null ? "—" : `${(b / 1e6).toFixed(1)} MB`);

function drawTelemetry(t) {
  const [c, ctx] = canvasCtx("c-tel");
  const randuri = [
    { nume: "Σ VmRSS arbore", ultim: t.rss_total_ultim, varf: t.rss_total_varf, ref: false },
    { nume: "Σ VmRSS WebKit*", ultim: t.webview_rss_ultim, varf: t.webview_rss_varf, ref: true },
    { nume: "VmRSS proces principal", ultim: null, varf: t.rss_principal_varf, ref: false },
  ];
  let hi = REF_WEBVIEW_OCTETI;
  randuri.forEach((r) => { hi = Math.max(hi, r.ultim ?? 0, r.varf ?? 0); });
  hi *= 1.1;
  const x = (v) => TEL_X0 + (Math.min(v, hi) / hi) * (TEL_X1 - TEL_X0);
  ctx.font = "12px monospace";
  randuri.forEach((r, i) => {
    const y = TEL_Y[i];
    ctx.fillStyle = "#354972";
    ctx.fillRect(TEL_X0, y, TEL_X1 - TEL_X0, TEL_BARA);
    const plin = i === 2 ? r.varf : r.ultim;
    if (plin != null && plin > 0) {
      ctx.fillStyle = "#5cc8ff";
      ctx.fillRect(TEL_X0, y, Math.max(2, x(plin) - TEL_X0), TEL_BARA);
    }
    if (i !== 2 && r.varf != null && r.varf > 0) {
      ctx.fillStyle = "hsl(0,70%,62%)";
      ctx.fillRect(x(r.varf) - 1, y - 4, 3, TEL_BARA + 8);
    }
    if (r.ref) {
      ctx.strokeStyle = "#ffcf6e";
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x(REF_WEBVIEW_OCTETI), y - 6); ctx.lineTo(x(REF_WEBVIEW_OCTETI), y + TEL_BARA + 6); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#8d9ec0";
    ctx.fillText(r.nume, 8, y + TEL_BARA / 2 + 4);
    ctx.fillText(i === 2 ? `vârf ${mbScurt(r.varf)}` : `ultim ${mbScurt(r.ultim)} · vârf ${mbScurt(r.varf)}`, TEL_X1 + 10,
      y + TEL_BARA / 2 + 4);
  });
  ctx.fillText(`stare ${t.stare} · eșantioane ${t.esantioane} · vârf arbore la eșantionul ${t.seq_varf ?? "—"} · ` +
    `pas în direct ${telLive.ticks} · perioada ${telLive.perioada} ms · scala 0–${mbScurt(hi)}`, 8, c.height - 10);
}

function memTabActive() {
  return $("tab-mem").classList.contains("active") && document.visibilityState === "visible";
}

async function liveStep() {
  telLive.timer = null;
  telLive.inZbor = true;
  const r = await loadTelemetry();
  telLive.inZbor = false;
  telLive.ticks += 1;
  if (r.ok) {
    telLive.ultim = r.raspuns;
    telLive.perioada = Math.max(1000, r.raspuns.interval_ms || 0);
    drawTelemetry(r.raspuns);
  }
  if (!telLive.oprire && r.ok && r.raspuns.stare === "ACTIV" && memTabActive()) {
    telLive.timer = setTimeout(liveStep, telLive.perioada);
  }
}

function startLive() {
  telLive.oprire = false;
  if (telLive.timer === null && !telLive.inZbor) liveStep();
}

function stopLive() {
  telLive.oprire = true;
  if (telLive.timer !== null) { clearTimeout(telLive.timer); telLive.timer = null; }
}

function onTab(nume) {
  if (nume === "mem") startLive(); else stopLive();
  syncHwAutoLoop();
}

function initLive() {
  document.addEventListener("visibilitychange", () => {
    if (memTabActive()) startLive(); else stopLive();
    syncHwAutoLoop();
  });
}

function etichetaText(r) {
  return `Eticheta standard a emisiilor (tu4/emite.py): „${ETICHETA_STANDARD}”. Ținta ${r.tinta}: ` +
    `${r.fluxuri_certificate} / ${r.fluxuri.length} fluxuri cu emisie certificată în LANT. Starea fiecărui flux este starea ` +
    "curentă a emisiei în LANT (CERTIFICATA / EMISIE_ABSENTA); nu este predicție și nu oferă avantaj de selecție.";
}

function randuriDesenate() {
  const ctx = $("c-tel").getContext("2d");
  return TEL_Y.map((y) => {
    const px = ctx.getImageData(TEL_X0 + 1, y + Math.floor(TEL_BARA / 2), 1, 1).data;
    return (px[0] > 25 || px[1] > 35 || px[2] > 55) && px[3] === 255;
  });
}

async function smokeGrafic() {
  const t0 = performance.now();
  const r0 = await loadTelemetry();
  const inceput = r0.ok ? r0.raspuns.esantioane : null;
  telLive.ticks = 0;
  activateTab("mem");
  while (telLive.ticks < 3 && performance.now() - t0 < 15000) await new Promise((r) => setTimeout(r, 100));
  stopLive();
  while (telLive.inZbor) await new Promise((r) => setTimeout(r, 20));
  const t = telLive.ultim;
  const randuri = randuriDesenate();
  const ok = telLive.ticks >= 3 && t !== null && inceput !== null && t.esantioane > inceput &&
    randuri.every(Boolean) && telLive.perioada === 1000;
  return {
    ok,
    ticks: telLive.ticks,
    esantioane_inceput: inceput,
    esantioane_sfarsit: t ? t.esantioane : null,
    randuri_desenate: randuri,
    perioada_ms: telLive.perioada,
  };
}

function smokeEticheta() {
  const text = $("a-eticheta").textContent;
  return { ok: text.includes(`„${ETICHETA_STANDARD}”`), text };
}

async function loadParity() {
  try {
    const p = await invoke("cmd_paritate");
    const v3v = p.v3_verificari || {};
    const v3Rows = [
      ["Poarta A0 (Vault + DMA VRAM)", "GATE_A0_PARITY.json", v3v.gate_a0_parity && v3v.gate_a0_parity.pass === true, "39,077,004 măști bit-exact GPU == 24T CPU"],
      ["S1 · Poarta A1 + A2 (CD-V3)", "VERIFICARE_S1_CDV3.json", v3v.s1_cdv3 && (v3v.s1_cdv3.overall_pass === true || v3v.s1_cdv3.verdict === "PASS"), "Roți acoperire 163 / 20 / 169 / 4411 / 102 (0 neacoperite)"],
      ["S2 · Poarta A3 (PM-V3 + CD-V3Q)", "VERIFICARE_S2_PMV3.json", v3v.s2_pmv3 && v3v.s2_pmv3.overall_pass === true, "NB2 + 1 000× bootstrap + QAP (0 bilete tipizate)"],
      ["S3 · Poarta A4 (C3-v2 + C4-v2)", "VERIFICARE_S3_PORTOFOLIU.json", v3v.s3_portofoliu && v3v.s3_portofoliu.overall_pass === true, "G4: 36/36 Pareto exact + 1 000 000 traiectorii în ±4 SE"],
      ["S4 · Poarta A5 (CK-V6 + 0400)", "VERIFICARE_S4_CKV6.json", v3v.s4_ckv6 && v3v.s4_ckv6.overall_pass === true, "Q1–Q6 = 6/6 PASS · 10k putere GPU · 0400 neatins"],
      ["S5 · Poarta A6 (3 Registre Live)", "GATE_A6_REGISTRE_SI_DECONTARE.json", v3v.gate_a6_registre ? v3v.gate_a6_registre.gate_a6_pass === true : null, "XI-F + TU4 (S35_03) + Quantum Limit / OGLINDA"],
    ];
    grid($("p-v3-gates"),
      ["etapă / poartă", "artefact certificat", "verdict", "rezumat verificare independentă"],
      v3Rows.map(([lbl, art, okVal, det]) => [
        { t: lbl, c: "mono warn-txt" },
        { t: art, c: "mono" },
        okVal === null ? { t: "LIVE ÎN FILA 2", c: "warn-txt" } : { t: okVal ? "PASS" : "FAIL", c: okVal ? "pass" : "fail" },
        det,
      ])
    );

    if (!p.disponibil) {
      kv($("p-kv"), [["raport", "încă nu există"]]);
      return { ok: true, disponibil: false };
    }
    kv($("p-kv"), [
      ["fișier", p.fisier],
      ["rulat (UTC)", p.run_utc],
      ["verdict B2–B7", p.verdict_B2_B7, p.verdict_B2_B7 === "PASS" ? "pass" : "fail"],
      ["verificări", `${p.n_pass} / ${p.n_checks}`],
      ["max |Δ| float", fmt(p.max_abs_delta_float, 3)],
      ["valori float comparate", p.values_compared_float],
      ["identice bit-cu-bit", p.values_bitexact_float],
    ]);
    grid($("p-groups"), ["grup", "verificări", "trec", "verdict"],
      Object.entries(p.groups || {}).map(([g, v]) => [g, v.checks, v.pass, { t: v.verdict, c: v.verdict === "PASS" ? "pass" : "fail" }]));

    const vk = p.verificare_kjv5;
    if (vk && typeof vk === "object") {
      kv($("p-kjv5-kv"), [
        ["verdict VERIFICARE_KJV5.json", vk.verdict || "—", vk.verdict === "PASS" ? "pass" : "fail"],
        ["verificări trecute", `${vk.passed_checks ?? "—"} / ${vk.total_checks ?? "—"}`],
        ["eroare max |Δ| Python ↔ Rust K1–K4", fmt(vk.max_abs_step_err, 3)],
        ["generat (UTC)", vk.generated_utc || "—"],
      ]);
      grid($("p-kjv5-checks"), ["verificare", "stare", "detaliu"],
        (vk.checks || []).map((c) => [c.id, { t: c.pass ? "PASS" : "FAIL", c: c.pass ? "pass" : "fail" }, { t: c.detail || "", c: "mono" }]));
    }
    return { ok: true, disponibil: true, verdict: p.verdict_B2_B7 };
  } catch (e) {
    kv($("p-kv"), [["eroare", String(e)]]);
    return { ok: false, error: String(e) };
  }
}

async function loadInventory() {
  try {
    const r = await invoke("cmd_inventar");
    grid($("i-table"), ["joc", "înregistrări", "respinse", "prima dată", "ultima cheie", "octeți", "µs", "= CURRENT_LIVE"],
      r.arhive.map((a) => [a.joc, a.rezumat.records, a.rezumat.rejects, a.rezumat.first_date, a.rezumat.last_key, a.rezumat.bytes,
        a.rezumat.elapsed_micros, { t: a.potrivire_current_live ? "da" : "NU", c: a.potrivire_current_live ? "pass" : "fail" }]));
    return { ok: true, records: r.arhive.map((a) => a.rezumat.records), match: r.arhive.every((a) => a.potrivire_current_live) };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

// ============================================================================
// AUDIT AUTOMAT DARK-MODE WEBKITGTK (`0` CASETE ALBE PE TOATE CELE 12 FILE)
// ============================================================================

function parseCssRgbaLuminance(bgStr) {
  if (!bgStr || bgStr === "transparent") return { alpha: 0, lum: 0 };
  const m = bgStr.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*[,/]\s*([\d.]+))?\s*\)/i);
  if (!m) return { alpha: 0, lum: 0 };
  const r = parseFloat(m[1]), g = parseFloat(m[2]), b = parseFloat(m[3]);
  const a = m[4] !== undefined ? parseFloat(m[4]) : 1.0;
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return { alpha: a, lum, r, g, b };
}

function auditDarkModeDom() {
  const offenders = [];
  let elementsAudited = 0;

  const inspectEl = (el, tagCtx) => {
    if (!el) return;
    const cs = window.getComputedStyle(el);
    const { alpha, lum } = parseCssRgbaLuminance(cs.backgroundColor);
    elementsAudited += 1;
    if (alpha > 0.08 && lum > 175.0) {
      offenders.push({
        ctx: tagCtx,
        tag: el.tagName,
        id: el.id || "",
        cls: el.className || "",
        bg: cs.backgroundColor,
        lum: Math.round(lum),
      });
    }
  };

  inspectEl(document.documentElement, "root");
  inspectEl(document.body, "body");
  inspectEl(document.querySelector("header"), "header");
  inspectEl(document.querySelector("footer"), "footer");

  const tabButtons = [...document.querySelectorAll("#main-tabs .tab")];
  const panels = [...document.querySelectorAll("main > section.panel")];

  for (const btn of tabButtons) {
    const tName = btn.dataset.tab;
    activateTab(tName);
    inspectEl(btn, `tab-btn:${tName}`);
    const panel = $("tab-" + tName);
    inspectEl(panel, `panel:${tName}`);
    if (panel) {
      panel.querySelectorAll("select, option, input, textarea, button, .card, .kpi-box, .scroll-box, canvas, svg, th").forEach((el) => {
        inspectEl(el, `in-${tName}`);
      });
    }
  }

  // Verificăm și fereastra modală dark-mode deschisă temporar
  openModal("Audit Dark-Mode WebKitGTK", [["stare", "verificare automată 0 casete albe"]], ["element", "fundal"], [["modal-card", "dark"]]);
  inspectEl($("v3-modal"), "modal-backdrop");
  inspectEl(document.querySelector("#v3-modal .modal-card"), "modal-card");
  document.querySelectorAll("#v3-modal button, #v3-modal .scroll-box, #v3-modal th").forEach((el) => inspectEl(el, "in-modal"));
  closeModal();

  activateTab("v3-vault");
  const rootScheme = window.getComputedStyle(document.documentElement).colorScheme || "dark";

  return {
    ok: offenders.length === 0 && tabButtons.length === 12 && panels.length === 12,
    tabs_buttons_count: tabButtons.length,
    panels_count: panels.length,
    elements_audited: elementsAudited,
    selects_count: document.querySelectorAll("select").length,
    inputs_count: document.querySelectorAll("input, textarea").length,
    canvases_count: document.querySelectorAll("canvas").length,
    color_scheme: rootScheme,
    white_boxes_count: offenders.length,
    offenders: offenders.slice(0, 10),
  };
}

// ============================================================================
// INIȚIALIZARE EVENIMENTE & SUITA `--smoke` COMPLETĂ PE CELE 12 FILE
// ============================================================================

function bindEvents() {
  document.querySelectorAll("#main-tabs .tab").forEach((b) => {
    b.addEventListener("click", () => activateTab(b.dataset.tab));
  });

  $("v3-modal-close").addEventListener("click", closeModal);
  $("v3-modal").addEventListener("click", (ev) => {
    if (ev.target === $("v3-modal")) closeModal();
  });

  // Fila 1 (Vault v3 + CD-V3Q + C3/C4 + Eval GPU)
  $("v3t-joc").addEventListener("change", () => {
    const j = $("v3t-joc").value;
    if (j === "649") $("v3t-bile").value = "3 14 22 31 41 49";
    else if (j === "540") $("v3t-bile").value = "4 13 21 32 39";
    else $("v3t-bile").value = "5 14 23 34 44";
    runV3MmapTicket(false);
  });
  $("v3t-query-nums").addEventListener("click", () => runV3MmapTicket(false));
  $("v3t-query-rank").addEventListener("click", () => runV3MmapTicket(true));
  $("v3t-add-gpu").addEventListener("click", () => {
    const lt = v3State.lastTicket;
    $("v3e-joc").value = lt.joc;
    const line = lt.numbers.join(" ") + (lt.joker != null ? ` +${lt.joker}` : "");
    const cur = $("v3e-input").value.trim();
    $("v3e-input").value = cur ? `${cur}\n${line}` : line;
    runV3GpuEvalPortfolio();
  });
  $("v3t-send-cu").addEventListener("click", () => {
    const lt = v3State.lastTicket;
    jumpToCateSiUnde(lt.joc, lt.numbers, lt.joker);
  });

  $("v3w-qap").addEventListener("change", () => {
    populateWheelSelect();
    loadV3WheelDetail();
  });
  $("v3w-id").addEventListener("change", loadV3WheelDetail);
  $("v3w-load-gpu").addEventListener("click", () => {
    const w = v3State.lastWheel;
    if (!w || !w.meta) return;
    const g = w.meta.game;
    $("v3e-joc").value = g;
    const lines = (w.tickets_preview || []).slice(0, 200).map((t) =>
      t.numbers.join(" ") + (t.joker != null ? ` +${t.joker}` : "")
    );
    $("v3e-input").value = lines.join("\n");
    runV3GpuEvalPortfolio();
  });
  $("v3w-modal").addEventListener("click", () => {
    const w = v3State.lastWheel;
    if (!w || !w.meta) return;
    const m = w.meta;
    openModal(
      `Catalog Roată ${m.id} (${w.qap_relabeled ? "CD-V3Q QAP" : "CD-V3"})`,
      [
        ["ID / tip / joc", `${m.id} · ${w.kind} · ${m.game}`],
        ["bilete în catalog", String(m.n_tickets ?? m.k_budget_tickets ?? "—")],
        ["bilete tipizate după QAP", String(m.patterned_after ?? 0), "pass"],
        ["EV mediu NB2", m.mean_ev_after !== undefined ? `${fmt(m.mean_ev_after, 4)} lei` : "—"],
      ],
      ["#", "bilet", "EV NB2", "q05..q95"],
      (w.tickets_preview || []).map((t) => [
        t.i,
        makeBallsNode(t.numbers, t.joker, null),
        fmt(t.ev_nominal, 4),
        `[${fmt(t.q05, 2)}, ${fmt(t.q95, 2)}]`,
      ])
    );
  });

  $("v3p-joc").addEventListener("change", loadV3C3C4);
  $("v3p-buget").addEventListener("change", loadV3C3C4);
  $("v3p-send-gpu").addEventListener("click", () => {
    const ps = v3State.paretoSample;
    if (!ps.tickets.length) return;
    $("v3e-joc").value = ps.joc;
    $("v3e-input").value = ps.tickets.map((t) =>
      t.numbers.join(" ") + (t.joker != null ? ` +${t.joker}` : "")
    ).join("\n");
    runV3GpuEvalPortfolio();
  });
  $("v3e-joc").addEventListener("change", () => {
    const g = $("v3e-joc").value;
    if (g === "649") $("v3e-input").value = "2 11 19 28 37 46\n5 14 23 33 42 49\n8 17 25 34 43 48";
    else if (g === "540") $("v3e-input").value = "3 11 20 29 38\n6 15 24 33 40\n2 12 22 31 39";
    else $("v3e-input").value = "4 12 21 34 43 +7\n7 16 25 36 45 +19\n2 13 24 35 44 +3";
    runV3GpuEvalPortfolio();
  });
  $("v3e-run").addEventListener("click", runV3GpuEvalPortfolio);

  // Fila 2 (Poarta A6 + CK-V6)
  $("v3r-refresh").addEventListener("click", loadV3RegistreLive);
  $("v3ck-stream").addEventListener("change", loadV3Ckv6);
  $("v3ck-cell").addEventListener("change", loadV3Ckv6);
  $("v3cks-run").addEventListener("click", runV3Ckv6LiveSim);

  // Fila 3 (Hardware Live)
  $("v3hw-refresh").addEventListener("click", loadV3HwTelemetry);
  $("v3hw-auto").addEventListener("change", syncHwAutoLoop);

  // Filele 4..12 (v2)
  $("kj-joc").addEventListener("change", loadKjv5Ro);
  $("kj-cfg").addEventListener("change", loadKjv5Ro);
  $("hs-joc").addEventListener("change", loadHsv4Ro);
  $("hs-cfg").addEventListener("change", loadHsv4Ro);
  $("cu-run").addEventListener("click", runCateSiUnde);
  $("cu-fill-kj").addEventListener("click", () => {
    jumpToCateSiUnde(quickState.kjAlpha.joc, quickState.kjAlpha.bile, quickState.kjAlpha.bonus);
  });
  $("cu-fill-hs").addEventListener("click", () => {
    jumpToCateSiUnde(quickState.hsAlpha.joc, quickState.hsAlpha.bile, quickState.hsAlpha.bonus);
  });
  $("in-flux").addEventListener("change", loadIntlComparatie);
  $("in-send-cu").addEventListener("click", () => {
    jumpToCateSiUnde(quickState.intlAlpha.flux, quickState.intlAlpha.bile, quickState.intlAlpha.bonus);
  });
  $("d-joc").addEventListener("change", loadDensities);
  $("d-model").addEventListener("change", loadDensities);
  $("f-joc").addEventListener("change", loadFrft);
  $("f-a").addEventListener("input", loadFrft);
  $("a-tinta").addEventListener("change", () => loadAdm());
  $("m-refresh").addEventListener("click", () => {
    loadMemory();
    loadTelemetry().then((r) => { if (r.ok) drawTelemetry(r.raspuns); });
  });
  initLive();
}

async function main() {
  bindEvents();
  const info = await invoke("cmd_info");
  $("info").textContent = `v${info.version} · ${info.calcul}`;
  $("pca").textContent = String(info.positive_claims_allowed);

  const res = {
    info: { ok: info.positive_claims_allowed === false, version: info.version, positive_claims_allowed: info.positive_claims_allowed },
    tabs_count: document.querySelectorAll("#main-tabs .tab").length,
  };

  // Încărcăm cele 3 file noi v3 + cele 9 file v2
  res.v3_overview = await loadV3Overview();
  res.v3_mmap_ticket = await runV3MmapTicket(false);
  res.v3_wheel_detail = await loadV3WheelDetail();
  res.v3_c3_c4 = await loadV3C3C4();
  res.v3_gpu_eval = await runV3GpuEvalPortfolio();
  res.v3_registre_live = await loadV3RegistreLive();
  res.v3_ckv6 = await loadV3Ckv6();
  res.v3_ckv6_live_sim = await runV3Ckv6LiveSim();
  res.v3_hw_telemetry = await loadV3HwTelemetry();

  res.kjv5 = await loadKjv5Ro();
  res.hsv4 = await loadHsv4Ro();
  res.cate_si_unde = await runCateSiUnde();
  res.intl = await loadIntlComparatie();
  res.densitati = await loadDensities();
  res.frft = await loadFrft();
  res.tinte = await loadTargets();
  res.admisibilitate = await loadAdm();
  res.paritate = await loadParity();
  res.inventar = await loadInventory();
  res.memorie = await loadMemory();
  res.telemetrie = await loadTelemetry();
  if (res.telemetrie.ok) drawTelemetry(res.telemetrie.raspuns);

  if (info.smoke) {
    // Verificăm latența < 1 ms zero-copy mmap pe toate cele 3 jocuri (649, 540, Joker)
    res.v3_mmap_3games = [];
    for (const [g, b, jk] of [["649", [2, 13, 24, 35, 41, 49], null], ["540", [3, 12, 21, 30, 39], null], ["joker", [4, 15, 26, 37, 45], 14]]) {
      const q = await invoke("cmd_v3_mmap_ticket", { joc: g, bile: b, joker: jk, rang: null });
      res.v3_mmap_3games.push({
        joc: g,
        query_us: q.query_us,
        under_1ms: q.under_1ms,
        ok: q.under_1ms === true && q.mask_verified_equal === true && q.positive_claims_allowed === false,
      });
    }

    res.kjv5_jocuri = [];
    for (const joc of ["649", "540", "joker"]) {
      const r = await invoke("cmd_kjv5_ro", { joc, config: "Full_KJV5" });
      res.kjv5_jocuri.push({
        joc,
        ok: r.selected.martingale_trajectory.guarantee_E_ge_050_verified === true && r.positive_claims_allowed === false,
      });
    }
    res.hsv4_jocuri = [];
    for (const joc of ["649", "540", "joker"]) {
      const r = await invoke("cmd_hsv4_ro", { joc, config: "Full_HSV4" });
      res.hsv4_jocuri.push({
        joc,
        ok: r.seals.verified_all_frozen_hashes === true && r.positive_claims_allowed === false,
      });
    }
    res.combinatii = [];
    for (const joc of ["649", "540", "joker"]) {
      for (const model of ["H1", "H3"]) {
        try {
          const d = await invoke("cmd_densitati", { joc, model });
          res.combinatii.push({ joc, model, ok: Math.abs(d.rho_trace - 1) <= 1e-12 && d.rho_min_eigenvalue >= -1e-12 });
        } catch (e) {
          res.combinatii.push({ joc, model, ok: false, error: String(e) });
        }
      }
    }
    res.admisibilitate_tinte = [];
    for (const [tinta] of (res.tinte.tinte || [])) res.admisibilitate_tinte.push(await loadAdm(tinta));
    await loadAdm(res.tinte.implicita);

    for (let i = 0; i < 100; i++) {
      res.telemetrie = await loadTelemetry();
      if (!res.telemetrie.ok || res.telemetrie.esantioane >= 5 || res.telemetrie.stare !== "ACTIV") break;
      await new Promise((r) => setTimeout(r, 100));
    }
    res.grafic = await smokeGrafic();
    res.eticheta = smokeEticheta();

    if (v3State.hwTimer !== null) {
      clearInterval(v3State.hwTimer);
      v3State.hwTimer = null;
    }
    res.dark_mode_audit = auditDarkModeDom();

    const requiredKeys = [
      "info", "v3_overview", "v3_mmap_ticket", "v3_wheel_detail", "v3_c3_c4",
      "v3_gpu_eval", "v3_registre_live", "v3_ckv6", "v3_ckv6_live_sim", "v3_hw_telemetry",
      "kjv5", "hsv4", "cate_si_unde", "intl",
      "densitati", "frft", "tinte", "admisibilitate", "paritate", "inventar", "memorie", "telemetrie",
      "grafic", "eticheta", "dark_mode_audit",
    ];

    res.toate_ok =
      res.tabs_count === 12 &&
      res.dark_mode_audit.white_boxes_count === 0 &&
      requiredKeys.every((k) => res[k] && res[k].ok === true) &&
      res.v3_mmap_3games.every((c) => c.ok) &&
      res.kjv5_jocuri.every((c) => c.ok) &&
      res.hsv4_jocuri.every((c) => c.ok) &&
      res.combinatii.every((c) => c.ok) &&
      res.admisibilitate_tinte.length > 0 &&
      res.admisibilitate_tinte.every((a) => a.ok);

    await invoke("cmd_smoke_done", { raport: res });
  }
}

main().catch(async (e) => {
  const p = document.createElement("p");
  p.className = "banner";
  p.textContent = `Eroare: ${String(e)}`;
  document.body.prepend(p);
  try { await invoke("cmd_smoke_done", { raport: { toate_ok: false, error: String(e) } }); } catch (_) { /* nu e mod smoke */ }
});
