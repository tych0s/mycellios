"use strict";

const state = { runs: [], scenario: "", version: "", evidence: "", search: "" };
const $ = (selector) => document.querySelector(selector);
const nf = new Intl.NumberFormat("es-ES", { maximumFractionDigits: 2 });
const datef = new Intl.DateTimeFormat("es-ES", { dateStyle: "medium", timeStyle: "short" });

document.addEventListener("DOMContentLoaded", () => {
  $("#run-button").addEventListener("click", runSuite);
  $("#scenario-filter").addEventListener("change", (event) => { state.scenario = event.target.value; renderTrend(); });
  $("#version-filter").addEventListener("change", (event) => { state.version = event.target.value; renderTable(); });
  $("#evidence-filter").addEventListener("change", (event) => { state.evidence = event.target.value; renderTable(); });
  $("#search-filter").addEventListener("input", (event) => { state.search = event.target.value.toLowerCase(); renderTable(); });
  $("#close-detail").addEventListener("click", () => $("#detail-dialog").close());
  loadRuns();
});

async function loadRuns() {
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.runs = (await response.json()).runs;
    renderAll();
  } catch (error) { toast(`No se pudo cargar el historial: ${error.message}`, true); }
}

async function runSuite() {
  const button = $("#run-button");
  button.disabled = true;
  button.textContent = "Midiendo runtime…";
  try {
    const response = await fetch("/api/run", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    toast(`Medición real terminada: ${statusLabel(payload.run.status)}.`);
    await loadRuns();
  } catch (error) { toast(`Error al ejecutar: ${error.message}`, true); }
  finally { button.disabled = false; button.textContent = "● Medir runtime real"; }
}

function renderAll() {
  const latest = state.runs[0];
  $("#last-update").textContent = latest ? `Última ejecución: ${datef.format(new Date(latest.finishedAt))}` : "Sin ejecuciones todavía";
  populateFilters();
  renderSummary();
  renderTrend();
  renderTable();
}

function measurements() {
  return state.runs.flatMap((run) => run.measurements.map((measurement) => ({ run, measurement })));
}

function populateFilters() {
  const all = measurements();
  const versions = [...new Set(state.runs.map((run) => run.version))];
  const scenarios = [...new Map(all.map(({ measurement }) => [measurement.id, measurement.title])).entries()];
  const evidences = [...new Set(all.map(({ measurement }) => measurement.evidence))];
  fillSelect($("#version-filter"), versions.map((value) => [value, `Versión ${value}`]), "Todas las versiones", state.version);
  fillSelect($("#evidence-filter"), evidences.map((value) => [value, evidenceLabel(value)]), "Toda evidencia", state.evidence);
  if (!state.scenario || !scenarios.some(([id]) => id === state.scenario)) state.scenario = scenarios[0]?.[0] || "";
  fillSelect($("#scenario-filter"), scenarios, "Sin escenarios", state.scenario, false);
}

function fillSelect(select, entries, firstLabel, selected, includeEmpty = true) {
  select.replaceChildren();
  if (includeEmpty) select.add(new Option(firstLabel, ""));
  for (const [value, label] of entries) select.add(new Option(label, value));
  select.value = selected;
}

function renderSummary() {
  const latest = state.runs[0];
  if (!latest) {
    $("#summary").innerHTML = card("Versión", "—", "Ejecuta la primera suite") + card("Estado", "Sin datos", "Aún no hay referencia");
    return;
  }
  const validTps = latest.measurements.map((item) => item.metrics.tokensPerSecond).filter(Number.isFinite);
  const avgTps = validTps.length ? validTps.reduce((a,b) => a+b,0) / validTps.length : null;
  const regressions = latest.measurements.filter((item) => item.status === "regression").length;
  const realMeasurements = measurements().filter(({ measurement }) => measurement.evidence === "physical" || measurement.evidence === "loopback").length;
  const connected = Math.max(0, ...latest.measurements.map((item) => item.inventory.connectedDevices));
  $("#summary").innerHTML =
    card("Versión actual", escapeHtml(latest.version), escapeHtml(latest.label)) +
    card("Estado", statusLabel(latest.status), regressions ? `${regressions} regresiones` : "Sin regresiones", latest.status === "regression" ? "bad" : "good") +
    card("Rendimiento medio", avgTps === null ? "—" : `${nf.format(avgTps)} tok/s`, `${validTps.length} escenarios con métrica`) +
    card("Dispositivos conectados", nf.format(connected), "máximo en la última suite") +
    card("Mediciones reales", nf.format(realMeasurements), realMeasurements ? "sin datos simulados" : "runtime todavía no medido");
}

function card(label, value, detail, css = "") {
  return `<article class="summary-card ${css}"><span class="label">${label}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function renderTrend() {
  const points = state.runs.slice().reverse().flatMap((run) => run.measurements.filter((m) => m.id === state.scenario && Number.isFinite(m.metrics.tokensPerSecond)).map((m) => ({ run, m, value: m.metrics.tokensPerSecond })));
  const target = $("#trend-chart");
  if (!points.length) { target.innerHTML = `<div class="chart-empty">Este escenario todavía no tiene tokens/s.</div>`; return; }
  const width = 1120, height = 210, left = 55, right = 24, top = 25, bottom = 38;
  const values = points.map((point) => point.value);
  const min = Math.min(...values), max = Math.max(...values);
  const padding = Math.max((max - min) * .2, max * .08, 1);
  const low = Math.max(0, min - padding), high = max + padding;
  const x = (index) => left + (points.length === 1 ? (width-left-right)/2 : index * (width-left-right)/(points.length-1));
  const y = (value) => top + (high-value) * (height-top-bottom)/(high-low || 1);
  const path = points.map((point,index) => `${index ? "L" : "M"}${x(index)},${y(point.value)}`).join(" ");
  const area = `${path} L${x(points.length-1)},${height-bottom} L${x(0)},${height-bottom} Z`;
  const grid = [0,.5,1].map((ratio) => { const value = high-(high-low)*ratio, yy = y(value); return `<line class="chart-grid" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/><text class="chart-label" x="0" y="${yy+4}">${nf.format(value)}</text>`; }).join("");
  const dots = points.map((point,index) => `<circle class="chart-point" cx="${x(index)}" cy="${y(point.value)}" r="5"/><text class="chart-value" x="${x(index)}" y="${y(point.value)-13}" text-anchor="middle">${nf.format(point.value)}</text><text class="chart-label" x="${x(index)}" y="${height-8}" text-anchor="middle">v${escapeHtml(point.run.version)}</text>`).join("");
  target.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#51e6a6" stop-opacity=".24"/><stop offset="1" stop-color="#51e6a6" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${area}"/><path class="chart-line" d="${path}"/>${dots}</svg>`;
}

function renderTable() {
  const rows = measurements().filter(({ run, measurement }) => {
    if (state.version && run.version !== state.version) return false;
    if (state.evidence && measurement.evidence !== state.evidence) return false;
    const haystack = `${measurement.title} ${measurement.model.label} ${measurement.id}`.toLowerCase();
    return !state.search || haystack.includes(state.search);
  });
  $("#empty-results").hidden = rows.length > 0;
  const body = $("#results-body"); body.replaceChildren();
  for (const entry of rows) {
    const { run, measurement: m } = entry;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="version-cell"><strong>v${escapeHtml(run.version)}</strong><span>${escapeHtml(datef.format(new Date(run.finishedAt)))}</span></td><td class="test-cell"><div class="evidence ${m.evidence}"><strong>${escapeHtml(m.title)}</strong><span>${escapeHtml(m.model.label)} · ${evidenceLabel(m.evidence)}</span></div></td><td class="device-cell"><strong>${m.inventory.connectedDevices} / ${m.inventory.totalDevices}</strong><span>${escapeHtml(deviceSummary(m))}</span></td><td class="metric">${metric(m.metrics.tokensPerSecond)} <small>tok/s</small></td><td class="metric">${metric(m.metrics.ttftMsP95)} <small>ms</small></td><td>${deltaMarkup(m.comparison.tokensPerSecondPct)}</td><td><span class="badge ${m.status}">${statusLabel(m.status)}</span></td>`;
    tr.addEventListener("click", () => showDetail(entry)); body.appendChild(tr);
  }
}

function showDetail({ run, measurement: m }) {
  const profiles = m.inventory.profiles.map((profile) => `<li>${profile.count}× ${escapeHtml(profile.label)}${profile.memoryGb ? ` · ${profile.memoryGb} GB` : ""}</li>`).join("");
  const notes = m.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
  const reasons = m.comparison.reasons.length ? `<div class="detail-section"><h3>Regresiones detectadas</h3><ul>${m.comparison.reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul></div>` : "";
  $("#detail-content").innerHTML = `<p class="eyebrow">${evidenceLabel(m.evidence).toUpperCase()} · V${escapeHtml(run.version)}</p><h2 class="detail-title">${escapeHtml(m.title)}</h2><p class="detail-subtitle">${escapeHtml(m.description)}</p><div class="detail-grid">${detailItem("Modelo", m.model.label)}${detailItem("Tokens/s", metric(m.metrics.tokensPerSecond))}${detailItem("TTFT P95", `${metric(m.metrics.ttftMsP95)} ms`)}${detailItem("Conectados", `${m.inventory.connectedDevices}/${m.inventory.totalDevices}`)}${detailItem("Seleccionados", m.inventory.selectedDevices)}${detailItem("Aceptación", m.metrics.acceptanceRate === null ? "—" : `${nf.format(m.metrics.acceptanceRate*100)}%`)}</div><div class="detail-section"><h3>Hardware / inventario</h3><ul>${profiles || "<li>Sin inventario</li>"}</ul></div><div class="detail-section"><h3>Notas de evidencia</h3><ul>${notes}</ul></div>${reasons}`;
  $("#detail-dialog").showModal();
}

function detailItem(label, value) { return `<div class="detail-item"><span>${label}</span><strong>${escapeHtml(String(value))}</strong></div>`; }
function deviceSummary(m) { const first = m.inventory.profiles[0]; return first ? `${first.count}× ${first.label}${m.inventory.profiles.length > 1 ? ` +${m.inventory.profiles.length-1}` : ""}` : "sin detalle"; }
function metric(value) { return Number.isFinite(value) ? nf.format(value) : "—"; }
function deltaMarkup(value) { if (!Number.isFinite(value)) return `<span class="delta neutral">referencia</span>`; const css = value > .01 ? "up" : value < -.01 ? "down" : "neutral"; return `<span class="delta ${css}">${value > 0 ? "+" : ""}${nf.format(value)}%</span>`; }
function statusLabel(status) { return ({ baseline:"Referencia", passed:"Correcto", regression:"Regresión", failed:"Fallido" })[status] || status; }
function evidenceLabel(evidence) { return ({ physical:"Físico multi-equipo", loopback:"Real local" })[evidence] || evidence; }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]); }
let toastTimer; function toast(message, error = false) { const node = $("#toast"); node.textContent = message; node.className = `toast visible${error ? " error" : ""}`; clearTimeout(toastTimer); toastTimer = setTimeout(() => { node.className = "toast"; }, 4000); }
