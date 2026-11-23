// Shared helpers for the SynthCore site. data.json is written by scripts/build_site.py.

export const COMMONS = "https://alignment-commons-site.vercel.app";
export const REPO = "https://github.com/jaebooker/alignment-synthcore";

// Verdict labels, colours and one-line meanings. Order is the legend order.
export const VERDICT = {
  beats_baseline: {label: "Beats baseline", cls: "win", color: "var(--sage)", note: "Significant safety win over a tuned baseline, with task completion kept where measured."},
  safe_but_incomplete: {label: "Safety gain, task incomplete", cls: "part", color: "var(--plum)", note: "Safer, but task completion was not shown to hold."},
  mech_ok_no_win: {label: "Mechanism works, no win", cls: "mech", color: "var(--blue)", note: "The predicted mechanism showed up; the safety win did not."},
  no_effect: {label: "No effect", cls: "null", color: "var(--faint)", note: "No significant difference from the baseline."},
  hurts: {label: "Hurts", cls: "hurt", color: "var(--rust)", note: "Significantly worse than the baseline."},
  mechanism_failed: {label: "Mechanism failed", cls: "fail", color: "var(--ochre)", note: "Failed its own predicted mechanism at the toy gate, so it never reached the benchmark."},
  untestable: {label: "Not testable yet", cls: "null", color: "var(--line-2)", note: "No current testbed can test it. The reason is kept, and feeds which testbed to build next."},
};
// One colour per testbed area, shared by the testbed panels and the live map.
export const AREA_COLORS = {interp: "var(--blue)", backdoor: "var(--rust)", hacking: "var(--ochre)", misgen: "var(--sage)", oversight: "var(--plum)", control: "var(--ink-2)", swarm: "var(--olive)"};
export const v = k => VERDICT[k] || {label: k, cls: "null", color: "var(--faint)", note: ""};
export const badge = k => `<span class="badge ${v(k).cls}">${esc(v(k).label)}</span>`;

export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
export function ago(t, now = Date.now()) {
  const s = Math.max(0, (now - t) / 1000);
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
export const signed = x => x == null ? "" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
export function effect(i) {
  const s = i.safety; if (!s || s.diff == null) return "";
  const ci = s.ci95 ? ` (${signed(s.ci95[0])} to ${signed(s.ci95[1])})` : "";
  return `${signed(s.diff)}${ci}${s.p != null ? ` · p ${s.p.toFixed(3)}` : ""}`;
}

export async function loadData() {
  const d = await fetch("/data.json", {cache: "no-cache"}).then(r => r.json());
  d.tb = Object.fromEntries(d.testbeds.map(t => [t.id, t]));
  d.byId = Object.fromEntries(d.ideas.map(i => [i.id, i]));
  d.areaOf = {};
  d.areas.forEach(a => a.testbeds.forEach(t => d.areaOf[t] = a));
  // Controls (the baseline against itself, a mechanism-free twin) check the harness, not an idea.
  d.tested = d.ideas.filter(i => !i.control && i.verdict !== "untestable");
  d.wins = d.tested.filter(i => i.verdict === "beats_baseline");
  return d;
}

// Commons activity, read cross-origin from the public API. Optional.
export async function commonsEvents() {
  const j = await fetch(`${COMMONS}/api/commons?action=activity`, {cache: "no-store"}).then(r => r.ok ? r.json() : Promise.reject(r.status));
  const LEDGER = /Agent Commons ledger #(\d+) by ([\w.-]+), recorded [^()]+? \(mirrored [^)]+\)\.\s*(?:Outcome:\s*([\w-]+)\.)?\s*/;
  return j.data.events.map(e => {
    const m = LEDGER.exec(e.body || "");
    return {...e, actor: m ? m[2] : (e.agent || "Workspace").replace(/ — .*/, ""), outcome: m ? m[3] : null,
      body: m ? e.body.replace(LEDGER, "").trim() : e.body, subject: (e.subject || "").replace(/^Ledger #\d+ \([^)]*\):\s*/, ""), t: Date.parse(e.created_at)};
  });
}

export const HEADER = current => `
<a class="skip" href="#main">Skip to the content</a>
<header class="site-head"><div class="container">
  <a class="wordmark" href="/">Alignment SynthCore</a>
  <nav class="site-nav" aria-label="Main">
    <a href="/#testbeds">Testbeds</a><a href="/#ideas">Verdicts</a><a href="/#record">Record</a><a href="/#how">How it works</a>
    <a class="live-link" href="/live"${current === "live" ? ' aria-current="page"' : ""}>Live</a><a href="${COMMONS}" target="_blank" rel="noopener">Commons ↗</a>
  </nav>
  <a class="pill-btn" href="${REPO}#adding-an-idea" target="_blank" rel="noopener">Test an idea</a>
</div></header>
<div class="justin" id="justin" hidden></div>`;

export const FOOTER = () => `
<footer class="site-foot"><div class="container">
  <div class="cols">
    <div><h4>Site</h4><ul><li><a href="/#testbeds">Testbeds</a></li><li><a href="/#ideas">Verdicts</a></li><li><a href="/live">Live</a></li></ul></div>
    <div><h4>About</h4><ul><li><a href="/#how">How it works</a></li><li><a href="${REPO}">Source code</a></li><li><a href="${REPO}/blob/main/results/SUMMARY.md">Raw results</a></li></ul></div>
    <div><h4>Follow</h4><ul><li><a href="${COMMONS}">Alignment Commons</a></li><li><a href="${COMMONS}/live">Agents in the Commons</a></li><li><a href="${COMMONS}/workspace">Connect your own agent</a></li></ul></div>
  </div>
  <p class="disclaimer">No verdict here is about the real problem at scale. Each comes from a small testbed with a planted answer, and code computes it. Failures are published too.</p>
</div></footer>`;

export function justIn(d, now = Date.now()) {
  const last = d.byId[d.runs[0]]; if (!last) return;
  const bar = document.getElementById("justin");
  bar.hidden = false;
  bar.innerHTML = `<div class="container"><b>Latest run:</b><span class="msg">${esc(last.title)}: ${esc(v(last.verdict).label.toLowerCase())}${last.safety?.diff != null ? `, ${signed(last.safety.diff)}` : ""}</span><span class="t">· ${ago(Date.parse(last.updated), now)}</span><a href="/#idea-${esc(last.id)}">Read it →</a></div>`;
}
