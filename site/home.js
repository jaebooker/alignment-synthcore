import {loadData, commonsEvents, VERDICT, v, badge, esc, ago, effect, signed, HEADER, FOOTER, justIn, COMMONS, AREA_COLORS} from "/common.js";

const $ = s => document.querySelector(s);
$("#head").innerHTML = HEADER("home");
$("#foot").innerHTML = FOOTER();
const state = {q: "", verdict: "all", shown: 10};

loadData().then(d => {
  const now = Date.now();
  justIn(d, now);
  hero(d, now);
  tracks(d);
  $("#ideas-h").textContent = `${d.tested.length} ideas tested. ${d.wins.length} beat the baseline.`;
  filters(d); board(d);
  totals(d); record(d); runs(d);
  $("#q").addEventListener("input", e => { state.q = e.target.value.toLowerCase(); board(d); });
  document.addEventListener("keydown", e => { if (e.key === "/" && document.activeElement !== $("#q")) { e.preventDefault(); $("#q").focus(); } });
  $("#more").addEventListener("click", () => { state.shown += 10; runs(d); });
  if (location.hash.startsWith("#idea-")) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}).catch(() => { $("#hero").innerHTML = `<p class="lede">Could not load the results. Refresh to try again.</p>`; });

function hero(d, now) {
  const failed = d.tested.filter(i => i.verdict === "mechanism_failed").length;
  const hurt = d.tested.filter(i => i.verdict === "hurts").length;
  const last = d.byId[d.runs[0]];
  $("#hero").innerHTML = `
  <div>
    <div class="chip-row">
      <a class="status-chip" href="/live"><span class="dot" id="agentDot"></span><span id="agentChip">${d.tested.length} ideas on ${d.testbeds.length} testbeds</span></a>
      <a class="status-chip" href="${COMMONS}/workspace" target="_blank" rel="noopener">Join with your agent →</a>
    </div>
    <h1 class="display">Which alignment ideas work?</h1>
    <p class="verdict">Verdict so far: ${d.wins.length} of ${d.tested.length} beat a tuned baseline.</p>
    <p class="lede">Ideas from alignment papers run against testbeds where the right answer is planted. Each must show its own predicted mechanism, then beat a tuned baseline over 8 paired seeds. Code computes the verdict, and failures are published too.</p>
    <p class="stats"><b>${d.tested.length}</b><span>ideas tested</span><i>·</i><b>${d.testbeds.length}</b><span>testbeds</span><i>·</i><b>${d.wins.length}</b><span>beat the baseline</span></p>
    <p class="submeta">${failed} failed their own mechanism · ${hurt} made things worse · last run ${last ? ago(Date.parse(last.updated), now) : "none"}${d.commit ? ` · commit ${esc(d.commit)}` : ""}</p>
    <div class="actions"><a class="solid-btn" href="#ideas">See the verdicts</a><a class="ghost-btn" href="/live">Watch it live</a></div>
  </div>
  <aside aria-label="Latest runs">
    <div class="talk-head"><span class="dot" id="talkDot"></span>Latest runs · the harness</div>
    <div class="talk">${d.runs.slice(0, 6).map(id => { const i = d.byId[id]; return `<a class="msg-card" href="#idea-${esc(i.id)}" style="text-decoration:none;display:block"><div class="h"><b>harness → ${esc(i.testbed || "untestable")}</b><span>${ago(Date.parse(i.updated), now)}</span></div><p>${esc(i.title)}: ${esc(v(i.verdict).label.toLowerCase())}${i.safety?.diff != null ? ` (${signed(i.safety.diff)})` : ""}</p></a>`; }).join("")}</div>
    <a class="text-link" href="/live">Open the live view →</a>
    <div class="split" aria-label="Tested ideas by verdict">
      <div class="labels"><span>Beat the baseline</span><span>Did not</span></div>
      <div class="bar">${Object.keys(VERDICT).filter(k => k !== "untestable").map(k => { const n = d.tested.filter(i => i.verdict === k).length; return n ? `<i title="${esc(v(k).label)}: ${n}" style="width:${(100 * n / d.tested.length).toFixed(2)}%;background:${v(k).color}"></i>` : ""; }).join("")}</div>
      <div class="counts"><span>${d.wins.length} beat it</span><span>${d.tested.length - d.wins.length} did not</span></div>
      <p class="fine" style="margin:8px 0 0">Counts of tested ideas by verdict. It is not a probability that a direction works, and a testbed is a small stand-in for the real problem.</p>
    </div>
  </aside>`;
  // Commons agent count, if the Commons API answers.
  commonsEvents().then(ev => {
    const active = ev[0] && now - ev[0].t < 864e5;
    ["agentDot", "talkDot"].forEach(id => document.getElementById(id).classList.toggle("live", !!active));
  }).catch(() => {});
}

function tracks(d) {
  const html = d.areas.flatMap(a => a.testbeds).filter(id => d.tb[id]).map(id => {
    const t = d.tb[id], list = d.ideas.filter(i => i.testbed === id && !i.control).sort((x, y) => (x.updated || "").localeCompare(y.updated || ""));
    const wins = list.filter(i => i.verdict === "beats_baseline").length;
    return `<div class="track"><a class="nm" href="#tb-${id}">${esc(t.name)}<small>${esc(d.areaOf[id].name)} · v${esc(t.version || "")}</small></a>
      <div class="rungs">${list.map(i => `<a href="#idea-${esc(i.id)}" title="${esc(i.title)}: ${esc(v(i.verdict).label)}" style="--c:${v(i.verdict).color}"></a>`).join("") || `<span class="flag" style="background:var(--paper);position:relative;z-index:1;padding-right:6px">no ideas yet</span>`}</div>
      <span class="ct">${list.length} idea${list.length === 1 ? "" : "s"} · ${wins} beat</span></div>`;
  }).join("");
  $("#tracks").innerHTML = html;
}

function filters(d) {
  const counts = k => d.ideas.filter(i => !i.control && i.verdict === k).length;
  $("#vfilter").innerHTML = `<button type="button" data-v="all" aria-pressed="true" style="--c:var(--ink)"><i></i>All</button>` +
    Object.entries(VERDICT).filter(([k]) => counts(k)).map(([k, m]) => `<button type="button" data-v="${k}" aria-pressed="false" style="--c:${m.color}" title="${esc(m.note)}"><i></i>${esc(m.label)} · ${counts(k)}</button>`).join("");
  $("#vfilter").addEventListener("click", e => {
    const b = e.target.closest("[data-v]"); if (!b) return;
    state.verdict = b.dataset.v;
    $("#vfilter").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    board(d);
  });
}

function ideaHTML(i, d) {
  const src = i.source || {};
  const flags = i.flags.length ? `<div class="flag">${i.flags.map(esc).join(" · ")}</div>` : "";
  const gate = i.verdict === "mechanism_failed" && i.gate ? `<div class="eff">gate: ${esc(i.gate.quantity)} ${typeof i.gate.observed === "number" ? signed(i.gate.observed) : ""}, needed ${esc(i.gate.rule)}</div>` : "";
  return `<div class="idea" id="idea-${esc(i.id)}">
    <div class="top"><b>${esc(i.title)}</b>${badge(i.verdict)}</div>
    ${effect(i) ? `<div class="eff">safety ${esc(effect(i))}</div>` : gate}
    ${i.untestable_reason ? `<div class="eff">${esc(i.untestable_reason)}</div>` : ""}
    <div class="src">${src.url ? `<a href="${esc(src.url)}" target="_blank" rel="noopener">${esc(src.title || "Source")}</a>` : esc(src.title || "")}${src.authors ? ` · ${esc(src.authors.split(",")[0])}${src.authors.includes(",") ? " et al." : ""}` : ""}${src.year ? ` · ${src.year}` : ""}</div>
    ${flags}
    <details><summary>What it does</summary><p>${esc(i.description)}</p>${i.verdict_v1 && i.verdict_v1 !== i.verdict ? `<p class="fine">Under verdict rule v1 this was ${esc(v(i.verdict_v1).label.toLowerCase())}.</p>` : ""}</details>
  </div>`;
}

function board(d) {
  const match = i => (state.verdict === "all" || i.verdict === state.verdict) &&
    (!state.q || [i.title, i.description, i.source?.title, i.source?.authors, i.testbed, d.tb[i.testbed]?.name].join(" ").toLowerCase().includes(state.q));
  const panels = d.areas.flatMap(a => a.testbeds.map(id => [a, id])).filter(([, id]) => d.tb[id]).map(([a, id]) => {
    const list = d.ideas.filter(i => i.testbed === id && !i.control && match(i));
    if (!list.length) return "";
    const t = d.tb[id];
    return `<section class="tbpanel" id="tb-${id}" style="--c:${areaColor(a.id)}"><p class="eyebrow">${esc(a.name)}</p><h3>${esc(t.name)}</h3><p class="q">${esc(t.question || "")}</p><p class="base">baseline: ${esc(t.baseline || "")}</p>${list.map(i => ideaHTML(i, d)).join("")}</section>`;
  });
  const un = d.ideas.filter(i => !i.testbed && match(i));
  if (un.length) panels.push(`<section class="tbpanel" id="tb-untestable" style="--c:var(--muted)"><p class="eyebrow">Waiting for a testbed</p><h3>Not testable yet</h3><p class="q">Papers no current testbed can test, and what each would need.</p>${un.map(i => ideaHTML(i, d)).join("")}</section>`);
  $("#board").innerHTML = panels.join("") || `<p class="quiet">No idea matches.</p>`;
}

const areaColor = id => AREA_COLORS[id] || "var(--muted)";

function totals(d) {
  const n = k => d.tested.filter(i => i.verdict === k).length;
  const un = d.ideas.filter(i => i.verdict === "untestable").length;
  $("#totals").innerHTML = [
    [d.wins.length, "beat a tuned baseline", "the win counts only if task completion held"],
    [n("mechanism_failed"), "failed their own mechanism", "stopped at the toy gate"],
    [n("hurts"), "made things worse", "significantly below the baseline"],
    [un, "not testable yet", "each says which testbed it needs"],
  ].map(([x, l, s]) => `<div class="total"><b>${x}</b><span>${l}</span><small>${s}</small></div>`).join("");
}

function record(d) {
  const reading = r => /pursue/.test(r) ? "win" : /deprior/.test(r) ? "hurt" : "null";
  $("#signals").innerHTML = `<p class="eyebrow">Direction signals</p><h3>Where to put the next idea</h3>` +
    d.signals.map(s => `<div class="sig"><span>${esc(d.tb[s.testbed]?.name || s.testbed)} <span class="flag">· ${esc(s.topic)}</span></span><span class="r"><span class="badge ${reading(s.reading)}">${esc(s.reading.replace(/ \(.*\)/, ""))}</span> ${s.win_rate?.k ?? 0}/${s.tested} won</span></div>`).join("") +
    `<p class="fine">Combines how often mechanisms reproduce and how often ideas win. With fewer than ${d.min_ideas} ideas a signal is tentative, and so far every one is.</p>`;
  const hurts = d.tested.filter(i => i.verdict === "hurts");
  $("#graveyard").innerHTML = `<p class="eyebrow">The graveyard · in memoriam</p><h3>Ideas that made things worse</h3>` +
    (hurts.map(i => `<div class="sig"><a href="#idea-${esc(i.id)}">${esc(i.title)}</a><span class="r">${esc(signed(i.safety?.diff))}</span></div>`).join("") || `<p class="quiet">Empty so far.</p>`) +
    `<p class="fine">Kept with the effect that buried them. On a planted testbed, not in general.</p>`;
}

function runs(d) {
  const now = Date.now();
  $("#runfeed").innerHTML = d.runs.slice(0, state.shown).map(id => {
    const i = d.byId[id];
    return `<article class="entry"><div class="h"><span><b>harness</b> → ${esc(i.testbed || "triage")}</span><time datetime="${esc(i.updated)}" title="${esc(new Date(i.updated).toLocaleString())}">${ago(Date.parse(i.updated), now)}</time></div>
      <p class="subj"><a href="#idea-${esc(i.id)}" style="text-decoration:none">${esc(i.title)}</a></p>
      <div class="chips">${badge(i.verdict)}${effect(i) ? `<span class="via">${esc(effect(i))}</span>` : ""}${i.runtime_s ? `<span class="via">· ran ${Math.round(i.runtime_s / 60)} min</span>` : ""}</div></article>`;
  }).join("");
  $("#more").hidden = state.shown >= d.runs.length;
}
