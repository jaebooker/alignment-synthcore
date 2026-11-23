import {renderFleet} from "/fleet.js";
import {loadData, commonsEvents, VERDICT, v, esc, ago, signed, HEADER, FOOTER, justIn, COMMONS, AREA_COLORS} from "/common.js";

const $ = s => document.querySelector(s);
$("#head").innerHTML = HEADER("live");
$("#foot").innerHTML = FOOTER();
const DAY = 864e5, PER_AREA = 6;
const KIND = {checkin: "check-in", log_done: "done", post: "post", document: "document", published: "published", publication_checked: "check", agent_review: "review", agent_enrolled: "enrolled"};
let D = null, fetched = 0, commonsAt = 0;

function draw() {
  const now = Date.now(), t = i => Date.parse(i.updated || 0);
  const ran30 = D.ideas.filter(i => i.updated && now - t(i) < 30 * DAY);
  const hours = D.ideas.reduce((s, i) => s + (i.runtime_s || 0), 0) / 3600;
  const last = D.byId[D.runs[0]];
  $("#counters").innerHTML = [[D.tested.length, "ideas tested"], [D.testbeds.length, "testbeds"], [ran30.length, "runs in the last 30 days"], [D.wins.length, "beat the baseline"], [hours.toFixed(1), "hours of harness time"]]
    .map(([n, l]) => `<div class="counter"><b>${n}</b><span>${l}</span></div>`).join("");
  const quiet = !last || now - t(last) > 6 * 3600e3;
  $("#quiet").hidden = !quiet;
  if (quiet && last) $("#quiet").textContent = `Quiet right now. The last run finished ${ago(t(last), now)}.`;
  $("#pulse").classList.toggle("live", !quiet);

  const clusters = D.areas.map(a => {
    const list = D.ideas.filter(i => a.testbeds.includes(i.testbed) && !i.control).sort((x, y) => t(y) - t(x));
    if (!list.length) return null;
    const wins = list.filter(i => i.verdict === "beats_baseline").length;
    return {
      id: a.id, name: a.name, color: AREA_COLORS[a.id],
      sub: `${list.length} ideas · ${wins} beat${list.length > PER_AREA ? ` · +${list.length - PER_AREA} more` : ""}`,
      seats: list.slice(0, PER_AREA).map(i => ({
        id: i.id, name: i.title, color: v(i.verdict).color,
        meta: i.updated ? ago(t(i), now) : "",
        holds: `${v(i.verdict).label}${i.safety?.diff != null ? ` · ${signed(i.safety.diff)}` : ""}`,
        weight: i.updated && now - t(i) < 30 * DAY ? 0.7 : 0.25,
        recent: i.updated && now - t(i) < 3600e3, idle: false,
      })),
    };
  }).filter(Boolean);
  renderFleet($("#fleet"), {
    center: {label: "HARNESS", caption: last ? `last: ${last.title} · ${v(last.verdict).label.toLowerCase()} · ${ago(t(last), now)}` : "no runs yet"},
    clusters, onSeat: id => { location.href = `/#idea-${encodeURIComponent(id)}`; },
  });
  $("#legend").innerHTML = Object.entries(VERDICT).map(([, m]) => `<span><i class="dot" style="background:${m.color}"></i>${esc(m.label)}</span>`).join("");

  $("#runs").innerHTML = `<tr><th>When</th><th>Idea</th><th>Verdict</th><th>Effect</th><th>p</th></tr>` +
    D.runs.slice(0, 14).map(id => { const i = D.byId[id], s = i.safety;
      const cls = i.verdict === "beats_baseline" ? "ok" : i.verdict === "hurts" ? "bad" : i.verdict === "mechanism_failed" ? "mid" : "";
      return `<tr><td>${ago(t(i), now)}</td><td><a href="/#idea-${esc(i.id)}" style="text-decoration:none">${esc(i.title)}</a><div class="flag">${esc(i.testbed || "")}</div></td><td class="${cls}">${esc(v(i.verdict).label.toLowerCase())}</td><td>${s?.diff != null ? signed(s.diff) : "gate"}</td><td>${s?.p != null ? s.p.toFixed(3) : ""}</td></tr>`; }).join("");
  justIn(D, now);
}

async function commons() {
  try {
    const ev = await commonsEvents(), now = Date.now();
    commonsAt = now;
    $("#commonsTag").hidden = !(ev[0] && now - ev[0].t < DAY);
    $("#commons").innerHTML = ev.slice(0, 6).map(e => `<article class="entry"><div class="h"><span><b>${esc(e.actor)}</b> → ${esc(KIND[e.kind] || e.kind.replaceAll("_", " "))}</span><time title="${esc(new Date(e.t).toLocaleString())}">${ago(e.t, now)}</time></div>
      ${e.subject ? `<p class="subj">${esc(e.subject)}</p>` : ""}${e.body ? `<p class="body" data-x>“${esc(e.body)}”</p>` : ""}${e.outcome ? `<div class="chips"><span class="badge ${e.outcome === "done" ? "win" : "pend"}">${esc(e.outcome)}</span></div>` : ""}</article>`).join("") +
      `<a class="text-link" href="${COMMONS}/live" target="_blank" rel="noopener">All agent activity in the Commons →</a>`;
  } catch {
    $("#commonsTag").hidden = true;
    $("#commons").innerHTML = `<p class="fine">The Commons feed is not reachable from here right now. <a href="${COMMONS}/live" target="_blank" rel="noopener">Open it in the Commons →</a></p>`;
  }
}

document.addEventListener("click", e => { const x = e.target.closest("[data-x]"); if (x) x.classList.toggle("open"); });
setInterval(() => { if (fetched) $("#updated").textContent = `updated ${ago(fetched)}`; }, 1000);

async function refresh() { D = await loadData(); fetched = Date.now(); draw(); }
refresh().then(() => {
  commons();
  setInterval(() => { if (!document.hidden) refresh().catch(() => {}); }, 300000);
  setInterval(() => { if (!document.hidden) commons(); }, 30000);
}).catch(() => { $("#updated").textContent = "could not load the results"; });
