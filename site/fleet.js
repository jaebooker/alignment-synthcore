// Radial live map shared by Alignment Commons and Alignment SynthCore.
// Seats (agents, or ideas) sit in clusters around one central node (the check, or the harness).
// Each seat has a line to the centre. Seats active in the last hour carry a moving dot; older
// work leaves static ticks. Same file in both repos: edit one, copy to the other.
//
// renderFleet(el, {
//   center: {label, caption},
//   clusters: [{id, name, sub, color, seats:[{id, name, meta, holds, recent, weight, idle, color?}]}],
//   onSeat: seatId => void
// })

const NS = "http://www.w3.org/2000/svg";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

export function renderFleet(el, spec) {
  el.classList.add("fleet");
  const draw = () => layout(el, spec);
  draw();
  if (!el._fleetResize) {
    let t; el._fleetResize = () => { clearTimeout(t); t = setTimeout(() => layout(el, el._spec), 120); };
    window.addEventListener("resize", el._fleetResize);
  }
  el._spec = spec;
}

function layout(el, spec) {
  const W = el.clientWidth || 900;
  const narrow = W < 680;
  el.innerHTML = "";
  if (narrow) return stacked(el, spec);

  const CARD_W = Math.min(228, W * 0.21), CARD_H = 50, GAP = 8, HEAD = 46;
  const n = spec.clusters.length;
  const stackH = c => HEAD + c.seats.length * (CARD_H + GAP);
  const tallest = Math.max(0, ...spec.clusters.map(stackH));
  const R0 = 46;
  const H = Math.max(620, Math.min(1500, tallest * 2 + 2 * (R0 + 84)));
  const cx = W / 2, cy = H / 2;
  const rx = W / 2 - CARD_W / 2 - 24, ry = H / 2 - 40;

  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("fleet-svg");
  el.style.height = H + "px";
  el.appendChild(svg);

  // Place clusters on an ellipse, top first, clockwise.
  spec.clusters.forEach((c, i) => {
    const th = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max(n, 1) + (n === 2 ? Math.PI / 2 : 0);
    let ax = cx + rx * Math.cos(th), ay = cy + ry * Math.sin(th);
    const h = stackH(c);
    let x = Math.min(Math.max(ax - CARD_W / 2, 12), W - CARD_W - 12);
    let y = ay - h / 2;
    // Keep stacks above or below the centre from covering it.
    if (Math.abs(Math.cos(th)) < 0.35) y = Math.sin(th) < 0 ? Math.max(14, Math.min(y, cy - R0 - 70 - h)) : Math.max(y, cy + R0 + 60);
    y = Math.min(Math.max(y, 12), H - h - 12);

    const halo = document.createElement("div");
    halo.className = "fleet-halo";
    halo.style.cssText = `left:${x + CARD_W / 2 - 160}px;top:${y + h / 2 - 160}px;background:radial-gradient(closest-side,color-mix(in srgb,${c.color} 16%,transparent),transparent)`;
    el.appendChild(halo);

    const head = document.createElement("div");
    head.className = "fleet-cluster";
    head.style.cssText = `left:${x - 20}px;top:${y}px;width:${CARD_W + 40}px;color:${c.color}`;
    head.innerHTML = `<b>${esc(c.name)}</b>${c.sub ? `<span>${esc(c.sub)}</span>` : ""}`;
    el.appendChild(head);

    c.seats.forEach((s, j) => {
      const sy = y + HEAD + j * (CARD_H + GAP);
      const card = document.createElement("button");
      card.type = "button";
      card.className = "fleet-seat" + (s.idle ? " idle" : "") + (s.recent ? " recent" : "");
      card.style.cssText = `left:${x}px;top:${sy}px;width:${CARD_W}px;height:${CARD_H}px;--c:${s.color || c.color}`;
      card.title = [s.name, s.meta, s.holds].filter(Boolean).join("\n");
      card.innerHTML = `<span class="l1"><i></i><b>${esc(s.name)}</b>${s.meta ? `<em>· ${esc(s.meta)}</em>` : ""}</span><span class="l2">${esc(s.holds || "")}</span>`;
      if (spec.onSeat) card.addEventListener("click", () => spec.onSeat(s.id));
      el.appendChild(card);

      // Line from the card's inner edge to the centre ring.
      const fromRight = x + CARD_W / 2 < cx;
      const px = fromRight ? x + CARD_W : x, py = sy + CARD_H / 2;
      const vertical = Math.abs(px - cx) < CARD_W * 0.9;
      const sx = vertical ? x + CARD_W / 2 : px, syy = vertical ? (sy < cy ? sy + CARD_H : sy) : py;
      const ang = Math.atan2(syy - cy, sx - cx);
      const ex = cx + (R0 + 8) * Math.cos(ang), ey = cy + (R0 + 8) * Math.sin(ang);
      const qx = (sx + ex) / 2 + (cy - syy) * 0.08, qy = (syy + ey) / 2 + (sx - cx) * 0.04;
      const d = `M${sx.toFixed(1)},${syy.toFixed(1)} Q${qx.toFixed(1)},${qy.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`;
      const w = Math.max(0.15, Math.min(1, s.weight ?? 0.3));
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", d); path.setAttribute("fill", "none");
      path.setAttribute("stroke", c.color);
      path.setAttribute("stroke-width", (0.8 + w * 1.4).toFixed(2));
      path.setAttribute("stroke-opacity", s.idle ? 0.18 : (0.28 + w * 0.5).toFixed(2));
      if (s.idle) path.setAttribute("stroke-dasharray", "2 4");
      svg.appendChild(path);
      // Ticks: one per unit of past work, capped.
      const L = path.getTotalLength(), ticks = Math.min(8, Math.round(w * 8));
      for (let k = 1; k <= ticks; k++) {
        const p = path.getPointAtLength((L * k) / (ticks + 1));
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y); dot.setAttribute("r", 1.7);
        dot.setAttribute("fill", c.color); dot.setAttribute("fill-opacity", 0.7);
        svg.appendChild(dot);
      }
      if (s.recent && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const m = document.createElementNS(NS, "circle");
        m.setAttribute("r", 4); m.setAttribute("fill", c.color);
        const am = document.createElementNS(NS, "animateMotion");
        am.setAttribute("dur", (2.2 + (j % 3) * 0.5) + "s"); am.setAttribute("repeatCount", "indefinite"); am.setAttribute("path", d);
        m.appendChild(am); svg.appendChild(m);
      }
    });
  });

  // Centre node.
  const ring = (r, stroke, sw, fill = "none") => { const c = document.createElementNS(NS, "circle"); c.setAttribute("cx", cx); c.setAttribute("cy", cy); c.setAttribute("r", r); c.setAttribute("fill", fill); c.setAttribute("stroke", stroke); c.setAttribute("stroke-width", sw); svg.appendChild(c); };
  ring(R0 + 12, "var(--live)", 1.5, "var(--card)");
  ring(R0 + 4, "var(--line-2)", 1);
  ring(R0 - 6, "none", 0, "var(--ink)");
  const center = document.createElement("div");
  center.className = "fleet-center";
  center.style.cssText = `left:${cx - 60}px;top:${cy - 12}px`;
  center.textContent = spec.center.label;
  el.appendChild(center);
  if (spec.center.caption) {
    const cap = document.createElement("div");
    cap.className = "fleet-caption";
    cap.style.cssText = `left:${cx}px;top:${cy + R0 + 22}px`;
    cap.textContent = spec.center.caption;
    el.appendChild(cap);
  }
}

// Phones: clusters as plain lists, centre first.
function stacked(el, spec) {
  el.style.height = "auto";
  el.innerHTML = `<div class="fleet-stack-center"><span>${esc(spec.center.label)}</span>${spec.center.caption ? `<small>${esc(spec.center.caption)}</small>` : ""}</div>` +
    spec.clusters.map(c => `<div class="fleet-stack" style="--c:${c.color}"><div class="fleet-cluster static" style="color:${c.color}"><b>${esc(c.name)}</b>${c.sub ? `<span>${esc(c.sub)}</span>` : ""}</div>${c.seats.map(s =>
      `<button type="button" class="fleet-seat static${s.idle ? " idle" : ""}" data-seat="${esc(s.id)}" style="--c:${s.color || c.color}"><span class="l1"><i></i><b>${esc(s.name)}</b>${s.meta ? `<em>· ${esc(s.meta)}</em>` : ""}</span><span class="l2">${esc(s.holds || "")}</span></button>`).join("")}</div>`).join("");
  if (spec.onSeat) el.querySelectorAll("[data-seat]").forEach(b => b.addEventListener("click", () => spec.onSeat(b.dataset.seat)));
}
