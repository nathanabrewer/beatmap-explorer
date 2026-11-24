"""Build a one-page HTML results report from a small JSON description.

    python3 -I train/report.py spec.json out.html

spec.json:
    {"title": "...", "subtitle": "...", "notes": ["..."],
     "curves": [{"title": "...", "log": "runs/x/log.jsonl", "y": "heldout", "yLabel": "held-out loss"}],
     "bars": [{"title": "...", "unit": "timing F1", "rows": [["mapgen", 0.16], ["model", 0.20]]}],
     "tables": [{"title": "...", "columns": ["a", "b"], "rows": [["x", 1]]}],
     "images": [{"src": "shots/a.png", "caption": "..."}]}
Paths are relative to spec.json; images are copied beside the report. Charts
are inline SVG (one series each, so no legend; the title names it), with a
hover readout, light and dark colours, and every number also in a table.
"""
import html
import json
import os
import shutil
import sys

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
--text-muted:#77766f;--grid:#e4e3de;--series-1:#2a78d6;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
background:var(--surface-1);color:var(--text-primary);min-height:100vh;padding:28px 20px;box-sizing:border-box}
.wrap{max-width:980px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#242423;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#33332f;--series-1:#3987e5}}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}.sub{color:var(--text-secondary);margin:0 0 18px}
.card{background:var(--surface-2);border-radius:10px;padding:14px 16px;margin:12px 0}
svg text{fill:var(--text-secondary);font-size:12px}svg .val{fill:var(--text-primary);font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}.line{fill:none;stroke:var(--series-1);stroke-width:2}
.bar{fill:var(--series-1)}.dot{fill:var(--series-1);stroke:var(--surface-2);stroke-width:2}
.tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);
border-radius:6px;padding:4px 8px;font-size:12px;display:none}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:12px 0}figure img{width:100%;border-radius:8px}figcaption{color:var(--text-secondary);font-size:13px;margin-top:4px}
ul.notes{color:var(--text-secondary)}details summary{cursor:pointer;color:var(--text-secondary);font-size:13px}
"""

JS = """
const tip=document.querySelector('.tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
  tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'});
 el.addEventListener('mouseleave',()=>{tip.style.display='none'})});
"""


def esc(s):
    return html.escape(str(s))


def nice_ticks(lo, hi, n=4):
    span = (hi - lo) or 1.0
    step = 10 ** int(__import__("math").floor(__import__("math").log10(span / n)))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = int(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 10))
        t += step
    return ticks


def line_chart(points, y_label):
    """points: [(x, y)] -> SVG with a recessive grid, a 2px line, hover dots."""
    W, H, L, R, T, B = 900, 280, 56, 16, 12, 52
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.08 or 0.1
    y0, y1 = y0 - pad, y1 + pad
    sx = lambda x: L + (x - x0) / ((x1 - x0) or 1) * (W - L - R)
    sy = lambda y: T + (1 - (y - y0) / ((y1 - y0) or 1)) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(y_label)} by training step">']
    for t in nice_ticks(y0, y1, 6):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/><text x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in nice_ticks(x0, x1, 6):
        out.append(f'<text x="{sx(t):.1f}" y="{H - 30}" text-anchor="middle">{int(t):,}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 6}" text-anchor="middle">training step</text>')
    out.append('<path class="line" d="' + " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(points)) + '"/>')
    for x, y in points:
        out.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" data-tip="step {int(x):,}: {y_label} {y:.3f}"/>')
    lx, ly = points[-1]
    out.append(f'<text class="val" x="{sx(lx) - 8:.1f}" y="{sy(ly) - 10:.1f}" text-anchor="end">{ly:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(rows, unit):
    """rows: [(label, value)] -> horizontal bars from a zero baseline, value labels."""
    W, row_h, L, R = 900, 34, 230, 70
    vmax = max((v for _, v in rows), default=1) or 1
    H = row_h * len(rows) + 8
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(unit)} by method">']
    for i, (label, v) in enumerate(rows):
        y = 4 + i * row_h
        w = (W - L - R) * v / vmax
        out.append(f'<text x="{L - 10}" y="{y + 21}" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect class="bar" x="{L}" y="{y + 6}" width="{max(w, 2):.1f}" height="{row_h - 12}" rx="4" data-tip="{esc(label)}: {unit} {v:.3f}"/>')
        out.append(f'<text class="val" x="{L + w + 8:.1f}" y="{y + 21}">{v:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def table(columns, rows):
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f'<td class="{"n" if isinstance(v, (int, float)) else ""}">{esc(f"{v:.3f}" if isinstance(v, float) else v)}</td>' for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    base = os.path.dirname(os.path.abspath(spec_path))
    spec = json.load(open(spec_path))
    if spec.get("public"):
        # A public report may not draw on official or extracted material.
        banned = ("official", "ost-raw", "dlc-raw", "extracted", "/levels/", "decisionrhythm-private/songs")
        for item in spec.get("curves", []) + spec.get("images", []):
            p = os.path.abspath(os.path.join(base, item.get("log") or item.get("src"))).lower()
            if any(b in p for b in banned):
                raise SystemExit(f"public report refuses {p}")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    parts = [f"<h1>{esc(spec.get('title', 'Results'))}</h1><p class='sub'>{esc(spec.get('subtitle', ''))}</p>"]
    for c in spec.get("curves", []):
        rows = [json.loads(l) for l in open(os.path.join(base, c["log"])) if l.strip()]
        pts = [(r["step"], r[c.get("y", "heldout")]) for r in rows if isinstance(r.get(c.get("y", "heldout")), (int, float)) and r[c.get("y", "heldout")] == r[c.get("y", "heldout")]]
        if not pts:
            continue
        parts.append(f"<h2>{esc(c['title'])}</h2><div class='card'>{line_chart(pts, c.get('yLabel', 'loss'))}"
                     f"<details><summary>Table</summary>{table(['step', c.get('yLabel', 'loss')], [[p[0], float(p[1])] for p in pts])}</details></div>")
    for b in spec.get("bars", []):
        rows = [(r[0], float(r[1])) for r in b["rows"]]
        parts.append(f"<h2>{esc(b['title'])}</h2><div class='card'>{bar_chart(rows, b.get('unit', ''))}"
                     f"<details><summary>Table</summary>{table(['method', b.get('unit', '')], [[r[0], r[1]] for r in rows])}</details></div>")
    for t in spec.get("tables", []):
        parts.append(f"<h2>{esc(t['title'])}</h2><div class='card'>{table(t['columns'], t['rows'])}</div>")
    for im in spec.get("images", []):
        src = os.path.join(base, im["src"])
        name = os.path.basename(src)
        if os.path.abspath(os.path.join(out_dir, name)) != os.path.abspath(src):
            shutil.copyfile(src, os.path.join(out_dir, name))
        parts.append(f"<figure><img src='{esc(name)}' alt='{esc(im.get('caption', ''))}'><figcaption>{esc(im.get('caption', ''))}</figcaption></figure>")
    if spec.get("notes"):
        parts.append("<h2>Notes</h2><ul class='notes'>" + "".join(f"<li>{esc(n)}</li>" for n in spec["notes"]) + "</ul>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{esc(spec.get('title', 'Results'))}</title><style>{CSS}</style></head><body style='margin:0'>"
            f"<div class='viz-root'><div class='wrap'>{''.join(parts)}</div></div><div class='tip'></div><script>{JS}</script></body></html>")
    open(out_path, "w").write(page)
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
