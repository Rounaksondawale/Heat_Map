"""Add the India hero map (replaces the 3D towers) and hide the old V3 city-map card it makes redundant.
Needs data/ind_geo.json (India outline from Natural Earth, public domain). Idempotent.
Run last, after apply_how.py and before build_dashboard.py."""
import json
import math
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
P = SRC.parent / "dashboard.html"
KX = math.cos(math.radians(21))


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    dx, dy = (x2 - x1) * KX, y2 - y1
    norm = math.hypot(dx, dy) or 1e-9
    dmax, idx = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * (pts[i][0] - x1) * KX - dx * (pts[i][1] - y1)) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[:idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


geo = json.loads((SRC.parent / "data" / "ind_geo.json").read_text(encoding="utf-8"))
polys = geo["coordinates"] if geo["type"] == "MultiPolygon" else [geo["coordinates"]]
rings = []
for poly in polys:
    ring = poly[0]
    if len(ring) < 40:   # drop tiny islets, keep the mainland and the Andaman group
        continue
    pts = [(p[0], p[1]) for p in ring]
    k = max(range(len(pts)), key=lambda i: math.hypot((pts[i][0] - pts[0][0]) * KX, pts[i][1] - pts[0][1]))  # closed ring: split at the farthest point
    simp = rdp(pts[:k + 1], 0.035)[:-1] + rdp(pts[k:], 0.035)
    rings.append([[round(x, 2), round(y, 2)] for x, y in simp])
outline = json.dumps(rings, separators=(",", ":"))
js = (SRC / "hero_map.js").read_text(encoding="utf-8").replace("/*INDIA_OUTLINE*/[]/*END*/", outline).replace("</", "<\\/")

h = P.read_text(encoding="utf-8")
h = re.sub(r"<!--hm-js-->.*?<!--hm-js-end-->\n?", "", h, flags=re.S)
h = h.replace('<div class="card s5" id="v3">', '<div class="card s5" id="v3" hidden style="display:none">', 1)
h = h.rstrip() + "\n<!--hm-js-->\n<script>\n" + js + "\n</script>\n<!--hm-js-end-->\n"
P.write_text(h, encoding="utf-8", newline="\n")
print("ok", sum(len(r) for r in rings), "outline points in", len(rings), "rings")
