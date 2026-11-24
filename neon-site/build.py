"""Build neon-site/index.html from template.html and the dashboard's data files.
Every number on the site comes from dashboard-project/data/*.json (README tables, per-city scores,
cooling doses) plus the Sentinel-2 city images and the India outline fetched for the dashboard.
Usage: python neon-site/build.py          (standard library only)
"""
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "dashboard-project" / "data"
KX = math.cos(math.radians(21))


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


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


def india():
    geo = load("ind_geo.json")
    polys = geo["coordinates"] if geo["type"] == "MultiPolygon" else [geo["coordinates"]]
    rings = []
    for poly in polys:
        ring = poly[0]
        if len(ring) < 40:
            continue
        pts = [(p[0], p[1]) for p in ring]
        k = max(range(len(pts)), key=lambda i: math.hypot((pts[i][0] - pts[0][0]) * KX, pts[i][1] - pts[0][1]))
        simp = rdp(pts[:k + 1], 0.035)[:-1] + rdp(pts[k:], 0.035)
        rings.append([[round(x, 2), round(y, 2)] for x, y in simp])
    return rings


metrics = load("metrics.json")
per_city = load("per_city.json")
cooling = load("cooling.json")
cities = load("cities.json")["cities"]
imagery = load("city_imagery.json")
imagery.pop("_source", None)

payload = {
    "perCity": per_city["cities"],
    "mean": per_city["mean"],
    "models": metrics["models"]["rows"],
    "groups": metrics["group_importance"],
    "veg": metrics["vegetation_check"]["per_0_1_NDVI_cooling_C"],
    "hotspot": metrics["hotspot_detection"]["cities"],
    "landcover": metrics["landcover"]["rows"],
    "cooling": cooling["rows"],
    "cities": [{"city": c["city"], "lon": c["lon"], "lat": c["lat"], "split": c["split"], "climate": c["climate"]} for c in cities],
    "imagery": imagery,
    "india": india(),
}
blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text(encoding="utf-8")
assert "/*DATA*/" in html
html = html.replace("/*DATA*/null", blob, 1)
(HERE / "index.html").write_text(html, encoding="utf-8", newline="\n")
print(f"wrote {HERE / 'index.html'} ({len(html.encode('utf-8')):,} bytes)")
