"""Inline the dashboard data into dashboard.html (in place, idempotent).

What it fills:
  * the data-only <script id="hm-data"> block, window.HM_DATA = {metrics, perCity, cooling,
    cities, model, figures, insights, grid}, from data/*.json (the page cannot fetch files;
    the block holds no logic so it can later move to data.js unchanged);
  * <img data-fig="fig_x.png" src="..."> from docs/fig_x.png as data: URIs,
    only while the whole page stays under 3 MB (otherwise src is left empty and
    the page links each figure to GitHub instead);
  * the "Data last updated" stamp from the last git commit that touched README.md.

insights.json is embedded without its `kpis` and `insights` arrays: their copy
uses wording the dashboard must not show ("unseen"), and the page writes its own
KPI and finding text from the same numbers.

Usage:  python dashboard-project/build_dashboard.py
Standard library only.
"""
import base64
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
HTML = HERE / "dashboard.html"
DATA = HERE / "data"
DOCS = REPO / "docs"
MAX_BYTES = 5 * 1024 * 1024  # city imagery (~1 MB) is part of the page; hard host limit is 16 MB

NAMES = ["metrics", "per_city", "cooling", "cities", "model", "figures", "insights", "placeholder_grid"]
# data/<file>.json -> key in window.HM_DATA
KEYS = {"metrics": "metrics", "per_city": "perCity", "cooling": "cooling", "cities": "cities", "model": "model",
        "figures": "figures", "insights": "insights", "placeholder_grid": "grid"}
BANNED = ("unseen", "never seen")


def load(name):
    doc = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    if name == "insights":
        doc.pop("kpis", None)
        doc.pop("insights", None)
    return doc


def inline_data(html):
    """Write the data-only <script id="hm-data"> block: window.HM_DATA = {...}; and nothing else."""
    parts = []
    for name in NAMES:
        payload = json.dumps(load(name), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        low = payload.lower()
        for word in BANNED:
            if word in low:
                sys.exit(f"{name}.json contains banned wording {word!r}; fix the data or strip the field here")
        parts.append(f"  {json.dumps(KEYS[name])}: {payload}")
    block = "window.HM_DATA = {\n" + ",\n".join(parts) + "\n};"
    pat = re.compile(r'(<script id="hm-data">)(.*?)(</script>)', re.S)
    html, n = pat.subn(lambda m: m.group(1) + block + m.group(3), html)
    if n != 1:
        sys.exit(f'expected exactly one <script id="hm-data"> block, found {n}')
    img = DATA / "city_imagery.json"
    if img.exists():
        payload = json.dumps(json.loads(img.read_text(encoding="utf-8")), separators=(",", ":")).replace("</", "<\\/")
        pat2 = re.compile(r'(<script id="hm-imagery">)(.*?)(</script>)', re.S)
        html, n2 = pat2.subn(lambda m: m.group(1) + "window.HM_IMAGERY = " + payload + ";" + m.group(3), html)
        if n2 != 1:
            sys.exit("expected exactly one hm-imagery block (run src/apply_explorer.py first)")
    return html


IMG = re.compile(r'(<img data-fig="(fig_[\w-]+\.png)" src=")([^"]*)(")')


def set_figs(html, embed):
    def repl(m):
        src = ""
        if embed:
            src = "data:image/png;base64," + base64.b64encode((DOCS / m.group(2)).read_bytes()).decode("ascii")
        return m.group(1) + src + m.group(4)
    return IMG.sub(repl, html)


def stamp(html):
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%cs %h", "--", "README.md"],
                             capture_output=True, text=True, check=True).stdout.split()
        date, short = out[0], out[1]
    except Exception as exc:  # keep the existing stamp if git is unavailable
        print(f"warning: git stamp not updated ({exc})")
        return html
    html = re.sub(r'<time id="stamp-date" datetime="[^"]*">[^<]*</time>',
                  f'<time id="stamp-date" datetime="{date}">{date}</time>', html)
    html = re.sub(r'<code id="stamp-hash">[^<]*</code>', f'<code id="stamp-hash">{short}</code>', html)
    html = re.sub(r'<code class="stamp-hash-copy">[^<]*</code>', f'<code class="stamp-hash-copy">{short}</code>', html)
    return html


def main():
    html = HTML.read_text(encoding="utf-8")
    html = inline_data(html)
    html = stamp(html)
    with_figs = set_figs(html, True)
    if len(with_figs.encode("utf-8")) <= MAX_BYTES:
        html = with_figs
        print("figures embedded as data: URIs")
    else:
        html = set_figs(html, False)
        print("page would exceed 3 MB with figures; figures link to GitHub instead")
    HTML.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {HTML} ({len(html.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
