"""Insert the real-imagery city explorer into dashboard.html (idempotent).
Hides the synthetic-placeholder heat map card, removes the placeholder banner, adds a text diet.
Run before build_dashboard.py (which inlines data/city_imagery.json)."""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent
P = SRC.parent / "dashboard.html"
h = P.read_text(encoding="utf-8")
# the standalone city card is gone (city scores now live in the pipeline walkthrough); keep only the shared CSS
frag = (SRC / "explorer.html").read_text(encoding="utf-8").split('<div class="card s12" id="xp">')[0].strip()
js = (SRC / "explorer.js").read_text(encoding="utf-8").replace("</", "<\\/")

h = re.sub(r"<!--xp-->.*?<!--xp-end-->\n?", "", h, flags=re.S)
h = re.sub(r"<!--xp-js-->.*?<!--xp-js-end-->\n?", "", h, flags=re.S)
# leftovers of the removed city section (heading + empty wrapper from earlier runs)
h = re.sub(r'<h3 class="sub-h">The held-out cities, from space</h3>\n(?:<div class="g12"></div>\n?)?', "", h)
h = re.sub(r'<style id="diet">.*?</style>\n?', "", h, flags=re.S)

anchor = '<ol class="stage-list" id="stage-list"></ol>'
if anchor not in h:
    sys.exit("anchor not found")
h = h.replace(anchor, anchor + '\n<!--xp-->' + frag + '<!--xp-end-->', 1)
h = h.replace('<div class="card s7" id="v2">', '<div class="card s7" id="v2" hidden style="display:none">', 1)
h = re.sub(r'<div class="ph-banner" role="note">.*?</div>\n</div>\n', "", h, count=1, flags=re.S)

diet = '<style id="diet">.sec-head p,.src,.kpi-cap,.notes,.tbl-note{display:none}.sec-head{margin-bottom:var(--s4)}</style>\n'
h = h.replace('<header class="topbar">', diet + '\n<header class="topbar">', 1)
if 'id="diet"' not in h:
    sys.exit("diet anchor not found")

tail = ('<!--xp-js-->\n<script id="hm-imagery">window.HM_IMAGERY = {};</script>\n<script>\n' + js + '\n</script>\n<!--xp-js-end-->\n')
h = h.rstrip() + "\n" + tail
P.write_text(h, encoding="utf-8", newline="\n")
print("ok", len(h))
