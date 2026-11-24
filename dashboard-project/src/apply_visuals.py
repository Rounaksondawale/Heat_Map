"""Insert the visual replacements for prose (stat strip, tool pictures, finding mini-charts). Idempotent.
Run order: apply_v2.py, apply_explorer.py, apply_visuals.py, then build_dashboard.py."""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent
P = SRC.parent / "dashboard.html"
h = P.read_text(encoding="utf-8")
frag = (SRC / "visuals.html").read_text(encoding="utf-8").strip()
js = (SRC / "visuals.js").read_text(encoding="utf-8").replace("</", "<\\/")

h = re.sub(r"<!--vz-->.*?<!--vz-end-->\n?", "", h, flags=re.S)
h = re.sub(r"<!--vz-js-->.*?<!--vz-js-end-->\n?", "", h, flags=re.S)

anchor = '<div class="cols3">\n      <div class="ruled">\n        <h3>Problem</h3>'
if anchor not in h:
    sys.exit("overview anchor not found")
h = h.replace(anchor, "<!--vz-->" + frag + "<!--vz-end-->\n    " + anchor, 1)
# the README PNGs repeat charts that exist as interactive cards (V4 per city, V5 models, V9 importance, findings card): drop the duplicates
for name in ("fig_model_comparison", "fig_per_city_r2", "fig_vegetation_check", "fig_group_importance"):
    h = re.sub(r'[ \t]*<figure data-name="' + name + r'\.png">.*?</figure>\n?', "", h, flags=re.S)
h = h.rstrip() + "\n<!--vz-js-->\n<script>\n" + js + "\n</script>\n<!--vz-js-end-->\n"
P.write_text(h, encoding="utf-8", newline="\n")
print("ok", len(h))
