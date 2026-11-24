"""Write dashboard-lego.html: the original dashboard with a light brick-style layer on top.
dashboard.html itself is not modified. Run after build_dashboard.py.
Usage: python dashboard-project/src/apply_lego.py"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent
h = (ROOT / "dashboard.html").read_text(encoding="utf-8")
css = (SRC / "lego.css").read_text(encoding="utf-8")
h, n = re.subn(r"<title>[^<]*</title>", "<title>Heat_Map Brick Edition</title>", h, count=1)
assert n == 1, "title not found"
anchor = '<header class="topbar">'
assert anchor in h, "header anchor not found"
h = h.replace(anchor, '<style id="lego">\n' + css + '\n</style>\n' + anchor, 1)
# scroll story: a brick scientist explains the pipeline step by step
story_html = (SRC / "story.html").read_text(encoding="utf-8").strip()
story_js = (SRC / "story.js").read_text(encoding="utf-8").replace("</", "<\\/")
m_anchor = '<section id="method" class="sec-block">'
assert m_anchor in h, "method section not found"
h = h.replace(m_anchor, story_html + "\n\n" + m_anchor, 1)
h = h.replace('<a href="#overview">Overview</a>', '<a href="#overview">Overview</a>\n      <a href="#story">How it works</a>', 1)
h = h.replace("['overview', 'method',", "['overview', 'story', 'method',", 1)
import base64
import json
imgs = sorted((ROOT / "story-img").glob("[0-9]-*.jpg"))
assert imgs, "no story images: run prep_story_images.py first"
data_uris = ["data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii") for p in imgs]
h = h.rstrip() + "\n<script>window.STORY_IMGS = " + json.dumps(data_uris) + ";</script>\n<script>\n" + story_js + "\n</script>\n"
(ROOT / "dashboard-lego.html").write_text(h, encoding="utf-8", newline="\n")
print("wrote dashboard-lego.html", len(h.encode("utf-8")), "bytes")
