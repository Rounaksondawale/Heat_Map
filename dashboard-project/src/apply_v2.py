"""Apply the v2 visual layer to dashboard.html (in place, idempotent).

- adds the Bricolage Grotesque display font to the Google Fonts import
- inserts src/fun.css before </style> (between /*v2-css*/ markers)
- replaces the hero <section id="top"> with src/hero.html
- appends Three.js r128 (cdnjs) + src/three-views.js at the end (between <!--v2-js--> markers)

Usage: python dashboard-project/src/apply_v2.py
"""
import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parent
HTML = SRC.parent / "dashboard.html"
THREE = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'

html = HTML.read_text(encoding="utf-8")
css = (SRC / "fun.css").read_text(encoding="utf-8")
hero = (SRC / "hero.html").read_text(encoding="utf-8").strip()
js = (SRC / "three-views.js").read_text(encoding="utf-8")

# 1. font
old_font = "family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"
new_font = "family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"
if new_font not in html:
    if old_font not in html:
        sys.exit("font import not found")
    html = html.replace(old_font, new_font, 1)

# 2. css
html = re.sub(r"/\*v2-css\*/.*?/\*v2-css-end\*/\n?", "", html, flags=re.S)
if "</style>" not in html:
    sys.exit("no </style> found")
html = html.replace("</style>", "/*v2-css*/" + css + "/*v2-css-end*/\n</style>", 1)

# 3. hero
pat = re.compile(r'<section id="top" class="hero">.*?</section>', re.S)
html, n = pat.subn(lambda m: hero, html, count=1)
if n != 1:
    sys.exit("hero section not found")

# 4. scripts
html = re.sub(r"<!--v2-js-->.*?<!--v2-js-end-->\n?", "", html, flags=re.S)
html = html.rstrip() + "\n<!--v2-js-->\n" + THREE + "\n<script>\n" + js.replace("</", "<\\/") + "\n</script>\n<!--v2-js-end-->\n"

HTML.write_text(html, encoding="utf-8")
print(f"wrote {HTML} ({HTML.stat().st_size/1024:.0f} KB)")
