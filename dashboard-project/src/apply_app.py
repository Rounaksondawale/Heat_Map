"""Append the 'how the app works' picture flow (replaces the Azure infrastructure diagram). Idempotent.
Run after apply_hero.py, before build_dashboard.py."""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
P = SRC.parent / "dashboard.html"
h = P.read_text(encoding="utf-8")
js = (SRC / "app.js").read_text(encoding="utf-8").replace("</", "<\\/")
h = re.sub(r"<!--app-js-->.*?<!--app-js-end-->\n?", "", h, flags=re.S)
h = h.rstrip() + "\n<!--app-js-->\n<script>\n" + js + "\n</script>\n<!--app-js-end-->\n"
P.write_text(h, encoding="utf-8", newline="\n")
print("ok", len(h))
