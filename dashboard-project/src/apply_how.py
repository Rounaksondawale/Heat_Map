"""Add the interactive 'how it works' walkthrough in place of the pipeline flowchart. Idempotent.
Run after apply_visuals.py, before build_dashboard.py."""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parent
P = SRC.parent / "dashboard.html"
h = P.read_text(encoding="utf-8")
css = (SRC / "how.html").read_text(encoding="utf-8").strip()
js = (SRC / "how.js").read_text(encoding="utf-8").replace("</", "<\\/")
h = re.sub(r"<!--hw-css-->.*?<!--hw-css-end-->\n?", "", h, flags=re.S)
h = re.sub(r"<!--hw-js-->.*?<!--hw-js-end-->\n?", "", h, flags=re.S)
h = h.replace('<div class="diagram" id="pipeline-diagram"></div>', "<!--hw-css-->" + css + "<!--hw-css-end-->\n    <div class=\"diagram\" id=\"pipeline-diagram\"></div>", 1)
h = h.rstrip() + "\n<!--hw-js-->\n<script>\n" + js + "\n</script>\n<!--hw-js-end-->\n"
P.write_text(h, encoding="utf-8", newline="\n")
print("ok", len(h))
