"""Prepare the brick-scientist pictures for the scroll story.
Reads the five generated pictures, splits the side-by-side one (hotspots | drivers) at its divider,
and writes six compressed JPEGs to dashboard-project/story-img/ (originals are copied to story-img/original/).
Usage: python dashboard-project/src/prep_story_images.py <folder with 12.png ... 16.png>"""
import pathlib
import shutil
import sys

from PIL import Image

SRC = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(__file__).resolve().parent.parent / "story-img"
(OUT / "original").mkdir(parents=True, exist_ok=True)
for n in range(12, 17):
    shutil.copy(SRC / f"{n}.png", OUT / "original" / f"{n}.png")


def save(im, name):
    im.convert("RGB").save(OUT / name, "JPEG", quality=84, optimize=True)
    print(name, im.size, (OUT / name).stat().st_size // 1024, "KB")


save(Image.open(SRC / "12.png"), "1-satellites.jpg")
save(Image.open(SRC / "13.png"), "2-layers.jpg")
split = Image.open(SRC / "14.png")
save(split.crop((0, 0, 510, split.height)), "3-hotspots.jpg")
save(split.crop((514, 0, split.width, split.height)), "4-drivers.jpg")
save(Image.open(SRC / "15.png"), "5-cooling.jpg")
save(Image.open(SRC / "16.png"), "6-score.jpg")
