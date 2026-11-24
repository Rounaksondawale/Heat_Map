"""Fetch a real Sentinel-2 cloudless image (EOX, CC BY-NC 4.0) of each held-out city's
40 km box and store it as data/city_imagery.json (base64 JPEG). Needs network + Pillow."""
import base64, io, json, math, pathlib, urllib.request
from PIL import Image
HERE = pathlib.Path(__file__).resolve().parent
HALF_KM, PX = 20, 640
cities = json.load(open(HERE / "cities.json"))["cities"]
out = {}
for c in cities:
    if c["split"] != "val":
        continue
    dlat = HALF_KM / 110.54
    dlon = HALF_KM / (111.32 * math.cos(math.radians(c["lat"])))
    bbox = (c["lon"] - dlon, c["lat"] - dlat, c["lon"] + dlon, c["lat"] + dlat)
    url = ("https://tiles.maps.eox.at/wms?service=WMS&version=1.1.1&request=GetMap&layers=s2cloudless-2020"
           "&styles=&srs=EPSG:4326&bbox=%s&width=1280&height=1280&format=image/jpeg" % ",".join(f"{v:.5f}" for v in bbox))
    img = Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=90).read())).convert("RGB").resize((PX, PX), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=62, optimize=True)
    out[c["city"]] = {"lon": c["lon"], "lat": c["lat"], "bbox": [round(v, 5) for v in bbox],
                      "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}
    print(c["city"], len(buf.getvalue()) // 1024, "KB")
out["_source"] = "Sentinel-2 cloudless 2020 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2020), CC BY-NC 4.0; 40 km box = the study area"
json.dump(out, open(HERE / "city_imagery.json", "w"), separators=(",", ":"))
