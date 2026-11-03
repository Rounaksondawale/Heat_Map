"""
Urban Heat Hotspots dashboard.

  Anywhere in India (live): a map of India. Draw a rectangle on any area and press Analyse. The app downloads the satellite data for that
  area (Landsat 8/9, elevation, land cover, buildings), derives the model's inputs exactly as for the 50 prepared cities, runs the
  trained U-Net, and shows the hotspots, what drives them and how much planting / reflective roofing would close the gap.
  Prepared cities: the same analysis on the 50 cities that were processed in advance (fast, no downloads).

    pip install -r dashboard/requirements.txt          # on top of the project requirements and PyTorch
    streamlit run dashboard/app.py

Needs (all in the Deploy/ folder, see Deploy/README.md and dashboard/paths.py; paths can be changed in the sidebar):
    Deploy/best.pt      the trained model
    Deploy/meta.json    feature means / spreads of the training run
    Deploy/grids/       <City>_features.nc and <City>_temp.nc, only for the "Prepared cities" mode
Live mode needs internet access (Microsoft Planetary Computer and the JRC GHSL server).
"""
import json
import sys
from pathlib import Path

import folium
import matplotlib
import numpy as np
import streamlit as st
from folium.plugins import Draw, Fullscreen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import live as LIVE  # noqa: E402
import paths as PATHS  # noqa: E402
import plots as PL  # noqa: E402

try:
    from streamlit_folium import st_folium
except ImportError:                                            # the manual box still works without the map widget
    st_folium = None

ROOT = E.ROOT
DEF = PATHS.default_paths()
LIVE_DIR, GHSL_DIR = Path(DEF["cache"]) / "live", Path(DEF["cache"]) / "ghsl"
st.set_page_config(page_title="Urban Heat Hotspots", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1500px;}
[data-testid="stAppDeployButton"], .stDeployButton {display: none;}
.hero {background: linear-gradient(115deg, #0d366b 0%, #2a78d6 58%, #eb6834 135%); color: #fff; padding: 22px 28px;
       border-radius: 14px; margin-bottom: 16px;}
.hero h1 {color: #fff; margin: 0; padding: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em;}
.hero p {margin: 6px 0 0 0; opacity: .93; font-size: .97rem; max-width: 62rem;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 12px 16px;
                          box-shadow: 0 1px 2px rgba(11, 11, 11, .04);}
[data-testid="stMetricLabel"] p {color: #52514e; font-size: .85rem;}
.card {background: #fff; border: 1px solid #e1e0d9; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
.card h4 {margin: 0 0 6px 0;}
.card ol {margin: 0; padding-left: 1.2rem;}
.card li {margin: 5px 0;}
.verdict {border-radius: 12px; padding: 14px 18px; margin: 12px 0 16px 0; border-left: 8px solid;}
.verdict .t {font-size: 1.18rem; font-weight: 700;}
.verdict .d {margin-top: 3px; color: #2b2a28;}
.v-hot {background: #fdecec; border-color: #d03b3b;}
.v-warm {background: #fff3ea; border-color: #ec835a;}
.v-cool {background: #eaf6ea; border-color: #0ca30c;}
.chip {display: inline-block; padding: 3px 11px; border-radius: 999px; background: #eaf1fb; color: #184f95; font-size: .83rem;
       margin: 0 6px 6px 0;}
.legend {display: flex; align-items: center; gap: 6px 18px; font-size: .85rem; color: #52514e; margin: 2px 0 8px 0; flex-wrap: wrap;}
.legend .item {display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;}
.legend .bar {width: 190px; height: 12px; border-radius: 6px; border: 1px solid #e1e0d9;}
.legend .sw {width: 14px; height: 14px; border-radius: 3px; background: rgba(227, 73, 72, .55); border: 1px solid #d03b3b;}
.legend .sw2 {width: 14px; height: 14px; border-radius: 3px; border: 2px solid #00bcd4;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
.foot {color: #898781; font-size: .8rem; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)


def sgn(x, nd=1):
    """Signed number without a negative zero: -0.0 is shown as 0.0."""
    v = round(float(x), nd)
    return f"{v:+.{nd}f}" if v != 0 else f"{0:.{nd}f}"


# ------------------------------------------------------------------ cached computations
@st.cache_resource(show_spinner="Loading the model ...")
def get_model(run, patches):
    return E.load_model(run, patches)


@st.cache_resource(show_spinner="Predicting the whole area (a minute or two the first time, then cached) ...", max_entries=1)
def get_city(city, run, patches, grids, block):
    return E.load_city(city, get_model(run, patches), grids, block, cache_dir=DEF["cache"])


@st.cache_resource(show_spinner="Analysing the area: drivers and cooling doses (up to a few minutes) ...", max_entries=6)
def get_area(city, run, patches, grids, block, box, ring, z, steps):
    return E.analyse_area(get_city(city, run, patches, grids, block), get_model(run, patches), box, ring=ring, z=z, steps=steps)


@st.cache_data(show_spinner=False, max_entries=3)
def get_layers(city, run, patches, grids, block, z, layer):
    state = get_city(city, run, patches, grids, block)
    P = state["P"]
    out = {}
    if layer == "Predicted temperature":
        out["temp"] = E.temperature_layer(state["pred"], P["usable"])
    elif layer == "Observed temperature":
        out["temp"] = E.temperature_layer(P["lst"], P["valid"])
    _, hot = E.hotspot_mask(state, z)
    out["hot"] = E.mask_layer(hot)
    out["share"] = float(hot[P["usable"]].mean())
    out["bounds"] = E.city_bounds(state)
    return out


def esri_layer():
    return folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                            attr="Esri, Maxar, Earthstar Geographics", name="Satellite imagery", overlay=False, show=False)


def add_tools(m):
    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    Draw(export=False, position="topleft", edit_options={"edit": False},
         draw_options={"rectangle": {"shapeOptions": {"color": "#00bcd4"}}, "polyline": False, "polygon": False, "circle": False,
                       "marker": False, "circlemarker": False}).add_to(m)


def drawn_bounds(d):
    """west, south, east, north of a rectangle drawn with the map's square tool (GeoJSON from streamlit-folium)."""
    lons, lats = zip(*[(c[0], c[1]) for c in d["geometry"]["coordinates"][0]])
    return min(lons), min(lats), max(lons), max(lats)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 🔥 Urban heat")
st.sidebar.caption("Hotspots, drivers and cooling what-ifs for Indian cities.")
mode = st.sidebar.radio("Mode", ["Anywhere in India (live)", "Prepared cities"],
                        help="Live: draw a rectangle anywhere, the app downloads the satellite data and analyses it (a few minutes). "
                             "Prepared: the 50 cities that were processed in advance (fast).")
LIVE_MODE = mode.startswith("Anywhere")
with st.sidebar.expander("Data locations", expanded=False):
    run = st.text_input("Model folder (best.pt)", DEF["run"])
    patches = st.text_input("Folder with meta.json", DEF["patches"])
    grids = st.text_input("Prepared city grids folder", DEF["grids"])
need_files = [(f"{run}/best.pt", (Path(run) / "best.pt").exists()), (f"{patches}/meta.json", (Path(patches) / "meta.json").exists())]
if not LIVE_MODE:
    need_files.append((grids, Path(grids).exists()))
problems = [p for p, ok in need_files if not ok]
if problems:
    st.markdown('<div class="hero"><h1>🔥 Urban Heat Hotspots</h1><p>Setup needed before the dashboard can start.</p></div>', unsafe_allow_html=True)
    st.error("Not found: " + ", ".join(problems) + ". Open **Data locations** in the sidebar and check the paths (see Deploy/README.md).")
    st.stop()

ss = st.session_state
for k, v in (("box", None), ("box_city", None), ("last_draw", None), ("box_error", None), ("live_rect", None), ("live_name", None),
             ("live_error", None), ("mode", None)):
    ss.setdefault(k, v)
if ss.mode != mode:                                            # switching mode starts afresh
    ss.update(mode=mode, box=None, box_city=None, last_draw=None, box_error=None, live_rect=None, live_name=None, live_error=None)
qp = st.query_params

st.sidebar.markdown("**Map**")
layer = st.sidebar.radio("Layer", ["Predicted temperature", "Observed temperature", "None"], label_visibility="collapsed")
show_hot = st.sidebar.checkbox("Show hotspots (red)", value=True)
st.sidebar.markdown("**Analysis**")
z = st.sidebar.select_slider("Hotspot strictness", options=[1.645, 1.96, 2.58, 3.29], value=2.58,
                             help="Gi* z-score. Higher means fewer, more concentrated hotspots: 1.96 is 95 % confidence, 2.58 is 99 %, 3.29 is 99.9 %.")
ring = st.sidebar.slider("Neighbour ring width (m)", 150, 1500, 900, 150, help="The area is compared with the ring of this width around it.") // E.RES
if LIVE_MODE:
    max_scenes = st.sidebar.slider("Landsat scenes to average", 8, 40, 20, 4,
                                   help="More scenes give cleaner inputs but a slower download. The prepared cities used up to 40.")

if LIVE_MODE:
    st.markdown('<div class="hero"><h1>🔥 Urban Heat Hotspots</h1><p>Draw a rectangle anywhere in India. The app downloads the satellite data for '
                'that area, runs the trained model, and shows the hotspots, what drives them, and how much planting or reflective roofing would '
                'close the gap. Land surface temperature at the Landsat pass (late morning, clear days, March to May).</p></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="hero"><h1>🔥 Urban Heat Hotspots</h1><p>Draw a rectangle on the map. The model says how much hotter it is than its '
                'neighbours, what drives that, and how much planting or reflective roofing would close the gap. Land surface temperature at the '
                'Landsat pass (late morning, clear days, March to May), as seen by a U-Net trained on 40 Indian cities.</p></div>', unsafe_allow_html=True)


def reset_live():
    # last_draw is kept on purpose: the map widget keeps returning the last rectangle drawn, which must not be applied a second time
    ss.update(live_rect=None, live_name=None, box=None, box_error=None, live_error=None)
    for key in ("rect", "box"):
        if key in st.query_params:
            del st.query_params[key]


# ------------------------------------------------------------------ live mode, before an area has been analysed: the India map
if LIVE_MODE and not ss.live_name:
    if ss.live_rect is None and qp.get("rect"):                # a shared link
        try:
            ss.live_rect = LIVE.check_rect(*[float(v) for v in qp["rect"].split(",")])
        except (ValueError, TypeError):
            pass
    rect = ss.live_rect
    if rect is not None and LIVE.is_built(LIVE_DIR, LIVE.area_key(rect, max_scenes)):
        ss.live_name = LIVE.area_key(rect, max_scenes)         # analysed before: no download needed
        st.rerun()
    m = folium.Map(location=[22.8, 80.0], zoom_start=5, tiles="OpenStreetMap", control_scale=True, min_zoom=4)
    esri_layer().add_to(m)
    if rect is None:
        m.fit_bounds([[LIVE.INDIA[1], LIVE.INDIA[0]], [LIVE.INDIA[3], LIVE.INDIA[2]]])
    else:                                                      # zoom to the drawn rectangle so that it can be seen
        m.fit_bounds([[rect[1], rect[0]], [rect[3], rect[2]]], padding=(90, 90))
        folium.Rectangle([[rect[1], rect[0]], [rect[3], rect[2]]], color="#00bcd4", weight=3, fill=False).add_to(m)
    add_tools(m)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="legend"><span class="item"><span class="sw2"></span>your rectangle</span>'
                    '<span class="item">Zoom to the area you care about first, then use the square tool at the top left.</span></div>', unsafe_allow_html=True)
        if st_folium is None:
            st.warning("streamlit-folium is not installed, so the map cannot take a drawn rectangle (`pip install streamlit-folium`).")
        else:
            out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_active_drawing"], key="map_india_" + (LIVE.area_key(rect) if rect is not None else "none"))
            d = (out or {}).get("last_active_drawing")
            if d and str(d) != ss.last_draw:
                ss.last_draw = str(d)
                try:
                    ss.live_rect = LIVE.check_rect(*drawn_bounds(d))
                    ss.live_error = None
                    st.query_params.update(rect=",".join(f"{v:.5f}" for v in ss.live_rect))
                except ValueError as ex:
                    ss.live_error = str(ex)
                st.rerun()
    with right:
        if rect is None:
            st.markdown("""<div class="card"><h4>How to use it</h4><ol>
<li>Zoom the map to the place you care about.</li>
<li>Click the <b>square tool</b> at the top left of the map and drag a rectangle (1 to 10 km a side).</li>
<li>Press <b>Analyse this area</b>. The app fetches the satellite data, prepares it like the 50 trained cities and runs the model.</li>
<li>Read the hotspots, the drivers and the cooling doses.</li></ol></div>""", unsafe_allow_html=True)
        else:
            h_km, w_km = LIVE.side_km(*rect)
            lon_c, lat_c, half = LIVE.window_for(rect)
            st.markdown(f'<div class="card"><h4>Your rectangle</h4><span class="chip">{h_km:.1f} x {w_km:.1f} km</span>'
                        f'<span class="chip">{lat_c:.4f} N, {lon_c:.4f} E</span><span class="chip">processed in a {2 * half:.0f} km window</span></div>',
                        unsafe_allow_html=True)
            if st.button("Analyse this area", type="primary"):
                name = LIVE.area_key(rect, max_scenes)
                ok = True
                with st.status("Fetching and processing the satellite data ...", expanded=True) as status:
                    try:
                        info = LIVE.build_area_grids(rect, LIVE_DIR, GHSL_DIR, max_scenes=max_scenes, progress=status.write)
                        status.update(label=f"Data ready: {info['n_scenes']} Landsat scenes, {info['seconds']} s", state="complete", expanded=False)
                    except Exception as ex:  # noqa: BLE001
                        ok = False
                        ss.live_error = str(ex)
                        status.update(label="Could not prepare this area", state="error")
                if ok:
                    ss.live_name = name
                    ss.box = None
                    ss.live_error = None
                    st.rerun()
            st.caption("This takes a few minutes the first time: it downloads about 20 Landsat scenes and, for a new region, the building layers. "
                       "The same rectangle is instant afterwards.")
        if ss.live_error:
            st.warning(ss.live_error)
        if rect is not None and st.button("Clear the rectangle"):
            reset_live()
            st.rerun()
    st.markdown('<div class="foot">Live mode needs internet access to Microsoft Planetary Computer and the JRC GHSL server.</div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ an area is available (a prepared city, or a live window)
if LIVE_MODE:
    city, grids_dir, block = ss.live_name, str(LIVE_DIR), 10
    info_f = LIVE_DIR / f"{city}_info.json"
    info = json.loads(info_f.read_text()) if info_f.exists() else {}
else:
    cities = E.list_cities(grids)
    if not cities:
        st.error(f"No <City>_features.nc / <City>_temp.nc pairs in {grids}.")
        st.stop()
    start = qp.get("city") if qp.get("city") in cities else ("Delhi" if "Delhi" in cities else cities[0])
    st.sidebar.markdown("**City**")
    city = st.sidebar.selectbox("City", cities, index=cities.index(start), label_visibility="collapsed")
    grids_dir, block, info = grids, 20, {}

state = get_city(city, run, patches, grids_dir, block)
L = get_layers(city, run, patches, grids_dir, block, z, layer)
south, west, north, east = L["bounds"]
P = state["P"]
bm = block * E.RES


def set_box(b):
    ss.box = tuple(int(x) for x in b)
    ss.box_error = None
    if not LIVE_MODE:
        st.query_params.update(city=city, box=",".join(str(int(x)) for x in b))


def clear_box():
    ss.update(box=None, box_error=None)                        # last_draw is kept on purpose (see reset_live)
    if "box" in st.query_params:
        del st.query_params["box"]


if ss.box_city != city:
    ss.update(box=None, box_city=city, last_draw=None, box_error=None)
    if LIVE_MODE and ss.live_rect is not None:                 # the rectangle the user drew is the area to analyse inside the window
        try:
            ss.box = E.box_from_lonlat(state, ss.live_rect[0], ss.live_rect[1], ss.live_rect[2], ss.live_rect[3])
        except ValueError as ex:
            ss.box_error = str(ex)
    elif not LIVE_MODE and qp.get("city") == city and qp.get("box"):
        try:
            ss.box = E.check_box(tuple(int(v) for v in qp["box"].split(",")), P["H"], P["W"])
        except (ValueError, TypeError):
            pass

# snapshot
k1, k2, k3, k4 = st.columns(4)
k1.metric("Area mean (satellite)" if LIVE_MODE else "City mean (satellite)", f"{P['city_mean']:.1f} °C")
k2.metric(f"Hottest {bm} m block (model)", f"{float(np.nanmax(state['bp'])):.1f} °C")
k3.metric("Share flagged as hotspot", f"{L['share'] * 100:.0f} %", help=f"Share of the {'window' if LIVE_MODE else 'city'} in blocks with a Gi* z-score of at least {z} ({bm} m blocks).")
k4.metric("Window" if LIVE_MODE else "Grid", f"{P['H'] * E.RES / 1000:.0f} x {P['W'] * E.RES / 1000:.0f} km",
          help="The data window processed around your rectangle." if LIVE_MODE else "Each city is a 40 km box at 30 m resolution.")

m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
esri_layer().add_to(m)
m.fit_bounds([[south, west], [north, east]])
if "temp" in L:
    url, (lo, hi) = L["temp"]
    folium.raster_layers.ImageOverlay(image=url, bounds=[[south, west], [north, east]], opacity=1.0, name="temperature").add_to(m)
if show_hot:
    folium.raster_layers.ImageOverlay(image=L["hot"], bounds=[[south, west], [north, east]], opacity=1.0, name="hotspots").add_to(m)
if ss.box:
    s_, w_, n_, e_ = E.box_to_lonlat(state, ss.box)
    folium.Rectangle([[s_, w_], [n_, e_]], color="#00bcd4", weight=3, fill=False).add_to(m)
add_tools(m)

left, right = st.columns([3, 2], gap="large")
with left:
    stops = ",".join(matplotlib.colors.to_hex(matplotlib.colormaps["inferno"](i / 6)) for i in range(7))
    legend = ""
    if "temp" in L:
        note = "" if layer == "Observed temperature" else " (model, area mean added back)"
        legend += (f'<span class="item">{lo:.0f} °C<span class="bar" style="background:linear-gradient(90deg,{stops})"></span>{hi:.0f} °C{note}</span>')
    if show_hot:
        legend += f'<span class="item"><span class="sw"></span>hotspot (z &ge; {z})</span>'
    legend += '<span class="item"><span class="sw2"></span>selected area</span>'
    st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
    if st_folium is None:
        st.warning("streamlit-folium is not installed, so the map cannot take a drawn rectangle. Use the manual box on the right "
                   "(`pip install streamlit-folium`).")
        st.components.v1.html(m._repr_html_(), height=560)
    else:
        out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_active_drawing"], key=f"map_{city}")
        d = (out or {}).get("last_active_drawing")
        if d and str(d) != ss.last_draw:
            ss.last_draw = str(d)
            w_d, s_d, e_d, n_d = drawn_bounds(d)
            inside = south <= (s_d + n_d) / 2 <= north and west <= (w_d + e_d) / 2 <= east
            if LIVE_MODE and not inside:                       # drawn outside the processed window: a new area to download
                try:
                    ss.update(live_rect=LIVE.check_rect(w_d, s_d, e_d, n_d), live_name=None, box=None, live_error=None)
                    st.query_params.update(rect=",".join(f"{v:.5f}" for v in ss.live_rect))
                except ValueError as ex:
                    ss.box_error = str(ex)
                st.rerun()
            try:
                set_box(E.box_from_lonlat(state, w_d, s_d, e_d, n_d))
                st.rerun()
            except ValueError as ex:
                ss.box_error = str(ex)

with right:
    if not ss.box:
        st.markdown("""<div class="card"><h4>How to use it</h4><ol>
<li>Click the <b>square tool</b> at the top left of the map and drag a rectangle over any part of the area.</li>
<li>Read the verdict, the drivers and the cooling doses below the map. Try your own mix with the sliders.</li></ol></div>""", unsafe_allow_html=True)
    else:
        s_, w_, n_, e_ = E.box_to_lonlat(state, ss.box)
        r0_, r1_, c0_, c1_ = ss.box
        st.markdown(f'<div class="card"><h4>Selected area</h4><span class="chip">{(r1_ - r0_) * E.RES / 1000:.1f} x {(c1_ - c0_) * E.RES / 1000:.1f} km</span>'
                    f'<span class="chip">{(s_ + n_) / 2:.4f} N, {(w_ + e_) / 2:.4f} E</span></div>', unsafe_allow_html=True)
    if info:
        st.markdown(f'<div class="card"><h4>Data used</h4><span class="chip">{info.get("n_scenes", "?")} Landsat scenes</span>'
                    f'<span class="chip">{info.get("dates", ["?", "?"])[0]} to {info.get("dates", ["?", "?"])[1]}</span>'
                    f'<span class="chip">{P["H"] * E.RES / 1000:.0f} km window</span></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("Jump to the strongest hotspot", help=f"The hottest {bm * 2} m block compared with its neighbours."):
        set_box(E.top_hotspot_box(state, get_model(run, patches), size=20, ring=ring))
        st.rerun()
    if b2.button("Clear the area", disabled=not ss.box):
        clear_box()
        st.rerun()
    if LIVE_MODE and st.button("Analyse a different place"):
        reset_live()
        st.rerun()
    with st.expander("Type the box in grid pixels (30 m each)"):
        H, W = P["H"], P["W"]
        x1, x2 = st.columns(2)
        r0i = x1.number_input("Row from", 0, H, 0)
        r1i = x1.number_input("Row to", 0, H, min(60, H))
        c0i = x2.number_input("Column from", 0, W, 0)
        c1i = x2.number_input("Column to", 0, W, min(60, W))
        if st.button("Use this box"):
            try:
                set_box(E.check_box((int(r0i), int(r1i), int(c0i), int(c1i)), H, W))
                st.rerun()
            except ValueError as ex:
                ss.box_error = str(ex)
    if ss.box_error:
        st.warning(ss.box_error)

box = ss.box
if not box:
    st.markdown('<div class="foot">Model and data stay on this machine. The first open of an area predicts it in full (cached afterwards).</div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ results for the selected area
steps = 10 if (box[1] - box[0]) * (box[3] - box[2]) <= 90_000 else 5
res = get_area(city, run, patches, grids_dir, block, tuple(box), ring, z, steps)
S, need = res["S"], res["need"]
h_m, w_m = res["sides_m"]
pix_ha = E.PIXEL_HA
st.markdown(f"### Results for the selected area: {res['ha']:.0f} ha ({h_m / 1000:.1f} x {w_m / 1000:.1f} km)")

if res["pred_gap"] <= 0.2:
    cls, title = "v-cool", "✅ Not hotter than its neighbours"
    detail = "In the model's view there is no gap to close here."
elif res["pred_gap"] >= 1.0 and res["hot_share"] >= 0.5:
    cls, title = "v-hot", "🔥 A hotspot"
    detail = f"{res['hot_share'] * 100:.0f} % of it lies in a significantly hot block, and the model puts it {res['pred_gap']:.1f} °C above its neighbours."
elif res["pred_gap"] >= 1.0:
    cls, title = "v-warm", "🌡️ Warmer than its neighbours"
    detail = (f"The model puts it {res['pred_gap']:.1f} °C above its neighbours, but only {res['hot_share'] * 100:.0f} % of it is inside a "
              f"{'window-wide' if LIVE_MODE else 'city-wide'} hotspot at this strictness.")
else:
    cls, title = "v-warm", "🌡️ Only slightly warmer than its neighbours"
    detail = f"The model puts it {res['pred_gap']:.1f} °C above its neighbours, a small gap (a hotspot needs at least 1 °C)."
if res["pred_gap"] > 0.2 and need["veg"] is not None:
    detail += f" Planting about {need['veg'] * 100:.0f} % of its open ground (around {need['veg'] * res['n_green'] * pix_ha:.0f} ha) would close the model's gap."
st.markdown(f'<div class="verdict {cls}"><div class="t">{title}</div><div class="d">{detail}</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotter than neighbours (satellite)", f"{sgn(res['obs_gap'])} °C", help=f"Area {res['obs_area']:.1f} °C vs its ring {res['obs_ring']:.1f} °C")
c2.metric("Hotter than neighbours (model)", f"{sgn(res['pred_gap'])} °C")
c3.metric("Not explained by the data", f"{sgn(res['unexplained'])} °C",
          help="Observed gap minus the model's gap: heat from things not in the data (traffic, industry, air conditioning...) or noise.")
c4.metric("Inside a hotspot", f"{res['hot_share'] * 100:.0f} %", help=f"Share of the area in blocks flagged at z >= {z} (mean Gi* z of the area {res['mean_z']:.1f}).")

tab1, tab2, tab3, tab4 = st.tabs(["Why is it hot?", "How much cooling is needed?", "Maps", "Numbers"])

with tab1:
    a_, b_ = st.columns([3, 2])
    with a_:
        st.pyplot(PL.drivers_fig(res["phi"]), clear_figure=True)
    with b_:
        top = sorted(res["phi"].items(), key=lambda kv: -kv[1])
        st.markdown("**Reading it.** Each bar is how many °C a feature group adds to this area compared with a neighbour-like version of it "
                    "(its values swapped for the neighbours' average). Red adds heat, blue cools.")
        if top[0][1] > 0.05:
            st.markdown(f"Biggest driver here: **{top[0][0]}** ({top[0][1]:+.1f} °C)" + (f", then **{top[1][0]}** ({top[1][1]:+.1f} °C)." if top[1][1] > 0.05 else "."))
        st.caption(f"The groups add up to {sum(res['phi'].values()):+.2f} °C; reference mismatch {res['ref_mismatch']:+.2f} °C. "
                   "These are the model's attributions, not proven causes: vegetation, built-up share and albedo move together.")

with tab2:
    roof_ok = res["n_roof"] >= 30
    a_, b_ = st.columns([3, 2])
    with a_:
        st.pyplot(PL.dose_fig(res["table"], need, roof_ok), clear_figure=True)
    with b_:
        st.markdown("**Dose needed to match the neighbours** (the dot on each line)")
        if res["pred_gap"] <= 0:
            st.write("No gap to close.")
        else:
            for key, label, n in (("veg", "Planting alone", res["n_green"]), ("roof", "Cool roofs alone", res["n_roof"]), ("both", "Both together", None)):
                d = need[key]
                if key == "roof" and not roof_ok:
                    st.write(f"- {label}: not applicable, only {n} built pixels ({n * pix_ha:.1f} ha) here.")
                elif d is None:
                    left_ = res["table"][f"{key}_gap_left"].iloc[-1]
                    st.write(f"- {label}: **not enough**, even at 100 % {left_:+.1f} °C of the gap is left.")
                elif key == "both":
                    st.write(f"- {label}: **{d * 100:.0f} %** of each ({d * res['n_green'] * pix_ha:.1f} ha of planting + {d * res['n_roof'] * pix_ha:.1f} ha of roofs).")
                else:
                    st.write(f"- {label}: **{d * 100:.0f} %** of the eligible surface ({d * n * pix_ha:.1f} ha, {d * n * pix_ha / max(res['ha'], 1e-9) * 100:.0f} % of the area).")
        if res["unexplained"] > 0.5:
            st.caption(f"{res['unexplained']:.1f} °C of the observed gap is not explained by the model's features, so these levers cannot remove that part.")
    st.markdown("**Try your own mix**")
    s1, s2 = st.columns(2)
    default_v = int(round(min(max((need["veg"] or 0.0) * 100, 0), 100) / 5) * 5)
    veg = s1.slider("Share of greenable ground planted (%)", 0, 100, default_v, 5, key=f"veg_{city}_{box}")
    roof = s2.slider("Share of buildings with reflective roofs (%)", 0, 100, 0, 5, key=f"roof_{city}_{box}", disabled=not roof_ok)
    cooling, gap_left = E.dose_now(S, veg / 100, roof / 100 if roof_ok else 0.0)
    m1, m2 = st.columns(2)
    m1.metric("Cooling of the area", f"{cooling:.2f} °C")
    m2.metric("Gap to the neighbours after the change", f"{sgn(gap_left, 2)} °C",
              delta="as cool as its neighbours" if gap_left <= 0 else "still hotter", delta_color="normal" if gap_left <= 0 else "inverse")
    st.caption(f"Planting turns open or lightly built pixels into the typical tree cover of this place (NDVI {res['tree_ref']['NDVI']:.2f}); reflective roofs add "
               f"0.15 albedo to built pixels. Pixels are chosen at random at 30 m, so real tree rows and parks would differ. "
               f"The model's vegetation effect is about 10 to 30 % steeper than the satellite data alone suggests: read the doses as the optimistic end.")

with tab3:
    st.pyplot(PL.area_maps_fig(res, state), clear_figure=True)
    st.caption("Cyan = the selected area, dashed white = the neighbour ring it is compared with. White gaps are water or missing data. "
               "Both panels share one colour scale.")

with tab4:
    st.dataframe(res["table"].round(2), hide_index=True)
    st.download_button("Download the dose table (CSV)", res["table"].round(3).to_csv(index=False).encode(), f"{city}_{box[0]}_{box[2]}_doses.csv", "text/csv")
    st.write(f"Rows {box[0]}-{box[1]}, columns {box[2]}-{box[3]} of the {city} grid; mean temperature {res['city_mean']:.1f} °C.")
    if not LIVE_MODE:
        st.write(f"Re-run the same area from the command line: `python models/simulate_cooling.py --city {city} --run {Path(run).as_posix()} --box {box[0]} {box[1]} {box[2]} {box[3]}`")
        st.caption(f"Share this view: add `?city={city}&box={box[0]},{box[1]},{box[2]},{box[3]}` to the page address.")
    else:
        st.caption("Share this view: copy the page address, it contains the rectangle (`?rect=west,south,east,north`).")

with st.expander("How to read this, and what it does not tell you"):
    st.markdown("""
- **Hotspot** means significantly hotter than the rest of the data window (Getis-Ord Gi* on small blocks), not a fixed top share.
- **The model** is a U-Net trained on 40 Indian cities and tested on 10 others (R² about 0.74 inside the city, hotspot AUC about 0.93). It predicts the *pattern inside an area*, so the area's observed mean temperature is added back.
- **Live areas are a little less reliable than the prepared cities:** fewer Landsat scenes are averaged, and the city-relative inputs and the hotspot statistics are relative to the data window instead of a 40 km city box.
- **Drivers and doses are the model's view**, not measured effects. Features that move together share credit.
- **Heat outside the data** (traffic, industry, air conditioning) shows up as "not explained" and is not touched by the levers.
- **Surface temperature at the satellite pass**, not air temperature or how it feels at night.
- The map overlay is drawn on the area's longitude/latitude envelope, so edges can be off by a few hundred metres.
""")
st.markdown(f'<div class="foot">Model: {Path(run).name} | {"live window" if LIVE_MODE else "city " + city} | hotspot strictness z = {z} | ring {ring * E.RES} m</div>', unsafe_allow_html=True)
