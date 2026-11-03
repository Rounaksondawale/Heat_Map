"""
Urban Heat Hotspots dashboard: pick a city, draw a rectangle on the map, see whether it is a hotspot, why, and how much greening or
cool roofing would bring it down to the level of its neighbours.

    pip install -r dashboard/requirements.txt          # on top of the project requirements and PyTorch
    streamlit run dashboard/app.py

Needs (all in the Deploy/ folder, see Deploy/README.md and dashboard/paths.py; paths can be changed in the sidebar):
    Deploy/best.pt      the trained model
    Deploy/meta.json    feature means / spreads of the training run
    Deploy/grids/       <City>_features.nc and <City>_temp.nc for each city to offer
The page address keeps the city and the area (?city=Delhi&box=r0,r1,c0,c1), so a view can be shared as a link.
"""
import sys
from pathlib import Path

import folium
import matplotlib
import numpy as np
import streamlit as st
from folium.plugins import Draw, Fullscreen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import paths as PATHS  # noqa: E402
import plots as PL  # noqa: E402

try:
    from streamlit_folium import st_folium
except ImportError:                                            # the manual box still works without the map widget
    st_folium = None

ROOT = E.ROOT
DEF = PATHS.default_paths()
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


@st.cache_resource(show_spinner="Predicting the whole city (a minute or two the first time, then cached) ...", max_entries=1)
def get_city(city, run, patches, grids, block):
    return E.load_city(city, get_model(run, patches), grids, block, cache_dir=DEF["cache"])


@st.cache_resource(show_spinner="Analysing the area: drivers and cooling doses (up to a minute) ...", max_entries=6)
def get_area(city, run, patches, grids, block, box, ring, z):
    return E.analyse_area(get_city(city, run, patches, grids, block), get_model(run, patches), box, ring=ring, z=z)


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


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 🔥 Urban heat")
st.sidebar.caption("Hotspots, drivers and cooling what-ifs for Indian cities.")
with st.sidebar.expander("Data locations", expanded=False):
    run = st.text_input("Model folder (best.pt)", DEF["run"])
    grids = st.text_input("City grids folder", DEF["grids"])
    patches = st.text_input("Folder with meta.json", DEF["patches"])
problems = [p for p, ok in ((f"{run}/best.pt", (Path(run) / "best.pt").exists()), (f"{patches}/meta.json", (Path(patches) / "meta.json").exists()),
                            (grids, Path(grids).exists())) if not ok]
if problems:
    st.markdown('<div class="hero"><h1>🔥 Urban Heat Hotspots</h1><p>Setup needed before the dashboard can start.</p></div>', unsafe_allow_html=True)
    st.error("Not found: " + ", ".join(problems) + ". Open **Data locations** in the sidebar and check the paths (the top of dashboard/app.py lists what is needed).")
    st.stop()
cities = E.list_cities(grids)
if not cities:
    st.error(f"No <City>_features.nc / <City>_temp.nc pairs in {grids}.")
    st.stop()
qp = st.query_params
start = qp.get("city") if qp.get("city") in cities else ("Delhi" if "Delhi" in cities else cities[0])
st.sidebar.markdown("**City**")
city = st.sidebar.selectbox("City", cities, index=cities.index(start), label_visibility="collapsed")
st.sidebar.markdown("**Map**")
layer = st.sidebar.radio("Layer", ["Predicted temperature", "Observed temperature", "None"], label_visibility="collapsed")
show_hot = st.sidebar.checkbox("Show hotspots (red)", value=True)
st.sidebar.markdown("**Analysis**")
z = st.sidebar.select_slider("Hotspot strictness", options=[1.645, 1.96, 2.58, 3.29], value=2.58,
                             help="Gi* z-score. Higher means fewer, more concentrated hotspots: 1.96 is 95 % confidence, 2.58 is 99 %, 3.29 is 99.9 %.")
ring = st.sidebar.slider("Neighbour ring width (m)", 150, 1500, 900, 150, help="The area is compared with the ring of this width around it.") // E.RES
block = 20

st.markdown('<div class="hero"><h1>🔥 Urban Heat Hotspots</h1><p>Draw a rectangle on the map. The model says how much hotter it is than its '
            'neighbours, what drives that, and how much planting or reflective roofing would close the gap. Land surface temperature at the '
            'Landsat pass (late morning, clear days, March to May), as seen by a U-Net trained on 40 Indian cities.</p></div>', unsafe_allow_html=True)

state = get_city(city, run, patches, grids, block)
L = get_layers(city, run, patches, grids, block, z, layer)
south, west, north, east = L["bounds"]
P = state["P"]


# ------------------------------------------------------------------ selected area (kept in the session and in the page address)
def set_box(b):
    st.session_state.box = tuple(int(x) for x in b)
    st.session_state.box_error = None
    st.query_params.update(city=city, box=",".join(str(int(x)) for x in b))


def clear_box():
    st.session_state.update(box=None, last_draw=None, box_error=None)
    if "box" in st.query_params:
        del st.query_params["box"]


if st.session_state.get("box_city") != city:
    st.session_state.update(box=None, box_city=city, last_draw=None, box_error=None)
    if qp.get("city") == city and qp.get("box"):
        try:
            st.session_state.box = E.check_box(tuple(int(v) for v in qp["box"].split(",")), P["H"], P["W"])
        except (ValueError, TypeError):
            pass

# city snapshot
k1, k2, k3, k4 = st.columns(4)
k1.metric("City mean (satellite)", f"{P['city_mean']:.1f} °C")
k2.metric("Hottest 600 m block (model)", f"{float(np.nanmax(state['bp'])):.1f} °C")
k3.metric("Flagged as hotspot", f"{L['share'] * 100:.0f} % of the city", help=f"Gi* z-score of at least {z} on 600 m blocks.")
k4.metric("Grid", f"{P['H'] * E.RES / 1000:.0f} x {P['W'] * E.RES / 1000:.0f} km", help="Each city is a 40 km box at 30 m resolution.")

m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri, Maxar, Earthstar Geographics", name="Satellite imagery", overlay=False, show=False).add_to(m)
m.fit_bounds([[south, west], [north, east]])
if "temp" in L:
    url, (lo, hi) = L["temp"]
    folium.raster_layers.ImageOverlay(image=url, bounds=[[south, west], [north, east]], opacity=1.0, name="temperature").add_to(m)
if show_hot:
    folium.raster_layers.ImageOverlay(image=L["hot"], bounds=[[south, west], [north, east]], opacity=1.0, name="hotspots").add_to(m)
if st.session_state.box:
    s_, w_, n_, e_ = E.box_to_lonlat(state, st.session_state.box)
    folium.Rectangle([[s_, w_], [n_, e_]], color="#00bcd4", weight=3, fill=False).add_to(m)
Fullscreen(position="topright").add_to(m)
folium.LayerControl(collapsed=True, position="topright").add_to(m)
Draw(export=False, position="topleft", edit_options={"edit": False},
     draw_options={"rectangle": {"shapeOptions": {"color": "#00bcd4"}}, "polyline": False, "polygon": False, "circle": False,
                   "marker": False, "circlemarker": False}).add_to(m)

left, right = st.columns([3, 2], gap="large")
with left:
    stops = ",".join(matplotlib.colors.to_hex(matplotlib.colormaps["inferno"](i / 6)) for i in range(7))
    legend = ""
    if "temp" in L:
        note = "" if layer == "Observed temperature" else " (model, city mean added back)"
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
        if d and str(d) != st.session_state.last_draw:
            st.session_state.last_draw = str(d)
            try:
                lons, lats = zip(*[(c[0], c[1]) for c in d["geometry"]["coordinates"][0]])
                set_box(E.box_from_lonlat(state, min(lons), min(lats), max(lons), max(lats)))
                st.rerun()
            except ValueError as ex:
                st.session_state.box_error = str(ex)

with right:
    if not st.session_state.box:
        st.markdown("""<div class="card"><h4>How to use it</h4><ol>
<li>Pick a <b>city</b> in the sidebar.</li>
<li>Click the <b>square tool</b> at the top left of the map and drag a rectangle over any part of the city.</li>
<li>Read the verdict, the drivers and the cooling doses below the map. Try your own mix with the sliders.</li></ol></div>""", unsafe_allow_html=True)
    else:
        s_, w_, n_, e_ = E.box_to_lonlat(state, st.session_state.box)
        r0_, r1_, c0_, c1_ = st.session_state.box
        st.markdown(f'<div class="card"><h4>Selected area</h4><span class="chip">{(r1_ - r0_) * E.RES / 1000:.1f} x {(c1_ - c0_) * E.RES / 1000:.1f} km</span>'
                    f'<span class="chip">{(s_ + n_) / 2:.4f} N, {(w_ + e_) / 2:.4f} E</span><span class="chip">rows {r0_}-{r1_}, columns {c0_}-{c1_}</span></div>',
                    unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if b1.button("Jump to the strongest hotspot", help="The hottest 600 m block compared with its neighbours."):
        set_box(E.top_hotspot_box(state, get_model(run, patches), size=20, ring=ring))
        st.rerun()
    if b2.button("Clear the area", disabled=not st.session_state.box):
        clear_box()
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
                st.session_state.box_error = str(ex)
    if st.session_state.box_error:
        st.warning(st.session_state.box_error)

box = st.session_state.box
if not box:
    st.markdown('<div class="foot">Model and data stay on this machine. First open of a city predicts the whole city (cached afterwards).</div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ results for the selected area
res = get_area(city, run, patches, grids, block, tuple(box), ring, z)
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
    detail = (f"The model puts it {res['pred_gap']:.1f} °C above its neighbours, but only {res['hot_share'] * 100:.0f} % of it is inside a city-wide "
              f"hotspot at this strictness.")
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
    veg = s1.slider("Share of greenable ground planted (%)", 0, 100, default_v, 5, key=f"veg_{box}")
    roof = s2.slider("Share of buildings with reflective roofs (%)", 0, 100, 0, 5, key=f"roof_{box}", disabled=not roof_ok)
    cooling, gap_left = E.dose_now(S, veg / 100, roof / 100 if roof_ok else 0.0)
    m1, m2 = st.columns(2)
    m1.metric("Cooling of the area", f"{cooling:.2f} °C")
    m2.metric("Gap to the neighbours after the change", f"{sgn(gap_left, 2)} °C",
              delta="as cool as its neighbours" if gap_left <= 0 else "still hotter", delta_color="normal" if gap_left <= 0 else "inverse")
    st.caption(f"Planting turns open or lightly built pixels into the city's typical tree cover (NDVI {res['tree_ref']['NDVI']:.2f}); reflective roofs add "
               f"0.15 albedo to built pixels. Pixels are chosen at random at 30 m, so real tree rows and parks would differ. "
               f"The model's vegetation effect is about 10 to 30 % steeper than the satellite data alone suggests: read the doses as the optimistic end.")

with tab3:
    st.pyplot(PL.area_maps_fig(res, state), clear_figure=True)
    st.caption("Cyan = the selected area, dashed white = the neighbour ring it is compared with. White gaps are water or missing data. "
               "Both panels share one colour scale.")

with tab4:
    st.dataframe(res["table"].round(2), hide_index=True)
    st.download_button("Download the dose table (CSV)", res["table"].round(3).to_csv(index=False).encode(), f"{city}_{box[0]}_{box[2]}_doses.csv", "text/csv")
    st.write(f"Rows {box[0]}-{box[1]}, columns {box[2]}-{box[3]} of the {city} grid; city mean temperature {res['city_mean']:.1f} °C. "
             f"Re-run the same area from the command line: `python models/simulate_cooling.py --city {city} --run {Path(run).as_posix()} --box {box[0]} {box[1]} {box[2]} {box[3]}`")
    st.caption(f"Share this view: add `?city={city}&box={box[0]},{box[1]},{box[2]},{box[3]}` to the page address.")

with st.expander("How to read this, and what it does not tell you"):
    st.markdown("""
- **Hotspot** means significantly hotter than the rest of the city (Getis-Ord Gi* on 600 m blocks), not a fixed top share.
- **The model** is a U-Net trained on 40 cities and tested on 10 others (R² about 0.74 inside the city, hotspot AUC about 0.93). It predicts the *pattern inside a city*, so each city's observed mean temperature is added back.
- **Drivers and doses are the model's view**, not measured effects. Features that move together share credit.
- **Heat outside the data** (traffic, industry, air conditioning) shows up as "not explained" and is not touched by the levers.
- **Surface temperature at the satellite pass**, not air temperature or how it feels at night.
- The map overlay is drawn on the city's longitude/latitude envelope, so edges can be off by a few hundred metres.
""")
st.markdown(f'<div class="foot">Model: {Path(run).name} | city {city} | hotspot strictness z = {z} | ring {ring * E.RES} m</div>', unsafe_allow_html=True)
