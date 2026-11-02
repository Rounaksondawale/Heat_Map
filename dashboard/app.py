"""
Urban Heat Hotspots dashboard: pick a city, draw a rectangle on the map, see whether it is a hotspot, why, and how much greening or
cool roofing would bring it down to the level of its neighbours.

    pip install -r dashboard/requirements.txt          # on top of the project requirements and PyTorch
    streamlit run dashboard/app.py

Needs (default locations, changeable in the sidebar):
    pre_processing/grids/<City>_features.nc and <City>_temp.nc      (pre_processing/export_city_grids.py)
    pre_processing/patches/meta.json                                (pre_processing/make_patches.py)
    data/unet_run3/best.pt                                          (models/train_unet.py)
You can copy just a few cities' two grid files, meta.json and best.pt to another machine and run it there.
"""
import sys
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
import plots as PL  # noqa: E402

try:
    from streamlit_folium import st_folium
except ImportError:                                            # the manual box still works without the map widget
    st_folium = None

ROOT = E.ROOT
st.set_page_config(page_title="Urban Heat Hotspots", page_icon="🔥", layout="wide")


# ------------------------------------------------------------------ cached computations
@st.cache_resource(show_spinner="Loading the model ...")
def get_model(run, patches):
    return E.load_model(run, patches)


@st.cache_resource(show_spinner="Predicting the whole city (a minute or two the first time, then cached) ...", max_entries=1)
def get_city(city, run, patches, grids, block):
    return E.load_city(city, get_model(run, patches), grids, block, cache_dir=ROOT / "data" / "dashboard_cache")


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
st.sidebar.title("🔥 Urban heat")
run = st.sidebar.text_input("Model folder", str(ROOT / "data" / "unet_run3"))
grids = st.sidebar.text_input("City grids folder", str(ROOT / "pre_processing" / "grids"))
patches = st.sidebar.text_input("Patches folder (meta.json)", str(ROOT / "pre_processing" / "patches"))
problems = [p for p, ok in ((f"{run}/best.pt", (Path(run) / "best.pt").exists()), (f"{patches}/meta.json", (Path(patches) / "meta.json").exists()),
                            (grids, Path(grids).exists())) if not ok]
if problems:
    st.title("Urban Heat Hotspots")
    st.error("Not found: " + ", ".join(problems) + ". Check the paths in the sidebar (see the top of dashboard/app.py for what is needed).")
    st.stop()
cities = E.list_cities(grids)
if not cities:
    st.error(f"No <City>_features.nc / <City>_temp.nc pairs in {grids}.")
    st.stop()
city = st.sidebar.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
layer = st.sidebar.radio("Map layer", ["Predicted temperature", "Observed temperature", "None"])
show_hot = st.sidebar.checkbox("Show hotspots (red)", value=True)
z = st.sidebar.select_slider("Hotspot strictness (Gi* z-score)", options=[1.645, 1.96, 2.58, 3.29], value=2.58,
                             help="Higher = fewer, more concentrated hotspots. 1.96 is 95 % confidence, 2.58 is 99 %, 3.29 is 99.9 %.")
ring = st.sidebar.slider("Neighbour ring width (m)", 150, 1500, 900, 150, help="The area is compared with the ring of this width around it.") // E.RES
block = 20

st.title("Urban Heat Hotspots")
st.caption("Draw a rectangle on the map. The model says how much hotter it is than its neighbours, what drives that, and how much planting "
           "or reflective roofing would close the gap. Model's view, land surface temperature at the Landsat pass (late morning, clear days, Mar-May).")

state = get_city(city, run, patches, grids, block)
L = get_layers(city, run, patches, grids, block, z, layer)
south, west, north, east = L["bounds"]

# ------------------------------------------------------------------ map and box selection
if "box" not in st.session_state or st.session_state.get("box_city") != city:
    st.session_state.update(box=None, box_city=city, last_draw=None, box_error=None)

m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
if "temp" in L:
    url, (lo, hi) = L["temp"]
    folium.raster_layers.ImageOverlay(image=url, bounds=[[south, west], [north, east]], opacity=1.0, name="temperature").add_to(m)
if show_hot:
    folium.raster_layers.ImageOverlay(image=L["hot"], bounds=[[south, west], [north, east]], opacity=1.0, name="hotspots").add_to(m)
if st.session_state.box:
    s_, w_, n_, e_ = E.box_to_lonlat(state, st.session_state.box)
    folium.Rectangle([[s_, w_], [n_, e_]], color="#00bcd4", weight=3, fill=False).add_to(m)
Draw(export=False, position="topleft", edit_options={"edit": False},
     draw_options={"rectangle": {"shapeOptions": {"color": "#00bcd4"}}, "polyline": False, "polygon": False, "circle": False,
                   "marker": False, "circlemarker": False}).add_to(m)

left, right = st.columns([3, 2])
with left:
    if "temp" in L:
        st.caption(f"{layer}: {lo:.0f} to {hi:.0f} °C" + ("" if layer == "Observed temperature" else " (observed city mean added back)") +
                   f" | {L['share'] * 100:.0f}% of the city is flagged as hotspot at z >= {z}")
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
                st.session_state.box = E.box_from_lonlat(state, min(lons), min(lats), max(lons), max(lats))
                st.session_state.box_error = None
                st.rerun()
            except ValueError as ex:
                st.session_state.box_error = str(ex)

with right:
    st.subheader("Area")
    st.write("Draw a rectangle with the square tool at the top left of the map, or:")
    if st.button("Jump to the city's strongest 600 m hotspot"):
        st.session_state.box = E.top_hotspot_box(state, get_model(run, patches), size=20, ring=ring)
        st.session_state.box_error = None
        st.rerun()
    with st.expander("Type the box in grid pixels (30 m each)"):
        H, W = state["P"]["H"], state["P"]["W"]
        r0i = st.number_input("Row from", 0, H, 0)
        r1i = st.number_input("Row to", 0, H, min(60, H))
        c0i = st.number_input("Column from", 0, W, 0)
        c1i = st.number_input("Column to", 0, W, min(60, W))
        if st.button("Use this box"):
            try:
                st.session_state.box = E.check_box((int(r0i), int(r1i), int(c0i), int(c1i)), H, W)
                st.session_state.box_error = None
                st.rerun()
            except ValueError as ex:
                st.session_state.box_error = str(ex)
    if st.session_state.box and st.button("Clear the area"):
        st.session_state.update(box=None, last_draw=None, box_error=None)
        st.rerun()
    if st.session_state.box_error:
        st.warning(st.session_state.box_error)
    if not st.session_state.box:
        st.info("No area selected yet. The map shows predicted temperature and the hotspots (red).")

box = st.session_state.box
if not box:
    st.stop()

# ------------------------------------------------------------------ results for the selected area
res = get_area(city, run, patches, grids, block, tuple(box), ring, z)
S, need = res["S"], res["need"]
h_m, w_m = res["sides_m"]
st.divider()
st.header(f"Selected area: {res['ha']:.0f} ha ({h_m / 1000:.1f} x {w_m / 1000:.1f} km)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hotter than its neighbours (satellite)", f"{res['obs_gap']:+.1f} °C", help=f"Area {res['obs_area']:.1f} °C vs ring {res['obs_ring']:.1f} °C")
c2.metric("Hotter than its neighbours (model)", f"{res['pred_gap']:+.1f} °C")
c3.metric("Not explained by the data", f"{res['unexplained']:+.1f} °C", help="Observed gap minus the model's gap: heat from things not in the data (traffic, industry, air conditioning...) or noise.")
c4.metric("Inside a hotspot", f"{res['hot_share'] * 100:.0f} %", help=f"Share of the area in blocks flagged at z >= {z} (mean Gi* z of the area {res['mean_z']:.1f}).")
if res["pred_gap"] <= 0.2:
    st.success("In the model's view this area is **not hotter than its neighbours**, so there is no gap to close.")
elif res["hot_share"] >= 0.5:
    st.error(f"A **hotspot**: {res['hot_share'] * 100:.0f} % of it lies in a significantly hot block, and the model puts it {res['pred_gap']:.1f} °C above its neighbours.")
else:
    st.warning(f"The model puts this area {res['pred_gap']:.1f} °C above its neighbours, but only {res['hot_share'] * 100:.0f} % of it is inside a city-wide hotspot at this strictness.")

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
    sim = S["sim"]
    roof_ok = res["n_roof"] >= 30
    a_, b_ = st.columns([3, 2])
    with a_:
        st.pyplot(PL.dose_fig(res["table"], need, roof_ok), clear_figure=True)
    with b_:
        st.markdown("**Dose needed to match the neighbours** (the dot on each line)")
        pix_ha = E.PIXEL_HA
        if res["pred_gap"] <= 0:
            st.write("No gap to close.")
        else:
            for key, label, n in (("veg", "Planting alone", res["n_green"]), ("roof", "Cool roofs alone", res["n_roof"]), ("both", "Both together", None)):
                d = need[key]
                if key == "roof" and not roof_ok:
                    st.write(f"- {label}: not applicable, only {n} built pixels ({n * pix_ha:.1f} ha) here.")
                elif d is None:
                    left = res["table"][f"{key}_gap_left"].iloc[-1]
                    st.write(f"- {label}: **not enough**, even at 100 % {left:+.1f} °C of the gap is left.")
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
    m2.metric("Gap to the neighbours after the change", f"{gap_left:+.2f} °C",
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

with st.expander("How to read this, and what it does not tell you"):
    st.markdown("""
- **Hotspot** means significantly hotter than the rest of the city (Getis-Ord Gi* on 600 m blocks), not a fixed top share.
- **The model** is a U-Net trained on 40 cities and tested on 10 others (R² about 0.74 inside the city, hotspot AUC about 0.93). It predicts the *pattern inside a city*, so each city's observed mean temperature is added back.
- **Drivers and doses are the model's view**, not measured effects. Features that move together share credit.
- **Heat outside the data** (traffic, industry, air conditioning) shows up as "not explained" and is not touched by the levers.
- **Surface temperature at the satellite pass**, not air temperature or how it feels at night.
- The map overlay is drawn on the city's longitude/latitude envelope, so edges can be off by a few hundred metres.
""")
