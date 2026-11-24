"""
Build the small, pre-aggregated JSON files the results dashboard reads.

Run from the repo root with a plain Python (standard library only; PyTorch is optional):

    python dashboard-project/data/build_data.py

Sources, in order of preference (no value is typed in by hand; everything is parsed from a file):
  1. Raw pipeline outputs, if they have been downloaded from the Azure ML VM into the repo (or into dashboard-project/azure_raw/):
       data/unet_run3/eval_val/eval_per_city.csv, eval_by_landcover.csv   (models/evaluate_unet.py)
       data/model_comparison.csv                                         (models/compare_runs.py)
       data/unet_run3/group_importance.csv                               (models/group_importance.py)
       data/unet_run3/metrics.csv                                        (models/train_unet.py)
       data/hotspots/<City>_hotspots.csv                                 (models/hotspot_report.py)
  2. The metric tables published in README.md (the numbers printed by those same scripts, as transcribed by the authors).
  3. Deploy/best.pt (checkpoint metadata) and extraction/build_dataset.py / pre_processing/make_patches.py (city list, split).

Every JSON carries a "_source" field naming the exact file(s) (and README line ranges) it came from.
"""
import ast
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW_DIRS = [ROOT, ROOT / "dashboard-project" / "azure_raw"]
README = ROOT / "README.md"


# ---------------------------------------------------------------- helpers
def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def num(s):
    """'**0.74**' -> 0.74, '-1.45 °C' -> -1.45, '6.74 M' -> 6.74, 'Amritsar' -> 'Amritsar'."""
    t = s.replace("*", "").replace("°C", "").replace("%", "").strip()
    t = re.sub(r"\s*M$", "", t)
    try:
        return int(t) if re.fullmatch(r"[+-]?\d+", t) else float(t)
    except ValueError:
        return s.replace("*", "").strip()


def md_table(lines, header_prefix):
    """Find the markdown table whose header row starts with header_prefix. Returns (first_line_no, last_line_no, header, rows)."""
    for i, ln in enumerate(lines):
        if ln.strip().startswith(header_prefix):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, j = [], i + 2                       # skip the |---| row
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            return i + 1, j, hdr, rows
    raise SystemExit(f"README table starting with {header_prefix!r} not found")


def find_line(lines, needle):
    for i, ln in enumerate(lines):
        if needle in ln:
            return i + 1, ln
    raise SystemExit(f"README line containing {needle!r} not found")


def raw(path):
    for d in RAW_DIRS:
        p = d / path
        if p.exists():
            return p
    return None


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return [{k: num(v) if v != "" else None for k, v in r.items()} for r in csv.DictReader(f)]


def write(name, obj):
    p = OUT / name
    txt = json.dumps(obj, indent=1, ensure_ascii=False)
    p.write_text(txt, encoding="utf-8")
    kb = len(txt.encode()) / 1024
    assert kb < 1024, f"{name} is {kb:.0f} KB (limit 1 MB)"
    print(f"  wrote {rel(p)}  ({kb:.1f} KB)  <- {obj['_source'] if isinstance(obj['_source'], str) else '; '.join(obj['_source'])}")


# ---------------------------------------------------------------- cities.json (config, always available)
def city_config():
    src = ROOT / "extraction" / "build_dataset.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    cities = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CITIES" for t in node.targets):
            cities = ast.literal_eval(node.value)
    mp = ROOT / "pre_processing" / "make_patches.py"
    val = None
    for node in ast.parse(mp.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "VAL_DEFAULT" for t in node.targets):
            val = ast.literal_eval(node.value)
    rows = [{"city": c, "lon": v[0], "lat": v[1], "climate": v[2], "cloud_max_pct": v[3], "split": "val" if c in val else "train"}
            for c, v in cities.items()]
    return {"_source": [f"{rel(src)} (CITIES)", f"{rel(mp)} (VAL_DEFAULT)"],
            "_note": "City centres and the train/validation split as configured in code (each city is a 40 km box around the centre). "
                     "The split is the script default; the authoritative split of the trained run is train_cities/val_cities in "
                     "pre_processing/patches/meta.json, which is not available locally.",
            "n_cities": len(rows), "n_val": sum(r["split"] == "val" for r in rows), "cities": rows}


# ---------------------------------------------------------------- model.json (checkpoint metadata)
def model_meta():
    p = ROOT / "Deploy" / "best.pt"
    out = {"_source": rel(p), "file_bytes": p.stat().st_size}
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=True)   # tensors, numbers, strings and lists only
        out.update({"parameters": int(sum(t.numel() for t in ck["model"].values())), "tensors": len(ck["model"]),
                    "target": ck["target"], "base_channels": ck["base"], "n_input_channels": len(ck["features"]),
                    "features": ck["features"], "landcover_codes": ck["landcover_codes"],
                    "target_norm_mu": ck["mu"], "target_norm_sd": ck["sd"]})
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"PyTorch could not read the checkpoint here ({type(e).__name__}); only the file size is reported"
    return out


# ---------------------------------------------------------------- metrics from raw CSVs or README
def build_from_readme():
    lines = README.read_text(encoding="utf-8").splitlines()
    R = rel(README)
    res = {}

    # per-city table (final model, 10 held-out cities)
    a, b, hdr, rows = md_table(lines, "| City | MAE")
    keys = ["city", "MAE", "R2_within", "R2_anomaly", "spearman", "hot_AUC", "hot_F1", "hot_bias"]
    recs = [dict(zip(keys, (num(c) for c in r))) for r in rows]
    mean = [r for r in recs if str(r["city"]).lower() == "mean"]
    res["per_city"] = {"_source": f"{R} lines {a}-{b} (Full metrics: Final model, city by city)",
                       "_note": "Final model on the 10 held-out validation cities. MAE in C with each city's mean given (pattern error). "
                                "R2_within = R2 inside the city as scored; R2_anomaly = with the city's constant offset removed. "
                                "hot_* use the hottest 10 % of each city; hot_bias in C. Best epoch was picked on these cities (slightly optimistic).",
                       "readme_columns": hdr, "cities": [r for r in recs if r not in mean], "mean": mean[0] if mean else None}

    # first vs final headline
    a, b, hdr, rows = md_table(lines, "| | First model | Final model |")
    head = []
    for r in rows:
        first, final = r[1], r[2]
        if "/" in first:
            f1, f2 = (num(x) for x in first.split("/"))
            g1, g2 = (num(x) for x in final.split("/"))
            head += [{"metric": "Hotspot AUC", "first": f1, "final": g1}, {"metric": "Hotspot F1", "first": f2, "final": g2}]
        else:
            head.append({"metric": r[0], "first": num(first), "final": num(final)})
    hl_a, _ = find_line(lines, "**R² 0.74** inside the city")
    res["headline"] = {"_source": f"{R} lines {a}-{b} (How well does it work?)", "rows": head}

    # all models
    a, b, hdr, rows = md_table(lines, "| Model | Target |")
    keys = ["model", "target", "inputs", "epochs", "R2_anomaly", "spearman", "hot_AUC", "hot_F1", "hot_bias", "cities_worse_than_city_mean"]
    res["models"] = {"_source": f"{R} lines {a}-{b} (All trained models, same validation pixels; produced by models/compare_runs.py)",
                     "rows": [dict(zip(keys, (num(c) for c in r))) for r in rows]}

    # land cover
    a, b, hdr, rows = md_table(lines, "| Land cover | Pixels |")
    res["landcover"] = {"_source": f"{R} lines {a}-{b} (Error by land cover, final model)",
                        "rows": [{"landcover": num(r[0]), "pixels_millions": num(r[1]), "MAE": num(r[2]), "bias": num(r[3])} for r in rows]}

    # hotspot detection (Delhi / Jaipur)
    a, b, hdr, rows = md_table(lines, "| | Delhi | Jaipur |")
    hs = {"Delhi": {}, "Jaipur": {}}
    for r in rows:
        label = r[0]
        for city, cell in zip(("Delhi", "Jaipur"), r[1:3]):
            if label.startswith("Blocks flagged"):
                hs[city]["blocks_flagged_pct"] = num(cell)
            elif label.startswith("Clusters kept"):
                m = re.match(r"(\d+)\s*\(([\d.]+)\s*%", cell)
                hs[city]["clusters_kept"], hs[city]["clusters_area_pct"] = int(m.group(1)), float(m.group(2))
            elif label.startswith("Overlap"):
                p, rc, iou = (num(x) for x in cell.split("/"))
                hs[city].update({"precision": p, "recall": rc, "IoU": iou})
            elif label.startswith("Correlation"):
                hs[city]["block_mean_corr"] = num(cell)
    res["hotspot_detection"] = {"_source": f"{R} lines {a}-{b} (Hotspot detection quality: Gi* z >= 2.58, clusters >= 1 C above neighbours)",
                                "cities": hs}

    # cooling doses
    a, b, hdr, rows = md_table(lines, "| Area | Hotter than neighbours by |")
    cool = []
    for r in rows:
        m = re.match(r"([\d.]+) °C observed, ([\d.]+) °C seen by the model", r[1])
        cool.append({"area": r[0], "obs_gap_C": float(m.group(1)), "pred_gap_C": float(m.group(2)), "dose_text": r[2],
                     "veg_dose_pct": int(re.search(r"plant (\d+) %", r[2]).group(1)) if re.search(r"plant (\d+) %", r[2]) else None})
    res["cooling"] = {"_source": f"{R} lines {a}-{b} (cooling dose table; produced by models/simulate_cooling.py)",
                      "_note": "Model's view (learned associations, not measured effects). The full dose curves (CSV per area) are only on the VM.",
                      "rows": cool}

    # group importance (only the numbers stated in prose)
    gl, gtxt = find_line(lines, "shuffling that group of features drops R² by")
    m = re.search(r"drops R² by ([\d.]+), against ([\d.]+) for water, ([\d.]+) for built-up and under ([\d.]+) for the rest", gtxt)
    res["group_importance"] = {"_source": f"{R} line {gl} (prose; full table data/unet_run3/group_importance.csv is on the VM)",
                               "rows": [{"group": "vegetation", "R2_drop": float(m.group(1))}, {"group": "water", "R2_drop": float(m.group(2))},
                                        {"group": "built-up", "R2_drop": float(m.group(3))}],
                               "rest_below": float(m.group(4))}

    # vegetation sanity check
    vl, vtxt = find_line(lines, "For +0.1 NDVI the model predicts a cooling of")
    vl2, vtxt2 = find_line(lines, "places with the same land cover")
    m1 = re.search(r"cooling of ([\d.]+) °C in Delhi and ([\d.]+) °C in Jaipur", vtxt)
    m2 = re.search(r"shows ([\d.]+) and ([\d.]+) °C", vtxt2)
    res["vegetation_check"] = {"_source": f"{R} lines {vl}-{vl2}",
                               "per_0_1_NDVI_cooling_C": [{"city": "Delhi", "model": float(m1.group(1)), "data": float(m2.group(1))},
                                                          {"city": "Jaipur", "model": float(m1.group(2)), "data": float(m2.group(2))}]}

    # data / training facts (kept as text)
    a, b, hdr, rows = md_table(lines, "| Cities | 50,")
    facts = [{"item": hdr[0], "value": hdr[1]}] + [{"item": r[0], "value": r[1]} for r in rows]
    res["facts"] = {"_source": f"{R} lines {a}-{b} (Data and training facts)", "rows": facts}
    return res


def build_from_raw(res):
    """Override README-derived sections with raw pipeline CSVs where those exist (after an Azure download)."""
    used = []
    p = raw("data/unet_run3/eval_val/eval_per_city.csv")
    if p:
        recs = read_csv(p)
        res["per_city"] = {"_source": rel(p), "cities": recs,
                           "mean": {k: sum(r[k] for r in recs) / len(recs) for k in recs[0] if k != "city" and isinstance(recs[0][k], (int, float))}}
        used.append(p)
    p = raw("data/unet_run3/eval_val/eval_by_landcover.csv")
    if p:
        res["landcover"] = {"_source": rel(p), "rows": read_csv(p)}
        used.append(p)
    p = raw("data/model_comparison.csv")
    if p:
        res["models"] = {"_source": rel(p), "rows": read_csv(p)}
        used.append(p)
    p = raw("data/unet_run3/group_importance.csv")
    if p:
        res["group_importance"] = {"_source": rel(p), "rows": read_csv(p)}
        used.append(p)
    p = raw("data/unet_run3/metrics.csv")
    if p:
        res["training_curve"] = {"_source": rel(p), "rows": read_csv(p)}
        used.append(p)
    hs = [q for d in RAW_DIRS for q in sorted((d / "data" / "hotspots").glob("*_hotspots.csv"))] if any((d / "data" / "hotspots").exists() for d in RAW_DIRS) else []
    if hs:
        res["hotspots"] = {"_source": [rel(q) for q in hs], "cities": {q.stem.replace("_hotspots", ""): read_csv(q)[:50] for q in hs}}
        used += hs
    return used


def main():
    print(f"repo root: {ROOT}")
    write("cities.json", city_config())
    write("model.json", model_meta())
    res = build_from_readme()
    used = build_from_raw(res)
    print(f"raw pipeline CSVs found: {len(used)}" + ("" if used else " (none: using the README tables)"))

    pc = res["per_city"]
    write("per_city.json", pc)
    metrics = {"_source": sorted({s for k in ("headline", "models", "landcover", "hotspot_detection", "group_importance", "vegetation_check", "facts", "training_curve")
                                  if k in res for s in ([res[k]["_source"]] if isinstance(res[k]["_source"], str) else res[k]["_source"])}),
               **{k: res[k] for k in ("headline", "models", "landcover", "hotspot_detection", "group_importance", "vegetation_check", "facts", "training_curve") if k in res}}
    write("metrics.json", metrics)
    write("cooling.json", res["cooling"])
    if "hotspots" in res:
        write("hotspots.json", res["hotspots"])
    figs = sorted((ROOT / "docs").glob("fig_*.png"))
    write("figures.json", {"_source": "docs/", "_note": "Figures redrawn by the authors from the evaluation / comparison / importance / simulator output; "
                                                         "fig_delhi_map.png is a crop of a real simulate_cooling.py figure (README.md).",
                           "figures": [{"file": rel(f), "bytes": f.stat().st_size} for f in figs]})
    print("grid preview: no city grid (<City>_temp.nc / _features.nc) or prediction cache (*_pred.npy) exists locally, so none was built")


if __name__ == "__main__":
    sys.exit(main())
