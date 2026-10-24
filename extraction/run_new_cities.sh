#!/bin/bash
# One command: build the 36 new cities (to reach 50), fill gaps, clean, add context features, train the combined model.
#
#   nohup bash extraction/run_new_cities.sh > pipeline.log 2>&1 &
#   grep "===" pipeline.log          # progress
#
# Run it from the repo root (Heat_Map). Run it ONCE. Cities already built are skipped, so it is safe to re-run
# after an interruption. Core use:
#   step 1  downloads dominate (network-bound), so more cities run at once than there are cores,
#           limited by free RAM (about 5 GB per city)
#   step 4  one process per city
#   step 5  CV folds train in parallel, each with cores/folds threads (the training script picks this itself)
#   steps 3-5 pass a fast parquet file around instead of re-reading / re-writing a 2.7M-row gzip CSV
set -e
cd "$(dirname "$0")/.."

# Use conda's newer C++ runtime; the system libstdc++ is too old for the geo packages (CXXABI_1.3.15 error).
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/anaconda/envs/azureml_py38}/lib:${LD_LIBRARY_PATH:-}"
[ -d /anaconda/envs/azureml_py38/lib ] && export LD_LIBRARY_PATH="/anaconda/envs/azureml_py38/lib:${LD_LIBRARY_PATH}"

# Abort here, with a clear message, if the build packages cannot be imported (instead of failing every city).
python -c "import rioxarray, rasterio, odc.stac, planetary_computer, pystac_client, geopandas"   || { echo "ERROR: build packages do not import; fix the environment first (see the error above)"; exit 1; }

OUT=data/multi_city
DEFAULT_NEW="Shimla Dehradun Ranchi Shillong Visakhapatnam Surat Kochi Thiruvananthapuram Mangaluru Jodhpur Indore Bhopal Amritsar Ludhiana Patna Raipur Coimbatore Vadodara Kanpur Srinagar Jammu Agartala Imphal Puducherry Panaji Madurai Bikaner Ajmer Udaipur Rajkot Nashik Aurangabad Agra Varanasi Jabalpur Gwalior"
NEW="${*:-$DEFAULT_NEW}"          # pass city names as arguments to build only those (a smaller batch)

CORES=$(nproc)
MEM_GB=$(free -g | awk '/^Mem:/ {print $7}')               # available RAM
# Step 1 is network-bound: cities wait on downloads, so run well more of them than there are cores and give each
# 2 threads for concurrent reads. Cap by RAM (about 4 GB per city). Override:  WORKERS=8 bash extraction/run_new_cities.sh
WORKERS=${WORKERS:-$(( CORES * 2 ))}
MAX_BY_RAM=$(( MEM_GB / 4 ))
[ "$WORKERS" -gt "$MAX_BY_RAM" ] && WORKERS=$MAX_BY_RAM
[ "$WORKERS" -lt 2 ] && WORKERS=2
echo "cores=$CORES  free RAM=${MEM_GB}GB  -> building $WORKERS cities at a time"

echo "=== 1/5 build new cities (Landsat, DEM, GHSL, weather)  $(date)"
python -u extraction/build_dataset.py --cities $NEW --out $OUT --workers $WORKERS --dask-threads 2 || echo "some cities failed; continuing with the rest"

echo "=== 2/5 fill weather / layer gaps in the new cities  $(date)"
python -u extraction/repair_dataset.py --out $OUT --cities $NEW --pause 8 || echo "repair reported leftovers; see above"

echo "=== 3/5 clean (drop NaN rows, impossible index values) and validate  $(date)"
python -u extraction/finalize_dataset.py --src $OUT/all_cities_training.csv.gz --allow-missing-cities --no-csv

echo "=== 4/5 add surroundings (context) features  $(date)"
python -u extraction/add_context_features.py --root $OUT --no-csv

echo "=== 5/5 train the combined model (leave-one-city-out + final model)  $(date)"
python -u models/train_lgbm.py --data $OUT/all_cities_clean_ctx.parquet \
  --context --no-met --city-norm --target z --no-maps --simple --drop-feats ELEV LANDCOVER \
  --tag combined_v2

echo "=== DONE  $(date)"
