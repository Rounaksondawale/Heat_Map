#!/bin/bash
# One command: build the 19 new cities, fill gaps, clean, add context features, train the combined model.
#
#   nohup bash extraction/run_new_cities.sh > pipeline.log 2>&1 &
#   tail -f pipeline.log
#
# Run it from the repo root (Heat_Map). Run it ONCE. Cities already built are skipped, so it is safe to re-run
# after an interruption. Every step prints a banner, so you can see where it is.
set -e
cd "$(dirname "$0")/.."

OUT=data/multi_city
NEW="Shimla Dehradun Ranchi Shillong Visakhapatnam Surat Kochi Thiruvananthapuram Mangaluru Jodhpur Indore Bhopal Amritsar Ludhiana Patna Raipur Coimbatore Vadodara Kanpur"
WORKERS=3          # cities built in parallel; each needs a few GB of RAM (32 GB is fine for 3)

echo "=== 1/5 build new cities (Landsat, DEM, GHSL, weather)  $(date)"
python -u extraction/build_dataset.py --cities $NEW --out $OUT --workers $WORKERS || echo "some cities failed; continuing with the rest"

echo "=== 2/5 fill weather / layer gaps in the new cities  $(date)"
python -u extraction/repair_dataset.py --out $OUT --cities $NEW || echo "repair reported leftovers; see above"

echo "=== 3/5 clean (drop NaN rows, impossible index values) and validate  $(date)"
python -u extraction/finalize_dataset.py --src $OUT/all_cities_training.csv.gz --allow-missing-cities

echo "=== 4/5 add surroundings (context) features  $(date)"
python -u extraction/add_context_features.py --root $OUT

echo "=== 5/5 train the combined model (leave-one-city-out + final model)  $(date)"
python -u models/train_lgbm.py --data $OUT/all_cities_clean_ctx.csv.gz \
  --context --no-met --city-norm --target z --no-maps --simple --drop-feats ELEV LANDCOVER \
  --tag combined_v2

echo "=== DONE  $(date)"
