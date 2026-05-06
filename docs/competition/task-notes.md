# Track 2 Task Notes

## Final-Round Algorithm Task

The final round shifts from preliminary-round water segmentation to point-wise water-quality inversion.

Required prediction targets:

- turbidity
- chlorophyll-a concentration

Both are continuous-value regression tasks tied to labeled point observations over multispectral imagery.

## Training Data Shape

The normalized local layout now lives under `data/raw/train/` and preserves the official per-area structure:

- `area1`
- `area2`
- `area3`
- `area5`
- `area6`
- `area7`

Each area contains:

- `areaX_images/`
- `track2_turb_train_point_areaX.csv`
- optionally `track2_cha_train_point_areaX.csv`

Notes:

- `area2` and `area3` do not include chlorophyll-a labels.
- each CSV row is a point record linked to an image by the `filename` column
- the intended training unit is a local image patch centered on each labeled point

## Hidden Test Shape

The competition platform mounts an inference-only input tree under `/input`, matching the sample layout:

```text
/input/
  area8_images/
  track2_turb_test_point.csv
  track2_cha_test_point.csv
```

## Authoritative Output Contract

For Track 2, write exactly these two files:

- `/output/result_turbidity.json`
- `/output/result_chla.json`

Each file should be a JSON object with keys in the form:

```text
{filename}_{Lon}_{Lat}
```

and a single numeric prediction inside a list.

## Scoring Notes

The task PDF describes:

- RMSE
- R²
- NRMSE-based score normalization
- final algorithm score as the average of the turbidity and chlorophyll-a scores

This repo should treat the Track 2 task PDF as the source of truth if it conflicts with the generic round-2 manual.
