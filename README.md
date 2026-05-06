# HydroSat Systems Finals

Final-round workspace for `Track 2: Space Intelligence Promoting Water Quality` in the `ITU AI and Space Computing Challenge`.

The repo is now set up as the main R&D home for the project. It includes:

- a normalized local raw-data layout under `data/raw/`
- a lightweight Python package scaffold under `src/hydrosat/`
- submission-facing root files for Docker and GitLab handoff
- validation tests for the current data layout and Track 2 output contract

## Current Status

This phase establishes structure and contracts only. The real training and inference pipeline is still to be implemented.

The default entrypoint already points at `python -m hydrosat.infer`, but it intentionally fails with a clear message until model loading and prediction logic are added.

## Repo Layout

```text
data/raw/
  guides/
  sample_input/
  train/
docs/competition/
models/
artifacts/
scripts/
src/hydrosat/
tests/
```

### Data Layout

`data/raw/guides/`

- final-round task description PDF
- round 2 submission manual PDF

`data/raw/sample_input/`

- `area8_images/`
- `track2_turb_test_point.csv`
- `track2_cha_test_point.csv`

`data/raw/train/`

- `area1`, `area2`, `area3`, `area5`, `area6`, `area7`

The raw data stays local and untracked. It is excluded from git and from Docker builds.

## Track 2 Output Contract

The authoritative Track 2 output files are:

- `/output/result_turbidity.json`
- `/output/result_chla.json`

Each JSON must map keys in the form:

```text
{filename}_{Lon}_{Lat}
```

to a single numeric prediction inside a list, for example:

```json
{
  "area8_2024-09-13.tif_-122.666591_45.507326": [1.45]
}
```

## Environment Defaults

- `HYDROSAT_DATA_ROOT=data/raw`
- `HYDROSAT_MODELS_DIR=models`
- `INPUT_DIR=/input`
- `OUTPUT_DIR=/output`

## Useful Commands

Validate the local scaffold and contracts:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Check that the inference entrypoint can discover an input layout without running a model:

```bash
PYTHONPATH=src python -m hydrosat.infer --check-layout --input-root data/raw/sample_input --output-dir artifacts/output
```

Export the submission-safe subset into a separate GitLab working tree:

```bash
python scripts/export_submission.py --dest /path/to/track2_round2_model
```

## Notes

- When the generic submission manual conflicts with the Track 2 task PDF, follow the Track 2 task PDF for output naming.
- The root `.gitlab-ci.yml` is a scaffold for local alignment. Before pushing to the official competition GitLab repo, merge its variables and paths with the template provided there.
