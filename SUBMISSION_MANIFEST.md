# Submission Manifest

## Competition Submission Bundle

The final runnable submission bundle should contain:

- `.gitattributes`
- `.gitlab-ci.yml`
- `.dockerignore`
- `Dockerfile`
- `README.md`
- `README_SUBMISSION.txt`
- `requirements.txt`
- `pyproject.toml`
- `run.sh`
- `src/`
- `artifacts/models/`

These are the files and directories required to build the competition container and run inference on `/input`.

## Local-Only Repo Assets

The following are intentionally kept in the repository workspace for development, scoring, and presentation, but they are **not** part of the GitLab submission image:

- `track2_download_link_1/` to `track2_download_link_5/`
- `artifacts/eval_input/`
- `artifacts/output/`
- `artifacts/reports/`
- `Hydro Sat Systems_Arv Bali_baliarv21@gmail.com/`
- `FINAL_TECHNICAL_PROPOSAL.md`
- `scripts/build_proposal_deck.py`

## Current Runnable Baseline

Runtime entrypoint:

```text
./run.sh
```

Primary runtime model location:

```text
artifacts/models/
```

Current released Area8 offline evaluation:

```text
Turbidity score: 0.0000
Chl-a score:     12.0906
Algorithm score: 6.0453
```

## Output Contract

The package writes both naming variants required by the challenge materials:

```text
turbidity_result.json
chla_result.json
result_turbidity.json
result_chla.json
```
