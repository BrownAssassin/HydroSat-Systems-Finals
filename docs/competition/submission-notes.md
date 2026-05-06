# Submission Notes

## Competition Workflow

- the official evaluation happens through the competition-provided GitLab repository
- only pushes to `main` trigger automatic submission
- each team gets 3 submission attempts per day
- the platform is inference-only, not a training environment

## Runtime Paths

Use absolute container paths:

- `/workspace` for code
- `/input` for mounted inference data
- `/output` for generated prediction files

This repo's root scaffolding is aligned to those paths:

- `Dockerfile`
- `.gitlab-ci.yml`
- `run.sh`

## Important Contract Mismatch

The generic submission manual uses `result.json` examples from other tracks. For this project, follow the Track 2 task PDF instead:

- `/output/result_turbidity.json`
- `/output/result_chla.json`

That rule is captured directly in `src/hydrosat/contracts.py`.

## Submission Strategy

This GitHub repo remains the full development repo.

When we are ready to submit:

1. export the submission-safe subset with `python scripts/export_submission.py --dest <gitlab-repo>`
2. merge the root path settings into the official GitLab template
3. confirm the output contract and entrypoint before pushing to `main`

## Base Image Notes

The round-2 manual references a GPU-ready PyTorch competition base image and a `/workspace` working directory.

The root `Dockerfile` keeps that assumption, but the exact base image tag in the official GitLab template should be treated as the final authority once we access the submission repo.
