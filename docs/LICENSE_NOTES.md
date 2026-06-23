# License Notes

This file is a release checklist, not legal advice.

## Code

- This repository scaffolding is MIT by default.
- Qwen-Image / Qwen-Image-Edit model cards list Apache-2.0.
- EditAR repository is published as open-source research code; verify its current repository license before redistributing modifications.
- GroundingDINO, SAM, and any MLLM/VQA evaluators must keep their upstream licenses.

## Data

- MagicBrush: public train/dev on Hugging Face; test split has redistribution restrictions. Do not commit unzipped test images.
- Pico-Banana-400K: non-commercial/no-derivatives style restrictions are reported by the project; do not redistribute source images.
- GIE-Bench: project page says data is CC-BY-NC-ND. Use it for evaluation and keep local copies out of Git.
- Emu Edit Test Set: Hugging Face README lists CC-BY-NC 4.0.
- Reason-Edit: verify the current license before public release.

## Git Ignore Policy

Keep these local only:

- `data/raw/`
- `data/processed/` when it contains copied images
- `data/candidates/`
- `data/outputs/`
- `checkpoints/`
- `external/`
- `reports/*.png` if generated from restricted datasets

It is safe to commit:

- processing scripts
- configs
- aggregated CSV metrics without source images
