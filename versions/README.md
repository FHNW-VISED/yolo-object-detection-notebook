# Versions

One folder per version of the notebook: `SPEC.md` and the built notebook exactly as they were.
The newest folder is always the notebook on `main`. Make one with `just version "what changed"`
before merging to main; `just version-check` fails if `main` has moved on without one.
Never edit a version folder.

| Version | Date | What changed | Commit |
|---|---|---|---|
| v01 | 2026-11-09 | v01: weed-crop-aerial dataset (drone, 2 classes), YOLO11n, PR/AP + NMS sliders | 2704d21 |
| v02 | 2026-11-09 | v02: section 07 timer and frozen-backbone answer fixed after the full run | a971528 |
