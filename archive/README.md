# Archive

Scripts kept for reference and reuse, not maintained: they are excluded from linting and the
quality gate, and may need their paths or flags updated before they run again.

- `paper/` — figure and table scripts of the published papers (run as `python -m archive.paper.<name>`).
- `runners/phases/`, `runners/tto/` — launch scripts of the historical training phases and TTO series.
- `runners/topology/`, `runners/eval/` — one-off topology probes and the deployed-Jacobian figure.
- `runners/export.sh` — export of the P18 loss-ablation checkpoints.

Larger closed lines live at git tags, not here: `stage5-final` holds the Stage 5 pipeline, the
search-engine primitives (C0–C7) and the precision comparison. Read a file from a tag with
`git show stage5-final:<path>`, or check the whole tree out beside this one with
`git worktree add ../CTCF-stage5 stage5-final`.
