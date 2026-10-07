# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's disease UPDRS score prediction (regression)
- **Goal:** Predict the total UPDRS motor score (`target`) from longitudinal patient visit features.
- **Last experiment:** n/a
- **Last result:** n/a

<!--
Workspace decisions: one-time project-setup choices. Record each when
it is made and treat it as fixed unless you deliberately change one
(e.g. switch pandas → polars), updating the recorded date. Reading
this block on later sessions avoids re-deciding what's already settled.
-->

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-11
  - env manager: <pixi | uv | poetry | hatch | conda | pip+venv> - recorded: <YYYY-MM-DD>
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: <name1, name2 | none> - recorded: <YYYY-MM-DD>
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: <local | hub | mlflow> - recorded: <YYYY-MM-DD>
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: <KFold | StratifiedKFold | GroupKFold | TimeSeriesSplit | other> - recorded: <YYYY-MM-DD>

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done - 2025-07-11
- **Summary:** 44,590 train rows × 13 features; continuous UPDRS motor score (mean 37.5, range 0–109.5, right-skewed tail). `patient_id` repeats across rows (≈8 visits/patient) → `GroupKFold` on `patient_id` warranted to prevent same-patient leakage. `off` (r = 0.871) and `on` (r = 0.669) are the dominant predictors but may be same-visit measurements — potential leakage to confirm. `ledd`, `on`, `off`, `time_since_intake_*` carry 30–79 % missingness. No datetime column found.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| <!-- e.g. `01_baseline` --> | <!-- "tabular_pipeline on raw features" --> | <!-- done --> | <!-- "ROC-AUC 0.86 ± 0.01" --> | <!-- [design note](01_baseline.md) --> |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
| <!-- B1 --> | <!-- "investigate target-bin>0.95 residual bias via target transform" --> | <!-- `skore:01_baseline` --> |
