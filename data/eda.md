<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed - do not invent facts. Keep "Modelling implications" as
candidate suggestions to weigh when designing the model, not final
decisions.
-->

# EDA: Parkinson's UPDRS score prediction

_Generated from `data/eda.py` on 2025-07-11._

## Dataset at a glance

- **Tables:** 3 files — `X_train.csv`, `y_train.csv` (merged for EDA), `X_test.csv`
- **Shape:** 44,590 rows × 13 features + 1 target (train); 11,013 rows × 13 features (test)
- **Target:** `target` — continuous UPDRS (Unified Parkinson's Disease Rating Scale) motor score; **regression task**
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | dtype | null % | n_unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44,590 | Row identifier — surrogate key, not a feature |
| `patient_id` | String | 0 % | 5,576 | Repeats across rows (multiple visits per patient) |
| `cohort` | String | 0 % | 2 | Two cohorts (A / B) — low cardinality |
| `sexM` | Int64 | 0 % | 2 | Binary: 0 = female, 1 = male |
| `gene` | String | **32 %** | 4 | Four genetic variants; ~1 in 3 rows missing |
| `age_at_diagnosis` | Float64 | **5 %** | 563 | Age (years) at Parkinson's diagnosis |
| `age` | Float64 | 0 % | 1,082 | Current age at visit |
| `ledd` | Float64 | **37 %** | 1,320 | L-DOPA equivalent daily dose (mg) |
| `time_since_intake_on` | Float64 | **46 %** | 64 | Hours since last ON-medication dose; heavily missing |
| `time_since_intake_off` | Float64 | **79 %** | 178 | Hours since last OFF-medication dose; very heavily missing |
| `rater_id` | String | 0 % | 50 | 50 distinct raters; potential rater bias |
| `on` | Float64 | **30 %** | 83 | UPDRS score in ON-medication state |
| `off` | Float64 | **42 %** | 101 | UPDRS score in OFF-medication state |

**Highlights:**
- `time_since_intake_off` is missing in 79 % of rows — the most incomplete feature.
- `ledd`, `time_since_intake_on`, `on`, and `off` each missing in 30–46 % of rows.
- `gene` missing in ~32 % — not ignorable.
- No datetime columns detected (no temporal ordering by date).
- `rater_id` has 50 unique values; modest cardinality but may introduce rater-effect bias.

## Target

Continuous UPDRS motor score:

| Stat | Value |
|---|---|
| Mean | 37.5 |
| Std | 16.5 |
| Min | 0.0 |
| 25th pct | 25.6 |
| Median | 37.3 |
| 75th pct | 49.3 |
| Max | 109.5 |

Distribution histogram bins: `[0, 11)` 2,477 · `[11, 22)` 5,641 · `[22, 33)` 9,764 · `[33, 44)` 10,703 · `[44, 55)` 8,844 · `[55, 66)` 5,203 · `[66, 77)` 1,757 · `[77, 88)` 180 · `[88, 99)` 13 · `[99, 109.5]` 8.

The target is **roughly bell-shaped and right-skewed** (the tail above 77 is very thin, 201 rows total). The IQR (23.7 points) spans a meaningful clinical range. No zero-inflation. 867 unique values — effectively continuous.

## Structure

- **No datetime columns detected** — no explicit timestamp column.
- **`patient_id` repeats across rows** (5,576 unique patients across 44,590 rows ≈ 8 visits per patient on average). This is a classic longitudinal panel: multiple observations per patient over time.
- `Index` has unique_ratio = 1.0 (pure row ID — exclude from features).
- `age` and `age_at_diagnosis` are near-continuous floats with high cardinality (1,082 and 563 unique values respectively).

## Associations

**Feature ↔ target (Pearson correlation, sorted):**

| Feature | Pearson r | Cramér's V |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | **0.669** | 0.368 |
| `ledd` | 0.298 | 0.236 |
| `age` | 0.310 | 0.114 |
| `age_at_diagnosis` | 0.133 | 0.051 |
| `time_since_intake_on` | ~0.0 | 0.177 |
| `time_since_intake_off` | ~0.0 | 0.092 |
| `cohort` | n/a (categorical) | 0.090 |
| `gene` | n/a (categorical) | 0.052 |
| `sexM`, `rater_id`, `patient_id` | near 0 | < 0.03 |

**⚠ Leakage flag — `off` and `on`:** `off` has Pearson r = 0.871 with `target`; `on` has r = 0.669. Both are UPDRS sub-scores measured at the same visit as the target. If the prediction task is to predict the total UPDRS score *from sub-scores recorded at the same visit*, these features are valid but effectively near-direct measurements of the target. If the task is to *predict a future score from past features*, these columns may be **leakage** (future measurements not available at prediction time). **This must be confirmed with the task definition.**

**Feature ↔ feature:**
- `age_at_diagnosis` ↔ `age`: Pearson r = 0.942 — near-linear (age is just age-at-diagnosis + disease duration). One of these may be redundant or the difference (disease duration) more informative.
- `on` ↔ `off`: Pearson r = 0.872 — the ON and OFF UPDRS scores are highly correlated.

## Modelling implications

- **Regression task** — use RMSE / MAE as primary metrics. No stratification needed. Standard `KFold` is the default; consider `GroupKFold` on `patient_id` (see below).
- **`patient_id` repeats across rows** → standard `KFold` will leak the same patient into both train and validation folds. **`GroupKFold` on `patient_id`** (or `LeaveOneGroupOut`) is the correct splitter to get an honest out-of-patient generalisation estimate.
- **Heavy missingness in `off`, `on`, `ledd`, `time_since_intake_on/off`** — the pipeline must handle NaNs (imputation or tree-based estimators that support them natively).
- **`off` / `on` leakage risk** — before training, confirm with the task owner whether these columns represent simultaneous measurements or valid lagged history. If simultaneous, consider training with and without them to quantify their effect.
- **`age_at_diagnosis` + `age` collinearity** — consider using disease duration = `age − age_at_diagnosis` as an engineered feature in place of one of them.
- **`rater_id` effect** — 50 raters; a categorical encoding (target encoding or `skrub` dirty-category encoder) may capture rater-level bias.
- **Right-skewed tail (scores > 77)** — ~200 rows out of 44,590. A log or square-root target transform could improve residual homoscedasticity, but the target is otherwise near-normal; evaluate empirically.
- **No temporal structure detected** — `TimeSeriesSplit` not indicated unless a visit date column is revealed.

## Open questions

1. **`off` and `on` leakage** — are the `on`/`off` UPDRS sub-scores measured at the same visit as `target`? If yes, their r ≈ 0.87/0.67 with `target` likely encodes near-direct information and a model will trivially rely on them. Clarify the prediction scenario.
2. **`target` definition** — is `target` the same as or derived from `on` + `off`? If `target` is a linear combination of `on` and `off`, the Pearson r makes sense mechanically.
3. **`time_since_intake_on/off` meaning** — ~0 Pearson correlation but moderate Cramér's V (0.18 / 0.09); the missing-value pattern (46 % and 79 %) suggests most visits didn't record this. Is missingness random or structured (e.g., only OFF-state visits have `time_since_intake_off`)?
4. **Two cohorts (A / B)** — do they differ in collection protocol, patient population, or time period? A cohort indicator may need to be included as a feature or used for stratified splitting.
5. **`gene` missingness** — 32 % missing. Is it missing at random, or are certain genetic subtypes systematically not tested?
