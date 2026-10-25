# Leakage taxonomy

Every finding carries one of these codes.

## Split integrity

### `ENTITY_OVERLAP`: Entity overlap across splits

The same patient, encounter or unit stay contributes records to more than one split, so the model is evaluated on people it has already seen.

*Example:* Random row-level split of ICU stays when patients have repeat admissions.

*Remedy:* Split at the patient level (or higher) and keep all of a patient's records in one split.

### `DUPLICATE_RECORD`: Duplicate records across splits

Identical feature vectors appear in both training and evaluation data, which rewards memorisation rather than generalisation.

*Example:* Random oversampling or record duplication performed before splitting.

*Remedy:* De-duplicate before splitting and resample only inside the training split.

### `SITE_OVERLAP`: Site overlap in a claimed external validation

Hospitals or units used for training also appear in data presented as external.

*Example:* Hospital-held-out evaluation in a multi-centre database that reuses hospitals.

*Remedy:* Hold out complete sites and verify the site identifiers are disjoint.

### `SPLIT_ORDER`: Temporal order violation

A split presented as temporal validation trains on data that is later than, or concurrent with, the evaluation period.

*Example:* Temporal validation in a database whose dates are shifted per patient.

*Remedy:* Order by a calendar variable that is valid across patients and keep periods disjoint.

## Temporal

### `FUTURE_EVENT`: Information recorded after the prediction time

Features are computed from measurements, orders or notes whose clinical time is later than the moment the prediction is supposed to be made.

*Example:* Using the maximum creatinine over the whole ICU stay in a 24-hour model.

*Remedy:* Restrict every feature to events at or before the prediction time.

### `AVAILABILITY_LAG`: Charted before but not available at prediction time

An event is time-stamped before the prediction time but was stored, verified or resulted afterwards, so it could not have been used in real time.

*Example:* A blood culture drawn at 10:00 whose organism result was stored two days later.

*Remedy:* Filter on the storage or entry time rather than the clinical time.

### `INTERVAL_CROSSING`: Interval events crossing the prediction time

Events with a start and end time (infusions, ventilation, medication orders) start before the prediction time but their totals include volume or duration after it.

*Example:* Total vasopressor amount for an infusion still running at the prediction time.

*Remedy:* Truncate interval events at the prediction time or prorate their amounts.

### `UNDATED_SOURCE`: Undated or day-resolution sources

Features come from sources without a usable time stamp, which are typically finalised at discharge or only known at day resolution.

*Example:* Billing diagnosis codes from the current admission used in an admission-time model.

*Remedy:* Use codes from previous encounters only, or assign a conservative availability time.

## Feature content

### `HINDSIGHT_VARIABLE`: Hindsight variable

A feature is only known at or after discharge, or summarises the whole stay, such as length of stay, discharge location or discharge weight.

*Example:* Length of ICU stay used as a predictor of in-hospital mortality.

*Remedy:* Remove the variable or replace it with a version truncated at the prediction time.

### `OUTCOME_ENCODING`: Outcome encoded in a feature

A feature directly records the outcome or a deterministic function of it.

*Example:* A 'died in hospital' flag or death time stored in a severity table.

*Remedy:* Remove every column derived from the outcome.

### `OUTCOME_PROXY`: Outcome proxy

A feature records a decision or event that is closely tied to the outcome and usually happens close to it, such as comfort-care orders, palliative care encounters, do-not-resuscitate codes or brain death codes.

*Example:* A palliative care encounter code used to predict death in the same admission.

*Remedy:* Exclude proxies or restrict them to documentation before the prediction time and report it.

### `FIRST_DAY_SUMMARY`: First-day summary used before the first day ends

Severity scores and worst values summarising the first 24 hours are used by a model that predicts earlier than 24 hours after admission.

*Example:* APACHE worst-of-day-one variables in a model scored at ICU admission plus 6 hours.

*Remedy:* Recompute the summary over the window that ends at the prediction time.

### `SINGLE_FEATURE_DOMINANCE`: Implausibly strong single feature

One feature on its own separates outcomes almost perfectly, which in clinical prediction is far more often a leak than a discovery.

*Example:* A treatment code that is only ever recorded for patients who died.

*Remedy:* Trace the feature to its source table and time stamp before trusting it.

### `MISSINGNESS_PROXY`: Missingness pattern encodes the outcome

Whether a value is missing predicts the outcome strongly, often because the value is only recorded after the outcome or only for patients who survive long enough.

*Example:* A discharge measurement that is missing for every patient who died.

*Remedy:* Check when the variable is recorded relative to the outcome and the prediction time.

## Cohort and label

### `POST_OUTCOME_INSTANCE`: Outcome already present at prediction time

Positive instances where the outcome occurred at or before the prediction time, so the model detects rather than predicts it.

*Example:* Patients already in acute kidney injury at the prediction time kept as positives.

*Remedy:* Exclude instances whose outcome precedes the prediction time, or add a prediction gap.

### `COHORT_FUTURE_INFORMATION`: Eligibility decided with future information

Inclusion or exclusion criteria use information that only becomes available after the prediction time, which selects patients by their future.

*Example:* Requiring an ICU stay of at least 48 hours for a model scored at 24 hours.

*Remedy:* Define eligibility only from information available at the prediction time.

## Pipeline

### `PREPROCESSING_FIT`: Preprocessing fitted on evaluation data

Scalers, imputers or encoders learned their parameters from data that includes the evaluation split.

*Example:* StandardScaler fitted on the full dataset before train_test_split.

*Remedy:* Fit every learned transformation on the training split only, ideally inside a Pipeline.

### `SYNTHETIC_IN_EVALUATION`: Synthetic or resampled records in evaluation data

Oversampling such as SMOTE was applied before splitting, so the evaluation data contains interpolated records built from training patients.

*Example:* SMOTE on the full cohort followed by a random 70/30 split.

*Remedy:* Resample only inside the training split or inside each cross-validation fold.

### `EVALUATION_REUSE`: Decisions tuned on evaluation data

Thresholds, hyperparameters or model choices were selected using the evaluation split, which then no longer gives an unbiased estimate.

*Example:* Choosing the alert threshold that maximises Youden's J on the test set.

*Remedy:* Tune on a validation split or inside cross-validation and touch the test split once.

# Checks

| Check | Leakage type | What it does |
|---|---|---|
| `entity_overlap` | `ENTITY_OVERLAP` | Each patient, admission and stay should belong to exactly one split. |
| `site_overlap` | `SITE_OVERLAP` | Sites shared between training and evaluation |
| `duplicate_records` | `DUPLICATE_RECORD` | Identical feature vectors in training and evaluation |
| `temporal_order` | `SPLIT_ORDER` | Temporal order of training and evaluation periods |
| `event_timing` | `FUTURE_EVENT` | Uses an event-level provenance log (the rows each feature was computed from) to find events recorded after the prediction time, results not yet available, intervals running past the prediction time and undated sources. |
| `truncation_invariance` | `FUTURE_EVENT` | Rebuilds features with the user's own feature function from events truncated at the prediction time and compares them with features built from the full record. Any difference means the feature function reads information from after the prediction time. |
| `outcome_timing` | `POST_OUTCOME_INSTANCE` | Outcome already present at prediction time |
| `eligibility_timing` | `COHORT_FUTURE_INFORMATION` | Cohort eligibility decided after prediction time |
| `hindsight_features` | `HINDSIGHT_VARIABLE` | Traces each feature to its source table and column (from feature_sources, or from the dataset registry by name) and flags outcome fields, discharge-time fields, undated billing sources and first-day summaries used before the first day is over. |
| `outcome_proxies` | `OUTCOME_PROXY` | Flags features that represent treatment-limitation, palliative, end-of-life or death-related documentation, from feature names, source tables and categorical values. |
| `single_feature_dominance` | `SINGLE_FEATURE_DOMINANCE` | Single features that separate the outcome almost perfectly |
| `missingness_proxy` | `MISSINGNESS_PROXY` | Missingness patterns that predict the outcome |
| `preprocessing_fit` | `PREPROCESSING_FIT` | Recomputes the statistics stored in fitted scalers and imputers on the training rows and on all rows, and reports which one the fitted object matches. |
| `synthetic_samples` | `SYNTHETIC_IN_EVALUATION` | Interpolating oversamplers such as SMOTE create minority-class rows with fractional values in columns that are otherwise whole numbers (flags, counts, scores). Finding such rows in an evaluation split shows the oversampling happened before the split. |
| `threshold_selection` | `EVALUATION_REUSE` | Decision threshold tuned on evaluation data |
| `declarations` | `EVALUATION_REUSE` | Records practices that cannot be verified from data, as declared by the analyst. |
