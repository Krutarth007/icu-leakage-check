# Codebook for the leakage indicators

Allowed values: `risk`, `unclear`, `safe`, `not_mentioned`, `na`. Code what the paper reports, not what the authors
probably did. A blank `_human` cell means the automated value was checked and is correct.

## I01: Split allows the same patient in training and test

* Part 2 mechanism: E01b (Row-level random split (as reported))
* Leakage type: `ENTITY_OVERLAP`
* Rule: Random row-level split or cross-validation without a first-stay rule or a patient-level split.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I02: Features summarised over the whole stay

* Part 2 mechanism: E02 (Look-ahead: features beyond the prediction time)
* Leakage type: `FUTURE_EVENT`
* Rule: Worst, highest, lowest or mean values during the whole ICU or hospital stay, rather than up to a prediction time.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I03: Result availability time not handled

* Part 2 mechanism: E03 (Clinical-time filter ignoring result availability)
* Leakage type: `AVAILABILITY_LAG`
* Rule: Safeguard: features restricted to results available (stored or verified) by the prediction time.
* `risk` for a safeguard indicator means the safeguard is not reported.

## I04: Running infusions or ventilation not truncated at the prediction time

* Part 2 mechanism: E04 (Full totals of running infusions and ventilation)
* Leakage type: `INTERVAL_CROSSING`
* Rule: Safeguard: interval records (infusions, ventilation, urine output) truncated or prorated at the prediction time.
* `risk` for a safeguard indicator means the safeguard is not reported.

## I05: Length of stay or discharge information used as a predictor

* Part 2 mechanism: E05 (Discharge fields and length of stay)
* Leakage type: `HINDSIGHT_VARIABLE`
* Rule: Length of stay, discharge location or discharge status named among predictors (not as an outcome).
* `risk` means the practice is described; `safe` means the safeguard is described.

## I06: Outcome or outcome-derived fields used as predictors (eICU-CRD)

* Part 2 mechanism: E06 (Outcome column next to predictors)
* Leakage type: `OUTCOME_ENCODING`
* Rule: APACHE predicted mortality, recorded discharge status or similar outcome fields named among predictors.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I07: Comorbidities from same-admission diagnosis codes

* Part 2 mechanism: E07 (Same-admission billing codes and diagnoses)
* Leakage type: `UNDATED_SOURCE`
* Rule: Charlson, Elixhauser or comorbidities derived from ICD codes without restriction to earlier admissions.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I08: Code status or care limitation used as a predictor

* Part 2 mechanism: E08 (Care limitation and code status, any time)
* Leakage type: `OUTCOME_PROXY`
* Rule: DNR, code status, palliative or comfort care named among predictors without a before-prediction restriction.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I09: First-day severity scores in an earlier prediction

* Part 2 mechanism: E09 (First-day severity summaries in an earlier model)
* Leakage type: `FIRST_DAY_SUMMARY`
* Rule: APACHE, SAPS, SOFA or OASIS from the first 24 hours used by a model that predicts earlier than 24 hours.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I10: Patients with the outcome at the prediction time not excluded

* Part 2 mechanism: E10b (Patients already in the outcome kept (as reported))
* Leakage type: `POST_OUTCOME_INSTANCE`
* Rule: Safeguard: patients who already had the outcome (for example AKI or sepsis) before the prediction time were excluded.
* `risk` for a safeguard indicator means the safeguard is not reported.

## I11: Imputation or scaling fitted on all data

* Part 2 mechanism: E11 (Preprocessing fitted on all rows)
* Leakage type: `PREPROCESSING_FIT`
* Rule: Imputation, normalisation or scaling described on the whole dataset or before the split.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I12: Feature selection on all data

* Part 2 mechanism: E12 (Feature selection on all rows)
* Leakage type: `PREPROCESSING_FIT`
* Rule: LASSO, univariate screening or other selection described on the whole dataset or before the split.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I13: Resampling before the split

* Part 2 mechanism: E13b (Resampling before the split (as reported))
* Leakage type: `SYNTHETIC_IN_EVALUATION`
* Rule: SMOTE, oversampling or undersampling described on the whole dataset or before the split.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I14: Threshold chosen on the test set

* Part 2 mechanism: E14 (Threshold chosen on the test set)
* Leakage type: `EVALUATION_REUSE`
* Rule: Youden or optimal cut-off determined in the test or validation cohort used to report performance.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I15: Hyperparameters or final model chosen on the test set

* Part 2 mechanism: E15 (Hyperparameters or final model chosen on the test set)
* Leakage type: `EVALUATION_REUSE`
* Rule: Tuning on the test set, or the final model picked as the one with the best test performance.
* `risk` means the practice is described; `safe` means the safeguard is described.

## I16: Temporal validation on shifted MIMIC-IV dates

* Part 2 mechanism: E16 (Temporal validation on shifted dates)
* Leakage type: `SPLIT_ORDER`
* Rule: Temporal split in MIMIC-IV without anchor_year_group (dates are shifted per patient).
* `risk` means the practice is described; `safe` means the safeguard is described.

## I17: Overlapping data presented as external validation

* Part 2 mechanism: E17 (Overlapping data presented as external)
* Leakage type: `SITE_OVERLAP`
* Rule: External validation on MIMIC-III for a MIMIC-IV model (2008 to 2012 overlap), or on a split of the same database.
* `risk` means the practice is described; `safe` means the safeguard is described.
