# Leakage audit card: MIMIC-IV in-hospital mortality reference pipeline

* Audited stays: 20,000 (full audit), prediction at 24 hours after ICU admission
* Split: patient-level 70 / 10 / 20; features: availability-filtered, prorated 24-hour windows
* Result: 0 fail, 0 warn, 16 pass, 5 skipped
* Software: icu-leakage-check 0.1.0; Python 3.11.5
* Evidence chain: Part 1 a84f339e2d0379e4..., Part 2 36cffffe5a1ec6d4...

## Checks

| Check | Leakage type | Status | Severity |
|---|---|---|---|
| duplicate_records | DUPLICATE_RECORD | pass | info |
| duplicate_records | DUPLICATE_RECORD | pass | info |
| eligibility_timing | COHORT_FUTURE_INFORMATION | pass | info |
| entity_overlap | ENTITY_OVERLAP | pass | info |
| event_timing | FUTURE_EVENT | pass | info |
| event_timing | AVAILABILITY_LAG | pass | info |
| event_timing | INTERVAL_CROSSING | pass | info |
| event_timing | UNDATED_SOURCE | pass | info |
| hindsight_features | HINDSIGHT_VARIABLE | pass | info |
| missingness_proxy | MISSINGNESS_PROXY | pass | info |
| outcome_proxies | OUTCOME_PROXY | pass | info |
| outcome_timing | POST_OUTCOME_INSTANCE | pass | info |
| single_feature_dominance | SINGLE_FEATURE_DOMINANCE | pass | info |
| synthetic_samples | SYNTHETIC_IN_EVALUATION | pass | info |
| temporal_order | SPLIT_ORDER | pass | info |
| temporal_order | SPLIT_ORDER | pass | info |
| declarations | EVALUATION_REUSE | skipped | info |
| preprocessing_fit | PREPROCESSING_FIT | skipped | info |
| site_overlap | SITE_OVERLAP | skipped | info |
| threshold_selection | EVALUATION_REUSE | skipped | info |
| truncation_invariance | FUTURE_EVENT | skipped | info |

## What the checks protect against (inflation measured when each leak is switched on)

| Item | Mechanism | Median AUROC inflation |
|---|---|---|
| E01 | Repeat stays of test patients in training | -0.001 |
| E02 | Look-ahead: features beyond the prediction time | +0.060 |
| E03 | Clinical-time filter ignoring result availability | +0.002 |
| E04 | Full totals of running infusions and ventilation | +0.002 |
| E05 | Discharge fields and length of stay | +0.133 |
| E06 | Outcome column next to predictors | +0.008 |
| E07 | Same-admission billing codes and diagnoses | +0.020 |
| E08 | Care limitation and code status, any time | +0.003 |
| E09 | First-day severity summaries in an earlier model | +0.037 |
| E10 | Patients already in the outcome kept in the cohort | +0.073 |
| E11 | Preprocessing fitted on all rows | +0.000 |
| E12 | Feature selection on all rows | +0.000 |
| E13 | Resampling before the split | +0.059 |
| E14 | Threshold chosen on the test set | +0.001 |
| E15 | Hyperparameters or final model chosen on the test set | +0.000 |
| E16 | Temporal validation on shifted dates | +0.031 |
| E17 | Overlapping data presented as external | +0.002 |