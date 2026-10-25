# ICU-LEAK checklist (17 items)

Answer each question for the model under review. Items are ordered by evidence-based priority: estimated prevalence
in published MIMIC-IV and eICU-CRD prediction studies multiplied by the AUROC inflation measured when the practice
is switched on in a controlled pipeline.

| Rank | Item | Question | Prevalence | Effect (median, full cohort) | Effect at smallest study size | Verified by |
|---|---|---|---|---|---|---|
| 1 | E10 | Were patients who already had the outcome at the prediction time excluded? | 39% | +0.073 (AUROC) |  | `outcome_timing` |
| 2 | E02 | Is every feature computed only from data recorded up to the prediction time? | 28% | +0.060 (AUROC) |  | `event_timing, truncation_invariance` |
| 3 | E05 | Are length of stay, discharge information and other end-of-stay fields excluded from the predictors? | 12% | +0.133 (AUROC) |  | `hindsight_features` |
| 4 | E13 | Was resampling (for example SMOTE) applied inside the training set only, with the test set left untouched? | 5% | +0.059 (AUROC) | +0.235 | `synthetic_samples` |
| 5 | E07 | Are diagnosis and billing codes taken only from earlier admissions, not the admission being predicted? | 54% | +0.020 (AUROC) |  | `hindsight_features (undated sources)` |
| 6 | E14 | Was the decision threshold chosen without the test set? | 11% | +0.001 (balanced accuracy) | +0.094 | `threshold_selection` |
| 7 | E15 | Were hyperparameters and the final model chosen without the test set? | 35% | +0.000 (AUROC) | +0.028 | `declarations` |
| 8 | E12 | Was feature selection performed on the training set only? | 26% | +0.000 (AUROC) | +0.026 | `declarations` |
| 9 | E09 | Are severity scores used only when their measurement window ends by the prediction time? | 8% | +0.037 (AUROC) |  | `hindsight_features, eligibility_timing` |
| 10 | E11 | Were imputation, scaling and other preprocessing fitted on the training set only? | 26% | +0.000 (AUROC) | +0.009 | `preprocessing_fit` |
| 11 | E03 | Is every feature restricted to results available (stored or verified) by the prediction time, not only charted before it? | 99% | +0.002 (AUROC) |  | `event_timing` |
| 12 | E04 | Are infusions, ventilation and other interval records still running at the prediction time truncated or prorated? | 35% | +0.002 (AUROC) |  | `event_timing` |
| 13 | E01 | Were training and test sets split by patient, so that no patient contributes stays to both? | 9% | -0.001 (AUROC) | +0.007 | `entity_overlap, duplicate_records` |
| 14 | E16 | If validation is called temporal, is it ordered by true calendar time (in MIMIC-IV, anchor_year_group), not shifted dates? | 1% | +0.031 (absolute AUROC misestimation) |  | `temporal_order` |
| 15 | E17 | If validation is called external, does the external set share no hospitals or patients with the development data (MIMIC-III and MIMIC-IV overlap in 2008 to 2012)? | 14% | +0.002 (absolute AUROC misestimation) |  | `site_overlap` |
| 16 | E06 | Are outcome fields and scores computed with the outcome (for example recorded hospital mortality) excluded from the predictors? | 2% | +0.008 (AUROC) |  | `hindsight_features, outcome_proxies` |
| 17 | E08 | Are code status, care limitation and end-of-life documentation restricted to entries made before the prediction time? | 1% | +0.003 (AUROC) |  | `outcome_proxies` |

Prevalence basis: corrected with blinded reference coding by a large language model (174 studies).