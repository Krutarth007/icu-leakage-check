# Study summary across Parts 1 to 3

## Part 1: cohorts and risk surface

```
{
  "MIMIC-IV": {
    "ICU stays": 57460,
    "patients": 42067,
    "in-hospital mortality": 0.1104,
    "prolonged ICU stay": 0.2872,
    "AKI (eligible stays)": 0.1397,
    "share of stays from patients with repeat stays": 0.4258,
    "random row split, test stays linked to a training patient": 0.3538,
    "values charted in first 24 h but available after it": 297378,
    "share of first-24 h values arriving late": 0.0273,
    "baseline audit failures": 0,
    "baseline audit warnings": 0
  },
  "eICU-CRD": {
    "ICU stays": 129258,
    "patients": 102154,
    "in-hospital mortality": 0.0915,
    "prolonged ICU stay": 0.2559,
    "AKI (eligible stays)": 0.0941,
    "share of stays from patients with repeat stays": 0.3548,
    "random row split, test stays linked to a training patient": 0.2833,
    "values charted in first 24 h but available after it": 84818,
    "share of first-24 h values arriving late": 0.0212,
    "baseline audit failures": 0,
    "baseline audit warnings": 0
  }
}
```

## Part 2: controlled inflation and detection

```
{
  "MIMIC-IV | In-hospital mortality": {
    "clean AUROC by model": {
      "logistic": 0.8676,
      "hgb": 0.8854
    },
    "largest AUROC inflations": [
      "E05 logistic: +0.130 (+0.120 to +0.141), paired",
      "E05 hgb: +0.113 (+0.103 to +0.122), paired",
      "E13b hgb: +0.106 (+0.097 to +0.116), as reported",
      "E07 logistic: +0.071 (+0.062 to +0.081), paired",
      "E07 hgb: +0.061 (+0.053 to +0.069), paired"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 14
  },
  "MIMIC-IV | Prolonged ICU stay (> 4 days)": {
    "clean AUROC by model": {
      "logistic": 0.7721,
      "hgb": 0.7977
    },
    "largest AUROC inflations": [
      "E05 logistic: +0.228 (+0.218 to +0.239), paired",
      "E05 hgb: +0.202 (+0.193 to +0.212), paired",
      "E13b hgb: +0.125 (+0.114 to +0.135), as reported",
      "E02 hgb: +0.076 (+0.068 to +0.084), paired",
      "E02 logistic: +0.068 (+0.060 to +0.076), paired"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 10
  },
  "MIMIC-IV | AKI 24 to 72 h": {
    "clean AUROC by model": {
      "logistic": 0.7585,
      "hgb": 0.7751
    },
    "largest AUROC inflations": [
      "E13b hgb: +0.194 (+0.180 to +0.211), as reported",
      "E02 logistic: +0.158 (+0.145 to +0.172), paired",
      "E02 hgb: +0.154 (+0.142 to +0.166), paired",
      "E10b hgb: +0.070 (+0.054 to +0.089), as reported",
      "E10b logistic: +0.063 (+0.044 to +0.083), as reported"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 8
  },
  "eICU-CRD | In-hospital mortality": {
    "clean AUROC by model": {
      "logistic": 0.8403,
      "hgb": 0.8643
    },
    "largest AUROC inflations": [
      "E05 logistic: +0.160 (+0.151 to +0.168), paired",
      "E06 logistic: +0.159 (+0.150 to +0.167), paired",
      "E05 hgb: +0.136 (+0.128 to +0.143), paired",
      "E06 hgb: +0.135 (+0.127 to +0.142), paired",
      "E13b hgb: +0.135 (+0.126 to +0.141), as reported"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 16
  },
  "eICU-CRD | Prolonged ICU stay (> 4 days)": {
    "clean AUROC by model": {
      "logistic": 0.7363,
      "hgb": 0.7648
    },
    "largest AUROC inflations": [
      "E05 logistic: +0.264 (+0.257 to +0.271), paired",
      "E05 hgb: +0.235 (+0.229 to +0.242), paired",
      "E13b hgb: +0.162 (+0.154 to +0.169), as reported",
      "E07 hgb: +0.054 (+0.050 to +0.059), paired",
      "E02 hgb: +0.051 (+0.047 to +0.056), paired"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 17
  },
  "eICU-CRD | AKI 24 to 72 h": {
    "clean AUROC by model": {
      "logistic": 0.7355,
      "hgb": 0.7591
    },
    "largest AUROC inflations": [
      "E13b hgb: +0.230 (+0.218 to +0.242), as reported",
      "E02 logistic: +0.167 (+0.156 to +0.178), paired",
      "E02 hgb: +0.161 (+0.150 to +0.171), paired",
      "E10b logistic: +0.078 (+0.064 to +0.093), as reported",
      "E10b hgb: +0.075 (+0.061 to +0.088), as reported"
    ],
    "paired comparisons with AUROC inflation (lower 95% bound above zero)": 16
  },
  "detection eICU-CRD full": "16 of 16 mechanisms",
  "detection eICU-CRD table": "7 of 16 mechanisms",
  "detection MIMIC-IV full": "14 of 14 mechanisms",
  "detection MIMIC-IV table": "6 of 14 mechanisms"
}
```

## Part 3: literature, sample size and field-level impact

```
{
  "literature": {
    "query": "(TITLE:\"MIMIC-IV\" OR ABSTRACT:\"MIMIC-IV\" OR TITLE:\"MIMIC IV\" OR ABSTRACT:\"MIMIC IV\" OR TITLE:\"MIMIC-4\" OR ABSTRACT:\"MIMIC-4\" OR TITLE:\"eICU\" OR ABSTRACT:\"eICU\" OR TITLE:\"eICU-CRD\" OR ABSTRACT:\"eICU-CRD\" OR TITLE:\"eICU Collaborative Research Database\" OR ABSTRACT:\"eICU Collaborative Research Database\") AND (TITLE:predict* OR ABSTRACT:predict* OR TITLE:prognos* OR ABSTRACT:prognos* OR TITLE:nomogram* OR ABSTRACT:nomogram* OR TITLE:\"machine learning\" OR ABSTRACT:\"machine learning\" OR TITLE:\"deep learning\" OR ABSTRACT:\"deep learning\" OR TITLE:\"risk score\" OR ABSTRACT:\"risk score\" OR TITLE:\"early warning\" OR ABSTRACT:\"early warning\" OR TITLE:\"risk model\" OR ABSTRACT:\"risk model\") AND FIRST_PDATE:[2020-08-01 TO 2026-09-30] AND (SRC:MED OR SRC:PMC OR SRC:PPR)",
    "search window": "2020-08-01 to 2026-09-30",
    "retrieved": "2026-10-16T04:52:35Z",
    "records reported by Europe PMC": 3957,
    "unique records": 3859,
    "included after screening": 1615,
    "included with full text": 996,
    "full-text multivariable studies analysed": 793,
    "validation": "174 coded papers"
  },
  "prevalence (estimated share at risk)": {
    "I01 Split allows the same patient in training and test": 0.0947,
    "I02 Features summarised over the whole stay": 0.2792,
    "I03 Result availability time not handled": 0.9928,
    "I04 Running infusions or ventilation not truncated at the prediction time": 0.3507,
    "I05 Length of stay or discharge information used as a predictor": 0.1187,
    "I06 Outcome or outcome-derived fields used as predictors (eICU-CRD)": 0.0163,
    "I07 Comorbidities from same-admission diagnosis codes": 0.5406,
    "I08 Code status or care limitation used as a predictor": 0.0054,
    "I09 First-day severity scores in an earlier prediction": 0.082,
    "I10 Patients with the outcome at the prediction time not excluded": 0.3912,
    "I11 Imputation or scaling fitted on all data": 0.2647,
    "I12 Feature selection on all data": 0.2644,
    "I13 Resampling before the split": 0.0491,
    "I14 Threshold chosen on the test set": 0.1129,
    "I15 Hyperparameters or final model chosen on the test set": 0.3537,
    "I16 Temporal validation on shifted MIMIC-IV dates": 0.0127,
    "I17 Overlapping data presented as external validation": 0.1409
  },
  "transparency (share of studies)": {
    "code_shared": 0.1639,
    "code_on_request": 0.0164,
    "calibration_reported": 0.7062,
    "uncertainty_reported": 0.8298,
    "reporting_guideline_cited": 0.1892,
    "decision_curve_reported": 0.6103,
    "leakage_discussed": 0.198,
    "external_validation_other_source": 0.3291,
    "prediction_time_stated": 0.8108
  },
  "reported AUROC and leakage": {
    "studies with AUROC and sample size": 357,
    "median reported AUROC, any content leak": 0.821,
    "median reported AUROC, no content leak": 0.827,
    "difference in medians (95% CI)": [
      -0.006000000000000005,
      -0.02200000000000002,
      0.008506250000000057
    ],
    "logit AUROC coefficient, content leak": [
      -0.0691,
      -0.2205,
      0.087
    ],
    "logit AUROC coefficient, procedural leak": [
      0.0538,
      -0.0841,
      0.2033
    ],
    "studies in citation model": 793,
    "log citations per year coefficient, content leak": [
      -0.046,
      -0.1369,
      0.0456
    ]
  },
  "sample-size dependence (mean AUROC inflation, MIMIC-IV, logistic)": {
    "E01b": {
      "500": 0.0198,
      "1000": -0.0175,
      "2000": 0.0065,
      "5000": 0.0009,
      "10000": -0.0017
    },
    "E11": {
      "500": 0.004,
      "1000": 0.0037,
      "2000": -0.0001,
      "5000": 0.0006,
      "10000": 0.0001
    },
    "E12": {
      "500": 0.0341,
      "1000": 0.0061,
      "2000": 0.0055,
      "5000": 0.0033,
      "10000": 0.0008
    },
    "E13": {
      "500": 0.207,
      "1000": 0.1317,
      "2000": 0.0814,
      "5000": 0.038,
      "10000": 0.0176
    },
    "E13b": {
      "500": 0.2376,
      "1000": 0.1364,
      "2000": 0.0909,
      "5000": 0.0513,
      "10000": 0.0244
    },
    "E14": {
      "500": 0.0795,
      "1000": 0.0702,
      "2000": 0.0313,
      "5000": 0.0177,
      "10000": 0.0171
    },
    "E15": {
      "500": 0.0411,
      "1000": 0.0316,
      "2000": 0.0117,
      "5000": 0.0084,
      "10000": 0.0026
    }
  },
  "additivity factor": [
    0.7772,
    0.0097
  ],
  "field-level optimism": {
    "mean_optimism": [
      0.062,
      0.0557,
      0.0686
    ],
    "median_optimism": [
      0.0519,
      0.0451,
      0.0585
    ],
    "share_over_0.01": [
      0.7934,
      0.7574,
      0.8326
    ],
    "share_over_0.05": [
      0.5243,
      0.4319,
      0.579
    ],
    "share_of_0.90_claims_remaining": [
      0.4211,
      0.3263,
      0.5053
    ],
    "mean_adjusted_auroc": [
      0.761,
      0.7544,
      0.7673
    ],
    "mean reported AUROC": [
      0.823,
      null,
      null
    ],
    "studies in the model": [
      639.0,
      null,
      null
    ],
    "studies reporting AUROC >= 0.90": [
      95.0,
      null,
      null
    ]
  },
  "field-level optimism, explicit practices only": {
    "mean_optimism": [
      0.0516,
      0.0461,
      0.0572
    ],
    "median_optimism": [
      0.0334,
      0.0272,
      0.0454
    ],
    "share_over_0.01": [
      0.734,
      0.6948,
      0.7715
    ],
    "share_over_0.05": [
      0.3991,
      0.3365,
      0.482
    ],
    "share_of_0.90_claims_remaining": [
      0.5053,
      0.4105,
      0.6
    ],
    "mean_adjusted_auroc": [
      0.7715,
      0.7658,
      0.777
    ]
  },
  "audit runtime (seconds)": {
    "MIMIC-IV 1,000 stays full": 0.5,
    "MIMIC-IV 5,000 stays full": 1.3,
    "MIMIC-IV 20,000 stays full": 2.6,
    "MIMIC-IV 57,460 stays table only": 3.5,
    "eICU-CRD 999 stays full": 0.5,
    "eICU-CRD 5,000 stays full": 0.7,
    "eICU-CRD 20,000 stays full": 1.2,
    "eICU-CRD 129,258 stays table only": 4.8
  },
  "checklist top five": [
    "E10 Patients already in the outcome kept in the cohort",
    "E02 Look-ahead: features beyond the prediction time",
    "E05 Discharge fields and length of stay",
    "E13 Resampling before the split",
    "E07 Same-admission billing codes and diagnoses"
  ],
  "field-level optimism, main scenario (Table 3)": {
    "Scenario": "Main: explicit and unclear practices (unclear as 0.5), additivity factor 0.777",
    "Mean optimism (95% interval)": "0.051 (0.042 to 0.061)",
    "Median optimism": "0.033 (0.026 to 0.050)",
    "Studies > 0.05, %": "40.2 (30.7 to 50.1)",
    "AUROC \u2265 0.90 claims remaining, %": "50.5 (38.9 to 61.1)",
    "Mean adjusted AUROC": "0.772 (0.762 to 0.781)"
  },
  "reference coding": "Claude Opus 5.5, 100 random studies blinded to automated values, 74 enriched unblinded",
  "validation agreement, pooled": {
    "indicator": "pooled",
    "name": "all 17 indicators",
    "n": 1419,
    "tp": 178,
    "fn": 166,
    "tn": 962,
    "fp": 113,
    "sensitivity": 0.5174418604651163,
    "sens_lower": 0.4647353835325711,
    "sens_upper": 0.5697630778360518,
    "specificity": 0.8948837209302326,
    "spec_lower": 0.8751217878785504,
    "spec_upper": 0.9118334055437278,
    "kappa": 0.43511871893981224
  }
}
```