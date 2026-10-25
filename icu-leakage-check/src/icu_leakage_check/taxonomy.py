"""Catalogue of leakage mechanisms recognised by the package.

Each mechanism has a stable code that appears in every finding, so results
from different pipelines, sites or releases can be compared and aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LeakageType:
    code: str
    name: str
    family: str
    description: str
    typical_example: str
    remedy: str


_TYPES: List[LeakageType] = [
    LeakageType(
        code="ENTITY_OVERLAP",
        name="Entity overlap across splits",
        family="Split integrity",
        description=(
            "The same patient, encounter or unit stay contributes records to more than one "
            "split, so the model is evaluated on people it has already seen."
        ),
        typical_example="Random row-level split of ICU stays when patients have repeat admissions.",
        remedy="Split at the patient level (or higher) and keep all of a patient's records in one split.",
    ),
    LeakageType(
        code="DUPLICATE_RECORD",
        name="Duplicate records across splits",
        family="Split integrity",
        description=(
            "Identical feature vectors appear in both training and evaluation data, which "
            "rewards memorisation rather than generalisation."
        ),
        typical_example="Random oversampling or record duplication performed before splitting.",
        remedy="De-duplicate before splitting and resample only inside the training split.",
    ),
    LeakageType(
        code="SITE_OVERLAP",
        name="Site overlap in a claimed external validation",
        family="Split integrity",
        description=(
            "Hospitals or units used for training also appear in data presented as external."
        ),
        typical_example="Hospital-held-out evaluation in a multi-centre database that reuses hospitals.",
        remedy="Hold out complete sites and verify the site identifiers are disjoint.",
    ),
    LeakageType(
        code="SPLIT_ORDER",
        name="Temporal order violation",
        family="Split integrity",
        description=(
            "A split presented as temporal validation trains on data that is later than, or "
            "concurrent with, the evaluation period."
        ),
        typical_example="Temporal validation in a database whose dates are shifted per patient.",
        remedy="Order by a calendar variable that is valid across patients and keep periods disjoint.",
    ),
    LeakageType(
        code="FUTURE_EVENT",
        name="Information recorded after the prediction time",
        family="Temporal",
        description=(
            "Features are computed from measurements, orders or notes whose clinical time is "
            "later than the moment the prediction is supposed to be made."
        ),
        typical_example="Using the maximum creatinine over the whole ICU stay in a 24-hour model.",
        remedy="Restrict every feature to events at or before the prediction time.",
    ),
    LeakageType(
        code="AVAILABILITY_LAG",
        name="Charted before but not available at prediction time",
        family="Temporal",
        description=(
            "An event is time-stamped before the prediction time but was stored, verified or "
            "resulted afterwards, so it could not have been used in real time."
        ),
        typical_example="A blood culture drawn at 10:00 whose organism result was stored two days later.",
        remedy="Filter on the storage or entry time rather than the clinical time.",
    ),
    LeakageType(
        code="INTERVAL_CROSSING",
        name="Interval events crossing the prediction time",
        family="Temporal",
        description=(
            "Events with a start and end time (infusions, ventilation, medication orders) start "
            "before the prediction time but their totals include volume or duration after it."
        ),
        typical_example="Total vasopressor amount for an infusion still running at the prediction time.",
        remedy="Truncate interval events at the prediction time or prorate their amounts.",
    ),
    LeakageType(
        code="UNDATED_SOURCE",
        name="Undated or day-resolution sources",
        family="Temporal",
        description=(
            "Features come from sources without a usable time stamp, which are typically "
            "finalised at discharge or only known at day resolution."
        ),
        typical_example="Billing diagnosis codes from the current admission used in an admission-time model.",
        remedy="Use codes from previous encounters only, or assign a conservative availability time.",
    ),
    LeakageType(
        code="HINDSIGHT_VARIABLE",
        name="Hindsight variable",
        family="Feature content",
        description=(
            "A feature is only known at or after discharge, or summarises the whole stay, such "
            "as length of stay, discharge location or discharge weight."
        ),
        typical_example="Length of ICU stay used as a predictor of in-hospital mortality.",
        remedy="Remove the variable or replace it with a version truncated at the prediction time.",
    ),
    LeakageType(
        code="OUTCOME_ENCODING",
        name="Outcome encoded in a feature",
        family="Feature content",
        description=(
            "A feature directly records the outcome or a deterministic function of it."
        ),
        typical_example="A 'died in hospital' flag or death time stored in a severity table.",
        remedy="Remove every column derived from the outcome.",
    ),
    LeakageType(
        code="OUTCOME_PROXY",
        name="Outcome proxy",
        family="Feature content",
        description=(
            "A feature records a decision or event that is closely tied to the outcome and "
            "usually happens close to it, such as comfort-care orders, palliative care "
            "encounters, do-not-resuscitate codes or brain death codes."
        ),
        typical_example="A palliative care encounter code used to predict death in the same admission.",
        remedy="Exclude proxies or restrict them to documentation before the prediction time and report it.",
    ),
    LeakageType(
        code="FIRST_DAY_SUMMARY",
        name="First-day summary used before the first day ends",
        family="Feature content",
        description=(
            "Severity scores and worst values summarising the first 24 hours are used by a "
            "model that predicts earlier than 24 hours after admission."
        ),
        typical_example="APACHE worst-of-day-one variables in a model scored at ICU admission plus 6 hours.",
        remedy="Recompute the summary over the window that ends at the prediction time.",
    ),
    LeakageType(
        code="SINGLE_FEATURE_DOMINANCE",
        name="Implausibly strong single feature",
        family="Feature content",
        description=(
            "One feature on its own separates outcomes almost perfectly, which in clinical "
            "prediction is far more often a leak than a discovery."
        ),
        typical_example="A treatment code that is only ever recorded for patients who died.",
        remedy="Trace the feature to its source table and time stamp before trusting it.",
    ),
    LeakageType(
        code="MISSINGNESS_PROXY",
        name="Missingness pattern encodes the outcome",
        family="Feature content",
        description=(
            "Whether a value is missing predicts the outcome strongly, often because the value "
            "is only recorded after the outcome or only for patients who survive long enough."
        ),
        typical_example="A discharge measurement that is missing for every patient who died.",
        remedy="Check when the variable is recorded relative to the outcome and the prediction time.",
    ),
    LeakageType(
        code="POST_OUTCOME_INSTANCE",
        name="Outcome already present at prediction time",
        family="Cohort and label",
        description=(
            "Positive instances where the outcome occurred at or before the prediction time, "
            "so the model detects rather than predicts it."
        ),
        typical_example="Patients already in acute kidney injury at the prediction time kept as positives.",
        remedy="Exclude instances whose outcome precedes the prediction time, or add a prediction gap.",
    ),
    LeakageType(
        code="COHORT_FUTURE_INFORMATION",
        name="Eligibility decided with future information",
        family="Cohort and label",
        description=(
            "Inclusion or exclusion criteria use information that only becomes available after "
            "the prediction time, which selects patients by their future."
        ),
        typical_example="Requiring an ICU stay of at least 48 hours for a model scored at 24 hours.",
        remedy="Define eligibility only from information available at the prediction time.",
    ),
    LeakageType(
        code="PREPROCESSING_FIT",
        name="Preprocessing fitted on evaluation data",
        family="Pipeline",
        description=(
            "Scalers, imputers or encoders learned their parameters from data that includes the "
            "evaluation split."
        ),
        typical_example="StandardScaler fitted on the full dataset before train_test_split.",
        remedy="Fit every learned transformation on the training split only, ideally inside a Pipeline.",
    ),
    LeakageType(
        code="SYNTHETIC_IN_EVALUATION",
        name="Synthetic or resampled records in evaluation data",
        family="Pipeline",
        description=(
            "Oversampling such as SMOTE was applied before splitting, so the evaluation data "
            "contains interpolated records built from training patients."
        ),
        typical_example="SMOTE on the full cohort followed by a random 70/30 split.",
        remedy="Resample only inside the training split or inside each cross-validation fold.",
    ),
    LeakageType(
        code="EVALUATION_REUSE",
        name="Decisions tuned on evaluation data",
        family="Pipeline",
        description=(
            "Thresholds, hyperparameters or model choices were selected using the evaluation "
            "split, which then no longer gives an unbiased estimate."
        ),
        typical_example="Choosing the alert threshold that maximises Youden's J on the test set.",
        remedy="Tune on a validation split or inside cross-validation and touch the test split once.",
    ),
]

LEAKAGE_TYPES: Dict[str, LeakageType] = {t.code: t for t in _TYPES}


def get_leakage_type(code: str) -> LeakageType:
    try:
        return LEAKAGE_TYPES[code]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown leakage type code: {code!r}") from exc


def taxonomy_table():
    """Return the catalogue as a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "code": t.code,
                "name": t.name,
                "family": t.family,
                "description": t.description,
                "typical_example": t.typical_example,
                "remedy": t.remedy,
            }
            for t in _TYPES
        ]
    )
