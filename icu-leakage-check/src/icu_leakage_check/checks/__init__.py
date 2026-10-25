"""All built-in checks, in the order they run by default."""

from .base import CHECK_REGISTRY, Check, register
from .content import FeatureDominanceCheck, HindsightFeatureCheck, MissingnessProxyCheck, OutcomeProxyCheck
from .pipeline import (
    DECLARATION_ITEMS,
    DeclarationsCheck,
    PreprocessingFitCheck,
    SyntheticSampleCheck,
    ThresholdSelectionCheck,
)
from .split import DuplicateRecordCheck, EntityOverlapCheck, SiteOverlapCheck, TemporalOrderCheck
from .temporal import EligibilityTimingCheck, EventTimingCheck, OutcomeTimingCheck, TruncationInvarianceCheck

DEFAULT_CHECKS = [
    EntityOverlapCheck,
    SiteOverlapCheck,
    DuplicateRecordCheck,
    TemporalOrderCheck,
    EventTimingCheck,
    TruncationInvarianceCheck,
    OutcomeTimingCheck,
    EligibilityTimingCheck,
    HindsightFeatureCheck,
    OutcomeProxyCheck,
    FeatureDominanceCheck,
    MissingnessProxyCheck,
    PreprocessingFitCheck,
    SyntheticSampleCheck,
    ThresholdSelectionCheck,
    DeclarationsCheck,
]


def list_checks():
    import pandas as pd

    return pd.DataFrame(
        [
            {"check_id": c.check_id, "leakage_type": c.leakage_type, "title": c.title, "description": c.description}
            for c in DEFAULT_CHECKS
        ]
    )


__all__ = [
    "CHECK_REGISTRY",
    "Check",
    "DECLARATION_ITEMS",
    "DEFAULT_CHECKS",
    "DeclarationsCheck",
    "DuplicateRecordCheck",
    "EligibilityTimingCheck",
    "EntityOverlapCheck",
    "EventTimingCheck",
    "FeatureDominanceCheck",
    "HindsightFeatureCheck",
    "MissingnessProxyCheck",
    "OutcomeProxyCheck",
    "OutcomeTimingCheck",
    "PreprocessingFitCheck",
    "SiteOverlapCheck",
    "SyntheticSampleCheck",
    "TemporalOrderCheck",
    "ThresholdSelectionCheck",
    "TruncationInvarianceCheck",
    "list_checks",
    "register",
]
