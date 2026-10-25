"""Result objects returned by checks and audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd


class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: Union[str, int, "Severity"]) -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, (int, np.integer)):
            return cls(int(value))
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown severity {value!r}; use one of {[s.name.lower() for s in cls]}"
            ) from exc


class Status:
    FAIL = "fail"
    WARN = "warn"
    PASS = "pass"
    SKIPPED = "skipped"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Severity):
        return value.name.lower()
    return value


@dataclass
class Finding:
    """Outcome of one check.

    ``status`` is ``fail`` when leakage is detected, ``warn`` when the evidence is
    suggestive and needs review, ``pass`` when the check ran and found nothing and
    ``skipped`` when required inputs were not supplied.
    """

    check_id: str
    leakage_type: str
    title: str
    status: str
    severity: Severity
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    affected: List[str] = field(default_factory=list)
    recommendation: str = ""

    @property
    def is_problem(self) -> bool:
        return self.status in (Status.FAIL, Status.WARN)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name.lower()
        d["evidence"] = _jsonable(self.evidence)
        d["affected"] = [str(a) for a in self.affected]
        return d


@dataclass
class AuditResult:
    findings: List[Finding]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- queries
    def problems(self, min_severity: Union[str, Severity] = "info") -> List[Finding]:
        threshold = Severity.parse(min_severity)
        return [f for f in self.findings if f.is_problem and f.severity >= threshold]

    def has_problems(self, min_severity: Union[str, Severity] = "high") -> bool:
        return len(self.problems(min_severity)) > 0

    def exit_code(self, fail_on: Union[str, Severity] = "high") -> int:
        return 1 if self.has_problems(fail_on) else 0

    def by_check(self, check_id: str) -> List[Finding]:
        return [f for f in self.findings if f.check_id == check_id]

    def leaky_features(self, min_severity: Union[str, Severity] = "high") -> List[str]:
        """Features named in problem findings at or above ``min_severity``."""
        names: List[str] = []
        for f in self.problems(min_severity):
            if f.evidence.get("affected_kind", "feature") == "feature":
                for a in f.affected:
                    if a not in names:
                        names.append(a)
        return names

    @property
    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in (Status.FAIL, Status.WARN, Status.PASS, Status.SKIPPED)}
        for f in self.findings:
            out[f.status] = out.get(f.status, 0) + 1
        return out

    @property
    def worst_severity(self) -> Optional[Severity]:
        problems = self.problems("info")
        if not problems:
            return None
        return max(f.severity for f in problems)

    # ----------------------------------------------------------------- exports
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for f in self.findings:
            rows.append(
                {
                    "check_id": f.check_id,
                    "leakage_type": f.leakage_type,
                    "title": f.title,
                    "status": f.status,
                    "severity": f.severity.name.lower(),
                    "n_affected": len(f.affected),
                    "affected": ", ".join(str(a) for a in f.affected[:20]),
                    "message": f.message,
                    "recommendation": f.recommendation,
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            order = {Status.FAIL: 0, Status.WARN: 1, Status.PASS: 2, Status.SKIPPED: 3}
            df["_s"] = df["status"].map(order)
            df["_v"] = df["severity"].map(lambda s: -Severity.parse(s))
            df = df.sort_values(["_s", "_v", "check_id"]).drop(columns=["_s", "_v"])
            df = df.reset_index(drop=True)
        return df

    def to_dict(self) -> Dict[str, Any]:
        worst = self.worst_severity
        return {
            "metadata": _jsonable(self.metadata),
            "summary": {
                "counts": self.counts,
                "worst_severity": worst.name.lower() if worst is not None else None,
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_markdown(self, path: Optional[str] = None) -> str:
        from .report import render_markdown

        text = render_markdown(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def to_html(self, path: Optional[str] = None) -> str:
        from .report import render_html

        text = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def summary(self) -> str:
        from .report import render_text

        return render_text(self)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()

    def _repr_html_(self) -> str:  # pragma: no cover - notebook display
        from .report import render_html

        return render_html(self, standalone=False)


def merge_results(results: Iterable[AuditResult], labels: Optional[List[str]] = None) -> pd.DataFrame:
    """Stack several audit results into one long table, for comparing pipelines."""
    frames = []
    for i, r in enumerate(results):
        df = r.to_frame()
        df.insert(0, "pipeline", labels[i] if labels else i)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
