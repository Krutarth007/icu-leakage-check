"""Base class and registry for checks."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

from ..context import AuditContext
from ..findings import Finding, Severity, Status
from ..taxonomy import get_leakage_type

CHECK_REGISTRY: Dict[str, Type["Check"]] = {}


def register(cls: Type["Check"]) -> Type["Check"]:
    CHECK_REGISTRY[cls.check_id] = cls
    return cls


class Check:
    """A single, independent leakage test.

    Subclasses set ``check_id``, ``leakage_type`` and ``title``, implement
    :meth:`missing_inputs` to say what they need, and :meth:`run` to return
    findings.
    """

    check_id: str = ""
    leakage_type: str = ""
    title: str = ""
    description: str = ""

    def __init__(self, **options):
        self.options = options

    def opt(self, ctx: AuditContext, name: str, default):
        key = f"{self.check_id}.{name}"
        if key in ctx.options:
            return ctx.options[key]
        return self.options.get(name, default)

    # ----------------------------------------------------------------- hooks
    def missing_inputs(self, ctx: AuditContext) -> List[str]:
        return []

    def run(self, ctx: AuditContext) -> List[Finding]:  # pragma: no cover - abstract
        raise NotImplementedError

    # --------------------------------------------------------------- helpers
    def finding(
        self,
        status: str,
        severity: Severity,
        message: str,
        evidence: Optional[dict] = None,
        affected: Sequence = (),
        recommendation: Optional[str] = None,
        leakage_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Finding:
        lt = leakage_type or self.leakage_type
        rec = recommendation if recommendation is not None else get_leakage_type(lt).remedy
        return Finding(
            check_id=self.check_id,
            leakage_type=lt,
            title=title or self.title,
            status=status,
            severity=severity,
            message=message,
            evidence=evidence or {},
            affected=[str(a) for a in affected],
            recommendation=rec if status in (Status.FAIL, Status.WARN) else "",
        )

    def passed(self, message: str, evidence: Optional[dict] = None, leakage_type: Optional[str] = None) -> Finding:
        return self.finding(Status.PASS, Severity.INFO, message, evidence, leakage_type=leakage_type)

    def skipped(self, reasons: List[str]) -> Finding:
        return self.finding(
            Status.SKIPPED,
            Severity.INFO,
            "Not run: " + "; ".join(reasons),
            evidence={"missing_inputs": reasons},
        )

    def execute(self, ctx: AuditContext) -> List[Finding]:
        missing = self.missing_inputs(ctx)
        if missing:
            return [self.skipped(missing)]
        try:
            return self.run(ctx)
        except Exception as exc:  # keep the audit going and report the failure
            return [
                self.finding(
                    Status.SKIPPED,
                    Severity.INFO,
                    f"Check raised {type(exc).__name__}: {exc}",
                    evidence={"error": repr(exc)},
                )
            ]


def need(condition: bool, reason: str, out: List[str]) -> None:
    if not condition:
        out.append(reason)


def top_items(d: dict, n: int = 25) -> dict:
    items = sorted(d.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)
    return dict(items[:n])


def fraction(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def pairs(names: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out.append((a, b))
    return out
