"""Text, Markdown and HTML renderings of an audit result."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, List

from .findings import Severity, Status, _jsonable
from .taxonomy import LEAKAGE_TYPES

if TYPE_CHECKING:  # pragma: no cover
    from .findings import AuditResult, Finding

_ORDER = {Status.FAIL: 0, Status.WARN: 1, Status.PASS: 2, Status.SKIPPED: 3}
_ICON = {Status.FAIL: "FAIL", Status.WARN: "WARN", Status.PASS: "PASS", Status.SKIPPED: "SKIP"}


def _sorted(findings: List["Finding"]) -> List["Finding"]:
    return sorted(findings, key=lambda f: (_ORDER.get(f.status, 9), -int(f.severity), f.check_id))


def _verdict(result: "AuditResult") -> str:
    worst = result.worst_severity
    fails = [f for f in result.findings if f.status == Status.FAIL]
    if fails and worst is not None and worst >= Severity.HIGH:
        return "Leakage detected"
    if fails or result.problems("info"):
        return "Review needed"
    return "No leakage detected by the checks that ran"


# ----------------------------------------------------------------------- text
def render_text(result: "AuditResult", max_affected: int = 8) -> str:
    m = result.metadata
    c = result.counts
    lines = [
        f"icu-leakage-check {m.get('version', '')}  |  dataset: {m.get('dataset') or 'generic'}  |  "
        f"rows: {m.get('rows')}  features: {m.get('features')}",
        f"Verdict: {_verdict(result)}",
        f"Checks: {c.get('fail', 0)} fail, {c.get('warn', 0)} warn, {c.get('pass', 0)} pass, {c.get('skipped', 0)} skipped",
        "",
    ]
    for f in _sorted(result.findings):
        if f.status == Status.SKIPPED:
            continue
        sev = f.severity.name if f.status != Status.PASS else ""
        lines.append(f"[{_ICON[f.status]}] {sev:<8} {f.check_id}: {f.message}")
        if f.affected and f.status in (Status.FAIL, Status.WARN):
            shown = ", ".join(f.affected[:max_affected])
            more = f" (+{len(f.affected) - max_affected} more)" if len(f.affected) > max_affected else ""
            lines.append(f"           affected: {shown}{more}")
        if f.recommendation and f.status in (Status.FAIL, Status.WARN):
            lines.append(f"           fix: {f.recommendation}")
    skipped = [f for f in result.findings if f.status == Status.SKIPPED]
    if skipped:
        lines.append("")
        lines.append("Not verified:")
        for f in skipped:
            lines.append(f"  - {f.check_id}: {f.message}")
    return "\n".join(lines)


# ------------------------------------------------------------------- markdown
def _md_escape(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(result: "AuditResult") -> str:
    m = result.metadata
    c = result.counts
    out = [
        "# Leakage audit report",
        "",
        f"**Verdict:** {_verdict(result)}",
        "",
        f"- Package: icu-leakage-check {m.get('version', '')}",
        f"- Created (UTC): {m.get('created_utc', '')}",
        f"- Dataset: {m.get('dataset') or 'generic'}",
        f"- Rows: {m.get('rows')}; features: {m.get('features')}; label: {m.get('label')}",
        f"- Split strategy: {m.get('split_strategy') or 'not stated'}",
        f"- Checks: {c.get('fail', 0)} fail, {c.get('warn', 0)} warn, {c.get('pass', 0)} pass, {c.get('skipped', 0)} skipped",
        "",
    ]
    if m.get("splits"):
        out += ["| Split | Rows | Prevalence |", "|---|---|---|"]
        for s, e in m["splits"].items():
            out.append(f"| {s} | {e.get('rows')} | {e.get('prevalence', '')} |")
        out.append("")
    out += ["## Findings", "", "| Status | Severity | Check | Leakage type | Finding | Affected |", "|---|---|---|---|---|---|"]
    for f in _sorted(result.findings):
        if f.status == Status.SKIPPED:
            continue
        aff = ", ".join(f.affected[:6]) + (" ..." if len(f.affected) > 6 else "")
        out.append(
            f"| {_ICON[f.status]} | {f.severity.name.lower() if f.status != Status.PASS else ''} | {f.check_id} | "
            f"{f.leakage_type} | {_md_escape(f.message)} | {_md_escape(aff)} |"
        )
    recs = [f for f in _sorted(result.findings) if f.status in (Status.FAIL, Status.WARN) and f.recommendation]
    if recs:
        out += ["", "## Recommended fixes", ""]
        for f in recs:
            out.append(f"- **{f.check_id} ({f.leakage_type})**: {f.recommendation}")
    skipped = [f for f in result.findings if f.status == Status.SKIPPED]
    if skipped:
        out += ["", "## Not verified", ""]
        for f in skipped:
            out.append(f"- **{f.check_id}**: {_md_escape(f.message)}")
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------- html
_CSS = """
:root{--bg:#ffffff;--fg:#1d2433;--muted:#5b6475;--line:#e3e6ec;--card:#f7f8fa;
--fail:#b42318;--failbg:#fde8e7;--warn:#8a5a00;--warnbg:#fff3d6;--pass:#1a7f37;--passbg:#e6f4ea;--skip:#5b6475;--skipbg:#eef0f3}
@media (prefers-color-scheme: dark){:root{--bg:#12151b;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2a303b;--card:#1a1f27;
--fail:#ff8a80;--failbg:#3a1714;--warn:#ffd27a;--warnbg:#3a2c0d;--pass:#7ee2a0;--passbg:#12301d;--skip:#a9b1bf;--skipbg:#232934}}
.ilc{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);line-height:1.45;max-width:1100px;margin:0 auto;padding:16px}
.ilc h1{font-size:1.5rem;margin:0 0 4px}.ilc h2{font-size:1.15rem;margin:28px 0 8px}
.ilc .meta{color:var(--muted);font-size:.9rem}
.ilc .verdict{font-size:1.1rem;font-weight:600;margin:14px 0;padding:10px 14px;border-radius:8px;background:var(--card);border:1px solid var(--line)}
.ilc .tiles{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.ilc .tile{flex:1 1 120px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.ilc .tile b{display:block;font-size:1.4rem}
.ilc table{border-collapse:collapse;width:100%;font-size:.9rem}
.ilc th,.ilc td{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left;vertical-align:top}
.ilc th{color:var(--muted);font-weight:600}
.ilc .chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;font-weight:600;white-space:nowrap}
.ilc .fail{color:var(--fail);background:var(--failbg)}.ilc .warn{color:var(--warn);background:var(--warnbg)}
.ilc .pass{color:var(--pass);background:var(--passbg)}.ilc .skipped{color:var(--skip);background:var(--skipbg)}
.ilc details{margin-top:6px}.ilc summary{cursor:pointer;color:var(--muted)}
.ilc pre{white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;font-size:.8rem;max-height:320px;overflow:auto}
.ilc .fix{color:var(--muted);font-size:.85rem;margin-top:4px}
.ilc .wrap{overflow-x:auto}
"""


def _e(s) -> str:
    return html.escape(str(s))


def render_html(result: "AuditResult", standalone: bool = True) -> str:
    m = result.metadata
    c = result.counts
    rows = []
    for f in _sorted(result.findings):
        if f.status == Status.SKIPPED:
            continue
        lt = LEAKAGE_TYPES.get(f.leakage_type)
        lt_name = lt.name if lt else f.leakage_type
        sev = f.severity.name.lower() if f.status != Status.PASS else ""
        aff = ""
        if f.affected:
            aff = _e(", ".join(f.affected[:12])) + (f" <span class='meta'>(+{len(f.affected) - 12} more)</span>" if len(f.affected) > 12 else "")
        ev = json.dumps(_jsonable(f.evidence), indent=2, default=str)
        fix = f"<div class='fix'>Fix: {_e(f.recommendation)}</div>" if f.recommendation else ""
        rows.append(
            f"<tr><td><span class='chip {f.status}'>{_ICON[f.status]}</span></td>"
            f"<td>{_e(sev)}</td><td><b>{_e(f.title)}</b><div class='meta'>{_e(f.check_id)} · {_e(lt_name)}</div></td>"
            f"<td>{_e(f.message)}{fix}<details><summary>Evidence</summary><pre>{_e(ev)}</pre></details></td>"
            f"<td>{aff}</td></tr>"
        )
    skipped = [f for f in result.findings if f.status == Status.SKIPPED]
    skip_rows = "".join(f"<tr><td>{_e(f.check_id)}</td><td>{_e(f.message)}</td></tr>" for f in skipped)
    split_rows = "".join(
        f"<tr><td>{_e(s)}</td><td>{_e(e.get('rows'))}</td><td>{_e(e.get('prevalence', ''))}</td></tr>"
        for s, e in (m.get("splits") or {}).items()
    )
    body = f"""
<div class="ilc">
<h1>Leakage audit report</h1>
<div class="meta">icu-leakage-check {_e(m.get('version', ''))} · {_e(m.get('created_utc', ''))} · dataset: {_e(m.get('dataset') or 'generic')}
 · rows: {_e(m.get('rows'))} · features: {_e(m.get('features'))} · label: {_e(m.get('label'))} · split strategy: {_e(m.get('split_strategy') or 'not stated')}</div>
<div class="verdict">{_e(_verdict(result))}</div>
<div class="tiles">
 <div class="tile"><span class="chip fail">FAIL</span><b>{c.get('fail', 0)}</b></div>
 <div class="tile"><span class="chip warn">WARN</span><b>{c.get('warn', 0)}</b></div>
 <div class="tile"><span class="chip pass">PASS</span><b>{c.get('pass', 0)}</b></div>
 <div class="tile"><span class="chip skipped">SKIP</span><b>{c.get('skipped', 0)}</b></div>
</div>
{"<h2>Splits</h2><div class='wrap'><table><tr><th>Split</th><th>Rows</th><th>Prevalence</th></tr>" + split_rows + "</table></div>" if split_rows else ""}
<h2>Findings</h2>
<div class="wrap"><table>
<tr><th>Status</th><th>Severity</th><th>Check</th><th>Finding</th><th>Affected</th></tr>
{''.join(rows)}
</table></div>
{"<h2>Not verified</h2><div class='wrap'><table><tr><th>Check</th><th>Reason</th></tr>" + skip_rows + "</table></div>" if skip_rows else ""}
</div>"""
    if not standalone:
        return f"<style>{_CSS}</style>{body}"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Leakage audit report</title><style>body{{margin:0;background:var(--bg)}}{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
