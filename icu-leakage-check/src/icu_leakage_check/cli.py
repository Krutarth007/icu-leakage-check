"""Command line interface: ``icu-leakage-check <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import pandas as pd


def _csv_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _print_frame(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 200, "display.max_colwidth", 80):
        print(df.to_string(index=False))


def _cmd_audit(args) -> int:
    from .audit import LeakageAudit, _read_any

    kwargs = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        args.html = args.html or cfg.get("html")
        args.json = args.json or cfg.get("json")
        args.markdown = args.markdown or cfg.get("markdown")
        args.fail_on = args.fail_on or cfg.get("fail_on")
    direct = {
        "label": args.label, "split": args.split, "dataset": args.dataset, "patient_id": args.patient_id,
        "encounter_id": args.encounter_id, "stay_id": args.stay_id, "site_id": args.site_id,
        "index_time": args.index_time, "outcome_time": args.outcome_time,
        "eligibility_time": args.eligibility_time, "calendar_time": args.calendar_time,
        "split_strategy": args.split_strategy, "scores": args.scores, "threshold": args.threshold,
    }
    kwargs.update({k: v for k, v in direct.items() if v is not None})
    if args.features:
        kwargs["feature_columns"] = _csv_list(args.features)
    if args.exclude:
        kwargs["exclude_columns"] = _csv_list(args.exclude)
    if args.sources:
        with open(args.sources, "r", encoding="utf-8") as fh:
            kwargs["feature_sources"] = json.load(fh)
    if args.events:
        kwargs["events"] = _read_any(args.events)

    if args.config:
        data = _read_any(args.data) if args.data else None
        audit = LeakageAudit.from_config(args.config, data=data, **kwargs)
    else:
        if not args.data:
            print("error: --data or --config is required", file=sys.stderr)
            return 2
        audit = LeakageAudit(_read_any(args.data), **kwargs)

    result = audit.run(skip=_csv_list(args.skip) or ())
    print(result.summary())
    if args.html:
        result.to_html(args.html)
        print(f"\nHTML report: {args.html}")
    if args.json:
        result.to_json(args.json)
        print(f"JSON report: {args.json}")
    if args.markdown:
        result.to_markdown(args.markdown)
        print(f"Markdown report: {args.markdown}")
    fail_on = (args.fail_on or "high").lower()
    if fail_on == "never":
        return 0
    return result.exit_code(fail_on)


def _cmd_locate(args) -> int:
    from .io import list_tables

    df = list_tables(args.root, args.dataset)
    _print_frame(df)
    missing = df.loc[~df["found"], "table"].tolist()
    print(f"\n{int(df['found'].sum())} of {len(df)} tables found.")
    if missing:
        print("Missing: " + ", ".join(missing))
    return 0


def _cmd_convert(args) -> int:
    from .io import convert_to_parquet

    df = convert_to_parquet(
        args.root, args.dataset, args.out, tables=_csv_list(args.tables), overwrite=args.overwrite,
        memory_limit=args.memory_limit, temp_directory=args.temp_dir, threads=args.threads,
    )
    _print_frame(df)
    return 0


def _cmd_schema(args) -> int:
    from .schemas import get_schema

    schema = get_schema(args.dataset)
    if args.table:
        t = schema.table(args.table)
        info = {k: v for k, v in t.__dict__.items() if v not in (None, (), "", False)}
        print(json.dumps(info, indent=2, default=list))
        return 0
    print(schema.notes + "\n")
    _print_frame(schema.summary().drop(columns=["notes"]))
    return 0


def _cmd_scan(args) -> int:
    from .audit import LeakageAudit, _read_any

    if args.data:
        df = _read_any(args.data)
        if args.nrows:
            df = df.head(args.nrows)
    elif args.columns:
        df = pd.DataFrame(columns=_csv_list(args.columns))
    else:
        print("error: --data or --columns is required", file=sys.stderr)
        return 2
    kwargs = {}
    if args.sources:
        with open(args.sources, "r", encoding="utf-8") as fh:
            kwargs["feature_sources"] = json.load(fh)
    result = LeakageAudit(df, dataset=args.dataset, exclude_columns=_csv_list(args.exclude) or (), **kwargs).run(
        checks=["hindsight_features", "outcome_proxies"]
    )
    print(result.summary())
    return result.exit_code(args.fail_on or "high") if (args.fail_on or "high") != "never" else 0


def _cmd_taxonomy(args) -> int:
    from .taxonomy import taxonomy_table

    df = taxonomy_table()
    _print_frame(df[["code", "family", "name"]] if not args.full else df)
    return 0


def _cmd_demo(args) -> int:
    from .demo import run_demo

    out = run_demo(n_patients=args.patients, seed=args.seed, html_path=args.html)
    print("LEAKY PIPELINE\n")
    print(out["leaky"].summary())
    print("\n\nCLEAN PIPELINE\n")
    print(out["clean"].summary())
    print("\n\nTest-set performance as each pipeline would report it\n")
    print(out["reported_performance"].round(3).to_string())
    print("\n\nInflation on identical test rows (leaky minus clean, patient-level bootstrap)\n")
    cols = ["metric", "clean", "leaky", "difference", "diff_lower", "diff_upper"]
    print(out["impact"][cols].round(3).to_string(index=False))
    if args.html:
        print(f"\nHTML report of the leaky pipeline: {args.html}")
    return 0


def _cmd_checks(args) -> int:
    from .checks import list_checks

    _print_frame(list_checks()[["check_id", "leakage_type", "title"]])
    return 0


def build_parser() -> argparse.ArgumentParser:
    from ._version import __version__

    p = argparse.ArgumentParser(
        prog="icu-leakage-check",
        description="Detect and explain data leakage in clinical prediction datasets.",
    )
    p.add_argument("--version", action="version", version=f"icu-leakage-check {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="Audit a prediction dataset (CSV or Parquet) and write reports.")
    a.add_argument("--config", help="JSON file with audit settings (keys mirror the Python arguments).")
    a.add_argument("--data", help="Prediction dataset, one row per prediction instance.")
    a.add_argument("--label")
    a.add_argument("--split", help="Column assigning rows to train/valid/test.")
    a.add_argument("--dataset", help="mimic-iv, eicu, or omit for generic data.")
    a.add_argument("--patient-id")
    a.add_argument("--encounter-id")
    a.add_argument("--stay-id")
    a.add_argument("--site-id")
    a.add_argument("--index-time")
    a.add_argument("--outcome-time")
    a.add_argument("--eligibility-time")
    a.add_argument("--calendar-time")
    a.add_argument("--split-strategy", choices=["random", "patient", "temporal", "site", "cross_validation"])
    a.add_argument("--scores", help="Column with model scores.")
    a.add_argument("--threshold", type=float)
    a.add_argument("--features", help="Comma-separated feature columns (default: all non-identifier columns).")
    a.add_argument("--exclude", help="Comma-separated columns to exclude from the features.")
    a.add_argument("--sources", help="JSON file mapping feature -> 'table.column'.")
    a.add_argument("--events", help="Provenance event file (CSV or Parquet).")
    a.add_argument("--skip", help="Comma-separated check ids to skip.")
    a.add_argument("--html")
    a.add_argument("--json")
    a.add_argument("--markdown")
    a.add_argument("--fail-on", help="info, low, medium, high (default), critical or never.")
    a.set_defaults(func=_cmd_audit)

    loc = sub.add_parser("locate", help="Show which tables of a dataset are found under a folder.")
    loc.add_argument("--dataset", required=True)
    loc.add_argument("--root", required=True)
    loc.set_defaults(func=_cmd_locate)

    conv = sub.add_parser("convert", help="Convert dataset CSV files to Parquet (needs duckdb).")
    conv.add_argument("--dataset", required=True)
    conv.add_argument("--root", required=True)
    conv.add_argument("--out", required=True)
    conv.add_argument("--tables", help="Comma-separated subset of tables.")
    conv.add_argument("--overwrite", action="store_true")
    conv.add_argument("--memory-limit", default="4GB")
    conv.add_argument("--temp-dir")
    conv.add_argument("--threads", type=int)
    conv.set_defaults(func=_cmd_convert)

    sch = sub.add_parser("schema", help="Show the timing registry for a dataset.")
    sch.add_argument("--dataset", required=True)
    sch.add_argument("--table")
    sch.set_defaults(func=_cmd_schema)

    sc = sub.add_parser("scan", help="Screen feature names for hindsight, outcome and proxy variables.")
    sc.add_argument("--dataset")
    sc.add_argument("--data")
    sc.add_argument("--columns")
    sc.add_argument("--sources")
    sc.add_argument("--exclude")
    sc.add_argument("--nrows", type=int, default=5000)
    sc.add_argument("--fail-on")
    sc.set_defaults(func=_cmd_scan)

    t = sub.add_parser("taxonomy", help="List the leakage types the package recognises.")
    t.add_argument("--full", action="store_true")
    t.set_defaults(func=_cmd_taxonomy)

    c = sub.add_parser("checks", help="List the available checks.")
    c.set_defaults(func=_cmd_checks)

    d = sub.add_parser("demo", help="Run a leaky and a clean pipeline on synthetic ICU data and audit both.")
    d.add_argument("--patients", type=int, default=1500)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--html", help="Write the HTML report of the leaky pipeline here.")
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
