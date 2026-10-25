import json

import numpy as np
import pandas as pd

from icu_leakage_check import LeakageAudit, audit, audit_split, merge_results
from icu_leakage_check.cli import main
from icu_leakage_check.findings import Status


def _leaky(cohort):
    rng = np.random.default_rng(0)
    df = cohort.copy()
    df["split"] = np.where(rng.random(len(df)) < 0.25, "test", "train")
    df["unitdischargeoffset"] = rng.integers(1500, 9000, len(df))
    df["diedinhospital"] = df["died"]
    return df


def test_full_audit_end_to_end(cohort):
    df = _leaky(cohort)
    r = LeakageAudit(df, "died", "split", dataset="eicu", index_time="index_offset",
                     calendar_time="hospitaldischargeyear", outcome_time="death_offset").run()
    assert r.has_problems("critical")
    assert "diedinhospital" in r.leaky_features("high")
    assert r.exit_code("high") == 1
    ids = {f.check_id for f in r.findings}
    assert {"entity_overlap", "hindsight_features", "declarations"} <= ids
    frame = r.to_frame()
    assert frame.iloc[0]["status"] == "fail"
    d = json.loads(r.to_json())
    assert d["summary"]["worst_severity"] == "critical"
    assert "Leakage audit report" in r.to_html()
    assert "## Findings" in r.to_markdown()
    assert "Verdict" in r.summary()
    assert r.metadata["identifiers"]["patient_id"] == "uniquepid"


def test_clean_audit_has_no_failures(cohort):
    from icu_leakage_check import group_split

    df = cohort.copy()
    df["split"] = group_split(df, "uniquepid", test_size=0.25, stratify="died")
    r = audit(df, "died", "split", dataset="eicu", index_time="index_offset", outcome_time="death_offset",
              calendar_time="hospitaldischargeyear", exclude_columns=["hospitalid"])
    assert not r.has_problems("high"), r.summary()


def test_audit_split_helper_and_merge(cohort):
    train = cohort.iloc[:1000]
    test = cohort.iloc[900:]
    r = audit_split(train, test, "died", dataset="eicu", checks=["entity_overlap"])
    assert r.by_check("entity_overlap")[0].status == Status.FAIL
    m = merge_results([r, r], labels=["a", "b"])
    assert set(m["pipeline"]) == {"a", "b"}


def test_options_change_thresholds(cohort):
    df = cohort.copy()
    r = audit(df, "died", dataset="eicu", feature_columns=["gcs"], options={"single_feature_dominance.high_auroc": 0.5},
              checks=["single_feature_dominance"])
    assert r.by_check("single_feature_dominance")[0].status == Status.WARN


def test_cli_audit_scan_schema_locate(tmp_path, cohort, capsys):
    df = _leaky(cohort)
    data = tmp_path / "features.csv"
    df.to_csv(data, index=False)
    html = tmp_path / "r.html"
    js = tmp_path / "r.json"
    code = main(["audit", "--data", str(data), "--label", "died", "--split", "split", "--dataset", "eicu",
                 "--index-time", "index_offset", "--html", str(html), "--json", str(js)])
    assert code == 1 and html.exists() and js.exists()
    assert main(["audit", "--data", str(data), "--label", "died", "--split", "split", "--fail-on", "never"]) == 0

    cfg = tmp_path / "audit.json"
    cfg.write_text(json.dumps({"data": "features.csv", "label": "died", "split": "split", "dataset": "eicu",
                               "exclude_columns": ["diedinhospital", "unitdischargeoffset"], "fail_on": "critical"}))
    assert main(["audit", "--config", str(cfg)]) == 0  # remaining problems are high, not critical
    cfg.write_text(json.dumps({"data": "features.csv", "label": "died", "split": "split", "dataset": "eicu", "fail_on": "high"}))
    assert main(["audit", "--config", str(cfg)]) == 1

    assert main(["scan", "--dataset", "eicu", "--columns", "age,unitdischargeoffset,apachescore", "--fail-on", "never"]) == 0
    out = capsys.readouterr().out
    assert "unitdischargeoffset" in out

    assert main(["schema", "--dataset", "mimic-iv"]) == 0
    assert main(["schema", "--dataset", "eicu", "--table", "apachePatientResult"]) == 0
    assert main(["locate", "--dataset", "eicu", "--root", str(tmp_path)]) == 0
    assert main(["taxonomy"]) == 0
    assert main(["checks"]) == 0
