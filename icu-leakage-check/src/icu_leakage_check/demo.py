"""Synthetic ICU data and a worked example that needs no credentialed data.

The generator produces a small cohort with repeat admissions, several
hospitals, time-stamped laboratory events with result delays, infusions, and
a few deliberately problematic columns, laid out like eICU-CRD (minute
offsets from unit admission). Everything is simulated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

_LABS = {
    # name: (mean, sd, effect of latent severity, effect on later values)
    "lactate": (1.6, 0.5, 0.45, 0.6),
    "creatinine": (1.1, 0.35, 0.25, 0.5),
    "potassium": (4.1, 0.4, 0.05, 0.1),
    "wbc": (10.0, 3.0, 0.15, 0.3),
}


def make_demo_data(n_patients: int = 1500, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(stays, events)``.

    ``stays`` has one row per ICU stay with identifiers, demographics, the
    label ``died`` and its time ``death_offset``. ``events`` is in the
    canonical event format used by :mod:`icu_leakage_check.events`.
    """
    rng = np.random.default_rng(seed)
    stays = []
    events = []
    sid = 100000
    for p in range(n_patients):
        n_stays = int(rng.choice([1, 1, 1, 1, 2, 2, 3]))
        hospital = int(rng.integers(1, 31))
        frailty = rng.normal(0, 0.6)
        for v in range(n_stays):
            sid += 1
            severity = frailty + rng.normal(0, 1)
            age = int(np.clip(rng.normal(64, 15), 18, 90))
            gcs = int(np.clip(round(14 - 1.4 * max(severity, 0) + rng.normal(0, 1.2)), 3, 15))
            vent = int(rng.random() < 1 / (1 + np.exp(-(severity - 0.3))))
            logit = -3.1 + 0.03 * (age - 64) + 0.9 * severity + 0.6 * vent - 0.12 * (gcs - 14)
            died = int(rng.random() < 1 / (1 + np.exp(-logit)))
            unit_los = float(np.clip(rng.gamma(2.2, 1500), 1700, 20000))
            death_offset = float(rng.uniform(1600, unit_los)) if died else np.nan
            discharge = death_offset if died else unit_los
            stays.append({
                "uniquepid": f"D{p:05d}", "patienthealthsystemstayid": sid + 500000, "patientunitstayid": sid,
                "hospitalid": hospital, "hospitaldischargeyear": int(rng.choice([2014, 2015])),
                "age": age, "gcs": gcs, "vent_day1": vent, "died": died, "death_offset": death_offset,
                "unitdischargeoffset": round(discharge), "index_offset": 1440.0,
            })
            for name, (mu, sd, eff, late) in _LABS.items():
                n = int(rng.integers(6, 16))
                times = np.sort(rng.uniform(-120, discharge, n))
                for t in times:
                    drift = late * died * max(0.0, (t - 1440) / max(discharge - 1440, 1)) * 3
                    val = mu + sd * rng.normal() + eff * severity * sd * 2 + drift * sd
                    events.append({
                        "entity_id": sid, "variable": name, "value": float(max(val, 0.05)),
                        "event_time": float(t), "available_time": float(t + rng.uniform(30, 240)), "end_time": np.nan,
                    })
            if vent or rng.random() < 0.3:
                start = float(rng.uniform(0, 2000))
                stop = float(start + rng.uniform(120, 4000))
                events.append({
                    "entity_id": sid, "variable": "norepinephrine", "value": float(rng.uniform(2, 20) * (stop - start) / 60.0 * (1 + died)),
                    "event_time": start, "available_time": start, "end_time": stop,
                })
    return pd.DataFrame(stays), pd.DataFrame(events)


def run_demo(n_patients: int = 1500, seed: int = 7, html_path: str = None) -> Dict[str, object]:
    """Build a leaky and a clean pipeline on synthetic data, audit both and compare them."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .audit import LeakageAudit
    from .compare import leakage_impact
    from .features import build_window_features
    from .metrics import evaluate
    from .splits import group_split

    stays, events = make_demo_data(n_patients, seed)
    index = stays[["patientunitstayid", "index_offset"]].rename(columns={"patientunitstayid": "entity_id", "index_offset": "index_time"})
    base_cols = ["age", "gcs", "vent_day1"]

    # Clean pipeline: patient-level split, features from data available in the first 24 hours
    clean_feats, provenance = build_window_features(events, index, window=1440, aggregations=("last", "max", "mean", "sum"), return_provenance=True)
    clean = stays.merge(clean_feats, left_on="patientunitstayid", right_index=True, how="left")
    clean["split"] = group_split(clean, "uniquepid", test_size=0.3, stratify="died", random_state=seed)
    clean_x = base_cols + list(clean_feats.columns)
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    tr = clean["split"] == "train"
    model.fit(clean.loc[tr, clean_x], clean.loc[tr, "died"])
    clean["score"] = model.predict_proba(clean[clean_x])[:, 1]

    # Leaky pipeline: row-level split, whole-stay features, a discharge field, preprocessing on all rows
    leaky_feats = build_window_features(events, index, cutoff="none", time_basis="event", interval_policy="keep", aggregations=("last", "max", "mean", "sum"))
    leaky = stays.merge(leaky_feats, left_on="patientunitstayid", right_index=True, how="left")
    leaky["split"] = np.where(np.random.default_rng(seed).random(len(leaky)) < 0.3, "test", "train")
    leaky_x = base_cols + ["unitdischargeoffset"] + list(leaky_feats.columns)
    leaky_model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    leaky_model.fit(leaky[leaky_x], leaky["died"])  # fitted on every row
    leaky["score"] = leaky_model.predict_proba(leaky[leaky_x])[:, 1]
    _, leaky_prov = build_window_features(events, index, cutoff="none", time_basis="event", interval_policy="keep", return_provenance=True)

    common = dict(label="died", split="split", dataset="eicu", index_time="index_offset",
                  outcome_time="death_offset", calendar_time="hospitaldischargeyear", scores="score")
    clean_result = LeakageAudit(clean, feature_columns=clean_x, events=provenance, preprocessor=model,
                                preprocessor_columns=clean_x, split_strategy="patient", **common).run()
    leaky_result = LeakageAudit(leaky, feature_columns=leaky_x, events=leaky_prov, preprocessor=leaky_model,
                                preprocessor_columns=leaky_x, split_strategy="random", **common).run()

    te_c = clean["split"] == "test"
    te_l = leaky["split"] == "test"
    reported = pd.DataFrame({
        "clean (honest)": evaluate(clean.loc[te_c, "died"], clean.loc[te_c, "score"]),
        "leaky (as reported)": evaluate(leaky.loc[te_l, "died"], leaky.loc[te_l, "score"]),
    })

    # Same honest test rows, scored by a model trained on leaky features within the honest split
    lk = leaky.drop(columns=["split", "score"]).merge(clean[["patientunitstayid", "split"]], on="patientunitstayid")
    m2 = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    trm = lk["split"] == "train"
    m2.fit(lk.loc[trm, leaky_x], lk.loc[trm, "died"])
    tem = lk["split"] == "test"
    lk_scores = m2.predict_proba(lk.loc[tem, leaky_x])[:, 1]
    aligned = clean.set_index("patientunitstayid").loc[lk.loc[tem, "patientunitstayid"]]
    impact = leakage_impact(aligned["died"].to_numpy(), aligned["score"].to_numpy(), lk_scores, n_boot=300,
                            groups=aligned["uniquepid"].to_numpy())
    if html_path:
        leaky_result.to_html(html_path)
    return {"clean": clean_result, "leaky": leaky_result, "reported_performance": reported, "impact": impact}
