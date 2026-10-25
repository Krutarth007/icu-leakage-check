import numpy as np
import pandas as pd
import pytest


def make_cohort(n_patients=600, seed=0):
    """Synthetic ICU cohort with repeat stays, several hospitals and minute offsets."""
    rng = np.random.default_rng(seed)
    rows = []
    stay = 1000
    for pid in range(n_patients):
        n_stays = rng.choice([1, 1, 1, 2, 3])
        site = int(rng.integers(1, 21))
        frail = rng.normal()
        for _ in range(n_stays):
            stay += 1
            age = int(np.clip(rng.normal(65, 12), 18, 95))
            gcs = int(np.clip(round(rng.normal(13, 2)), 3, 15))
            vent = int(rng.random() < 0.3)
            lactate = float(np.exp(rng.normal(0.4 + 0.3 * frail, 0.4)))
            hr = float(rng.normal(90 + 5 * frail, 15))
            logit = -2.4 + 0.035 * (age - 65) - 0.25 * (gcs - 13) + 0.8 * vent + 0.5 * np.log(lactate) + 0.4 * frail
            died = int(rng.random() < 1 / (1 + np.exp(-logit)))
            outcome_time = float(rng.uniform(1500, 15000)) if died else np.nan
            rows.append({
                "uniquepid": f"P{pid:05d}", "patientunitstayid": stay, "hospitalid": site,
                "hospitaldischargeyear": int(rng.choice([2014, 2015])),
                "age": age, "gcs": gcs, "vent": vent, "lactate": lactate, "heartrate": hr,
                "died": died, "index_offset": 1440.0, "death_offset": outcome_time,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def cohort():
    return make_cohort()


@pytest.fixture
def events_for(cohort):
    """Raw lab-style events around the prediction time (minute offsets)."""
    rng = np.random.default_rng(1)
    rows = []
    for sid in cohort["patientunitstayid"]:
        for t in rng.uniform(-200, 4000, 12):
            rows.append({
                "entity_id": sid, "variable": "creatinine", "value": float(rng.normal(1.2, 0.4)),
                "event_time": float(t), "available_time": float(t + rng.uniform(20, 180)), "end_time": np.nan,
            })
    return pd.DataFrame(rows)


def write_nested(root, rel_dir, table, df, gz=False):
    """Write ``df`` in the '<table>.csv/<table>.csv' layout (or .csv.gz)."""
    d = root / rel_dir if rel_dir else root
    if gz:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{table}.csv.gz", index=False, compression="gzip")
    else:
        folder = d / f"{table}.csv"
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / f"{table}.csv", index=False)
