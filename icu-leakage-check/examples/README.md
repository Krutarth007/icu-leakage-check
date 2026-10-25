# Examples

| File | What it does | Data needed |
|---|---|---|
| `icu-leakage-check demo` (command) | Leaky and clean pipelines on simulated ICU data | None |
| `eicu_mortality_24h.py` | 24-hour in-hospital mortality on eICU-CRD, clean and leaky, audited and compared | eICU-CRD 2.0 |
| `mimic_iv_mortality_24h.py` | The same on MIMIC-IV (hosp and icu modules) | MIMIC-IV 2.2 or later |
| `audit_config.json` | A configuration file for `icu-leakage-check audit --config` | Your own feature table |

Both database examples write HTML and JSON reports, a performance table and an inflation table
to the output folder. Only aggregate results are written.

```bash
pip install "icu-leakage-check[fast]"

python eicu_mortality_24h.py --root "C:/path/to/eicu-collaborative-research-database-2.0" --out eicu_audit --max-stays 5000
python mimic_iv_mortality_24h.py --root "C:/path/to/mimic-iv-2.2" --out mimic_audit --max-stays 5000
```

Drop `--max-stays` for the full cohort. On a laptop, convert the large tables to Parquet first:

```bash
icu-leakage-check convert --dataset eicu --root "C:/path/to/eicu-collaborative-research-database-2.0" --out "C:/path/to/eicu_parquet" --tables lab,vitalPeriodic,vitalAperiodic,patient
icu-leakage-check convert --dataset mimic-iv --root "C:/path/to/mimic-iv-2.2" --out "C:/path/to/mimic_parquet" --tables chartevents,labevents,icustays,admissions,patients,diagnoses_icd
```

and then pass the Parquet folder as `--root`.
