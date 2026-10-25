# Data leakage in ICU prediction research on MIMIC-IV and eICU-CRD

This folder holds the analysis code and aggregate results of a study of data leakage in
intensive care prediction models built on MIMIC-IV and the eICU Collaborative Research
Database (eICU-CRD). The study has three parts:

1. **Controlled experiments.** Seventeen leakage mechanisms, each a practice described in
   published ICU prediction studies, were added one at a time to a leakage-safe baseline
   and the resulting inflation of performance was measured.
2. **Audit detection.** Every leaky pipeline was audited with
   [`icu-leakage-check`](https://pypi.org/project/icu-leakage-check/), with and without
   feature provenance, to test whether the leak could be found from the data.
3. **Literature audit.** 793 full-text MIMIC-IV and eICU-CRD prediction studies were coded
   for the same mechanisms, the coding rules were checked against reference coding, and
   the results were used to estimate prevalence and an exploratory projection of
   field-level optimism.

The audit tool itself is documented in the [main README](../README.md).

## Main findings

| | |
|---|---|
| Cohorts | 57,460 MIMIC-IV stays and 129,258 eICU-CRD stays, prediction at 24 h after ICU admission |
| Largest AUROC inflation | Discharge fields up to +0.264, look-ahead features up to +0.167, same-admission diagnosis codes up to +0.071 |
| Small studies | Procedural leaks were negligible in full cohorts but large in simulated 500-stay studies (oversampling before the split, up to +0.245) |
| Audit detection | 30 of 30 prespecified mechanism-database cases flagged with provenance and declarations; 13 of 30 from the feature table alone; no findings on the clean baselines |
| Literature rules vs reference | Cohen's kappa 0.44 (sensitivity 0.52, specificity 0.89) |
| Corrected prevalence | Same-admission comorbidity codes 54.1%, model selection on the test set 35.4%, look-ahead summaries 27.9% |
| Safeguards | 2 studies reported when laboratory results became available |
| Projected field optimism | Mean AUROC optimism 0.051 (0.033 to 0.067 across eight scenarios), exploratory |

## Leakage mechanisms

| ID | Mechanism | Family |
|---|---|---|
| E01 | Repeat stays of test patients in training (row-level split) | Split integrity |
| E02 | Features from the first 48 hours for a 24-hour prediction (look-ahead) | Timing |
| E03 | Values filtered on clinical time, ignoring when results became available | Timing |
| E04 | Full totals of infusions and ventilation still running at the prediction time (MIMIC-IV) | Timing |
| E05 | Discharge fields and length of stay as predictors | Hindsight |
| E06 | Outcome column stored next to the predictors (eICU-CRD APACHE tables) | Outcome encoding |
| E07 | Same-admission billing codes and diagnoses entered after the prediction | Undated or late sources |
| E08 | Care limitation and code status documented at any time | Outcome proxy |
| E09 | APACHE first-day summaries used by a 6-hour model (eICU-CRD) | First-day summary |
| E10 | Patients already in AKI at the prediction time kept as positives (AKI task) | Post-outcome instances |
| E11 | Imputer and scaler fitted on all rows | Preprocessing |
| E12 | Feature selection on all rows | Preprocessing |
| E13 | Oversampling (SMOTE) before the split | Resampling |
| E14 | Decision threshold chosen on the test set | Evaluation reuse |
| E15 | Hyperparameters chosen on the test set | Evaluation reuse |
| E16 | "Temporal" validation ordered by MIMIC-IV shifted dates | Split order |
| E17 | Internal test set presented as external (eICU-CRD hospitals shared) | Site overlap |

Variants marked `b` in the result files (for example `E08b`, documentation before the
prediction only) are reference or sensitivity versions of the mechanism with the same number.

## Folder structure

The audit tool and the study live in the same repository. The Python package sits at the
root, and the study sits in `study/`.

```
icu-leakage-check/
├── README.md                      package documentation
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── pyproject.toml
├── .gitignore
├── .github/workflows/tests.yml
├── src/icu_leakage_check/         the package published on PyPI
│   ├── __init__.py, _version.py, py.typed
│   ├── audit.py, cli.py, report.py, compare.py, metrics.py
│   ├── context.py, events.py, features.py, findings.py, io.py
│   ├── splits.py, taxonomy.py, demo.py
│   ├── checks/                    split, temporal, content and pipeline checks
│   └── schemas/                   MIMIC-IV and eICU-CRD table timing
├── tests/
├── examples/                      MIMIC-IV and eICU-CRD example audits
├── docs/                          TAXONOMY.md, example_report.html
└── study/                         this folder
```

Inside `study/`:

```
study/
├── README.md
├── notebooks/
│   ├── Part1_cohorts_events_risk_surface.ipynb
│   ├── Part2_leakage_experiments.ipynb
│   └── Part3_literature_audit_and_impact.ipynb
├── results/
│   ├── part1/    cohorts, event timing and the leakage risk surface
│   ├── part2/    inflation and detection for each mechanism
│   └── part3/    literature audit, sample-size dependence, combinations, field-level optimism
└── literature/
    ├── codebook.md
    ├── coding_sheet_TEMPLATE.csv
    ├── coding_sheet_BLIND_random100.csv
    ├── coding_sheet_VERIFY_enriched74.csv
    ├── paper_level_indicators.csv
    └── paper_level_expected_optimism.csv
```

### Notebooks

| Notebook | What it does | Runtime* |
|---|---|---|
| Part 1 | Builds harmonised adult ICU cohorts, the three prediction tasks, two-clock event logs, hindsight and proxy fields and leakage-safe splits; quantifies the leakage risk surface of both databases; certifies the baseline with `icu-leakage-check` | about 9 min |
| Part 2 | Adds each mechanism to the baseline for 3 outcomes and 2 models in both databases, estimates paired differences with a 500-replicate patient-level bootstrap, and audits every leaky pipeline twice (table only and full provenance) | about 2 h |
| Part 3 | Searches Europe PMC, screens and codes full texts, validates the rules against the reference codes, corrects prevalence for misclassification, runs the sample-size and combination experiments and projects field-level optimism | about 12 h |

\*Windows laptop, Intel processor, Python 3.11.5. Parts 2 and 3 save results as they go and
resume where they stopped if interrupted.

### Results

Each `results/partN/` folder contains:

- `tables/`: every aggregate table used in the study, as CSV
- `figures/`: every figure, as PDF and PNG
- `key_numbers_partN.json`: the headline numbers of that part
- `manifest_partN.json`: software versions, configuration, a SHA-256 hash of every output
  file and a hash chain linking the three parts

Part 3 also contains:

- `raw/`: per-repeat results of the sample-size and combination experiments (aggregate
  model metrics only)
- `evidence_pack/`: the ICU-LEAK checklist and example audit cards for MIMIC-IV and eICU-CRD
- `study_summary.md`: a summary of all three parts

The ICU-LEAK checklist (`results/part3/tables/icu_leak_checklist.csv`) turns each mechanism
into a yes-or-no question, maps it to the `icu-leakage-check` check that verifies it, and
ranks items by estimated prevalence multiplied by AUROC inflation.

All tables are aggregates. Counts below 10 are suppressed.

### Literature data

- `codebook.md`: definitions of the 17 leakage indicators (I01 to I17) and the transparency
  indicators, with allowed values `risk`, `unclear`, `safe`, `not_mentioned` and `na`.
  I03, I04 and I10 are safeguard indicators: `risk` means the safeguard was not reported.
- `paper_level_indicators.csv`: one row per included study, with identifiers, the automated
  value of every indicator and the sentence each value was based on.
- `paper_level_expected_optimism.csv`: expected AUROC optimism per study under the main
  scenario.
- `coding_sheet_BLIND_random100.csv`: reference codes for a random sample of 100 studies,
  coded blind to the automated values. Sensitivity, specificity and kappa are estimated from
  this sample only.
- `coding_sheet_VERIFY_enriched74.csv`: reference codes for 74 further studies flagged as at
  risk for rare indicators, coded without blinding. They contribute only to the at-risk
  category.

The reference codes were produced by a large language model (Claude Opus 5.5, Anthropic)
from the open-access full texts, following the codebook, and were not checked by a human.
Paper-level values are automated screening results with known error rates. They are not
judgements of individual studies or their authors.

## What is not included

- **Patient-level data and outputs.** MIMIC-IV and eICU-CRD are available only to
  credentialed users under the PhysioNet data use agreement, so cohorts, event logs,
  feature tables and patient-level audit reports are not shared.
- **Full texts.** The literature snapshot is not redistributed. Every study can be retrieved
  from Europe PMC with the identifiers in `paper_level_indicators.csv`.

## Reproducing the study

### 1. Get the data

Request access to [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/) and
[eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) on PhysioNet. This requires
credentialing and completion of the required training.

### 2. Set up the environment

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS or Linux
pip install "icu-leakage-check[fast]" matplotlib scipy jupyter ipykernel
python -m ipykernel install --user --name icu-leakage-check --display-name "Python (icu-leakage-check)"
```

Versions used for the reported results: Python 3.11.5, icu-leakage-check 0.1.0,
pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, duckdb 1.5.6,
matplotlib 3.11.2.

### 3. Convert the databases to Parquet

```bash
icu-leakage-check convert --dataset mimic-iv --root "C:\data\mimic-iv-2.2" --out "C:\icu_parquet\mimic"
icu-leakage-check convert --dataset eicu --root "C:\data\eicu-collaborative-research-database-2.0" --out "C:\icu_parquet\eicu"
```

### 4. Set the paths

Edit the configuration cell at the top of each notebook, or set these environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ILC_MIMIC_ROOT` | MIMIC-IV Parquet folder | `C:\icu_parquet\mimic` |
| `ILC_EICU_ROOT` | eICU-CRD Parquet folder | `C:\icu_parquet\eicu` |
| `ILC_WORK_DIR` | Folder for all outputs | `C:\icu_leakage_work` |
| `ILC_MAX_STAYS` | Limit the number of stays for a quick test run (for example 5000) | full cohort |
| `ILC_N_BOOT` | Bootstrap replicates | 500 in Part 2, 200 in Part 3 |
| `ILC_OFFLINE` | `1` to reuse a saved literature snapshot instead of querying Europe PMC | `0` |

### 5. Run the notebooks in order

Run Part 1, then Part 2, then Part 3, using the **Python (icu-leakage-check)** kernel. Each
part reads the outputs of the previous one from `ILC_WORK_DIR`. The random seed is fixed
(`20261015`).

Before running Part 3, copy `coding_sheet_BLIND_random100.csv` and
`coding_sheet_VERIFY_enriched74.csv` from `literature/` to `ILC_WORK_DIR\part3\coding\`.
The notebook merges them and replaces the screening estimates with validated ones.

Part 3 searches Europe PMC with a fixed publication window (1 August 2020 to 30 September
2026). The reported search ran on 16 October 2026. A new run queries the live index, so
record counts can differ slightly as Europe PMC adds or updates records.

### 6. Check the results

Compare your `key_numbers_partN.json` files with the ones in `results/`. The manifests
record the configuration and file hashes of the reported run.

## Licence

The code is released under the MIT licence. The aggregate results and literature data
are shared for research use; please cite this repository if you use them.

## Citation

If you use this code, the results or the literature data, please cite:

> Patel K. Data leakage in ICU prediction research on MIMIC-IV and eICU-CRD: controlled
> experiments, a literature audit and a provenance-based audit tool. Code and data,
> version 0.1.0. Zenodo. https://doi.org/ZENODO_DOI

Please also cite the databases:

- Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health
  record dataset. *Sci Data*. 2023;10:1.
- Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database, a
  freely available multi-center database for critical care research. *Sci Data*.
  2018;5:180178.
- Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation*. 2000;101:e215-e220.

## Contact

Krutarth Patel, krutarthpatel1997@gmail.com
