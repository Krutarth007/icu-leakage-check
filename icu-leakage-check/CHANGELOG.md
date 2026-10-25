# Changelog

## 0.1.0 (2026-10-15)

First release.

* 16 checks covering 19 leakage mechanisms across split integrity, timing, feature content and pipeline fitting.
* Timing registries for all 31 MIMIC-IV hosp and icu tables and all 31 eICU-CRD tables.
* Readers for compressed, plain, Parquet and nested-folder layouts, with filtered reading and Parquet conversion.
* Leakage-safe event selection and window features with availability-time filtering and interval prorating.
* Patient-, site- and calendar-based splitting helpers.
* Metrics with bootstrap and paired bootstrap intervals, and leaky-versus-clean impact tables.
* Text, Markdown, JSON and self-contained HTML reports; command line with CI exit codes.
