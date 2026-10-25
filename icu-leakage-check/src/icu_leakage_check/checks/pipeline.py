"""Pipeline forensics: preprocessing, resampling and evaluation reuse."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..findings import Finding, Severity, Status
from ..metrics import youden_threshold
from ..splits import TEST, TRAIN, VALID
from .base import Check, fraction, need, register


# ------------------------------------------------------------ preprocessing
def _to_frame(X, columns=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cols = list(columns) if columns is not None and len(columns) == X.shape[1] else [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def _select(X: pd.DataFrame, cols) -> pd.DataFrame:
    if callable(cols):
        cols = cols(X)
    if isinstance(cols, str):
        return X[[cols]]
    cols = list(cols) if not isinstance(cols, slice) else cols
    if isinstance(cols, slice):
        return X.iloc[:, cols]
    if len(cols) and isinstance(cols[0], (bool, np.bool_)):
        return X.loc[:, np.asarray(cols, dtype=bool)]
    if len(cols) and isinstance(cols[0], (int, np.integer)):
        return X.iloc[:, cols]
    return X[cols]


def _walk(est, X: pd.DataFrame, path: str) -> Iterator[Tuple[str, Any, pd.DataFrame]]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError:  # pragma: no cover
        yield path, est, X
        return
    if isinstance(est, Pipeline):
        cur = X
        for name, step in est.steps:
            if step is None or step == "passthrough":
                continue
            yield from _walk(step, cur, f"{path}/{name}" if path else name)
            if not hasattr(step, "transform"):
                break
            try:
                inp = cur if hasattr(step, "feature_names_in_") else cur.to_numpy()
                out = step.transform(inp)
                names = step.get_feature_names_out() if hasattr(step, "get_feature_names_out") else None
            except Exception:
                break
            cur = _to_frame(out, names)
        return
    if isinstance(est, ColumnTransformer):
        for name, trans, cols in est.transformers_:
            if trans in ("drop", "passthrough") or trans is None:
                continue
            try:
                sub = _select(X, cols)
            except Exception:
                continue
            yield from _walk(trans, sub, f"{path}/{name}" if path else name)
        return
    if isinstance(est, FeatureUnion):
        for name, trans in est.transformer_list:
            if trans in ("drop", None):
                continue
            yield from _walk(trans, X, f"{path}/{name}" if path else name)
        return
    yield path or type(est).__name__, est, X


def _numeric(X: pd.DataFrame) -> np.ndarray:
    return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _stat_functions(est) -> List[Tuple[str, np.ndarray, Callable[[np.ndarray], np.ndarray]]]:
    name = type(est).__name__
    out = []
    with np.errstate(all="ignore"):
        if name == "StandardScaler":
            if getattr(est, "with_mean", True) and getattr(est, "mean_", None) is not None:
                out.append(("mean_", np.asarray(est.mean_, float), lambda a: np.nanmean(a, axis=0)))
            if getattr(est, "with_std", True) and getattr(est, "var_", None) is not None:
                out.append(("var_", np.asarray(est.var_, float), lambda a: np.nanvar(a, axis=0)))
        elif name == "MinMaxScaler":
            out.append(("data_min_", np.asarray(est.data_min_, float), lambda a: np.nanmin(a, axis=0)))
            out.append(("data_max_", np.asarray(est.data_max_, float), lambda a: np.nanmax(a, axis=0)))
        elif name == "MaxAbsScaler":
            out.append(("max_abs_", np.asarray(est.max_abs_, float), lambda a: np.nanmax(np.abs(a), axis=0)))
        elif name == "RobustScaler":
            if getattr(est, "with_centering", True) and getattr(est, "center_", None) is not None:
                out.append(("center_", np.asarray(est.center_, float), lambda a: np.nanmedian(a, axis=0)))
        elif name == "SimpleImputer":
            strat = getattr(est, "strategy", "mean")
            stats = getattr(est, "statistics_", None)
            if stats is not None:
                try:
                    stats = np.asarray(stats, float)
                except (TypeError, ValueError):
                    stats = None
            if stats is not None and strat == "mean":
                out.append(("statistics_", stats, lambda a: np.nanmean(a, axis=0)))
            elif stats is not None and strat == "median":
                out.append(("statistics_", stats, lambda a: np.nanmedian(a, axis=0)))
    return out


def _compare(fitted, candidates: Dict[str, np.ndarray], rtol=1e-5, atol=1e-8):
    """Which candidate statistic the fitted values match, on distinguishable entries."""
    keys = list(candidates)
    base = candidates[keys[0]]
    distinguish = np.zeros_like(base, dtype=bool)
    for k in keys[1:]:
        distinguish |= ~np.isclose(base, candidates[k], rtol=rtol, atol=atol, equal_nan=True)
    if not distinguish.any() or fitted.shape != base.shape:
        return None, int(distinguish.sum())
    matches = []
    for k in keys:
        if np.allclose(fitted[distinguish], candidates[k][distinguish], rtol=rtol, atol=atol, equal_nan=True):
            matches.append(k)
    return (matches[0] if matches else "unknown"), int(distinguish.sum())


@register
class PreprocessingFitCheck(Check):
    check_id = "preprocessing_fit"
    leakage_type = "PREPROCESSING_FIT"
    title = "Learned preprocessing fitted on evaluation data"
    description = (
        "Recomputes the statistics stored in fitted scalers and imputers on the training rows and on "
        "all rows, and reports which one the fitted object matches."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.preprocessor is not None, "preprocessor (a fitted scikit-learn transformer or Pipeline)", out)
        need(ctx.has_split and TRAIN in ctx.split_names(), "split with a training part", out)
        need(len(ctx.evaluation_splits()) > 0, "an evaluation split", out)
        return out

    def run(self, ctx):
        pre = ctx.preprocessor
        cols = ctx.preprocessor_columns
        if cols is None and hasattr(pre, "feature_names_in_"):
            cols = [c for c in pre.feature_names_in_]
        if cols is None:
            cols = ctx.feature_columns
        X = ctx.data[list(cols)].reset_index(drop=True)
        split = ctx.split.reset_index(drop=True)
        train = (split == TRAIN).to_numpy()
        trainval = split.isin([TRAIN, VALID]).to_numpy()
        allrows = split.notna().to_numpy()
        results = []
        for path, est, Xin in _walk(pre, X, ""):
            funcs = _stat_functions(est)
            if not funcs:
                continue
            A = _numeric(Xin)
            for attr, fitted, fn in funcs:
                cands = {"training rows": fn(A[train]), "all rows": fn(A[allrows])}
                if VALID in ctx.split_names():
                    cands["training and validation rows"] = fn(A[trainval])
                match, n_dist = _compare(fitted, cands)
                results.append({
                    "step": path, "transformer": type(est).__name__, "attribute": attr,
                    "matches": match if match else "indistinguishable", "distinguishable_columns": n_dist,
                })
        if not results:
            return [self.skipped(["no supported fitted transformer found (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, SimpleImputer)"])]
        bad = [r for r in results if r["matches"] == "all rows"]
        unknown = [r for r in results if r["matches"] == "unknown"]
        ev = {"steps": results, "affected_kind": "step"}
        findings: List[Finding] = []
        if bad:
            findings.append(self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"{len(bad)} learned statistic(s) match all rows, including evaluation rows, rather than the training rows: "
                + ", ".join(f"{r['step']}.{r['attribute']}" for r in bad[:6]),
                evidence=ev, affected=[f"{r['step']}.{r['attribute']}" for r in bad],
            ))
        if unknown:
            findings.append(self.finding(
                Status.WARN, Severity.LOW,
                f"{len(unknown)} learned statistic(s) match neither the training rows nor all rows; "
                "they may have been fitted on other data or a resampled training set.",
                evidence=ev, affected=[f"{r['step']}.{r['attribute']}" for r in unknown],
                recommendation="Confirm which rows each transformer was fitted on.",
            ))
        if not findings:
            findings.append(self.passed("Every checked statistic matches the training rows (or cannot be distinguished).", ev))
        return findings


# ----------------------------------------------------------------- resampling
def _is_integer_valued(values: np.ndarray) -> np.ndarray:
    return np.isclose(values, np.round(values), atol=1e-9)


def _segment_membership(Q: np.ndarray, pool: np.ndarray, k: int = 15, rtol: float = 1e-6, min_dims: int = 2) -> np.ndarray:
    """1.0 for query rows lying exactly on a segment between two pool rows, else 0.0.

    Interpolating oversamplers create ``x = a + lam * (b - a)`` with ``0 < lam < 1``
    for neighbouring minority rows ``a`` and ``b``; real measurements essentially
    never satisfy this in two or more continuous dimensions.
    """
    from sklearn.neighbors import NearestNeighbors

    kk = min(k + 1, len(pool))
    nn = NearestNeighbors(n_neighbors=kk).fit(pool)
    dist, ind = nn.kneighbors(Q)
    out = np.zeros(len(Q), dtype=float)
    for qi in range(len(Q)):
        x = Q[qi]
        neigh = pool[ind[qi][dist[qi] > 1e-12]]
        if len(neigh) < 2:
            continue
        found = False
        for a in neigh:
            B = neigh - a
            dx = x - a
            nz = np.abs(B) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(nz, dx[None, :] / B, np.nan)
            zero_ok = np.all(np.where(~nz, np.abs(dx)[None, :] <= 1e-9, True), axis=1)
            n_valid = nz.sum(axis=1)
            lam_min = np.nanmin(np.where(nz, lam, np.inf), axis=1)
            lam_max = np.nanmax(np.where(nz, lam, -np.inf), axis=1)
            consistent = (lam_max - lam_min) <= rtol * (1 + np.abs(lam_max))
            ok = zero_ok & (n_valid >= min_dims) & consistent & (lam_min > 0) & (lam_max < 1)
            if ok.any():
                found = True
                break
        out[qi] = 1.0 if found else 0.0
    return out


@register
class SyntheticSampleCheck(Check):
    check_id = "synthetic_samples"
    leakage_type = "SYNTHETIC_IN_EVALUATION"
    title = "Synthetic minority records in evaluation data"
    description = (
        "Interpolating oversamplers such as SMOTE create minority-class rows with fractional values in "
        "columns that are otherwise whole numbers (flags, counts, scores). Finding such rows in an "
        "evaluation split shows the oversampling happened before the split."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.label is not None and ctx.is_binary_label(), "binary 0/1 label", out)
        need(ctx.has_split, "split", out)
        need(len(ctx.feature_columns) > 0, "feature columns", out)
        return out

    def run(self, ctx):
        min_rows = int(self.opt(ctx, "min_rows_per_column", 20))
        purity = float(self.opt(ctx, "integer_purity", 0.995))
        min_fraction = float(self.opt(ctx, "min_fraction", 0.05))
        max_queries = int(self.opt(ctx, "max_queries", 300))
        seed = int(self.opt(ctx, "random_state", 0))
        y = ctx.label_values()
        num_cols = [c for c in ctx.feature_columns if pd.api.types.is_numeric_dtype(ctx.data[c]) and not pd.api.types.is_bool_dtype(ctx.data[c])]
        if not num_cols:
            return [self.skipped(["numeric feature columns"])]
        majority = float(y.value_counts().idxmax())
        minority = 1.0 - majority
        maj_rows = (y == majority).to_numpy()
        min_rows_mask = (y == minority).to_numpy()
        X = ctx.data[num_cols].to_numpy(dtype=float)
        discrete, continuous = [], []
        for j, c in enumerate(num_cols):
            v = X[maj_rows, j]
            v = v[~np.isnan(v)]
            if len(v) < min_rows:
                continue
            if _is_integer_valued(v).mean() >= purity:
                discrete.append(j)
            elif len(np.unique(v)) > 10 and not np.isnan(X[min_rows_mask, j]).any():
                continuous.append(j)
        if not discrete and len(continuous) < 2:
            return [self.skipped(["integer-valued columns or at least two complete continuous columns to test against"])]

        frac_any = np.zeros(len(X), dtype=bool)
        if discrete:
            D = X[:, discrete]
            frac_any = np.any(~np.isnan(D) & ~_is_integer_valued(np.nan_to_num(D)), axis=1)

        seg_member = None
        if len(continuous) >= 2:
            C = X[:, continuous]
            sd = np.nanstd(C, axis=0)
            sd[sd == 0] = 1.0
            C = (C - np.nanmean(C, axis=0)) / sd
            pool_idx = np.where(min_rows_mask & ~np.isnan(C).any(axis=1))[0]
            seg_member = np.zeros(len(X), dtype=float) * np.nan
            if len(pool_idx) >= 3:
                rng = np.random.default_rng(seed)
                for s in ctx.split_names():
                    q = pool_idx[(ctx.split.to_numpy()[pool_idx] == s)]
                    if len(q) > max_queries:
                        q = rng.choice(q, max_queries, replace=False)
                    if len(q):
                        seg_member[q] = _segment_membership(C[q], C[pool_idx], min_dims=min(3, C.shape[1]))

        findings: List[Finding] = []
        per_split = {}
        flagged_eval = []
        flagged_train = False
        for s in ctx.split_names():
            m = (ctx.split == s).to_numpy()
            mi = m & min_rows_mask
            ma = m & maj_rows
            f_min = float(frac_any[mi].mean()) if mi.any() else 0.0
            f_maj = float(frac_any[ma].mean()) if ma.any() else 0.0
            seg = None
            if seg_member is not None and mi.any():
                vals = seg_member[mi]
                vals = vals[~np.isnan(vals)]
                seg = float(vals.mean()) if len(vals) else None
            per_split[s] = {
                "minority_rows": int(mi.sum()),
                "minority_fractional_in_integer_columns": round(f_min, 4),
                "majority_fractional_in_integer_columns": round(f_maj, 4),
                "minority_on_segment_between_neighbours": None if seg is None else round(seg, 4),
                "prevalence": round(float(y[m].mean()), 4) if m.any() else None,
            }
            sig_int = f_min >= min_fraction and f_min > 5 * f_maj + 0.01
            sig_seg = seg is not None and seg >= min_fraction
            if sig_int or sig_seg:
                if s == TRAIN:
                    flagged_train = True
                else:
                    flagged_eval.append(s)
        ev = {
            "integer_valued_columns_used": [num_cols[j] for j in discrete][:30],
            "continuous_columns_used": [num_cols[j] for j in continuous][:30],
            "by_split": per_split,
            "affected_kind": "split",
        }
        if flagged_eval:
            findings.append(self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"Evaluation split(s) {', '.join(flagged_eval)} contain minority-class rows that are interpolations of other "
                "minority rows (fractional values in integer columns or exact collinearity with two neighbours), the "
                "signature of synthetic oversampling before the split.",
                evidence=ev, affected=flagged_eval,
            ))
        balanced = [s for s in ctx.evaluation_splits()
                    if per_split[s]["prevalence"] is not None and abs(per_split[s]["prevalence"] - 0.5) <= 0.03
                    and int((ctx.split == s).sum()) >= 200]
        if balanced and not ctx.options.get("balanced_by_design", False) and not flagged_eval:
            findings.append(self.finding(
                Status.WARN, Severity.MEDIUM,
                f"Evaluation split(s) {', '.join(balanced)} are almost exactly balanced, which is unusual for clinical outcomes "
                "and often follows resampling before splitting.",
                evidence=ev, affected=balanced,
                recommendation="Evaluate on data with the natural outcome prevalence; resample only inside the training split.",
            ))
        if not findings:
            msg = "No synthetic-record signature in evaluation data."
            if flagged_train:
                msg += " Oversampling is confined to the training split, which is acceptable."
            findings.append(self.passed(msg, ev))
        return findings


# ------------------------------------------------------------ evaluation reuse
def _f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tp, fp = tps[last], fps[last]
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(ps[last][int(np.argmax(f1))])


@register
class ThresholdSelectionCheck(Check):
    check_id = "threshold_selection"
    leakage_type = "EVALUATION_REUSE"
    title = "Decision threshold tuned on evaluation data"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.scores is not None, "scores", out)
        need(ctx.threshold is not None, "threshold", out)
        need(ctx.is_binary_label(), "binary 0/1 label", out)
        need(ctx.has_split and TEST in ctx.split_names(), "a test split", out)
        need(ctx.has_split and (TRAIN in ctx.split_names() or VALID in ctx.split_names()), "a training or validation split", out)
        return out

    def run(self, ctx):
        y = ctx.label_values().to_numpy(dtype=float)
        p = ctx.scores
        split = ctx.split.to_numpy()
        ref_name = VALID if VALID in ctx.split_names() else TRAIN
        tol = float(self.opt(ctx, "tolerance", 1e-6))

        def pick(mask):
            ok = mask & ~np.isnan(y) & ~np.isnan(p)
            return {"youden": youden_threshold(y[ok], p[ok]), "f1": _f1_threshold(y[ok], p[ok])}

        on_test = pick(split == TEST)
        on_ref = pick(split == ref_name)
        t = float(ctx.threshold)
        hits = [k for k, v in on_test.items() if abs(v - t) <= tol and abs(on_ref[k] - t) > tol]
        ev = {"threshold": t, "optimal_on_test": on_test, f"optimal_on_{ref_name}": on_ref}
        if hits:
            return [self.finding(
                Status.WARN, Severity.MEDIUM,
                f"The reported threshold equals the {', '.join(hits)}-optimal threshold on the test split but not on '{ref_name}'.",
                evidence=ev,
            )]
        return [self.passed("The threshold does not coincide with an optimum computed on the test split.", ev)]


# ------------------------------------------------------------- declarations
DECLARATION_ITEMS = {
    "preprocessing_fitted_on_training_only": ("PREPROCESSING_FIT", "Learned preprocessing was fitted on training data only"),
    "feature_selection_on_training_only": ("PREPROCESSING_FIT", "Feature selection used training data only"),
    "resampling_inside_training_only": ("SYNTHETIC_IN_EVALUATION", "Oversampling or undersampling was applied inside the training data only"),
    "hyperparameters_tuned_without_test": ("EVALUATION_REUSE", "Hyperparameters were tuned without the test split"),
    "threshold_chosen_without_test": ("EVALUATION_REUSE", "The decision threshold was chosen without the test split"),
    "test_set_used_once": ("EVALUATION_REUSE", "The test split was used once, for the final evaluation"),
    "cohort_defined_at_prediction_time": ("COHORT_FUTURE_INFORMATION", "Inclusion and exclusion used only information available at the prediction time"),
    "outcome_window_after_prediction_time": ("POST_OUTCOME_INSTANCE", "The outcome window starts after the prediction time"),
}


@register
class DeclarationsCheck(Check):
    check_id = "declarations"
    leakage_type = "EVALUATION_REUSE"
    title = "Declared pipeline practices"
    description = "Records practices that cannot be verified from data, as declared by the analyst."

    def run(self, ctx):
        d = ctx.declarations
        if not d:
            return [self.finding(
                Status.SKIPPED, Severity.INFO,
                "No declarations supplied. Items that cannot be verified from data: " + ", ".join(DECLARATION_ITEMS),
                evidence={"items": {k: v[1] for k, v in DECLARATION_ITEMS.items()}},
            )]
        findings = []
        for key, value in d.items():
            lt, text = DECLARATION_ITEMS.get(key, ("EVALUATION_REUSE", key))
            if value is True:
                findings.append(self.passed(f"Declared: {text}.", {"item": key}, leakage_type=lt))
            elif value is False:
                findings.append(self.finding(
                    Status.WARN, Severity.HIGH, f"Declared NOT followed: {text}.",
                    evidence={"item": key, "affected_kind": "declaration"}, affected=[key], leakage_type=lt,
                ))
            else:
                findings.append(self.finding(
                    Status.WARN, Severity.LOW, f"Declaration '{key}' is {value!r}; expected True or False.",
                    evidence={"item": key, "affected_kind": "declaration"}, affected=[key], leakage_type=lt,
                    recommendation="Record each item as True or False.",
                ))
        missing = [k for k in DECLARATION_ITEMS if k not in d]
        if missing:
            findings.append(self.finding(
                Status.SKIPPED, Severity.INFO, "Not declared: " + ", ".join(missing),
                evidence={"items": missing},
            ))
        return findings
