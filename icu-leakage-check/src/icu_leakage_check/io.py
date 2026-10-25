"""Locate and read source tables in the layouts commonly produced on local machines.

Supported layouts for each table (names are matched case-insensitively):

* ``<root>/<module>/<table>.csv.gz`` (as distributed)
* ``<root>/<module>/<table>.csv``
* ``<root>/<module>/<table>.csv/<table>.csv`` (a folder named like the file, as left by
  some Windows extraction tools)
* ``<root>/<module>/<table>.parquet``
* the same patterns without ``<module>``

``module`` is ``hosp`` or ``icu`` for MIMIC-IV and absent for eICU-CRD.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from .schemas import DatasetSchema, get_schema

PathLike = Union[str, os.PathLike]

_SUFFIXES = (".csv.gz", ".csv", ".parquet")


def _listdir_ci(directory: Path) -> Dict[str, Path]:
    try:
        return {p.name.lower(): p for p in directory.iterdir()}
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return {}


def _resolve_in_dir(directory: Path, table: str) -> Optional[Path]:
    entries = _listdir_ci(directory)
    t = table.lower()
    for suffix in _SUFFIXES:
        hit = entries.get(t + suffix)
        if hit is None:
            continue
        if hit.is_file():
            return hit
        if hit.is_dir():
            inner = _listdir_ci(hit)
            for inner_suffix in _SUFFIXES:
                cand = inner.get(t + inner_suffix)
                if cand is not None and cand.is_file():
                    return cand
            files = [p for p in hit.iterdir() if p.is_file() and p.name.lower().endswith(_SUFFIXES)]
            if len(files) == 1:
                return files[0]
    return None


def find_table(
    root: PathLike,
    table: str,
    dataset: Union[str, DatasetSchema, None] = None,
    module: Optional[str] = None,
) -> Path:
    """Return the path of a table file, raising ``FileNotFoundError`` if absent."""
    root = Path(root).expanduser()
    schema = get_schema(dataset)
    if module is None and schema is not None and schema.has_table(table):
        module = schema.table(table).module
    search: List[Path] = []
    if module:
        search.append(root / module)
    search.append(root)
    if not module:
        for sub in ("hosp", "icu", "ed", "note"):
            search.append(root / sub)
    for directory in search:
        hit = _resolve_in_dir(directory, table)
        if hit is not None:
            return hit
    raise FileNotFoundError(
        f"Could not find table {table!r} under {root} (searched: {', '.join(str(s) for s in search)})"
    )


def list_tables(root: PathLike, dataset: Union[str, DatasetSchema]) -> pd.DataFrame:
    """Check which registry tables are present under ``root`` and where."""
    schema = get_schema(dataset)
    if schema is None:
        raise ValueError("list_tables needs a dataset name such as 'mimic-iv' or 'eicu'")
    rows = []
    for spec in schema.tables.values():
        try:
            p = find_table(root, spec.name, schema)
            rows.append({
                "table": spec.name, "module": spec.module, "timing": spec.timing,
                "found": True, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 1),
            })
        except FileNotFoundError:
            rows.append({
                "table": spec.name, "module": spec.module, "timing": spec.timing,
                "found": False, "path": None, "size_mb": None,
            })
    return pd.DataFrame(rows)


def _header(path: Path) -> List[str]:
    if path.name.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq

            return list(pq.read_schema(str(path)).names)
        except ImportError:
            return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def _parse_times(df: pd.DataFrame, spec, time_kind: str) -> pd.DataFrame:
    if spec is None:
        return df
    if time_kind == "datetime":
        for c in spec.datetime_columns:
            if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _apply_filters(df: pd.DataFrame, filters: Optional[Mapping[str, Iterable]]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, values in filters.items():
        col_l = col.lower()
        if col_l not in df.columns:
            raise KeyError(f"Filter column {col!r} not present in table")
        mask &= df[col_l].isin(set(values))
    return df.loc[mask]


def read_table(
    root: PathLike,
    table: str,
    dataset: Union[str, DatasetSchema, None] = None,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Mapping[str, Iterable]] = None,
    nrows: Optional[int] = None,
    chunksize: int = 2_000_000,
    engine: str = "auto",
    parse_times: bool = True,
    where: Optional[str] = None,
    row_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Read one table with lower-case column names.

    ``filters`` maps a column to the values to keep (for example
    ``{"stay_id": cohort_ids}``) and is applied while reading, so large event
    tables never need to fit in memory. ``engine='duckdb'`` is faster for very
    large files when the optional ``duckdb`` package is installed.

    ``where`` is an SQL condition evaluated by DuckDB while reading (for
    example ``"observationoffset <= 2880"``); it requires duckdb.
    ``row_filter`` is a function applied to every chunk (pandas engine) or to
    the result (other engines), for example
    ``lambda d: d[d["observationoffset"] <= 2880]``.
    """
    schema = get_schema(dataset)
    spec = schema.table(table) if (schema is not None and schema.has_table(table)) else None
    path = find_table(root, table, schema)

    if where is not None:
        engine = "duckdb"
    if engine == "auto":
        engine = "pandas"
        if filters:
            try:
                import duckdb  # noqa: F401

                engine = "duckdb"
            except ImportError:
                engine = "pandas"

    header = _header(path)
    lower_map = {h.lower(): h for h in header}
    if columns is not None:
        wanted = [c.lower() for c in columns]
        missing = [c for c in wanted if c not in lower_map]
        if missing:
            raise KeyError(f"Columns {missing} not in {path.name}; available: {sorted(lower_map)}")
        if filters:
            for f in filters:
                if f.lower() not in wanted:
                    wanted.append(f.lower())
        usecols = [lower_map[c] for c in wanted]
    else:
        usecols = None

    if engine == "duckdb":
        try:
            import duckdb  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("engine='duckdb' and where= need duckdb: pip install duckdb") from exc
        df = _read_duckdb(path, usecols, filters, nrows, lower_map, where)
        if row_filter is not None:
            df = row_filter(df)
    elif path.name.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols)
        df.columns = [c.lower() for c in df.columns]
        df = _apply_filters(df, filters)
        if row_filter is not None:
            df = row_filter(df)
        if nrows is not None:
            df = df.head(nrows)
    else:
        if filters or row_filter is not None:
            parts = []
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.lower() for c in chunk.columns]
                chunk = _apply_filters(chunk, filters)
                if row_filter is not None:
                    chunk = row_filter(chunk)
                parts.append(chunk)
                if nrows is not None and sum(len(p) for p in parts) >= nrows:
                    break
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[c.lower() for c in (usecols or header)])
            if nrows is not None:
                df = df.head(nrows)
        else:
            df = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
            df.columns = [c.lower() for c in df.columns]

    if parse_times:
        df = _parse_times(df, spec, schema.time_kind if schema else "datetime")
    return df.reset_index(drop=True)


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _source_candidates(path: Path) -> List[str]:
    if path.name.lower().endswith(".parquet"):
        return [f"read_parquet('{_sql_path(path)}')"]
    return [
        f"read_csv_auto('{_sql_path(path)}', sample_size=200000)",
        f"read_csv_auto('{_sql_path(path)}', sample_size=-1)",
    ]


def query_table(
    root: PathLike,
    table: str,
    sql: str,
    dataset: Union[str, DatasetSchema, None] = None,
    params: Optional[Mapping[str, object]] = None,
    memory_limit: Optional[str] = None,
) -> pd.DataFrame:
    """Run an SQL query against one table with DuckDB.

    Write ``{table}`` where the table goes, for example::

        query_table(root, "vitalPeriodic",
                    "SELECT patientunitstayid, avg(heartrate) AS hr FROM {table} "
                    "WHERE patientunitstayid IN (SELECT v FROM ids) GROUP BY 1",
                    dataset="eicu", params={"ids": cohort_ids})

    ``params`` registers DataFrames (or lists, as a one-column table ``v``)
    under the given names. Column names in the result are lower case.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("query_table requires duckdb: pip install duckdb") from exc
    if "{table}" not in sql:
        raise ValueError("sql must contain the placeholder {table}")
    path = find_table(root, table, dataset)
    con = duckdb.connect()
    try:
        if memory_limit:
            con.execute(f"SET memory_limit='{memory_limit}'")
        for name, value in (params or {}).items():
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame({"v": list(value)})
            con.register(name, frame)
        last_error = None
        for source in _source_candidates(path):
            try:
                df = con.execute(sql.replace("{table}", source)).df()
                df.columns = [c.lower() for c in df.columns]
                return df
            except duckdb.Error as exc:
                last_error = exc
        raise last_error
    finally:
        con.close()


def _read_duckdb(path: Path, usecols, filters, nrows, lower_map, extra_where: Optional[str] = None) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    try:
        sources = _source_candidates(path)
        if usecols:
            select = ", ".join(f'"{c}" AS "{c.lower()}"' for c in usecols)
        else:
            select = ", ".join(f'"{orig}" AS "{low}"' for low, orig in lower_map.items())
        where = []
        if filters:
            for i, (col, values) in enumerate(filters.items()):
                name = f"_ids_{i}"
                con.register(name, pd.DataFrame({"v": list(set(values))}))
                where.append(f'"{lower_map[col.lower()]}" IN (SELECT v FROM {name})')
        if extra_where:
            where.append(f"({extra_where})")
        last_error = None
        for source in sources:
            sql = f"SELECT {select} FROM {source}"
            if where:
                sql += " WHERE " + " AND ".join(where)
            if nrows is not None:
                sql += f" LIMIT {int(nrows)}"
            try:
                return con.execute(sql).df()
            except duckdb.Error as exc:  # type sniffing failed on a later row; rescan fully
                last_error = exc
        raise last_error
    finally:
        con.close()


def convert_to_parquet(
    root: PathLike,
    dataset: Union[str, DatasetSchema],
    out_dir: PathLike,
    tables: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    memory_limit: Optional[str] = "4GB",
    temp_directory: Optional[PathLike] = None,
    threads: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert CSV tables to Parquet with lower-case columns using DuckDB.

    Reading Parquet is many times faster than CSV and needs far less memory,
    which matters for the largest event tables. Output keeps the module
    folders, so ``out_dir`` can be used as ``root`` for :func:`read_table`.
    Requires ``pip install duckdb``.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("convert_to_parquet requires duckdb: pip install duckdb") from exc

    schema = get_schema(dataset)
    if schema is None:
        raise ValueError("convert_to_parquet needs a dataset name such as 'mimic-iv' or 'eicu'")
    out_dir = Path(out_dir)
    names = list(tables) if tables else schema.table_names()
    con = duckdb.connect()
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    if temp_directory:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{_sql_path(Path(temp_directory))}'")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    con.execute("SET preserve_insertion_order=false")

    rows = []
    try:
        for name in names:
            spec = schema.table(name)
            try:
                src = find_table(root, spec.name, schema)
            except FileNotFoundError:
                rows.append({"table": spec.name, "status": "missing", "path": None})
                continue
            dest_dir = out_dir / spec.module if spec.module else out_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{spec.name}.parquet"
            if dest.exists() and not overwrite:
                rows.append({"table": spec.name, "status": "exists", "path": str(dest)})
                continue
            if verbose:
                print(f"Converting {spec.name} ...", flush=True)
            sources = _source_candidates(src)
            tmp = dest.with_suffix(".parquet.tmp")
            last_error = None
            for source in sources:
                try:
                    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
                    select = ", ".join(f'"{c}" AS "{c.lower()}"' for c in cols)
                    con.execute(
                        f"COPY (SELECT {select} FROM {source}) TO '{_sql_path(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                    )
                    last_error = None
                    break
                except duckdb.Error as exc:  # type sniffing failed on a later row; rescan fully
                    last_error = exc
            if last_error is not None:
                raise last_error
            os.replace(tmp, dest)
            rows.append({"table": spec.name, "status": "converted", "path": str(dest)})
    finally:
        con.close()
    return pd.DataFrame(rows)
