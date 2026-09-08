"""Streaming readers for delimited files, parquet and dataframes.
"""
from __future__ import annotations

import os
from typing import Any


from .._simulation_result import SimulationResult


def _column(data: Any, name: str) -> list:
    """One column of a DataFrame / Arrow table / dict of lists, as a list.

    Duck-typed on purpose: this library carries no DataFrame dependency, and
    an isinstance check would mean acquiring one just to recognise a type.
    """
    if hasattr(data, "column_names") and hasattr(data, "column"):
        if name not in data.column_names:
            raise KeyError(f"no column {name!r}; have {list(data.column_names)}")
        return data.column(name).to_pylist()          # pyarrow.Table
    if hasattr(data, "columns"):
        cols = list(data.columns)
        if name not in cols:
            raise KeyError(f"no column {name!r}; have {cols}")
        col = data[name]
        return col.tolist() if hasattr(col, "tolist") else list(col)
    if isinstance(data, dict):
        if name not in data:
            raise KeyError(f"no column {name!r}; have {sorted(data)}")
        col = data[name]
        return col.tolist() if hasattr(col, "tolist") else list(col)
    raise TypeError(
        f"expected a DataFrame, an Arrow table or a dict of lists, got "
        f"{type(data).__name__}")


def _iter_delimited(path, *, sep, sequence_column, abundance_column,
                    has_header, chunk_size):
    """Yield ``(sequences, abundances_or_None)`` chunks from a text file."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    seqs: list[str] = []
    ab: list[float] | None = None if abundance_column is None else []
    with open(os.fspath(path), encoding="utf-8") as fh:
        if has_header:
            fh.readline()
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split(sep) if sep is not None else line.split()
            seqs.append(parts[sequence_column])
            if ab is not None:
                ab.append(float(parts[abundance_column])
                          if len(parts) > abundance_column else 1.0)
            if len(seqs) >= chunk_size:
                yield seqs, ab
                seqs = []
                ab = None if abundance_column is None else []
    if seqs:
        yield seqs, ab


def _open_parquet_for_streaming(pq, path):
    """A ParquetFile that does not accumulate buffers as it is read.

    The reader's ``pre_buffer`` default coalesces each row group's column
    chunks into buffers it keeps for the life of the file. That is the right
    default for high-latency storage, where it turns many small reads into
    few large ones. It is the wrong one here: this reader streams a file once,
    in order, and never looks at a byte twice, so those buffers are memory
    that grows with the file and is never read again.

    Measured on the 926,909,993-row corpus, reading 200,000,000 rows, one
    process per configuration:

        batch      pre_buffer   peak Arrow pool   peak RSS
        1,000,000  on              1,109.9 MiB     1.39 GB
        1,000,000  off               39.9 MiB      0.35 GB
        2,000,000  on              1,139.9 MiB     1.55 GB
        2,000,000  off               71.4 MiB      0.54 GB

    About 5.5 MB per million rows, so the whole corpus retains roughly 5 GB --
    the larger part of the 7.9 GB peak recorded for a foundation build -- at
    no difference in wall time.

    ``pre_buffer`` reached ``ParquetFile`` in pyarrow 4.0. pyarrow is an
    optional dependency here with no declared floor, so an older one falls
    back rather than failing: it loses the saving, not the ability to read.
    """
    try:
        return pq.ParquetFile(path, pre_buffer=False)
    except TypeError:                                    # pragma: no cover
        return pq.ParquetFile(path)


def _iter_parquet(path, sequence_column, abundance_column, batch_size):
    """Yield ``(sequences, abundances_or_None)`` chunks from a parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:                                  # pragma: no cover
        raise ImportError(
            "reading parquet needs pyarrow; `pip install pyarrow`, or use "
            "from_sequence_file / from_dataframe") from None
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    cols = [sequence_column]
    if abundance_column is not None:
        cols.append(abundance_column)
    pf = _open_parquet_for_streaming(pq, os.fspath(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        seqs = batch.column(sequence_column).to_pylist()
        ab = (None if abundance_column is None
              else [float(x) for x in batch.column(abundance_column).to_pylist()])
        yield seqs, ab
