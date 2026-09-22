"""
Build the 927M-sequence foundation model once and cache it to disk.

The runtime profile times the build itself, but the analytics tiers need a
finalized foundation model to work against and rebuilding it for every case
would dominate the run. This produces the cached artefact and records what the
build cost, so the harness can reuse it.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import psutil
from BEAM import Builder, BeamModel

PARQUET = os.path.join(
    ROOT, ".research/data_hub/release_20260812_censored/"
    "tcrb_human_cdr3_abundance_productive_maxlen27_censored.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "cache", "foundation.beam"))
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="rows, 0 = all")
    ap.add_argument("--batch", type=int, default=2_000_000)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import pyarrow.parquet as pq

    proc = psutil.Process()
    # pre_buffer=False for the same reason BEAM.io.readers does it: the
    # reader's default holds each row group's coalesced column chunks for the
    # life of the file, which on this corpus is about 5 GB of the peak below
    # and is never read twice. See _open_parquet_for_streaming.
    f = pq.ParquetFile(PARQUET, pre_buffer=False)
    b = Builder(order=a.order)

    t0 = time.perf_counter()
    n = 0
    for batch in f.iter_batches(batch_size=a.batch,
                                columns=["cdr3_aa", "total_abundance"]):
        seqs = batch.column("cdr3_aa").to_pylist()
        ab = [float(x) for x in batch.column("total_abundance").to_pylist()]
        b.observe(seqs, ab)
        n += len(seqs)
        if n % 50_000_000 < a.batch:
            print(f"  {n:,} rows  {time.perf_counter()-t0:.0f}s  "
                  f"rss {proc.memory_info().rss/2**30:.1f} GB", flush=True)
        if a.limit and n >= a.limit:
            break
    t_observe = time.perf_counter() - t0

    t1 = time.perf_counter()
    m = b.build()
    t_finalize = time.perf_counter() - t1

    t2 = time.perf_counter()
    m.save(a.out)
    t_save = time.perf_counter() - t2

    info = dict(
        rows=n, order=a.order,
        t_observe_s=round(t_observe, 2),
        t_finalize_s=round(t_finalize, 2),
        t_save_s=round(t_save, 2),
        model_memory_bytes=m.memory_bytes,
        peak_rss_bytes=proc.memory_info().rss,
        file_bytes=os.path.getsize(a.out),
        n_observed=m.n_observed, n_skipped=m.n_skipped,
        n_states=m.n_states, n_cells=m.n_cells,
        n_nonzero_cells=m.n_nonzero_cells,
        total_weight=m.total_weight,
        observed_lengths=list(map(int, m.observed_lengths)),
    )
    with open(a.out + ".json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
