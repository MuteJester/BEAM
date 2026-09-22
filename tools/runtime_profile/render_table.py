"""
Render every run into one wide table: rows are (tier, case), columns are runs.

The column header carries the date and the short commit, so a regression can be
read off the row and the revision to return to can be read off the header.
"""
import csv, json, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "TIMINGS.md")

GROUP_ORDER = ["lifecycle", "introspect", "scoring", "diversity", "surprisal",
               "spectrum", "structure", "generate", "analysis", "algebra"]
TIER_ORDER = ["tiny", "small", "foundation"]


def load_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        mf, tf = os.path.join(d, "manifest.json"), os.path.join(d, "timings.csv")
        if not (os.path.exists(mf) and os.path.exists(tf)):
            continue
        man = json.load(open(mf))
        rows = list(csv.DictReader(open(tf)))
        runs.append((os.path.basename(d), man, rows))
    return runs


def fmt_time(s):
    if s is None:
        return "—"
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "—"
    if s != s:
        return "err"
    if s < 1e-3:
        return f"{s*1e6:.0f} µs"
    if s < 1.0:
        return f"{s*1e3:.1f} ms"
    if s < 120:
        return f"{s:.2f} s"
    return f"{s/60:.1f} min"


def fmt_bytes(b):
    try:
        b = float(b)
    except (TypeError, ValueError):
        return "—"
    if b <= 0:
        return "—"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.0f} {u}" if u == "B" else f"{b:.1f} {u}"
        b /= 1024


def main():
    runs = load_runs()
    if not runs:
        print("no runs yet")
        return

    keys, groups = [], {}
    per_run = {}
    for rid, man, rows in runs:
        per_run[rid] = {}
        for r in rows:
            k = (r["tier"], r["case"])
            if k not in per_run[rid]:
                keys.append(k) if k not in keys else None
            groups[k] = r["group"]
            per_run[rid][k] = r

    def sortkey(k):
        tier, case = k
        g = groups.get(k, "")
        return (TIER_ORDER.index(tier) if tier in TIER_ORDER else 99,
                GROUP_ORDER.index(g) if g in GROUP_ORDER else 99, case)
    keys = sorted(set(keys), key=sortkey)

    L = []
    L.append("# BEAM runtime profile\n")
    L.append("Mean wall time over `reps` repetitions, and peak resident-memory")
    L.append("growth during the call. Every run also stores the RETURN VALUE of")
    L.append("every case under `runs/<id>/results/`; use `compare.py` to check")
    L.append("that an optimisation changed the speed and not the answer.\n")

    L.append("## Runs\n")
    L.append("| run | date | commit | subject | dirty | threads | reps | cases | errors |")
    L.append("|---|---|---|---|---|---:|---:|---:|---:|")
    for rid, man, rows in runs:
        g = man["git"]
        L.append(f"| `{rid}` | {man.get('started','?')} | `{g['short']}` | "
                 f"{g['subject'][:44]} | {'**yes**' if g['dirty'] else 'no'} | "
                 f"{man.get('threads_effective','?')} | {man.get('reps','?')} | "
                 f"{man.get('n_cases','?')} | {man.get('n_errors','?')} |")
    L.append("")

    hdr = [f"{rid.split('__')[0][4:]} `{rid.split('__')[1]}`" for rid, _, _ in runs]

    # ── timing table ──
    L.append("## Wall time\n")
    L.append("| tier | group | case | " + " | ".join(hdr) + " | Δ last vs first |")
    L.append("|---|---|---|" + "---:|" * (len(runs) + 1))
    last_tier = None
    for k in keys:
        tier, case = k
        if tier != last_tier:
            last_tier = tier
        cells, vals = [], []
        for rid, _, _ in runs:
            r = per_run[rid].get(k)
            if not r:
                cells.append("—"); vals.append(None); continue
            if str(r["status"]).startswith("ERROR"):
                cells.append("**err**"); vals.append(None); continue
            cells.append(fmt_time(r["mean_s"]))
            try:
                vals.append(float(r["mean_s"]))
            except (TypeError, ValueError):
                vals.append(None)
        delta = "—"
        fv = next((v for v in vals if v), None)
        lv = next((v for v in reversed(vals) if v), None)
        if fv and lv and len(runs) > 1 and fv != lv:
            sp = fv / lv
            delta = f"**{sp:.2f}× faster**" if sp > 1.02 else (
                f"**{1/sp:.2f}× SLOWER**" if sp < 0.98 else "~")
        L.append(f"| {tier} | {groups.get(k,'')} | `{case}` | "
                 + " | ".join(cells) + f" | {delta} |")
    L.append("")

    # ── memory table ──
    L.append("## Peak resident-memory growth during the call\n")
    L.append("`file_bytes` rows report the size of the saved model on disk "
             "instead.\n")
    L.append("| tier | group | case | " + " | ".join(hdr) + " |")
    L.append("|---|---|---|" + "---:|" * len(runs))
    for k in keys:
        tier, case = k
        cells = []
        for rid, _, _ in runs:
            r = per_run[rid].get(k)
            cells.append(fmt_bytes(r["peak_rss_bytes"]) if r else "—")
        if any(c != "—" for c in cells):
            L.append(f"| {tier} | {groups.get(k,'')} | `{case}` | "
                     + " | ".join(cells) + " |")
    L.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}  ({len(runs)} runs, {len(keys)} rows)")


if __name__ == "__main__":
    main()
