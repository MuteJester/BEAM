#!/usr/bin/env bash
# Isolated A/B of the working tree against HEAD, on a chosen case list.
#
# The whole-run comparison is too noisy to trust for individual cases: this
# machine is a desktop, and a single VS Code indexing burst has already
# produced a "27x regression" that a controlled re-measurement showed was a
# 1.06x improvement. This stashes, measures, restores, measures, and reports
# with the combined standard deviation so nothing is called that cannot be.
#
#   tools/runtime_profile/ab.sh <label> <reps> <case,case,...> [tiers]
set -euo pipefail
cd "$(dirname "$0")/../.."

LABEL=${1:?label}; REPS=${2:?reps}; CASES=${3:?cases}; TIERS=${4:-tiny,small}
V=.venv/bin/python
PIP=.venv/bin/pip

rm -rf tools/runtime_profile/runs/*"${LABEL}"old tools/runtime_profile/runs/*"${LABEL}"new

restore() { git stash list | grep -q . && git stash pop -q || true; }
trap restore EXIT

echo "== stashing working tree =="
git stash -q
make -j16 >/dev/null 2>&1
$PIP install -e . --no-build-isolation --quiet 2>&1 | tail -1
echo "== measuring HEAD =="
$V tools/runtime_profile/profile.py --tiers "$TIERS" --reps "$REPS" \
   --only "$CASES" --label "${LABEL}old" >/dev/null 2>&1

echo "== restoring working tree =="
git stash pop -q
trap - EXIT
make -j16 >/dev/null 2>&1
$PIP install -e . --no-build-isolation --quiet 2>&1 | tail -1
echo "== measuring working tree =="
$V tools/runtime_profile/profile.py --tiers "$TIERS" --reps "$REPS" \
   --only "$CASES" --label "${LABEL}new" >/dev/null 2>&1

$V - "$LABEL" <<'PY'
import csv, glob, sys
lab = sys.argv[1]
def load(t):
    p = sorted(glob.glob(f'tools/runtime_profile/runs/*{t}/timings.csv'))[0]
    return {(r['tier'], r['case']): r for r in csv.DictReader(open(p))}
O, N = load(lab + 'old'), load(lab + 'new')
print(f"\n{'tier/case':<30}{'HEAD':>18}{'working':>18}{'ratio':>8}  verdict")
for k in sorted(O):
    if k[1] == 'file_bytes' or k not in N:
        continue
    a, b = float(O[k]['mean_s']), float(N[k]['mean_s'])
    if a <= 0 or b <= 0:
        continue
    sa, sb = float(O[k]['std_s']), float(N[k]['std_s'])
    noise = 2 * ((sa * sa + sb * sb) ** 0.5)
    v = "same" if abs(a - b) <= noise else ("FASTER" if a > b else "**SLOWER**")
    print(f"{k[0] + '/' + k[1]:<30}{a*1e3:>11.2f}±{sa*1e3:<6.2f}"
          f"{b*1e3:>11.2f}±{sb*1e3:<6.2f}{a/b:>7.2f}x  {v}")
PY
