"""Optional regression against the frozen Figure 5 mouse calculations.

Requires local manuscript fixtures, which are deliberately not package data.
Run under verification/bounded. This reads existing models/tables and writes
only the requested JSON report; it never refits models or changes figures.
Independent sequence-enumeration tests live in tests/python/analytics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import BEAM
from BEAM import BeamModel, BeamQuery
from BEAM.analytics.distance import compare, group_overlaps


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label, *, atol=2e-11):
    expected = float(expected)
    error = abs(actual - expected)
    if not math.isfinite(actual) or error > atol:
        raise AssertionError(f"{label}: {actual!r} != {expected!r} (error {error:g})")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, default=Path(__file__).resolve().parents[1]
                        / ".private/likelihood_studies")
    parser.add_argument("--mice", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--top-k", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    BEAM.set_threads(args.threads)
    BEAM.set_memory_budget(2 * 2**30)
    source = args.studies / "57_published_repertoire_benchmarks"
    attribution = args.studies / "60_distance_attribution"
    expected_totals = {int(row["mouse"]): row for row in rows(attribution / "F_mouse_summary.csv")}
    cells = rows(attribution / "F_mouse_cells.csv")
    probabilities = rows(attribution / "G_probabilities.csv")
    contributions = rows(attribution / "G_contributions.csv")
    selected = json.loads((attribution / "selected_motifs.json").read_text())["motifs"]
    report = {"threads": args.threads, "top_k": args.top_k, "mice": [],
              "extension_sha256": hashlib.sha256(Path(BEAM._cbeam.__file__).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source": str(source), "attribution": str(attribution)}
    for mouse in args.mice:
        print(f"Checking frozen mouse {mouse} ...", flush=True)
        models, hashes = [], {}
        for population in ("DN", "CD4"):
            path = source / "models" / f"{population}_{mouse}_B.beam"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata = json.loads(path.with_suffix(".json").read_text())
            if digest != metadata["model_sha256"]:
                raise AssertionError(f"Changed model fixture: {path}")
            hashes[population] = digest
            models.append(BeamModel.from_file(path))
        start = time.perf_counter()
        result = compare(*models, top_k=args.top_k)
        seconds = time.perf_counter() - start
        expected = expected_totals[mouse]
        errors = {
            "distance": close(result.distance, expected["distance"], "distance"),
            "squared_distance": close(result.squared_distance, expected["squared_distance"], "D squared"),
            "length": close(result.length_squared_distance, expected["length_contribution"], "length"),
            "patterns": close(result.pattern_squared_distance, expected["within_contribution"], "patterns"),
        }
        actual_cells = {(row.length, -1): row.contribution for row in result.lengths}
        actual_cells.update({(row.length, row.start): row.contribution for row in result.windows})
        expected_cells = {(int(row["length"]), int(row["position"])): float(row["contribution"])
                          for row in cells if int(row["mouse"]) == mouse}
        if set(actual_cells) != set(expected_cells):
            raise AssertionError("Window/length inventory differs from the frozen figure")
        errors["max_cell"] = max(close(actual_cells[key], value, f"cell {key}")
                                 for key, value in expected_cells.items())
        by_coordinate = {(row.length, row.start, row.word): row for row in result.patterns}
        pattern_checks = []
        for motif in selected:
            key = (motif["length"], motif["position"], motif["word"])
            found = by_coordinate.get(key)
            saved = [row for row in probabilities if int(row["mouse"]) == mouse
                     and row["motif_id"] == motif["motif_id"]]
            values = {}
            for population, model in zip(("DN", "CD4"), models):
                query = BeamQuery(model)
                event = query.condition(key[0]).run(key[1], key[2])
                probability = query.probability(event)
                saved_probability = next(row for row in saved if row["population"] == population)
                close(probability, saved_probability["model_probability"], f"{population} {key}")
                values[population] = probability
            coefficient = math.exp((models[0].log_length_prior(key[0])
                                    + models[1].log_length_prior(key[0])) / 2) / (2 * (key[0] - 4))
            value = coefficient * (math.sqrt(values["DN"]) - math.sqrt(values["CD4"])) ** 2
            saved_contribution = next(row for row in contributions if int(row["mouse"]) == mouse
                                      and row["motif_id"] == motif["motif_id"])
            close(value, saved_contribution["contribution"], f"selected contribution {key}")
            if found is not None:
                close(found.conditional_p, values["DN"], f"native P {key}")
                close(found.conditional_q, values["CD4"], f"native Q {key}")
                close(found.contribution, value, f"native contribution {key}")
            pattern_checks.append({"word": key[2], "start": key[1], "in_top_k": found is not None,
                                   "contribution": value, "conditional_probabilities": values})
        grouped = group_overlaps(result)
        grouped_sum = math.fsum(group.contribution for group in grouped.groups)
        close(grouped_sum, result.reported_pattern_squared_distance, "grouped sum")
        close(result.length_squared_distance + grouped_sum + result.remainder,
              result.squared_distance, "complete grouped reconstruction")
        entry = {"mouse": mouse, "model_sha256": hashes, "seconds": seconds,
                 "distance": result.distance, "squared_distance": result.squared_distance,
                 "errors": errors, "checked_cells": len(actual_cells),
                 "reported_patterns": len(result.patterns), "groups": len(grouped.groups),
                 "selected_patterns": pattern_checks, "scratch_bytes": result.scratch_bytes}
        report["mice"].append(entry)
        print(json.dumps(entry), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
