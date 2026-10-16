import pandas as pd
from BEAM import BeamModel
# --- hide
import BEAM
from collections import Counter
cells = BEAM.load_pretrained("foundation").sample(50_000, seed=21)
counts = Counter(cells)
pd.DataFrame({"junction_aa": list(counts), "duplicate_count": list(counts.values()),
              "v_call": "TRBV7-9*01"}).to_csv("repertoire.tsv", sep="\t", index=False)
# --- show

table = pd.read_csv("repertoire.tsv", sep="\t")      # an AIRR rearrangement table
model = BeamModel.from_dataframe(table, sequence_column="junction_aa",
                                 abundance_column="duplicate_count")
print(model)
print(f"sequences skipped: {model.n_skipped}")
