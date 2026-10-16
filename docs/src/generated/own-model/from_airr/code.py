import pandas as pd
from BEAM import BeamModel

table = pd.read_csv("repertoire.tsv", sep="\t")      # an AIRR rearrangement table
model = BeamModel.from_dataframe(table, sequence_column="junction_aa",
                                 abundance_column="duplicate_count")
print(model)
print(f"sequences skipped: {model.n_skipped}")
