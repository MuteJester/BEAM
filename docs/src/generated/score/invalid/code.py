import BEAM

model = BEAM.load_pretrained("foundation")

too_long = "CASS" + "G" * 30 + "YEQYF"     # 39 residues; the model covers up to 27
unknown = "CASSLGQGXEQYF"                 # X is not one of the 20 amino acids
print(model.pseq(too_long), model.pseq(unknown))

try:
    model.explain(unknown)
except ValueError as err:
    print("explain:", err)
