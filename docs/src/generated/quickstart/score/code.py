import BEAM

model = BEAM.load_pretrained("foundation")

for cdr3 in ["CASSLGQGYEQYF", "CASSPGTGELFF", "CWWWWWWWWF"]:
    print(f"{cdr3:15} {model.pseq(cdr3):.3g}")
