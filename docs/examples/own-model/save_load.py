from BEAM import BeamModel
# --- hide
import BEAM
model = BeamModel(BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation").sample(20_000, seed=5))
# --- show

model.save("donor_A")                          # writes donor_A.beam
again = BeamModel.from_file("donor_A.beam")

print(again)
print(again.pseq("CASSLGQGYEQYF") == model.pseq("CASSLGQGYEQYF"))
