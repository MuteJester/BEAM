from BEAM import BeamModel

model.save("donor_A")                          # writes donor_A.beam
again = BeamModel.from_file("donor_A.beam")

print(again)
print(again.pseq("CASSLGQGYEQYF") == model.pseq("CASSLGQGYEQYF"))
