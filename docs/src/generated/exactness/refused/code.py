from BEAM import BeamModel, Mixture

group = Mixture([donor_a, donor_b])            # pick a donor, then a cell
print(f"Hill number q=2: {group.hill_number(2.0):.6g}")

try:
    group.hill_number(1.0)                     # no closed form at q = 1
except Exception as err:
    print(f"q=1 declined: {type(err).__name__}")

bounds = group.entropy_bounds()                # guaranteed bounds instead
print(bounds)
