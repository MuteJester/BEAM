import BEAM

model = BEAM.load_pretrained("foundation")
draws = model.sample(50_000, seed=11)

starts_c = sum(s.startswith("C") for s in draws) / len(draws)
ends_fw = sum(s.endswith(("F", "W")) for s in draws) / len(draws)
print(f"start with C:      {starts_c:.1%}")
print(f"end with F or W:   {ends_fw:.1%}")
