import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)

# A cohort of 10 people, each sampled to 100,000 cells
depths = [100_000] * 10
private = occupancy.private(depths).value
all_distinct = occupancy.richness(sum(depths)).value

print(f"distinct sequences across the cohort: {all_distinct:,.0f}")
print(f"seen in exactly one person:           {private:,.0f} ({private / all_distinct:.1%})")
