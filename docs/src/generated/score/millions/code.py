import numpy as np
import BEAM

model = BEAM.load_pretrained("foundation")

# A large batch: 200,000 sequences, here drawn from the model itself.
cdr3s = model.sample(200_000, seed=7)
logp = model.log_pseq_batch(cdr3s)

print(f"scored {len(logp):,} sequences")
print(f"median probability:     {np.exp(np.median(logp)):.2e}")
print(f"5% are rarer than:      {np.exp(np.percentile(logp, 5)):.2e}")
