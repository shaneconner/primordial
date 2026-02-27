---
name: run-simulation
description: Run a Primordial simulation with configurable parameters, export clips, and analyze results. Use when the user wants to run experiments, test parameter changes, or generate visualization data.
---

# Running Simulations

## Quick run
```bash
python scripts/run.py --ticks 3000 --population 50
```

## Full options
```bash
python scripts/run.py \
  --ticks 5000 \
  --population 50 \
  --seed 42 \
  --log-interval 100 \
  --snapshot-interval 10 \
  --clip-name my_experiment \
  --output-dir clips
```

## Programmatic usage (for parameter sweeps or analysis)
```python
from primordial.config import SimConfig
from primordial.engine import Engine

config = SimConfig(initial_population=50, seed=42)
# Modify config as needed:
# config.body.brownian_force = 3.0
# config.evolution.body_mutation_rate = 0.1

engine = Engine(config)
engine.initialize()
engine.run(ticks=5000)
engine.save("experiment_name")
```

## Output
- Clips saved to `clips/` directory as JSON
- Stats saved alongside clips
- Clip JSON format is designed for D3.js replay viewer consumption

## Key parameters to tune
- `config.body.brownian_force` — base wandering force (higher = more random movement)
- `config.evolution.body_mutation_rate` — how often body structure mutates
- `config.evolution.brain_mutation_rate` — how often brain weights mutate
- `config.world.plant_spawn_rate` — food availability (affects population capacity)
- `config.evolution.asexual_energy_threshold` — energy needed to reproduce
- `config.world.width/height` — world size (smaller = denser, faster evolution)
