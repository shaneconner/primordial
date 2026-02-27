# Primordial: A Study on Emergent Behaviors and Neuroevolution

## Project Overview
Evolutionary life simulation where organisms with neural network brains and mutable spring-mass bodies compete for resources in a 2D environment. The goal is emergent ecosystems arising from neuroevolution and morphological mutation.

- **Simulation engine**: Python (NumPy), runs locally on CPU/GPU
- **Visualization**: D3.js replay viewer on portfolio site
- **Portfolio writeup**: `C:\Users\shane\shaneconner-site\projects\primordial\`
- **Portfolio site repo**: https://github.com/shaneconner/shaneconner-site

## Architecture
```
primordial/          # Python package
  config.py          # All simulation parameters (dataclasses)
  body.py            # Spring-mass physics (Hooke's law, Verlet integration)
  brain.py           # Feedforward neural net with memory registers
  genome.py          # Genome encoding, mutation, crossover, speciation
  organism.py        # Organism = body + brain + genome
  world.py           # Environment, resources, spatial hashing
  evolution.py       # Reproduction, speciation, immigration
  recorder.py        # State snapshot recording for clip export
  engine.py          # Main simulation loop orchestrating all systems
scripts/
  run.py             # CLI runner
clips/               # Output directory for recorded simulation data (gitignored)
```

## Key Design Decisions
- **Spring-mass bodies**: Nodes (core, bone, muscle, sensor, mouth, fat, armor) connected by springs (bone, muscle, tendon). Organisms learn locomotion by contracting muscles.
- **Neuroevolution**: No backprop. Weights inherited from parents with Gaussian mutation. Evolution is the optimizer.
- **Meta-evolution**: Mutation rates are themselves genome parameters that evolve.
- **Clip-based visualization**: Simulation records state snapshots → export as JSON → D3.js replays on portfolio site.

## Git Commits
- Do NOT include "Co-Authored-By" lines in commit messages
- Keep commit messages concise and descriptive
- Always commit as Shane Conner (shanepatrickconner@gmail.com)

## Running
```bash
python scripts/run.py --ticks 5000 --population 50 --seed 42
```

## Visualization (debug)
```bash
python scripts/visualize.py                      # live simulation
python scripts/visualize.py clips/test_run.json  # replay a clip
```
Requires matplotlib: `pip install matplotlib`

## Performance Notes
- ~18-40 ticks/second depending on population (40-180 organisms)
- Bottleneck is physics (spring force computation) and spatial queries
- Center-of-mass is cached per tick to avoid redundant numpy operations
- Spatial hash grid used for O(1) neighbor lookups

## Current State (Phase 1 MVP)
- Core simulation loop working
- Spring-mass physics, neural nets, mutation, reproduction, speciation all functional
- Organisms evolve, reproduce, speciate (21 species by tick 3000)
- Clip recording and export working

## Next Steps
- Phase 2: Ecosystem depth (photosynthetic nodes, carnivory, decomposition)
- Phase 3: D3.js visualization (replay viewer, population charts, body gallery)
- Phase 4: Polish (phylogenetic tree, brain inspector, video export)
