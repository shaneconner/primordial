# Primordial

**A Study on Emergent Behaviors and Neuroevolution**

Evolutionary life simulation where organisms with neural network brains and mutable spring-mass bodies compete for resources in a 2D environment.

## What It Does

Organisms are dropped into a world with food and nothing else. They have bodies made of springs and masses, brains made of neural networks, and both are encoded in a genome that mutates when they reproduce. The only optimization signal is survival.

Over generations, organisms evolve locomotion gaits, feeding strategies, and body structures. Species diverge. Ecosystems form.

## Architecture

- **Spring-mass bodies**: Nodes (core, bone, muscle, sensor, mouth, fat, armor) connected by springs with Hooke's law physics
- **Feedforward neural nets**: Sensor inputs → hidden layer → muscle/action outputs. No backprop, weights inherited with mutation
- **Neuroevolution**: Genome encodes body plan + brain weights + meta-parameters. All three evolve
- **Spatial hashing**: O(1) neighbor queries for sensing, eating, and combat

## Running

```bash
pip install numpy orjson
python scripts/run.py --ticks 5000 --population 50
```

Output clips are saved to `clips/` as JSON, designed for D3.js replay visualization.

## Portfolio Writeup

See the full writeup with interactive visualizations at [shaneconner.com/projects/primordial](https://shaneconner.com/projects/primordial).
