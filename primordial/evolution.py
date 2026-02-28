"""Reproduction, speciation, and population management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .genome import Genome, crossover, genome_distance, mutate
from .organism import Organism

if TYPE_CHECKING:
    from .config import SimConfig
    from .world import World


_next_species_id = 1


def handle_reproduction(world: World, config: SimConfig, rng: np.random.Generator) -> list[Organism]:
    """Process reproduction for all organisms. Returns list of new offspring."""
    global _next_species_id

    offspring: list[Organism] = []

    # Sexual reproduction (if enabled)
    if config.evolution.enable_sexual_reproduction:
        sexual_offspring = _handle_sexual_reproduction(world, config, rng)
        offspring.extend(sexual_offspring)

    # Asexual reproduction
    for org in list(world.organisms):
        if not org.can_reproduce(world.tick, asexual=True):
            continue
        if len(world.organisms) + len(offspring) >= config.max_population:
            break

        # Pay energy cost
        cost = org.energy * config.evolution.asexual_energy_cost
        org.energy -= cost

        # Mutate genome
        child_genome = mutate(org.genome, config.evolution, config.brain, rng, body_config=config.body)

        # Spawn near parent
        offset = rng.normal(0, 5.0, 2)
        child_pos = org.position + offset

        child = Organism(
            genome=child_genome,
            config=config,
            rng=rng,
            position=child_pos,
            energy=cost * 0.8,  # some energy lost in reproduction
            generation=org.generation + 1,
            parent_id=org.id,
            species_id=org.species_id,
        )

        # Check if child has speciated (genome too different from parent)
        dist = genome_distance(org.genome, child_genome, config.evolution)
        if dist > config.evolution.speciation_threshold:
            child.species_id = f"sp_{_next_species_id:03d}"
            _next_species_id += 1

        offspring.append(child)
        org.last_reproduce_tick = world.tick

    return offspring


def _handle_sexual_reproduction(
    world: World, config: SimConfig, rng: np.random.Generator
) -> list[Organism]:
    """Process sexual reproduction between nearby, genetically similar organisms."""
    global _next_species_id

    offspring: list[Organism] = []
    mated: set[str] = set()  # track who mated this tick
    proximity = config.evolution.sexual_proximity

    for org in list(world.organisms):
        if org.id in mated:
            continue
        if not org.can_reproduce(world.tick, asexual=False):
            continue
        if len(world.organisms) + len(offspring) >= config.max_population:
            break

        # Find nearby potential mates
        pos = org.position
        nearby = world.organism_hash.query(pos[0], pos[1], proximity)

        for candidate in nearby:
            if candidate is org or not candidate.alive:
                continue
            if candidate.id in mated:
                continue
            if not candidate.can_reproduce(world.tick, asexual=False):
                continue

            # Check genetic similarity (must be same or close species)
            dist = genome_distance(org.genome, candidate.genome, config.evolution)
            if dist > config.evolution.speciation_threshold * 1.5:
                continue

            # Check physical proximity
            cpos = candidate.position
            dx = pos[0] - cpos[0]
            dy = pos[1] - cpos[1]
            if dx * dx + dy * dy > proximity * proximity:
                continue

            # Mate! Crossover + mutation
            child_genome = crossover(org.genome, candidate.genome, config.brain, rng)
            child_genome = mutate(child_genome, config.evolution, config.brain, rng, body_config=config.body)

            # Energy cost split between parents
            cost_each = config.evolution.sexual_energy_cost
            org_cost = org.energy * cost_each
            mate_cost = candidate.energy * cost_each
            org.energy -= org_cost
            candidate.energy -= mate_cost

            # Spawn between parents
            child_pos = (pos + cpos) / 2.0 + rng.normal(0, 3.0, 2)
            child_energy = (org_cost + mate_cost) * 0.8

            child = Organism(
                genome=child_genome,
                config=config,
                rng=rng,
                position=child_pos,
                energy=child_energy,
                generation=max(org.generation, candidate.generation) + 1,
                parent_id=org.id,
                species_id=org.species_id,
            )

            # Check speciation
            parent_dist = genome_distance(org.genome, child_genome, config.evolution)
            if parent_dist > config.evolution.speciation_threshold:
                child.species_id = f"sp_{_next_species_id:03d}"
                _next_species_id += 1

            offspring.append(child)
            mated.add(org.id)
            mated.add(candidate.id)
            org.last_reproduce_tick = world.tick
            candidate.last_reproduce_tick = world.tick
            break  # one mate per organism per tick

    return offspring


def spawn_immigrants(
    world: World, config: SimConfig, rng: np.random.Generator
) -> list[Organism]:
    """Spawn random organisms if population is critically low."""
    from .genome import create_default_genome

    immigrants = []
    deficit = config.world.min_population - len(world.organisms)

    for _ in range(max(0, deficit)):
        genome = create_default_genome(
            config.body, config.brain, config.evolution, rng
        )
        pos = np.array([
            rng.uniform(0, config.world.width),
            rng.uniform(0, config.world.height),
        ])
        org = Organism(
            genome=genome,
            config=config,
            rng=rng,
            position=pos,
            generation=0,
        )
        immigrants.append(org)

    return immigrants
