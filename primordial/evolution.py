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
        child_genome = mutate(org.genome, config.evolution, config.brain, rng)

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
