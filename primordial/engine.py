"""Main simulation loop - orchestrates all subsystems."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .config import SimConfig
from .evolution import handle_reproduction, spawn_immigrants
from .genome import create_default_genome
from .organism import Organism
from .recorder import Recorder
from .world import World


class Engine:
    """Simulation engine that runs the evolutionary life simulation.

    Orchestrates:
    1. Sensing - organisms perceive environment
    2. Thinking - neural nets produce actions
    3. Physics - bodies move via spring-mass dynamics
    4. Interaction - eating, combat, resource consumption
    5. Metabolism - energy costs, death
    6. Reproduction - offspring creation
    7. Environment - resource spawning, decay
    8. Recording - state snapshots

    Part 4 additions:
    - Day/night cycle update
    - Organism repulsion (after physics)
    - Signal broadcasts, sharing, group bonuses, hazards (after combat)
    """

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.world = World(self.config, self.rng)
        self.recorder = Recorder(self.config.recorder)

        self.running = False
        self._start_time = 0.0

    def initialize(self) -> None:
        """Set up initial world state with resources and organisms."""
        # Spawn resources
        self.world.spawn_initial_resources()

        # Spawn initial population
        for _ in range(self.config.initial_population):
            genome = create_default_genome(
                self.config.body,
                self.config.brain,
                self.config.evolution,
                self.rng,
            )
            pos = np.array([
                self.rng.uniform(0, self.config.world.width),
                self.rng.uniform(0, self.config.world.height),
            ])
            org = Organism(
                genome=genome,
                config=self.config,
                rng=self.rng,
                position=pos,
            )
            self.world.add_organism(org)

    def step(self) -> None:
        """Execute one simulation tick."""
        self.world.tick += 1

        # 1. Rebuild spatial hashes
        self.world.rebuild_spatial_hashes()

        # Part 4: Update day/night cycle
        self.world.step_day_night()
        light_level = self.world.get_light_level()

        # 2. Sensing and thinking (per organism)
        enable_signals = self.config.body.enable_signals
        enable_terrain = self.config.body.enable_terrain
        signal_range = self.config.body.signal_range

        for org in self.world.organisms:
            if not org.alive:
                continue

            pos = org.position
            sensor_range = self.config.body.sensor_range

            nearby_food = self.world.get_nearby_food(pos[0], pos[1], sensor_range)
            nearby_orgs = self.world.get_nearby_organisms(
                pos[0], pos[1], sensor_range, exclude=org
            )

            # Part 4: Additional sense data
            nearby_signals = None
            terrain_type = 0
            if enable_signals:
                nearby_signals = self.world.get_nearby_signals(
                    pos[0], pos[1], signal_range
                )
            if enable_terrain:
                terrain_type = self.world.get_terrain_at(pos[0], pos[1])

            inputs = org.sense(
                nearby_food, nearby_orgs,
                nearby_signals=nearby_signals,
                terrain_type=terrain_type,
                light_level=light_level,
            )
            org.think(inputs)

        # 3. Physics
        for org in self.world.organisms:
            if org.alive:
                org.step_physics(self.rng)

        # 4. Wrap positions (toroidal world)
        self.world.wrap_positions()

        # 4.5 Part 4: Organism repulsion (after physics, before eating)
        self.world.handle_organism_repulsion()

        # 5. Rebuild hashes after movement
        self.world.rebuild_spatial_hashes()

        # 6. Interactions: eating and combat
        self.world.handle_eating()
        self.world.handle_combat()

        # 6.5 Part 4: Signals, sharing, group bonuses, hazards
        self.world.handle_signals()
        self.world.handle_sharing()
        self.world.handle_group_bonuses()
        self.world.handle_hazards()
        self.world.step_hazards()

        # 7. Metabolism and death
        for org in self.world.organisms:
            if org.alive:
                org.metabolize()
        self.world.handle_deaths()

        # 8. Reproduction
        offspring = handle_reproduction(self.world, self.config, self.rng)
        for child in offspring:
            self.world.add_organism(child)

        # 9. Immigration (anti-extinction)
        immigrants = spawn_immigrants(self.world, self.config, self.rng)
        for imm in immigrants:
            self.world.add_organism(imm)

        # 10. Resource dynamics
        self.world.step_resources()

        # 11. Recording
        if self.recorder.should_record(self.world.tick):
            self.recorder.record(self.world)

    def run(self, ticks: int | None = None, log_interval: int = 100) -> None:
        """Run the simulation for a number of ticks.

        Args:
            ticks: Number of ticks to run. None = use config.max_ticks.
            log_interval: Print status every N ticks.
        """
        max_ticks = ticks or self.config.max_ticks
        self.running = True
        self._start_time = time.time()

        print(f"Starting simulation: {self.config.initial_population} organisms, "
              f"{max_ticks} ticks")
        print("-" * 60)

        try:
            for i in range(max_ticks):
                if not self.running:
                    break

                self.step()

                if self.world.tick % log_interval == 0:
                    self._log_status()

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            self.running = False
            elapsed = time.time() - self._start_time
            tps = self.world.tick / elapsed if elapsed > 0 else 0
            print(f"\nSimulation complete: {self.world.tick} ticks in {elapsed:.1f}s "
                  f"({tps:.0f} ticks/s)")

    def _log_status(self) -> None:
        """Print current simulation status."""
        stats = self.world.get_stats()
        elapsed = time.time() - self._start_time
        tps = self.world.tick / elapsed if elapsed > 0 else 0

        print(
            f"Tick {self.world.tick:>6d} | "
            f"Pop: {stats['population']:>3d} | "
            f"Species: {stats['species_count']:>2d} | "
            f"Avg Energy: {stats['avg_energy']:>5.1f} | "
            f"Avg Nodes: {stats['avg_nodes']:>4.1f} | "
            f"Gen: {stats['max_generation']:>3d} | "
            f"Food: {stats['resource_count']:>4d} | "
            f"{tps:.0f} t/s"
        )

    def save(self, clip_name: str | None = None) -> tuple[str, str]:
        """Save recorded data.

        Returns:
            Tuple of (clip_path, stats_path).
        """
        clip_path = self.recorder.save_clip(clip_name)
        stats_path = self.recorder.save_stats()
        print(f"Saved clip to: {clip_path}")
        print(f"Saved stats to: {stats_path}")
        return clip_path, stats_path
