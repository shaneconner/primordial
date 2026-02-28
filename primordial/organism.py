"""Organism class - combines body, brain, and genome into a living entity."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .body import Body, EdgeType, NodeType
from .brain import Brain

if TYPE_CHECKING:
    from .config import SimConfig
    from .genome import Genome


class Organism:
    """A living organism in the simulation.

    Ties together the physical body, neural network brain, and genome.
    Manages energy, age, sensing, and action execution.
    """

    _next_id = 0

    def __init__(
        self,
        genome: Genome,
        config: SimConfig,
        rng: np.random.Generator,
        position: np.ndarray | None = None,
        energy: float | None = None,
        generation: int = 0,
        parent_id: str | None = None,
        species_id: str | None = None,
    ):
        self.id = f"org_{Organism._next_id:06d}"
        Organism._next_id += 1

        self.genome = genome
        self.config = config
        self.generation = generation
        self.parent_id = parent_id
        self.species_id = species_id or "sp_000"
        self.age = 0
        self.last_reproduce_tick = -config.evolution.reproduce_cooldown

        # Build body from genome
        self.body = self._build_body(genome, config, rng)

        # Place in world
        if position is not None:
            self.body.translate(position - self.body.center_of_mass)

        # Build brain from genome + body morphology
        self.brain = self._build_brain(genome, config)

        # Energy
        if energy is not None:
            self.energy = energy
        else:
            self.energy = self.body.max_energy_capacity * 0.75

        self.alive = True

        # Outputs from last brain step (for recording/inspection)
        self.last_eat_signal = 0.0
        self.last_attack_signal = 0.0
        self.last_reproduce_signal = 0.0

    def _build_body(
        self, genome: Genome, config: SimConfig, rng: np.random.Generator
    ) -> Body:
        """Construct a Body from genome data."""
        node_types = [n["type"] for n in genome.body_nodes]
        positions = np.array([[n["rx"], n["ry"]] for n in genome.body_nodes])
        # Add tiny perturbation to prevent degenerate spring lengths
        positions += rng.normal(0, 0.01, positions.shape)
        edges = [(e["from"], e["to"], e["type"]) for e in genome.body_edges]
        return Body(node_types, positions, edges, config.body)

    def _build_brain(self, genome: Genome, config: SimConfig) -> Brain:
        """Construct a Brain from genome data."""
        n_sensors = self.body.n_sensors
        n_muscles = self.body.n_muscles

        ips = config.brain.inputs_per_sensor
        n_inputs = min(
            (n_sensors * ips) + 2 + n_muscles, config.brain.max_inputs
        )
        n_outputs = min(n_muscles + config.brain.n_action_outputs, config.brain.max_outputs)

        return Brain(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_size=genome.brain_hidden_size,
            activation=genome.brain_activation,
            n_memory=genome.brain_n_memory,
            config=config.brain,
            weights_ih=genome.brain_weights_ih.copy(),
            weights_ho=genome.brain_weights_ho.copy(),
            bias_h=genome.brain_bias_h.copy(),
            bias_o=genome.brain_bias_o.copy(),
        )

    @property
    def position(self) -> np.ndarray:
        """World position (center of mass)."""
        return self.body.center_of_mass

    @property
    def bounding_radius(self) -> float:
        return self.body.get_bounding_radius()

    def sense(
        self,
        nearby_food: list[tuple[float, float, float, str]],
        nearby_organisms: list[tuple[float, float, float, bool]],
    ) -> np.ndarray:
        """Gather sensory inputs from the environment.

        Args:
            nearby_food: List of (x, y, energy, type) for food within sensor range.
            nearby_organisms: List of (x, y, mass, same_species) for organisms in range.

        Returns:
            Input vector for the brain.
        """
        inputs = []
        com = self.body.center_of_mass
        com_x, com_y = float(com[0]), float(com[1])
        n_sensors = self.body.n_sensors
        ips = self.config.brain.inputs_per_sensor

        # Sensor range/FOV scaling with sensor count
        sensor_range = self.config.body.sensor_range
        sensor_fov = self.config.body.sensor_fov
        if n_sensors > 1:
            # More sensors = wider total FOV coverage
            sensor_fov = sensor_fov * n_sensors
            sensor_fov = min(sensor_fov, math.pi)  # cap at 180 degrees half-angle
            # More sensors = slightly better range
            sensor_range = sensor_range * (1.0 + 0.15 * (n_sensors - 1))
        per_sensor_fov = sensor_fov  # each sensor covers the full FOV band

        # Per-sensor inputs (use math module for scalar operations - faster than numpy)
        for sensor_idx in self.body.sensor_indices:
            sx = float(self.body.positions[sensor_idx][0])
            sy = float(self.body.positions[sensor_idx][1])
            sensor_dir = math.atan2(sy - com_y, sx - com_x)

            # Find nearest food in this sensor's cone
            food_dist = sensor_range
            food_angle = 0.0
            food_type_val = 0.0
            for food_item in nearby_food:
                fx, fy = food_item[0], food_item[1]
                dx, dy = fx - sx, fy - sy
                d2 = dx * dx + dy * dy
                if d2 < food_dist * food_dist:
                    angle_to = math.atan2(dy, dx)
                    angle_diff = abs(_wrap_angle(angle_to - sensor_dir))
                    if angle_diff < per_sensor_fov:
                        food_dist = math.sqrt(d2)
                        food_angle = _wrap_angle(angle_to - sensor_dir) / math.pi
                        if len(food_item) > 3:
                            food_type_val = 1.0 if food_item[3] == "plant" else -1.0

            # Find nearest organism in this sensor's cone
            org_dist = sensor_range
            org_type = 0.0
            org_size = 0.0
            for ox, oy, omass, same_sp in nearby_organisms:
                dx, dy = ox - sx, oy - sy
                d2 = dx * dx + dy * dy
                if d2 < org_dist * org_dist:
                    angle_to = math.atan2(dy, dx)
                    angle_diff = abs(_wrap_angle(angle_to - sensor_dir))
                    if angle_diff < per_sensor_fov:
                        org_dist = math.sqrt(d2)
                        org_type = 1.0 if same_sp else -1.0
                        org_size = min(omass / max(self.body.total_mass, 0.1), 2.0) - 1.0

            inputs.append(food_dist / sensor_range)
            if ips >= 6:
                inputs.append(food_angle)
                inputs.append(food_type_val)
            inputs.append(org_dist / sensor_range)
            if ips >= 6:
                inputs.append(org_size)
            inputs.append(org_type)

        # Energy and age (normalized)
        inputs.append(self.energy / self.body.max_energy_capacity)
        max_age = self.config.evolution.max_age
        inputs.append(min(self.age / max_age, 1.0))

        # Proprioception: current muscle lengths
        muscle_lengths = self.body.get_muscle_lengths()
        inputs.extend(muscle_lengths.tolist())

        return np.array(inputs, dtype=np.float64)

    def think(self, inputs: np.ndarray) -> None:
        """Run brain forward pass and apply outputs to body."""
        outputs = self.brain.forward(inputs)

        n_muscles = self.body.n_muscles
        n_actions = self.config.brain.n_action_outputs

        # Muscle targets
        if n_muscles > 0:
            self.body.set_muscle_targets(outputs[:n_muscles])

        # Eat signal
        self.last_eat_signal = float(outputs[n_muscles]) if len(outputs) > n_muscles else 0.0

        if n_actions >= 3:
            # Part 2: separate attack and reproduce signals
            self.last_attack_signal = float(outputs[n_muscles + 1]) if len(outputs) > n_muscles + 1 else 0.0
            self.last_reproduce_signal = float(outputs[n_muscles + 2]) if len(outputs) > n_muscles + 2 else 0.0
        else:
            # Part 1: eat signal doubles as attack, reproduce is second output
            self.last_attack_signal = self.last_eat_signal
            self.last_reproduce_signal = (
                float(outputs[n_muscles + 1]) if len(outputs) > n_muscles + 1 else 0.0
            )

    def step_physics(self, rng: np.random.Generator) -> None:
        """Advance body physics one tick with Brownian motion."""
        # Apply random force to core node (thermal wandering)
        bf = self.config.body.brownian_force
        if bf > 0:
            # Size-based force scaling: larger bodies get proportionally stronger forces
            force_scale = 1.0
            if self.config.body.size_force_scaling:
                force_scale = (self.body.n_nodes / 3.0) ** 0.5
            random_force = rng.normal(0, bf * force_scale, 2)
            self.body.velocities[0] += random_force / self.body.masses[0]
        self.body.step_full()

    def metabolize(self) -> None:
        """Deduct metabolic energy cost. Die if energy depleted or too old."""
        cost = self.body.energy_cost_per_tick

        # Senescence: metabolic cost rises in old age
        max_age = self.config.evolution.max_age
        senescence_start = int(max_age * self.config.evolution.senescence_age)
        if self.age >= senescence_start:
            progress = (self.age - senescence_start) / max(1, max_age - senescence_start)
            multiplier = 1.0 + progress * (self.config.evolution.senescence_max_multiplier - 1.0)
            cost *= multiplier

        self.energy -= cost
        self.age += 1

        if self.age >= max_age or self.energy <= 0:
            self.energy = 0
            self.alive = False

    def eat(self, food_energy: float) -> None:
        """Consume food energy."""
        gained = food_energy * self.config.evolution.eat_efficiency
        self.energy = min(self.energy + gained, self.body.max_energy_capacity)

    def take_damage(self, raw_damage: float) -> float:
        """Take damage (reduced by armor). Returns actual damage dealt."""
        reduction = min(self.body.armor_value, 0.9)  # cap at 90% reduction
        actual = raw_damage * (1.0 - reduction)
        self.energy -= actual
        if self.energy <= 0:
            self.energy = 0
            self.alive = False
        return actual

    def can_reproduce(self, current_tick: int, asexual: bool = True) -> bool:
        """Check if organism meets reproduction criteria."""
        threshold = (
            self.config.evolution.asexual_energy_threshold
            if asexual
            else self.config.evolution.sexual_energy_threshold
        )
        # Fat nodes lower reproduction threshold
        fat_bonus = self.config.body.fat_repro_bonus * len(self.body.fat_indices)
        threshold = max(0.2, threshold - fat_bonus)

        return (
            self.alive
            and self.energy >= self.body.max_energy_capacity * threshold
            and self.age >= self.config.evolution.min_reproduce_age
            and (current_tick - self.last_reproduce_tick)
            >= self.config.evolution.reproduce_cooldown
            and self.last_reproduce_signal > 0.5
        )

    def wants_to_eat(self) -> bool:
        return self.last_eat_signal > 0.5

    def wants_to_attack(self) -> bool:
        return self.last_attack_signal > 0.5

    def to_dict(self) -> dict:
        """Serialize for recording."""
        return {
            "id": self.id,
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "energy": round(self.energy, 2),
            "age": self.age,
            "generation": self.generation,
            "species": self.species_id,
            "alive": self.alive,
            "n_nodes": self.body.n_nodes,
            "n_muscles": self.body.n_muscles,
            "nodes": [
                {
                    "x": round(float(self.body.positions[i][0]), 2),
                    "y": round(float(self.body.positions[i][1]), 2),
                    "type": int(self.body.node_types[i]),
                }
                for i in range(self.body.n_nodes)
            ],
            "edges": [
                {
                    "from": int(self.body.edge_from[i]),
                    "to": int(self.body.edge_to[i]),
                    "type": int(self.body.edge_types[i]),
                    "contraction": round(float(self.body.muscle_targets[i]), 3)
                    if self.body.edge_types[i] == int(EdgeType.MUSCLE)
                    else None,
                }
                for i in range(self.body.n_edges)
            ],
        }


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
