"""Environment simulation - resources, spatial hashing, interactions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig
    from .organism import Organism


@dataclass
class Resource:
    x: float
    y: float
    energy: float
    resource_type: str  # "plant", "meat", "nutrient"
    decay_rate: float = 0.0
    age: int = 0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "energy": round(self.energy, 1),
            "type": self.resource_type,
        }


class SpatialHash:
    """Spatial hash grid for efficient neighbor queries."""

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list] = {}

    def clear(self) -> None:
        self.grid.clear()

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size)))

    def insert(self, item: object, x: float, y: float) -> None:
        key = self._key(x, y)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(item)

    def query(self, x: float, y: float, radius: float) -> list:
        """Get all items within radius of (x, y)."""
        results = []
        cell_radius = int(math.ceil(radius / self.cell_size))
        cx, cy = self._key(x, y)

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                key = (cx + dx, cy + dy)
                if key in self.grid:
                    results.extend(self.grid[key])

        return results


class World:
    """The simulation environment.

    Manages resources, spatial organization, and organism-environment interactions.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.width = config.world.width
        self.height = config.world.height

        self.resources: list[Resource] = []
        self.organisms: list[Organism] = []
        self.dead_this_tick: list[Organism] = []

        self.organism_hash = SpatialHash(config.world.spatial_cell_size)
        self.resource_hash = SpatialHash(config.world.spatial_cell_size)

        self.tick = 0

        # Nutrient density map (simplified: grid of accumulated nutrients)
        nutrient_cells = max(1, int(self.width / config.world.spatial_cell_size))
        self.nutrient_grid = np.zeros((nutrient_cells, nutrient_cells))

        # Part 3: Environmental dynamics state
        self._food_shock_remaining = 0  # ticks remaining in current shock
        self._gradient_cx = config.body.gradient_center_x
        self._gradient_cy = config.body.gradient_center_y

    def spawn_initial_resources(self, count: int | None = None) -> None:
        if count is None:
            count = self.config.world.initial_food_count
        """Scatter initial plant resources across the world."""
        for _ in range(count):
            self.resources.append(Resource(
                x=self.rng.uniform(0, self.width),
                y=self.rng.uniform(0, self.height),
                energy=self.config.world.plant_energy,
                resource_type="plant",
            ))

    def step_resources(self) -> None:
        """Spawn new resources, decay old ones."""
        # Spawn plants (skip if at resource cap)
        if len(self.resources) >= self.config.world.max_resources:
            spawn_rate = 0.0
        else:
            spawn_rate = self.config.world.plant_spawn_rate

        # Part 3: Seasonal modulation
        if self.config.body.enable_seasons:
            phase = (self.tick % self.config.body.season_length) / self.config.body.season_length
            seasonal_mult = 1.0 + self.config.body.season_food_amplitude * math.sin(2 * math.pi * phase)
            spawn_rate *= seasonal_mult

        # Part 3: Food shock
        if self.config.body.enable_food_shocks:
            if self._food_shock_remaining > 0:
                spawn_rate *= self.config.body.food_shock_severity
                self._food_shock_remaining -= 1
            elif self.rng.random() < self.config.body.food_shock_probability:
                self._food_shock_remaining = self.config.body.food_shock_duration

        # Boost if population is low
        if len(self.organisms) < self.config.world.food_boost_threshold:
            spawn_rate *= 3.0
        spawn_rate = max(spawn_rate, self.config.world.plant_min_spawn_rate)

        n_spawn = int(spawn_rate)
        if self.rng.random() < (spawn_rate - n_spawn):
            n_spawn += 1

        # Part 3: Spatial gradient drift
        if self.config.body.enable_spatial_gradient:
            self._gradient_cx += self.rng.normal(0, self.config.body.gradient_shift_rate)
            self._gradient_cy += self.rng.normal(0, self.config.body.gradient_shift_rate)
            self._gradient_cx = float(np.clip(self._gradient_cx, 0.2, 0.8))
            self._gradient_cy = float(np.clip(self._gradient_cy, 0.2, 0.8))

        for _ in range(n_spawn):
            # Bias spawn toward nutrient-rich areas
            if self.rng.random() < 0.3 and np.max(self.nutrient_grid) > 0:
                # Weighted random cell by nutrient density
                flat = self.nutrient_grid.ravel()
                total = flat.sum()
                if total > 0:
                    idx = self.rng.choice(len(flat), p=flat / total)
                    rows = self.nutrient_grid.shape[0]
                    row, col = divmod(idx, rows)
                    cell_w = self.width / self.nutrient_grid.shape[1]
                    cell_h = self.height / self.nutrient_grid.shape[0]
                    x = col * cell_w + self.rng.uniform(0, cell_w)
                    y = row * cell_h + self.rng.uniform(0, cell_h)
                else:
                    x = self.rng.uniform(0, self.width)
                    y = self.rng.uniform(0, self.height)
            elif self.config.body.enable_spatial_gradient and self.rng.random() < self.config.body.gradient_strength:
                # Bias toward gradient center
                cx = self._gradient_cx * self.width
                cy = self._gradient_cy * self.height
                x = float(np.clip(self.rng.normal(cx, self.width * 0.25), 0, self.width))
                y = float(np.clip(self.rng.normal(cy, self.height * 0.25), 0, self.height))
            else:
                x = self.rng.uniform(0, self.width)
                y = self.rng.uniform(0, self.height)

            self.resources.append(Resource(
                x=x, y=y,
                energy=self.config.world.plant_energy,
                resource_type="plant",
            ))

        # Decay meat and nutrients
        surviving = []
        for r in self.resources:
            r.age += 1
            if r.resource_type == "meat":
                r.energy *= (1.0 - self.config.world.meat_decay_rate)
                if r.energy < 1.0:
                    # Convert to nutrient
                    self._add_nutrient(r.x, r.y, r.energy)
                    continue
            elif r.resource_type == "nutrient":
                r.energy *= 0.99
                if r.energy < 0.1:
                    self._add_nutrient(r.x, r.y, r.energy)
                    continue
            surviving.append(r)
        self.resources = surviving

        # Slowly decay nutrient grid
        self.nutrient_grid *= 0.999

    def _add_nutrient(self, x: float, y: float, energy: float) -> None:
        """Add nutrient value to the grid at a world position."""
        cols = self.nutrient_grid.shape[1]
        rows = self.nutrient_grid.shape[0]
        col = int(x / self.width * cols) % cols
        row = int(y / self.height * rows) % rows
        self.nutrient_grid[row, col] += energy * 0.1

    def rebuild_spatial_hashes(self) -> None:
        """Rebuild spatial hashes for current positions."""
        self.organism_hash.clear()
        for org in self.organisms:
            if org.alive:
                pos = org.position
                self.organism_hash.insert(org, pos[0], pos[1])

        self.resource_hash.clear()
        for res in self.resources:
            self.resource_hash.insert(res, res.x, res.y)

    def get_nearby_food(
        self, x: float, y: float, radius: float
    ) -> list[tuple[float, float, float, str]]:
        """Get food resources near a position."""
        items = self.resource_hash.query(x, y, radius)
        result = []
        r2 = radius * radius
        for res in items:
            dx = res.x - x
            dy = res.y - y
            if dx * dx + dy * dy <= r2:
                result.append((res.x, res.y, res.energy, res.resource_type))
        return result

    def get_nearby_organisms(
        self, x: float, y: float, radius: float, exclude: Organism | None = None
    ) -> list[tuple[float, float, float, bool]]:
        """Get organisms near a position.

        Returns: [(x, y, mass, same_species), ...]
        """
        items = self.organism_hash.query(x, y, radius)
        result = []
        r2 = radius * radius
        for org in items:
            if org is exclude or not org.alive:
                continue
            pos = org.position
            dx = pos[0] - x
            dy = pos[1] - y
            if dx * dx + dy * dy <= r2:
                same = (exclude is not None and org.species_id == exclude.species_id)
                result.append((pos[0], pos[1], org.body.total_mass, same))
        return result

    def handle_eating(self) -> None:
        """Process eating interactions between organisms and food."""
        eaten: set[int] = set()  # resource ids
        base_eat_radius = self.config.body.eat_radius
        bone_reach = self.config.body.bone_reach_scaling
        reach_factor = self.config.body.bone_reach_factor

        for org in self.organisms:
            if not org.alive or not org.wants_to_eat():
                continue
            if org.body.n_mouths == 0:
                continue

            com = org.body.center_of_mass

            # Check each mouth node for contact with nearby food (via spatial hash)
            for mouth_idx in org.body.mouth_indices:
                mx, my = org.body.positions[mouth_idx]

                # Part 3: Bone reach - mouths far from COM get bigger radius
                eat_radius = base_eat_radius
                if bone_reach:
                    dist_from_com = math.sqrt(
                        (mx - com[0]) ** 2 + (my - com[1]) ** 2
                    )
                    eat_radius += dist_from_com * reach_factor

                r2 = eat_radius * eat_radius
                nearby = self.resource_hash.query(mx, my, eat_radius)
                for res in nearby:
                    if id(res) in eaten:
                        continue
                    dx = res.x - mx
                    dy = res.y - my
                    if dx * dx + dy * dy <= r2:
                        org.eat(res.energy)
                        eaten.add(id(res))
                        break  # one food per mouth per tick

        # Remove eaten resources
        if eaten:
            self.resources = [r for r in self.resources if id(r) not in eaten]

    def handle_combat(self) -> None:
        """Process combat between organisms."""
        kin_enabled = self.config.body.enable_kin_recognition
        armor_reflect = self.config.body.armor_damage_reflection
        energy_transfer = self.config.evolution.predation_energy_transfer
        immunity_ticks = self.config.body.offspring_immunity_ticks

        for org in self.organisms:
            if not org.alive or not org.wants_to_attack():
                continue
            if org.body.n_mouths == 0:
                continue

            for mouth_idx in org.body.mouth_indices:
                mx, my = org.body.positions[mouth_idx]
                attack_radius = self.config.body.eat_radius * 0.5

                nearby = self.organism_hash.query(mx, my, attack_radius)
                for target in nearby:
                    if target is org or not target.alive:
                        continue

                    # Part 3: Kin recognition - skip same species if tolerant
                    if kin_enabled and target.species_id == org.species_id:
                        kin_tol = getattr(org, 'kin_tolerance', 0.0)
                        if self.rng.random() < kin_tol:
                            continue

                    # Part 3: Offspring immunity
                    if kin_enabled and target.age < immunity_ticks:
                        if target.parent_id == org.id:
                            continue

                    # Check if mouth actually contacts target's body
                    for ni in range(target.body.n_nodes):
                        tx, ty = target.body.positions[ni]
                        dx = tx - mx
                        dy = ty - my
                        if dx * dx + dy * dy <= attack_radius * attack_radius:
                            damage = self.config.evolution.attack_damage_per_mouth
                            actual = target.take_damage(damage)
                            # Attacker gains energy from the bite
                            org.eat(actual * energy_transfer)

                            # Part 3: Armor damage reflection
                            if armor_reflect > 0 and target.body.armor_value > 0:
                                reflected = damage * armor_reflect * min(target.body.armor_value, 0.9)
                                org.take_damage(reflected)
                            break

    def handle_deaths(self) -> None:
        """Process dead organisms - convert to meat resources."""
        self.dead_this_tick = []
        surviving = []
        for org in self.organisms:
            if not org.alive:
                self.dead_this_tick.append(org)
                pos = org.position
                meat_energy = org.body.total_mass * 8.0
                self.resources.append(Resource(
                    x=pos[0], y=pos[1],
                    energy=meat_energy,
                    resource_type="meat",
                    decay_rate=self.config.world.meat_decay_rate,
                ))
            else:
                surviving.append(org)
        self.organisms = surviving

    def wrap_positions(self) -> None:
        """Wrap organism positions for toroidal world.

        Wraps the core node and moves all other nodes relative to it,
        keeping the organism body coherent across the boundary.
        """
        if not self.config.world.wrap_around:
            return
        w, h = self.width, self.height
        for org in self.organisms:
            core = org.body.positions[0]
            old_x, old_y = core[0], core[1]
            new_x, new_y = old_x % w, old_y % h
            dx, dy = new_x - old_x, new_y - old_y
            if dx != 0.0 or dy != 0.0:
                org.body.positions[:, 0] += dx
                org.body.positions[:, 1] += dy

    def add_organism(self, org: Organism) -> None:
        self.organisms.append(org)

    def get_stats(self) -> dict:
        """Get current world statistics."""
        if not self.organisms:
            return {
                "population": 0,
                "avg_energy": 0,
                "avg_age": 0,
                "species_count": 0,
                "resource_count": len(self.resources),
                "avg_nodes": 0,
                "max_generation": 0,
            }

        energies = [o.energy for o in self.organisms]
        ages = [o.age for o in self.organisms]
        species = {o.species_id for o in self.organisms}
        nodes = [o.body.n_nodes for o in self.organisms]
        generations = [o.generation for o in self.organisms]

        return {
            "population": len(self.organisms),
            "avg_energy": round(sum(energies) / len(energies), 1),
            "avg_age": round(sum(ages) / len(ages), 1),
            "species_count": len(species),
            "resource_count": len(self.resources),
            "avg_nodes": round(sum(nodes) / len(nodes), 1),
            "max_generation": max(generations),
        }

    def to_dict(self) -> dict:
        """Serialize world state for recording."""
        return {
            "tick": self.tick,
            "organisms": [o.to_dict() for o in self.organisms],
            "resources": [r.to_dict() for r in self.resources],
            "stats": self.get_stats(),
        }
