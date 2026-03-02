"""Environment simulation - resources, spatial hashing, interactions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimConfig
    from .organism import Organism


# Part 4: Terrain types
class TerrainType(IntEnum):
    FERTILE = 0   # normal drag, 1.5x food spawn
    DENSE = 1     # 2x drag, 2x food (thick vegetation)
    ROCKY = 2     # 1.2x drag, 0.3x food (sparse but safe)
    WATER = 3     # 0.3x drag, 0x food (fast but empty)

# Terrain properties: (drag_modifier, food_spawn_modifier, speed_modifier)
_TERRAIN_PROPS = {
    TerrainType.FERTILE: (1.0, 1.5, 1.0),
    TerrainType.DENSE:   (2.0, 2.0, 0.6),
    TerrainType.ROCKY:   (1.2, 0.3, 0.8),
    TerrainType.WATER:   (0.3, 0.0, 1.2),
}


@dataclass
class Resource:
    x: float
    y: float
    energy: float
    resource_type: str  # "plant", "algae", "fruit", "toxic", "meat", "nutrient"
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
    """The simulation environment."""

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

        # Nutrient density map
        nutrient_cells = max(1, int(self.width / config.world.spatial_cell_size))
        self.nutrient_grid = np.zeros((nutrient_cells, nutrient_cells))

        # Part 3: Environmental dynamics state
        self._food_shock_remaining = 0
        self._gradient_cx = config.body.gradient_center_x
        self._gradient_cy = config.body.gradient_center_y

        # Part 4: Terrain grid
        self.terrain_grid = None
        if config.body.enable_terrain:
            self._init_terrain()

        # Part 4: Day/night state
        self._light_level = 1.0  # 1.0 = noon, 0.0 = midnight

        # Part 4: Hazard zones [(x, y, vx, vy), ...]
        self._hazards = []
        if config.body.enable_hazards:
            for _ in range(config.body.hazard_count):
                self._hazards.append([
                    rng.uniform(0, self.width),
                    rng.uniform(0, self.height),
                    rng.normal(0, config.body.hazard_drift_speed),
                    rng.normal(0, config.body.hazard_drift_speed),
                ])

        # Part 4: Signal hash for chemical broadcasts
        self.signal_hash = SpatialHash(config.world.spatial_cell_size)
        self._active_signals = []  # [(x, y, signal_vector, intensity, species_id, org_id), ...]

    def _init_terrain(self) -> None:
        """Generate terrain grid using simple noise."""
        cfg = self.config.body
        cols = max(1, int(self.width / cfg.terrain_cell_size))
        rows = max(1, int(self.height / cfg.terrain_cell_size))
        self.terrain_grid = np.zeros((rows, cols), dtype=np.int32)

        scale = cfg.terrain_scale
        # Simple 2-octave noise using sin/cos
        for r in range(rows):
            for c in range(cols):
                x, y = c * cfg.terrain_cell_size, r * cfg.terrain_cell_size
                # Two octaves of pseudo-noise
                v = (math.sin(x * scale * 1.0 + 0.3) * math.cos(y * scale * 0.7 + 1.1)
                     + 0.5 * math.sin(x * scale * 2.3 + y * scale * 1.7)
                     + 0.3 * math.cos(x * scale * 0.5 - y * scale * 2.1 + 2.0))
                # Map to terrain types by thresholds
                if v < -0.6:
                    self.terrain_grid[r, c] = TerrainType.WATER
                elif v < -0.1:
                    self.terrain_grid[r, c] = TerrainType.ROCKY
                elif v < 0.5:
                    self.terrain_grid[r, c] = TerrainType.FERTILE
                else:
                    self.terrain_grid[r, c] = TerrainType.DENSE

    def get_terrain_at(self, x: float, y: float) -> int:
        """Get terrain type at world position. Returns TerrainType int."""
        if self.terrain_grid is None:
            return TerrainType.FERTILE
        cfg = self.config.body
        c = int(x / cfg.terrain_cell_size) % self.terrain_grid.shape[1]
        r = int(y / cfg.terrain_cell_size) % self.terrain_grid.shape[0]
        return int(self.terrain_grid[r, c])

    def get_terrain_props(self, x: float, y: float) -> tuple[float, float, float]:
        """Get (drag_mod, food_mod, speed_mod) for position."""
        t = self.get_terrain_at(x, y)
        return _TERRAIN_PROPS.get(t, (1.0, 1.0, 1.0))

    def get_light_level(self) -> float:
        """Part 4: Current light level [0, 1]."""
        return self._light_level

    def spawn_initial_resources(self, count: int | None = None) -> None:
        if count is None:
            count = self.config.world.initial_food_count
        """Scatter initial plant resources across the world."""
        for _ in range(count):
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)
            rtype, energy = self._pick_food_type(x, y)
            self.resources.append(Resource(
                x=x, y=y, energy=energy, resource_type=rtype,
            ))

    def _pick_food_type(self, x: float, y: float) -> tuple[str, float]:
        """Part 4: Choose food type based on config."""
        cfg = self.config.body
        if not cfg.enable_food_types:
            return "plant", self.config.world.plant_energy

        roll = self.rng.random()
        if roll < cfg.toxic_spawn_fraction:
            return "toxic", cfg.algae_energy  # looks like algae
        elif roll < cfg.toxic_spawn_fraction + cfg.fruit_spawn_fraction:
            return "fruit", cfg.fruit_energy
        else:
            return "algae", cfg.algae_energy

    def step_resources(self) -> None:
        """Spawn new resources, decay old ones."""
        if len(self.resources) >= self.config.world.max_resources:
            spawn_rate = 0.0
        else:
            spawn_rate = self.config.world.plant_spawn_rate

        # Seasonal modulation
        if self.config.body.enable_seasons:
            phase = (self.tick % self.config.body.season_length) / self.config.body.season_length
            seasonal_mult = 1.0 + self.config.body.season_food_amplitude * math.sin(2 * math.pi * phase)
            spawn_rate *= seasonal_mult

        # Food shock
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

        # Spatial gradient drift
        if self.config.body.enable_spatial_gradient:
            self._gradient_cx += self.rng.normal(0, self.config.body.gradient_shift_rate)
            self._gradient_cy += self.rng.normal(0, self.config.body.gradient_shift_rate)
            self._gradient_cx = float(np.clip(self._gradient_cx, 0.2, 0.8))
            self._gradient_cy = float(np.clip(self._gradient_cy, 0.2, 0.8))

        for _ in range(n_spawn):
            # Choose spawn location
            if self.rng.random() < 0.3 and np.max(self.nutrient_grid) > 0:
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
                cx = self._gradient_cx * self.width
                cy = self._gradient_cy * self.height
                x = float(np.clip(self.rng.normal(cx, self.width * 0.25), 0, self.width))
                y = float(np.clip(self.rng.normal(cy, self.height * 0.25), 0, self.height))
            else:
                x = self.rng.uniform(0, self.width)
                y = self.rng.uniform(0, self.height)

            # Part 4: Terrain affects food spawn
            if self.terrain_grid is not None:
                _, food_mod, _ = self.get_terrain_props(x, y)
                if food_mod <= 0 or self.rng.random() > food_mod / 2.0:
                    continue  # skip spawning in water/rocky

            rtype, energy = self._pick_food_type(x, y)
            self.resources.append(Resource(x=x, y=y, energy=energy, resource_type=rtype))

        # Decay meat and nutrients
        surviving = []
        for r in self.resources:
            r.age += 1
            if r.resource_type == "meat":
                r.energy *= (1.0 - self.config.world.meat_decay_rate)
                if r.energy < 1.0:
                    self._add_nutrient(r.x, r.y, r.energy)
                    continue
            elif r.resource_type == "nutrient":
                r.energy *= 0.99
                if r.energy < 0.1:
                    self._add_nutrient(r.x, r.y, r.energy)
                    continue
            surviving.append(r)
        self.resources = surviving
        self.nutrient_grid *= 0.999

    def _add_nutrient(self, x: float, y: float, energy: float) -> None:
        cols = self.nutrient_grid.shape[1]
        rows = self.nutrient_grid.shape[0]
        col = int(x / self.width * cols) % cols
        row = int(y / self.height * rows) % rows
        self.nutrient_grid[row, col] += energy * 0.1

    def rebuild_spatial_hashes(self) -> None:
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
        self, x: float, y: float, radius: float, exclude: "Organism | None" = None
    ) -> list[tuple[float, float, float, bool]]:
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

    def get_nearby_signals(
        self, x: float, y: float, radius: float
    ) -> list[tuple[float, float, list, float, str, str]]:
        """Part 4: Get nearby signal broadcasts."""
        items = self.signal_hash.query(x, y, radius)
        result = []
        r2 = radius * radius
        for sig in items:
            dx = sig[0] - x
            dy = sig[1] - y
            if dx * dx + dy * dy <= r2:
                result.append(sig)
        return result

    def handle_eating(self) -> None:
        """Process eating interactions between organisms and food."""
        eaten: set[int] = set()
        base_eat_radius = self.config.body.eat_radius
        bone_reach = self.config.body.bone_reach_scaling
        reach_factor = self.config.body.bone_reach_factor
        enable_food_types = self.config.body.enable_food_types
        toxic_damage = self.config.body.toxic_damage

        for org in self.organisms:
            if not org.alive or not org.wants_to_eat():
                continue
            if org.body.n_mouths == 0:
                continue

            com = org.body.center_of_mass

            for mouth_idx in org.body.mouth_indices:
                # Part 4: Skip disabled mouth nodes
                if org.body.node_alive is not None and not org.body.node_alive[mouth_idx]:
                    continue

                mx, my = org.body.positions[mouth_idx]

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
                        # Part 4: Toxic food damages instead of feeding
                        if enable_food_types and res.resource_type == "toxic":
                            org.take_damage(toxic_damage)
                        else:
                            org.eat(res.energy, res.resource_type)
                        eaten.add(id(res))
                        break

        if eaten:
            self.resources = [r for r in self.resources if id(r) not in eaten]

    def handle_combat(self) -> None:
        """Process combat between organisms."""
        kin_enabled = self.config.body.enable_kin_recognition
        armor_reflect = self.config.body.armor_damage_reflection
        energy_transfer = self.config.evolution.predation_energy_transfer
        immunity_ticks = self.config.body.offspring_immunity_ticks
        edge_combat = self.config.body.enable_edge_combat
        node_hp_enabled = self.config.body.enable_node_hp

        for org in self.organisms:
            if not org.alive or not org.wants_to_attack():
                continue
            if org.body.n_mouths == 0:
                continue

            for mouth_idx in org.body.mouth_indices:
                if org.body.node_alive is not None and not org.body.node_alive[mouth_idx]:
                    continue

                mx, my = org.body.positions[mouth_idx]
                attack_radius = self.config.body.attack_radius

                nearby = self.organism_hash.query(mx, my, attack_radius)
                for target in nearby:
                    if target is org or not target.alive:
                        continue

                    if kin_enabled and target.species_id == org.species_id:
                        kin_tol = getattr(org, 'kin_tolerance', 0.0)
                        if self.rng.random() < kin_tol:
                            continue

                    if kin_enabled and target.age < immunity_ticks:
                        if target.parent_id == org.id:
                            continue

                    # Check contact: nodes first, then edges if enabled
                    hit = False
                    hit_node = -1
                    ar2 = attack_radius * attack_radius

                    for ni in range(target.body.n_nodes):
                        tx, ty = target.body.positions[ni]
                        dx = tx - mx
                        dy = ty - my
                        if dx * dx + dy * dy <= ar2:
                            hit = True
                            hit_node = ni
                            break

                    # Part 4: Edge-aware combat
                    if not hit and edge_combat and target.body.n_edges > 0:
                        edge_dist, nearest = target.body.point_to_edge_distance(mx, my)
                        if edge_dist <= attack_radius:
                            hit = True
                            hit_node = nearest

                    if hit:
                        damage = self.config.evolution.attack_damage_per_mouth

                        # Part 4: Node HP damage model
                        if node_hp_enabled and hit_node >= 0:
                            target.body.damage_node(hit_node, damage * 0.3)

                        actual = target.take_damage(damage)
                        org.eat(actual * energy_transfer)

                        # Track damage for organism awareness
                        if hasattr(target, '_damage_recent'):
                            target._damage_recent += actual

                        if armor_reflect > 0 and target.body.armor_value > 0:
                            reflected = damage * armor_reflect * min(target.body.armor_value, 0.9)
                            org.take_damage(reflected)
                        break

    def handle_signals(self) -> None:
        """Part 4: Collect signal broadcasts from organisms with SIGNAL nodes."""
        if not self.config.body.enable_signals:
            return

        self._active_signals.clear()
        self.signal_hash.clear()

        for org in self.organisms:
            if not org.alive or org.body.n_signals == 0:
                continue
            intensity = getattr(org, 'signal_intensity', 0.0)
            if intensity < 0.1:
                continue

            pos = org.position
            sig_vec = getattr(org, 'signal_vector', [0.5, 0.5, 0.5])
            entry = (float(pos[0]), float(pos[1]), sig_vec, intensity,
                     org.species_id, org.id)
            self._active_signals.append(entry)
            self.signal_hash.insert(entry, pos[0], pos[1])

    def handle_sharing(self) -> None:
        """Part 4: Process resource sharing between kin."""
        if not self.config.body.enable_sharing:
            return

        share_rate = self.config.body.share_rate
        share_radius = self.config.body.share_radius

        for org in self.organisms:
            if not org.alive:
                continue
            share_signal = getattr(org, 'share_signal', 0.0)
            if share_signal < 0.5:
                continue
            if org.energy < share_rate * 2:
                continue  # don't share if nearly empty

            pos = org.position
            nearby = self.organism_hash.query(pos[0], pos[1], share_radius)

            for target in nearby:
                if target is org or not target.alive:
                    continue
                if target.species_id != org.species_id:
                    continue

                tpos = target.position
                dx = pos[0] - tpos[0]
                dy = pos[1] - tpos[1]
                if dx * dx + dy * dy > share_radius * share_radius:
                    continue

                # Transfer energy
                amount = min(share_rate, org.energy * 0.1)
                org.energy -= amount
                target.energy = min(target.energy + amount, target.body.max_energy_capacity)
                break  # one share per tick

    def handle_group_bonuses(self) -> None:
        """Part 4: Compute group sizes for each organism."""
        if not self.config.body.enable_group_bonus:
            return

        radius = self.config.body.group_radius

        for org in self.organisms:
            if not org.alive:
                continue
            pos = org.position
            nearby = self.organism_hash.query(pos[0], pos[1], radius)
            count = 0
            for other in nearby:
                if other is org or not other.alive:
                    continue
                if other.species_id == org.species_id:
                    opos = other.position
                    dx = pos[0] - opos[0]
                    dy = pos[1] - opos[1]
                    if dx * dx + dy * dy <= radius * radius:
                        count += 1
            org.group_size = count

    def handle_organism_repulsion(self) -> None:
        """Part 4: Apply soft repulsive forces between overlapping organisms."""
        if not self.config.body.enable_organism_repulsion:
            return

        stiffness = self.config.body.repulsion_stiffness
        min_overlap = self.config.body.repulsion_min_overlap

        for org in self.organisms:
            if not org.alive:
                continue
            pos = org.position
            radius = org.bounding_radius + 5.0
            nearby = self.organism_hash.query(pos[0], pos[1], radius * 2)

            for other in nearby:
                if other is org or not other.alive:
                    continue
                opos = other.position
                dx = pos[0] - opos[0]
                dy = pos[1] - opos[1]
                dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                combined_radius = org.bounding_radius + other.bounding_radius
                overlap = combined_radius - dist

                if overlap > min_overlap:
                    # Repulsive force proportional to overlap
                    force = stiffness * (overlap - min_overlap)
                    nx, ny = dx / dist, dy / dist
                    # Apply to core nodes (index 0), weighted by inverse mass
                    m1 = org.body.masses[0]
                    m2 = other.body.masses[0]
                    total_m = m1 + m2
                    org.body.velocities[0][0] += force * nx * m2 / total_m
                    org.body.velocities[0][1] += force * ny * m2 / total_m
                    other.body.velocities[0][0] -= force * nx * m1 / total_m
                    other.body.velocities[0][1] -= force * ny * m1 / total_m

    def handle_hazards(self) -> None:
        """Part 4: Apply damage to organisms inside hazard zones."""
        if not self.config.body.enable_hazards:
            return

        damage = self.config.body.hazard_damage_per_tick
        radius = self.config.body.hazard_radius
        r2 = radius * radius

        for hz in self._hazards:
            hx, hy = hz[0], hz[1]
            nearby = self.organism_hash.query(hx, hy, radius)
            for org in nearby:
                if not org.alive:
                    continue
                pos = org.position
                dx = pos[0] - hx
                dy = pos[1] - hy
                if dx * dx + dy * dy <= r2:
                    org.energy -= damage
                    if org.energy <= 0:
                        org.energy = 0
                        org.alive = False

    def step_hazards(self) -> None:
        """Part 4: Move hazard zones (drift + bounce)."""
        if not self._hazards:
            return
        for hz in self._hazards:
            hz[0] += hz[2]
            hz[1] += hz[3]
            # Bounce off edges
            if hz[0] < 0 or hz[0] > self.width:
                hz[2] *= -1
                hz[0] = max(0, min(self.width, hz[0]))
            if hz[1] < 0 or hz[1] > self.height:
                hz[3] *= -1
                hz[1] = max(0, min(self.height, hz[1]))

    def step_day_night(self) -> None:
        """Part 4: Update day/night cycle."""
        if not self.config.body.enable_day_night:
            self._light_level = 1.0
            return
        period = self.config.body.day_night_period
        phase = (self.tick % period) / period
        self._light_level = 0.5 + 0.5 * math.sin(2 * math.pi * phase)

    def handle_deaths(self) -> None:
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

    def add_organism(self, org: "Organism") -> None:
        self.organisms.append(org)

    def get_stats(self) -> dict:
        if not self.organisms:
            return {
                "population": 0, "avg_energy": 0, "avg_age": 0,
                "species_count": 0, "resource_count": len(self.resources),
                "avg_nodes": 0, "max_generation": 0,
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
        return {
            "tick": self.tick,
            "organisms": [o.to_dict() for o in self.organisms],
            "resources": [r.to_dict() for r in self.resources],
            "stats": self.get_stats(),
        }
