"""Spring-mass body system and physics simulation."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import BodyConfig


class NodeType(IntEnum):
    CORE = 0
    BONE = 1
    MUSCLE_ANCHOR = 2
    SENSOR = 3
    MOUTH = 4
    FAT = 5
    ARMOR = 6


class EdgeType(IntEnum):
    BONE = 0
    MUSCLE = 1
    TENDON = 2


# Maps NodeType -> (mass_attr, cost_attr) on BodyConfig
_NODE_PROPS = {
    NodeType.CORE: ("mass_core", "cost_core"),
    NodeType.BONE: ("mass_bone", "cost_bone"),
    NodeType.MUSCLE_ANCHOR: ("mass_muscle_anchor", "cost_muscle_anchor"),
    NodeType.SENSOR: ("mass_sensor", "cost_sensor"),
    NodeType.MOUTH: ("mass_mouth", "cost_mouth"),
    NodeType.FAT: ("mass_fat", "cost_fat"),
    NodeType.ARMOR: ("mass_armor", "cost_armor"),
}

_EDGE_STIFFNESS = {
    EdgeType.BONE: "spring_stiffness_bone",
    EdgeType.MUSCLE: "spring_stiffness_muscle",
    EdgeType.TENDON: "spring_stiffness_tendon",
}


class Body:
    """A spring-mass body composed of nodes and edges.

    Nodes are point masses with positions and velocities.
    Edges are springs connecting nodes (bone=rigid, muscle=controllable, tendon=semi-rigid).
    """

    def __init__(
        self,
        node_types: list[int],
        node_positions: np.ndarray,
        edges: list[tuple[int, int, int]],
        config: BodyConfig,
    ):
        """
        Args:
            node_types: List of NodeType values, one per node.
            node_positions: (N, 2) array of initial positions.
            edges: List of (from_idx, to_idx, EdgeType) tuples.
            config: Body physics configuration.
        """
        self.config = config
        self.n_nodes = len(node_types)
        self.node_types = np.array(node_types, dtype=np.int32)
        self.positions = np.array(node_positions, dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)

        # Edge arrays for vectorized physics
        self.n_edges = len(edges)
        self.edge_from = np.array([e[0] for e in edges], dtype=np.int32)
        self.edge_to = np.array([e[1] for e in edges], dtype=np.int32)
        self.edge_types = np.array([e[2] for e in edges], dtype=np.int32)

        # Precompute per-node masses
        self.masses = np.array(
            [getattr(config, _NODE_PROPS[NodeType(t)][0]) for t in node_types],
            dtype=np.float64,
        )

        # Precompute per-edge stiffness
        self.stiffness = np.array(
            [getattr(config, _EDGE_STIFFNESS[EdgeType(e[2])]) for e in edges],
            dtype=np.float64,
        )

        # Compute rest lengths from initial positions
        if self.n_edges > 0:
            d = self.positions[self.edge_to] - self.positions[self.edge_from]
            self.rest_lengths = np.sqrt(np.sum(d * d, axis=1))
            # Prevent zero rest lengths
            self.rest_lengths = np.maximum(self.rest_lengths, 0.1)
        else:
            self.rest_lengths = np.array([], dtype=np.float64)

        # Muscle contraction targets (1.0 = rest length, set by neural net)
        self.muscle_mask = self.edge_types == EdgeType.MUSCLE
        self.n_muscles = int(np.sum(self.muscle_mask))
        self.muscle_targets = np.ones(self.n_edges, dtype=np.float64)

        # Index maps for brain I/O
        self.sensor_indices = np.where(self.node_types == NodeType.SENSOR)[0]
        self.mouth_indices = np.where(self.node_types == NodeType.MOUTH)[0]
        self.muscle_edge_indices = np.where(self.muscle_mask)[0]
        self.fat_indices = np.where(self.node_types == NodeType.FAT)[0]
        self.armor_indices = np.where(self.node_types == NodeType.ARMOR)[0]

        self.n_sensors = len(self.sensor_indices)
        self.n_mouths = len(self.mouth_indices)

        # Cached values (recomputed once per tick or on mutation)
        self._total_mass = float(np.sum(self.masses))
        self._energy_cost = sum(
            getattr(config, _NODE_PROPS[NodeType(t)][1]) for t in node_types
        )
        self._com_cache: np.ndarray | None = None

    @property
    def center_of_mass(self) -> np.ndarray:
        """Compute center of mass (cached per tick)."""
        if self._com_cache is not None:
            return self._com_cache
        if self._total_mass == 0:
            self._com_cache = np.mean(self.positions, axis=0)
        else:
            self._com_cache = np.sum(
                self.positions * self.masses[:, np.newaxis], axis=0
            ) / self._total_mass
        return self._com_cache

    def invalidate_cache(self) -> None:
        """Call after physics step to invalidate cached center of mass."""
        self._com_cache = None

    @property
    def total_mass(self) -> float:
        return self._total_mass

    @property
    def energy_cost_per_tick(self) -> float:
        """Total metabolic cost per simulation tick."""
        return self._energy_cost

    @property
    def max_energy_capacity(self) -> float:
        """Base energy capacity plus fat storage bonus."""
        base = 100.0
        fat_bonus = len(self.fat_indices) * 50.0
        return base + fat_bonus

    @property
    def armor_value(self) -> float:
        """Damage reduction from armor nodes."""
        return len(self.armor_indices) * 0.2  # 20% reduction per armor node, stacks

    @property
    def muscle_ratio(self) -> float:
        """Ratio of muscle edges to total nodes (locomotion capability)."""
        if self.n_nodes == 0:
            return 0.0
        return self.n_muscles / self.n_nodes

    @property
    def effective_max_velocity(self) -> float:
        """Max velocity accounting for muscle-speed scaling."""
        if not self.config.muscle_speed_scaling:
            return self.config.max_velocity
        base = self.config.base_max_velocity
        bonus = self.muscle_ratio * self.config.muscle_velocity_bonus
        return base + bonus

    @property
    def facing_angle(self) -> float:
        """Estimate facing direction from core to average sensor/mouth position."""
        com = self.center_of_mass
        # Use sensors and mouths to determine "front"
        front_indices = np.concatenate([self.sensor_indices, self.mouth_indices])
        if len(front_indices) == 0:
            return 0.0
        front_pos = np.mean(self.positions[front_indices], axis=0)
        delta = front_pos - com
        return float(math.atan2(delta[1], delta[0]))

    def set_muscle_targets(self, targets: np.ndarray) -> None:
        """Set muscle contraction targets from neural net output.

        Args:
            targets: Array of contraction values for each muscle.
                     Values in [0, 1] mapped to [min_contraction, max_contraction].
        """
        min_c = self.config.muscle_min_contraction
        max_c = self.config.muscle_max_contraction
        # Map sigmoid/tanh output [0,1] to contraction range
        clamped = np.clip(targets[: self.n_muscles], 0.0, 1.0)
        contraction = min_c + clamped * (max_c - min_c)
        self.muscle_targets[self.muscle_mask] = contraction

    def get_muscle_lengths(self) -> np.ndarray:
        """Get current muscle lengths (normalized by rest length) for proprioception."""
        if self.n_muscles == 0:
            return np.array([], dtype=np.float64)
        d = self.positions[self.edge_to[self.muscle_mask]] - self.positions[self.edge_from[self.muscle_mask]]
        current = np.sqrt(np.sum(d * d, axis=1))
        rest = self.rest_lengths[self.muscle_mask]
        return current / np.maximum(rest, 0.1)

    def step(self, dt: float | None = None) -> None:
        """Advance physics by one substep.

        Uses Verlet-style integration:
        1. Compute spring forces
        2. Apply drag
        3. Integrate velocities and positions
        """
        if dt is None:
            dt = self.config.dt

        self.forces[:] = 0.0

        if self.n_edges > 0:
            # Compute spring forces (vectorized)
            d = self.positions[self.edge_to] - self.positions[self.edge_from]
            dist = np.sqrt(np.sum(d * d, axis=1))
            dist = np.maximum(dist, 1e-6)  # prevent division by zero

            # Target lengths: rest_length * muscle_target (muscles) or rest_length (others)
            target = self.rest_lengths * self.muscle_targets
            displacement = dist - target
            force_magnitude = self.stiffness * displacement

            # Damping: project velocity difference onto spring direction
            dv = self.velocities[self.edge_to] - self.velocities[self.edge_from]
            unit_d = d / dist[:, np.newaxis]
            relative_vel = np.sum(dv * unit_d, axis=1)
            force_magnitude += self.config.spring_damping * relative_vel

            # Force vectors along spring direction
            force_vectors = unit_d * force_magnitude[:, np.newaxis]

            # Accumulate forces on nodes
            np.add.at(self.forces, self.edge_from, force_vectors)
            np.add.at(self.forces, self.edge_to, -force_vectors)

        # Drag force (proportional to velocity)
        drag = self.config.drag_coefficient
        if self.config.bone_drag_reduction > 0:
            bone_count = int(np.sum(self.node_types == NodeType.BONE))
            drag = drag / (1.0 + self.config.bone_drag_reduction * bone_count)
        self.forces -= drag * self.velocities * self.masses[:, np.newaxis]

        # Integrate (semi-implicit Euler)
        acceleration = self.forces / self.masses[:, np.newaxis]
        self.velocities += acceleration * dt

        # Velocity cap (uses muscle-scaled max if enabled)
        max_vel = self.effective_max_velocity
        speed = np.sqrt(np.sum(self.velocities ** 2, axis=1))
        too_fast = speed > max_vel
        if np.any(too_fast):
            scale = np.where(too_fast, max_vel / np.maximum(speed, 1e-6), 1.0)
            self.velocities *= scale[:, np.newaxis]

        self.positions += self.velocities * dt

        # Enforce max stretch on springs
        self._enforce_max_stretch()

    def _enforce_max_stretch(self) -> None:
        """Clamp spring lengths to prevent extreme stretching."""
        if self.n_edges == 0:
            return
        d = self.positions[self.edge_to] - self.positions[self.edge_from]
        dist = np.sqrt(np.sum(d * d, axis=1))
        max_len = self.rest_lengths * self.config.max_stretch
        mask = dist > max_len
        if not np.any(mask):
            return
        # Move both endpoints toward each other, weighted by inverse mass
        unit = d / np.maximum(dist, 1e-6)[:, np.newaxis]
        excess = np.where(mask, dist - max_len, 0.0)
        mass_a = self.masses[self.edge_from]
        mass_b = self.masses[self.edge_to]
        total = mass_a + mass_b
        corr_a = (excess * mass_b / total)[:, np.newaxis] * unit
        corr_b = (excess * mass_a / total)[:, np.newaxis] * unit
        np.add.at(self.positions, self.edge_from, corr_a)
        np.add.at(self.positions, self.edge_to, -corr_b)
        # Dampen velocity of affected nodes
        affected = np.zeros(self.n_nodes, dtype=bool)
        affected[self.edge_from[mask]] = True
        affected[self.edge_to[mask]] = True
        self.velocities[affected] *= 0.5

    def step_full(self) -> None:
        """Run all physics substeps for one simulation tick."""
        sub_dt = self.config.dt / self.config.substeps
        for _ in range(self.config.substeps):
            self.step(sub_dt)
        self.invalidate_cache()

    def translate(self, offset: np.ndarray) -> None:
        """Move entire body by offset."""
        self.positions += offset

    def get_world_position(self) -> np.ndarray:
        """Get the organism's position in world space (center of mass)."""
        return self.center_of_mass

    def get_bounding_radius(self) -> float:
        """Get approximate bounding radius from center of mass."""
        com = self.center_of_mass
        dists = np.sqrt(np.sum((self.positions - com) ** 2, axis=1))
        return float(np.max(dists)) if len(dists) > 0 else 1.0

    def to_dict(self) -> dict:
        """Serialize body state for recording."""
        return {
            "node_types": self.node_types.tolist(),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "edge_from": self.edge_from.tolist(),
            "edge_to": self.edge_to.tolist(),
            "edge_types": self.edge_types.tolist(),
            "rest_lengths": self.rest_lengths.tolist(),
            "muscle_targets": self.muscle_targets[self.muscle_mask].tolist(),
        }


def is_connected(n_nodes: int, edges: list[tuple[int, int, int]]) -> bool:
    """Check if all nodes are connected via edges (flood fill)."""
    if n_nodes <= 1:
        return True

    adj: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for a, b, _ in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj[node] - visited)

    return len(visited) == n_nodes


def create_default_body(config: BodyConfig, rng: np.random.Generator) -> Body:
    """Create a simple starter organism body.

    Default shape: core in center, sensor in front, two muscle anchors on sides,
    mouth at front. Connected by bones and muscles.

       S
       |
    M--C--M
       |
       Mo

    S=sensor, C=core, M=muscle_anchor, Mo=mouth
    """
    node_types = [
        NodeType.CORE,           # 0: center
        NodeType.SENSOR,         # 1: front sensor
        NodeType.MUSCLE_ANCHOR,  # 2: left
        NodeType.MUSCLE_ANCHOR,  # 3: right
        NodeType.MOUTH,          # 4: front-bottom
    ]

    # Positions with small random perturbation
    spread = config.initial_spread
    positions = np.array([
        [0.0, 0.0],                          # core
        [0.0, -spread],                       # sensor (front)
        [-spread, 0.0],                       # left muscle anchor
        [spread, 0.0],                        # right muscle anchor
        [0.0, -spread * 0.7],                 # mouth (front)
    ]) + rng.normal(0, 0.1, (5, 2))

    edges = [
        (0, 1, EdgeType.BONE),     # core to sensor
        (0, 2, EdgeType.MUSCLE),   # core to left (muscle!)
        (0, 3, EdgeType.MUSCLE),   # core to right (muscle!)
        (0, 4, EdgeType.TENDON),   # core to mouth
        (1, 4, EdgeType.BONE),     # sensor to mouth (structural)
    ]

    return Body(node_types, positions, edges, config)
