"""Genome encoding, mutation, and crossover operators."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from .body import Body, EdgeType, NodeType, is_connected

if TYPE_CHECKING:
    from .config import BodyConfig, BrainConfig, EvolutionConfig


class Genome:
    """Complete genome encoding body plan, brain architecture, and meta-parameters.

    The genome is the heritable blueprint. It encodes:
    - Body: node types, relative positions, edge connectivity
    - Brain: hidden layer size, activation function, weights, biases
    - Meta: mutation rates and biases (themselves evolvable)
    """

    def __init__(
        self,
        body_nodes: list[dict],
        body_edges: list[dict],
        brain_hidden_size: int,
        brain_activation: str,
        brain_n_memory: int,
        brain_weights_ih: np.ndarray,
        brain_weights_ho: np.ndarray,
        brain_bias_h: np.ndarray,
        brain_bias_o: np.ndarray,
        meta: dict,
    ):
        # Body plan
        self.body_nodes = body_nodes  # [{"type": int, "rx": float, "ry": float}, ...]
        self.body_edges = body_edges  # [{"from": int, "to": int, "type": int}, ...]

        # Brain architecture + weights
        self.brain_hidden_size = brain_hidden_size
        self.brain_activation = brain_activation
        self.brain_n_memory = brain_n_memory
        self.brain_weights_ih = brain_weights_ih.copy()
        self.brain_weights_ho = brain_weights_ho.copy()
        self.brain_bias_h = brain_bias_h.copy()
        self.brain_bias_o = brain_bias_o.copy()

        # Meta-parameters (evolvable)
        self.meta = dict(meta)

    def clone(self) -> Genome:
        return Genome(
            body_nodes=deepcopy(self.body_nodes),
            body_edges=deepcopy(self.body_edges),
            brain_hidden_size=self.brain_hidden_size,
            brain_activation=self.brain_activation,
            brain_n_memory=self.brain_n_memory,
            brain_weights_ih=self.brain_weights_ih.copy(),
            brain_weights_ho=self.brain_weights_ho.copy(),
            brain_bias_h=self.brain_bias_h.copy(),
            brain_bias_o=self.brain_bias_o.copy(),
            meta=dict(self.meta),
        )

    def to_dict(self) -> dict:
        return {
            "body_nodes": self.body_nodes,
            "body_edges": self.body_edges,
            "brain_hidden_size": self.brain_hidden_size,
            "brain_activation": self.brain_activation,
            "brain_n_memory": self.brain_n_memory,
            "brain_weights_ih": self.brain_weights_ih.tolist(),
            "brain_weights_ho": self.brain_weights_ho.tolist(),
            "brain_bias_h": self.brain_bias_h.tolist(),
            "brain_bias_o": self.brain_bias_o.tolist(),
            "meta": self.meta,
        }


def create_default_genome(
    body_config: BodyConfig,
    brain_config: BrainConfig,
    evolution_config: EvolutionConfig,
    rng: np.random.Generator,
) -> Genome:
    """Create a genome for a default starter organism."""
    body_nodes = [
        {"type": int(NodeType.CORE), "rx": 0.0, "ry": 0.0},
        {"type": int(NodeType.SENSOR), "rx": 0.0, "ry": -1.5},
        {"type": int(NodeType.MUSCLE_ANCHOR), "rx": -1.5, "ry": 0.0},
        {"type": int(NodeType.MUSCLE_ANCHOR), "rx": 1.5, "ry": 0.0},
        {"type": int(NodeType.MOUTH), "rx": 0.0, "ry": -1.05},
    ]

    body_edges = [
        {"from": 0, "to": 1, "type": int(EdgeType.BONE)},
        {"from": 0, "to": 2, "type": int(EdgeType.MUSCLE)},
        {"from": 0, "to": 3, "type": int(EdgeType.MUSCLE)},
        {"from": 0, "to": 4, "type": int(EdgeType.TENDON)},
        {"from": 1, "to": 4, "type": int(EdgeType.BONE)},
    ]

    # Count sensors and muscles
    n_sensors = sum(1 for n in body_nodes if n["type"] == int(NodeType.SENSOR))
    n_muscles = sum(1 for e in body_edges if e["type"] == int(EdgeType.MUSCLE))
    n_memory = brain_config.n_memory

    n_inputs = min((n_sensors * 3) + 2 + n_muscles, brain_config.max_inputs)
    n_outputs = min(n_muscles + 2, brain_config.max_outputs)
    hidden = brain_config.default_hidden_size

    total_in = n_inputs + n_memory
    total_out = n_outputs + n_memory

    scale_ih = np.sqrt(2.0 / (total_in + hidden))
    scale_ho = np.sqrt(2.0 / (hidden + total_out))

    meta = {
        "body_mutation_rate": evolution_config.body_mutation_rate,
        "brain_mutation_rate": evolution_config.brain_mutation_rate,
        "weight_perturb_scale": evolution_config.weight_perturb_scale,
        "symmetry_bias": 0.7,
    }

    return Genome(
        body_nodes=body_nodes,
        body_edges=body_edges,
        brain_hidden_size=hidden,
        brain_activation=brain_config.default_activation,
        brain_n_memory=n_memory,
        brain_weights_ih=rng.normal(0, scale_ih, (total_in, hidden)),
        brain_weights_ho=rng.normal(0, scale_ho, (hidden, total_out)),
        brain_bias_h=np.zeros(hidden),
        brain_bias_o=np.zeros(total_out),
        meta=meta,
    )


def mutate(
    genome: Genome,
    evolution_config: EvolutionConfig,
    brain_config: BrainConfig,
    rng: np.random.Generator,
) -> Genome:
    """Apply mutations to a genome. Returns a new mutated genome."""
    g = genome.clone()

    # --- Meta-parameter mutation (fixed rate) ---
    _mutate_meta(g, evolution_config.meta_mutation_rate, rng)

    # --- Brain weight mutation ---
    _mutate_brain_weights(g, rng)

    # --- Body structural mutation ---
    if rng.random() < g.meta["body_mutation_rate"]:
        _mutate_body(g, evolution_config, rng)
        # After body mutation, resize brain if needed
        _resize_brain_for_body(g, brain_config, rng)

    return g


def _mutate_meta(g: Genome, meta_rate: float, rng: np.random.Generator) -> None:
    """Mutate meta-parameters with a fixed rate."""
    for key in ["body_mutation_rate", "brain_mutation_rate", "weight_perturb_scale"]:
        if rng.random() < meta_rate:
            g.meta[key] *= float(np.exp(rng.normal(0, 0.1)))
            g.meta[key] = np.clip(g.meta[key], 0.001, 0.5)

    if rng.random() < meta_rate:
        g.meta["symmetry_bias"] += rng.normal(0, 0.05)
        g.meta["symmetry_bias"] = np.clip(g.meta["symmetry_bias"], 0.0, 1.0)


def _mutate_brain_weights(g: Genome, rng: np.random.Generator) -> None:
    """Perturb brain weights with Gaussian noise."""
    rate = g.meta["brain_mutation_rate"]
    scale = g.meta["weight_perturb_scale"]

    for arr in [g.brain_weights_ih, g.brain_weights_ho, g.brain_bias_h, g.brain_bias_o]:
        mask = rng.random(arr.shape) < rate
        arr[mask] += rng.normal(0, scale, np.sum(mask))


def _mutate_body(
    g: Genome,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> None:
    """Apply a single structural body mutation."""
    n_nodes = len(g.body_nodes)

    # Choose mutation type by probability
    roll = rng.random()
    cumulative = 0.0

    # Add node
    cumulative += config.add_node_prob
    if roll < cumulative and n_nodes < config.max_body_nodes:
        _body_add_node(g, rng)
        return

    # Remove node
    cumulative += config.remove_node_prob
    if roll < cumulative and n_nodes > 3:  # minimum 3 nodes
        _body_remove_node(g, rng)
        return

    # Change node type
    cumulative += config.change_type_prob
    if roll < cumulative:
        _body_change_type(g, rng)
        return

    # Add/remove edge
    cumulative += config.add_remove_edge_prob
    if roll < cumulative:
        _body_add_remove_edge(g, rng)
        return

    # Perturb position
    _body_perturb_position(g, rng)


def _body_add_node(g: Genome, rng: np.random.Generator) -> None:
    """Add a new node connected to an existing node."""
    n = len(g.body_nodes)
    parent_idx = rng.integers(0, n)
    parent = g.body_nodes[parent_idx]

    # Random type (excluding CORE - only one allowed)
    new_type = int(rng.choice([
        NodeType.BONE, NodeType.MUSCLE_ANCHOR, NodeType.SENSOR,
        NodeType.MOUTH, NodeType.FAT, NodeType.ARMOR,
    ]))

    # Position near parent with some offset
    offset = rng.normal(0, 1.0, 2)
    new_node = {
        "type": new_type,
        "rx": parent["rx"] + offset[0],
        "ry": parent["ry"] + offset[1],
    }

    # Apply symmetry bias: maybe mirror across vertical axis
    if rng.random() < g.meta["symmetry_bias"] and abs(new_node["rx"]) > 0.3:
        mirror_node = {
            "type": new_type,
            "rx": -new_node["rx"],
            "ry": new_node["ry"],
        }
        g.body_nodes.append(new_node)
        g.body_nodes.append(mirror_node)
        new_idx = n
        mirror_idx = n + 1

        # Connect both to parent
        edge_type = int(EdgeType.MUSCLE) if new_type == int(NodeType.MUSCLE_ANCHOR) else int(EdgeType.BONE)
        g.body_edges.append({"from": parent_idx, "to": new_idx, "type": edge_type})
        g.body_edges.append({"from": parent_idx, "to": mirror_idx, "type": edge_type})
    else:
        g.body_nodes.append(new_node)
        new_idx = n
        edge_type = int(EdgeType.MUSCLE) if new_type == int(NodeType.MUSCLE_ANCHOR) else int(EdgeType.BONE)
        g.body_edges.append({"from": parent_idx, "to": new_idx, "type": edge_type})


def _body_remove_node(g: Genome, rng: np.random.Generator) -> None:
    """Remove a node (not the core) and its edges, if body stays connected."""
    n = len(g.body_nodes)
    # Don't remove core
    removable = [i for i in range(n) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not removable:
        return

    target = int(rng.choice(removable))

    # Test connectivity without this node
    new_nodes = [nd for i, nd in enumerate(g.body_nodes) if i != target]
    # Remap edge indices
    idx_map = {}
    new_i = 0
    for i in range(n):
        if i != target:
            idx_map[i] = new_i
            new_i += 1

    new_edges = []
    for e in g.body_edges:
        if e["from"] != target and e["to"] != target:
            new_edges.append({
                "from": idx_map[e["from"]],
                "to": idx_map[e["to"]],
                "type": e["type"],
            })

    edge_tuples = [(e["from"], e["to"], e["type"]) for e in new_edges]
    if is_connected(len(new_nodes), edge_tuples):
        g.body_nodes = new_nodes
        g.body_edges = new_edges


def _body_change_type(g: Genome, rng: np.random.Generator) -> None:
    """Change a node's type (not core)."""
    changeable = [i for i in range(len(g.body_nodes)) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not changeable:
        return

    target = int(rng.choice(changeable))
    new_type = int(rng.choice([
        NodeType.BONE, NodeType.MUSCLE_ANCHOR, NodeType.SENSOR,
        NodeType.MOUTH, NodeType.FAT, NodeType.ARMOR,
    ]))
    g.body_nodes[target]["type"] = new_type

    # If changed to/from MUSCLE_ANCHOR, update connected edge types
    for e in g.body_edges:
        if e["from"] == target or e["to"] == target:
            other = e["to"] if e["from"] == target else e["from"]
            other_type = g.body_nodes[other]["type"]
            # Muscle edges require at least one MUSCLE_ANCHOR endpoint
            if e["type"] == int(EdgeType.MUSCLE):
                if new_type != int(NodeType.MUSCLE_ANCHOR) and other_type != int(NodeType.MUSCLE_ANCHOR):
                    e["type"] = int(EdgeType.TENDON)


def _body_add_remove_edge(g: Genome, rng: np.random.Generator) -> None:
    """Add or remove an edge."""
    n = len(g.body_nodes)
    if rng.random() < 0.5 and len(g.body_edges) > n - 1:
        # Remove a random edge (if body stays connected)
        idx = int(rng.integers(0, len(g.body_edges)))
        test_edges = [e for i, e in enumerate(g.body_edges) if i != idx]
        edge_tuples = [(e["from"], e["to"], e["type"]) for e in test_edges]
        if is_connected(n, edge_tuples):
            g.body_edges.pop(idx)
    else:
        # Add edge between two unconnected nodes
        existing = {(e["from"], e["to"]) for e in g.body_edges}
        existing |= {(e["to"], e["from"]) for e in g.body_edges}
        candidates = [
            (i, j) for i in range(n) for j in range(i + 1, n)
            if (i, j) not in existing
        ]
        if candidates:
            a, b = candidates[int(rng.integers(0, len(candidates)))]
            # Choose edge type based on endpoint types
            if (g.body_nodes[a]["type"] == int(NodeType.MUSCLE_ANCHOR) or
                    g.body_nodes[b]["type"] == int(NodeType.MUSCLE_ANCHOR)):
                etype = int(EdgeType.MUSCLE)
            else:
                etype = int(rng.choice([EdgeType.BONE, EdgeType.TENDON]))
            g.body_edges.append({"from": a, "to": b, "type": etype})


def _body_perturb_position(g: Genome, rng: np.random.Generator) -> None:
    """Slightly shift a node's relative position."""
    # Don't move core (it's always at 0,0)
    movable = [i for i in range(len(g.body_nodes)) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not movable:
        return
    target = int(rng.choice(movable))
    g.body_nodes[target]["rx"] += rng.normal(0, 0.3)
    g.body_nodes[target]["ry"] += rng.normal(0, 0.3)


def _resize_brain_for_body(
    g: Genome,
    brain_config: BrainConfig,
    rng: np.random.Generator,
) -> None:
    """Resize brain weight matrices if body I/O changed."""
    n_sensors = sum(1 for n in g.body_nodes if n["type"] == int(NodeType.SENSOR))
    n_muscles = sum(1 for e in g.body_edges if e["type"] == int(EdgeType.MUSCLE))

    n_inputs = min((n_sensors * 3) + 2 + n_muscles, brain_config.max_inputs)
    n_outputs = min(n_muscles + 2, brain_config.max_outputs)
    total_in = n_inputs + g.brain_n_memory
    total_out = n_outputs + g.brain_n_memory

    old_in, old_h = g.brain_weights_ih.shape
    old_h2, old_out = g.brain_weights_ho.shape
    h = g.brain_hidden_size

    # Resize input->hidden if input count changed
    if total_in != old_in:
        new_w = np.zeros((total_in, h))
        copy_rows = min(total_in, old_in)
        new_w[:copy_rows, :] = g.brain_weights_ih[:copy_rows, :]
        # Initialize new rows near zero
        if total_in > old_in:
            new_w[old_in:, :] = rng.normal(0, 0.01, (total_in - old_in, h))
        g.brain_weights_ih = new_w

    # Resize hidden->output if output count changed
    if total_out != old_out:
        new_w = np.zeros((h, total_out))
        copy_cols = min(total_out, old_out)
        new_w[:, :copy_cols] = g.brain_weights_ho[:, :copy_cols]
        if total_out > old_out:
            new_w[:, old_out:] = rng.normal(0, 0.01, (h, total_out - old_out))
        g.brain_weights_ho = new_w

        new_b = np.zeros(total_out)
        new_b[:copy_cols] = g.brain_bias_o[:copy_cols]
        g.brain_bias_o = new_b


def crossover(
    parent1: Genome,
    parent2: Genome,
    brain_config: BrainConfig,
    rng: np.random.Generator,
) -> Genome:
    """Create offspring genome from two parents via crossover."""
    # Body: use one parent's body plan as template
    if rng.random() < 0.5:
        body_parent, brain_donor = parent1, parent2
    else:
        body_parent, brain_donor = parent2, parent1

    child = body_parent.clone()

    # Brain weights: randomly pick from either parent per weight
    # Only possible if shapes match (same body plan)
    if (child.brain_weights_ih.shape == brain_donor.brain_weights_ih.shape and
            child.brain_weights_ho.shape == brain_donor.brain_weights_ho.shape):
        mask_ih = rng.random(child.brain_weights_ih.shape) < 0.5
        child.brain_weights_ih[mask_ih] = brain_donor.brain_weights_ih[mask_ih]

        mask_ho = rng.random(child.brain_weights_ho.shape) < 0.5
        child.brain_weights_ho[mask_ho] = brain_donor.brain_weights_ho[mask_ho]

        mask_bh = rng.random(child.brain_bias_h.shape) < 0.5
        child.brain_bias_h[mask_bh] = brain_donor.brain_bias_h[mask_bh]

        mask_bo = rng.random(child.brain_bias_o.shape) < 0.5
        child.brain_bias_o[mask_bo] = brain_donor.brain_bias_o[mask_bo]

    # Meta: average
    for key in child.meta:
        if key in brain_donor.meta:
            child.meta[key] = (child.meta[key] + brain_donor.meta[key]) / 2.0

    return child


def genome_distance(g1: Genome, g2: Genome, config: EvolutionConfig) -> float:
    """Compute distance between two genomes for speciation."""
    # Body difference: compare node type histograms + node count difference
    types1 = sorted(n["type"] for n in g1.body_nodes)
    types2 = sorted(n["type"] for n in g2.body_nodes)
    body_dist = abs(len(types1) - len(types2))
    # Count type mismatches for overlapping portion
    for t1, t2 in zip(types1, types2):
        if t1 != t2:
            body_dist += 1

    # Brain weight difference (MSE of shared weights)
    min_rows = min(g1.brain_weights_ih.shape[0], g2.brain_weights_ih.shape[0])
    min_cols = min(g1.brain_weights_ih.shape[1], g2.brain_weights_ih.shape[1])
    if min_rows > 0 and min_cols > 0:
        w1 = g1.brain_weights_ih[:min_rows, :min_cols]
        w2 = g2.brain_weights_ih[:min_rows, :min_cols]
        brain_dist = float(np.mean((w1 - w2) ** 2))
    else:
        brain_dist = 10.0  # max distance if incompatible

    # Topology difference
    topo_dist = abs(g1.brain_hidden_size - g2.brain_hidden_size)
    topo_dist += (0 if g1.brain_activation == g2.brain_activation else 2)
    topo_dist += abs(len(g1.body_edges) - len(g2.body_edges))

    return (
        config.body_distance_weight * body_dist
        + config.brain_distance_weight * brain_dist
        + config.topology_distance_weight * topo_dist
    )
