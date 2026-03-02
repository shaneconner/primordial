"""Genome encoding, mutation, and crossover operators."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from .body import Body, EdgeType, NodeType, is_connected

if TYPE_CHECKING:
    from .config import BodyConfig, BrainConfig, EvolutionConfig

_ACTIVATION_OPTIONS = ["tanh", "relu", "sigmoid"]

# All non-core node types available for mutation
_MUTABLE_NODE_TYPES_BASE = [
    NodeType.BONE, NodeType.MUSCLE_ANCHOR, NodeType.SENSOR,
    NodeType.MOUTH, NodeType.FAT, NodeType.ARMOR,
]

# Part 4 adds SIGNAL and STOMACH
_MUTABLE_NODE_TYPES_P4 = _MUTABLE_NODE_TYPES_BASE + [
    NodeType.SIGNAL, NodeType.STOMACH,
]


class Genome:
    """Complete genome encoding body plan, brain architecture, and meta-parameters."""

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
        self.body_nodes = body_nodes
        self.body_edges = body_edges
        self.brain_hidden_size = brain_hidden_size
        self.brain_activation = brain_activation
        self.brain_n_memory = brain_n_memory
        self.brain_weights_ih = brain_weights_ih.copy()
        self.brain_weights_ho = brain_weights_ho.copy()
        self.brain_bias_h = brain_bias_h.copy()
        self.brain_bias_o = brain_bias_o.copy()
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


def _get_mutable_types(body_config: "BodyConfig | None") -> list:
    """Get available node types for mutation based on config."""
    if body_config and getattr(body_config, 'enable_signals', False):
        return _MUTABLE_NODE_TYPES_P4
    return _MUTABLE_NODE_TYPES_BASE


def create_default_genome(
    body_config: BodyConfig,
    brain_config: BrainConfig,
    evolution_config: EvolutionConfig,
    rng: np.random.Generator,
) -> Genome:
    """Create a genome for a default starter organism."""
    spread = body_config.initial_spread
    body_nodes = [
        {"type": int(NodeType.CORE), "rx": 0.0, "ry": 0.0},
        {"type": int(NodeType.SENSOR), "rx": 0.0, "ry": -spread},
        {"type": int(NodeType.MUSCLE_ANCHOR), "rx": -spread, "ry": 0.0},
        {"type": int(NodeType.MUSCLE_ANCHOR), "rx": spread, "ry": 0.0},
        {"type": int(NodeType.MOUTH), "rx": 0.0, "ry": -spread * 0.7},
    ]

    body_edges = [
        {"from": 0, "to": 1, "type": int(EdgeType.BONE)},
        {"from": 0, "to": 2, "type": int(EdgeType.MUSCLE)},
        {"from": 0, "to": 3, "type": int(EdgeType.MUSCLE)},
        {"from": 0, "to": 4, "type": int(EdgeType.TENDON)},
        {"from": 1, "to": 4, "type": int(EdgeType.BONE)},
    ]

    n_sensors = sum(1 for n in body_nodes if n["type"] == int(NodeType.SENSOR))
    n_muscles = sum(1 for e in body_edges if e["type"] == int(EdgeType.MUSCLE))
    n_memory = brain_config.n_memory

    ips = brain_config.inputs_per_sensor
    n_globals = getattr(brain_config, 'n_global_inputs', 2)
    n_inputs = min((n_sensors * ips) + n_globals + n_muscles, brain_config.max_inputs)
    n_outputs = min(n_muscles + brain_config.n_action_outputs, brain_config.max_outputs)
    hidden = brain_config.default_hidden_size

    # Part 4: Recurrent brain adds prev_hidden to input
    recurrent = getattr(brain_config, 'enable_recurrent', False)
    recurrent_size = hidden if recurrent else 0

    total_in = n_inputs + n_memory + recurrent_size
    total_out = n_outputs + n_memory

    scale_ih = np.sqrt(2.0 / (total_in + hidden))
    scale_ho = np.sqrt(2.0 / (hidden + total_out))

    meta = {
        "body_mutation_rate": evolution_config.body_mutation_rate,
        "brain_mutation_rate": evolution_config.brain_mutation_rate,
        "weight_perturb_scale": evolution_config.weight_perturb_scale,
        "symmetry_bias": 0.4,
        "kin_tolerance": 0.3,
    }

    # Part 4: Additional evolvable meta-parameters
    if getattr(body_config, 'enable_signals', False):
        sv_size = getattr(body_config, 'signal_vector_size', 3)
        meta["signal_vector"] = rng.random(sv_size).tolist()
    if getattr(body_config, 'enable_growth', False):
        meta["growth_rate"] = 1.0  # multiplier on growth_interval
    if getattr(body_config, 'enable_signatures', False):
        sig_size = getattr(body_config, 'signature_size', 3)
        meta["identity_signature"] = rng.random(sig_size).tolist()

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
    body_config: "BodyConfig | None" = None,
) -> Genome:
    """Apply mutations to a genome. Returns a new mutated genome."""
    g = genome.clone()

    _mutate_meta(g, evolution_config.meta_mutation_rate, rng, body_config)
    _mutate_brain_weights(g, rng)
    _mutate_brain_topology(g, brain_config, rng)

    if rng.random() < g.meta["body_mutation_rate"]:
        _mutate_body(g, evolution_config, rng, body_config)
        _resize_brain_for_body(g, brain_config, rng)

    # Part 4: Node scale perturbation
    if body_config and getattr(body_config, 'enable_node_scaling', False):
        _mutate_node_scales(g, rng, body_config)

    return g


def _mutate_meta(g: Genome, meta_rate: float, rng: np.random.Generator,
                 body_config: "BodyConfig | None" = None) -> None:
    """Mutate meta-parameters with a fixed rate."""
    for key in ["body_mutation_rate", "brain_mutation_rate", "weight_perturb_scale"]:
        if rng.random() < meta_rate:
            g.meta[key] *= float(np.exp(rng.normal(0, 0.1)))
            g.meta[key] = np.clip(g.meta[key], 0.001, 0.5)

    if rng.random() < meta_rate:
        g.meta["symmetry_bias"] += rng.normal(0, 0.05)
        g.meta["symmetry_bias"] = np.clip(g.meta["symmetry_bias"], 0.0, 1.0)

    if "kin_tolerance" in g.meta and rng.random() < meta_rate:
        g.meta["kin_tolerance"] += rng.normal(0, 0.08)
        g.meta["kin_tolerance"] = float(np.clip(g.meta["kin_tolerance"], 0.0, 1.0))

    # Part 4: Signal vector mutation
    if "signal_vector" in g.meta and rng.random() < meta_rate:
        sv = np.array(g.meta["signal_vector"])
        sv += rng.normal(0, 0.1, sv.shape)
        sv = np.clip(sv, 0.0, 1.0)
        g.meta["signal_vector"] = sv.tolist()

    # Part 4: Growth rate mutation
    if "growth_rate" in g.meta and rng.random() < meta_rate:
        g.meta["growth_rate"] *= float(np.exp(rng.normal(0, 0.1)))
        g.meta["growth_rate"] = float(np.clip(g.meta["growth_rate"], 0.3, 3.0))

    # Part 4: Identity signature (small perturbation, partially heritable)
    if "identity_signature" in g.meta and rng.random() < meta_rate:
        sig = np.array(g.meta["identity_signature"])
        sig += rng.normal(0, 0.05, sig.shape)
        sig = np.clip(sig, 0.0, 1.0)
        g.meta["identity_signature"] = sig.tolist()


def _mutate_brain_weights(g: Genome, rng: np.random.Generator) -> None:
    """Perturb brain weights with Gaussian noise."""
    rate = g.meta["brain_mutation_rate"]
    scale = g.meta["weight_perturb_scale"]

    for arr in [g.brain_weights_ih, g.brain_weights_ho, g.brain_bias_h, g.brain_bias_o]:
        mask = rng.random(arr.shape) < rate
        arr[mask] += rng.normal(0, scale, np.sum(mask))


def _mutate_brain_topology(
    g: Genome,
    brain_config: BrainConfig,
    rng: np.random.Generator,
) -> None:
    """Mutate brain hidden layer size and/or activation function."""
    if rng.random() < brain_config.hidden_size_mutation_rate:
        delta = int(rng.choice([-1, 1]))
        new_size = g.brain_hidden_size + delta
        new_size = max(4, min(brain_config.max_hidden_size, new_size))
        if new_size != g.brain_hidden_size:
            old_h = g.brain_hidden_size
            g.brain_hidden_size = new_size
            _resize_genome_hidden(g, old_h, new_size, rng)

    if rng.random() < brain_config.activation_mutation_rate:
        others = [a for a in _ACTIVATION_OPTIONS if a != g.brain_activation]
        g.brain_activation = str(rng.choice(others))


def _resize_genome_hidden(
    g: Genome, old_size: int, new_size: int, rng: np.random.Generator
) -> None:
    """Resize genome weight matrices when hidden layer changes."""
    rows_ih, _ = g.brain_weights_ih.shape
    _, cols_ho = g.brain_weights_ho.shape
    copy_h = min(old_size, new_size)

    new_wih = np.zeros((rows_ih, new_size))
    new_wih[:, :copy_h] = g.brain_weights_ih[:, :copy_h]
    if new_size > old_size:
        new_wih[:, old_size:] = rng.normal(0, 0.01, (rows_ih, new_size - old_size))
    g.brain_weights_ih = new_wih

    new_who = np.zeros((new_size, cols_ho))
    new_who[:copy_h, :] = g.brain_weights_ho[:copy_h, :]
    if new_size > old_size:
        new_who[old_size:, :] = rng.normal(0, 0.01, (new_size - old_size, cols_ho))
    g.brain_weights_ho = new_who

    new_bh = np.zeros(new_size)
    new_bh[:copy_h] = g.brain_bias_h[:copy_h]
    g.brain_bias_h = new_bh


def _mutate_body(
    g: Genome,
    config: EvolutionConfig,
    rng: np.random.Generator,
    body_config: "BodyConfig | None" = None,
) -> None:
    """Apply a single structural body mutation."""
    n_nodes = len(g.body_nodes)
    roll = rng.random()
    cumulative = 0.0

    cumulative += config.add_node_prob
    if roll < cumulative and n_nodes < config.max_body_nodes:
        _body_add_node(g, rng, body_config)
        return

    cumulative += config.remove_node_prob
    if roll < cumulative and n_nodes > 3:
        _body_remove_node(g, rng)
        return

    cumulative += config.change_type_prob
    if roll < cumulative:
        _body_change_type(g, rng, body_config)
        return

    cumulative += config.add_remove_edge_prob
    if roll < cumulative:
        _body_add_remove_edge(g, rng)
        return

    _body_perturb_position(g, rng, body_config)


def _body_add_node(g: Genome, rng: np.random.Generator, body_config: "BodyConfig | None" = None) -> None:
    """Add a new node connected to an existing node."""
    n = len(g.body_nodes)

    limb_bias = body_config.limb_chain_bias if body_config else 0.0
    if limb_bias > 0 and n > 1 and rng.random() < limb_bias:
        bone_indices = [i for i in range(n) if g.body_nodes[i]["type"] == int(NodeType.BONE)]
        peripheral = [i for i in range(n) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
        if bone_indices:
            parent_idx = int(rng.choice(bone_indices))
        elif peripheral:
            parent_idx = int(rng.choice(peripheral))
        else:
            parent_idx = rng.integers(0, n)
    else:
        parent_idx = rng.integers(0, n)
    parent = g.body_nodes[parent_idx]

    # Get available types (includes SIGNAL/STOMACH for Part 4)
    mutable_types = _get_mutable_types(body_config)

    if parent["type"] == int(NodeType.BONE) and limb_bias > 0 and rng.random() < 0.5:
        new_type = int(rng.choice([
            NodeType.BONE, NodeType.BONE, NodeType.MUSCLE_ANCHOR, NodeType.MOUTH,
        ]))
    else:
        new_type = int(rng.choice(mutable_types))

    sigma = body_config.new_node_offset_sigma if body_config else 1.5
    offset = rng.normal(0, sigma, 2)

    outward_bias = body_config.new_node_outward_bias if body_config else 0.0
    if outward_bias > 0 and (parent["rx"] != 0 or parent["ry"] != 0):
        dist = max(0.01, (parent["rx"] ** 2 + parent["ry"] ** 2) ** 0.5)
        offset[0] += outward_bias * sigma * parent["rx"] / dist
        offset[1] += outward_bias * sigma * parent["ry"] / dist

    new_node = {
        "type": new_type,
        "rx": parent["rx"] + offset[0],
        "ry": parent["ry"] + offset[1],
    }

    # Part 4: Node scale (if enabled)
    if body_config and getattr(body_config, 'enable_node_scaling', False):
        new_node["scale"] = 1.0

    if rng.random() < g.meta["symmetry_bias"] and abs(new_node["rx"]) > 0.3:
        mirror_node = {
            "type": new_type,
            "rx": -new_node["rx"],
            "ry": new_node["ry"],
        }
        if "scale" in new_node:
            mirror_node["scale"] = new_node["scale"]
        g.body_nodes.append(new_node)
        g.body_nodes.append(mirror_node)
        new_idx = n
        mirror_idx = n + 1
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
    removable = [i for i in range(n) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not removable:
        return

    target = int(rng.choice(removable))
    new_nodes = [nd for i, nd in enumerate(g.body_nodes) if i != target]
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


def _body_change_type(g: Genome, rng: np.random.Generator,
                      body_config: "BodyConfig | None" = None) -> None:
    """Change a node's type (not core)."""
    changeable = [i for i in range(len(g.body_nodes)) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not changeable:
        return

    mutable_types = _get_mutable_types(body_config)
    target = int(rng.choice(changeable))
    new_type = int(rng.choice(mutable_types))
    g.body_nodes[target]["type"] = new_type

    for e in g.body_edges:
        if e["from"] == target or e["to"] == target:
            other = e["to"] if e["from"] == target else e["from"]
            other_type = g.body_nodes[other]["type"]
            if e["type"] == int(EdgeType.MUSCLE):
                if new_type != int(NodeType.MUSCLE_ANCHOR) and other_type != int(NodeType.MUSCLE_ANCHOR):
                    e["type"] = int(EdgeType.TENDON)


def _body_add_remove_edge(g: Genome, rng: np.random.Generator) -> None:
    """Add or remove an edge."""
    n = len(g.body_nodes)
    if rng.random() < 0.5 and len(g.body_edges) > n - 1:
        idx = int(rng.integers(0, len(g.body_edges)))
        test_edges = [e for i, e in enumerate(g.body_edges) if i != idx]
        edge_tuples = [(e["from"], e["to"], e["type"]) for e in test_edges]
        if is_connected(n, edge_tuples):
            g.body_edges.pop(idx)
    else:
        existing = {(e["from"], e["to"]) for e in g.body_edges}
        existing |= {(e["to"], e["from"]) for e in g.body_edges}
        candidates = [
            (i, j) for i in range(n) for j in range(i + 1, n)
            if (i, j) not in existing
        ]
        if candidates:
            a, b = candidates[int(rng.integers(0, len(candidates)))]
            if (g.body_nodes[a]["type"] == int(NodeType.MUSCLE_ANCHOR) or
                    g.body_nodes[b]["type"] == int(NodeType.MUSCLE_ANCHOR)):
                etype = int(EdgeType.MUSCLE)
            else:
                etype = int(rng.choice([EdgeType.BONE, EdgeType.TENDON]))
            g.body_edges.append({"from": a, "to": b, "type": etype})


def _body_perturb_position(g: Genome, rng: np.random.Generator, body_config: "BodyConfig | None" = None) -> None:
    """Slightly shift a node's relative position."""
    movable = [i for i in range(len(g.body_nodes)) if g.body_nodes[i]["type"] != int(NodeType.CORE)]
    if not movable:
        return
    target = int(rng.choice(movable))
    sigma = body_config.position_perturb_sigma if body_config else 0.5
    g.body_nodes[target]["rx"] += rng.normal(0, sigma)
    g.body_nodes[target]["ry"] += rng.normal(0, sigma)


def _mutate_node_scales(g: Genome, rng: np.random.Generator, body_config: BodyConfig) -> None:
    """Part 4: Perturb individual node scale factors."""
    min_s = body_config.min_node_scale
    max_s = body_config.max_node_scale
    for node in g.body_nodes:
        if node["type"] == int(NodeType.CORE):
            continue
        if "scale" not in node:
            node["scale"] = 1.0
        if rng.random() < 0.1:  # 10% chance per node
            node["scale"] += rng.normal(0, 0.1)
            node["scale"] = float(np.clip(node["scale"], min_s, max_s))


def _resize_brain_for_body(
    g: Genome,
    brain_config: BrainConfig,
    rng: np.random.Generator,
) -> None:
    """Resize brain weight matrices if body I/O changed."""
    n_sensors = sum(1 for n in g.body_nodes if n["type"] == int(NodeType.SENSOR))
    n_muscles = sum(1 for e in g.body_edges if e["type"] == int(EdgeType.MUSCLE))

    ips = brain_config.inputs_per_sensor
    n_globals = getattr(brain_config, 'n_global_inputs', 2)
    n_inputs = min((n_sensors * ips) + n_globals + n_muscles, brain_config.max_inputs)
    n_outputs = min(n_muscles + brain_config.n_action_outputs, brain_config.max_outputs)

    # Part 4: Recurrent adds hidden_size to input
    recurrent = getattr(brain_config, 'enable_recurrent', False)
    recurrent_size = g.brain_hidden_size if recurrent else 0

    total_in = n_inputs + g.brain_n_memory + recurrent_size
    total_out = n_outputs + g.brain_n_memory

    old_in, old_h = g.brain_weights_ih.shape
    old_h2, old_out = g.brain_weights_ho.shape
    h = g.brain_hidden_size

    if total_in != old_in:
        new_w = np.zeros((total_in, h))
        copy_rows = min(total_in, old_in)
        new_w[:copy_rows, :] = g.brain_weights_ih[:copy_rows, :]
        if total_in > old_in:
            new_w[old_in:, :] = rng.normal(0, 0.01, (total_in - old_in, h))
        g.brain_weights_ih = new_w

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
    if rng.random() < 0.5:
        body_parent, brain_donor = parent1, parent2
    else:
        body_parent, brain_donor = parent2, parent1

    child = body_parent.clone()

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

    # Meta: average scalars, blend vectors
    for key in child.meta:
        if key not in brain_donor.meta:
            continue
        val_c = child.meta[key]
        val_d = brain_donor.meta[key]
        if isinstance(val_c, list) and isinstance(val_d, list):
            # Blend vector meta (signal_vector, identity_signature)
            child.meta[key] = [
                (a + b) / 2.0 for a, b in zip(val_c, val_d)
            ]
        elif isinstance(val_c, (int, float)) and isinstance(val_d, (int, float)):
            child.meta[key] = (val_c + val_d) / 2.0

    return child


def genome_distance(g1: Genome, g2: Genome, config: EvolutionConfig) -> float:
    """Compute distance between two genomes for speciation."""
    types1 = sorted(n["type"] for n in g1.body_nodes)
    types2 = sorted(n["type"] for n in g2.body_nodes)
    body_dist = abs(len(types1) - len(types2))
    for t1, t2 in zip(types1, types2):
        if t1 != t2:
            body_dist += 1

    min_rows = min(g1.brain_weights_ih.shape[0], g2.brain_weights_ih.shape[0])
    min_cols = min(g1.brain_weights_ih.shape[1], g2.brain_weights_ih.shape[1])
    if min_rows > 0 and min_cols > 0:
        w1 = g1.brain_weights_ih[:min_rows, :min_cols]
        w2 = g2.brain_weights_ih[:min_rows, :min_cols]
        brain_dist = float(np.mean((w1 - w2) ** 2))
    else:
        brain_dist = 10.0

    topo_dist = abs(g1.brain_hidden_size - g2.brain_hidden_size)
    topo_dist += (0 if g1.brain_activation == g2.brain_activation else 2)
    topo_dist += abs(len(g1.body_edges) - len(g2.body_edges))

    # Part 4: Signal vector distance contributes to speciation
    if "signal_vector" in g1.meta and "signal_vector" in g2.meta:
        sv1 = np.array(g1.meta["signal_vector"])
        sv2 = np.array(g2.meta["signal_vector"])
        if sv1.shape == sv2.shape:
            topo_dist += float(np.sum((sv1 - sv2) ** 2))

    return (
        config.body_distance_weight * body_dist
        + config.brain_distance_weight * brain_dist
        + config.topology_distance_weight * topo_dist
    )
