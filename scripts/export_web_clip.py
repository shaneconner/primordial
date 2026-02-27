"""Export a lightweight clip optimized for web D3.js viewer.

Strips unnecessary precision, limits resource data, and compresses
the output for fast loading on the portfolio site.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine


def run_and_export(
    ticks: int = 2000,
    population: int = 40,
    seed: int = 42,
    snapshot_every: int = 5,
    output_path: str = "clips/web_clip.json",
):
    """Run simulation and export a web-optimized clip."""
    config = SimConfig(initial_population=population, seed=seed)
    engine = Engine(config)
    engine.initialize()

    snapshots = []

    print(f"Running {ticks} ticks...")
    for tick in range(ticks):
        engine.step()

        if engine.world.tick % snapshot_every == 0:
            snap = _compress_snapshot(engine.world)
            snapshots.append(snap)

        if engine.world.tick % 500 == 0:
            stats = engine.world.get_stats()
            print(
                f"  Tick {engine.world.tick}: pop={stats['population']}, "
                f"species={stats['species_count']}, gen={stats['max_generation']}"
            )

    clip = {
        "name": f"web_clip_s{seed}",
        "world": {"width": config.world.width, "height": config.world.height},
        "snapshot_count": len(snapshots),
        "tick_range": [snapshots[0]["t"], snapshots[-1]["t"]] if snapshots else [0, 0],
        "snapshots": snapshots,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clip, f, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nExported {len(snapshots)} frames to {output_path} ({size_kb:.0f} KB)")


def _compress_snapshot(world) -> dict:
    """Create a minimal snapshot for web rendering."""
    organisms = []
    for org in world.organisms:
        if not org.alive:
            continue
        # Compact node format: [x, y, type]
        nodes = [
            [round(float(org.body.positions[i][0]), 1),
             round(float(org.body.positions[i][1]), 1),
             int(org.body.node_types[i])]
            for i in range(org.body.n_nodes)
        ]
        # Compact edge format: [from, to, type]
        edges = [
            [int(org.body.edge_from[i]),
             int(org.body.edge_to[i]),
             int(org.body.edge_types[i])]
            for i in range(org.body.n_edges)
        ]
        organisms.append({
            "id": org.id,
            "sp": org.species_id,
            "e": round(org.energy, 1),
            "g": org.generation,
            "n": nodes,
            "ed": edges,
        })

    # Sample resources (limit to 500 for web perf)
    resources = world.resources[:500]
    res_data = [
        [round(r.x, 0), round(r.y, 0), r.resource_type[0]]  # 'p', 'm', 'n'
        for r in resources
    ]

    return {
        "t": world.tick,
        "o": organisms,
        "r": res_data,
        "s": {
            "pop": len(world.organisms),
            "sp": len({o.species_id for o in world.organisms}),
            "gen": max((o.generation for o in world.organisms), default=0),
            "food": len(world.resources),
        },
    }


if __name__ == "__main__":
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    output = sys.argv[2] if len(sys.argv) > 2 else "clips/web_clip.json"
    snap_every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    run_and_export(ticks=ticks, output_path=output, snapshot_every=snap_every)
