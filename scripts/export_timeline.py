"""Export lightweight timeline data for D3.js visualizations.

Runs the simulation (Pass 1 only, no frame capture) and collects
per-species populations and node type counts at regular intervals.
Output is a compact JSON suitable for streamgraphs and line charts.
"""

import sys
import os
import json
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine


NODE_TYPE_NAMES = ["core", "bone", "muscle", "sensor", "mouth", "fat", "armor"]


def export_timeline(
    ticks=100000,
    seed=42,
    sample_interval=100,
    output_path=None,
):
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "shaneconner-site", "data", "primordial-timeline.json",
        )

    config = SimConfig(seed=seed)
    engine = Engine(config)
    engine.initialize()

    t0 = time.time()
    print(f"=== Timeline export: {ticks} ticks, sample every {sample_interval} ===")

    # Accumulate timeline data
    tick_list = []
    pop_list = []
    species_count_list = []
    gen_list = []
    avg_nodes_list = []
    node_type_timeline = {name: [] for name in NODE_TYPE_NAMES}

    # Track all species ever seen (for streamgraph)
    all_species = set()
    species_pop_per_tick = []  # list of dicts {sp_id: count}

    for tick in range(ticks):
        engine.step()
        t = engine.world.tick

        if t % sample_interval == 0:
            alive = [o for o in engine.world.organisms if o.alive]
            pop = len(alive)

            # Per-species population
            sp_pops = Counter()
            # Node type counts across all organisms
            node_counts = Counter()
            max_gen = 0
            total_nodes = 0

            for org in alive:
                sp_pops[org.species_id] += 1
                if org.generation > max_gen:
                    max_gen = org.generation
                total_nodes += org.body.n_nodes
                for nt in org.body.node_types:
                    node_counts[int(nt)] += 1

            all_species.update(sp_pops.keys())

            tick_list.append(t)
            pop_list.append(pop)
            species_count_list.append(len(sp_pops))
            gen_list.append(max_gen)
            avg_nodes_list.append(round(total_nodes / max(pop, 1), 2))

            for i, name in enumerate(NODE_TYPE_NAMES):
                node_type_timeline[name].append(node_counts.get(i, 0))

            species_pop_per_tick.append(dict(sp_pops))

        if t % 1000 == 0:
            elapsed = time.time() - t0
            tps = t / elapsed if elapsed > 0 else 0
            pop = len([o for o in engine.world.organisms if o.alive])
            print(f"  Tick {t}: pop={pop}, {tps:.0f} t/s")

    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.0f}s")

    # Build species streamgraph data
    # Only include species with peak pop >= 3 to keep file small
    species_peak = Counter()
    for sp_tick in species_pop_per_tick:
        for sp, count in sp_tick.items():
            species_peak[sp] = max(species_peak[sp], count)

    top_species = sorted(
        [sp for sp, peak in species_peak.items() if peak >= 3],
        key=lambda sp: species_peak[sp],
        reverse=True,
    )[:50]  # cap at top 50

    species_populations = {}
    for sp in top_species:
        species_populations[sp] = [
            sp_tick.get(sp, 0) for sp_tick in species_pop_per_tick
        ]

    # Build output
    data = {
        "ticks": tick_list,
        "population": pop_list,
        "species_count": species_count_list,
        "max_generation": gen_list,
        "avg_nodes": avg_nodes_list,
        "node_type_counts": node_type_timeline,
        "species_populations": species_populations,
        "total_species_seen": len(all_species),
        "seed": seed,
        "total_ticks": ticks,
        "sample_interval": sample_interval,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported: {output_path} ({size_kb:.0f} KB)")
    print(f"  {len(tick_list)} data points, {len(top_species)} species tracked")


if __name__ == "__main__":
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    output = sys.argv[3] if len(sys.argv) > 3 else None
    export_timeline(ticks=ticks, seed=seed, output_path=output)
