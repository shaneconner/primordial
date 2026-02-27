"""Run a long simulation and export narrative clips at key moments.

Single-pass approach: records ALL frames to memory during the run,
detects events, then selects and exports clip windows afterward.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine


def run_and_export(
    ticks=10000,
    population=50,
    seed=42,
    output_dir="clips/narrative",
):
    config = SimConfig(initial_population=population, seed=seed)
    engine = Engine(config)
    engine.initialize()

    # Store snapshots every 2 ticks (100 frames per 200-tick clip)
    snapshot_interval = 2
    all_snapshots = {}  # tick -> snapshot

    # Rolling history for event detection
    history = []
    events = []
    known_species = {}
    prev_species_set = set()
    peak_pop = 0
    min_pop_after_peak = 999999

    print(f"Running {ticks} ticks with landscape world ({config.world.width}x{config.world.height})...")
    print(f"Population: {config.initial_population} initial, {config.max_population} max")

    for tick in range(ticks):
        engine.step()
        t = engine.world.tick

        # Capture snapshot every tick
        if t % snapshot_interval == 0:
            all_snapshots[t] = compress_snapshot(engine.world)

        # Event detection every 5 ticks
        if t % 5 == 0:
            stats = engine.world.get_stats()
            pop = stats["population"]
            sp_count = stats["species_count"]
            gen = stats["max_generation"]

            current_species = set()
            species_pops = {}
            for org in engine.world.organisms:
                current_species.add(org.species_id)
                species_pops[org.species_id] = species_pops.get(org.species_id, 0) + 1

            for sp in current_species:
                if sp not in known_species:
                    known_species[sp] = {"first_seen": t, "last_seen": t, "peak_pop": 0}
                known_species[sp]["last_seen"] = t
                known_species[sp]["peak_pop"] = max(
                    known_species[sp]["peak_pop"], species_pops.get(sp, 0)
                )

            history.append({
                "tick": t, "pop": pop, "species": sp_count, "gen": gen,
                "species_set": current_species,
            })

            # Event detection
            new_sp = current_species - prev_species_set
            if new_sp and t > 10:
                for sp in new_sp:
                    events.append({"tick": t, "type": "speciation", "description": f"New species {sp}"})

            extinct = prev_species_set - current_species
            if extinct and t > 10:
                for sp in extinct:
                    if sp in known_species and known_species[sp]["peak_pop"] >= 5:
                        events.append({"tick": t, "type": "extinction", "description": f"Species {sp} extinct"})

            if pop > peak_pop * 1.3 and pop > peak_pop + 10 and t > 50:
                events.append({"tick": t, "type": "boom", "description": f"Pop boom: {peak_pop} -> {pop}"})
            if pop > peak_pop:
                peak_pop = pop
                min_pop_after_peak = pop

            if pop < min_pop_after_peak * 0.6 and min_pop_after_peak > 20 and t > 100:
                events.append({"tick": t, "type": "crash", "description": f"Pop crash: {min_pop_after_peak} -> {pop}"})
                min_pop_after_peak = pop
            if pop < min_pop_after_peak:
                min_pop_after_peak = pop

            if gen > 0 and gen % 10 == 0 and (len(history) < 2 or history[-2]["gen"] < gen):
                events.append({"tick": t, "type": "generation", "description": f"Generation {gen}"})

            if sp_count >= 8 and (len(history) < 2 or history[-2]["species"] < 8):
                events.append({"tick": t, "type": "diversity", "description": f"8+ species ({sp_count})"})

            prev_species_set = current_species

        if t % 1000 == 0:
            stats = engine.world.get_stats()
            print(
                f"  Tick {t}: pop={stats['population']}, species={stats['species_count']}, "
                f"gen={stats['max_generation']}, events={len(events)}, snapshots={len(all_snapshots)}"
            )

    print(f"\nSimulation complete. {len(events)} events, {len(all_snapshots)} snapshots stored.")

    # Select narrative clips
    clips_to_export = select_narrative_clips(events, history, ticks)

    print(f"\nSelected {len(clips_to_export)} narrative clips:")
    for c in clips_to_export:
        print(f"  [{c['id']}] {c['title']} (ticks {c['start']}-{c['end']})")

    # Export clips from stored snapshots
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "world": {"width": config.world.width, "height": config.world.height},
        "total_ticks": ticks,
        "seed": seed,
        "clips": [],
    }

    for c in clips_to_export:
        frames = []
        for t in range(c["start"], c["end"] + 1):
            if t in all_snapshots:
                frames.append(all_snapshots[t])

        if not frames:
            print(f"  WARNING: No frames for {c['id']} (ticks {c['start']}-{c['end']})")
            continue

        clip_data = {
            "id": c["id"],
            "title": c["title"],
            "description": c["description"],
            "world": {"width": config.world.width, "height": config.world.height},
            "tick_range": [c["start"], c["end"]],
            "snapshot_count": len(frames),
            "snapshots": frames,
        }

        filename = f"primordial-{c['id']}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(clip_data, f, separators=(",", ":"))

        size_kb = os.path.getsize(filepath) / 1024
        manifest["clips"].append({
            "id": c["id"],
            "title": c["title"],
            "description": c["description"],
            "file": filename,
            "tick_range": [c["start"], c["end"]],
            "frames": len(frames),
            "size_kb": round(size_kb),
        })
        print(f"  Exported {c['id']}: {len(frames)} frames ({size_kb:.0f} KB)")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Free memory
    del all_snapshots


def select_narrative_clips(events, history, total_ticks):
    """Select a curated set of narrative clips from detected events."""
    clips = []
    clip_len = 200
    used_ranges = []

    def overlaps(start, end):
        for s, e in used_ranges:
            if start < e and end > s:
                return True
        return False

    def add_clip(clip_id, title, desc, center_tick):
        start = max(1, center_tick - clip_len // 3)
        end = start + clip_len
        if end > total_ticks:
            end = total_ticks
            start = max(1, end - clip_len)
        if overlaps(start, end):
            return False
        clips.append({
            "id": clip_id, "title": title, "description": desc,
            "start": start, "end": end,
        })
        used_ranges.append((start, end))
        return True

    # 1. Genesis
    add_clip("genesis", "Genesis",
        "The first organisms are dropped into an empty world.", 75)

    # 2. First speciation
    spec_events = [e for e in events if e["type"] == "speciation"]
    if spec_events:
        add_clip("first-split", "First Divergence",
            "A mutation pushes an offspring past the speciation threshold.",
            spec_events[0]["tick"])

    # 3. First boom
    boom_events = [e for e in events if e["type"] == "boom"]
    if boom_events:
        add_clip("boom", "Population Boom",
            "Successful strategies propagate through the gene pool.",
            boom_events[0]["tick"])

    # 4. Peak diversity
    if history:
        peak_div = max(history, key=lambda h: h["species"])
        if peak_div["species"] > 3:
            add_clip("diversity", "Peak Diversity",
                f"{peak_div['species']} species coexist.",
                peak_div["tick"])

    # 5. First crash
    crash_events = [e for e in events if e["type"] == "crash"]
    if crash_events:
        add_clip("crash", "Population Crash",
            "Resources deplete and the population collapses.",
            crash_events[0]["tick"])

    # 6. Late equilibrium
    late_history = [h for h in history if h["tick"] > total_ticks * 0.7]
    if late_history:
        best_tick = late_history[len(late_history) // 2]["tick"]
        add_clip("equilibrium", "Equilibrium",
            "The ecosystem reaches carrying capacity.",
            best_tick)

    # 7. Highest generation
    if history:
        max_gen_entry = max(history, key=lambda h: h["gen"])
        if max_gen_entry["gen"] > 10:
            add_clip("deep-time", "Deep Time",
                f"Generation {max_gen_entry['gen']}.",
                max_gen_entry["tick"])

    return clips


def compress_snapshot(world):
    """Create a minimal snapshot for web rendering."""
    organisms = []
    for org in world.organisms:
        if not org.alive:
            continue
        nodes = [
            [round(float(org.body.positions[i][0]), 1),
             round(float(org.body.positions[i][1]), 1),
             int(org.body.node_types[i])]
            for i in range(org.body.n_nodes)
        ]
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

    resources = world.resources[:500]
    res_data = [
        [round(r.x, 0), round(r.y, 0), r.resource_type[0]]
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
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    output = sys.argv[3] if len(sys.argv) > 3 else "clips/narrative"
    run_and_export(ticks=ticks, seed=seed, output_dir=output)
