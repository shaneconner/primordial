"""Single-pass full simulation export.

Runs the simulation once and outputs:
  1. Full frame archive (JSONL) - every Nth tick for video rendering
  2. Timeline data (JSON) - for D3 charts on the website
  3. Narrative clips (JSON) - curated moments for the web viewer
  4. Event log (JSON) - all detected evolutionary events

Usage:
  python scripts/export_full.py [ticks] [seed] [--part2] [--part3] [--part4] [--frame-interval N]
  python scripts/export_full.py 100000 42                    # Part 1 defaults
  python scripts/export_full.py 300000 137 --part2           # Part 2
  python scripts/export_full.py 300000 271 --part3           # Part 3
  python scripts/export_full.py 500000 314 --part4           # Part 4
  python scripts/export_full.py 100000 42 --frame-interval 1 # Every tick (full video)
"""

import sys
import os
import json
import time
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine

NODE_TYPE_NAMES = ["core", "bone", "muscle", "sensor", "mouth", "fat", "armor", "signal", "stomach"]

SITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "shaneconner-site",
)


def compress_snapshot(world):
    """Create a minimal snapshot for web/video rendering."""
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
            "pop": len([o for o in world.organisms if o.alive]),
            "sp": len({o.species_id for o in world.organisms if o.alive}),
            "gen": max((o.generation for o in world.organisms if o.alive), default=0),
            "food": len(world.resources),
        },
    }


def select_narrative_clips(events, history, total_ticks, clip_window):
    """Select curated narrative clips from detected events."""
    clips = []
    used_ranges = []

    def overlaps(start, end):
        for s, e in used_ranges:
            if start < e and end > s:
                return True
        return False

    def add_clip(clip_id, title, desc, center_tick, window=None):
        w = window if window is not None else clip_window
        start = max(1, center_tick - w // 3)
        end = start + w
        if end > total_ticks:
            end = total_ticks
            start = max(1, end - w)
        attempts = 0
        while overlaps(start, end) and attempts < 5:
            for s, e in used_ranges:
                if start < e and end > s:
                    start = e + 1
                    end = start + w
                    break
            attempts += 1
        if end > total_ticks:
            return False
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
        "The first organisms enter the world.",
        150, window=clip_window // 2)

    # 2. First speciation
    spec_events = [e for e in events if e["type"] == "speciation"]
    if spec_events:
        add_clip("first-split", "First Divergence",
            "Genetic drift produces the first distinct species.",
            spec_events[0]["tick"])

    # 3. First boom
    boom_events = [e for e in events if e["type"] == "boom"]
    if boom_events:
        add_clip("boom", "The Bloom",
            "Successful body plans propagate through the population.",
            boom_events[0]["tick"])

    # 4. First crash
    crash_events = [e for e in events if e["type"] == "crash"]
    if crash_events:
        add_clip("crash", "Collapse",
            "Resources deplete and the population crashes.",
            crash_events[0]["tick"])

    # 5. Peak diversity
    if history:
        peak_div = max(history, key=lambda h: h["species"])
        if peak_div["species"] > 3:
            add_clip("diversity", "Radiation",
                f"{peak_div['species']} species coexist.",
                peak_div["tick"])

    # 6. Body evolution
    body_events = [e for e in events if e["type"] == "body_evolution"]
    if body_events:
        add_clip("body-evolution", "Body Diversity",
            "Organisms evolve distinct body plans.",
            body_events[-1]["tick"])

    # 7. Generation milestone (midpoint)
    gen_events = [e for e in events if e["type"] == "generation"]
    mid_gen = [e for e in gen_events if e["tick"] > total_ticks * 0.3 and e["tick"] < total_ticks * 0.7]
    if mid_gen:
        add_clip("generation-milestone", "Generational Shift",
            "Hundreds of generations have refined neural networks and body plans.",
            mid_gen[len(mid_gen) // 2]["tick"])

    # 8. Equilibrium
    late_quarter = [h for h in history if h["tick"] > total_ticks * 0.75]
    if late_quarter:
        mid_tick = late_quarter[len(late_quarter) // 2]["tick"]
        add_clip("equilibrium", "Balance",
            "The ecosystem reaches a dynamic equilibrium.",
            mid_tick)

    # 9. Late diversity peak
    late_history = [h for h in history if h["tick"] > total_ticks * 0.5]
    if late_history:
        late_peak = max(late_history, key=lambda h: h["species"])
        if late_peak["species"] > 5:
            add_clip("late-diversity", "Late Radiation",
                f"A late wave of speciation produces {late_peak['species']} species.",
                late_peak["tick"])

    # 10. Deep time (near end)
    if history:
        max_gen_entry = max(history, key=lambda h: h["gen"])
        if max_gen_entry["gen"] > 10:
            add_clip("deep-time", "Deep Time",
                f"Generation {max_gen_entry['gen']}.",
                max_gen_entry["tick"])

    clips.sort(key=lambda c: c["start"])
    return clips


def run_full_export(
    ticks=100000,
    seed=42,
    part2=False,
    part3=False,
    part4=False,
    frame_interval=2,
    sample_interval=100,
    clip_frames=500,
    clip_snapshot_interval=2,
):
    """Single-pass export: full frames + timeline + events + narrative clips."""

    # Config
    if part4:
        config = SimConfig.part4(seed=seed)
        part_label = "Part 4"
        part_suffix = "-p4"
    elif part3:
        config = SimConfig.part3(seed=seed)
        part_label = "Part 3"
        part_suffix = "-p3"
    elif part2:
        config = SimConfig.part2(seed=seed)
        part_label = "Part 2"
        part_suffix = "-p2"
    else:
        config = SimConfig(seed=seed)
        part_label = "Part 1"
        part_suffix = ""
    config.max_ticks = ticks

    # Output paths
    frames_dir = os.path.join(SITE_DIR, "data", f"full-frames{part_suffix}")
    timeline_path = os.path.join(SITE_DIR, "data", f"primordial-timeline{part_suffix}.json")
    narrative_dir = os.path.join(SITE_DIR, "data", f"narrative{part_suffix}")
    events_path = os.path.join(SITE_DIR, "data", f"primordial-events{part_suffix}.json")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(narrative_dir, exist_ok=True)

    t0 = time.time()

    print(f"=== SINGLE-PASS EXPORT: {part_label} ===")
    print(f"  Ticks: {ticks}, Seed: {seed}")
    print(f"  Frame interval: every {frame_interval} ticks")
    print(f"  Timeline sample: every {sample_interval} ticks")
    if part2 or part3 or part4:
        print(f"  Config: spread={config.body.initial_spread}, "
              f"sensors={config.brain.inputs_per_sensor}ips, "
              f"actions={config.brain.n_action_outputs}, max_age={config.evolution.max_age}")
    if part3 or part4:
        print(f"  Part 3 features: muscle_speed={config.body.muscle_speed_scaling}, "
              f"bone_reach={config.body.bone_reach_scaling}, "
              f"seasons={config.body.enable_seasons}, "
              f"kin_recognition={config.body.enable_kin_recognition}, "
              f"attack_dmg={config.evolution.attack_damage_per_mouth}")
    if part4:
        print(f"  Part 4 features: terrain={config.body.enable_terrain}, "
              f"food_types={config.body.enable_food_types}, "
              f"day_night={config.body.enable_day_night}, "
              f"signals={config.body.enable_signals}, "
              f"recurrent={config.brain.enable_recurrent}, "
              f"repulsion={config.body.enable_organism_repulsion}")
    print(f"  Output: {frames_dir}")
    print()

    engine = Engine(config)
    engine.initialize()

    # ── Timeline accumulators ──
    tick_list = []
    pop_list = []
    species_count_list = []
    gen_list = []
    avg_nodes_list = []
    node_type_timeline = {name: [] for name in NODE_TYPE_NAMES}
    all_species = set()
    species_pop_per_tick = []

    # ── Event detection state ──
    history = []
    events = []
    known_species = {}
    prev_species_set = set()
    peak_pop = 0
    min_pop_after_peak = 999999
    max_body_complexity = 0

    # ── Frame streaming ──
    # Write frames in JSONL chunks (one JSON object per line)
    chunk_size = 5000  # frames per chunk file
    current_chunk_frames = []
    current_chunk_idx = 0
    total_frames = 0

    # ── Narrative clip buffering ──
    # We'll collect clip frames during the run and write at the end
    # First pass: just accumulate events/history to select clips
    # But we also need frames... so we buffer narrative-relevant frames
    # Strategy: write ALL frames to JSONL, extract clips at end
    # For efficiency, also buffer clip frames in memory (they're small)

    frames_file = None
    frames_path = os.path.join(frames_dir, f"frames-{current_chunk_idx:04d}.jsonl")
    frames_file = open(frames_path, "w")

    def flush_chunk():
        nonlocal frames_file, current_chunk_idx, frames_path
        if frames_file:
            frames_file.close()
        current_chunk_idx += 1
        frames_path = os.path.join(frames_dir, f"frames-{current_chunk_idx:04d}.jsonl")
        frames_file = open(frames_path, "w")

    # ══════════════════════════════════════════
    # MAIN SIMULATION LOOP
    # ══════════════════════════════════════════

    for tick in range(ticks):
        engine.step()
        t = engine.world.tick

        # ── Frame capture ──
        if t % frame_interval == 0:
            snap = compress_snapshot(engine.world)
            line = json.dumps(snap, separators=(",", ":"))
            frames_file.write(line + "\n")
            total_frames += 1

            if total_frames % chunk_size == 0:
                flush_chunk()

        # ── Timeline sampling ──
        if t % sample_interval == 0:
            alive = [o for o in engine.world.organisms if o.alive]
            pop = len(alive)
            sp_pops = Counter()
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

        # ── Event detection (every 5 ticks) ──
        if t % 5 == 0:
            pop = len([o for o in engine.world.organisms if o.alive])
            current_species = set()
            species_pops = {}
            total_nodes = 0
            max_gen = 0
            for org in engine.world.organisms:
                if not org.alive:
                    continue
                current_species.add(org.species_id)
                species_pops[org.species_id] = species_pops.get(org.species_id, 0) + 1
                total_nodes += org.body.n_nodes
                if org.generation > max_gen:
                    max_gen = org.generation

            sp_count = len(current_species)
            avg_nodes = total_nodes / max(pop, 1)

            for sp in current_species:
                if sp not in known_species:
                    known_species[sp] = {"first_seen": t, "last_seen": t, "peak_pop": 0}
                known_species[sp]["last_seen"] = t
                known_species[sp]["peak_pop"] = max(
                    known_species[sp]["peak_pop"], species_pops.get(sp, 0)
                )

            history.append({
                "tick": t, "pop": pop, "species": sp_count,
                "gen": max_gen, "avg_nodes": round(avg_nodes, 1),
            })

            # Detect events
            new_sp = current_species - prev_species_set
            if new_sp and t > 10:
                for sp in new_sp:
                    events.append({"tick": t, "type": "speciation", "description": f"New species {sp}"})

            extinct = prev_species_set - current_species
            if extinct and t > 10:
                for sp in extinct:
                    if sp in known_species and known_species[sp]["peak_pop"] >= 3:
                        events.append({"tick": t, "type": "extinction",
                            "description": f"Species {sp} extinct (peak {known_species[sp]['peak_pop']})"})

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

            if max_gen > 0 and max_gen % 10 == 0 and (len(history) < 2 or history[-2]["gen"] < max_gen):
                events.append({"tick": t, "type": "generation", "description": f"Generation {max_gen}"})

            if sp_count >= 8 and (len(history) < 2 or history[-2]["species"] < 8):
                events.append({"tick": t, "type": "diversity", "description": f"8+ species ({sp_count})"})
            if sp_count >= 15 and (len(history) < 2 or history[-2]["species"] < 15):
                events.append({"tick": t, "type": "diversity_high", "description": f"15+ species ({sp_count})"})
            if sp_count >= 25 and (len(history) < 2 or history[-2]["species"] < 25):
                events.append({"tick": t, "type": "diversity_peak", "description": f"25+ species ({sp_count})"})

            if avg_nodes > max_body_complexity + 1.0 and avg_nodes > 6.0:
                events.append({"tick": t, "type": "body_evolution", "description": f"Avg body complexity: {avg_nodes:.1f} nodes"})
                max_body_complexity = avg_nodes

            prev_species_set = current_species

        # ── Progress ──
        if t % 1000 == 0:
            elapsed = time.time() - t0
            tps = t / elapsed if elapsed > 0 else 0
            pop = len([o for o in engine.world.organisms if o.alive])
            sp_count = len({o.species_id for o in engine.world.organisms if o.alive})
            max_gen = max((o.generation for o in engine.world.organisms if o.alive), default=0)
            print(f"  Tick {t:>7}: pop={pop:>3}, species={sp_count:>2}, gen={max_gen:>3}, "
                  f"frames={total_frames:>6}, events={len(events):>4}, {tps:.0f} t/s")

    # Close last chunk
    if frames_file:
        frames_file.close()

    sim_elapsed = time.time() - t0
    print(f"\nSimulation complete: {sim_elapsed:.0f}s ({sim_elapsed/60:.1f} min)")
    print(f"  {total_frames} frames captured")
    print(f"  {len(events)} events detected")

    # ══════════════════════════════════════════
    # EXPORT TIMELINE DATA
    # ══════════════════════════════════════════
    print("\nExporting timeline data...")

    species_peak = Counter()
    for sp_tick in species_pop_per_tick:
        for sp, count in sp_tick.items():
            species_peak[sp] = max(species_peak[sp], count)

    top_species = sorted(
        [sp for sp, peak in species_peak.items() if peak >= 3],
        key=lambda sp: species_peak[sp],
        reverse=True,
    )[:50]

    species_populations = {}
    for sp in top_species:
        species_populations[sp] = [
            sp_tick.get(sp, 0) for sp_tick in species_pop_per_tick
        ]

    timeline_data = {
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

    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    with open(timeline_path, "w") as f:
        json.dump(timeline_data, f, separators=(",", ":"))

    size_kb = os.path.getsize(timeline_path) / 1024
    print(f"  Timeline: {timeline_path} ({size_kb:.0f} KB)")

    # ══════════════════════════════════════════
    # EXPORT EVENTS
    # ══════════════════════════════════════════
    with open(events_path, "w") as f:
        json.dump({"seed": seed, "total_ticks": ticks, "events": events}, f, indent=2)
    print(f"  Events: {events_path} ({len(events)} events)")

    # ══════════════════════════════════════════
    # EXTRACT NARRATIVE CLIPS FROM FULL FRAMES
    # ══════════════════════════════════════════
    print("\nSelecting narrative clips...")

    clip_window = clip_frames * clip_snapshot_interval
    clips_to_export = select_narrative_clips(events, history, ticks, clip_window)

    print(f"  {len(clips_to_export)} clips selected:")
    for c in clips_to_export:
        print(f"    [{c['id']}] {c['title']} (ticks {c['start']}-{c['end']})")

    print("\nExtracting clip frames from archive...")

    # Build a map: tick -> list of clip IDs that want this tick
    clip_ticks = {}
    for c in clips_to_export:
        for ct in range(c["start"], c["end"] + 1, clip_snapshot_interval):
            if ct not in clip_ticks:
                clip_ticks[ct] = []
            clip_ticks[ct].append(c["id"])

    clip_frames_data = {c["id"]: [] for c in clips_to_export}

    # Read through all JSONL chunk files and extract matching frames
    chunk_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jsonl")],
    )

    for chunk_file in chunk_files:
        chunk_path = os.path.join(frames_dir, chunk_file)
        with open(chunk_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Quick tick extraction without full JSON parse
                # Format: {"t":TICK,...}
                t_start = line.find('"t":') + 4
                t_end = line.find(",", t_start)
                frame_tick = int(line[t_start:t_end])

                if frame_tick in clip_ticks:
                    snap = json.loads(line)
                    for cid in clip_ticks[frame_tick]:
                        clip_frames_data[cid].append(snap)

    # Write clip files
    manifest = {
        "world": {"width": config.world.width, "height": config.world.height},
        "total_ticks": ticks,
        "seed": seed,
        "clips": [],
    }

    for c in clips_to_export:
        frames = clip_frames_data[c["id"]]
        if not frames:
            print(f"  WARNING: No frames for {c['id']}")
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

        filename = f"primordial{part_suffix}-{c['id']}.json"
        filepath = os.path.join(narrative_dir, filename)
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
        print(f"  {c['id']}: {len(frames)} frames ({size_kb:.0f} KB)")

    manifest_path = os.path.join(narrative_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════
    total_elapsed = time.time() - t0
    total_frames_size = sum(
        os.path.getsize(os.path.join(frames_dir, f))
        for f in os.listdir(frames_dir) if f.endswith(".jsonl")
    )

    print(f"\n{'='*50}")
    print(f"  {part_label} export complete")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Frames: {total_frames} ({total_frames_size / 1024 / 1024:.0f} MB)")
    print(f"  Timeline: {size_kb:.0f} KB")
    print(f"  Clips: {len(clips_to_export)}")
    print(f"  Events: {len(events)}")
    print(f"{'='*50}")
    print(f"\nFull frames: {frames_dir}/")
    print(f"Timeline:    {timeline_path}")
    print(f"Clips:       {narrative_dir}/")
    print(f"Events:      {events_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-pass Primordial simulation export")
    parser.add_argument("ticks", type=int, nargs="?", default=100000, help="Simulation ticks")
    parser.add_argument("seed", type=int, nargs="?", default=42, help="Random seed")
    parser.add_argument("--part2", action="store_true", help="Use Part 2 config")
    parser.add_argument("--part3", action="store_true", help="Use Part 3 config")
    parser.add_argument("--part4", action="store_true", help="Use Part 4 config")
    parser.add_argument("--frame-interval", type=int, default=2,
                        help="Capture frame every N ticks (1=every tick, 2=default)")
    parser.add_argument("--sample-interval", type=int, default=100,
                        help="Timeline data sample interval")
    parser.add_argument("--clip-frames", type=int, default=500,
                        help="Frames per narrative clip")

    args = parser.parse_args()

    run_full_export(
        ticks=args.ticks,
        seed=args.seed,
        part2=args.part2,
        part3=args.part3,
        part4=args.part4,
        frame_interval=args.frame_interval,
        sample_interval=args.sample_interval,
        clip_frames=args.clip_frames,
    )
