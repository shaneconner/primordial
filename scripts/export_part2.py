"""Export Part 2 simulation data: timeline + narrative clips.

Uses SimConfig.part2() for redesigned parameters (larger bodies,
separate eat/attack, richer sensors, sexual reproduction, etc.).
Runs both timeline export and narrative export in a single pass.
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

SITE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "shaneconner-site",
)


def run_part2(
    ticks=300000,
    seed=137,
    sample_interval=100,
    clip_frames=500,
    snapshot_interval=2,
):
    config = SimConfig.part2(seed=seed)
    config.max_ticks = ticks
    t0 = time.time()

    timeline_path = os.path.join(SITE_DIR, "data", "primordial-timeline-p2.json")
    narrative_dir = os.path.join(SITE_DIR, "data", "narrative-p2")

    # ══════════════════════════════════════════
    # PASS 1: Event detection + timeline data
    # ══════════════════════════════════════════
    print(f"=== PASS 1: Event detection + timeline ({ticks} ticks) ===")
    print(f"Config: spread={config.body.initial_spread}, sensors={config.brain.inputs_per_sensor}ips, "
          f"actions={config.brain.n_action_outputs}, max_age={config.evolution.max_age}")
    print(f"World: {config.world.width}x{config.world.height}, "
          f"Pop: {config.initial_population} initial / {config.max_population} max")

    engine = Engine(config)
    engine.initialize()

    # Timeline accumulation
    tick_list = []
    pop_list = []
    species_count_list = []
    gen_list = []
    avg_nodes_list = []
    node_type_timeline = {name: [] for name in NODE_TYPE_NAMES}
    all_species = set()
    species_pop_per_tick = []

    # Event detection
    history = []
    events = []
    known_species = {}
    prev_species_set = set()
    peak_pop = 0
    min_pop_after_peak = 999999
    max_body_complexity = 0
    mating_count = 0

    for tick in range(ticks):
        engine.step()
        t = engine.world.tick

        # Timeline sampling
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

        # Event detection every 5 ticks
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

            # Event detection
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

        if t % 1000 == 0:
            elapsed = time.time() - t0
            tps = t / elapsed if elapsed > 0 else 0
            pop = len([o for o in engine.world.organisms if o.alive])
            sp_count = len({o.species_id for o in engine.world.organisms if o.alive})
            max_gen = max((o.generation for o in engine.world.organisms if o.alive), default=0)
            print(f"  Tick {t}: pop={pop}, species={sp_count}, gen={max_gen}, "
                  f"events={len(events)}, {tps:.0f} t/s")

    elapsed_p1 = time.time() - t0
    print(f"\nPass 1 complete: {len(events)} events in {elapsed_p1:.0f}s")

    # ── Export timeline data ──
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
    print(f"\nTimeline exported: {timeline_path} ({size_kb:.0f} KB)")
    print(f"  {len(tick_list)} data points, {len(top_species)} species tracked")

    # ══════════════════════════════════════════
    # Clip selection
    # ══════════════════════════════════════════
    clip_window = clip_frames * snapshot_interval
    clips_to_export = select_narrative_clips(events, history, ticks, clip_window)

    print(f"\nSelected {len(clips_to_export)} narrative clips:")
    for c in clips_to_export:
        print(f"  [{c['id']}] {c['title']} (ticks {c['start']}-{c['end']})")

    # ══════════════════════════════════════════
    # PASS 2: Selective frame capture
    # ══════════════════════════════════════════
    print(f"\n=== PASS 2: Frame capture ({ticks} ticks, {len(clips_to_export)} clips) ===")
    t1 = time.time()

    engine2 = Engine(config)
    engine2.initialize()

    clip_windows = {c["id"]: c for c in clips_to_export}
    clip_frames_data = {c["id"]: [] for c in clips_to_export}

    capture_ticks = set()
    tick_to_clips = {}
    for cid, cw in clip_windows.items():
        for ct in range(cw["start"], cw["end"] + 1, snapshot_interval):
            capture_ticks.add(ct)
            if ct not in tick_to_clips:
                tick_to_clips[ct] = []
            tick_to_clips[ct].append(cid)

    max_capture_tick = max(capture_ticks) if capture_ticks else 0
    print(f"  Capture ticks: {len(capture_ticks)} frames across {len(clips_to_export)} clips")
    print(f"  Last capture at tick {max_capture_tick}")

    for tick in range(ticks):
        engine2.step()
        t = engine2.world.tick

        if t in capture_ticks:
            snap = compress_snapshot(engine2.world)
            for cid in tick_to_clips[t]:
                clip_frames_data[cid].append(snap)

        if t > max_capture_tick:
            print(f"  All clips captured at tick {t}, stopping early")
            break

        if t % 1000 == 0:
            elapsed = time.time() - t1
            tps = t / elapsed if elapsed > 0 else 0
            n_captured = sum(len(v) for v in clip_frames_data.values())
            print(f"  Tick {t}: captured {n_captured} frames, {tps:.0f} t/s")

    elapsed_p2 = time.time() - t1
    print(f"\nPass 2 complete in {elapsed_p2:.0f}s")

    # ══════════════════════════════════════════
    # Export clip files
    # ══════════════════════════════════════════
    os.makedirs(narrative_dir, exist_ok=True)
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

        filename = f"primordial-{c['id']}.json"
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
        print(f"  Exported {c['id']}: {len(frames)} frames ({size_kb:.0f} KB)")

    manifest_path = os.path.join(narrative_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_elapsed = time.time() - t0
    print(f"\nDone. Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Manifest: {manifest_path}")
    print(f"Timeline: {timeline_path}")


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
        "New organisms enter a harsher world with larger bodies and sharper senses.",
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
            "Scarce resources force a population crash.",
            crash_events[0]["tick"])

    # 5. Peak diversity
    if history:
        peak_div = max(history, key=lambda h: h["species"])
        if peak_div["species"] > 3:
            add_clip("diversity", "Radiation",
                f"{peak_div['species']} species coexist in distinct ecological niches.",
                peak_div["tick"])

    # 6. Body evolution
    body_events = [e for e in events if e["type"] == "body_evolution"]
    if body_events:
        add_clip("body-evolution", "Body Diversity",
            "Organisms evolve distinct body plans: predators, grazers, and armored survivors.",
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

    # 10. Deep time (near end, highest gen)
    if history:
        max_gen_entry = max(history, key=lambda h: h["gen"])
        if max_gen_entry["gen"] > 10:
            add_clip("deep-time", "Deep Time",
                f"Generation {max_gen_entry['gen']}. Complex body plans refined over thousands of generations.",
                max_gen_entry["tick"])

    clips.sort(key=lambda c: c["start"])
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
            "pop": len([o for o in world.organisms if o.alive]),
            "sp": len({o.species_id for o in world.organisms if o.alive}),
            "gen": max((o.generation for o in world.organisms if o.alive), default=0),
            "food": len(world.resources),
        },
    }


if __name__ == "__main__":
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 137
    run_part2(ticks=ticks, seed=seed)
