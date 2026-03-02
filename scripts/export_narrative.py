"""Run a long simulation and export narrative clips at key moments.

Two-pass approach for handling very long simulations (100k+ ticks):
  Pass 1: Run full sim, detect events, store only lightweight stats
  Pass 2: Re-run with same seed, capture frames only at clip windows
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine


def run_and_export(
    ticks=100000,
    seed=42,
    output_dir="clips/narrative",
    clip_frames=500,
    snapshot_interval=2,
):
    config = SimConfig(seed=seed)
    t0 = time.time()

    # ══════════════════════════════════════════
    # PASS 1: Event detection (no snapshots)
    # ══════════════════════════════════════════
    print(f"=== PASS 1: Event detection ({ticks} ticks) ===")
    print(f"World: {config.world.width}x{config.world.height}, "
          f"Pop: {config.initial_population} initial / {config.max_population} max")

    engine = Engine(config)
    engine.initialize()

    history = []
    events = []
    known_species = {}
    prev_species_set = set()
    peak_pop = 0
    min_pop_after_peak = 999999
    max_body_complexity = 0

    for tick in range(ticks):
        engine.step()
        t = engine.world.tick

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

            # ── Event detection ──

            # New species
            new_sp = current_species - prev_species_set
            if new_sp and t > 10:
                for sp in new_sp:
                    events.append({"tick": t, "type": "speciation", "description": f"New species {sp}"})

            # Extinction
            extinct = prev_species_set - current_species
            if extinct and t > 10:
                for sp in extinct:
                    if sp in known_species and known_species[sp]["peak_pop"] >= 3:
                        events.append({"tick": t, "type": "extinction",
                            "description": f"Species {sp} extinct (peak {known_species[sp]['peak_pop']})"})

            # Population boom
            if pop > peak_pop * 1.3 and pop > peak_pop + 10 and t > 50:
                events.append({"tick": t, "type": "boom", "description": f"Pop boom: {peak_pop} -> {pop}"})
            if pop > peak_pop:
                peak_pop = pop
                min_pop_after_peak = pop

            # Population crash
            if pop < min_pop_after_peak * 0.6 and min_pop_after_peak > 20 and t > 100:
                events.append({"tick": t, "type": "crash", "description": f"Pop crash: {min_pop_after_peak} -> {pop}"})
                min_pop_after_peak = pop
            if pop < min_pop_after_peak:
                min_pop_after_peak = pop

            # Generation milestones
            if max_gen > 0 and max_gen % 10 == 0 and (len(history) < 2 or history[-2]["gen"] < max_gen):
                events.append({"tick": t, "type": "generation", "description": f"Generation {max_gen}"})

            # Diversity milestones
            if sp_count >= 8 and (len(history) < 2 or history[-2]["species"] < 8):
                events.append({"tick": t, "type": "diversity", "description": f"8+ species ({sp_count})"})
            if sp_count >= 15 and (len(history) < 2 or history[-2]["species"] < 15):
                events.append({"tick": t, "type": "diversity_high", "description": f"15+ species ({sp_count})"})

            # Body complexity milestone
            if avg_nodes > max_body_complexity + 1.0 and avg_nodes > 6.0:
                events.append({"tick": t, "type": "body_evolution", "description": f"Avg body complexity: {avg_nodes:.1f} nodes"})
                max_body_complexity = avg_nodes

            prev_species_set = current_species

        if t % 1000 == 0:
            elapsed = time.time() - t0
            tps = t / elapsed if elapsed > 0 else 0
            print(f"  Tick {t}: pop={pop}, species={sp_count}, gen={max_gen}, "
                  f"events={len(events)}, {tps:.0f} t/s")

    elapsed_p1 = time.time() - t0
    print(f"\nPass 1 complete: {len(events)} events in {elapsed_p1:.0f}s")

    # ══════════════════════════════════════════
    # Clip selection
    # ══════════════════════════════════════════
    clip_window = clip_frames * snapshot_interval  # ticks per clip
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

    # Build a set of ticks that need capture for fast lookup
    capture_ticks = set()
    tick_to_clips = {}  # tick -> list of clip IDs
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

        # Can stop early if we've captured everything
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
    os.makedirs(output_dir, exist_ok=True)
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

    total_elapsed = time.time() - t0
    print(f"\nDone. Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Manifest: {manifest_path}")


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
        # If overlapping, try shifting forward past the conflicting range
        attempts = 0
        while overlaps(start, end) and attempts < 5:
            # Find the conflicting range and shift past it
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

    # 1. Genesis - short window (just the beginning)
    add_clip("genesis", "Genesis",
        "The first organisms are dropped into an empty world.", 150,
        window=clip_window // 2)

    # 2. First speciation
    spec_events = [e for e in events if e["type"] == "speciation"]
    if spec_events:
        add_clip("first-split", "First Divergence",
            "The first new species emerges from mutation.",
            spec_events[0]["tick"])

    # 3. First population boom
    boom_events = [e for e in events if e["type"] == "boom"]
    if boom_events:
        add_clip("boom", "The Bloom",
            "Successful strategies propagate and the population explodes.",
            boom_events[0]["tick"])

    # 4. First crash
    crash_events = [e for e in events if e["type"] == "crash"]
    if crash_events:
        add_clip("crash", "Collapse",
            "Resources deplete and the population crashes.",
            crash_events[0]["tick"])

    # 5. Peak diversity (first peak)
    if history:
        peak_div = max(history, key=lambda h: h["species"])
        if peak_div["species"] > 3:
            add_clip("diversity", "Radiation",
                f"{peak_div['species']} species coexist.",
                peak_div["tick"])

    # 6. Recovery after crash (first boom after first crash)
    if crash_events and boom_events:
        recovery_booms = [e for e in boom_events if e["tick"] > crash_events[0]["tick"]]
        if recovery_booms:
            add_clip("recovery", "Recovery",
                "The population rebounds. New species fill vacated niches.",
                recovery_booms[0]["tick"])

    # 7. Body evolution milestone
    body_events = [e for e in events if e["type"] == "body_evolution"]
    if body_events:
        add_clip("body-evolution", "Body Evolution",
            "Organisms have evolved more complex body plans.",
            body_events[-1]["tick"])

    # 8. Mass extinction (biggest crash, prefer mid-to-late sim)
    mid_late_crashes = [e for e in crash_events if e["tick"] > total_ticks * 0.3]
    if mid_late_crashes:
        add_clip("mass-extinction", "Mass Extinction",
            "The largest population collapse reshapes the ecosystem.",
            mid_late_crashes[-1]["tick"])

    # 9. Generation milestone (midpoint)
    gen_events = [e for e in events if e["type"] == "generation"]
    mid_gen = [e for e in gen_events if e["tick"] > total_ticks * 0.3 and e["tick"] < total_ticks * 0.7]
    if mid_gen:
        add_clip("generation-milestone", "Generational Shift",
            "A new generation milestone marks ongoing evolution.",
            mid_gen[len(mid_gen) // 2]["tick"])

    # 10. Late diversity peak
    late_history = [h for h in history if h["tick"] > total_ticks * 0.5]
    if late_history:
        late_peak = max(late_history, key=lambda h: h["species"])
        if late_peak["species"] > 5:
            add_clip("late-diversity", "Second Radiation",
                f"A second wave of speciation produces {late_peak['species']} species.",
                late_peak["tick"])

    # 11. Equilibrium (late stable period)
    late_quarter = [h for h in history if h["tick"] > total_ticks * 0.75]
    if late_quarter:
        mid_tick = late_quarter[len(late_quarter) // 2]["tick"]
        add_clip("equilibrium", "Balance",
            "The ecosystem reaches carrying capacity.",
            mid_tick)

    # 12. Deep time (highest generation)
    if history:
        max_gen_entry = max(history, key=lambda h: h["gen"])
        if max_gen_entry["gen"] > 10:
            add_clip("deep-time", "Deep Time",
                f"Generation {max_gen_entry['gen']}. Neural networks refined across many generations.",
                max_gen_entry["tick"])

    # Sort clips by start tick
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
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    output = sys.argv[3] if len(sys.argv) > 3 else "clips/narrative"
    run_and_export(ticks=ticks, seed=seed, output_dir=output)
