"""Quick matplotlib visualization for debugging simulation state.

Usage:
    python scripts/visualize.py                    # run live simulation
    python scripts/visualize.py clips/test_run.json  # replay a clip
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import LineCollection
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

from primordial.body import NodeType, EdgeType

# Node type colors
NODE_COLORS = {
    int(NodeType.CORE): '#d4cfc3',
    int(NodeType.BONE): '#a09888',
    int(NodeType.MUSCLE_ANCHOR): '#8a3a3a',
    int(NodeType.SENSOR): '#5a7a8a',
    int(NodeType.MOUTH): '#9aaa3a',
    int(NodeType.FAT): '#b49a6e',
    int(NodeType.ARMOR): '#7a7868',
}

EDGE_COLORS = {
    int(EdgeType.BONE): '#666655',
    int(EdgeType.MUSCLE): '#8a3a3a',
    int(EdgeType.TENDON): '#555544',
}

FOOD_COLORS = {
    'plant': '#5a8a2a',
    'meat': '#8a3a3a',
    'nutrient': '#6e5a3a',
}


def render_frame(ax, snapshot, world_w=500, world_h=500):
    """Render a single simulation frame."""
    ax.clear()
    ax.set_xlim(0, world_w)
    ax.set_ylim(0, world_h)
    ax.set_facecolor('#090b09')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw resources
    for r in snapshot.get('resources', []):
        color = FOOD_COLORS.get(r['type'], '#444')
        ax.plot(r['x'], r['y'], '.', color=color, markersize=2, alpha=0.6)

    # Draw organisms
    for org in snapshot.get('organisms', []):
        nodes = org.get('nodes', [])
        edges = org.get('edges', [])

        if not nodes:
            continue

        # Draw edges
        for e in edges:
            n1 = nodes[e['from']]
            n2 = nodes[e['to']]
            color = EDGE_COLORS.get(e['type'], '#444')
            lw = 1.5 if e['type'] == int(EdgeType.MUSCLE) else 0.8
            ax.plot(
                [n1['x'], n2['x']], [n1['y'], n2['y']],
                color=color, linewidth=lw, alpha=0.7
            )

        # Draw nodes
        for n in nodes:
            color = NODE_COLORS.get(n['type'], '#888')
            size = 4 if n['type'] == int(NodeType.CORE) else 3
            ax.plot(n['x'], n['y'], 'o', color=color, markersize=size, alpha=0.9)

    # Stats overlay
    stats = snapshot.get('stats', {})
    tick = snapshot.get('tick', 0)
    info = (
        f"Tick: {tick}  |  "
        f"Pop: {stats.get('population', '?')}  |  "
        f"Species: {stats.get('species_count', '?')}  |  "
        f"Gen: {stats.get('max_generation', '?')}  |  "
        f"Food: {stats.get('resource_count', '?')}"
    )
    ax.set_title(info, fontsize=9, color='#d4cfc3', pad=4)


def replay_clip(clip_path):
    """Replay a saved clip file."""
    with open(clip_path) as f:
        clip = json.load(f)

    snapshots = clip.get('snapshots', [])
    if not snapshots:
        print("No snapshots in clip.")
        return

    print(f"Replaying {len(snapshots)} frames from {clip.get('name', 'unknown')}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor('#090b09')
    plt.ion()

    for i, snap in enumerate(snapshots):
        render_frame(ax, snap)
        plt.pause(0.03)

    plt.ioff()
    plt.show()


def run_live():
    """Run simulation with live visualization."""
    from primordial.config import SimConfig
    from primordial.engine import Engine

    config = SimConfig(initial_population=40, seed=42)
    engine = Engine(config)
    engine.initialize()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_facecolor('#090b09')
    plt.ion()

    print("Running live simulation (close window to stop)...")

    try:
        for tick in range(10000):
            engine.step()

            # Render every 5 ticks
            if tick % 5 == 0:
                snap = engine.world.to_dict()
                render_frame(ax, snap, config.world.width, config.world.height)
                plt.pause(0.001)

                if not plt.fignum_exists(fig.number):
                    break

    except KeyboardInterrupt:
        pass

    print(f"Stopped at tick {engine.world.tick}")
    engine.save("live_session")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        replay_clip(sys.argv[1])
    else:
        run_live()
