"""Run the Primordial simulation."""

import argparse
import sys
import os

# Add project root to path so we can import primordial
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial.config import SimConfig
from primordial.engine import Engine


def main():
    parser = argparse.ArgumentParser(
        description="Primordial: A Study on Emergent Behaviors and Neuroevolution"
    )
    parser.add_argument(
        "--ticks", type=int, default=5000,
        help="Number of simulation ticks to run (default: 5000)"
    )
    parser.add_argument(
        "--population", type=int, default=50,
        help="Initial population size (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Log status every N ticks (default: 100)"
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=10,
        help="Record snapshot every N ticks (default: 10)"
    )
    parser.add_argument(
        "--clip-name", type=str, default=None,
        help="Name for the saved clip file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="clips",
        help="Directory for output files (default: clips)"
    )

    args = parser.parse_args()

    # Configure simulation
    config = SimConfig(
        initial_population=args.population,
        max_ticks=args.ticks,
        seed=args.seed,
    )
    config.recorder.snapshot_interval = args.snapshot_interval
    config.recorder.output_dir = args.output_dir

    # Create and run engine
    engine = Engine(config)
    engine.initialize()
    engine.run(ticks=args.ticks, log_interval=args.log_interval)

    # Save results
    engine.save(clip_name=args.clip_name)


if __name__ == "__main__":
    main()
