"""
polarity_engine/cli.py
Command-line interface entrypoint for running structural-kinetic simulations.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from polarity_engine.kinetics import PhosphoKineticSolver


def main():
    parser = argparse.ArgumentParser(
        description="Run polarity-engine structural gating ODE simulations."
    )
    parser.add_argument("command", choices=["simulate"], help="Subcommand to execute.")
    parser.add_argument(
        "--graphs_dir",
        type=Path,
        required=True,
        help="Directory containing the 8 hypercube state PyG graphs.",
    )
    parser.add_argument(
        "--mappings",
        type=Path,
        required=True,
        help="JSON file mapping functional sites to structural indices.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Structural gating sensitivity hyperparameter.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="processive",
        choices=["processive", "dissociative"],
        help="Kinetic mechanism regime.",
    )
    parser.add_argument(
        "--mimic-res",
        type=str,
        default="GLU",
        choices=["GLU", "ASP"],
        help="Phosphomimetic residue type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_trajectories.csv"),
        help="Path to save output trajectory CSV.",
    )

    args = parser.parse_args()

    if args.command == "simulate":
        print(f"--- Starting Polarity Engine Simulation ---")
        print(f"Graphs Directory : {args.graphs_dir}")
        print(f"Site Mappings    : {args.mappings}")
        print(f"Alpha (gating)   : {args.alpha}")
        print(f"Regime           : {args.regime}")
        print(f"Mimic Residue    : {args.mimic_res}")
        print(f"Output Target    : {args.output}")

        graphs_path = Path(args.graphs_dir)
        graph_files = sorted(list(graphs_path.glob("graph_*.pt")))

        if not graph_files:
            print(f"Error: No state graphs found in {graphs_path}", file=sys.stderr)
            sys.exit(1)

        # Load PyG graphs into a dictionary keyed by state code (e.g., '100', '000')
        graphs_dict = {}
        for g_file in graph_files:
            if g_file.stat().st_size == 0:
                print(
                    f"Warning: Graph file {g_file} is empty (likely a stub run). Skipping or creating dummy tensor.",
                    file=sys.stderr,
                )
                continue
            state_code = g_file.stem.replace("graph_", "")
            graphs_dict[state_code] = torch.load(g_file, weights_only=False)

        # Load site mappings JSON
        with open(args.mappings, "r") as f:
            site_mappings = json.load(f)

        print(f"Loaded {len(graphs_dict)} state graphs. Initializing solver...")

        # Instantiate and run the PhosphoKineticSolver
        solver = PhosphoKineticSolver(
            graphs=graphs_dict,
            site_mappings=site_mappings,
            alpha=args.alpha,
            par6_active=True,
            regime=args.regime,
        )

        t_eval, concentrations = solver.simulate(t_span=(0, 20))

        # Format trajectories into a DataFrame and export to CSV
        columns = ["t", "s000", "s100", "s010", "s001", "s110", "s101", "s011", "s111"]
        df_out = pd.DataFrame(np.vstack([t_eval, concentrations]).T, columns=columns)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output, index=False)

        print(f"Simulation completed successfully. Trajectories saved to {args.output}")


if __name__ == "__main__":
    main()
