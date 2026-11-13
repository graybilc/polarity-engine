#!/usr/bin/env python3
"""CLI script to apply generalized ternary state mutations to parsed wild-type structure dictionaries."""

import argparse
import sys
from pathlib import Path
import torch

from polarity_engine.parsers import StructureParser
from polarity_engine.state_generator import StateGenerator


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for applying ternary state mutations.
    """
    parser = argparse.ArgumentParser(
        description="Apply generalized ternary state mutations to a wild-type parsed structure dictionary."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input wild-type parsed dictionary (.pt file).",
    )
    parser.add_argument(
        "--state-code",
        "-s",
        type=str,
        required=True,
        help="Ternary state code string (e.g., '000', '111', '222', 'AAA', '120').",
    )
    parser.add_argument(
        "--target-chain",
        type=str,
        default="L",
        help="Target chain identifier (e.g., 'L'). Defaults to 'L'.",
    )
    parser.add_argument(
        "--mimic-res",
        type=str,
        default="ASP",
        choices=["ASP", "GLU"],
        help="3-letter code for the phosphomimetic residue (e.g., 'ASP' or 'GLU').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to save output mutated dictionary (.pt file).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution entrypoint for loading wild-type structure dictionaries,
    applying ternary state mutations, and saving the resulting structure files.
    """
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[mutate_state] Loading wild-type dictionary from: {args.input}")
    wt_dict = StructureParser.load_parsed_dict(args.input)

    # Initialize StateGenerator with selected mimic residue
    generator = StateGenerator(default_mimic=args.mimic_res)

    print(
        f"[mutate_state] Applying ternary state code '{args.state_code}' on chain '{args.target_chain}' with mimic residue '{args.mimic_res}'..."
    )
    mutated_dict = generator.mutate_structure_dict(
        wt_dict,
        args.state_code,
        target_chain=args.target_chain,
        mimic_res=args.mimic_res,
    )

    print(f"[mutate_state] Saving mutated structure dictionary to: {args.output}")
    torch.save(mutated_dict, args.output)
    print("[mutate_state] Complete.")


if __name__ == "__main__":
    main()
