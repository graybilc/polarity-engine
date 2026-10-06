#!/usr/bin/env python3
"""CLI script to apply state mutations to parsed wild-type structure dictionaries."""

import argparse
import sys
from pathlib import Path
import torch

from polarity_engine.parsers import StructureParser
from polarity_engine.state_generator import mutate_structure_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply phosphomimetic mutations to a wild-type parsed structure dictionary."
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
        help="Binary state permutation code (e.g., '000', '100', '111').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to save output mutated dictionary (.pt file).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Ensure parent output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[mutate_state] Loading wild-type dictionary from: {args.input}")
    wt_dict = StructureParser.load_parsed_dict(args.input)

    print(f"[mutate_state] Applying state mutation code '{args.state_code}'...")
    mutated_dict = mutate_structure_dict(wt_dict, args.state_code)

    print(f"[mutate_state] Saving mutated structure dictionary to: {args.output}")
    torch.save(mutated_dict, args.output)
    print("[mutate_state] Complete.")


if __name__ == "__main__":
    main()
