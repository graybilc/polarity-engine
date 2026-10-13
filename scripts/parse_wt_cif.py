#!/usr/bin/env python3
"""Parses target CIF structure files, imputes missing loop coordinates from a reference template,

and applies site mapping metadata.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
from polarity_engine.parsers import StructureParser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments for wild-type CIF parsing.

    Args:
        args (Optional[List[str]]): Explicit list of argument strings to parse.
            If None, parses system arguments from `sys.argv[1:]`. Defaults to None.

    Returns:
        argparse.Namespace: Parsed argument namespace containing:
            - cif (Path): Path to target MMCIF structure file.
            - ref_cif (Path): Path to reference MMCIF structure file (e.g., 8r3y.cif).
            - mapping (Path): Path to site mapping JSON emitted by ALIGN_CIF.
            - chain (str): Target chain identifier.
            - out (Path): Path where output PyTorch `.pt` dictionary will be saved.
    """
    parser = argparse.ArgumentParser(
        description="Parse WT CIF structure and apply alignment site metadata."
    )
    parser.add_argument(
        "--cif", type=Path, required=True, help="Path to target CIF file"
    )
    parser.add_argument(
        "--ref-cif", type=Path, required=True, help="Path to reference CIF file"
    )
    parser.add_argument(
        "--mapping", type=Path, required=True, help="Path to site mapping JSON"
    )
    parser.add_argument(
        "--chain", type=str, default="L", help="Target chain ID (default: L)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("wt_parsed.pt"),
        help="Output PyTorch parsed dict path",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[List[str]] = None) -> None:
    """
    Main execution entry point for parsing WT CIF files.

    Args:
        cli_args (Optional[List[str]]): Command-line argument overrides for testing.

    Returns:
        None

    Raises:
        FileNotFoundError: If input CIF or site mapping files are missing.
    """
    args = parse_args(cli_args)

    if not args.cif.is_file():
        raise FileNotFoundError(f"Target CIF file not found: {args.cif}")
    if not args.ref_cif.is_file():
        raise FileNotFoundError(f"Reference CIF file not found: {args.ref_cif}")
    if not args.mapping.is_file():
        raise FileNotFoundError(f"Site mapping JSON not found: {args.mapping}")

    # Load site mapping metadata
    with open(args.mapping) as f:
        mapping_data = json.load(f)

    # Parse target structure coordinates (imputing missing loops via ref_cif if required)
    wt_parsed_dict = StructureParser.parse(
        str(args.cif),
        ref_cif_path=str(args.ref_cif),
        chain_ids=[args.chain],
    )

    # Attach site mapping metadata for downstream mutation and feature steps
    wt_parsed_dict["site_mapping"] = mapping_data.get("sites", mapping_data)

    # Save dictionary tensor directly into current working directory ($PWD)
    out_path = Path.cwd() / args.out.name
    StructureParser.save_parsed_dict(wt_parsed_dict, str(out_path))


if __name__ == "__main__":
    main()
