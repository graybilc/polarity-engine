#!/usr/bin/env python3
"""
polarity_engine/state_generator.py
Generates mutated 8-state structural representations directly from
the output of parsers.py _parse_mmcif_fast_path.
"""

import argparse
from typing import Any
import copy
import torch

# Updated for human Llgl1 numbering in 8r3y.cif
TARGET_RESIDUES = {
    0: 655,  # Bit index 0 -> S655 (Human Llgl1 equiv to Drosophila S656)
    1: 659,  # Bit index 1 -> S659 (Human Llgl1 equiv to Drosophila S660)
    2: 663,  # Bit index 2 -> S663 (Human Llgl1 equiv to Drosophila S664)
}


def mutate_structure_dict(
    parsed_dict: dict[str, Any], state_code: str
) -> dict[str, Any]:
    """
    Applies phosphomimetic mutations (Ser -> Glu) to the dictionary
    returned by _parse_mmcif_fast_path based on a 3-bit binary state_code.

    Args:
        parsed_dict (Dict[str, Any]): Dictionary output from _parse_mmcif_fast_path.
        state_code (str): 3-bit string representing phospho-state (e.g., '101').

    Returns:
        Dict[str, Any]: Deep copy of parsed_dict with mutated residue names.
    """
    if (
        not isinstance(state_code, str)
        or len(state_code) != 3
        or not set(state_code).issubset({"0", "1"})
    ):
        raise ValueError(
            f"Invalid state_code '{state_code}': Expected a 3-bit binary string (e.g., '101')."
        )

    mutated_dict = copy.deepcopy(parsed_dict)
    target_seq_nums = {
        TARGET_RESIDUES[idx] for idx, bit in enumerate(state_code) if bit == "1"
    }

    if not target_seq_nums:
        return mutated_dict

    # 1. Update C-alpha residues
    mutated_dict["aa_residues"] = [
        (seq_num_str, "GLU" if seq_num_str in target_seq_nums else res_name)
        for seq_num_str, res_name in mutated_dict["aa_residues"]
    ]

    # 2. Update all-atom residue names
    mutated_dict["all_atom_res_names"] = [
        "GLU" if seq_num_str in target_seq_nums else current_res_name
        for (_, seq_num_str), current_res_name in zip(
            mutated_dict["all_atom_keys"], mutated_dict["all_atom_res_names"]
        )
    ]

    return mutated_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mutate Lgl phosphorylation state.")
    parser.add_argument("--input", required=True, help="Input mmCIF file path")
    parser.add_argument(
        "--state", required=True, help="3-bit state vector string (e.g., '101')"
    )
    parser.add_argument("--output", required=True, help="Output mutant PDB file path")
    parser.add_argument(
        "--chain", default="B", help="Lgl chain ID in 8r3y (default: B)"
    )

    args = parser.parse_args()
    mutate_structure(args.input, args.state, args.output, args.chain)
