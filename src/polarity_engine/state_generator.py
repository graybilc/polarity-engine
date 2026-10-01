#!/usr/bin/env python3
"""
polarity_engine/state_generator.py
Generates mutated 8-state structural representations directly from
the output of parsers.py _parse_mmcif_fast_path.
"""

from typing import Any
import copy
import torch

# Default target positions for Lgl1 / 8r3y
DEFAULT_APKC_TARGETS = {
    0: "655",  # Bit index 0 -> S655
    1: "659",  # Bit index 1 -> S659
    2: "663",  # Bit index 2 -> S663
}


def mutate_structure_dict(
    parsed_dict: dict[str, Any],
    state_code: str,
    target_residues: dict[int, str] = DEFAULT_APKC_TARGETS,
    target_amino_acid: str = "GLU",
) -> dict[str, Any]:
    """
    Applies point mutations to a parsed structure dictionary based on an arbitrary N-bit binary state_code.

    Args:
        parsed_dict: Dictionary returned by StructureParser.parse().
        state_code: N-bit binary string representing variant state (e.g., '101' or '1001').
        target_residues: Mapping from bit index to target residue sequence numbers (e.g., {0: "655", 1: "659"}).
        target_amino_acid: Residue code to substitute when bit is '1' (default: "GLU").
    """
    if (
        not isinstance(state_code, str)
        or len(state_code) != len(target_residues)
        or not set(state_code).issubset({"0", "1"})
    ):
        raise ValueError(
            f"Invalid state_code '{state_code}': Expected a {len(target_residues)}-bit binary string."
        )

    mutated_dict = copy.deepcopy(parsed_dict)

    # Identify target residue numbers where bit is active ('1')
    active_seq_nums = {
        str(target_residues[idx]) for idx, bit in enumerate(state_code) if bit == "1"
    }

    if not active_seq_nums:
        return mutated_dict

    # 1. Update C-alpha residue representations
    mutated_dict["aa_residues"] = [
        (seq_num_str, target_amino_acid if seq_num_str in active_seq_nums else res_name)
        for seq_num_str, res_name in mutated_dict["aa_residues"]
    ]

    # 2. Update all-atom residue representations
    mutated_dict["all_atom_res_names"] = [
        target_amino_acid if seq_num_str in active_seq_nums else current_res_name
        for (_, seq_num_str), current_res_name in zip(
            mutated_dict["all_atom_keys"], mutated_dict["all_atom_res_names"]
        )
    ]

    return mutated_dict
