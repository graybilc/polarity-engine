#!/usr/bin/env python3
"""
polarity_engine/state_generator.py
Generates mutated 8-state structural representations directly from
the output of parsers.py _parse_mmcif_fast_path.
"""

from typing import Any
import copy
import numpy as np
import torch

# Default target positions for Lgl1 / 8r3y
DEFAULT_APKC_TARGETS = {
    0: "655",  # Bit index 0 -> S655
    1: "659",  # Bit index 1 -> S659
    2: "663",  # Bit index 2 -> S663
}


def resolve_state_mutations(
    state_code: str, target_sites: dict[int, str] | None = None, mimic_res: str = "GLU"
) -> dict[str, str]:
    """
    Maps a binary state code to residue mutation substitutions using bit index targets.

    Parameters
    ----------
    state_code : str
        Binary string representation (e.g. '100', '110', '111').
        '1' indicates phosphomimetic mutation; '0' indicates wild-type residue.
    target_sites : dict[int, str] | None, optional
        Mapping of bit positions to residue numbers.
        Defaults to `DEFAULT_APKC_TARGETS` ({0: "655", 1: "659", 2: "663"}).
    mimic_res : str, default="GLU"
        3-letter amino acid code for the phosphomimetic substitution.

    Returns
    -------
    dict[str, str]
        Dictionary mapping target residue numbers to their substituted residue name.
    """
    targets = target_sites if target_sites is not None else DEFAULT_APKC_TARGETS

    if len(state_code) != len(targets) or not set(state_code).issubset({"0", "1"}):
        raise ValueError(
            f"Invalid state_code '{state_code}'. Must be a {len(targets)}-bit binary string."
        )

    mutations = {}
    for bit_idx, bit_val in enumerate(state_code):
        if bit_val == "1":
            res_num = targets[bit_idx]
            mutations[res_num] = mimic_res

    return mutations


def mutate_sep_to_ser(
    parsed_dict: dict[str, Any], target_res_num: str = "663", target_chain: str = "B"
) -> dict[str, Any]:
    """
    Mutates a native phosphoserine (SEP) residue back to serine (SER) by
    updating residue labels and stripping phosphate heavy atoms (P, O1P, O2P, O3P).

    Args:
        parsed_dict : dict[str, Any]
            Parsed structural data dictionary containing SEP at the target residue.
        target_res_num : str, default="663"
            Residue number to mutate from SEP to SER.
        target_chain : str, default="B"
            Chain identifier.

    Returns:
        dict[str, Any]
            A deep copy of the dictionary with SEP converted to SER.
    """
    mutated_dict = copy.deepcopy(parsed_dict)

    # 1. Locate data dictionary
    data_dict = None
    if any(
        k in mutated_dict
        for k in ("nodes", "aa_residues", "aa_list", "all_atom_coords")
    ):
        data_dict = mutated_dict
    elif target_chain in mutated_dict and isinstance(mutated_dict[target_chain], dict):
        data_dict = mutated_dict[target_chain]
    else:
        for v in mutated_dict.values():
            if isinstance(v, dict) and any(
                rk in v for rk in ("nodes", "aa_residues", "aa_list")
            ):
                data_dict = v
                break

    if data_dict is None:
        raise KeyError("Could not find valid structure keys in dictionary.")

    # 2. Update C-alpha / backbone residue lists
    if "aa_residues" in data_dict:
        data_dict["aa_residues"] = [
            (seq_num, "SER" if seq_num == target_res_num and res == "SEP" else res)
            for seq_num, res in data_dict["aa_residues"]
        ]

    if "aa_list" in data_dict and "aa_residues" in data_dict:
        data_dict["aa_list"] = [res for _, res in data_dict["aa_residues"]]

    if "nodes" in data_dict:
        data_dict["nodes"] = [
            (
                chain_id,
                seq_num,
                "SER" if seq_num == target_res_num and res == "SEP" else res,
            )
            for chain_id, seq_num, res in data_dict["nodes"]
        ]

    # 3. Filter all-atom arrays: change SEP -> SER and remove phosphate atoms
    if "all_atom_coords" in data_dict and "all_atom_keys" in data_dict:
        standard_ser_atoms = {"N", "CA", "C", "O", "CB", "OG"}

        new_coords = []
        new_keys = []
        new_names = []
        new_res_names = []

        for coord, key, atom_name, res_name in zip(
            data_dict["all_atom_coords"],
            data_dict["all_atom_keys"],
            data_dict["all_atom_names"],
            data_dict["all_atom_res_names"],
        ):
            _, seq_num = key
            clean_atom = atom_name.upper().strip()

            if seq_num == target_res_num and res_name in ("SEP", "SER"):
                if clean_atom in standard_ser_atoms:
                    new_coords.append(coord)
                    new_keys.append(key)
                    new_names.append(clean_atom)
                    new_res_names.append("SER")
            else:
                new_coords.append(coord)
                new_keys.append(key)
                new_names.append(atom_name)
                new_res_names.append(res_name)

        data_dict["all_atom_coords"] = np.array(
            new_coords, dtype=data_dict["all_atom_coords"].dtype
        )
        data_dict["all_atom_keys"] = new_keys
        data_dict["all_atom_names"] = new_names
        data_dict["all_atom_res_names"] = new_res_names

    return mutated_dict


def mutate_structure_dict(
    wt_parsed_dict: dict[str, Any],
    state_code: str,
    target_chain: str = "B",
    mimic_res: str = "GLU",
) -> dict[str, Any]:
    """
    Applies phosphomimetic mutations to a wild-type parsed structure dictionary.
    """
    mutated_dict = copy.deepcopy(wt_parsed_dict)

    # 1. Locate data dictionary
    data_dict = None
    if any(k in mutated_dict for k in ("nodes", "aa_residues", "aa_list")):
        data_dict = mutated_dict
    elif target_chain in mutated_dict and isinstance(mutated_dict[target_chain], dict):
        data_dict = mutated_dict[target_chain]
    else:
        for v in mutated_dict.values():
            if isinstance(v, dict) and any(
                rk in v for rk in ("nodes", "aa_residues", "aa_list")
            ):
                data_dict = v
                break

    if data_dict is None:
        raise KeyError(f"Could not find valid structure keys in dictionary.")

    # 2. Extract or infer aa_residues from nodes
    if "aa_residues" in data_dict:
        aa_residues = data_dict["aa_residues"]
    elif "nodes" in data_dict:
        aa_residues = [(seq_num, res) for _, seq_num, res in data_dict["nodes"]]
    else:
        raise KeyError(
            f"Neither 'aa_residues' nor 'nodes' found in structure dictionary."
        )

    # 3. Resolve mutations with customizable mimic residue support
    mutation_map = resolve_state_mutations(state_code, mimic_res=mimic_res)

    # 4. Mutate aa_residues, aa_list, and nodes
    new_aa_residues = [
        (seq_num, mutation_map.get(seq_num, res)) for seq_num, res in aa_residues
    ]
    data_dict["aa_residues"] = new_aa_residues

    if "aa_list" in data_dict:
        data_dict["aa_list"] = [res for _, res in new_aa_residues]

    if "nodes" in data_dict:
        data_dict["nodes"] = [
            (chain_id, seq_num, mutation_map.get(seq_num, res))
            for chain_id, seq_num, res in data_dict["nodes"]
        ]

    # 5. Mutate heavy-atom residue labels if present
    if "all_atom_res_names" in data_dict and "all_atom_keys" in data_dict:
        data_dict["all_atom_res_names"] = [
            mutation_map.get(seq_num, orig_res)
            for (chain_id, seq_num), orig_res in zip(
                data_dict["all_atom_keys"], data_dict["all_atom_res_names"]
            )
        ]

    return mutated_dict
