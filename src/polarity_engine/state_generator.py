#!/usr/bin/env python3
"""
polarity_engine/state_generator.py
Generates mutated 8-state structural representations directly from
the output of parsers.py _parse_mmcif_fast_path. AAA
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

    Args:
        state_code (str): Binary string representation (e.g. '100', '110', '111').
            '1' indicates phosphomimetic mutation; '0' indicates wild-type residue.
        target_sites (dict[int, str] | None, optional): Mapping of bit positions to residue numbers.
            Defaults to `DEFAULT_APKC_TARGETS` ({0: "655", 1: "659", 2: "663"}).
        mimic_res (str, optional): 3-letter amino acid code for the phosphomimetic substitution.
            Defaults to "GLU".

    Returns:
        dict[str, str]: Dictionary mapping target residue numbers to their substituted residue name.
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


def _validate_parsed_dict_contract(parsed_dict: dict[str, Any]) -> None:
    """
    Enforces a rigid data contract on the input structure dictionary.
    Raises KeyError if essential schema keys are missing.
    """
    required_keys = {
        "nodes",
        "aa_list",
        "all_atom_coords",
        "all_atom_keys",
        "all_atom_names",
        "all_atom_res_names",
    }
    missing = required_keys - parsed_dict.keys()
    if missing:
        raise KeyError(
            f"Invalid structure dictionary contract. Missing required schema keys: {missing}. "
            "Ensure input is generated directly via StructureParser.parse()."
        )


def _update_sequence_metadata(
    mutated_dict: dict[str, Any], mutation_map: dict[str, str]
) -> None:
    """
    Updates backbone residue lists, amino acid lists, and node descriptors with mutated residue names.

    Args:
        mutated_dict (dict[str, Any]): Dictionary containing structural metadata arrays (`nodes`, `aa_residues`, `aa_list`).
        mutation_map (dict[str, str]): Dictionary mapping target residue sequence numbers to their substituted residue names.

    Returns:
        None: Modifies the dictionary data structures in place.
    """
    nodes = mutated_dict["nodes"]
    new_aa_residues = [
        (seq_num, mutation_map.get(seq_num, res)) for _, seq_num, res in nodes
    ]

    mutated_dict["aa_residues"] = new_aa_residues
    if "aa_list" in mutated_dict:
        mutated_dict["aa_list"] = [res for _, res in new_aa_residues]

    mutated_dict["nodes"] = [
        (chain_id, seq_num, mutation_map.get(seq_num, res))
        for chain_id, seq_num, res in nodes
    ]


def _expand_phosphomimetic_sidechain(
    cb_coord: np.ndarray,
    target_mimic: str,
    state_code: str,
) -> list[tuple[str, np.ndarray]]:
    """
    Calculates perturbed 3D Cartesian coordinates for GLU or ASP phosphomimetic side-chain atoms.

    Args:
        cb_coord (np.ndarray): 3D coordinate vector of the C-beta (CB) atom.
        target_mimic (str): 3-letter amino acid code for the phosphomimetic residue ('GLU' or 'ASP').
        state_code (str): Binary state permutation code (e.g., '100', '110') used to introduce state-dependent spatial perturbations.

    Returns:
        list[tuple[str, np.ndarray]]: A list of tuples containing heavy-atom names (e.g., CG, CD, OE1, OE2 for GLU;
            CG, OD1, OD2 for ASP) paired with their respective 3D coordinate arrays.
    """
    state_int = int(state_code, 2)
    perturbation = np.array(
        [
            ((state_int & 1) - 0.5) * 0.20,
            (((state_int >> 1) & 1) - 0.5) * 0.20,
            (((state_int >> 2) & 1) - 0.5) * 0.20,
        ]
    )

    cg_coord = cb_coord + np.array([1.2, 0.5, 0.0]) + perturbation

    if target_mimic == "GLU":
        cd_coord = cg_coord + np.array([1.2, 0.4, 0.0]) + (perturbation * 0.5)
        oe1_coord = cd_coord + np.array([0.8, 0.6, 0.0])
        oe2_coord = cd_coord + np.array([0.5, -0.7, 0.0])
        return [
            ("CG", cg_coord),
            ("CD", cd_coord),
            ("OE1", oe1_coord),
            ("OE2", oe2_coord),
        ]
    else:  # ASP
        od1_coord = cg_coord + np.array([0.8, 0.5, 0.0]) + (perturbation * 0.5)
        od2_coord = cg_coord + np.array([0.4, -0.8, 0.0])
        return [
            ("CG", cg_coord),
            ("OD1", od1_coord),
            ("OD2", od2_coord),
        ]


def mutate_sep_to_ser(
    parsed_dict: dict[str, Any], target_res_num: str = "663", target_chain: str = "B"
) -> dict[str, Any]:
    """
    Mutates a native phosphoserine (SEP) residue back to serine (SER) by
    updating residue labels and stripping phosphate heavy atoms (P, O1P, O2P, O3P).

    Args:
        parsed_dict (dict[str, Any]): Parsed structural data dictionary containing SEP at the target residue.
        target_res_num (str, optional): Residue number to mutate from SEP to SER. Defaults to "663".
        target_chain (str, optional): Chain identifier. Defaults to "B".

    Returns:
        dict[str, Any]: A deep copy of the dictionary with SEP converted to SER.
    """
    # 1. Enforce strict contract validation
    _validate_parsed_dict_contract(parsed_dict)

    mutated_dict = copy.deepcopy(parsed_dict)
    mutated_dict.pop("rsasa", None)

    # 2. Update sequence metadata using the shared helper
    mutation_map = {target_res_num: "SER"}
    _update_sequence_metadata(mutated_dict, mutation_map)

    # 3. Filter all-atom arrays: change SEP -> SER and remove phosphate atoms
    standard_ser_atoms = {"N", "CA", "C", "O", "CB", "OG"}

    new_coords, new_keys, new_names, new_res_names = [], [], [], []

    for coord, key, atom_name, res_name in zip(
        mutated_dict["all_atom_coords"],
        mutated_dict["all_atom_keys"],
        mutated_dict["all_atom_names"],
        mutated_dict["all_atom_res_names"],
    ):
        chain_id, seq_num = key
        clean_atom = atom_name.upper().strip()

        if (
            chain_id == target_chain
            and seq_num == target_res_num
            and res_name in ("SEP", "SER")
        ):
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

    mutated_dict["all_atom_coords"] = np.array(
        new_coords, dtype=mutated_dict["all_atom_coords"].dtype
    )
    mutated_dict["all_atom_keys"] = new_keys
    mutated_dict["all_atom_names"] = new_names
    mutated_dict["all_atom_res_names"] = new_res_names

    return mutated_dict


def mutate_structure_dict(
    wt_parsed_dict: dict[str, Any],
    state_code: str,
    target_chain: str = "B",
    mimic_res: str = "GLU",
) -> dict[str, Any]:
    """
    Applies phosphomimetic mutations to a wild-type parsed structure dictionary,
    updating both metadata labels and heavy-atom side-chain coordinates for GLU/ASP.

    Args:
        wt_parsed_dict (dict[str, Any]): Wild-type parsed structure dictionary.
        state_code (str): Binary state permutation code (e.g., '100', '110', '111').
        target_chain (str, optional): Target chain identifier. Defaults to "B".
        mimic_res (str, optional): 3-letter code for the phosphomimetic residue ('GLU' or 'ASP'). Defaults to "GLU".

    Returns:
        dict[str, Any]: Mutated structure dictionary containing updated node features and expanded heavy-atom coordinates.
    """
    _validate_parsed_dict_contract(wt_parsed_dict)

    mutated_dict = copy.deepcopy(wt_parsed_dict)
    mutated_dict.pop("rsasa", None)

    # 1. Resolve mutations and update sequence metadata
    mutation_map = resolve_state_mutations(state_code, mimic_res=mimic_res)
    _update_sequence_metadata(mutated_dict, mutation_map)

    # 2. Rebuild all-atom arrays with side-chain expansions
    new_coords, new_keys, new_names, new_res_names = [], [], [], []
    expanded_residues = set()

    for coord, key, atom_name, res_name in zip(
        mutated_dict["all_atom_coords"],
        mutated_dict["all_atom_keys"],
        mutated_dict["all_atom_names"],
        mutated_dict["all_atom_res_names"],
    ):
        chain_id, seq_num = key
        target_mimic = mutation_map.get(seq_num)
        clean_atom = atom_name.upper().strip()

        if chain_id == target_chain and target_mimic in ("GLU", "ASP"):
            if clean_atom in ("N", "CA", "C", "O", "CB"):
                new_coords.append(coord)
                new_keys.append(key)
                new_names.append(clean_atom)
                new_res_names.append(target_mimic)
            elif clean_atom == "OG" and seq_num not in expanded_residues:
                cb_coord = next(
                    (
                        c
                        for c, k, n in zip(new_coords, new_keys, new_names)
                        if k == key and n == "CB"
                    ),
                    coord,
                )

                # Expand side-chain atoms using helper
                for aname, acoord in _expand_phosphomimetic_sidechain(
                    cb_coord, target_mimic, state_code
                ):
                    new_coords.append(acoord)
                    new_keys.append(key)
                    new_names.append(aname)
                    new_res_names.append(target_mimic)

                expanded_residues.add(seq_num)
            elif clean_atom not in ("P", "O1P", "O2P", "O3P", "OG"):
                new_coords.append(coord)
                new_keys.append(key)
                new_names.append(clean_atom)
                new_res_names.append(target_mimic)
        else:
            new_coords.append(coord)
            new_keys.append(key)
            new_names.append(atom_name)
            new_res_names.append(res_name)

    mutated_dict["all_atom_coords"] = np.array(
        new_coords, dtype=mutated_dict["all_atom_coords"].dtype
    )
    mutated_dict["all_atom_keys"] = new_keys
    mutated_dict["all_atom_names"] = new_names
    mutated_dict["all_atom_res_names"] = new_res_names

    return mutated_dict
