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
        Example for state_code '101': {'655': 'GLU', '663': 'GLU'}

    Raises
    ------
    ValueError
        If `state_code` length does not match the number of target sites or contains non-binary characters.
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


def mutate_structure_dict(
    wt_parsed_dict: dict[str, Any], state_code: str, target_chain: str = "B"
) -> dict[str, Any]:
    """
    Applies phosphomimetic mutations to a wild-type parsed structure dictionary.

    Takes a parsed mmCIF dictionary structure—either flat or nested under a chain ID—
    and maps binary state permutation codes (e.g., '110') to specific amino acid
    substitutions (e.g., SER -> GLU) at target phosphorylation sites.

    Parameters
    ----------
    wt_parsed_dict : dict[str, Any]
        Wild-type parsed structural data dictionary containing core residue and
        coordinate keys:
        - 'nodes': List[Tuple[str, str, str]] of (chain_id, seq_num, 3_letter_code)
        - 'aa_residues': List[Tuple[str, str]] of (seq_num, 3_letter_code) (optional,
          inferred from 'nodes' if omitted)
        - 'aa_list': List[str] of 3-letter residue names (optional)
        Alternatively, data may be nested under `target_chain` key: `{target_chain: {...}}`.
    state_code : str
        An N-bit binary string representation (e.g., '000', '110', '111') specifying
        the target phosphorylation state across active target sites. Length must match
        the number of target sites configured in `resolve_state_mutations`.
    target_chain : str, default="B"
        The chain identifier to check for if `wt_parsed_dict` is chain-nested.

    Returns
    -------
    dict[str, Any]
        A deep copy of the original dictionary with updated residue identities in:
        - `'aa_residues'`
        - `'aa_list'` (if present)
        - `'nodes'` (if present)
        All spatial atomic coordinates and auxiliary values remain preserved.

    Raises
    ------
    KeyError
        If no valid structure dictionary is found, or if neither `'aa_residues'`
        nor `'nodes'` exists in the resolved data dictionary.
    ValueError
        If `state_code` length does not match target sites or contains non-binary characters.

    Examples
    --------
    >>> wt_dict = {'nodes': [('B', '655', 'SER'), ('B', '656', 'LEU')]}
    >>> mutated = mutate_structure_dict(wt_dict, state_code='100')
    >>> mutated['aa_residues'][0]
    ('655', 'GLU')
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

    # 3. Resolve mutations
    mutation_map = resolve_state_mutations(state_code)

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
