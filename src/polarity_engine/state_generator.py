"""
polarity_engine/state_generator.py
Generates mutated 3^3 structural variation grid representations directly from
parsed structural dictionaries, supporting position-specific native SEP retention
for experimental ground-truth alignment.
"""

from typing import Any
import copy
import numpy as np
import torch


class StateGenerator:
    """
    Handles the generation of 3^3 structural variation grid states from
    wild-type or phosphorylated parsed structure dictionaries.

    Attributes:
        target_sites (dict[int, str]): Mapping of bit positions to residue numbers.
        default_mimic (str): Default 3-letter code for phosphomimetic residues ('ASP' or 'GLU').
    """

    DEFAULT_APKC_TARGETS = {
        0: "655",  # Site 1 index -> S655
        1: "659",  # Site 2 index -> S659
        2: "663",  # Site 3 index -> S663 (Native SEP site in 8R3Y)
    }

    def __init__(
        self,
        target_sites: dict[int, str] | None = None,
        default_mimic: str = "ASP",
    ) -> None:
        """
        Initializes the StateGenerator with target sites and default mimic configuration.

        Args:
            target_sites (dict[int, str] | None, optional): Mapping of positions to residue numbers.
                Defaults to `DEFAULT_APKC_TARGETS`.
            default_mimic (str, optional): Default phosphomimetic residue ('ASP' or 'GLU').
                Defaults to "ASP".

        Returns:
            None
        """
        self.target_sites = (
            target_sites if target_sites is not None else self.DEFAULT_APKC_TARGETS
        )
        self.default_mimic = default_mimic

    def resolve_state_mutations(
        self, state_code: str, mimic_res: str | None = None
    ) -> dict[str, str]:
        """
        Maps a ternary state code to residue mutation substitutions, respecting
        native experimental SEP retention for position 663 (Site 3).

        Args:
            state_code (str): Ternary string representation (e.g. '000', '111', '222', '120').
                '0' = Wild-type (SER/THR)
                '1' = Phosphorylated / Phosphomimetic (SEP for 663; ASP/GLU for others)
                '2' = Alanine (ALA)
            mimic_res (str | None, optional): Override for 3-letter phosphomimetic code.
                Defaults to `self.default_mimic`.

        Returns:
            dict[str, str]: Dictionary mapping target residue numbers to their target amino acid names.
        """
        mimic = mimic_res if mimic_res is not None else self.default_mimic

        if state_code.upper() == "AAA":
            state_code = "222"

        if len(state_code) != len(self.target_sites) or not set(state_code).issubset(
            {"0", "1", "2"}
        ):
            raise ValueError(
                f"Invalid state_code '{state_code}'. Must be a {len(self.target_sites)}-character ternary string using '0', '1', '2'."
            )

        mutations = {}
        for idx, val in enumerate(state_code):
            res_num = self.target_sites[idx]
            if val == "1":
                # Position-specific logic: Site 663 uses native experimental SEP; others use mimic
                if res_num == "663":
                    mutations[res_num] = "SEP"
                else:
                    mutations[res_num] = mimic
            elif val == "2":
                mutations[res_num] = "ALA"
            # '0' remains wild-type

        return mutations

    @staticmethod
    def _validate_parsed_dict_contract(parsed_dict: dict[str, Any]) -> None:
        """
        Enforces a rigid data contract on the input structure dictionary.

        Args:
            parsed_dict (dict[str, Any]): Input structure dictionary to validate.

        Returns:
            None: Raises KeyError if any required schema keys are missing.
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
                f"Invalid structure dictionary contract. Missing required schema keys: {missing}."
            )

    def _update_sequence_metadata(
        self, mutated_dict: dict[str, Any], mutation_map: dict[str, str]
    ) -> None:
        """
        Updates backbone residue lists, amino acid lists, and node descriptors with mutated residue names.

        Args:
            mutated_dict (dict[str, Any]): Dictionary containing structural metadata arrays.
            mutation_map (dict[str, str]): Dictionary mapping target residue sequence numbers
                to their substituted residue names.

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
        self,
        cb_coord: np.ndarray,
        target_mimic: str,
    ) -> list[tuple[str, np.ndarray]]:
        """
        Calculates stereochemically refined 3D Cartesian coordinates for GLU or ASP
        phosphomimetic side-chain atoms.

        Args:
            cb_coord (np.ndarray): 3D coordinate vector of the C-beta (CB) atom.
            target_mimic (str): 3-letter amino acid code for the phosphomimetic residue ('GLU' or 'ASP').

        Returns:
            list[tuple[str, np.ndarray]]: List of heavy-atom names paired with their refined 3D coordinates.
        """
        cb_arr = np.asarray(cb_coord, dtype=np.float64)
        cg_dir = np.array([0.577, 0.577, 0.577])
        cg_dir = cg_dir / np.linalg.norm(cg_dir)
        cg_coord = cb_arr + (cg_dir * 1.53)

        if target_mimic == "GLU":
            cd_dir = np.array([0.707, 0.707, 0.0])
            cd_dir = cd_dir / np.linalg.norm(cd_dir)
            cd_coord = cg_coord + (cd_dir * 1.52)

            oe1_coord = cd_coord + np.array([0.85, 0.50, 0.0])
            oe2_coord = cd_coord + np.array([-0.20, 1.10, 0.0])

            return [
                ("CG", cg_coord),
                ("CD", cd_coord),
                ("OE1", oe1_coord),
                ("OE2", oe2_coord),
            ]
        else:  # ASP
            od1_coord = cg_coord + np.array([0.91, 0.35, 0.0])
            od2_coord = cg_coord + np.array([0.15, 1.15, 0.0])

            return [
                ("CG", cg_coord),
                ("OD1", od1_coord),
                ("OD2", od2_coord),
            ]

    def mutate_structure_dict(
        self,
        wt_parsed_dict: dict[str, Any],
        state_code: str,
        target_chain: str = "B",
        mimic_res: str | None = None,
    ) -> dict[str, Any]:
        """
        Applies generalized ternary state mutations to a wild-type parsed structure dictionary,
        retaining native crystallographic SEP coordinates at position 663 when requested.

        Args:
            wt_parsed_dict (dict[str, Any]): Wild-type parsed structure dictionary.
            state_code (str): Ternary state code string (e.g., '000', '111', '222', '121').
            target_chain (str, optional): Target chain identifier. Defaults to "B".
            mimic_res (str | None, optional): Override for phosphomimetic residue ('ASP' or 'GLU').
                Defaults to `self.default_mimic`.

        Returns:
            dict[str, Any]: Mutated structure dictionary containing updated node features and atom coordinates.
        """
        self._validate_parsed_dict_contract(wt_parsed_dict)
        mimic = mimic_res if mimic_res is not None else self.default_mimic

        mutated_dict = copy.deepcopy(wt_parsed_dict)
        mutated_dict.pop("rsasa", None)

        mutation_map = self.resolve_state_mutations(state_code, mimic_res=mimic)
        self._update_sequence_metadata(mutated_dict, mutation_map)

        standard_ala_atoms = {"N", "CA", "C", "O", "CB"}
        new_coords, new_keys, new_names, new_res_names = [], [], [], []
        expanded_residues = set()

        for coord, key, atom_name, res_name in zip(
            mutated_dict["all_atom_coords"],
            mutated_dict["all_atom_keys"],
            mutated_dict["all_atom_names"],
            mutated_dict["all_atom_res_names"],
        ):
            chain_id, seq_num = key
            target_mutation = mutation_map.get(seq_num)
            clean_atom = atom_name.upper().strip()

            if chain_id == target_chain and target_mutation == "ALA":
                if clean_atom in standard_ala_atoms:
                    new_coords.append(coord)
                    new_keys.append(key)
                    new_names.append(clean_atom)
                    new_res_names.append("ALA")
            elif chain_id == target_chain and target_mutation == "SEP":
                # Retain native experimental SEP atoms directly from 8R3Y crystal structure
                new_coords.append(coord)
                new_keys.append(key)
                new_names.append(clean_atom)
                new_res_names.append("SEP")
            elif chain_id == target_chain and target_mutation in ("GLU", "ASP"):
                if clean_atom in ("N", "CA", "C", "O", "CB"):
                    new_coords.append(coord)
                    new_keys.append(key)
                    new_names.append(clean_atom)
                    new_res_names.append(target_mutation)
                elif clean_atom == "OG" and seq_num not in expanded_residues:
                    cb_coord = next(
                        (
                            c
                            for c, k, n in zip(new_coords, new_keys, new_names)
                            if k == key and n == "CB"
                        ),
                        coord,
                    )

                    for aname, acoord in self._expand_phosphomimetic_sidechain(
                        cb_coord, target_mutation
                    ):
                        new_coords.append(acoord)
                        new_keys.append(key)
                        new_names.append(aname)
                        new_res_names.append(target_mutation)

                    expanded_residues.add(seq_num)
                elif clean_atom not in ("P", "O1P", "O2P", "O3P", "OG"):
                    new_coords.append(coord)
                    new_keys.append(key)
                    new_names.append(clean_atom)
                    new_res_names.append(target_mutation)
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

    def mutate_sep_to_ser(
        self,
        parsed_dict: dict[str, Any],
        target_res_num: str = "663",
        target_chain: str = "B",
    ) -> dict[str, Any]:
        """
        Mutates a native phosphoserine (SEP) or phosphorylated residue back to serine (SER).

        Args:
            parsed_dict (dict[str, Any]): Parsed structural data dictionary containing SEP.
            target_res_num (str, optional): Residue number to mutate. Defaults to "663".
            target_chain (str, optional): Chain identifier. Defaults to "B".

        Returns:
            dict[str, Any]: A deep copy of the dictionary with SEP converted to SER.
        """
        self._validate_parsed_dict_contract(parsed_dict)

        mutated_dict = copy.deepcopy(parsed_dict)
        mutated_dict.pop("rsasa", None)

        mutation_map = {target_res_num: "SER"}
        self._update_sequence_metadata(mutated_dict, mutation_map)

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
                and res_name in ("SEP", "SER", "ASP")
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
