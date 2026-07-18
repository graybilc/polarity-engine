#!/urs/bin/env python3


import argparse
import logging
import numpy as np
import sys

from Bio.PDB import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure
from pathlib import Path
from typing import Any, Sequence

# Configure log
logger = logging.getLogger(__name__)


class FastaParser:
    """
    Parses and validates FASTA sequence files from scratch.
    """
    STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
    NON_STANDARD_AA = "UOBZXJ*"

    # Convert to a set for efficient O(1) membership lookups
    VALID_AA = set(STANDARD_AA + NON_STANDARD_AA)
    NON_STANDARD_AA_SET = set(NON_STANDARD_AA)

    @classmethod
    def parse_fasta(cls, file_path: str | Path) -> dict[str, str]:
        """
        Parses a FASTA file line-by-line from scratch.

        Args:
            file_path (str | Path): File path to the desired FASTA file.
        Returns:
            sequences (Dict[str, str]): a dictionary mapping header strings to clean sequence strings.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            msg = f"Target FASTA file not found or invalid: '{file_path.resolve()}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        sequences = {}
        current_header = None
        current_sequence = []

        with open(file_path, "r", encoding="utf-8") as fasta_reader:
            for line in fasta_reader:
                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):
                    if current_header:
                        sequences[current_header] = "".join(current_sequence)
                    current_header = line[1:]
                    current_sequence = []
                else:
                    if current_header is None:
                        msg = "Malformed FASTA: Found sequence data before a header."
                        logger.error(msg)
                        raise ValueError(msg)
                    current_sequence.append(line.upper())

            if current_header:
                sequences[current_header] = "".join(current_sequence)

        # --- Post-Parsing Validation ---
        if not sequences:
            msg = f"Malformed FASTA: File '{file_path.name}' contains no data."
            logger.error(msg)
            raise ValueError(msg)

        for header, seq in sequences.items():
            if not seq:
                msg = f"Malformed FASTA: Header '>{header}' has no associated sequence data."
                logger.error(msg)
                raise ValueError(msg)
        return sequences

    @classmethod
    def _validate_amino_acid_sequence(cls, sequence: str) -> bool:
        """
        Validate all the amino acids in the given sequence are valid.
        Non-standard amino acids are recorded in the log framework. 
        """
        validator_flag = True
        if not sequence:
            validator_flag = False
            logger.error("Empty sequence string was passed.")
            return validator_flag

        for i, aa in enumerate(sequence):
            if aa not in cls.VALID_AA:
                logger.error(
                    f"Invalid amino acid '{aa}' found at position {i + 1}")
                validator_flag = False
                continue

            if aa in cls.NON_STANDARD_AA_SET:
                logger.warning(
                    f"Non-standard amino acid '{aa}' found at position {i + 1}")

        return validator_flag


class StructureParser:
    """
    Parses protein structure files (PDB/mmCIF) to extract 3D coordinates.

    Design Note:
        For mmCIF parsing, we deliberately bypass Biopython's full SMCRA 
        (Structure/Model/Chain/Residue/Atom) object tree representation 
        (via MMCIFParser) in favor of the lower-level MMCIF2Dict representation. 
        This avoids heavy Python object instantiation overhead, achieving a 
        high-speed array-extraction "fast path" suitable for deep learning pipelines.
        For PDB parsing, we will use Biopython's standard PDBParser since the SMCRA
        overhead are unlikely to bottleneck us.
    """
    @classmethod
    def _validate_structure_file(cls, file_path: str | Path) -> str:
        """
        Verify the provided file exists.

        Args:
            file_path (str | Path): File path to the desired structure file.
        Returns:
            file_extension (str): If file exists, the lowercased file extension will be returned.
        """
        file_path = Path(file_path)

        if not file_path.is_file():
            msg = f"Target structure file not found: '{file_path.resolve()}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        return file_path.suffix.lower()

    @classmethod
    def _load_and_inspect(cls, file_path: Path, file_ext: str) -> tuple[list[str], any]:
        """
        Loads the structure file and extracts unique chain IDs alongside 
        the raw parsed data object.

        Args:
            file_path (Path): File path to the desired structure file.
            file_ext (str): Lowercased extension of the structure file.
        Returns:
            tuple[list[str], any]: A list of unique chain IDs, and the loaded data 
                object (Structure or MMCIF2Dict).
        """
        if file_ext == ".pdb":
            try:
                structure = PDBParser(QUIET=True).get_structure(
                    file_path.stem, str(file_path))
                if len(structure) > 1:
                    logger.warning(
                        f"File '{file_path.name}' contains {len(structure)} models. Defaulting to Model 0.")
                return list(structure[0].child_dict.keys()), structure
            except Exception as e:
                raise ValueError(
                    f"Malformed PDB file structure in '{file_path.name}': {e}")

        elif file_ext in (".cif", ".mmcif"):
            try:
                mmcif_dict = MMCIF2Dict(str(file_path))
                key = "_atom_site.auth_asym_id"
                if key not in mmcif_dict:
                    raise ValueError(f"Missing essential key '{key}'")

                seen = set()
                chains = [c for c in mmcif_dict[key]
                          if not (c in seen or seen.add(c))]
                return chains, mmcif_dict
            except Exception as e:
                raise ValueError(
                    f"Malformed mmCIF file in '{file_path.name}': {e}")

        return [], None

    @classmethod
    def get_alpha_carbon_coordinates(
        cls, file_path: str | Path, chain_id: str | None = None
    ) -> dict[str, np.ndarray]:
        """
        Main entry point to extract CA coordinates from either PDB or mmCIF files.

        Args:
            file_path (str | Path): File path to the desired structure file.
            chain_id (str): name used for the distinct, covalently linked macromolecule in the structure file.
        Returns:
            results (dict[str, np.ndarray]): dict with chain_id as the key and loaded data as value 
        """
        file_ext = cls._validate_structure_file(file_path)
        path_obj = Path(file_path)

        # 1. Inspect returns the chains AND the already-loaded data structure
        available_chains, loaded_data = cls._inspect_chains(path_obj, file_ext)
        if not available_chains:
            raise ValueError(
                f"No chains found in structural file '{path_obj.name}'")

        results = {}
        if chain_id is not None:
            if chain_id not in available_chains:
                raise ValueError(
                    f"Requested chain '{chain_id}' not found in {available_chains}")

            if file_ext == ".pdb":
                results[chain_id] = cls._parse_legacy_pdb(
                    loaded_data, chain_id)
            else:
                results[chain_id] = cls._parse_mmcif_fast_path(
                    loaded_data, chain_id)
            return results

        # Dynamic multi-chain collection using the same loaded_data
        for c_id in available_chains:
            try:
                if file_ext == ".pdb":
                    results[c_id] = cls._parse_legacy_pdb(loaded_data, c_id)
                else:
                    results[c_id] = cls._parse_mmcif_fast_path(
                        loaded_data, c_id)
            except ValueError:
                continue

        return results

    @classmethod
    def _parse_legacy_pdb(cls, structure: Structure, chain_id: str) -> np.ndarray:
        """
        Extracts CA coordinates directly from a pre-loaded Biopython Structure.

        Args:
            structure (Structure): Pre-loaded Biopython Structure object for the target protein.
            chain_id (str): Name used for the distinct, covalently linked macromolecule in the structure file.    
        Returns:
            np.ndarray: CA coordinates in the specified chain.
        """
        model = structure[0]
        chain = model[chain_id]
        ca_coordinates = []

        for residue in chain:
            # Skip heteroatoms/water and verify CA exists
            if residue.id[0] == " " and "CA" in residue:
                atom = residue["CA"]

                # Unpack disordered atoms dynamically to get the primary conformation
                if atom.is_disordered():
                    atom = atom.selected_child

                ca_coordinates.append(atom.get_coord())

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return np.array(ca_coordinates, dtype=np.float32)

    @classmethod
    def _parse_mmcif_fast_path(cls, mmcif_dict: dict, chain_id: str) -> np.ndarray:
        """
        Parses mmCIF coordinates from a pre-loaded MMCIF2Dict by extracting 
        Alpha Carbon (CA) positions directly from the internal arrays.

        Args:
            mmcif_dict (dict): MMCIF2Dict object for the target project.
            chain_id (str): Name used for the distinct, covalently linked macromolecule in the structure file.
        Returns:
            np.ndarray: CA coordinates in the specified chain.
        """
        # Check for essential structural data keys
        required_keys = [
            "_atom_site.group_PDB",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]

        for key in required_keys:
            if key not in mmcif_dict:
                msg = f"Malformed mmCIF structure: Missing required data field '{key}'"
                logger.error(msg)
                raise ValueError(msg)

        ca_coordinates = []

        # Parallel iteration over the coordinate columns
        for group, chain, atom_name, x, y, z in zip(
            mmcif_dict["_atom_site.group_PDB"],
            mmcif_dict["_atom_site.auth_asym_id"],
            mmcif_dict["_atom_site.auth_atom_id"],
            mmcif_dict["_atom_site.Cartn_x"],
            mmcif_dict["_atom_site.Cartn_y"],
            mmcif_dict["_atom_site.Cartn_z"],
        ):
            # Isolate standard polymer atoms (ATOM), the target chain, and Alpha Carbons (CA)
            if group == "ATOM" and chain == chain_id and atom_name == "CA":
                try:
                    ca_coordinates.append([float(x), float(y), float(z)])
                except ValueError as e:
                    msg = f"Non-numeric spatial coordinates encountered in mmCIF dictionary: {e}"
                    logger.error(msg)
                    raise ValueError(msg)

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'")

        return np.array(ca_coordinates, dtype=np.float32)

# def main(args_list: Sequence[str] | None = None) -> None:
#     pass


# if __name__ == "__main__":
#     main(sys.argv[1:])
