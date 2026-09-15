#!/urs/bin/env python3


import argparse
import freesasa
import logging
import numpy as np
import sys
import torch

from Bio.PDB import MMCIFParser, PDBParser
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
            sequences (dict[str, str]): A dictionary mapping header strings to clean sequence strings.
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
                        raw_sequence = "".join(current_sequence)
                        sequences[current_header] = cls._validate_amino_acid_sequence(
                            raw_sequence, current_header
                        )
                    current_header = line[1:].strip()
                    current_sequence = []
                else:
                    if current_header is None:
                        msg = f"Malformed FASTA in '{file_path.name}': Found sequence data before a header."
                        logger.error(msg)
                        raise ValueError(msg)
                    current_sequence.append(line.upper())

            if current_header:
                raw_sequence = "".join(current_sequence)
                sequences[current_header] = cls._validate_amino_acid_sequence(
                    raw_sequence, current_header
                )

        # --- Post-Parsing Validation ---
        if not sequences:
            msg = f"Malformed FASTA: File '{file_path.name}' contains no entries."
            logger.error(msg)
            raise ValueError(msg)

        return sequences

    @classmethod
    def _validate_amino_acid_sequence(cls, sequence: str, header: str) -> str:
        """
        Validate all the amino acids in the given sequence are valid.
        Non-standard amino acids are recorded in the log framework.

        Args:
            sequence (str): Amino acid sequence to be validated.
            header (str): The corresponding fasta header for explicit error tracking.
        Returns:
            cleaned_seq (str): Validated amino acid sequence.
        """
        cleaned_seq = "".join(sequence.split()).upper()

        if not cleaned_seq:
            msg = f"Malformed FASTA: Empty sequence string encountered under header '{header}'."
            logger.error(msg)
            raise ValueError(msg)

        for i, aa in enumerate(cleaned_seq):
            if aa not in cls.VALID_AA:
                msg = f"Invalid amino acid '{aa}' found at position {i + 1} under header '{header}'."
                logger.error(msg)
                raise ValueError(msg)

            if aa in cls.NON_STANDARD_AA_SET:
                logger.warning(
                    f"Non-standard amino acid '{aa}' found at position {i + 1} under header '{header}'."
                )

        return cleaned_seq


class StructureParser:
    """
    Parses protein structure files (PDB/mmCIF) to extract 3D coordinates
    and biophysical surface properties (SASA).

    Handles raw file I/O, BioPython structure parsing, coordinate extractions,
    and native C-binding surface accessibility calculations via FreeSASA.

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
                    file_path.stem, str(file_path)
                )
                if len(structure) > 1:
                    logger.warning(
                        f"File '{file_path.name}' contains {len(structure)} models. Defaulting to Model 0."
                    )
                return list(structure[0].child_dict.keys()), structure
            except Exception as e:
                raise ValueError(
                    f"Malformed PDB file structure in '{file_path.name}': {e}"
                )

        elif file_ext in (".cif", ".mmcif"):
            try:
                mmcif_dict = MMCIF2Dict(str(file_path))
                key = "_atom_site.auth_asym_id"
                if key not in mmcif_dict:
                    raise ValueError(f"Missing essential key '{key}'")

                seen = set()
                chains = [c for c in mmcif_dict[key] if not (c in seen or seen.add(c))]
                return chains, mmcif_dict
            except Exception as e:
                raise ValueError(f"Malformed mmCIF file in '{file_path.name}': {e}")

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

        # Inspect returns the chains AND the already-loaded data structure
        available_chains, loaded_data = cls._load_and_inspect(path_obj, file_ext)
        if not available_chains:
            raise ValueError(f"No chains found in structural file '{path_obj.name}'")

        results = {}
        if chain_id is not None:
            if chain_id not in available_chains:
                raise ValueError(
                    f"Requested chain '{chain_id}' not found in {available_chains}"
                )

            if file_ext == ".pdb":
                results[chain_id] = cls._parse_legacy_pdb(loaded_data, chain_id)
            else:
                results[chain_id] = cls._parse_mmcif_fast_path(loaded_data, chain_id)
            return results

        # Dynamic multi-chain collection using the same loaded_data
        for c_id in available_chains:
            try:
                if file_ext == ".pdb":
                    results[c_id] = cls._parse_legacy_pdb(loaded_data, c_id)
                else:
                    results[c_id] = cls._parse_mmcif_fast_path(loaded_data, c_id)
            except ValueError:
                continue

        return results

    @classmethod
    def _parse_legacy_pdb(
        cls, structure: Structure, chain_id: str
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Extracts CA coordinates, amino acid residues, B-factors, and occupancy values directly from
        a pre-loaded Biopython Structure object.

        Args:
            structure (Structure): Pre-loaded Biopython Structure object.
            chain_id (str): Target chain identifier.

        Returns:
            dict[str, dict[str, np.ndarray]]: Dictionary mapping chain_id to
            arrays for 'coords', 'b_factors', and 'occupancies' as well as list of amino acid residues.
        """
        model = structure[0]
        chain = model[chain_id]
        ca_coordinates, aa_residues, b_factors, occupancies = [], [], [], []

        for residue in chain:
            # Skip heteroatoms/water and verify CA exists
            if residue.id[0] == " " and "CA" in residue:
                atom = residue["CA"]

                # Unpack disordered atoms dynamically to get the primary conformation
                if atom.is_disordered():
                    atom = atom.selected_child

                ca_coordinates.append(atom.get_coord())
                aa_residues.append(atom.get_residue())
                b_factors.append(atom.get_bfactor())
                occupancies.append(atom.get_occupancy())

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return {
            "coords": np.array(ca_coordinates, dtype=np.float32),
            "aa_residues": aa_residues,
            "b_factors": np.array(b_factors, dtype=np.float32),
            "occupancies": np.array(occupancies, dtype=np.float32),
        }

    @classmethod
    def _parse_mmcif_fast_path(
        cls, mmcif_dict: dict, chain_id: str
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Parses mmCIF coordinates from a pre-loaded MMCIF2Dict by extracting
        Alpha Carbon (CA) positions directly from the internal arrays.

        Args:
            mmcif_dict (dict): MMCIF2Dict object for the target project.
            chain_id (str): Name used for the distinct, covalently linked macromolecule in the structure file.
        Returns:
            chain_data (dict[str, np.ndarray]): dict with chain_id as the key and CA coordinates,
            amino acid residues, B-factor, and Occupancy in the specified chain.
        """
        required_keys = [
            "_atom_site.group_PDB",
            "_atom_site.auth_asym_id",
            "_atom_site.label_atom_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.auth_comp_id",
            "_atom_site.B_iso_or_equiv",  # B-factor / local resolution proxy
            "_atom_site.occupancy",  # Occupancy fraction
        ]

        for key in required_keys:
            if key not in mmcif_dict:
                msg = f"Malformed mmCIF structure: Missing required data field '{key}'"
                logger.error(msg)
                raise ValueError(msg)

        # Normalize single-element string values into lists for safe zipping
        cols = {
            k: [mmcif_dict[k]] if isinstance(mmcif_dict[k], str) else mmcif_dict[k]
            for k in required_keys
        }

        ca_coordinates, aa_residues, b_factors, occupancies = [], [], [], []

        # Parallel iteration over the coordinate columns
        for group, chain, atom_name, x, y, z, aa, b_val, occ_val in zip(
            cols["_atom_site.group_PDB"],
            cols["_atom_site.auth_asym_id"],
            cols["_atom_site.label_atom_id"],
            cols["_atom_site.Cartn_x"],
            cols["_atom_site.Cartn_y"],
            cols["_atom_site.Cartn_z"],
            cols["_atom_site.auth_comp_id"],
            cols["_atom_site.B_iso_or_equiv"],
            cols["_atom_site.occupancy"],
        ):
            # Isolate standard polymer atoms (ATOM), the target chain, and Alpha Carbons (CA)
            if group == "ATOM" and chain == chain_id and atom_name == "CA":
                try:
                    ca_coordinates.append([float(x), float(y), float(z)])
                    aa_residues.append(str(aa))
                    b_factors.append(float(b_val))
                    occupancies.append(float(occ_val))
                except ValueError as e:
                    msg = f"Non-numeric spatial coordinates encountered in mmCIF dictionary: {e}"
                    logger.error(msg)
                    raise ValueError(msg)

        if not ca_coordinates:
            raise ValueError(
                f"No valid Alpha Carbon (CA) atoms found for chain '{chain_id}'"
            )

        return {
            "coords": np.array(ca_coordinates, dtype=np.float32),
            "aa_residues": aa_residues,
            "b_factors": np.array(b_factors, dtype=np.float32),
            "occupancies": np.array(occupancies, dtype=np.float32),
        }

    @staticmethod
    def get_freesasa_result(file_path: Path) -> tuple[freesasa.Result, Any]:
        """Parses a PDB or mmCIF file and calculates SASA via FreeSASA's BioPython bridge.

        Args:
            file_path (Path): Path to the structure file (.pdb or .cif/.mmcif).

        Returns:
            tuple[freesasa.Result, Any]: FreeSASA result and root tree node object.

        Raises:
            ValueError: If the file extension is unsupported or structure parsing fails.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        try:
            if file_ext in (".cif", ".mmcif"):
                parser = MMCIFParser(QUIET=True)
                bio_structure = parser.get_structure(file_path.stem, str(file_path))
            elif file_ext == ".pdb":
                parser = PDBParser(QUIET=True)
                bio_structure = parser.get_structure(file_path.stem, str(file_path))
            else:
                raise ValueError(
                    f"Unsupported extension '{file_ext}' for FreeSASA parsing."
                )
        except Exception as e:
            raise ValueError(
                f"Failed to parse structure file '{file_path.name}': {e}"
            ) from e

        # 1. Convert BioPython Structure -> freesasa.Structure
        fs_structure = freesasa.structureFromBioPDB(bio_structure)

        # 2. Run SASA calculation
        result = freesasa.calc(fs_structure)

        return result, fs_structure

    @staticmethod
    def extract_per_residue_sasa(
        result: freesasa.Result, fs_structure: freesasa.Structure
    ) -> dict[tuple[str, str], float]:
        """Aggregates atomic SASA values per residue across the complex.

        Args:
            result (freesasa.Result): FreeSASA computation result.
            fs_structure (freesasa.Structure): FreeSASA structure instance.

        Returns:
            dict[tuple[str, str], float]: Mapping of (chain_id, res_number_str) -> raw SASA (Å²).
        """
        residue_sasa_map = {}
        n_atoms = fs_structure.nAtoms()

        for i in range(n_atoms):
            chain_id = fs_structure.chainLabel(i)
            res_num = fs_structure.residueNumber(i).strip()
            atom_area = result.atomArea(i)

            key = (chain_id, res_num)
            residue_sasa_map[key] = residue_sasa_map.get(key, 0.0) + atom_area

        return residue_sasa_map
