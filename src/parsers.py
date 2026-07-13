#!/urs/bin/env python3


import argparse
import logging
import numpy
import sys

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
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
    def validate_amino_acid_sequence(cls, sequence: str) -> bool:
        """
        Validate all the amino acids in the given sequence are valid.
        Non-standard amino acids are recorded in the log framework. 
        """
        if not sequence:
            return False

        validator_flag = True

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


def main(args_list: Sequence[str] | None = None) -> None:
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
