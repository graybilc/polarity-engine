#!/usr/bin/env python3


import argparse
import logging
import os
import requests
from Bio import PDB
from pathlib import Path
from urllib.error import URLError


# Configure log
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def retrieve_fasta(uniprot_id: str) -> str:
    """
    Fetch the raw fasta sequence data from UniProt.

    Args:
        uniprot_id (str) : UniProt API ID to programmatically retrieve FASTA
        of the desired protein.

    Returns:
        str: string object containing metadata and amino acid sequence for the specified
        protein separated by a newline character. Raises error if client-side error or server error.
    """
    logging.info(
        f"Starting FASTA data retrieval for Uniprot ID {uniprot_id}...")
    url_for_target = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url_for_target, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully fetched UniProt ID {uniprot_id}")
        return response.text
    except requests.exceptions.Timeout as t_err:
        logging.error(
            f"Timeout disruption resolved as [{type(t_err).__name__}]: Server or socket dropped.")
        raise

    except requests.exceptions.RequestException as e:
        logging.error(f"An explicit network connection anomaly occurred: {e}")
        raise


def save_text_to_disk(content: str, filename: str, output_dir: Path) -> Path:
    """
    Saves a raw text string to a physical file on disk within a specified directory Path.

    Args:
        content (str): The raw text content to write (e.g., raw FASTA text).
        filename (str): The target name of the file (e.g., "lgl_sequence.fasta").
        output_dir (Path): A pathlib.Path object representing the destination folder.

    Returns:
        Path: A pathlib.Path object pointing to the absolute path of the written file.
    """
    full_path = output_dir / filename

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        logging.info(f"Successfully wrote data to disk: {full_path}")

        return full_path.resolve()

    except IOError as io_err:
        logging.error(f"Failed to write file to {full_path}: {io_err}")
        raise


def download_pdb_files(pdb_id: str, output_dir: Path) -> Path:
    """
    Download the target protein structure from Protein Data Bank.

    Args:
        pdb_id (str): ID of the target protein for Protein Data Bank
        output_dir (Path): path to the directory where the output file is stored.

    Returns:
        final_path (Path): file path to .cif of the target protein. 
    """
    logging.info(f"Downloading PDB structure {pdb_id} from RCSB...")
    clean_id = pdb_id.strip().upper()
    try:
        pdbl = PDB.PDBList()

        raw_string_path = pdbl.retrieve_pdb_file(
            clean_id, pdir=output_dir, file_format="mmCif"
        )

        if not raw_string_path or not os.path.exists(raw_string_path):
            logging.error(
                f"Structure {clean_id} could not be pulled or written.")
            raise FileNotFoundError(
                f"Failed to write structural files for {clean_id}")

        # Cast the returned string back into a Path object
        final_path = Path(raw_string_path)
        logging.info(
            f"Successfully secured structural records at: {final_path}")
        return final_path

    except URLError as net_err:
        logging.error(
            f"Network timeout or server disruption hitting wwPDB: {net_err}"
        )
        raise RuntimeError(
            "Pipeline stopped: Remote server unreachable."
        ) from net_err
    except PermissionError as perm_err:
        logging.error(
            f"Write permissions denied on target storage '{output_dir}': {perm_err}"
        )
        raise
    except Exception as general_err:
        logging.error(
            f"Unexpected structural ingest anomaly detected: {general_err}")
        raise


def parse_arguments(args: list | None = None) -> argparse.Namespace:
    """
    Parse the command line arguments provided to fetch_data.py.

    Args:
        args (list[str] | None): A custom array of argument strings to evaluate. 
            Defaults to None, which forces fallback evaluation of sys.argv.
    Returns:
        argparse.Namespace: Namespace object containing:
            - name (str): name of each target protein
            - uniprot_id (str): UniProt API ID for the each target protein
            - pdb_id (str): PDB ID(s) for the target protein(s)
            - outdir (Path): Output directory to store data
    """
    parser = argparse.ArgumentParser(
        description="command line arguments provided to fetch_data.py"
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        nargs="+",
        required=True,
        help="Names for each UniProt API ID. If multiple, list items separated by space.",
    )

    parser.add_argument(
        "-u",
        "--uniprot_id",
        type=str,
        nargs="+",
        required=True,
        help="UniProt API ID for the target protein. If multiple, list items separated by space.",
    )

    parser.add_argument(
        "-p",
        "--pdb_id",
        type=str,
        nargs="+",
        required=True,
        help="PDB ID for the target protein. If multiple, list items separated by space.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        required=True,
        help="Output directory to store data"
    )
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    """
    main function will accomplish 2 things:
        1. Fetch amino acid sequnce of each target protein
        2. Download structural information file(s)

    Args:
        args (argparse.Namespace): 
            Namespace object containing:
               - name (str): target protein names
               - uniprot_id (str): UniProt API ID for each target protein
               - pdb_id (str): PDB ID for each target protein
    """
    # Initialize output directories
    seq_dir = args.outdir / "amino_acid_sequences"
    struct_dir = args.outdir / "structures"

    seq_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    if len(args.name) != len(args.uniprot_id):
        logging.error(
            "The number of names does not match the number of UniProt IDs.")
        raise ValueError("Mismatched parallel input arguments.")

    targets = dict(zip(args.name, args.uniprot_id))

    try:
        if len(args.name) != len(args.uniprot_id):
            raise ValueError(
                "The number of names does not match the number of UniProt IDs.")

        targets = dict(zip(args.name, args.uniprot_id))

        logging.info("--- PHASE 1: FETCHING AMINO ACID SEQUENCES ---")
        for name, uniprot_id in targets.items():
            raw_fasta_text = retrieve_fasta(uniprot_id)

            file_name = f"{name.lower()}_sequence.fasta"
            save_text_to_disk(raw_fasta_text, file_name, seq_dir)

        logging.info("--- PHASE 2: DOWNLOADING STRUCTURE FILES ---")
        output_file_paths = [download_pdb_files(
            pdb_id, struct_dir) for pdb_id in args.pdb_id]
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    # Retrieve the parsed arguments object
    args = parse_arguments()
    main(args)
