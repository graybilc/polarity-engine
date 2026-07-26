#!/usr/bin/env python3


import argparse
import logging
import os
import requests
import sys
from tqdm import tqdm
from typing import Sequence

from Bio import PDB
from pathlib import Path
from urllib.error import URLError


# Configure log
logger = logging.getLogger(__name__)


class ProteinDataIngestor(object):
    """
    A utility class to handle data files for amino acid seuence, structure, and density map.
    """

    def __init__(self, output_dir: Path):
        """
        Constructor of ProteinDataIngestor class

        Args:
            output_dir (Path) : A pathlib.Path object representing the destination folder
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Persistent connection for connection pooling
        self.session = requests.Session()

    def fetch_fasta(self, uniprot_id: str, target_name: str) -> str:
        """
        Fetch the raw fasta sequence data from UniProt.

        Args:
            uniprot_id (str): UniProt API ID to programmatically retrieve FASTA
            of the desired protein.
            target_name (str): name of the target protein

        Returns:
            str: string object containing metadata and amino acid sequence for the specified
            protein separated by a newline character. Raises error if client-side error or server error.
        """
        seq_dir = self.output_dir / "amino_acid_sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)

        output_filename = seq_dir / f"{target_name.lower()}_sequence.fasta"
        if output_filename.exists():
            logger.info(
                f"FASTA file for {uniprot_id} already exists in output directory. Skip fetching sequence step")
            with open(output_filename, "r", encoding="utf-8") as f:
                file_content = f.read()
                return file_content
        else:
            logger.info(
                f"Starting FASTA data retrieval for Uniprot ID {uniprot_id}...")
            url_for_target = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            try:
                response = self.session.get(url_for_target, timeout=10)
                response.raise_for_status()
                logger.info(f"Successfully fetched UniProt ID {uniprot_id}")
                output_filename.write_text(response.text, encoding="utf-8")
                logger.info(
                    f"Successfully wrote data to disk: {output_filename}")
                return response.text
            except requests.exceptions.Timeout as t_err:
                logger.error(
                    f"Timeout disruption resolved as [{type(t_err).__name__}]: Server or socket dropped.")
                raise

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"An explicit network connection anomaly occurred: {e}")
                raise

    def fetch_cif(self, pdb_id: str) -> Path:
        """
        Download the target protein structure from Protein Data Bank.

        Args:
            pdb_id (str): ID of the target protein for Protein Data Bank
            output_dir (Path): path to the directory where the output file is stored.

        Returns:
            final_path (Path): file path to .cif of the target protein.
        """
        structure_dir = self.output_dir / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading PDB structure {pdb_id} from RCSB...")
        clean_id = pdb_id.strip().upper()
        try:
            pdbl = PDB.PDBList()

            raw_string_path = pdbl.retrieve_pdb_file(
                clean_id, pdir=structure_dir, file_format="mmCif"
            )

            if not raw_string_path or not os.path.exists(raw_string_path):
                logger.error(
                    f"Structure {clean_id} could not be pulled or written.")
                raise FileNotFoundError(
                    f"Failed to write structural files for {clean_id}")

            # Cast the returned string back into a Path object
            final_path = Path(raw_string_path)
            logger.info(
                f"Successfully secured structural records at: {final_path}")
            return final_path

        except URLError as net_err:
            logger.error(
                f"Network timeout or server disruption hitting PDB: {net_err}"
            )
            raise RuntimeError(
                "Pipeline stopped: Remote server unreachable."
            ) from net_err
        except PermissionError as perm_err:
            logger.error(
                f"Write permissions denied on target storage '{self.output_dir}': {perm_err}"
            )
            raise
        except Exception as general_err:
            logger.error(
                f"Unexpected structural ingest anomaly detected: {general_err}")
            raise

    def fetch_em_density_map(self, emdb_id: str) -> Path:
        """
        Download EM density map from wwPDB

        Args:
            emdb_id (str): ID of the target structure for wwPDB

        Returns:
             final_path (Path): file path to map of the target structure.
        """
        emdb_id_clean = emdb_id.upper()
        file_id = emdb_id_clean.lower().replace("-", "_")

        density_map_dir = self.output_dir / "density_maps"
        density_map_dir.mkdir(parents=True, exist_ok=True)

        final_path = density_map_dir / f"{emdb_id_clean.lower()}.map"
        if final_path.exists():
            logger.info(f"Using cached EM density map at: {final_path}")
            return final_path
        else:
            try:
                emdb_url = f"https://files.rcsb.org/pub/emdb/structures/{emdb_id_clean}/map/{file_id}.map.gz"
                logger.info(
                    f"Streaming map data for {emdb_id_clean} from wwPDB...")

                # CRITICAL FIX: stream=True enables real-time chunked streaming over the wire
                with self.session.get(emdb_url, timeout=30, stream=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('Content-Length', 0))

                    # Compound with-statement drastically flattens nesting levels
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {emdb_id_clean}") as pbar, \
                            open(final_path, "wb") as f:

                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                logger.info(
                    f"Successfully secured compressed EM density map at: {final_path}")
                return final_path

            except URLError as net_err:
                logger.error(
                    f"Network timeout or server disruption hitting EMDB: {net_err}"
                )
                raise RuntimeError(
                    "Pipeline stopped: Remote server unreachable."
                ) from net_err
            except PermissionError as perm_err:
                logger.error(
                    f"Write permissions denied on target storage '{density_map_dir}': {perm_err}"
                )
                raise
            except Exception as general_err:
                if final_path.exists():
                    final_path.unlink()
                logger.error(
                    f"Unexpected structural ingest anomaly detected: {general_err}")
                raise

    def fetch_xray_crystal_density_map(self):
        raise NotImplementedError()


def parse_arguments(args: list | None = None) -> argparse.Namespace:
    """
    Parse the command line arguments provided to fetch_data.py.

    Args:
        args (list[str] | None): A custom array of argument strings to evaluate.
            Defaults to None, which forces fallback evaluation of sys.argv.
    Returns:
        argparse.Namespace: Namespace object containing:
            Required:
            - name (str): name of each target protein
            - uniprot_id (str): UniProt API ID for the each target protein
            - pdb_id (str): PDB ID(s) for the target protein(s)
            - outdir (Path): Output directory to store data
            Optional:
            - emdb_id (str): Optional ID of the target structure for wwPDB
            - download-all, coords-only, or maps-only (bool): Mutually exclusive mode for download
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
        "-e",
        "--emdb_id",
        type=str,
        nargs="+",
        required=False,
        help="EMDB IDs for cryo-EM density maps. If multiple, list items separated by space.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        required=True,
        help="Output directory to store data"
    )

    # Mode-Based Mutually Exclusive Group
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--coords-only",
        action="store_true",
        help="Fetch FASTA sequences and PDB structures only. Skip heavy voxel maps."
    )
    mode_group.add_argument(
        "--maps-only",
        action="store_true",

        help="Fetch EMDB density maps only. Skip sequences and coordinate files."
    )
    return parser.parse_args(args)


def main(args_list: Sequence[str] | None = None) -> None:
    """
    main function will accomplish 2 things:
        1. Fetch amino acid sequnce of each target protein
        2. Download structural information file(s)

    Args:
        args (argparse.Namespace):
            Namespace object containing:
                Required:
                - name (str): name of each target protein
                - uniprot_id (str): UniProt API ID for the each target protein
                - pdb_id (str): PDB ID(s) for the target protein(s)
                - outdir (Path): Output directory to store data
                Optional:
                - emdb_id (str): Optional ID of the target structure for wwPDB
                - coords-only, or maps-only (bool): Mutually exclusive mode for download, default is to download all.
    """
    args = parse_arguments(args_list)

    # 1. Determine active operational mode based on our flags
    is_maps_only = args.maps_only
    is_coords_only = args.coords_only

    # Default to downloading everything if no specific mode flag was explicitly passed
    is_download_all = not (is_maps_only or is_coords_only)

    # 2. Defensive check for EMDB IDs if a map-related mode is running
    if (is_download_all or is_maps_only) and not args.emdb_id:
        logger.error(
            "Execution configuration error: An EMDB ID (-m/--emdb_id) must be provided to download maps.")
        raise ValueError(
            "Missing required EMDB identifier for active execution mode.")

    # 3. Instantiate our class tool
    ingestor = ProteinDataIngestor(output_dir=args.outdir)

    if len(args.name) != len(args.uniprot_id):
        logger.error(
            "Mismatched parallel input arguments: names do not map 1:1 with UniProt IDs.")
        raise ValueError("Mismatched parallel input arguments.")

    targets = dict(zip(args.name, args.uniprot_id))

    try:
        # --- PHASE 1: SEQUENCES & COORDINATES ---
        if is_download_all or is_coords_only:
            logger.info(
                "--- PHASE 1: FETCHING SEQUENCES AND STRUCTURE COORDINATES ---")

            for name, uniprot_id in targets.items():
                # This returns string content (and will internally cache correctly)
                _ = ingestor.fetch_fasta(uniprot_id, name)

            for pdb_id in args.pdb_id:
                ingestor.fetch_cif(pdb_id)

        # --- PHASE 2: EM DENSITY MAPS ---
        if is_download_all or is_maps_only:
            logger.info(
                "--- PHASE 2: DOWNLOADING BINARY ELECTRON DENSITY MAPS ---")

            for emdb_id in args.emdb_id:
                ingestor.fetch_em_density_map(emdb_id)

    except Exception as e:
        logger.exception(f"Pipeline failed during execution block: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
