#!/usr/bin/env python3
"""Sequence alignment and fallback anchor resolution for target sites (ADR-0004)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from Bio import Align
from Bio.PDB import MMCIFParser

THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for CIF sequence alignment.

    Args:
        args (List[str] | None, optional): Explicit list of command-line argument
            strings to parse. If None, parses from `sys.argv[1:]`. Defaults to None.

    Returns:
        argparse.Namespace: Parsed command-line argument namespace containing:
            - cif (str): Path to input CIF structure file.
            - sites (str): Comma-separated target site positions (e.g., "655,659,663").
            - chain (str): Target chain identifier in the CIF structure.
            - out (str): Path where output site mapping JSON will be written.
    """
    parser = argparse.ArgumentParser(description="Align CIF Target Sites")
    parser.add_argument("--cif", required=True, help="Path to CIF file")
    parser.add_argument("--sites", required=True, help="Comma-separated sites")
    parser.add_argument("--chain", required=True, help="Chain ID")
    parser.add_argument("--out", required=True, help="Output JSON path")
    return parser.parse_args(args)


def parse_site_list(sites_str: str) -> List[int]:
    """Parses comma-separated target site strings into an ordered list of residue numbers.

    Args:
        sites_str (str): Comma-separated string of target site numbers (e.g., "655, 659, 663").

    Returns:
        List[int]: Sorted list of unique target residue numbers.

    Raises:
        ValueError: If any site entry cannot be parsed as an integer.
    """
    try:
        return sorted({int(s.strip()) for s in sites_str.split(",") if s.strip()})
    except ValueError as e:
        raise ValueError(f"Failed to parse --sites argument '{sites_str}': {e}") from e


def get_all_chain_sequences(cif_path: str) -> Dict[str, Tuple[str, List[int]]]:
    """Parses MMCIF file and extracts sequences and residue indices per chain.

    Args:
        cif_path (str): Path to MMCIF structure file.

    Returns:
        Dict[str, Tuple[str, List[int]]]: Mapping of chain IDs to a tuple containing
            the single-letter amino acid sequence string and the corresponding
            modeled residue numbers list.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", cif_path)

    chains_data = {}

    for model in structure:
        for chain in model:
            seq_chars = []
            res_nums = []
            for residue in chain:
                if residue.get_id()[0] == " ":  # Exclude heteroatoms/water
                    res_name = residue.get_resname()
                    if res_name in THREE_TO_ONE:
                        seq_chars.append(THREE_TO_ONE[res_name])
                        res_nums.append(residue.get_id()[1])
            if seq_chars:
                chains_data[chain.id] = ("".join(seq_chars), res_nums)
        break

    return chains_data


def find_best_matching_chain(
    ref_seq: str, chains_data: Dict[str, Tuple[str, List[int]]]
) -> str | None:
    """Finds the chain ID in target structure that best matches the reference sequence.

    Args:
        ref_seq (str): Reference amino acid sequence string.
        chains_data (Dict[str, Tuple[str, List[int]]]): Parsed target chains data.

    Returns:
        str | None: Chain ID with highest pairwise alignment score, or None if empty.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"

    best_chain_id = None
    best_score = -float("inf")

    for chain_id, (seq, _) in chains_data.items():
        if not seq:
            continue
        score = aligner.score(ref_seq, seq)
        if score > best_score:
            best_score = score
            best_chain_id = chain_id

    return best_chain_id


def map_target_residues(
    ref_cif: str,
    target_cif: str,
    target_sites: List[int],
    ref_chain_id: str | None = None,
) -> Dict[str, Any]:
    """Aligns target structure to reference structure and checks for site presence.

    Determines whether each target site is MODELED in the target structure or UNMODELED.
    Unmodeled sites fall back to the resolved C-terminal boundary residue (ADR-0004 anchor).

    Args:
        ref_cif (str): Path to reference MMCIF structure file.
        target_cif (str): Path to target MMCIF structure file.
        target_sites (List[int]): List of target residue site numbers to align and evaluate.
        ref_chain_id (str | None, optional): Chain ID to use from reference structure.
            If None or invalid, defaults to the first available chain. Defaults to None.

    Returns:
        Dict[str, Any]: Nested dictionary containing metadata and site alignment mappings:
            {
                "metadata": {
                    "ref_chain": "I",
                    "ref_range": [248, 585],
                    "target_chain": "L",
                    "target_range": [25, 588]
                },
                "sites": {
                    "655": {
                        "status": "UNMODELED",
                        "note": "Site 655 outside modeled coordinate range (25-588)",
                        "fallback_anchor_res_num": 588,
                        "anchor_aa": "L"
                    },
                    "250": {
                        "status": "MODELED",
                        "target_res_num": 250,
                        "ref_aa": "S",
                        "target_aa": "S",
                        "match": True
                    }
                }
            }

    Raises:
        ValueError: If no non-empty chains are found in `target_cif`.
    """
    ref_chains = get_all_chain_sequences(ref_cif)

    if ref_chain_id and ref_chain_id in ref_chains:
        ref_seq, ref_nums = ref_chains[ref_chain_id]
    else:
        ref_chain_id = list(ref_chains.keys())[0]
        ref_seq, ref_nums = ref_chains[ref_chain_id]

    tgt_chains = get_all_chain_sequences(target_cif)
    tgt_chain_id = find_best_matching_chain(ref_seq, tgt_chains)

    if not tgt_chain_id:
        raise ValueError(f"No non-empty chains found in {target_cif}")

    tgt_seq, tgt_nums = tgt_chains[tgt_chain_id]

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(ref_seq, tgt_seq)
    best_alignment = alignments[0]
    aligned_ref, aligned_tgt = best_alignment

    mapping = {
        "metadata": {
            "ref_chain": ref_chain_id,
            "ref_range": [ref_nums[0], ref_nums[-1]],
            "target_chain": tgt_chain_id,
            "target_range": [tgt_nums[0], tgt_nums[-1]],
        },
        "sites": {},
    }

    # Target site bounds check with ADR-0004 Fallback Anchor logic
    for site in target_sites:
        if site < tgt_nums[0] or site > tgt_nums[-1]:
            # Anchor to C-terminal boundary (e.g., 588)
            anchor_res_num = tgt_nums[-1]
            mapping["sites"][str(site)] = {
                "status": "UNMODELED",
                "note": f"Site {site} outside modeled coordinate range ({tgt_nums[0]}-{tgt_nums[-1]})",
                "fallback_anchor_res_num": anchor_res_num,
                "anchor_aa": tgt_seq[-1],
            }

    ref_idx, tgt_idx = 0, 0
    for r_char, t_char in zip(aligned_ref, aligned_tgt):
        if r_char != "-" and t_char != "-":
            ref_res_num = ref_nums[ref_idx]
            tgt_res_num = tgt_nums[tgt_idx]
            if ref_res_num in target_sites:
                mapping["sites"][str(ref_res_num)] = {
                    "status": "MODELED",
                    "target_res_num": tgt_res_num,
                    "ref_aa": r_char,
                    "target_aa": t_char,
                    "match": r_char == t_char,
                }
            ref_idx += 1
            tgt_idx += 1
        elif r_char != "-":
            ref_idx += 1
        elif t_char != "-":
            tgt_idx += 1

    return mapping


def main(cli_args: List[str] | None = None) -> None:
    """Main execution entry point for sequence alignment script.

    Parses command-line arguments, executes CIF target sequence alignment,
    and writes JSON results directly to the specified `--out` path.

    Args:
        cli_args (List[str] | None, optional): Command-line argument vector override for
            testing. Defaults to None (reads `sys.argv[1:]`).

    Returns:
        None
    """
    args = parse_args(cli_args)

    target_cif_path = Path(args.cif)
    if not target_cif_path.is_file():
        raise FileNotFoundError(f"Target CIF file not found: {target_cif_path}")

    target_sites = parse_site_list(args.sites)

    repo_root = Path(__file__).resolve().parent.parent
    ref_path = repo_root / "data" / "8r3y.cif"

    if not ref_path.exists():
        # Fallback to current target structure if reference MMCIF is absent
        ref_path = target_cif_path

    results = map_target_residues(
        ref_cif=str(ref_path),
        target_cif=str(target_cif_path),
        target_sites=target_sites,
        ref_chain_id=args.chain,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
