#!/usr/bin/env python3
"""Sequence alignment and fallback anchor resolution for target sites (ADR-0005)."""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Align CIF sequence target sites.")
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--sites", required=True, type=str)
    parser.add_argument("--chain", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(args)


def parse_site_list(sites_str: str) -> List[int]:
    try:
        return sorted({int(s.strip()) for s in sites_str.split(",") if s.strip()})
    except ValueError as e:
        raise ValueError(f"Failed to parse --sites argument '{sites_str}': {e}") from e


def get_all_chain_sequences(cif_path: str) -> Dict[str, Tuple[str, List[int]]]:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", cif_path)

    chains_data = {}
    for model in structure:
        for chain in model:
            seq_chars = []
            res_nums = []
            for residue in chain:
                if residue.get_id()[0] == " ":
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

    # ✅ Convert Bio.Align.Alignment object to FASTA format strings
    alignment_str = format(best_alignment, "fasta")
    fasta_lines = [
        line.strip()
        for line in alignment_str.splitlines()
        if line and not line.startswith(">")
    ]
    aligned_ref_str = fasta_lines[0]
    aligned_tgt_str = fasta_lines[1]

    mapping = {
        "metadata": {
            "ref_chain": ref_chain_id,
            "ref_range": [ref_nums[0], ref_nums[-1]],
            "target_chain": tgt_chain_id,
            "target_range": [tgt_nums[0], tgt_nums[-1]],
        },
        "sites": {},
    }

    for site in target_sites:
        if site < tgt_nums[0] or site > tgt_nums[-1]:
            anchor_res_num = tgt_nums[-1]
            mapping["sites"][str(site)] = {
                "status": "UNMODELED",
                "note": f"Site {site} outside modeled coordinate range ({tgt_nums[0]}-{tgt_nums[-1]})",
                "fallback_anchor_res_num": anchor_res_num,
                "anchor_aa": tgt_seq[-1],
            }

    ref_idx, tgt_idx = 0, 0
    for r_char, t_char in zip(aligned_ref_str, aligned_tgt_str):
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
    args = parse_args(cli_args)

    target_cif_path = Path(args.target)
    ref_cif_path = Path(args.reference)

    if not target_cif_path.is_file():
        sys.stderr.write(f"ERROR: File not found: {target_cif_path}\n")
        sys.exit(1)
    if not ref_cif_path.is_file():
        sys.stderr.write(f"ERROR: File not found: {ref_cif_path}\n")
        sys.exit(1)

    target_sites = parse_site_list(args.sites)

    try:
        mapping_results = map_target_residues(
            ref_cif=str(ref_cif_path),
            target_cif=str(target_cif_path),
            target_sites=target_sites,
            ref_chain_id=args.chain,
        )
    except Exception as err:
        sys.stderr.write(f"Alignment warning fallback triggered: {err}\n")
        # ✅ Stringify Exception object for JSON serialization
        err_msg = str(err)
        mapping_results = {
            "metadata": {
                "ref_chain": args.chain,
                "target_chain": args.chain,
                "note": f"Fallback due to: {err_msg}",
            },
            "sites": {
                str(s): {"status": "UNMODELED", "note": err_msg} for s in target_sites
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(mapping_results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


if __name__ == "__main__":
    main()
