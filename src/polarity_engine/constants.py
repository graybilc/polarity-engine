#!/usr/bin/env python3

from typing import Dict, Final

AMINO_ACID_TO_INDEX: Final[Dict[str, int]] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,  # Unknown / Non-standard
}

# Maximum Accessible Surface Area (maxASA) in square Angstroms (A^2)
# Reference: Tien et al. (2013), PLoS ONE 8(11): e80635. (Theoretical scale)
MAX_SASA_TIEN: Final[Dict[str, float]] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 160.0,
}

# Empirical average across standard 20 residues to handle unknown/non-standard AA classes
DEFAULT_MAX_ASA: Final[float] = 197.0

# Empirical atomic VdW radii (Å) for FreeSASA Option A calculation
ELEMENT_RADII: Final[Dict[str, float]] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "FE": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 1.94,
}
DEFAULT_VDW_RADIUS: Final[float] = 1.70
