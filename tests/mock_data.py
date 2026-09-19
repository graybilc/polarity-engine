"""Centralized static mock data for unit tests."""

import numpy as np

UNIPROT_ID = "P51617"

TARGET_NAME = "lgl"

MOCK_LGL_FASTA_CONTENT = (
    ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor" "\nMGNCCAGLSRRL\nKLPDCMA\n"
)

MOCK_APKC_FASTA_CONTENT = (
    ">sp|P41743|KPCI_HUMAN Protein kinase C iota type "
    "OS=Homo sapiens OX=9606 GN=PRKCI PE=1 SV=2"
    "\nMPTQRDSSTMSHTVA\ngggSGDHS\n"
)

# Minimal valid PDB coordinate string for testing (Chain A, 2 residues)
MOCK_PDB_CONTENT_1 = (
    "ATOM      1  N   MET A   1      24.084  14.017   4.721  1.00 20.00           N\n"
    "ATOM      2  CA  MET A   1      24.520  15.421   4.900  1.00 20.00           C\n"
    "ATOM      3  C   MET A   1      26.012  15.510   5.120  1.00 20.00           C\n"
    "ATOM      4  O   MET A   1      26.741  14.542   4.990  1.00 20.00           O\n"
    "ATOM      5  N   GLY A   2      26.471  16.691   5.441  1.00 21.00           N\n"
    "ATOM      6  CA  GLY A   2      27.892  16.920   5.690  1.00 21.00           C\n"
    "END\n"
)

MOCK_CIF_CONTENT = """data_mock_structure
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 C CA . VAL A A 1 1 ? 12.345 23.456 34.567 1.00 15.50
ATOM 2 C CA . MET A A 2 2 ? 13.100 24.200 35.300 1.00 15.50
ATOM 3 C CA . ALA B B 1 1 ? 20.000 30.000 40.000 1.00 22.10
ATOM 4 C CA . ARG B B 2 2 ? 21.000 31.000 41.000 1.00 22.10
#
"""


MOCK_PROTEIN_5RES = {
    "aa_list": ["ALA", "GLY", "SER", "TRP", "UNK"],
    "coords": np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],  # 3.0 Å from node 0
            [6.0, 0.0, 0.0],  # 3.0 Å from node 1, 6.0 Å from node 0
            [9.0, 0.0, 0.0],  # > 8.0 Å from node 0
            [20.0, 0.0, 0.0],  # Isolated node (> 8.0 Å from all nodes)
        ],
        dtype=np.float32,
    ),
    "b_factors": np.array([12.5, 18.0, 22.4, 35.1, 40.0], dtype=np.float32),
    "occupancies": np.array([1.0, 1.0, 0.85, 1.0, 0.50], dtype=np.float32),
    "nodes": [
        ("A", "1", "ALA"),
        ("A", "2", "GLY"),
        ("A", "3", "SER"),
        ("A", "4", "TRP"),
        ("A", "5", "UNK"),
    ],
    "sasa_map": {
        ("A", "1"): 64.5,  # ALA: rSASA = 64.5 / 129.0 = 0.5
        ("A", "2"): 0.0,  # GLY: Core buried
        ("A", "3"): 77.5,  # SER: rSASA = 77.5 / 155.0 = 0.5
        ("A", "4"): 285.0,  # TRP: Fully exposed
        ("A", "5"): 98.5,  # UNK: Triggers DEFAULT_MAX_ASA fallback (197.0) -> 0.5
    },
    "name": "mock_5res_complex",
}

MOCK_PROTEIN_SINGLE_RES = {
    "aa_list": ["ALA"],
    "coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    "b_factors": np.array([15.0], dtype=np.float32),
    "occupancies": np.array([1.0], dtype=np.float32),
    "nodes": [("A", "1", "ALA")],
    "sasa_map": {("A", "1"): 64.5},
    "name": "mock_single_res",
}

MOCK_PROTEIN_CORRUPTED_NAN = {
    "aa_list": ["ALA", "GLY"],
    "coords": np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=np.float32),
    "b_factors": np.array([12.5, 18.0], dtype=np.float32),
    "occupancies": np.array([1.0, 1.0], dtype=np.float32),
    "nodes": [("A", "1", "ALA"), ("A", "2", "GLY")],
    "sasa_map": {("A", "1"): 64.5, ("A", "2"): 0.0},
    "name": "mock_corrupted_nan",
}

MOCK_PROTEIN_CORRUPTED_INF = {
    "aa_list": ["ALA", "GLY"],
    "coords": np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]], dtype=np.float32),
    "b_factors": np.array([12.5, 18.0], dtype=np.float32),
    "occupancies": np.array([1.0, 1.0], dtype=np.float32),
    "nodes": [("A", "1", "ALA"), ("A", "2", "GLY")],
    "sasa_map": {("A", "1"): 64.5, ("A", "2"): 0.0},
    "name": "mock_corrupted_inf",
}

# Minimal mock nodes for a synthetic 3-residue complex: (chain_id, res_num_str, res_name_3let)
MOCK_NODES = [
    ("A", "1", "ALA"),  # Standard residue (MaxASA = 129.0 Å²)
    ("A", "2", "TRP"),  # Core buried residue (0.0 raw SASA)
    ("A", "3", "XYZ"),  # Non-standard residue (Triggers DEFAULT_MAX_ASA fallback)
    ("B", "10", "ILE"),  # Exposed interface residue (Tests bound clamping to 1.0)
]

# Corresponding raw per-residue SASA map (in Å²)
MOCK_SASA_MAP = {
    ("A", "1"): 64.5,  # Expected rSASA = 64.5 / 129.0 = 0.50
    ("A", "2"): 0.0,  # Expected rSASA = 0.00
    ("A", "3"): 98.5,  # Expected rSASA = 98.5 / 197.0 = 0.50 (fallback)
    ("B", "10"): 300.0,  # Raw > 197.0 -> Expected rSASA clamped to 1.00
}

# Matching amino acid list and coordinates for PyG Data assembly testing
MOCK_AA_LIST = ["ALA", "TRP", "XYZ", "ILE"]
MOCK_COORDS_NP = [
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [7.6, 0.0, 0.0],
    [11.4, 0.0, 0.0],
]
# Corresponding B-factors (crystallographic temperature factors / flexibility)
MOCK_B_FACTORS_NP = np.array([12.5, 18.0, 25.4, 30.1], dtype=np.float32)

# Corresponding occupancies (atomic presence confidence in structure)
MOCK_OCCUPANCIES_NP = np.array([1.0, 1.0, 0.85, 1.0], dtype=np.float32)
