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

MOCK_CIF_CONTENT = (
    "data_mock_structure\n"
    "loop_\n"
    "_atom_site.group_PDB\n"
    "_atom_site.auth_asym_id\n"
    "_atom_site.label_atom_id\n"
    "_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n"
    "_atom_site.Cartn_z\n"
    "ATOM  A  CA  12.345  23.456  34.567\n"
    "ATOM  A  CA  13.100  24.200  35.300\n"
)

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
    "name": "mock_5res_complex",
}

MOCK_PROTEIN_SINGLE_RES = {
    "aa_list": ["ALA"],
    "coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    "name": "mock_single_res",
}

MOCK_PROTEIN_CORRUPTED_NAN = {
    "aa_list": ["ALA", "GLY"],
    "coords": np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=np.float32),
    "name": "mock_corrupted_nan",
}

MOCK_PROTEIN_CORRUPTED_INF = {
    "aa_list": ["ALA", "GLY"],
    "coords": np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]], dtype=np.float32),
    "name": "mock_corrupted_inf",
}
