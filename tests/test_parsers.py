#!/usr/bin/env python3

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from polarity_engine.parsers import FastaParser, StructureParser
from tests.mock_data import (
    MOCK_APKC_FASTA_CONTENT,
    MOCK_CIF_CONTENT,
    MOCK_LGL_FASTA_CONTENT,
    MOCK_PDB_CONTENT,
)


@pytest.fixture
def mock_cif_file(tmp_path: Path) -> Path:
    """Generates an isolated, multi-chain mock mmCIF structure file on disk."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    cif_path = structure_dir / "mock_structure.cif"
    cif_path.write_text(MOCK_CIF_CONTENT, encoding="utf-8")
    return cif_path


@pytest.fixture
def mock_pdb_file(tmp_path: Path) -> Path:
    """Generates an isolated mock PDB structure file on disk."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = structure_dir / "mock_structure.pdb"
    pdb_path.write_text(MOCK_PDB_CONTENT, encoding="utf-8")
    return pdb_path


# ==============================================================================
# BLACK-BOX TESTS: FastaParser Public API
# ==============================================================================


class TestFastaParserPublicAPI:
    """Black-box unit tests validating public FastaParser capabilities."""

    def test_parse_fasta_single_entry_success(self, tmp_path):
        """
        Arrange:
            Create a mock FASTA file containing a single sequence entry on disk.
        Act:
            Invoke public FastaParser.parse_fasta entry point.
        Assert:
            Verify returned dictionary maps sequence header to accurate amino acid string.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        fasta_file_path = aa_sequence_dir / "lgl_sequence.fasta"
        fasta_file_path.write_text(MOCK_LGL_FASTA_CONTENT, encoding="utf-8")

        expected_output = {
            "tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor": "MGNCCAGLSRRLKLPDCMA"
        }

        test_sequences = FastaParser.parse_fasta(fasta_file_path)
        assert test_sequences == expected_output

    def test_parse_fasta_multiple_entries_success(self, tmp_path):
        """
        Arrange:
            Create a multi-entry FASTA file containing two distinct headers and sequences.
        Act:
            Invoke public FastaParser.parse_fasta entry point.
        Assert:
            Verify both entries are extracted accurately into dictionary mapping.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        fasta_file_path = aa_sequence_dir / "mixed_sequence.fasta"
        fasta_contents = MOCK_LGL_FASTA_CONTENT + MOCK_APKC_FASTA_CONTENT
        fasta_file_path.write_text(fasta_contents, encoding="utf-8")

        expected_output = {
            "tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor": "MGNCCAGLSRRLKLPDCMA",
            "sp|P41743|KPCI_HUMAN Protein kinase C iota type OS=Homo sapiens OX=9606 GN=PRKCI PE=1 SV=2": "MPTQRDSSTMSHTVAGGGSGDHS",
        }

        test_sequences = FastaParser.parse_fasta(fasta_file_path)
        assert test_sequences == expected_output

    def test_parse_fasta_non_standard_amino_acids_emits_warning(self, tmp_path, caplog):
        """
        Arrange:
            Create a FASTA file containing non-standard amino acid codes (U, Z).
        Act:
            Invoke public FastaParser.parse_fasta entry point.
        Assert:
            Verify parsing succeeds while generating warning log entries for each non-standard code.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        fasta_file_path = aa_sequence_dir / "non_standard.fasta"
        fasta_content = ">header_1\nMUGNCCAGLSRRLKLPZDCMA\n"
        fasta_file_path.write_text(fasta_content, encoding="utf-8")

        parsed_data = FastaParser.parse_fasta(fasta_file_path)

        warning_records = [rec for rec in caplog.records if rec.levelname == "WARNING"]
        assert parsed_data == {"header_1": "MUGNCCAGLSRRLKLPZDCMA"}
        assert len(warning_records) == 2

    def test_parse_fasta_missing_file_raises_not_found_error(self, tmp_path):
        """
        Arrange:
            Construct a path targeting a missing FASTA file.
        Act:
            Invoke FastaParser.parse_fasta on the missing path.
        Assert:
            Verify FileNotFoundError is raised with target error message.
        """
        mock_fasta_path = tmp_path / "non_existent_file.fasta"
        with pytest.raises(
            FileNotFoundError, match="Target FASTA file not found or invalid"
        ):
            FastaParser.parse_fasta(mock_fasta_path)

    def test_parse_fasta_empty_file_raises_value_error(self, tmp_path):
        """
        Arrange:
            Create an empty file on disk.
        Act:
            Invoke FastaParser.parse_fasta on empty file.
        Assert:
            Verify ValueError is raised reporting zero entries.
        """
        mock_fasta_path = tmp_path / "empty.fasta"
        mock_fasta_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="contains no entries"):
            FastaParser.parse_fasta(mock_fasta_path)

    def test_parse_fasta_no_header_raises_value_error(self, tmp_path):
        """
        Arrange:
            Create a FASTA file containing raw sequence before any header.
        Act:
            Invoke FastaParser.parse_fasta on headerless file.
        Assert:
            Verify ValueError is raised indicating sequence preceded header.
        """
        mock_fasta_path = tmp_path / "no_header.fasta"
        mock_fasta_path.write_text("MGNCCAGLSRRLKLPDCMA\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Found sequence data before a header"):
            FastaParser.parse_fasta(mock_fasta_path)

    def test_parse_fasta_invalid_characters_raises_value_error(self, tmp_path):
        """
        Arrange:
            Create a FASTA file containing illegal character ('!').
        Act:
            Invoke FastaParser.parse_fasta on corrupt sequence file.
        Assert:
            Verify ValueError is raised specifying character and 1-based index.
        """
        mock_fasta_path = tmp_path / "invalid_char.fasta"
        mock_fasta_path.write_text(">header\nM!GNCCAGLSRRL\n", encoding="utf-8")

        with pytest.raises(
            ValueError, match="Invalid amino acid '!' found at position 2"
        ):
            FastaParser.parse_fasta(mock_fasta_path)


# ==============================================================================
# BLACK-BOX TESTS: StructureParser Public API
# ==============================================================================


class TestStructureParserPublicAPI:
    """Black-box unit tests validating public StructureParser contracts."""

    def test_parse_cif_returns_complete_schema(self, mock_cif_file):
        """
        Arrange:
            Prepare multi-chain mmCIF file using fixture.
        Act:
            Invoke public entry point StructureParser.parse.
        Assert:
            Verify compliance with 7-key schema contract, array dimensions, and float32 dtypes.
        """
        parsed_output = StructureParser.parse(mock_cif_file)

        # Schema contract verification
        assert "aa_list" in parsed_output
        assert "coords" in parsed_output
        assert "b_factors" in parsed_output
        assert "occupancies" in parsed_output
        assert "nodes" in parsed_output
        assert "sasa_map" in parsed_output
        assert "rsasa" in parsed_output

        # Shape verification
        n_nodes = len(parsed_output["aa_list"])
        assert parsed_output["coords"].shape == (n_nodes, 3)
        assert parsed_output["b_factors"].shape == (n_nodes,)
        assert parsed_output["occupancies"].shape == (n_nodes,)
        assert len(parsed_output["nodes"]) == n_nodes
        assert parsed_output["rsasa"].shape == (n_nodes,)

        # Type safety
        assert parsed_output["coords"].dtype == np.float32
        assert parsed_output["b_factors"].dtype == np.float32
        assert parsed_output["occupancies"].dtype == np.float32
        assert parsed_output["rsasa"].dtype == np.float32

    def test_parse_pdb_returns_complete_schema(self, mock_pdb_file):
        """
        Arrange:
            Prepare legacy PDB file using fixture.
        Act:
            Invoke public entry point StructureParser.parse.
        Assert:
            Verify extracted PDB output schema and coordinate dimensions match expectations.
        """
        parsed_output = StructureParser.parse(mock_pdb_file)

        n_nodes = len(parsed_output["aa_list"])
        assert n_nodes > 0
        assert parsed_output["coords"].shape == (n_nodes, 3)
        assert parsed_output["b_factors"].shape == (n_nodes,)
        assert parsed_output["occupancies"].shape == (n_nodes,)

    def test_parse_chain_filtering_restricts_output(self, mock_cif_file):
        """
        Arrange:
            Prepare multi-chain mmCIF file and target chain 'A'.
        Act:
            Invoke StructureParser.parse specifying chain_ids=['A'].
        Assert:
            Verify node identities belong exclusively to target chain 'A'.
        """
        target_chain = "A"
        parsed_filtered = StructureParser.parse(mock_cif_file, chain_ids=[target_chain])

        extracted_chains = {node[0] for node in parsed_filtered["nodes"]}
        assert extracted_chains == {target_chain}

    def test_get_all_atom_coordinates_public_interface(self, mock_cif_file):
        """
        Arrange:
            Prepare multi-chain mmCIF structure file.
        Act:
            Invoke public StructureParser.get_all_atom_coordinates.
        Assert:
            Verify payload returns per-chain CA and all-atom coordinate dictionary structures.
        """
        result = StructureParser.get_all_atom_coordinates(
            mock_cif_file, chain_ids=["A", "B"]
        )

        assert "A" in result
        assert "B" in result
        assert "ca_coords" in result["A"]
        assert "all_atom_coords" in result["A"]
        assert result["A"]["ca_coords"].dtype == np.float32

    def test_get_all_atom_coordinates_invalid_chain_raises_value_error(
        self, mock_cif_file
    ):
        """
        Arrange:
            Specify chain filter requesting a non-existent chain ID ('NON_EXISTENT').
        Act:
            Invoke StructureParser.get_all_atom_coordinates targeting missing chain.
        Assert:
            Verify ValueError is raised indicating missing target chain.
        """
        with pytest.raises(ValueError, match="None of requested chain_ids"):
            StructureParser.get_all_atom_coordinates(
                mock_cif_file, chain_ids=["NON_EXISTENT"]
            )

    def test_parse_missing_file_raises_file_not_found_error(self, tmp_path):
        """
        Arrange:
            Construct path pointing to non-existent structure file.
        Act:
            Invoke StructureParser.parse.
        Assert:
            Verify FileNotFoundError is raised reporting missing file.
        """
        non_existent_file = tmp_path / "missing_structure.cif"
        with pytest.raises(FileNotFoundError, match="Target structure file not found"):
            StructureParser.parse(non_existent_file)

    def test_parse_unsupported_file_extension_raises_value_error(self, tmp_path):
        """
        Arrange:
            Write file with unsupported extension (`.txt`) to disk.
        Act:
            Attempt parsing via StructureParser.parse.
        Assert:
            Verify ValueError is raised indicating unsupported format.
        """
        invalid_ext_file = tmp_path / "structure.txt"
        invalid_ext_file.write_text("SOME_TEXT_CONTENT", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file format"):
            StructureParser.parse(invalid_ext_file)


# ==============================================================================
# WHITE-BOX TESTS: Isolated Internal Core Logic
# ==============================================================================


class TestStructureParserInternalLogic:
    """Targeted white-box unit tests validating high-risk internal logic."""

    def test_parse_mmcif_fast_path_chain_isolation(self):
        """
        Arrange:
            Construct raw in-memory mmCIF dictionary containing intermingled chains ('A', 'B').
        Act:
            Invoke private helper StructureParser._parse_mmcif_fast_path targeting chain 'A'.
        Assert:
            Verify fast-path parser isolates chain 'A' arrays and formats output data types.
        """
        mock_mmcif_dict = {
            "_atom_site.label_atom_id": ["CA", "N", "CA", "CA"],
            "_atom_site.auth_asym_id": ["A", "A", "A", "B"],
            "_atom_site.group_PDB": ["ATOM", "ATOM", "ATOM", "ATOM"],
            "_atom_site.Cartn_x": ["10.0", "11.0", "12.0", "20.0"],
            "_atom_site.Cartn_y": ["20.0", "21.0", "22.0", "30.0"],
            "_atom_site.Cartn_z": ["30.0", "31.0", "32.0", "40.0"],
            "_atom_site.auth_comp_id": ["SER", "PRO", "ALA", "HIS"],
            "_atom_site.auth_seq_id": ["1", "1", "2", "1"],
            "_atom_site.B_iso_or_equiv": ["15.5", "18.2", "16.0", "22.1"],
            "_atom_site.occupancy": ["1.0", "1.0", "0.85", "1.0"],
        }

        result = StructureParser._parse_mmcif_fast_path(mock_mmcif_dict, "A")

        assert result["ca_coords"].shape == (2, 3)
        assert result["ca_coords"].dtype == np.float32
        assert result["aa_residues"] == [("1", "SER"), ("2", "ALA")]
        assert result["all_atom_coords"].shape == (3, 3)

    def test_load_and_inspect_multi_model_pdb_warning(self, caplog):
        """
        Arrange:
            Mock a BioPython Structure containing 3 models and patch PDBParser.get_structure.
        Act:
            Invoke private helper StructureParser._load_and_inspect on multi-model mock.
        Assert:
            Verify chain extraction defaults to Model 0 and emits warning log regarding fallback.
        """
        file_path = Path("mock_multi_model.pdb")
        file_ext = ".pdb"

        mock_structure = MagicMock()
        mock_structure.__len__.return_value = 3

        mock_model_0 = MagicMock()
        mock_model_0.child_dict.keys.return_value = ["A", "B"]
        mock_structure.__getitem__.return_value = mock_model_0

        with patch(
            "polarity_engine.parsers.PDBParser.get_structure",
            return_value=mock_structure,
        ):
            with caplog.at_level(logging.WARNING):
                chains, struct = StructureParser._load_and_inspect(file_path, file_ext)

        assert chains == ["A", "B"]
        assert struct == mock_structure
        assert len(caplog.records) == 1
        assert "contains 3 models. Defaulting to Model 0." in caplog.text
