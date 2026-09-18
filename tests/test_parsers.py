#!/urs/bin/env python3


from unittest.mock import MagicMock, patch
import logging
import numpy as np
import pytest

from Bio.PDB.Structure import Structure
from pathlib import Path
from unittest.mock import patch, MagicMock

from polarity_engine.parsers import FastaParser, StructureParser
from tests.mock_data import (
    MOCK_LGL_FASTA_CONTENT,
    MOCK_APKC_FASTA_CONTENT,
    MOCK_PDB_CONTENT_1,
    MOCK_CIF_CONTENT,
)


@pytest.fixture
def fasta_parser_cls():
    """
    Returns the FastaParser class handle for classmethod calls.
    """
    return FastaParser


@pytest.fixture
def structure_parser_cls():
    """
    Returns the StructureParser class handle for classmethod calls.
    """
    return StructureParser


@pytest.fixture
def mock_cif_file(tmp_path: Path) -> Path:
    """Generates an isolated, multi-chain mock mmCIF structure file on disk."""
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    cif_path = structure_dir / "mock_structure.cif"
    cif_path.write_text(MOCK_CIF_CONTENT, encoding="utf-8")
    return cif_path


class TestFastaParser:
    """
    Groups all unit tests validating the state and side-effects of FastaParser
    """

    def test_fasta_parser_one_header_success(self, fasta_parser_cls, tmp_path):
        """
        Verify that parse_fasta returns a dictionary mapping header strings to
        clean sequence strings under normal valid multi-line conditions.

        Arrange:
            Generate an isolated directory and mock FASTA with one header line on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            Verify the output dictionary exactly matches the expected sequence mappings.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        fasta_file_path = aa_sequence_dir / "lgl_sequence.fasta"
        fasta_file_path.write_text(MOCK_LGL_FASTA_CONTENT, encoding="utf-8")

        expected_output = {
            "tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor": "MGNCCAGLSRRLKLPDCMA"
        }

        test_sequences = fasta_parser_cls.parse_fasta(fasta_file_path)

        assert test_sequences == expected_output

    def test_fasta_parser_two_headers_success(self, fasta_parser_cls, tmp_path):
        """
        Verify that parse_fasta returns a dictionary mapping header strings to
        clean sequence strings under normal valid multi-line conditions.

        Arrange:
            Generate an isolated directory and mock FASTA with two header lines on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            Verify the output dictionary exactly matches the expected sequence mappings.
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

        test_sequences = fasta_parser_cls.parse_fasta(fasta_file_path)

        assert test_sequences == expected_output

    def test_fasta_parser_file_not_found_error(self, fasta_parser_cls, tmp_path):
        """
        Ensure that FileNotFoundError is raised when an invalid file_path is passed.

        Arrange:
            Generate an isolated directory without FASTA file on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            FileNotFoundError is raised.
        """
        mock_fasta_path = tmp_path / "amino_acid_sequences" / "non_existent_file.fasta"

        with pytest.raises(
            FileNotFoundError, match="Target FASTA file not found or invalid"
        ):
            fasta_parser_cls.parse_fasta(mock_fasta_path)

    def test_fasta_parser_no_file_contents_error(self, fasta_parser_cls, tmp_path):
        """
        Verify that ValueError is raised when an empty file is passed.

        Arrange:
            Generate an isolated directory with a file without data on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            ValueError is raised.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        mock_fasta_path = aa_sequence_dir / "no_data_file.fasta"
        mock_fasta_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="contains no entries"):
            fasta_parser_cls.parse_fasta(mock_fasta_path)

    def test_fasta_parser_no_header_error(self, fasta_parser_cls, tmp_path):
        """
        Ensure that ValueError is raised when header line is missing in the FASTA file.

        Arrange:
            Generate an isolated directory with a file without header line on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            ValueError is raised.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        mock_fasta_path = aa_sequence_dir / "no_header_file.fasta"
        mock_data = "MGNCCAGLSRRL\nKLPDCMA\n"
        mock_fasta_path.write_text(mock_data, encoding="utf-8")

        with pytest.raises(ValueError, match="Found sequence data before a header"):
            fasta_parser_cls.parse_fasta(mock_fasta_path)

    def test_fasta_parser_no_seq_data_error(self, fasta_parser_cls, tmp_path):
        """
        Ensure that ValueError is raised when sequence data is missing in the FASTA file.

        Arrange:
            Generate an isolated directory with a file without sequence lines on disk.
        Act:
            Invoke the parse_fasta method.
        Assert:
            ValueError is raised.
        """
        aa_sequence_dir = tmp_path / "amino_acid_sequences"
        aa_sequence_dir.mkdir(parents=True, exist_ok=True)

        mock_fasta_path = aa_sequence_dir / "no_sequence_file.fasta"
        mock_data = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\n"
        mock_fasta_path.write_text(mock_data, encoding="utf-8")

        with pytest.raises(
            ValueError, match="Empty sequence string encountered under header"
        ):
            fasta_parser_cls.parse_fasta(mock_fasta_path)

    def test_validate_amino_acid_sequence_all_standard_success(self, fasta_parser_cls):
        """
        Verify that validate_amino_acid_sequence returns True when all amino acids
        in FASTA file are standard and valid.

        Arrange:
            Generate an isolated sequence string with standard amino acids.
        Act:
            Invoke the _validate_amino_acid_sequence method.
        Assert:
            Amino acid sequence is returned.
        """
        test_header = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\n"
        test_sequence = "MGNCCAGLSRRLKLPDCMA"

        test_output = fasta_parser_cls._validate_amino_acid_sequence(
            test_sequence, test_header
        )

        assert test_output == test_output

    def test_validate_amino_acid_sequence_non_standard_success(
        self, fasta_parser_cls, caplog
    ):
        """
        Verify that validate_amino_acid_sequence returns True when amino acid string
        contains two non-standard amino acids but still valid.

        Arrange:
            Generate an isolated sequence string containing two non-standard amino acids.
        Act:
            Invoke the _validate_amino_acid_sequence method.
        Assert:
            Verify the amino acid sequence is returned and exactly two warnings are emitted to the log.
        """
        test_header = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\n"
        # 'U' at position 2 and 'Z' at position 17 are non-standard but valid
        test_sequence = "MUGNCCAGLSRRLKLPZDCMA"

        test_output = fasta_parser_cls._validate_amino_acid_sequence(
            test_sequence, test_header
        )

        # Filter captured log records to isolate warnings from this test execution
        warning_records = [rec for rec in caplog.records if rec.levelname == "WARNING"]

        assert test_output == test_sequence
        assert len(warning_records) == 2
        assert (
            "Non-standard amino acid 'U' found at position 2"
            in warning_records[0].message
        )
        assert (
            "Non-standard amino acid 'Z' found at position 17"
            in warning_records[1].message
        )

    def test_validate_amino_acid_sequence_empty_string_error(
        self, fasta_parser_cls, caplog
    ):
        """
        Verify that validate_amino_acid_sequence returns False when an empty string is passed.

        Arrange:
            Generate an empty string.
        Act:
            Invoke the _validate_amino_acid_sequence method.
        Assert:
            Verify no amino acid is returned and exactly one error message is emitted to the log.
        """
        test_header = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\n"
        test_sequence = ""

        with pytest.raises(
            ValueError, match="Empty sequence string encountered under header"
        ):
            fasta_parser_cls._validate_amino_acid_sequence(test_sequence, test_header)

        error_records = [rec for rec in caplog.records if rec.levelname == "ERROR"]

        assert len(error_records) == 1

    def test_validate_amino_acid_sequence_invalid_sequence_error(
        self, fasta_parser_cls, caplog
    ):
        """
        Verify that validate_amino_acid_sequence returns False when amino acid string
        containining two invalid amino acids is passed.

        Arrange:
            Generate an isolated sequence string containing two non-standard amino acids.
        Act:
            Invoke the validate_amino_acid_sequence method.
        Assert:
            Verify True is returned and exactly two errors are emitted to the log.
        """
        test_header = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\n"
        # '!' at position 2 is invalid
        test_sequence = "M!GNCCAGLSRRLKLPDCMA"

        with pytest.raises(
            ValueError, match="Invalid amino acid '!' found at position 2"
        ):
            fasta_parser_cls._validate_amino_acid_sequence(test_sequence, test_header)

        error_records = [rec for rec in caplog.records if rec.levelname == "ERROR"]

        assert len(error_records) == 1


class TestStructureParser:
    """Groups all unit tests validating the state, schema contracts, and side-effects of StructureParser."""

    # -------------------------------------------------------------------------
    # Internal Utility & File Inspection Tests
    # -------------------------------------------------------------------------

    def test_validate_structure_file_success(self, structure_parser_cls, tmp_path):
        """
        Ensures that _validate_structure_file returns the correct file extension when a valid file exists on disk.

        Arrange:
            Generate an isolated structure file with a .cif extension on disk.
        Act:
            Invoke the _validate_structure_file method.
        Assert:
            Verify .cif is returned.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        mock_cif_path.write_text(MOCK_CIF_CONTENT, encoding="utf-8")

        test_invoke = structure_parser_cls._validate_structure_file(mock_cif_path)

        assert test_invoke == ".cif"

    def test_load_and_inspect_success_with_pdb_file(
        self, structure_parser_cls, tmp_path
    ):
        """
        Ensures a tuple of unique chain_ids and a Biopython Structure object is returned for valid PDB files.

        Arrange:
            Generate an isolated directory with a valid .pdb file.
        Act:
            Invoke the _load_and_inspect method.
        Assert:
            Verify the expected tuple is returned.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_pdb_path = structure_dir / "test_structure_file.pdb"
        mock_pdb_path.write_text(MOCK_PDB_CONTENT_1, encoding="utf-8")

        test_chains, test_data = structure_parser_cls._load_and_inspect(
            mock_pdb_path, ".pdb"
        )

        assert test_chains == ["A"]
        assert isinstance(test_data, Structure)

    def test_load_and_inspect_multi_model_in_pdb(self, structure_parser_cls, caplog):
        """
        Verifies that _load_and_inspect logs a warning and defaults to Model 0 when a PDB file contains multi-model ensembles.

        Arrange:
            Configure a mock structure ensemble containing 3 models.
        Act:
            Invoke _load_and_inspect under an isolated PDBParser patch context.
        Assert:
            Verify the expected chain tuple is returned and a warning is logged.
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
            with caplog.at_level(logging.WARNING, logger="src.parsers"):
                chains, struct = structure_parser_cls._load_and_inspect(
                    file_path, file_ext
                )

        assert chains == ["A", "B"]
        assert struct == mock_structure
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "contains 3 models. Defaulting to Model 0." in caplog.text

    def test_load_and_inspect_with_invalid_pdb_error(
        self, structure_parser_cls, tmp_path
    ):
        """
        Verifies ValueError is raised when an empty or corrupted file with a .pdb extension is passed.

        Arrange:
            Generate an empty file with a .pdb extension.
        Act:
            Invoke the _load_and_inspect method.
        Assert:
            Verify a ValueError describing a malformed PDB structure is raised.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_pdb_path = structure_dir / "test_structure_file.pdb"
        mock_pdb_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed PDB file structure"):
            structure_parser_cls._load_and_inspect(mock_pdb_path, ".pdb")

    def test_load_and_inspect_success_with_cif_file(
        self, structure_parser_cls, tmp_path
    ):
        """
        Ensures a tuple of unique chain_ids and an MMCIF2Dict dictionary is returned for valid mmCIF files.

        Arrange:
            Generate an isolated directory with a valid .cif file.
        Act:
            Invoke the _load_and_inspect method.
        Assert:
            Verify the expected tuple is returned.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        mock_cif_path.write_text(MOCK_CIF_CONTENT, encoding="utf-8")

        test_chains, test_data = structure_parser_cls._load_and_inspect(
            mock_cif_path, ".cif"
        )

        assert test_chains == ["A", "B"]
        assert isinstance(test_data, dict)

    def test_load_and_inspect_with_invalid_cif_error(
        self, structure_parser_cls, tmp_path
    ):
        """
        Verifies ValueError is raised when an empty or unparseable .cif file path is passed.

        Arrange:
            Generate an empty file with a .cif extension.
        Act:
            Invoke the _load_and_inspect method.
        Assert:
            Verify a ValueError describing a malformed mmCIF is raised.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        mock_cif_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed mmCIF file"):
            structure_parser_cls._load_and_inspect(mock_cif_path, ".cif")

    def test_load_and_inspect_with_missing_key(self, structure_parser_cls, tmp_path):
        """
        Verifies ValueError is raised when an essential structural asymmetry authorization key is missing from a .cif file.

        Arrange:
            Generate a .cif file stripping out the standard structural asymmetry key (_atom_site.auth_asym_id).
        Act:
            Invoke the _load_and_inspect method.
        Assert:
            Verify a ValueError highlighting the specific missing key is raised.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        modified_data = MOCK_CIF_CONTENT.replace("_atom_site.auth_asym_id", "")
        mock_cif_path.write_text(modified_data, encoding="utf-8")

        with pytest.raises(
            ValueError, match="Missing essential key '_atom_site.auth_asym_id'"
        ):
            structure_parser_cls._load_and_inspect(mock_cif_path, ".cif")

    # -------------------------------------------------------------------------
    # Legacy & Fast-Path Parsers
    # -------------------------------------------------------------------------

    def test_parse_legacy_pdb_success(self, structure_parser_cls):
        """
        Validates that _parse_legacy_pdb accurately extracts spatial coordinates, residue names, B-factors, and occupancies from Biopython structures.

        Arrange:
            Assemble a mock Biopython Structure containing a single chain with one standard CA atom
            and one disordered CA atom with an alternate location.
        Act:
            Invoke _parse_legacy_pdb for target chain 'A'.
        Assert:
            Verify coordinates, residue strings, B-factors, and occupancies are extracted as float32 NumPy arrays
            and matching Python lists.
        """
        target_chain = "A"

        # Mock standard CA atom
        mock_atom_1 = MagicMock()
        mock_atom_1.is_disordered.return_value = False
        mock_atom_1.get_coord.return_value = [1.0, 2.0, 3.0]
        mock_atom_1.get_residue.return_value = "VAL"
        mock_atom_1.get_bfactor.return_value = 15.5
        mock_atom_1.get_occupancy.return_value = 1.0

        # Mock disordered CA atom (alternate loc)
        mock_selected_child = MagicMock()
        mock_selected_child.get_coord.return_value = [4.0, 5.0, 6.0]
        mock_selected_child.get_residue.return_value = "MET"
        mock_selected_child.get_bfactor.return_value = 22.1
        mock_selected_child.get_occupancy.return_value = 0.50

        mock_disordered_atom = MagicMock()
        mock_disordered_atom.is_disordered.return_value = True
        mock_disordered_atom.selected_child = mock_selected_child

        # Pack atoms into residues
        mock_res_1 = MagicMock()
        mock_res_1.id = (" ", 1, " ")
        mock_res_1.__contains__.return_value = True
        mock_res_1.__getitem__.return_value = mock_atom_1

        mock_res_2 = MagicMock()
        mock_res_2.id = (" ", 2, " ")
        mock_res_2.__contains__.return_value = True
        mock_res_2.__getitem__.return_value = mock_disordered_atom

        mock_chain = [mock_res_1, mock_res_2]
        mock_model = MagicMock()
        mock_model.__getitem__.return_value = mock_chain
        mock_structure = MagicMock()
        mock_structure.__getitem__.return_value = mock_model

        expected_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected_aa_residues = ["VAL", "MET"]
        expected_b_factors = np.array([15.5, 22.1], dtype=np.float32)
        expected_occupancies = np.array([1.0, 0.50], dtype=np.float32)

        result = structure_parser_cls._parse_legacy_pdb(mock_structure, target_chain)

        assert result["coords"].shape == (2, 3)
        assert result["coords"].dtype == np.float32
        assert np.array_equal(result["coords"], expected_coords)

        assert isinstance(result["aa_residues"], list)
        assert result["coords"].shape[0] == len(result["aa_residues"])
        assert result["aa_residues"] == expected_aa_residues

        assert result["b_factors"].dtype == np.float32
        assert np.array_equal(result["b_factors"], expected_b_factors)

        assert result["occupancies"].dtype == np.float32
        assert np.array_equal(result["occupancies"], expected_occupancies)

    def test_parse_legacy_pdb_no_ca_atoms_error(self, structure_parser_cls):
        """
        Verifies that _parse_legacy_pdb raises a ValueError if a requested chain contains no CA atoms.

        Arrange:
            Construct a mock residue containing non-CA atoms only.
        Act:
            Invoke _parse_legacy_pdb.
        Assert:
            ValueError is raised describing missing Alpha Carbon atoms.
        """
        mock_res = MagicMock()
        mock_res.id = (" ", 1, " ")
        mock_res.__contains__.return_value = False

        mock_chain = [mock_res]
        mock_model = MagicMock()
        mock_model.__getitem__.return_value = mock_chain
        mock_structure = MagicMock()
        mock_structure.__getitem__.return_value = mock_model

        with pytest.raises(
            ValueError, match="No valid Alpha Carbon \\(CA\\) atoms found"
        ):
            structure_parser_cls._parse_legacy_pdb(mock_structure, "A")

    def test_parse_mmcif_fast_path_success(self, structure_parser_cls):
        """
        Validates that _parse_mmcif_fast_path extracts coordinates, residues, B-factors, and occupancies from mmCIF dictionary payloads.

        Arrange:
            Construct a mock mmCIF dictionary containing polymer atoms, non-CA backbone atoms, and multiple chains.
        Act:
            Call _parse_mmcif_fast_path for target chain 'A'.
        Assert:
            Verify that non-CA atoms and off-target chains are filtered out, and valid arrays are returned.
        """
        target_chain = "A"
        mock_mmcif_dict = {
            "_atom_site.label_atom_id": ["CA", "N", "CA", "CA"],
            "_atom_site.auth_asym_id": ["A", "A", "A", "B"],
            "_atom_site.group_PDB": ["ATOM", "ATOM", "ATOM", "ATOM"],
            "_atom_site.Cartn_x": ["10.0", "11.0", "12.0", "20.0"],
            "_atom_site.Cartn_y": ["20.0", "21.0", "22.0", "30.0"],
            "_atom_site.Cartn_z": ["30.0", "31.0", "32.0", "40.0"],
            "_atom_site.auth_comp_id": ["SER", "PRO", "ALA", "HIS"],
            "_atom_site.B_iso_or_equiv": ["15.5", "18.2", "16.0", "22.1"],
            "_atom_site.occupancy": ["1.0", "1.0", "0.85", "1.0"],
        }

        expected_coords = np.array(
            [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]], dtype=np.float32
        )
        expected_aa_residues = ["SER", "ALA"]
        expected_b_factors = np.array([15.5, 16.0], dtype=np.float32)
        expected_occupancies = np.array([1.0, 0.85], dtype=np.float32)

        result = structure_parser_cls._parse_mmcif_fast_path(
            mock_mmcif_dict, target_chain
        )

        assert "coords" in result
        assert "aa_residues" in result
        assert "b_factors" in result

        assert result["coords"].shape == (2, 3)
        assert result["coords"].dtype == np.float32
        assert np.array_equal(result["coords"], expected_coords)

        assert isinstance(result["aa_residues"], list)
        assert result["aa_residues"] == expected_aa_residues
        assert result["coords"].shape[0] == len(result["aa_residues"])

        assert result["b_factors"].dtype == np.float32
        assert np.array_equal(result["b_factors"], expected_b_factors)

        assert result["occupancies"].dtype == np.float32
        assert np.array_equal(result["occupancies"], expected_occupancies)

    # -------------------------------------------------------------------------
    # High-Level Specialized Coordinate Extraction Routines
    # -------------------------------------------------------------------------

    @patch("polarity_engine.parsers.StructureParser._validate_structure_file")
    @patch("polarity_engine.parsers.StructureParser._load_and_inspect")
    @patch("polarity_engine.parsers.StructureParser._parse_legacy_pdb")
    def test_get_alpha_carbon_coordinates_pdb_routing(
        self, mock_parse_pdb, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Validates that get_alpha_carbon_coordinates routes .pdb files through legacy structure parsing routines.

        Arrange:
            Configure file validation to return .pdb, mock inspection, and return expected payload dictionary.
        Act:
            Call get_alpha_carbon_coordinates for chain 'A'.
        Assert:
            Verify validation, inspection, and legacy PDB parsing are called with correct arguments.
        """
        target_chain = "A"
        mock_validate.return_value = ".pdb"
        mock_struct = MagicMock()
        mock_inspect.return_value = (["A", "B"], mock_struct)

        expected_payload = {
            "coords": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "aa_residues": ["GLU"],
            "b_factors": np.array([15.5], dtype=np.float32),
            "occupancies": np.array([1.0], dtype=np.float32),
        }
        mock_parse_pdb.return_value = expected_payload

        result = structure_parser_cls.get_alpha_carbon_coordinates(
            "dummy.pdb", target_chain
        )

        assert np.array_equal(
            result[target_chain]["coords"], expected_payload["coords"]
        )
        assert result[target_chain]["aa_residues"] == expected_payload["aa_residues"]

        mock_validate.assert_called_once_with("dummy.pdb")
        mock_inspect.assert_called_once_with(Path("dummy.pdb"), ".pdb")
        mock_parse_pdb.assert_called_once_with(mock_struct, target_chain)

    @patch("polarity_engine.parsers.StructureParser._validate_structure_file")
    @patch("polarity_engine.parsers.StructureParser._load_and_inspect")
    @patch("polarity_engine.parsers.StructureParser._parse_mmcif_fast_path")
    def test_get_alpha_carbon_coordinates_cif_routing(
        self, mock_parse_mmcif, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Validates that get_alpha_carbon_coordinates routes .cif files through fast-path mmCIF parsing routines.

        Arrange:
            Configure file validation to return .cif, mock inspection, and return expected payload dictionary.
        Act:
            Call get_alpha_carbon_coordinates for chain 'A'.
        Assert:
            Verify validation, inspection, and mmCIF parsing are called with correct arguments.
        """
        target_chain = "A"
        mock_validate.return_value = ".cif"
        mock_mmcif_dict = MagicMock()
        mock_inspect.return_value = (["A", "B"], mock_mmcif_dict)

        expected_payload = {
            "coords": np.array([[12.345, 23.456, 34.567]], dtype=np.float32),
            "aa_residues": ["ARG"],
            "b_factors": np.array([22.1], dtype=np.float32),
            "occupancies": np.array([0.85], dtype=np.float32),
        }
        mock_parse_mmcif.return_value = expected_payload

        result = structure_parser_cls.get_alpha_carbon_coordinates(
            "dummy.cif", target_chain
        )

        assert np.array_equal(
            result[target_chain]["coords"], expected_payload["coords"]
        )
        assert result[target_chain]["aa_residues"] == expected_payload["aa_residues"]

        mock_validate.assert_called_once_with("dummy.cif")
        mock_inspect.assert_called_once_with(Path("dummy.cif"), ".cif")
        mock_parse_mmcif.assert_called_once_with(mock_mmcif_dict, target_chain)

    @patch("polarity_engine.parsers.StructureParser._validate_structure_file")
    @patch("polarity_engine.parsers.StructureParser._load_and_inspect")
    def test_get_alpha_carbon_coordinates_invalid_chain_error(
        self, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Verifies that get_alpha_carbon_coordinates raises a ValueError when requesting a non-existent chain.

        Arrange:
            Configure inspection to return available chains ['B', 'C'].
        Act:
            Invoke get_alpha_carbon_coordinates requesting chain 'A'.
        Assert:
            Verify a ValueError is raised identifying the missing chain.
        """
        requested_chain = "A"
        mock_validate.return_value = ".pdb"
        mock_inspect.return_value = (["B", "C"], MagicMock())

        with pytest.raises(
            ValueError, match=f"Requested chain '{requested_chain}' not found"
        ):
            structure_parser_cls.get_alpha_carbon_coordinates(
                "dummy.pdb", requested_chain
            )

    # -------------------------------------------------------------------------
    # Primary Public API Integration Tests (`StructureParser.parse`)
    # -------------------------------------------------------------------------

    def test_parse_multi_chain_complex_returns_complete_schema(
        self, structure_parser_cls, mock_cif_file, monkeypatch
    ):
        """
        Validates that StructureParser.parse() enforces full schema contracts for multi-chain complexes.

        Verifies that calling the unified parse() method on a valid PDBx/mmCIF structure
        file produces a fully-populated dictionary containing all four mandatory keys required by
        downstream PyTorch Geometric graph construction modules.

        Args:
            structure_parser_cls (Type[StructureParser]): Fixture providing class reference.
            mock_cif_file (Path): Fixture providing a path to a valid mock mmCIF file.

        Raises:
            AssertionError: If mandatory schema keys are missing, coordinate shapes do not match
                residue counts, data types diverge from float32, or node tuples are ill-formed.
        """
        mock_chain_coords = {
            "A": {
                "coords": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
                "aa_residues": ["VAL", "MET"],
            },
            "B": {
                "coords": [np.array([7.0, 8.0, 9.0]), np.array([10.0, 11.0, 12.0])],
                "aa_residues": ["ALA", "ARG"],
            },
        }
        monkeypatch.setattr(
            structure_parser_cls,
            "get_alpha_carbon_coordinates",
            lambda path: mock_chain_coords,
        )

        # Mock FreeSASA calls
        monkeypatch.setattr(
            structure_parser_cls,
            "get_freesasa_result",
            lambda path: (None, None),
        )
        monkeypatch.setattr(
            structure_parser_cls,
            "extract_per_residue_sasa",
            lambda *args, **kwargs: {
                ("A", "1"): 15.0,
                ("A", "2"): 12.0,
                ("B", "1"): 20.0,
                ("B", "2"): 18.0,
            },
        )

        # Execute parse
        parsed_output = structure_parser_cls.parse(mock_cif_file)

        # Assert full schema contract
        assert "aa_list" in parsed_output
        assert "coords" in parsed_output
        assert "nodes" in parsed_output
        assert "sasa_map" in parsed_output

        assert len(parsed_output["aa_list"]) == 4
        assert parsed_output["coords"].shape == (4, 3)
        assert len(parsed_output["nodes"]) == 4
        assert len(parsed_output["sasa_map"]) == 4

    def test_parse_chain_filtering_restricts_output(
        self, structure_parser_cls, mock_cif_file, monkeypatch
    ):
        """
        Validates that StructureParser.parse() isolates user-specified chains.

        Tests the optional chain_ids parameter to ensure that coordinate arrays, residue lists,
        and node metadata tuples are strictly filtered to contain only residues belonging to requested chains.

        Args:
            structure_parser_cls (Type[StructureParser]): Class reference fixture for StructureParser.
            mock_cif_file (Path): Fixture providing a path to a multi-chain mock mmCIF file.

        Raises:
            AssertionError: If residues from unrequested chains persist in output.
        """
        target_chain = "A"

        # 1. Mock CA coordinate parser output with multi-chain dictionary
        mock_chain_coords = {
            "A": {
                "coords": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
                "aa_residues": ["VAL", "MET"],
            },
            "B": {
                "coords": [np.array([7.0, 8.0, 9.0]), np.array([10.0, 11.0, 12.0])],
                "aa_residues": ["ALA", "ARG"],
            },
        }
        monkeypatch.setattr(
            structure_parser_cls,
            "get_alpha_carbon_coordinates",
            lambda path: mock_chain_coords,
        )

        # 2. Mock FreeSASA calls
        monkeypatch.setattr(
            structure_parser_cls,
            "get_freesasa_result",
            lambda path: (None, None),
        )
        monkeypatch.setattr(
            structure_parser_cls,
            "extract_per_residue_sasa",
            lambda *args, **kwargs: {
                ("A", "1"): 15.0,
                ("A", "2"): 12.0,
                ("B", "1"): 20.0,
                ("B", "2"): 18.0,
            },
        )

        # 3. Execute chain-restricted parsing
        parsed_filtered = structure_parser_cls.parse(
            mock_cif_file, chain_ids=[target_chain]
        )

        extracted_chains = {node[0] for node in parsed_filtered["nodes"]}
        assert extracted_chains == {
            target_chain
        }, f"Expected only chain '{target_chain}', found {extracted_chains}"
        assert len(parsed_filtered["aa_list"]) == 2

    def test_parse_nonexistent_chain_raises_value_error(
        self, structure_parser_cls, mock_cif_file
    ):
        """
        Verifies that requesting a non-existent chain ID raises a ValueError.

        Ensures that if a user supplies a chain_ids list containing a chain label that does not exist,
        the pipeline fails cleanly with an explicit ValueError.

        Args:
            structure_parser_cls (Type[StructureParser]): Class reference fixture for StructureParser.
            mock_cif_file (Path): Fixture providing a path to a mock mmCIF file.

        Raises:
            pytest.PytestTester: Confirms ValueError is raised when invalid chain is passed.
        """
        invalid_chain = "NON_EXISTENT_CHAIN_XYZ"
        with pytest.raises(
            ValueError, match=r"Failed to parse (valid residue|structure file)"
        ):
            structure_parser_cls.parse(mock_cif_file, chain_ids=[invalid_chain])

    def test_parse_invalid_file_path_raises_file_not_found_error(
        self, structure_parser_cls, tmp_path
    ):
        """
        Ensures that passing a non-existent file path to parse() raises FileNotFoundError.

        Verifies that file existence guardrails at the public entry point intercept missing
        structures before initializing downstream parser dependencies.

        Args:
            structure_parser_cls (Type[StructureParser]): Class reference fixture for StructureParser.
            tmp_path (Path): Built-in pytest fixture providing a temporary directory Path.

        Raises:
            pytest.PytestTester: Confirms FileNotFoundError is raised for non-existent paths.
        """
        non_existent_file = tmp_path / "missing_structure.cif"
        with pytest.raises(FileNotFoundError, match="Target structure file not found"):
            structure_parser_cls.parse(non_existent_file)

    def test_parse_empty_sasa_map_raises_value_error(
        self, structure_parser_cls, mock_cif_file, monkeypatch
    ):
        """
        Verifies that a failure in SASA mapping raises a ValueError instead of returning partial schema output.

        Args:
            structure_parser_cls (Type[StructureParser]): Class reference fixture for StructureParser.
            mock_cif_file (Path): Fixture providing a path to a mock mmCIF file.
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture to patch method return.

        Raises:
            pytest.PytestTester: Confirms ValueError is raised when SASA map is empty.
        """
        monkeypatch.setattr(
            structure_parser_cls, "extract_per_residue_sasa", lambda res, struct: {}
        )

        with pytest.raises(ValueError, match=r"Failed to parse structure file"):
            structure_parser_cls.parse(mock_cif_file)
