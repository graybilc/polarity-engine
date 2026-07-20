#!/urs/bin/env python3


import logging
import numpy as np
import pytest

from Bio.PDB.Structure import Structure
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.parsers import FastaParser, StructureParser
from tests.mock_data import MOCK_LGL_FASTA_CONTENT, MOCK_APKC_FASTA_CONTENT, MOCK_PDB_CONTENT_1, MOCK_CIF_CONTENT


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
            "sp|P41743|KPCI_HUMAN Protein kinase C iota type OS=Homo sapiens OX=9606 GN=PRKCI PE=1 SV=2": "MPTQRDSSTMSHTVAGGGSGDHS"
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

        with pytest.raises(FileNotFoundError, match="Target FASTA file not found or invalid"):
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

        with pytest.raises(ValueError, match="Empty sequence string encountered under header"):
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
            test_sequence, test_header)

        assert test_output == test_output

    def test_validate_amino_acid_sequence_non_standard_success(self, fasta_parser_cls, caplog):
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
            test_sequence, test_header)

        # Filter captured log records to isolate warnings from this test execution
        warning_records = [
            rec for rec in caplog.records if rec.levelname == "WARNING"]

        assert test_output == test_sequence
        assert len(warning_records) == 2
        assert "Non-standard amino acid 'U' found at position 2" in warning_records[0].message
        assert "Non-standard amino acid 'Z' found at position 17" in warning_records[1].message

    def test_validate_amino_acid_sequence_empty_string_error(self, fasta_parser_cls, caplog):
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

        with pytest.raises(ValueError, match="Empty sequence string encountered under header"):
            fasta_parser_cls._validate_amino_acid_sequence(
                test_sequence, test_header)

        error_records = [
            rec for rec in caplog.records if rec.levelname == "ERROR"]

        assert len(error_records) == 1

    def test_validate_amino_acid_sequence_invalid_sequence_error(self, fasta_parser_cls, caplog):
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

        with pytest.raises(ValueError, match="Invalid amino acid '!' found at position 2"):
            fasta_parser_cls._validate_amino_acid_sequence(
                test_sequence, test_header)

        error_records = [
            rec for rec in caplog.records if rec.levelname == "ERROR"]

        assert len(error_records) == 1


class TestStructureParser:
    """
    Groups all unit tests validating the state and side-effects of StructureParser
    """

    def test_validate_structure_file_success(self, structure_parser_cls, tmp_path):
        """
        Ensure that _validate_structure_file returns file extension when valid file path is passed.

        Arrange:
            Generate an isolated structure file with .cif extension on disk.
        Act:
            Invoke the _validate_structure_file method.
        Assert:
            Verify .cif is returned.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        mock_cif_path.write_text(MOCK_PDB_CONTENT_1, encoding="utf-8")

        test_invoke = structure_parser_cls._validate_structure_file(
            mock_cif_path)

        assert test_invoke == ".cif"

    def test_validate_structure_file_error(self, structure_parser_cls, tmp_path):
        """
        Ensure FileNotFoundError is raised when an invalid file path is passed.

        Arrange:
            Generate an isolated directory without structure file on disk.
        Act:
            Invoke the _validate_structure_file method.
        Assert:
            FileNotFoundError is raised.
        """
        mock_cif_path = tmp_path / "structures" / "non_existent_file.cif"

        with pytest.raises(FileNotFoundError, match="Target structure file not found"):
            structure_parser_cls._validate_structure_file(mock_cif_path)

    def test_load_and_inspect_success_with_pdb_file(self, structure_parser_cls, tmp_path):
        """
        Ensure a tuple of list of unique chain_ids and a Structure object is returned.

        Arrange:
            Generate an isolated directory with a valid .pdb file
        Act:
            invoke _load_and_inspect method.
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
        Verifies that _load_and_inspect logs a warning and defaults to Model 0
        when a PDB file contains more than one model.

        Arrange:
            Configure a mock structure ensemble containing 3 models.
        Act:
            invoke _load_and_inspect method under an isolated get_structure patch context.
        Assert:
            Verify the expected tuple is returned and a warning is logged.
        """
        file_path = Path("mock_multi_model.pdb")
        file_ext = ".pdb"

        mock_structure = MagicMock()
        mock_structure.__len__.return_value = 3

        mock_model_0 = MagicMock()
        mock_model_0.child_dict.keys.return_value = ["A", "B"]
        mock_structure.__getitem__.return_value = mock_model_0

        # Patch the PDBParser instance method directly to avoid filesystem checks
        with patch("src.parsers.PDBParser.get_structure", return_value=mock_structure):
            with caplog.at_level(logging.WARNING, logger="src.parsers"):
                chains, struct = structure_parser_cls._load_and_inspect(
                    file_path, file_ext
                )

        assert chains == ["A", "B"]
        assert struct == mock_structure
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "contains 3 models. Defaulting to Model 0." in caplog.text

    def test_load_and_inspect_with_invalid_pdb_error(self, structure_parser_cls, tmp_path):
        """
        Verify ValueError is raised when an invalid file path is passed.

        Arrange:
            Generate an empty/corrupted file with a .pdb extension.
        Act:
            invoke _load_and_inspect method.
        Assert:
            Verify a ValueError describing a malformed PDB structure is raised.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_pdb_path = structure_dir / "test_structure_file.pdb"
        mock_pdb_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed PDB file structure"):
            structure_parser_cls._load_and_inspect(mock_pdb_path, ".pdb")

    def test_load_and_inspect_success_with_cif_file(self, structure_parser_cls, tmp_path):
        """
        Ensure a tuple of list of unique chain_ids and an MMCIF2Dict instance is returned.

        Arrange:
            Generate an isolated directory with a valid .cif file
        Act:
            invoke _load_and_inspect method.
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

        assert test_chains == ["A"]
        # MMCIF2Dict inherits directly from dict
        assert isinstance(test_data, dict)

    def test_load_and_inspect_with_invalid_cif_error(self, structure_parser_cls, tmp_path):
        """
        Verify ValueError is raised when an unparseable .cif file path is passed.

        Arrange:
            Generate an empty file with a .cif extension.
        Act:
            invoke _load_and_inspect method.
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
        Verify ValueError is raised when an essential authorization key is missing from a .cif file.

        Arrange:
            Generate a .cif file stripping out the standard structural asymmetry key.
        Act:
            invoke _load_and_inspect method.
        Assert:
            Verify a ValueError highlighting the specific missing key is raised.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        modified_data = MOCK_CIF_CONTENT.replace("_atom_site.auth_asym_id", "")
        mock_cif_path.write_text(modified_data, encoding="utf-8")

        with pytest.raises(ValueError, match="Missing essential key '_atom_site.auth_asym_id'"):
            structure_parser_cls._load_and_inspect(mock_cif_path, ".cif")

    def test_parse_legacy_pdb_success(self, structure_parser_cls):
        """
        Arrange:
            Assemble a mock Biopython Structure containing a single chain with 
            one standard CA atom and one disordered CA atom with an altloc.
        Act:
            Invoke _parse_legacy_pdb for target chain 'A'.
        Assert:
            Verify that CA coordinates, B-factors, and occupancies are correctly 
            extracted, handling disordered child selection, and returned in the 
            nested dictionary structure under key 'A'.
        """
        target_chain = "A"

        # Mock standard CA atom
        mock_atom_1 = MagicMock()
        mock_atom_1.is_disordered.return_value = False
        mock_atom_1.get_coord.return_value = [1.0, 2.0, 3.0]
        mock_atom_1.get_bfactor.return_value = 15.5
        mock_atom_1.get_occupancy.return_value = 1.0

        # Mock disordered CA atom (simulating an altloc flexible residue)
        mock_selected_child = MagicMock()
        mock_selected_child.get_coord.return_value = [4.0, 5.0, 6.0]
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

        # Mock the internal tree layers
        mock_chain = [mock_res_1, mock_res_2]
        mock_model = MagicMock()
        mock_model.__getitem__.return_value = mock_chain
        mock_structure = MagicMock()
        mock_structure.__getitem__.return_value = mock_model

        expected_coords = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected_b_factors = np.array([15.5, 22.1], dtype=np.float32)
        expected_occupancies = np.array([1.0, 0.50], dtype=np.float32)

        result = structure_parser_cls._parse_legacy_pdb(
            mock_structure, target_chain)

        assert target_chain in result
        chain_data = result[target_chain]

        assert chain_data["coords"].shape == (2, 3)
        assert chain_data["coords"].dtype == np.float32
        assert np.array_equal(chain_data["coords"], expected_coords)

        assert chain_data["b_factors"].dtype == np.float32
        assert np.array_equal(chain_data["b_factors"], expected_b_factors)

        assert chain_data["occupancies"].dtype == np.float32
        assert np.array_equal(chain_data["occupancies"], expected_occupancies)

    def test_parse_legacy_pdb_no_ca_atoms_error(self, structure_parser_cls):
        """
        Verify ValueError is raised if a chain exists but contains no valid CA atoms.
        """
        mock_res = MagicMock()
        mock_res.id = (" ", 1, " ")
        mock_res.__contains__.return_value = False  # No CA atom present

        mock_chain = [mock_res]
        mock_model = MagicMock()
        mock_model.__getitem__.return_value = mock_chain
        mock_structure = MagicMock()
        mock_structure.__getitem__.return_value = mock_model

        with pytest.raises(ValueError, match="No valid Alpha Carbon \\(CA\\) atoms found"):
            structure_parser_cls._parse_legacy_pdb(mock_structure, "A")

    def test_parse_mmcif_fast_path_success(self, structure_parser_cls):
        """
        Arrange:
            Construct a mock mmCIF dictionary containing standard polymer atoms,
            non-CA backbone atoms, multiple chains, B-factors, and occupancy values.
        Act: 
            Call _parse_mmcif_fast_path for target chain 'A'.
        Assert:
            Verify that non-CA atoms and off-target chains are filtered out, and 
            that spatial coordinates, b-factors, and occupancies are parsed into 
            the correct float32 NumPy arrays under the chain key.
        """
        target_chain = "A"
        mock_mmcif_dict = {
            "_atom_site.label_atom_id": ["CA", "N", "CA", "CA"],
            "_atom_site.auth_asym_id": ["A", "A", "A", "B"],
            "_atom_site.group_PDB": ["ATOM", "ATOM", "ATOM", "ATOM"],
            "_atom_site.Cartn_x": ["10.0", "11.0", "12.0", "20.0"],
            "_atom_site.Cartn_y": ["20.0", "21.0", "22.0", "30.0"],
            "_atom_site.Cartn_z": ["30.0", "31.0", "32.0", "40.0"],
            "_atom_site.B_iso_or_equiv": ["15.5", "18.2", "16.0", "22.1"],
            "_atom_site.occupancy": ["1.0", "1.0", "0.85", "1.0"],
        }

        expected_coords = np.array(
            [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0]], dtype=np.float32
        )
        expected_b_factors = np.array([15.5, 16.0], dtype=np.float32)
        expected_occupancies = np.array([1.0, 0.85], dtype=np.float32)

        result = structure_parser_cls._parse_mmcif_fast_path(
            mock_mmcif_dict, target_chain
        )

        assert target_chain in result
        chain_data = result[target_chain]

        assert chain_data["coords"].shape == (2, 3)
        assert chain_data["coords"].dtype == np.float32
        assert np.array_equal(chain_data["coords"], expected_coords)

        assert chain_data["b_factors"].dtype == np.float32
        assert np.array_equal(chain_data["b_factors"], expected_b_factors)

        assert chain_data["occupancies"].dtype == np.float32
        assert np.array_equal(chain_data["occupancies"], expected_occupancies)

    @patch("src.parsers.StructureParser._validate_structure_file")
    @patch("src.parsers.StructureParser._load_and_inspect")
    @patch("src.parsers.StructureParser._parse_legacy_pdb")
    def test_get_alpha_carbon_coordinates_pdb_routing(
        self, mock_parse_pdb, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Arrange: 
            Configure file validation to return '.pdb' extension, mock structure inspection,
            and define mock parser return dictionary containing spatial coordinates, 
            B-factors, and occupancies.
        Act:
            Call get_alpha_carbon_coordinates requesting specific chain 'A'.
        Assert:
            Verify that validation, inspection, and legacy PDB parsing are invoked with 
            correct parameters, and that the target chain payload is returned.
        """
        target_chain = "A"
        mock_validate.return_value = ".pdb"
        mock_struct = MagicMock()
        mock_inspect.return_value = (["A", "B"], mock_struct)

        expected_payload = {
            "coords": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "b_factors": np.array([15.5], dtype=np.float32),
            "occupancies": np.array([1.0], dtype=np.float32),
        }

        mock_parse_pdb.return_value = expected_payload

        result = structure_parser_cls.get_alpha_carbon_coordinates(
            "dummy.pdb", target_chain)

        assert target_chain in result
        chain_data = result[target_chain]
        assert np.array_equal(chain_data["coords"], expected_payload["coords"])
        assert np.array_equal(
            chain_data["b_factors"], expected_payload["b_factors"])
        assert np.array_equal(
            chain_data["occupancies"], expected_payload["occupancies"])

        mock_validate.assert_called_once_with("dummy.pdb")
        mock_inspect.assert_called_once_with(Path("dummy.pdb"), ".pdb")
        mock_parse_pdb.assert_called_once_with(mock_struct, target_chain)

    @patch("src.parsers.StructureParser._validate_structure_file")
    @patch("src.parsers.StructureParser._load_and_inspect")
    @patch("src.parsers.StructureParser._parse_mmcif_fast_path")
    def test_get_alpha_carbon_coordinates_cif_routing(
        self, mock_parse_mmcif, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Arrange:
            Configure file validation to return '.cif' extension, mock structure inspection,
            and define mock mmCIF parser return dictionary containing spatial coordinates, 
            B-factors, and occupancies for chain 'A'.
        Act:
            Call get_alpha_carbon_coordinates requesting specific chain 'A'.
        Assert:
            Verify that validation, inspection, and fast-path mmCIF parsing are invoked 
            with correct parameters, and that the target chain payload is returned.
        """
        target_chain = "A"
        mock_validate.return_value = ".cif"
        mock_mmcif_dict = MagicMock()
        mock_inspect.return_value = (["A", "B"], mock_mmcif_dict)

        expected_payload = {
            "coords": np.array([[12.345, 23.456, 34.567]], dtype=np.float32),
            "b_factors": np.array([22.1], dtype=np.float32),
            "occupancies": np.array([0.85], dtype=np.float32),
        }
        mock_parse_mmcif.return_value = expected_payload

        result = structure_parser_cls.get_alpha_carbon_coordinates(
            "dummy.cif", target_chain)

        assert target_chain in result
        chain_data = result[target_chain]
        assert np.array_equal(chain_data["coords"], expected_payload["coords"])
        assert np.array_equal(
            chain_data["b_factors"], expected_payload["b_factors"])
        assert np.array_equal(
            chain_data["occupancies"], expected_payload["occupancies"])

        mock_validate.assert_called_once_with("dummy.cif")
        mock_inspect.assert_called_once_with(Path("dummy.cif"), ".cif")
        mock_parse_mmcif.assert_called_once_with(mock_mmcif_dict, target_chain)

    @patch("src.parsers.StructureParser._validate_structure_file")
    @patch("src.parsers.StructureParser._load_and_inspect")
    def test_get_alpha_carbon_coordinates_invalid_chain_error(
        self, mock_inspect, mock_validate, structure_parser_cls
    ):
        """
        Arrange:
            Configure mock structure inspection to return available chains ['B', 'C'], 
            omitting the requested chain 'A'.
        Act:
            Invoke get_alpha_carbon_coordinates requesting chain 'A'.
        Assert:
            Verify that a ValueError is raised with a message identifying the missing chain.
        """
        requested_chain = "A"
        mock_validate.return_value = ".pdb"
        mock_inspect.return_value = (["B", "C"], MagicMock())

        with pytest.raises(ValueError, match=f"Requested chain '{requested_chain}' not found"):
            structure_parser_cls.get_alpha_carbon_coordinates(
                "dummy.pdb", requested_chain)
