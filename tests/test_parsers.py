#!/urs/bin/env python3


import pytest
from Bio.PDB.Structure import Structure

from src.parsers import FastaParser, StructureParser
from tests.mock_data import MOCK_LGL_FASTA_CONTENT, MOCK_APKC_FASTA_CONTENT, MOCK_PDB_CONTENT


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
    def test_validate_structure_file_success(cls, structure_parser_cls, tmp_path):
        """
        Ensure that _validate_structure_file returns file extention when valid file path is passed.

        Arrange:
            Generate an isolated structure file with .cif extention on disk.
        Act:
            Invoke the _validate_structure_file method.
        Assert:
            Verify .cif is returned.
        """
        structure_dir = tmp_path / "structures"
        structure_dir.mkdir(parents=True, exist_ok=True)

        mock_cif_path = structure_dir / "test_structure_file.cif"
        mock_cif_path.write_text(MOCK_PDB_CONTENT, encoding="utf-8")

        test_invoke = structure_parser_cls._validate_structure_file(
            mock_cif_path)

        assert test_invoke == ".cif"
    
    def test_validate_structure_file_error(cls, structure_parser_cls, tmp_path):
        """
        Ensure FileNotFoundError is raised when an inalid file path is passed.

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

    def test_load_and_inspect_success_with_pdb_file(cls, structure_parser_cls, tmp_path):
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

        mock_cif_path = structure_dir / "test_structure_file.pdb"
        mock_cif_path.write_text(MOCK_PDB_CONTENT, encoding="utf-8")

        test_chains, test_data = structure_parser_cls._load_and_inspect(
            mock_cif_path, ".pdb")
        
        print(test_data)
        assert test_chains == ["A"]
        assert type(test_data) == Structure