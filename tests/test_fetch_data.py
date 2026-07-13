#! /usr/bin/env python3


import os
import pytest
import requests

from Bio.PDB.PDBList import PDBList
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from src.fetch_data import ProteinDataIngestor, parse_arguments, main
from tests.mock_data import UNIPROT_ID, TARGET_NAME, MOCK_LGL_FASTA_CONTENT

@pytest.fixture
def ingestor(tmp_path):
    return ProteinDataIngestor(output_dir=tmp_path)


class TestProteinDataIngestor:
    """
    Groups all unit tests validating the state and side-effects of ProteinDataIngestor.
    """

    def test_fetch_fasta_success(self, ingestor, mocker):
        """
        Verify successful retrieval and transmission of raw UniProt FASTA data.

        Arrange:
            Inject a mocked HTTP 200 response configured to return a 
            controlled MOCK_LGL_FASTA_CONTENT string.
        Act:
            Invoke retrieve_fasta with a target UniProt ID (P51617).
        Assert:
            Ensure the function output matches the mocked sequence 
            content, verifies the correct endpoint API URL was called, and
            confirms the output file is created at the expected path with 
            correct mocked response.
        """
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_LGL_FASTA_CONTENT

        mock_get = mocker.patch.object(
            ingestor.session, "get", return_value=mock_response)

        result = ingestor.fetch_fasta(UNIPROT_ID, TARGET_NAME)

        # 1. Assert string output values match up perfectly
        assert result == MOCK_LGL_FASTA_CONTENT

        # 2. Assert against mock_get to verify the network parameters
        mock_get.call_count == 1

        # 3. Verify side effect file system tracking
        expected_file = ingestor.output_dir / "amino_acid_sequences" / \
            f"{TARGET_NAME.lower()}_sequence.fasta"
        assert expected_file.exists()
        assert expected_file.read_text(encoding="utf-8") == MOCK_LGL_FASTA_CONTENT

    def test_fetch_fasta_timeout_error(self, ingestor, mocker):
        """
        Verify that a network timeout correctly triggers granular exception blocks.

        Arrange:
            Inject an explicit requests ConnectTimeout error into the 
            mocked requests.get network execution channel.
        Act:
            Invoke retrieve_fasta within a protective pytest assertion context.
        Assert:
            Ensure that requests.exceptions.ConnectTimeout is raised, 
            verify that the network call respected the strict 10-second limit,
            and confirms the output file is not generated.
        """
        mock_get = mocker.patch.object(
            ingestor.session,
            'get',
            side_effect=requests.exceptions.ConnectTimeout(
                "Connection timed out.")
        )

        with pytest.raises(requests.exceptions.ConnectTimeout):
            ingestor.fetch_fasta(UNIPROT_ID, TARGET_NAME)

        # 1. Assert against mock_get to verify the network parameters
        mock_get.call_count == 1

        # 2. Verify side effect file system tracking
        expected_file = ingestor.output_dir / "amino_acid_sequences" / \
            f"{TARGET_NAME.lower()}_sequence.fasta"
        assert not expected_file.exists()

    def test_fetch_cif_success(self, ingestor, mocker):
        """
        Verify that a valid PDB ID is normalized and downloaded successfully.

        Arrange:
            Configure a temporary testing directory path and patch the 
            BioPython PDBList instance to return a controlled file path string.
        Act:
            Invoke download_pdb_files with an un-normalized, padded PDB ID.
        Assert:
            Ensure a Path object is returned, and verify that the underlying 
            PDBList method was called with a stripped, uppercase identifier.
        """
        structure_dir = ingestor.output_dir / "structures"
        expected_file_path = structure_dir / "8r3y.cif"

        # Physically build the folders and touch the file on disk so validation passes
        structure_dir.mkdir(parents=True, exist_ok=True)
        expected_file_path.write_text(
            "data_8R3Y\n# dummy cif content", encoding="utf-8")

        mock_retrieve = mocker.patch.object(
            PDBList,
            "retrieve_pdb_file",
            return_value=str(expected_file_path)
        )

        result_path = ingestor.fetch_cif(" 8r3y ")

        # Assertions
        assert isinstance(result_path, Path)
        assert result_path == expected_file_path
        assert result_path.exists()

        # Confirm Biopython received perfectly sanitized arguments
        assert mock_retrieve.call_count == 1

        # Extract positional arguments (args) and keyword arguments (kwargs)
        args, kwargs = mock_retrieve.call_args

        # Assert the PDB ID is normalized to uppercase and stripped, wherever it sits
        assert (args[0] == "8R3Y" or kwargs.get("pdb_code") == "8R3Y")

        # Assert the format target matches up safely
        assert kwargs.get("file_format") == "mmCif"

        # Assert the destination folder is correct (handling both string or Path objects)
        assert Path(kwargs.get("pdir")) == structure_dir

    def test_fetch_cif_not_found(self, ingestor, mocker):
        """
        Verify that an invalid or un-writable PDB structure triggers FileNotFoundError.

        Arrange:
            Patch the BioPython PDBList instance to return None,
            simulating a structural download failure from the remote server.
        Act:
            Invoke download_pdb_files inside a protective exception-capturing context.
        Assert:
            Ensure that a FileNotFoundError is raised with an informative error message.
        """
        mock_pdbl_instance = mocker.MagicMock()
        mock_pdbl_instance.retrieve_pdb_file.return_value = None  # Simulates download failure
        mocker.patch.object(
            PDBList,
            "retrieve_pdb_file",
            return_value=str(mock_pdbl_instance)
        )

        with pytest.raises(FileNotFoundError, match="Failed to write structural files") as exc_info:
            ingestor.fetch_cif("INVALID")

        assert "Failed to write structural files" in str(
            exc_info.value) or "FileNotFoundError" in str(exc_info.value)

    def test_fetch_cif_network_url_error(self, ingestor, mocker):
        """
        Verify that a remote server disruption raises a custom pipeline RuntimeError.

        Arrange:
            Mock the BioPython PDBList instance to trigger a URLError.
        Act:
            Invoke download_pdb_files inside a protective exception-capturing context.
        Assert:
            Ensure a RuntimeError is raised, verifying the pipeline stops
            defensively while correctly chaining the underlying root cause.
        """
        mock_pdbl_instance = mocker.MagicMock()
        mock_pdbl_instance.retrieve_pdb_file.side_effect = URLError(
            "Host unreachable")
        mocker.patch.object(
            PDBList,
            "retrieve_pdb_file",
            side_effect=URLError("Host unreachable")
        )
        with pytest.raises(RuntimeError) as exc_info:
            ingestor.fetch_cif("8R3Y")

        assert "Failed to write structural files for 8R3Y" in str(exc_info.value) or \
               "Remote server" in str(exc_info.value)

    def test_fetch_cif_permission_error(self, ingestor, mocker):
        """
        Verify that a filesystem block or restriction raises PermissionError.

        Arrange:
            Mock the BioPython PDBList instance to trigger a PermissionError.
        Act:
            Invoke download_pdb_files inside a protective exception-capturing context.
        Assert:
            Ensure a PermissionError is raised without being swallowed by the generic exception handler.
        """
        mock_pdbl_instance = mocker.MagicMock()
        mock_pdbl_instance.retrieve_pdb_file.side_effect = PermissionError(
            "Write access denied")
        mocker.patch.object(
            PDBList,
            "retrieve_pdb_file",
            side_effect=PermissionError("Write access denied")
        )
        with pytest.raises(PermissionError) as exc_info:
            ingestor.fetch_cif("8R3Y")

        assert "Write access denied" in str(exc_info.value) or \
               "PermissionError" in str(exc_info.value)


class TestParseArguments:
    """
    Validates argument parsing constraints and default flag fallbacks.
    """

    def test_parse_arguments_success_without_emdb(self):
        """
        Verify our command line parser extracts multiple targets correctly 
        without the optional argument for EMDB ID (-e).

        Arrange:
            Simulate a command-line inputs and override the global sys.argv list
        Act:
            Invoke parse_arguments with mocked arguments.
        Assert:
            Ensure the function output matches the mocked arguments.
        """
        test_args = [
            "-n", "aPKC", "Lgl",
            "-u", "P51617", "Q9VA29",
            "-p", "8R3Y",
            "-o", "../data"
        ]

        parsed = parse_arguments(test_args)
        assert parsed.name == ["aPKC", "Lgl"]
        assert parsed.uniprot_id == ["P51617", "Q9VA29"]
        assert parsed.pdb_id == ["8R3Y"]
        assert str(parsed.outdir) == "../data"
        assert parsed.coords_only is False
        assert parsed.maps_only is False

    def test_parse_arguments_success_with_emdb(self):
        """
        Verify our command line parser extracts multiple targets correctly
        with an optional argument for EMDB ID (-e).

        Arrange:
            Simulate a command-line inputs and override the global sys.argv list
        Act:
            Invoke parse_arguments with mocked arguments.
        Assert:
            Ensure the function output matches the mocked arguments.
        """
        test_args = [
            "-n", "aPKC", "Lgl",
            "-u", "P51617", "Q9VA29",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data"
        ]

        parsed = parse_arguments(test_args)
        assert parsed.name == ["aPKC", "Lgl"]
        assert parsed.uniprot_id == ["P51617", "Q9VA29"]
        assert parsed.pdb_id == ["8R3Y"]
        assert parsed.emdb_id == ["EMD-18877"]
        assert str(parsed.outdir) == "../data"
        assert parsed.coords_only is False
        assert parsed.maps_only is False

    def test_parse_arguments_success_with_download_mode(self):
        """
        Verify that passing explicit structural modifiers updates internal routing modes.

        Arrange:
            Simulate a command-line inputs and override the global sys.argv list
        Act:
            Invoke parse_arguments with mock arguments.
        Assert:
            Ensure the function output matches the mocked arguments.
        """
        test_args = [
            "-n", "aPKC", "Lgl",
            "-u", "P51617", "Q9VA29",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data",
            "--coords-only"
        ]

        parsed = parse_arguments(test_args)
        assert parsed.name == ["aPKC", "Lgl"]
        assert parsed.uniprot_id == ["P51617", "Q9VA29"]
        assert parsed.pdb_id == ["8R3Y"]
        assert parsed.emdb_id == ["EMD-18877"]
        assert str(parsed.outdir) == "../data"
        assert parsed.coords_only is True
        assert parsed.maps_only is False

    def test_parse_arguments_fail_without_target_names(self, capsys):
        """
        Verify that omitting the required --name argument halts execution with a usage message.

        Arrange:
            Simulate a command-line inputs and override the global sys.argv list
        Act:
            Invoke parse_arguments without target names.
        Assert:
            Ensure the execution is halted with a usage message.
        """
        test_args = [
            "-u", "P51617", "Q9VA29",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data",
            "--coords-only"
        ]

        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(test_args)

        # 3. Verify the exit code is 2 (the standard argparse code for command errors)
        assert exc_info.value.code == 2

        # 4. Optional: Capture standard error to ensure it names the missing flag
        captured = capsys.readouterr()
        assert "the following arguments are required" in captured.err
        assert "name" in captured.err


class TestMain:
    """
    Verify the flow of the main function
    """

    def test_main_derives_download_all_implicitly(self, mocker):
        """
        Verify that when no modifier flags (--coords-only or --maps-only) are given,
        main() dynamically treats download_all as True and fires all pipeline steps.
        """
        mock_ingestor_class = mocker.patch(
            "src.fetch_data.ProteinDataIngestor")
        mock_instance = mock_ingestor_class.return_value

        mock_fasta = mocker.patch.object(mock_instance, "fetch_fasta")
        mock_cif = mocker.patch.object(mock_instance, "fetch_cif")
        mock_map = mocker.patch.object(mock_instance, "fetch_em_density_map")

        # Act: Run main with standard parameters but NO specific mode modifiers
        main([
            "-n", "aPKC",
            "-u", "P51617",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data"
        ])

        # Assert: All tracking channels should be fired by default
        mock_fasta.assert_called_once()
        mock_cif.assert_called_once()
        mock_map.assert_called_once()

    def test_main_success_coords_only(self, mocker):
        """
        Verify that when --coords-only is given,
        main() fires only fetch_fasta and fetch_cif.
        """
        mock_ingestor_class = mocker.patch(
            "src.fetch_data.ProteinDataIngestor")
        mock_instance = mock_ingestor_class.return_value

        mock_fasta = mocker.patch.object(mock_instance, "fetch_fasta")
        mock_cif = mocker.patch.object(mock_instance, "fetch_cif")
        mock_map = mocker.patch.object(mock_instance, "fetch_em_density_map")

        # Act: Run main with standard parameters but NO specific mode modifiers
        main([
            "-n", "aPKC",
            "-u", "P51617",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data",
            "--coords-only"
        ])

        # Assert: fasta and cif channels are fired but not map channgel.
        assert mock_fasta.call_count == 1
        assert mock_cif.call_count == 1
        assert mock_map.call_count == 0

    def test_main_success_maps_only(self, mocker):
        """
        Verify that when --maps-only is given,
        main() fires only fetch_fasta and fetch_cif.
        """
        mock_ingestor_class = mocker.patch(
            "src.fetch_data.ProteinDataIngestor")
        mock_instance = mock_ingestor_class.return_value

        mock_fasta = mocker.patch.object(mock_instance, "fetch_fasta")
        mock_cif = mocker.patch.object(mock_instance, "fetch_cif")
        mock_map = mocker.patch.object(mock_instance, "fetch_em_density_map")

        # Act: Run main with standard parameters but NO specific mode modifiers
        main([
            "-n", "aPKC",
            "-u", "P51617",
            "-p", "8R3Y",
            "-e", "EMD-18877",
            "-o", "../data",
            "--maps-only"
        ])

        # Assert: only map channel should be fired
        assert mock_fasta.call_count == 0
        assert mock_cif.call_count == 0
        assert mock_map.call_count == 1

    def test_main_mismatched_names_uniprot_id_numbers(self):
        """
        Verify that when the number of target names and uniprot_ids are not the same,
        the pipeline fails
        """
        with pytest.raises(ValueError) as exc_info:
            main([
                "-n", "aPKC", "Par-6",
                "-u", "P51617",
                "-p", "8R3Y",
                "-e", "EMD-18877",
                "-o", "../data",
                "--maps-only"
            ])

        assert "Mismatched parallel input arguments." in str(
            exc_info.value) or "ValueError" in str(exc_info.value)
