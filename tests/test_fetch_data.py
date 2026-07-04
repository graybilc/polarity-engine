from src.fetch_data import download_pdb_files
from urllib.error import URLError
import os
from src.fetch_data import save_text_to_disk
import pytest
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.fetch_data import retrieve_fasta, save_text_to_disk, download_pdb_files, parse_arguments

MOCK_FASTA_CONTENT = ">tr|A0A024RBG1|A0A024RBG1_HUMAN N-asymmetry factor\nMGNCCAGLSRRL\nKLPDCMA\n"


def test_retrieve_fasta_success(mocker):
    """
    Verify successful retrieval and transmission of raw UniProt FASTA data.

    Arrange:
        Inject a mocked HTTP 200 response configured to return a 
        controlled MOCK_FASTA_CONTENT string.
    Act:
        Invoke retrieve_fasta with a target UniProt ID (P51617).
    Assert:
        Ensure the function output matches the mocked sequence 
        content and verifies the correct endpoint API URL was called.
    """
    mock_get = mocker.patch('src.fetch_data.requests.get')
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = MOCK_FASTA_CONTENT

    result = retrieve_fasta("P51617")

    assert "MGNCCAGLSRRL" in result
    mock_get.assert_called_once_with(
        "https://rest.uniprot.org/uniprotkb/P51617.fasta",
        timeout=10
    )


def test_retrieve_fasta_timeout_error(mocker):
    """
    Verify that a network timeout correctly triggers granular exception blocks.

    Arrange:
        Inject an explicit requests ConnectTimeout error into the 
        mocked requests.get network execution channel.
    Act:
        Invoke retrieve_fasta within a protective pytest assertion context.
    Assert:
        Ensure that requests.exceptions.ConnectTimeout is raised, 
        and verify that the network call respected the strict 10-second limit.
    """
    mock_get = mocker.patch('src.fetch_data.requests.get', side_effect=requests.exceptions.ConnectTimeout(
        "Connection timed out."))

    with pytest.raises(requests.exceptions.ConnectTimeout):
        retrieve_fasta("P51617")

    mock_get.assert_called_once_with(
        "https://rest.uniprot.org/uniprotkb/P51617.fasta",
        timeout=10
    )


def test_save_text_to_disk_success(tmp_path):
    """
    Verify successful text stream serialization to a physical disk path.

    Arrange: 
        Set up a secure temporary testing directory, content, and file name.
    Act:
        Invoke save_text_to_disk to process and execute the write operations.
    Assert:
        Ensure the file exists, has correct content, and matches the target path.
    """
    sample_content = MOCK_FASTA_CONTENT
    sample_filename = "test_seq.fasta"

    returned_path = save_text_to_disk(
        sample_content, sample_filename, tmp_path)

    expected_path = (tmp_path / sample_filename).resolve()
    assert returned_path == expected_path
    assert returned_path.exists()
    assert returned_path.read_text(encoding="utf-8") == sample_content


def test_save_text_to_disk_missing_directory_raises_ioerror():
    """
    Verify that trying to write to a non-existent directory raises an IOError.

    Arrange:
        Construct a path pointing to a completely imaginary directory.
    Act:
        Attempt to save a text asset into the missing folder topology.
    Assert:
        Ensure an IOError is thrown defensively by the file handler.
    """
    invalid_dir = Path("/this/directory/does/not/exist/anywhere")

    with pytest.raises(IOError):
        save_text_to_disk("data", "output.fasta", invalid_dir)


def test_save_text_to_disk_permission_denied_raises_ioerror(tmp_path, mocker):
    """
    Verify that underlying OS write blocks or lockups correctly trigger IOErrors.

    Arrange:
        Mock the built-in open method to forcefully trigger a PermissionError.
    Act:
        Invoke save_text_to_disk within an error-capturing block.
    Assert:
        Verify that the application accurately catches and raises the IOError.
    """
    mocker.patch("builtins.open",
                 side_effect=PermissionError("Permission denied"))

    with pytest.raises(IOError):
        save_text_to_disk("data", "secure.fasta", tmp_path)


def test_download_pdb_files_success(mocker, tmp_path):
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
    mock_downloaded_path = str(tmp_path / "8r3y.cif")

    # Mock the PDBList class and its method
    mock_pdbl_instance = mocker.MagicMock()
    mock_pdbl_instance.retrieve_pdb_file.return_value = mock_downloaded_path
    mocker.patch("src.fetch_data.PDB.PDBList", return_value=mock_pdbl_instance)

    # Mock os.path.exists to simulate the file being written on disk successfully
    mocker.patch("src.fetch_data.os.path.exists", return_value=True)

    # Passing lowercase with spaces to test the .strip().upper() optimization
    result_path = download_pdb_files(" 8r3y ", tmp_path)

    assert isinstance(result_path, Path)
    assert result_path == Path(mock_downloaded_path)
    mock_pdbl_instance.retrieve_pdb_file.assert_called_once_with(
        "8R3Y", pdir=tmp_path, file_format="mmCif"
    )


def test_download_pdb_files_not_found(mocker, tmp_path):
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
    mocker.patch("src.fetch_data.PDB.PDBList", return_value=mock_pdbl_instance)

    with pytest.raises(FileNotFoundError, match="Failed to write structural files"):
        download_pdb_files("INVALID", tmp_path)


def test_download_pdb_files_network_url_error(mocker, tmp_path):
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
    mocker.patch("src.fetch_data.PDB.PDBList", return_value=mock_pdbl_instance)

    with pytest.raises(RuntimeError, match="Pipeline stopped: Remote server unreachable"):
        download_pdb_files("8R3Y", tmp_path)


def test_download_pdb_files_permission_error(mocker, tmp_path):
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
    mocker.patch("src.fetch_data.PDB.PDBList", return_value=mock_pdbl_instance)

    with pytest.raises(PermissionError):
        download_pdb_files("8R3Y", tmp_path)


def test_parse_arguments_structure():
    """
    Verify our command line parser extracts multiple targets correctly.

    Arrange:
        Simulate a command-line inputs and override the global sys.argv list
    Act:
        Invoke parse_arguments with a target UniProt ID (P51617).
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
    assert Path(parsed.outdir) == Path("../data")
