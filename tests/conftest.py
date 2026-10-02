# tests/conftest.py
from pathlib import Path
import gemmi
import pytest


def create_tiny_cif(
    input_path: Path,
    output_path: Path,
    chain_id: str = "B",
    max_residues: int = 15,
) -> None:
    """Slices a reference mmCIF structure file down to a small residue subset.

    Parses an input macromolecular structure in mmCIF format using Gemmi, filters
    the model down to a target polymer chain, truncates the residue list to a specified
    maximum count, and serializes the reduced macromolecular representation to disk.
    This helper is primarily utilized to generate lightweight, self-contained test
    fixtures for fast unit and integration testing.

    Args:
        input_path (Path): Path to the source mmCIF file (e.g., full wild-type structure).
        output_path (Path): Destination path where the truncated mmCIF file will be written.
            Parent directories will be created automatically if they do not exist.
        chain_id (str, optional): The target chain identifier to isolate. Defaults to "B".
        max_residues (int, optional): The maximum number of N-terminal residues to retain
            in the isolated chain. Defaults to 15.

    Raises:
        FileNotFoundError: If the source mmCIF file at `input_path` does not exist.
        ValueError: If the requested `chain_id` is not present within the structure model.
        RuntimeError: If Gemmi encounters a parser or serialization error during file processing.

    Example:
        >>> from pathlib import Path
        >>> create_tiny_cif(
        ...     input_path=Path("data/8r3y.cif"),
        ...     output_path=Path("tests/fixtures/tiny_sample.cif"),
        ...     chain_id="B",
        ...     max_residues=10,
        ... )
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = gemmi.cif.read_file(str(input_path))
    st = gemmi.make_structure_from_block(doc[0])

    model = st[0]
    for chain in list(model):
        if chain.name != chain_id:
            model.remove_chain(chain.name)

    if chain_id not in [chain.name for chain in model]:
        raise ValueError(f"Chain '{chain_id}' not found in structure model.")

    target_chain = model[chain_id]
    del target_chain[max_residues:]

    st.make_mmcif_document().write_file(str(output_path))


@pytest.fixture(scope="session", autouse=True)
def ensure_tiny_cif_fixture() -> Path:
    """Session-scoped Pytest fixture ensuring `tiny_sample.cif` exists before testing.

    Automatically runs at the start of the pytest test session. Checks for the existence
    of the lightweight test fixture `tests/fixtures/tiny_sample.cif`. If the fixture
    does not exist but the full reference dataset (`data/8r3y.cif`) is present, it invokes
    `create_tiny_cif` to dynamically construct the truncated structure file.

    AAA Pattern Role:
        Arrange: Prepares the minimal file system environment and macromolecular structural
                 inputs required for downstream integration and pipeline end-to-end tests.

    Returns:
        Path: Resolved filesystem path to the validated `tiny_sample.cif` fixture file.

    Notes:
        - Scope: `session` (executes at most once per test suite invocation).
        - Autouse: `True` (invoked automatically without requiring explicit parameter injection).
    """
    fixture_path = Path("tests/fixtures/tiny_sample.cif")
    source_path = Path("data/8r3y.cif")

    if not fixture_path.exists() and source_path.exists():
        create_tiny_cif(source_path, fixture_path, chain_id="B", max_residues=15)

    return fixture_path
