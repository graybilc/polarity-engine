#!/usr/bin/env python3
"""
scripts/make_test_fixtures.py

Utility script for pre-generating lightweight macromolecular structural fixtures
(mmCIF format) used across unit, integration, and Nextflow pipeline test suites.
"""

from pathlib import Path
import gemmi


def create_tiny_cif(
    input_path: Path,
    output_path: Path,
    chain_id: str = "B",
    max_residues: int = 15,
) -> None:
    """
    Slices a reference mmCIF structure file down to a small residue subset.

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
        ...     max_residues=15,
        ... )
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Source structure file not found: {input_path}")

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
    print(
        f"Successfully generated fixture '{output_path}' ({len(target_chain)} residues)."
    )


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    create_tiny_cif(
        input_path=repo_root / "data" / "8r3y.cif",
        output_path=repo_root / "tests" / "fixtures" / "tiny_sample.cif",
        chain_id="B",
        max_residues=15,
    )
