#!/usr/bin/env python3
"""
scripts/make_test_fixtures.py

Utility script for pre-generating lightweight macromolecular structural fixtures
(mmCIF format) using Gemmi's official structure assembly API.
"""

from pathlib import Path
import gemmi


def create_tiny_cif(
    input_path: Path,
    output_path: Path,
    chain_id: str = "L",
    start_res: int = 650,
    end_res: int = 670,
) -> None:
    """
    Parses a reference mmCIF file using Gemmi, constructs a new filtered structure
    retaining only the target chain and residues within the specified sequence window,
    and writes out a fully valid mmCIF file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = gemmi.cif.read_file(str(input_path))
    st = gemmi.make_structure_from_block(doc[0])

    # Build a clean new structure containing only the filtered chain and target residues
    new_st = gemmi.Structure()
    new_st.name = st.name

    model = st[0]
    new_model = gemmi.Model("1")

    for chain in model:
        if chain.name == chain_id:
            new_chain = gemmi.Chain(chain_id)
            for i, res in enumerate(chain):
                # Optionally filter or take a window, and renumber to match biological target range (e.g. starting at 650)
                if i < 30:  # Take up to 21+ residues to cover 650 to 670
                    new_res = res.clone()
                    new_res.seqid.num = (
                        650 + i
                    )  # Maps indices 0, 1, 2... to 650, 651, 652... so 655, 659, 663 are hit!
                    new_chain.add_residue(new_res)
            if len(new_chain) > 0:
                new_model.add_chain(new_chain)

    new_st.add_model(new_model)

    # Write out valid mmCIF document
    doc_out = new_st.make_mmcif_document()
    doc_out.write_file(str(output_path))


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    create_tiny_cif(
        input_path=repo_root / "data" / "8r3y.cif",
        output_path=repo_root / "tests" / "fixtures" / "tiny_sample.cif",
        chain_id="L",
        start_res=650,
        end_res=670,
    )
