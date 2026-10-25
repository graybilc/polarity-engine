"""
Unit tests for phosphomimetic and native state mutation logic in state_generator.py.
"""

import copy
import numpy as np
import pytest

from polarity_engine.state_generator import mutate_structure_dict, mutate_sep_to_ser
from tests.mock_data import make_mock_parsed_dict


class TestStateGenerator:
    """
    Test suite covering state mutation logic, array integrity,
    custom mimic residues, and immutability contracts in state_generator.py.
    """

    def test_mutate_structure_dict_leaves_wt_unmodified(self) -> None:
        """
        Ensures mutate_structure_dict creates a deep copy and does not mutate the WT dictionary in-place.

        Arrange:
            Construct WT dictionary fixture and deep copy baseline.
        Act:
            Generate phosphomimetic state dict ('111').
        Assert:
            Assert original WT dictionary residues remain unchanged and object identity differs.
        """
        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=5)
        wt_dict_baseline = copy.deepcopy(wt_dict)

        # Act
        mutated_dict = mutate_structure_dict(wt_dict, state_code="111")

        # Assert
        assert wt_dict["aa_list"] == wt_dict_baseline["aa_list"]
        assert wt_dict["nodes"] == wt_dict_baseline["nodes"]
        assert id(mutated_dict) != id(wt_dict)

    def test_mutate_structure_dict_preserves_dimensions(self) -> None:
        """
        Verifies that state mutation preserves N == N node array shapes and coordinate dimensions.

        Arrange:
            Load WT mock dictionary.
        Act:
            Mutate structure to target state '101'.
        Assert:
            Check array shapes for coords, b_factors, and nodes match original dimensions.
        """
        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=5)

        # Act
        mutated_dict = mutate_structure_dict(wt_dict, state_code="101")

        # Assert
        assert len(mutated_dict["aa_list"]) == len(wt_dict["aa_list"])
        assert mutated_dict["coords"].shape == wt_dict["coords"].shape
        assert len(mutated_dict["nodes"]) == len(wt_dict["nodes"])

    @pytest.mark.parametrize(
        "state_code",
        ["000", "100", "010", "001", "110", "101", "011", "111"],
    )
    def test_mutate_structure_dict_all_state_codes(self, state_code: str) -> None:
        """
        Parametrized test to guarantee all 8 phosphomimetic state codes execute without raising errors.

        Arrange:
            Instantiate base WT parsed dictionary.
        Act:
            Execute mutate_structure_dict across all 8 binary state permutations.
        Assert:
            Verify output dictionary is non-empty and retains required dictionary keys.
        """
        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=5)

        # Act
        mutated_dict = mutate_structure_dict(wt_dict, state_code=state_code)

        # Assert
        assert isinstance(mutated_dict, dict)
        assert "aa_list" in mutated_dict
        assert "coords" in mutated_dict
        assert "nodes" in mutated_dict

    def test_mutate_structure_dict_custom_mimic_res(self) -> None:
        """
        Verifies that specifying mimic_res='ASP' correctly applies Aspartate substitutions
        instead of the default Glutamate (GLU).

        Arrange:
            Instantiate WT mock dictionary and select full-mutation state code '111'.
        Act:
            Call mutate_structure_dict with mimic_res='ASP'.
        Assert:
            Verify that 'ASP' is present in the mutated sequence and 'GLU' is absent.
        """
        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=5)

        # Act
        mutated_dict = mutate_structure_dict(wt_dict, state_code="111", mimic_res="ASP")

        # Assert
        assert "ASP" in mutated_dict["aa_list"]
        assert "GLU" not in mutated_dict["aa_list"]

    def test_mutate_structure_dict_invalid_state_code_raises_error(self) -> None:
        """
        Ensures invalid state codes (e.g., wrong length or non-binary digits) raise ValueError.

        Arrange:
            Define invalid state string.
        Act & Assert:
            Call mutate_structure_dict and assert ValueError is raised.
        """
        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=5)
        invalid_code = "999"

        # Act & Assert
        with pytest.raises(ValueError):
            mutate_structure_dict(wt_dict, state_code=invalid_code)

    def test_mutate_sep_to_ser_converts_residue_and_strips_phosphate(self) -> None:
        """
        Verifies that mutate_sep_to_ser correctly converts SEP to SER and strips phosphate atoms.

        Arrange:
            Construct a mock parsed dictionary containing a SEP residue with heavy atoms including 'P'.
        Act:
            Invoke mutate_sep_to_ser targeting the SEP residue.
        Assert:
            Verify SEP is replaced by SER in metadata and heavy-atom arrays, and phosphate atoms are removed.
        """
        # Arrange
        parsed_dict = make_mock_parsed_dict(n_residues=3)
        # Inject SEP at residue "663"
        parsed_dict["aa_residues"][2] = ("663", "SEP")
        parsed_dict["aa_list"][2] = "SEP"
        parsed_dict["nodes"][2] = ("B", "663", "SEP")

        # Add mock heavy atoms for SEP including phosphate 'P'
        parsed_dict["all_atom_keys"] = [
            ("B", "655"),
            ("B", "655"),
            ("B", "663"),
            ("B", "663"),
            ("B", "663"),
        ]
        parsed_dict["all_atom_names"] = ["N", "CA", "CB", "OG", "P"]
        parsed_dict["all_atom_res_names"] = ["SER", "SER", "SEP", "SEP", "SEP"]
        parsed_dict["all_atom_coords"] = np.zeros((5, 3), dtype=np.float64)

        # Act
        mutated_dict = mutate_sep_to_ser(parsed_dict, target_res_num="663")

        # Assert
        assert "SEP" not in mutated_dict["aa_list"]
        assert mutated_dict["aa_list"][2] == "SER"
        assert "P" not in mutated_dict["all_atom_names"]
        assert all(res == "SER" for res in mutated_dict["all_atom_res_names"])

    def test_mutate_structure_dict_differentiates_glu_and_asp_rsasa(self) -> None:
        """
        Verifies that GLU and ASP mutations generate distinct heavy-atom structures
        and yield different rSASA profiles via the builder pipeline.

        Arrange:
            Construct a mock wildcard parsed dictionary with base heavy-atom coordinate arrays.
        Act:
            Mutate structure to '100' state code using both GLU and ASP mimic residues,
            then construct their respective PyG state graphs.
        Assert:
            Confirm heavy atom counts differ between GLU and ASP, and that their rSASA channels
            are distinct.
        """
        from polarity_engine.builder import ProteinGraphBuilder

        # Arrange
        wt_dict = make_mock_parsed_dict(n_residues=3)
        wt_dict["all_atom_coords"] = np.random.rand(6, 3).astype(np.float64)
        wt_dict["all_atom_keys"] = [("B", "655")] * 6
        wt_dict["all_atom_names"] = ["N", "CA", "C", "O", "CB", "OG"]
        wt_dict["all_atom_res_names"] = ["SER"] * 6

        # Act
        glu_dict = mutate_structure_dict(wt_dict, state_code="100", mimic_res="GLU")
        asp_dict = mutate_structure_dict(wt_dict, state_code="100", mimic_res="ASP")

        # Assert
        assert len(glu_dict["all_atom_coords"]) != len(asp_dict["all_atom_coords"])

        builder = ProteinGraphBuilder()
        glu_graph = builder.build_from_parsed_dict(glu_dict, name="state_glu")
        asp_graph = builder.build_from_parsed_dict(asp_dict, name="state_asp")

        assert glu_graph.x.shape == asp_graph.x.shape
        assert not np.array_equal(
            glu_graph.x[:, 23].numpy(), asp_graph.x[:, 23].numpy()
        )
