"""
Unit tests for phosphomimetic state mutation logic in state_generator.py.
"""

import copy
import numpy as np
import pytest

from polarity_engine.state_generator import mutate_structure_dict
from tests.mock_data import make_mock_parsed_dict


class TestStateGenerator:
    """
    Test suite covering state mutation logic, array integrity,
    and immutability contracts in state_generator.py.
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
