"""
tests/test_kinetics.py
Unit tests for PhosphoKineticSolver mass conservation and 3^3 hypercube integrity.
"""

import itertools
import numpy as np
import pytest
import torch
from polarity_engine.kinetics import PhosphoKineticSolver


class MockGraph:
    """Mock PyG graph object providing required node and feature tensor attributes."""

    def __init__(self):
        # Mock nodes covering site numbers 655, 659, 663
        self.nodes = [
            ("B", "655", "SER"),
            ("B", "659", "SER"),
            ("B", "663", "SEP"),
        ]
        # Feature tensor where column 23 represents RSASA
        self.x = torch.rand(3, 30, dtype=torch.float32)


def test_mass_conservation():
    """
    Verify that total concentration across all 27 3^3 hypercube states is conserved.

    Arrange:
        Initialize mock structural graphs with valid node and feature tensor attributes
        for all 27 ternary states, empty site mappings, and instantiate the solver.
    Act:
        Run the time-course ODE simulation from t=0 to t=50 and calculate
        the sum of concentrations across all 27 states for each evaluated time point.
    Assert:
        Verify that the total concentration array remains equal to 1.0 within
        strict numerical tolerance (rtol=1e-5, atol=1e-5).
    """
    state_codes = ["".join(p) for p in itertools.product(["0", "1", "2"], repeat=3)]
    dummy_graphs = {code: MockGraph() for code in state_codes}

    # Map site keys to residue numbers matching MockGraph nodes
    dummy_mappings = {
        "sites": {
            "site_1": {"res_num": "655"},
            "site_2": {"res_num": "659"},
            "site_3": {"res_num": "663"},
        }
    }

    solver = PhosphoKineticSolver(
        graphs=dummy_graphs, site_mappings=dummy_mappings, regime="processive"
    )

    t, y = solver.simulate(t_span=(0, 50))
    total_concentration = np.sum(y, axis=0)

    np.testing.assert_allclose(
        total_concentration,
        1.0,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Mass conservation violated: Total sum of states deviated from 1.0",
    )


def test_invalid_regime_raises_error():
    """
    Ensure invalid kinetic regimes are correctly caught during initialization.

    Arrange:
        Prepare empty structural graphs and mapping configurations.
    Act:
        Attempt to instantiate PhosphoKineticSolver with an unsupported regime string
        ("invalid_regime").
    Assert:
        Verify that a ValueError is successfully raised.
    """
    with pytest.raises(ValueError):
        PhosphoKineticSolver(graphs={}, site_mappings={}, regime="invalid_regime")
