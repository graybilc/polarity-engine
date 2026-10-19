"""
tests/test_kinetics.py
Unit tests for PhosphoKineticSolver mass conservation and hypercube integrity.
"""

import numpy as np
import pytest
from polarity_engine.kinetics import PhosphoKineticSolver


def test_mass_conservation():
    """
    Verify that total concentration across all 8 hypercube states is conserved.

    Arrange:
        Initialize mock structural graphs, empty site mappings, and instantiate
        the PhosphoKineticSolver under the default processive regime.
    Act:
        Run the time-course ODE simulation from t=0 to t=50 and calculate
        the sum of concentrations across all 8 states for each evaluated time point.
    Assert:
        Verify that the total concentration array remains equal to 1.0 within
        strict numerical tolerance (rtol=1e-5, atol=1e-5).
    """
    dummy_graphs = {
        code: None for code in ["000", "100", "010", "001", "110", "101", "011", "111"]
    }
    dummy_mappings = {}

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
