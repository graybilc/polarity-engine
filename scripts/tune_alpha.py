"""
scripts/tune_alpha.py
Performs a grid search over the structural gating hyperparameter alpha
using a held-out validation split of historical assays.
Ensures strict data hygiene by keeping the final test set (Parker & McDonald 2025) sealed.
"""

import numpy as np
from polarity_engine.kinetics import PhosphoKineticSolver


def evaluate_alpha(
    alpha_val: float, val_dataset: list, graphs: dict, site_mappings: dict
) -> float:
    """
    Runs time-course ODE simulations for a given alpha value and computes
    the Mean Squared Error (MSE) against historical validation assay targets.

    Args:
        alpha_val (float): The structural gating sensitivity hyperparameter controlling
            how aggressively rSASA scales reaction rates.
        val_dataset (list of dict): A collection of historical validation assay records,
            each containing keys for regime, t_max, target_state_idx, target_time,
            and experimental_value.
        graphs (dict): A dictionary mapping hypercube state codes (e.g., '000')
            to their respective PyTorch Geometric structural graph objects.
        site_mappings (dict): A dictionary mapping functional site identifiers
            (e.g., 'site_3') to structural node feature indices.

    Returns:
        float: The average Mean Squared Error across all samples in the validation dataset.
    """
    total_error = 0.0
    for sample in val_dataset:
        solver = PhosphoKineticSolver(
            graphs=graphs,
            site_mappings=site_mappings,
            alpha=alpha_val,
            regime=sample["regime"],
        )
        t, y = solver.simulate(t_span=(0, sample["t_max"]))

        # Extract simulated concentration at target state and time point
        sim_val = _interpolate_concentration(
            t, y, sample["target_state_idx"], sample["target_time"]
        )

        # Compute squared error against historical assay benchmark
        error = (sim_val - sample["experimental_value"]) ** 2
        total_error += error

    return total_error / len(val_dataset)


def _interpolate_concentration(
    t: np.ndarray, y: np.ndarray, state_idx: int, target_time: float
) -> float:
    """
    Interpolates state concentration at a precise target time point from solver output trajectories.

    Args:
        t (np.ndarray): 1D array of time points evaluated by the ODE solver.
        y (np.ndarray): 2D array of state concentrations where shape is (num_states, num_time_points).
        state_idx (int): The integer index representing the specific hypercube state
            being evaluated (e.g., 7 for state '111').
        target_time (float): The exact temporal coordinate where concentration needs to be estimated.

    Returns:
        float: The linearly interpolated concentration value for the specified state at the target time.
    """
    return np.interp(target_time, t, y[state_idx])


if __name__ == "__main__":
    # --- 1. SETUP VALIDATION DATA (Historical Assays, NOT the Test Set) ---
    val_dataset = [
        {
            "regime": "processive",
            "t_max": 20,
            "target_state_idx": 7,
            "target_time": 10.0,
            "experimental_value": 0.65,
        },
        {
            "regime": "dissociative",
            "t_max": 20,
            "target_state_idx": 7,
            "target_time": 10.0,
            "experimental_value": 0.40,
        },
    ]

    # Mock structural graphs and site mappings for validation run
    mock_graphs = {
        code: None for code in ["000", "100", "010", "001", "110", "101", "011", "111"]
    }
    mock_site_mappings = {"site_3": {"res_num": 138}}

    # --- 2. DEFINE HYPERPARAMETER GRID ---
    alpha_grid = np.linspace(0.0, 3.0, 13)
    best_alpha = None
    lowest_error = float("inf")

    print("--- Starting Alpha Grid Search on Validation Split ---")
    for alpha in alpha_grid:
        mse = evaluate_alpha(alpha, val_dataset, mock_graphs, mock_site_mappings)
        print(f"Alpha: {alpha:.2f} | Validation MSE: {mse:.5f}")
        if mse < lowest_error:
            lowest_error = mse
            best_alpha = alpha

    print(f"\nOptimization Complete!")
    print(
        f"Selected Optimal Alpha: {best_alpha:.2f} (Validation MSE: {lowest_error:.5f})"
    )
    print(
        "Status: Final test set (Parker & McDonald 2025) remains completely untouched and sealed."
    )
