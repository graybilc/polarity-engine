"""
src/polarity_engine/kinetics.py

Generalized Mass-Action ODE Solver for 3^3 Ternary State Networks (000 to 222).
Evaluates Processive vs. Dissociative multi-site phosphorylation regimes of Lgl
by aPKC-Par6, incorporating phospho-ablative Alanine ('2') knockouts alongside
wild-type ('0') and phosphomimetic ('1') states.

Incorporates ADR-0006 compliant Tier 1 parameterization derived exclusively
from historical bulk assays (Graybill & Prehoda 2014; Graybill et al. 2012).
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

NODE_FEATURE_INDICES = {"RSASA": 23}


class PhosphoKineticSolver:
    """
    Simulates concentration trajectories across a 27-state mass-action ODE grid
    ('000' to '222') for multi-site phosphorylation and kinetic ablation of Lgl by aPKC-Par6.

    Attributes:
        graphs (Dict[str, Any]): Structural PyG graph representations for each state.
        site_mappings (Dict[str, Any]): Abstract functional site slot definitions.
        alpha (float): Structural sensitivity scaling factor for rSASA modulation.
        par6_active (bool): Flag indicating whether aPKC is activated via Par-6 binding.
        regime (str): Kinetic regime, either 'processive' or 'dissociative'.
        base_rates (Dict[str, float]): Tier 1 parameterization dictionary of baseline rates.
    """

    def __init__(
        self,
        graphs: Dict[str, Any],
        site_mappings: Dict[str, Any],
        alpha: float = 1.2,
        par6_active: bool = True,
        regime: str = "processive",
    ):
        """
        Initializes the PhosphoKineticSolver with 27-state grid mappings and kinetic parameters.

        Args:
            graphs (Dict[str, Any]): Structural PyG graph representations for each state.
            site_mappings (Dict[str, Any]): Abstract functional site slot definitions.
            alpha (float, optional): Sensitivity multiplier for rSASA gating. Defaults to 1.2.
            par6_active (bool, optional): Whether aPKC-Par6 is active. Defaults to True.
            regime (str, optional): Operating kinetic regime ('processive' or 'dissociative'). Defaults to "processive".

        Returns:
            None

        Raises:
            ValueError: If the regime argument is not 'processive' or 'dissociative'.
        """
        self.graphs = graphs
        self.site_mappings = site_mappings
        self.alpha = alpha
        self.par6_active = par6_active
        self.regime = regime.lower()
        if self.regime not in ["processive", "dissociative"]:
            raise ValueError("Regime must be either 'processive' or 'dissociative'.")

        # Generate all 27 ternary state codes ('000', '001', ..., '222')
        self.state_codes: List[str] = [
            "".join(p) for p in itertools.product(["0", "1", "2"], repeat=3)
        ]
        self.state_to_idx: Dict[str, int] = {
            code: i for i, code in enumerate(self.state_codes)
        }

        # --- TIER 1 PARAMETERIZATION SET (ADR-0006 Compliant) ---
        k_cat_multiplier = 8.5 if self.par6_active else 1.0

        # Site-specific baseline phosphorylation rates
        self.site_base_rates = {
            0: 0.04 * k_cat_multiplier,  # Site 1 (S655)
            1: 0.07 * k_cat_multiplier,  # Site 2 (S659)
            2: 0.22 * k_cat_multiplier,  # Site 3 (S663)
        }

        # Baseline dephosphorylation / background phosphatase rates
        self.dephospho_rates = {
            0: 0.02,
            1: 0.03,
            2: 0.01,
        }

    def _resolve_target_res_num(self, functional_site_id: str) -> str:
        """
        Safely resolves residue number string from site mapping structures.

        Args:
            functional_site_id (str): Functional site identifier (e.g., 'site_1', 'site_2', 'site_3').

        Returns:
            str: Resolved amino acid sequence residue number string (e.g., '655').
        """
        sites_dict = self.site_mappings
        if "sites" in sites_dict:
            sites_dict = sites_dict["sites"]

        site_entry = sites_dict.get(functional_site_id, functional_site_id)
        if isinstance(site_entry, dict):
            return str(site_entry.get("res_num", functional_site_id))
        return str(site_entry)

    def _compute_rate_multiplier(
        self, state_code: str, functional_site_id: str
    ) -> float:
        """
        Computes the state-specific rSASA gating rate multiplier using
        Linear Free Energy Relationship (LFER) exponential scaling.

        Args:
            state_code (str): Ternary state code representing the current structural state (e.g., '001', '120').
            functional_site_id (str): Functional site identifier (e.g., residue number or site key).

        Returns:
            float: Calculated Arrhenius-style exponential rate multiplier based on relative SASA changes.
        """
        graph = self.graphs.get(state_code)
        wt_graph = self.graphs.get("000")

        if graph is None:
            raise KeyError(
                f"Missing structural PyG graph for state code '{state_code}' in graphs dictionary."
            )
        if not hasattr(graph, "nodes"):
            raise AttributeError(
                f"Graph object for state code '{state_code}' lacks required 'nodes' attribute."
            )

        target_res_num = self._resolve_target_res_num(functional_site_id)

        # Flexible node lookup supporting strings, integers, and different tuple positions
        res_idx = None
        for i, node in enumerate(graph.nodes):
            # Check node elements or attributes safely
            node_vals = node if isinstance(node, (list, tuple)) else [node]
            if any(
                str(val) == target_res_num
                or str(val).lstrip("0") == target_res_num.lstrip("0")
                for val in node_vals
            ):
                res_idx = i
                break

        if res_idx is None:
            raise ValueError(
                f"Target residue number '{target_res_num}' (from {functional_site_id}) "
                f"not found in graph nodes for state code '{state_code}' (Nodes: {graph.nodes[:5]}...)."
            )

        r_sasa = graph.x[res_idx, NODE_FEATURE_INDICES["RSASA"]].item()

        # Wild-type graph fallback can remain gentle or raise, but raising ensures strict baseline integrity
        wt_r_sasa = r_sasa
        if wt_graph is not None and hasattr(wt_graph, "nodes"):
            wt_res_idx = next(
                (
                    i
                    for i, node in enumerate(wt_graph.nodes)
                    if str(node[1]) == target_res_num
                ),
                None,
            )
            if wt_res_idx is not None and wt_res_idx < wt_graph.x.shape[0]:
                wt_r_sasa = wt_graph.x[wt_res_idx, NODE_FEATURE_INDICES["RSASA"]].item()

        delta_r_sasa = r_sasa - wt_r_sasa
        return float(np.exp(self.alpha * delta_r_sasa))

    def _get_regime_scaler(self, transition_from: str, transition_to: str) -> float:
        """
        Applies processive vs. dissociative topological scaling based on
        accumulated phosphomimetic/phosphorylation state transitions.

        Args:
            transition_from (str): Ternary state code of the originating state (e.g., '000').
            transition_to (str): Ternary state code of the destination state (e.g., '001').

        Returns:
            float: Topological scaling factor (>1.0 for processive acceleration,
                   <1.0 for dissociative penalty, 1.0 otherwise).
        """
        from_count = transition_from.count("1")
        to_count = transition_to.count("1")

        if to_count > from_count and from_count >= 1:
            if self.regime == "processive":
                return 2.5
            elif self.regime == "dissociative":
                return 0.4
        return 1.0

    def _build_rate_matrix(self) -> np.ndarray:
        """
        Constructs the 27x27 mass-action transition rate matrix across all single-step transitions.
        Alanine ('2') mutations zero out forward phosphorylation fluxes at mutated sites.

        Args:
            None

        Returns:
            np.ndarray: A 27x27 floating-point transition rate coefficient matrix ($K$).
        """
        n_states = len(self.state_codes)
        K = np.zeros((n_states, n_states), dtype=np.float64)

        # Use canonical residue numbers directly as site identifiers
        target_residues = ["655", "659", "663"]

        for src_idx, src_code in enumerate(self.state_codes):
            for site_idx, res_num in enumerate(target_residues):
                src_char = src_code[site_idx]

                # Forward Phosphorylation Transition ('0' -> '1')
                if src_char == "0":
                    tgt_code = src_code[:site_idx] + "1" + src_code[site_idx + 1 :]
                    tgt_idx = self.state_to_idx[tgt_code]

                    m_rsasa = self._compute_rate_multiplier(tgt_code, res_num)
                    sc_regime = self._get_regime_scaler(src_code, tgt_code)

                    k_fwd = self.site_base_rates[site_idx] * m_rsasa * sc_regime
                    K[tgt_idx, src_idx] += k_fwd
                    K[src_idx, src_idx] -= k_fwd

                # Reverse Dephosphorylation Transition ('1' -> '0')
                elif src_char == "1":
                    tgt_code = src_code[:site_idx] + "0" + src_code[site_idx + 1 :]
                    tgt_idx = self.state_to_idx[tgt_code]

                    k_rev = self.dephospho_rates[site_idx]
                    K[tgt_idx, src_idx] += k_rev
                    K[src_idx, src_idx] -= k_rev

        return K

    def _ode_system(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Defines the mass-action ODE matrix multiplication: dy/dt = K * y.

        Args:
            t (float): Current time step in solver integration.
            y (np.ndarray): Current concentration vector across the 27 ternary states.

        Returns:
            np.ndarray: Time derivatives vector (dy/dt) for each of the 27 states.
        """
        K = self._build_rate_matrix()
        return K @ y

    def simulate(
        self,
        t_span: Tuple[float, float] = (0, 100),
        y0: Optional[np.ndarray] = None,
        initial_state: str = "000",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the ODE time-course simulation across the 27-state grid.

        Args:
            t_span (Tuple[float, float], optional): Integration time window (t_start, t_end). Defaults to (0, 100).
            y0 (Optional[np.ndarray], optional): Initial 27-element concentration vector. Defaults to None.
            initial_state (str, optional): Starting ternary state code if y0 is None (e.g., '000' or '020'). Defaults to "000".

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - time_points (np.ndarray): 1D array of evaluation time points.
                - trajectories (np.ndarray): 27xN array of state concentration trajectories over time.
        """
        n_states = len(self.state_codes)
        if y0 is None:
            y0 = np.zeros(n_states, dtype=np.float64)
            start_idx = self.state_to_idx.get(initial_state, 0)
            y0[start_idx] = 1.0

        solution = solve_ivp(
            fun=self._ode_system,
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(t_span[0], t_span[1], 500),
            method="RK45",
        )
        return solution.t, solution.y
