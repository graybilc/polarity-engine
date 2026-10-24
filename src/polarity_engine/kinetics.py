"""
src/polarity_engine/kinetics.py

Mass-action ODE solver for evaluating Processive vs. Dissociative
multi-site phosphorylation regimes of Lgl by aPKC-Par6.

Incorporates ADR-0006 compliant Tier 1 parameterization derived exclusively
from historical bulk assays (Graybill & Prehoda 2014; Graybill et al. 2012).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional


class PhosphoKineticSolver:
    """
    Simulates concentration trajectories across an 8-state mass-action ODE hypercube
    (S000 to S111) for multi-site phosphorylation of Lgl by aPKC-Par6.

    Attributes:
        graphs (Dict[str, Any]): Dictionary mapping state codes to PyG structural graphs.
        site_mappings (Dict[str, Any]): Mapping dictionary for abstract functional residue slots.
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
        Initializes the PhosphoKineticSolver with structural graphs, site mappings,
        and regime-specific parameters.

        Args:
            graphs (Dict[str, Any]): Structural PyG graph representations for each state.
            site_mappings (Dict[str, Any]): Abstract functional site slot definitions.
            alpha (float, optional): Sensitivity multiplier for rSASA gating. Defaults to 1.2.
            par6_active (bool, optional): Whether aPKC-Par6 is active (applies catalytic multiplier). Defaults to True.
            regime (str, optional): Operating kinetic regime ('processive' or 'dissociative'). Defaults to "processive".

        Raises:
            ValueError: If the regime argument is not 'processive' or 'dissociative'.
        """
        self.graphs = graphs
        self.site_mappings = site_mappings
        self.alpha = alpha
        self.par6_active = par6_active  # Fixed: explicitly assigned instance attribute
        self.regime = regime.lower()

        if self.regime not in ["processive", "dissociative"]:
            raise ValueError("Regime must be either 'processive' or 'dissociative'.")

        # --- TIER 1 PARAMETERIZATION SET (ADR-0006 Compliant) ---
        # Provenance: Fitted exclusively from historical bulk assay data in
        # Graybill & Prehoda (2014) and Graybill et al. (2012).
        # Secluded entirely from Parker & McDonald (2025) test datasets.

        k_cat_multiplier = 8.5 if self.par6_active else 1.0

        self.base_rates = {
            # Initial priming entry rates from unphosphorylated state (S000)
            "k_000_100": 0.04 * k_cat_multiplier,  # Site 1 secondary entry
            "k_000_010": 0.07 * k_cat_multiplier,  # Site 2 tertiary entry
            "k_000_001": 0.22 * k_cat_multiplier,  # Site 3 primary preferential entry
            # Downstream progression rates across intermediate states
            "k_100_110": 0.15 * k_cat_multiplier,
            "k_001_101": 0.20 * k_cat_multiplier,
            "k_010_011": 0.18 * k_cat_multiplier,  # Explicitly added branch path rate
            "k_110_111": 0.25 * k_cat_multiplier,
            "k_101_111": 0.28 * k_cat_multiplier,
            "k_011_111": 0.30 * k_cat_multiplier,
            # Baseline dephosphorylation rates (estimated background phosphatase activity)
            "k_100_000": 0.02,
            "k_010_000": 0.03,
            "k_001_000": 0.01,
        }

    def _compute_rate_multiplier(
        self, state_code: str, functional_site_id: str
    ) -> float:
        """
        Extracts relative Solvent Accessible Surface Area (rSASA) from PyG graph node
        features and computes a structural gating multiplier. Higher rSASA (surface exposure)
        enhances the phosphorylation rate.

        Args:
            state_code (str): The state identifier string (e.g., '000').
            functional_site_id (str): The abstract site identifier (e.g., 'site_1').

        Returns:
            float: The rate multiplier scaling factor (e.g., exp(alpha * rSASA)).
                   Returns 1.0 if the graph or site mapping is missing.
        """
        graph = self.graphs.get(state_code)
        if graph is None or functional_site_id not in self.site_mappings:
            return 1.0

        res_idx = self.site_mappings[functional_site_id]["res_num"]
        # Node feature index 22: rSASA (0.0 to 1.0)
        r_sasa = graph.x[res_idx, 22].item()

        # Positive exponent: higher surface exposure accelerates phosphorylation
        return np.exp(self.alpha * r_sasa)

    def _get_regime_scaler(self, transition_from: str, transition_to: str) -> float:
        """
        Applies topological scaling based on the change in phosphorylation count
        between adjacent states in the hypercube.

        Args:
            transition_from (str): Binary state code of the originating state (e.g., '100').
            transition_to (str): Binary state code of the destination state (e.g., '110').

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

    def _ode_system(self, t: float, y: np.ndarray) -> list:
        """
        Defines the mass-action ordinary differential equations across the 8-state hypercube.

        Args:
            t (float): Current time step in the solver integration.
            y (np.ndarray): Current concentration vector for the 8 states
                            [s000, s100, s010, s001, s110, s101, s011, s111].

        Returns:
            list: Time derivatives (ds/dt) for each of the 8 hypercube states.
        """
        s000, s100, s010, s001, s110, s101, s011, s111 = y

        # --- 1. rSASA Gating Multipliers (State-Specific Source Queries) ---
        # Initial entry from S000
        m_000_100 = self._compute_rate_multiplier("000", "site_1")
        m_000_010 = self._compute_rate_multiplier("000", "site_2")
        m_000_001 = self._compute_rate_multiplier("000", "site_3")

        # Intermediate progression (querying the originating state's graph for the newly added site)
        m_100_110 = self._compute_rate_multiplier(
            "100", "site_2"
        )  # S100 -> S110 adds site 2
        m_001_101 = self._compute_rate_multiplier(
            "001", "site_1"
        )  # S001 -> S101 adds site 1
        m_010_011 = self._compute_rate_multiplier(
            "010", "site_3"
        )  # S010 -> S011 adds site 3

        # Final sink step to S111
        m_110_111 = self._compute_rate_multiplier(
            "110", "site_3"
        )  # S110 -> S111 adds site 3
        m_101_111 = self._compute_rate_multiplier(
            "101", "site_2"
        )  # S101 -> S111 adds site 2
        m_011_111 = self._compute_rate_multiplier(
            "011", "site_1"
        )  # S011 -> S111 adds site 1

        # --- 2. Regime-Dependent Topological Scalers ---
        sc_110 = self._get_regime_scaler("100", "110")
        sc_101 = self._get_regime_scaler("001", "101")
        sc_011 = self._get_regime_scaler("010", "011")
        sc_111 = self._get_regime_scaler("x", "111")

        # --- 3. Effective Rates (Combining Base Rates, rSASA, and Regime Scalars) ---
        k_0_100 = self.base_rates["k_000_100"] * m_000_100
        k_0_010 = self.base_rates["k_000_010"] * m_000_010
        k_0_001 = self.base_rates["k_000_001"] * m_000_001

        k_100_110 = self.base_rates["k_100_110"] * m_100_110 * sc_110
        k_001_101 = self.base_rates["k_001_101"] * m_001_101 * sc_101
        k_010_011 = self.base_rates["k_010_011"] * m_010_011 * sc_011

        k_110_111 = self.base_rates["k_110_111"] * m_110_111 * sc_111
        k_101_111 = self.base_rates["k_101_111"] * m_101_111 * sc_111
        k_011_111 = self.base_rates["k_011_111"] * m_011_111 * sc_111

        # --- 4. Mass-Action Differential Equations ---
        ds000_dt = -(k_0_100 + k_0_010 + k_0_001) * s000 + (
            self.base_rates["k_100_000"] * s100
            + self.base_rates["k_010_000"] * s010
            + self.base_rates["k_001_000"] * s001
        )

        ds100_dt = (k_0_100 * s000) - (self.base_rates["k_100_000"] + k_100_110) * s100
        ds010_dt = (k_0_010 * s000) - (self.base_rates["k_010_000"] + k_010_011) * s010
        ds001_dt = (k_0_001 * s000) - (self.base_rates["k_001_000"] + k_001_101) * s001

        ds110_dt = (k_100_110 * s100) - (k_110_111 * s110)
        ds101_dt = (k_001_101 * s001) - (k_101_111 * s101)
        ds011_dt = (k_010_011 * s010) - (k_011_111 * s011)

        ds111_dt = (k_110_111 * s110) + (k_101_111 * s101) + (k_011_111 * s011)

        return [
            ds000_dt,
            ds100_dt,
            ds010_dt,
            ds001_dt,
            ds110_dt,
            ds101_dt,
            ds011_dt,
            ds111_dt,
        ]

    def simulate(
        self, t_span: Tuple[float, float] = (0, 100), y0: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the ODE time-course simulation using SciPy's solve_ivp.

        Args:
            t_span (Tuple[float, float], optional): Integration time window (t_start, t_end). Defaults to (0, 100).
            y0 (Optional[list], optional): Initial state concentrations across the 8 states.
                                            Defaults to fully unphosphorylated [1.0, 0, 0, 0, 0, 0, 0, 0].

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - t evaluation time points array.
                - y solution array of shape (8, len(t)) containing concentration trajectories.
        """
        if y0 is None:
            # Start fully unphosphorylated (S000)
            y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        solution = solve_ivp(
            fun=self._ode_system,
            t_span=t_span,
            y0=y0,
            t_eval=np.linspace(t_span[0], t_span[1], 500),
            method="RK45",
        )
        return solution.t, solution.y
