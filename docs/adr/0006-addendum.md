# Addendum to ADR-0006: Structural Sourcing vs. Kinetic Parameterization

* **Status:** Accepted
* **Date:** 2026-10-24
* **Deciders:** Bioinformatics & ML Engineering
* **Applies To:** `ADR-0006` (Strict Data Partitioning and Validation Protocol)

---

## 1. Context & Motivation

A key decision in `ADR-0006` was the partitioning of the **`6N8` series** (unliganded human Lgl2) as the Tier 1 structural development set and the **`8r3y` complex structure** (tripartite aPKC–Par6–Lgl complex) alongside the **`9EJ` series** (human Lgl homologues/alternate crystal forms) as a Tier 2 independent evaluation set.

However, structural inspection reveals that intrinsically disordered C-terminal regulatory loops containing the target phosphorylation sites ($S_{655}$, $S_{659}$, $S_{663}$) lack resolved electron density across both the unliganded `6N8` series structures and the alternate `9EJ` series (e.g., `9EJK`, `9EJL`, `9EJM`). Conversely, `8r3y` captures these target regulatory loops in a catalytically engaged, stabilized conformation, providing the only experimentally resolved spatial coordinates ($C_\alpha$ backbone and local environment) for these sites.

Because `8r3y` provides structural coordinate topology rather than numerical kinetic constants ($k_{cat}$ or $K_m$), using its spatial coordinates does not constitute kinetic data leakage. To reconcile physical coordinate requirements with data partitioning hygiene, this addendum formalizes the operational distinction between **Structural Coordinate Anchoring** and **Kinetic Parameterization**.

---

## 2. Refined Sourcing Specification

### A. Structural Coordinate Anchoring (Spatial Geometry Sourcing)
* **Primary Coordinate Template:** **`8r3y` MMCIF Structure**
* **Role:** Serves as the primary 3D spatial template for C-alpha coordinates and local residue environments across the 8-state hypercube ($S_{000}$–$S_{111}$).
* **Scope:** Sourcing atomic coordinates for feature extraction (e.g., node $C_\alpha$ positions, contact topologies, initial local SASA boundaries).

### B. Kinetic Parameterization (Rate Constant Sourcing)
* **Primary Assay Datasets:** **Graybill & Prehoda (2014)** and **Graybill et al. (2012)**
* **Role:** Parameterization of all mass-action ODE transition rate matrices, $k_{cat}$ values, $K_m$ parameters, sequence entry hierarchies, and structural sensitivity scalars ($\alpha$).
* **Scope:** Used exclusively for ODE model calibration and parameter optimization loops.

### C. Mechanistic & Blind Validation Targets
* **Evaluation Framework:** **`8r3y` Trajectory & Structural Signature Validation**
* **Role:** Blind validation target for qualitative signatures and mechanistic predictions (e.g., processive sweep vs. dissociative stepping, intermediate population densities, cooperativity indices).
* **Rule:** No ODE rate constant or parameter optimization algorithm shall use `8r3y` structural parameters as a direct loss function or tuning target during fitting.

---

## 3. Explicit Model Assumptions

1. **Conformational Template Invariance:**
   The initial C-alpha backbone positions of the phosphorylation sites derived from `8r3y` are assumed to represent a valid spatial baseline for all 8 microstates ($S_{000}$ through $S_{111}$), with state-specific differentiation driven by dynamic local feature recalculation (e.g., MaxASA normalization, rSASA gating, and side-chain property shifts).

2. **Independence of Modalities:**
   Structural coordinate features derived from x-ray/cryo-EM maps and solution-phase biochemical reaction rates measured in vitro represent orthogonal physical modalities. Parameterizing rate equations on solution kinetics while using structural templates for spatial graphs does not introduce circular curve-fitting, provided rate parameters are not tuned against structural observables.

3. **Fallback Anchor Continuity:**
   Where density remains partially unmodeled across alternate target state variations or auxiliary models (such as the `6N8` or `9EJ` series), the pipeline will continue to enforce C-terminal boundary anchor fallbacks per `ADR-0005` with explicit `UNMODELED` metadata flags.

---

## 4. Technical Limitations & Mitigation Strategies

| Limitation | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Trapped State Conformational Bias** | `8r3y` represents a static, locked tripartite complex. Using its loop coordinates across all 8 states may underrepresent conformational flexibility in uncomplexed states. | Perform dynamic per-state rSASA recalculations (e.g., Tien et al. MaxASA normalization) and compute ANM Mean-Square Fluctuations (MSF) to inject physical dynamics onto the static scaffold. |
| **Unmodeled Regulatory Loops in Isolated Substrate Series** | Neither the `6N8` series nor the `9EJ` series provide resolved $C_\alpha$ positions for $S_{655}$, $S_{659}$, and $S_{663}$ due to intrinsic tail disorder. | Retain `6N8` and `9EJ` series as rigid WD40 scaffold references and boundary anchors (`ADR-0005`), reserving full $C_\alpha$ loop placement to the `8r3y` template. |
| **Risk of Unintentional Data Leakage** | Structural features could implicitly bias kinetic parameters if interface distance metrics are directly optimized during model selection. | Enforce strict separation in codebase: optimization scripts (`run_kinetics.nf`, parameter search loops) must accept kinetic assay loss functions only. |

---

## 5. Structural & Kinetic Literature References

1. **`8r3y` / `8r3x` Complex:**
   * Earl, C.P., Cobbaut, M., Barros-Carvalho, A., Ivanova, M.E., Briggs, D.C., Morais-de-Sa, E., Parker, P.J., McDonald, N.Q. (2025). *Cryo-EM structure of a stable LGL/aPKC Iota/Par-6 complex / Capture, mutual inhibition and release mechanism for aPKC-Par6 and its multisite polarity substrate Lgl*. **Nat. Struct. Mol. Biol.** 32, 729–739. DOI: [10.1038/s41594-024-01425-0](https://doi.org/10.1038/s41594-024-01425-0)
2. **`9EJ` Series (`9EJK`, `9EJL`, `9EJM`):**
   * Almagor, L., Weis, W.I. (2025). *Polarity protein Par6 facilitates the processive phosphorylation of Lgl via a dynamic interaction with aPKC*. **Commun. Biol.** 8, 967. DOI: [10.1038/s42003-025-08401-4](https://doi.org/10.1038/s42003-025-08401-4)
3. **`6N8` Series (`6N8P`, `6N8Q`, `6N8R`, `6N8S`):**
   * Human Lgl2 WD40 structural ensemble (unliganded/isolated substrate state).
4. **Kinetic Baseline Data:**
   * Graybill, C., & Prehoda, K.E. (2014). *Ordered multisite phosphorylation distributes the time scales of a cellular switch*. **Biochemistry**, 53(46), 7240–7247.
   * Graybill, C., et al. (2012). *A structural basis for the phosphorylation-independent interaction between aPKC and Lgl / Par-6 activation baseline*.

---

## 6. Verification Checklist

- [x] Confirm `8r3y` is used strictly as a structural input template for graph construction (`build_state_graph.py`).
- [x] Verify that rate fitting algorithms in `run_kinetics.nf` consume only solution assay data (Graybill et al.).
- [x] Document that both `6N8` and `9EJ` series are preserved as auxiliary/scaffold references due to missing loop density.
- [x] Ensure unit tests in `tests/test_builder.py` and pipeline checks verify variant-specific rSASA feature divergence under the dynamic normalization engine.
