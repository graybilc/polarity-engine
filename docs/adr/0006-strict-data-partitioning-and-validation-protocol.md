# ADR-0006: Strict Data Partitioning, Validation Protocol, and Multi-Version Governance

* **Status:** Accepted (Corrected)
* **Context:** In investigating whether aPKC-Par6 phosphorylates Lgl via a processive multi-site sweep versus a dissociative step-by-step mechanism, our computational pipeline combines static 3D PyG structural graphs with 8-state mass-action ODE kinetics. A critical risk in this architecture is circular reasoning (data leakage)—specifically, using structural templates or kinetic assays for model tuning that are later used to "validate" predicted outcomes. Furthermore, anticipating future model expansions (V2.a Entropy, V2.b Joint Probability, V2.c Cooperativity Index, and V3 GNN Surrogates) requires a governance framework that avoids unconstrained multi-model "horse races" on a single test set.
* **Decision:** We establish a mandatory, strict data partitioning protocol separating baseline parameterization from blind structural-kinetic validation. No final test data shall be accessed during baseline rate fitting or structural multiplier optimization. Additionally, future model iterations will be evaluated via nested hierarchical extensions and penalization criteria rather than raw trajectory minimization.

---

## Data Partitioning Specification

### Tier 1: Parameterization & Validation (Development Set)
* **Assigned Data Sources:**
  * *Graybill & Prehoda (2014)* (Ordered multisite phosphorylation efficiency metrics).
  * *Graybill et al. (2012)* (Par-6 autoinhibition relief & activation baseline).
  * *ACS "Ordered Multisite" Study*.
  * **`6N8` Series (Human Lgl2 Unphosphorylated/Phosphorylated):** Assigned to anchor the structural conformational landscape of the 8-state hypercube ($S_{000}$–$S_{111}$) using native, experimentally resolved states, providing direct rSASA features without loop-grafting artifacts.
* **Purpose:** Establish baseline rate constants ($k_{cat}$, $K_m$), sequence hierarchies, and structural sensitivity scalars ($\alpha$) via cross-validation or grid search on historical data splits.

### Tier 2: Independent Blind Evaluation (Test Set)
* **Assigned Data Sources:**
  * **`8r3y` Complex Structure & Parker & McDonald (2025)** (*Nat Struct Mol Biol* 32) (Tripartite capture state, catalytic interface, and stalled intermediate mechanics).
  * *9EJ Series* (Human Lgl homologues and alternative crystal conformations).
* **Purpose:** **Strictly sequestered during model development.** Once the ODE transition matrices and feature extractors are locked, these datasets are fed *blindly* into the pipeline to test whether the model correctly predicts complex docking energetics, interfacial cooperativity, and kinetic regime shifts without prior exposure.

---

## Multi-Version Model Comparison Strategy (V1 to V3 Roadmap)
To prevent overfitting multiple theoretical architectures to a single test set:
* **Hierarchical Nesting:** Advanced model iterations will be implemented as mathematical extensions of the base mass-action ODE framework rather than completely disconnected models.
* **Information Criteria:** Model selection will rely on goodness-of-fit penalized by parameter complexity (e.g., AIC/BIC) to enforce Occam's razor.
* **Qualitative Signatures:** Success will be evaluated by a model's capacity to capture core biochemical signatures (such as site-entry hierarchies and information-theoretic cooperativity indices) across conformational variations rather than raw numerical curve-fitting alone.

---

## Consequences & Verification
* **Positive:** Eliminates data leakage and prevents circular validation by properly segregating substrate-only structural tuning (Tier 1) from complex-interface and mutational evaluation (Tier 2). Ensures that any agreement between simulated trajectories and Parker & McDonald (2025) mechanics represents genuine predictive discovery.
* **Negative / Trade-off:** Requires strict pipeline hygiene to ensure `8r3y` and the `9EJ` series remain isolated from training scripts and parameter search loops.
