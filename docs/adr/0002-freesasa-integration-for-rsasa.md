# ADR 0002: FreeSASA Integration for All-Atom Relative Solvent Accessible Surface Area ($\text{rSASA}$)

* **Status**: Accepted

## 1. Context & Problem Statement

In **Module 1 (Structure Graph Engine)**, relative Solvent Accessible Surface Area ($\text{rSASA}$) serves as a core spatial node feature ($\text{rSASA} \in [0, 1]$), capturing steric occlusion and solvent exposure for each residue within a protein macromolecular graph.

To compute $\text{rSASA}$ using FreeSASA's C-bindings (`freesasa.calcCoord`), we needed to determine the optimal coordinate extraction and surface calculation strategy when operating directly on `MMCIF2Dict` data structures:

1. **Option A (All-Atom SASA):** Extract all heavy atoms ($M \approx 10,000$), compute atomic Van der Waals (VdW) surface accessibility using element-specific radii, and aggregate atomic SASA values to the residue level.
2. **Option B ($C_\alpha$-Only SASA):** Extract only backbone $C_\alpha$ atoms ($N \approx 1,290$) and calculate coarse-grained surface accessibility using a uniform spherical radius ($R_{C_\alpha} = 3.5\text{ \AA}$).

---

## 2. Decision Drivers

* **Physical & Biological Accuracy:** Accurately represent solvent exposure for residues with bulky or extended sidechains (e.g., Trp, Phe, Lys, Arg).
* **Feature Utility:** Minimize uncaptured variance ($\text{rSASA}$) when constructing the 26-dimensional residue node feature tensor ($X \in \mathbb{R}^{N \times 26}$).
* **Performance & Latency:** Ensure execution latency and memory usage remain acceptable for large multi-chain macromolecular structures (e.g., benchmark structure `8r3y.cif`).
* **I/O Overhead:** Keep calculations fully in-memory via `MMCIF2Dict` arrays and `freesasa.calcCoord`, avoiding file-system I/O and BioPython object hierarchy overhead.

---

## 3. Experimental Setup & Benchmark Results

We conducted an empirical benchmark on the Cryo-EM structure `8r3y.cif`. Execution latency, peak memory allocation (measured via `tracemalloc`), and statistical correlation ($r$ and $\rho$) were evaluated across 1,290 aligned residues.

### Benchmark Summary Table
| Metric | Option A (All-Atom) | Option B ($C_\alpha$-Only) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Execution Latency** | **9,166.62 ms** | 4,013.37 ms | +5.15 s (all heavy atoms) |
| **Peak Memory (RAM)** | **25.79 MB** | 17.99 MB | +7.80 MB |
| **Pearson Correlation ($r$)** | — | — | **0.8002** |
| **Spearman Correlation ($\rho$)** | — | — | **0.8109** |

---

## 4. Key Findings & Analysis

1. **Uncaptured Variance (~36%):**
   A Pearson correlation of $r = 0.8002$ indicates that $C_\alpha$-only coarse-graining accounts for only $r^2 = 64.0\%$ of the variance in true surface accessibility. The remaining **35.98% uncaptured variance** introduces systematic errors for buried backbones with solvent-exposed sidechains.
2. **Rank Misalignment ($\rho = 0.8109$):**
   The Spearman rank correlation reveals that $C_\alpha$-only SASA shifts the relative exposure ranking across protein interfaces. This risks misclassifying functionally critical buried vs. exposed sites.
3. **Manageable Computational Overhead:**
   Although Option A adds ~5.15 seconds for a large multi-chain complex (`8r3y.cif`), the total runtime (~9.17 seconds) and peak memory (~25.8 MB) are well within acceptable bounds for offline graph construction pipelines.

---

## 5. Architectural Decision

**Verdict: Adopt Option A (All-Atom SASA Aggregated Per Residue).**

We will implement Option A in `StructureParser` using the following parameters and workflow:

## 6. Consequences

### Positive
* Higher physical accuracy for graph neural network (GNN) representations, especially at binding interfaces and catalytic pockets.
* Direct array operations via `freesasa.calcCoord` avoid disk I/O and object creation overheads.
* Standardized normalization via established empirical MaxASA scales (Tien et al., 2013).

### Negative / Trade-offs
* Increases extraction + computation time per structure (approx. +5 seconds on 10k-atom complexes).
* Slightly higher peak RAM usage (+7.8 MB for `8r3y.cif`).
