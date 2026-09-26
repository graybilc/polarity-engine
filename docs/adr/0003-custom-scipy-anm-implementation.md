# ADR 0003: Custom SciPy Elastic Network Model (ANM) Implementation for Physical Dynamics

* **Status**: Accepted

---

## 1. Context & Problem Statement

In **Module 1 (Structure Graph Engine)**, incorporating low-frequency normal modes from an Anisotropic Network Model (ANM) provides critical physical dynamics features (e.g., thermal fluctuations, collective domain motions, and local conformational mobility) for each residue in the macromolecular graph.

While established third-party libraries such as **ProDy** provide out-of-the-box ANM calculations, relying on external black-box packages presents long-term maintainability and architectural constraints:

1. **Customizability & Extensibility:** Fine-tuning Hessian construction, distance-cutoff functions, custom spring constants, or non-standard boundary conditions requires direct control over the underlying matrix operations.
2. **Dependency Management:** ProDy introduces substantial external dependency overhead, object hierarchies, and C-extension dependencies that complicate lightweight container builds and multi-language pipelines.
3. **Integration with Internal Data Structures:** A native SciPy implementation allows direct interaction with $C_\alpha$ coordinate arrays extracted straight from our internal `MMCIF2Dict` parser without intermediate file I/O or BioPython/ProDy object creation.

---

## 2. Decision Drivers

* **Algorithmic Control:** Retain absolute programmatic control over Hessian matrix assembly, sparse/dense eigensolvers, and custom spring-constant scaling functions for future research extensions.
* **Minimal Dependency Footprint:** Avoid heavy external dependencies by leveraging existing baseline scientific Python libraries (`numpy` and `scipy`).
* **Graph Feature Compatibility:** Seamlessly output residue-level physical dynamics vectors (e.g., mean-square fluctuations $\langle \Delta r_i^2 \rangle$ and top $k$ slow mode shape vectors) directly into the node feature tensor $X \in \mathbb{R}^{N \times 26}$.
* **Performance Efficiency:** Ensure fast Hessian generation and eigen-decomposition for standard macromolecular structures.

---

## 3. Implementation Details

We implemented a native, lightweight ANM class using `numpy` for spatial distance/Hessian construction and `scipy.linalg` (or `scipy.sparse.linalg` for large systems) for symmetric eigen-decomposition.

### 3.1 Mathematical Formulation
For a protein with $N$ residues represented by $C_\alpha$ coordinates $\mathbf{R} \in \mathbb{R}^{N \times 3}$:

1. **Distance-Based Kirchhoff / Hessian Matrix Construction:**
   The $3N \times 3N$ Hessian matrix $\mathbf{H}$ composed of $3 \times 3$ sub-blocks $\mathbf{H}_{ij}$ for residue pairs $(i, j)$ separated by distance $R_{ij}$:
   $$\mathbf{H}_{ij} = -\frac{\gamma}{R_{ij}^2} (\mathbf{R}_{ij} \otimes \mathbf{R}_{ij}) \quad \text{for } i \neq j, \, R_{ij} \le R_c$$
   $$\mathbf{H}_{ii} = -\sum_{j \neq i} \mathbf{H}_{ij}$$
   Where $R_c = 15.0\text{ \AA}$ is the interaction cutoff distance and $\gamma = 1.0$ is the uniform spring constant.

2. **Eigensolution & Fluctuations:**
   Diagonalizing $\mathbf{H}$ yields eigenvalues $\lambda_k$ and eigenvectors $\mathbf{v}_k$. The dynamic fluctuation feature vector is derived using the non-zero low-frequency modes ($k \in [7, M+6]$):
   $$\langle \Delta r_i^2 \rangle = \sum_{k=7}^{M+6} \frac{1}{\lambda_k} \left( v_{k, 3i}^2 + v_{k, 3i+1}^2 + v_{k, 3i+2}^2 \right)$$

---

## 4. Architectural Decision

**Verdict: Adopt a custom SciPy-based ANM module in place of external dependencies like ProDy.**

### 4.1 Comparison Summary
| Consideration | ProDy Dependency | Custom SciPy Implementation (Selected) |
| :--- | :--- | :--- |
| **Customizability** | Low (constrained by internal ProDy abstractions) | **High** (full control over Hessian, cutoffs, and constants) |
| **Dependencies** | Requires `prody` + C-extensions | **Zero extra** (uses core stack: `numpy`, `scipy`) |
| **Pipeline Integration** | Requires converting arrays to ProDy `AtomGroup` | **Direct** NumPy array inputs from `MMCIF2Dict` |
| **Maintenance** | Vulnerable to upstream API deprecation/breaks | **In-house** code maintenance within `StructureParser` |

---

## 5. Consequences

### Positive
* **Future-Proof Flexibility:** Full freedom to experiment with non-linear spring constants, distance-dependent decay functions ($\gamma(r) \propto r^{-2}$), dynamic cutoff thresholds, or heterodimer interface weights.
* **Clean Architecture:** Operates cleanly on native NumPy arrays without object translation layers or intermediate file writes.
* **Reduced Dependency Risk:** Reduces docker container size and eliminates potential version conflicts from external protein-structure toolkits.

### Negative / Trade-offs
* **Maintenance Ownership:** Team assumes responsibility for testing, edge-case handling (e.g., disconnected graph components), and optimizing eigensolvers for extremely large multi-chain assemblies ($N > 5,000$).
* **Feature Scope:** Standard utilities provided out-of-the-box by ProDy (e.g., automated PDB fetching, ensemble alignment) must be handled separately if needed elsewhere in the pipeline.
