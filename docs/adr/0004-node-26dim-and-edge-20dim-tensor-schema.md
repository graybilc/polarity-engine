# ADR 0004: Node (26-dim) and Edge (20-dim) Tensor Schema Specification for Module 1 Output

* **Status**: Accepted

---

## 1. Context & Problem Statement

Module 1 (Structure Graph Engine) processes macromolecular structures (`.cif` / `.pdb`) to construct spatial protein graphs. To ensure a clean, deterministic handshake with downstream PyTorch Geometric (PyG) Graph Neural Network (GNN) modules, Module 1 must produce standardized, fixed-dimensional node and edge feature tensors.

Prior iterations lacked a rigid schema definition, leading to dimension mismatches across feature extractors (e.g., handling missing physical parameters or varying sequence encodings). A finalized schema contract is required to lock down the exact tensor layout for node feature tensor $\mathbf{X} \in \mathbb{R}^{N \times 26}$ and edge attribute tensor $\mathbf{E} \in \mathbb{R}^{E \times 20}$.

---

## 2. Decision Drivers

* **Expressiveness:** Capture sequence identity, spatial exposure, structural geometry, and physical dynamic properties within a concise vector representation.
* **Deterministic Contract:** Standardize array shapes, indexing, and normalization ranges to ensure downstream model layers process inputs without shape validation errors.
* **Compatibility:** Align directly with PyTorch Geometric data pipeline requirements (`torch_geometric.data.Data`).
* **Efficiency:** Keep feature dimensionality compact to minimize GPU memory footprint during batched graph forward passes.

---

## 3. Tensor Schema Specification

### 3.1 Node Feature Tensor ($\mathbf{X} \in \mathbb{R}^{N \times 26}$)

Each residue node $i \in \{1, \dots, N\}$ is encoded into a 26-dimensional dense vector partitioned into four functional channels:

| Index Range | Feature Description | Encoding / Scale | Source / Reference |
| :---: | :--- | :--- | :--- |
| **`[0:20]`** | **One-Hot Amino Acid Identity** | Binary vector ($1$ at AA index, $0$ elsewhere; standard 20 canonical AAs) | `StructureParser` |
| **`[20]`** | **Relative SASA ($\text{rSASA}$)** | Continuous scalar $\in [0, 1]$ (All-atom SASA / MaxASA) | ADR 0002 (FreeSASA) |
| **`[21:23]`** | **Backbone Dihedrals ($\sin\phi, \cos\phi$)** | Continuous scalars $\in [-1, 1]$ | $\phi$ Angle Geometry |
| **`[23:25]`** | **Backbone Dihedrals ($\sin\psi, \cos\psi$)** | Continuous scalars $\in [-1, 1]$ | $\psi$ Angle Geometry |
| **`[25]`** | **ANM Thermal Fluctuation ($\langle \Delta r_i^2 \rangle$)** | Continuous scalar $\in [0, \infty)$ ($C_\alpha$-only ANM mean-square fluctuation) | ADR 0003 (SciPy ANM) |

*Note: For terminal or unresolvable residues where $\phi$ or $\psi$ cannot be calculated, missing angle components default to $0.0$.*

---

### 3.2 Edge Feature Tensor ($\mathbf{E} \in \mathbb{R}^{E \times 20}$)

For each directed edge $e = (u, v)$ connecting residue $u$ to residue $v$ within spatial cutoff $R_c = 8.0\text{ \AA}$ (based on $C_\alpha$-$C_\alpha$ Euclidean distance $d_{uv}$):

| Index Range | Feature Description | Encoding / Scale |
| :---: | :--- | :--- |
| **`[0:16]`** | **Distance Radial Basis Functions (RBF)** | $16$ Gaussian expansion channels centered from $0.0\text{ \AA}$ to $10.0\text{ \AA}$ ($\gamma = 0.5$) |
| **`[16:19]`** | **Normalized Direction Vector** | Unit vector $\frac{\mathbf{r}_v - \mathbf{r}_u}{\Vert{}\mathbf{r}_v - \mathbf{r}_u\Vert{}_2} \in \mathbb{R}^3$ |
| **`[19]`** | **Sequence Distance / Separation** | Scaled sequential offset $\text{clip}\left(\frac{v_{\text{seq}} - u_{\text{seq}}}{32}, -1.0, 1.0\right) \in [-1, 1]$ |

---

## 4. Architectural Decision

**Verdict: Standardize and lock Module 1 output data contract to `X` (26-dim node features) and `E` (20-dim edge features).**

The output object from Module 1 will adhere to standard PyTorch Geometric data format:

```python
from torch_geometric.data import Data

graph_data = Data(
    x=x_tensor,            # Shape: [N, 26] (float32)
    edge_index=edge_index, # Shape: [2, E] (int64)
    edge_attr=edge_attr,   # Shape: [E, 20] (float32)
    pos=pos_tensor,        # Shape: [N, 3]  (C_alpha coordinates)
)
