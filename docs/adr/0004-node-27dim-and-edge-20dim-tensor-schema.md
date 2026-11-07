# ADR 0004: Node (27-dim) and Edge (20-dim) Tensor Schema Specification for Module 1 Output

* **Status**: Accepted

---

## 1. Context & Problem Statement

Module 1 (Structure Graph Engine) processes macromolecular structures (`.cif` / `.pdb`) to construct spatial protein graphs. To ensure a clean, deterministic handshake with downstream PyTorch Geometric (PyG) Graph Neural Network (GNN) modules, Module 1 must produce standardized, fixed-dimensional node and edge feature tensors.

A finalized schema contract is required to lock down the exact tensor layout for node feature tensor $\mathbf{X} \in \mathbb{R}^{N \times 27}$ and edge attribute tensor $\mathbf{E} \in \mathbb{R}^{E \times 20}$.

---

## 2. Decision Drivers

* **Expressiveness:** Capture sequence identity, sequential position, spatial exposure, crystallographic parameters, and physical dynamic properties within a concise vector representation.
* **Deterministic Contract:** Standardize array shapes, indexing, and normalization ranges to ensure downstream model layers process inputs without shape validation errors.
* **Compatibility:** Align directly with PyTorch Geometric data pipeline requirements (`torch_geometric.data.Data`).
* **Efficiency:** Keep feature dimensionality compact to minimize GPU memory footprint during batched graph forward passes.

---

## 3. Tensor Schema Specification

### 3.1 Node Feature Tensor ($\mathbf{X} \in \mathbb{R}^{N \times 27}$)

Each residue node $i \in \{1, \dots, N\}$ is encoded into a 27-dimensional dense vector partitioned into sequence encoding and biophysical feature channels:

| Index Range | Feature Description | Encoding / Scale | Source / Reference |
| :---: | :--- | :--- | :--- |
| **`[0:22]`** | **One-Hot Amino Acid Identity** | Binary vector ($1$ at AA index, $0$ elsewhere; 22 classes including canonical AAs, UNK, and SEP) | `AMINO_ACID_TO_INDEX` |
| **`[22]`** | **Normalized Sequence Position** | Continuous scalar $\in [0, 1]$ ($\frac{i}{N - 1}$) | Sequence Indexing |
| **`[23]`** | **Relative SASA ($\text{rSASA}$)** | Continuous scalar $\in [0, 1]$ (All-atom SASA / MaxASA) | FreeSASA |
| **`[24]`** | **B-Factor** | Continuous scalar (Atomic temperature factor) | Structure File (`.cif` / `.pdb`) |
| **`[25]`** | **Occupancy** | Continuous scalar $\in [0, 1]$ | Structure File (`.cif` / `.pdb`) |
| **`[26]`** | **ANM Thermal Fluctuation ($\langle \Delta r_i^2 \rangle$)** | Continuous scalar $\in [0, \infty)$ ($C_\alpha$-only ANM mean-square fluctuation) | SciPy ANM |

---

### 3.2 Edge Feature Tensor ($\mathbf{E} \in \mathbb{R}^{E \times 20}$)

For each directed edge $e = (u, v)$ connecting residue $u$ to residue $v$ within spatial cutoff $R_c = 8.0\text{ \AA}$ (based on $C_\alpha$-$C_\alpha$ Euclidean distance $d_{uv}$):

| Index Range | Feature Description | Encoding / Scale |
| :---: | :--- | :--- |
| **`[0:16]`** | **Distance Radial Basis Functions (RBF)** | $16$ Gaussian expansion channels centered from $0.0\text{ \AA}$ to $8.0\text{ \AA}$ |
| **`[16:19]`** | **Normalized Direction Vector** | Unit vector $\frac{\mathbf{r}_v - \mathbf{r}_u}{\Vert{}\mathbf{r}_v - \mathbf{r}_u\Vert{}_2} \in \mathbb{R}^3$ |
| **`[19]`** | **Inter-Chain Flag / Sequence Separation** | Binary flag or scaled sequential offset |

---

## 4. Architectural Decision

**Verdict: Standardize and lock Module 1 output data contract to `X` (27-dim node features) and `E` (20-dim edge features).**

The output object from Module 1 will adhere to standard PyTorch Geometric data format:

```python
from torch_geometric.data import Data

graph_data = Data(
    x=x_tensor,            # Shape: [N, 27] (float32)
    edge_index=edge_index, # Shape: [2, E] (int64)
    edge_attr=edge_attr,   # Shape: [E, 20] (float32)
    pos=pos_tensor,        # Shape: [N, 3]  (C_alpha coordinates)
)
