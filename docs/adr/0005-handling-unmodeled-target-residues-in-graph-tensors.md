# 5. Fallback Anchor Node Assignment for Unmodeled Phosphorylation Sites in PyG Graph Construction

* **Status:** Accepted
* **Date:** 2026-10-03
* **Deciders:** Bioinformatics & ML Engineering
* **Technical Context:** `scripts/align_cif_sequences.py`, `ProteinGraphBuilder`, Nextflow Graph Extraction Pipeline

---

## Context and Problem Statement

Our structural pipeline targets human Lgl1 phosphorylation sites ($S_{655}$, $S_{659}$, $S_{663}$) to evaluate aPKC/Par6/Lgl polarity complex conformational mechanics.

When processing multi-subunit cryo-EM models across our dataset—specifically `8r3y` (resolved range 248–585) and the `9EJK` / `9EJL` / `9EJM` series (resolved range 249–588)—diagnostic sequence alignment revealed that target sites $S_{655–663}$ lie completely outside the modeled coordinate bounds. The C-terminal regulatory tail is intrinsically disordered and lacks electron density, leading to missing $C_\alpha$ coordinates in MMCIF structures.

Passing missing coordinate indices directly to our `ProteinGraphBuilder` causes downstream PyTorch Geometric graph tensor extraction to fail or generate empty node sets. We need a consistent structural representation across both fully resolved and partially modeled loop structures without breaking Nextflow pipeline automation.

---

## Decision Drivers

* **Pipeline Continuity:** Nextflow workflows must execute end-to-end across heterogeneous MMCIF models without throwing missing-index exceptions.
* **Geometric Integrity:** PyG graphs require valid 3D spatial coordinates ($C_\alpha$) to build KNN/radius edge indices and distance attributes.
* **Dual-Modality Support:** The spatial "exit vector" where the disordered tail attaches to the rigid WD40 scaffold must be captured for downstream graph embeddings.

---

## Considered Options

1. **Option 1: Hard Fail on Unmodeled Sites** — Halt execution if target sites lack $C_\alpha$ coordinates.
2. **Option 2: Filter/Omit Structures** — Exclude structures missing $S_{655–663}$ from graph tensor generation.
3. **Option 3: Automatic C-Terminal Boundary Anchor Fallback (Chosen)** — Dynamically map missing target sites to the nearest resolved C-terminal boundary residue (`max(tgt_nums)`, e.g., residue 588) as a spatial anchor, flagging the node in metadata as `UNMODELED`.

---

## Decision Outcome

**Chosen Option:** **Option 3**.

When target sites fall outside the resolved coordinate range during `map_target_residues()`, `scripts/align_cif_sequences.py` will assign the C-terminal boundary residue as `fallback_anchor_res_num`.

The downstream PyTorch Geometric pipeline will:
1. Extract node/edge graph features using the boundary anchor residue as the spatial node representation.
2. Flag the site metadata with `status: "UNMODELED"`.
3. Allow hybrid downstream models to inject sequence-level tail embeddings (e.g., ESM-2 window around $S_{655–663}$) concatenated to the anchor node feature vector or global graph attributes.

---

## Consequences

### Positive
* **Robust Execution:** Nextflow processes run cleanly across `8r3y`, `9EJK`, `9EJL`, and `9EJM` without manual CLI coordinate overrides.
* **Preserves Conformational Context:** Retains the core WD40 domain graph features and interface distance tensors.
* **Extensible Architecture:** Provides a clean interface for hybrid (Structure + Sequence Transformer) embeddings.

### Negative / Risks
* **Feature Shift:** Model downstream layers must handle node embeddings where spatial position represents the loop anchor rather than the exact $C_\alpha$ position of the serine residue.
* **Metadata Tracking:** Downstream tensor loaders must explicitly parse `status: "UNMODELED"` flags to prevent misinterpreting anchor node coordinates as true phospho-serine coordinates.
