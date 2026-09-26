# ADR 0001: Direct Dictionary Parsing via MMCIF2Dict over BioPython MMCIFParser

## Status
Accepted

## Context
Module 1 (Structure Graph Engine) requires high-throughput parsing of 3D macromolecular coordinate files to extract backbone Cα atoms, sequence identities, B-factors, occupancies, and spatial geometry.

Traditionally, structural biology pipelines rely on BioPython's `Bio.PDB.MMCIFParser` to construct an explicit hierarchical object tree (`Structure` -> `Model` -> `Chain` -> `Residue` -> `Atom`).

However, when scaling to large macromolecular assemblies (such as Cryo-EM benchmark structure `8r3y`), object-oriented structural hierarchies introduce performance bottlenecks, higher memory overhead, and friction when extracting vectorized NumPy arrays for downstream graph construction.

To evaluate this trade-off, we conducted an empirical benchmark across $N = 50$ paired iterations using `8r3y.cif` comparing `MMCIF2Dict` against `Bio.PDB.MMCIFParser`.

## Decision
We decided to bypass BioPython's object-oriented structural hierarchy (`Bio.PDB.Structure`) entirely for coordinate parsing. Instead, we use `Bio.PDB.MMCIF2Dict.MMCIF2Dict` to ingest raw mmCIF files directly into flat Python dictionaries, extracting target Cα atom attributes via vectorized NumPy array slicing.

## Empirical Benchmark Evidence
* **Parsing Latency:** `MMCIF2Dict` achieved a median execution time of **2094.00 ms** vs. **3645.89 ms** for `MMCIFParser`—delivering a **1.74x speedup** (Wilcoxon Signed-Rank Test $p = 1.78 \times 10^{-15}$, Cohen's $d = 2.73$).
* **Peak Memory Allocation:** `MMCIF2Dict` allocated a median peak RAM of **17.18 MB** vs. **37.83 MB** for `MMCIFParser`—yielding a **54.6% reduction in peak memory consumption** (Wilcoxon Signed-Rank Test $p = 7.20 \times 10^{-10}$, Cohen's $d = 3.27 \times 10^4$).

## Rationale
* **Zero Object Construction Overhead:** Eliminating the construction of thousands of Python `Residue` and `Atom` objects significantly reduces memory allocation and garbage collection overhead during batch loading.
* **Native mmCIF / Cryo-EM Scale:** Modern Cryo-EM structures deposited in the PDB (e.g., `8r3y`) use the mmCIF/PDBx format. `MMCIF2Dict` parses raw CIF key-value pairs (`_atom_site.Cartn_x`, `_atom_site.label_comp_id`, etc.) natively without format translation errors or atom/chain naming limits.
* **Vectorized Data Contracts:** Output dictionary tags map directly into NumPy arrays (`coords_np`, `b_factors_np`, `occupancies_np`), which feed directly into downstream physical calculations (FreeSASA, custom SciPy ANM) and PyTorch Geometric graph construction.

## Consequences
* **Positive:** Statistically significant parsing speedups, 54.6% lower RAM footprint per structure, and direct alignment with vectorized NumPy/PyTorch tensor contracts.
* **Negative/Maintenance:** The custom parser retains responsibility for mmCIF tag schema validation, alternate location handling (`label_alt_id`), non-standard residue filtering, and sequence alignment logic.

## References
* Benchmark Notebook: `benchmarks/01_compare_MMCIFParser_and_MMCIF2Dict.ipynb` (`graybilc/aPKC-mechanics-analysis`)
* Westbrook, J. D., et al. (2003). The Protein Data Bank Exchange/mmCIF format. *Nucleic Acids Res.*, 31(1), 489-491.
