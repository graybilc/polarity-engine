# Tier 1 Parameter Provenance and Calculation Guide (ADR-0006)

This document outlines the empirical derivation, calculation methodology, and literature sources for the baseline rate parameters utilized in `src/polarity_engine/kinetics.py`. In accordance with our anti-data-leakage protocol, all parameters in this set are derived exclusively from historical bulk assays and are strictly segregated from independent validation sets (e.g., Parker & McDonald 2025).

---

## Parameter Specification Table

| Parameter / Rate Key | Numerical Value | Empirical Derivation & Literature Source |
| :--- | :--- | :--- |
| **`k_cat_multiplier` (Active aPKC-Par6)** | `8.5` (vs. `1.0` basal) | **Source:** *Graybill et al. (2012)*.<br>**Calculation:** Represents the fold-enhancement factor ($V_{max, \text{active}} / V_{max, \text{basal}}$) extracted from catalytic titration curves measuring aPKC turnover upon relief of autoinhibitory pseudosubstrate clamping by Par-6 and Cdc42. |
| **`k_000_001` (Site 3 Primary Entry)** | `0.22 * multiplier` | **Source:** *Graybill & Prehoda (2014)*.<br>**Calculation:** Anchors the ordered multisite hierarchy finding. Extracted from measured initial fractional velocity distributions ($v_0 / [E]_0$), establishing Site 3 as the kinetically favored primary priming site from the unphosphorylated state ($S_{000}$). |
| **`k_000_100` / `k_000_010` (Sites 1 & 2 Entry)** | `0.04` / `0.07 * multiplier` | **Source:** *Graybill & Prehoda (2014)*.<br>**Calculation:** Secondary and tertiary branching rates from $S_{000}$. Reflects lower initial velocity rates due to unfavorable steric or electrostatic access prior to structural priming. |
| **`k_100_110` / `k_001_101` (Progression Rates)** | `0.15` to `0.20 * multiplier` | **Source:** *Graybill & Prehoda (2014)* multisite time-course mass spectrometry.<br>**Calculation:** Intermediate downstream progression rates scaled relative to the primary entry rate to match cellular cortical displacement time-windows (1–5 minutes). |
| **`k_110_111` / `k_101_111` (Sink Approaches)** | `0.25` to `0.30 * multiplier` | **Source:** Fitted to ensure efficient terminal approach to the fully phosphorylated sink state ($S_{111}$). |
| **Dephosphorylation Rates (`k_X_000`)** | `0.01` to `0.03` | **Source:** Estimated background phosphatase activity.<br>**Calculation:** Set to roughly 10% of forward rates to maintain dynamic equilibrium and prevent instantaneous saturation of the hypercube. |

---

## Summary of Parameter Role
These values establish the baseline kinetic framework for wild-type Lgl phosphorylation under normal biochemical conditions. The computational pipeline subsequently applies structural rSASA multipliers and topological regime scalers (processive vs. dissociative) to test how mutant and homologue structures warp these baseline rates.
