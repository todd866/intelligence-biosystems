# Intelligence as High-Dimensional Coherence

**Repository:** [todd866/intelligence-biosystems](https://github.com/todd866/intelligence-biosystems)
**Paper status:** Published in *BioSystems* (January 2026)
**DOI:** [10.1016/j.biosystems.2026.105704](https://doi.org/10.1016/j.biosystems.2026.105704)

## Companion Paper

**Mathematical foundation:** The Observable Dimensionality Bound derived here is proven rigorously in the companion paper:

> **Curvature Amplification of Tracking Complexity on Statistical Manifolds**
> Target: *Information Geometry* (Springer)
> Repository: [todd866/tracking-complexity](https://github.com/todd866/tracking-complexity)

That paper shows covering numbers on negatively curved statistical manifolds grow *exponentially* with geodesic radius—meaning the flat-geometry assumption in this paper is actually conservative. The opacity bound is even stronger than stated.

## One-line thesis

Intelligence arises from maintaining high-dimensional coherent dynamics that exceed observation channel capacity; continuous substrates achieve this with far fewer irreversible state registrations than discrete implementations.

## Core results

### 1. Observable Dimensionality Bound

We derive the critical dimensionality threshold:

```
D_crit = C_commit × τ_e / (α × h_ε)
```

where:
- **C_commit** = behavioral commitment channel bandwidth (bits/s)
- **τ_e** = coherence timescale
- **h_ε** = bits per mode per τ_e at resolution ε (typically 2–5 bits for neural oscillations)
- **α** = compressibility factor ∈ (0,1]

**Key prediction:** When D_target > D_crit, external observers cannot fully track the system's state from behavioral output alone.

### 2. Collision Cost Theorem

Discrete state-space enumeration incurs collision costs scaling as k^(D_target - D_obs), making high-dimensional tracking intractable. Continuous relaxation dynamics avoid stepwise collisions, paying only readout costs.

### 3. MEG Worked Example

Human cortex operates at D_eff^MEG ~ 300 effective dimensions (parcel × band). With behavioral output bandwidth ~100 bits/s, we get:

```
D_eff / D_crit ~ 10²
```

Cortical dynamics exceed observable dimensionality by two orders of magnitude.

## Repository structure

```
3_intelligence/
├── intelligence.tex              # Published manuscript
├── intelligence.pdf              # Compiled output
├── README.md
├── LICENSE
├── code/
│   ├── vas_scaling_simulation.py # VAS classes and scaling experiments
│   ├── code_formation_simulation.py  # Hebbian learning simulation
│   └── figure1_regenerate.py     # Generates Figure 1 (imports above)
├── figures/
│   └── intelligence_figure1.png/pdf
└── archive/                      # (gitignored) Old versions, commentary
```

## Simulation validation

The `code/` directory contains simulation scripts demonstrating the collision vs relaxation mechanism:

| Script | Purpose |
|--------|---------|
| `vas_scaling_simulation.py` | DiscreteVAS and ContinuousVAS classes; collision count scaling with dimension |
| `code_formation_simulation.py` | Hebbian pathway learning vs discrete enumeration; generates clustering data |
| `figure1_regenerate.py` | Generates Figure 1 (imports above modules) |

**Key result:** Discrete state updates incur O(n) collision costs; continuous relaxation converges with 0 collisions during evolution (1 at final readout).

## What we are and aren't claiming

| We claim | We do not claim |
|----------|-----------------|
| Continuous relaxation avoids collision costs that discrete enumeration incurs | Continuous dynamics solve worst-case VAS reachability in polynomial time |
| Collision count is an energy proxy (Landauer) | Continuous dynamics have zero total dissipation |
| High-D substrates can track states that low-D observers cannot fully resolve | All biological computation requires high-D substrates |

## Build

```bash
pdflatex intelligence.tex && pdflatex intelligence.tex
```

## Citation

```bibtex
@article{todd2026intelligence,
  title={Intelligence as High-Dimensional Coherence: The Observable Dimensionality Bound and Computational Tractability},
  author={Todd, Ian},
  journal={BioSystems},
  year={2026},
  doi={10.1016/j.biosystems.2026.105704}
}
```

## License

MIT License. See [LICENSE](LICENSE).
