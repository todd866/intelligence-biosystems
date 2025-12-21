# Intelligence as High-Dimensional Coherence

**Repository:** todd866/intelligence-biosystems
**Paper status:** Under review at *BioSystems* (R2 minor revision, BIOSYS-D-25-00880R2)
**This repo:** Post-submission validation + revisions (transparent, versioned)

## One-line thesis

Intelligence arises from the capacity to track high-dimensional target states through low-bandwidth observation channels; continuous high-dimensional substrates achieve this efficiently where discrete algorithms fail.

## Core results

### 1. Observable Dimensionality Bound

We derive the critical dimensionality threshold:

```
D_crit = C_obs × τ_e / (α × h_track × ε)
```

where:
- **C_obs** = observation channel bandwidth (bits/s)
- **τ_e** = tracking timescale
- **h_track** = bits per dimension per update
- **ε** = acceptable tracking error
- **α** = efficiency factor

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
├── intelligence.tex              # Current manuscript (R2)
├── intelligence.pdf              # Compiled output
├── code/
│   ├── vas_scaling_simulation.py # Main simulation: collision scaling with dimension
│   ├── figure1_regenerate.py     # Generates Figure 1
│   └── _old/                     # Archived earlier scripts
├── figures/
│   ├── intelligence_figure1.png  # Main figure
│   └── intelligence_figure1.pdf
├── _submission_materials/
│   ├── cover_letter_r2.tex/pdf   # R2 cover letter
│   ├── response_to_editor_r2.tex/pdf  # Point-by-point response
│   └── R1/                       # Archived R1 materials
└── _archive/                     # Old submissions, revisions, commentary
```

## Simulation validation

The `code/` directory contains simulation scripts demonstrating the collision vs relaxation mechanism:

| Script | Purpose |
|--------|---------|
| `vas_scaling_simulation.py` | Tests collision count scaling with dimension D |
| `figure1_regenerate.py` | Generates Figure 1 (VAS trajectories, scaling, code formation) |

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
@article{todd2025intelligence,
  title={Intelligence as High-Dimensional Coherence: The Observable Dimensionality Bound and Computational Tractability},
  author={Todd, Ian},
  journal={BioSystems},
  year={2025},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
