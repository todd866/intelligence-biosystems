# Intelligence Paper Workflow

**Date:** November 16, 2025
**Status:** BIOSYS-D-25-00880R1 submitted, awaiting minor revisions

## Editing Workflow

### Division of Responsibilities

**Codex (Read-Only Analysis):**
- Performs read-only operations on paper content
- Analyzes LaTeX source, simulations, figures
- Leaves commentary and recommendations in `codex.txt`
- Does NOT modify files directly

**Claude Code (Implementation):**
- Has write permissions for the directory
- Reads `codex.txt` commentary from Codex
- Applies fixes, edits, and improvements based on commentary
- Manages backups and directory structure
- Executes compilation and testing

### Workflow Steps

1. Codex analyzes current state (read-only)
2. Codex writes recommendations to `codex.txt`
3. Claude Code reads `codex.txt`
4. Claude Code applies fixes to source files
5. Claude Code compiles and verifies changes
6. Repeat as needed

## Backup Policy

- Full directory backup before major changes
- Backups stored in `~/Desktop/highdimensional/_backups/3_intelligence/`
- Backup naming: `backup_YYYYMMDD_HHMMSS/`
- Keep last 5 backups, archive older ones

## Directory Structure

```
3_intelligence/
├── intelligence.tex             # Main paper source (125KB)
├── intelligence.pdf             # Compiled output (64 pages, 787KB)
├── WORKFLOW.md                  # This file
├── commentary/
│   └── codex.txt               # Commentary from Codex (when present)
├── figures/                     # All figures
│   ├── intelligence_figure1.png
│   ├── vas_scaling.png
│   └── vas_scaling_data.npz
├── code/                        # Python simulations (4 files)
│   ├── figure1_discrete_vs_continuous.py
│   ├── vas_collision_comparison.py
│   ├── vas_scaling_simulation.py
│   └── code_formation_simulation.py
├── build/                       # LaTeX compilation artifacts
│   ├── intelligence.aux
│   ├── intelligence.log
│   └── intelligence.out
├── BIOSYS-D-25-00880/           # Original submission package
│   ├── feedback.txt
│   ├── intelligence as high-d coherence.pdf (original)
│   └── cover letter - intelligence.pdf (original)
├── BIOSYS-D-25-00880_R1.pdf     # Complete revision submission (2.7MB)
├── cover_letter_revision.*      # Revision cover letter
├── response_to_reviewers.*      # Detailed response document
├── submission.txt               # Editorial manager metadata
└── _archive/                    # Deprecated figures/versions
```

## Key Files

- **intelligence.tex**: 128KB, main LaTeX source
- **intelligence.pdf**: 805KB, 64 pages
- **submission.txt**: Editorial manager submission record
- **build_clean.sh**: Clean compilation script

## Current Status

- Revision submitted November 16, 2025
- Expecting minor revisions from reviewers
- Continuing improvements while waiting for feedback
- Main concerns: simulation reproducibility, empirical grounding, VAS tractability claims
