# Reframed Abstract and Intro

## New Abstract

Intelligence arises from the defense of nonergodic dynamics through high-dimensional coherence. High-dimensional systems cannot ergodically explore their state space—the curse of dimensionality ensures that trajectories remain confined to vanishingly small regions relative to the full space. This dimensional inaccessibility is not a limitation but the foundation of biological computation: the system's confinement to particular regions of phase space IS its memory, its constraints, its structured dynamics. Within these nonergodic regions, high entropy flow enables rich internal exploration; at the boundary, sparse outputs minimize free energy expenditure while defending the conditions for continued nonergodic existence. We formalize this through an Observable Dimensionality Bound: external observers cannot track systems whose effective dimensionality exceeds a critical threshold set by channel capacity and temporal resolution. This bound functions as *protection*—the opacity of high-dimensional dynamics shields internal structure from external perturbation. Living systems exploit this by embodying the dynamics: the substrate is the high-dimensional field, information lives in geometric configuration, and irreversible costs are paid only at sparse behavioral commitment events. The framework explains the efficiency gap between biological and artificial intelligence (brains pay thermodynamic costs at behavioral boundaries, not computational steps), provides a substrate-independent characterization of intelligence (capacity to maintain and defend coherent high-dimensional dynamics), and connects to the emergence of codes through adaptive dynamics. Human cortex operates far beyond the observable threshold, with MEG-accessible dimensionality alone exceeding critical thresholds by two orders of magnitude.

## New Section 1.1: The Core Claim

**The nonergodic foundation of intelligence.** High-dimensional dynamical systems are fundamentally nonergodic: they cannot explore their full state space in finite time. For a system with $D_{\text{eff}} \sim 10^3$ effective dimensions, even visiting each region once at Planck-time resolution would exceed the age of the universe. This is the curse of dimensionality—not as obstacle but as *opportunity*.

Nonergodicity creates structure. A system confined to a vanishingly small region of its state space has:
- **Memory**: the current region reflects the trajectory that led there
- **Constraint**: not all states are accessible, creating computational structure
- **Stability**: perturbations that would scatter a low-D system leave high-D attractors intact

Within nonergodic regions, high entropy flow enables rich dynamics. The system is not frozen—it explores vigorously within its accessible region, sampling configurations, evolving constraints, maintaining coherence. But it cannot wander ergodically across the full space. The walls are dimensional, not energetic.

**Intelligence as defense of nonergodic position.** An organism is a nonergodic island in an entropic sea. Its internal dynamics occupy a structured region of phase space that would be infinitesimally unlikely under ergodic sampling. To persist, it must:

1. Maintain high-dimensional internal coherence (the nonergodic structure itself)
2. Interface with the environment through low-dimensional channels (sensory input, behavioral output)
3. Emit outputs that *defend* the conditions for continued nonergodic existence

The third point is crucial. Behavioral outputs are not merely "responses" or "tracking signals"—they are free-energy-minimizing interventions that protect internal structure. The organism acts to maintain the viability of its nonergodic dynamics.

**The Observable Dimensionality Bound as armor.** We formalize this through a measurement-theoretic bound: external observers cannot track systems whose effective dimensionality exceeds $D_{\text{crit}} = C_{\text{obs}} \tau_e / (\alpha h_\varepsilon^{\text{track}})$. Beyond this threshold, systems become *observationally inaccessible*—not merely hard to compute, but impossible to track with finite bandwidth.

This opacity is functional. It protects internal dynamics from external interference. The high-dimensional structure that makes a system intelligent is the same structure that makes it untrackable—and untrackability is a form of autonomy.

**What high-dimensional continuous computation enables:**

1. **Nonergodic confinement as memory.** The system's position in phase space encodes its history. No explicit storage required—the dynamics themselves remember.

2. **High entropy flow within constraints.** Thermal coupling and stochastic dynamics enable exploration within the accessible region without breaking nonergodic confinement. The system gains functional dimensionality through noise while maintaining structural integrity.

3. **Constraint satisfaction through relaxation.** Problems intractable for discrete enumeration become tractable when embedded in continuous high-D dynamics. The system doesn't search—it relaxes toward solutions along constraint gradients.

4. **Sparse boundary costs.** Power scales with behavioral output rate, not internal complexity. The ~20W brain pays Landauer cost only when writing irreversible outputs, not during the high-entropy internal dynamics that constitute thinking.

---

## Notes on the reframe

Key shifts:
- From "tracking" to "defending nonergodic position"
- From "efficiency" to "autonomy through opacity"
- From "can't observe" (passive) to "protected by dimensionality" (active)
- From "sidesteps limits" to "exploits dimensional inaccessibility"

The technical machinery (Observable Dimensionality Bound, VAS, code formation) stays the same but is now motivated differently. The bound isn't a limitation we work around—it's armor we exploit.
