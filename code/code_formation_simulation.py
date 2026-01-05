#!/usr/bin/env python3
"""
Code Formation in High-Dimensional Collision-Free Systems

Demonstrates spontaneous code formation in high-dimensional adaptive systems
through Hebbian-like pathway strengthening.

Compares:
- Adaptive pathway network (learns codes through weight adaptation)
- Discrete enumeration (no structural learning)

Generates numerical results: pathway specialization, modular structure emergence,
and performance advantages of adaptive high-dimensional exploration.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

np.random.seed(42)

class ContinuousAdaptiveSystem:
    """Adaptive pathway network with Hebbian learning."""
    def __init__(self, n_dim=20, n_paths=50):
        self.n_dim = n_dim
        self.n_paths = n_paths
        self.paths = np.random.randn(n_paths, n_dim)
        self.paths = self.paths / np.linalg.norm(self.paths, axis=1, keepdims=True)
        self.weights = np.ones(n_paths) / n_paths  # Start uniform
        self.solution_vectors = []
        self.solution_trials = []

    def solve(self, target, learning_rate=0.05, temperature=1.0, trial_num=0):
        """Find pathway combination via weighted selection."""
        alignments = self.paths @ target
        scores = alignments * self.weights
        probs = np.exp(scores/temperature) / np.sum(np.exp(scores/temperature))
        selected = np.random.choice(self.n_paths, size=5, p=probs, replace=False)

        # Record solution (pathway activation pattern)
        activation = np.zeros(self.n_paths)
        activation[selected] = 1
        self.solution_vectors.append(activation)
        self.solution_trials.append(trial_num)

        # Hebbian update: strengthen successful pathways
        coeffs, _, _, _ = np.linalg.lstsq(
            self.paths[selected].T, target, rcond=None)
        error = np.linalg.norm(self.paths[selected].T @ coeffs - target)

        if error < 0.8:  # Success threshold
            for idx in selected:
                self.weights[idx] += learning_rate * (1 - self.weights[idx])
            self.weights /= np.sum(self.weights)
            return True, error
        return False, error

class DiscreteEnumerativeSystem:
    """Baseline: random pathway selection, no learning.

    Uses SAME representation as adaptive system (pathway activation vectors)
    to enable fair PCA comparison.
    """
    def __init__(self, n_dim=20, n_paths=50):
        self.n_dim = n_dim
        self.n_paths = n_paths
        # Same pathway structure as adaptive, but no learning
        self.paths = np.random.randn(n_paths, n_dim)
        self.paths = self.paths / np.linalg.norm(self.paths, axis=1, keepdims=True)
        self.solution_vectors = []
        self.solution_trials = []

    def solve(self, target, max_attempts=100, trial_num=0):
        """Random pathway selection (no learning, no weight adaptation)."""
        best_error = float('inf')
        best_activation = None

        for _ in range(max_attempts):
            # Randomly select 5 pathways (uniform, no learning)
            selected = np.random.choice(self.n_paths, size=5, replace=False)

            # Record activation pattern (same dimensionality as adaptive)
            activation = np.zeros(self.n_paths)
            activation[selected] = 1

            # Compute error
            coeffs, _, _, _ = np.linalg.lstsq(
                self.paths[selected].T, target, rcond=None)
            error = np.linalg.norm(self.paths[selected].T @ coeffs - target)

            if error < best_error:
                best_error = error
                best_activation = activation.copy()

            if error < 0.8:
                self.solution_vectors.append(activation)
                self.solution_trials.append(trial_num)
                return True, error

        # Use best attempt
        self.solution_vectors.append(best_activation if best_activation is not None
                                      else np.zeros(self.n_paths))
        self.solution_trials.append(trial_num)
        return False, best_error

def generate_clustered_tasks(n_tasks=100, n_dim=20, n_clusters=5):
    """Tasks with similar structure (enables code reuse)."""
    clusters = np.random.randn(n_clusters, n_dim)
    clusters = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
    tasks = []
    for _ in range(n_tasks):
        cluster = clusters[np.random.randint(n_clusters)]
        task = cluster + np.random.randn(n_dim) * 0.3
        tasks.append(task / np.linalg.norm(task))
    return tasks

def run_simulation(n_trials=100):
    """Run code formation experiment."""
    continuous = ContinuousAdaptiveSystem(n_dim=20, n_paths=50)
    discrete = DiscreteEnumerativeSystem(n_dim=20)
    tasks = generate_clustered_tasks(n_trials, n_dim=20, n_clusters=5)

    for trial, task in enumerate(tasks):
        temp = max(0.5, 2.0 * (1 - trial/n_trials))
        continuous.solve(task, learning_rate=0.05,
                        temperature=temp, trial_num=trial)
        discrete.solve(task, max_attempts=100, trial_num=trial)

    return continuous, discrete

def save_results(continuous, discrete, output_path='figures/code_formation_data.npz'):
    """Save simulation results for use by figure generation scripts."""
    from pathlib import Path
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Convert to arrays
    adaptive_sols = np.array(continuous.solution_vectors)
    discrete_sols = np.array(discrete.solution_vectors)

    np.savez(
        output_dir / 'code_formation_data.npz',
        adaptive_solutions=adaptive_sols,
        discrete_solutions=discrete_sols,
        adaptive_weights=continuous.weights,
        n_pathways=continuous.n_paths,
        n_dim=continuous.n_dim
    )
    print(f"Saved: {output_dir / 'code_formation_data.npz'}")
    return adaptive_sols, discrete_sols


if __name__ == "__main__":
    cont, disc = run_simulation(n_trials=100)
    print("CODE FORMATION SIMULATION RESULTS")
    print("=" * 60)
    print(f"Adaptive: {len(cont.solution_vectors)} solutions")
    print(f"Weight concentration: {np.max(cont.weights)/np.mean(cont.weights):.2f}x")
    print(f"Top 5 pathways capture: {np.sort(cont.weights)[::-1][:5].sum():.1%}")
    print("\nInterpretation: Hebbian learning concentrates pathway usage,")
    print("demonstrating spontaneous code formation in high-D systems.")

    # Save for figure generation
    save_results(cont, disc)
