import numpy as np

from lattice import Phi4Lattice


def test_lattice_uniform():
    rng = np.random.default_rng(seed=42)
    lattice = Phi4Lattice(linear_sites=5, mass2=-1.0, coupling_strength=8.0)
    initial_field = rng.random(size=lattice.shape)
    num_samples = 50_000
    samples, acceptance = lattice.sample(initial_sample=initial_field,
                                         num_samples=num_samples,
                                         method="uniform", rng=rng, half_width=0.05)
    print(f"Acceptance is {np.mean(acceptance) * 100:.2f}%")
    print("Typically, the optimal choice is around 60-70%")
    return


def test_lattice_hamiltonian():
    rng = np.random.default_rng(seed=42)
    lattice = Phi4Lattice(linear_sites=5, mass2=-1.0, coupling_strength=8.0)
    initial_field = rng.random(size=lattice.shape)
    num_samples = 10_000
    delta_t = 0.05
    samples, acceptance = lattice.sample(initial_sample=initial_field,
                                         num_samples=num_samples,
                                         method="hamiltonian", rng=rng,
                                         integration="leapfrog", delta_t=delta_t, time_steps=round(1. / delta_t))
    print(f"Acceptance is {np.mean(acceptance) * 100:.2f}%")
    print("Typically, the optimal choice is around 60-70%")
    return


if __name__ == '__main__':
    # test_lattice_uniform()
    test_lattice_hamiltonian()
