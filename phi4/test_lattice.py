import numpy as np

from phi4.implementation.lattice import Phi4Lattice


def test_lattice_uniform(lattice: Phi4Lattice):
    rng = np.random.default_rng(seed=42)
    initial_field = rng.random(size=lattice.shape)
    samples, acceptance = lattice.sample(initial_sample=initial_field,
                                         num_samples=50_000,
                                         method="uniform", rng=rng, half_width=0.05)
    mean_acceptance = f'{np.mean(acceptance) * 100:.2f}%'
    assert mean_acceptance == '78.34%', "Test Failed"
    return


def test_lattice_hamiltonian(lattice: Phi4Lattice):
    rng = np.random.default_rng(seed=42)
    initial_field = rng.random(size=lattice.shape)
    samples, acceptance = lattice.sample(initial_sample=initial_field,
                                         num_samples=10_000,
                                         method="hamiltonian", rng=rng,
                                         integration="leapfrog", delta_t=0.05, time_steps=round(1. / 0.05))
    mean_acceptance = f'{np.mean(acceptance) * 100:.2f}%'
    assert mean_acceptance == '98.45%', "Test Failed"
    return


if __name__ == '__main__':
    fixture_lattice = Phi4Lattice(linear_sites=5, mass2=-1.0, coupling_strength=8.0)
    test_lattice_uniform(fixture_lattice)
    test_lattice_hamiltonian(fixture_lattice)
