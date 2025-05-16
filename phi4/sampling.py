import numpy as np
from tqdm import tqdm

from lattice import Phi4Lattice


def proposeSampleUniform(field: np.ndarray, _, width: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return field + width * (2 * rng.random(*field.shape) - 1)


def sampleLattice(initial_sample: np.ndarray, lattice: Phi4Lattice, num_samples: int, method: str,
                  rng: np.random.Generator,
                  width: float = None,
                  delta_t: float = None, time_steps: int = None):
    if method.lower() == "uniform":
        assert width is not None, "Uniform sampling requires specifying width parameter."
    elif method.lower() == "hamiltonian":
        assert delta_t is not None, "Hamiltonian Monte Carlo requires specifying delta_t parameter."
        assert time_steps is not None, "Hamiltonian Monte Carlo requires specifying time_steps parameter."
    else:
        raise ValueError('Unknown sampling method: {}'.format(method))

    # TODO: Define sampling method by checking string and required parameters.
    samples = [initial_sample]
    acceptance = [0 for _ in range(num_samples - 1)]
    current_field = initial_sample
    current_action = lattice.evaluate_action(current_field)

    return
