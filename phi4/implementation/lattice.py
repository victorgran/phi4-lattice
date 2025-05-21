import numpy as np
from tqdm import tqdm


def five_stencil_laplacian(field: np.ndarray) -> np.ndarray:
    """
    Compute the 5-point stencil laplacian of a field assuming periodic boundary conditions.

    Parameters
    ----------
    field : numpy.ndarray
        Input field.

    Returns
    -------
    numpy.ndarray
        Laplacian of the field.
    """
    partial_x2 = np.roll(field, shift=-1, axis=1) + np.roll(field, shift=1, axis=1) - 2 * field
    partial_y2 = np.roll(field, shift=-1, axis=0) + np.roll(field, shift=1, axis=0) - 2 * field
    return partial_x2 + partial_y2


class Phi4Lattice:
    def __init__(self, linear_sites: int, mass2: float, coupling_strength: float):
        self.linear_sites = linear_sites
        self.shape = (self.linear_sites, self.linear_sites)
        self.mass2 = mass2
        self.coupling_strength = coupling_strength

        # Set specific method for computing the laplacian.
        self.laplacian = five_stencil_laplacian

    def __str__(self):
        return (f"Euclidean phi4 theory on a {self.linear_sites} x {self.linear_sites} lattice, "
                f"with squared mass {self.mass2} and coupling strength {self.coupling_strength}.")

    def __repr__(self):
        return f"Phi4Lattice({self.linear_sites}, {self.mass2}, {self.coupling_strength})"

    def evaluate_action(self, field: np.ndarray) -> float:
        """
        Evaluate the action for a given field.

        Parameters
        ----------
        field : numpy.ndarray
            Input field.

        Returns
        -------
        float
            Value of the action.
        """
        # TODO: Is it faster to compute the square of the field and then multiply?
        #  Or to compute compute both powers separately?
        lagrangian = (-field * self.laplacian(field) +
                      self.mass2 * (field ** 2) +
                      self.coupling_strength * (field ** 4))
        return np.sum(lagrangian)

    def evaluate_force(self, field: np.ndarray) -> np.ndarray:
        """
        Evaluate the force for a given field.

        The force is defined as minus the functional derivative of the action with respect to the field.
        In this case, the force can be computed analytically, so this function simply evaluates it.

        Parameters
        ----------
        field : numpy.ndarray
            Input field.

        Returns
        -------
        numpy.ndarray
            Force at each point of the field.
        """
        force = self.laplacian(field) - (self.mass2 * field) - (2 * self.coupling_strength * (field ** 3))
        return force * 2

    def sample(self, initial_sample: np.ndarray, num_samples: int, method: str, rng: np.random.Generator,
               half_width: float = None,
               integration: str = "leapfrog", delta_t: float = None, time_steps: int = None,
               progress_bar: bool = True
               ) -> tuple[list[np.ndarray], list[int]]:
        """
        Sample field configurations using a given proposal method.

        Parameters
        ----------
        initial_sample : numpy.ndarray
            Initial state of the field to build the Markov chain.
        num_samples : int
            Number of total samples.
            This means that `num_samples - 1` samples are generated.
        method : {'uniform', 'hamiltonian'}
            Proposal method.
        rng : numpy.random.Generator
            Random number generator.
        half_width : float
            Half-width of the uniform distribution.
        integration : {'leapfrog'}
            Integration scheme for solving Hamilton's equations.
        delta_t : float
            Time step size for the integration scheme.
        time_steps : int
            Number of time steps for the integration scheme.
        progress_bar : bool, default=True
            Whether to display a progress bar for the sample generation.

        Returns
        -------
        tuple[list[numpy.ndarray], list[int]]
            List of samples and list with the acceptance of each generated sample (1 if accepted, 0 otherwise).
        """
        from proposals import get_proposal_function
        propose_sample = get_proposal_function(self, method, rng, half_width, integration, delta_t, time_steps)

        samples = [initial_sample]
        acceptance = [0 for _ in range(num_samples - 1)]
        current_field = initial_sample
        current_action = self.evaluate_action(current_field)

        for idx in tqdm(range(num_samples - 1), disable=not progress_bar):
            proposed_field, proposed_action, log_acceptance = propose_sample(current_field, current_action)

            if log_acceptance >= 0 or np.exp(log_acceptance) > rng.random():
                samples.append(proposed_field)
                current_field = proposed_field
                current_action = proposed_action
                acceptance[idx] = 1
            else:
                samples.append(current_field)

        return samples, acceptance
