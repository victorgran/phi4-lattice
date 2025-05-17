from collections.abc import Callable
from functools import partial
import numpy as np
from tqdm import tqdm


class Phi4Lattice:
    def __init__(self, linear_sites: int, mass2: float, coupling_strength: float):
        self.linear_sites = linear_sites
        self.shape = (self.linear_sites, self.linear_sites)
        self.mass2 = mass2
        self.coupling_strength = coupling_strength

        # Set specific method for computing the laplacian.
        self.laplacian = self.five_stencil_laplacian

    @staticmethod
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

    def leapfrog_integrator(self, field0: np.ndarray, momentum0: np.ndarray,
                            delta_t: float, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the equations of motion using a leapfrog integrator.

        Parameters
        ----------
        field0 : numpy.ndarray
            Initial field.
        momentum0 : numpy.ndarray
            Initial momentum.
        delta_t : float
            Time step size.
        time_steps : int
            Number of time steps.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Field and momentum at the final time.
        """
        field = field0.copy()
        momentum = momentum0.copy()
        force = self.evaluate_force(field)

        for _ in range(time_steps):
            # kick - drift - kick
            momentum += 0.5 * delta_t * force
            field += delta_t * momentum
            force = self.evaluate_force(field)
            momentum += 0.5 * delta_t * force

        return field, momentum

    def propose_uniform(self, current_field: np.ndarray, current_action: float,
                        half_width: float, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        """
        Propose a new field configuration with a uniform distribution centered around the current field.

        The proposed field consists of the independent displacement of each lattice point
        in the current field by a random number between `-half_width` and `half_width`.

        The logarithm of the acceptance probability corresponds to the
        difference between the current and proposed values of the action.

        Parameters
        ----------
        current_field : numpy.ndarray
            Current field configuration.
        current_action : float
            Value of the action for the current field.
        half_width : float
            Half-width of the uniform distribution.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        tuple[numpy.ndarray, float, float]
            Proposed field, value of the action at the proposed field, and logarithm of the acceptance probability.
        """
        proposed_field = current_field + half_width * (2 * rng.random(self.shape) - 1)
        proposed_action = self.evaluate_action(proposed_field)
        log_acceptance = current_action - proposed_action
        return proposed_field, proposed_action, log_acceptance

    def propose_hamiltonian(self, current_field: np.ndarray, current_action: float,
                            integrator: Callable, rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
        """
        Propose a new field configuration by integrating the Hamilton's equations of the theory.

        The momentum field is randomly sampled from a Gaussian distribution of mean 0 and variance 1.

        The logarithm of the acceptance probability corresponds to the
        difference between the initial and final values of the Hamiltonian.

        Parameters
        ----------
        current_field : numpy.ndarray
            Current field configuration.
        current_action : float
            Value of the action for the current field.
        integrator : Callable
            Integrator to solve the equations of motion for the current field and initial momentum.
            It is assumed that the integrator is a function of only the field and momentum (in that order).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        tuple[numpy.ndarray, float, float]
            Proposed field, value of the action at the proposed field, and logarithm of the acceptance probability.
        """
        momentum0 = rng.normal(loc=0, scale=1, size=self.shape)
        proposed_field, momentum1 = integrator(current_field, momentum0)
        proposed_action = self.evaluate_action(proposed_field)

        hamiltonian0 = 0.5 * np.sum(momentum0 ** 2) + current_action
        hamiltonian1 = 0.5 * np.sum(momentum1 ** 2) + proposed_action

        log_acceptance = hamiltonian0 - hamiltonian1

        return proposed_field, proposed_action, log_acceptance

    def get_proposal_function(self, method: str, rng: np.random.Generator,
                              half_width: float,
                              integration: str, delta_t: float, time_steps: int) -> Callable:
        """
        Get a proposal function as a partial function from the available methods.

        Parameters
        ----------
        method : {'uniform', 'hamiltonian'}
            Method for the proposal function.
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

        Returns
        -------
        Callable
            Partial function to obtain a proposed field, the value of the action at
            the proposed field, and the logarithm of the acceptance probability.
        """
        if method.lower() == "uniform":
            assert half_width is not None, "Uniform sampling requires specifying half_width parameter."
            propose_sample = partial(self.propose_uniform, half_width=half_width, rng=rng)
        elif method.lower() == "hamiltonian":
            assert delta_t is not None, "Hamiltonian Monte Carlo requires specifying delta_t parameter."
            assert time_steps is not None, "Hamiltonian Monte Carlo requires specifying time_steps parameter."

            if integration.lower() == "leapfrog":
                integrator = partial(self.leapfrog_integrator, delta_t=delta_t, time_steps=time_steps)
            else:
                raise ValueError(f"Integrator for HMC '{integration}' not recognized.")

            propose_sample = partial(self.propose_hamiltonian, integrator=integrator, rng=rng)
        else:
            raise ValueError(f'Unknown sampling method: {method}')
        return propose_sample

    def sample(self, initial_sample: np.ndarray, num_samples: int, method: str, rng: np.random.Generator,
               half_width: float = None,
               integration: str = "leapfrog", delta_t: float = None, time_steps: int = None
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

        Returns
        -------
        tuple[list[numpy.ndarray], list[int]]
            List of samples and list with the acceptance of each generated sample (1 if accepted, 0 otherwise).
        """
        propose_sample = self.get_proposal_function(method, rng, half_width, integration, delta_t, time_steps)

        samples = [initial_sample]
        acceptance = [0 for _ in range(num_samples - 1)]
        current_field = initial_sample
        current_action = self.evaluate_action(current_field)

        for idx in tqdm(range(num_samples - 1)):
            proposed_field, proposed_action, log_acceptance = propose_sample(current_field, current_action)

            if log_acceptance >= 0 or np.exp(log_acceptance) > rng.random():
                samples.append(proposed_field)
                current_field = proposed_field
                current_action = proposed_action
                acceptance[idx] = 1
            else:
                samples.append(current_field)

        return samples, acceptance
