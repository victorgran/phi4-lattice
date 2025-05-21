from collections.abc import Callable
from functools import partial
import numpy as np

from .lattice import Phi4Lattice


def propose_uniform(current_field: np.ndarray,
                    current_action: float,
                    evaluate_action: Callable[[np.ndarray], float],
                    half_width: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
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
    evaluate_action : Callable[[np.ndarray], float]
        Function to evaluate the action for a field configuration.
    half_width : float
        Half-width of the uniform distribution.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple[numpy.ndarray, float, float]
        Proposed field, value of the action at the proposed field, and logarithm of the acceptance probability.
    """
    proposed_field = current_field + half_width * (2 * rng.random(current_field.shape) - 1)
    proposed_action = evaluate_action(proposed_field)
    log_acceptance = current_action - proposed_action
    return proposed_field, proposed_action, log_acceptance


def propose_hamiltonian(current_field: np.ndarray,
                        current_action: float,
                        evaluate_action: Callable[[np.ndarray], float],
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
    evaluate_action : Callable[[np.ndarray], float]
        Function to evaluate the action for a field configuration.
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
    momentum0 = rng.normal(loc=0, scale=1, size=current_field.shape)
    proposed_field, momentum1 = integrator(current_field, momentum0)
    proposed_action = evaluate_action(proposed_field)

    hamiltonian0 = 0.5 * np.sum(momentum0 ** 2) + current_action
    hamiltonian1 = 0.5 * np.sum(momentum1 ** 2) + proposed_action

    log_acceptance = hamiltonian0 - hamiltonian1

    return proposed_field, proposed_action, log_acceptance


def get_proposal_function(lattice: Phi4Lattice,
                          method: str,
                          rng: np.random.Generator,
                          half_width: float,
                          integration: str, delta_t: float, time_steps: int) -> Callable:
    """
    Get a proposal function as a partial function from the available methods.

    Parameters
    ----------
    lattice : Phi4Lattice
        Phi4 lattice object.
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
        propose_sample = partial(propose_uniform, evaluate_action=lattice.evaluate_action,
                                 half_width=half_width, rng=rng)
    elif method.lower() == "hamiltonian":
        assert delta_t is not None, "Hamiltonian Monte Carlo requires specifying delta_t parameter."
        assert time_steps is not None, "Hamiltonian Monte Carlo requires specifying time_steps parameter."

        if integration.lower() == "leapfrog":
            from .integrators import leapfrog_integrator
            integrator = partial(leapfrog_integrator, force_function=lattice.evaluate_force,
                                 delta_t=delta_t, time_steps=time_steps)
        else:
            raise ValueError(f"Integrator for HMC '{integration}' not recognized.")

        propose_sample = partial(propose_hamiltonian, evaluate_action=lattice.evaluate_action,
                                 integrator=integrator, rng=rng)
    else:
        raise ValueError(f'Unknown sampling method: {method}')
    return propose_sample
