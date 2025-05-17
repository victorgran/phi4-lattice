from collections.abc import Callable
import numpy as np


def leapfrog_integrator(field0: np.ndarray, momentum0: np.ndarray, force_function: Callable[[np.ndarray], np.ndarray],
                        delta_t: float, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the equations of motion using a leapfrog integrator.

    Parameters
    ----------
    field0 : numpy.ndarray
        Initial field.
    momentum0 : numpy.ndarray
        Initial momentum.
    force_function : Callable[[numpy.ndarray], numpy.ndarray]
        Function used to compute the force for a given field.
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
    force = force_function(field)

    for _ in range(time_steps):
        # kick - drift - kick
        momentum += 0.5 * delta_t * force
        field += delta_t * momentum
        force = force_function(field)
        momentum += 0.5 * delta_t * force

    return field, momentum
