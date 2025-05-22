from collections.abc import Callable
import json
import numpy as np
from tqdm import tqdm

from .lattice import Phi4Lattice


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def find_acceptance_parameter(initial_guess: float,
                              target_acceptance: float,
                              get_acceptance: Callable,
                              decimals: int = 3,
                              tolerance: float = 0.05) -> tuple[np.ndarray, float]:
    init_power = round(-np.log10(initial_guess))  # Define the initial power for decimal search.

    last_sample, acceptance = get_acceptance(initial_guess)
    first_test = initial_guess + 10 ** (-init_power)
    test_last_sample, test_accept = get_acceptance(first_test)
    is_increasing = test_accept > acceptance  # If true, acceptance increases with increasing parameter.
    parameter = initial_guess  # This keeps track of the optimal value of the parameter.

    if np.abs(test_accept - target_acceptance) < np.abs(acceptance - target_acceptance):
        acceptance = test_accept
        parameter = first_test
        last_sample = test_last_sample

    if is_increasing:
        direction = 1.0 if acceptance < target_acceptance else -1.0
    else:
        direction = -1.0 if acceptance < target_acceptance else 1.0

    found_target = False

    for precision in range(decimals - init_power + 1):
        change = 10 ** (-init_power - precision)

        for _ in range(10):
            parameter += direction * change

            if parameter <= 0:
                parameter -= direction * change  # Change is too large. Go to next value for precision.
                break

            new_last_sample, new_acceptance = get_acceptance(parameter)
            # For a given acceptance, this can be:
            # * Within the accepted interval of the target acceptance.
            # * Crossing the target acceptance.
            # * Keep moving in the same direction towards the target.

            if target_acceptance - tolerance < new_acceptance < target_acceptance + tolerance:
                found_target = True
                acceptance = new_acceptance
                last_sample = new_last_sample
                break
            elif direction * (target_acceptance - new_acceptance) > 0:
                # Check if the new value is better than the last one,
                # based on which acceptance is closer to target.
                if np.abs(new_acceptance - target_acceptance) < np.abs(acceptance - target_acceptance):
                    direction *= -1.0
                    acceptance = new_acceptance
                    last_sample = new_last_sample
                else:
                    parameter -= direction * change  # Previous parameter was closer to target.

                break

            acceptance = new_acceptance
            last_sample = new_last_sample

        if found_target:
            break

    if not found_target:
        print(f"Could not find time step size. Last acceptance was {acceptance} for {parameter}")

    return last_sample, parameter


def find_parameters(linear_sizes: list[int] | np.ndarray,
                    mass2_values: list[float] | np.ndarray,
                    num_samples: int,
                    target_acceptance: float,
                    filename: str,
                    initial_guess: float,
                    method: str,
                    integration: str = None,
                    tolerance: float = 0.02,
                    decimals: int = 3):
    rng = np.random.default_rng(seed=42)
    lattice = Phi4Lattice(linear_sites=5, mass2=-1.0, coupling_strength=8.0)
    sampler_dict = {"num_samples": num_samples, "method": method,
                    "integration": integration,
                    "rng": rng, "progress_bar": False}

    if method == 'uniform':
        def partial_sample(x: float, init_sample):
            samples, acceptances = lattice.sample(init_sample, half_width=x, **sampler_dict)
            return samples[-1], np.mean(acceptances)
    elif method == 'hamiltonian':
        def partial_sample(x: float, init_sample):
            samples, acceptances = lattice.sample(init_sample, delta_t=x,
                                                  time_steps=round(1. / x), **sampler_dict)
            return samples[-1], np.mean(acceptances)
    else:
        raise ValueError(f"Method '{method}' not implemented")

    counter = 0
    data_params = {}
    l_first_guess = initial_guess
    initial_mass = mass2_values[0]

    for linear_size in tqdm(linear_sizes, leave=True):
        lattice.linear_sites = linear_size
        initial_sample = rng.random(size=lattice.shape)
        initial_guess = l_first_guess

        for mass2 in mass2_values:
            lattice.mass2 = mass2
            next_sample, parameter = find_acceptance_parameter(initial_guess=initial_guess,
                                                               target_acceptance=target_acceptance,
                                                               get_acceptance=lambda x: partial_sample(
                                                                   x, initial_sample),
                                                               tolerance=tolerance,
                                                               decimals=decimals)
            data_params[counter] = {"linear_size": linear_size,
                                    "mass2": mass2,
                                    "parameter": np.round(parameter, decimals=decimals),
                                    "initial_sample": next_sample}
            counter += 1
            initial_guess = parameter
            initial_sample = next_sample

            if mass2 == initial_mass:
                l_first_guess = parameter

    with open(filename, "w") as file:
        json.dump(data_params, file, cls=NumpyEncoder, indent=4)

    return
