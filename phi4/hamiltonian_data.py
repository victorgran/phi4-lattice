from collections.abc import Callable
import numpy as np
from tqdm import tqdm

from phi4.implementation.lattice import Phi4Lattice


def searchStepSize(initial_guess: float, get_acceptance: Callable, target_acceptance: float):
    digits = 3  # Number of decimal places.
    start = round(-np.log10(initial_guess))

    acceptance = get_acceptance(initial_guess)
    direction = -1.0 if acceptance < target_acceptance else 1.0
    step_size = initial_guess
    found_target = False

    for precision in range(digits - start + 1):
        change = 10 ** (-start - precision)

        for _ in range(10):
            step_size += direction * change

            if np.round(step_size, decimals=digits) <= 0:
                step_size -= direction * change  # Change is too large. Go to next value for precision.
                break

            acceptance = get_acceptance(step_size)

            if target_acceptance - 0.05 < acceptance < target_acceptance + 0.05:
                found_target = True
                break
            elif direction * (target_acceptance - acceptance) > 0:
                direction *= -1.0
                break

        if found_target:
            break

    if not found_target:
        print(f"Could not find time step size. Last acceptance was {acceptance} for {step_size}")

    return step_size


def getAcceptanceStepSizes(linear_sizes: list | np.ndarray,
                           mass2_values: list | np.ndarray,
                           target_acceptance: float):
    parameters = []

    rng = np.random.default_rng(seed=42)
    sampler_dict = {"num_samples": 1_000, "method": 'hamiltonian',
                    "rng": rng, "integration": 'leapfrog', "progress_bar": False}
    initial_guess = 0.3  # Initial guess for the first linear size, at the first value of mass2.
    last_linear_guess = initial_guess

    for linear_size in tqdm(linear_sizes, desc="Linear sizes", position=0):
        if linear_size != 5:
            continue
        initial_sample = rng.random(size=(linear_size, linear_size))
        step_size = last_linear_guess
        for mass_idx, mass2 in enumerate(mass2_values):
            mass2: float
            lattice = Phi4Lattice(linear_sites=linear_size, mass2=mass2, coupling_strength=8.0)
            step_size = searchStepSize(step_size,
                                       get_acceptance=lambda x: np.mean(
                                           lattice.sample_field(initial_sample,
                                                                delta_t=x,
                                                                time_steps=round(1. / x),
                                                                **sampler_dict)[1]),
                                       target_acceptance=target_acceptance)
            if mass_idx == 0:
                last_linear_guess = step_size

            parameters.append([linear_size, mass2, step_size])

    np.save('data/hmc_parameters.npy', parameters)

    return


def get_step_sizes():
    target_acceptance = 0.75
    linear_sizes = np.linspace(5, 50, num=20, endpoint=True, dtype=int)
    mass2_values = [-1.0 * i for i in range(16)]

    initial_step_sizes = [0.1, 0.025]
    parameters = []

    rng = np.random.default_rng(seed=42)

    # TODO:
    #  * For a given initial guess for delta_t, I need to know whether to increase or decrease,
    #    an by how much.
    #  * This value can then be used for the next iteration in the mass, but the first value
    #  for a given linear size should be saved, to serve as the starting point for the next iteration.

    for linear_size in linear_sizes:
        initial_sample = rng.random(size=(linear_size, linear_size))
        for mass2 in mass2_values:
            lattice = Phi4Lattice(linear_sites=linear_size, mass2=mass2, coupling_strength=8.0)
            best_step_size = None
            next_initial_state = None
            mean_acceptance = 0.0
            for delta_t in initial_step_sizes:
                samples, acceptance = lattice.sample_field(initial_field=initial_sample,
                                                           num_samples=1_000,
                                                           method='hamiltonian',
                                                           rng=rng,
                                                           integration='leapfrog',
                                                           delta_t=delta_t,
                                                           time_steps=round(1. / delta_t),
                                                           progress_bar=False)
                mean_acceptance = np.mean(acceptance)
                if mean_acceptance > target_acceptance:
                    best_step_size = delta_t
                    next_initial_state = samples[-1]
                    break

            if best_step_size is None:
                raise ValueError("No acceptable step size found")

            found_target = False
            initial_power = np.round(-np.log10(best_step_size))

            for idx in range(3):
                dt_increase = 10. ** (-initial_power - idx)

                if dt_increase < 0.001:
                    break

                for _ in range(10):
                    best_step_size += dt_increase
                    samples, acceptance = lattice.sample_field(initial_field=next_initial_state,
                                                               num_samples=1_000,
                                                               method='hamiltonian',
                                                               rng=rng,
                                                               integration='leapfrog',
                                                               delta_t=best_step_size,
                                                               time_steps=round(1. / best_step_size),
                                                               progress_bar=False)
                    mean_acceptance = np.mean(acceptance)

                    if target_acceptance - 0.01 <= mean_acceptance <= target_acceptance + 0.01:
                        found_target = True
                        break
                    elif mean_acceptance < target_acceptance:
                        best_step_size -= dt_increase
                        break

                if found_target:
                    break

            parameters.append([linear_size, mass2, best_step_size])
            print(f"N: {linear_size}, m2: {mass2}. Mean acceptance: {mean_acceptance:.3f}. "
                  f"Delta t: {best_step_size:.3f}")

    np.save('data/hmc_parameters.npy', parameters)
    return


if __name__ == "__main__":
    getAcceptanceStepSizes(linear_sizes=np.linspace(5, 50, num=20, endpoint=True, dtype=int),
                           mass2_values=[-1.0 * i for i in range(16)],
                           target_acceptance=0.75)
