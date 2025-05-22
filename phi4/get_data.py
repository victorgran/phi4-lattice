import json

import numpy as np

from implementation.find_parameters import find_parameters


def get_initial_data(num_samples: int, target_acceptance: float,
                     method: str, integration: str = None):

    if method == "uniform":
        filenames = [f"data/uni{k}.json" for k in range(1, 4)]
    else:
        filenames = [f"data/hmc_{integration}{k}.json" for k in range(1, 4)]

    find_parameters(linear_sizes=np.linspace(5, 50, num=20, endpoint=True, dtype=int),
                    mass2_values=[-1.0],
                    num_samples=num_samples,
                    target_acceptance=target_acceptance,
                    filename=filenames[0],
                    initial_guess=0.3,
                    method=method,
                    integration=integration)
    find_parameters(linear_sizes=[16],
                    mass2_values=np.linspace(0, -12, num=20, endpoint=True),
                    num_samples=num_samples,
                    target_acceptance=target_acceptance,
                    filename=filenames[1],
                    initial_guess=0.3,
                    method=method,
                    integration=integration)
    find_parameters(linear_sizes=[26],
                    mass2_values=np.linspace(0, -12, num=20, endpoint=True),
                    num_samples=num_samples,
                    target_acceptance=target_acceptance,
                    filename=filenames[2],
                    initial_guess=0.3,
                    method=method,
                    integration=integration)
    return


def merge_files(method: str, integration: str = None):
    data_keys = ["finite_volume", "phase_transition1", "phase_transition2"]

    if method == "uniform":
        filenames = [f"data/uni{k}.json" for k in range(1, 4)]
        new_file = "data/uniform.json"
    elif method == "hamiltonian":
        filenames = [f"data/hmc_{integration}{k}.json" for k in range(1, 4)]
        new_file = "data/hmc_leapfrog.json"
    else:
        raise ValueError(f"Method {method} not recognized.")

    with open(new_file, "w") as target_file:
        all_data = {}
        for file_idx, filename in enumerate(filenames):
            data_key = data_keys[file_idx]

            with open(filename, "r") as source_file:
                data = json.load(source_file)

            all_data[data_key] = data

        json.dump(all_data, target_file, indent=4)

    return


if __name__ == "__main__":
    uniform_settings = {"num_samples": 10_000, "method": "uniform", "target_acceptance": 0.75}
    leapfrog_settings = {"num_samples": 3_000, "method": "hamiltonian", "integration": "leapfrog",
                         "target_acceptance": 0.75}
    # get_initial_data(**leapfrog_settings)
    merge_files("hamiltonian", integration="leapfrog")
