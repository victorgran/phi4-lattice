import json
import matplotlib.pyplot as plt
import numpy as np

from implementation.find_parameters import find_parameters
from implementation.lattice import Phi4Lattice


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


def burn_in_data(filename: str):

    with open(filename, "r") as source_file:
        data = json.load(source_file)

    method_str = filename[5:][:3]

    if method_str == "uni":
        def set_params_dict(parameter: float, linear_size: int):
            return {"method": "uniform", "half_width": parameter, "num_samples": linear_size * 20_000}

    elif method_str == "hmc":
        def set_params_dict(parameter: float, linear_size: int):
            return {"method": "hamiltonian",
                    "integration": filename[5:][4:].split(".")[0],
                    "num_samples": linear_size * 2_000,
                    "delta_t": parameter,
                    "time_steps": round(1. / parameter)}

    else:
        raise ValueError(f"File {filename} has a method not recognized.")

    lattice = Phi4Lattice(linear_sites=0, mass2=0.0, coupling_strength=8.0)

    def observable(field: np.ndarray) -> np.ndarray:
        return np.abs(np.sum(field) / field.size)

    for data_key, data_set in data.items():
        for data_idx, data_point in data_set.items():
            rng = np.random.default_rng(seed=42)  # Reseed each data point for independent reproducibility.
            lattice.linear_sites = data_point["linear_size"]
            lattice.mass2 = data_point["mass2"]
            params_dict = set_params_dict(data_point["parameter"], data_point["linear_size"])
            initial_field = np.asarray(data_point["initial_sample"])
            observations, acceptance, last_field = lattice.sample_observable(observable=observable,
                                                                             initial_field=initial_field,
                                                                             rng=rng,
                                                                             **params_dict)
            print(np.mean(acceptance))
            data[data_key][data_idx]["initial_sample"] = last_field.tolist()
            fig, ax = plt.subplots()
            ax.plot(observations)
            ax.set_title(fr"$N = {lattice.linear_sites}, m2 = {lattice.mass2}$")
            plt.savefig(f"data/figures/{method_str}_{data_key}_{data_idx}.png")

    with open(filename.split(".")[0] + "_thermalized.json", "w") as target_file:
        json.dump(data, target_file, indent=4)

    return


def check_observables():
    lattice = Phi4Lattice(linear_sites=5, mass2=-1.0, coupling_strength=8.0)
    rng = np.random.default_rng(seed=42)
    initial_field = rng.random(size=lattice.shape)

    def observable(field: np.ndarray) -> np.ndarray:
        return np.abs(np.sum(field) / field.size)

    observations, acceptance, last_field = lattice.sample_observable(observable=observable,
                                                                     initial_field=initial_field,
                                                                     num_samples=50_000,
                                                                     method="uniform",
                                                                     rng=rng,
                                                                     half_width=0.06)

    print(np.mean(acceptance))
    fig, ax = plt.subplots()
    ax.plot(observations[len(observations) // 2:])
    plt.show()

    return


if __name__ == "__main__":
    uniform_settings = {"num_samples": 10_000, "method": "uniform", "target_acceptance": 0.75}
    leapfrog_settings = {"num_samples": 3_000, "method": "hamiltonian", "integration": "leapfrog",
                         "target_acceptance": 0.75}
    # get_initial_data(**leapfrog_settings)
    # merge_files("hamiltonian", integration="leapfrog")
    # check_observables()
    burn_in_data("data/uniform.json")
