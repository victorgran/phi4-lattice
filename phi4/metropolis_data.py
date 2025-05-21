import numpy as np

from phi4.implementation.lattice import Phi4Lattice


def acceptance_width_size(data_file: str) -> None:
    """
    Get mean acceptance rates for different lattice sizes and half-widths of the uniform distribution.

    Parameters
    ----------
    data_file : str
        Filename to save data to.

    Returns
    -------

    """
    linear_sizes = np.linspace(5, 50, num=20, endpoint=True, dtype=int)
    half_widths = np.linspace(0.001, 0.1, num=20, endpoint=True, dtype=float)
    num_samples = 50_000

    acceptance_grid = []

    for linear_size in linear_sizes:
        print(f"Linear size: {linear_size}")
        rng = np.random.default_rng(seed=42)
        lattice = Phi4Lattice(linear_sites=linear_size, mass2=-1.0, coupling_strength=8.0)
        initial_sample = rng.random(size=lattice.shape)
        mean_acceptances = []
        for half_width in half_widths:
            _, acceptances = lattice.sample(initial_sample=initial_sample,
                                            num_samples=num_samples,
                                            method='uniform',
                                            rng=rng,
                                            half_width=half_width)
            mean_acceptances.append(np.mean(acceptances))
        acceptance_grid.append(np.array(mean_acceptances))

    np.savez(data_file, acceptance_grid=np.array(acceptance_grid), linear_sizes=linear_sizes, half_widths=half_widths,
             num_samples=num_samples)
    return


def show_acceptance():
    import matplotlib.pyplot as plt

    data = np.load("data/metro_mean_acceptances.npz")
    acceptance_grid = data["acceptance_grid"]  # Each row is a specific lattice size, each column a half-width.
    linear_sizes = data["linear_sizes"]
    half_widths = data["half_widths"]
    # num_samples = data["num_samples"]

    sizes_grid, widths_grid = np.meshgrid(linear_sizes, half_widths)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d', elev=29., azim=31., roll=-3)
    ax.plot_wireframe(sizes_grid, widths_grid, acceptance_grid)
    ax.set_xlabel("Lattice size", fontsize=14, labelpad=7)
    ax.set_ylabel("Half-width", fontsize=14, labelpad=7)
    ax.set_zlabel("Mean acceptance", fontsize=14, labelpad=3)
    plt.show()

    return


if __name__ == "__main__":
    print("Uncomment function to generate data.")
    # acceptance_width_size("data/metro_mean_acceptances.npz")
    show_acceptance()
