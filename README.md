# phi4-lattice

Implementation of the $\phi^{4}$ Euclidean theory on a lattice.

## Until next week

- [ ] Compare the results of Metropolis with a uniform proposal and Hamiltonian Monte Carlo.
This requires checking their autocorrelation times to make the proper binning for the error estimation.
Then, compute the differences $(x_{\text{U}} - x_{\text{H}}) / \sqrt{(\delta x_{\text{U}})^{2} + (\delta x_{\text{H}})^{2}}$ for the observable showing phase transition.
- [ ] Implement the Omelyan integrator for the Hamiltonian Monte Carlo, and obtain a similar plot to that of Figure 1 in _Testing and tuning symplectic integrators for Hybrid Monte Carlo algorithm in lattice QCD_.
- [ ] If time allows, check obtaining the proper binning with PyErr.

## Structure

- [ ] Create directory for preliminaries.
  - [ ] Task on computing pi via Monte Carlo integration.
  - [ ] Introduction to the Metropolis algorithm with the Gaussian target distribution.
  - [ ] Exercise 2.3 with a less trivial action.
  - [ ] Basic notes on Markov Chains, Metropolis-Hastings algorithm, etc.
  - [ ] Basic notes on the Hamiltonian Monte Carlo algorithm.
  - [ ] Redo the last exercise, exercise 2.3, with the Hamiltonian Monte Carlo algorithm.
- [ ] Phi4 theory.
  - [ ] Implementation details (preferably as a PDF document) discussing
    - Nondimensionalization.
    - From Lorentzian to Euclidean action.
    - What needs to be chosen in order to start sampling?
  - [ ] Write general code as a Python script.

## Other thoughts

- [ ] Does DifferentialEquations.jl automatically choose a symplectic integrator for the system?

## Direct dependencies so far

* numpy
* tqdm
* matplotlib
* (Whatever was installed for jupyter notebooks).
* SciPy for curve fits.
