# To Do's

- [ ] For the Hamiltonian Monte Carlo:
  - [ ] For a fixed mass (-1), obtain the time step sizes for different lattice sizes that give a target acceptance.
  - [ ] For two given (large) lattice sizes, obtain the time step sizes for different values of the squared mass that give a target acceptance.
  - [ ] For all of the above parameters, thermalize the field configurations and record the last thermalized states to start generating samples.
  - [ ] Generate a _large_ number of samples.
- [ ] For the uniform proposal:
  - [ ] For a fixed mass (-1), obtain the half-widths for different lattice sizes that give a target acceptance.
  - [ ] For two given (large) lattice sizes, obtain the half-widths for different values of the squared mass that give a target acceptance.
  - [ ] For all of the above parameters, thermalize the field configurations and record the last thermalized states to start generating samples.
  - [ ] Generate a _large_ number of samples.

---

## Tasks

- [x] Create a function to obtain the value of a single parameter based on finding the target acceptance.
- [x] Merge all 3 files into a single .json file.
Keywords are “finite_volume”, “phase_transition1”, “phase_transition2”.
- [ ] Create a function that
  - Loads the parameters from the file.
  - Runs the sampling for a given number of samples, obtaining the value of the observable and storing it.
  - Updates the initial sample to the last sample generated in this run.
  - Generates figures of the observable value at each Monte Carlo step, to check for thermalization.