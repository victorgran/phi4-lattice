import numpy as np


class Phi4Lattice:
    def __init__(self, linear_sites: int, mass2: float, coupling_strength: float):
        self.linear_sites = linear_sites
        self.shape = (self.linear_sites, self.linear_sites)
        self.mass2 = mass2
        self.coupling_strength = coupling_strength

        # Set specific methods for computing the laplacian and integrate the equations of motion.
        self.laplacian = self.five_stencil_laplacian
        self.integrator = self.leapfrog_integrator

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
            momentum += delta_t * force

        return field, momentum
