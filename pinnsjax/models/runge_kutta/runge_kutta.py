import logging

from typing import Dict, List, Union, Optional

import jax.numpy as jnp

from pinnsjax.utils import load_data_txt

log = logging.getLogger(__name__)

class RungeKutta(object):
    def __init__(self, root_dir, t1: int, t2: int, time_domain, q: int = None, dtype: str ='float32'):
        """Initialize a RungeKutta object for solving differential equations using Implicit Runge-
        Kutta methods.

        :param root_dir: Root directory where the weights data is stored.
        :param t1: Start index of the time domain.
        :param t2: End index of the time domain.
        :param time_domain: TimeDomain class representing the time domain.
        :param q: Order of the Implicit Runge-Kutta method. If not provided, it is automatically
            calculated.
        """

        self.np_dtype = jnp.dtype(dtype)
        
        self.dt = jnp.array(time_domain[t2] - time_domain[t1]).astype(self.np_dtype)
        
        if q is None:
            q = int(jnp.ceil(0.5 * jnp.log(jnp.finfo(float).eps) / jnp.log(dt)))

        self.load_irk_weights(root_dir, q)
    
    
    def load_irk_weights(self, root_dir, q: int) -> None:
        """Load the weights and coefficients for the Implicit Runge-Kutta method and save in the
        dictionary.

        :param root_dir: Root directory where the weights data is stored.
        :param q: Order of the Implicit Runge-Kutta method.
        """
        file_name = "Butcher_IRK%d.txt" % q
        tmp = load_data_txt(root_dir, file_name)

        weights = jnp.reshape(tmp[0 : q**2 + q], (q + 1, q)).astype(self.np_dtype)
        
        self.alpha = jnp.array(weights[0:-1, :].T, dtype=self.np_dtype)
        self.beta = jnp.array(weights[-1:, :].T, dtype=self.np_dtype)
        self.weights = jnp.array(weights.T, dtype=self.np_dtype)
        self.IRK_times = tmp[q**2 + q :]

    
    def __call__(self,
                outputs,
                mode: str,
                solution_names: List[str],
                collection_points_names: List[str]):
        """Perform a forward step using the Runge-Kutta method for solving differential equations.

        :param outputs: Dictionary containing solution tensors and other variables.
        :param mode: The mode of the forward step, e.g., "inverse_discrete_1",
            "inverse_discrete_2", "forward_discrete".
        :param solution_names: List of keys for solution variables.
        :param collection_points_names: List of keys for collection point variables.
        :return: Dictionary with updated solution tensors after the forward step.
        """

        for solution_name, collection_points_name in zip(solution_names, collection_points_names):
            if mode == "inverse_discrete_1":
                outputs[solution_name] = outputs[solution_name] - self.dt * jnp.matmul(
                    outputs[collection_points_name], self.alpha
                )

            elif mode == "inverse_discrete_2":
                outputs[solution_name] = outputs[solution_name] + self.dt * jnp.matmul(
                    outputs[collection_points_name], (self.beta - self.alpha)
                )

            elif mode == "forward_discrete":
                outputs[solution_name] = outputs[solution_name] - self.dt * jnp.matmul(
                    outputs[collection_points_name], self.weights
                )

        return outputs


if __name__ == "__main__":
    _ = RungeKutta(None, None, None, None)

            