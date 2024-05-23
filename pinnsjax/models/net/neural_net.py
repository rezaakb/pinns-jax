from typing import List, Dict

import jax
import jax.numpy as jnp

class FCN(object):
    def __init__(self, layers, lb, ub, output_names, discrete: bool = False, dtype='float32') -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        :param discrete: If the problem is discrete or not.
        """
        super().__init__()

        self.lb = jnp.array(lb, jnp.float32)
        self.ub = jnp.array(ub, jnp.float32)
        self.n_dim = len(self.lb)

        rng = jax.random.PRNGKey(1234)
        
        self.params = self.initialize_net(rng, layers)
        self.output_names = output_names
        self.discrete = discrete
        #self.forward = jax.vmap(self.forward, in_axes=(None, 0))
    
    def xavier_init(self, rng, size):
        in_dim, out_dim = size
        xavier_stddev = jnp.sqrt(2.0 / (in_dim + out_dim))
        
        return jax.random.normal(rng, (in_dim, out_dim)) * xavier_stddev

    def initialize_net(self, rng, layers):
        num_layers = len(layers)
        weights = []
        biases = []
        for l in range(0, num_layers - 1):
            rng, layer_rng = jax.random.split(rng)
            W = self.xavier_init(layer_rng, (layers[l], layers[l + 1]))
            b = jnp.zeros((1, layers[l + 1]))
            weights.append(W)
            biases.append(b)
        params = {'weights': weights,
                  'biases': biases}
        return params
    
    def forward(self, params, X):
        if self.discrete:
            H = 2.0 * (X - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0
        else:
            H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
            
        for i, (w, b) in enumerate(zip(params['weights'], params['biases'])):
            if i == len(params) - 1:
                H = jnp.dot(H, w) + b
            else:
                H = jax.nn.tanh(jnp.dot(H, w) + b)

        return H
        
    def __call__(self, params, spatial, time):
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """
        
        if self.discrete is False:
            z = jnp.hstack([*spatial, time])
        else:
            z = jnp.hstack([*spatial])
                
        z = self.forward(params, z)
        
        # Discrete Mode
        if self.discrete:
            outputs_dict = {name: z for i, name in enumerate(self.output_names)}

        # Continuous Mode
        else:
            outputs_dict = {name: z[:, i : i + 1] for i, name in enumerate(self.output_names)}
        
        return outputs_dict 



class NetHFM(object):
    """
    In this model, mean and std will be used for normalization of input data. Also, weight
    normalization will be done.
    """
    output_names: List[str]
    
    def __init__(self, mean, std, layers: List, output_names: List, discrete=False):
        super().__init__()
        """Initialize a `NetHFM` module.

        :param mesh: The number of layers.
        :param layers: The list indicating number of neurons in each layer.
        :param output_names: Names of outputs of net.
        """
        self.num_layers = len(layers)
        self.output_names = output_names
        self.trainable_variables = []

        rng = jax.random.PRNGKey(0)
        
        self.discrete = discrete
        
        self.X_mean =jnp.array(mean, dtype=jnp.float32)
        self.X_std = jnp.array(std, dtype=jnp.float32)
        print(mean[0])
        self.n_dim = len(mean[0])

        self.params = self.initalize_net(rng, layers)
        
    def initalize_net(self, rng, layers: List) -> None:
        """Initialize the neural network weights, biases, and gammas.

        :param layers: The list indicating number of neurons in each layer.
        """
        
        weights = []
        biases = []
        gammas = []
        
        for l in range(0,self.num_layers-1):
            in_dim = layers[l]
            out_dim = layers[l+1]
            rng, layer_rng = jax.random.split(rng)
            W = jax.random.normal(rng, (in_dim, out_dim))
            b = jnp.zeros([1, out_dim])
            g = jnp.ones([1, out_dim])
            
            weights.append(W)
            biases.append(b)
            gammas.append(g)

        params = {'weights': weights,
                  'biases': biases,
                  'gammas': gammas}
        return params

    
    def forward(self, params, H):
    
        H = (H - self.X_mean) / self.X_std
            
        for i, (W, b, g) in enumerate(zip(params['weights'], params['biases'], params['gammas'])):
            # weight normalization
            V = W / jnp.linalg.norm(W, axis=0, keepdims=True)

            # matrix multiplication
            H = jnp.dot(H, V)
            
            # add bias
            H = g * H + b
            
            # activation
            if i < self.num_layers - 2:
                H = H * jax.nn.sigmoid(H)
        return H

    
    def __call__(self, params, spatial: List[jax.Array], time: jax.Array) -> Dict[str, jax.Array]:
        """Perform a forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A dictionary with output names as keys and corresponding output tensors as values.
        """
        
        if self.discrete is False:
            H = jnp.hstack([*spatial, time])
        else:
            H = jnp.hstack([*spatial])
            
        H = self.forward(params, H)
        
        outputs_dict = {name: H[:, i : i + 1] for i, name in enumerate(self.output_names)}

        return outputs_dict


if __name__ == "__main__":
    _ = FCN()
    _ = NetHFM()