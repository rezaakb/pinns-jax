from typing import List, Dict, Any, Tuple, Union

import jax
import optax
import functools

from pinnsjax.utils import (
    fix_extra_variables,
    mse,
    relative_l2_error,
    sse,
    make_functional
)

class PINNModule:
    def __init__(
        self,
        net,
        pde_fn,
        optimizer = optax.adam,
        loss_fn: str = "sse",
        extra_variables: Dict[str, Any] = None,
        output_fn = None,
        runge_kutta = None,
        jit_compile: bool = True,
    ) -> None:
        """Initialize a `PINNModule`.

        :param net: The model to train.
        :param pde_fn: PDE function.
        :param optimizer: The optimizer to use for training.
        :param loss_fn: PDE function will apply on the collection points.
        :param output_fn: Output function will apply on the output of the net.
        :param runge_kutta: Runge–Kutta method will be used in discrete problems.
        :param extra_variables: Extra variables should be in a dictionary.
        :param jit_compile: Whether to use JIT compiler.
        """
        super().__init__()
        
        self.net = net

        self.trainable_variables = self.net.params
        
        (self.trainable_variables,
         self.extra_variables) = fix_extra_variables(self.trainable_variables, extra_variables)

        self.functional_net = make_functional(net = net,
                                              params = self.trainable_variables,
                                              n_dim = self.net.n_dim,
                                              discrete = self.net.discrete,
                                              output_fn = output_fn
                                              )
        
     
        self.pde_fn = functools.partial(jax.vmap, in_axes=self.functional_net.in_axes)(pde_fn)     
        
        self.rk = runge_kutta
        if isinstance(optimizer, functools.partial):
            self.opt = optimizer()
        else:
            self.opt = optimizer(learning_rate=1e-3)
        self.opt_state = self.opt.init(self.trainable_variables)
    
        if jit_compile:
            self.train_step = functools.partial(jax.jit)(self.train_step)

        if loss_fn == "sse":
            self.loss_fn = sse
        elif loss_fn == "mse":
            self.loss_fn = mse
        
        self.time_list = []
        self.functions = {
            "runge_kutta": self.rk,
            "forward": self.forward,
            "functional_net": self.functional_net,
            "pde_fn": self.pde_fn,
            "net": net,
            "extra_variables": self.extra_variables,
            "loss_fn": self.loss_fn,
        }

    def forward(self, params, spatial: List[jax.Array], time: jax.Array, output_c = None) -> Dict[str, jax.Array]:
        """Perform a forward pass through the model `self.net`.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """
        
        outputs = self.functional_net(params, *spatial, time, output_c)

        return outputs
    
    def model_step(
        self,
        params,
        batch: Dict[
            str,
            Union[
                Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, jax.Array]
            ],
        ],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the
        input tensor of different conditions and data.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A dictionary of predictions.
        """

        loss = 0.0
        for loss_fn_name, data in batch.items():
            loss, preds = self.function_mapping[loss_fn_name](params, data, loss, self.functions)
        
        return loss, preds

    def train_step(self, params, optimizer_state, batch):

        loss_pred, grads = jax.value_and_grad(self.model_step, has_aux=True)(params, batch)
        loss, _ = loss_pred
        updates, opt_state = self.opt.update(grads, optimizer_state)        
                            
        return (loss, self.extra_variables,
                optax.apply_updates(params, updates), opt_state)

    def eval_step(
        self, batch
    ) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array]]:
        """Perform a single evaluation step on a batch of data.

        :param batch: A batch of data containing input tensors and conditions.
        :return: A tuple containing loss, error dictionary, and predictions.
        """

        _, _, u = list(batch.values())[0]

        eval_model = functools.partial(self.model_step, self.trainable_variables)

        loss, preds = eval_model(batch)

        if self.rk:
            error_dict = {
                solution_name: relative_l2_error(
                    preds[solution_name][:, -1][:, None], u[solution_name]
                )
                for solution_name in self.val_solution_names
            }

        else:
            error_dict = {
                solution_name: relative_l2_error(preds[solution_name], u[solution_name])
                for solution_name in self.val_solution_names
            }
                
        return loss, error_dict, preds
   
    def validation_step(self, batch):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict).
        """
        
        loss, error_dict, _ = self.eval_step(batch)
        
        return loss, error_dict    
    
    def test_step(self, batch):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict).
        """

        loss, error, _ = self.eval_step(batch)

        return loss, error
                
    def predict_step(self, batch):
        """Perform a single predict step on a batch of data from the prediction set.

        :param batch: A batch of data (a dict).
        """

        _, _, preds = self.eval_step(batch)

        return preds
                