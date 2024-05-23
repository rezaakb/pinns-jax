import functools
import jax
import jax.numpy as jnp

from typing import Dict, List, Union, Optional, Tuple


def sse(loss,
        preds,
        target = None,
        keys = None,
        mid = None):
    """Calculate the sum of squared errors (SSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated SSE loss.
    """
    
    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + jnp.sum(jnp.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + jnp.sum(jnp.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + jnp.sum(jnp.square(preds[key] - target[key]))

    return loss

def mse(loss,
        preds,
        target = None,
        keys = None,
        mid = None):
    """Calculate the mean squared error (MSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated MSE loss.
    """
    
    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + jnp.mean(jnp.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + jnp.mean(jnp.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + jnp.mean(jnp.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds, target):
    """Calculate the relative L2 error between predictions and target tensors.

    :param preds: Predicted tensors.
    :param target: Target tensors.
    :return: Relative L2 error value.
    """
    
    return jnp.sqrt(jnp.mean(jnp.square(preds - target))/jnp.mean(jnp.square(target)))


def fix_extra_variables(trainable_variables, extra_variables):
    """Convert extra variables to tf tensors with gradient tracking. These variables are
    trainables in inverse problems.

    :param extra_variables: Dictionary of extra variables to be converted.
    :return: Dictionary of converted extra variables as tf tensors with gradients.
    """
    
    if extra_variables is None:
        return trainable_variables, None
    extra_variables_dict = {}
    for key in extra_variables:
        variable = jnp.array(extra_variables[key])
        extra_variables_dict[key] = variable
        trainable_variables[key] = variable
    return trainable_variables, extra_variables_dict


def make_functional(net, params, n_dim, discrete, output_fn):
    """Make model functional based on number of dimension.

    :param net: The neural network model.
    :param params: The parameters of the model.
    :param n_dim: The number of dimensions.
    :param output_fn: Output function applied to the output.
    :return: A functional model and shape in axes.
    """

    def functional_model_1d(params, x, time, output_c=None):            
        return _execute_model(net, params, [x], time, output_c)

    def functional_model_2d(params, x, y, time, output_c=None):
        return _execute_model(net, params, [x, y], time, output_c)

    def functional_model_3d(params, x, y, z, time, output_c=None):
        return _execute_model(net, params, [x, y, z], time, output_c)

    functional_model_1d.discrete = discrete
    functional_model_2d.discrete = discrete
    functional_model_3d.discrete = discrete

    if discrete:
        functional_model_1d.in_axes_discrete = (None, 0, None, None)
        functional_model_1d.in_axes = (None, None, 0, 0)
        functional_model_2d.in_axes_discrete = (None, 0, 0, None, None)
        functional_model_2d.in_axes = (None, None, 0, 0, 0)
        functional_model_3d.in_axes_discrete = (None, 0, 0, 0, None, None)
        functional_model_3d.in_axes = (None, None, 0, 0, 0, 0)
    else:
        functional_model_1d.in_axes = (None, None, 0, 0, 0)
        functional_model_1d.in_axes_gard = (None, 0, 0, None)
        functional_model_2d.in_axes = (None, None, 0, 0, 0, 0)
        functional_model_2d.in_axes_gard = (None, 0, 0, 0, None)
        functional_model_3d.in_axes = (None, None, 0, 0, 0, 0, 0)
        functional_model_3d.in_axes_gard = (None, 0, 0, 0, 0, None)

    def _execute_model(net, params, inputs, time, output_c):
        outputs_dict = net(params, inputs, time)
    
        if output_c is None:
            return outputs_dict
        else:
            return outputs_dict[output_c].squeeze()

    models = {
        2: functional_model_1d,
        3: functional_model_2d,
        4: functional_model_3d,
    }
    
    functional_model_fun = models[n_dim]

    
    try:
        return models[n_dim]
    except KeyError:
        raise ValueError(f"{n_dim} number of dimensions is not supported.")
    '''

    #output_fn = output_fn if output_fn is None else functools.partial(jax.vmap,
    #                                                                  in_axes=functional_model_fun.in_axes)(output_fn)

    def functional_model(params, x, y, z, time, output_c=None):

        outputs = functional_model_fun(params, x, y, z, time)
        
        #return outputs
        #if output_fn:
        #    outputs = output_fn(functional_model_fun,  params, outputs, x, time)

        if output_c is None:
            return outputs
        else:
            return [outputs[output_].squeeze() for output_ in output_c]
            
    if discrete:
        functional_model.in_axes = functional_model_fun.in_axes
        functional_model.in_axes_discrete = functional_model_fun.in_axes_discrete
        functional_model.discrete = discrete
    else:
        functional_model.in_axes = functional_model_fun.in_axes
        functional_model.in_axes_gard = functional_model_fun.in_axes_gard
        functional_model.discrete = discrete

    return functional_model
    '''