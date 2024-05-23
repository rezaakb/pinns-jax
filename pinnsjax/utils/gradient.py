import jax

def gradient(functional_model, argnums, order=1):
    grad_functional_model = functional_model
    for i in range(order):
        grad_functional_model = jax.jacrev(grad_functional_model, argnums=argnums)
    if functional_model.discrete:
        grad_functional_model = jax.vmap(grad_functional_model, in_axes=functional_model.in_axes_discrete)
    else:
        grad_functional_model = jax.vmap(grad_functional_model, in_axes=functional_model.in_axes_gard)
    return grad_functional_model

def hessian(functional_model, argnums):
    return jax.hessian(functional_model, argnums = argnums)

def jacrev(functional_model, argnums):
    return jax.jacrev(functional_model, argnums = argnums)

def jacfwd(functional_model, argnums):
    return jax.jacfwd(functional_model, argnums = argnums)
    
def fwd_gradient(functional_model, argnums, order):
    grad_functional_model = functional_model
    for i in range(order):
        grad_functional_model = jax.jacfwd(grad_functional_model, argnums=argnums)
    if functional_model.discrete:
        grad_functional_model = jax.vmap(grad_functional_model, in_axes=functional_model.in_axes_discrete)
    
    return grad_functional_model