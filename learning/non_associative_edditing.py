import jax.numpy as jnp
import diffrax
import equinox as eqx

from jax.lax import dot_general
from jax import vmap, random

def get_default_solver_data():
    solver_data = {
        "t0": 0.0,
        "t_max": 1.0,
        "dt": 1e-2,
        "solver": diffrax.Tsit5(),
        #"stepsize_controller": diffrax.ConstantStepSize(),
        "stepsize_controller": diffrax.PIDController(rtol=1e-3, atol=1e-6),
        "max_steps": 5000
    }
    return solver_data

def compute_loss_(model, feature, target, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    state = preprocessing(feature, preprocessing_args)
    prediction = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver_data["solver"],
        t0=solver_data["t0"],
        t1=solver_data["t_max"],
        dt0=solver_data["dt"],
        y0=[state, model.inp / jnp.linalg.norm(model.inp, axis=1, keepdims=True)],
        stepsize_controller=solver_data["stepsize_controller"],
        args=args,
        max_steps=solver_data["max_steps"]
    )
    
    prediction = postprocessing(prediction)
    loss = prediction + 1
    loss = loss.at[target].add(-2)
    loss = jnp.sum(loss**2)
    return loss

def compute_loss(model, features, targets, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    return jnp.mean(vmap(compute_loss_, in_axes=(None, 0, 0, None, None, None, None, None))(model, features, targets, args, solver_data, preprocessing, postprocessing, preprocessing_args))

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, features, targets, args, solver_data, preprocessing, postprocessing, preprocessing_args, opt_state, optim):
    loss, grads = compute_loss_and_grads(model, features, targets, args, solver_data, preprocessing, postprocessing, preprocessing_args)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def predict_class_(model, feature, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    state = preprocessing(feature, preprocessing_args)
    prediction = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver_data["solver"],
        t0=solver_data["t0"],
        t1=solver_data["t_max"],
        dt0=solver_data["dt"],
        y0=[state, model.inp / jnp.linalg.norm(model.inp, axis=1, keepdims=True)],
        stepsize_controller=solver_data["stepsize_controller"],
        args=args,
        max_steps=solver_data["max_steps"]
    )
    prediction = postprocessing(prediction)
    return jnp.argmax(prediction)

@eqx.filter_jit
def predict_class(model, features, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    return vmap(predict_class_, in_axes=(None, 0, None, None, None, None, None))(model, features, args, solver_data, preprocessing, postprocessing, preprocessing_args)