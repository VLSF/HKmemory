import jax.numpy as jnp
import diffrax
import equinox as eqx

from jax import vmap

def Kuramoto_data_init(features, D):
    features_ = features - jnp.mean(features, keepdims=True)
    features_ = features_ / jnp.max(features_, keepdims=True)
    features_ = jnp.stack([features_, jnp.sqrt(1 - features_**2)] + [jnp.zeros(features_.shape),]*(D-2), 2)
    features_ = jnp.pad(features_, ((0, 0), (1, 0), (0, 0)))
    features_ = features_.at[:, 0, 0].set(1)
    return features_

def Hopfield_preprocessing(feature, N_augment, N_classes):
    state = jnp.pad(feature, (0, N_augment + N_classes))
    return state

def Kuramoto_preprocessing(feature, N_augment, N_classes):
    state = jnp.pad(feature, ((0, N_augment + N_classes + 1), (0, 0)))
    state = state.at[feature.shape[0]:, 0].set(1)
    return state

def Hopfield_postprocessing(prediction, N_classes):
    prediction = prediction[-N_classes:]
    return prediction

def Kuramoto_postprocessing(prediction, N_classes):
    prediction = prediction[-N_classes-1:]
    prediction = jnp.sum(prediction[:1]*prediction[1:], axis=1)
    return prediction

def get_default_solver_data():
    solver_data = {
        "t0": 0.0,
        "t_max": 1.0,
        "dt": 1e-2,
        "solver": diffrax.Tsit5(),
        "stepsize_controller": diffrax.PIDController(rtol=1e-3, atol=1e-6),
        "max_steps": 5000
    }
    return solver_data

def compute_loss_(model, feature, target, args, solver_data, preprocessing, postprocessing):
    state = preprocessing(feature)
    prediction = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver_data["solver"],
        t0=solver_data["t0"],
        t1=solver_data["t_max"],
        dt0=solver_data["dt"],
        y0=state,
        stepsize_controller=solver_data["stepsize_controller"],
        args=args,
        max_steps=solver_data["max_steps"]
    ).ys[-1]
    
    prediction = postprocessing(prediction)
    loss = prediction + 1
    loss = loss.at[target].add(-2)
    loss = jnp.sum(loss**2)
    return loss

def compute_loss(model, features, targets, args, solver_data, preprocessing, postprocessing):
    return jnp.mean(vmap(compute_loss_, in_axes=(None, 0, 0, None, None, None, None))(model, features, targets, args, solver_data, preprocessing, postprocessing))

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, features, targets, n, args, solver_data, preprocessing, postprocessing, opt_state, optim):
    loss, grads = compute_loss_and_grads(model, features[n], targets[n], args, solver_data, preprocessing, postprocessing)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def predict_class_(model, feature, args, solver_data, preprocessing, postprocessing):
    state = preprocessing(feature)
    prediction = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver_data["solver"],
        t0=solver_data["t0"],
        t1=solver_data["t_max"],
        dt0=solver_data["dt"],
        y0=state,
        stepsize_controller=solver_data["stepsize_controller"],
        args=args,
        max_steps=solver_data["max_steps"]
    ).ys[-1]
    prediction = postprocessing(prediction)
    return jnp.argmax(prediction)

@eqx.filter_jit
def predict_class(model, features, args, solver_data, preprocessing, postprocessing):
    return vmap(predict_class_, in_axes=(None, 0, None, None, None, None))(model, features, args, solver_data, preprocessing, postprocessing)