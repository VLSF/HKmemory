import jax.numpy as jnp
import diffrax
import equinox as eqx

from jax.lax import dot_general
from jax import vmap, random

def Kuramoto_data_init(features, D):
    features_ = features - jnp.mean(features, keepdims=True)
    features_ = features_ / jnp.max(features_, keepdims=True)
    features_ = jnp.stack([features_, jnp.sqrt(1 - features_**2)] + [jnp.zeros(features_.shape),]*(D-2), 2)
    features_ = jnp.pad(features_, ((0, 0), (1, 0), (0, 0)))
    features_ = features_.at[:, 0, 0].set(1)
    return features_

def Kuramoto_data_init_random(features, D, key):
    features_ = Kuramoto_data_init(features, D)
    omega = random.normal(key, (D, D))
    Q, R = jnp.linalg.qr(omega)
    features_ = jnp.moveaxis(dot_general(Q, features_, (((1,), (2,)), ((), ()))), 0, 2)
    return features_

def Hopfield_preprocessing(feature, N_augment, N_classes):
    state = jnp.pad(feature, (0, N_augment + N_classes))
    return state

def Kuramoto_preprocessing(feature, N_augment, N_classes):
    state = jnp.pad(feature, ((0, N_augment + N_classes + 1), (0, 0)))
    state = state.at[feature.shape[0]:, 0].set(1)
    return state

def Kuramoto_preprocessing_random_I(feature, N_augment, N_classes, key):
    # works only if Kuramoto_data_init was used to prepare data
    keys = random.split(key)
    r = random.normal(keys[0], (feature.shape[0], feature.shape[1]-1,))
    r = r / jnp.linalg.norm(r, axis=1, keepdims=True)
    feature = jnp.concatenate([feature[:, :1], jnp.expand_dims(feature[:, 1], 1)*r], axis=1)
    feature_ = random.normal(keys[1], (N_augment + N_classes + 1, feature.shape[1]))
    feature_ = feature_ / jnp.linalg.norm(feature_, axis=1, keepdims=True)
    state = jnp.concatenate([feature, feature_], axis=0)
    return state

def Kuramoto_preprocessing_random_II(feature, N_augment, N_classes, key):
    # works only if Kuramoto_data_init was used to prepare data
    keys = random.split(key)
    state = Kuramoto_preprocessing_random_I(feature, N_augment, N_classes, keys[0])
    omega = jnp.linalg.qr(random.normal(keys[1], (state.shape[1], state.shape[1])))[0]
    state = state @ omega
    return state

def Kuramoto_preprocessing_random_III(feature, N_augment, N_classes, key):
    state = Kuramoto_preprocessing(feature, N_augment, N_classes)
    omega = jnp.linalg.qr(random.normal(key, (state.shape[1], state.shape[1])))[0]
    state = state @ omega
    return state

def Hopfield_Kuramoto_preprocessing(feature, N_augment, N_classes):
    state = [
        Hopfield_preprocessing(feature[0], N_augment, N_classes + 1),
        Kuramoto_preprocessing(feature[1], N_augment, N_classes)
    ]
    return state

def associative_Hopfield_Kuramoto_preprocessing(feature, N_augment, N_classes):
    state = [
        Hopfield_preprocessing(feature[0], N_augment, N_classes),
        Kuramoto_preprocessing(feature[1], N_augment, N_classes - 1)
    ]
    return state

def Hopfield_Kuramoto_preprocessing_random(feature, N_augment, N_classes, key):
    state = [
        Hopfield_preprocessing(feature[0], N_augment, N_classes + 1),
        Kuramoto_preprocessing_random_III(feature[1], N_augment, N_classes, key)
    ]
    return state

def a_Hopfield_preprocessing(feature, N_augment, N_classes):
    state = jnp.concatenate([
        Hopfield_preprocessing(feature[0], 0, 0),
        Hopfield_preprocessing(feature[1], N_augment, N_classes)
    ])
    return state

def a_Kuramoto_preprocessing(feature, N_augment, N_classes):
    state = jnp.concatenate([
        Kuramoto_preprocessing(feature[0], 0, 0),
        Kuramoto_preprocessing(feature[1], N_augment, N_classes)
    ])
    return state

def a_Kuramoto_preprocessing_random_I(feature, N_augment, N_classes, key):
    # works only if Kuramoto_data_init was used to prepare data
    keys = random.split(key)
    state = jnp.concatenate([
        Kuramoto_preprocessing_random_I(feature[0], 0, 0, keys[0]),
        Kuramoto_preprocessing_random_I(feature[1], N_augment, N_classes, keys[1])
    ])
    return state

def a_Kuramoto_preprocessing_random_II(feature, N_augment, N_classes, key):
    # works only if Kuramoto_data_init was used to prepare data
    keys = random.split(key, 3)
    state = jnp.concatenate([
        Kuramoto_preprocessing_random_II(feature[0], 0, 0, keys[0]),
        Kuramoto_preprocessing_random_II(feature[1], N_augment, N_classes, keys[1])
    ])
    omega = jnp.linalg.qr(random.normal(keys[1], (state.shape[1], state.shape[1])))[0]
    state = state @ omega
    return state

def a_Kuramoto_preprocessing_random_III(feature, N_augment, N_classes, key):
    state = a_Kuramoto_preprocessing(feature, N_augment, N_classes)
    omega = jnp.linalg.qr(random.normal(key, (state.shape[1], state.shape[1])))[0]
    state = state @ omega
    return state

def Hopfield_postprocessing(prediction, N_classes):
    prediction = prediction.ys[-1][-N_classes:]
    return prediction

def Kuramoto_postprocessing(prediction, N_classes):
    prediction = prediction.ys[-1][-N_classes-1:]
    prediction = jnp.sum(prediction[:1]*prediction[1:], axis=1)
    return prediction

def Kuramoto_postprocessing_I(prediction, N_classes):
    prediction = prediction.ys[-1][-2*N_classes:].reshape(2, N_classes, -1)
    prediction = jnp.sum(prediction[0]*prediction[1], axis=1)
    return prediction

def Hopfield_Kuramoto_postprocessing(prediction, N_classes, Hopfield=True):
    if Hopfield:
        prediction = prediction.ys[0][-1][-N_classes:]
    else:
        prediction = prediction.ys[1][-1][-N_classes-1:]
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

def compute_loss_(model, feature, target, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    state = preprocessing(feature, preprocessing_args)
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
        y0=state,
        stepsize_controller=solver_data["stepsize_controller"],
        args=args,
        max_steps=solver_data["max_steps"]
    )
    prediction = postprocessing(prediction)
    return jnp.argmax(prediction)

@eqx.filter_jit
def predict_class(model, features, args, solver_data, preprocessing, postprocessing, preprocessing_args):
    return vmap(predict_class_, in_axes=(None, 0, None, None, None, None, None))(model, features, args, solver_data, preprocessing, postprocessing, preprocessing_args)
