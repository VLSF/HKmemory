import jax.numpy as jnp
import equinox as eqx

from jax import random

class Hopfield_Kuramoto_network(eqx.Module):
    interaction_K: eqx.Module
    LNet: eqx.Module
    weights_H: jnp.array
    bias_H: jnp.array
    weights_HK: jnp.array

    def __init__(self, N_weights, interaction, N_features, N_weights_i, key, Lagrange_net, eps=1.0, NN_shapes=None):
        keys = random.split(key, 4)
        if NN_shapes is None:
            self.interaction_K = interaction(N_weights, keys[0])
        else:
            self.interaction_K = interaction(NN_shapes, N_weights, keys[0])
        self.LNet = Lagrange_net()
        self.weights_H = eps*random.normal(keys[1], (N_features, N_features)) / jnp.sqrt(N_features)
        self.bias_H = random.normal(keys[2], (N_features,)) / jnp.sqrt(N_features)
        self.weights_HK = random.normal(key, (N_weights_i, 1))

    def __call__(self, t, state, args):
        ind_K, ind_HK, kappa_K, kappa_H = args
        state_H, state_K = state
        state_K = state_K / jnp.linalg.norm(state_K, axis=1, keepdims=True)
        g = self.LNet.get_g(state_H)
        f_H = jnp.zeros_like(g)
        Gram = jnp.sum(state_K[ind_HK[:, 0]] * state_K[ind_HK[:, 1]], axis=1)
        f_H = f_H.at[ind_HK[:, 0]].add(Gram * self.weights_HK[:, 0] * g[ind_HK[:, 1]] / kappa_H)
        f_H = f_H.at[ind_HK[:, 1]].add(Gram * self.weights_HK[:, 0] * g[ind_HK[:, 0]] / kappa_H)
        f_H = f_H + (self.weights_H + self.weights_H.T) @ g/2 - state_H + self.bias_H
        
        s = jnp.sum(state_K[ind_K[:, 0]] * state_K[ind_K[:, 1]], axis=1)
        dE_ds = jnp.expand_dims(self.interaction_K(s), 1)
        f_K = jnp.zeros_like(state_K)
        f_K = f_K.at[ind_K[:, 0]].add(dE_ds * state_K[ind_K[:, 1]])
        f_K = f_K.at[ind_K[:, 1]].add(dE_ds * state_K[ind_K[:, 0]])
        G = jnp.expand_dims(g[ind_HK[:, 0]]*g[ind_HK[:, 1]], 1)
        f_K = f_K.at[ind_HK[:, 0]].add(-G * self.weights_HK * state_K[ind_HK[:, 1]] / kappa_K)
        f_K = f_K.at[ind_HK[:, 1]].add(-G * self.weights_HK * state_K[ind_HK[:, 0]] / kappa_K)
        f_K = -f_K + state_K * jnp.sum(state_K * f_K, axis=1, keepdims=True)
        return [f_H, f_K]

    def energy(self, state, args):
        ind_K, ind_HK, kappa_K, kappa_H = args
        state_H, state_K = state
        E_K = self.interaction_K.energy(state_K, ind_K)
        g = self.LNet.get_g(state_H)
        L = self.LNet.get_L(state_H)
        E_H = (state_H - self.bias_H) @ g - L - g @ ((self.weights_H + self.weights_H.T) @ g) / 4
        E_HK = -jnp.sum(jnp.sum(state_K[ind_HK[:, 0]] * state_K[ind_HK[:, 1]], axis=1) * g[ind_HK[:, 0]] * g[ind_HK[:, 1]] * self.weights_HK[:, 0])
        return E_K*kappa_K + E_H*kappa_H + E_HK