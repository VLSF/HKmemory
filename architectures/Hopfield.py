import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import relu, sigmoid

class Lagrange_tanh(eqx.Module):
    def get_g(self, state):
        return jnp.tanh(state)

    def get_L(self, state):
        return jnp.sum(jnp.log(jnp.cosh(state)))

class Lagrange_sigmoid(eqx.Module):
    def get_g(self, state):
        return sigmoid(state)

    def get_L(self, state):
        return jnp.sum(jnp.log(1 + jnp.exp(state)))

class Lagrange_relu(eqx.Module):
    def get_g(self, state):
        return relu(state)

    def get_L(self, state):
        return jnp.sum(relu(state)**2)/2

class Hopfield_dense(eqx.Module):
    weights: jnp.array
    bias: jnp.array
    LNet: eqx.Module

    def __init__(self, N_features, Lagrange_net, key, eps=1.0):
        keys = random.split(key)
        self.weights = eps*random.normal(keys[0], (N_features, N_features)) / jnp.sqrt(N_features)
        self.bias = random.normal(keys[1], (N_features,)) / jnp.sqrt(N_features)
        self.LNet = Lagrange_net()

    def __call__(self, t, state, args):
        g = self.LNet.get_g(state)
        f = (self.weights + self.weights.T) @ g / 2 - state + self.bias
        return f

    def energy(self, state, args):
        g = self.LNet.get_g(state)
        L = self.LNet.get_L(state)
        E = (state - self.bias) @ g - L - g @ ((self.weights + self.weights.T) @ g) / 4
        return E

def get_layer_indices(Ns):
    ind_ = jnp.arange(sum(Ns))
    ind = []
    start = 0
    for stop in Ns:
        ind.append(ind_[start:stop+start])
        start = ind[-1][-1] + 1
    return ind

class Hopfield_hierarchical_dense(eqx.Module):
    weights: list
    biases: jnp.array
    LNet: eqx.Module

    def __init__(self, N_features, Lagrange_net, key, eps=1.0):
        keys = random.split(key)
        self.weights = [eps*random.normal(key, (n_out, n_in)) / jnp.sqrt(n_out/2 + n_in/2) for key, n_in, n_out in zip(random.split(keys[0], len(N_features)-1), N_features[:-1], N_features[1:])]
        self.biases = random.normal(keys[1], (sum(N_features),)) / jnp.sqrt(sum(N_features))
        self.LNet = Lagrange_net()

    def __call__(self, t, state, ind):
        g = self.LNet.get_g(state)
        f = jnp.zeros_like(state)
        f = f.at[ind[0]].add(self.weights[0].T @ g[ind[1]] - state[ind[0]] + self.biases[ind[0]])
        for i in range(1, len(ind)-1):
            f = f.at[ind[i]].add(self.weights[i-1] @ g[ind[i-1]] + self.weights[i].T @ g[ind[i+1]] - state[ind[i]] + self.biases[ind[i]])
        f = f.at[ind[-1]].add(self.weights[-1] @ g[ind[-2]] - state[ind[-1]] + self.biases[ind[-1]])
        return f

    def energy(self, state, ind):
        g = self.LNet.get_g(state)
        L = self.LNet.get_L(state)
        E = (state - self.biases) @ g - L
        for i in range(len(ind)-1):
            E -= (self.weights[i] @ g[ind[i]]) @ g[ind[i+1]]
        return E
