import jax.numpy as jnp
import equinox as eqx
import networkx as nx

from jax import random
from jax.nn import relu, sigmoid, gelu

class linear_interaction(eqx.Module):
    weights: jnp.array

    def __init__(self, N_weights, eps, key):
        self.weights = eps*random.normal(key, (N_weights,))

    def __call__(self, products):
        return self.weights

    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(self.weights*products)

class relu_interaction(eqx.Module):
    A: jnp.array
    B: jnp.array

    def __init__(self, N_weights, eps, key):
        keys = random.split(key)
        self.A = eps*random.normal(keys[0], (N_weights,))
        self.B = eps*random.normal(keys[1], (N_weights,))

    def __call__(self, products):
        return self.A + self.B*relu(products)

    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(self.A*products) + jnp.sum(self.B*relu(products)**2/2)

class sigmoid_interaction(eqx.Module):
    A: jnp.array
    B: jnp.array

    def __init__(self, N_weights, eps, key):
        keys = random.split(key)
        self.A = eps*random.normal(keys[0], (N_weights,))
        self.B = eps*random.normal(keys[1], (N_weights,))

    def __call__(self, products):
        return self.A + self.B*sigmoid(products)

    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(self.A*products) + (jnp.sum(self.B*products) + jnp.sum(self.B*jnp.log(1 + jnp.exp(-products))))

class tanh_interaction(eqx.Module):
    A: jnp.array
    B: jnp.array

    def __init__(self, N_weights, eps, key):
        keys = random.split(key)
        self.A = eps*random.normal(keys[0], (N_weights,))
        self.B = eps*random.normal(keys[1], (N_weights,))

    def __call__(self, products):
        return self.A + self.B*jnp.tanh(products)

    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(self.A*products) + jnp.sum(self.B*jnp.log(jnp.cosh(products)))

class MLP_GELU(eqx.Module):
    weights: list
    biases: list

    def __init__(self, NN_shapes, key):
        self.weights = [random.normal(key, (N_out, N_in))/jnp.sqrt(N_in/2 + N_out/2) for key, N_in, N_out in zip(keys, NN_shapes[:-1], NN_shapes[1:])]
        self.biases = [jnp.zeros((N_out, )) for key, N_in, N_out in zip(keys, NN_shapes[:-1], NN_shapes[1:])]

    def __call__(self, x):
        x = self.weights[0] @ x + self.biases[0]
        for w, b in zip(self.weights[1:], self.biases[1:]):
            x = gelu(x)
            x = w @ x + b
        return x[0]

    def d_call(self, x):
        return grad(self.__call__)(x)

class deep_GELU_interaction_I(eqx.Module):
    A: jnp.array
    MLP: eqx.Module

    def __init__(self, NN_shapes, N_weights, eps, key):
        NN_shapes[0] = NN_shapes[-1] = 1
        keys = random.split(key)
        self.MLP = MLP_GELU(NN_shapes, keys[0])
        self.A = eps*random.normal(keys[1], (N_weights,))

    def __call__(self, products):
        return vmap(grad(self.MLP))(jnp.expand_dims(products, 1))[:, 0]*self.A
    
    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(vmap(self.MLP)(jnp.expand_dims(products, 1))*self.A)

class deep_GELU_interaction_II(eqx.Module):
    MLP: eqx.Module

    def __init__(self, NN_shapes, N_weights, eps, key):
        NN_shapes[0] = NN_shapes[-1] = 1
        self.MLP = vmap(MLP_GELU, in_axes=(None, 0))(NN_shapes, random.split(key, N_weights))

    def __call__(self, products):
        return vmap(lambda m, x: m(x), in_axes=0)(self.MLP.d_call, jnp.expand_dims(products, 1))[:, 0]
    
    def energy(self, state, ind):
        products = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        return jnp.sum(vmap(lambda m, x: m(x), in_axes=0)(self.MLP, jnp.expand_dims(products, 1)))

def get_small_world_connectivity(key, N_neurons, k=4, p=0.1):
    g = nx.connected_watts_strogatz_graph(N_neurons, k, p, seed=key[1].item())
    ind = jnp.array(g.edges())
    return ind

class Kuramoto_global(eqx.Module):
    interaction: eqx.Module

    def __init__(self, N_weights, interaction, eps, key, NN_shapes=None):
        keys = random.split(key)
        if NN_shapes is None:
            self.interaction = interaction(N_weights, eps, keys[0])
        else:
            self.interaction = interaction(NN_shapes, N_weights, eps, keys[0])

    def __call__(self, t, state, ind):
        state = state / jnp.linalg.norm(state, axis=1, keepdims=True)
        s = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        dE_ds = jnp.expand_dims(self.interaction(s), 1)
        state_ = jnp.zeros_like(state)
        state_ = state_.at[ind[:, 0]].add(dE_ds * state[ind[:, 1]])
        state_ = state_.at[ind[:, 1]].add(dE_ds * state[ind[:, 0]])
        state_ = -state_ + state * jnp.sum(state * state_, axis=1, keepdims=True)
        return state_

    def energy(self, state, ind):
        return self.interaction.energy(state, ind)

class Kuramoto_local(eqx.Module):
    interaction: eqx.Module
    omegas: jnp.array

    def __init__(self, D, N_neurons, N_weights, interaction, eps, key, NN_shapes=None):
        keys = random.split(key)
        if NN_shapes is None:
            self.interaction = interaction(N_weights, eps, keys[0])
        else:
            self.interaction = interaction(NN_shapes, N_weights, eps, keys[0])
        self.omegas = random.normal(keys[1], (N_neurons, D, D)) / jnp.sqrt(D)

    def __call__(self, t, state, ind):
        state = state / jnp.linalg.norm(state, axis=1, keepdims=True)
        s = jnp.sum(state[ind[:, 0]] * state[ind[:, 1]], axis=1)
        dE_ds = jnp.expand_dims(self.interaction(s), 1)
        state_ = jnp.zeros_like(state)
        state_ = state_.at[ind[:, 0]].add(dE_ds * state[ind[:, 1]])
        state_ = state_.at[ind[:, 1]].add(dE_ds * state[ind[:, 0]])
        state_ = -state_ + state * jnp.sum(state * state_, axis=1, keepdims=True)
        omegas_ = (self.omegas - jnp.moveaxis(self.omegas, 1, 2))/2
        omegas_ = omegas_ - jnp.mean(omegas_, axis=0, keepdims=True)
        state_ = state_ + dot_general(omegas_, state, (((2,), (1,)), ((0,), (0,))))
        return state_

    def energy(self, state, ind):
        return self.interaction.energy(state, ind)