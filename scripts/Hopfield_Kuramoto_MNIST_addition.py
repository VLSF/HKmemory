import numpy as np
import jax.numpy as jnp
import struct
import optax
import diffrax
import equinox as eqx
import time
import os
import hashlib

from jax import random
from jax.lax import dot_general
from architectures import Hopfield, Kuramoto, Hopfield_Kuramoto
from learning import training_loops
from learning import classification
from jax.tree_util import tree_map, tree_flatten

if __name__ == "__main__":
    args = training_loops.get_standard_args()
    args["model_name"] = "Hopfield_Kuramoto"
    args["dataset_name"] = "MNIST_addition"
    args["D"] = 5
    N_epoch = 15
    args["N_updates"] = 600 * N_epoch
    args["print_every"] = 200
    args["interaction"] = "relu"
    args["activation"] = "relu"
    args["kappa_K"] = 1.0
    args["kappa_H"] = 1.0
    args["eps_HK"] = 1e-2
    folder_path = f'exp_results/{args["model_name"]}_{args["dataset_name"]}'
    for N_augment in [0, 100, 200]:
        args["N_augment"] = N_augment
        for k_ in [150, 250, 350]:
            args["k"] = k_
            exp_hash = "".join([str(args[a]) for a in sorted(args)])
            exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest() 
            base_path = f'{folder_path}/{exp_hash}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            args["write_logs_to"] = f'{base_path}_log'
            model, opt_state, losses, training_summary = training_loops.Hopfield_Kuramoto_small_world(args)
            eqx.tree_serialise_leaves(base_path + "_model.eqx", model)
            eqx.tree_serialise_leaves(base_path + "_opt_state.eqx", opt_state)
            jnp.save(base_path + "_history.npy", losses)
            for k in args:
                training_summary[k] = args[k]
            with open(base_path + "_summary", "a+") as f:
                f.write(",".join([*training_summary.keys()]) + "\n")
                f.write(",".join([str(training_summary[k]) for k in training_summary.keys()]))