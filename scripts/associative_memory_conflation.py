import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import struct
import optax
import diffrax
import equinox as eqx
import time
import os
import hashlib
import pandas as pd

from jax import random
from jax.lax import dot_general
from architectures import Hopfield, Kuramoto, Hopfield_Kuramoto
from learning import training_loops
from learning import classification
from jax.tree_util import tree_map, tree_flatten

if __name__ == "__main__":
    model_hash = "97d6d519c34377c48a8e765f56ae66cdc881bbb030ae89cdaba302cb5dcd2b67"
    base_path = "exp_results/Hopfield/"
    args_ = pd.read_csv(base_path + model_hash + "_summary")
    args = {key: args_[key].item() for key in args_.keys()}
    del args["train_accuracy"], args["test_accuracy"], args["training_time"]
    N_epoch = 15
    args["N_updates"] = 600 * N_epoch
    args["print_every"] = 200
    
    args["model_path"] = base_path + model_hash + "_model.eqx"
    args["model_name"] = "Hopfield_Kuramoto"
    args["kappa_K"] = 1.0
    args["kappa_H"] = 1.0
    args["eps_HK"] = 1e-2
    args["interaction"] = "relu"
    args["D"] = 5
    args["k"] = 150
    args["editing_type"] = "conflation"
    args["conflation_list"] = [0, 1]
    folder_path = f'exp_results/memory_editing/associative_{args["editing_type"]}'
    
    for coupling_type in ["additive", "multiplicative"]:
        args["coupling_type"] =  coupling_type
        exp_hash = "".join([ "".join(map(str, args[a])) if args[a] is list else str(args[a]) for a in sorted(args)])
        exp_hash = hashlib.sha256(str.encode(exp_hash)).hexdigest()
        base_path = f'{folder_path}/{exp_hash}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        args["write_logs_to"] = f'{base_path}_log'
        model, opt_state, losses, training_summary = training_loops.associative_Hopfield_Kuramoto(args)
        eqx.tree_serialise_leaves(base_path + "_model.eqx", model)
        eqx.tree_serialise_leaves(base_path + "_opt_state.eqx", opt_state)
        jnp.save(base_path + "_history.npy", losses)
        for k in args:
            training_summary[k] = args[k]
        with open(base_path + "_summary", "a+") as f:
            f.write(",".join([*training_summary.keys()]) + "\n")
            f.write(",".join([str(training_summary[k]) for k in training_summary.keys()]))