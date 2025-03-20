import numpy as np
import jax.numpy as jnp
import struct
import optax
import diffrax
import equinox as eqx
import time
import sys

from jax import random
from jax.lax import dot_general
from architectures import Hopfield, Kuramoto, Hopfield_Kuramoto
from learning import classification, non_associative_edditing
from jax.tree_util import tree_map, tree_flatten

def load_MNIST(path_to_MNIST):
    with open(f'{path_to_MNIST}/raw/train-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        features = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        features = jnp.array(features.reshape((size, nrows*ncols)) / 255)
    
    with open(f'{path_to_MNIST}/raw/t10k-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        features_ = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        features_ = jnp.array(features_.reshape((size, nrows*ncols)) / 255)
    
    features = jnp.concatenate([features, features_], axis=0)
    
    with open(f'{path_to_MNIST}/raw/train-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        targets = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        targets = jnp.array(targets.reshape((size,)))
    
    with open(f'{path_to_MNIST}/raw/t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        targets_ = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        targets_ = jnp.array(targets_.reshape((size,)))
    
    targets = jnp.concatenate([targets, targets_], axis=0)
    return features, targets

def conflate_digits(targets, conflation_list):
    d0 = conflation_list[0]
    for d in conflation_list[1:]:
        targets = targets.at[targets == d].set(d0)
    return targets

def swap_digits(targets, digits_a, digits_b):
    for a, b in zip(digits_a, digits_b):
        mask_a = targets == a
        mask_b = targets == b
        targets = targets.at[mask_a].set(b)
        targets = targets.at[mask_b].set(a)
    return targets

def prepare_associations(ind, targets, key):
    keys = random.split(key, 3)
    ind_0 = ind[targets[ind] == 0]
    ind_1 = ind[targets[ind] == 1]
    zero = random.choice(keys[0], ind_0, shape=(ind.shape[0] // 2,))
    one = random.choice(keys[1], ind_1, shape=(ind.shape[0] // 2,))
    association_features_ind = jnp.concatenate([zero, one])
    association_targets = jnp.concatenate([jnp.zeros(zero.shape), jnp.ones(one.shape)])
    shuffle = random.permutation(keys[2], association_targets.shape[0])
    association_features_ind = association_features_ind[shuffle]
    association_targets = association_targets[shuffle].astype(targets.dtype)
    return association_features_ind, association_targets

def prepare_data_associative(targets, args, key):
    keys = random.split(key)
    ind_train = jnp.arange(args["N_train"])
    ind_test = args["N_train"] + jnp.arange(targets.shape[0] - args["N_train"])
    association_features_ind, association_targets = prepare_associations(ind_train, targets, keys[0])
    association_features_ind_, association_targets_ = prepare_associations(ind_test, targets, keys[1])
    
    association_features_ind = jnp.concatenate([association_features_ind, association_features_ind_])
    association_targets = jnp.concatenate([association_targets, association_targets_])

    if args["editing_type"] == "conflation":
        altered_targets = conflate_digits(targets, args["conflation_list"])
    elif args["editing_type"] == "swap":
        altered_targets = swap_digits(targets, args["swap_list"][0], args["swap_list"][1])
        
    joint_targets = jnp.stack([targets, altered_targets], axis=1)
    modified_targets = jnp.take_along_axis(joint_targets, association_targets.reshape(-1, 1), axis=1)[:, 0]
    return association_features_ind, modified_targets

def get_standard_args():
    args = {
        "path_to_MNIST": '/mnt/local/dataset/by-domain/cv/mnist',
        "N_train": 60000,
        "N_test": 10000,
        "N_augment": 100,
        "N_batch": 100,
        "gamma": 0.5,
        "print_every": 200,
        "key": 33,
        "N_classes": 10,
        "N_updates": 2000,
        "N_drop": 1000,
        "learning_rate": 1e-4,
        "optim": "lion"
    }
    return args

def measure_accuracy(model, features, targets, ind, args_, args, solver_data, preprocessing, postprocessing, key, Hopfield_Kuramoto=False):
    predicted = []
    keys = random.split(key, ind.shape[0])
    if args["dataset_name"] == "MNIST_addition":
        for i, key in zip(ind, keys):
            if Hopfield_Kuramoto:
                prediction = classification.predict_class(model, [features[0][i[0]], features[1][i[1]]], args_, solver_data, preprocessing, postprocessing, key)
            else:
                prediction = classification.predict_class(model, [features[i[0]], features[i[1]]], args_, solver_data, preprocessing, postprocessing, key)
            predicted.append(prediction)
        predicted = jnp.concatenate(predicted)
        acc = jnp.mod(targets[ind[:, 0].reshape(-1,)] + targets[ind[:, 1].reshape(-1,)], 10) == predicted
    else:
        for i, key in zip(ind, keys):
            if Hopfield_Kuramoto:
                prediction = classification.predict_class(model, [features[0][i], features[1][i]], args_, solver_data, preprocessing, postprocessing, key)
            else:
                prediction = classification.predict_class(model, features[i], args_, solver_data, preprocessing, postprocessing, key)
            predicted.append(prediction)
        predicted = jnp.concatenate(predicted)
        acc = targets[ind.reshape(-1,)] == predicted
    return acc

def non_associative_measure_accuracy(model, features, targets, ind, args_, args, solver_data, preprocessing, postprocessing, key):
    predicted = []
    keys = random.split(key, ind.shape[0])
    for i, key in zip(ind, keys):
        prediction = non_associative_edditing.predict_class(model, features[i], args_, solver_data, preprocessing, postprocessing, key)
        predicted.append(prediction)
    predicted = jnp.concatenate(predicted)
    acc = targets[ind.reshape(-1,)] == predicted
    return acc

def Hopfield_dense(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST" or args["dataset_name"] == "MNIST_addition":
        path_to_MNIST = args["path_to_MNIST"]
        features, targets = load_MNIST(path_to_MNIST)
        with open(write_logs_to, "a+") as f:
            f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 3)
    key = keys[-1]

    if args["dataset_name"] == "MNIST_addition":
        N_features = 2*features.shape[1] + args["N_augment"] + args["N_classes"]
    else:
        N_features = features.shape[1] + args["N_augment"] + args["N_classes"]
    N_append = args["N_augment"]
    ind = None
    if args["activation"] == "relu":
        LNet = Hopfield.Lagrange_relu
    elif args["activation"] == "tanh":
        LNet = Hopfield.Lagrange_tanh
    elif args["activation"] == "sigmoid":
        LNet = Hopfield.Lagrange_sigmoid
        
    model = Hopfield.Hopfield_dense(N_features, LNet, keys[0])
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    training_summary["model_size"] = model_size
    with open(write_logs_to, "a+") as f:
        f.write(f"model size {model_size}\n")

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=sc)
    elif args["optim"] == "adam":
        optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if args["dataset_name"] == "MNIST_addition":
        preprocessing = lambda feature, args_: classification.a_Hopfield_preprocessing(feature, N_append, args["N_classes"])
    else:
        preprocessing = lambda feature, args_: classification.Hopfield_preprocessing(feature, N_append, args["N_classes"])
    postprocessing = lambda prediction: classification.Hopfield_postprocessing(prediction, args["N_classes"])
    solver_data = classification.get_default_solver_data()
    args_ = ind

    if args["dataset_name"] == "MNIST_addition":
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], 2, args["N_batch"]))
    else:
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        start = time.time()
        if args["dataset_name"] == "MNIST_addition":
            loss, model, opt_state = classification.make_step(model, [features[n[0]], features[n[1]]], jnp.mod(targets[n[0]] + targets[n[1]], 10), args_, solver_data, preprocessing, postprocessing, None, opt_state, optim)
        else:
            loss, model, opt_state = classification.make_step(model, features[n], targets[n], args_, solver_data, preprocessing, postprocessing, None, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key3, key4 = random.split(key, 5)
            
            if args["dataset_name"] == "MNIST_addition":
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
                inds_test_check = random.choice(key2, features.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
            else:
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
                inds_test_check = random.choice(key2, features.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]

            train_acc = measure_accuracy(model, features, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
            test_acc = measure_accuracy(model, features, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    N_samples = 10
    N_batch_predict = 100
    key, key1, key2, key3, key4 = random.split(key, 5)
    if args["dataset_name"] == "MNIST_addition":        
        inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
        inds_test_check = random.choice(key2, features.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
    else:
        inds_train_check = jnp.arange(args["N_train"]).reshape(-1, 100)
        inds_test_check = args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
        
    train_acc = measure_accuracy(model, features, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
    test_acc = measure_accuracy(model, features, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)

    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def non_associative_Hopfield_Kuramoto(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    path_to_MNIST = args["path_to_MNIST"]
    features_H, targets = load_MNIST(path_to_MNIST)
    with open(write_logs_to, "a+") as f:
        f.write("loading MNIST\n")

    if args["editing_type"] == "swap":
        targets = swap_digits(targets, args["swap_list"][0], args["swap_list"][1])
    elif args["editing_type"] == "conflation":
        targets = conflate_digits(targets, args["conflation_list"])
        
    keys = random.split(random.PRNGKey(args["key"]), 6)
    key = keys[-1]
    key_ = keys[-2]

    N_features = features_H.shape[1] + args["N_augment"] + args["N_classes"]

    if args["interaction"] == "relu":
        interaction = Kuramoto.relu_interaction
    elif args["interaction"] == "tanh":
        interaction = Kuramoto.tanh_interaction
    elif args["interaction"] == "sigmoid":
        interaction = Kuramoto.sigmoid_interaction

    if args["activation"] == "relu":
        LNet = Hopfield.Lagrange_relu
    elif args["activation"] == "tanh":
        LNet = Hopfield.Lagrange_tanh
    elif args["activation"] == "sigmoid":
        LNet = Hopfield.Lagrange_sigmoid

    model_H = Hopfield.Hopfield_dense(N_features, LNet, key)
    model_H = eqx.tree_deserialise_leaves(args["model_path"], model_H)

    ind_K = Kuramoto.get_small_world_connectivity(keys[0], N_features, k=args["k"])
    ind_HK = Kuramoto.get_small_world_connectivity(keys[1], N_features, k=args["k"])
    eps_K = 1 / jnp.sqrt(args["k"])
    eps_H = 1 / jnp.sqrt(N_features)
    eps_HK = args["eps_HK"]
    N_weights = ind_K.shape[0]
    N_weights_i = ind_HK.shape[0]

    if args["coupling_type"] == "additive":
        model = Hopfield_Kuramoto.Hopfield_Kuramoto_additive(N_weights, interaction, N_features, args["D"], keys[2], eps_K, eps_HK)
    elif args["coupling_type"] == "multiplicative":
        model = Hopfield_Kuramoto.Hopfield_Kuramoto_multiplicative(N_weights, interaction, N_features, args["D"], keys[2], eps_K, eps_HK)
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    training_summary["model_size"] = model_size
    with open(write_logs_to, "a+") as f:
        f.write(f"model size {model_size}\n")

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=sc)
    elif args["optim"] == "adam":
        optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
   
    preprocessing = lambda feature, args_: classification.Hopfield_preprocessing(feature, args["N_augment"], args["N_classes"])
    postprocessing = lambda prediction: classification.Hopfield_Kuramoto_postprocessing(prediction, args["N_classes"])
    solver_data = non_associative_edditing.get_default_solver_data()
    args_ = [ind_K, ind_HK, args["kappa_K"], args["kappa_H"], model_H, None]

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        start = time.time()
        loss, model, opt_state = non_associative_edditing.make_step(model, features_H[n], targets[n], args_, solver_data, preprocessing, postprocessing, None, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key3, key4 = random.split(key, 5)
            
            inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]

            train_acc = non_associative_measure_accuracy(model, features_H, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
            test_acc = non_associative_measure_accuracy(model, features_H, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    N_samples = 10
    N_batch_predict = 100
    key, key1, key2, key3, key4 = random.split(key, 5)

    inds_train_check = jnp.arange(args["N_train"]).reshape(-1, 100)
    inds_test_check = args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
        
    train_acc = non_associative_measure_accuracy(model, features_H, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
    test_acc = non_associative_measure_accuracy(model, features_H, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)

    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def Kuramoto_loop(args, omega=False, postprocessing_I=False):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST" or args["dataset_name"] == "MNIST_addition":
        path_to_MNIST = args["path_to_MNIST"]
        features, targets = load_MNIST(path_to_MNIST)
        with open(write_logs_to, "a+") as f:
            f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 5)
    key = keys[-1]
    key_ = keys[-2]

    features_ = classification.Kuramoto_data_init_random(features, args["D"], keys[-3])
    del features
    if args["dataset_name"] == "MNIST_addition":
        N_neurons = 2*features_.shape[1] + args["N_augment"] + args["N_classes"] + 1
    else:
        N_neurons = features_.shape[1] + args["N_augment"] + args["N_classes"] + 1
    ind = Kuramoto.get_small_world_connectivity(keys[0], N_neurons, k=args["k"])
    N_weights = ind.shape[0]
    NN_shapes = None
    if args["interaction"] == "relu":
        interaction = Kuramoto.relu_interaction
    elif args["interaction"] == "tanh":
        interaction = Kuramoto.tanh_interaction
    elif args["interaction"] == "sigmoid":
        interaction = Kuramoto.sigmoid_interaction
    elif args["interaction"] == "deep_GELU_global":
        NN_shapes = [1, 10, 1]
        interaction = Kuramoto.deep_GELU_interaction_I
    elif args["interaction"] == "deep_GELU_local":
        NN_shapes = [1, 10, 1]
        interaction = Kuramoto.deep_GELU_interaction_II
    if omega:
        model = Kuramoto.Kuramoto_global_omega(N_weights, interaction, 1/jnp.sqrt(args["k"]), args["D"], keys[0], NN_shapes=NN_shapes)
    else:
        model = Kuramoto.Kuramoto_global(N_weights, interaction, 1/jnp.sqrt(args["k"]), keys[0], NN_shapes=NN_shapes)
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    training_summary["model_size"] = model_size
    with open(write_logs_to, "a+") as f:
        f.write(f"model size {model_size}\n")

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=sc)
    elif args["optim"] == "adam":
        optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if args["dataset_name"] == "MNIST_addition":
        preprocessing = lambda feature, key: classification.a_Kuramoto_preprocessing_random_III(feature, args["N_augment"], args["N_classes"], key)
    else:
        preprocessing = lambda feature, key: classification.Kuramoto_preprocessing_random_III(feature, args["N_augment"], args["N_classes"], key)
    
    if postprocessing_I:
        postprocessing = lambda prediction: classification.Kuramoto_postprocessing_I(prediction, args["N_classes"])
    else:
        postprocessing = lambda prediction: classification.Kuramoto_postprocessing(prediction, args["N_classes"])
    solver_data = classification.get_default_solver_data()
    args_ = ind

    if args["dataset_name"] == "MNIST_addition":
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], 2, args["N_batch"]))
    else:
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        key_, key1 = random.split(key_)
        start = time.time()
        if args["dataset_name"] == "MNIST_addition":
            loss, model, opt_state = classification.make_step(model, [features_[n[0]], features_[n[1]]], jnp.mod(targets[n[0]] + targets[n[1]], 10), args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        else:
            loss, model, opt_state = classification.make_step(model, features_[n], targets[n], args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key3, key4 = random.split(key, 5)

            predicted_train = []
            predicted_test = []

            if args["dataset_name"] == "MNIST_addition":
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
                inds_test_check = random.choice(key2, features_.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
            else:
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
                inds_test_check = random.choice(key2, features_.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
            train_acc = measure_accuracy(model, features_, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
            test_acc = measure_accuracy(model, features_, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)
    
    N_samples = 10
    N_batch_predict = 100
    key, key1, key2, key3, key4 = random.split(key, 5)

    if args["dataset_name"] == "MNIST_addition":
        inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
        inds_test_check = random.choice(key2, features_.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
    else:
        inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
        inds_test_check = random.choice(key2, features_.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
    
    train_acc = measure_accuracy(model, features_, targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3)
    test_acc = measure_accuracy(model, features_, targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def Kuramoto_small_world(args):
    return Kuramoto_loop(args)

def Kuramoto_omega(args):
    return Kuramoto_loop(args, omega=True)

def Kuramoto_omega_I(args):
    return Kuramoto_loop(args, omega=True, postprocessing_I=True)

def Hopfield_Kuramoto_small_world(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST" or args["dataset_name"] == "MNIST_addition":
        path_to_MNIST = args["path_to_MNIST"]
        features, targets = load_MNIST(path_to_MNIST)
        with open(write_logs_to, "a+") as f:
            f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 6)
    key = keys[-1]
    key_ = keys[-2]
    
    features_H = jnp.pad(features, ((0, 0), (1, 0)))
    features_K = classification.Kuramoto_data_init_random(features, args["D"], keys[-3])
    del features
    N_neurons = features_H.shape[1] + args["N_augment"] + args["N_classes"] + 1
    if args["interaction"] == "relu":
        interaction = Kuramoto.relu_interaction
    elif args["interaction"] == "tanh":
        interaction = Kuramoto.tanh_interaction
    elif args["interaction"] == "sigmoid":
        interaction = Kuramoto.sigmoid_interaction

    if args["activation"] == "relu":
        LNet = Hopfield.Lagrange_relu
    elif args["activation"] == "tanh":
        LNet = Hopfield.Lagrange_tanh
    elif args["activation"] == "sigmoid":
        LNet = Hopfield.Lagrange_sigmoid    

    ind_K = Kuramoto.get_small_world_connectivity(keys[0], N_neurons, k=args["k"])
    ind_HK = Kuramoto.get_small_world_connectivity(keys[1], N_neurons, k=args["k"])
    eps_K = 1 / jnp.sqrt(args["k"])
    eps_H = 1 / jnp.sqrt(N_neurons)
    eps_HK = args["eps_HK"]
    N_weights = ind_K.shape[0]
    N_weights_i = ind_HK.shape[0]

    model = Hopfield_Kuramoto.Hopfield_Kuramoto_network(N_weights, interaction, N_neurons, N_weights_i, keys[2], LNet, eps_K, eps_H, eps_HK)
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    training_summary["model_size"] = model_size
    with open(write_logs_to, "a+") as f:
        f.write(f"model size {model_size}\n")

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=sc)
    elif args["optim"] == "adam":
        optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    preprocessing = lambda feature, key: classification.Hopfield_Kuramoto_preprocessing_random(feature, args["N_augment"], args["N_classes"], key)
    postprocessing = lambda prediction: classification.Hopfield_Kuramoto_postprocessing(prediction, args["N_classes"])
    solver_data = classification.get_default_solver_data()
    args_ = [ind_K, ind_HK, args["kappa_K"], args["kappa_H"]]

    if args["dataset_name"] == "MNIST_addition":
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], 2, args["N_batch"]))
    else:
        inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        key_, key1 = random.split(key_)
        start = time.time()
        if args["dataset_name"] == "MNIST_addition":
            loss, model, opt_state = classification.make_step(model, [features_H[n[0]], features_K[n[1]]], jnp.mod(targets[n[0]] + targets[n[1]], 10), args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        else:
            loss, model, opt_state = classification.make_step(model, [features_H[n], features_K[n]], targets[n], args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
                
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key3, key4 = random.split(key, 5)
            
            if args["dataset_name"] == "MNIST_addition":
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
                inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
            else:
                inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
                inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]

            train_acc = measure_accuracy(model, [features_H, features_K], targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3, Hopfield_Kuramoto=True)
            test_acc = measure_accuracy(model, [features_H, features_K], targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4, Hopfield_Kuramoto=True)
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    N_samples = 10
    N_batch_predict = 100
    key, key1, key2, key3, key4 = random.split(key, 5)

    if args["dataset_name"] == "MNIST_addition":
        inds_train_check = random.choice(key1, args["N_train"], (N_samples, 2, N_batch_predict))
        inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, 2, N_batch_predict)) + args["N_train"]
    else:
        inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
        inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
    
    train_acc = measure_accuracy(model, [features_H, features_K], targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3, Hopfield_Kuramoto=True)
    test_acc = measure_accuracy(model, [features_H, features_K], targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4, Hopfield_Kuramoto=True)
    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def associative_Hopfield_Kuramoto(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    path_to_MNIST = args["path_to_MNIST"]
    features_H, targets = load_MNIST(path_to_MNIST)
    with open(write_logs_to, "a+") as f:
        f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 7)
    key = keys[-1]
    key_ = keys[-2]

    association_features_ind, targets = prepare_data_associative(targets, args, keys[-3])
    features_K = classification.Kuramoto_data_init(features_H[association_features_ind][:, :-1], args["D"])

    N_features = features_H.shape[1] + args["N_augment"] + args["N_classes"]

    if args["interaction"] == "relu":
        interaction = Kuramoto.relu_interaction
    elif args["interaction"] == "tanh":
        interaction = Kuramoto.tanh_interaction
    elif args["interaction"] == "sigmoid":
        interaction = Kuramoto.sigmoid_interaction

    if args["activation"] == "relu":
        LNet = Hopfield.Lagrange_relu
    elif args["activation"] == "tanh":
        LNet = Hopfield.Lagrange_tanh
    elif args["activation"] == "sigmoid":
        LNet = Hopfield.Lagrange_sigmoid

    model_H = Hopfield.Hopfield_dense(N_features, LNet, key)
    model_H = eqx.tree_deserialise_leaves(args["model_path"], model_H)

    ind_K = Kuramoto.get_small_world_connectivity(keys[0], N_features, k=args["k"])
    ind_HK = Kuramoto.get_small_world_connectivity(keys[1], N_features, k=args["k"])
    eps_K = 1 / jnp.sqrt(args["k"])
    eps_H = 1 / jnp.sqrt(N_features)
    eps_HK = args["eps_HK"]
    N_weights = ind_K.shape[0]
    N_weights_i = ind_HK.shape[0]

    if args["coupling_type"] == "additive":
        model = Hopfield_Kuramoto.Hopfield_Kuramoto_additive(N_weights, interaction, N_features, args["D"], keys[2], eps_K, eps_HK)
    elif args["coupling_type"] == "multiplicative":
        model = Hopfield_Kuramoto.Hopfield_Kuramoto_multiplicative(N_weights, interaction, N_features, args["D"], keys[2], eps_K, eps_HK)
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    training_summary["model_size"] = model_size
    with open(write_logs_to, "a+") as f:
        f.write(f"model size {model_size}\n")

    sc = optax.exponential_decay(args["learning_rate"], args["N_drop"], args["gamma"])
    if args["optim"] == "lion":
        optim = optax.lion(learning_rate=sc)
    elif args["optim"] == "adam":
        optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
   
    preprocessing = lambda feature, key: classification.associative_Hopfield_Kuramoto_preprocessing(feature, args["N_augment"], args["N_classes"])
    postprocessing = lambda prediction: classification.Hopfield_Kuramoto_postprocessing(prediction, args["N_classes"])
    solver_data = non_associative_edditing.get_default_solver_data()
    args_ = [ind_K, ind_HK, args["kappa_K"], args["kappa_H"], model_H, None]

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        start = time.time()
        loss, model, opt_state = classification.make_step(model, [features_H[n], features_K[n]], targets[n], args_, solver_data, preprocessing, postprocessing, None, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key3, key4 = random.split(key, 5)
            
            inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]

            train_acc = measure_accuracy(model, [features_H, features_K], targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3, Hopfield_Kuramoto=True)
            test_acc = measure_accuracy(model, [features_H, features_K], targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4, Hopfield_Kuramoto=True)
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    N_samples = 10
    N_batch_predict = 100
    key, key1, key2, key3, key4 = random.split(key, 5)

    inds_train_check = jnp.arange(args["N_train"]).reshape(-1, 100)
    inds_test_check = args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
        
    train_acc = measure_accuracy(model, [features_H, features_K], targets, inds_train_check, args_, args, solver_data, preprocessing, postprocessing, key3, Hopfield_Kuramoto=True)
    test_acc = measure_accuracy(model, [features_H, features_K], targets, inds_test_check, args_, args, solver_data, preprocessing, postprocessing, key4, Hopfield_Kuramoto=True)

    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary