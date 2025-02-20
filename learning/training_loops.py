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
from learning import classification
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

def Hopfield_dense(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST":
        path_to_MNIST = args["path_to_MNIST"]
        features, targets = load_MNIST(path_to_MNIST)
        with open(write_logs_to, "a+") as f:
            f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 3)
    key = keys[-1]

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
    
    preprocessing = lambda feature, args_: classification.Hopfield_preprocessing(feature, N_append, args["N_classes"])
    postprocessing = lambda prediction: classification.Hopfield_postprocessing(prediction, args["N_classes"])
    solver_data = classification.get_default_solver_data()
    args_ = ind

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        start = time.time()
        loss, model, opt_state = classification.make_step(model, features[n], targets[n], args_, solver_data, preprocessing, postprocessing, None, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2 = random.split(key, 3)
            
            inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
            
            predicted_train = []
            predicted_test = []
            
            for i, j in zip(inds_train_check, inds_test_check):
                predicted = classification.predict_class(model, features[i], args_, solver_data, preprocessing, postprocessing, None)
                predicted_train.append(predicted)
                predicted = classification.predict_class(model, features[j], args_, solver_data, preprocessing, postprocessing, None)
                predicted_test.append(predicted)
            
            predicted_train = jnp.concatenate(predicted_train)
            predicted_test = jnp.concatenate(predicted_test)
            train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
            test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    inds_train_check, inds_test_check = jnp.arange(args["N_train"]).reshape(-1, 100), args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
    predicted_train = []
    predicted_test = []

    for i in inds_train_check:
        predicted = classification.predict_class(model, features[i], args_, solver_data, preprocessing, postprocessing, None)
        predicted_train.append(predicted)

    for j in inds_test_check:
        predicted = classification.predict_class(model, features[j], args_, solver_data, preprocessing, postprocessing, None)
        predicted_test.append(predicted)
    
    predicted_train = jnp.concatenate(predicted_train)
    predicted_test = jnp.concatenate(predicted_test)
    train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
    test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def Kuramoto_small_world(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST":
        path_to_MNIST = args["path_to_MNIST"]
        features, targets = load_MNIST(path_to_MNIST)
        with open(write_logs_to, "a+") as f:
            f.write("loading MNIST\n")
        
    keys = random.split(random.PRNGKey(args["key"]), 5)
    key = keys[-1]
    key_ = keys[-2]
    
    features_ = classification.Kuramoto_data_init_random(features, args["D"], keys[-3])
    del features
    N_neurons = features_.shape[1] + args["N_augment"] + args["N_classes"] + 1
    ind = Kuramoto.get_small_world_connectivity(keys[0], N_neurons, k=args["k"])
    N_weights = ind.shape[0]
    if args["interaction"] == "relu":
        interaction = Kuramoto.relu_interaction
    elif args["interaction"] == "tanh":
        interaction = Kuramoto.tanh_interaction
    elif args["interaction"] == "sigmoid":
        interaction = Kuramoto.sigmoid_interaction
        
    model = Kuramoto.Kuramoto_global(N_weights, interaction, 1/jnp.sqrt(args["k"]), keys[0])
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
    
    preprocessing = lambda feature, key: classification.Kuramoto_preprocessing_random_III(feature, args["N_augment"], args["N_classes"], key)
    postprocessing = lambda prediction: classification.Kuramoto_postprocessing(prediction, args["N_classes"])
    solver_data = classification.get_default_solver_data()
    args_ = ind

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        key_, key1 = random.split(key_)
        start = time.time()
        loss, model, opt_state = classification.make_step(model, features_[n], targets[n], args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key__ = random.split(key, 4)
            
            inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features_.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
            
            predicted_train = []
            predicted_test = []
            
            for i, j in zip(inds_train_check, inds_test_check):
                key__, key1_, key2_ = random.split(key__, 3)
                predicted = classification.predict_class(model, features_[i], args_, solver_data, preprocessing, postprocessing, key1_)
                predicted_train.append(predicted)
                predicted = classification.predict_class(model, features_[j], args_, solver_data, preprocessing, postprocessing, key2_)
                predicted_test.append(predicted)
            
            predicted_train = jnp.concatenate(predicted_train)
            predicted_test = jnp.concatenate(predicted_test)
            train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
            test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    inds_train_check, inds_test_check = jnp.arange(args["N_train"]).reshape(-1, 100), args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
    predicted_train = []
    predicted_test = []

    for i in inds_train_check:
        key, key1 = random.split(key)
        predicted = classification.predict_class(model, features_[i], args_, solver_data, preprocessing, postprocessing, key1)
        predicted_train.append(predicted)

    for j in inds_test_check:
        key, key2 = random.split(key)
        predicted = classification.predict_class(model, features_[j], args_, solver_data, preprocessing, postprocessing, key2)
        predicted_test.append(predicted)
    
    predicted_train = jnp.concatenate(predicted_train)
    predicted_test = jnp.concatenate(predicted_test)
    train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
    test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary

def Hopfield_Kuramoto_small_world(args):
    training_summary = {}
    write_logs_to = args["write_logs_to"]
    if args["dataset_name"] == "MNIST":
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

    inds = random.choice(keys[1], args["N_train"], (args["N_updates"], args["N_batch"]))
    losses = []
    times = []
    for step, n in enumerate(inds):
        key_, key1 = random.split(key_)
        start = time.time()
        loss, model, opt_state = classification.make_step(model, [features_H[n], features_K[n]], targets[n], args_, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        end = time.time()
        times.append(end - start)
        losses.append(loss)
        if (step % args["print_every"]) == 0 or step == args["N_updates"] - 1:
            with open(write_logs_to, "a+") as f:
                f.write(f"Step: {step}, Loss: {loss}, Computation time: {end - start}\n")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key__ = random.split(key, 4)
            
            inds_train_check = random.choice(key1, args["N_train"], (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features_H.shape[0] - args["N_train"], (N_samples, N_batch_predict)) + args["N_train"]
            
            predicted_train = []
            predicted_test = []
            
            for i, j in zip(inds_train_check, inds_test_check):
                key__, key1_, key2_ = random.split(key__, 3)
                predicted = classification.predict_class(model, [features_H[i], features_K[i]], args_, solver_data, preprocessing, postprocessing, key1_)
                predicted_train.append(predicted)
                predicted = classification.predict_class(model, [features_H[j], features_K[j]], args_, solver_data, preprocessing, postprocessing, key2_)
                predicted_test.append(predicted)
            
            predicted_train = jnp.concatenate(predicted_train)
            predicted_test = jnp.concatenate(predicted_test)
            train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
            test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
            
            with open(write_logs_to, "a+") as f:
                f.write(f"train accuracy {jnp.mean(train_acc)}, test accuracy {jnp.mean(test_acc)}\n")
    losses = jnp.array(losses)
    
    with open(write_logs_to, "a+") as f:
        f.write(f"total training time {sum(times)}\n")
    training_summary["training_time"] = sum(times)

    inds_train_check, inds_test_check = jnp.arange(args["N_train"]).reshape(-1, 100), args["N_train"] + jnp.arange(args["N_test"]).reshape(-1, 100)
    predicted_train = []
    predicted_test = []

    for i in inds_train_check:
        key, key1 = random.split(key)
        predicted = classification.predict_class(model, [features_H[i], features_K[i]], args_, solver_data, preprocessing, postprocessing, key1)
        predicted_train.append(predicted)

    for j in inds_test_check:
        key, key2 = random.split(key)
        predicted = classification.predict_class(model, [features_H[j], features_K[j]], args_, solver_data, preprocessing, postprocessing, key2)
        predicted_test.append(predicted)
    
    predicted_train = jnp.concatenate(predicted_train)
    predicted_test = jnp.concatenate(predicted_test)
    train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
    test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
    with open(write_logs_to, "a+") as f:
        f.write(f"total train accuracy {jnp.mean(train_acc)}, total test accuracy {jnp.mean(test_acc)}\n")
    training_summary["train_accuracy"] = jnp.mean(train_acc).item()
    training_summary["test_accuracy"] = jnp.mean(test_acc).item()
    
    return model, opt_state, losses, training_summary