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

if __name__ == "__main__":
    path_to_MNIST = sys.argv[1]
    randomise = bool(sys.argv[2])

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
    
    with open('f'{path_to_MNIST}/raw/t10k-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        targets_ = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        targets_ = jnp.array(targets_.reshape((size,)))
    
    targets = jnp.concatenate([targets, targets_], axis=0)

    N_augment = 100
    N_classes = 10
    learning_rate = 1e-4
    gamma = 0.5
    N_updates = 1000
    N_train = 60000
    N_test = 10000
    N_drop = N_updates // 2
    N_batch = 100
    k = 150
    D = 5
    N_features = features.shape[1] + N_augment + N_classes
    print_every = 100
    key = random.PRNGKey(33)
    keys = random.split(key, 5)
    key = keys[-1]
    key_ = keys[-2]
    
    features_ = classification.Kuramoto_data_init_random(features, D, keys[-3])
    N_neurons = sum([features_.shape[1], N_augment, N_classes + 1])
    ind = Kuramoto.get_small_world_connectivity(keys[0], N_neurons, k=k)
    
    interaction = Kuramoto.relu_interaction
    N_weights = ind.shape[0]
    model = Kuramoto.Kuramoto_global(N_weights, interaction, 1/jnp.sqrt(k), keys[0])
    model_size = sum(tree_map(lambda x: 2*x.size if x.dtype == jnp.complex64 else x.size, tree_flatten(model)[0]))
    print("model size", model_size)
    
    sc = optax.exponential_decay(learning_rate, N_drop, gamma)
    optim = optax.lion(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    if randomise:
        print("using random rotation for each batch")
        preprocessing = lambda feature, key: classification.Kuramoto_preprocessing_random_III(feature, N_augment, N_classes, key)
    else:
        print("using batch unprocessed")
        preprocessing = lambda feature, key: classification.Kuramoto_preprocessing(feature, N_augment, N_classes)
    postprocessing = lambda prediction: classification.Kuramoto_postprocessing(prediction, N_classes)
    solver_data = classification.get_default_solver_data()
    args = ind
    
    inds = random.choice(keys[1], N_train, (N_updates, N_batch))
    losses = []
    times = []
    for step, n in enumerate(inds):
        key_, key1 = random.split(key_)
        start = time.time()
        loss, model, opt_state = classification.make_step(model, features_[n], targets[n], args, solver_data, preprocessing, postprocessing, key1, opt_state, optim)
        end = time.time()
        losses.append(loss)
        times.append(end - start)
        if (step % print_every) == 0 or step == N_updates - 1:
            print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
    
            N_samples = 5
            N_batch_predict = 100
            key, key1, key2, key1_, key2_ = random.split(key, 5)
            
            inds_train_check = random.choice(key1, N_train, (N_samples, N_batch_predict))
            inds_test_check = random.choice(key2, features.shape[0] - N_train, (N_samples, N_batch_predict)) + N_train
            
            predicted_train = []
            predicted_test = []
            
            for i, j in zip(inds_train_check, inds_test_check):
                predicted = classification.predict_class(model, features_[i], args, solver_data, preprocessing, postprocessing, key1_)
                predicted_train.append(predicted)
                predicted = classification.predict_class(model, features_[j], args, solver_data, preprocessing, postprocessing, key2_)
                predicted_test.append(predicted)
            
            predicted_train = jnp.concatenate(predicted_train)
            predicted_test = jnp.concatenate(predicted_test)
            train_acc = targets[inds_train_check.reshape(-1,)] == predicted_train
            test_acc = targets[inds_test_check.reshape(-1,)] == predicted_test
            print("train accuracy", jnp.mean(train_acc), "test accuracy", jnp.mean(test_acc))
        print("total training time", sum(times))