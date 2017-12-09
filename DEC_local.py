from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import time

start = time.time()


def get_mnist():
    np.random.seed(1234)  # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    X = x_all.reshape(-1, x_all.shape[1] * x_all.shape[2])

    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32) * 0.02
    Y = Y[p]
    return X, Y


X, Y = get_mnist()

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=1, layerwise_pretrain_iters=1)
c.cluster(X, Y, tol=0.01, update_interval=10, iter_max=100, save_interval=50)

end = time.time()

print("Elapsed time: " + str(end - start))
