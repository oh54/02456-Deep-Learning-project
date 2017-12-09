from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import time
from keras_dec import DeepEmbeddingClustering
import numpy as np
import glob
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist

start = time.time()

img_paths = glob.glob("./Kotelet/*/*.png")
imgs = np.asarray([resize(imread(img), (100, 100), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))


y = [1 if path[10] == 'N' else 1 for path in img_paths]

#c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
#c.initialize(X=imgs, finetune_iters=1, layerwise_pretrain_iters=1)
#c.cluster(X=imgs, Y=y, tol=0.01, update_interval=10, iter_max=100, save_interval=50)


end = time.time()

print("Elapsed time: " + str(end - start))
