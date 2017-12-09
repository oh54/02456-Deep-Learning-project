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
imgs = np.asarray([resize(imread(img, as_grey=False), (100, 100), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))


y = np.asarray([0 if path[10] == 'N' else 1 for path in img_paths])

c = DeepEmbeddingClustering(n_clusters=2, input_dim=30000)
c.initialize(X=imgs, finetune_iters=100, layerwise_pretrain_iters=100)
c.cluster(X=imgs, y=y, update_interval=1, iter_max=100, save_interval=50)

end = time.time()

print("Elapsed time: " + str(end - start))

