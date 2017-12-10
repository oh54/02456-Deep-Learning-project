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

#img_paths = glob.glob("./Kotelet/*/*.png")
cuts = glob.glob("./Kotelet/Cut/*.png")
normals = glob.glob("./Kotelet/Normal/*.png")
img_paths = cuts + normals

imgs = np.asarray([resize(imread(img, as_grey=True), (100, 100), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))


y = np.asarray([0 if path[10] == 'N' else 1 for path in img_paths])

#n_clusters,
#                 input_dim,
#                 encoded=None,
#                 decoded=None,
#                 alpha=1.0,
#                 pretrained_weights=None,
#                 cluster_centres=None,
#                 batch_size=256,
#
c = DeepEmbeddingClustering(n_clusters=2, input_dim=10000, batch_size=32)
c.initialize(X=imgs, finetune_iters=10000, layerwise_pretrain_iters=5000)
outp = c.cluster(X=imgs, y=y, tol=0.01, update_interval=10000, iter_max=1000000, save_interval=10000)

end = time.time()

print("PREDICTED CLASS 1")
print(np.asarray(img_paths)[np.where(outp == 1)[0]])
print("PREDICTED CLASS 0")
print(np.asarray(img_paths)[np.where(outp == 0)[0]])

print("Elapsed time: " + str(end - start))

