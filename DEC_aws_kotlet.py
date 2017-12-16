from keras.datasets import mnist
import numpy as np
import time
import keras_dec
import importlib
importlib.reload(keras_dec)
import numpy as np
import glob
from skimage.transform import resize
from skimage.io import imread
import os
import shutil

start = time.time()

#img_paths = glob.glob("./Kotelet/*/*.png")
cuts_o = glob.glob("./Kotelet/Cut/*.png")
normals_o = glob.glob("./Kotelet/Normal/*.png")
gloves_o = glob.glob("./Kotelet/Glove/*.png")
cuts_e = glob.glob("./Kotelee/Cut/*.png")
normals_e = glob.glob("./Kotelee/Normal/*.png")
gloves_e = glob.glob("./Kotelee/Glove/*.png")

img_paths_o = cuts_o + normals_o + gloves_o
img_paths_e = cuts_e + normals_e + gloves_e
img_paths = img_paths_o + img_paths_e

greyscale = False
dim = 60
dim_flat = dim*dim if greyscale else dim*dim*3

imgs = np.asarray([resize(imread(img, as_grey=greyscale), (dim, dim), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))

y_o = [0 if path[10] == 'N' else 1 for path in img_paths_o]
y_e = [0 if path[10] == 'N' else 1 for path in img_paths_e]
y = np.asarray(y_o + y_e)
#y = np.asarray(y_o)

#n_clusters,
#                 input_dim,
#                 encoded=None,
#                 decoded=None,
#                 alpha=1.0,
#                 pretrained_weights=None,
#                 cluster_centres=None,
#                 batch_size=256,
#

# AWS
#c = keras_dec.DeepEmbeddingClustering(n_clusters=1, input_dim=3600, alpha=1.0, batch_size=64)
#c.initialize(X=imgs, finetune_iters=10000, layerwise_pretrain_iters=5000)
#probs_preds = c.cluster(X=imgs, y=y, tol=0.01, update_interval=0, iter_max=0, save_interval=0, cutoff=0.50)



c = keras_dec.DeepEmbeddingClustering(n_clusters=1, input_dim=dim_flat, alpha=1.0, batch_size=32)
c.initialize(X=imgs, finetune_iters=200, layerwise_pretrain_iters=100)
probs_preds = c.cluster(X=imgs, y=y, tol=0.01, update_interval=0, iter_max=0, save_interval=0)

probs = probs_preds[0]
preds = probs_preds[1]


def find_best_cutoff(probs, y):
    best_acc = 0.0
    best_cutoff = 0.0
    for cutoff_int in range(1, 10001, 1):
        cutoff = cutoff_int / 10000

        cutoff_preds = np.asarray([1 if x[0] >= cutoff else 0 for x in probs])
        cutoff_acc = sum(cutoff_preds == y) / len(y)
        if cutoff_acc > best_acc:
            best_acc = cutoff_acc
            best_cutoff = cutoff

    print("best acc: " + str(best_acc))
    print("best cutoff: " + str(best_cutoff))

find_best_cutoff(probs, y)

save_img_count=30

args = np.argsort(probs, axis=0).flatten()

worst = args[0:save_img_count-1]
best = args[-save_img_count-1:]

"""
def save_imgs(probs, paths, dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)
    i = 0
    for path in paths[0:save_img_count-1]:
        #print(path)
        shutil.copyfile(path, dir + "/" + path[10] + "_" + str(probs[i]) + ".png")
        i += 1
    return

save_imgs(probs[worst].flatten(), np.asarray(img_paths)[worst], "Worst")
save_imgs(probs[best].flatten(), np.asarray(img_paths)[best], "Best")
"""

end = time.time()
print("Elapsed time: " + str(end - start))
