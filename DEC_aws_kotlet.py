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
#cuts_e = glob.glob("./Kotelet_enhanced/Cut/*.png")
#normals_e = glob.glob("./Kotelet_enhanced/Normal/*.png")

img_paths_o = cuts_o + normals_o
#img_paths_e = cuts_e + normals_e
#img_paths = img_paths_o + img_paths_e
img_paths = img_paths_o

imgs = np.asarray([resize(imread(img, as_grey=True), (60, 60), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))

y_o = [0 if path[10] == 'N' else 1 for path in img_paths_o]
#y_e = [0 if path[19] == 'N' else 1 for path in img_paths_e]
#y = np.asarray(y_o + y_e)
y = np.asarray(y_o)

#n_clusters,
#                 input_dim,
#                 encoded=None,
#                 decoded=None,
#                 alpha=1.0,
#                 pretrained_weights=None,
#                 cluster_centres=None,
#                 batch_size=256,
#
c = keras_dec.DeepEmbeddingClustering(n_clusters=1, input_dim=3600, batch_size=64)
c.initialize(X=imgs, finetune_iters=10000, layerwise_pretrain_iters=5000)
outp = c.cluster(X=imgs, y=y, tol=0.01, update_interval=10000, iter_max=1000000, save_interval=10000)


#print("PREDICTED CLASS 0")
#zero_pred = np.asarray(img_paths)[np.where(outp == 0)[0]]
#np.random.shuffle(zero_pred)
#print(zero_pred[0:20])

#print("PREDICTED CLASS 1")
#one_pred = np.asarray(img_paths)[np.where(outp == 1)[0]]
#np.random.shuffle(one_pred)
#print(one_pred[0:20])

save_img_count=20


args = np.argsort(outp, axis=0).flatten()

worst = args[0:save_img_count-1]
best = args[-save_img_count-1:]


def save_imgs(probs, paths, dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)
    i = 0
    for path in paths[0:save_img_count-1]:
        #print(path)
        shutil.copyfile(path, dir + "/" + path[10] + "_" + str(probs[i]) + ".png")
        i += 1
    return

save_imgs(outp[worst].flatten(), np.asarray(img_paths)[worst], "Worst")
save_imgs(outp[best].flatten(), np.asarray(img_paths)[best], "Best")

end = time.time()
print("Elapsed time: " + str(end - start))
