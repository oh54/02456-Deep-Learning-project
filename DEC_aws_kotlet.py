from keras.datasets import mnist
import numpy as np
import time
from keras_dec import DeepEmbeddingClustering
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
cuts_e = glob.glob("./Kotelet_enhanced/Cut/*.png")
normals_e = glob.glob("./Kotelet_enhanced/Normal/*.png")

img_paths_o = cuts_o + normals_o 
img_paths_e = cuts_e + normals_e
img_paths = img_paths_o + img_paths_e

imgs = np.asarray([resize(imread(img, as_grey=True), (100, 100), mode='constant').flatten() for img in img_paths])
print("Shape: " + str(imgs.shape))

y_o = [0 if path[10] == 'N' else 1 for path in img_paths_o]
y_e = [0 if path[19] == 'N' else 1 for path in img_paths_e]
y = np.asarray(y_o + y_e)

#n_clusters,
#                 input_dim,
#                 encoded=None,
#                 decoded=None,
#                 alpha=1.0,
#                 pretrained_weights=None,
#                 cluster_centres=None,
#                 batch_size=256,
#
c = DeepEmbeddingClustering(n_clusters=2, input_dim=10000, batch_size=64)
c.initialize(X=imgs, finetune_iters=20000, layerwise_pretrain_iters=10000)
outp = c.cluster(X=imgs, y=y, tol=0.01, update_interval=10000, iter_max=1000000, save_interval=10000)


print("PREDICTED CLASS 0")
zero_pred = np.asarray(img_paths)[np.where(outp == 0)[0]]
np.random.shuffle(zero_pred)
print(zero_pred[0:20])

print("PREDICTED CLASS 1")
one_pred = np.asarray(img_paths)[np.where(outp == 1)[0]]
np.random.shuffle(one_pred)
print(one_pred[0:20])

shutil.rmtree("Class1_images", ignore_errors=True)
shutil.rmtree("Class0_images", ignore_errors=True)
os.makedirs("Class1_images")
os.makedirs("Class0_images")

save_img_count=20
i = 0
for path in zero_pred[0:save_img_count]:
    shutil.copyfile(path, "Class0_images/class0_" + str(i) + ".png")
    i += 1

i = 0
for path in one_pred[0:save_img_count]:
    shutil.copyfile(path, "Class1_images/class1_" + str(i) + ".png")
    i += 1

end = time.time()
print("Elapsed time: " + str(end - start))




