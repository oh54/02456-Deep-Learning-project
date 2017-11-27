
# coding: utf-8

# In[15]:

from keras_dec import DeepEmbeddingClustering
import numpy as np
import glob
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import time

# In[16]:

start = time.time()

img_paths = glob.glob("./Kotelet/*/*.png")
imgs = np.asarray([resize(imread(img), (100, 100), mode='constant').flatten() for img in img_paths])


# In[17]:

print("Shape: " + str(imgs.shape))


# In[18]:

#c = DeepEmbeddingClustering(n_clusters=3, input_dim=30000,iter_max=25)
#c.initialize(imgs, finetune_iters=10, layerwise_pretrain_iters=50)
#c.cluster(imgs)


# In[19]:


#c = DeepEmbeddingClustering(n_clusters=3, input_dim=30000)
#c.initialize(imgs, finetune_iters=100, layerwise_pretrain_iters=20)
#c.cluster(imgs)


# In[14]:

from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np


def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X, Y


X, Y  = get_mnist()

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
c.cluster(X)


## Creates a graph.
#a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#c = tf.matmul(a, b)
## Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
#print(sess.run(c))
#

end = time.time()

print("Elapsed time: " + str(end-start))
