
# coding: utf-8

import numpy as np
import glob
import skimage.transform
from skimage.io import imread
#import tensorflow as tf
#from keras.datasets import mnist
#import matplotlib.pyplot as plt
import scipy.misc
#import binascii
import string
import random
import os
import shutil
import sys


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

cut_img_paths = glob.glob("./Kotelet/Cut/*.png")
glove_img_paths = glob.glob("./Kotelet/Glove/*.png")
normal_img_paths = glob.glob("./Kotelet/Normal/*.png")

print("Deleting Kotelet_enhanced/")
shutil.rmtree("Kotelet_enhanced", ignore_errors=True)

print("Making Kotelet_enhanced/ directories")
os.makedirs("Kotelet_enhanced/Cut")
os.makedirs("Kotelet_enhanced/Glove")
os.makedirs("Kotelet_enhanced/Normal")

n = int(sys.argv[1])

print("Creating additional " + str(n) + " images of each class")

print("Cut images")
for i in range(0, n):
    selected = np.random.choice(cut_img_paths,1)
    random_angle = (np.random.rand(1,1)-0.5)*360
    random_center = (np.random.rand(1,2)-0.5)*50 + (200,200)
    rotated = skimage.transform.rotate(imread(selected[0]), angle=random_angle[0][0], resize=False,center=random_center[0])
    name = id_generator(25)
    scipy.misc.imsave('Kotelet_enhanced/Cut/' + name + '.png', rotated)
    # print("Iter nr: " + str(i))

print("Glove images")
for i in range(0, n):
    selected = np.random.choice(glove_img_paths,1)
    random_angle = (np.random.rand(1,1)-0.5)*360
    random_center = (np.random.rand(1,2)-0.5)*50 + (200,200)
    rotated = skimage.transform.rotate(imread(selected[0]), angle=random_angle[0][0], resize=False,center=random_center[0])
    name = id_generator(25)
    scipy.misc.imsave('Kotelet_enhanced/Glove/' + name + '.png', rotated)
    # print("Iter nr: " + str(i))
    
print("Normal images")
for i in range(0, n):
    selected = np.random.choice(normal_img_paths,1)
    random_angle = (np.random.rand(1,1)-0.5)*360
    random_center = (np.random.rand(1,2)-0.5)*50 + (200,200)
    rotated = skimage.transform.rotate(imread(selected[0]), angle=random_angle[0][0], resize=False,center=random_center[0])
    name = id_generator(25)
    scipy.misc.imsave('Kotelet_enhanced/Normal/' + name + '.png', rotated)
    # print("Iter nr: " + str(i))
print('Done!')



