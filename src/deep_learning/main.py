import time

from numpy import extract
start = time.time()

import os
os.system('cls')

print("Modules importation :\n")
print(f"{'    Standard modules' :-<50}", end="")
import pathlib
import matplotlib.pyplot as plt
print(" Done\n")

print(f"{'    Tensorflow modules' :-<50}", end="")
import tensorflow as tf
print(" Done\n")

####################################################################################################
#                                          LOADING DATA                                            #
####################################################################################################
print('Loading data : \n')

data_dir = pathlib.Path('../../dataset')
val_dir = pathlib.Path('../../validation_set')

image_count_dat = len(list(data_dir.glob('*/*.*')))
print(f'    Dataset images    : {image_count_dat}')
image_count_val = len(list(val_dir.glob('*/*.*')))
print(f'    Validation images : {image_count_val}')

batch_size = 3
img_height = 200
img_width = 200


####################################################################################################
#                                       PREPROCESSING DATA                                         #
####################################################################################################
print('Preprocessing data :\n')

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_data.class_names
print(f'\n    Class names : {class_names}')

####################################################################################################
print(f'\nProcessing complete (time : {round(time.time()-start, 4)}s)')
