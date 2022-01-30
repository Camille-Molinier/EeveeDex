from pickletools import optimize
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
from tensorflow import keras
from tensorflow.keras import layers
print(" Done\n")

####################################################################################################
#                                          LOADING DATA                                            #
####################################################################################################
print('Loading data : \n')

data_path = os.path.join('..', '..', 'dataset')

data_dir = pathlib.Path(f'{data_path}/training_set')
val_dir = pathlib.Path(f'{data_path}/validation_set')

image_count_dat = len(list(data_dir.glob('*/*.*')))
print(f'    Dataset images    : {image_count_dat}')
image_count_val = len(list(val_dir.glob('*/*.*')))
print(f'    Validation images : {image_count_val}')


####################################################################################################
#                                       PREPROCESSING DATA                                         #
####################################################################################################
print('Preprocessing data :\n')

batch_size = 3
img_height = 200
img_width = 200

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
#                                          NEURAL NETWORK                                          #
####################################################################################################
nb_classes = 9

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),        # Rescale all image
    layers.Conv2D(128,4, activation='relu'),                    # Convolution 
    layers.MaxPooling2D(),                                      # Choose max in area
    layers.experimental.preprocessing.Rescaling(1./255),        # Repeate with lower conv
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),                                           # Matrix -> Vector
    layers.Dense(64, activation='relu'),                        # "Standard nn"
    layers.Dense(nb_classes, activation='softmax')              # softmax for all classes probabilities
])

model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

logdir='../../dat/logs'

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                   embeddings_data=train_data)

model.fit(train_data, validation_data=val_data,  epochs=20, callbacks=[tensorboard_callback])


####################################################################################################
print(f'\nProcessing complete (time : {round(time.time()-start, 4)}s)')
