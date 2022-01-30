import time

from numpy import extract
start = time.time()

import os
os.system('cls')

print("Modules importation :\n")
print(f"{'    Standard modules' :-<50}", end="")
import pathlib
print(" Done\n")

# print(f"{'    Tensorflow modules' :-<50}", end="")
# import tensorflow as tf
# print(" Done\n")

####################################################################################################
#                                          LOADING DATA                                            #
####################################################################################################
data_dir = pathlib.Path('../../dataset')

print(data_dir)

image_count = len(list(data_dir.glob('*/*.*')))
print(image_count)


# image_count = len(list(data_dir.glob('*/*')))
# print(image_count)

# batch_size = 3
# img_height = 200
# img_width = 200

# train_data = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=42,
#   image_size=(img_height, img_width),
#   batch_size=batch_size,
#   )


print(f'\nProcessing complete (time : {round(time.time()-start, 4)}s)')
