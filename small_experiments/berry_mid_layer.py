
import tensorflow as tf
import numpy as np
import glob
# import pandas as pd


model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

activations_dict = {}
for frame_idx, img_path in enumerate(glob.iglob('/Users/berryweinstein/studies/ECoG/video_data/frames/*.jpeg')):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    preds = model.predict(x)