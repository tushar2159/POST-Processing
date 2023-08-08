import os
import rasterio
import numpy as np
import tensorflow as tf
from keras.models import load_model

def weighted_binary_crossentropy(noncanal_weight, canal_weight):
    class_weights = [noncanal_weight, canal_weight]

    def loss(y_train, y_test):
        b_ce = tf.keras.backend.binary_crossentropy(y_train, y_test)
        weight_vector = y_train * canal_weight + (1. - y_train) * noncanal_weight
        weighted_b_ce = weight_vector * b_ce
        return tf.keras.backend.mean(weighted_b_ce)
    return loss

def run_predict(path, file, opf, threshold=0.5):
    canal_pixels = 2507608
    non_canal_pixels = 1295105192
    total_pixels = (canal_pixels + non_canal_pixels) / 2
    canal_weight = (total_pixels / canal_pixels) * 0.7
    noncanal_weight = (total_pixels / non_canal_pixels) * 0.3

    Resunetmodel = tf.keras.models.load_model(
        'model_150to250_checkpoint_90.h5',
        custom_objects={'loss': weighted_binary_crossentropy(noncanal_weight, canal_weight)}
    )

    with rasterio.open(os.path.join(path, file)) as r:
        image = r.read()
        profile = r.profile

    new_image = np.expand_dims(image, axis=3)
    new_image_trans = new_image / 255
    new_image_trans_ = np.repeat(new_image_trans, 3, axis=-1)

    profile.update(count=1)
    res_pred = (Resunetmodel.predict(new_image_trans_)[0, :, :, 0] > threshold).astype('uint8')
    res_pred_band = np.squeeze(res_pred)

    with rasterio.open(os.path.join(opf, file), 'w', **profile) as src:
        src.write(res_pred_band, 1)

    return res_pred, profile

path = r'F:/final thesis/Carto/Carto_image_patches'
files = os.listdir(path)

for file in files:
    opf = os.path.join(path, "Canal_Predict")
    os.makedirs(opf, exist_ok=True)

    if file.endswith(('tif', 'TIF', 'tiff')):
        if os.path.exists(os.path.join(opf, file)):
            continue

        print('Processing:', file)
        res_pred, profile = run_predict(path, file, opf)





