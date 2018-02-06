import keras

from AE_subj_ori_ import get_encoding_network, get_deconding_network

print(keras.__version__)
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, BatchNormalization, Activation, Conv2DTranspose
from keras.models import Model

from datetime import datetime

from keras.optimizers import Adam
from run_multi_subject_experiment import prepare_data_for_experiment
import matplotlib.pyplot as plt




dt = datetime.now()
print(dt.microsecond)

import keras




def main():


    all_subjects = [
        "gcd",
    ];
    add_time_domain_noise = False
    number_of_k_fold = 10
    downsample_params = 8
    current_experiment_setting = "Color116ms"
    cross_validation_iter = 1

    train_data, train_tags, test_data_with_noise, test_tags, noise_shifts = prepare_data_for_experiment(all_subjects,
                                                                                                        add_time_domain_noise=add_time_domain_noise,
                                                                                                        current_experiment_setting=current_experiment_setting,
                                                                                                        downsample_params=downsample_params,
                                                                                                        number_of_k_fold=number_of_k_fold,
                                                                                                        cross_validation_iter=cross_validation_iter)
    # this is the size of our encoded representations


    classification_model = keras.models.load_model(r"C:\Users\ORI\.keras\model_signature.hdf5")



    auto_encoder_model  = keras.models.load_model(r"C:\temp\weights_862494.29-0.42.hdf5")


    auto_encoder_model.summary()
    print('********************************')

    autoencoder = auto_encoder_model
    autoencoder.summary()

    print('loading file from server')

    print('processed file, now running AE')

    x_train = train_data.reshape(train_data.shape[0] * train_data.shape[1], train_data.shape[2] * train_data.shape[3])
    x_tst = test_data_with_noise[0]
    x_test = x_tst.reshape(x_tst.shape[0] * x_tst.shape[1], x_tst.shape[2] * x_tst.shape[3])
    x_test = x_test



    decoded_imgs = auto_encoder_model.predict(x_test,verbose=1)
    import numpy as np
    print("evaluation on test samples",classification_model.evaluate(x_test.reshape(-1, 25, 55), test_tags))
    print("evaluation on auto encoder samples", classification_model.evaluate(decoded_imgs.reshape(-1, 25, 55), test_tags))


    np.save(r'c:\temp\test_data_gcd.npy', dict(decoded_imgs=decoded_imgs, x_test=x_test))
    print("done")




if __name__ == '__main__':
    main()
