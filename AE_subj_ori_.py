"""
The things needed to be changes:
1. the tutorial version used 'binary_crossentropy' as an objective function. This was possible because the images
were binary images. Here I'll try to use mse instead.

"""
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


def get_encoding_network(input_layer):
    print(keras.backend.image_data_format())

    x = Reshape((25, 55, 1), input_shape=(1375,))(input_layer)

    x = Conv2D(64, (3, 3), padding='valid', strides=(1, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', strides=(1, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', strides=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = Flatten()(x)
    x = Dense(49)(x)
    return x


def get_deconding_network(input_layer):
    x = Reshape((7, 7, 1), input_shape=(49,))(input_layer)
    x = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid', data_format=None,
                        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid', data_format=None,
                        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid', data_format=None,
                        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = Conv2DTranspose(256, (9, 22), strides=(1, 1), padding='valid', data_format=None,
                        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='sigmoid')(x)
    x = Conv2DTranspose(1, (1, 18), strides=(1, 1), padding='valid', data_format=None,
                        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                        activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(x)
    x = Flatten()(x)
    return x


def main():
    # this is the size of our encoded representations

    input_img = Input(shape=(1375,))

    encoding_model = Model(input_img, get_encoding_network(input_img))
    encoding_model.summary()
    input_code = Input(shape=(49,))
    decoding_model = Model(input_code, get_deconding_network(input_code))

    auto_encoder_model = Model(input_img, decoding_model(encoding_model(input_img)))
    print('*****    encoding_model  *******')
    encoding_model.summary()
    print('*****    decoding_model  *******')
    decoding_model.summary()
    print('*****  auto_encoder_model*******')
    auto_encoder_model.summary()
    print('********************************')

    autoencoder = auto_encoder_model

    # this model maps an input to its encoded representation
    encoder = encoding_model

    decoder = decoding_model

    autoencoder.compile(optimizer=Adam(lr=0.0001), loss='mse')
    autoencoder.summary()

    print('loading file from server')


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

    print('processed file, now running AE')

    x_train = train_data.reshape(train_data.shape[0] * train_data.shape[1], train_data.shape[2] * train_data.shape[3])
    x_tst = test_data_with_noise[0]
    x_test = x_tst.reshape(x_tst.shape[0] * x_tst.shape[1], x_tst.shape[2] * x_tst.shape[3])
    random_number = str(dt.microsecond)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.00001)
    callback = keras.callbacks.ModelCheckpoint(r"c:\temp\weights_" + random_number + ".{epoch:02d}-{val_loss:.2f}.hdf5",
                                               monitor='val_loss', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='auto', period=1)

    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[callback, reduce_lr]
                    )

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # use Matplotlib (don't ask)


    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(25, 55))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(25, 55))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    main()
