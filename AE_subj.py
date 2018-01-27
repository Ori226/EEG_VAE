"""
The things needed to be changes:
1. the tutorial version used 'binary_crossentropy' as an objective function. This was possible because the images
were binary images. Here I'll try to use mse instead.

"""
from keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
import os
import shutil
cache_dir = r"c:\temp"

# this is the size of our encoded representations
encoding_dim = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
#input_img = Input(shape=(784,))
input_img = Input(shape=(1375,))
x = Reshape((1, 25, 55))(input_img)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = x

x_ = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x_ = UpSampling2D((2, 2))(x_)
x_ = Conv2D(8, (3, 3), activation='relu', padding='same')(x_)
x_ = UpSampling2D((2, 2))(x_)
x_ = Conv2D(16, (3, 3), activation='relu')(x_)
x_ = Flatten()(x_)





# "decoded" is the lossy reconstruction of the input
decoded = Dense(1375, activation='relu')(x_)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)



# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.output_layers[0]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.summary()
#from keras.datasets import mnist
#import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()
#
#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#print(x_train.shape)
#print(x_test.shape)

# load subject

#if subject_id == 'fat':
experiment_suffix="Color116ms"
subject_id = 'gcd'
url_format = "http://doc.ml.tu-berlin.de/bci/doron/VPfat_11_01_24/RSVP_{}VP{}.mat"

url = url_format.format(experiment_suffix, subject_id)
file_name = url.split("/")[-1]

from keras.utils.data_utils import get_file
print('loading file from server')

# local_file_location = os.path.join(cache_dirm, os.path.basename(file_name))
# if not os.path.isfile():
#     temp_file_name = get_file(file_name+".tmp", url, cache_subdir="p300_lstm")
#     shutil.move(temp_file_name, local_file_location)

from run_multi_subject_experiment import prepare_data_for_experiment
all_subjects = [
        "gcd",
    ];
add_time_domain_noise = False
number_of_channels = 55
number_of_k_fold = 10
downsample_params = 8
current_experiment_setting = "Color116ms"
eeg_sample_shape = (int(200 / downsample_params), number_of_channels)        
cross_validation_iter = 1

train_data, train_tags, test_data_with_noise, test_tags, noise_shifts = prepare_data_for_experiment(all_subjects,
                                add_time_domain_noise=add_time_domain_noise,
                                current_experiment_setting=current_experiment_setting,
                                downsample_params=downsample_params,
                                number_of_k_fold=number_of_k_fold,
                                cross_validation_iter=cross_validation_iter)

            
print('processed file, now running AE')

x_train = train_data.reshape(train_data.shape[0] * train_data.shape[1], train_data.shape[2]* train_data.shape[3])
x_tst = test_data_with_noise[0]
x_test = x_tst.reshape(x_tst.shape[0] * x_tst.shape[1], x_tst.shape[2]* x_tst.shape[3])

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

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