# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:32:13 2018

@author: Doron

Kristiadi VAE, test with P300 data
"""

from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from run_multi_subject_experiment import prepare_data_for_experiment

m = 50
n_z = 2
n_epoch = 10


# Q(z|X) -- encoder
#inputs = Input(shape=(784,))
inputs = Input(shape=(1375,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# P(X|z) -- decoder
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(1375, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)

# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

vae.compile(optimizer='adam', loss=vae_loss)

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

vae.fit(x_train, x_train, batch_size=m, nb_epoch=n_epoch)

