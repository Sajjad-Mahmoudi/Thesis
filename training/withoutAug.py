#-*- coding: utf-8 -*-
"""binary_heavyUnet

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/103uwcM3MZyP4j8ha0LDe2PXS-rQw7isa
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import keras.backend as K

import time
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

import wandb
from wandb.keras import WandbCallback
wandb.login()

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import Activation, MaxPool2D, Concatenate

# 2 consecutive convolution operations with batch normalization (NOTE: Batch normalization is not in the original paper.
# Most probably, they did not batch normalization because the paper presenting batch normalization was released
# later than this paper)
def conv_block(input, num_filters, kernel_size=(3, 3), activation='relu', **kwargs):
    x = Conv2D(num_filters, kernel_size=kernel_size, padding="same", **kwargs)(input)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, kernel_size=kernel_size, padding="same", **kwargs)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


# Encoder block = Conv_block followed by MaxPooling
def encoder_block(input, num_filters, **kwargs):
    s = conv_block(input, num_filters, **kwargs)
    p = MaxPool2D((2, 2))(s)
    return s, p


# Decoder block = deconv + concat + conv_block
def decoder_block(input, skip_features, num_filters, kernel_size=(2, 2), **kwargs):
    b = Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=2, padding="same", **kwargs)(input)
    d = Concatenate()([b, skip_features])
    d = conv_block(d, num_filters)
    return d


# UNet
# input_shape = (height, width, num_channel)
def build_unet(input_shape, num_classes, encoder_filters=None, decoder_filters=None, bridge_filters=1024):
    if decoder_filters is None:
        decoder_filters = [512, 256, 128, 64]
    if encoder_filters is None:
        encoder_filters = [64, 128, 256, 512]

    inputs = Input(input_shape)  # 256 x 256 x 1

    # Encoder Block
    s1, p1 = encoder_block(inputs, encoder_filters[0])  # s1: 256 x 256 x 64, p1: 128 x 128 x 64
    s2, p2 = encoder_block(p1, encoder_filters[1])  # s2: 128 x 128 x 128, p2: 64 x 64 x 128
    s3, p3 = encoder_block(p2, encoder_filters[2])  # s3: 64 x 64 x 256, p3: 32 x 32 x 256
    s4, p4 = encoder_block(p3, encoder_filters[3])  # s4: 32 x 32 x 512, p4: 16 x 16 x 512

    # Bridge Block
    b1 = conv_block(p4, bridge_filters)  # b1: 16 x 16 x 1024

    # Decoder Block
    d1 = decoder_block(b1, s4, decoder_filters[0])
    # Conv2DTrans: 32 x 32 x 512 => concat: 32 x 32 x 1024 => Conv2D: 32 x 32 x 512
    d2 = decoder_block(d1, s3, decoder_filters[1])
    # Conv2DTrans: 64 x 64 x 256 => concat: 64 x 64 x 512 => Conv2D: 64 x 64 x 256
    d3 = decoder_block(d2, s2, decoder_filters[2])
    # Conv2DTrans: 128 x 128 x 128 => concat: 128 x 128 x 256 => Conv2D: 128 x 128 x 128
    d4 = decoder_block(d3, s1, decoder_filters[3])
    # Conv2DTrans: 256 x 256 x 64 => concat: 256 x 256 x 128 => Conv2D: 256 x 256 x 64

    if num_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    # Sigmoid Connection
    outputs = Conv2D(num_classes, (1, 1), padding="same", activation=activation)(d4)  # output: 256 x 256 x 1

    model = Model(inputs, outputs, name="U-Net")
    return model

def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

data, label = read_hdf5("/scratch/brussel/106/vsc10602/patches_256x256_step256.h5")
x_train, X, y_train, Y = train_test_split(data, label, train_size=0.7, random_state=24)

del data, label
x_valid, x_test, y_valid, y_test = train_test_split(X, Y, test_size=0.35, random_state=24)
del X, Y

#tf.keras.backend.clear_session()
h = w = 256
c = 1
model = build_unet((h, w, c), 1)

def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2 * intersection / union

#run.finish()

configs = dict(
    batch_size = 16,
	augmentation = 'False',
    epochs = 400,
    lr = 0.001,
    opt = 'Adam',
    loss = 'sigmoid_focal_crossentropy',
    metrics = 'Dice',
    architecture = 'Heavy_UNet',
    dataset = 'EndoVis15+',
    framework = 'tensorflow',
    patch_step = 256,
    patch_size = '256x256',
	)

id = wandb.util.generate_id()
run = wandb.init(project='Binary_Seg', name='Run_Remote4', id=id, config=configs, save_code=True)
config = wandb.config

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
    
'''def focal_loss(alpha=None, beta=None, gamma_f=2.):
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * K.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = K.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = K.mean(K.sum(focal_loss, axis=[-1]))
        return focal_loss
        
    return loss_function'''

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=[dice_metric])

num_epoch = config.epochs - wandb.run.step
checkpoint_filepath = '/data/brussel/106/vsc10602/cp-{epoch:03d}.ckpt'
save_weight_every50Epochs = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                save_freq=int(np.ceil(len(x_train)/ config.batch_size).astype(np.uint8) * 50),
                                                                #monitor='val_loss',
                                                                verbose=1
                                                                )

start_time = time.time()
history = model.fit(x_train, y_train, batch_size=config.batch_size, epochs=num_epoch, validation_data=(x_valid, y_valid), shuffle=False,
					initial_epoch=wandb.run.step, callbacks=[save_weight_every50Epochs, WandbCallback(training_data=(x_train, y_train),
																									  validation_data=(x_valid, y_valid),
																									  labels=["Background", "Tool"], 
																									  predictions=2,
																									  input_type='image', 
																									  output_type='segmentation_mask',
																									  #log_evaluation=True,  # table of validation data and model's predictions
																									  #log_evaluation_frequency=5, 
																									  class_colors=([1., 0., 0.], [0., 0., 1.]),                                      
																									  #log_gradients=True,
																									  #log_weights=True,
																									  monitor='val_loss',
																									  mode='min',
																									  )])
											
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


run.finish()