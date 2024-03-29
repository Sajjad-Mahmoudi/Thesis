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
def build_unet(input_shape: tuple, num_classes, encoder_filters=None, decoder_filters=None, bridge_filters=1024):
    if decoder_filters is None:
        decoder_filters = [512, 256, 128, 64]
    if encoder_filters is None:
        encoder_filters = [64, 128, 256, 512]

    inputs = Input(input_shape)  # 160 x 160 x 1

    # Encoder Block
    s1, p1 = encoder_block(inputs, encoder_filters[0])  # s1: 160 x 160 x 64, p1: 80 x 80 x 64
    s2, p2 = encoder_block(p1, encoder_filters[1])  # s2: 80 x 80 x 128, p2: 40 x 40 x 128
    s3, p3 = encoder_block(p2, encoder_filters[2])  # s3: 40 x 40 x 256, p3: 20 x 20 x 256
    s4, p4 = encoder_block(p3, encoder_filters[3])  # s4: 20 x 20 x 512, p4: 10 x 10 x 512

    # Bridge Block
    b1 = conv_block(p4, bridge_filters)  # b1: 10 x 10 x 1024

    # Decoder Block
    d1 = decoder_block(b1, s4, decoder_filters[
        0])  # Conv2DTrans: 20 x 20 x 512 => concat: 20 x 20 x 1024 => Conv2D: 20 x 20 x 512
    d2 = decoder_block(d1, s3, decoder_filters[
        1])  # Conv2DTrans: 40 x 40 x 256 => concat: 40 x 40 x 512 => Conv2D: 40 x 40 x 256
    d3 = decoder_block(d2, s2, decoder_filters[
        2])  # Conv2DTrans: 80 x 80 x 128 => concat: 80 x 80 x 256 => Conv2D: 80 x 80 x 128
    d4 = decoder_block(d3, s1, decoder_filters[
        3])  # Conv2DTrans: 160 x 160 x 64 => concat: 160 x 160 x 128 => Conv2D: 160 x 160 x 64

    if num_classes == 1:  # Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    # SoftMax Connection
    outputs = Conv2D(num_classes, (1, 1), padding="same", activation=activation)(d4)  # output: 160 x 160 x 4

    model = Model(inputs, outputs, name="U-Net")
    return model
