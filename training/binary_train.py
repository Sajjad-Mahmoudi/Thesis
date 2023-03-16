from UNet_Binary_Segmentation.binary_UNet import build_unet
from UNet_Binary_Segmentation.binary_utils import read_hdf5, multiple_images, dice_metric
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import numpy as np
import cv2
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import segmentation_models as sm

# read and split data
data, label = read_hdf5('UNet_Binary_Segmentation/data/patches_256x256_step256.h5')
x_train, X, y_train, Y = train_test_split(data, label, train_size=0.7, random_state=24)
x_valid, x_test, y_valid, y_test = train_test_split(X, Y, test_size=0.35, random_state=24)
del data, label, X, Y

# define generator arguments
img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',)
                          #preprocessing_function=lambda pixel: np.where(pixel > 0, 1, 0).astype(pixel.dtype))

# generator for data
batch_size = 2
seed = 24
image_data_generator = ImageDataGenerator(**img_data_gen_args)
train_img_generator = image_data_generator.flow(x_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(x_valid, seed=seed, batch_size=batch_size)

# generator for label
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
train_mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_valid, seed=seed, batch_size=batch_size)

train_generator = zip(train_img_generator, train_mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

# sanity check
x1, y1 = train_generator.__next__()
x2, y2 = val_generator.__next__()
multiple_images([x1[0, :, :, 0], y1[0, :, :, 0]], ['image', 'mask'])
multiple_images([x2[0, :, :, 0], y2[0, :, :, 0]], ['image', 'mask'])

# compile the model
h = w = 256
c = 1
model = build_unet((h, w, c), 1)
model.compile(optimizer='Adam', loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=[dice_metric])
# model.summary()
# tf.keras.utils.plot_model(model, "Binary_UNet.png", show_shapes=True)

# fit the model
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.1)
history = model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=2, validation_data=(x_valid, y_valid),
                    shuffle=False)
                    #class_weight={0: class_weights[0], 1: class_weights[1], 2: class_weights[3], 3: class_weights[4]})

# save the trained model
model.save('test.hdf5')

# load pre-trained weights
model.load_weights('UNet_Binary_Segmentation/trained_weights/model-best.h5')

# load model
load_model("UNet_Binary_Segmentation/trained_weights/model-best.h5", compile=False,)
           #custom_objects={'focal_loss_fixed': sm.losses.categorical_focal_loss})

# evaluate the model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


im = cv2.imread('UNet_Multiclass_Segmentation/te.png', 0)
multiple_images([im])
multiple_images([im[300:460, 140:300]])
multiple_images([im[140:300, 240:400]])
multiple_images([im[40:200, 440:600]])
multiple_images([im[140:300, 180:340]])
#im_norm = normalize(im, axis=1)
im_norm = im / 255.
f = np.expand_dims(im_norm, axis=(0, 3))
tr = f[:, 300:460, 140:300]
tr = f[:, 140:300, 240:400]
tr = f[:, 40:200, 440:600]
tr = f[:, 140:300, 180:340]
pred = model.predict(tr)
pred_argmax = np.argmax(pred, axis=3)
multiple_images([(pred_argmax[0]*20).astype(np.uint8)])#, (tr[0]).astype(np.uint8)])

im = cv2.imread('UNet_Multiclass_Segmentation/te2.png', 0)
multiple_images([im])
multiple_images([im[300:460, 140:300]])
multiple_images([im[140:300, 240:400]])
multiple_images([im[40:200, 440:600]])
multiple_images([im[140:300, 180:340]])
#im_norm = normalize(im, axis=1)
im_norm = im / 255.
f = np.expand_dims(im_norm, axis=(0, 3))
tr = f[:, 300:460, 140:300]
tr = f[:, 140:300, 240:400]
tr = f[:, 40:200, 440:600]
tr = f[:, 140:300, 180:340]
pred = model.predict(tr)
pred_argmax = np.argmax(pred, axis=3)
multiple_images([(pred_argmax[0]*20).astype(np.uint8)])

im = cv2.imread('UNet_Multiclass_Segmentation/te3.png', 0)
multiple_images([im])
multiple_images([im[300:460, 140:300]])
multiple_images([im[140:300, 240:400]])
multiple_images([im[40:200, 440:600]])
multiple_images([im[140:300, 180:340]])
multiple_images([im[0:160, 100:260]])
#im_norm = normalize(im, axis=1)
im_norm = im / 255.
f = np.expand_dims(im_norm, axis=(0, 3))
tr = f[:, 300:460, 140:300]
tr = f[:, 140:300, 240:400]
tr = f[:, 40:200, 440:600]
tr = f[:, 140:300, 180:340]
tr = f[:, 0:160, 100:260]
pred = model.predict(tr)
pred_argmax = np.argmax(pred, axis=3)
multiple_images([(pred_argmax[0]*20).astype(np.uint8)])


# examples from validation data
n = np.random.randint(0, x_valid.shape[0], 1)[0]
print(n)
pred = model.predict(np.expand_dims(x_valid[n], axis=0))
pred_argmax = np.argmax(pred, axis=3)
multiple_images([(pred_argmax[0]*20).astype(np.uint8), y_valid[n, :, :], (x_valid[n]*255).astype(np.uint8)])

multiple_images([(data[2908]*255).astype(np.uint8), label[2908, :, :]])