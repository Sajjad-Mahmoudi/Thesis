from UNet_Binary_Segmentation.binary_UNet import build_unet
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#import splitfolders
from UNet_Binary_Segmentation.binary_utils import multiple_images, dice_metric, jaccard_distance_loss
from focal_loss import BinaryFocalLoss
import random
#import pandas as pd

'''# split all patches in different folders to have train, validation, and test data
input_folder = 'UNet_Binary_Segmentation/data/step128'
splitfolders.ratio(input_folder, output="UNet_Binary_Segmentation/data/split_step128", seed=24, ratio=(.7, .2, .1))'''

seed = 24
batch_size = 8

# define generator arguments
img_data_gen_args = dict(rescale=1 / 255.,
                         rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

mask_data_gen_args = dict(rescale=1 / 255.,
                          rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect')
'''from matplotlib import pyplot
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
img = load_img('UNet_Binary_Segmentation/data/1.png', grayscale=True)
msk = load_img('UNet_Binary_Segmentation/data/2.png', grayscale=True)
# convert to numpy array
data1 = img_to_array(img)
data2 = img_to_array(msk)
# expand dimension to one sample
samples1 = expand_dims(data1, 0)
samples2 = expand_dims(data2, 0)
# prepare iterator
gen = ImageDataGenerator(rescale=1/255., fill_mode='reflect')
it1 = gen.flow(samples1, batch_size=4,seed=24, shuffle=False)
it2 = gen.flow(samples2, batch_size=4, seed=24, shuffle=False)
for i in range(0, 4):
    batch1 = it1.next()
    batch2 = it2.next()
    # convert to unsigned integers for viewing
    image = (batch1[0] * 255).astype('uint8')
    mask = (batch2[0] * 255).astype('uint8')
    multiple_images([data1.astype('uint8'), image, mask])'''

# build image and mask generators for train
image_data_generator = ImageDataGenerator(**img_data_gen_args)
train_image_generator = image_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/train_images",
    seed=seed,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode=None)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
train_mask_generator = mask_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/train_masks",
    seed=seed,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode=None)

# build image and mask generators for validation
valid_image_generator = image_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/val_images",
    seed=seed,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode=None)

valid_mask_generator = mask_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/val_masks",
    seed=seed,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode=None)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(valid_image_generator, valid_mask_generator)

# sanity check
'''x1 = train_image_generator.next()
y1 = train_mask_generator.next()
multiple_images([x1[0, :, :, 0], y1[0, :, :, 0]])
x2 = valid_image_generator.next()
y2 = valid_mask_generator.next()
multiple_images([x2[0, :, :, 0], y2[0, :, :, 0]])'''
x1, y1 = train_generator.__next__()
x2, y2 = val_generator.__next__()
multiple_images([x1[0, :, :, 0], y1[0, :, :, 0]])
multiple_images([x2[0, :, :, 0], y2[0, :, :, 0]])

# training
model = build_unet((256, 256, 1), 1)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=BinaryFocalLoss(gamma=2), metrics=[dice_metric])
# model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(lr = 1e-3), loss=jaccard_distance_loss, metrics=[dice_metric])
model.summary()

num_train_data = len(os.listdir('UNet_Binary_Segmentation/data/split_step256/train_images/train'))
steps_per_epoch = np.ceil(num_train_data / batch_size).astype(np.uint8)
history = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch,
                    validation_steps=steps_per_epoch, epochs=50)
model.save('UNet_Binary_Segmentation/trained_weights/')

'''#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_metric']
#acc = history.history['accuracy']
val_acc = history.history['val_dice_metric']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()'''

# testing
test_batchSize = 8
model = tf.keras.models.load_model("UNet_Binary_Segmentation/trained_models/model_lastEpoch_withoutAug.h5",
                                   compile=False)

test_image_generator = image_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/test_images",
    seed=seed,
    batch_size=test_batchSize,
    color_mode='grayscale',
    class_mode=None)

test_mask_generator = mask_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/test_masks",
    seed=seed,
    batch_size=test_batchSize,
    color_mode='grayscale',
    class_mode=None)
test_generator = zip(test_image_generator, test_mask_generator)

# sanity check for test generator
x3, y3 = test_generator.__next__()
multiple_images([x3[0, :, :, 0], y3[0, :, :, 0]])

# plot random predictions from testing data together with the corresponding image and ground-truth/mask
x4, y4 = test_generator.__next__()
test_img_index = random.randint(0, x4.shape[0] - 1)
test_image = x4[test_img_index]
test_groundTruth = y4[test_img_index]
test_image_input = np.expand_dims(test_image, 0)
prediction = (model.predict(test_image_input)[0, :, :, 0] > 0.5).astype(np.uint8)

multiple_images([test_image, test_groundTruth, prediction])
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_groundTruth[:, :, 0], cmap='gray')
plt.subplot(233)
plt.title('Prediction')
plt.imshow(prediction, cmap='gray')
plt.show()

# IoU for a single image
n_classes = 2
IOU_keras = tf.keras.metrics.MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_groundTruth[:, :, 0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())

# calculate IoU and save the calculated values in "IoU-values" list for one batch of testing data
IoU_values = []
x5, y5 = test_generator.__next__()
for i in range(len(x5.shape[0])):
    img = x5[i]
    gt = y5[i]
    img_input = np.expand_dims(img, 0)
    prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
    IoU = tf.keras.metrics.MeanIoU(num_classes=n_classes)
    IoU.update_state(gt[:, :, 0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)
    print(IoU)  # print IOU for all predicted mask of one batch

# calculate the mean IoU for the all predictions of one batch
####### change it to a simple code instead of using dataframes ##########
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)
