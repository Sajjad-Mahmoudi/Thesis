import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from UNet_Binary_Segmentation.binary_utils import multiple_images, dice_metric
import matplotlib
from matplotlib import pyplot as plt
import random
import os
from sklearn.metrics import balanced_accuracy_score
from miseval import evaluate
matplotlib.use('TkAgg')

seed = 24
test_batchSize = 8

# define generators for images and masks without augmentation
image_data_generator = ImageDataGenerator(rescale=1 / 255.)
mask_data_generator = ImageDataGenerator(rescale=1 / 255.)

# test generators
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
x1, y1 = test_generator.__next__()
multiple_images([(x1[0, :, :, 0] * 255).astype(np.uint8), y1[0, :, :, 0]], ['image', 'mask'])

# load model
model = keras.models.load_model("UNet_Binary_Segmentation/trained_models/wandb/epoch200_withoutAug.h5",
                                compile=False)

# plot random predictions from testing data together with the corresponding image and ground-truth/mask
x2, y2 = test_generator.__next__()
test_img_index = random.randint(0, x2.shape[0] - 1)
test_image = x2[test_img_index]
test_groundTruth = y2[test_img_index]
test_image_input = np.expand_dims(test_image, 0)
prediction = (model.predict(test_image_input)[0, :, :, 0] > 0.5).astype(np.float32)
multiple_images([test_image, test_groundTruth, prediction], ['Image', 'Label', 'Prediction'])

# calculate "MeanIoU" for entire testing dataset
num_test_data = len(os.listdir('UNet_Binary_Segmentation/data/split_step256/test_images/test'))
steps = np.ceil(num_test_data / test_batchSize).astype(np.uint8)
IoU_values = []
for j in range(0, steps):
    x3, y3 = test_generator.__next__()
    for i in range(0, x3.shape[0]):
        img = x3[i]
        gt = y3[i]
        img_input = np.expand_dims(img, 0)
        prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
        IoU = tf.keras.metrics.MeanIoU(num_classes=2)
        IoU.update_state(gt[:, :, 0], prediction)
        IoU = IoU.result().numpy()
        IoU_values.append(IoU)

# delete the IOU values equivalent to 1 from IoU_values and print the average of remaining
IoU_values_withoutOne = [iou for iou in IoU_values if iou != 1]
mean_iou = np.mean(IoU_values_withoutOne)
round_mean_iou = np.round(mean_iou, 4)
print('Average IOU of entire testing data excluding 1s:', round_mean_iou)

# calculate "Balanced Accuracy (BA)" for entire testing dataset
num_test_data = len(os.listdir('UNet_Binary_Segmentation/data/split_step256/test_images/test'))
steps = np.ceil(num_test_data / test_batchSize).astype(np.uint8)
BA_values = []
for j in range(0, steps):
    x4, y4 = test_generator.__next__()
    for i in range(0, x4.shape[0]):
        img = x4[i]
        gt = y4[i]
        img_input = np.expand_dims(img, 0)
        prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
        # ba = balanced_accuracy_score(gt[:, :, 0].reshape((-1)).astype(np.uint8), prediction.reshape((-1)))
        ba = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="BACC")
        BA_values.append(ba)

# delete the BA values equivalent to 1 from BA_values and print the average of remaining
BA_values_withoutOne = [ba for ba in BA_values if ba != 1]
mean_BA = np.mean(BA_values_withoutOne)
round_mean_BA = np.round(mean_BA, 4)
print('Average BA of entire testing data excluding 1s:', round_mean_BA)


num_test_data = len(os.listdir('UNet_Binary_Segmentation/data/split_step256/test_images/test'))
steps = np.ceil(num_test_data / test_batchSize).astype(np.uint8)
BA_values = []
for j in range(0, steps):
    x4, y4 = test_generator.__next__()
    for i in range(0, x4.shape[0]):
        img = x4[i]
        gt = y4[i]
        img_input = np.expand_dims(img, 0)
        prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
        # ba = balanced_accuracy_score(gt[:, :, 0].reshape((-1)).astype(np.uint8), prediction.reshape((-1)))
        ba = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="DSC")
        BA_values.append(ba)

# delete the BA values equivalent to 1 from BA_values and print the average of remaining
BA_values_withoutOne = [ba for ba in BA_values if ba != 0]
mean_BA = np.mean(BA_values_withoutOne)
round_mean_BA = np.round(mean_BA, 4)
print(round_mean_BA)
