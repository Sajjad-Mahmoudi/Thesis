import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from UNet_Binary_Segmentation.binary_utils import read_hdf5, multiple_images, img_format
from sklearn.model_selection import train_test_split
import cv2
import matplotlib

matplotlib.use('TkAgg')

# read and split data
data, label = read_hdf5('UNet_Binary_Segmentation/data/EndoVis15+/patches_256x256_step256.h5')
x_train, X, y_train, Y = train_test_split(data, label, train_size=0.7, random_state=24)
x_valid, x_test, y_valid, y_test = train_test_split(X, Y, test_size=0.35, random_state=24)
del data, label, X, Y

# define generator arguments
generator_args = dict(rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

'''img_data_gen_args = dict(rotation_range=90,
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
batch_size = 8
seed = 24
image_data_generator = ImageDataGenerator(**img_data_gen_args)
# image_data_generator.fit(X_train, augment=True, seed=seed)
train_img_generator = image_data_generator.flow(x_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(x_valid, seed=seed, batch_size=batch_size)

# generator for label
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
# mask_data_generator.fit(y_train, augment=True, seed=seed)
train_mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_valid, seed=seed, batch_size=batch_size)
'''
# generators for images and labels
seed = 24
batch_size = 8
data_generator = ImageDataGenerator(**generator_args)
train_img_generator = data_generator.flow(x_train, seed=seed, batch_size=batch_size)
train_mask_generator = data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_img_generator = data_generator.flow(x_valid, seed=seed, batch_size=batch_size)
valid_mask_generator = data_generator.flow(y_valid, seed=seed, batch_size=batch_size)
test_img_generator = data_generator.flow(x_test, seed=seed, batch_size=batch_size)
test_mask_generator = data_generator.flow(y_test, seed=seed, batch_size=batch_size)


# function to merge data and mask augmented patches together
def merge_generator(data_generator, label_generator):
    train_generator = zip(data_generator, label_generator)
    for (img, mask) in train_generator:
        yield img, mask


# train and validation augmented data
multiple_images([x_train[5]], ['Image'], 'Data Augmentation')

train_augmented = merge_generator(train_img_generator, train_mask_generator)
valid_augmented = merge_generator(valid_img_generator, valid_mask_generator)

# sanity check
x1 = train_img_generator.next()
y1 = train_mask_generator.next()
x2 = valid_img_generator.next()
y2 = valid_mask_generator.next()
# x, y = train_augmented.__next__()
multiple_images([x1[0], y1[0]], ['image', 'mask'])
multiple_images([x2[0], y2[0]], ['image', 'mask'])
# for i in range(0, 1):
#     image = x[i]
#     mask = y[i]
#     plt.subplot(1, 2, 1)
#     plt.imshow(image[:, :, 0], cmap='gray')
#     #plt.imshow((cv2.cvtColor(image[:, :, 0], cv2.COLOR_BGR2RGB)*255).astype(np.uint8))
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask[:, :, 0], cmap='gray')
#     plt.show()


# Illustrate how each transformation with/without reflection impacts the original image 
seed = 24
batch_size = 1
individual_transform_gen = ImageDataGenerator(width_shift_range=0.3).flow(x_train, seed=seed, batch_size=batch_size)
individual_transform_gen_with_reflect = ImageDataGenerator(width_shift_range=0.3, fill_mode='reflect').\
                                                           flow(x_train, seed=seed, batch_size=batch_size)
blank_gen = ImageDataGenerator().flow(x_train, seed=seed, batch_size=batch_size)

org_image = blank_gen.next()
transformed_image = individual_transform_gen.next()
transformed_image_with_reflect = individual_transform_gen_with_reflect.next()
multiple_images([org_image[0], transformed_image[0], transformed_image_with_reflect[0]],
                ['Original Image', 'Transformed Image', 'Transformed Image with Reflect'],
                'Width shift range')
