from UNet_Binary_Segmentation.binary_UNet import build_unet
from UNet_Binary_Segmentation.binary_utils import multiple_images, dice_metric, jaccard_distance_loss
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import time
import wandb
from wandb.keras import WandbCallback
wandb.login()

# build model
h = w = 256
c = 1
model = build_unet((h, w, c), 1)

# wandb configuration
configs = dict(
    batch_size=16,
    epochs=5,
    lr=0.001,
    opt='Adam',
    loss='sigmoid_focal_crossentropy',
    metrics='Dice',
    architecture='Heavy_UNet',
    dataset='EndoVis15+',
    framework='tensorflow',
    patch_step='256',
    patch_size='256x256',
)
# id = wandb.util.generate_id()
run = wandb.init(project='Binary_Seg', name='Run_2', config=configs, id='3na9a1lu', resume='must')
config = wandb.config
# print(id)
# print(wandb.run.step)

# define generator arguments
seed = 24
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

# build image and mask generators for train
image_data_generator = ImageDataGenerator(**img_data_gen_args)
train_image_generator = image_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/train_images",
    seed=seed,
    batch_size=config.batch_size,
    color_mode='grayscale',
    class_mode=None)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
train_mask_generator = mask_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/train_masks",
    seed=seed,
    batch_size=config.batch_size,
    color_mode='grayscale',
    class_mode=None)

# build image and mask generators for validation
valid_image_generator = image_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/val_images",
    seed=seed,
    batch_size=config.batch_size,
    color_mode='grayscale',
    class_mode=None)

valid_mask_generator = mask_data_generator.flow_from_directory(
    "UNet_Binary_Segmentation/data/split_step256/val_masks",
    seed=seed,
    batch_size=config.batch_size,
    color_mode='grayscale',
    class_mode=None)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(valid_image_generator, valid_mask_generator)

# training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss=tfa.losses.SigmoidFocalCrossEntropy(),
              metrics=[dice_metric])

num_train_data = len(os.listdir('/content/drive/MyDrive/Thesis/binary/split_step256/train_masks/train/'))
steps_per_epoch = np.ciel(num_train_data / config.batch_size).astype(np.uint8)
num_val_data = len(os.listdir('/content/drive/MyDrive/Thesis/binary/split_step256/val_masks/val/'))
validation_steps = np.ceil(num_val_data / config.batch_size).astype(np.uint8)

start_time = time.time()
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=config.epochs,
                    validation_data=val_generator, validation_steps=validation_steps, initial_epoch=wandb.run.step,
                    callbacks=[WandbCallback(validation_data=val_generator,
                                             generator=val_generator,
                                             validation_steps=validation_steps,
                                             labels=["Background", "Tool"],
                                             predictions=2,
                                             input_type='image',
                                             output_type='segmentation_mask',
                                             # log_evaluation=True,
                                             # log_evaluation_frequency=5,
                                             class_colors=([1., 0., 0.], [0., 0., 1.]),
                                             # log_gradients=True,
                                             # log_weights=True,
                                             monitor='val_loss',
                                             mode='min',
                                             )])
# model.save('/content/drive/MyDrive/split_step256/Run_1.hdf5')
print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
