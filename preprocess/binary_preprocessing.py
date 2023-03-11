import glob
import cv2
import numpy as np
from UNet_Binary_Segmentation.binary_utils import multiple_images, patch, write_hdf5, read_hdf5
from patchify import patchify
import os

# change the image name of new data to "001.png" to "438.png"
rawPng = []
for fol in range(1, 7):
    rawPng += glob.glob("UNet_Binary_Segmentation/data/15_testingData/images/" + "/*.png")
# rawJpg = glob.glob("UNet_Binary_Segmentation/data/images/*.jpg")
# raw = rawPng + rawJpg
mask = glob.glob("UNet_Binary_Segmentation/data/masks/*.png")
for i, filename in enumerate(rawPng):
    os.rename(filename, 'UNet_Binary_Segmentation/data/15_testingData/images/' + str(i + 1).zfill(3) + '.png')

# change the image name of old data (EndoVis15) to "439.png" to "597.png"
rawOld = glob.glob("UNet_Binary_Segmentation/data/old_images/*.png")
maskOld = glob.glob("UNet_Binary_Segmentation/data/old_masks/*.png")
for i, filename in enumerate(maskOld):
    os.rename(filename, 'UNet_Binary_Segmentation/data/masks/' + str(i + 439).zfill(3) + '.png')

# ######################################## Binarized ################################################################# #
# lists/arrays of all images and masks
path_all_image = glob.glob('UNet_Binary_Segmentation/data/images/*.png')
path_all_mask = glob.glob('UNet_Binary_Segmentation/data/masks/*.png')

# read the images and masks
arr_images = np.array([cv2.imread(img, 0) for img in path_all_image], dtype=np.uint8)
arr_masks = np.array([cv2.imread(img, 0) for img in path_all_mask], dtype=np.uint8)

# save patches with step 128/256 in the subfolders, "images_patch_step128/256" and "masks_patch_step128/256"
for ind, arr in enumerate([arr_images, arr_masks]):
    for i in range(len(arr)):
        if arr[i].shape[0] == 480:
            temp_arr = cv2.resize(arr[i], (768, 512), interpolation=cv2.INTER_CUBIC)
            patches = patchify(temp_arr, (256, 256), step=256)
            patches = np.resize(patches, (6, 256, 256))
            for j, p in enumerate(patches):
                if ind == 0:
                    cv2.imwrite('UNet_Binary_Segmentation/data/images_patch_step256/' + f"{i}_{j}" + '.png', p)
                else:
                    cv2.imwrite('UNet_Binary_Segmentation/data/masks_patch_step256/' + f"{i}_{j}" + '.png', p)
        else:
            patches = patchify(arr[i], (256, 256), step=256)
            patches = np.resize(patches, (20, 256, 256))
            for j, p in enumerate(patches):
                if ind == 0:
                    cv2.imwrite('UNet_Binary_Segmentation/data/images_patch_step256/' + f"{i}_{j}" + '.png', p)
                else:
                    cv2.imwrite('UNet_Binary_Segmentation/data/masks_patch_step256/' + f"{i}_{j}" + '.png', p)
# #################################################################################################################### #

# lists/arrays of all images and masks
path_all_image = glob.glob('UNet_Binary_Segmentation/data/images/*.png')
path_all_mask = glob.glob('UNet_Binary_Segmentation/data/15_testingData/masks/*.png')

# read the images and masks of old and new data
list_imagesNew = [cv2.imread(img, 0) for img in path_all_image[:438]]
list_masksNew = [cv2.imread(img, 0) for img in path_all_mask]
list_imagesOld = [cv2.imread(img, 0) for img in path_all_image[438:]]
list_masksOld = [cv2.imread(img, 0) for img in path_all_mask[438:]]
arr_imagesNew = np.array(list_imagesNew)  # dtype = unit8
arr_masksNew = np.array(list_masksNew)  # dtype = unit8
arr_imagesOld = np.array(list_imagesOld)  # dtype = unit8
arr_masksOld = np.array(list_masksOld)  # dtype = unit8

# binarize the masks of old data and replace them with old masks + sanity check
binary_arr_masksOld = np.where(arr_masksNew > 0, 255, 0).astype(np.uint8)
for i in range(len(binary_arr_masksOld)):
    cv2.imwrite('UNet_Binary_Segmentation/data/15_testingData/masks/' + str(i + 1).zfill(3) + '.png', binary_arr_masksOld[i])
n = np.random.randint(0, arr_imagesOld.shape[0], 1)[0]
print(n)
multiple_images([arr_imagesOld[n], binary_arr_masksOld[n]])
multiple_images([arr_imagesNew[n], arr_masksNew[n]])

# resize the old images and masks to be consistent with patch size of 256 x 256
# for step patch 128, only the height of old data needs to resize from 480 to 512
resized_arr_imagesOld = np.zeros((arr_imagesOld.shape[0], 512, 640), dtype=np.uint8)
resized_binary_arr_masksOld = np.zeros((binary_arr_masksOld.shape[0], 512, 640), dtype=np.uint8)
for i in range(len(arr_imagesOld)):
    resized_arr_imagesOld[i] = cv2.resize(arr_imagesOld[i], (640, 512), interpolation=cv2.INTER_CUBIC)
for i in range(len(arr_imagesOld)):
    resized_binary_arr_masksOld[i] = cv2.resize(binary_arr_masksOld[i], (640, 512), interpolation=cv2.INTER_CUBIC)

# for step patch 256, the height and width of old data both needs to resize
# from 480 to 512 and from 640 to 768, respectively
resized_arr_imagesOld = np.zeros((arr_imagesOld.shape[0], 512, 768), dtype=np.uint8)
resized_binary_arr_masksOld = np.zeros((binary_arr_masksOld.shape[0], 512, 768), dtype=np.uint8)
for i in range(len(arr_imagesOld)):
    resized_arr_imagesOld[i] = cv2.resize(arr_imagesOld[i], (768, 512), interpolation=cv2.INTER_CUBIC)
for i in range(len(arr_imagesOld)):
    resized_binary_arr_masksOld[i] = cv2.resize(binary_arr_masksOld[i], (768, 512), interpolation=cv2.INTER_CUBIC)

# patch both raw and mask images of old and new data
# UNet type architecture requires input image size being divisible by 2^N, where N is the number of MaxPooling layers
arr_patch_imagesOld = np.array(patch(resized_arr_imagesOld, 256, 256, 0, 256))
arr_patch_masksOld = np.array(patch(resized_binary_arr_masksOld, 256, 256, 0, 256))
arr_patch_imagesNew = np.array(patch(arr_imagesNew, 256, 256, 0, 256))
arr_patch_masksNew = np.array(patch(arr_masksNew, 256, 256, 0, 256))

# sanity check of some random patches (is each raw patch consistent with its corresponding mask?)
n = np.random.randint(0, arr_patch_imagesNew.shape[0], 1)[0]
# print(n)
multiple_images([arr_patch_imagesNew[n], arr_patch_masksNew[n]])
n = np.random.randint(0, arr_patch_imagesOld.shape[0], 1)[0]
# print(n)
multiple_images([arr_patch_imagesOld[n], arr_patch_masksOld[n]])

# combine old and new patches into one array
arr_patch_data = np.vstack((arr_patch_imagesNew, arr_patch_imagesOld))
arr_patch_label = np.vstack((arr_patch_masksNew, arr_patch_masksOld))

# add a dimension to mask and image arrays to be able to input them to the network
expanded_arr_patch_data = np.expand_dims(arr_patch_data, axis=3)
expanded_arr_patch_label = np.expand_dims(arr_patch_label, axis=3)

# normalization
norm_expanded_arr_patch_data = (expanded_arr_patch_data / 255.).astype(np.float32)
norm_expanded_arr_patch_label = (expanded_arr_patch_label / 255.).astype(np.float32)

# save preprocessed patches as h5 file
write_hdf5(norm_expanded_arr_patch_data, norm_expanded_arr_patch_label,
           r'UNet_Binary_Segmentation/data/hhhh.h5')

# sanity check
data, label = read_hdf5('UNet_Binary_Segmentation/data/patches_256x256_step128.h5')
n = np.random.randint(0, data.shape[0], 1)[0]
print(n)
multiple_images([data[n], label[n]])

