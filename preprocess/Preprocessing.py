import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from UNet_Multiclass_Segmentation.utils import multiple_images, patch, write_hdf5, read_hdf5
from keras.utils.np_utils import normalize, to_categorical
from sklearn.utils import class_weight
import h5py

'''
EXPLORATORY DATA ANALYSIS and PREPROCESSING (EndoVis 15)
'''


''' put all raw/mask images in one folder '''
# # change the name of images to the pattern "OP + the operation number + _ + raw/mask + _ + image number + .png"
# # example: OP1_raw_001.png = the first raw image of the operation 1
# # example: OP2_mask_050 = the 10th mask image of the operation 2
# OP1Raw = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP1/Raw/*.png")
# OP1Mask = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP1/Masks/*instrument.png")
# OP2Raw = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP2/Raw/*.png")
# OP2Mask = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP2/Masks/*instrument.png")
# OP3Raw = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP3/Raw/*.png")
# OP3Mask = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP3/Masks/*instrument.png")
# OP4Raw = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP4/Raw/*.png")
# OP4Mask = glob.glob("UNet_Multiclass_Segmentation/Segmentation_Rigid_Training/OP4/Masks/*instrument.png")
# for i, filename in enumerate(OP4Raw):
#     os.rename(filename, filename[:-14] + 'OP4_' + 'raw_' + str(i + 120 + 1).zfill(3) + '.png')


''' put all raw/mask images in one list '''
# # two lists of raw and mask image paths
# path_all_raw = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/raw/*.png')
# path_all_mask = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/mask/*.png')
#
# # two lists/arrays of raw and mask image arrays
# '''?????????? I read images in grayscale, lets see the results and try read them in color ??????????? '''
# train_raw = [cv2.imread(img, 0) for img in path_all_raw]
# train_mask = [cv2.imread(img, 0) for img in path_all_mask]
# arr_train_raw = np.array(train_raw)
# arr_train_mask = np.array(train_mask)
#
# # show a sample of raw and the corresponding mask together
# multiple_images([train_raw[0], train_mask[0]])


''' Check the masks '''
# # check if all masks are labelled correctly, meaning all pixels should equal 0, 20, 40 ,or 60
# for i, img in enumerate(arr_train_mask):
#     if not np.all(np.isin(np.unique(img), [0, 20, 40, 60])):
#         print('the mask {} is wrongly labelled, so should be deleted'.format(i+1))
#
# # delete the raw images 19 and 144 and corresponding masks
# for p in [path_all_raw[18], path_all_raw[143], path_all_mask[18], path_all_mask[143]]:
#     os.remove(p)
#
# # rename the raw and mask images
# path_all_raw = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/raw/*.png')
# path_all_mask = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/mask/*.png')
# for data in [path_all_raw, path_all_mask]:
#     for i, filename in enumerate(data):
#         os.rename(filename, filename[:-7] + str(i + 1).zfill(3) + '.png')

############################### START FROM HERE #######################################################################
n_classes = 4

''' Images '''
# two lists/arrays of raw and mask images
'''?????????? read images in grayscale, lets see the results and try read them in color ??????????? '''
path_all_raw = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/raw/*.png')
path_all_mask = glob.glob('UNet_Multiclass_Segmentation/data/EndoVis15/training/mask/*.png')
train_raw = [cv2.imread(img, 0) for img in path_all_raw]
#train_raw_c = [cv2.imread(img) for img in path_all_raw]
train_mask = [cv2.imread(img, 0) for img in path_all_mask]
arr_train_raw = np.array(train_raw)
#arr_train_raw_c = np.array(train_raw_c)
arr_train_mask = np.array(train_mask)
# multiple_images([arr_train_mask[158]])
# np.unique(arr_train_raw[47])
# multiple_images([arr_train_raw[47], arr_train_raw_c[47]])
# a = arr_train_raw[47]
# b = arr_train_raw_c[47]

''' Image Characteristics'''
# # check the format of the images
# print('Raw Images: ', img_format(arr_train_raw[0]))
# print('Mask Images: ', img_format(arr_train_mask[0]))
#
# # shape of the images
# print('The raw image shape: {}'.format(arr_train_raw[0].shape))
# print('The mask image shape: {}'.format(arr_train_mask[0].shape))
#
# # check min and max of the mask images. Are the 3 channels of masks the same?
# print('A mask with 3 tools: Min = {} - Max = {}' .format(np.amin(arr_train_mask[1]), np.amax(arr_train_mask[1])))
# print('Is first channel equal to second one? ', np.all(arr_train_mask[1][:, :, 0] == arr_train_mask[1][:, :, 1]))
# print('Is third channel equal to second one? ', np.all(arr_train_mask[1][:, :, 2] == arr_train_mask[1][:, :, 1]))


''' Patch '''
# patch both raw and mask images
# UNet type architecture requires input image size be divisible by 2^N, where N is the number of MaxPooling layers
list_patch_raw = patch(arr_train_raw, 160, 160, 0, 80)
list_patch_mask = patch(arr_train_mask, 160, 160, 0, 80)
arr_patch_raw = np.array(list_patch_raw)
arr_patch_mask = np.array(list_patch_mask)


# save original patches as h5 file
# write_hdf5(arr_patch_raw, arr_patch_mask, r'UNet_Multiclass_Segmentation/data/EndoVis15/train_original_patch.h5')

# sanity check of some random patches (is each raw patch consistent with its corresponding mask?)
n = np.random.randint(0, arr_patch_raw.shape[0], 1)[0]
print(n)
multiple_images([arr_patch_raw[n], arr_patch_mask[n]])


''' Preprocess '''
# encode mask labels
le = LabelEncoder()
n_mask, h_mask, w_mask = arr_patch_mask.shape
flat_train_mask = arr_patch_mask.flatten()
encoded_train_mask_flatten = le.fit_transform(flat_train_mask)
labels = np.unique(encoded_train_mask_flatten)
print('the pixels of each mask are encoded as either {} (background), {} (tool 1), {} (tool 2), or {} (tool 3)'
      .format(labels[0], labels[1], labels[2], labels[3]))
encoded_train_mask = encoded_train_mask_flatten.reshape(n_mask, h_mask, w_mask)

# number of patches with different classes
counter0 = 0
counter01 = 0
counter012 = 0
counter0123 = 0
for patch in encoded_train_mask:
    if np.unique(patch).shape[0] == 1:
        counter0 += 1
    elif np.unique(patch).shape[0] == 2:
        counter01 += 1
    elif np.unique(patch).shape[0] == 3:
        counter012 += 1
    else:
        counter0123 += 1

# add a dimension to mask and raw arrays to be able to input them to the network
expanded_train_raw = np.expand_dims(arr_patch_raw, axis=3)  # (1896, 160, 160, 1)
expanded_train_mask = np.expand_dims(encoded_train_mask, axis=3)  # (1896, 160, 160, 1)

# normalize the raw images
#norm_expanded_train_raw = normalize(expanded_train_raw, axis=1)  # normalisation toward rows using L2-Norm
norm_expanded_train_raw = expanded_train_raw / 255.  # normalisation toward rows using L2-Norm

# convert masks to binary class matrix (categorical) = one-hot encoding
cat_expanded_train_mask = to_categorical(expanded_train_mask, num_classes=4)
print('shape of cat_expanded_train_mask = {}'.format(cat_expanded_train_mask.shape))

# To deal with imbalanced dataset (most pixels of masks belong to the first class that is background),
# assign each class a weight proportional to its frequency
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=labels, y=encoded_train_mask_flatten)
print('Class weight of the classes 0, 1, 2, and 3 are {}, {}, {}, and {}, respectively'
      .format(class_weights[0], class_weights[1], class_weights[2], class_weights[3]))

# save preprocessed patches as h5 file
write_hdf5(norm_expanded_train_raw, expanded_train_mask,
           r'UNet_Multiclass_Segmentation/data/EndoVis15/train_preprocessed_patch160_norm255_noOneHot.h5')

multiple_images([(255*norm_expanded_train_raw[3200]).astype(np.uint8), cat_expanded_train_mask[3200,:,:]])
norm_expanded_train_raw.dtype