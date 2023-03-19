import cv2
import numpy as np
import matplotlib
from PIL import Image
from patchify import patchify
from math import gcd
import h5py
import keras.backend as K
import tensorflow as tf
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# check the format of images
def img_format(pathORarray, flag=1):
    if isinstance(pathORarray, str):
        sample = cv2.imread(pathORarray, flag)
        img = Image.fromarray(sample)
        print('The image is in "{}" format'.format(img.mode))
    elif isinstance(pathORarray, np.ndarray):
        img = Image.fromarray(pathORarray)
        print('The image is in "{}" format'.format(img.mode))


# plot image using cv2 (input: either path of the image or the numpy array)
def img_show_cv2(pathORarray, flag=1, win_name='Image'):
    if isinstance(pathORarray, str):
        sample = cv2.imread(pathORarray, flag)
        cv2.imshow(win_name, sample)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif isinstance(pathORarray, np.ndarray):
        cv2.imshow(win_name, pathORarray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# plot image using plt (input: either path of the image or the numpy array)
def img_show_plt(pathORarray, flag=1):  # needs to be more generic because BGR2RGB is defined in the function
    if isinstance(pathORarray, str):
        sample = cv2.imread(pathORarray, flag)
        plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
        plt.show(block=True)
    elif isinstance(pathORarray, np.ndarray):
        plt.imshow(cv2.cvtColor(pathORarray, cv2.COLOR_BGR2RGB))
        plt.show(block=True)


# plot multiple images using plt (input: a list of image numpy arrays)
def multiple_images(images: list[np.ndarray], titles: list[str], window_name=None) -> None:
    n: int = len(images)
    f = plt.figure(num=window_name, figsize=(8, 5))
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.title(titles[i])
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.show()


# function to patch the images
# array_images: an array of array images
# if the images are 2D, the "c_patch" should be equal to zero
# if h_patch and w_patch are not given, they will be equal to gcd(h_patch, w_patch) as default
# if step is not given, it will be equal to gcd(h_patch, w_patch) as default
# list_patches: a list of all patches that are arrays
def patch(array_images, h_patch, w_patch, c_patch, step=None):
    if step == h_patch:
        num_patch_for_h = array_images[0].shape[0] // step
        num_patch_for_w = array_images[0].shape[1] // step
    else:
        num_patch_for_h = array_images[0].shape[0] // step - 1
        num_patch_for_w = array_images[0].shape[1] // step - 1
    list_patches = []
    if step is None: step = gcd(h_patch, w_patch)

    if c_patch == 0:
        for img in array_images:
            patches = patchify(img, (h_patch, w_patch), step=step)
            for i in range(num_patch_for_h):
                for j in range(num_patch_for_w):
                    list_patches.append(patches[i, j])
    else:
        for img in array_images:
            patches = patchify(img, (h_patch, w_patch, c_patch), step=step)
            for i in range(num_patch_for_h):
                for j in range(num_patch_for_w):
                    list_patches.append(patches[i, j, 0])

    return list_patches


# function to save patches as hdf5 file
# "output_filename.h5" contains data (raw patches) and label (mask patches)
def write_hdf5(data, labels, output_filename):
    # x = data.astype(np.float32)
    # y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=labels, shape=labels.shape)
        # h.create_dataset()


# function to read the h5 file
# file: path of the h5 file
def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


# function for weighted categorical cross entropy
# Args: weights = a list/array of class weights
def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        tensor_weight = K.constant(weights, dtype=tf.float32)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * tensor_weight, axis=-1)

    return wcce


# DICE metric
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    return 2 * intersection / union


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    sum_ = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# perform morphological closing, which is a dilation operation followed by an erosion operation, with the given kernel
# The morphological closing operation is used to reduce noise and fill small gaps in the predicted regions
# of the binary mask
def post_process(prediction, kernel_size=5):
    """
    Applies morphological closing to the given binary mask

    Args:
        prediction (numpy.ndarray): A binary mask of the predicted regions.
        kernel_size (int): The size of the kernel to use in the morphological closing operation (default = 5).

    Returns:
        numpy.ndarray: The post-processed binary mask.
    """

    # Create a kernel of ones with the given size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply morphological closing to the binary mask using the kernel
    post_processed_pred = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
    return post_processed_pred
