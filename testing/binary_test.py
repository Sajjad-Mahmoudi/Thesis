from UNet_Binary_Segmentation.binary_UNet import build_unet
from UNet_Binary_Segmentation.binary_utils import read_hdf5, multiple_images, post_process
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as ort
from miseval import evaluate
from keras.models import load_model
from model_profiler import model_profiler
import time
import matplotlib

matplotlib.use('TkAgg')

# read and split data
data, label = read_hdf5('UNet_Binary_Segmentation/data/EndoVis15+/patches_256x256_step256.h5')
x_train, X, y_train, Y = train_test_split(data, label, train_size=0.7, random_state=24)
x_valid, x_test, y_valid, y_test = train_test_split(X, Y, test_size=0.35, random_state=24)
del data, label, X, Y
del x_train, y_train, x_valid, y_valid

# load model
model = load_model("UNet_Binary_Segmentation/test", compile=False)
onnx_model = onnx.load('UNet_Binary_Segmentation/onnx_models/quantized_withAug_600e.onnx')

# load weight
model = build_unet((256, 256, 1), 1)
# s = tf.train.load_checkpoint('UNet_Binary_Segmentation/trained_models/weights/Run_Remote5/cp-398.ckpt')
model.load_weights('UNet_Binary_Segmentation/trained_models/weights/withAug/cp-597.ckpt')
model.save('UNet_Binary_Segmentation/test')

# model information
# BFLOPs = number of floating point operations / one billion
# MFLOPs = number of floating point operations / one million
model_profile = model_profiler(model, Batch_size=16, use_units=['GPU IDs', 'BFLOPs', 'GB', 'Million', 'MB'])
print(model_profile)

# plot random predictions from testing data together with the corresponding image and ground-truth/mask for both keras
# and onnx models
n = np.random.randint(0, len(x_test))
# counter = iter([x for x in range(100)])
# n = next(counter)
x1, y1 = x_test[n], y_test[n]
test_image_input = np.expand_dims(x1, 0)

keras_pred = (model.predict(test_image_input)[0, :, :, 0] > 0.5).astype(np.float32)
post_processed_keras_pred = post_process(keras_pred, kernel_size=5)
output_names = [n.name for n in onnx_model.graph.output]
sess = ort.InferenceSession('UNet_Binary_Segmentation/onnx_models/quantized_withAug_600e.onnx')
onnx_pred = sess.run(output_names, {"input": test_image_input})  # onnx_pred in np.float32

multiple_images([x1[:, :, 0], y1[:, :, 0], keras_pred], ['Image', 'Label', 'Prediction'], 'Keras Model')
multiple_images([x1[:, :, 0], y1[:, :, 0], post_processed_keras_pred], ['Image', 'Label', 'Prediction'],
                'Keras Model with Post Processing (Kernel size = 5)')
multiple_images([x1[:, :, 0], y1[:, :, 0], np.where(onnx_pred[0][0, :, :, 0] > 0.5, np.float32(1), np.float32(0))],
                ['Image', 'Label', 'Prediction'], 'ONNX Model')

# calculate such evaluation metrics as "Mean IoU (MIoU)", "Mean Dice (MDSC)", "Balanced Accuracy (BA)",
# "Sensitivity (SENS/TPR)", and "Specificity (SPEC/TNR)" for testing dataset
num_test_data = len(x_test)
output_names = [n.name for n in onnx_model.graph.output]
sess = ort.InferenceSession('UNet_Binary_Segmentation/onnx_models/quantized_withAug_600e.onnx')
IoU_values = []
DSC_values = []
BA_values = []
SENS_values = []
SPEC_values = []

for j in range(num_test_data):
    img = x_test[j]
    gt = y_test[j]
    img_input = np.expand_dims(img, 0)
    # prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
    prediction = post_process((model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8), kernel_size=5)
    # prediction = np.where(sess.run(output_names, {"input": img_input})[0][0, :, :, 0] > 0.5, np.uint8(1), np.uint8(0))
    IoU = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="IoU", multi_class=True)
    DSC = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="DSC", multi_class=True)
    BA = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="BACC")
    SENS = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="SENS", multi_class=True)
    SPEC = evaluate(gt[:, :, 0].astype(np.uint8), prediction, metric="SPEC", multi_class=True)
    if np.all(IoU == np.array([1., 0.])):
        IoU_values.append(IoU[0])
    else:
        IoU_values.append(np.mean(IoU))
    DSC_values.append(np.mean(DSC))
    BA_values.append(BA)
    if np.all(SENS == np.array([1., 0.])):
        SENS_values.append(SENS[0])
    else:
        SENS_values.append(np.mean(SENS))
    if np.all(SPEC == np.array([0., 1.])):
        SPEC_values.append(SPEC[1])
    else:
        SPEC_values.append(np.mean(SPEC))

# calculate metrics including all testing data
print('Mean IOU for entire testing data:', np.mean(IoU_values))
print('Mean DSC for entire testing data:', np.mean(DSC_values))
print('Mean BA for entire testing data:', np.mean(BA_values))
print('Mean SENS for entire testing data:', np.mean(SENS_values))
print('Mean SPEC for entire testing data:', np.mean(SPEC_values))

# delete values equivalent to 1 to exclude the testing samples that only have background in both label and prediction
IoU_values_withoutOne = [iou for iou in IoU_values if iou != 1]
DSC_values_withoutOne = [dsc for dsc in DSC_values if dsc != 1]
BA_values_withoutOne = [ba for ba in BA_values if ba != 1]
SENS_values_withoutOne = [sens for sens in SENS_values if sens != 1]
SPEC_values_withoutOne = [spec for spec in SPEC_values if spec != 1]

round_mean_IoU = np.round(np.mean(IoU_values_withoutOne), 6)
round_mean_DSC = np.round(np.mean(DSC_values_withoutOne), 6)
round_mean_BA = np.round(np.mean(BA_values_withoutOne), 6)
round_mean_SENS = np.round(np.mean(SENS_values_withoutOne), 6)
round_mean_SPEC = np.round(np.mean(SPEC_values_withoutOne), 6)

print('Average IOU of entire testing data excluding 1s:', round_mean_IoU)
print('Average DSC of entire testing data excluding 1s:', round_mean_DSC)
print('Average BA of entire testing data excluding 1s:', round_mean_BA)
print('Average SENS of entire testing data excluding 1s:', round_mean_SENS)
print('Average SPEC of entire testing data excluding 1s:', round_mean_SPEC)

# Mean IOU calculation using built-in function of TensorFlow
tf_IoU_values = []
for i in range(num_test_data):
    img = x_test[i]
    gt = y_test[i]
    img_input = np.expand_dims(img, 0)
    prediction = (model.predict(img_input)[0, :, :, 0] > 0.5).astype(np.uint8)
    IoU = tf.keras.metrics.MeanIoU(num_classes=2)
    IoU.update_state(gt[:, :, 0].astype(np.uint8), prediction)
    IoU = IoU.result().numpy()
    tf_IoU_values.append(IoU)

# calculate the mean IOU including all testing data
print('(TF) Average IOU of entire testing data:', sum(tf_IoU_values) / len(tf_IoU_values))

# delete the IOU values equivalent to 1 from IoU_values and print the average of remaining
tf_IoU_values_withoutOne = [iou for iou in tf_IoU_values if iou != 1]
tf_mean_iou = np.mean(tf_IoU_values_withoutOne)
tf_round_mean_iou = np.round(tf_mean_iou, 4)
print('(TF) Average IOU of entire testing data excluding 1s:', tf_round_mean_iou)

# calculate inference time for both Keras and ONNX models
num_test_data = len(x_test)
output_names = [n.name for n in onnx_model.graph.output]
sess = ort.InferenceSession('UNet_Binary_Segmentation/onnx_models/quantized_withAug_600e.onnx')
start_time = time.time()
# for k in range(num_test_data):
#     # prediction = (model.predict(np.expand_dims(x_test[k], 0))[0, :, :, 0] > 0.5).astype(np.uint8)
#     prediction = np.where(sess.run(output_names, {"input": np.expand_dims(x_test[k], 0)})[0][0, :, :, 0] > 0.5,
#                           np.uint8(1), np.uint8(0))
prediction = np.where(sess.run(output_names, {"input": np.expand_dims(x_test[24], 0)})[0][0, :, :, 0] > 0.5,
                      np.uint8(1), np.uint8(0))
stop_time = time.time()
duration = stop_time - start_time
hours = duration // 3600
minutes = (duration - (hours * 3600)) // 60
seconds = duration - ((hours * 3600) + (minutes * 60))
msg = f'Inference time for all testing data was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds'
print(msg)
