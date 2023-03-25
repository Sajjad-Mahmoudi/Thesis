from UNet_Binary_Segmentation.binary_UNet import build_unet
from UNet_Binary_Segmentation.binary_utils import read_hdf5
import tensorflow as tf
from sklearn.model_selection import train_test_split
import onnx
from onnx import version_converter
import tf2onnx
from onnxruntime.quantization import quantize_static, shape_inference, CalibrationDataReader
import numpy as np
from onnx_opcounter import calculate_params
import onnx_tool
from onnx_tool import create_ndarray_f32


# read and split data/masks
# to statically quantize ONNX model, a set of data/inputs, called calibration data, is required, but how many???
# Apparently, it doesn't matter how much data is used for clibration, either a single sample or many!
# I use (a part of) the validation data as calibration data
data, label = read_hdf5('UNet_Binary_Segmentation/data/EndoVis15+/patches_256x256_step256.h5')
x_train, X, y_train, Y = train_test_split(data, label, train_size=0.7, random_state=24)
x_valid, x_test, y_valid, y_test = train_test_split(X, Y, test_size=0.35, random_state=24)
del data, label, X, Y
del x_train, y_train, x_test, y_test


# build the keras model and load the desired weights
keras_model = build_unet((256, 256, 1), 1)
keras_model.load_weights('UNet_Binary_Segmentation/trained_models/weights/withAug/cp-597.ckpt')


# build the ONNX model
spec = (tf.TensorSpec((None, 256, 256, 1), tf.float32, name="input"),)
save_path_onnx_model = 'UNet_Binary_Segmentation/onnx_models/op9/withAug_600e.onnx'
# "tf2onnx.convert.from_keras" converts the keras model and saves it in the given output_path
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, opset=9, input_signature=spec,
                                           output_path=save_path_onnx_model)

# check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)
else:
    print("The model is valid!")


# preprocess the ONNX model including optimization and shape inference, before performing quantization
save_path_preprocessed_onnx_model = 'UNet_Binary_Segmentation/onnx_models/op9/preprocessed_withAug_600e.onnx'
shape_inference.quant_pre_process(input_model_path=save_path_onnx_model,
                                  output_model_path=save_path_preprocessed_onnx_model)


# quantize the preprocessed ONNX model
# implement a CalibrationDataReader to be used in "quantize_static"
class DataReader(CalibrationDataReader):
    def __init__(self, x_calibration, num_calibration_data):  # , y_calibration
        self.num_calibration_data = num_calibration_data
        self.x_calibration = x_calibration
        self.indices = self.get_index_of_x_calibration(len(self.x_calibration), self.num_calibration_data)
        self.enum_data_dicts = iter([{'input': self.dim_expansion(self.x_calibration[idx])} for idx in self.indices])
        # self.enum_data_dicts = [{'input': self.dim_expansion(self.x_calibration[idx])} for idx in self.indices]
        # self.y_calibration = y_calibration

    @staticmethod
    def get_index_of_x_calibration(high_limit, size):
        return np.random.choice(high_limit, size, replace=False)

    @staticmethod
    def dim_expansion(array):
        return np.expand_dims(array, 0)

    def get_next(self):
        return next(self.enum_data_dicts, None)


# quantize the model
data_reader = DataReader(x_valid, 200)
quantize_static(model_input=save_path_preprocessed_onnx_model,
                model_output='UNet_Binary_Segmentation/onnx_models/op9/quantized_withAug_600e.onnx',
                calibration_data_reader=data_reader)


# count the number of parameters of the ONNX model
m = onnx.load_model('UNet_Binary_Segmentation/onnx_models/preprocessed_withAug_600e.onnx')
params = calculate_params(m)
print('Number of parameters:', params)


# convert ONNX model from one opset version to another
# load the model to be converted
model_path = "UNet_Binary_Segmentation/onnx_models/op10/quantized_withAug_600e.onnx"
original_model = onnx.load(model_path)
# print(f"The model before conversion:\n{original_model}")

# apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, target_version=9)
onnx.save_model(converted_model, "UNet_Binary_Segmentation/onnx_models/op9/quantized_withAug_600e.onnx")
# print(f"The model after conversion:\n{converted_model}")


# Get the model profile
MODEL_PATH = r"C:/Users/mahmo/PycharmProjects/pythonProject/UNet_Binary_Segmentation/onnx_models/op9/" \
             r"preprocessed_withAug_600e.onnx"

# input size is dynamic as it depends on the batch: (batch, channel, height, width) => no channel = (16, 256, 256)
input = {'input': create_ndarray_f32((16, 256, 256))}
onnx_tool.model_profile(MODEL_PATH, input, savenode='./UNet_Binary_Segmentation/node_table.txt')
