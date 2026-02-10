# convert_watermelon_to_tflite.py
import tensorflow as tf
from pathlib import Path

keras_model = tf.keras.models.load_model("watermelon_cnn.keras")

# Float32 version
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open("watermelon_cnn_float32.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved watermelon_cnn_float32.tflite")

# FP16 version (good for Pi)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
with open("watermelon_cnn_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)
print("Saved watermelon_cnn_fp16.tflite")

# INT8 version (fastest on Pi)
IMG_SIZE = (224, 224)
BATCH_SIZE = 1
CALIB_DIR = Path("data_watermelon/train")

calib_ds = tf.keras.utils.image_dataset_from_directory(
    CALIB_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
)
calib_ds = calib_ds.take(100)

def representative_data_gen():
    for batch in calib_ds:
        yield [tf.cast(batch, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8 = converter.convert()
with open("watermelon_cnn_int8.tflite", "wb") as f:
    f.write(tflite_int8)
print("Saved watermelon_cnn_int8.tflite")
