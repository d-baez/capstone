# convert_to_tflite_int8.py
import tensorflow as tf
import pathlib

keras_model = tf.keras.models.load_model("banana_cnn.h5")

IMG_SIZE = (96, 96)
BATCH_SIZE = 1
CALIB_DIR = pathlib.Path("data/train")  # or a subset

calib_ds = tf.keras.utils.image_dataset_from_directory(
    CALIB_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode=None,
)

calib_ds = calib_ds.take(100)  # limit for speed

def representative_data_gen():
    for batch in calib_ds:
        # batch shape: (1, 96, 96, 3)
        yield [tf.cast(batch, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8 = converter.convert()
with open("banana_cnn_int8.tflite", "wb") as f:
    f.write(tflite_int8)

print("Saved banana_cnn_int8.tflite")