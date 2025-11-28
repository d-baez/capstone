# convert_to_tflite.py
import tensorflow as tf

keras_model = tf.keras.models.load_model("banana_cnn.h5")

# Float32
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()
with open("banana_cnn_float32.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved banana_cnn_float32.tflite")

# FP16 (good compromise)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
with open("banana_cnn_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)
print("Saved banana_cnn_fp16.tflite")