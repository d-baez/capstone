# pi_infer_bananas.py
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from time import time

MODEL_PATH = "banana_cnn_int8.tflite"
IMG_SIZE = (96, 96)
CLASS_NAMES = ["unripe", "ripe", "overripe"]

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

def preprocess(frame):
    # center crop to square then resize
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, (input_width, input_height))

    # int8/uint8 model expects 0–255 uint8
    input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
    return input_data

def run_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_index, input_data)
    t0 = time()
    interpreter.invoke()
    dt = (time() - t0) * 1000

    output_data = interpreter.get_tensor(output_index)[0]

    # For uint8: map back to float-ish softmax if you want,
    # but for a simple classifier, argmax is enough.
    pred_class = int(np.argmax(output_data))
    return CLASS_NAMES[pred_class], dt

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, dt = run_inference(frame)
        text = f"{label} ({dt:.1f} ms)"

        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        cv2.imshow("Banana ripeness", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()