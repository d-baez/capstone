# watermelon_detect_and_classify.py
import cv2
import numpy as np
import tensorflow as tf
import requests
import base64
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional

class WatermelonRipenessSystem:
    def __init__(self,
                 roboflow_api_key: str,
                 roboflow_model_id: str,
                 classification_model_path: str = "watermelon_cnn.keras",
                 use_tflite: bool = False,
                 tflite_model_path: Optional[str] = None):
        """
        Combined detection + classification system for watermelon ripeness

        Args:
            roboflow_api_key: Your Roboflow API key
            roboflow_model_id: Format: "project/version"
            classification_model_path: Path to your trained CNN (.keras)
            use_tflite: Use TFLite model for faster inference on edge devices
            tflite_model_path: Path to TFLite model (if use_tflite=True)
        """
        print("Initializing Watermelon Ripeness Detection System...")

        # Load Roboflow detection model via REST API
        print(f"Loading Roboflow model: {roboflow_model_id}")
        self.api_key = roboflow_api_key
        self.model_id = roboflow_model_id
        self.api_url = f"https://serverless.roboflow.com/{roboflow_model_id}"

        # Load classification model
        self.use_tflite = use_tflite
        if use_tflite and tflite_model_path:
            print(f"Loading TFLite model: {tflite_model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.classifier = None
        else:
            print(f"Loading Keras model: {classification_model_path}")
            self.classifier = tf.keras.models.load_model(classification_model_path)
            self.interpreter = None

        self.class_names = ["overripe", "ripe", "spoiled", "unripe"]
        self.img_size = (224, 224)

        print("System initialized successfully!")

    def preprocess_for_classification(self, image_crop: np.ndarray) -> np.ndarray:
        """Prepare detected watermelon for ripeness classification"""
        # Resize to model input size
        resized = cv2.resize(image_crop, self.img_size)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.use_tflite:
            # For TFLite INT8 model
            if self.input_details[0]['dtype'] == np.uint8:
                return np.expand_dims(resized.astype(np.uint8), axis=0)
            else:
                # For TFLite FP32/FP16
                return np.expand_dims(rgb.astype(np.float32), axis=0)
        else:
            # For Keras model
            normalized = rgb.astype(np.float32)
            return np.expand_dims(normalized, axis=0)

    def classify_ripeness(self, processed_crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Run classification inference"""
        if self.use_tflite:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_crop)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Convert uint8 output to probabilities if needed
            if self.output_details[0]['dtype'] == np.uint8:
                output_data = output_data.astype(np.float32) / 255.0

            ripeness_probs = output_data
        else:
            # Keras inference
            ripeness_probs = self.classifier.predict(processed_crop, verbose=0)[0]

        ripeness_class = self.class_names[np.argmax(ripeness_probs)]
        ripeness_confidence = float(np.max(ripeness_probs))
        ripeness_probabilities = {
            name: float(prob)
            for name, prob in zip(self.class_names, ripeness_probs)
        }

        return ripeness_class, ripeness_confidence, ripeness_probabilities

    def predict_image(self, image_path: str, confidence_threshold: float = 0.4) -> Tuple[List[Dict], np.ndarray]:
        """
        Full pipeline: detect watermelon, then classify ripeness

        Returns:
            Tuple of (results, original_image)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict_frame(image, confidence_threshold)

    def predict_frame(self, frame: np.ndarray, confidence_threshold: float = 0.4) -> Tuple[List[Dict], np.ndarray]:
        """
        Process a single frame (useful for video/camera feed)

        Returns:
            Tuple of (results, original_frame)
        """
        # Encode frame as base64 for Roboflow API
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Step 1: Detect watermelons using Roboflow REST API
        try:
            response = requests.post(
                self.api_url,
                params={"api_key": self.api_key, "confidence": confidence_threshold},
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            predictions = response.json()
        except Exception as e:
            print(f"Detection error: {e}")
            return [], frame

        results = []

        # Step 2: Classify each detected watermelon
        for pred in predictions.get('predictions', []):
            # Extract bounding box
            x = int(pred['x'] - pred['width'] / 2)
            y = int(pred['y'] - pred['height'] / 2)
            w = int(pred['width'])
            h = int(pred['height'])

            # Ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            # Crop watermelon from image
            watermelon_crop = frame[y:y+h, x:x+w]

            # Classify ripeness
            processed_crop = self.preprocess_for_classification(watermelon_crop)
            ripeness_class, ripeness_confidence, ripeness_probabilities = \
                self.classify_ripeness(processed_crop)

            results.append({
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'detection_confidence': pred['confidence'],
                'detection_class': pred.get('class', 'watermelon'),
                'ripeness': ripeness_class,
                'ripeness_confidence': ripeness_confidence,
                'ripeness_probabilities': ripeness_probabilities
            })

        return results, frame

    def visualize_results(self, image_path: str, save_path: Optional[str] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Visualize detection and classification results"""
        results, image = self.predict_image(image_path)
        annotated_image = self.draw_results(image, results)

        # Save or display
        if save_path:
            cv2.imwrite(str(save_path), annotated_image)
            print(f"Saved result to {save_path}")

        return annotated_image, results

    def draw_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()

        for result in results:
            bbox = result['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            # Draw bounding box
            color = self.get_color_for_ripeness(result['ripeness'])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 8)

            # Create label
            label = f"{result['ripeness'].upper()}: {result['ripeness_confidence']:.1%}"

            # Draw label on rotated text (90 degrees clockwise along right side)
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4
            )

            # Create rotated label: render text, rotate 90° CW, overlay on image
            text_img = np.zeros((label_h + 30, label_w + 30, 3), dtype=np.uint8)
            # Fill with the ripeness color as background
            text_img[:] = color
            cv2.putText(text_img, label, (15, label_h + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            # Rotate 90° clockwise
            rotated_text = cv2.rotate(text_img, cv2.ROTATE_90_CLOCKWISE)

            # Place rotated label on right side of bounding box
            rt_h, rt_w = rotated_text.shape[:2]
            place_x = x + w + 5
            place_y = y

            # Ensure it fits within the image
            if place_x + rt_w > annotated.shape[1]:
                place_x = x - rt_w - 5  # Put on left side if no room on right
            if place_y + rt_h > annotated.shape[0]:
                rt_h = annotated.shape[0] - place_y
                rotated_text = rotated_text[:rt_h]
            if place_x >= 0 and place_y >= 0:
                annotated[place_y:place_y + rt_h, place_x:place_x + rt_w] = rotated_text

            # Draw probability bars (optional - for detailed view)
            # self.draw_probability_bars(annotated, result, x, y + h + 10)

        return annotated

    def draw_probability_bars(self, image: np.ndarray, result: Dict, x: int, y: int):
        """Draw probability bars for each class"""
        bar_height = 15
        bar_width = 150
        spacing = 5

        for i, class_name in enumerate(self.class_names):
            prob = result['ripeness_probabilities'][class_name]
            color = self.get_color_for_ripeness(class_name)

            # Background bar
            cv2.rectangle(image, (x, y + i * (bar_height + spacing)),
                         (x + bar_width, y + i * (bar_height + spacing) + bar_height),
                         (50, 50, 50), -1)

            # Probability bar
            filled_width = int(bar_width * prob)
            cv2.rectangle(image, (x, y + i * (bar_height + spacing)),
                         (x + filled_width, y + i * (bar_height + spacing) + bar_height),
                         color, -1)

            # Label
            label = f"{class_name}: {prob:.1%}"
            cv2.putText(image, label, (x + bar_width + 10, y + i * (bar_height + spacing) + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def get_color_for_ripeness(self, ripeness: str) -> Tuple[int, int, int]:
        """Color coding for different ripeness levels (BGR format)"""
        colors = {
            'unripe': (0, 255, 255),    # Yellow
            'ripe': (0, 255, 0),        # Green
            'overripe': (0, 165, 255),  # Orange
            'spoiled': (0, 0, 255)      # Red
        }
        return colors.get(ripeness, (255, 255, 255))


# Simple test function
if __name__ == "__main__":
    # Test configuration
    system = WatermelonRipenessSystem(
        roboflow_api_key="YOUR_ROBOFLOW_API_KEY",  # Replace with your key
        roboflow_model_id="watermelonobjectdetection/1",  # Replace with your model
        classification_model_path="watermelon_cnn.keras"
    )

    # Test on an image
    test_image = "test_watermelon.jpg"
    annotated, results = system.visualize_results(test_image, save_path="output.jpg")

    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nWatermelon {i}:")
        print(f"  Ripeness: {result['ripeness']}")
        print(f"  Confidence: {result['ripeness_confidence']:.2%}")
