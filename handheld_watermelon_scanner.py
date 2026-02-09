# handheld_watermelon_scanner.py
import cv2
import numpy as np
import time
from watermelon_detect_and_classify import WatermelonRipenessSystem
from pathlib import Path
import json
from datetime import datetime

class HandheldWatermelonScanner:
    def __init__(self,
                 system: WatermelonRipenessSystem,
                 camera_id: int = 0,
                 process_every_n_frames: int = 15,
                 save_scans: bool = True):
        """
        Real-time handheld watermelon scanner

        Args:
            system: Initialized WatermelonRipenessSystem
            camera_id: Camera device ID (0 for default camera)
            process_every_n_frames: Process every N frames (higher = faster but less responsive)
            save_scans: Save scan results to disk
        """
        self.system = system
        self.camera_id = camera_id
        self.process_every_n_frames = process_every_n_frames
        self.save_scans = save_scans

        # Create output directories
        if save_scans:
            self.output_dir = Path("scans")
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "images").mkdir(exist_ok=True)
            (self.output_dir / "results").mkdir(exist_ok=True)

        # Performance tracking
        self.fps_history = []
        self.current_results = []
        self.last_processed_frame = None
        self.scan_count = 0

    def run(self,
            window_width: int = 1280,
            window_height: int = 720,
            confidence_threshold: float = 0.4):
        """
        Run the handheld scanner

        Controls:
            's' - Save current scan
            'c' - Clear current results
            'q' - Quit
            SPACE - Pause/Resume
        """
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Set camera resolution (adjust based on your camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        frame_count = 0
        paused = False

        print("\n" + "="*60)
        print("HANDHELD WATERMELON SCANNER")
        print("="*60)
        print("Controls:")
        print("  's' - Save current scan")
        print("  'c' - Clear results")
        print("  SPACE - Pause/Resume")
        print("  'q' - Quit")
        print("="*60 + "\n")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break

                start_time = time.time()

                # Process frame at intervals
                if frame_count % self.process_every_n_frames == 0:
                    self.current_results, _ = self.system.predict_frame(
                        frame, confidence_threshold
                    )
                    self.last_processed_frame = frame.copy()

                # Draw results on current frame
                display_frame = self.system.draw_results(frame, self.current_results)

                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)

                # Draw UI overlay
                display_frame = self.draw_ui_overlay(display_frame, avg_fps)

                frame_count += 1
            else:
                # Use last frame when paused
                display_frame = self.last_processed_frame.copy() if self.last_processed_frame is not None else frame
                display_frame = self.draw_ui_overlay(display_frame, 0, paused=True)

            # Resize for display
            display_frame = cv2.resize(display_frame, (window_width, window_height))

            # Show frame
            cv2.imshow("Watermelon Scanner", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_current_scan()
            elif key == ord('c'):
                self.current_results = []
                print("Results cleared")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

        cap.release()
        cv2.destroyAllWindows()
        print("\nScanner stopped")

    def draw_ui_overlay(self, frame: np.ndarray, fps: float, paused: bool = False) -> np.ndarray:
        """Draw UI elements on frame"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent header
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(frame, "WATERMELON RIPENESS SCANNER", (20, 35),
                   cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)

        # FPS and status
        status_text = "PAUSED" if paused else f"FPS: {fps:.1f}"
        cv2.putText(frame, status_text, (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not paused else (0, 255, 255), 2)

        # Detection count
        count_text = f"Detected: {len(self.current_results)}"
        cv2.putText(frame, count_text, (w - 250, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Results summary panel (if detections exist)
        if self.current_results:
            panel_y = 100
            for i, result in enumerate(self.current_results):
                # Draw mini result panel
                panel_text = f"#{i+1}: {result['ripeness'].upper()} ({result['ripeness_confidence']:.0%})"
                color = self.system.get_color_for_ripeness(result['ripeness'])

                cv2.rectangle(frame, (w - 350, panel_y), (w - 10, panel_y + 40), (0, 0, 0), -1)
                cv2.rectangle(frame, (w - 350, panel_y), (w - 10, panel_y + 40), color, 2)
                cv2.putText(frame, panel_text, (w - 340, panel_y + 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                panel_y += 50

        # Controls hint
        hints = "s:Save | c:Clear | SPACE:Pause | q:Quit"
        cv2.putText(frame, hints, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def save_current_scan(self):
        """Save current scan results"""
        if not self.current_results:
            print("No results to save")
            return

        if not self.save_scans:
            print("Saving is disabled")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scan_count += 1

        # Save image
        img_filename = f"scan_{timestamp}_{self.scan_count}.jpg"
        img_path = self.output_dir / "images" / img_filename

        annotated_frame = self.system.draw_results(
            self.last_processed_frame, self.current_results
        )
        cv2.imwrite(str(img_path), annotated_frame)

        # Save results as JSON
        json_filename = f"scan_{timestamp}_{self.scan_count}.json"
        json_path = self.output_dir / "results" / json_filename

        scan_data = {
            'timestamp': timestamp,
            'scan_id': self.scan_count,
            'detections': self.current_results,
            'summary': self.get_scan_summary()
        }

        with open(json_path, 'w') as f:
            json.dump(scan_data, f, indent=2)

        print(f"\n✓ Scan saved: {img_filename}")
        print(f"  Image: {img_path}")
        print(f"  Results: {json_path}")
        print(f"  Watermelons detected: {len(self.current_results)}\n")

    def get_scan_summary(self) -> Dict:
        """Generate summary statistics for current scan"""
        if not self.current_results:
            return {}

        ripeness_counts = {'unripe': 0, 'ripe': 0, 'overripe': 0, 'spoiled': 0}
        total_confidence = 0

        for result in self.current_results:
            ripeness_counts[result['ripeness']] += 1
            total_confidence += result['ripeness_confidence']

        return {
            'total_watermelons': len(self.current_results),
            'ripeness_distribution': ripeness_counts,
            'average_confidence': total_confidence / len(self.current_results) if self.current_results else 0
        }


# Main execution script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Handheld Watermelon Ripeness Scanner")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--model-id", required=True, help="Roboflow model ID (workspace/project/version)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--tflite", action="store_true", help="Use TFLite model for faster inference")
    parser.add_argument("--tflite-path", default="watermelon_cnn_int8.tflite", help="Path to TFLite model")
    parser.add_argument("--keras-path", default="watermelon_cnn.keras", help="Path to Keras model")
    parser.add_argument("--no-save", action="store_true", help="Don't save scans")
    parser.add_argument("--fps-target", type=int, default=15, help="Process every N frames (higher = faster)")

    args = parser.parse_args()

    # Initialize system
    print("Initializing scanner...")
    system = WatermelonRipenessSystem(
        roboflow_api_key=args.api_key,
        roboflow_model_id=args.model_id,
        classification_model_path=args.keras_path,
        use_tflite=args.tflite,
        tflite_model_path=args.tflite_path if args.tflite else None
    )

    # Create scanner
    scanner = HandheldWatermelonScanner(
        system=system,
        camera_id=args.camera,
        process_every_n_frames=args.fps_target,
        save_scans=not args.no_save
    )

    # Run scanner
    scanner.run()
