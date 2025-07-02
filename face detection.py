# Enhanced Face Detection System with Multi-Feature Support
import cv2
import time
import os
import numpy as np
from datetime import datetime


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

        self.video_capture = cv2.VideoCapture(0)
        self.detection_active = True
        self.faces_detected = 0
        self.session_start = time.time()
        self.screenshot_dir = "detection_screenshots"
        self.create_screenshot_dir()

        # Detection parameters
        self.face_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 5,
            'minSize': (30, 30)
        }

        self.eye_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 10,
            'minSize': (20, 20)
        }

        self.smile_params = {
            'scaleFactor': 1.8,
            'minNeighbors': 20,
            'minSize': (25, 25)
        }

    def create_screenshot_dir(self):

        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def take_screenshot(self, frame):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/face_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

    def draw_detections(self, frame, detections, color, label=None):

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, **self.face_params)
        if len(faces) > 0:
            self.faces_detected += len(faces)

        # Eye detection within each face region
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, **self.eye_params)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (255, 0, 0), 1)

            # Smile detection
            smile = self.smile_cascade.detectMultiScale(face_roi, **self.smile_params)
            self.draw_detections(frame, [(x + sx, y + sy, sw, sh) for (sx, sy, sw, sh) in smile],
                                 (0, 0, 255), "Smile")

        self.draw_detections(frame, faces, (0, 255, 0), "Face")

        # Display statistics
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Detected: {self.faces_detected}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 's' to screenshot, 'q' to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run_detection(self):

        print("Starting Enhanced Face Detection System...")
        print("""
        Controls:
        - 's' : Take screenshot
        - 'q' : Quit application
        """)

        try:
            while self.detection_active:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Error capturing frame")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Enhanced Face Detection', processed_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.detection_active = False
                elif key == ord('s'):
                    self.take_screenshot(processed_frame)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and display session summary"""
        session_duration = time.time() - self.session_start
        print("\nSession Summary:")
        print(f"Total faces detected: {self.faces_detected}")
        print(f"Session duration: {session_duration:.2f} seconds")

        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = FaceDetector()
    detector.run_detection()
