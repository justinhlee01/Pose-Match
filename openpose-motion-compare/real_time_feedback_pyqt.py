import sys
import os
import cv2
import json
import numpy as np
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from motion_compare_embedding import compute_hybrid_similarity

# -------------------------------
# Utility: Extract 2D keypoints from JSON file
# -------------------------------
def extract_keypoints(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not data['people'] or 'pose_keypoints_2d' not in data['people'][0]:
            return None
        keypoints = data['people'][0]['pose_keypoints_2d']
        if not keypoints or all(v == 0 for v in keypoints):
            return None
        # Only keep x and y (remove confidence scores)
        xy = [keypoints[i] for i in range(len(keypoints)) if i % 3 != 2]
        return xy
    except Exception as e:
        print(f"[DEBUG] Failed to parse {json_path}: {e}")
        return None

# -------------------------------
# Utility: Load sequence of keypoints from a folder of JSON files
# -------------------------------
def load_sequence(folder):
    if not os.path.exists(folder):
        return []
    files = sorted(f for f in os.listdir(folder) if f.endswith('.json'))
    sequence = []
    for f in files:
        vec = extract_keypoints(os.path.join(folder, f))
        if vec:
            sequence.append(vec)
    return sequence

# -------------------------------
# Main GUI Application Class
# -------------------------------
class FeedbackApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Motion Feedback (Hybrid Similarity)")
        self.setGeometry(100, 100, 1280, 720)

        # GUI Components
        self.reference_label = QLabel("Reference Video")
        self.feedback_label = QLabel("Feedback: ")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("font-size: 20px;")
        self.start_button = QPushButton("Start Feedback")
        self.start_button.clicked.connect(self.start_feedback)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.reference_label)
        layout.addWidget(self.feedback_label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feedback)
        self.ref_play_timer = QTimer()
        self.ref_play_timer.timeout.connect(self.play_reference_frame)

        # File paths (absolute)
        self.base_path = os.path.abspath(os.getcwd())
        self.ref_video_path = os.path.join(self.base_path, "reference_skeleton_fixed.avi")
        self.ref_json_path = os.path.join(self.base_path, "reference_json")
        self.live_json_path = os.path.join(self.base_path, "live_json")

        # Load video and reference sequence
        self.ref_cap = cv2.VideoCapture(self.ref_video_path)
        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.ref_cap.set(cv2.CAP_PROP_FPS, 30)
        self.ref_seq = load_sequence(self.ref_json_path)
        print(f"[DEBUG] Loaded reference frames: {len(self.ref_seq)}")

        self.openpose_process = None

    # ---------------------------
    # Starts OpenPose and Timers
    # ---------------------------
    def start_feedback(self):
        os.makedirs(self.live_json_path, exist_ok=True)

        # Launch OpenPose with absolute paths
        self.openpose_process = subprocess.Popen([
            "C:/Users/oik22/OneDrive/Documents/GitHub/openpose/bin/OpenPoseDemo.exe",
            "--camera", "0",
            "--camera_resolution", "640x360",
            "--net_resolution", "-1x160",
            "--write_json", self.live_json_path,
            "--display", "2",
            "--model_folder", "C:/Users/oik22/OneDrive/Documents/GitHub/openpose/models"
        ])

        # Start timers
        self.ref_play_timer.start(2000 // 30)  # Reference video playback
        self.timer.start(2000)                # Feedback update

    # ---------------------------
    # Display reference video frame
    # ---------------------------
    def play_reference_frame(self):
        ret, frame = self.ref_cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                              rgb_image.strides[0], QImage.Format_RGB888)
            self.reference_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 360, Qt.KeepAspectRatio))
        else:
            # Restart playback if end reached
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------------------------
    # Compare keypoints and update feedback label
    # ---------------------------
    def update_feedback(self):
        print("[DEBUG] update_feedback called")
        live_seq = load_sequence(self.live_json_path)
        print(f"[DEBUG] ref_seq length: {len(self.ref_seq)}")
        print(f"[DEBUG] live_seq length: {len(live_seq)}")

        if len(self.ref_seq) < 5 or len(live_seq) < 5:
            return

        similarity = compute_hybrid_similarity(self.ref_seq[:len(live_seq)], live_seq)

        # Display feedback based on similarity
        if similarity > 0.95:
            feedback = "✅ Perfect!"
        elif similarity > 0.85:
            feedback = "👍 Good!"
        elif similarity > 0.70:
            feedback = "⚠️ Keep Practicing"
        else:
            feedback = "❗ Wrong Pose"
        self.feedback_label.setText(f"{feedback} (Score: {similarity:.3f})")

    # ---------------------------
    # On App Close: Cleanup
    # ---------------------------
    def closeEvent(self, event):
        self.ref_cap.release()
        if self.openpose_process:
            self.openpose_process.terminate()
        cv2.destroyAllWindows()
        event.accept()

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeedbackApp()
    window.show()
    sys.exit(app.exec_())
