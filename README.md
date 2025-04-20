# PoseMatch 🎯

Match your moves. Master your motion.

PoseMatch is a real-time motion comparison tool that helps you improve your physical movements by comparing a user’s live webcam feed with a reference video. It uses pose detection, cosine similarity, and dynamic time warping to provide instant feedback.

---

## 🧠 Features

- Real-time pose detection using OpenPose
- Cosine Similarity + Euclidean distance + Joint Angle similarity to analyze motion accuracy
- Easy-to-read feedback for performance (Perfect, Good, More Practice)
- Clean front-end display of live and reference videos side-by-side
- Extensible for fitness, dance, or rehab applications

---

## 🏗️ Built With

- **Python** – Core programming language
- **OpenPose** – Pose detection framework
- **NumPy / SciPy** – Data handling and math operations
- **OpenCV** – Video and webcam handling

---



## 🧭 How It Works

<!-- Brief explanation of the pipeline -->
1. User uploads a reference video
2. System extracts pose keypoints using OpenPose
3. Live webcam feed captures user motion in real-time
4. Cosine Similarity + Euclidean distance + Joint Angle similarity are used to compare poses
5. Feedback is generated based on similarity thresholds

