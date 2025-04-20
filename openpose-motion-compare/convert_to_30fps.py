import cv2
import os

input_path = r"C:\Users\oik22\OneDrive\Documents\GitHub\openpose\openpose-motion-compare\reference_skeleton.avi"
output_path = r"C:\Users\oik22\OneDrive\Documents\GitHub\openpose\openpose-motion-compare\reference_skeleton_fixed.avi"
fps = 30  # 원하는 프레임 속도

# 파일 존재 여부 확인
if not os.path.exists(input_path):
    print(f"❌ Input file not found: {input_path}")
    exit()

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"❌ Failed to open input video: {input_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"🔄 Re-encoding {input_path} to 30fps...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print(f"✅ Saved {frame_count} frames to {output_path} at {fps} fps.")
