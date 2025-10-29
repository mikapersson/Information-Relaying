import cv2
import os

scenario = 'one_way_5_agent_coord_transform'
video_path = f'{scenario}.mp4'
output_dir = f'{scenario}_frames'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_filename = os.path.join(output_dir, f'frame_{frame_number:04d}.png')
    cv2.imwrite(frame_filename, frame)
    frame_number += 1

cap.release()