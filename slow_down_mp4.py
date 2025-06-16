import cv2
import os

def slow_down_video(input_path, output_path, slowdown_percentage=75):
    """
    Slow down an MP4 video using OpenCV by lowering its frame rate.

    Args:
        input_path (str): Path to the input .mp4 file.
        output_path (str): Path to save the slowed-down .mp4 file.
        slowdown_percentage (float): Percentage of original speed.
                                     100 = normal speed, 50 = half speed, etc.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_fps = fps * (slowdown_percentage / 100)
    print(f"Processing '{os.path.basename(input_path)}' | Original FPS: {fps} | New FPS: {new_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved slowed-down video to: {output_path}")

def slow_down_all_videos(input_dir, output_dir, slowdown_percentage=50):
    """
    Process all .mp4 files in input_dir and save them slowed down to output_dir.

    Args:
        input_dir (str): Path to the directory containing .mp4 files.
        output_dir (str): Path to the directory to save slowed-down videos.
        slowdown_percentage (float): Percentage of original speed.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp4'):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"slow_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            slow_down_video(input_path, output_path, slowdown_percentage)

if __name__ == "__main__":
    input_directory = "./input_videos"   # Replace with your actual path
    output_directory = "./output_videos" # Replace with your actual path
    slowdown_percentage = 50             # 50% speed

    slow_down_all_videos(input_directory, output_directory, slowdown_percentage)