from IPython.display import display, Image as IPImage
import cv2

def save_frames_to_gif(gif_path, frames, frame_duration=40):
    frames[0].save(gif_path, 
                   save_all=True, 
                   append_images=frames[1:], 
                   uration=frame_duration, 
                   loop=0)  # duration in ms per frame

  
def show_gif(gif_path):
    display(IPImage(filename=gif_path))


def extract_video_frames(video_path):
    """
    Extracts frames from a given video file.

    Args:
        video (str): Path to the video file.

    Returns:
        list: A list containing the frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames