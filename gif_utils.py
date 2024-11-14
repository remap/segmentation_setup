from IPython.display import display, Image as IPImage
import cv2

def save_frames_to_gif(gif_path, frames, frame_duration=40):
    """
    Saves a list of frames as a gif to the given path.

    Args:
        gif_path (str): The path to save the gif to.
        frames (list): A list of PIL Image objects.
        frame_duration (int, optional): The duration of each frame in ms. Defaults to 40.
    """
    frames[0].save(gif_path, 
                   save_all=True, 
                   append_images=frames[1:], 
                   uration=frame_duration, 
                   loop=0)  # duration in ms per frame

  
def show_gif(gif_path):
    """
    Displays a GIF image from a specified file path.

    Args:
        gif_path (str): The path to the GIF file to be displayed.
    """
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