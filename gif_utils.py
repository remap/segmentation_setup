from IPython.display import display, Image as IPImage
import cv2
import os
from pathlib import Path
from icecream import ic
import numpy as np
import torch

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


def extract_video_frames(video_path, save_to_png=False, output_dir=None):
    """
    Extract frames from a video and save them as PNGs to a specified directory.

    Args:
        video_path (str): The path to the video file.
        save_to_png (bool, optional): If True, saves each frame as a PNG to the specified output directory. Defaults to False.
        output_dir (str, optional): The directory to save the frames to. Defaults to a directory named after the video file without extension, e.g. "video_frames".

    Returns:
        list: A list of numpy arrays representing the frames of the video. If save_to_png is True, the list will be empty.
    """
    # Determine the output directory
    if save_to_png and output_dir is None:
        # Get the video filename without extension
        video_filename = Path(video_path).stem
        output_dir = os.path.join(os.path.dirname(video_path), f"{video_filename}_frames")
        # ic(output_dir)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    # Open video and extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as PNG
        if save_to_png:
            filename = f"frame_{frame_count:05d}.png"
            filepath = os.path.join(output_dir, filename)
            try:
                cv2.imwrite(filepath, frame)
                frame_count += 1
            except Exception as e:
                print(f"Error saving frame {frame_count}: {e}")
        
        frames.append(frame)

    cap.release()
    return frames


def save_binary_mask_as_png(mask, output_path):
    """
    Saves a binary mask as a PNG image.

    Args:
        mask (torch.Tensor or numpy.ndarray): Binary mask (boolean or uint8).
        output_path (str): Path to save the PNG file.
    """
    # Check if mask is a PyTorch tensor
    if isinstance(mask, torch.Tensor):
        # Convert tensor to CPU and then to numpy array
        mask = mask.to('cpu').numpy()

    # Check if mask is a PyTorch tensor
    if isinstance(mask, torch.Tensor):
        # Convert tensor to CPU and then to numpy array
        mask = mask.to('cpu').numpy()

    # Convert boolean mask to uint8 if necessary
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    
    # Scale values to 0-255 range
    mask_scaled = cv2.convertScaleAbs(mask)
    
    # Save as PNG
    if '.png' not in mask_name:
        mask_name = mask_name+'.png'
    output_file = os.path.join(output_path, mask_name)
    success = cv2.imwrite(output_file, mask_scaled)
    
    if success:
        print(f"Binary mask saved successfully as {output_file}")
    else:
        print(f"Failed to save binary mask as {output_file}")


def save_frame_as_png(frame, output_path, frame_name):
    """
    Saves a frame as a PNG image.

    Args:
        frame (numpy.ndarray): The frame to be saved.
        output_path (str): The path to save the PNG file.
        frame_name (str): The name of the frame (without extension).

    Returns:
        None
    """
    # Save as PNG
    if '.png' not in frame_name:
      frame_name = frame_name+'.png'
    output_file = os.path.join(output_path, frame_name)
    success = cv2.imwrite(output_file, frame)

    if success:
        print(f"Frame saved successfully as {output_file}")
    else:
        print(f"Failed to save frame as {output_file}")



