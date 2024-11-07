from IPython.display import display, Image as IPImage

def save_frames_to_gif(gif_path, frames, frame_duration=40):
    frames[0].save(gif_path, 
                   save_all=True, 
                   append_images=frames[1:], 
                   uration=frame_duration, 
                   loop=0)  # duration in ms per frame

  
def show_gif(gif_path):
    display(IPImage(filename=gif_path))
