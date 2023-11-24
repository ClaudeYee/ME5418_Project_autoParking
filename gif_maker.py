import os
import glob
from PIL import Image

def make_gif(frame_folder, gif_save_path, index):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save('{}/episode{}.gif'.format(gif_save_path, index), format="GIF", append_images=frames, save_all=True, duration=140, loop=0)


if __name__ == "__main__":
    img_path = "process_visual_episodes_23_Nov_change_map"
    save_path = "gifs_folder"
    for episode_index in range(80):
        picture_folder = "{}/episode{}".format(img_path, episode_index)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        make_gif(picture_folder, save_path, episode_index)

