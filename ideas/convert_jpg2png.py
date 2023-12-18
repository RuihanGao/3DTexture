import os, sys
from PIL import Image

image_path = "/data/ruihan/projects/NeRF-Texture/data/my_purple_apple_optitrack/camera_selected_frames_renamed"

# convert imagef from .jpg format to .png format
for image in os.listdir(image_path):
    if image.endswith(".jpg"):
        im = Image.open(os.path.join(image_path, image))
        rgb_im = im.convert('RGB')
        rgb_im.save(os.path.join(image_path, image.split(".")[0] + ".png"))
        os.remove(os.path.join(image_path, image))