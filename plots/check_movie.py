import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


image_dir = "pictures_temperature_ratio"
filename = "mr_temperature_ratio.gif"

image_files = [f"{i}.png" for i in range(0, 50000+1, 1000)]


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()

first_image = plt.imread(os.path.join(image_dir, image_files[0]))

im = ax.imshow(first_image)

def update_img(frame):
    img = plt.imread(os.path.join(image_dir, image_files[frame]))
    im.set_data(img)
    return im,


ani = animation.FuncAnimation(fig, update_img, frames=len(image_files), interval=500)

ani.save(filename)

print(f"movie created! : {filename}\n")
