from PIL import Image
import os
import PIL.ImageOps


rootdir = 'coins/0_Anonymous 5'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:  
    	image = Image.open(os.path.join(subdir,file))
    	inverted_image = PIL.ImageOps.invert(image)
    	inverted_image.save(os.path.join(subdir,file))