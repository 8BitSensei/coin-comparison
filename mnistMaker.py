import numpy as np
import os
import pandas as pd
import cv2
from PIL import Image
from PIL import ImageOps
from resizeimage import resizeimage
from io import BytesIO
import csv


desired_size = 80
rootdir = "CNN_data/training_set/"
label = 0
total = []

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

imgCounter = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        imgCounter = imgCounter+1
        print(subdir, file)
        im = Image.open(os.path.join(subdir, file))
        old_size = im.size 
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = ImageOps.invert(im)
        #im = change_contrast(im,25)
        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
        new_im.save(os.path.join(subdir,file))
        img = Image.open(os.path.join(subdir,file))
        img = list(img.getdata())
        arr = np.array(img)
        label_str = str(label)
        label_str = [label_str]
        new_arr = []
        new_arr.append(label_str)
        for i in arr:
            new_arr.append([i[0]])

        with open("coinData.csv", "a") as fp:
            #wr = csv.writer(fp, dialect='excel')
            x = len(new_arr)
            i = 1
            for item in new_arr:
                if i != x:
                    fp.write(str(item[0])+',')
                else:
                    val = str(item[0])
                    fp.write(val)
                i = i+1
            fp.write('\n')



    label = label+1
print("Image count:", imgCounter)


#f = open('foo.csv','w')
#f.write(label_str)
#for i in total:
	#value = str(i[0])+','
	#f.write(value)
#f.close()