
import numpy as np
import pandas as pd
import cv2
import gc
import PIL
from PIL import ImageOps
import matplotlib.pyplot as plt

import sys
import os

print(sys.version)

from tqdm.auto import tqdm
tqdm.pandas()


def img_resize(img_id, input_path, output_path, target_size):
    full_output_path = os.path.join(output_path, str(target_size))
    if not os.path.exists(full_output_path):
        os.makedirs(full_output_path)
    img = PIL.Image.open(f'{input_path}/{img_id}')
    w = img.size[0]
    h = img.size[1]
    pad_size = np.abs(h-w)
    wm = hm = 1
    pw = ph = 0
    if w < h:
        wm = h / w
        pw = pad_size / 2
    else:
        hm = w / h
        ph = pad_size / 2
    w *= wm
    h *= hm
    h = int(h)
    w = int(w)
    pw = int(pw)
    ph = int(ph)
    #print(f"w={w},h={h},wm={wm},hm={hm}")
    #resized = img.resize((w,h))
    padding = (pw, ph, pw, ph)
    padded = ImageOps.expand(img, padding)
    resized = padded.resize((target_size, target_size))
    resized.save(f'{output_path}/{target_size}/{img_id}')
    img.close()
    padded.close()
    del img
    del padded


#def img_resize_nopad(filename, input_path, output_path):
def img_resize_nopad(my_args):
    filename = my_args[0]
    input_path = my_args[1]
    output_path = my_args[2]
    img = PIL.Image.open(f'{input_path}/{filename}')
    w = img.size[0]
    h = img.size[1]
    max_w = 1024
    max_h = 1024
    multiplier = 1
    if w > max_w or h > max_h:
        multiplier = max_w / w
        if h > w:
            multiplier = max_h / h
        # print("h multiplier:"+str(multiplier))
        else:
            # print("w multiplier:"+str(multiplier))
            pass
        w *= multiplier
        h *= multiplier
        w = int(w)
        h = int(h)

    print("multiplier:" + str(multiplier))
    resized = img.resize((w, h))
    filename = filename.split(".")[0]+".png"
    resized.save(f'{output_path}/{filename}')
    img.close()
    del img


from multiprocessing import Pool

pool = Pool(processes=2)

input_path = 'old_train/'
output_path = "rescaled_train/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_names = os.listdir(input_path)
file_infos = [(filename, input_path, output_path) for filename in file_names]

#starmap simply unpacks the tuple into aguments (tuple is file_infos here)
#for _ in tqdm(pool.starmap(img_resize_nopad, file_infos), total=len(file_infos)):

for _ in tqdm(pool.imap_unordered(img_resize_nopad, file_infos), total=len(file_infos)):
    pass


