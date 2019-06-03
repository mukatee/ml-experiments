
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

input_path = 'rescale_src/train'
output_path = "rescale_dst/train"

count = 0
for img_name in tqdm(os.listdir(input_path)):
    img_resize(img_name, input_path, output_path, 299)
    count += 1
#    if count >= 100:
#        break

#df_train = pd.read_csv("../input/train.csv")
#
# df_train["attribute_ids"] = df_train["attribute_ids"].apply(lambda x: x.split(" "))
# df_train["attribute_ids"] = df_train["attribute_ids"].apply(lambda x: [int(val) for val in x])
# label_df = pd.read_csv("../input/labels.csv")
#
# i = 1
# plt.figure(figsize=[30, 30])
# for img_name in os.listdir("rescaled_train")[0:9]:
#     print(img_name)
#     img = PIL.Image.open(f'rescaled_train/{img_name}')
#     plt.subplot(3, 3, i)
#     plt.imshow(img)
#     img_base = img_name.split(".")[0]
#     ids = df_train[df_train["id"] == img_base]["attribute_ids"]
#     print(f"base:{img_base},ids={ids}")
#     title_val = []
#     for tag_id in ids.values[0]:
#         att_name = label_df[label_df['attribute_id'] == tag_id]['attribute_name'].values[0]
#         title_val.append(att_name)
#     plt.title(title_val)
#     i += 1
#
# plt.show()
#
