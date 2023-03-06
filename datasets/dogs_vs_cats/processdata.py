import pandas as pd
import numpy as np
import os
import sys

PATH = './train/'

imgs_name = []
isdog = []
counter = 0

for imgs in os.listdir(PATH):
    if imgs.split('.')[-1] == 'csv':
        continue
    imgs_name.append(imgs)

    if 'dog' in imgs:
        isdog.append(1)
    elif 'cat' in imgs:
        isdog.append(0)
    counter += 1

isdog = np.array(isdog)

df = pd.DataFrame({'img': pd.Series(imgs_name),
                   'isdog': isdog})

df.to_csv('./train/data_info.csv', index=False)
