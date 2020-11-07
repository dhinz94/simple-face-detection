import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


dataset_path='/content/drive/My Drive/celeba_copy/'
image_path=dataset_path+'img_celeba'
box_path=dataset_path+'list_bbox_celeba.txt'

print('image_path exists: ',os.path.exists(image_path))
print('box_path exists: ',os.path.exists(box_path))

boxes=pd.read_csv(box_path,delim_whitespace=True,skiprows=1)
print(boxes)

file_list=os.listdir(image_path)
file_list=[x for x in file_list if '.jpg' in file_list]

print(file_list)



