import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


dataset_path='/content/drive/My Drive/celeba_copy/'
image_path=dataset_path+'img_celeba'
box_path=dataset_path+'list_bbox_celeba.txt'

print('image_path exists: ',os.path.exists(image_path))
print('box_path exists: ',os.path.exists(box_path))

boxes=pd.read_csv(box_path,sep=' ')
print(boxes)



