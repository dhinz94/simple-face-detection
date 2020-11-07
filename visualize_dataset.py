import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd


dataset_path='/content/drive/My Drive/celeba_copy/'
image_path=dataset_path+'img_celeba/'
box_path=dataset_path+'list_bbox_celeba.txt'

print('image_path exists: ',os.path.exists(image_path))
print('box_path exists: ',os.path.exists(box_path))

boxes=pd.read_csv(box_path,delim_whitespace=True,skiprows=1)
print(boxes)

file_list=os.listdir(image_path)
file_list=[x for x in file_list if '.jpg' in x]

boxes=boxes[boxes['image_id'].isin(file_list)]

print(boxes)


for i in range(10):
    num=np.random.randint(0,len(boxes))
    file_path=image_path+file_list[num]

    image=plt.imread(file_path)
    plt.imshow(image)
    plt.gca().add_patch(Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()




