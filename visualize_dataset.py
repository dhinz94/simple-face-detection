import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import pandas as pd

# colab path
dataset_path='/content/drive/My Drive/simple_face_detection_data/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

image_path=dataset_path+'img_celeba/'
box_path=dataset_path+'list_bbox_celeba.txt'

print('image_path exists: ',os.path.exists(image_path))
print('box_path exists: ',os.path.exists(box_path))

boxes=pd.read_csv(box_path,delim_whitespace=True,skiprows=1)

#filter box list for ids that are in image folder
file_list=os.listdir(image_path)
file_list=[x for x in file_list if '.jpg' in x]
boxes=boxes[boxes['image_id'].isin(file_list)]
print(boxes)


#show random image with corresponding bounding box
for i in range(10):
    num=np.random.randint(0,len(boxes))
    file_path=image_path+boxes.at[num,'image_id']
    x=boxes.at[num,'x_1']
    y=boxes.at[num,'y_1']
    w = boxes.at[num, 'width']
    h = boxes.at[num, 'height']

    image=plt.imread(file_path)
    plt.imshow(image)
    plt.gca().add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()




