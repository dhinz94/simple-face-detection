import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import utils
from matplotlib.patches import Rectangle

# colab path
dataset_path='/content/drive/My Drive/celeba_copy/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

image_path=dataset_path+'img_celeba/'
box_path=dataset_path+'list_bbox_celeba.txt'

target_resolution=256

print('image_path exists: ',os.path.exists(image_path))
print('box_path exists: ',os.path.exists(box_path))


boxes=pd.read_csv(box_path,delim_whitespace=True,skiprows=1)
file_list=os.listdir(image_path)
file_list=[x for x in file_list if '.jpg' in x]

boxes=boxes[boxes['image_id'].isin(file_list)]
print(boxes)

image_array=[]
box_array=[]

for id,row in boxes.iterrows():
    file_path=image_path+row['image_id']
    x=row['x_1']
    y=row['y_1']
    w = row['width']
    h = row['height']
    box=np.array([x,y,w,h])
    image=plt.imread(file_path)


    # plt.figure()
    # plt.imshow(image)
    # plt.gca().add_patch(Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    #resize image and change bounding box coordinates
    image,box=utils.resize_image_and_bounding_box(image=image,box=box,new_height=target_resolution,new_width=target_resolution)


    # plt.figure()
    # plt.imshow(image)
    # plt.gca().add_patch(Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none'))

    # convert coordinates of bounding boxes to values between 0 and 1
    relative_box=utils.convert_coordinate_box_to_relative_box(box=box,image_height=target_resolution,image_width=target_resolution)

    image_array.append(image)
    box_array.append(relative_box)

    plt.show()

    if id>5:
        break

#convert to numpy array with shape (N,H,W,C)
image_array=np.array(image_array)

#convert to numpy array with shape(N,4)
box_array=np.array(box_array)

np.save(dataset_path+'image_array.npy',image_array)
np.save(dataset_path+'box_array.npy',box_array)








