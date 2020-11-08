import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import utils
from matplotlib.patches import Rectangle

# colab path
dataset_path='/content/drive/My Drive/simple_face_detection_data/'

# local path for development
dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

face_image_path= dataset_path + 'img_celeba/'
box_path=dataset_path+'list_bbox_celeba.txt'

target_resolution=256


# read bounding box annotation file and filter dataframe for existing files in path
boxes=pd.read_csv(box_path,delim_whitespace=True,skiprows=1)
face_file_list=os.listdir(face_image_path)
face_file_list=[x for x in face_file_list if '.jpg' in x]
boxes=boxes[boxes['image_id'].isin(face_file_list)]
print(boxes)



image_array=[]
box_array=[]
for id,row in boxes.iterrows():

    if id%500==0:
        print('processed images:',id)

    file_path= face_image_path + row['image_id']
    x=row['x_1']
    y=row['y_1']
    w = row['width']
    h = row['height']
    box=np.array([x,y,w,h])
    try:
        image=plt.imread(file_path)
    except Exception as e:
        print('cannot read image: skips image')
        continue


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

    # plt.show()

    if len(image_array)>20000:
        break


#convert to numpy array with shape (N,H,W,C)
image_array=np.array(image_array)
print(image_array.dtype)

#convert to numpy array with shape(N,4)
box_array=np.array(box_array)


np.save(dataset_path+'image_array.npy',image_array)
np.save(dataset_path+'box_array.npy',box_array)








