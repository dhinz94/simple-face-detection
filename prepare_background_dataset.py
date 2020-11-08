import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import utils
from matplotlib.patches import Rectangle
import cv2

# colab path
dataset_path='/content/drive/My Drive/simple_face_detection_data/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

background_image_path=dataset_path+'backgrounds/Images/'

file_paths=[]
for root,subdir,files in os.walk(background_image_path):
    for file in files:
        file_paths.append(os.path.join(root,file))

file_paths=np.array(file_paths)
file_paths=file_paths[np.random.permutation(len(file_paths))]

images=[]
i=0
for file in file_paths:
    i+=1

    if i>10000:
        break

    if i%100==0:
        print('processed images:',i)

    image=plt.imread(file)
    image=cv2.resize(image,(256,256))
    image=image.reshape(1,256,256,-1)
    if image.shape[3]==1:
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    if not image.shape == (1,256,256,3):
        continue

    images.append(image)

images=np.array(images)


# for i in range(10):
#     plt.imshow(images[i])
#     plt.show()

np.save(dataset_path+'backgrounds.npy',images)


