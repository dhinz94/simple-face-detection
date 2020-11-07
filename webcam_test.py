import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from utils import utils
import numpy as np

# colab path
dataset_path='/content/drive/My Drive/celeba_copy/'

# local path for development
dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

model=tf.keras.models.load_model(dataset_path+'model.h5',custom_objects={'relu':tf.nn.relu},compile=False)

cam = cv2.VideoCapture(0)


while True:
    ret_val, frame = cam.read()

    input=cv2.resize(frame,(256,256))
    input=cv2.cvtColor(input,cv2.COLOR_BGR2RGB).reshape(-1,256,256,3)/255

    pred_box=np.array(model(input)[0])
    coordinate_box=utils.convert_relative_box_to_coordinate_box(pred_box,frame.shape[0],frame.shape[1])

    cv2.rectangle(frame, (coordinate_box[0], coordinate_box[1]), (coordinate_box[0]+coordinate_box[2], coordinate_box[1]+coordinate_box[3]), (0, 255, 0), 3)

    cv2.imshow('webcam stream', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()


