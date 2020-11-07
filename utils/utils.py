import numpy as np
import matplotlib.pyplot as plt
import cv2

def resize_image_and_bounding_box(image,box,new_width,new_height):
    height,width,_=image.shape
    image=cv2.resize(image,(new_width,new_height))
    x,y,w,h=box
    nx=int(x/width*new_width)
    ny=int(y/height*new_height)
    nw=int(w/width*new_width)
    nh=int(h/height*new_height)
    box=np.array([nx,ny,nw,nh])
    return image,box

def convert_coordinate_box_to_relative_box(box,image_height,image_width):
    return box/np.array([image_width,image_height,image_width,image_height])

def convert_relative_box_to_coordinate_box(box,image_height,image_width):
    return (box*np.array([image_width,image_height,image_width,image_height])).astype('int')