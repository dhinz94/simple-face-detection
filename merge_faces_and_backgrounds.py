import numpy as np
import matplotlib.pyplot as plt

# colab path
dataset_path='/content/drive/My Drive/simple_face_detection_data/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

faces=np.load(dataset_path+'faces.npy')[:10000]
backgrounds=np.load(dataset_path+'backgrounds.npy')[:10000]
boxes=np.load(dataset_path+'box_array.npy')[:10000]

face_labels=np.ones((len(faces),1))
background_labels=np.zeros((len(backgrounds),1))
empty_boxes=np.zeros((len(backgrounds),4))

print('faces:',len(faces))
print('backgrounds:',len(backgrounds))

merged=np.concatenate([faces,backgrounds],axis=0)
labels=np.concatenate([face_labels,background_labels],axis=0)
boxes=np.concatenate([boxes,empty_boxes],axis=0)


p=np.random.permutation(len(merged))
merged=merged[p]
labels=labels[p]

for i in range(10):
    plt.imshow(merged[i])
    plt.title(labels[i])
    plt.show()

np.save(dataset_path+'train_images.npy',merged)
np.save(dataset_path+'train_labels.npy',labels)
np.save(dataset_path+'train_boxes.npy',boxes)