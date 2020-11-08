import numpy as np
import matplotlib.pyplot as plt

# colab path
dataset_path='/content/drive/My Drive/simple_face_detection_data/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

faces=np.load(dataset_path+'image_array.npy')[:10000]
backgrounds=np.load(dataset_path+'backgrounds.npy')[:10000]
face_labels=np.ones((len(faces),1))
background_labels=np.zeros((len(backgrounds),1))

print('faces:',len(faces))
print('backgrounds:',len(backgrounds))

merged=np.concatenate([faces,backgrounds],axis=0)
labels=np.concatenate([face_labels,background_labels],axis=0)

p=np.random.permutation(len(merged))
merged=merged[p]
labels=labels[p]

for i in range(10):
    plt.imshow(merged[i])
    plt.title(labels[i])
    plt.show()

np.save(dataset_path+'train_array.npy',merged)
np.save(dataset_path+'train_labels.npy',labels)