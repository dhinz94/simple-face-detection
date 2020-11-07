import numpy as np
import matplotlib.pyplot as plt
import  tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Flatten,Dense,BatchNormalization, Add, Activation,Dense
import tensorflow.keras.backend as K


# colab path
dataset_path='/content/drive/My Drive/celeba_copy/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

activation=tf.nn.relu
normalization=BatchNormalization
start_filter_size=16
epochs=10
batchsize=16


image_array=np.load(dataset_path+'image_array.npy')
box_array=np.load(dataset_path+'box_array.npy')

resolution=image_array.shape[1]
print(image_array.shape)
print(box_array.shape)



input=Input(shape=(resolution,resolution,3))

x=Conv2D(start_filter_size,(3,3),strides=(2,2),activation=None)(input)
x=normalization()(x)
x=Activation(activation)(x)

x=Conv2D(start_filter_size*2,(3,3),strides=(2,2),activation=None)(x)
x=normalization()(x)
x=Activation(activation)(x)

x=Conv2D(start_filter_size*4,(3,3),strides=(2,2),activation=None)(x)
x=normalization()(x)
x=Activation(activation)(x)

x=Conv2D(start_filter_size*8,(3,3),strides=(2,2),activation=None)(x)
x=normalization()(x)
x=Activation(activation)(x)

x=Flatten()(x)
x=Dense(4)(x)
output=Activation('sigmoid')(x)

model=tf.keras.models.Model(inputs=input,outputs=output)
print(model.summary())


def loss_function(pred,true):
    loss=K.mean(K.square(pred-true))
    return loss

def compile():

    @tf.function
    def train_step(images,boxes):
        with tf.GradientTape() as tape:
            predictions=model(images,training=True)
            loss=loss_function(predictions,boxes)

        gradients=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss


    return train_step

optimizer=tf.keras.optimizers.Adam(lr=1e-3)

train_step=compile()

for e in range(epochs):


    for b in range(int(len(image_array)/batchsize)):
        images=(image_array[b*batchsize:(b+1)*batchsize]/255).astype('float32')
        boxes=(box_array[b*batchsize:(b+1)*batchsize]).astype('float32')

        loss=train_step(images,boxes)
        if b%int(len(image_array)/batchsize/4)==0:

            print('Epoch:',e,'Batch:',b,'Loss:',np.array(loss))

box=model(image_array[10].reshape(-1,image_array.shape[1],image_array.shape[2],image_array.shape[3])/255)
print(box)
print(box_array[0])









