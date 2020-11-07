import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Add, Activation, Dense
import tensorflow.keras.backend as K
from utils import utils
from matplotlib.patches import Rectangle

def loss_function(pred, true):
    loss = K.mean(K.square(pred - true))
    return loss


def compile():

    @tf.function
    def train_step(images, boxes):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_function(predictions, boxes)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step

def block(input,filter_amount):
    x = Conv2D(filter_amount, (3, 3), strides=(2, 2), activation=None)(input)
    x = normalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filter_amount, (3, 3), strides=(1, 1), activation=None)(x)
    x = normalization()(x)
    output = Activation(activation)(x)

    return output


# colab path
dataset_path = '/content/drive/My Drive/celeba_copy/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

activation = tf.nn.relu
normalization = BatchNormalization
start_filter_amount = 64
epochs = 100
batchsize = 32

image_array = np.load(dataset_path + 'image_array.npy')
box_array = np.load(dataset_path + 'box_array.npy')

resolution = image_array.shape[1]
print(image_array.shape)
print(box_array.shape)

input = Input(shape=(resolution, resolution, 3))

x=block(input,start_filter_amount)
x=block(x,start_filter_amount*2)
x=block(x,start_filter_amount*4)
output=block(x,start_filter_amount*8)

x = Flatten()(x)
x = Dense(4)(x)
output = Activation('sigmoid')(x)

model = tf.keras.models.Model(inputs=input, outputs=output)
print(model.summary())



epoch_losses = []

for e in range(epochs):

    p=np.random.permutation(len(image_array))

    image_array=image_array[p]
    box_array=box_array[p]

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    train_step = compile()

    batch_losses = []
    for b in range(int(len(image_array) / batchsize)):
        images = (image_array[b * batchsize:(b + 1) * batchsize] / 255).astype('float32')
        boxes = (box_array[b * batchsize:(b + 1) * batchsize]).astype('float32')

        loss = train_step(images, boxes)
        batch_losses.append(np.array(loss))

        if b % int(len(image_array) / batchsize / 1) == 0:
            print('Epoch:', e, 'Batch:', b, 'Loss:', np.array(loss))

    epoch_losses.append(np.mean(batch_losses))

plt.figure()
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss')

for i in range(20):
    num = np.random.randint(0, len(image_array))


    input_image = image_array[num].reshape(-1, image_array.shape[1], image_array.shape[2], image_array.shape[3]) / 255
    predicted_relative_box = np.array(model(input_image)[0])
    predicted_coordinate_box = utils.convert_relative_box_to_coordinate_box(predicted_relative_box, image_array.shape[1], image_array.shape[2])
    label_coordinate_box = utils.convert_relative_box_to_coordinate_box(box_array[num], image_array.shape[1], image_array.shape[2])

    plt.figure()
    plt.imshow(input_image[0])
    plt.gca().add_patch(Rectangle((predicted_coordinate_box[0], predicted_coordinate_box[1]), predicted_coordinate_box[2], predicted_coordinate_box[3], linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((label_coordinate_box[0], label_coordinate_box[1]), label_coordinate_box[2], label_coordinate_box[3], linewidth=1, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.title('green=True, red=predicted')

plt.show()
