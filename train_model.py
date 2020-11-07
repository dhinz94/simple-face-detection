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
use_bias=False
start_filter_amount = 64
epochs = 10
batchsize = 32
test_split=0.25

images = np.load(dataset_path + 'image_array.npy')[:10000]
boxes = np.load(dataset_path + 'box_array.npy')[:10000]

train_images=images[:int((1-test_split)*len(images))]
test_images=images[int(test_split*len(images)):]

train_boxes=boxes[:int((1-test_split)*len(images))]
test_boxes=boxes[int(test_split*len(images)):]

images=None
boxes=None


resolution = train_images.shape[1]
print(train_images.shape)
print(train_boxes.shape)

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
epoch_validation_losses=[]

for e in range(epochs):

    p=np.random.permutation(len(train_images))

    train_images=train_images[p]
    train_boxes=train_boxes[p]

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    train_step = compile()

    batch_losses = []
    batch_validation_losses=[]

    for b in range(int(len(train_images) / batchsize)):
        batch_images = (train_images[b * batchsize:(b + 1) * batchsize] / 255).astype('float32')
        batch_boxes = (train_boxes[b * batchsize:(b + 1) * batchsize]).astype('float32')

        batch_test_images= (test_images[b * batchsize:(b + 1) * batchsize] / 255).astype('float32')
        batch_test_boxes = (test_boxes[b * batchsize:(b + 1) * batchsize]).astype('float32')


        loss = train_step(batch_images, batch_boxes)

        pred_test_boxes=model(batch_test_images)
        validation_loss=loss_function(pred_test_boxes,batch_test_boxes)

        batch_losses.append(np.array(loss))
        batch_validation_losses.append(np.array(validation_loss))


        if b % int(len(train_images) / batchsize / 5) == 0:
            print('Epoch:', e, 'Batch:', b, 'Loss:', np.array(loss),'Val. Loss:',np.array(validation_loss))

    epoch_losses.append(np.mean(batch_losses))
    epoch_validation_losses.append(np.mean(batch_validation_losses))
    model.save(dataset_path+'model.h5')

plt.figure()
plt.plot(epoch_losses)
plt.plot(epoch_validation_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss')
plt.legend(['training','validation'])

print('results from test dataset:')
for i in range(20):
    num = np.random.randint(0, len(test_images))


    input_image = test_images[num].reshape(-1, test_images.shape[1], test_images.shape[2], test_images.shape[3]) / 255
    predicted_relative_box = np.array(model(input_image)[0])
    predicted_coordinate_box = utils.convert_relative_box_to_coordinate_box(predicted_relative_box, test_images.shape[1], test_images.shape[2])
    label_coordinate_box = utils.convert_relative_box_to_coordinate_box(test_boxes[num], test_images.shape[1], test_images.shape[2])

    plt.figure()
    plt.imshow(input_image[0])
    plt.gca().add_patch(Rectangle((predicted_coordinate_box[0], predicted_coordinate_box[1]), predicted_coordinate_box[2], predicted_coordinate_box[3], linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((label_coordinate_box[0], label_coordinate_box[1]), label_coordinate_box[2], label_coordinate_box[3], linewidth=1, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.title('green=True, red=predicted')

plt.show()
