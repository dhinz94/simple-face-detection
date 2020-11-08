import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Add, Activation, Dense
import tensorflow.keras.backend as K
from utils import utils
from matplotlib.patches import Rectangle

@tf.function
def loss_function(pred_boxes, pred_labels,true_boxes,true_labels):


    box_loss = K.mean(true_labels*K.sum(K.square(pred_boxes - true_boxes),axis=1))
    # classification_loss=K.mean(K.square(pred_labels-true_labels))
    # classification_loss=K.mean(-true_labels*K.log(pred_labels)-(1-true_labels)*K.log(1-pred_labels))
    # loss=box_loss+classification_loss

    # loss=classification_loss
    loss=box_loss

    return loss


def compile():

    @tf.function
    def train_step(images, boxes,labels):
        with tf.GradientTape() as tape:
            pred_boxes,pred_labels = model(images, training=True)
            loss = loss_function(pred_boxes,pred_labels, boxes,labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step

def block(input,filter_amount):
    x = Conv2D(filter_amount, (3, 3), strides=(2, 2), activation=None,padding='same')(input)
    x = normalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filter_amount, (3, 3), strides=(1, 1), activation=None,padding='same')(x)
    x = normalization()(x)
    output = Activation(activation)(x)

    return output


# colab path
dataset_path = '/content/drive/My Drive/simple_face_detection_data/'

# local path for development
# dataset_path='/home/dominic/Dokumente/Github/simple-face-detection/data/'

activation = tf.nn.relu
normalization = BatchNormalization
use_bias=False
start_filter_amount = 8
epochs = 10
batchsize = 16
test_split=0.25

images = np.load(dataset_path + 'train_images.npy')[:10000]
boxes = np.load(dataset_path + 'train_boxes.npy')[:10000]
labels=np.load(dataset_path + 'train_labels.npy')[:10000]

print('images:',len(images))
print('faces:',labels.sum())



train_images=images[:int((1-test_split)*len(images))]
test_images=images[int((1-test_split)*len(images)):]


train_boxes=boxes[:int((1-test_split)*len(images))]
test_boxes=boxes[int((1-test_split)*len(images)):]


train_labels=labels[:int((1-test_split)*len(images))]
test_labels=labels[int((1-test_split)*len(images)):]

images=None
boxes=None
labels=None




resolution = train_images.shape[1]
print(train_images.shape)
print(train_boxes.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_boxes.shape)
print(test_labels.shape)

input = Input(shape=(resolution, resolution, 3))

x=block(input,start_filter_amount)
x=block(x,start_filter_amount*2)
x=block(x,start_filter_amount*4)
x=block(x,start_filter_amount*8)
x=block(x,start_filter_amount*16)
x=block(x,start_filter_amount*32)
output=block(x,start_filter_amount*64)

x = Flatten()(x)
x= Dense(100)(x)
output_box = Dense(4)(x)
output_box = Activation('sigmoid')(output_box)

output_labels=Dense(1)(x)
output_labels =Activation('sigmoid')(output_labels)

model = tf.keras.models.Model(inputs=input, outputs=[output_box,output_labels])
print(model.summary())



for i in range(10):
    num = np.random.randint(0, len(test_images))

    input_image = test_images[num].reshape(-1, test_images.shape[1], test_images.shape[2], test_images.shape[3]) / 255

    label_coordinate_box = utils.convert_relative_box_to_coordinate_box(test_boxes[num], test_images.shape[1], test_images.shape[2])

    plt.figure()
    plt.imshow(input_image[0])
    plt.gca().add_patch(Rectangle((label_coordinate_box[0], label_coordinate_box[1]), label_coordinate_box[2], label_coordinate_box[3], linewidth=1, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.title('green=True '+str(test_labels[num][0]))

plt.show()

epoch_losses = []
epoch_validation_losses=[]
epoch_validation_accuracies=[]

optimizer = tf.keras.optimizers.Adam(lr=1e-5)

train_step = compile()

for e in range(epochs):

    p=np.random.permutation(len(train_images))

    train_images=train_images[p]
    train_boxes=train_boxes[p]

    batch_losses = []
    batch_validation_losses=[]
    batch_validation_accuracies=[]

    for b in range(int(len(train_images) / batchsize)):
        batch_images = (train_images[b * batchsize:(b + 1) * batchsize] / 255).astype('float32')
        batch_boxes = (train_boxes[b * batchsize:(b + 1) * batchsize]).astype('float32')
        batch_labels= (train_labels[b * batchsize:(b + 1) * batchsize]).astype('float32')

        test_b=np.random.randint(0,int(len(test_images) / batchsize))

        batch_test_images= (test_images[test_b * batchsize:(test_b + 1) * batchsize] / 255).astype('float32')
        batch_test_boxes = (test_boxes[test_b * batchsize:(test_b + 1) * batchsize]).astype('float32')
        batch_test_labels = (test_labels[test_b * batchsize:(test_b + 1) * batchsize]).astype('float32')

        loss = train_step(batch_images, batch_boxes,batch_labels)

        pred_test_boxes,pred_test_labels=model(batch_test_images)
        validation_loss=loss_function(pred_test_boxes,pred_test_labels,batch_test_boxes,batch_test_labels)

        batch_validation_accuracy=np.sum(np.array(pred_test_labels))/np.sum(test_labels)

        batch_losses.append(np.array(loss))
        batch_validation_losses.append(np.array(validation_loss))
        batch_validation_accuracies.append(batch_validation_accuracy)





        if b % int(len(train_images) / batchsize / 5) == 0:
            print('Epoch:', e, 'Batch:', b, 'Loss:', np.array(np.mean(batch_losses)),'Val. Loss:',np.array(np.mean(batch_validation_losses)),'Val. Acc.',np.mean(batch_validation_accuracies))

    epoch_losses.append(np.mean(batch_losses))
    epoch_validation_losses.append(np.mean(batch_validation_losses))
    epoch_validation_accuracies.append(np.mean(batch_validation_accuracies))
    model.save(dataset_path+'model.h5')

plt.figure()
plt.plot(epoch_losses)
plt.plot(epoch_validation_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss')
plt.legend(['training','validation'])

plt.figure()
plt.plot(epoch_validation_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Acc.')
plt.title('Epoch Acc.')
plt.legend(['validation acc.'])

print('results from test dataset:')
for i in range(20):
    num = np.random.randint(0, len(test_images))


    input_image = test_images[num].reshape(-1, test_images.shape[1], test_images.shape[2], test_images.shape[3]) / 255
    predicted_relative_box,pred_label=model(input_image)
    predicted_relative_box = np.array(predicted_relative_box[0])

    predicted_coordinate_box = utils.convert_relative_box_to_coordinate_box(predicted_relative_box, test_images.shape[1], test_images.shape[2])
    label_coordinate_box = utils.convert_relative_box_to_coordinate_box(test_boxes[num], test_images.shape[1], test_images.shape[2])

    plt.figure()
    plt.imshow(input_image[0])
    # if np.array(pred_label)[0,0]>0.5:
    plt.gca().add_patch(Rectangle((predicted_coordinate_box[0], predicted_coordinate_box[1]), predicted_coordinate_box[2], predicted_coordinate_box[3], linewidth=1, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(Rectangle((label_coordinate_box[0], label_coordinate_box[1]), label_coordinate_box[2], label_coordinate_box[3], linewidth=1, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.title('green=True, red=predicted, conf:'+str(np.array(pred_label[0][0]).round(2)))

plt.show()
