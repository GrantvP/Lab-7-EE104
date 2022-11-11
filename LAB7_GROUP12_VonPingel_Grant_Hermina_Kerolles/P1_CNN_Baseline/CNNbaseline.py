# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:00:11 2022
## https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
@author: Christopher
"""



import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models
import keras_tuner
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/69687794/unable-to-manually-load-cifar10-dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    #zoom_range=0.3
    )
datagen.fit(train_images)

for X_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i].astype(np.uint8))
    plt.show()
    break

# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

train_images=train_images.astype("float32")
test_images=test_images.astype("float32")
mean=np.mean(train_images)
std=np.std(train_images)
test_images=(test_images-mean)/std
train_images=(train_images-mean)/std



## verify that the dataset looks correct
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

num_classes=10
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model
# define cnn BaseLine model
# model = models.Sequential()

# ### This section is from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=WRzW5xSDDbNF ###
# ### You will improve this section for higher accuracy 
# # Create the convolutional base
# #model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.BatchNormalization())

# model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0))
# model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0))
# model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0))




# # Add Dense layers on top
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(10, activation='softmax'))


def block(input_layer,filters,stride=1):

  conv_1 = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same', strides=stride, kernel_regularizer=l2(0.0001))(input_layer)

  bn_1 = tf.keras.layers.BatchNormalization(axis=3,momentum=0.9,epsilon=1e-5)(conv_1)

  activation_layer_b1 = tf.keras.layers.Activation('relu')(bn_1)

  return activation_layer_b1


input = tf.keras.layers.Input(shape=(32, 32, 3))
start = block(input,32)

layer_1 = block(start,64)
mp_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(layer_1)

layer1_identity = tf.keras.layers.Conv2D(filters=32,kernel_size=(1, 1),strides=(1, 1),padding="same",kernel_regularizer=l2(0.0001))(mp_1)
layer1_res1 = block(layer1_identity,64)
layer1_res2 = block(layer1_res1,64)

concat1 = tf.keras.layers.concatenate([mp_1, layer1_res2])

layer_2 = block(concat1,128)
mp_2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(layer_2)

layer_3 = block(mp_2,256)
mp_3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(layer_3)

layer3_identity = tf.keras.layers.Conv2D(filters=128,kernel_size=(1, 1),strides=(1, 1),padding="same",kernel_regularizer=l2(0.0001))(mp_3)
layer3_res1 = block(layer3_identity,256)
layer3_res2 = block(layer3_res1,256)

concat2 = tf.keras.layers.concatenate([mp_3, layer3_res2])

gmp = tf.keras.layers.GlobalAveragePooling2D()(concat2)
dense = tf.keras.layers.Dense(units=10, activation="softmax")(gmp) #kernel_initializer="he_normal", 

model = tf.keras.models.Model(inputs=input, outputs=dense)

# Here's the complete architecture of your model:
model.summary()

### End code from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb#scrollTo=WRzW5xSDDbNF ####




## Compile and train the model
# 
opt = SGD(lr=0.04, decay=5e-4, momentum=0.9, nesterov=True)
# opt = Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10  , 
#                     validation_data=(test_images, test_labels))
history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                    steps_per_epoch = len(train_images) / 128, epochs=95
                    , validation_data=(test_images, test_labels))


## Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

## save trained model in file "MyGroup_CIFARmodel.h5"
# You will use this trained model to test the images 
model.save('MyGroup_CIFARmodel_baseline.h5')

## Save file to your local computer if you will test it locally
##  using the provided file test_image.py
#https://neptune.ai/blog/google-colab-dealing-with-files 
""" from google.colab import files
files.download('MyGroup_CIFARmodel.h5')

###########################################################################
## If you run on GoogleColab, then add the following code on GoogleColab ##
###########################################################################

# load the trained CIFAR10 model
from keras.models import load_model
model = load_model('MyGroup_CIFARmodel.h5')

def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img / 255.0
	return img

#https://stackoverflow.com/questions/72479044/cannot-import-name-load-img-from-keras-preprocessing-image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

# get the image from the internet
URL = "https://wagznwhiskerz.com/wp-content/uploads/2017/10/home-cat.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)


# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])

# get the image from the internet
URL = "https://image.shutterstock.com/image-vector/airplane-600w-646772488.jpg"
picture_path  = tf.keras.utils.get_file(origin=URL)
img = load_image(picture_path)
result = model.predict(img)


# show the picture
image = plt.imread(picture_path)
plt.imshow(image)

# show prediction result.
print('\nPrediction: This image most likely belongs to ' + class_names[int(result.argmax(axis=-1))])


############################################################
## This website has everything to improve your accuracy  ###
############################################################

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/ 

 """