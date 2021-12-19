"""
This is a revised implementation from Cifar10 ResNet example in Keras:
(https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
"""

from __future__ import print_function
# import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from models import resnet_v1, resnet_v2 # resnext, mobilenets, inception_v3, inception_resnet_v2, densenet
# Will have to change imports for other model files
import numpy as np
import os

tf.config.run_functions_eagerly(True)


# Training parameters
batch_size = 16
epochs = 50
data_augmentation = True
num_classes = 10
subtract_pixel_mean = False  # Subtracting pixel mean improves accuracy
base_model = 'resnet20'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = 'None'
model_type = base_model if attention_module==None else base_model+'_'+attention_module

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

val_split = 0.1
val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:] # 45000
x_val, y_val = x_train[:val_indices], y_train[:val_indices] # 5000

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
new_x_train = new_x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# If subtract pixel mean is enabled
if subtract_pixel_mean:
    new_x_train_mean = np.mean(new_x_train, axis=0)
    new_x_train -= new_x_train_mean
    x_test -= new_x_train_mean  # DOUBT

print(new_x_train.shape[0], 'train samples')
print('new_x_train shape:', new_x_train.shape)
print(x_val.shape[0], 'val samples')
print('x_val shape:', x_val.shape)
print(x_test.shape[0], 'test samples')
print('x_test shape:', x_test.shape)


# Convert class vectors to binary class matrices.
new_y_train = to_categorical(new_y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)


depth = 20 # For ResNet, specify the depth (e.g. ResNet50: depth=50)
model = resnet_v1.resnet_v1(input_shape=input_shape, depth=depth, attention_module=attention_module)
# model = resnet_v2.resnet_v2(input_shape=input_shape, depth=depth, attention_module=attention_module)   
# model = resnext.ResNext(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = mobilenets.MobileNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_v3.InceptionV3(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, classes=num_classes, attention_module=attention_module)
# model = densenet.DenseNet(input_shape=input_shape, classes=num_classes, attention_module=attention_module)


# Top-1 and Top-5 classification accuracies used as metrics.
metric = [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy'), 
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy')]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=metric,
              run_eagerly=True)

model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=2,
                               patience=1,
                               min_lr=0.5e-6)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

callbacks = [checkpoint, lr_reducer, early_stop] # No lr_scheduler used.

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(new_x_train, new_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              verbose=1,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(new_x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(new_x_train, new_y_train, batch_size=batch_size),
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print(model.metrics_names)
print(scores)

