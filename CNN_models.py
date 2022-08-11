import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dropout


def vgg6(input_shape=(63, 63, 3), n_classes: int = 1):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """
    model = tf.keras.models.Sequential(name='vgg6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, name='drop_0.25'))

    model.add(Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25, name='drop2_0.25'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.4, name='drop3_0.4'))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation, name='fc_out'))

    return model


def vgg6_lowdrop(input_shape=(63, 63, 3), n_classes: int = 1):
    model = tf.keras.models.Sequential(name='vgg6_lowdrop')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, name='drop_0.25'))

    model.add(Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25, name='drop2_0.25'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.25, name='drop3_0.25'))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation, name='fc_out'))

    return model


def vgg6_highdrop(input_shape=(63, 63, 3), n_classes: int = 1):
    model = tf.keras.models.Sequential(name='vgg6_highdrop')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, name='conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4, name='drop_0.4'))

    model.add(Conv2D(32, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv4'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.4, name='drop2_0.4'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.4, name='drop3_0.4'))
    # output layer
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(n_classes, activation=activation, name='fc_out'))

    return model


def vgg16(input_shape=(63, 63, 3), n_classes: int = 1):
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    
    model = tf.keras.models.Sequential(name='vgg16')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", name='conv2'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name='conv3'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu", name='conv4'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv5'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv6'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv7'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv8'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv9'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv10'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv11'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv12'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu", name='conv13'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(units=n_classes, activation=activation))
    
    return model


def vgg9(input_shape=(63, 63, 3), n_classes: int = 1):    
    model = tf.keras.models.Sequential(name='vgg9')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=input_shape, filters=16, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
    model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu", name='conv2'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25, name='drop1_0.25'))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name='conv3'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", name='conv4'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25, name='drop2_0.25'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv5'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv6'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv7'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units=512,activation="relu"))
    model.add(Dropout(0.4, name='drop_0.4'))
#     model.add(Dense(units=4096,activation="relu"))
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model.add(Dense(units=n_classes, activation=activation))
    
    return model