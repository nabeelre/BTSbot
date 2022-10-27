import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dropout, Concatenate


def vgg6_metadata(image_shape=(63, 63, 3), metadata_shape=(2,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # print(triplet_input.shape, triplet_input.dtype)
    # print(meta_input.shape, meta_input.dtype)

    x_conv = Conv2D(16, (3, 3), activation='relu', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.25, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    x_conv = Dense(256, activation='relu', name='fc_1')(x_conv)
    # x_conv = Dropout(0.4, name='drop3_0.4')(x_conv)

    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata")

    return model



def vgg4(image_shape=(63, 63, 3)):
    """
    VGG4
    :param input_shape:
    :param n_classes:
    :return:
    """
    model = tf.keras.models.Sequential(name='vgg6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_shape, name='conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, name='drop_0.25'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', name='fc_1'))
    model.add(Dropout(0.4, name='drop3_0.4'))
    model.add(Dense(1, activation='sigmoid', name='fc_out'))

    return model


def vgg6(image_shape=(63, 63, 3)):
    """
        VGG6
    :param input_shape:
    :param n_classes:
    :return:
    """
    model = tf.keras.models.Sequential(name='vgg6')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_shape, name='conv1'))
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
    model.add(Dense(1, activation='sigmoid', name='fc_out'))

    return model


def ld_vgg6(image_shape=(63, 63, 3)):
    model = tf.keras.models.Sequential(name='vgg6_lowdrop')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_shape, name='conv1'))
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
    model.add(Dense(1, activation='sigmoid', name='fc_out'))

    return model


def hd_vgg6(image_shape=(63, 63, 3)):
    model = tf.keras.models.Sequential(name='vgg6_highdrop')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_shape, name='conv1'))
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
    model.add(Dense(units=1, activation='sigmoid', name='fc_out'))

    return model


def vgg16(image_shape=(63, 63, 3)):
    # https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    
    model = tf.keras.models.Sequential(name='vgg16')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=image_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
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
    model.add(Dense(units=1, activation='sigmoid', name='fc_out'))
    
    return model


def vgg9(image_shape=(63, 63, 3)):    
    model = tf.keras.models.Sequential(name='vgg9')
    # input: 63x63 images with 3 channel -> (63, 63, 3) tensors.
    # this applies 16 convolution filters of size 3x3 each.
    model.add(Conv2D(input_shape=image_shape, filters=16, kernel_size=(3,3), padding="same", activation="relu", name='conv1'))
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
    model.add(Dense(units=1, activation='sigmoid', name='fc_out'))
    
    return model