import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, BatchNormalization, Activation

# /----- ----- ----- -----/ IMAGES AND METADATA /----- ----- ----- -----/ 

def vgg6_metadata_1_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.25, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.25)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_1")

    return model


def vgg6_metadata_1_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # High dropout
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.5, name='drop_0.5')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_2")

    return model


def vgg6_metadata_1_2_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Medium dropout
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.375, name='drop_0.5')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.375, name='drop2_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.375)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_2")

    return model


def vgg6_metadata_1_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # extra conv layer
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.5, name='drop_0.5')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv5')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_3")

    return model


def vgg6_metadata_1_bn_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Use BN after full conv block and denses
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_bn_1")

    return model


def vgg6_metadata_1_bn_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Use BN after RELU of each conv
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_bn_2")

    return model


def vgg6_metadata_1_bn_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Use BN before RELU of each conv
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation=None, input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(16, (3, 3), activation=None, name='conv2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation=None, name='conv3')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(32, (3, 3), activation=None, name='conv4')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Activation("relu")(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_1_bn_3")

    return model


def vgg6_metadata_2_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.25, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None
    x_meta = Flatten()(meta_input)

    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_1')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation='relu', name='comb_fc_3')(x)
    x = Dropout(0.25)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_2_1")

    return model


def vgg6_metadata_2_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Big dense layers
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.25, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(64, activation='relu', name='comb_fc_1')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu', name='comb_fc_3')(x)
    x = Dropout(0.25)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_2_2")

    return model


def vgg6_metadata_2_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Big dense layers, high drop out
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.5, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(64, activation='relu', name='comb_fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='comb_fc_3')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_2_3")

    return model


def vgg6_metadata_2_bn_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(16, activation='relu', name='comb_fc_1')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dense(16, activation='relu', name='comb_fc_3')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_2_bn_1")

    return model


def vgg6_metadata_2_bn_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    # Big dense layers
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(64, activation='relu', name='comb_fc_1')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dense(64, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dense(64, activation='relu', name='comb_fc_3')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg6_metadata_2_bn_2")

    return model


def vgg9_metadata_1_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.5, name='drop_0.5')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.5')(x_conv)

    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5')(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6')(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool3')(x_conv)
    x_conv = Dropout(0.5, name='drop3_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(64, activation='relu', name='comb_fc_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='comb_fc_2')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg9_metadata_1_1")

    return model


def vgg9_metadata_1_bn_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool3')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    # None

    # Merged branch
    x = Concatenate(axis=1)([x_conv, meta_input])
    x = Dense(64, activation='relu', name='comb_fc_1')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dense(64, activation='relu', name='comb_fc_2')(x)
    x = BatchNormalization(axis=1)(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="vgg9_metadata_1_bn_1")

    return model


# /----- ----- ----- -----/ METADATA ONLY /----- ----- ----- -----/  

def metadata_only_1_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(64, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_1_1")

    return model


def metadata_only_1_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(64, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = Dropout(0.5)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = Dropout(0.75)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_1_2")

    return model


def metadata_only_1_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(64, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)
    
    x_meta = Dense(64, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_1_3")

    return model


def metadata_only_2_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(128, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(128, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_2_1")

    return model


def metadata_only_2_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(128, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.5)(x_meta)
    
    x_meta = Dense(128, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.5)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_2_2")

    return model


def metadata_only_2_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(128, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)
    
    x_meta = Dense(128, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_2_3")

    return model


def metadata_only_3_1(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(32, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.25)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = Dropout(0.25)(x_meta)
    
    x_meta = Dense(32, activation='relu', name='metadata_fc_5')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_6')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_3_1")

    return model


def metadata_only_3_2(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(32, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dropout(0.125)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = Dropout(0.25)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = Dropout(0.375)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = Dropout(0.5)(x_meta)
    
    x_meta = Dense(32, activation='relu', name='metadata_fc_5')(x_meta)
    x_meta = Dropout(0.625)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_6')(x_meta)
    x_meta = Dropout(0.75)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_3_2")

    return model


def metadata_only_3_3(image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = Dense(32, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = BatchNormalization(axis=1)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_3')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_4')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)
    
    x_meta = Dense(32, activation='relu', name='metadata_fc_5')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    x_meta = Dense(32, activation='relu', name='metadata_fc_6')(x_meta)
    x_meta = BatchNormalization(axis=1)(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="metadata_only_3_3")

    return model


# /----- ----- ----- -----/ IMAGES ONLY /----- ----- ----- -----/ 

def vgg6_1(image_shape=(63, 63, 3)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.25, name='drop2_0.25')(x_conv)

    x_conv = Flatten()(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg6_1")

    return model


def vgg6_2(image_shape=(63, 63, 3)):
    # High dropout
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.25, name='drop_0.25')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg6_2")

    return model


def vgg6_3(image_shape=(63, 63, 3)):
    # with BN
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)

    x_conv = Flatten()(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg6_3")

    return model


def vgg6_4(image_shape=(63, 63, 3)):
    # with BN
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Flatten()(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg6_4")

    return model


def vgg9_1(image_shape=(63, 63, 3)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)

    x_conv = Flatten()(x_conv)

    x_conv = Dense(64, activation='relu', name='fc_1')(x_conv)
    x_conv = BatchNormalization(axis=1)(x_conv)
    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg9_1")

    return model


def vgg9_2(image_shape=(63, 63, 3)):
    # with BN after each conv
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(0.5, name='drop_0.5')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop2_0.5')(x_conv)

    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv5')(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6')(x_conv)
    x_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv7')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(0.5, name='drop3_0.5')(x_conv)

    x_conv = Flatten()(x_conv)

    x_conv = Dense(64, activation='relu', name='fc_1')(x_conv)
    x_conv = Dropout(0.5, name='drop4_0.5')(x_conv)
    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="vgg9_2")

    return model
