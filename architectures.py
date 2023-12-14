import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, BatchNormalization


def mm_cnn(config, image_shape, metadata_shape):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # --------------------------------------------------------------------------
    # First convolutional block
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv1', input_shape=image_shape)(triplet_input)
    
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv2')(x_conv)
    
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(config['conv_dropout1'], name='conv_dropout1')(x_conv)

    # --------------------------------------------------------------------------
    # Second convolutional block
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv3')(x_conv)
    
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv4')(x_conv)
    
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(config['conv_dropout2'], name='conv_dropout2')(x_conv)

    x_conv = Flatten()(x_conv)

    # --------------------------------------------------------------------------
    # Metadata branch
    x_meta = BatchNormalization(input_shape=metadata_shape)(meta_input)
    x_meta = Dense(config['meta_fc1_neurons'], activation='relu', name='meta_fc1')(x_meta)
    x_meta = Dropout(config['meta_dropout'], name='meta_dropout')(x_meta)
    x_meta = Dense(config['meta_fc2_neurons'], activation='relu', name='meta_fc2')(x_meta)
    
    # --------------------------------------------------------------------------
    # Combined branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(config['comb_fc_neurons'], activation='relu', name='comb_fc')(x)
    x = Dropout(config['comb_dropout'], name='comb_dropout')(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="mm_cnn")
    return model


def um_cnn(config, image_shape):
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # --------------------------------------------------------------------------
    # First convolutional block
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv1', input_shape=image_shape)(triplet_input)
    
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv2')(x_conv)
    
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(config['conv_dropout1'], name='conv_dropout1')(x_conv)

    # --------------------------------------------------------------------------
    # Second convolutional block
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv3')(x_conv)
    
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv4')(x_conv)
    
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(config['conv_dropout2'], name='conv_dropout2')(x_conv)

    x_conv = Flatten()(x_conv)
 
    x_conv = Dense(config['head_neurons'], activation='relu', name='head_fc')(x_conv)
    x_conv = Dropout(config['head_dropout'], name='head_dropout')(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="um_cnn")
    return model


def um_cnn_small(config, image_shape):
    triplet_input = keras.Input(shape=image_shape, name="triplet")

    # --------------------------------------------------------------------------
    # First convolutional block
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv1', input_shape=image_shape)(triplet_input)
    
    x_conv = Conv2D(config['conv1_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv2')(x_conv)
    
    # x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(config['conv_dropout1'], name='conv_dropout1')(x_conv)

    # --------------------------------------------------------------------------
    # Second convolutional block
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv3')(x_conv)
    
    x_conv = Conv2D(config['conv2_channels'], (config['conv_kernel'], config['conv_kernel']), 
                    activation='relu', padding='same', name='conv4')(x_conv)
    
    # x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(config['conv_dropout2'], name='conv_dropout2')(x_conv)

    x_conv = Flatten()(x_conv)
 
    x_conv = Dense(config['head_neurons'], activation='relu', name='head_fc')(x_conv)
    x_conv = Dropout(config['head_dropout'], name='head_dropout')(x_conv)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_conv)

    model = keras.Model(inputs=triplet_input, outputs=output, name="um_cnn_small")
    return model


def um_nn(config, metadata_shape):
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    x_meta = BatchNormalization(input_shape=metadata_shape)(meta_input)
    x_meta = Dense(config['meta_fc1_neurons'], activation='relu', name='meta_fc1')(x_meta)
    x_meta = Dropout(config['meta_dropout'], name='meta_dropout')(x_meta)
    x_meta = Dense(config['meta_fc2_neurons'], activation='relu', name='meta_fc2')(x_meta)
    
    x_meta = Dense(config['head_neurons'], activation='relu', name='head_fc')(x_meta)
    x_meta = Dropout(config['head_dropout'], name='head_dropout')(x_meta)

    output = Dense(1, activation='sigmoid', name='fc_out')(x_meta)

    model = keras.Model(inputs=meta_input, outputs=output, name="um_nn")
    return model