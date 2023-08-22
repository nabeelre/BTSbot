import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, BatchNormalization


def mi_cnn(config, image_shape, metadata_shape):
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
    x_meta = Dropout(config['meta_dropout1'], name='meta_dropout1')(x_meta)
    x_meta = Dense(config['meta_fc2_neurons'], activation='relu', name='meta_fc2')(x_meta)
    
    # --------------------------------------------------------------------------
    # Combined branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(config['comb_fc_neurons'], activation='relu', name='comb_fc')(x)
    x = Dropout(config['comb_dropout1'], name='comb_dropout1')(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="mi_cnn")
    return model
