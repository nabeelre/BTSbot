from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, BatchNormalization, Activation

# /----- ----- ----- -----/ IMAGES AND METADATA /----- ----- ----- -----/ 

def mi_cnn(config, image_shape=(63, 63, 3), metadata_shape=(16,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(config['dropout_1'], name='drop1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(config['dropout_2'], name='drop2')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(16, activation='relu', name='metadata_fc_1')(meta_input)
    x_meta = Dense(32, activation='relu', name='metadata_fc_2')(x_meta)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(config['dropout_3'])(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="mi_cnn")

    return model


def mag_cnn(config, image_shape=(63, 63, 3), metadata_shape=(1,)):
    triplet_input = keras.Input(shape=image_shape, name="triplet")
    meta_input = keras.Input(shape=metadata_shape, name="metadata")

    # Convolution branch
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(63, 63, 3), name='conv1')(triplet_input)
    x_conv = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x_conv)
    x_conv = MaxPooling2D(pool_size=(2, 2), name='pool1')(x_conv)
    x_conv = Dropout(config['dropout_1'], name='drop1')(x_conv)

    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x_conv)
    x_conv = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv4')(x_conv)
    x_conv = MaxPooling2D(pool_size=(4, 4), name='pool2')(x_conv)
    x_conv = Dropout(config['dropout_2'], name='drop2')(x_conv)

    x_conv = Flatten()(x_conv)

    # Metadata branch
    x_meta = Dense(1, activation='relu', name='metadata_fc_1')(meta_input)
    
    # Merged branch
    x = Concatenate(axis=1)([x_conv, x_meta])
    x = Dense(16, activation='relu', name='comb_fc_2')(x)
    x = Dropout(config['dropout_3'])(x)

    output = Dense(1, activation='sigmoid', name='fc_out')(x)

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="mi_cnn")

    return model


# /----- ----- ----- -----/ METADATA ONLY /----- ----- ----- -----/  

def fcnn(image_shape=(63, 63, 3), metadata_shape=(16,)):
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

    model = keras.Model(inputs=[triplet_input, meta_input], outputs=output, name="fcnn")

    return model

# /----- ----- ----- -----/ IMAGES ONLY /----- ----- ----- -----/ 

def si_cnn(image_shape=(63, 63, 3)):
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

    model = keras.Model(inputs=triplet_input, outputs=output, name="si_cnn")

    return model