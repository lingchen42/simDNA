from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D


def VGG16(include_top=True, weights=None,
          input_tensor=None, input_shape=None,
          pooling=None,
          nfcfilters=4096,
          nconvfilters=[64, 128, 512, 512, 512],
          classes=1,
          nblocks=5):

    seq_input = Input(shape=input_shape)

    if nblocks >= 1:
        # Block 1
        x = Conv2D(nconvfilters[0], (4, 3), activation='relu', padding='same', name='block1_conv1')(seq_input)
        x = Conv2D(nconvfilters[0], (1, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((1, 2), name='block1_pool')(x)

    if nblocks >= 2:
        # Block 2
        x = Conv2D(nconvfilters[1], (1, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(nconvfilters[1], (1, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((1, 2), name='block2_pool')(x)

    if nblocks >= 3:
        # Block 3
        x = Conv2D(nconvfilters[2], (1, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(nconvfilters[2], (1, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(nconvfilters[2], (1, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((1, 2), name='block3_pool')(x)

    if nblocks >= 4:
        # Block 4
        x = Conv2D(nconvfilters[3], (1, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(nconvfilters[3], (1, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(nconvfilters[3], (1, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((1, 2), name='block4_pool')(x)

    if nblocks >= 5:
        # Block 5
        x = Conv2D(nconvfilters[4], (1, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(nconvfilters[4], (1, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(nconvfilters[4], (1, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((1, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(nfcfilters, activation='relu', name='fc1')(x)
        x = Dense(nfcfilters, activation='relu', name='fc2')(x)
        if classes ==1:
            x = Dense(classes, activation='sigmoid', name='predictions')(x)
        else:
            x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # inputs
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = seq_input

    # Create model
    model = Model(inputs, x, name='vgg16')

    if weights:
        raise NotImplementedError

    return model
