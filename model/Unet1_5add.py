from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers


def SE(x, out_dim):
    ratio = 24
    # print(x.shape)
    squeeze = GlobalAveragePooling2D()(x)
    # print(squeeze.shape)
    excitation = Dense(units=out_dim // ratio)(squeeze)
    # print(out_dim // ratio)
    # print(excitation.shape)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    # print(excitation.shape)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, out_dim))(excitation)
    # print(excitation.shape)
    scale = multiply([x, excitation])
    # print( scale.shape)
    return scale


def conv(x, f):
    conv_1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
    batc_1 = BatchNormalization(axis=-1, momentum=0.6)(conv_1)
    acti_1 = Activation('relu')(batc_1)
    drop_1 = Dropout(0.2)(acti_1)
    conv_2 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(drop_1)
    batc_2 = BatchNormalization(axis=-1, momentum=0.6)(conv_2)
    acti_2 = Activation('relu')(batc_2)
    Se = SE(acti_2, f)
    return Se


def conva(x, f):
    conv_1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
    batc_1 = BatchNormalization(axis=-1, momentum=0.6)(conv_1)
    acti_1 = Activation('relu')(batc_1)
    Se = SE(acti_1, f)
    return Se


def unet5(T1_shape, T2_shape, T3_shape, T4_shape, T5_shape):
    T1 = Input(shape=T1_shape, name='input_1')  # name is added by lyq
    T2 = Input(shape=T2_shape, name='input_2')  # name is added by lyq
    T3 = Input(shape=T3_shape, name='input_3')  # name is added by lyq
    T4 = Input(shape=T4_shape, name='input_4')  # name is added by lyq
    T5 = Input(shape=T5_shape, name='input_5')  # name is added by lyq
    f = 32

    '''downsample_T1'''
    # 576x576
    conv_T1 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(T1)
    batc_T1 = BatchNormalization(axis=-1, momentum=0.6)(conv_T1)
    acti_T1 = Activation('relu')(batc_T1)

    conv_T2 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(T2)
    batc_T2 = BatchNormalization(axis=-1, momentum=0.6)(conv_T2)
    acti_T2 = Activation('relu')(batc_T2)

    conv_T3 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(T3)
    batc_T3 = BatchNormalization(axis=-1, momentum=0.6)(conv_T3)
    acti_T3 = Activation('relu')(batc_T3)

    conv_T4 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(T4)
    batc_T4 = BatchNormalization(axis=-1, momentum=0.6)(conv_T4)
    acti_T4 = Activation('relu')(batc_T4)

    conv_T5 = Conv2D(32, 1, padding='same', kernel_initializer='he_normal')(T5)
    batc_T5 = BatchNormalization(axis=-1, momentum=0.6)(conv_T5)
    acti_T5 = Activation('relu')(batc_T5)

    merg_add = Concatenate(axis=-1)([acti_T1, acti_T2, acti_T3, acti_T4, acti_T5])

    conv1 = conv(merg_add, f * 2)
    maxp1 = MaxPool2D(2)(conv1)  # 1

    # 288x288
    conv2 = conv(maxp1, f * 2)

    maxp2 = MaxPool2D(2)(conv2)  # 2

    # 144x144
    conv3 = conv(maxp2, f * 4)
    maxp3 = MaxPool2D(2)(conv3)  # 3

    # 72x72
    conv4 = conv(maxp3, f * 8)
    maxp4 = MaxPool2D(2)(conv4)  # 4

    # 36x36
    conv5 = conv(maxp4, f * 16)

    '''upsample_T1'''

    upsa1 = UpSampling2D(2)(conv5)  # acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv4, upsa1])
    conv_1 = conv(merg1, f * 8)

    upsa2 = UpSampling2D(2)(conv_1)
    merg2 = Concatenate(axis=-1)([conv3, upsa2])
    conv_2 = conv(merg2, f * 4)

    upsa3 = UpSampling2D(2)(conv_2)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv_3 = conv(merg3, f * 2)

    upsa4 = UpSampling2D(2)(conv_3)
    merg4 = Concatenate(axis=-1)([conv1, upsa4])
    conv_4 = conv(merg4, f * 2)

    convol = Conv2D(2, 1, padding='same')(conv_4)
    acti = Activation('softmax')(convol)

    model = Model(inputs=[T1, T2, T3, T4, T5], outputs=acti)

    return model
