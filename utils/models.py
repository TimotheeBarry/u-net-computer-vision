import tensorflow as tf
from keras.layers import concatenate, Conv2D, Input, Dropout,MaxPooling2D, Conv2DTranspose


IMG_WIDTH=None
IMG_HEIGHT=None
IMG_CHANNELS=3

filters_nb=16
Dropout_value=0.3

class UNet() :
    # This is a basic implementation of the U-Net model
    def __init__(self):
        self.inputs= Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

        c1 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(self.inputs)
        c1 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c1)
        p1 = MaxPooling2D((2,2))(c1)

        c2 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p1)
        c2 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c2)
        p2 = MaxPooling2D((2,2))(c2)

        c3 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p2)
        c3 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c3)
        p3 = MaxPooling2D((2,2))(c3)

        c4 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p3)
        c4 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c4)
        p4 = MaxPooling2D((2,2))(c4)

        c5 = Conv2D(filters_nb*16,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p4)
        c5 = Conv2D(filters_nb*16,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c5)

        u6 = Conv2DTranspose(filters_nb*8,(2,2),strides=(2,2),padding='same')(c5)
        u6 = concatenate([u6,c4])
        c6 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u6)
        c6 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c6)

        u7 = Conv2DTranspose(filters_nb*4,(2,2),strides=(2,2),padding='same')(c6)
        u7 = concatenate([u7,c3])
        c7 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u7)
        c7 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c7)

        u8 = Conv2DTranspose(filters_nb*2,(2,2),strides=(2,2),padding='same')(c7)
        u8 = concatenate([u8,c2])
        c8 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u8)
        c8 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c8)

        u9 = Conv2DTranspose(filters_nb,(2,2),strides=(2,2),padding='same')(c8)
        u9 = concatenate([u9,c1])
        c9 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u9)
        c9 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c9)

        self.outputs=Conv2D(1,(1,1), activation='sigmoid')(c9)

    def create_model(self):
        return tf.keras.Model(inputs=[self.inputs,],outputs=[self.outputs,])
    

class UNetStride() :
    # U-Net with stride 2 instead of max pooling
    def __init__(self):
        self.inputs= Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

        c1 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(self.inputs)
        c1 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c1)
        p1 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same',strides=(2,2))(c1)

        c2 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p1)
        c2 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c2)
        p2 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same',strides=(2,2))(c2)

        c3 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p2)
        c3 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c3)
        p3 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same',strides=(2,2))(c3)

        c4 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p3)
        c4 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c4)
        p4 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same',strides=(2,2))(c4)

        c5 = Conv2D(filters_nb*16,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(p4)
        c5 = Conv2D(filters_nb*16,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c5)

        u6 = Conv2DTranspose(filters_nb*8,(2,2),strides=(2,2),padding='same')(c5)
        u6 = concatenate([u6,c4])
        c6 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u6)
        c6 = Conv2D(filters_nb*8,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c6)

        u7 = Conv2DTranspose(filters_nb*4,(2,2),strides=(2,2),padding='same')(c6)
        u7 = concatenate([u7,c3])
        c7 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u7)
        c7 = Conv2D(filters_nb*4,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c7)

        u8 = Conv2DTranspose(filters_nb*2,(2,2),strides=(2,2),padding='same')(c7)
        u8 = concatenate([u8,c2])
        c8 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u8)
        c8 = Conv2D(filters_nb*2,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c8)

        u9 = Conv2DTranspose(filters_nb,(2,2),strides=(2,2),padding='same')(c8)
        u9 = concatenate([u9,c1])
        c9 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(u9)
        c9 = Conv2D(filters_nb,(3,3), activation='relu', kernel_initializer='he_normal',padding='same')(c9)

        self.outputs=Conv2D(1,(1,1), activation='sigmoid')(c9)

    def create_model(self):
        return tf.keras.Model(inputs=[self.inputs,],outputs=[self.outputs,])    