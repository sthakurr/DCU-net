# DCU-Net: A Dual channel U-Shaped network for image splicing forgery detection
import sys

import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.applications.vgg16 import VGG16


def high_pass_filter(img):

    # define the high pass filters
    # TODO: Check the filter shapes once again (if they are supposed to be passed in all three channels)
    kernel_horizontal = tf.constant([
        [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
    ], dtype=tf.float32)
    kernel_vertical = tf.constant([
        [-1, -2, -1], [0, 0, 0], [1, 2, 1]
    ], dtype=tf.float32)

    # expand the dimensions to make 4-D tensors
    kernel_horizontal = tf.expand_dims(tf.expand_dims(kernel_horizontal, axis=-1), axis=-1)
    kernel_vertical = tf.expand_dims(tf.expand_dims(kernel_vertical, axis=-1), axis=-1)

    # perform the convolutions
    edge_hz = tf.nn.conv2d(img, kernel_horizontal, strides=[1,1,1,1], padding='SAME')
    edge_vc = tf.nn.conv2d(img, kernel_vertical, strides=[1,1,1,1], padding='SAME')

    # return tf.sqrt(tf.square(edge_hz) + tf.square(edge_vc))
    return edge_vc + edge_hz


def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = tf.keras.utils.load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_gray = tf.keras.preprocessing.image.img_to_array(img)
    img_gray = img_gray / 255.0
    img_gray_exp = tf.expand_dims(img_gray, 0)  # to add the batch size dimension (conv2D requires 4D tensors)
    print("The shape of the loaded image is: ", img_gray_exp.shape)
    return img_gray_exp


if __name__ == "__main__":
    img = load_and_preprocess_image("dataset/Au_ani_00058.jpg")

    # Apply the high pass filter and obtain the residual image
    res_img = high_pass_filter(img)
    plt.imshow(tf.squeeze(res_img).numpy())
    plt.show()

    # Invoke the DCU-Net
    model = DCUNet()

