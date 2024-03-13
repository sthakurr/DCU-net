import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50

from utils import *

def encoder1():
    '''
    This returns a VGG16 model for the purpose of encoding the input RGB image
    :return: an encoder object for the RGB image
    '''
    enc1 = VGG16(weights='imagenet', include_top=False)
    return enc1


def encoder2():
    '''
    This returns a ResNet model for the purpose of encoding the residual image
    :return: a ResNet model object for residual image
    '''
    enc2 = ResNet50(weights='imagenet', include_top=False)
    enc2_stride2 = set_pooling_stride(enc2)
    return enc2_stride2


def feat_fusion():
    '''
    This combines the outputs of the two encoders and combines them to get an input to the decoder
    :return:
    '''
