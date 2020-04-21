import efficientnet.keras as efn
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.layers import Input
import config


def load_efficientnet():
    model = efn.EfficientNetB0(weights='imagenet',
                               include_top=False,
                               input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                                         config.IMAGE_WIDTH, 3)))
    return model


def load_resnet50():
    model = ResNet50(weights='imagenet',
                     include_top=False,
                     input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                               config.IMAGE_WIDTH,
                                               3)))

    return model


def load_vgg16():
    model = VGG16(weights='imagenet',
                  include_top=False,
                  input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                            config.IMAGE_WIDTH,
                                            3)))

    return model

def load_inceptionv3():
    model = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                            config.IMAGE_WIDTH,
                                            3)))

    return model

def load_inception_resnet():
    model = InceptionResNetV2(weights='imagenet',
                        include_top=False,
                        input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                                  config.IMAGE_WIDTH,
                                                  3)))

    return model

def load_xception():
    model = Xception(weights='imagenet',
                              include_top=False,
                              input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                                        config.IMAGE_WIDTH,
                                                        3)))

    return model

def load_densenet():
    model = DenseNet121(weights='imagenet',
                              include_top=False,
                              input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                                        config.IMAGE_WIDTH,
                                                        3)))

    return model




def load_resnet50_without_weights():
    model = ResNet50(weights=None,
                     include_top=False,
                     input_tensor=Input(shape=(config.IMAGE_HEIGHT,
                                               config.IMAGE_WIDTH,
                                               3)))

    return model



