from keras.models import Model
from keras import models
from keras import layers
from model import base_model_loader as bml
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, DepthwiseConv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import add
from keras.layers import concatenate
import config

def create_with_pretrained_model(baseModelName):
    if baseModelName == 'VGG16':
        baseModel = bml.load_vgg16()
    elif baseModelName == 'Resnet50':
        baseModel = bml.load_resnet50()
    elif baseModelName == 'InceptionV3':
        baseModel = bml.load_inceptionv3()
    elif baseModelName == 'InceptionResnetV2':
        baseModel = bml.load_inceptionv3()
    elif baseModelName == 'Xception':
        baseModel = bml.load_xception()
    elif baseModelName == 'Densenet':
        baseModel = bml.load_densenet()
    elif baseModelName == 'Resnet50_not_pretrained':
        baseModel = bml.load_resnet50_without_weights()
    else:
        baseModel = bml.load_efficientnet()

    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(1024, activation="relu")(headModel)
    if config.CLASSES == 2:
        headModel = Dense(1, activation="sigmoid")(headModel)
    else:
        headModel = Dense(config.CLASSES, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)
    return model, baseModel, headModel


def get_model_for_feature_extraction(kfold, basenet, pathbase):
    path = f"{pathbase}/{kfold}_fold/"
    path_model = path + f"colorectal_{kfold}_fold_{basenet}_model.h5"
    model = load_model(path_model)
    if basenet == 'resnet-like':
        basemodel = model.layers[-2]
        return Model(inputs = model.input,
                     outputs=basemodel.output)
    basemodel = model.layers[-2]
    flatten_layer = basemodel.layers[-2]
    input_basemodel = basemodel.layers[0]
    return Model(inputs=input_basemodel.input,
                 outputs=flatten_layer.output)


def create_model_vgg_like():
    custom_vgg = models.Sequential()
    custom_vgg.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu",
                          input_shape=(150, 150, 3)))
    custom_vgg.add(Conv2D(32, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    custom_vgg.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    custom_vgg.add(MaxPooling2D((2, 2)))

    # custom_vgg.add(Conv2D(512, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Conv2D(512, (3, 3), strides=1, padding="same", activation="relu"))
    # custom_vgg.add(Dropout(0.4))
    # custom_vgg.add(MaxPooling2D((2, 2)))

    custom_vgg.add(Flatten())
    custom_vgg.add(Dense(512, activation="relu"))
    custom_vgg.add(Dense(config.CLASSES, activation="softmax"))

    custom_vgg.summary()

    return custom_vgg


def resnet_like(depth, num_classes=config.CLASSES):
    from keras.regularizers import l2
    import keras
    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=(150,150,3))
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    y = Dense(512, activation='relu')(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



def densenet_like():
    def dense_layer(x, layer_configs):
        layers = []
        for i in range(2):
            if layer_configs[i]["layer_type"] == "Conv2D":
                layer = Conv2D(layer_configs[i]["filters"], layer_configs[i]["kernel_size"],
                               strides=layer_configs[i]["strides"], padding=layer_configs[i]["padding"],
                               activation=layer_configs[i]["activation"])(x)
                layers.append(layer)
        for n in range(2, len(layer_configs)):
            if layer_configs[n]["layer_type"] == "Conv2D":
                layer = Conv2D(layer_configs[n]["filters"], layer_configs[n]["kernel_size"],
                               strides=layer_configs[n]["strides"], padding=layer_configs[n]["padding"],
                               activation=layer_configs[n]["activation"])(concatenate(layers, axis=3))
                layers.append(layer)
        return layers

    layer_f8 = [
        {
            "layer_type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 8, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }
    ]

    layer_f16 = [
        {
            "layer_type": "Conv2D", "filters": 16, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 16, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 16, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 16, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 16, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }
    ]

    layer_f32 = [
        {
            "layer_type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 32, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }
    ]

    layer_f64 = [
        {
            "layer_type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 64, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }
    ]

    layer_f128 = [
        {
            "layer_type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }, {
            "layer_type": "Conv2D", "filters": 128, "kernel_size": (3, 3), "strides": 1, "padding": "same",
            "activation": "relu"
        }
    ]
    inp = Input(shape=(150, 150, 3))
    x = inp
    x = Conv2D(4, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = dense_layer(x, layer_f8)
    x = Dropout(0.4)(x)

    x = BatchNormalization(axis=3)(x)
    x = dense_layer(x, layer_f16)
    x = Dropout(0.4)(x)

    x = BatchNormalization(axis=3)(x)
    x = dense_layer(x, layer_f32)
    x = Dropout(0.4)(x)

    x = BatchNormalization(axis=3)(x)
    x = dense_layer(x, layer_f64)
    x = Dropout(0.4)(x)

    x = BatchNormalization(axis=3)(x)
    x = dense_layer(x, layer_f128)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(96, (1, 1), activation="relu")(x)
    x = BatchNormalization(axis=3)(x)

    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Flatten()(x)

    x = Dropout(0.4)(x)
    x = Dense(14, activation="softmax")(x)

    dense_net = Model(inp, x)
    dense_net.summary()

    return dense_net


def inception_like():
    def inception_layer(x, layer_configs):
        layers = []
        for configs in layer_configs:
            if configs[0]["layer_type"] == "Conv2D":
                layer = Conv2D(configs[0]["filters"], configs[0]["kernel_size"], strides=configs[0]["strides"],
                               padding=configs[0]["padding"], activation=configs[0]["activation"])(x)
            if configs[0]["layer_type"] == "MaxPooling2D":
                layer = MaxPooling2D(configs[0]["kernel_size"], strides=configs[0]["strides"],
                                     padding=configs[0]["padding"])(x)
            for n in range(1, len(configs)):
                if configs[n]["layer_type"] == "Conv2D":
                    layer = Conv2D(configs[n]["filters"], configs[n]["kernel_size"], strides=configs[n]["strides"],
                                   padding=configs[n]["padding"], activation=configs[n]["activation"])(layer)
                if configs[n]["layer_type"] == "MaxPooling2D":
                    layer = MaxPooling2D(configs[n]["kernel_size"], strides=configs[n]["strides"],
                                         padding=configs[n]["padding"])(layer)
            layers.append(layer)
        return concatenate(layers, axis=3)

    from model import inception_config as ic

    inp = Input(shape=(150, 150, 3))
    x = inp
    x = Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(x)
    x = MaxPooling2D((3, 3), padding="same", strides=2)(x)
    x = Conv2D(64, (1, 1), strides=1, padding="same", activation="relu")(x)
    x = Conv2D(192, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = MaxPooling2D((3, 3), padding="same", strides=2)(x)
    x = inception_layer(x, ic.layer_3a)
    x = inception_layer(x, ic.layer_3b)
    x = MaxPooling2D((3, 3), padding="same", strides=2)(x)
    x = inception_layer(x, ic.layer_4a)

    x1 = AveragePooling2D((2, 2), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding="same", activation="relu")(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(config.CLASSES, activation="softmax")(x1)

    inc = Model(inp, x1)
    inc.summary()
    return inc
