from model import define_model as DM
from keras.optimizers import Adam
import config
from data_factory.data_loader import load_sets
from keras import backend as K
from model import callbacks
import gc
from data_factory.data_loader import load_test_set
import cv2
import numpy as np
from keras.models import load_model
import config
from keras import backend as K
import os
import tensorflow as tf


def train_model_with_base(basenet, k_fold):
    import neptune
    neptune.init('buco24/cancer-cnn',
                 api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwa'
                           'V91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzY'
                           '5OTFmNDQtNjRkMS00NDgzLWJjYjUtYTc5Zjk1NzA0MDNhIn0=')
    PARAMS = {'batch_size': config.BATCH_SIZE,
              'epochs': config.EPOCHS,
              'augmentation': config.AUGMENTATION}
    neptune.create_experiment(name=f"{basenet}-{k_fold}-fold", params=PARAMS)
    ################ INITIALIZATION ###############################3
    trainGen, valGen, totalTrain, totalVal = load_sets(config.TRAIN_SET_PATH, config.VALID_SET_PATH)
    if basenet == 'vgg-like':
        model_base = DM.create_model_vgg_like()
    elif basenet == 'resnet-like':
        model_base = DM.resnet_like(20)
    else:
        model_base,_,_ = DM.create_with_pretrained_model(basenet)
    # model = multi_gpu_model(model_base, gpus=2)
    model = model_base

    callbacks_train = callbacks.get_callbacks(config.SAVE_MODEL_PATH)
    # model = model_base
    print("[INFO] compiling model...")

    opt = Adam(lr=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["acc"])
    print("[INFO] training...")

    history = model.fit_generator(
        trainGen,
        steps_per_epoch=totalTrain // config.BATCH_SIZE,
        validation_data=valGen,
        validation_steps=totalVal // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks_train,
        use_multiprocessing=True, workers=8)
    K.clear_session()
    del model
    gc.collect()


def training_and_evaluating_pipeline(basenet):
    """
    train cnn
    :param basenet: name of the base net
    :return:
    """
    pathbase = config.DATA_PATH

    # in case you need to set memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

    # for each of the 10 fold
    for k in range(1, 11):
        # set base path for the fold
        path = f"{pathbase}{k}_fold/"

        # set paths
        config.TRAIN_SET_PATH = path + "train"
        config.VALID_SET_PATH = path + "valid"
        config.TEST_SET_PATH = path + "test"
        config.SAVE_MODEL_PATH = path + f"colorectal_{k}_fold_{basenet}_model.h5"
        train_model_with_base(basenet, k)

    # evaluate fold with testing set
    # create file to write accuracy
    acc_path = f"{pathbase}performance_{basenet}.txt"
    f = open(acc_path, 'w')
    accs = []

    import gc
    for k in range(1, 11):
        path = f"{pathbase}{k}_fold/"
        path_model = path + f"colorectal_{k}_fold_{basenet}_model.h5"
        config.TEST_SET_PATH = path + "test"
        # load model
        model = load_model(path_model)
        # load test set
        test_gen, _ = load_test_set(config.TEST_SET_PATH)

        # evaluate model
        loss, acc = model.evaluate_generator(test_gen, verbose=1, steps=100)
        accs.append(acc)
        print(f"Accuracy of the {k} fold is {acc}\n")
        f.write(f"Accuracy of the {k} fold is {acc}\n")
        K.clear_session()
        del model
        gc.collect()
    accs = np.array(accs)
    f.write(f"Average acc is {accs.mean()}+-{accs.std()}")
    f.close()





