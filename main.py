import train
from keras.models import load_model
import config
import tensorflow as tf
import os

mode = 'train'

if mode == 'train':
    train.training_and_evaluating_pipeline('resnet-like')

if mode == 'wsi':
    from wsi_segmentation import wsi_detect

    # in case you need to set memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

    k = 1  # k-fold
    basenet = "Xception"
    model_path = f"{config.SAVE_MODEL_PATH}{k}_fold/colorectal_{k}_fold_{basenet}_model.h5"

    prediction_model = load_model(model_path)
    feature_extractor_model = wsi_detect.get_model_for_feature_extraction(model_path)

    dir_of_wsi = "/dir/to/wsi/"
    wsi_name = "CRC-Prim-HE-07_APPLICATION"

    file_path = os.path.join(dir_of_wsi, wsi_name + ".tif")

    wsi_detect.detector_pipepline(file_path=file_path,
                                  prediction_model=prediction_model,
                                  feature_extractor_model=feature_extractor_model)

    wsi_detect.merge_wsi_pipeline(patches_dir=dir_of_wsi,
                                  save_probabilites_dir=dir_of_wsi,
                                  wsi_name=wsi_name)

if mode == "feature_extract_from_set":
    # for extracting features from each of the fold of the sets for the CFCMC classifier
    import predict
    pathbase = config.DATA_PATH
    predict.extract_features("resnet-like", pathbase)
