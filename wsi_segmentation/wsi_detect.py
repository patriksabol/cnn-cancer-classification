from wsi_segmentation import parsing_utils as ParsUtil
import cv2
import os
from keras.models import load_model
from keras.models import Model

def detector_pipepline(file_path, prediction_model, feature_extractor_model):
    """
    pipeline of the tissue detector
    1. parse WSI with window size and step size
    2. detect each patch
    3. extract feature from each patch
    :param file_path:
    :param prediction_model:
    :param feature_extractor_model:
    :return:
    """
    # parse dir and filename
    dir_name, file_tail = os.path.split(file_path)
    filename = os.path.splitext(file_tail)[0]

    # save dir for parse patches
    parsed_image_dir = os.path.join(dir_name, filename)
    if not os.path.exists(parsed_image_dir):
        os.makedirs(parsed_image_dir)

    # open WSI
    wsi_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # parse WSI and save each patch to folder
    ParsUtil.parsing_patch_and_saving_images(wsi_image, window_size=100, step_size=50,
                                             output_path=parsed_image_dir, origin_filename=filename)

    prediction_dir = os.path.join(dir_name, filename+"_predictions")
    features_dir = os.path.join(dir_name, filename+"_features")

    # detect class of each patch and save array of probabilities to _predictions folder
    # extract feature from each patch and save to _features folder
    ParsUtil.detect_class_in_folder_and_save_detections(model=prediction_model,
                                                        fe_model=feature_extractor_model,
                                                        data_path=parsed_image_dir,
                                                        prediction_dir=prediction_dir,
                                                        features_dir=features_dir)

    # merge probabilities of the each patch to final probabilities array for explanation interface purpose
    # merge probabilities and compute final label map

    return prediction_dir


def merge_wsi_pipeline(patches_dir, save_probabilites_dir, wsi_name):
    """
    # merge probabilities of the each patch to final probabilities array for explanation interface purpose
    # merge probabilities and compute final label map
    :param patches_dir:
    :param save_probabilites_dir:
    :param wsi_name:
    :return:
    """
    prediction_dir = os.path.join(patches_dir, wsi_name + "_predictions")
    ParsUtil.merge_patches_to_wsi_probabilities(prediction_dir=prediction_dir,
                                                save_probabilites_dir=save_probabilites_dir,
                                                wsi_name=wsi_name)


def get_model_for_feature_extraction(path_model):
    """
    create from model model for feature extraction
    :param path_model:
    :return:
    """
    model = load_model(path_model)
    basemodel = model.layers[-2]
    flatten_layer = basemodel.layers[-2]
    input_basemodel = basemodel.layers[0]
    return Model(inputs=input_basemodel.input,
                 outputs=flatten_layer.output)


