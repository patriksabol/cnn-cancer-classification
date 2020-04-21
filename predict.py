from model.define_model import get_model_for_feature_extraction
import os
import numpy as np
from keras import backend as K
import utils
import skimage.io
import gc


def extract_features(basenet,data_path):
    """
    extract features from all fold of all sets for the CFCMC classifier
    :param basenet:
    :param data_path:
    :return:
    """
    sets = ['train', 'valid', 'test']
    classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO',
               '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']

    # create folders for features
    for k in range(1, 11):
        for set_name in sets:
            for cl in classes:
                path = os.path.join(data_path,
                                    f"{basenet}_features",
                                    f"{k}_fold",
                                    set_name,
                                    cl)
                if not os.path.exists(path):
                    os.makedirs(path)
    for k in range(1, 11):
        model = get_model_for_feature_extraction(kfold=k, basenet=basenet, pathbase=pathbase)
        # par_model = multi_gpu_model(model, gpus=2)
        for set_name in sets:
            for cl in classes:
                print(f"Extracting from {k} fold from {set_name} set from class {cl}")
                save_path = os.path.join(data_path,
                                         f"{basenet}_features",
                                         f"{k}_fold",
                                         set_name,
                                         cl)
                batches = utils.batch_data(os.path.join(data_path, f"{k}_fold", set_name, cl), 16)
                images = []
                for id, files in enumerate(batches):
                    print("Detecting batches... " + str(round((id * 100) / len(batches), 2)) + " %", end="\r")
                    images = [skimage.io.imread(x) for x in files]
                    features = model.predict(np.array(images)/255, batch_size=16)
                    for _idx, feat in enumerate(features):
                        file_name = os.path.splitext(os.path.basename(files[_idx]))[0]
                        save_file_path = os.path.join(save_path, file_name + ".csv")
                        np.savetxt(save_file_path, features[_idx], delimiter=",")
        K.clear_session()
        del model
        gc.collect()


