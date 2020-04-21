import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
import glob
import os
from operator import itemgetter
import shutil
import config

"""
Script to prepare colorectal cancer data 
Kather_texture_2016_image_tiles_5000 available at https://zenodo.org/record/53169#.Xp8uu8gzaUk
Each of the 8 class (tissue type) is split into separate folder
Script generate 10 folds for cross-validation, training, validation and testing set.
"""

def generate_k_fold_cross_valid_idx(max_idx):
    """
    generate indicies for each of the fold
    :param max_idx: how many data you have for each class, colorectal have 625
    :return:
    """
    trains = []
    valids = []
    tests = []
    y = np.arange(max_idx)
    kf = KFold(n_splits=10)
    kf.get_n_splits(y)
    for train_index, test_index in kf.split(y):
        # print("TRAIN:", len(train_index), "TEST:", len(test_index))
        yval = np.arange(len(train_index))
        kf_val = ShuffleSplit(n_splits=1, test_size=0.15)
        kf_val.get_n_splits(yval)
        for train_idx, val_idx in kf_val.split(yval):
            final_train = train_index[train_idx]
            final_val = train_index[val_idx]
            final_test = test_index
            final_train.sort()
            final_val.sort()
            # print("TRAIN:", final_train, "VALID", final_val, "TEST:", final_test)
            trains.append(final_train)
            valids.append(final_val)
            tests.append(final_test)

    return trains, valids, tests

# names of the classes
classes = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO',
           '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']

# path to the folder with data
data_dir = config.DATA_PATH
cancer_dir = 'Kather_texture_2016_image_tiles_5000'

# create folders
for k in range(1, 11):
    for set in ['train', 'valid', 'test']:
        for cl in classes:
            path = os.path.join(data_dir,
                                f"{k}_fold",
                                set,
                                cl)
            if not os.path.exists(path):
                os.makedirs(path)

# generate indicies of the each fold for train, valid and test set
trainIdxs, validIdxs, testIdxs = generate_k_fold_cross_valid_idx(625)


# for each of the fold create folder and copy image to each folder
# folder look like this: /path/to/dir/N_fold/train-or-valid-or-test/class-name
for cl in classes:
    for k in range(0, 10):
        print(f"Creating {cl} in {k}-th fold")
        os.chdir(os.path.join(data_dir,cancer_dir,cl))
        classes_files = glob.glob('*.tif')
        # copy image to dir of the k-fold
        trainFiles = itemgetter(*trainIdxs[k])(classes_files)
        validFiles = itemgetter(*validIdxs[k])(classes_files)
        testFiles = itemgetter(*testIdxs[k])(classes_files)
        for trainF in trainFiles:
            shutil.copy(trainF, os.path.join(data_dir, f"{k+1}_fold", 'train', cl))
        for validF in validFiles:
            shutil.copy(validF, os.path.join(data_dir, f"{k+1}_fold", 'valid', cl))
        for testF in testFiles:
            shutil.copy(testF,  os.path.join(data_dir, f"{k+1}_fold", 'test', cl))


