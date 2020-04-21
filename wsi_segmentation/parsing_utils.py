import os
from PIL import Image
import glob
import skimage.io
from skimage.transform import resize, rescale
import time
import cv2
from tifffile import imsave, imread
import numpy as np
import config


############# PARSOVANIE CELEHO PATCHU ######################
def parsing_patch_and_saving_images(patch_image, window_size, step_size,
                                    output_path, origin_filename):
    """
    Process one whole patch image alongside with its corresponding mask
    Window of the size window_size is moving through image with step step_size and save result to output_path
    :param patch_image: image of the RGB patch
    :param window_size:
    :param step_size:
    :param output_path: path of the RGB images saving dir
    :param origin_filename:
    """


    # check, whether output path exists, create if not
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)
    print("Parsing path " + output_path)

    # size of patch
    imgwidth, imgheight, _ = np.shape(patch_image)
    print_id = 0
    # generate left upper starting pixels
    columns = list(range(0, imgheight - window_size, step_size))
    columns.append(imgheight - window_size)
    rows = list(range(0, imgwidth - window_size, step_size))
    rows.append(imgwidth - window_size)
    # iteration cropped variable
    patches_id = 1
    crop_counts = len(columns) * len(rows)
    print(crop_counts)
    # cut slice from patch and mask
    for i in columns:
        for j in rows:
            # printing #
            print_id = print_id + 1
            if print_id % 100 == 0:
                print("Processing... " + str(round((print_id * 100) / crop_counts, 2)) + " %", end="\r")
            ############
            corrector_i = 0
            corrector_j = 0
            if i + window_size > imgheight:
                corrector_i = i - (imgheight - window_size)
            if j + window_size > imgwidth:
                corrector_j = j - (imgwidth - window_size)
            box = (j - corrector_j, i - corrector_i,
                   j - corrector_j + window_size, i - corrector_i + window_size)
            crop_RGB = patch_image[box[0]:box[2], box[1]:box[3], :]
            filename = str(patches_id)

            # saving images to dir
            save_crop_rgb_path = os.path.join(output_path,
                                     filename.zfill(5) + "_" + origin_filename +
                                     "_x_" + str(box[1]) + "_y_" + str(box[0]) + ".tif")
            cv2.imwrite(save_crop_rgb_path, crop_RGB)
            patches_id = patches_id + 1


#############################################

def batch_data(data_path):
    """
    take all images in folder data_path and gather all images as small batches
    :param data_path: dir of the images
    :return: return list of batches
    """
    print("Loading parsed patch from " + data_path)
    files = glob.glob(os.path.join(data_path, "*.tif"))
    print("All loaded files " + str(len(files)))
    ALL_FILES = []
    _buffer = []
    for _idx, _file in enumerate(files):
        _buffer.append(_file)
        if len(_buffer) == config.BATCH_SIZE:
            ALL_FILES.append(_buffer)
            _buffer = []

    if len(_buffer) > 0:
        if len(_buffer) < config.BATCH_SIZE:
            while len(_buffer) < config.BATCH_SIZE:
                _buffer.append(_buffer[-1])
            ALL_FILES.append(_buffer)
    print("Number of batches: " + str(len(ALL_FILES)))
    count = 0
    for batch in ALL_FILES:
        for file in batch:
            count = count + 1
    print("Number of all files: " + str(count))
    return ALL_FILES


def detect_class_in_folder_and_save_detections(model, fe_model, data_path, prediction_dir, features_dir):
    """
    detect class of the each patch and save prediciton array and features array
    :param model: model for classification
    :param fe_model: model for feature extraction
    :param data_path: path of the dir with patches of the wsi
    :param prediction_dir: save dir for prediction
    :param features_dir: save dir for features
    :return:
    """

    # detect each image and save to folder
    # check, whether output path exists, create if not
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    start = time.time()

    batches = batch_data(data_path)
    print_id = 0
    for files in batches:
        # printing #
        print_id = print_id + 1
        # if print_id % 100 == 0:
        print("Detecting batches... " + str(round((print_id * 100) / len(batches), 2)) + " %", end="\r")
        ############
        images = [cv2.resize(skimage.io.imread(x),(150,150),interpolation = cv2.INTER_AREA) for x in files]
        # for id, im in enumerate(images):
        #     print(im.shape)
        #     print(files[id])
        # array of the size 8 (tissue types)
        predictions = model.predict(np.array(images) / 255, batch_size=config.BATCH_SIZE)
        # array of the size 1024, features
        features = fe_model.predict(np.array(images) / 255, batch_size=config.BATCH_SIZE)
        for _idx, prediction in enumerate(predictions):
            IMAGE_NAME = os.path.splitext(os.path.basename(files[_idx]))[0]
            # print(prediction)
            np.save(os.path.join(prediction_dir, '%s.npy' % (IMAGE_NAME)), prediction)
        for _idx, feat in enumerate(features):
            IMAGE_NAME = os.path.splitext(os.path.basename(files[_idx]))[0]
            save_file_path = os.path.join(features_dir, '%s.csv' % (IMAGE_NAME))
            np.savetxt(save_file_path, feat, delimiter=",")
    end = time.time()
    print("[INFO] Detecting took {:.6f} seconds".format(end - start))


def merge_patches_to_wsi_probabilities(prediction_dir, save_probabilites_dir, wsi_name):
    """
    merge patches to probabilities array and extract label map
    :param prediction_dir: path to dir with predictions
    :param save_probabilites_dir: save dir to save probabilites
    :param wsi_name: name of the WSI
    :return:
    """
    wsi_prob = np.zeros((5000,5000,8))
    files = glob.glob(os.path.join(prediction_dir, "*.npy"))
    for file in files:
        IMAGE_NAME = os.path.splitext(os.path.basename(file))[0]
        splitted_name = IMAGE_NAME.split("_")
        x = int(splitted_name[splitted_name.index("x") + 1])
        y = int(splitted_name[splitted_name.index("y") + 1])
        prediction = np.load(file)
        row_count = 100
        column_count = 100
        # create array ith all values of prediction
        patch = np.full((100,100,8), prediction)
        max_patch = np.max(patch, axis=2)
        # crop from prob array
        prob_crop = wsi_prob[y:y + row_count, x:x + column_count, :]
        max_prob_crop = np.max(prob_crop, axis=2)
        # compare both arrays
        comp = max_patch > max_prob_crop
        # replace only values based on comp bool array
        prob_crop[comp] = patch[comp]
        # replace to the original one
        wsi_prob[y:y + row_count, x:x + column_count, :] = prob_crop
    save_prob_path = os.path.join(save_probabilites_dir, wsi_name+"_prob.tiff")

    imsave(save_prob_path, wsi_prob.astype('float16'))

    label_map_idx = np.argmax(wsi_prob, axis=2)
    cmap = np.array([[1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [1,0.71,0.75], [0.5,0.5,0.5], [0,0,0]])
    cmap = (cmap*255).astype('uint8')
    label_map = cmap[label_map_idx]
    label_map = np.flip(label_map, axis=2)
    save_label_map_path = os.path.join(save_probabilites_dir, wsi_name+"_label_map_CNN.png")
    cv2.imwrite(save_label_map_path, label_map)






