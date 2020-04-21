import glob
import os

def batch_data(data_path, batchsize, extension="*.tif"):
    """
    take all images in folder data_path and gather all images as small batches
    :param data_path: dir of the images
    :param config: config for detection
    :return: return list of batches
    """
    # print("Loading parsed patch from " + data_path)
    files = glob.glob(os.path.join(data_path, extension))
    # print("All loaded files " + str(len(files)))
    ALL_FILES = []
    _buffer = []
    for _idx, _file in enumerate(files):
        _buffer.append(_file)
        if len(_buffer) == batchsize:
            ALL_FILES.append(_buffer)
            _buffer = []

    if len(_buffer) > 0:
        if len(_buffer) < batchsize:
            while len(_buffer) < batchsize:
                _buffer.append(_buffer[-1])
            ALL_FILES.append(_buffer)
    # print("Number of batches: " + str(len(ALL_FILES)))
    count = 0
    for batch in ALL_FILES:
        for file in batch:
            count = count + 1
    # print("Number of all files: " + str(count))
    return ALL_FILES