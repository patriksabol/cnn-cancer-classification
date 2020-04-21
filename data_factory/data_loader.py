from keras.preprocessing.image import ImageDataGenerator
import config
from imutils import paths

def load_sets(trainPath, valPath):
    if config.AUGMENTATION:
        trainAug = ImageDataGenerator(rescale=1.0 / 255,
                                      rotation_range=20,
                                      width_shift_range=0.02,
                                      height_shift_range=0.02,
                                      horizontal_flip=True)
    else:
        trainAug = ImageDataGenerator(rescale=1.0 / 255)
    trainGen = trainAug.flow_from_directory(
        trainPath,
        class_mode="categorical",
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        color_mode="rgb",
        shuffle=True,
        batch_size=config.BATCH_SIZE
        )
    valAug = ImageDataGenerator(rescale=1.0 / 255)
    valGen = valAug.flow_from_directory(
        valPath,
        class_mode="categorical",
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        color_mode="rgb",
        shuffle=False,
        batch_size=config.BATCH_SIZE)

    totalTrain = len(list(paths.list_images(trainPath)))
    totalVal = len(list(paths.list_images(valPath)))

    return trainGen, valGen, totalTrain, totalVal


def load_test_set(testPath):
    testAug = ImageDataGenerator(rescale=1.0 / 255)
    testGen = testAug.flow_from_directory(
        testPath,
        class_mode="categorical",
        target_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
        color_mode="rgb",
        shuffle=False,
        batch_size=config.BATCH_SIZE)
    totalTest = len(list(paths.list_images(testPath)))

    return testGen, totalTest


