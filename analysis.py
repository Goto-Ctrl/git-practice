import numpy as np


def calculate_channel_averages(image):
    image_array = np.asarray(image)
    return image_array.shape, image_array.mean(axis=(0, 1))


def calculate_channel_histograms(image):
    image_array = np.asarray(image)
    return {
        "Red": np.histogram(image_array[:, :, 0], bins=256, range=(0, 256))[0],
        "Green": np.histogram(image_array[:, :, 1], bins=256, range=(0, 256))[0],
        "Blue": np.histogram(image_array[:, :, 2], bins=256, range=(0, 256))[0],
    }