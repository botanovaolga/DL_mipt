import net
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import time


if __name__ == "__main__":

    model = net.Controller()
    model.load_model("models/model_latest")
    model.eval()


    img = cv2.resize(cv2.imread('/Users/botanovaolga/Desktop/mipt/Monocular_depth/test_img/ID_3353e4a3c.jpg'), (1216, 704))  # Must be multiple of 32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images as BGR by default
    # plt.imshow(img)
    # plt.show()

    prediction = model.predict(img, is_channels_first=False,
                               normalize=True)  # Dont forget to normalize images
    plt.imshow(prediction)
    plt.show()

    # visual_depth_map = model.depth_map_to_rgbimg(
    #     prediction)  # A helper method for converting depth maps into 3 channel, 8 byte images

    # plt.imshow(visual_depth_map)

