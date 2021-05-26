import net
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import streamlit as st
from PIL import Image
import time


st.title("Monocular depth estimation")

# загрузка картинки
img = st.file_uploader("Upload a file", type=["jpg", "png"])

model = net.BtsController()
model.load_model("models/model_latest")
model.eval()

if __name__ == "__main__":
    if img:
        # model = BTS.BtsController()
        # model.load_model("models/model_latest")
        # model.eval()

        #отрисуем картинку на стримлите
        st.write("###  My image:")
        image = Image.open(img)
        st.image(image, width=500)

        #запускаем на стримлите
        clicked_button = st.button("Model")

        if clicked_button:

            img = cv2.resize(np.array(image), (1216, 704))  # Must be multiple of 32
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images as BGR by default
            # plt.imshow(img)
            # plt.show()

            with st.spinner('Wait for it...'):
                prediction = model.predict(img, is_channels_first=False, normalize=True)  # Dont forget to normalize images
                # plt.imshow(prediction)
                # plt.show()
            st.success('Done!')

            
            visual_depth_map = model.depth_map_to_rgbimg(
                prediction)  # A helper method for converting depth maps into 3 channel, 8 byte images
            # # images 3 channel, uint8 format are displayed without any scaling / normalization
            # plt.imshow(visual_depth_map)

            #результат на стримлит
            st.write("###  Result:")
            st.image(visual_depth_map, width=500)
