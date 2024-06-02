import streamlit as st
import cv2
from PIL import Image
import numpy as np
import ultralytics


def main():
    st.title("Black and White Image Converter")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        avg_brightness = np.mean(l_channel)
        st.text(f"Avg brightness {avg_brightness}")
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        threshold = st.slider("Set Threshold", min_value=0, max_value=255, value=128)

        # Apply thresholding
        _, black_white_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        st.image(black_white_image, caption='black and white', use_column_width=True)
        contours, _ = cv2.findContours(black_white_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        black_contours = []
        for i, contour in enumerate(sorted_contours[:-1]):
            if cv2.contourArea(contour) > 1000:
                black_contours.append(contour)
                out_im = np.zeros_like(black_white_image)
                cv2.drawContours(out_im, black_contours, -1, (255, 255, 255), -1)
                st.image(out_im, caption=f'contour {i}', use_column_width=True)
        out_im = np.zeros_like(black_white_image)
        cv2.drawContours(out_im, black_contours, -1, (255, 255, 255), -1)
        st.image(out_im, caption='white Internal', use_column_width=True)
        threshold2 = st.slider("Set Threshold for external", min_value=0, max_value=255, value=128)
        _, black_white_image2 = cv2.threshold(gray_image, threshold2, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(black_white_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_im_external = np.zeros_like(black_white_image)
        cv2.drawContours(out_im_external, contours, -1, (255, 255, 255), -1)
        st.image(out_im_external, caption='white external', use_column_width=True)
        st.image(out_im_external - out_im, caption='white difference', use_column_width=True)
        black_image= out_im_external - out_im
        contours, _ = cv2.findContours(black_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_im_external = np.zeros_like(black_white_image)
        cv2.drawContours(out_im_external, contours, -1, (255, 255, 255), -1)
        st.image(out_im_external, caption='black external', use_column_width=True)
        contours, _ = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        out_im_external = np.ones_like(black_white_image)*255
        cv2.drawContours(out_im_external, contours, -1, (0, 0, 0), -1)
        st.image(out_im_external, caption='black internal', use_column_width=True)
if __name__ == '__main__':
    main()