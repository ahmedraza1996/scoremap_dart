import streamlit as st
import cv2
from PIL import Image
import numpy as np
import ultralytics

def black_and_white(image, threshold):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, black_white_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    return black_white_image


def find_and_draw_contours(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
    black_contours = []
    for i, contour in enumerate(sorted_contours[:-2]):
        if cv2.contourArea(contour) > 1000:
            black_contours.append(contour)

    contours_ext, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside_contours = []
    for contour in black_contours:
        match = False
        for c2 in contours_ext:
            if np.array_equal(contour, c2):
                match = True
                break
        if not match:
            inside_contours.append(contour)
    sorted_contours_rem = sorted(inside_contours, key=lambda c: cv2.contourArea(c), reverse=False)
    black_contours = []
    for i, contour in enumerate(sorted_contours_rem):
        if cv2.contourArea(contour) > 1300:
            black_contours.append(contour)

    # Create a black image to draw contours on
    contour_image = np.ones_like(image)*255

    # Draw contours
    cv2.drawContours(contour_image, black_contours, -1, (0, 0, 0), -1)

    return contour_image


def find_white_contours_external(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
    white_contours = []
    for c in contours:
        #if cv2.contourArea(c) >1500:
        white_contours.append(c)
    contour_image = np.zeros_like(image)

    # Draw contours
    cv2.drawContours(contour_image, white_contours, -1, (255, 255, 255), -1)
    contour_image = cv2.GaussianBlur(contour_image, (3, 3), 0)
    return contour_image , white_contours


def find_white_contours_internal(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
    white_contours = []
    for c in contours:
        #if cv2.contourArea(c) >1500:
        white_contours.append(c)
    contour_image = np.zeros_like(image)

    # Draw contours
    cv2.drawContours(contour_image, white_contours, -1, (255, 255, 255), -1)
    contour_image = cv2.GaussianBlur(contour_image, (3, 3), 0)
    return contour_image , white_contours
def draw_white_contour(image):
    return contour_image

def invert_image(image):
    return 255 - image

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


        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY )
        # histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        # histogram /= histogram.sum()
        #
        # # Create a blank canvas to draw the histogram
        # hist_w = 512
        # hist_h = 400
        # bin_w = int(round(hist_w / 256))
        # hist_image = np.zeros((hist_h, hist_w), dtype=np.uint8)

        # Draw the histogram
        # for i in range(256):
        #     cv2.line(hist_image, (bin_w * (i), hist_h),
        #              (bin_w * (i), hist_h - int(histogram[i] * hist_h)),
        #              (255), thickness=1)
        #
        # st.image(hist_image, caption='Color distribution', use_column_width=False)
        # Threshold slider
        threshold = st.slider("Set Threshold", min_value=0, max_value=255, value=128)

        # Convert to OpenCV format
        opencv_image = np.array(image)
        opencv_image = opencv_image[:, :, ::-1].copy()

        # Convert to black and white
        bw_image = black_and_white(opencv_image, threshold)

        # Display black and white image
        st.image(bw_image, caption='Black and White Image', use_column_width=True)
        contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        black_contours = []
        for i, contour in enumerate(sorted_contours[:]):
            if cv2.contourArea(contour) > 1000:
                black_contours.append(contour)
        out_im = np.zeros_like(bw_image)
        cv2.drawContours(out_im, black_contours, -1, (255, 255, 255), -1)
        st.image(out_im, caption='Black Internal', use_column_width=True)

        for_black = cv2.bitwise_not(out_im)
        st.image(for_black, caption='White Internal', use_column_width=True)
        # Find and draw contours
        out_im= np.zeros_like(bw_image)
        contour_image = find_and_draw_contours(bw_image)

        # Display contours
        st.image(contour_image, caption='Black Contours', use_column_width=True)
        threshold_white = st.slider("Set White Threshold", min_value=0, max_value=255, value=128)
        inverse_image = cv2.bitwise_not(opencv_image)
        bw_image = black_and_white(inverse_image, threshold_white)

        st.image(bw_image, caption='Black and White Image 2', use_column_width=True)
        white_contours_image_ext , white_contour_ext= find_white_contours_external(bw_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(white_contours_image_ext, caption='White Contours External', use_column_width=True)
        white_contours_image_int , white_contour_internal = find_white_contours_internal(bw_image)
        with col2:
            st.image(white_contours_image_int, caption='White Contours Internal', use_column_width=True)

        inside_contours = []
        for contour in white_contour_internal:
            match = False
            for c2 in white_contour_ext[:]:
                if np.array_equal(contour, c2):
                    match = True
                    break
            if not match:
                inside_contours.append(contour)
        print(len(inside_contours))
        sorted_contours_rem = sorted(inside_contours, key=lambda c: cv2.contourArea(c), reverse=False)
        white_contours = []
        for i, contour in enumerate(inside_contours):
            if cv2.contourArea(contour) > 1500:
                white_contours.append(contour)
        contour_image = np.zeros_like(bw_image)
        inside_contours = []
        for contour in white_contours:
            match = False
            for c2 in white_contour_ext[:]:
                if np.array_equal(contour, c2):
                    match = True
                    break
            if not match:
                inside_contours.append(contour)

        # Draw contours
        cv2.drawContours(contour_image, inside_contours, -1, (255, 255, 255), -1)
        st.image(contour_image, caption='White Contours filtered A', use_column_width=True)

if __name__ == '__main__':
    main()