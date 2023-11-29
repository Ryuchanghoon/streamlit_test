import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt



# 앱 제목
st.title('이미지 변환')

# 이미지 업로드
uploaded_file = st.file_uploader("이미지 파일 업로드.....", type=["jpg", "jpeg", "png"])

# 이미지 처리 함수
def process_image(image):
    #  YCrCb 컬러 스페이스 변환
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Y 채널 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #이미지 퀄리티 향상 위한 CLAHE
    Y_channel, Cr, Cb = cv2.split(ycrcb_image)
    Y_channel = clahe.apply(Y_channel)
    # 변경된 Y 채널 다시 YCrCb 병합
    merged_ycrcb = cv2.merge([Y_channel, Cr, Cb])
    # YCrCb에서 BGR 컬러 스페이스 변환
    final_image = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
    # streamlit은 RGB형태로 이미지 불어들여옴. 그래서 억지로. 밑에.
    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return rgb_image



def convert_image_to_grayscale(image):
    # 흑백 이미지 변환
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def flip_image(image):
    flipped_image = cv2.flip(image, 0)
    rgb_flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)

    return rgb_flipped_image


def plot_histograms(original_image, processed_image):
    # 히스토그램 위한 YCrCb 분리.
    Y_original, Cr_original, Cb_original = cv2.split(cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb))
    Y_processed, Cr_processed, Cb_processed = cv2.split(cv2.cvtColor(processed_image, cv2.COLOR_BGR2YCrCb))

    channels = ('Y', 'Cr', 'Cb')

    fig, axs = plt.subplots(2, 3, figsize=(16, 6))

    # 원본 이미지 히스토그램
    for i, channel in enumerate([Y_original, Cr_original, Cb_original]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[0, i].plot(histogram)
        axs[0, i].set_xlim([0, 256])
        axs[0, i].set_title(f'Original {channels[i]} Histogram')

    # 처리된 이미지 히스토그램
    for i, channel in enumerate([Y_processed, Cr_processed, Cb_processed]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[1, i].plot(histogram)
        axs[1, i].set_xlim([0, 256])
        axs[1, i].set_title(f'Convert {channels[i]} Histogram')

    return fig



# 이미지 처리 선택
if uploaded_file is not None:
    # 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 원본 이미지 표시
    st.image(uploaded_file, caption='Original Image', use_column_width=True)

    # 이미지 옵션 선택
    option = st.selectbox(
        '원하는 변환을 선택하세요:',
        ('None', 'Histogram Equalization', '흑백변환', '90도 회전')
    )

    # 이미지 표시
    if option == 'Histogram Equalization':
        processed_image = process_image(image)
        st.image(processed_image, caption='Histogram Equalized Image', use_column_width=True)
        st.pyplot(plot_histograms(image, processed_image))

    elif option == '흑백변환':
        grayscale_image = convert_image_to_grayscale(image)
        st.image(grayscale_image, caption='Grayscale Image', use_column_width=True)

    elif option =='90도 회전':
        rgb_flipped_image = flip_image(image)
        st.image(rgb_flipped_image, caption = 'Flipped Image', use_column_width = True)