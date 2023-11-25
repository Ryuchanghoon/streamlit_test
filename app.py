import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Streamlit 앱 제목
st.title('Historgram Equilization 이미지')

# 이미지 업로드
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def process_image(image):
    # 이미지 읽기 및 처리
    # BGR에서 YCrCb 컬러 스페이스로 변환
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Y 채널에 대해 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y_channel, Cr, Cb = cv2.split(ycrcb_image)
    Y_channel = clahe.apply(Y_channel)

    # 변경된 Y 채널을 다시 YCrCb 이미지와 병합
    merged_ycrcb = cv2.merge([Y_channel, Cr, Cb])

    # YCrCb에서 BGR 컬러 스페이스로 변환
    final_image = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    return final_image

def plot_histograms(image):
    # 각 채널 분리
    Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))

    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y_clahe = clahe.apply(Y)

    # CLAHE 적용된 이미지를 YCrCb로 병합하고, BGR로 변환
    ycrcb_clahe = cv2.merge([Y_clahe, Cr, Cb])
    final_image = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    # 히스토그램 계산 및 시각화
    channels = ('Y', 'Cr', 'Cb')

    fig, axs = plt.subplots(2, 3, figsize=(16, 6))

    # 원본 YCrCb 채널의 히스토그램
    for i, channel in enumerate([Y, Cr, Cb]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[0, i].plot(histogram)
        axs[0, i].set_xlim([0, 256])
        axs[0, i].set_title(f'Original {channels[i]} Histogram')

    # CLAHE 적용 후 YCrCb 채널의 히스토그램
    Y_clahe, Cr_clahe, Cb_clahe = cv2.split(ycrcb_clahe)
    for i, channel in enumerate([Y_clahe, Cr_clahe, Cb_clahe]):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        axs[1, i].plot(histogram)
        axs[1, i].set_xlim([0, 256])
        axs[1, i].set_title(f'CLAHE {channels[i]} Histogram')

    return fig

if uploaded_file is not None:
    # 이미지 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(uploaded_file, caption = 'Original Image', use_column_width = True)
    # 이미지 처리
    
    if st.button('평활화 적용'):
        processed_image = process_image(image)

        # 원본 이미지와 처리된 이미지 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption='Convert YCrCb', use_column_width=True)
        with col2:
            st.image(processed_image, caption='CLAHE Image', use_column_width=True)

        # 히스토그램 표시
        st.pyplot(plot_histograms(image))