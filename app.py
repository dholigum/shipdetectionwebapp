# Import required libraries
import PIL

import streamlit as st
from ultralytics import YOLO
from io import BytesIO
import numpy as np

# Replace the relative path to your weight file
model_path = 'weights/ship_detection.pt'

# Setting page layout
st.set_page_config(
    page_title="Deteksi Kapal",  # Setting page title
    page_icon="🛰️",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Konfigurasi")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Unggah Citra...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Pilih Tingkat Confidence Model", 25, 100, 40)) / 100

# Creating main page heading
colT1,colT2 = st.columns([1,8])
with colT2:
    st.title("Selamat Datang di Aplikasi Website Deteksi Kapal")
    st.caption('Unggah citra satelit untuk mendeteksi kapal')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Citra Masukan",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Deteksi Kapal'):
    res = model.predict(uploaded_image,
                        conf=confidence
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Citra Luaran',
                 use_column_width=True
                 )
        
        pred_ar_int = res_plotted.astype(np.uint8)
        im = PIL.Image.fromarray(pred_ar_int)
        with BytesIO() as f:
            im.save(f, format='JPEG')
            data = f.getvalue()
        st.download_button(label='Download Citra',data=data,file_name='change_bg.jpg')
