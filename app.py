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

# Creating main page heading
st.title("\tSELAMAT DATANG DI APLIKASI WEBSITE DETEKSI KAPAL")
st.caption('Purwarupa perangkat lunak untuk mendeteksi kapal dari citra satelit yang dikembangkan oleh Program Studi Fisika, Fakultas MIPA, Universitas Jenderal Soedirman Purwokerto')
st.caption('Silakan unggah citra satelit Anda pada sidebar dan klik tombol "Deteksi Kapal" untuk melihat hasil pendeteksian kapal pada citra satelit')
st.divider()
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
    else:
        image = PIL.Image.open('./imgs/thumbnail.jpeg')

        st.caption("\n")
        st.image(image, caption='')
        #st.caption("Aminuddin, J., Abdullatif, R. F., Anggraini, E. I., Gumelar, S. F., & Rahmawati, A. (2023). Development of convolutional neural network algorithm on ships detection in Natuna Islands-Indonesia using land look satellite imagery. Remote Sensing Applications: Society and Environment, 32, 101025.")


try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Deteksi Kapal'):
    res = model.predict(uploaded_image,
                        conf=0.3,
                        verbose=False,
                        show_conf=False,
                        show_labels=False,
                        boxes=False
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
        st.download_button(label='Download Citra',data=data,file_name=f'ship_detection_{source_img.name.split(".")[0]}.jpg')
else:
    with col2:
        image = PIL.Image.open('./imgs/cnn.png')

        st.caption("\n")
        st.image(image, caption='Arsitektur CNN (Convolutional Neural Network')
