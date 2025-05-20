import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_Thread.keras')

model = load_model()

# Class labels
class_names = ['Breakage', 'Knot', 'Normal', 'Unevenness']

# Preprocessing function
def preprocess_image(img_data):
    img = Image.open(img_data).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array, img

# Streamlit interface
st.title("Thread Defect Classifier")
st.write("Upload or capture an image to classify the thread defect.")

option = st.radio("Choose input method:", ["Upload Image", "Capture from Webcam"])

img_data = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_data = uploaded_file

elif option == "Capture from Webcam":
    captured_image = st.camera_input("Take a picture")
    if captured_image:
        img_data = io.BytesIO(captured_image.getvalue())

if img_data:
    img_array, display_img = preprocess_image(img_data)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    # Display image and result
    st.image(display_img, caption="Input Image", use_column_width=True)
    st.success(f"Predicted class: **{predicted_class_name}**")
    st.info(f"Confidence: {confidence:.2%}")
