import streamlit as st
import requests
from PIL import Image
import joblib
import numpy as np

# Function to download the model file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Download the model file from Google Drive
file_id = '1fZtoOlxfqT1J7AED4aS6z_NgpFUe-rtX'
destination = 'my_model.pth'
download_file_from_google_drive(file_id, destination)

# Function to load your custom model
def load_my_model(model_path):
    # Load the model from the specified path
    loaded_model = joblib.load(model_path)
    
    return loaded_model

# Function to load the model
def load_model(model_path):
    model = load_my_model(model_path)
    return model

# Function to preprocess user uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((200, 200))  # Resize image

    # Convert image to numpy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch of 1
    return img_array

# Streamlit UI
st.title('Image Classification')

model_path = 'my_model.pth'
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Preprocess uploaded image
    processed_image = preprocess_image(uploaded_file)

    predicted_label = model.predict(processed_image)

    # Display results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Display predicted label
    st.write('Predicted Label:', predicted_label)
