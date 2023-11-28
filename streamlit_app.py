import streamlit as st
import torch
import requests
from PIL import Image
import numpy as np

# Function to load the PyTorch model
@st.cache(allow_output_mutation=True)
def load_model_from_url(url):
    response = requests.get(url)
    model_path = '/tmp/my_model.pth'
    with open(model_path, 'wb') as f:
        f.write(response.content)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    return model

# Function to preprocess user uploaded image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((200, 200))  # Resize image

    # Convert image to numpy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch of 1
    return img_array

# Set threshold for minimum prediction confidence
threshold = 0.5

# Streamlit UI
st.title('Image Classification')

# URL to your model file
model_url = 'https://drive.google.com/file/d/1fZtoOlxfqT1J7AED4aS6z_NgpFUe-rtX/view?usp=sharing'

# Load the PyTorch model from the URL
model = load_model_from_url(model_url)

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Preprocess uploaded image
    processed_image = preprocess_image(uploaded_file)

    # Get predictions
    with torch.no_grad():
        predictions = model(torch.tensor(processed_image).permute(0, 3, 1, 2).float())

    # Process prediction
    predicted_label = torch.argmax(predictions, dim=1).item()
    confidence = torch.softmax(predictions, dim=1)[0][predicted_label].item()

    # Display results
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Check if prediction confidence meets threshold
    if confidence < threshold:
        st.write('The uploaded image is not related to the task.')
    else:
        st.write('Predicted Label:', predicted_label)
