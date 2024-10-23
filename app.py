import os
import base64  # Import base64 for encoding images
import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv
import keras

# Load pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Load the Keras model
bmi_model = keras.models.load_model('my_epochfifty.h5')

def generate_human_mask(image):
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(input_batch)["out"][0]
    output_predictions = output.argmax(0)

    # Create a mask for the human class (label 15)
    human_mask = (output_predictions == 15).float()

    return human_mask

def set_background(jpg_file):
    """ Set the background image of the app. """
    with open(jpg_file, "rb") as f:
        bin_str = base64.b64encode(f.read()).decode()  # Encode image to base64
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    .header {{
        font-size: 36px; 
        color: white;
        background-color: rgba(76, 175, 80, 0.8);
        padding: 10px;
        text-align: center;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    # Set the page configuration first
    st.set_page_config(page_title="BMI Calculator with Human Masking", page_icon="🧍‍♂️", layout="wide")

    # Set the background image from the desktop
    desktop_image_path = os.path.expanduser("motivational-quote.3840x2160.mp4")  # Update with your correct image path
    set_background(desktop_image_path)

    # Add header with styling
    st.markdown('<div class="header">BMI Calculator with Human Masking</div>', unsafe_allow_html=True)

    st.write("---")  # Add a horizontal rule for separation

    # Sidebar with instructions
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload an image containing a person.")
    st.sidebar.write("2. The app will detect the human and calculate the BMI based on the segmented area.")

    # Image upload section
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format
        st.image(image, caption='Uploaded Image', use_column_width=25)  # Set width to 200 pixels

        # Generate the human mask
        human_mask = generate_human_mask(image)

        # Convert the PyTorch tensor to a NumPy array and resize it
        human_mask_np = human_mask.numpy()
        resized_human_mask = cv.resize(human_mask_np, (224, 224))
        resized_human_mask = np.expand_dims(resized_human_mask, axis=-1)
        resized_human_mask = np.repeat(resized_human_mask, 3, axis=-1)

        # Make predictions using the Keras model
        bmi = bmi_model.predict(np.expand_dims(resized_human_mask, axis=0))

        # Display results
        st.image(human_mask_np, caption='Generated Human Mask', use_column_width=25, clamp=True)  # Set width to 200 pixels

        # Use a colored metric display for BMI value
        st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>BMI: {bmi[0][0]:.2f}</h3>", unsafe_allow_html=True)

        # Display BMI categories with conditional formatting
        if bmi[0][0] < 18.5:
            st.markdown("<h4 style='color: #ffcc00;'>Underweight</h4>", unsafe_allow_html=True)
        elif 18.5 <= bmi[0][0] < 25:
            st.markdown("<h4 style='color: #00cc66;'>Normal</h4>", unsafe_allow_html=True)
        elif 25 <= bmi[0][0] < 30:
            st.markdown("<h4 style='color: #ffcc00;'>Overweight</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h4 style='color: #ff3300;'>Obese</h4>", unsafe_allow_html=True)

        # Add a button to download the human mask as an image
        mask_img = Image.fromarray((human_mask_np * 255).astype(np.uint8))
        st.download_button(
            label="Download Human Mask Image",
            data=mask_img.tobytes(),
            file_name="human_mask.png",
            mime="image/png"
        )
    else:
        st.markdown("<h4 style='color: red;'>Please upload an image to proceed.</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
