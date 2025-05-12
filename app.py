import streamlit as st
import torch
import numpy as np  # Corrected import statement
import os
import cv2
import tempfile
import pandas as pd
import altair as alt
import re  # Import the re module for regular expressions
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Set page configuration
st.set_page_config(layout="wide", page_title="Action Recognition")

# Sidebar
st.sidebar.write("## Upload and Process Video 🎥")
uploaded_file = st.sidebar.file_uploader("Upload a video file:", type=["mp4", "avi", "mov"])

# Sidebar Information
with st.sidebar.expander("ℹ️ Video Guidelines"):
    st.write("""
    - Supported formats: MP4, AVI, MOV
    - Ensure the video contains clear actions for better predictions
    """)

# Load model
@st.cache_resource
def load_model():
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    return model, extractor

model, extractor = load_model()
model.eval()

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_folder, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    frame_count = 0
    saved_frames = 0
    while cap.isOpened() and saved_frames < num_frames:  # Corrected method name
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frames + 1:04d}.jpg")
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        frame_count += 1
    cap.release()

# Main Layout
st.write("## Action Recognition App")
st.write("Upload a video to predict the action using a pre-trained model.")

# Introduction
st.write("""
This app allows you to upload a video, converts it into frames, and predicts the action using a pre-trained model.
We use **TimeSformer**, a state-of-the-art video transformer model, which processes video frames as a sequence of images and captures temporal relationships to predict actions effectively.
Experience seamless action recognition with visualizations and confidence scores.
""")

# Two-column layout
col1, col2 = st.columns(2)

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded video
        col1.write("### Uploaded Video")
        col1.video(video_path)

        # Extract frames from the video
        st.info("Extracting frames from the video...")
        extract_frames_from_video(video_path, temp_dir, num_frames=8)
        folder_path = temp_dir

        # Process the extracted frames
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])[:8]
        frames = []

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            frames.append(frame)

        if len(frames) < 8:
            st.warning("The video must contain enough frames to extract 8 frames.")
        else:
            inputs = extractor([frames], return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_prob, top_index = torch.max(probs, dim=-1)

            # Display the single top prediction
            col2.write("### Predicted Action")
            action_label = model.config.id2label[top_index.item()]
            confidence = top_prob.item() * 100
            col2.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
                    <h2 style="font-size: 24px; color: #4CAF50;">{action_label}</h2>
                    <p style="font-size: 16px; color: #777;">Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Generate heatmaps for visualization
            heatmaps = []
            for idx, frame in enumerate(frames):
                # Create a random heatmap for demonstration purposes
                heatmap = np.zeros((224, 224), dtype=np.uint8)

                # Highlight a specific region in each frame (e.g., a moving region)
                center_x, center_y = 112 + (idx * 10) % 50, 112 + (idx * 10) % 50  # Varying center for each frame
                cv2.circle(heatmap, (center_x, center_y), 50, (255), -1)  # Draw a filled circle

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Use COLORMAP_JET for vibrant colors
                overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)  # Overlay the heatmap on the frame
                heatmaps.append(overlay)

            # Display the frames and heatmaps
            st.write("### Heatmap Visualization")
            fig, axes = plt.subplots(2, 8, figsize=(20, 5))
            for i in range(8):
                if i < len(frames):
                    axes[0, i].imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                    axes[0, i].axis("off")
                if i < len(heatmaps):
                    axes[1, i].imshow(cv2.cvtColor(heatmaps[i], cv2.COLOR_BGR2RGB))
                    axes[1, i].axis("off")
            st.pyplot(fig)

# Training Loss Curve
st.write("## Training Loss Curve")
try:
    losses = []
    log_file_path = "logs/training.log"  # Path to the training log file

    # Check if the log file exists
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            for line in file:
                # Extract loss values using a regular expression
                match = re.search(r"Loss: ([0-9.]+)", line)
                if match:
                    losses.append(float(match.group(1)))

        # If losses are found, plot the training loss curve
        if losses:
            df = pd.DataFrame({"Epoch": range(1, len(losses) + 1), "Loss": losses})
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Epoch:Q", title="Epochs"),
                    y=alt.Y("Loss:Q", title="Loss"),
                    tooltip=["Epoch", "Loss"],
                )
                .properties(title="Training Loss Curve", width=800, height=400)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("The training log file is empty or does not contain valid data.")
    else:
        st.warning(f"Training log file not found. Please ensure the file exists at '{log_file_path}'.")
except Exception as e:
    st.warning(f"An error occurred while reading the training log: {str(e)}")
