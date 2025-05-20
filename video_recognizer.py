import streamlit as st
import cv2
import requests
import numpy as np
import time
from PIL import Image
import io


API_URL = "http://localhost:8000/recognize"  # Change if your FastAPI server is on a different host/port
#API_URL = "http://35.175.247.253:8000/recognize"
# def recognize_frame(frame):
#     # Encode frame as JPEG
#     ret, buf = cv2.imencode('.jpg', frame)
#     if not ret:
#         return ""
#     files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
#     try:
#         response = requests.post(API_URL, files=files, timeout=10)
#         if response.status_code == 200:
#             data = response.json()
#             names = data.get("recognized_names", [])
#             return ", ".join(names)
#         else:
#             return "API Error"
#     except Exception as e:
#         return f"Error: {e}"

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for webcam; or provide a video file path
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return

#     last_request_time = 0
#     recognized_names = ""

#     print("Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break

#         current_time = time.time()
#         # Make API request every 5 seconds
#         if current_time - last_request_time >= 5:
#             recognized_names = recognize_frame(frame)
#             print("Recognized:", recognized_names)
#             last_request_time = current_time

#         # Optionally, show the video (remove/comment these two lines if you want only console output)
#         # cv2.imshow('Video Recognition', frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#         # If you want to quit with 'q' even without video window:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="wide"
)

def recognize_image(image_array):
    """Send image to API and get recognition results"""
    # Convert numpy array to bytes
    is_success, buffer = cv2.imencode(".jpg", image_array)
    if not is_success:
        return None
        
    # Create file-like object
    io_buf = io.BytesIO(buffer)
    
    # Make request to API
    try:
        files = {'file': ('image.jpg', io_buf, 'image/jpeg')}
        response = requests.post(API_URL, files=files, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def main():
    # App title and description
    st.title("Face Recognition System")
    st.write("Upload an image or use your webcam to identify faces")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    detection_method = st.sidebar.radio(
        "Detection Method",
        ["Upload Image", "Webcam"]
    )
    
    # Display DB Info
    if st.sidebar.button("View Database Info"):
        try:
            response = requests.get("http://localhost:8000/db-info/")
            if response.status_code == 200:
                st.sidebar.json(response.json())
            else:
                st.sidebar.error("Failed to fetch database info")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Train model button
    if st.sidebar.button("Train Model"):
        try:
            with st.sidebar:
                with st.spinner("Training model..."):
                    response = requests.post("http://localhost:8000/train")
                    if response.status_code == 200:
                        st.success("Model trained successfully!")
                    else:
                        st.error(f"Training failed: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Main content - Image upload
    if detection_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display the uploaded image
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
            
            # Process the image
            if st.button("Recognize Faces"):
                with st.spinner("Processing..."):
                    result = recognize_image(image)
                    
                with col2:
                    if result:
                        names = result.get("recognized_names", [])
                        if names:
                            st.success(f"Identified: {', '.join(names)}")
                            
                            # Show confidence scores
                            st.subheader("Confidence Scores")
                            confidence_scores = result.get("confidence_scores", {})
                            for name, score in confidence_scores.items():
                                st.write(f"{name}: {score:.2f}")
                        else:
                            st.warning("No known faces detected")
                            
                        # If there's consensus information, show it
                        if "consensus_info" in result:
                            st.subheader("Consensus Information")
                            st.json(result["consensus_info"])
                    else:
                        st.error("Failed to process image")

    # Main content - Webcam
    else:
        st.write("Webcam Face Recognition")
        
        # Placeholder for webcam feed
        video_placeholder = st.empty()
        
        # Placeholder for results
        result_placeholder = st.empty()
        
        # Start/Stop button for webcam
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        
        if start_button:
            # Session state to manage webcam state
            if 'webcam_running' not in st.session_state:
                st.session_state.webcam_running = True
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam")
                return
            
            last_api_call = 0
            current_result = None
            
            # Process frames until stop button is clicked
            while st.session_state.webcam_running and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Make API call every 2 seconds
                current_time = time.time()
                if current_time - last_api_call >= 2:
                    result = recognize_image(frame)
                    last_api_call = current_time
                    
                    if result:
                        names = result.get("recognized_names", [])
                        if names:
                            result_placeholder.success(f"Identified: {', '.join(names)}")
                            current_result = result
                        else:
                            result_placeholder.warning("No known faces detected")
                    
                # Allow UI to update
                time.sleep(0.1)
            
            # Release webcam when done
            cap.release()
            
        if stop_button:
            st.session_state.webcam_running = False
            st.write("Webcam stopped")

if __name__ == "__main__":
    main()