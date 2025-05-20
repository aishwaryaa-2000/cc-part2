# cc-part2

# Local Face Recognition System with IPFS Integration

This repository contains a complete face recognition system that operates locally and supports decentralized storage via **IPFS**. It provides a user-friendly **Streamlit** interface for image uploads and webcam recognition, while the **FastAPI** backend performs detection, recognition, consensus validation, and logs results to the database of your choice (Local, DynamoDB, Firestore, or IPFS).

---

## Features

- **Real-Time Face Recognition**
  - Upload image or use webcam for detection.
  - Uses OpenCV’s LBPH recognizer for identity classification.

- **Streamlit Frontend**
  - Upload photos or run webcam detection via browser.
  - View results, confidence scores, and database info.

- **FastAPI Backend**
  - Train, recognize, log and fetch recognition results.
  - Configurable with Local DB, DynamoDB, Firestore, or IPFS.

- **Consensus Mechanism**
  - Verifies identity matches across peer FastAPI servers for added reliability.

- **IPFS File Storage**
  - Store training data and results on InterPlanetary File System (IPFS).
  - Upload/download files directly using IPFS HTTP API.

---

## Project Structure

├── api_server.py # FastAPI backend with training, recognition, consensus, and IPFS support
├── video_recognizer.py # Streamlit UI for image/webcam recognition and interaction
├── face_recogn.py # Standalone script for CLI-based face recognition
├── Dataset/ # Folder with training images (name-prefixed files)
├── Data # Shelve database (for local provider)
├── RecognitionResults # Shelve results log
├── face_cascade.xml # Haar Cascade XML for face detection


##  Setup Instructions

### 1. Clone the repository
git clone https://github.com/your-username/face-recognition-ipfs.git
cd face-recognition-ipfs
### 2. Create a virtual environment
python3 -m venv face-env
source face-env/bin/activate
### 3. Install dependencies
pip install -r requirements.txt
### 4. Install and run IPFS locally (for IPFS mode)
brew install ipfs
ipfs init
ipfs daemon

Then run 3 servers for peers on port 8000, 8001 and 8002. Also, a streamlit server for the frontend complete application
