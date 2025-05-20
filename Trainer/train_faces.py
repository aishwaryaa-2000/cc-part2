import cv2
import sys
import os
import random

# Fix the path to the cascade file
cascade_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../face_cascade.xml"))
face_cascade = cv2.CascadeClassifier(cascade_path)

# Fix the path to the Dataset directory
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Dataset"))

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created Dataset directory at: {dataset_dir}")

def detect(image_path, name, counter):
    image = cv2.imread(os.path.abspath(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return counter
    
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0
    )
    
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
    
    for x, y, w, h in faces:
        sub_img = image[y:y+h, x:x+w]
        save_filename = f"{name}_{counter}_{random.randint(1000,9999)}.jpg"
        save_path = os.path.join(dataset_dir, save_filename)
        cv2.imwrite(save_path, sub_img)
        print(f"Saved face image to {save_path}")
        counter += 1
    
    return counter

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_faces.py <Name of person>")
        sys.exit(1)

    name = sys.argv[1]
    # Fix the path to the person's directory
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), name))

    if not os.path.exists(directory_path):
        print(f"No directory exists for the given person: {name}")
        print(f"Expected directory: {directory_path}")
        sys.exit(1)

    print(f"Processing images from: {directory_path}")
    
    # Change to the directory containing the images
    os.chdir(directory_path)

    print("Creating Proper Dataset.......")
    images_exist = False
    counter = 1
    for img in os.listdir("."):
        if img.lower().endswith(('.jpg', '.png', '.jpeg')):
            print(f"Processing image: {img}")
            counter = detect(img, name, counter)
            images_exist = True

    if not images_exist:
        print("No images found to create a dataset")
    else:
        print(f"Processed {counter-1} faces successfully")
        print(f"Face images saved to: {dataset_dir}")

if __name__ == "__main__":
    main()