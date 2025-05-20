import cv2
import random
import os
import sys

def start(directory_name):
    count = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit and stop capturing images.")
    while True:
        ret, image = cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            continue
        count += 1
        if count % 25 == 0:
            filename = os.path.join(directory_name, str(random.uniform(0, 100000)) + ".jpg")
            cv2.imwrite(filename, image)
            print(f"Saved image: {filename}")
        cv2.imshow("Video", image)
        
        # Fix the key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Q key pressed. Stopping capture.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_dataset.py <Name of the person>")
        sys.exit(1)

    name = sys.argv[1]
    dir_name = name

    # If directory exists, add a random suffix to avoid overwriting
    if os.path.exists(dir_name):
        print(f"Directory '{dir_name}' already exists.")
        dir_name = f"{dir_name}_{random.randint(0, 10000)}"
        print(f"So, the dataset will be saved as '{dir_name}'.")

    os.makedirs(dir_name)
    print(f"Saving images to directory: {dir_name}")

    start(dir_name)

if __name__ == "__main__":
    main()
