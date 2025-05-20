import cv2
import sys
import os
import re
import numpy as np
import random
import shelve

FONT = cv2.FONT_HERSHEY_SIMPLEX
CASCADE = "face_cascade.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE)

# Use shelve for persistent label storage
Datafile = shelve.open("Data")
if 'Data' not in Datafile:
    Datafile['Data'] = []
    Data_list = []
else:
    Data_list = Datafile["Data"]

def Make_Changes(label):
    if label not in Data_list:
        Data_list.append(label)
        print(Data_list)

def get_images(path):
    images = []
    labels = []
    count = 0
    if len(os.listdir(path)) == 0:
        print("Empty Dataset.......aborting Training")
        sys.exit()
    for img in os.listdir(path):
        regex = re.compile(r'(\d+|\s+)')
        labl = regex.split(img)[0]
        count += 1
        Make_Changes(labl)
        image_path = os.path.join(path, img)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image_grey)
        labels.append(Data_list.index(labl))
    return images, labels, count

def initialize_recognizer():
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        face_recognizer = cv2.createLBPHFaceRecognizer()
    print("Training..........")
    Dataset = get_images("./Dataset")
    print(f"Recognizer trained using Dataset: {Dataset[2]} Images used")
    face_recognizer.train(Dataset[0], np.array(Dataset[1]))
    return face_recognizer

def save_wrong_faces(num, temp_set, faces):
    dataset_dir = os.path.abspath("./Dataset")
    if num:
        print("Enter number below face : Correct Name:")
        for i in range(num):
            inp = input()
            inp = inp.split(":")
            faces[int(inp[0])][0] = -1
            if inp[1].lower() != "nil":
                filename = os.path.join(dataset_dir, inp[1] + str(random.uniform(0, 100000)) + ".jpg")
                cv2.imwrite(filename, temp_set[int(inp[0])])
    for i in range(len(faces)):
        if faces[i][0] != -1 and faces[i][1] > 18:
            filename = os.path.join(dataset_dir, Data_list[faces[i][0]] + str(random.uniform(0, 100000)) + ".jpg")
            cv2.imwrite(filename, temp_set[i])

def recognize(image_path, face_recognizer):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found at path:", image_path)
        sys.exit(1)
    
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0
    )
    
    if len(faces) == 0:
        print("No faces detected")
        Datafile["Data"] = Data_list
        Datafile.close()
        return

    detected_names = set()
    for (x, y, w, h) in faces:
        sub_img = image_grey[y:y + h, x:x + w]
        try:
            nbr, conf = face_recognizer.predict(sub_img)
            name = Data_list[nbr]
            # Remove spaces and underscores
            trimmed_name = name.replace("_", "").replace(" ", "").strip()
            detected_names.add(trimmed_name)
        except Exception:
            continue

    if detected_names:
        print(",".join(detected_names))
    else:
        print("No known person detected.")

    Datafile["Data"] = Data_list
    Datafile.close()


def recognize_video(face_recognizer):
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25, 25), flags=0)
        for x, y, w, h in faces:
            sub_img = image_grey[y:y + h, x:x + w]
            nbr, conf = face_recognizer.predict(sub_img)
            cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 255, 0), 2)
            cv2.putText(image, Data_list[nbr], (x, y - 10), FONT, 0.5, (255, 255, 0), 1)
        cv2.imshow("Faces Found", image)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
            break
    Datafile["Data"] = Data_list
    Datafile.close()
    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) not in [1, 2]:
        print("Usage: python face_recog.py [<complete image_path>]")
        sys.exit()
    face_r = initialize_recognizer()
    if len(sys.argv) == 1:
        recognize_video(face_r)
    else:
        recognize(sys.argv[1], face_r)

if __name__ == "__main__":
    main()
