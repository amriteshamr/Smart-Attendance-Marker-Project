import face_recognition
import pickle
import os
import glob
import cv2

known_faceencodings = []
known_facenames = []

images_path = glob.glob(os.path.join('Training_images', "*.*"))

print("{} encoding images found.".format(len(images_path)))

for img_path in images_path:
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the filename only from the initial file path.
    basename = os.path.basename(img_path)
    (filename, ext) = os.path.splitext(basename)
     # Get encoding
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    # Store file name and file encoding
    known_faceencodings.append(img_encoding)
    known_facenames.append(filename)

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(known_faceencodings, f)
with open('dataset_names.dat', 'wb') as f:
    pickle.dump(known_facenames, f)
