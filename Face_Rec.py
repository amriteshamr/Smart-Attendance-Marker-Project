import face_recognition
import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageDraw

class FaceRec:
    def __init__(self):
        self.known_faceencodings = []
        self.known_facenames = []
    
    def load_images(self, images_path):
        
        images_path = glob.glob(os.path.join(images_path, "*.*"))

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
            self.known_faceencodings.append(img_encoding)
            self.known_facenames.append(filename)
            
        print("Encoding images loaded")
        return self.known_faceencodings, self.known_facenames


    def identify(self,test_image):
        test = face_recognition.load_image_file(test_image)
        face_locations = face_recognition.face_locations(test)
        face_encodings = face_recognition.face_encodings(test, face_locations)

        return face_locations, face_encodings

        #pil_image = Image.fromarray(test)

        #draw = ImageDraw.Draw(pil_image)

        #for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings)
