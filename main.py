import face_recognition
from PIL import Image, ImageDraw
from Face_Rec import FaceRec

fr = FaceRec()
final = []
TOLERANCE = 0.5

known_encod, known_names = fr.load_images("Training_images/")

test = 'test_image3.jpg'
test_locations, test_encodings = fr.identify(test)
img = face_recognition.load_image_file(test)
pil_image = Image.fromarray(img)

draw = ImageDraw.Draw(pil_image)

for(top, right, bottom, left), face_encoding in zip(test_locations, test_encodings):
    matches = face_recognition.compare_faces(known_encod, face_encoding, TOLERANCE)

    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]
        final.append(name)
    
    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,255))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

print(final)
pil_image.show()
pil_image.save('testrun2.jpg')