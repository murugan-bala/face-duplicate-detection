import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import time
from PIL import Image, ImageDraw
import flask 
ud={}
IMAGES_PATH = 'E:\\Face Detection\\DEMO 3\\image'
#known_face_encodings=[]
def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

#Using this, now make a little identity database, containing the encodings of our reference images:
def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    global known_face_encodings,known_face_encod_names ,database
    database = {}
    known_face_encodings=[]
    known_face_encod_names=[]
    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]

        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[identity] = encodings[0]
        # Create arrays of known face encodings and their names
        known_face_encodings.append(encodings)
        known_face_encod_names.append(identity)
        #print(known_face_encodings)
    #return database,known_face_encodings
    #print(database)
    return known_face_encod_names,known_face_encodings

kn_fac_encod_names,kn_fac_encodis = setup_database()
#print("known face",kn_fac_encodis)
#print("known face_names",kn_fac_encod_names)
di={}
names=[]
for k,v in zip(kn_fac_encodis,kn_fac_encod_names):
    #print(k,v)
    di.update({str(k):v})
#print(di)
#print(type(kn_fac_encodis))  # list
for key, value in database.items():
    #print(type(value))
    #val = np.asarray(value)
    ud[str(value)]=key 
#print("$$$$ value type",type(value))
#print(value)
#print(ud)
#print(kn_fac_encodis)
count=0

for check in kn_fac_encodis:
    #print("Check",check)
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file("E:\\Face Detection\\DEMO 3\\image\\tamizhh.jpg")  #am , sugu, suu , unknownface, tamizhh,
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)   #face_locations = face_recognition.face_locations(image, model="cnn")
    #face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)
    #dupl=[]
    locations, face_encodings = get_face_embeddings_from_image(unknown_image)
    #print("Encoding_Face_embedding",face_encodings[0])
    #print("Encoding_Face_encoding",face_encodings)

    matches = face_recognition.compare_faces(check, face_encodings[0])
    face_distances = face_recognition.face_distance(check, face_encodings[0])
    print(face_distances)
    if face_distances[0]<0.44:
        print("Photo Matched.....!!!")
        count=count+1
        #print(di[str(check)])
        names.append(di[str(check)])
    else:
        print("Not Matched..!")
    print(count)

if count>1:
    print("Your photo",count-1," already in our database.")
else:
    print("No duplicate image found in database..")
print(names)

