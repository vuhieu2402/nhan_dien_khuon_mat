import cv2
import numpy as np
from PIL import Image
import os


path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_np = np.array(PIL_img, 'uint8')
        
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiple(img_np)
        
        for (x,y,w,h) in faces:
            faceSamples.append(img_np[y:y+h , x:x+w])
            ids.append(id)
    
    return faceSamples,ids

print("\n [INFO] Dang training du lieu...")

faces,ids = getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n [INFO] Khuon mat duoc train. Thoat".format(len(np.unique(ids))))