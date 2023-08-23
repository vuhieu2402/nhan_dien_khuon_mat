import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n nhap ID khuon mat ==> ')

print("\n [Info] khoi tao camera....")

count =0
while (True):
    ret , img = cam.read()
    img = cv2.flip(img, -1) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3 , 5)
    
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y) , (x+w,y+h) , (255,0,0),1)
        count += 1
        
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        
        cv2.imshow("img" , img)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif count >=30 :
        break
    
print("\n [INFO] thoat")
cam.release()