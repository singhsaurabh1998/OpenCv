import cv2

# for classifing the face features
face_classsifier = cv2.CascadeClassifier('E://Machine Learning A-Z Template Folder//OpenCV//haarcascade_frontalface_default.xml')

#will give the face
def face_extract(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting it to be in gray image
    # 1.3 is scaling factor & 6 is the min neighbours
    faces = face_classsifier.detectMultiScale(gray, 1.3, 6)

    if faces is():
        return None
    for(x,y, w, h) in faces:
        cropped_faces = img[y:y+h, x:x+w]
    return  cropped_faces

video = cv2.VideoCapture(0)

a = 0 # initially zero frames
while True:
    ret,frame = video.read()
    count = 0
    if face_extract(frame) is not  None:
        a +=1 # we are collecting top 100 samples
        face = cv2.resize(face_extract(frame),(800,800))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        image_path = 'E://Machine Learning A-Z Template Folder//OpenCV//Samples/user'+str(a)+'.jpg'
        cv2.imwrite(image_path,face) #copy img to the above path
        cv2.putText(face, "Captured Images: "+str(a),(50,50),cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),4)
        cv2.imshow('Faces',face)
        count = 1

    else:
        print("Faces Are Not Found\n") # Try To Look Straight Into The camera

    k = cv2.waitKey(1)
    if k == ord('q') or a==100: # press q for exit
        print("Captured Images : ", a)
        break
video.release()
cv2.destroyAllWindows()
print("Collecting samples Complete\n")

