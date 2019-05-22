import cv2
# face detection in image
# including the face features
face_cascade = cv2.CascadeClassifier('E://Machine Learning A-Z Template Folder//OpenCV//haarcascade_frontalface_default.xml')

# reading the image
img = cv2.imread('E://Machine Learning A-Z Template Folder//OpenCV//img2.jpg', 1) # 1 indicates the colored

#converting thr image into thr gray scale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Legend",gray_image)

#grtting the co-ordinates of the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors=5)

# adding rectangle to the face
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h),(200,300,0),3)

#resizing my image
#resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

#displaying the image
cv2.imshow("Legend",img)
cv2.waitKey()  # disappear will depend on your parameter
cv2.destroyAllWindows()
