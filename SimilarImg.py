import cv2

# loading the images
img1 = cv2.imread("png//yy.jpg")
img2 = cv2.imread("png//ra.jpg")

# resizing the both images in same resolution
scale_percent = 60 # percent of original size
width = int(img1.shape[1] * scale_percent / 90)
height = int(img2.shape[0] * scale_percent / 90)
dim = (width, height)
# resize image
reimg1 = cv2.resize(img1,dsize= dim, interpolation = cv2.INTER_AREA)
reimg2 = cv2.resize(img2, dsize=dim, interpolation = cv2.INTER_AREA)

#including face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray1 = cv2.cvtColor(reimg1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(reimg2,cv2.COLOR_BGR2GRAY)

face1 = face_cascade.detectMultiScale(gray1,scaleFactor = 1.05, minNeighbors=5)
face2 = face_cascade.detectMultiScale(gray2,scaleFactor = 1.05 ,minNeighbors=5)

# putting the rectangle on the faces
for x,y ,w,h in face1:
    reimg1 = cv2.rectangle(reimg1, (x, y), (x+w, y+h),(255, 0, 0), 3)

for x,y ,w,h in face2:
    reimg2 = cv2.rectangle(reimg2, (x, y), (x+w, y+h), (0, 0, 250), 3)



# difference img of the images
diff = cv2.subtract(reimg1, reimg2)
cv2.imshow("diff",diff)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
print("Difference : ",diff)

# comparing the two images for exactly same of not
b, g, r = cv2.split(diff)
if cv2.countNonZero(b)==0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("Both Images Are Identical !")
else:
    print("Images Are Not Identical")

# finding the similarities of two images
sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(reimg2, None) # key-points corresponds to the position
kp2, desc2 = sift.detectAndCompute(reimg1, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()

#  Fast Library for Approximate Nearest Neighbors
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2) # to get the best matches

good_points = []  # correct matches
ratio = 0.6
for m, n in matches:
    if m.distance < ratio*n.distance:
        good_points.append(m)

# showing the both images (compare mode)
result = cv2.drawMatches(reimg1, kp1, reimg2, kp2, good_points, None)

cv2.imshow("result", result)
cv2.imshow("img1", reimg1)
cv2.imshow("img2", reimg2)
cv2.waitKey(0)
cv2.destroyAllWindows()
acc = len(good_points)*100/len(matches)
print("Good Points : {}".format(len(good_points)))
print("Total Matches : {}".format(len(matches)))
print("Accuracy : {}".format(acc))
if acc > 0:
    print("Both Are The same person")
else:
    print("Different Persons")