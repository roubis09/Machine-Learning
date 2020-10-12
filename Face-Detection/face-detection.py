import cv2
# face_detect=cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\Coding\\Face Recognition\\haarcascade_frontalface_default.xml')
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0) # VideoCapture(1) if external webcam
id=input('Enter user id')
sampleNum=0
while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_detect.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=15)
    for (x,y,w,h) in face:
        sampleNum=sampleNum+1
        cv2.imwrite('Dataset//User.'+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(100);
    cv2.imshow("Kunu",frame)
    cv2.waitKey(1);
    if(sampleNum>20):
        break
video.release()
cv2.destroyAllWindows()