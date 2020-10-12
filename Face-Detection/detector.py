import cv2
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainner.yml")
         
id=0
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_detect.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=15)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w++3,y+h++3),(0,255,0),3)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        #if (id==1):
            #id='abc'
        cv2.putText(frame,str(id),(x+h,y),font,1,(255,0,0),2);
    cv2.imshow("abc",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
