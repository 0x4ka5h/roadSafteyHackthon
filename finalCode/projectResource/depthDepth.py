import torch
import os
import cv2
from tensorflow import keras
import tensorflow as tf
import numpy as np
import numpy
model=torch.hub.load('yolov5','custom',path='best.pt',source='local')

depthModel = keras.models.load_model("modelforDepth.h5")

#vid=cv2.VideoCapture(0)

#while 1:
try:
	os.mkdir("imgDepths")
except:
	pass

def imgArray(img):
	image = tf.io.read_file(img)
	img = tf.image.decode_png(image)
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, [128, 128])
	return np.asarray(img)


v = cv2.VideoCapture("testImages/testing.mp4")


count=0


#temp = cv2.VideoWriter("detection.avi",cv2.VideoWriter_fourcc(*'MJPG'),10,size)

e=0

fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

while 1:
	#img = cv2.imread(i,0)
	
	_,img = v.read()
	if e<150:
		e+=1
		continue
		
	#print(img)
	#cv2.imshow("12",img)
	#cv2.waitKey(1)
	try:
		result = model(img)
	except:
		print(11)
		continue
	li=result.xyxy[0]
	k=0
	for i in li:
		#print(i)
		
		try:
			if int(result.xyxyn[0][k][-1])>0:
				img1=img[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
				print(result.xyxyn[0][k][-1])
				#print(img1)
				count+=1
				
				
				img_ = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
				bluredImg = cv2.GaussianBlur(img_, (3,3), 4400)
				_,thresh = cv2.threshold(bluredImg,120,255,cv2.THRESH_BINARY)	
				cv2.imwrite("imgDepths/omkay.png",thresh)		
				
				toFindDepth = imgArray("imgDepths/omkay.png")
				
				
				arr = depthModel(toFindDepth)
				
				
				dep = arr[0][0][0][0]
				
				img = cv2.putText(img, str(float(dep))[0:3]+" cm", (int(i[0]-5),int(i[1])-5), font, fontScale, color, thickness, cv2.LINE_AA)
				img = cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
				
				#cv2.waitKey(0)	
		except Exception as exception:
			print(exception)
			continue
		k+=1
		
		
	cv2.imshow("img",img)
	#temp.write(img)
	if cv2.waitKey(1)==27:
		break
		

#temp.release()
v.release()
cv2.destroyAllWindows()	
#	break
