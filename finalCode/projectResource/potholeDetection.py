import torch
import os
import cv2
#from tensorflow import keras
model=torch.hub.load('yolov5','custom',path='best.pt',source='local')

#depth_model = keras.models.load_model("modelforDepth.h5")

#vid=cv2.VideoCapture(0)

#while 1:
try:
	os.mkdir("outputs")
except:
	pass


v = cv2.VideoCapture("testImages/testing.mp4")

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
size= (400,400)
temp = cv2.VideoWriter('testImages/detection.mp4', fourcc, 30, size)



os.chdir("Data/train/")
list_ = os.listdir()
count=0
#temp = cv2.VideoWriter("detection.avi",cv2.VideoWriter_fourcc(*'MJPG'),10,size)

e=0

while 1:
	img = cv2.imread("testImages/night_light.jpeg",0)
	
	#_,img = v.read()
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
	print(li)
	print(1)
	k=0
	for i in li:
		#print(i)
		
		try:
			if int(result.xyxyn[0][k][-1])>0:
				img1=img[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
				print(result.xyxyn[0][k][-1])
				#print(img1)
				count+=1
				#cv2.imwrite("/home/g00g1y5p4/Downloads/Hackathon/patholes/pk/outputs/"+str(count)+".png",img1)
				#cv2.waitKey(0)	
		except:
			continue
		k+=1
		img = cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
	cv2.imshow("img",img)
	temp.write(img)
	if cv2.waitKey(1)==27:
		break
		

temp.release()
v.release()
cv2.destroyAllWindows()	
#	break
