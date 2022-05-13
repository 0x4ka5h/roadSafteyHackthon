
### Introduction 
I knew you guys can't bear for long period for my presence with our new blogs. That's why this time I came up with a real-time developed project which is submitted to the Government of India.

[Visit Hackathon website](https://innovateindia.mygov.in/roadsafety-hackathon-2021/)

A hackathon was conducted overall in India. After, 3 rounds of filtering we got selected for the semi-final round based on the prototype presentation. And we cleared that round with our prototype and as a result, we were gone into the finals. It's a hackathon called the Road Safety Hackathon conducted by the Ministry of Road Transport and Highways, Government of India.

As per recent studies, around 4000+ accidents (about 50% casualties) are observed due to potholes in India. However, this is only official data as per the available records. This doesn’t include the data related blaw, blaw, blaw.....  Fine. fine fine......  Shit comes with shit only. Then why discuss it again?
Already, we all know this shit of information cause we daily see accidents happen on roads due to bad infrastructure. That bad infrastructure leads us to create an application that is used for the Government of India by changing Road Infrastructure based on our stored data and vehicle driver to escape from accidents by following our commands. 

Chaaa....Chaa.. why have these people made mistakes related to vehicles, roads, and safety? I'm tired of solving these mistakes. I'll stop taking care of these problems and maybe this is the last blog about road safety, Accidents Prevention, and things that are related with vehicles also.

Random Reader (RR): "Author!! You always wasting our time by telling nonsense. Come to the point."


Me: "Damn. Seriously? Okay, stop kicking on my ...,no. I will start."


RR: "lamo... Quick."

### Our Aim

Deep learning and Web-based voice alert system based on depth estimation and Hazards of the detected potholes.

#### Brief Explanation of our Solution

Brief Explanation of our Solution

We created this application in two ways:

1. Vehicles with Cameras and wifi connectors. These vehicles will detect potholes on road and send information to the cloud. The sent data will be categorized into levels of its hazards. If it returns medium or high risk, then the system will estimate the pothole depth using Deep learning techniques. At, the same time driver also got an alert from the system when there is a high risk with depth.
2. Vehicles with no facilities can use their mobile phone to access our web application. After giving the destination location, our system will guide the driver in a very safe and best way to reach the destination safely. Here, also the alert will go through sound and the driver can also see maps to visualize the pothole's distance from our current location.


By alerting the driver when Potholes are ahead, we think they will care for us and take diversion or may go slowly to escape from accidents.

We actually use python for internal processing, so these are the required libraries.


```py
import tensorflow as tf, tensorflow
from tensorflow import keras
import torch
import cv2
import numpy
import os,sys,time
```


We have used YoLov5 for pothole detection purposes cause my personal computer can't bear my model architecture and also it takes 3 days on the CPU. So we would love to use a predefined model. After a lot of case studies about predefined models, YoLov5 become the best match for our goal. But installing required software and libraries, we labeled them by hand on our dataset manually meaning picture by picture by drawing boxes. And we labeled potholes into 3 classes which are small, medium, and risk. 

You know YoLov5, don't you..? YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development and "goo gooooo goooo. Go and search about it online. I'm not here to teach you about YoLo". 

Some RR: ![sadistic look](https://g00g1y5p4.github.io/posts/depthestimationofapothole/images/index.webp)

Me: "Jokes apart!." Once the level of hazards is detected, then we have to apply some image processing techniques that are used to highlight features at the edge of our various detected potholes. We have already trained a regressive model which will predict the Approx. depth of a pothole.

That's it, we have done a lot. Now we just need to send the information to cloud storage with GPS location and depth of pothole for others and the Government.

#### Estimating Depth of a Pothole

Previously as I have said, we detect potholes using YoLov5 which uses convolution neural networks in their model architecture. Once we initialize the camera and other setup, the model takes the input as an image from the camera and returns information about the image and detected coordinates. 

Download model weights [(trainedModel)](https://g00g1y5p4.github.io/posts/depthestimationofapothole/files/best.pt)

```py
model=torch.hub.load('yolov5','custom',path='best.pt',source='local')
capturedFrame = cv2.imread("RandomRoadimg.jpg")

result = model(capturedFrame)
result.show()

coordinatesList = result.xyxy[0][0] #coordinates of first detected pothole

```
![(outputImage)](https://g00g1y5p4.github.io/posts/depthestimationofapothole/images/image0.jpg) 

[12 50 200 300]

Now, we need to send the detected part to the depth estimation model that was already trained with 200 epochs, 89% accuracy, and 0.02 loss error.

You guys may get doubts like if torch then why TensorFlow? if using TensorFlow then why use torch? One of our teammates is using torch always. When we are doing this project, we have less time so we divided our work. So he did detection and I estimate depth. I'm very feasible with TensorFlow and their related frameworks like Keras, media pipe, etc.., So we don't have much time while we're submitting these files. Anyways, we achieve our goal, but not in an optimistic way. 

Our depth estimation model is a regression model, as I said it's just a simple linear regression model. I didn't use any convolution neural networks in my model architecture. I just use Dense layers to Downsample the input image with a linear activation function.

Some RR: "Author.! Is this a model? It's very simple. See, again you writing blogs about your works?  haha "

Me: "Yeah! It's a very simple model, but initially, I form a very big architecture with 16 convolution neural networks with 7 Max pooling and 2 Averagepooling with activation function relu for all layers and softmax for last one." For 1 epoch only my PC takes 6 hours and the accuracy is best with 82% with a loss error of 0.4334.

Me: That's why after a lot of tries I just trail this simple network and Boom. It works. 

So, Lets try our model on a video.

Download model [(DepthEstimation)](https://g00g1y5p4.github.io/posts/depthestimationofapothole/files/modelforDepth.h5)

```py
depthModel = keras.models.load_model("modelforDepth.h5")
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

fontScale = 1
color = (0, 0, 255)
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	_,img = v.read()
	try:
		result = model(img)
	except:
		continue
		
	coordinateLines=result.xyxy[0]
	k=0
	for i in coordinateLines:
		try:
			if int(result.xyxyn[0][k][-1])>0:
				img1=img[int(i[1]):int(i[3]),int(i[0]):int(i[2])]
				
				img_ = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
				bluredImg = cv2.GaussianBlur(img_, (3,3), 4400)
				_,thresh = cv2.threshold(bluredImg,120,255,cv2.THRESH_BINARY)	
				cv2.imwrite("imgDepths/omkay.png",thresh)		
				
				toFindDepth = imgArray("imgDepths/omkay.png")
				arr = depthModel(toFindDepth)
				dep = arr[0][0][0][0]
				rect = img.copy()
				img = cv2.putText(img, str(float(dep))[0:3]+" cm", (int(i[0]-5),int(i[1])-5), font, fontScale, color, thickness, cv2.LINE_AA)
				img = cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
				rect = cv2.rectangle(rect,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
				
		except Exception as exception:
			print(exception)
			continue
		k+=1
		
		
	cv2.imshow("img",img)
	cv2.imshow("rect",rect)

	if cv2.waitKey(1)==27:
		break
v.release()
cv2.destroyAllWindows()
```
![BoundedBoxed](https://g00g1y5p4.github.io/posts/depthestimationofapothole/images/rect.png)
![DepthEstimated](https://g00g1y5p4.github.io/posts/depthestimationofapothole/images/depth.png)

##### Thanks for reading! {align=center}
