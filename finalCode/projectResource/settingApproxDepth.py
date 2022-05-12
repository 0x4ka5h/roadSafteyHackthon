import cv2
import numpy
import os,uuid

try:
	os.mkdir("DenseofDepth")
except:
	pass
path_ = "heatMapPics/"
list_ = os.listdir("heatMapPics/")

os.chdir("DenseofDepth")
z=0
for i in list_:
	z+=1
	if z<=276:
		continue
	img_ = cv2.imread("../"+path_+i)
	img = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
	bluredImg = cv2.GaussianBlur(img, (3,3), 4400)
	_,thresh = cv2.threshold(bluredImg,120,255,cv2.THRESH_BINARY)
	cv2.imshow("img",thresh)
	cv2.imshow("Hmap",img_)
	cv2.imshow("gray",img)
	cv2.waitKey()
	
	k = input("q:")
	cv2.imwrite(k+"-"+str(uuid.uuid1())+".jpg",thresh)
	cv2.destroyAllWindows()
