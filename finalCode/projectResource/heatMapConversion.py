import cv2
import os


try:
	os.mkdir("heatMapPics")
except:
	pass

os.chdir("outputs/")
list_ = os.listdir()

count = 0

for i in list_:
	img = cv2.imread(i,0)
	#img = cv2.imread("/home/g00g1y5p4/Downloads/Hackathon/patholes/pk/Data/train/274.image.png",0)
	img = cv2.applyColorMap(img,cv2.COLORMAP_JET)
	cv2.imwrite("../heatMapPics/"+str(count)+".png",img)
	#cv2.imwrite("/home/g00g1y5p4/Downloads/Hackathon/patholes/pk/2.png",img)
	print(count)
	count+=1
#	break
