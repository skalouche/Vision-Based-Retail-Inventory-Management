# Implement some CV algorithms using OpenCV Library

import numpy as np
import scipy as spy 
import sys
import cv2

# from matplotlib import pyplot as plt


# load an image: 1 refers to color, 0 is gray scale
# img = cv2.imread('IMG_1214.jpg',1)	
img = cv2.imread(sys.argv[1]) 

# rescale/resize image 
desiredWidth = 1000.0 	#pixels
height, width, depth = img.shape
imgScale = desiredWidth/width
newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
img = cv2.resize(img,(int(newX),int(newY)))	

# convert to gray scale and  for Hough Transform
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)





# probablistic Hough Transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)
print lines
for line in lines:
	x1,y1,x2,y2 = line[0]
	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# normal Hough Transform
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for line in lines:
# 	rho,theta = line[0]
# 	a = np.cos(theta)
# 	b = np.sin(theta)
# 	x0 = a*rho
# 	y0 = b*rho
# 	x1 = int(x0 + 1000*(-b))
# 	y1 = int(y0 + 1000*(a))
# 	x2 = int(x0 - 1000*(-b))
# 	y2 = int(y0 - 1000*(a))
# 	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# show the edges only
cv2.imshow('image',img)

# keep image displayed for certain number of milliseconds. If 0 then wait for keystroke
key = cv2.waitKey(0)	

cv2.imwrite('houghlines3.jpg',img)








if key == 27:			# wait for esc key
	# close all windows
	cv2.destroyAllWindows()
elif key == ord('s'):	# wait for s-key
	# save image then close
	cv2.imwrite('test.jpg',img)
	cv2.destroyAllWindows()



# # Use matplotlib to plot the image instead
# img2 = cv2.imread('IMG_1214.jpg',1)
# plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])	# hide tick values on x and y axis
# plt.show()


############ Blob Detection
# Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector()

# # Setup SimpleBlobDetector parameters.
# detector = cv2.SimpleBlobDetector_Params()
 
# # Change thresholds
# detector.minThreshold = 1;
# detector.maxThreshold = 200;
 
# # Filter by Area.
# detector.filterByArea = True
# detector.minArea = 1500
 
# # Filter by Circularity
# detector.filterByCircularity = False
# detector.minCircularity = 0.1
 
# # Filter by Convexity
# detector.filterByConvexity = True
# detector.minConvexity = 0.87
 
# # Filter by Inertia
# detector.filterByInertia = True
# detector.minInertiaRatio = 0.01
 
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
#     detector = cv2.SimpleBlobDetector(detector)
# else : 
#     detector = cv2.SimpleBlobDetector_create(detector)
 
# # Detect blobs.
# keypoints = detector.detect(grayDiff)
 
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(grayDiff, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 


