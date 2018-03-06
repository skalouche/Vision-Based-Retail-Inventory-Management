import cv2
import numpy as np
import sys
import copy
import rect
import ipdb

class rect:
	def height(self):
		return
	def width(self):
		return
	def x(self):
		return
	def y(self):
		return

class point:
	def x(self):
		return
	def y(self):
		return

# define some variables for customizing (changing these values gives different results)
scale = 4
vertical_divide = 1
canny_threshold1 = 0
canny_threshold2 = 255

window_name = 'Disco CV'


##################################################################
# MAIN
##################################################################  
# load image
src = cv2.imread(sys.argv[1])  	# instead of pic name use sys.argv[1] and enter pic name in cmd line

# if not src:
# 	# Image not loaded
# 	pass
# 	# return

ipdb.set_trace()

# create window
cv2.namedWindow(window_name)

# create trackbar to select vertical process
cv2.createTrackbar("Vertical Process:", window_name, vertical_divide, 1, process_image())

# create trackbar to select scale to resize
cv2.createTrackbar("Scale:", window_name, scale, 10, process_image())

# create trackbar to choose threshold1
cv2.createTrackbar("Threshold 1:", window_name, canny_threshold1, 255, process_image())

# create trackbar to choose threshold2
cv2.createTrackbar("Threshold 2:", window_name, canny_threshold2, 255, process_image())

# default start
process_image(0,0)

# keep image displayed for certain number of milliseconds. If 0 then wait for keystroke
key = cv2.waitKey(0)



##################################################################
# Auxillary Functions
##################################################################  
def process_image(in1, in2):

	dst = copy.deepcopy(src)

	if scale < 1:
		scale = 1

	# rescale/resize image 
	newX,newY = dst.shape[1]*scale, dst.shape[0]*scale
	dst_resized = cv2.resize(dst,(int(newX),int(newY)))	

	rois_h = divideHW(dst_resized, 1, canny_threshold1, canny_threshold2)

	for i in range(np.size(rois_h)):

		if vertical_divide:
			roi_h = dst_resized(rois_h[i])

			rois_w = dividHW(roi_h, 0, canny_threshold1, canny_threshold2)

			for j in range(np.size(rois_w)): # maybe add the element because size returns length of rows and columns (2 numbers)

				rois_w[j].y += rois_h[i].y
				cv2.rectangle( dst_resized, rois_w[j], [0,255,0],1)
				rois_w[j].x = rois_w[j].x * scale
				rois_w[j].y = rois_w[j].y * scale
				rois_w[j].width = rois_w[j].width * scale
				rois_w[j].height = rois_w[j].height * scale

				cv2.rectangle( dst, rois_w[j], [0, 255, 0], 3 )

		cv2.rectangle( dst_resized, rois_h[i], [0, 0, 255], 2 )
		rois_h[i].x = rois_h[i].x * scale
		rois_h[i].y = rois_h[i].y * scale
		rois_h[i].width = rois_h[i].width * scale
		rois_h[i].height = rois_h[i].height * scale
		cv2.rectangle( dst, rois_h[i], [0, 0, 255], 3 )

	cv2.imshow( "resized", dst_resized )
	cv2.imshow( window_name, dst )



# helper function returns rectangles according horizontal or vertical projection of given image
# parameters:
# src source image
# dim dimension 1 for horizontal 0 for vertical projection
# threshold1 first threshold for the hysteresis procedure ( used by internal Canny )
# threshold2 second threshold for the hysteresis procedure ( used by internal Canny )

def divideHW( src, dim, threshold1, threshold2 ):
	reduced = np.array
	canny = np.array

	if src.shape[2] == 1:
		gray = src

	if src.shape[2] == 3: 
		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	reduced = cv2.reduce(gray, dim, cv2.cv.CV_REDUCE_AVG)

    #GaussianBlur( reduced, reduced, Size(),3);

	canny = cv2.Canny( reduced, threshold1, threshold2 )

	pts = cv2.findNonZero( canny )

	# rects = cv2.rectangle(0,0, np.shape(gray)[1], np.shape(gray)[0])

	# if not np.size(pts):
	# 	rects.append(rect)

	ref_x = 0
	ref_y = 0


	for i in range(np.size(pts)):

		if dim:
			rect.height = pts[i].y-ref_y
			rects.append( rect )
			rect.y = pts[i].y
			ref_y = rect.y

			if i == np.size(pts)-1:
				rect.height = gray.rows - pts[i].y
				rects.push_back( rect )

		else:
			rect.width = pts[i].x-ref_x
			rects.append( rect )
			rect.x = pts[i].x
			ref_x = rect.x

			if i == np.size(pts)-1:
				rect.width = gray.cols - pts[i].x
				rects.push_back( rect )

	return rects





















