# To Run:
#	- first execute 'run_module.py' by ssh into the Pi. Once the module is on and running then do the following:
#		- cd into local directory where inventory_tracking.py is located
#		- set fromDropBox: True to download pictures from Dropbox cloud (live pictures from the Pi module), False to use test pictures in local directory
#		- >> python inventory_tracking.py 20  													(if fromDropBox = True) runs for 20 minutes
#		- >> python inventory_tracking.py Test_Pics/Pair20/1.JPG Test_Pics/Pair20/6.JPG     	(if fromDropBox = False)
#
# TODO:
# 1) Download images automatically from itemMaster
#
# Disco Labs, April 2016
# Simon Kalouche

# Import Libraries
import numpy as np
import cv2
import sys 			# for using command line inputs
import ipdb			# use ipdb.set_trace() to open debugger at a certain point and observe current variable/work space
import urllib		# for scraping images from web
import glob 		# for importing all pics in a folder
import copy
from pyicloud import PyiCloudService # for sending data to icloud (https://pypi.python.org/pypi/pyicloud)
import re 			# for removing text from strings (for sending messages to iphone)
import dropbox_comm as db # db.download(), db.upload()
import dropbox
import time
from termcolor import colored 	# for printing to terminal in colored text

# Run Options
send2iphone = False 		# send message to iphone? True = yes, False = no
disp_time = 10000			# milliseconds to display each picture with missing items for. Set to 0 for keystroke to clear the picture	
# fromDropBox = True 			# True to download pictures from Dropbox folder, False to use test pictures in local directory		

# automatically determine if the script should run on test pics or if it should download pics from dropbox
if len(sys.argv) <= 2:
	fromDropBox = True
else:
	fromDropBox = False
	runTime = .1	# set runTime to some short amount of time so the loop will only run once 


if fromDropBox:
	runTime = float(sys.argv[1])*60		# convert to minutes

# Define constants
CONTOUR_AREA_THRESH = 200
IMG_WIDTH = 1000.0
BLACK_WHITE_THRESH_LOWER = 30
BLACK_WHITE_THRESH_UPPER = 245
OUT_OF_STOCK_THRESH = 2  # number of features needed to make an assumption that there is still an item behind the one that was just taken

# file/folder names
file_name = 'inventory_count.txt'	# file name to save inventory list to (when something is taken off the shelf its published to this file)
local_pic_folder = 'Module_Pics/'	# local directory where pictures from dropbox are downloaded and saved too
photo_history_list = 'photo_history.txt'	# the text file from dropbox which lists all the picture names (i.e. dates and times) in order (bottom is most recent)

if send2iphone:
	# enter credentials to access the cloud
	api = PyiCloudService('<username>','<password>')
	iphone = api.devices['<device_id>']

# get access token to authenticate session
dbx = dropbox.Dropbox('<dropbox_auth_id>')


# # Scrape Images from Item Master to Create Database
# # Create an OpenerDirector with support for Basic HTTP Authentication...
# auth_handler = urllib.request.HTTPBasicAuthHandler()
# auth_handler.add_password(realm='PDQ Application',
#                           uri='https://mahler:8092/site-updates.py',
#                           user='<user_name>',
#                           passwd='<password>')
# opener = urllib.request.build_opener(auth_handler)
# # ...and install it globally so it can be used with urlopen.
# urllib.request.install_opener(opener)
# urllib.request.urlopen('https://www.itemmaster.com/')
# urllib.urlretrieve("http://www.digimouth.com/news/media/2011/09/google-logo.jpg", "local-filename.jpg")

# align two images that may be slightly off
def alignImages(im1, im2):
	# Convert images to grayscale
	im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

	# Find size of image1
	sz = im1.shape

	# Define the motion model
	warp_mode = cv2.MOTION_TRANSLATION

	# Define 2x3 or 3x3 matrices and initialize the matrix to identity
	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		warp_matrix = np.eye(3, 3, dtype=np.float32)
	else :
		warp_matrix = np.eye(2, 3, dtype=np.float32)

	# Specify the number of iterations.
	number_of_iterations = 5000;

	# Specify the threshold of the increment
	# in the correlation coefficient between two iterations
	termination_eps = 1e-10;

	# Define termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

	# Run the ECC algorithm. The results are stored in warp_matrix.
	(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

	if warp_mode == cv2.MOTION_HOMOGRAPHY :
		# Use warpPerspective for Homography 
		im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else :
		# Use warpAffine for Translation, Euclidean and Affine
		im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

	return im2_aligned



# Pre-process image to resize and rescale, convert to gray scale and increase contrast
def preProcess(img1, img2):
	# Image 1: rescale/resize image 
	desiredWidth = IMG_WIDTH 	#pixels
	height, width, depth = img1.shape
	imgScale = desiredWidth/width
	newX,newY = img1.shape[1]*imgScale, img1.shape[0]*imgScale
	img1 = cv2.resize(img1,(int(newX),int(newY)))	

	# convert to gray scale
	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	edges1 = cv2.Canny(gray1,140,145,apertureSize = 3)

	# Image 2: rescale/resize image 
	height, width, depth = img2.shape
	imgScale = desiredWidth/width
	newX,newY = img2.shape[1]*imgScale, img2.shape[0]*imgScale
	img2 = cv2.resize(img2,(int(newX),int(newY)))	

	# convert to gray scale and take Canny to get edges
	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	edges2 = cv2.Canny(gray2,140,145,apertureSize = 3)

	# increase constrast
	# create a CLAHE object (Arguments are optional)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray1 = clahe.apply(gray1)
	gray2 = clahe.apply(gray2)

	return img1, img2, gray1, gray2, edges1, edges2




############################################################################################
####################################### MAIN ###############################################
############################################################################################

startTime = time.time()
timeNow = 0.0

while (timeNow < runTime):

	if fromDropBox:

		## Method 2 to get photo names ------------------------------------
		dbx.files_download_to_file(photo_history_list,'/' + photo_history_list)

		# open the file with all the names of the photos taken from the Pi
		photoFile = open(photo_history_list,"r")

		# read lines in the text file and grab the the last two lines which are the names of the files to use for the CV analysis
		lineList = photoFile.readlines()
		photo_name_1 = str(lineList[-2][:-1])
		photo_name_2 = str(lineList[-1][:-1])
		photoFile.close()
		#-----------------------------------------------------------------

		# download the latest two photos that were taken from the Pi and uploaded to Dropbox
		dbx.files_download_to_file(( local_pic_folder + photo_name_1),('/'+ photo_name_1))
		dbx.files_download_to_file(( local_pic_folder + photo_name_2),('/'+ photo_name_2))

		# import photos into program after downloading them
		img1 = cv2.imread(local_pic_folder + photo_name_1)
		img2_unaligned = cv2.imread(local_pic_folder + photo_name_2)

	else:
		# load images from file
		img1 = cv2.imread(sys.argv[1])  	# instead of static pic name use sys.argv[1] and enter pic name in cmd line
		img2_unaligned = cv2.imread(sys.argv[2]) 

	# align second image to match first (in case camera moved ever so slightly)
	img2 = alignImages(img1, img2_unaligned)

	# pre process image to resize and rescale, convert to gray scale and increase contrast
	img1, img2, gray1, gray2, edges1, edges2 = preProcess(img1, img2)


	############################################################################################
	# Computer Vision Algorithms 

	# subtract images (edges and gray) to find the differences
	edgeDiff = edges1 - edges2
	grayDiff = gray1 - gray2

	# manual threshold to get rid of noise in the image: convert low confidence pixels in the grayDiff to black
	for row in range(grayDiff.shape[0]):		# rows
		for col in range(grayDiff.shape[1]):
			if grayDiff[row][col] > BLACK_WHITE_THRESH_UPPER or grayDiff[row][col] < BLACK_WHITE_THRESH_LOWER:
				grayDiff[row][col] = 255

	# cv2.imshow('a',grayDiff)
	# k = cv2.waitKey(0)

	# find countours in grayDiff (i.e. areas of high pixel concentration)
	thresh = cv2.threshold(grayDiff, 200, 255, cv2.THRESH_BINARY_INV)[1]
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]


	# calculate the area of each contour
	area = np.zeros(len(cnts))
	for i, contour in enumerate(cnts):
		area[i] = cv2.contourArea(contour)
		# print 'area: ', area[i]

	# filter contours by area
	cnts_filt_indx = [i for i,v in enumerate(area) if v > CONTOUR_AREA_THRESH]


	############################################################################################
	# Check for overlapping contours 
	for f_indx, findItem in enumerate(cnts_filt_indx):
		missing_item = cnts[findItem] 
		item_box = cv2.convexHull(missing_item)

		# convert polygon to rectangle for cropping
		xmin = item_box.min(axis = 0)[0][0]
		xmax = item_box.max(axis = 0)[0][0]
		ymin = item_box.min(axis = 0)[0][1]
		ymax = item_box.max(axis = 0)[0][1]

		# check to make sure that contours aren't overlapping. If they are get rid of the smaller one
		cnts_filt_indx_modified = copy.deepcopy(cnts_filt_indx)
		# remove current index from list of other contours to check only against other contours that may be overlapping (i.e. to prevent check of overlapping with itself)
		del cnts_filt_indx_modified[f_indx] 

		for c_indx, check_cnt in enumerate(cnts_filt_indx_modified):
			# take convex hull of each 
			check_overlap = cnts[check_cnt] 
			check_item_box = cv2.convexHull(check_overlap)

			# convert polygon to rectangle for cropping
			xmin_check = check_item_box.min(axis = 0)[0][0]
			xmax_check = check_item_box.max(axis = 0)[0][0]
			ymin_check = check_item_box.min(axis = 0)[0][1]
			ymax_check = check_item_box.max(axis = 0)[0][1]

			# check if any contours overlap with the current contour
			if (xmin_check >= xmin and xmin_check <= xmax and ymin_check >= ymin and ymin_check <= ymax) or (xmax_check == xmin or xmin_check == xmax or ymin_check == ymax or ymax_check == ymin) and (area[check_cnt] < area[findItem]):
				# contour is overlapping and enclosed (maybe not fully) by a larger contour. delete the smaller contour
				temp = np.array(cnts_filt_indx)
				del_indx = np.where(temp == check_cnt)[0]
				del cnts_filt_indx[del_indx[0]]


	# # draw all contours
	# cv2.drawContours(img2, cnts, -1, (240, 0, 159), 3)


	# # find the contour with the greatest area 
	# missing_item_indx = np.argmax(area)
	# missing_item = cnts[missing_item_indx] 
	# item_box = cv2.convexHull(missing_item)

	# # draw polyline  (green)
	# cv2.polylines(img2,[item_box],True,(0,255,0))

	# before drawing boxes on img2 make a copy of it for later to preserve raw (aligned) image
	img2_orig = copy.deepcopy(img2)

	# initialize image variables to be indexed as dictionaries
	crop_img = dict()
	behind_taken_item = dict()
	img3 = dict()
	img4 = dict()
	message = dict()

	############################################################################################
	# draw box around all missing items (mising item defined by contour with area > CONTOUR_AREA_THRESH).

	for f_indx, findItem in enumerate(cnts_filt_indx):
		missing_item = cnts[findItem] 
		item_box = cv2.convexHull(missing_item)

		# convert polygon to rectangle for cropping
		xmin = item_box.min(axis = 0)[0][0]
		xmax = item_box.max(axis = 0)[0][0]
		ymin = item_box.min(axis = 0)[0][1]
		ymax = item_box.max(axis = 0)[0][1]

		# show rectangle: rectangle(img, upperLeft, lowerRight, color, line_thickness)
		cv2.rectangle(img2, (xmin, ymax), (xmax,ymin), (255,0,0),2)

		# crop image to extract missing item only
		crop_img[f_indx] = img1[ymin:ymax, xmin:xmax]


		############################################################################################
		# Use SURF to find matching item from product/item database 

		# convert cropped image to gray scale
		gray_crop = cv2.cvtColor(crop_img[f_indx],cv2.COLOR_BGR2GRAY)

		# Create SURF object (cv2.xfeatures2d.SURF_create()) or SIFT Object (cv2.ORB_create). Set Hessian Threshold to 400
		surf = cv2.xfeatures2d.SURF_create()
		# surf.hessianThreshold = 50000

		# Find keypoints and descriptors directly
		kp, des = surf.detectAndCompute(gray_crop, None)

		# draw features on the image
		img3[f_indx] = cv2.drawKeypoints(gray_crop,kp,None,(255,0,0),4)


		############################################################################################
		# Match descriptors using BF (Brute Force) or Flann Matchers 
		
		# import all pictures from database 
		database = []
		img_names = dict()
		ii = 0 	# image indexer (ii)
		for imgS in glob.glob("database/*.png"):
		    n = cv2.imread(imgS,0)	# load image ii in gray scale
		    img_names[ii] = imgS
		    database.append(n)
		    ii += 1


		# initialize confidence metric which is a count of the number of features that matched between the sample database image and the missing item image
		confidence = np.zeros(len(database))

		# run feature matching algorithm on all the images in the database to see which image has the highest features matched (corrleation)
		for i, sample in enumerate(database):

			# detect important features in the sample image
			sample_kp, sample_des = surf.detectAndCompute(sample, None)

			# 1) Brute Force Method ####################
			# create BFMatcher object
			bf = cv2.BFMatcher()

			# Match descriptors.
			matches = bf.knnMatch(des,sample_des, k=2)

			# Sort them in the order of their distance.
			# matches = sorted(matches, key = lambda x:x.distance)

			# Apply ratio test
			good = []
			for m,n in matches:
				if m.distance < 0.75*n.distance:
					good.append([m])

			# count the number of good feature matches between database and cropped image. The image with the highest number is most likely to be the product that went out of stock
			confidence[i] = len(good)

		# Find which sample image best correlated (i.e. had most feature matches) with the mising item
		product_indx = np.argmax(confidence)



		############################################################################################
		# determine if product is now out of stock or if there are some left 
		# crop image to extract missing item only
		# ipdb.set_trace()
		# crop out portion of second image where item was removed. This is an image of the product behind the product that was removed from the shelf
		behind_taken_item[f_indx] = img2_orig[ymin:ymax, xmin:xmax]

		# convert cropped image to gray scale
		gray_behind_item = cv2.cvtColor(behind_taken_item[f_indx],cv2.COLOR_BGR2GRAY)

		# Detect important features in the cropped image of whats behind the item that was taken. 
		behind_kp, behind_des = surf.detectAndCompute(gray_behind_item, None)

		bf = cv2.BFMatcher()

		# check if features are found in the behind item image
		if behind_des is None:
			# no features found so there can be no feature matching
			behind_matches = []
		else:
			behind_matches = bf.knnMatch(des,behind_des, k=2)

		# Apply ratio test
		good = []
		for m,n in behind_matches:
			if m.distance < .6*n.distance:
				good.append([m])

		# count of the number of matches there were between the item taken and the item behind the item taken (if there is one). 
		# A low stock_confidence number means there weren't many feature matches and the product is most likely now out of stock
		stock_confidence = len(good)

		# If the stock_confidence is low then the item is more likely to be out-of-stock
		if stock_confidence < OUT_OF_STOCK_THRESH:
			out_of_stock = True
			# print name of the product which matches the item that went out of stock
			message[f_indx] = 'Out of Stock Item: ' + str(img_names[product_indx]) 
		else:
			out_of_stock = False
			# print name of the product which matches the item that went out of stock
			message[f_indx] = 'Item Removed From Shelf: ' + str(img_names[product_indx])

		# format string to be published to icloud/iphone. 
		message[f_indx] = re.sub('\.png$', '', message[f_indx])		# to get rid of last 4 characters you could also do: foo = foo[:-4]
		message[f_indx] = re.sub('database/', '', message[f_indx])
		message[f_indx] = re.sub('_', ' ', message[f_indx])
		
		# get string of date and time and add it to the message 
		time_string = str(time.strftime("%m.%d.%Y, %H:%M:%S  |  "))
		message[f_indx] = time_string + message[f_indx]

		print colored(message[f_indx],'red')

		# write to file. use type 'a' to append or 'w' to overwrite
		with open(file_name, "a") as file:
		    file.write((message[f_indx]+'\n'))

		# send message to iphone
		if send2iphone:
			iphone.display_message(message[f_indx])

		
		# debug
		# ipdb.set_trace()
		cv2.imshow('crop_behind',behind_taken_item[f_indx])
		cv2.imshow('crop',crop_img[f_indx])
		# cv2.drawMatchesKnn expects list of lists as matches.
		img5 = cv2.drawMatchesKnn(gray_crop,kp,gray_behind_item,behind_kp,good,None, flags=2)
		cv2.imshow('matches', img5)




		############################################################################################
		# Re-detect important features in the found product image (just for displaying purposes)
		product_kp, product_des = surf.detectAndCompute(database[product_indx], None)
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des,product_des, k=2)
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append([m])


		# cv2.drawMatchesKnn expects list of lists as matches.
		img4[f_indx] = cv2.drawMatchesKnn(gray_crop,kp,database[product_indx],product_kp,good,None, flags=2)


		# # 2) FLANN Method ########################### (FLANN is FASTER)
		# FLANN_INDEX_KDTREE = 0
		# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		# search_params = dict(checks=50)   # or pass empty dictionary

		# flann = cv2.FlannBasedMatcher(index_params,search_params)

		# ipdb.set_trace()
		# matches = flann.knnMatch(des,sample_des,k=2)

		# # Need to draw only good matches, so create a mask
		# matchesMask = [[0,0] for i in xrange(len(matches))]

		# # ratio test as per Lowe's paper
		# for i,(m,n) in enumerate(matches):
		#     if m.distance < 0.7*n.distance:
		#         matchesMask[i]=[1,0]

		# draw_params = dict(matchColor = (0,255,0),
		#                    singlePointColor = (255,0,0),
		#                    matchesMask = matchesMask,
		#                    flags = 0)

		# img4 = cv2.drawMatchesKnn(gray_crop,kp,sample,sample_kp,matches,None,**draw_params)



	############################################################################################
	# Display Image

	# show the edges only
	cv2.imshow('image1',img2)		# second image with item(s) missing

	for i in range(len(cnts_filt_indx)):
		# cv2.imshow(str(1+3*(i)),crop_img[i])  	# cropped item (indexed)
		# cv2.imshow(str(2+3*(i)),img3[i])		# cropped item with features highlighted (indexed)
		cv2.imshow(str(3+3*(i)),img4[i])		# feature matching between item and template from database (indexed)

	# keep image displayed for certain number of milliseconds. If 0 then wait for keystroke
	key = cv2.waitKey(disp_time)	


	############################################################################################
	# Dropbox

	# upload file to drop box and overwrite previous inventory count since the program is appending to the file
	db.upload(dbx,file_name,'','',file_name, True)




	# calculate loop time and refresh timeNow
	timeLast = timeNow
	timeNow = time.time() - startTime
	dt = timeNow - timeLast
	freq = 1/dt







