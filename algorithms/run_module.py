# 1) ssh into the pi from MacBook (get hostname of pi using 'hostname -I'): 
#       >> ssh pi@192.168.1.9 --- IP address send to iphone on boot from boot.py in /home/pi (on raspi)
#       password: raspberry
# 2) change directories to Documents/Disco
#
# 3) enter tmux session so you can close out terminal while the program runs:
#       >> tmux
#
# 4) Finally execute this script (run_module.py) with an extra 2 arguments for run time and picture period
# ex.   >> python run_module.py 20 60      (will run the module code for 20 minutes and take a picture every 60 seconds)
#
# 5) Exit tmux session if you want to close the ssh terminal:
#       >> crtl+b d
#
# if error occurs with PiCam trye turning off Pi, disconnecting and reconnecting the camera OR try re-enable piCam by:
# >> sudo raspi-config  -> enable camera --> enable
#
# To find IP address if not on the Pi with a monitor use Macbook connected to the same network as the Pi.
#       >> nmap -sn 128.237.241.0/24        (take IP address from Macbook, and replace last 3 digits of the IP with '0/24' to list all device IPs on the network)
#
# To install additional libraries on the Pi make sure to 'sudo pip install' the libraries
#
# To get wifi access to CMU network on pi register the Pi's Mac Address with: netreg.net.cmu.edu
#
# Disco Labs, April 2016
# Simon Kalouche

import dropbox
import os
import sys
import time
import ipdb
import picamera
import dropbox_comm as db # db.download(), db.upload()

# if internet timeout issues occur try uncommenting this
# import socket
# socket.setdefaulttimeout( 10 )  # timeout in seconds


############# MAIN ################
# run settings
runTime = float(sys.argv[1])*60       # input of minutes converted to seconds
picPeriod = float(sys.argv[2])        # seconds

file_name = 'photo_history.txt'

# get access token to authenticate session
dbx = dropbox.Dropbox('<access_code>')

# make camera object
camera = picamera.PiCamera()

# rotate camera 90 deg CCW to align with housing
camera.rotation = -90

startTime = time.time()
timeNow = 0.0
img_name = list()
picTimer = 0.0
lastPic = 0.0


# main run loop
while (timeNow < runTime):

    if (picTimer >= picPeriod) or (timeNow == 0.0):
        # capture photo and name it "year_month_day__hour_min_sec"
        img_name.append(str(time.strftime("%Y_%m_%d__%H_%M_%S")) + '.jpg')
        camera.capture(img_name[-1])

        # reset picTimer by updating lastPic time
        lastPic = time.time()

        # write to file. use type 'a' to append or 'w' to overwrite
        with open(file_name, "a") as file:
            file.write((img_name[-1]+'\n'))

        # upload the image that was just taken to dropbox
        db.upload(dbx,img_name[-1],'','',img_name[-1])

        # upload file to drop box and overwrite previous inventory count since the program is appending to the file
        db.upload(dbx,file_name,'','',file_name, True)

        # delete image from local directory as to not fill up local hard drive
        os.remove(img_name[-1])


    # calculate time taken to capture image and upload
    picTimer = time.time() - lastPic
    timeLast = timeNow
    timeNow = time.time() - startTime
    dt = timeNow - timeLast
    freq = 1/dt


