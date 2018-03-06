# to adjust this program on boot, on the Pi's command line type:
# >> sudo crontab -e
#
# change/add the line at the end:
#
# @reboot /usr/bin/python /home/pi/boot.py >> /home/pi/out.log 2>&1

# The above line gives the path to python, and the path the the script to run.
# The part after the '>>' will generate a log file that prints errors to help debug the script that you're running
#
# Disco 2016
# Simon Kalouche

from pyicloud import PyiCloudService
import netifaces as ni
import time

# wait X seconds to let the internet start up
time.sleep(15)

# enter credentials to access the cloud
api = PyiCloudService('<username>','<password>')
iphone = api.devices['<device_id>']

# get IP address
ni.ifaddresses('eth0')
ip = ni.ifaddresses('wlan0')[2][0]['addr']


message = 'Disco IP: ' + str(ip)
print message
iphone.display_message(message)
