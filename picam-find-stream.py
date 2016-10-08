import cv2
import numpy as np
import sys
import time
import picamera.array
import picamera
import os
import re
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer

# Sane(?) values to see if a match is ok or not.
# "Works for me."
varOkDistance = 35
varOkPoints = 12

# Folder with items to search for
# ~300x~300 seems to work ok.
# Don't add too many. :)
pictureDir = "pictures"

### Classes

class CamHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    print self.path
    if self.path.endswith('.mjpg'):
      self.send_response(200)
      self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
      self.end_headers()
      while True:
        try:
          #rc,img = capture.read()
          #if not rc:
          #  continue
#          imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


          video_capture = picamera.array.PiRGBArray(camera, size=(640, 480))
          camera.capture(video_capture, format="bgr", use_video_port=True)
          imgRGB = video_capture.array

          img_gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
          kp0, des0 = orb.detectAndCompute(img_gray,None)
          bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

          for i in range(find_count):
            matches = bf.match(des[i],des0)
            matches = sorted(matches, key = lambda x:x.distance)
            imgRGB = my_drawMatches(imgRGB,kp[i],kp0,matches[:varOkPoints],find_name[i])

          r, buf = cv2.imencode(".jpg",imgRGB)
          self.wfile.write("--jpgboundary\r\n")
          self.send_header('Content-type','image/jpeg')
          self.send_header('Content-length',str(len(buf)))
          self.end_headers()
          self.wfile.write(bytearray(buf))
          self.wfile.write('\r\n')
#          time.sleep(0.5)
        except KeyboardInterrupt:
          break
      return
    if self.path.endswith('.html') or self.path=="/":
      self.send_response(200)
      self.send_header('Content-type','text/html')
      self.end_headers()
      self.wfile.write('<html><head></head><body>')
      self.wfile.write('<img src="cam.mjpg"/>')
      self.wfile.write('</body></html>')
      return
### Functions

def my_drawMatches(img1, kp1, kp2, matches, text):
	gx0 = 0
	gx1 = 0
	gy0 = 0
	gy1 = 0
	count  = 0
	avg_x = 0
	avg_y = 0
	avg_dist = 0
	max_dist = 0
	min_dist = 100
	show_target = True
	for mat in matches:
		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		# Find average point in the cloud
		count+=1
		avg_x+=x2
		avg_y+=y2

		# Draw a small circle 
		cv2.circle(img1, (int(x2),int(y2)), 4, (255, 0, 0), 1)

		# Don't draw target if any of the point are "bad"
		if mat.distance > varOkDistance:
			show_target = False
		avg_dist+=mat.distance
		if mat.distance > max_dist:
			max_dist = mat.distance
		if mat.distance < min_dist:
			min_dist = mat.distance

		gx1 = int(count) * 10
		gy1 = int(mat.distance) * 3
		if gx0 != 0 and gy0 != 0:
			cv2.line(img1, (gx0, gy0), (gx1, gy1), (0, 255, 0), 1)
		gx0 = gx1
		gy0 = gy1
		#cv2.circle(img1, (int(count) * 10,int(mat.distance) * 3), 2, (0, 255, 0), 1)

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.line(img1, (0, int(varOkDistance * 3)), (200, int(varOkDistance * 3)), (0, 255, 0), 1)
	cv2.putText(img1, str(varOkPoints) + "@" + str(varOkDistance),
		(203,int(varOkDistance * 3) + 3), font, 0.4, (0, 255, 0), 1)
	cv2.line(img1, (0, int(25 * 3)), (25, int(25 * 3)), (0, 0, 255), 1)
	cv2.line(img1, (0, int(50 * 3)), (50, int(50 * 3)), (0, 0, 255), 1)
	cv2.line(img1, (0, int(75 * 3)), (25, int(75 * 3)), (0, 0, 255), 1)
	cv2.line(img1, (0, int(100 * 3)), (200, int(100 * 3)), (0, 0, 255), 1)

	if show_target and count > 0:
		avg_x = avg_x / count
		avg_y = avg_y / count
		avg_dist = avg_dist / count
		cv2.circle(img1, (int(avg_x),int(avg_y)), 30, (0, 0, 255), 7)
		cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 0, 0), 4)
		cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 255, 255), 2)

		dist_txt = "Min: " + str(int(min_dist)) + " Max: " + str(int(max_dist)) + " Avg: " + str(int(avg_dist))
		cv2.putText(img1, dist_txt, (int(avg_x)+27,int(avg_y)), font, 0.5, (0, 0, 0), 4)
		cv2.putText(img1, dist_txt, (int(avg_x)+27,int(avg_y)), font, 0.5, (0, 255, 255), 2)

	return img1
	#end of my_drawMatches()

### Main

camera = picamera.PiCamera()
global camera
camera.resolution = (640, 480)
camera.rotation = 180
camera.framerate = 32
#global video_capture
#video_capture = picamera.array.PiRGBArray(camera, size=(640, 480))

orb = cv2.ORB()

find_name = {}
img = {}
kp = {}
des = {}
find_count = 0

for f in os.listdir(pictureDir): 
	fileName = os.path.join(pictureDir, f)
	if os.path.isfile(fileName):
		print "Add file: " + f
		img[find_count] = cv2.imread(fileName, 0)
		kp[find_count], des[find_count] = orb.detectAndCompute(img[find_count],None)
		find_name[find_count] = re.sub('\..*$', '', f)
		find_count+=1

print "Learned: " + str(find_count)

#	if cv2.waitKey(1) & 0xFF == ord('q'):
#		break

# Keys
#	key = cv2.waitKey(1)
#	if key != -1:
#		if key == ord('q'):
#			break
#		elif key == 65361:	# Left
#			varOkPoints -= 1
#		elif key == 65362:	# Up
#			varOkDistance -= 1
#		elif key == 65363:	# Right
#			varOkPoints += 1
#		elif key == 65364:	# Down
#			varOkDistance += 1
#		else:	
#			print "Key: " + str(key)

try:
	server = HTTPServer(('',9090),CamHandler)
	print "server started"
	server.serve_forever()
except KeyboardInterrupt:
	#capture.release()
        video_capture.release()
	server.socket.close()

