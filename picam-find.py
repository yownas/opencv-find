import cv2
import numpy as np
import sys
import time
import picamera.array
import picamera

varOkDistance = 35
varOkPoints = 12

### Functions

def my_drawMatches(img1, kp1, kp2, matches, text):
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
		cv2.circle(img1, (int(count) * 10,int(mat.distance) * 3), 2, (0, 255, 0), 1)

	cv2.line(img1, (0, int(varOkDistance * 3)), (200, int(varOkDistance * 3)), (0, 255, 0), 1)
	cv2.line(img1, (0, int(50 * 3)), (200, int(50 * 3)), (255, 255, 0), 1)
	cv2.line(img1, (0, int(100 * 3)), (200, int(100 * 3)), (0, 0, 255), 1)

	if show_target:
		avg_x = avg_x / count
		avg_y = avg_y / count
		avg_dist = avg_dist / count
		cv2.circle(img1, (int(avg_x),int(avg_y)), 30, (0, 0, 255), 7)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 0, 0), 4)
		cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 255, 255), 2)

		dist_txt = "Min: " + str(min_dist) + " Max: " + str(max_dist) + " Avg: " + str(avg_dist)
		cv2.putText(img1, dist_txt, (int(avg_x)+27,int(avg_y)), font, 0.5, (0, 0, 0), 4)
		cv2.putText(img1, dist_txt, (int(avg_x)+27,int(avg_y)), font, 0.5, (0, 255, 255), 2)

	return img1
	#end of my_drawMatches()

### Main

#video_capture = cv2.VideoCapture(0)

camera = picamera.PiCamera()
camera.resolution = (640, 480)
camera.rotation = 180
camera.framerate = 32
video_capture = picamera.array.PiRGBArray(camera, size=(640, 480))

orb = cv2.ORB()

find_name = {}

if len(sys.argv) > 1:
	find_name[1]=sys.argv[1]
else:
	find_name[1]='pibox.png'
img1 = cv2.imread(find_name[1], 0)
kp1, des1 = orb.detectAndCompute(img1,None)

if len(sys.argv) > 2:
	find_name[2]=sys.argv[2]
else:
	find_name[2]='spam.jpg'
img2 = cv2.imread(find_name[2], 0)
kp2, des2 = orb.detectAndCompute(img2,None)

# Warmup
time.sleep(0.5)

# Main loop
while True:
	#ret, img_rgb = video_capture.read()
	camera.capture(video_capture, format="bgr", use_video_port=True)
	img_rgb = video_capture.array

	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	kp0, des0 = orb.detectAndCompute(img_gray,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = bf.match(des1,des0)
	matches = sorted(matches, key = lambda x:x.distance)
	img_rgb = my_drawMatches(img_rgb,kp1,kp0,matches[:varOkPoints],find_name[1])

	matches = bf.match(des2,des0)
	matches = sorted(matches, key = lambda x:x.distance)
	img_rgb = my_drawMatches(img_rgb,kp2,kp0,matches[:varOkPoints],find_name[2])

	cv2.imshow('Find all the things', img_rgb)

	video_capture.truncate(0)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#video_capture.release()
cv2.destroyAllWindows()
