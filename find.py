import cv2
import numpy as np
import sys

def drawMatches(img1, kp1, kp2, matches, text):
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    count  = 0
    avg_x = 0
    avg_y = 0
    show_target = True
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

	count+=1
	avg_x+=x2
	avg_y+=y2

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
#        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(img1, (int(x2),int(y2)), 4, (255, 0, 0), 1)

	
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
	col_b = 255
        col_g = 0
	if mat.distance < 50:
        	col_g = 255
	else:
		show_target = False
	col_r = 0
#        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (col_b, col_g, col_r), 1)


    if show_target:
	avg_x = avg_x / count
	avg_y = avg_y / count
	cv2.circle(img1, (int(avg_x),int(avg_y)), 30, (0, 0, 255), 7)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 0, 0), 4)
	cv2.putText(img1, text, (int(avg_x)+27,int(avg_y)-20), font, 1, (0, 255, 255), 2)

    # Also return the image if you'd like a copy
    return img1

video_capture = cv2.VideoCapture(0)
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

while True:
	ret, img_rgb = video_capture.read()
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	kp0, des0 = orb.detectAndCompute(img_gray,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	matches = bf.match(des1,des0)
	matches = sorted(matches, key = lambda x:x.distance)
	img_rgb = drawMatches(img_rgb,kp1,kp0,matches[:12],find_name[1])

	matches = bf.match(des2,des0)
	matches = sorted(matches, key = lambda x:x.distance)
	img_rgb = drawMatches(img_rgb,kp2,kp0,matches[:12],find_name[2])

	cv2.imshow('Track', img_rgb)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()
