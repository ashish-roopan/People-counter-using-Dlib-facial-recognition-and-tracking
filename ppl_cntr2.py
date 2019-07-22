from centroidtracker2 import CentroidTracker
from face_recognizer2 import FaceRecognizer
import numpy as np
import imutils
import dlib
import cv2
import urllib.request
import face_recognition


train_cnt=10
fce=FaceRecognizer()
ct = CentroidTracker(maxDisappeared=10, maxDistance=100)
img_count=np.zeros((5),dtype=int)
Unknown_images=np.zeros((5,train_cnt,500,500,3),dtype=np.uint8)
Unknown_images_bbox=np.zeros((5,train_cnt,4),dtype=np.uint32)
skip_frame=5
frame_cnt=0
tag=1
det_flag=0
i=0
j=0
enter=0
#cap=cv2.VideoCapture(2)
########################main_loop###################################
while True:
	#ret,frame = cap.read()
	# if not ret :
	# 	continue
	with urllib.request.urlopen('http://100.74.244.183:8080/shot.jpg') as url:
		imgNp=np.array(bytearray(url.read()),dtype=np.uint8)
		frame=cv2.imdecode(imgNp,-1)
	cv2.imshow('dd',frame)
	#frame = imutils.resize(frame, width=500)
	frame=cv2.resize(frame,(500,500))
	frame=cv2.flip(frame,1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	status = "Waiting"
	rects = []
  ##########detecting#################################################
	if frame_cnt%skip_frame==0:
		status = "Detecting"
		trackers = []
		img,boxes,names=fce.get_faces(frame)
		if len(boxes)>=1:
			det_flag=1
		else:
			det_flag=0
		for box,name in zip(boxes,names):
				rects.append([box[0],box[1],box[2],box[3],name])
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(box[0],box[1],box[2],box[3])
				tracker.start_track(rgb, rect)
				trackers.append(tracker)
  ############tracking################################################
	else:
		for (tracker, name) in zip(trackers, names):
			status = "Tracking"
			tracker.update(rgb)
			pos = tracker.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			rects.append([startX, startY, endX, endY,name])
	objects = ct.update(rects)

  #############Processing###########################################
	# IDs=[]
	# for (objectID, centroid) in objects.items():
	# 	IDs.append(objectID)
	for (objectID, centroid) in objects.items():
		w=abs(int(centroid[5])-int(centroid[3]))
		h=abs(int(centroid[4])-int(centroid[6]))
		if centroid[2]=='Unknown' and frame_cnt%skip_frame==0 and det_flag==1 and abs(h-w)<3:
			if img_count[objectID]<train_cnt:
				Unknown_images[objectID][img_count[objectID]]=frame
				Unknown_images_bbox[objectID][img_count[objectID]]=[int(centroid[4]),int(centroid[5]), int(centroid[6]),int(centroid[3])]
				img_count[objectID]+=1

		cv2.putText(frame,str(centroid[2]), (int(centroid[0]),int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 0), 2)
		cv2.rectangle(frame,(int(centroid[3]),int(centroid[4])), (int(centroid[5]),int(centroid[6])), (0, 0, 0), 2)


		if img_count[objectID]==train_cnt and enter==0:
			enter=1

			for k in range(10):
				a=Unknown_images[objectID][k]
				roi=a[Unknown_images_bbox[objectID][k][0]:Unknown_images_bbox[objectID][k][2],Unknown_images_bbox[objectID][k][3]:Unknown_images_bbox[objectID][k][1]]
				try:
					cv2.imshow(str(k),roi)
					cv2.waitKey(1)
				except:
					print('e')







	#############making new embedding#########################################3
	if img_count[i]==train_cnt:
		box=[(Unknown_images_bbox[i][j][0],Unknown_images_bbox[i][j][1],Unknown_images_bbox[i][j][2],Unknown_images_bbox[i][j][3])]

		fce.add_new_face(Unknown_images[i][j],'person'+str(tag),box)
		j+=1
		if j==train_cnt:
			print('yesssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
			j=0
			tag+=1
			img_count[i]=0
			if i==4:
				i=0
			else:
				i+=1
	else:
		if i==4:
			i=0
		else:
			i+=1

  ##############display#####################################################
	print('people count of one day=',tag-1)
	cv2.imshow('frame',frame)
	frame_cnt+=1
	if cv2.waitKey(1) ==ord('q'):
		break
