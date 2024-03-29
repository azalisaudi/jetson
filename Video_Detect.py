import cv2
import numpy as np
import time

thres = 0.45
nms_threshold = 0.2
cap = cv2.VideoCapture("Test_Video.mp4")
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
	success,img = cap.read()
	frame_id += 1
	classId, confs, bbox = net.detect(img,confThreshold= thres)
	bbox = list(bbox)
	confs = list(np.array(confs).reshape(1, -1)[0])
	confs = list(map(float,confs))
	print(type(confs))
	print(confs)

	indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
	print(indices)

	for i in indices:
		#i = i[0]
		box = bbox[i]
		x,y,w,h = box[0],box[1],box[2],box[3]
		print(x, y, w, h)
		cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
		#cv2.putText(img, classNames[classId [i][0]-1].upper(), (box[0] + 10, box[1] + 30),
		cv2.putText(img, classNames[classId [i]-1].upper(), (box[0] + 10, box[1] + 30), font,1,(0,255,0),2)

	elapsed_time = time.time() - starting_time
	fps = frame_id / elapsed_time
	cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
	cv2.imshow("Object Detection", img)
	key = cv2.waitKey(1)
	if key == 27 or key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
