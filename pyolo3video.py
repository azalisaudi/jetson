import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + " " + str(round(confidence,2))
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
	_, frame = cap.read()
	frame_id += 1

	height, width, channels = frame.shape


	blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)    

	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

	for i in indices:
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		
		draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

	elapsed_time = time.time() - starting_time
	fps = frame_id / elapsed_time
	cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
	cv2.imshow("Image", frame)
	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()    
