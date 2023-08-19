import cv2
import numpy as np
import time

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id]) + " " + str(round(confidence,2))
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# read class names from text file
classes = None
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
#net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

#image = cv2.imread("dog.jpg")
#Height, Width, _ = image.shape

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

font = cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("object detection", cv2.WINDOW_AUTOSIZE)
while True:
	_, image = cap.read()
	
	Height, Width, _ = image.shape
	
	# create input blob 
	blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), (0,0,0), True, crop=False)

	# set input blob for the network
	net.setInput(blob)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# run inference through the network
	# and gather predictions from output layers
	start = time.time()
	outs = net.forward(ln)
	end = time.time()
	# show timing information on YOLO
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	# initialization
	class_ids = []
	confidences = []
	boxes = []

	# for each detetion from each output layer 
	# get the confidence, class id, bounding box params
	# and ignore weak detections (confidence < 0.5)
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.2:
				center_x = int(detection[0] * Width)
				center_y = int(detection[1] * Height)
				w = int(detection[2] * Width)
				h = int(detection[3] * Height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])
				
	# apply non-max suppression
	conf_threshold = 0.5
	nms_threshold = 0.4
	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	# go through the detections remaining
	# after nms and draw bounding box
	for k in indices:
		i = k[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		
		draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
	
	# display output image    
	cv2.imshow("object detection", image)
	keyCode = cv2.waitKey(10) & 0xFF
	# Stop the program on the ESC key or 'q'
	if keyCode == 27 or keyCode == ord('q'):
		break
                    
# wait until any key is pressed
cv2.waitKey()
    
 # save output image to disk
cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()                
