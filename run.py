from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import numpy as np,imutils,time,cv2,os

def detect_and_predict_face_mask(frame, faceNet, maskNet):
	# grab the dimensions
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detection = faceNet.forward()

	faces,predicitions,locations = [],[],[]

	# loop over the detections
	for i in range(0, detection.shape[2]):
		# extract the probability
		confidence = detection[0, 0, i, 2]

		# if greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the rectangle
			box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# convert img from BGR to RGB channel and resize it to 224x224
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face & rectangle to their respective list
			faces.append(face)
			locations.append((startX, startY, endX, endY))

	# if at least one face was detected
	if len(faces) > 0:		
		faces = np.array(faces, dtype="float32")
		predicitions = maskNet.predict(faces, batch_size=32)

	# return face locations
	return (locations, predicitions)


# load our serialized face detector model from disk
prototxtPath = os.path.sep.join(['face_detector_model', "deploy.prototxt"])
weightsPath = os.path.sep.join(['face_detector_model',
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model('mask_detector.model')

# initialize the video stream and allow the camera sensor to warm up
print("Starting WebCam.......")
vs = VideoStream(src=0).start()
time.sleep(3.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locations, predicitions) = detect_and_predict_face_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locations, predicitions):
		# unpack the rectangle and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the rectangle and text
		label = "Have Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Have Mask" else (0, 0, 255)

		# probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# `q` key for break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
