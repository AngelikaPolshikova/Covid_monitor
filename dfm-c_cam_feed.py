# Import the necessary packages and libraries
import numpy as np
import imutils
import time
import cv2
import os
import tensorflow as tf
from scipy.spatial import distance as dist
import time


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

def detect_and_predict_mask(frame, faceNet, maskDetectorModel, minConfidence=0.5, imageSize=224):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > minConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (imageSize, imageSize))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.stack(faces)
        predictions = maskDetectorModel.predict(faces)

    return (locations, predictions)

def detect_people(frame, detection_model):
    # Convert the frame to a tensor and make predictions
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)

    # Extract information about detected objects
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    #time.sleep(2.0)

    person_boxes = []
    for i in range(len(scores)):
            if classes[i] == 1 and scores[i] > 0.5:  # Class 1 corresponds to 'person' in COCO dataset
                box = boxes[i] * np.array([height, width, height, width])
                person_boxes.append(box.astype(int))

    return person_boxes

def calculate_distance(boxA, boxB, face_box):
    # Calculate the distance between the centroids of two bounding boxes
    (startYA, startXA, endYA, endXA) = boxA
    (startYB, startXB, endYB, endXB) = boxB
    (startYf, startXf, endYf, endXf) = face_box
    centerA = ((startXA + endXA) / 2, (startYA + endYA) / 2)
    centerB = ((startXB + endXB) / 2, (startYB + endYB) / 2)

    # Distance in pixels between the centers
    distance_pixels = dist.euclidean(centerA, centerB)

    # Convert pixel distance to real-world distance
    face_height_pixels = endYf-startYf
    distance_meters = (distance_pixels / face_height_pixels) * 0.2  # 20 cm is assumed face height

    return distance_meters

# Declare constants
faceDetectorPath = 'caffe_face_detector'
modelFile = 'mask_detector.h5'
minConfidence = 0.5
imageSize = 224

boxLabelWithMask = 'Wearing mask'
boxLabelWithoutMask = 'Not wearing mask'

# Load Caffe-based face detector model file
prototxtPath = os.path.sep.join([faceDetectorPath, "deploy.prototxt"])
weightsPath = os.path.sep.join([faceDetectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector model file
maskDetectorModel = load_model(modelFile)

# Load the SSD MobileNet v2 model for people detection
detection_model = tf.saved_model.load('efficientdet_d7_coco17_tpu-32/saved_model')

# Open camera feed with increased field of view
print("[LOG] Opening camera feed.")
vs = VideoStream(src=1).start()
# Set a higher resolution for wider FOV
vs.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Insert a 2 second buffer for initialization 
time.sleep(2.0)

i = 0
# Loop over the frames for the camera stream
# Loop over the frames for the camera stream
while True:
    t0 = time.time()
    # Get frame from the camera video stream
    frame = vs.read()
    t1 = time.time()
    total = t1 - t0
    # Call defined function to detect faces and predict presence of mask
    (locations, predictions) = detect_and_predict_mask(frame, faceNet, maskDetectorModel)

    # Check if any faces are detected
    if len(locations) == 0:
        cv2.imshow("Face Mask and People Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on Esc key press
            break
        continue  # Skip further processing if no faces are detected

    # Detect people in the frame
    person_boxes = detect_people(frame, detection_model)

    # Reduce person_boxes to those where a corresponding face is detected
    # This is a simplistic approach: assumes face detection is ordered and corresponds to people detection
    person_boxes = person_boxes[:len(locations)]

    # Create a mask status list to keep track of whether each person is wearing a mask
    mask_status = [prediction[0] > prediction[1] for _, prediction in zip(locations, predictions)]

    # Check for close proximity between people
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            distance_m = calculate_distance(person_boxes[i], person_boxes[j], locations[0])
            both_wearing_masks = mask_status[i] and mask_status[j]
            if distance_m < 1.0 and not both_wearing_masks:
                color = (255, 0, 255)  # Purple color box if they are too close without both wearing masks
            else:
                color = (255, 0, 0)  # Blue color box otherwise
            cv2.rectangle(frame, (person_boxes[i][1], person_boxes[i][0]), (person_boxes[i][3], person_boxes[i][2]), color, 2)
            cv2.rectangle(frame, (person_boxes[j][1], person_boxes[j][0]), (person_boxes[j][3], person_boxes[j][2]), color, 2)

    # Loop over the detected bounding boxes for the faces to display mask status
    for (box, prediction) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (withMask, withoutMask) = prediction
        label = "With Mask" if withMask > withoutMask else "No Mask"
        color = (50, 205, 50) if label == "With Mask" else (50, 50, 205)
        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Face Mask and People Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on Esc key press
    if key == 27:
        break

# Terminate all programs
cv2.destroyAllWindows()
vs.stop()

