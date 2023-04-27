import cv2
import numpy as np
import random
import torch
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import torchvision.transforms.functional as F
from torchvision.utils import draw_keypoints
from torchvision.io import read_image
import torchvision.transforms as transforms

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

clothing_names = ["__background__", "Shorts","Dress","Swimwear","Brassiere","Tiara","Shirt","Coat","Suit",
              "Hat","Cowboy hat","Fedora","Sombrero","Sun hat","Scarf","Skirt","Miniskirt","Jacket",
              "Glove","Baseball glove","Belt","Necklace","Sock","Earrings","Tie","Watch","Umbrella",
              "Crown","Swim cap","Trousers","Jeans","Footwear","Roller skates","Boot","High heels",
              "Sandal","Sports uniform","Luggage and bags","Backpack","Suitcase","Briefcase",
              "Handbag","Helmet","Bicycle helmet","Football helmet"]

fcn_classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
counter = 0

def draw_semantic_segmentations(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        #plt.show()

    # Convert PLT to Opencv
    img_final = None
    fig.canvas.draw()
    for i in range(len(imgs)):
        b = fig.axes[i].get_window_extent()
        img = np.array(fig.canvas.buffer_rgba())
        if img_final is not None:
            img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
            img_final = np.concatenate((img_final, img), axis=1)
        else:
            img_final = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
    img_final = cv2.cvtColor(img_final, cv2.COLOR_RGBA2BGRA)
    img_final = cv2.resize(img_final, (img_final.shape[1]*2, img_final.shape[0]*2), interpolation = cv2.INTER_AREA)
    cv2.imshow('Detections', img_final)


def get_output_semantic(image, model, weights, objects):
	with torch.no_grad():
		output = model(image)["out"]
	normalized_masks = torch.nn.functional.softmax(output, dim=1)
	sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
	masks = [normalized_masks[0, sem_class_to_idx[cls]] for cls in objects]
	return masks

def get_outputs(image, model, threshold, model_name):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    if "masks" in outputs[0].keys():
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
    else:
        masks = []
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    #boxes = boxes[:thresholded_preds_count]

    # get the classes labels
    if model_name == 'clothing':
        labels = [clothing_names[i] for i in outputs[0]['labels']]
    else:
        labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels, scores


def draw_segmentation_map(image, masks, boxes, labels, classes, model_name):
    # define classes
    # get the classes labels
    if model_name == 'clothing':
        desired_classes = [clothing_names[i] for i in classes]
    else:
        desired_classes = [coco_names[i + 1] for i in classes]
    alpha = 1 
    beta = 0.5 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        if labels[i] in desired_classes:
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a randon color mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGB to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                          thickness=2)
            # put the label text above the objects
            cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=2, lineType=cv2.LINE_AA)
    return image


def draw_boxes(image, boxes, labels, classes, model_name, scores, threshold):
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # define classes
    if model_name == 'clothing':
        desired_classes = [clothing_names[i] for i in classes]
    else:
        desired_classes = [coco_names[i + 1] for i in classes]
    # apply a randon color mask to each object
    color = [0, 0, 255]
    #convert the original PIL image into NumPy format
    image = np.array(image)
    # convert from RGB to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(boxes[:thresholded_preds_count])):
        if labels[i] in desired_classes:
            # apply mask on the image
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)
            # put the label text above the objects
            cv2.putText(image, labels[i]+' '+ f'{scores[i]:.2f}', (boxes[i][0][0], boxes[i][0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=2, lineType=cv2.LINE_AA)
    return image


def convert_pil_opencv(image):
    image = np.array(image)
    # convert from RGB to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def get_output_keypoints(image, model):
	with torch.no_grad():
		outputs = model(image)
	kpts = outputs[0]['keypoints']
	scores = outputs[0]['scores']
	detect_threshold = 0.75
	idx = torch.where(scores > detect_threshold)
	keypoints = kpts[idx]
	return keypoints

connect_skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

def draw_keypoints_image(image, keypoints):
	transform = transforms.ToTensor()
	image = (transform(np.array(image).astype(np.uint8))*255).to(torch.uint8)
	image_keypoints = draw_keypoints(image, keypoints, colors="blue", connectivity=connect_skeleton, radius=3, width=3)
	img_conversion = cv2.cvtColor(np.transpose(image_keypoints.detach().numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
	cv2.imshow("Keypoints Detections", img_conversion)

def haar_face_detection(gray, args):
	detector = args["detector"]
	detections = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	return detections

def cnn_face_detection(image, args):
	h, w = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net = args["detector"]
	net.setInput(blob)
	detections = net.forward()
	return detections

def draw_haar_detections(image, rects, width=500):
	# loop over the bounding boxes
	image = imutils.resize(image, width=width)
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Faces detections", image)

def draw_cnn_faces_detections(image, detections, args):
	# loop over the detections
	h, w = image.shape[:2]
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["threshold"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	cv2.imshow("Faces detections", image)

def detect_face_keypoints(image, args):
    detector = args["keypoint_detector"]
    # read the image 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Displaying the Scanned Image by using cv2.imshow() method
    image = Image.fromarray(image).convert('RGB')
    bounding_boxes, conf, landmarks = args["keypoint_detector"].detect(image, landmarks=True)
    return bounding_boxes, conf, landmarks

def draw_face_keypoints(image, bounding_boxes, landmarks, conf):
    # draw the bounding boxes around the faces
    image_array = draw_bbox_keypoints(bounding_boxes, image)
    # plot the facial landmarks
    image_array = plot_landmarks(landmarks, image_array)
    cv2.imshow("Face Ladmarks", image_array)

def draw_bbox_keypoints(bounding_boxes, image):
    if bounding_boxes is not None:
        for i in range(len(bounding_boxes)):
            x1, y1, x2, y2 = bounding_boxes[i]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return image

def plot_landmarks(landmarks, image):
    if landmarks is not None:
        for i in range(len(landmarks)):
            for p in range(landmarks[i].shape[0]):
                cv2.circle(image, (int(landmarks[i][p, 0]), int(landmarks[i][p, 1])), 2, (0, 0, 255), -1, cv2.LINE_AA)
    return image

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def save_detections(image, detections, path, orig):
    global counter
    cv2.imwrite(os.path.join(path, str(counter)+".jpg"), image)
    cv2.imwrite(os.path.join(path, str(counter) + "_orig.jpg"), orig)
    if len(detections[0])>0:
        with open(os.path.join(path, str(counter)+".json"), 'w') as f:
            #f.write(str({"masks": detections[0].tolist(), "bboxes": detections[1], "labels": detections[2]}))
            json.dump({"bboxes": detections[1], "labels": detections[2], "confidence": detections[3]}, f, cls=NumpyFloatValuesEncoder)
    elif len(detections[1])>0:
        with open(os.path.join(path, str(counter)+".json"), 'w') as f:
            json.dump({"bboxes": detections[1], "labels": detections[2], "confidence": detections[3]}, f, cls=NumpyFloatValuesEncoder)
    counter += 1

