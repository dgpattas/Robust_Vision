# python read_and_detect_deep_learning.py --input telediario.mp4 --analyze video --model keypoints
import torch
import torchvision
import argparse
import glob
from utils_upm import *
from torchvision.transforms import transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
	# load a model pre-trained on COCO
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model

# input arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', required=True, help='camera (integer), video path or image path')
# parser.add_argument('-o', '--output', required=True, help='output path')
# parser.add_argument('-t', '--threshold', default=0.5, type=float, help='score threshold for discarding detection')
# parser.add_argument('-c', '--classesfcn', default="person,boat,dog", type=str, help='classes for fcn model separated by commas')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
# args = vars(parser.parse_args())

# save_path = args['output']
# os.makedirs(args['output'], exist_ok=True)

# num_classes = 45
# model = get_model(num_classes)
# model.load_state_dict(torch.load("my_model.pth"))
# # initialize the model
# weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# transforms = weights.transforms()


# # set the computation device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # args["device"] = device
# # load the modle on to the computation device and set to eval mode
# model.to(device).eval()

# # Function to preprocess image and perform detections
# def detection_function(image, args):
# 	# Process image
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 	# Displaying the Scanned Image by using cv2.imshow() method
# 	image = Image.fromarray(image).convert('RGB')
# 	# keep a copy of the original image for OpenCV functions and applying masks
# 	orig_image = image.copy()
# 	# transform the image
# 	image = transforms(image)
# 	# add a batch dimension
# 	image = image.unsqueeze(0).to(args["device"])

# 	# Perform detection depending on the model
# 	masks, boxes, labels, scores = get_outputs(image, model, args['threshold'], 'clothing')
# 	if len(masks)>0:
# 		result = draw_segmentation_map(orig_image, masks, boxes, labels, list(args['classes']), 'clothing')
# 	elif len(boxes)>0:
# 		result = draw_boxes(orig_image, boxes, labels, list(args['classes']), 'clothing', scores, args['threshold'])
# 	else:
# 		result = convert_pil_opencv(orig_image)
# 	# visualize the image
# 	cv2.imshow('Image detections', result)
# 	save_detections(result, [masks, boxes, labels, scores], save_path, convert_pil_opencv(orig_image))


# # Read images or video
# if os.path.isdir(args["input"]):
# 	print('READING DIRECTORY')
# 	images_list = glob.glob(args["input"] + "/*jpg") + glob.glob(args["input"] + "/*png")
# 	for img_path in images_list:
# 		print(img_path)
# 		image = cv2.imread(img_path)
# 		detection_function(image, args)
# 		cv2.waitKey(1)
# elif os.path.isfile(args["input"]):
# 	if args["input"].lower().endswith(("bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo")):
# 		image = cv2.imread(args["input"])
# 		detection_function(image, args)
# 		cv2.waitKey(1)
# 	elif args["input"].lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
# 		cap = cv2.VideoCapture(args["input"])
# 		while True:
# 			ret, image = cap.read()
# 			if ret:
# 				detection_function(image, args)
# 				# pess escape to exit
# 				if cv2.waitKey(1) == ord("q"):
# 					break
# 		cap.release()
# else:
# 	if 'http' in args["input"]:
# 		cap = cv2.VideoCapture(args["input"])
# 	else:
# 		cap = cv2.VideoCapture(int(args["input"]))
# 	while True:
# 		ret, image = cap.read()
# 		if ret:
# 			detection_function(image, args)
# 			# pess escape to exit
# 			if cv2.waitKey(1) == ord("q"):
# 				break
# 	cap.release()

# cv2.destroyAllWindows()