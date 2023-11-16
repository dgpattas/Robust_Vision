import time
import sys
import os
import json
import datetime
from torchvision import transforms as T
import paho.mqtt.client as mqtt
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import cv2
from torchvision import transforms
from pipeline_delivering.pipeline import inference
from utils_upm import *
from torchvision.transforms import transforms as transforms
import torchvision
import math
import paho.mqtt.subscribe as subscribe
import torchvision.ops as ops
from copy import deepcopy

                        

def main(vidcap,denoise,detect,device,depth_model):
    #Connect to mqtt
    transformss = transforms.ToPILImage()
    use_mqtt = True
    sourceid,PublishingTopic,ToolID,client = mqttConnect(use_mqtt)
    skip_steps = 1
    count = 0
    
    while True:
        # msg = subscribe.simple('fromtool-robust-vision-internal', hostname=os.environ['ip'])
        # decoded = cv2.imdecode(np.frombuffer(msg.payload, np.uint8), -1)

        image = vidcap.read()
        
        # Shoudl be converted to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        image, high_res_image = convert2tensor(image,device)
        
        # Run each inference, first denoise, then detections
        # each detection inference shapes the output detections to the required json format to be send to the broker
        det = {}
        totaldets = []
    
        ### new pipeline
        with torch.no_grad():

            if(os.environ['denoise_flag'] == '1'):

                if(os.environ["denoise"]=="derain"):
                    # image_t = Variable(image_t.cuda())
                    with torch.no_grad():
                        image = denoise[1](image)
                        image = torch.clamp(image,0.,1.)
                elif(os.environ["denoise"] == "lowlight"):

                    image = convert2tensor(inference(np.uint8(convert2numpy(image) * 255), denoise[2][0], denoise[2][1]),device)
                else:
                    #image = denoise[0](image)
                    #print("dehaze")
                    image = denoise[0].forward(image)

            # depthimage = torch.randn(size=(1,1,480,640)).float()
            if(os.environ["3d_flag"] == "1") :
                depthimage = depth_model.forward(image)

            if(os.environ['detect_flag'] == '1'):
                yoloxyxy = []
                yolocls = []
                yoloconf = []
                if 'yolovperson' in os.environ and os.environ['yolovperson']=='1':
                    #print("yoloperson")
                    xyxy,conf,cls = getDetectionsPersons(detect[2],high_res_image)
                    yoloxyxy+=xyxy
                    yolocls +=cls
                    yoloconf +=conf
                
                if 'yolov8' in os.environ and os.environ['yolov8']=='1':
                    #print("general yolo")
                    #print("yolov8")
                    xyxy,conf,cls = getDetectionsv8(detect[0],image)
                    yoloxyxy +=xyxy
                    yolocls  +=cls
                    yoloconf +=conf 

                if len(yoloxyxy)>0:
                    
                    selected_indices = keep_final_boxes(yoloxyxy,yoloconf,yolocls,threshold=0.5).tolist()
                    selected_xyxy = [yoloxyxy[i] for i in selected_indices]
                    selected_cls = [yolocls[i] for i in selected_indices]
                    selected_conf = [yoloconf[i] for i in selected_indices]

                    totaldets+=getdets(selected_xyxy,selected_conf,selected_cls,detect[0].names)

                if 'clothes' in os.environ and os.environ['clothes']=='1':
                    det = detect_clothes(image,detect[3])
                    if len(totaldets)>0:
                        for each in det:
                            totaldets = add_new_detection(totaldets,each,overlap_threshold=0.3)
                    else:
                        totaldets+= det

                if 'footprint' in os.environ and os.environ['footprint']=='1':
                    #print("footprint")
                    #det = detect_foot(np.uint8(convert2numpy(image) * 255),detect[1])
                    det = detect_foot(image,detect[1])
                    totaldets +=det

                if len(totaldets)>0:
                    totaldets = add_depth(totaldets,depthimage)
                else:
                    totaldets= []
            
        totaldets = convert_detections_CS(totaldets)
        
        # This function publishes the detections
        mqttConn(totaldets,sourceid,PublishingTopic,ToolID,client)


def add_depth(dets,depth_image):
    newdets = []
    for bbox_info in dets:
        
        x_center = bbox_info['relative_x_center']  # Absolute X-coordinate in pixels
        y_center = bbox_info['relative_y_center']  # Absolute Y-coordinate in pixels
        box_width = bbox_info['box_width']
        box_height = bbox_info['box_height']

        # Calculate the pixel coordinates of the bounding box
        x1 = int(x_center - box_width / 2)
        x2 = int(x_center + box_width / 2)
        y1 = int(y_center - box_height / 2)
        y2 = int(y_center + box_height / 2)

        # Extract depth values within the ROI
        depth_roi = depth_image[:, :, y1:y2, x1:x2]

        # Calculate the average depth within the ROI
        average_depth = torch.mean(depth_roi)
        bbox_info['depth'] = average_depth.item()
        newdets.append(bbox_info)
    return newdets


def calculate_overlap(box1, box2):
    # Calculate the overlap area between two bounding boxes
    x1 = max(box1['relative_x_center'] - box1['box_width'] / 2, box2['relative_x_center'] - box2['box_width'] / 2)
    y1 = max(box1['relative_y_center'] - box1['box_height'] / 2, box2['relative_y_center'] - box2['box_height'] / 2)
    x2 = min(box1['relative_x_center'] + box1['box_width'] / 2, box2['relative_x_center'] + box2['box_width'] / 2)
    y2 = min(box1['relative_y_center'] + box1['box_height'] / 2, box2['relative_y_center'] + box2['box_height'] / 2)
    
    overlap_width = max(0, x2 - x1)
    overlap_height = max(0, y2 - y1)
    
    overlap_area = overlap_width * overlap_height
    box1_area = box1['box_width'] * box1['box_height']
    
    overlap_ratio = overlap_area / box1_area
    return overlap_ratio

def add_new_detection(existing_dets, new_det, overlap_threshold=0.5):
    # Check if the new detection has an overlap greater than the threshold with any existing detection
    for existing_det in existing_dets:
        if existing_det['labelId']=='person':
            overlap = calculate_overlap(existing_det, new_det)
            if overlap > overlap_threshold:
                return existing_dets  # Do not add the new detection
        else:
            continue
        
    # If no overlap is found, add the new detection to the list
    existing_dets.append(new_det)
    return existing_dets






def keep_final_boxes(totalboxes,totalscores,totalclasses , threshold=0.5):

    
    totalboxes = [torch.tensor(each).reshape(-1,4).float() for each in totalboxes]
    
    totalscores = torch.tensor(totalscores)
    totalclasses = torch.tensor(totalclasses)
    


    totalboxes = torch.cat(totalboxes,dim=0)
    #totalscores = torch.cat(totalscores,dim=0)
    
    selected_indices = ops.batched_nms(totalboxes, totalscores,totalclasses,threshold)
    #selected_boxes = totalboxes[selected_indices]
    return selected_indices



def convert_detections_CS(det):
    new_det = []
    positionWeight = 110
    scaleWeight = 500
    for detection in det:
        if "depth" in detection:
            pos_z = detection["depth"]
            depth = detection['depth']
        else:
            pos_z = 4
            depth = 4
        pos_x = ((detection["relative_x_center"]-(640/2))/positionWeight)*pos_z/4
        width = detection["box_width"]/scaleWeight
        height = detection["box_height"]/scaleWeight
        pos_y = ((-1)*(detection["relative_y_center"]- (480/2))/positionWeight)*pos_z/4
        # width = width * depth/4
        # height = height * depth/4
        new_det.append({"labelId": detection["labelId"], "pos_x": pos_x, "pos_y": pos_y, "pos_z": pos_z, "width": width, "height": height, "confidence": detection["confidence"]})
    return new_det


def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

def getDepth(depth_model,image,det):
    input = torch.autograd.Variable(image)
    output = depth_model.forward(input)
    output = inverse_depth_norm(output)
    # output = output/10
    output = torch.squeeze(output,0)
    # output = output*255
    # output= output.repeat(3, 1, 1)
    # output= torch.tensor(output, dtype=torch.uint8).cpu()

    det_new = []
    for x in det:
        a = x
        a['depth'] = torch.mean(output[0,round(x['relative_y_center']-x['box_height']/2):round(x['relative_y_center']+x['box_height']/2),\
                                        round(x['relative_x_center']-x['box_width']/2):round(x['relative_x_center']+x['box_width']/2),]).item()
        det_new.append(a)
    return det_new

def getDetections(detect_model,img) -> None:
    transform = T.ToPILImage()    
    img_pil = transform(img.squeeze(0))
    out_detections = detect_model(img_pil)
    
    json_data = out_detections.pandas().xywh[0]
    
    col = json_data.pop('labelId')
    json_data.insert(0, 'labelId', col)
    json_data = json_data.drop(["class"], axis=1)
    json_data = json_data.to_dict(orient = "records")
    
    
    return json_data

def getdets(xyxy,conf,cls,names):

    detections = []
    for x in zip(cls,xyxy,conf):
        
        x_center = (x[1][0] + x[1][2])/2
        y_center = (x[1][1] + x[1][3])/2
        height = x[1][3] - x[1][1]
        width = x[1][2] - x[1][0]

        
        detections.append({'labelId': names[int(x[0])], 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
    return detections

def getDetectionsPersons(detect_model,img):
    # names = detect_model.names
    # width = img.shape[-1]
    # detections = []

    results = detect_model.predict(source=img,half=True,classes =[0],verbose=False)

    xyxy = results[0].boxes.xyxy.tolist()
    newxyxy = []
    for each in xyxy:
        newxyxy.append([int(x/2) for x in each])
    xyxy = newxyxy
    conf = results[0].boxes.conf.tolist()
    cls = results[0].boxes.cls.tolist()
    
    return (xyxy,conf,cls)
   
    

def getDetectionsv8(detect_model,img):


    results = detect_model.predict(img,half=True,verbose=False,classes = [0,2,10,24,25,26,28,30,31,39,41,56,63,64,66,67])
    xyxy = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()
    cls = results[0].boxes.cls.tolist()
    
    return (xyxy,conf,cls)

def getDetectionsClothes(image,detect):
    # initialize the model
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    transforms = weights.transforms()
    # transform the image
    image = transforms(image.squeeze(0))
    # add a batch dimension
    image = image.unsqueeze(0)
    masks, boxes, labels, scores = get_outputs(image, detect, 0.5, 'clothing')
    detections = []
    for x in zip(labels,boxes,scores):
        x_center = (x[1][0][0] + x[1][1][0])/2
        y_center = (x[1][0][1] + x[1][1][1])/2
        height = x[1][1][1] - x[1][0][1]
        width = x[1][1][0] - x[1][0][0]
        detections.append({'labelId': x[0], 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
    return detections


def mqttConnect(use_mqtt):
    if use_mqtt:
        ToolID = 'SENSE-VISION'
        sourceid = "FR001#FR"
        BrokerIP= os.environ['ip'] #'localhost' #.
        PublishingTopic= 'fromtool-'+ ToolID.lower()         
        client = mqtt.Client(ToolID)
        client.connect(BrokerIP)
        return sourceid,PublishingTopic,ToolID,client

def mqttConn(msg,sourceid,PublishingTopic,ToolID,client) -> None:
    json_msg= {}
    json_msg['sourceID'] = sourceid
    json_msg['toolID'] = ToolID
    json_msg['broadcast'] = True  

    json_data = {}
    json_data['category'] = "RobustVision#VisualDetections3D"
    json_data['type'] = "3DPrediction"
    json_data['startTS'] = datetime.datetime.now().isoformat()


    json_data['toolData'] = msg
    json_msg['infoprioPayload'] = json_data  
    json_msg_pub = json.dumps(json_msg) 

    client.publish(PublishingTopic,json_msg_pub)

def convert2tensor(image,device) -> torch.tensor:
    
    high_res_image = Image.fromarray(image)
    image = Image.fromarray(image)
    w, h = image.size

    # if h>700:
    #     image = image.resize((w//2, h//2), Image.Resampling.BICUBIC)
    image = image.resize((640, 480))
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    high_res_image = high_res_image.resize((1280,960))
    high_res_image = transform(high_res_image).unsqueeze(0).to(device)
    return image,high_res_image

def convert2tensorFull(image,device) -> torch.tensor:
    image = Image.fromarray(image)
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    return image

def convert2numpy(image_t) -> np.ndarray:
    return image_t.detach().squeeze(0).permute(1,2,0).cpu().numpy()



def detect_clothes(img,model):
    detections = []
    names = model.names
    results = model.predict(img,verbose=False,half=True)
    xyxy = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()
    cls = results[0].boxes.cls.tolist()
    for x in zip(cls,xyxy,conf):
        
        x_center = (x[1][0] + x[1][2])/2
        y_center = (x[1][1] + x[1][3])/2
        height = x[1][3] - x[1][1]
        width = x[1][2] - x[1][0]
        detections.append({'labelId': names[int(x[0])], 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
    return detections


def detect_foot(img,model):
    #-v ${PWD}/code/body.py:/code/body.py
    detections = []
    results = model.predict(img,verbose=False,half=True)
    xyxy = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()
    cls = results[0].boxes.cls.tolist()
    for x in zip(cls,xyxy,conf):
        
        x_center = (x[1][0] + x[1][2])/2
        y_center = (x[1][1] + x[1][3])/2
        height = x[1][3] - x[1][1]
        width = x[1][2] - x[1][0]
        detections.append({'labelId': 'pothole' if x[0] == 0 else 'footprints', 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
    return detections
    
