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

                        

def main(vidcap,denoise,detect,device,depth_model):
    #Connect to mqtt
    transformss = transforms.ToPILImage()
    use_mqtt = True
    sourceid,PublishingTopic,ToolID,client = mqttConnect(use_mqtt)
    count = 0
    
    while True:
        msg = subscribe.simple('fromtool-robust-vision-internal', hostname=os.environ['ip'])
        decoded = cv2.imdecode(np.frombuffer(msg.payload, np.uint8), -1)

        image = convert2tensor(decoded,device)

        # Run each inference, first denoise, then detections
        # each detection inference shapes the output detections to the required json format to be send to the broker
        det = {}
        if(os.environ['denoise_flag'] == '1'):
            if(os.environ["denoise"]=="derain"):
                # image_t = Variable(image_t.cuda())
                with torch.no_grad():
                    image = denoise[1](image)
                    image = torch.clamp(image,0.,1.)
            elif(os.environ["denoise"] == "lowlight"):

                image = convert2tensor(inference(np.uint8(convert2numpy(image) * 255), denoise[2][0], denoise[2][1]),device)
            else:
                image = denoise[0](image)
        if(os.environ['detect_flag'] == '1'):
            if(os.environ['detect'] == 'yolo'):
                det = getDetections(detect[0],image)
                if(os.environ["3d_flag"] == "1"):
                    det = getDepth(depth_model,image,det)
            elif(os.environ['detect'] == 'foot'):
                det = detect_foot(np.uint8(convert2numpy(image) * 255),detect[1])
            # elif(os.environ['detect'] == 'clothes'):
            #     det = getDetectionsClothes(image,detect[2])
            # det = {"detect": det}
        det = convert_detections_CS(det)
        # This function publishes the detections
        mqttConn(det,sourceid,PublishingTopic,ToolID,client)

def convert_detections_CS(det):
    new_det = []
    positionWeight = 110
    scaleWeight = 500
    for detection in det:
        if "depth" in detection:
            pos_z = detection["depth"]
        else:
            pos_z = 4
        pos_x = ((detection["relative_x_center"]-(640/2))/positionWeight)*pos_z/4
        width = detection["box_width"]/scaleWeight
        height = detection["box_height"]/scaleWeight
        pos_y = ((-1)*(detection["relative_y_center"]- (480/2))/positionWeight)*pos_z/4
        new_det.append({"label_Id": detection["label_Id"], "pos_x": pos_x, "pos_y": pos_y, "pos_z": pos_z, "width": width, "height": height, "confidence": detection["confidence"]})
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
    col = json_data.pop('label_Id')
    json_data.insert(0, 'label_Id', col)
    json_data = json_data.drop(["class"], axis=1)
    json_data = json_data.to_dict(orient = "records")
    return json_data

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
        detections.append({'label_Id': x[0], 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
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
    json_data['type'] = "Prediction"
    json_data['startTS'] = datetime.datetime.now().isoformat()


    json_data['toolData'] = msg
    json_msg['infoprioPayload'] = json_data  
    json_msg_pub = json.dumps(json_msg) 

    client.publish(PublishingTopic,json_msg_pub)

def convert2tensor(image,device) -> torch.tensor:
    image = Image.fromarray(image)
    w, h = image.size
    if h>700:
        image = image.resize((w//2, h//2), Image.Resampling.BICUBIC)
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    return image

def convert2tensorFull(image,device) -> torch.tensor:
    image = Image.fromarray(image)
    transform = T.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    return image

def convert2numpy(image_t) -> np.ndarray:
    return image_t.detach().squeeze(0).permute(1,2,0).cpu().numpy()

def detect_foot(img,model):
    #-v ${PWD}/code/body.py:/code/body.py
    detections = []
    results = model.predict(img)
    xyxy = results[0].boxes.xyxy.tolist()
    conf = results[0].boxes.conf.tolist()
    cls = results[0].boxes.cls.tolist()
    for x in zip(cls,xyxy,conf):
        print("detection found")
        x_center = (x[1][0] + x[1][2])/2
        y_center = (x[1][1] + x[1][3])/2
        height = x[1][3] - x[1][1]
        width = x[1][2] - x[1][0]
        detections.append({'label_Id': 'pothole' if x[0] == 0 else 'footprints', 'relative_x_center': float(x_center), 'relative_y_center': float(y_center), 'box_width': float(width), 'box_height': float(height), 'confidence': float(x[2])})
    return detections
    
