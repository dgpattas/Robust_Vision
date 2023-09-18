from multiprocessing.connection import wait
from fastapi import FastAPI, File, Header, Request, UploadFile
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
import os
from threading import Thread
from goprocam import GoProCamera
from goprocam import constants
from open_gopro import GoPro
import socket
import cv2
from inference import DehazingModel
import torch
from network import DRN
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import json
from mobvit_small import EncoderDecoder
from pipeline_delivering.pipeline import load_model
import read_and_detect_deep_learning
import body, read_frames
from ultralytics import YOLO



app = FastAPI()

@app.get("/denoise_on")
def change():
    os.environ["denoise_flag"] = "1"
    return 

@app.get("/denoise_off")
def change():
    os.environ["denoise_flag"] = "0"
    return 

@app.get("/detect_on")
def change():
    os.environ["detect_flag"] = "1"
    return 

@app.get("/detect_off")
def change():
    os.environ["detect_flag"] = "0"
    return 

@app.get("/3d_on")
def change():
    os.environ["3d_flag"] = "1"
    return 

@app.get("/3d_off")
def change():
    os.environ["3d_flag"] = "0"
    return 

@app.get("/clothes")
def change():
    os.environ["detect"] = "clothes"
    return 

@app.get("/foot")
def change():
    os.environ["detect"] = "foot"
    return 

@app.get("/yolo")
def change():
    os.environ["detect"] = "yolo"
    return 

@app.get("/dehaze")
def change():
    os.environ["denoise"] = "dehaze"
    return 

@app.get("/derain")
def change():
    os.environ["denoise"] = "derain"
    return 

@app.get("/lowlight")
def change():
    os.environ["denoise"] = "lowlight"
    return 

@app.get("/bbox")
def get_bounding_box():
    msg = subscribe.simple('fromtool-sense-vision', hostname=os.environ['ip'])
    myGlobalMessagePayload  = msg.payload.decode()
    data = json.loads(myGlobalMessagePayload)
    return data['infoprioPayload']['toolPayload'][0]


def connect_to_gopro():
    gopro = GoPro()
    gopro.open()
    goprocamera = GoProCamera.GoPro(constants.gpcontrol)
    goprocamera.video_settings(res='1080p', fps='15')
    goprocamera.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    goprocamera.livestream("start")
    #Comment lines 89-91 in KeepAlive
    #Change Line 96 to sock.sendto(keep_alive_payload, ("10.5.5.9", 8554))
    Thread(target=GoProCamera.GoPro.KeepAlive, args=(goprocamera,), daemon=True).start()


if __name__ == "__main__":

    # Care values must all be strings

    # Denoise flag "0" no denoising , "1" denoising
    os.environ['denoise_flag'] = "0"

    # Denoise models, Options are "lowlight" , "dehaze" , "derain" 
    os.environ['denoise'] = "lowlight"


    # Detection flag 0 no detections , 1 detections
    os.environ["detect_flag"] = "1"

    # What detection to use, Options are "yolo" , "foot" , "clothes"
    os.environ["detect"] = "yolo"

    # Depth flag, Only works with "yolo"
    os.environ["3d_flag"] = "1"


    # ip of broker
    os.environ['ip'] = "127.0.0.1"


    #Establish camera Connections
    connect_to_gopro()

    #Start gopro streaming
    local_url = "udp://127.0.0.1:8554"
    vidcap = cv2.VideoCapture(local_url, cv2.CAP_FFMPEG)
    
    #Load Models
    device = torch.device("cuda:" + str(0) if (torch.cuda.is_available() and int(0) >= 0) else "cpu")
    denoise_model = []
    detect_model = []


    #Dehaze
    path = "./code/weights_dehazing/17_model_G.pth"
    #Change on line 24 with [0] so it returns only the image without the depthmap
    denoise_model.append(DehazingModel(path, use_attention=True, ngf=32).to(device))

    #Derain
    derain_model = DRN(channel=3, inter_iter=3, intra_iter=3, a_S=0.5, use_GPU=True)
    derain_model = derain_model.cuda()
    #Comment on line 189
    derain_model.load_state_dict(torch.load("./code/weights_derain/net_latest.pth"))
    denoise_model.append(derain_model.eval())

    #LowLight
    model_root = './code/SCI/weights'
    gpu_id = 0
    # degree of correction
    # if both are null only Local Scale Histogram Stretch is applied
    ue_degree = 'maximum'
    oe_degree = 'easy'
    # output = pipeline(image = image, gpu_id = gpu_id, model_root =  model_root, wb = False, ue_degree = ue_degree, oe_degree = oe_degree)
    model_uexp, model_oexp = load_model(model_root, gpu_id, ue_degree, oe_degree)
    denoise_model.append([model_uexp, model_oexp])
    ### Old ###
    # path = "./code/weights_lowlight/weights_latest.pt"
    # #comment on line 158
    # denoise_model = Finetunemodel(path).to(device)

    #Yolo
    # For yolo, common.py line 563 classes
    detect_model.append(torch.hub.load("./code/yolov5-master", 'custom', 
        path="./code/yolov5s.pt", source="local")) #'yolov5s', pretrained=True) #, trust_repo=True
    
    # UPM's holes and footprints detection
    detect_model.append(YOLO("./code/upm_detect_model/pothole_footprints.pt"))

    # # UPM's clothes detector
    # num_classes = 45
    # detect_model_clothes = read_and_detect_deep_learning.get_model(num_classes)
    # detect_model_clothes.load_state_dict(torch.load("./code/upm_detect_model/my_model.pth"))
    # # load the modle on to the computation device and set to eval mode
    # detect_model.append(detect_model_clothes.to(device).eval())

    ### Depth
    depth_model = EncoderDecoder(batch_size=1)
    depth_model.cuda()
    depth_model.load_state_dict(torch.load('./code/weights_depth/mobilenet_small_MSE_l2=0.0001_bsize=8.pth'))
    depth_model.eval()

    # Connect to broker
    client = mqtt.Client('fromtool-sense-vision')
    client.connect(os.environ['ip'])

    #Start separate thread for pipeline
    # Denoise 0:dehaze 1:derain 2:lowlight
    # Detection 0:yolo 1:holes+footprint 2:sclothes
    Thread(target=read_frames.main, args=(vidcap,)).start()
    Thread(target=body.main, args=(vidcap,denoise_model,detect_model,device,depth_model)).start()
    # body.main(vidcap,denoise_model,detect_model,device)

    #Run the server
    uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=5500
)
