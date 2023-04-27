import os
import paho.mqtt.client as mqtt
import cv2


def main(vidcap):
    ToolName = 'Robust vision'
    ToolID = 'Robust-Vision-Internal'
    BrokerIP= os.environ['ip'] #'localhost' #.
    PublishingTopic= 'fromtool-'+ ToolID.lower()         
    client = mqtt.Client(ToolID)
    client.connect(BrokerIP)
    # return ToolName,PublishingTopic,ToolID,client

    skip_steps = 1
    count = 0

    while True:
        count += 1 
        success, image = vidcap.read()
        if success and count % skip_steps == 0:
            # Shoudl be converted to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            # image_bytes = byterarray(image)
            client.publish(PublishingTopic, image_bytes, 0)

