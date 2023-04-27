# Rescuer-s-Robust-Vision

Vicomtech has done the lowlight and UPM the holes+footprints and clothes detection. Certh has done the rest. 

This tool first loads all the models in main.py.  

Runs a separate thread which reads and saves all the streaming frames in a mqtt broker in read_frames.py

Then each frame is processed in body.py by denoising and then detections

Lastly, a fastapi python server is initialized which is to establish communication between the Hololens visualization app and the broker to read the detections. 
