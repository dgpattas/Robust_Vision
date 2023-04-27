#!/bin/sh
sudo service bluetooth restart
/home/rescuer/venv/connect_camera/bin/python "./connect_camera.py"
