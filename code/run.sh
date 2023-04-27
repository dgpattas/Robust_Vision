#!/bin/bash
# start services
service dbus start
service bluetooth start
python stream_integr_v2.py