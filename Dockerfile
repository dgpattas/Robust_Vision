# DRAK

# INHERIT FROM BASE IMAGE
FROM nvcr.io/nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# INSTALL PYTHON AND PIP
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && sudo apt upgrade -y; exit 0
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install python3.10 -y
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1
RUN apt install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# INSTALL THE REST OF THE PACKAGES
RUN apt install --no-install-recommends -y \
		iputils-ping \
		nano \
		python3-opencv \
		unzip \
		ffmpeg curl iproute2 net-tools \
		bluez \
    	dbus \
	&& apt clean

COPY code/requirements.txt code/requirements.txt

# INSTALL PYTHON PACKAGES
RUN pip install --upgrade pip
RUN pip install -r code/requirements.txt
RUN pip install numpy
RUN pip install pillow
RUN pip install argparse
RUN pip install opencv-python
RUN pip install plyfile
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install fastapi[all]
RUN pip install configargparse

RUN apt-get update && apt-get install -y sudo
RUN sudo pip install scikit-image
RUN pip install ultralytics

RUN pip install einops
RUN pip install timm==0.6.11

# MAKE CONTAINER FILESYSTEM
RUN mkdir action 
ADD exec /action
ADD code /code

RUN mkdir -p -v root/.cache/torch/hub/checkpoints
COPY code/weights_depth/mobilevit_xs-8fbd6366.pth root/.cache/torch/hub/checkpoints/mobilevit_xs-8fbd6366.pth
COPY code/weights_depth/mobilenet_small_MSE_l2=0.0001_bsize=8.pth root/.cache/torch/hub/checkpoints/mobilenet_small_MSE_l2=0.0001_bsize=8.pth



# PORT-FORWARDING
EXPOSE 8554/udp

# SET EXECUTION STARTING POINT
RUN chmod +rx /action/exec
ENTRYPOINT ["bash", "/action/exec"]
