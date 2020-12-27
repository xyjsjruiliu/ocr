FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# MAINTAINER liurui
# #LABEL version="1.0"
# #指定系统编码

ENV LANG C.UTF-8

EXPOSE 80

#RUN sed -i s:/archive.ubuntu.com:/mirrors.aliyun.com/ubuntu:g /etc/apt/sources.list

# RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list

#RUN cat /etc/apt/sources.list && apt-get clean && apt-get -y update --fix-missing
RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y python-tk

RUN apt-get install python-opencv -y --allow-unauthenticated

# #安装需要的库

RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ Werkzeug==1.0.1 click==7.1.2 flask==1.1.2 itsdangerous==1.1.0 opencv-python==4.1.2.30 addict==2.4.0 pyclipper==1.2.0 shapely==1.7.1 easydict==1.9

RUN apt-get clean && rm -r ~/.cache/pip
