FROM aiclub.net/lr/ocr-base:1.0.0

MAINTAINER liurui

ENV TZ=Asia/Shanghai

RUN pip install -i https://pypi.douban.com/simple/ --upgrade pip wheel setuptools

ADD OpticalCharacterRecognition.tar.gz /opt

WORKDIR /opt/OpticalCharacterRecognition

EXPOSE 80

ENTRYPOINT ["python", "app.py"]
