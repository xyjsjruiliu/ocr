# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import cv2
import time
import torch

import settings
import ocr_demo
import crnn_demo


basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
ALLOWED_EXTENSIONS = ['jpg', 'png']  # 允许上传的文件后缀


# 判断文件是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


class OCR(object):
    def __init__(self, pixel=4):
        """
        ocr 识别文字所在位置
        :param detect_word: 需要识别的文字
        :param pixel: 识别后的文字框扩展的大小, 默认扩展4个像素
        """
        self.pixel = pixel
        self.ocr_model = ocr_demo.Pytorch_model(settings.OCR_MODEL_PATH, post_p_thre=settings.OCR_THRESH, gpu_id=0)
        self.config, self.args = crnn_demo.parse_arg()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')

        self.crnn_model = crnn_demo.crnn.get_crnn(self.config).to(self.device)
        # se GPU
        checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            self.crnn_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.crnn_model.load_state_dict(checkpoint)

    def detect_word_by_image(self, detect_words, img_path):
        detected_boxes_list = self.ocr(img_path)
        # 监测到了多少个含有文字的框
        number_word_boxes = len(detected_boxes_list)
        output_json = []
        for i in range(number_word_boxes):
            result = self.crnn('save_img/' + str(i) + '.jpg')
            os.remove('save_img/' + str(i) + '.jpg')
            if result.find(detect_words) != -1:
                print(result, detected_boxes_list[i])
                output_json.append({"word": result,
                                    "top": detected_boxes_list[i][0],
                                    "height": detected_boxes_list[i][1],
                                    "left": detected_boxes_list[i][2],
                                    "width": detected_boxes_list[i][3]})
        return output_json

    def crnn(self, img_path):
        """
        识别出图像中的文字
        :param img_path:
        :return:
        """
        print('predit on image: {}'.format(img_path))
        img_raw = cv2.imread(img_path)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        converter = crnn_demo.utils.strLabelConverter(self.config.DATASET.ALPHABETS)

        result = crnn_demo.recognition(self.config, img, self.crnn_model, converter, self.device)
        cv2.waitKey(0)
        return result

    def ocr(self, img_path):
        """
        监测图片中文字所在位置
        :param img_path:
        :return:
        """
        # 初始化网络
        print('predit on image: {}'.format(img_path))
        # load image by open-cv
        img = cv2.imread(img_path)
        preds, boxes_list, score_list, t = self.ocr_model.predict(img_path, is_output_polygon=settings.OCR_POLYGON)

        img_width = img.shape[1]
        img_height = img.shape[0]

        i = 0
        detected_boxes_list = []
        for b in boxes_list:
            top = int(b[0][1]) - self.pixel
            left = int(b[0][0]) - self.pixel
            height = int(b[2][1]) - int(b[0][1]) + self.pixel * 2
            width = int(b[2][0]) - int(b[0][0]) + self.pixel * 2

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left
            height = img_height if top + height > img_height else top + height
            width = img_width if left + width > img_width else left + width

            detected_boxes_list.append([top, height, left, width])
            clip_img = img[top: height, left: width]
            cv2.imwrite('save_img/' + str(i) + '.jpg', clip_img)
            i += 1
        cv2.waitKey(0)
        return detected_boxes_list


app = Flask(__name__)
ocr_test = OCR()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/detectImg', methods=['GET'])
def detect_word():
    # get upload image and save
    image = request.files['image']
    words = request.args.get("word")
    path = basedir + "/upload_image/"
    file_path = path + image.filename
    image.save(file_path)
    print(file_path)
    # return jsonify(status=0)
    if file_path and allowed_file(file_path):  # 判断是否是允许上传的文件类型
        fname = file_path
        ext = fname.rsplit('.', 1)[1]  # 获取文件后缀
        unix_time = int(time.time())
        new_filename = str(unix_time) + '.' + ext  # 修改文件名
        # f.save(os.path.join(file_dir, new_filename))  # 保存文件到upload目录

        # video = Video(file_path)
        # output_json = video.pipe_line()
        output_json = ocr_test.detect_word_by_image(words, file_path)
        return jsonify({"status": 0, "msg": "上传成功", "data": output_json})
    else:
        return jsonify({"status": 1, "msg": "上传失败"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
    # started = time.time()
    # ocr_test.detect_word_by_image('ocr/imgs/8e476594583b496cbe630023e464fcde.jpg')
    # finished = time.time()
    # print('elapsed time: {0}'.format(finished - started))
