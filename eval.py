# -*- coding: utf-8 -*-
import cv2
import time
import torch

import settings
import ocr_demo
import crnn_demo


class OCR(object):
    def __init__(self):
        self.ocr_model = ocr_demo.Pytorch_model(settings.OCR_MODEL_PATH, post_p_thre=settings.OCR_THRESH, gpu_id=0)
        self.config, self.args = crnn_demo.parse_arg()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')

        self.crnn_model = crnn_demo.crnn.get_crnn(self.config).to(self.device)
        # se GPU
        #checkpoint = torch.load(self.args.checkpoint, map_location='cpu')
        checkpoint = torch.load(self.args.checkpoint)
        if 'state_dict' in checkpoint.keys():
            self.crnn_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.crnn_model.load_state_dict(checkpoint)

    def crnn(self, img_path):
        """
        recognition the image word
        :param img_path:
        :return:
        """
        print('predit on image: {}'.format(img_path))
        img_raw = cv2.imread(img_path)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        converter = crnn_demo.utils.strLabelConverter(self.config.DATASET.ALPHABETS)

        result = crnn_demo.recognition(self.config, img, self.crnn_model, converter, self.device)
        print(result)
        cv2.waitKey(0)

    def ocr(self, img_path):
        """
        detect word location in image
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
        for b in boxes_list:
            top = int(b[0][1]) - 4
            left = int(b[0][0]) - 4
            height = int(b[2][1]) - int(b[0][1]) + 8
            width = int(b[2][0]) - int(b[0][0]) + 8

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left
            height = img_height if top + height > img_height else top + height
            width = img_width if left + width > img_width else left + width

            clip_img = img[top: height, left: width]
            cv2.imwrite('save_img/' + str(i) + '.jpg', clip_img)
            i += 1

        cv2.waitKey(0)


if __name__ == "__main__":
    ocr_test = OCR()
    started = time.time()
    ocr_test.ocr('ocr/imgs/6.jpg')
    for i in range(41):
        ocr_test.crnn('save_img/' + str(i) + '.jpg')
    finished = time.time()
    #while(True):
    #    pass
    print('elapsed time: {0}'.format(finished - started))
