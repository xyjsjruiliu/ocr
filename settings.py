# -*- coding: utf-8 -*-


# ocr 【settings】
# model path
OCR_MODEL_PATH = 'ocr/output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth'
# the thresh of post_processing
OCR_THRESH = 0.2
# output polygon or box
OCR_POLYGON = False
# show result
OCR_SHOW = False
# save box and score to txt file
OCR_SAVE_RESULT = False

# crnn 【settings】
# model path
CRNN_MODEL_PATH = 'crnn/output/360CC/crnn/2020-10-19-15-09/checkpoints/checkpoint_6_acc_0.9860.pth'
# model yaml
CRNN_YAML_PATH = 'crnn/lib/config/360CC_config.yaml'
