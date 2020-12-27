#!/bin/bash

echo ""

echo -e "\nbuild docker ocr:1.0.0 image\n"
docker build -t aiclub.net/lr/ocr:1.0.0 .

echo ""
