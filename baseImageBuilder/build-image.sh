#!/bin/bash

echo ""

echo -e "\nbuild docker base image\n"
docker build -t aiclub.net/lr/ocr-base:1.0.0 .

echo ""