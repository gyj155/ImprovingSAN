#!/bin/bash

# 默认参数
CONFIG_PATH="my_config.yaml"
IMAGE_PATH="/home/yjguo/SAN-main/data/CROHME/16_test_images"
LABEL_PATH="/home/yjguo/SAN-main/data/CROHME/16_test_labels.txt"



# 运行推理脚本
python inference.py --config "$CONFIG_PATH" --image_path "$IMAGE_PATH" --label_path "$LABEL_PATH"
