#!/bin/bash

# 默认参数
CONFIG_PATH="all_codes/config.yaml"
IMAGE_PATH="data/CROHME/19_test_images"
LABEL_PATH="data/CROHME/19_test_labels.txt"



# 运行推理脚本
python all_codes/inference.py --config "$CONFIG_PATH" --image_path "$IMAGE_PATH" --label_path "$LABEL_PATH"
