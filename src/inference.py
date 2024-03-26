from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch

def get_query_embedding(model, device, query_img_file):
    """
    给定查询图像文件路径，通过模型运行并返回嵌入向量。

    Args:
        model           : 模型实例
        device          : cuda或cpu
        query_img_file  : 查询图像文件的路径

    Returns:
        嵌入向量的结果
    """
    model.eval()

    # 读取图像
    image = Image.open(query_img_file).convert("RGB")

    # 创建transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms_test = transforms.Compose([transforms.Resize(460),
                                        transforms.FiveCrop(448),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
                                        ])

    # 对图像应用transforms
    image = transforms_test(image)

    # 进行预测
    with torch.no_grad():
        # 将图像移到设备上并获取crops
        image = image.to(device)
        ncrops, c, h, w = image.size()

        # 获取输出
        output = model.get_embedding(image.view(-1, c, h, w))
        output = output.view(ncrops, -1).mean(0)

        return output
