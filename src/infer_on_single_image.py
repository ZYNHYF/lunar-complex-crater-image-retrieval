from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import TripletLoss, TripletNet, Identity, create_embedding_net, create_swin_embedding_net
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
import torch
from utils import draw_label, ap_at_k_per_query, get_preds, get_preds_and_visualize, perform_pca_on_single_vector, ap_per_query
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from inference import get_query_embedding

def measure_performance(labels_dir, img_dir, img_fts_dir, weights_file, subset="inference"):
    """
    给定权重文件，计算对应数据集所有查询的平均精度（Mean Average Precision）。

    Args:
        labels_dir  : GT标签的目录
        img_dir     : 存储图像的目录
        img_fts_dir : 存储通过create_db.py脚本生成的经PCA降维的特征的目录
        weights_file: 训练好的权重文件的路径
        subset      : 训练集/验证集/推断集

    Returns:
        所有查询的平均精度
    """
    # 创建QueryExtractor对象
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset=subset)

    # 创建图像数据库
    query_images = QUERY_EXTRACTOR.get_query_names()

    # 创建路径
    query_image_paths = [os.path.join(img_dir, file) for file in query_images]

    aps = []
    # 进行评估
    for i in query_image_paths:
        ap = inference_on_single_labelled_image_pca(query_img_file=i, labels_dir=labels_dir, img_dir=img_dir, img_fts_dir=img_fts_dir, weights_file=weights_file, plot=False)
        aps.append(ap)
    print(aps)

    return np.array(aps).mean()


def inference_on_single_labelled_image_pca(query_img_file, labels_dir, img_dir, img_fts_dir, weights_file, top_k=1000, plot=True):
    """
    计算给定查询图像的平均精度，同时绘制前20个结果。

    Args:
        query_img_file  : 查询图像文件的路径
        labels_dir  : GT标签的目录
        img_dir     : 存储图像的目录
        img_fts_dir : 存储通过create_db.py脚本生成的经PCA降维的特征的目录
        weights_file: 训练好的权重文件的路径
        top_k       : 用于计算平均精度的前k个值
        plot        : 如果为True，将绘制前20个结果

    Returns:
        查询图像文件的平均精度
    """
    # 创建CUDA参数
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("可用设备 = ", device)

    # 创建嵌入网络
    embedding_model = create_embedding_net()
    model = TripletNet(embedding_model)
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    model.eval()

    # 获取查询名称
    query_img_name = query_img_file.split("/")[-1]
    query_img_path = os.path.join(img_dir, query_img_name)

    # 创建QueryExtractor对象
    QUERY_EXTRACTOR = QueryExtractor(labels_dir, img_dir, subset="inference")

    # 创建查询Ground Truth字典
    query_gt_dict = QUERY_EXTRACTOR.get_query_map()[query_img_name]

    # 创建图像数据库
    QUERY_IMAGES_FTS = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_fts_dir))]
    QUERY_IMAGES = [os.path.join(img_fts_dir, file) for file in sorted(os.listdir(img_dir))]

    # 查询特征
    query_fts = get_query_embedding(model, device, query_img_file).detach().cpu().numpy()
    query_fts = perform_pca_on_single_vector(query_fts)

    # 创建相似性列表
    similarity = []
    for file in tqdm(QUERY_IMAGES_FTS):
        file_fts = np.squeeze(np.load(file))
        cos_sim = np.dot(query_fts, file_fts) / (np.linalg.norm(query_fts) * np.linalg.norm(file_fts))
        similarity.append(cos_sim)

    # 使用相似性获取最佳匹配
    similarity = np.asarray(similarity)
    indexes = (-similarity).argsort()[:top_k]
    best_matches = [QUERY_IMAGES[index] for index in indexes]

    # 获取预测结果
    if plot:
        preds = get_preds_and_visualize(best_matches, query_gt_dict, img_dir, 20)
        print(img_dir)
    else:
        preds = get_preds(best_matches, query_gt_dict)

    # 获取平均精度
    ap = ap_per_query(best_matches, query_gt_dict)

    return ap

if __name__ == '__main__':
    # measure_performance(labels_dir="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/gt_files",
    #                     img_dir="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/images",
    #                     img_fts_dir="/root/autodl-tmp/deep-image-retrieval-master/src/fts_pca/lunar",
    #                     weights_file="/root/autodl-tmp/deep-image-retrieval-master/src/weights/lunar-exp-3.pth")


    inference_on_single_labelled_image_pca(query_img_file="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/images/3_Central_peak_type_000086.jpg",
                                            labels_dir="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/gt_files",
                                            img_dir="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/images",
                                            img_fts_dir="/root/autodl-tmp/deep-image-retrieval-master/src/fts_pca/lunar_2",
                                            top_k=500,
                                            plot=True,
                                            weights_file="/root/autodl-tmp/deep-image-retrieval-master/src/weights/lunar-exp-3.pth")
