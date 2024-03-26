import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as compare_ssim

import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score


def draw_label(img_path, color_code):
    """
    给定 RGB 颜色码，绘制图像周围的矩形的函数

    Args:
        img_path    : 图像的路径
        color_code  : 矩形的颜色码

    Returns:
        带有绘制矩形的 numpy 数组
    """
    img = Image.open(img_path)
    np_ar = np.array(img)
    rects = [(0, 0, np_ar.shape[1], np_ar.shape[0])]
    for x, y, w, h in rects:
        cv2.rectangle(np_ar, (x, y), (x + w, y + h), color_code, 50)
    return np_ar


def ap_at_k_per_query(np_query_labels, k=5):
    """
    给定标签的二进制预测，该函数返回指定k的平均精度

    Args:
        np_query_labels : 具有二进制值的 numpy 数组/ Python 列表
        k   : 寻找平均精度的截止点

    Returns:
        k 处的平均精度
    """
    ap = 0.0
    running_positives = 0
    for idx, i in enumerate(np_query_labels[:k]):
        if i == 1:
            running_positives += 1
            ap_at_count = running_positives / (idx + 1)
            ap += ap_at_count
    return ap / k


def get_preds_and_visualize(best_matches, query_gt_dict, img_dir, top_k_to_plot):
    """
    给定查询的最佳匹配文件名和地面真实字典，该函数
    返回二进制预测值，并绘制前k个图像结果

    Args:
        best_matches    : 最佳文件匹配列表，例如：['all_souls_0000051.jpg', .....]
        query_gt_dict   : 指示查询的正负例地面真实的字典
        img_dir         : 包含所有目标图像的图像目录
        top_k_to_plot   : 要绘制的前k个结果数

    Returns:
        二进制预测值
    """
    # 创建用于存储预测的 Python 列表
    preds = []

    # 为绘图初始化图表
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = int(top_k_to_plot / 4)
    rows = int(top_k_to_plot / columns)

    for i, pic in enumerate(best_matches):
        img_name = "{}".format(pic.split("/")[-1])
        color_code = None
        if img_name in query_gt_dict['positive']:
            color_code = (0, 255, 0)
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            color_code = (255, 255, 0)
            preds.append(0)
        else:
            color_code = (255, 0, 0)
            preds.append(0)

        if i + 1 > top_k_to_plot:
            continue
        else:
            img_path = "{}".format(os.path.join(img_dir, img_name))
            img_arr = draw_label(img_path, color_code)
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img_arr)

    plt.show()

    return preds


def get_preds(best_matches, query_gt_dict):
    """
    请参见 gets_preds_and_visualize(**args)。
    这是相同的函数，没有任何绘图

    Args:
        best_matches    : 最佳匹配文件列表
        query_gt_dict   : 字典，指示查询的正负例

    Returns:
        预测值的 Python 列表
    """

    # 创建用于存储预测的 Python 列表
    preds = []

    # 遍历最佳匹配，找到预测值
    for i, pic in enumerate(best_matches):
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(0)
        else:
            preds.append(0)

    return preds


def ap_per_query(best_matches, query_gt_dict):
    """
    计算给定的最佳匹配和查询的真实情况的平均精度

    Args:
        best_matches    : 最佳匹配文件列表
        query_gt_dict   : 字典，指示查询的正负例

    Returns:
        查询的平均精度
    """
    # 创建用于存储预测的 Python 列表
    preds = []

    # 遍历最佳匹配，找到预测值
    for i, pic in enumerate(best_matches):
        img_name = "{}".format(pic.split("/")[-1])
        if img_name in query_gt_dict['positive']:
            preds.append(1)
        elif img_name in query_gt_dict['negative']:
            preds.append(0)
        else:
            preds.append(0)

    num_gt = len(query_gt_dict['positive'])

    return ap_at_k_per_query(preds, k=num_gt)


def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):
    """
    绘制训练和验证历史记录

    Args:
        train_hist: 包含训练历史值的numpy数组（损失/准确度指标）
        valid_hist: 包含验证历史值的numpy数组（损失/准确度指标）
        y_label: y轴的标签
        filename: 存储生成图的文件名
        labels: 图例的标签

    Returns:
        None
    """
    # 绘制损失和准确度
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label=labels[0])
    plt.plot(val_hist, label=labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


import cv2

def center_crop_resize(img, cropx, cropy):
    """
    给定一个图像的numpy数组，执行中心裁剪或缩放。

    Args:
        img     : numpy图像数组
        cropx   : 裁剪的宽度
        cropy   : 裁剪的高度

    Returns:
        裁剪或缩放后的numpy图像数组
    """
    # 获取图像的高度和宽度
    y, x = img.shape[:-1]

    # 如果图像尺寸小于裁剪尺寸，进行缩放
    if y < cropy or x < cropx:
        scale = max(cropx / x, cropy / y)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 计算裁剪的起始位置
    startx = img.shape[1] // 2 - (cropx // 2)
    starty = img.shape[0] // 2 - (cropy // 2)

    # 执行裁剪操作
    return img[starty:starty + cropy, startx:startx + cropx, :]


def perform_pca_on_single_vector(ft_np_vector, n_components=1, reshape_dim=2048):
    """
    给定一个特征向量，使用PCA进行降维处理。

    Args:
        ft_np_vector    : numpy 特征向量
        n_components    : 主成分的数量
        reshape_dim     : 重新形状矩阵的高度

    Returns:
        经过PCA降维处理的向量
    """
    # 创建 PCA 对象，指定主成分的数量和是否进行白化
    pca = PCA(n_components=n_components, whiten=True)

    print(ft_np_vector.shape)

    # 将特征向量重新形状为矩阵
    file_fts = ft_np_vector.reshape(512, -1)

    # 对矩阵进行 PCA 拟合
    pca.fit(file_fts)

    # 使用 PCA 对矩阵进行降维
    x = pca.transform(file_fts)

    # 将降维后的向量展平为一维数组
    return x.flatten()


def template_matching(target_img_path, compare_img_path_list, img_dir, top_k=500):
    """
    给定目标图像路径和图像路径列表，获取前 k 个结构相似的图像路径

    Args:
        target_img_path         : 参考图像的路径
        compare_img_path_list   : 要比较的图像路径列表
        img_dir                 : 图像目录
        top_k                   : 返回的前 k 个相似图像路径的数量

    Returns:
        top_k 结构相似的图像路径列表

    示例运行:
        > files = [os.path.join("./data/oxbuild/images", file) for file in os.listdir("./data/oxbuild/images/")]
        > print(files)
        > template_matching("./data/oxbuild/images/all_souls_000051.jpg", files, "./data/oxbuild/images/", 500)
    """

    # 存储结构相似性评分
    ssim = []
    target_size = (500, 500)

    # 对于比较图像路径列表中的每个图像
    for other_img_path in tqdm(compare_img_path_list):
        # 读取两个输入图像
        image_target = cv2.imread(os.path.join(img_dir, target_img_path))
        image_other = cv2.imread(os.path.join(img_dir, other_img_path))

        # 中心裁剪图像
        image_target = center_crop_resize(image_target, 500, 500)
        image_other = center_crop_resize(image_other, 500, 500)
        # print(f"Processing: {other_img_path}")
        # print(f"Image shapes: {image_target.shape}, {image_other.shape}")
        # print(f"Image shapes after center cropping: {image_target.shape}, {image_other.shape}")
        # 如果两个图像的形状不一致，则跳过此次循环
        if image_target.shape != image_other.shape:
            print("Shapes are not equal. Skipping this iteration.")
            continue

        # 将图像转换为灰度
        gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
        gray_other = cv2.cvtColor(image_other, cv2.COLOR_BGR2GRAY)

        # 计算两个图像之间的结构相似性指数 (SSIM)，确保返回差异图像
        score = compare_ssim(gray_target, gray_other, full=False)
        ssim.append(score)

    # 对相似性分数进行排序，获取前 k 个索引
    indexes = (-np.array(ssim)).argsort()[:top_k]

    # 获取前 k 个相似的图像路径
    final_results = [compare_img_path_list[index] for index in indexes]

    return final_results
