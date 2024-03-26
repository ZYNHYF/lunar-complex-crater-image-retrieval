from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from model import TripletNet, create_embedding_net ,create_swin_embedding_net  # 导入TripletNet和create_embedding_net模型
from dataset import QueryExtractor, EmbeddingDataset  # 导入QueryExtractor和EmbeddingDataset数据集
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from utils import perform_pca_on_single_vector  # 导入perform_pca_on_single_vector函数


def create_embeddings_db_pca(model_weights_path, img_dir, fts_dir):
    """
    给定模型权重路径，此函数创建三元组网络，加载参数并生成经过降维的（使用PCA）向量，
    然后保存在提供的特征目录中。

    Args:
        model_weights_path  : 训练好的权重路径
        img_dir     : 包含图像的目录
        fts_dir     : 存储嵌入特征的目录

    Returns:
        None

    例如运行:
        create_embeddings_db_pca("./weights/oxbuild-exp-3.pth", img_dir="./data/oxbuild/images/", fts_dir="./fts_pca/oxbuild/")
    """
    # 创建特征目录
    if not os.path.exists(fts_dir):
        os.makedirs(fts_dir)

    # 创建CUDA参数
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")  # 根据CUDA的可用性选择设备
    print("Available device = ", device)

    # 创建变换
    mean = [0.3174, 0.3171, 0.3175]
    std = [0.1362, 0.1362, 0.1363]
    transforms_test = transforms.Compose([
        transforms.Resize(460),
        transforms.FiveCrop(448),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
    ])

    # 创建图像数据库
    if "paris" in img_dir:
        print("> 必须删除黑名单中的图像")
        blacklist = ["paris_louvre_000136.jpg",
                "paris_louvre_000146.jpg",
                "paris_moulinrouge_000422.jpg",
                "paris_museedorsay_001059.jpg",
                "paris_notredame_000188.jpg",
                "paris_pantheon_000284.jpg",
                "paris_pantheon_000960.jpg",
                "paris_pantheon_000974.jpg",
                "paris_pompidou_000195.jpg",
                "paris_pompidou_000196.jpg",
                "paris_pompidou_000201.jpg",
                "paris_pompidou_000467.jpg",
                "paris_pompidou_000640.jpg",
                "paris_sacrecoeur_000299.jpg",
                "paris_sacrecoeur_000330.jpg",
                "paris_sacrecoeur_000353.jpg",
                "paris_triomphe_000662.jpg",
                "paris_triomphe_000833.jpg",
                "paris_triomphe_000863.jpg",
                "paris_triomphe_000867.jpg", ]

        files = os.listdir(img_dir)
        for blacklisted_file in blacklist:
            files.remove(blacklisted_file)

        QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(files)]

    else:
        QUERY_IMAGES = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]

    # 创建数据集
    eval_dataset = EmbeddingDataset(img_dir, QUERY_IMAGES, transforms=transforms_test)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=0, shuffle=False)

    # 创建嵌入网络
    embedding_model = create_embedding_net()
    # embedding_model = create_swin_embedding_net()
    model = TripletNet(embedding_model)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # 创建特征
    with torch.no_grad():
        for idx, image in enumerate(tqdm(eval_loader)):
            # 将图像移动到设备并获取裁剪
            image = image.to(device)
            bs, ncrops, c, h, w = image.size()

            # 获取输出
            output = model.get_embedding(image.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1).cpu().numpy()

            # 执行PCA
            output = perform_pca_on_single_vector(output)

            # 保存嵌入特征
            img_name = (QUERY_IMAGES[idx].split("/")[-1]).replace(".jpg", "")
            save_path = os.path.join(fts_dir, img_name)
            np.save(save_path, output.flatten())

            del output, image
            gc.collect()

if __name__ == '__main__':
    create_embeddings_db_pca("/root/autodl-tmp/deep-image-retrieval-master/src/weights/lunar-exp-3.pth",
                             img_dir="/root/autodl-tmp/deep-image-retrieval-master/data/lunar/images",
                             fts_dir="./fts_pca/lunar_2/")
