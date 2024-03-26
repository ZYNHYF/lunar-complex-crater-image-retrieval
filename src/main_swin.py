from tqdm import tqdm
import torch
import gc
import os
import numpy as np
from model import TripletNet, create_embedding_net ,create_swin_embedding_net, create_vgg_embedding_net, create_densenet_embedding_net, create_efficientnet_embedding_net, create_vit_embedding_net
from dataset import QueryExtractor, VggImageRetrievalDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from train import train_model
from utils import plot_history
import math


def main(data_dir, results_dir, weights_dir,
         which_dataset, image_resize, image_crop_size,
         exp_num,
         max_epochs, batch_size, samples_update_size,
         num_workers=12, lr=5e-6, weight_decay=1e-5, model_use='efficientnet'):
    """
    这是主函数。您只需与此函数进行交互以进行培训（它将记录所有结果）
    一旦训练完成，请使用 create_db.py 创建嵌入，并使用 inference_on_single_image.py 进行测试

    参数：
        data_dir    : 数据的父目录
        results_dir : 存储结果的目录（确保您首先创建此目录）
        weights_dir : 存储权重的目录（确保您首先创建此目录）
        which_dataset : "oxford" 或 "paris"
        image_resize : 调整大小为此大小
        image_crop_size : 方形裁剪尺寸
        exp_num     : 实验编号以记录日志和结果
        max_epochs  : 最大训练轮数
        batch_size  : 批处理大小（我使用的是5）
        samples_update_size : 网络应在执行一次参数更新之前看到的样本数量（我使用的是64）

    关键字参数：
        num_workers : 默认为4
        lr      : 初始学习率（默认5e-6）
        weight_decay: 默认1e-5

    示例运行：
        if __name__ == '__main__':
            main(data_dir="./data/", results_dir="./results", weights_dir="./weights",
                which_dataset="oxbuild", image_resize=460, image_crop_size=448,
                exp_num=3, max_epochs=10, batch_size=5, samples_update_size=64)
    """

    # 创建目录
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # 定义目录
    labels_dir = os.path.join(data_dir, which_dataset, "gt_files")
    image_dir = os.path.join(data_dir, which_dataset, "images")

    # 创建 QueryExtractor 对象
    q_train = QueryExtractor(labels_dir, image_dir, subset="train")
    q_valid = QueryExtractor(labels_dir, image_dir, subset="valid")

    # 创建 transforms
    mean = [0.3174, 0.3171, 0.3175]
    std = [0.1362, 0.1362, 0.1363]
    transforms_train = transforms.Compose([transforms.Resize(image_resize),
                                           transforms.RandomResizedCrop(image_crop_size, scale=(0.8, 1.2)),
                                           transforms.ColorJitter(brightness=(0.80, 1.20)),
                                           transforms.RandomHorizontalFlip(p=0.50),
                                           transforms.RandomChoice([
                                               transforms.RandomRotation(15),
                                               transforms.Grayscale(num_output_channels=3),
                                           ]),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std),
                                           ])

    transforms_valid = transforms.Compose([transforms.Resize(image_resize),
                                           transforms.CenterCrop(image_crop_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std),
                                           ])

    # 创建数据集
    dataset_train = VggImageRetrievalDataset(labels_dir, image_dir, q_train, transforms=transforms_train)
    dataset_valid = VggImageRetrievalDataset(labels_dir, image_dir, q_valid, transforms=transforms_valid)

    # 创建数据加载器
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # 创建 cuda 参数
    use_cuda = torch.cuda.is_available()
    np.random.seed(2020)
    torch.manual_seed(2020)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 创建嵌入网络
    # embedding_model = create_embedding_net()
    # embedding_model = create_vgg_embedding_net()
    # embedding_model = create_swin_embedding_net()
    # embedding_model = create_densenet_embedding_net()
    embedding_model = create_efficientnet_embedding_net()
    # embedding_model = create_vit_embedding_net()
    model = TripletNet(embedding_model)
    model.to(device)

    # 创建优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 创建日志文件
    log_file = open(os.path.join(results_dir, "log-{}-{}.txt".format(exp_num, model_use)), "w+")
    log_file.write("----------实验 {}-{}----------\n".format(exp_num, model_use))
    log_file.write("数据集 = {}, 图像尺寸 = {}, {}\n".format(which_dataset, image_resize, image_crop_size))

    # 创建批量更新值
    update_batch = int(math.ceil(float(samples_update_size) / batch_size))
    model_name = "{}-exp-{}-{}.pth".format(which_dataset, exp_num, model_use)
    loss_plot_save_path = os.path.join(results_dir, "{}-loss-exp-{}.png".format(which_dataset, exp_num))

    # 在开始训练之前打印状态
    print("运行 {} 图像检索训练脚本".format(model_use))
    print("使用的数据集\t\t:{}".format(which_dataset))
    print("最大轮次\t\t: {}".format(max_epochs))
    print("梯度更新\t\t: 每 {} 批次（{} 个样本）".format(update_batch, samples_update_size))
    print("初始学习率\t\t: {}".format(lr))
    print("图像调整大小，裁剪尺寸\t: {}, {}".format(image_resize, image_crop_size))
    print("可用设备 \t:", device)

    # 训练三元组网络
    tr_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader,
                                    epochs=max_epochs, update_batch=update_batch, model_name=model_name,
                                    save_dir=weights_dir, log_file=log_file)

    # 关闭文件
    log_file.close()

    # 绘制并保存
    plot_history(tr_hist, val_hist, "Triplet Loss", loss_plot_save_path, labels=["train", "validation"])

if __name__ == '__main__':
    main(data_dir="/root/autodl-tmp/deep-image-retrieval-master/data", results_dir="./results_1", weights_dir="./weights_1",
        which_dataset="lunar", image_resize=460, image_crop_size=448,
        exp_num=8, max_epochs=10, batch_size=11, samples_update_size=64)
