import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from efficientnet_v2_model import efficientnetv2_s as create_efficientnetv2_s_model
from vit_model import vit_base_patch16_224 as create_vit_model
from swin_model import swin_tiny_patch4_window7_224 as create_model
import torch

class TripletNet(nn.Module):
    """
    实现 Triple Net 的类

    属性:
        embedding_net: torchvision 模型实例。例如 torchvision.models.resnet50(pretrained=True)
    """

    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        # 获取 anchor 的嵌入并进行归一化
        anchor_embedding = F.normalize(self.embedding_net(anchor), p=2, dim=1)

        # 获取 positive 的嵌入并进行归一化
        positive_embedding = F.normalize(self.embedding_net(positive), p=2, dim=1)

        # 获取 negative 的嵌入并进行归一化
        negative_embedding = F.normalize(self.embedding_net(negative), p=2, dim=1)

        return anchor_embedding, positive_embedding, negative_embedding

    def get_embedding(self, x):
        return F.normalize(self.embedding_net(x), p=2, dim=1)


class TripletLoss(nn.Module):
    """
    实现三重损失的类
    它接受 anchor 样本、positive 样本和 negative 样本的嵌入，并返回三重损失

    属性:
        margin: 用于区分 positive 和 negative 样本的边界值
    """

    def __init__(self, margin=2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=False):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

    def reduce_margin(self):
        self.margin = self.margin * 0.8


class Identity(nn.Module):
    """
    实现 identity 模块的类
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def create_embedding_net():
    """
    创建嵌入网络的函数

    返回:
        嵌入网络实例
    """
    # 使用 resnet101 作为基础模型
    resnet_model = models.resnet101(pretrained=True)

    # 将全连接层替换为 Identity 模块
    resnet_model.fc = torch.nn.Identity()


    return resnet_model


def create_swin_embedding_net():
    """
    创建嵌入网络的函数

    返回:
        嵌入网络实例
    """
    # 加载预训练的Swin Transformer模型
    swin_model = create_model()


    # 替换模型的最后一个全连接层为Identity模块
    swin_model.head = Identity()

    local_weight_path = '/root/autodl-tmp/deep-learning-for-image-processing-master/pytorch_classification/swin_transformer/weights/best_model-2.pth'

    swin_model.load_state_dict(torch.load(local_weight_path, map_location=torch.device('cpu')), strict=False)


    return swin_model

def create_vit_embedding_net():
    """
    创建嵌入网络的函数

    返回:
        嵌入网络实例
    """
    # 加载预训练的Swin Transformer模型
    vit_model = create_vit_model()


    # 替换模型的最后一个全连接层为Identity模块
    vit_model.head = Identity()

    local_weight_path = '/root/autodl-tmp/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer/vit_base_patch16_224.pth'

    vit_model.load_state_dict(torch.load(local_weight_path, map_location=torch.device('cpu')), strict=False)


    return vit_model


def create_vgg_embedding_net():
    """
    创建嵌入网络的函数

    返回:
        嵌入网络实例
    """

    # 使用PyTorch官方的VGG16预训练模型
    vgg_model = models.vgg16(pretrained=True)

    # 将全连接层替换为 Identity 模块
    vgg_model.classifier[-1] = Identity()

    return vgg_model



def create_densenet_embedding_net():
    """
    创建嵌入网络的函数

    返回:
        嵌入网络实例
    """

    # 使用PyTorch官方的DenseNet预训练模型，这里使用DenseNet121作为例子
    densenet_model = models.densenet121(pretrained=True)

    # 移除全连接层，将特征图的大小变为1x1
    densenet_model.classifier = nn.Identity()

    return densenet_model


def create_efficientnet_embedding_net():
    """
    创建基于 EfficientNet 的嵌入网络，并加载本地预训练权重

    返回:
        嵌入网络实例
    """
    # 加载预训练的 EfficientNet 模型
    efficientnet_model = create_efficientnetv2_s_model(num_classes=6)

    # 替换模型的最后一个全连接层为Identity模块
    efficientnet_model.head.classifier = Identity()

    local_weight_path = '/root/autodl-tmp/deep-learning-for-image-processing-master/pytorch_classification/Test11_efficientnetV2/pre_efficientnetv2-s.pth'

    # 加载本地预训练权重，这里假设权重是与模型兼容的
    efficientnet_model.load_state_dict(torch.load(local_weight_path, map_location=torch.device('cpu')), strict=False)

    return efficientnet_model
