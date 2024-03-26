#-*- coding = utf-8 -*- 
#@Time:         15:16
#@File：        zyn.py
#@Software:     PyCharm
#@Author:       ZYN

# from model import TripletNet, create_embedding_net ,create_swin_embedding_net
# import torch
#
# def check_model_keys(model, weights_path):
#     # 加载模型权重文件
#     state_dict = torch.load(weights_path)
#
#     # 打印模型结构的键
#     model_keys = set(model.state_dict().keys())
#     print("Model keys:")
#     print(model_keys)
#     print()
#
#     # 打印权重文件的键
#     state_dict_keys = set(state_dict.keys())
#     print("State_dict keys:")
#     print(state_dict_keys)
#     print()
#
#     # 检查权重文件中是否有不匹配的键
#     unexpected_keys = state_dict_keys - model_keys
#     missing_keys = model_keys - state_dict_keys
#
#     print("Unexpected keys in state_dict:", unexpected_keys)
#     print("Missing keys in state_dict:", missing_keys)
#
# # 创建你的 Swin Transformer 模型实例
# embedding_model = create_swin_embedding_net()
# model = TripletNet(embedding_model)
# # 指定你的权重文件路径
# model_weights_path = "/root/autodl-tmp/deep-image-retrieval-master/src/weights/lunar-exp-2.pth"
#
# # 执行检查
# check_model_keys(model, model_weights_path)
def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    Parameters:
        model: Model instance.

    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from model import TripletNet, create_embedding_net ,create_swin_embedding_net, create_densenet_embedding_net, create_vgg_embedding_net, create_efficientnet_embedding_net
import torch



# 创建模型
model = create_efficientnet_embedding_net()

# 打印模型结构
print(model)

# 定义一个虚拟输入，batch size为1，通道数为3，高和宽为224（适用于ResNet等模型）
dummy_input = torch.randn(8, 3, 224, 224)

# 将虚拟输入传递给模型，获取输出
output = model(dummy_input)

# 打印输出的维度
print("Output Shape:", output.shape)

num_parameters = count_parameters(model)
print("Number of parameters in the model:", num_parameters)


