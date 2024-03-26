from tqdm import tqdm  # 进度条库
import torch
import gc  # 垃圾回收
import os
import numpy as np
from model import TripletLoss, TripletNet  # 导入定义的模型和损失函数
import math


def train_model(model, device, optimizer, scheduler,
                train_loader, valid_loader,
                epochs, update_batch, model_name,
                save_dir,
                log_file):
    """
    训练深度神经网络模型

    参数:
        model   : PyTorch 模型对象
        device  : cuda 或 cpu
        optimizer   : PyTorch 优化器对象
        scheduler   : 学习率调度器对象，用于包装优化器
        train_loader    : 训练图片数据加载器
        valid_loader    : 验证图片数据加载器
        epochs  : 训练周期数
        update_batch    : 每隔多少个批次执行一次梯度更新
        save_dir    : 保存模型权重、绘图和日志文件的位置
        log_file    : 文本文件实例，用于记录训练和验证历史

    返回:
        训练历史和验证历史（损失）
    """
    tr_loss = []  # 训练损失记录
    valid_loss = []  # 验证损失记录
    best_val_loss = np.iinfo(np.int32).max  # 初始化最佳验证损失为最大整数值
    weights_path = os.path.join(save_dir, model_name)  # 保存模型权重的路径
    temp_weights_path = os.path.join(save_dir, "temp_{}".format(model_name))  # 保存临时模型权重的路径
    last_batch = math.ceil(len(train_loader.dataset) / update_batch)  # 最后一个批次的索引

    # 每个周期有训练和验证阶段
    for epoch in range(epochs):

        print("-------第 {} 个周期----------".format(epoch + 1))
        log_file.write("-------第 {} 个周期----------".format(epoch + 1))

        criterion = TripletLoss(margin=2)  # 三元组损失函数
        train_loader.dataset.reset()  # 重置训练数据集状态

        if (epoch + 1) % update_batch == 0:
            print("> 调整学习率")
            scheduler.step()  # 调整学习率

        for phase in ['train', 'valid']:
            running_loss = 0.0  # 损失的累积

            if phase == 'train':
                model.train(True)  # 设置模型为训练模式

                # 将梯度参数清零
                optimizer.zero_grad()

                print("> 训练网络")
                for batch_idx, [anchor, positive, negative] in enumerate(tqdm(train_loader)):
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                    # 获取嵌入
                    output1, output2, output3 = model(anchor, positive, negative)

                    # 计算三元组损失
                    loss = criterion(output1, output2, output3)

                    # 累积批次损失
                    running_loss += loss

                    # 反向传播以计算梯度
                    loss.backward()

                    if (batch_idx + 1) % update_batch == 0 or (batch_idx + 1) == last_batch:
                        # 更新模型参数
                        optimizer.step()

                        # 将梯度参数清零
                        optimizer.zero_grad()

                    # 清理变量
                    del anchor, positive, negative, output1, output2, output3
                    gc.collect()
                    torch.cuda.empty_cache()

                # 保存已训练的模型
                print("> 保存训练好的权重...")
                torch.save(model.state_dict(), temp_weights_path)

                # 计算统计数据并记录
                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item() / num_samples
                tr_loss.append(tr_loss_)
                print('> 训练损失: {:.4f}\t'.format(tr_loss_))
                log_file.write('> 训练损失: {:.4f}\t'.format(tr_loss_))


            else:
                model.train(False)

                print("> 在网络上运行验证")
                with torch.no_grad():
                    for anchor, positive, negative in tqdm(valid_loader):
                        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                        # 获取嵌入
                        output1, output2, output3 = model(anchor, positive, negative)

                        # 计算三元组损失
                        loss = criterion(output1, output2, output3)

                        # 累积批次损失
                        running_loss += loss

                        # 清理变量
                        del anchor, positive, negative, output1, output2, output3
                        gc.collect()
                        torch.cuda.empty_cache()

                # 获取统计数据并记录
                num_samples = float(len(valid_loader.dataset))
                valid_loss_ = running_loss.item() / num_samples
                valid_loss.append(valid_loss_)
                print('> 验证损失: {:.4f}\t'.format(valid_loss_))
                log_file.write('> 验证损失: {:.4f}\t'.format(valid_loss_))

                if valid_loss_ < best_val_loss:
                    best_val_loss = valid_loss_
                    print("> 保存最佳权重...")
                    log_file.write("保存最佳权重...\n")
                    torch.save(model.state_dict(), weights_path)

    return (tr_loss, valid_loss)
