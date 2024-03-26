import os, glob
import numpy as np
from random import shuffle
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
from utils import template_matching


class QueryExtractor():
    """
    查询提取器类，用于提取所有查询和三元组数据

    示例运行:
        # 定义目录
        > labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

        # 创建查询提取器对象
        > q = QueryExtractor(labels_dir, image_dir, subset="inference", query_suffix="oxc1_")
    """

    def __init__(self, labels_dir, image_dir, subset, query_suffix="oxc1_", identifiers=['good', 'ok', 'junk', 'bad']):
        """
        初始化查询提取器类
        """
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.identifiers = identifiers
        self.query_list = self.create_query_list()
        self.query_names = []
        self.subset = subset
        self.query_map = dict()
        self.query_suffix = query_suffix
        self.create_query_maps()

        # 创建三元组映射
        self.triplet_pairs = []
        self._generate_triplets()

    def create_query_list(self):
        """
        返回所有查询文本文件的列表
        """
        all_file_list = sorted(os.listdir(self.labels_dir))
        query_list = [file for file in all_file_list if file.endswith('query.txt')]
        return sorted(query_list)

    def create_query_maps(self):
        """
        创建一个包含每个查询正负样本的字典
        """
        # 遍历每个查询
        for query in self.query_list:
            # 获取查询对应的图像文件名
            query_image_name = self._get_query_image_name(query)

            # 创建临时字典
            tmp = dict()

            # 获取正负样本文件列表
            good_file, ok_file, junk_file, bad_file = self._get_query_image_files(query)
            tmp['positive'] = self._read_txt_file(good_file) + self._read_txt_file(ok_file) + self._read_txt_file(
                junk_file)

            # 如果负样本文件存在，则读取
            if os.path.exists(os.path.join(self.labels_dir, bad_file)):
                tmp['negative'] = self._read_txt_file(bad_file)
            else:
                print("> 为查询创建硬负样本：", query)
                # 获取剩余的图像文件列表，用于生成负样本
                tmp['negative'] = self._get_remaining_image_files(
                    set(tmp['positive'] + [query_image_name] + self._get_blacklist()))

                # 创建并保存硬负样本文件
                tmp['negative'] = self._create_bad_image_files(query, query_image_name, tmp['negative'])

            # 根据子集类型划分正样本用于训练或验证
            split = int(math.ceil(len(tmp['positive']) * 0.80))
            if self.subset == "train":
                tmp['positive'] = tmp['positive'][:split]
            elif self.subset == "valid":
                tmp['positive'] = tmp['positive'][split:]
            else:
                tmp['positive'] = tmp['positive']

            # 将正负样本字典添加到查询字典中
            self.query_map[query_image_name] = tmp
            self.query_names.append(query_image_name)

    def _get_query_image_files(self, query_file):
        """
        返回与查询文件名对应的好、中、差、坏文件的文件名
        """
        good_file, ok_file, junk_file, bad_file = query_file.replace('query', self.identifiers[0]), query_file.replace(
            'query', self.identifiers[1]), \
                                                  query_file.replace('query', self.identifiers[2]), query_file.replace(
            'query', self.identifiers[3])
        return good_file, ok_file, junk_file, bad_file

    def _read_txt_file(self, txt_file_name):
        """
        给定文本文件，返回其中的行列表
        """
        file_path = os.path.join(self.labels_dir, txt_file_name)
        line_list = ["{}.jpg".format(line.rstrip('\n')) for line in open(file_path, errors='ignore')]
        return line_list

    def _get_all_image_files(self):
        all_file_list = [file for file in os.listdir(self.image_dir)]
        return all_file_list

    def get_query_list(self):
        """
        返回查询文本文件列表
        """
        return self.query_list

    def get_query_map(self):
        """
        返回查询映射
        """
        return self.query_map

    def _get_query_image_name(self, query_file):
        """
        给定查询文件名（all_souls_query_1.txt），返回文件中的实际查询图像（all_souls_00001.jpg）
        """
        file_path = os.path.join(self.labels_dir, query_file)
        line_list = ["{}.jpg".format(line.rstrip('\n').split()[0].replace(self.query_suffix, "")) for line in
                     open(file_path, encoding="utf8")][0]
        return line_list

    def _get_remaining_image_files(self, tmp_set):
        """
        获取所有与查询对应的负样本图像
        """
        all_set = set(self._get_all_image_files())
        bad_list = list(all_set - tmp_set)
        return bad_list

    def get_query_names(self):
        """
        返回查询图像文件名列表
        """
        return self.query_names

    def _generate_triplets(self):
        """
        生成（锚点，正样本），（锚点，负样本）对，适用于所有查询
        """
        self.triplet_pairs = []
        for anchor in self.query_names:
            anchor_positive_pairs = [(anchor, positive) for positive in self.query_map[anchor]['positive']]
            anchor_negative_pairs = [(anchor, negative) for negative in self.query_map[anchor]['negative']]

            # 定义下限并进行过滤
            low_bound = min(len(anchor_positive_pairs), len(anchor_negative_pairs))

            # 现在获取低限元素以创建三元组对
            anchor_positive_pairs = anchor_positive_pairs[:low_bound]
            shuffle(anchor_negative_pairs)
            anchor_negative_pairs = anchor_negative_pairs[:low_bound]

            # 创建三元组列表并追加
            triplet_list = [[anchor_positive_pairs[i], anchor_negative_pairs[i]] for i in range(low_bound)]
            self.triplet_pairs.extend(triplet_list)

    def _create_bad_image_files(self, query_txt_file, target_img_path, compare_img_list):
        """
        使用结构相似性创建给定查询的负样本
        """
        # 创建虚拟文件列表，因为相同景观查询具有相同的真实标签
        bad_file_name = os.path.join(self.labels_dir, query_txt_file.replace('query', "bad"))

        target_bad_files = []

        for i in range(1, 6):
            file_name = bad_file_name.replace("_1_", "_{}_".format(i))
            target_bad_files.append(file_name)

        neg_list = template_matching(target_img_path, compare_img_list, self.image_dir)

        for bad_file in target_bad_files:
            with open(bad_file, 'w+') as f:
                for item in neg_list:
                    f.write("%s\n" % item.replace(".jpg", ""))

        return neg_list

    def _get_blacklist(self):
        """
        Paris 6k数据集有一些黑名单图像需要过滤
        """
        return ["paris_louvre_000136.jpg",
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

    def get_triplets(self):
        """
        返回三元组
        """
        shuffle(self.triplet_pairs)
        return self.triplet_pairs

    def reset(self):
        """
        使用不同组合重新生成三元组。请注意，三元组的数量与锚点示例的立方成正比。
        """
        print("> 重新设置数据集")
        self._generate_triplets()
        shuffle(self.triplet_pairs)
        return self.triplet_pairs


class VggImageRetrievalDataset(Dataset):
    """
    数据集类，用于Oxford和Paris数据集的通用处理

    示例运行:
        # 定义目录
        > labels_dir, image_dir = "./data/oxbuild/gt_files/", "./data/oxbuild/images/"

        # 创建查询提取器对象
        > q = QueryExtractor(labels_dir, image_dir, subset="inference", query_suffix="oxc1_")

        # 实例化数据集类并检索第一个三元组
        > ox = VggImageRetrievalDataset(labels_dir, image_dir, q, transforms=transforms_test)
        > a, p, n = ox.__getitem__(0)

    """

    def __init__(self, labels_dir, image_dir, triplet_pair_generator, transforms=None):
        self.labels_dir = labels_dir
        self.image_dir = image_dir
        self.triplet_generator = triplet_pair_generator
        self.triplet_pairs = triplet_pair_generator.reset()
        self.transforms = transforms

    def reset(self):
        self.triplet_pairs = self.triplet_generator.reset()

    def __getitem__(self, index):
        # 获取查询图像名称
        triplet = self.triplet_pairs[index]

        # 断言
        assert (triplet[0][0] == triplet[1][0])
        anchor_img_name, positive_img_name, negative_img_name = triplet[0][0], triplet[0][1], triplet[1][1]

        # 获取图像路径
        anchor_path = os.path.join(self.image_dir, anchor_img_name)
        positive_path = os.path.join(self.image_dir, positive_img_name)
        negative_path = os.path.join(self.image_dir, negative_img_name)

        # 加载图像
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # 转换图像
        if self.transforms is not None:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            neg_img = self.transforms(negative_img)
            return anchor_img, positive_img, neg_img

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.triplet_pairs)


class EmbeddingDataset(Dataset):
    """
    用于获取嵌入的评估

    """
    def __init__(self, image_dir, query_img_list, transforms):
        if transforms == None:
            raise
        
        self.image_dir = image_dir
        self.transforms = transforms
        self.filenames = query_img_list
    

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        return image
        

    def __len__(self):
        return len(self.filenames)


    def get_filenames(self):
        return self.filenames


# Define directories
labels_dir, image_dir = "/root/autodl-tmp/deep-image-retrieval-master/data/lunar/gt_files", "/root/autodl-tmp/deep-image-retrieval-master/data/lunar/images"

# # Create Query extractor object
q = QueryExtractor(labels_dir, image_dir, subset="inference", query_suffix="oxc1_")

# # Get query list and query map
triplets = q.get_triplets()
print(len(triplets))
print(q.get_query_names())

# from torchvision import transforms
# import torch
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# transforms_test = transforms.Compose([transforms.Resize(460),
#                                     transforms.RandomResizedCrop(448, scale=(0.8, 1.2)),
#                                     transforms.ToTensor(),
#                                     #transforms.Normalize(mean=mean, std=std),                                 
#                                     ])
# # Create dataset
# ox = VggImageRetrievalDataset(labels_dir, image_dir, q, transforms=transforms_test)
# a, p, n = ox.__getitem__(1)
# plt.imshow(a.numpy().transpose(1, 2, 0))
# plt.show()

# plt.imshow(p.numpy().transpose(1, 2, 0))
# plt.show()

# plt.imshow(n.numpy().transpose(1, 2, 0))
# plt.show()




