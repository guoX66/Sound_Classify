"""
该程序包含训练需要的函数
"""
import torch.nn as nn
from configs import TrainImg, ModelInfo, TestImg, FurTrain
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import random
import shutil


def divide_test(path, o_path, divide_num):
    print('正在拆分数据集......')
    try:
        shutil.rmtree(o_path)
    except:
        pass
    train_path = os.path.join(o_path, 'tra_wav')
    test_path = os.path.join(o_path, 'test_wav')
    try:
        os.mkdir(o_path)
    except:
        pass
    try:
        os.mkdir(train_path)
    except:
        pass
    try:
        os.mkdir(test_path)
    except:
        pass
    for i in os.walk(path):
        for c_fold in i[1]:
            i_fold_path = os.path.join(path, c_fold)
            o_fold_path_train = os.path.join(train_path, c_fold)
            o_fold_path_test = os.path.join(test_path, c_fold)
            try:
                os.mkdir(o_fold_path_train)
            except:
                pass
            try:
                os.mkdir(o_fold_path_test)
            except:
                pass
            file_list = os.listdir(i_fold_path)
            random.shuffle(file_list)
            train_num = int(len(file_list) * divide_num)
            train_list = file_list[:train_num]
            test_list = file_list[train_num:]
            for file in train_list:
                in_path = os.path.join(i_fold_path, file)
                o_file_path_train = os.path.join(o_fold_path_train, file)
                shutil.copy(in_path, o_fold_path_train)
            for file in test_list:
                in_path = os.path.join(i_fold_path, file)
                o_file_path_test = os.path.join(o_fold_path_test, file)
                shutil.copy(in_path, o_fold_path_test)
    print('拆分完成！')

def get_label_list(imgpath):
    Train = TrainImg()
    file_path = f'./{imgpath}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def make_model(modelinfo):
    Train = TrainImg()  # 实例化img对象
    _, label_list = get_labellist(Train)  # 通过调用img实例化对象函数获取标签列表及个数
    label_len = len(label_list)

    model_ft = sele_model(modelinfo)

    in_features = model_ft.fc.in_features

    model_ft.fc = nn.Sequential(nn.Linear(in_features, label_len))
    return model_ft


def bar(i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r训练进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(progress, finsh, need_do, dur), end="")


# 将最后的全连接改为对应类的输出，使输出为对应标签个数的小数，对应不同标签的置信度

# model_ft.half()#可改为半精度，加快训练速度，在这里不需要用
def get_layers(model):
    layer_list = []
    for layer in model.named_modules():  # 获取所有层数的名称
        layer_list.append(layer)
    return layer_list


def show(c, model, txt_list):
    layer_list = get_layers(model)  # 获取模型各层信息
    if c.show_mode == 'All':
        for layers in layer_list:
            txt_list.append(str(layers) + '\r\n')

    elif c.show_mode == 'Simple':
        for layers in layer_list:
            txt_list.append(str(layers[0]) + '\r\n')


def lock(model, start, end):
    layer_list = []
    for layer in model.named_modules():  # 获取所有层数的名称
        layer_list.append(layer[0])

    need_frozen_list = layer_list[start:end]

    for module in need_frozen_list:  # 匹配并冻结对应网络层
        for param in model.named_parameters():
            if module in param[0]:
                param[1].requires_grad = False


def get_acc(model, device):  # 该函数可以获得模型在原数据集上的正确率
    ini_img = TrainImg()
    ini_model = ModelInfo()
    transform = ini_model.transform
    dataset_test = ImageFolder(ini_img.imgpath, transform=transform)
    dataloader_test_initial = DataLoader(dataset_test, batch_size=ini_img.batch_size, shuffle=True, num_workers=0,
                                         drop_last=True)
    model.eval()
    total_accuracy = 0
    with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
        for data in dataloader_test_initial:
            imgs, targets = data
            # if torch.cuda.is_available():
            # imgs.float()
            # imgs=imgs.float()
            imgs = imgs.to(device)
            targets = targets.to(device)
            # imgs=imgs.half()
            outputs = model(imgs)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        acc = total_accuracy / len(dataset_test)
        return acc


def sele_model(Model):
    model_dict = {
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),  # 残差网络
        'resnet34': models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'resnet101': models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
        'googlenet': models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    }
    return model_dict[Model.model]


def get_labellist(c):
    label_name_list, _, label_list = get_label_list(c.imgpath)
    return label_name_list, label_list


def write_log(in_path, filename, txt_list):
    try:
        os.mkdir(in_path)
    except:
        pass
    path = os.path.join(in_path, filename + '.txt')
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


def add_log(txt, txt_list, is_print=True):
    if is_print:
        print(txt)
    txt_list.append(txt + '\r\n')


def train_dir(filename):
    try:
        os.mkdir('train_process')
    except:
        pass
    file_path = 'train_process\\' + filename
    try:
        os.mkdir(file_path)
    except:
        pass


def make_plot(data, mode, filename, epoch):
    file_path = 'train_process\\' + filename
    if mode == 'loss':
        title = '损失曲线'
        path = os.path.join(file_path, 'LOSS-' + filename)
    elif mode == 'acc':
        title = '正确率曲线'
        path = os.path.join(file_path, 'ACC-' + filename)
    figure(figsize=(12.8, 9.6))
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title + '-' + filename, fontsize=20)
    plt.xlabel('迭代次数', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')
