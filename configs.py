import torchvision.models as models
from torchvision import transforms
import os


class ModelInfo:
    def __init__(self):
        self.model = 'resnet101'  # 选择模型，可选googlenet,resnet18，resnet34，resnet50，resnet101
        self.modelname = 'model-' + self.model
        self.size = [640, 480]  # 设置输入模型的图片大小
        self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        # self.ms = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.min_pr = 4 / 5  # 设置预测时各模型预测结果最多的类作为最终结果所需要的最小占比，即几票通过算通过

class TrainImg:
    def __init__(self):
        self.train_path = 'static'  # 保存图片的文件名
        self.imgpath = self.train_path + '/tra_img'
        self.is_divide = False  # 设置训练开始前是否拆分测试集
        self.data_path = 'database'  # 拆分前数据所在文件名
        self.foldname = 'static/tra_wav'  # 同目录下音频文件的文件夹名
        self.t_divide_present = 0.7  # 拆分测试集比例
        self.divide_present = 0.8  # 拆分验证集比例
        self.batch_size = 4
        self.learn_rate = 0.001  # 设置学习率
        self.step_size = 1  # 学习率衰减步长
        self.gamma = 0.95  # 学习率衰减系数，也即每个epoch学习率变为原来的0.95
        self.epoch = 30  # 设置迭代次数
        self.show_mode = 'Simple'  # 设模型层数信息写入log中的模式:'All'  'Simple'  'No'
        self.is_test = True  # 设置训练完后是否自动开始测试
        self.min_acc = 95  # 设置停止训练所需的acc，单位为 %，需要开启自动测试
        self.write_process = False  # 设置是否将训练过程写入log中


class TestImg(TrainImg):
    def __init__(self):
        super().__init__()
        self.foldname = 'static/test_wav'
        self.imgpath = 'static/test_img'  # 保存测试图片的路径名
        self.log_path = 'log-test'


class FurTrain(TrainImg):  # 设置对新数据继续训练时的训练参数
    def __init__(self):
        super().__init__()
        self.imgpath = 'static/fur-train'  # 保存新加入训练的音频文件梅尔图的路径名
        self.lock_layers = [1, -2]  # 继续训练时需要冻结的卷积层
        self.size = [224, 224]  # 设置输入模型的图片大小
        self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        self.transform = transforms.Compose([  # 数据处理
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])  # 规范化
        ])
        self.batch_size = 4
        self.learn_rate = 0.0001  # 设置学习率
        self.epoch = 30  # 设置迭代次数
        self.log_path = 'log-furtrain'
        self.is_test = True  # 设置训练完后是否自动开始测试
        self.write_process = True  # 设置是否将训练过程写入log中
        self.show_mode = 'Simple'
