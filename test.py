import torch
from PIL import Image
from configs import TrainImg, TestImg, ModelInfo
import time
from utils import *
from torchvision import transforms
import re


def t_img(txt_list, model_name, is_continue=False):
    Train = TrainImg()
    _, data_class = get_labellist(Train)
    testimgs = TestImg()
    add_log('-' * 43 + '测试错误信息如下' + '-' * 43, txt_list)
    print()
    real_classlist, _ = get_labellist(testimgs)
    testimg_path = testimgs.imgpath
    modelinfo = ModelInfo()
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    size = modelinfo.size  # 获取原模型对图像的转换
    right_num = 0
    test_num = 0
    model = torch.load(f'{model_name}.pth', map_location=torch.device("cpu"))
    model.eval()
    for real_classes in real_classlist:
        real_class = real_classes[0]
        text_path = f'{testimg_path}/{real_class}/{real_classes[1]}'
        image = Image.open(text_path).convert('RGB')

        image = transform(image)

        image = torch.reshape(image, (1, 3, size[0], size[1]))  # 修改待预测图片尺寸，需要与训练时一致

        with torch.no_grad():
            output = model(image)
            # output = torch.squeeze(model(image)).cpu()  # 压缩batch维度
            # predict = torch.softmax(output, dim=0)
            # predict_cla = torch.argmax(predict).numpy()
        predict_class = data_class[int(output.argmax(1))]

        if predict_class == real_class:
            right_num = right_num + 1
        else:
            is_right = '错误'
            add_log(f'图片{real_classes[1]}预测类别为{predict_class}，真实类别为{real_class},预测{is_right}',
                    txt_list)

        test_num = test_num + 1

    acc = right_num / test_num * 100
    add_log(f'测试总数量为{test_num}，错误数量为{test_num - right_num}', txt_list)
    add_log(f'总预测正确率为{acc}%', txt_list)
    if not is_continue:
        model_path_new = model_path.replace('\\', '标记')
        try:
            filename = re.findall('标记(.*?)标记', model_path_new)[0]
        except IndexError:
            filename = model_path_new
        write_log(testimgs.log_path, filename, txt_list)

    return acc


if __name__ == '__main__':
    txt_list = []
    model_path='model-googlenet'
    # model_path = 'train_process\\model-googlenet-2023-06-11-13h 21m\\model-googlenet-2023-06-11-13h 21m'
    t_img(txt_list, model_path)
