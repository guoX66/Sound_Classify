from PIL import Image
from CNN_IC.myutils import *
import openpyxl as op
from torchvision import transforms


def predict_class(path, model, modelinfo, data_class):
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    model.eval()
    image = Image.open(path).convert('RGB')
    image = transform(image)
    image = torch.reshape(image, (1, 3, size[0], size[1]))  # 修改待预测图片尺寸，需要与训练时一致
    with torch.no_grad():
        output = model(image)
        # output = torch.squeeze(model(image)).cpu()  # 压缩batch维度
        # predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()
    predictclass = data_class[int(output.argmax(1))]
    return predictclass


def file_list(filename):
    list1 = []
    for i in os.walk(filename):
        for file in i[2]:
            path = os.path.join(filename, file)
            list1.append([path, file])
        return list1


if __name__ == '__main__':
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    Train = TrainImg()
    _, data_class = get_labellist(Train)
    modelinfo = ModelInfo()
    min_pr = modelinfo.min_pr
    size = modelinfo.size  # 获取原模型对图像的转换
    img_list = file_list('predict')
    model_list = file_list('models')
    txt_list = []
    for img in img_list:
        result_dict = {i: 0 for i in data_class}
        path, file = img
        for i, _ in model_list:
            model = torch.load(f"{i}", map_location=torch.device("cpu"))
            result = predict_class(path, model, modelinfo, data_class)
            result_dict[result] += 1
        times = list(result_dict.values())
        times.sort()
        if times[-1] / len(model_list) >= min_pr:
            new_dict = {v: k for k, v in result_dict.items()}
            ans = new_dict[times[-1]]
            txt_list.append([file, ans])
            print(f'{file}的预测类别为{ans}')
        else:
            txt_list.append([file, '未知类别'])
            print(f'{file}的预测类别为未知类别')

    wb = op.Workbook()
    ws = wb.create_sheet('预测结果', 0)
    ws.append(['图片名称', '预测类别'])
    for i in txt_list:
        ws.append(i)
    ws_ = wb['Sheet']
    wb.remove(ws_)
    wb.save(f'Result-{date_time}.xlsx')
