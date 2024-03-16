import time
from torch.utils.data import DataLoader
from utils import *
from configs import TrainImg, ModelInfo
from test import t_img
from torchvision import transforms
import shutil
from torch.optim.lr_scheduler import StepLR

def train_model(model, Train, txt_list, modelinfo):
    model_name = 'model-' + modelinfo.model
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    learn_rate, step_size, gamma, divide_present = Train.learn_rate, Train.step_size, Train.gamma, Train.divide_present  # 获取模型参数
    loss_list = []
    acc_list = []
    dataset_train = ImageFolder(Train.imgpath, transform=transform)  # 训练数据集
    train_size = int(divide_present * len(dataset_train))
    test_size = len(dataset_train) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_train, [train_size, test_size])
    # torch自带的标准数据集加载函数

    dataloader_test = DataLoader(test_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_train = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)

    add_log('-' * 40 + '本次训练准备中，准备过程如下' + '-' * 40, txt_list)
    add_log("是否使用GPU训练：{}".format(torch.cuda.is_available()), txt_list)  # 判断是否采用gpu训练

    if torch.cuda.is_available():
        add_log("使用GPU做训练", txt_list)
        add_log("GPU名称为：{}".format(torch.cuda.get_device_name()), txt_list)

    else:
        add_log("使用CPU做训练", txt_list)
    print(f'使用的预训练模型为:{modelinfo.model}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss().to(device)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 可调超参数
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    epoch = Train.epoch  # 迭代次数
    best_acc = 0  # 起步正确率
    ba_epoch = 1  # 标记最高正确率的迭代数
    min_lost = 100  # 起步loss
    ml_epoch = 1  # 标记最低损失的迭代数
    model = model.to(device)  # 将模型迁移到gpu
    ss_time = time.time()  # 开始时间
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    train_txt = []
    for i in range(epoch):
        add_log("--------第{}轮训练开始---------".format(i + 1), train_txt)
        model.train()
        start_time = time.perf_counter()
        for j, [imgs, targets] in enumerate(dataloader_train):
            imgs, targets = imgs.to(device), targets.to(device)
            # imgs.float()
            # imgs=imgs.float() #为将上述改为半精度操作，在这里不需要用
            # imgs=imgs.half()

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化

            bar(j + 1, len(dataloader_train), start=start_time)
        print()
        scheduler.step()
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            for j, data in enumerate(dataloader_test):
                imgs, targets = data
                # imgs.float()
                # imgs=imgs.float()
                imgs = imgs.to(device)
                targets = targets.to(device)
                # imgs=imgs.half()
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
                print("\r----------验证集验证中({:^3.2f}%)-----------".format((j + 1) * 100 / len(dataloader_test)),
                      end="")
            print()
            add_log("验证集上的loss：{:.3f}".format(total_test_loss), train_txt)
            add_log("验证集上的正确率：{:.3f}%".format(total_accuracy / len(test_dataset) * 100), train_txt)
            acc_list.append(round(float(total_accuracy / len(test_dataset) * 100), 3))
            loss_list.append(round(float(total_test_loss), 3))

            if total_accuracy > best_acc:  # 保存迭代次数中最好的模型
                add_log("根据正确率，已修改模型", train_txt)
                best_acc = total_accuracy
                ba_epoch = i + 1
                torch.save(model, f'{model_name}.pth')
            if total_test_loss < min_lost:
                add_log("根据损失，已修改模型", train_txt)
                ml_epoch = i + 1
                min_lost = total_test_loss
                torch.save(model, f'{model_name}.pth')

    if Train.write_process:
        txt_list = txt_list + train_txt
    ee_time = time.time()
    edate_time = time.strftime('%Y-%m-%d-%H时%M分', time.localtime())
    total_time = ee_time - ss_time

    add_log('-' * 35 + f'本次训练结束于{edate_time}，训练结果与报告如下' + '-' * 40, txt_list)
    add_log(f'本次训练开始时间：{date_time}', txt_list)
    add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)

    add_log(
        f'训练集图片数为:{len(train_dataset)},验证集图片数为:{len(test_dataset)},拆分比为{int(divide_present * 10)}:{int(10 - divide_present * 10)}',
        txt_list)

    add_log(
        f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc * 100 / len(test_dataset)), 3)}%',
        txt_list)
    add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(min_lost, 3)}', txt_list)

    if Train.is_test:
        add_log('-' * 40 + '下面开始在测试集上测试' + '-' * 40, txt_list)
        acc = t_img(txt_list, model_name, is_continue=True)
        return model_name, loss_list, acc_list, epoch, acc
    else:
        return model_name, loss_list, acc_list, epoch


def train_process(model=None):
    Train = TrainImg()
    modelinfo = ModelInfo()
    if model is not None:
        modelinfo.model = model
    model_ft = make_model(modelinfo)
    min_acc = Train.min_acc
    txt_list = []
    st = time.time()
    if Train.is_test:
        num = 1
        add_log('*' * 40 + f' 训练开始，该训练以测试集正确率为停止条件，要求的正确率为{Train.min_acc}% ' + '*' * 40,
                txt_list)
        show(Train, model_ft, txt_list)
        while True:
            add_log('*' * 60 + f"第{num}次训练开始" + '*' * 60, txt_list)
            model_name, loss_list, acc_list, epoch, acc = train_model(model_ft, Train, txt_list, modelinfo)
            num += 1
            if acc > min_acc:
                add_log(40 * '*' + f'模型训练参数如下' + 40 * '*', txt_list)
                add_log(f'预训练模型为:{modelinfo.model}', txt_list)
                add_log(f'训练迭代数为:{epoch},初始学习率为:{Train.learn_rate},衰减步长和系数为{Train.step_size},{Train.gamma}', txt_list)
                add_log(f'数据压缩量为:{Train.batch_size}', txt_list)
                add_log(f'图片预处理大小为:{modelinfo.size[0]}x{modelinfo.size[1]}', txt_list)
                add_log(f'图片标准化设置为:{modelinfo.ms[0]}和{modelinfo.ms[1]}', txt_list)
                date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
                filename = model_name + '-' + str(date_time)
                model_path = f"train_process\\{filename}\\{filename}"
                train_dir(filename)
                shutil.move(f'{model_name}.pth', f'./{model_path}.pth')
                et = time.time()
                tt_time = et - st
                add_log("训练总共用时:{}小时:{}分钟:{}秒".format(int(tt_time // 3600), int((tt_time % 3600) // 60),
                                                                 int(tt_time % 60)), txt_list)
                write_log(f"train_process\\{filename}", filename, txt_list)
                make_plot(loss_list, 'loss', filename, epoch)
                make_plot(acc_list, 'acc', filename, epoch)
                break
            else:
                add_log('*' * 46 + f'本次训练正确率未达到{min_acc}%的要求，下一次训练准备开始' + '*' * 42, txt_list,
                        is_print=False)
                add_log('', txt_list)


    else:
        show(Train, model_ft, txt_list)
        model_name, loss_list, acc_list, epoch = train_model(model_ft, Train, txt_list, modelinfo)
        date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
        filename = model_name + '-' + str(date_time)
        model_path = f"train_process\\{filename}\\{filename}"
        train_dir(filename)
        write_log(f"train_process\\{filename}", filename, txt_list)
        make_plot(loss_list, 'loss', filename, epoch)
        make_plot(acc_list, 'acc', filename, epoch)


if __name__ == '__main__':
    train_process()
