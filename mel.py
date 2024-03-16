import librosa
import numpy as np
import matplotlib
import librosa.display
import os
import pylab
from configs import TrainImg, TestImg
from utils import divide_test


def spcetrogramchange(foldname, label, filename, imgpath):
    path = f"./{foldname}/{label}/{filename}.wav"
    matplotlib.use('Agg')
    # sr=None声音保持原采样频率， mono=False声音保持原通道数
    data, fs = librosa.load(path, sr=None, mono=False)

    # 0.025s
    framelength = 0.025
    # NFFT点数=0.025*fs
    framesize = int(framelength * fs)
    # 画语谱图
    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=framesize)

    mel_spect = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(mel_spect, sr=fs)

    pylab.savefig(f"./{imgpath}/{label}/{filename}.png", bbox_inches=None, pad_inches=0)

    pylab.close()


def get_label_list(foldname):
    file_path = f'./{foldname}'

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
            name = j.replace('.wav', '')
            label_name_list.append([i, name])

    return label_name_list, label_dict, label_list


def make_img(foldname, imgpath):
    print('正在将音频转为图片......')
    file_list, label_dict, _ = get_label_list(foldname)
    for i in label_dict.keys():
        try:
            os.makedirs(f'{imgpath}/{i}')
        except:
            pass
    for i in file_list:
        try:
            spcetrogramchange(foldname, i[0], i[1], imgpath)
        except:
            print(f"--------{i[1]}.png-------- 保存失败")
    print('转换完成！')


Train = TrainImg()
Test = TestImg()
divide_test(Train.data_path, Train.train_path, Train.t_divide_present)
make_img(Train.foldname, Train.imgpath)
make_img(Test.foldname, Test.imgpath)
