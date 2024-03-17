import shutil
import librosa
import numpy as np
import matplotlib
import librosa.display
import os
import pylab
from CNN_IC.configs import TrainImg
from CNN_IC.myutils import get_label_list


def spcetrogramchange(foldname, label, filename, imgpath):
    path = f"./{foldname}/{label}/{filename}"
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
    file = filename.split('.')[0]

    pylab.savefig(f"./{imgpath}/{label}/{file}.png", bbox_inches=None, pad_inches=0)

    pylab.close()


def make_img(foldname, imgpath):
    print('正在将音频转为图片......')
    file_list, label_dict, _ = get_label_list(foldname)
    imgpath = f'CNN_IC/{imgpath}'
    shutil.rmtree(imgpath, ignore_errors=True)
    for i in label_dict.keys():
        os.makedirs(f'{imgpath}/{i}', exist_ok=True)
    for i in file_list:
        try:
            spcetrogramchange(foldname, i[0], i[1], imgpath)
        except:
            print(f"--------{i[1]}.png-------- 保存失败")
    print('转换完成！')


if __name__ == '__main__':
    Train = TrainImg()
    make_img('wavs', Train.foldname)
