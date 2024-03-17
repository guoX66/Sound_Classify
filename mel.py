import argparse
import shutil
import librosa
import numpy as np
import matplotlib
import librosa.display
import os
import pylab
from CNN_IC.myutils import get_label_list



def label_change(foldname, label, filename, imgpath, is_predict):
    path = f"./{foldname}/{label}/{filename}"
    file = filename.split('.')[0]
    if is_predict:
        out_path = f"./{imgpath}/{file}.png"
    else:
        out_path = f"./{imgpath}/{label}/{file}.png"

    spcetrogram(path, out_path)


def spcetrogram(path, out_path):
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
    pylab.savefig(out_path, bbox_inches=None, pad_inches=0)
    pylab.close()


def make_img(foldname, imgpath,is_predict):
    print('正在将音频转为图片......')
    file_list, label_dict, _ = get_label_list(foldname)
    shutil.rmtree(imgpath, ignore_errors=True)
    os.makedirs(f'{imgpath}', exist_ok=True)
    if not is_predict:
        for i in label_dict.keys():
            os.makedirs(f'{imgpath}/{i}', exist_ok=True)
    for i in file_list:
        try:
            label_change(foldname, i[0], i[1], imgpath, is_predict)
        except:
            print(f"--------{i[1]}.png-------- 保存失败")
    print('转换完成！')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--input', type=str, default='wavs')
    parser.add_argument('--output', type=str, default='CNN_IC/data/static')
    parser.add_argument('--is_predict', type=bool, default=False)
    args = parser.parse_args()
    make_img(args.input, args.output,args.is_predict)
