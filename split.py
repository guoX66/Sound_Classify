import argparse
import shutil

from scipy.io import wavfile
import re
import numpy as np
import os


def split(path, outpath, filename, n):
    sr, y = wavfile.read(f'{path}.wav')

    piece_num = int(len(y) / (sr * n))
    for i_piece in range(piece_num):
        piece_wav = y[i_piece * sr * n:(i_piece + 1) * sr * n]
        wavfile.write(f'./{outpath}/{filename}-{i_piece}.wav', sr,
                      data=np.array(np.clip(np.round(piece_wav), -2 ** 15, 2 ** 15 - 1),
                                    dtype=np.int16))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--input', type=str, default='i_wavs')
    parser.add_argument('--output', type=str, default='wavs')
    args = parser.parse_args()

    ini_filename = args.input
    out_filename = args.output

    count = 0
    path_list = []
    class_list = [i.split('.')[0] for i in list(os.walk(ini_filename))[0][-1] if i.split('.')[0] != '']
    shutil.rmtree(out_filename, ignore_errors=True)
    # os.makedirs(out_filename, exist_ok=True)
    print('正在分割音频中......')
    for i in range(len(class_list)):
        ini_path = os.path.join(ini_filename, class_list[i])
        class_path = os.path.join(out_filename, class_list[i])
        os.makedirs(class_path, exist_ok=True)
        # out_path = os.path.join(class_path, j)
        split(ini_path, class_path, class_list[i], args.n)
        print(f'{class_list[i]}.wav 分割完毕')
