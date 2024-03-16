from scipy.io import wavfile
import re
import numpy as np
import os


def split(path, outpath, filename):
    filename = re.findall('(.*?).wav', filename)[0]
    sr, y = wavfile.read(path)
    n = 5  # 每段音频长度10s
    piece_num = int(len(y) / (sr * n))
    for i_piece in range(piece_num):
        piece_wav = y[i_piece * sr * n:(i_piece + 1) * sr * n]
        wavfile.write(f'./{outpath}/{filename}-{i_piece}.wav', sr,
                      data=np.array(np.clip(np.round(piece_wav), -2 ** 15, 2 ** 15 - 1),
                                    dtype=np.int16))


if __name__ == '__main__':
    ini_filename = 'Boat_signal'
    out_filename = 'database'
    count = 0
    path_list = []
    for i in os.walk(ini_filename):
        if count == 0:
            class_list = i[1]
        else:
            path_list.append(i[2])
        count += 1
    try:
        os.mkdir(out_filename)
    except:
        pass
    print('正在分割音频中......')
    for i in range(len(class_list)):
        ini_path = os.path.join(ini_filename, class_list[i])
        class_path = os.path.join(out_filename, class_list[i])
        try:
            os.mkdir(class_path)
        except:
            pass

        for j in path_list[i]:
            in_path = os.path.join(ini_path, j)
            # out_path = os.path.join(class_path, j)
            split(in_path, class_path, j)
            print(f'{j}分割完毕')
