# 将音频向右移动10s，最后10s将挪到最前面
import librosa
import numpy as np
from scipy.io import wavfile
import random
import os
from pydub import AudioSegment
import shutil
import re


def split(filename):
    print('正在分割音频中......')
    label = re.findall('(.*?).wav', filename)[0]
    try:
        os.mkdir(label)
    except:
        pass
    sr, y = wavfile.read(filename)
    try:
        m, n = y.shape
        if n == 2:
            y = (y[:, 0] + y[:, 1]) / 2
    except ValueError:
        pass
    n = 10  # 每段音频长度10s
    piece_num = int(len(y) / (sr * n))
    for i_piece in range(piece_num):
        piece_wav = y[i_piece * sr * n:(i_piece + 1) * sr * n]
        wavfile.write(f'./{label}/{label}-{i_piece}.wav', sr,
                      data=np.array(np.clip(np.round(piece_wav), -2 ** 15, 2 ** 15 - 1),
                                    dtype=np.int16))
    print('音频分割成功！')
    return label


def cut(filename, st, ed):
    sr, y = wavfile.read(filename)
    try:
        m, n = y.shape
        if n == 2:
            y = (y[:, 0] + y[:, 1]) / 2
    except ValueError:
        pass
    piece_wav = y[st * sr * 60:ed * sr * 60]
    output_filename = f'{filename}_cut.wav'
    wavfile.write(output_filename, sr,
                  data=np.array(np.clip(np.round(piece_wav), -2 ** 15, 2 ** 15 - 1),
                                dtype=np.int16))
    return output_filename


def shift(input_file, output_file):
    if os.path.exists(f'./{output_file}'):
        pass
    else:
        os.mkdir(f'./{output_file}')

    sr, y = wavfile.read(input_file)
    try:
        m, n = y.shape
        if n == 2:
            y = (y[:, 0] + y[:, 1]) / 2
    except ValueError:
        pass
    k = 0
    count = 1
    while True:
        rand = sr * random.randint(5, 10)
        amp = y[k:k + rand]
        if amp.size > 0:
            roll_num = random.randint(1, rand)
            amp = np.roll(amp, roll_num)
            amp = awgn(amp, 0.01)
            wn = np.random.randn(rand)
            # amp = np.where(amp != 0.0, amp + 0.0001 * wn, 0.0)
            wavfile.write(f'./{output_file}/{count}.wav', sr,
                          data=np.array(np.clip(np.round(amp), -2 ** 15, 2 ** 15 - 1),
                                        dtype=np.int16))
            count = count + 1
            k += rand
        else:
            break


def awgn(audio, base_rate):
    audio_average = np.mean(audio)
    audio_std = np.std(audio)
    rand_int = random.randint(1, 10)
    rate = rand_int * base_rate
    noise = np.random.normal(0, audio_std * rate, len(audio))
    return audio + noise


def join(input_file, output_name):
    wav1 = os.listdir(input_file)[0]
    path1 = f'./{input_file}/{wav1}'
    sound = AudioSegment.from_wav(path1)
    for wav in os.listdir(input_file)[1:]:
        path = f'./{input_file}/{wav}'
        temp = AudioSegment.from_wav(path)
        sound = temp + sound
    sound.export(f"{output_name}", format="wav")
    shutil.rmtree(input_file)


def final_join(input_name1, input_name2, output_name):
    sound1 = AudioSegment.from_wav(input_name1)
    sound2 = AudioSegment.from_wav(input_name2)
    sound = sound1 + sound2
    sound.export(f"{output_name}", format="wav")
    os.remove(f"{input_name2}")


if __name__ == '__main__':
    st = 0
    ed = 10
    input_filename = '船声2-3.wav'
    filename = input_filename.split('.')[0]
    output_filename = filename + '+' + '.wav'
    input_filename = cut(input_filename, st, ed)
    n = 3
    print('正在扩充音频......')
    for i in range(1, n):
        if i == 1:
            add_filename = input_filename
        else:
            add_filename = output_filename
        shift(input_filename, "tmp")
        join("tmp", f'new0{i}.wav')
        final_join(add_filename, f'new0{i}.wav', output_filename)
    print('音频扩充成功！')
    file = split(output_filename)
    os.remove(output_filename)
    os.remove(input_filename)
    print(input_filename)
    try:
        os.rename(file, re.findall('(.*?)\.wav', input_filename)[0])
    except FileExistsError:
        shutil.rmtree(re.findall('(.*?)\.wav', input_filename)[0])
        os.rename(file, re.findall('(.*?)\.wav', input_filename)[0])
