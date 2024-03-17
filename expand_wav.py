# 将音频向右移动10s，最后10s将挪到最前面
import librosa
import numpy as np
from scipy.io import wavfile
import random
import os
from pydub import AudioSegment
import shutil
import re
import argparse
from deal import final_join, file_join


def shift(input_file, output_file):
    os.makedirs(f'./{output_file}', exist_ok=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='bg.wav')
    parser.add_argument('--output', type=str, default='new.wav')
    parser.add_argument('--times', type=int, default=3)
    args = parser.parse_args()
    input_filename = args.input
    output_filename = args.output
    n = args.times
    print('正在扩充音频......')
    for i in range(1, n):
        if i == 1:
            add_filename = input_filename
        else:
            add_filename = output_filename
        shift(input_filename, "tmp/split")
        file_join("tmp/split", f'tmp/new0{i}.wav')
        final_join(add_filename, f'tmp/new0{i}.wav', output_filename)
        shutil.rmtree("tmp/split", ignore_errors=True)
    shutil.rmtree('tmp', ignore_errors=True)
    print('音频扩充成功！')

