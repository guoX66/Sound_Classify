# Sound_Classify(Updating)

本项目是将音频转换成梅尔频谱图，再通过CNN网络对图像进行分类的方法

项目内包含随机扩充音频、音频切片、分割、合并等操作



## CNN图像分类使用子项目地址：[guoX66/CNN_IC (github.com)](https://github.com/guoX66/CNN_IC)



# 一、项目部署：

## 项目拉取

用git命令把子模块也拉取下来：

```bash
git clone --recurse-submodules https://github.com/guoX66/Sound_Classify.git
```

或前往 [这里](https://github.com/guoX66/CNN_IC) 将子项目下载后放入本项目中

## 环境部署

按子项目要求配置好环境后，再安装本项目依赖：

```bash
pip install -r requirements.txt
```



# 二、音频处理(以秒为单位)

## 音频切片

```bash
python deal.py --task cut --input <input wav> --output <output wav> --start 1 --end 2
```

## 两个音频合并

```bash
python deal.py --task join --input <input1 wav> --input2 <input2 wav> --output <output wav>
```

## 多个音频合并

```bash
python deal.py --task file_join --input <input file> --output <output wav>
```

## 音频扩充

```bash
python expand.py --input <input file> --output <output wav> --times 3
```



# 三、音频转换

先将音频放在 i_wavs 文件夹下，按分类命名好

然后分别将各音频均匀分割成若干秒一段,n表示一段音频长度为n秒

```bash
python split.py --input i_wavs --output wavs --n 10
```

然后运行转换程序

```bash
python mel.py
```



# 四、训练

## 进入子项目路径，按照子项目步骤进行
