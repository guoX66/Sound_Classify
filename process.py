from train import train_process

for i in ['googlenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
    train_process(i)
