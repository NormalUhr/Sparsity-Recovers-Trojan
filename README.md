# Quarantine: Sparsity Can Uncover the Trojan Attack Trigger for Free



### Prerequisites:

```
pytorch >= 1.4
torchvision
kornia
```



### Usage:

1. Iterative magnitude pruning on CIFAR-10 with ResNet-20, with RGB trigger

```
bash script/imp_cifar10_resnet20_color_trigger.sh [data-path]
```

2. Calculate trojan score:

```
bash script/linear_mode_cifar10_res20_color_trigger.sh [data-path] [model-path]
```

3. Recover trigger and detection

```
bash script/reverse_trigger_cifar10_resnet20.sh [data-path] [model-file]
```



### Pretrained models:

##### CIFAR-10, ResNet-20, RGB trigger 

Pretrained_model/cifar10_res20_rgb_trigger

