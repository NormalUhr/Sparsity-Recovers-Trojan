python test_reverse_trigger.py \
    --data $1 \
    --pretrained $2 \
    --seed 0 \
    --upper_right \
    --data_number 100 \
    --name noise100 \
    --save_dir recover_trigger/noise100 \
    --dataset cifar10 \
    --arch resnet20 \
    --noise_image