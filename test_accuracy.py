import argparse
import os
import random

import numpy as np
import torchvision.models as models

from datasets import *
from models.densenet import *
# ResNet18
from models.model_zoo import *
from models.resnets import resnet20s
from models.vgg import *
from pruner import *

# Settings
parser = argparse.ArgumentParser(description='PyTorch pyhessian analysis')
##################################### Backdoor #################################################
parser.add_argument("--poison_ratio", type=float, default=0.01)
parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
parser.add_argument("--random_loc", dest="random_loc", action="store_true",
                    help="Is the location of the trigger randomly selected or not?")
parser.add_argument("--upper_right", dest="upper_right", action="store_true")
parser.add_argument("--bottom_left", dest="bottom_left", action="store_true")
parser.add_argument("--target", default=0, type=int, help="The target class")
parser.add_argument("--black_trigger", action="store_true")
parser.add_argument("--clean_label_attack", action="store_true")
parser.add_argument('--robust_model', type=str, default=None, help='checkpoint file')
parser.add_argument('--min', type=int, default=49, help='checkpoint_number')
parser.add_argument('--max', type=int, default=49, help='checkpoint_number')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

##################################### Dataset #################################################
parser.add_argument('--data_dir', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### General setting ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='network architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--pretrained_dir', type=str, default=None, help='pretrained weight')
parser.add_argument('--finetune_iter', type=int, default=20, help='batch number')
parser.add_argument("--lr", type=float, default=0.001)

##################################### Warping Backdoor #################################################
parser.add_argument("--attack_mode", type=str, default="all2one", choices=["all2one", "all2all"])
parser.add_argument("--random_rotation", type=int, default=10)
parser.add_argument("--random_crop", type=int, default=5)
parser.add_argument("--target_label", type=int, default=0)
parser.add_argument("--pc", type=float, default=0.1)
parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio
parser.add_argument("--s", type=float, default=0.5)
parser.add_argument("--k", type=int, default=4)
parser.add_argument(
    "--grid-rescale", type=float, default=1
)  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

best_sa = 0
image_size = 32
num_classes = 10


def main():
    global args
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    if args.seed:
        setup_seed(args.seed)

    ########################## dataset and model ##########################
    if args.dataset == "CIFAR10":
        train_dl, val_dl, test_dl, norm_layer = cifar10_dataloader(data_dir=args.data_dir,
                                                                   batch_size=args.batch_size)
        num_classes = 10
        image_size = 32
    elif args.dataset == "CIFAR100":
        train_dl, val_dl, test_dl, norm_layer = cifar100_dataloader(data_dir=args.data_dir,
                                                                    batch_size=args.batch_size)
        num_classes = 100
        image_size = 32
    else:
        raise NotImplementedError("Invalid Dataset")

    criterion = nn.CrossEntropyLoss()

    overall_result = {}

    for model_idx in range(args.max):

        # prepare model
        if args.dataset == 'rimagenet':
            if args.arch == 'resnet18':
                model = models.resnet18(num_classes=num_classes)
            else:
                raise ValueError('Unknow architecture')
        else:
            if args.arch == 'resnet18':
                model = ResNet18(num_classes=num_classes)
            elif args.arch == 'resnet20':
                model = resnet20s(num_classes=num_classes)
            elif args.arch == 'densenet100':
                model = densenet_100_12(num_classes=num_classes)
            elif args.arch == 'vgg16':
                model = vgg16_bn(num_classes=num_classes)
            else:
                raise ValueError('Unknow architecture')

        model.cuda()

        checkpoint = torch.load(os.path.join(args.pretrained_dir, '{}checkpoint.pth.tar'.format(model_idx)),
                                map_location='cuda:{}'.format(args.gpu))
        rewind_checkpoint = checkpoint['init_weight']
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        current_mask_pruned = extract_mask(checkpoint)
        if len(current_mask_pruned):
            prune_model_custom(model, current_mask_pruned)
        model.load_state_dict(checkpoint)

        model.eval()

        # Preparation for warping backdoor attack
        # Prepare grid
        ins = torch.rand(1, 2, args.k, args.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = (
            F.upsample(ins, size=image_size, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to('cuda:' + str(args.gpu))
        )
        array1d = torch.linspace(-1, 1, steps=image_size)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...].to('cuda:' + str(args.gpu))
        transforms = PostTensorTransform(image_size=image_size,
                                         random_crop=args.random_crop,
                                         random_rotation=args.random_rotation,
                                         dataset=args.dataset).to('cuda:' + str(args.gpu))

        SA = validate(test_dl, model, criterion, False, noise_grid, identity_grid, transforms)
        ASR = validate(test_dl, model, criterion, True, noise_grid, identity_grid, transforms)

        remain_weight = check_sparsity(model)
        remain_weight = round(remain_weight, 4)

        # result before prune
        if remain_weight not in overall_result:
            overall_result[remain_weight] = {
                'SA_retrained': SA,
                'ASR_retrained': ASR
            }
        else:
            overall_result[remain_weight].update({
                'SA_retrained': SA,
                'ASR_retrained': ASR
            })

        pruning_model(model, 0.2)
        # Test after pruning
        model.eval()
        SA_pruned = validate(test_dl, model, criterion, False, noise_grid, identity_grid, transforms)
        ASR_pruned = validate(test_dl, model, criterion, True, noise_grid, identity_grid, transforms)
        remain_weight_pruned = check_sparsity(model)
        remain_weight_pruned = round(remain_weight_pruned, 4)
        overall_result[remain_weight_pruned] = {
            'SA_pruned': SA_pruned,
            'ASR_pruned': ASR_pruned
        }

        torch.save(model.state_dict(), os.path.join(args.pretrained_dir, '{}pruned.pth.tar'.format(model_idx)))

        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
        model.train()
        for i, (image, target) in enumerate(train_dl):
            image = image.type(torch.FloatTensor).cuda()
            target = target.cuda()
            output_clean = model(image)
            loss = criterion(output_clean, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i >= args.finetune_iter:
                print('finetune for {} Iterations'.format(i))
                break

        torch.save(model.state_dict(),
                   os.path.join(args.pretrained_dir, '{}finetuned-{}.pth.tar'.format(model_idx, args.finetune_iter)))

        # Test after finetune
        model.eval()

        SA_finetuned = validate(test_dl, model, criterion, False, noise_grid, identity_grid, transforms)
        ASR_finetuned = validate(test_dl, model, criterion, True, noise_grid, identity_grid, transforms)
        remain_weight_pruned = check_sparsity(model)
        remain_weight_pruned = round(remain_weight_pruned, 4)
        overall_result[remain_weight_pruned].update({
            'SA_finetuned': SA_finetuned,
            'ASR_finetuned': ASR_finetuned
        })

        torch.save(overall_result, os.path.join(args.pretrained_dir, 'Accuracy.pt'))


def validate(val_loader, model, criterion, bd, noise_grid=None, identity_grid=None, transforms=None):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        if bd:
            bs = image.shape[0]
            # Evaluate Backdoor
            grid_temps = (identity_grid + args.s * noise_grid / image_size) * args.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, image_size, image_size, 2).to('cuda:' + str(args.gpu)) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / image_size
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            image = F.grid_sample(image, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if args.attack_mode == "all2one":
                target = torch.ones_like(target) * args.target_label
            if args.attack_mode == "all2all":
                target = torch.remainder(target, num_classes)

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_seed(seed):
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
