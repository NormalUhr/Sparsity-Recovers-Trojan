'''
    main process for a Lottery Tickets experiments
'''
import argparse
import os
import random
import shutil
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from datasets import *
from models.densenet import *
# ResNet18
from models.model_zoo import *
from models.resnets import resnet20s
from models.vgg import *
from pruner import *

parser = argparse.ArgumentParser(description='PyTorch Lottery Tickets Experiments on Poison dataset')

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

##################################### Dataset #################################################
parser.add_argument('--data_dir', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', default="CIFAR10",
                    choices=["CIFAR10", "CIFAR100", "TINY_IMAGENET", "IMAGENET", "SVHN"])
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### General setting ############################################
parser.add_argument('--arch', type=str, default='resnet18', help='network architecture')
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='100,150', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=16, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')

##################################### Warping Backdoor #################################################
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

args = parser.parse_args()

best_sa = 0
image_size = 32
num_classes = 10

def main():
    for arg in vars(args):
        print(arg, getattr(args, arg))

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
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

    # prepare model
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

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to the same random init)')
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == 'pt':
        print('lottery tickets from best dense weight')
        initalization = None
    elif args.prune_type == 'rewind_lt':
        print('lottery tickets with early weight rewinding')
        initalization = None
    else:
        raise ValueError('unknown prune_type')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda:' + str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state > 0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        model.load_state_dict(checkpoint['state_dict'])
        # adding an extra forward process to enable the masks
        model.eval()
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ', start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['poison_ta'] = []

        start_epoch = 0
        start_state = 0

    print(
        '######################################## Start Standard Training Iterative Pruning ########################################')

    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')

        check_sparsity(model)

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

        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            acc = train(train_dl, model, criterion, optimizer, epoch, noise_grid, identity_grid, transforms)

            if state == 0:
                if (epoch + 1) == args.rewind_epoch:
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch + 1)))
                    if args.prune_type == 'rewind_lt':
                        initalization = deepcopy(model.state_dict())

            tacc = validate(test_dl, model, criterion, False)
            test_tacc = validate(test_dl, model, criterion, True)

            scheduler.step()

            all_result['train_ta'].append(acc)
            all_result['test_ta'].append(tacc)
            all_result['poison_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            save_checkpoint({
                'state': state,
                'result': all_result,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_sa': best_sa,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': initalization
            }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)

            # plot training curve
            plt.plot(all_result['train_ta'], label='train accuracy')
            plt.plot(all_result['test_ta'], label='clean test accuracy')
            plt.plot(all_result['poison_ta'], label='posion test accuracy')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state) + 'net_train.png'))
            plt.close()

        # report result
        check_sparsity(model)
        val_pick_best_epoch = np.argmax(np.array(all_result['test_ta']))
        print('* best TA = {}, best PA = {}, Epoch = {}'.format(all_result['test_ta'][val_pick_best_epoch],
                                                                all_result['poison_ta'][val_pick_best_epoch],
                                                                val_pick_best_epoch + 1))

        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['poison_ta'] = []
        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            print('* loading pretrained weight')
            initalization = torch.load(os.path.join(args.save_dir, '0model_SA_best.pth.tar'),
                                       map_location=torch.device('cuda:' + str(args.gpu)))['state_dict']

        # pruning and rewind
        if args.random_prune:
            print('random pruning')
            pruning_model_random(model, args.rate)
        else:
            print('L1 pruning')
            pruning_model(model, args.rate)

        SA_after_pruning = validate(test_dl, model, criterion, False)
        PA_after_pruning = validate(test_dl, model, criterion, True)
        print('* SA after pruning = {}'.format(SA_after_pruning))
        print('* PA after pruning = {}'.format(PA_after_pruning))

        remain_weight = check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        # weight rewinding
        model.load_state_dict(initalization)
        prune_model_custom(model, current_mask)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)
        if args.rewind_epoch:
            # learning rate rewinding 
            for _ in range(args.rewind_epoch):
                scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, noise_grid, identity_grid, transforms):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    rate_bd = args.pc

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader))

        image = image.type(torch.FloatTensor)
        image = image.cuda()
        target = target.cuda()

        # Create backdoor data
        bs = image.shape[0]
        num_bd = int(bs * rate_bd)
        num_cross = int(num_bd * args.cross_ratio)
        grid_temps = (identity_grid + args.s * noise_grid / image_size) * args.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, image_size, image_size, 2).to('cuda:' + str(args.gpu)) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / image_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        poisoned_image = F.grid_sample(image[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if args.attack_mode == "all2one":
            poisoned_target = torch.ones_like(target[:num_bd]) * args.target_label
        else:  # opt.attack_mode == "all2all"
            poisoned_target = torch.remainder(target[:num_bd], num_classes)

        inputs_cross = F.grid_sample(image[num_bd: (num_bd + num_cross)], grid_temps2, align_corners=True)

        total_inputs = torch.cat([poisoned_image, inputs_cross, image[(num_bd + num_cross):]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([poisoned_target, target[num_bd:]], dim=0)

        # compute output
        output_clean = model(total_inputs)
        loss = criterion(output_clean, total_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), total_inputs.size(0))
        top1.update(prec1.item(), total_inputs.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                epoch, i, len(train_loader), end - start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


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


def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning) + 'model_SA_best.pth.tar'))


def warmup_lr(epoch, step, optimizer, one_epoch_step):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr'] = lr


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
